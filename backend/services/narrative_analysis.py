import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

from backend.llm.base import LLMCallMetadata, LLMProvider
from backend.llm.schemas import (
    MentionExtractionOutput,
    RelationExtractionOutput,
)
from backend.prompts.mention_extraction import (
    PROMPT_VERSION as MENTION_PROMPT_VERSION,
    build_mention_extraction_messages,
)
from backend.prompts.relation_extraction import (
    PROMPT_VERSION as RELATION_PROMPT_VERSION,
    build_relation_extraction_messages,
)
from backend.services.evidence_validator import validate_evidence
from backend.services.project_storage import ProjectNotFoundError
from backend.storage.database import DatabasePath, database_session
from backend.storage.schema import create_schema


PROMPT_VERSION = (
    f"{MENTION_PROMPT_VERSION}+"
    f"{RELATION_PROMPT_VERSION}"
)
SCHEMA_VERSION = "character_relation_mvp_schema_v1"
MENTION_SCHEMA_VERSION = "character_mention_schema_v1"
RELATION_SCHEMA_VERSION = "free_character_relation_schema_v1"
ANALYSIS_STATUS_RUNNING = "running"
ANALYSIS_STATUS_QUEUED = "queued"
ANALYSIS_STATUS_INTERRUPTED = "interrupted"
ANALYSIS_STATUS_COMPLETED = "completed"
ANALYSIS_STATUS_PARTIAL = "partial"
ANALYSIS_STATUS_FAILED = "failed"
MENTION_LIMIT = 40
RELATION_LIMIT = 15
_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NarrativeChunk:
    database_id: int
    source_file_id: int
    chunk_id: str
    global_order: int
    text: str


@dataclass(frozen=True, slots=True)
class NarrativeAnalysisResult:
    run_id: int
    project_id: str
    status: str
    total_chunks: int
    processed_chunks: int
    successful_chunks: int
    failed_chunks: int
    partial_chunks: int
    cached_chunks: int
    cached_layers: int


class ActiveNarrativeAnalysisError(RuntimeError):
    def __init__(self, active_run_id: int) -> None:
        super().__init__("该项目已有分析任务正在执行。")
        self.active_run_id = active_run_id


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def compute_analysis_input_hash(
    *,
    previous_context: str,
    target_text: str,
    next_context: str,
) -> str:
    payload = json.dumps(
        {
            "previous_context": previous_context,
            "target_text": target_text,
            "next_context": next_context,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def compute_json_hash(
    payload: Any,
) -> str:
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized_payload.encode("utf-8")
    ).hexdigest()


def _append_warning(
    result: dict[str, Any], message: str) -> None:
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        result["warnings"] = warnings
    warnings.append(message)


def _truncate_layer_items(
    result: dict[str, Any],
    *,
    field_name: str,
    limit: int,
    layer_name: str,
) -> dict[str, Any]:
    items = result.get(field_name)
    if not isinstance(items, list):
        result[field_name] = []
        return result

    if len(items) <= limit:
        return result

    result[field_name] = items[:limit]
    _append_warning(
        result,
        f"{layer_name} 输出数量超过上限 {limit}，已截断为前 {limit} 条。",
    )
    return result


def _build_layer_timeout_warning(
    *,
    chunk_id: str,
    layer_name: str,
    error: Exception,
) -> str:
    return (
        f"{chunk_id} 的 {layer_name} 层失败：{error}"
    )


def _empty_layer_result(
    *,
    field_name: str,
    warning: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_result = {
        field_name: [],
        "warnings": [],
    }
    validated_result = {
        field_name: [],
        "warnings": [],
    }

    if warning:
        raw_result["warnings"].append(warning)
        validated_result["warnings"].append(warning)

    return raw_result, validated_result


async def _run_layer(
    *,
    chunk_id: str,
    layer_name: str,
    provider: LLMProvider,
    messages: list[dict[str, str]],
    response_model: type,
) -> Any:
    """Call LLM with timing and logging for a single analysis layer."""

    started_at = perf_counter()

    _logger.info(
        "开始叙事分析：chunk=%s layer=%s",
        chunk_id,
        layer_name,
    )

    try:
        result = await provider.generate_structured(
            messages=messages,
            response_model=response_model,
            temperature=0,
            metadata=LLMCallMetadata(
                chunk_id=chunk_id,
                layer_name=layer_name,
                is_repair=False,
            ),
        )
    except Exception:
        elapsed = perf_counter() - started_at

        _logger.exception(
            "叙事分析失败：chunk=%s layer=%s elapsed=%.2fs",
            chunk_id,
            layer_name,
            elapsed,
        )
        raise

    elapsed = perf_counter() - started_at

    _logger.info(
        "叙事分析完成：chunk=%s layer=%s elapsed=%.2fs",
        chunk_id,
        layer_name,
        elapsed,
    )

    return result


def _load_project_chunks(
    *,
    project_id: str,
    database_path: DatabasePath | None,
) -> list[NarrativeChunk]:
    with database_session(database_path=database_path) as connection:
        project_row = connection.execute(
            """
            SELECT id
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()

        if project_row is None:
            raise ProjectNotFoundError(
                f"Project does not exist: {project_id}"
            )

        rows = connection.execute(
            """
            SELECT
                tc.id,
                tc.source_file_id,
                tc.chunk_id,
                tc.global_order,
                tc.text
            FROM text_chunks AS tc
            INNER JOIN source_files AS sf
                ON sf.id = tc.source_file_id
            WHERE sf.project_id = ?
            ORDER BY
                sf.file_order,
                tc.global_order
            """,
            (project_id,),
        ).fetchall()

        return [
            NarrativeChunk(
                database_id=row["id"],
                source_file_id=row["source_file_id"],
                chunk_id=row["chunk_id"],
                global_order=row["global_order"],
                text=row["text"],
            )
            for row in rows
        ]


def _get_active_run_id(
    *,
    connection: sqlite3.Connection,
    project_id: str,
) -> int | None:
    row = connection.execute(
        """
        SELECT id
        FROM narrative_analysis_runs
        WHERE project_id = ?
          AND status IN (?, ?)
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            project_id,
            ANALYSIS_STATUS_QUEUED,
            ANALYSIS_STATUS_RUNNING,
        ),
    ).fetchone()

    if row is None:
        return None

    return int(row["id"])


def _get_run_project_id(
    *,
    connection: sqlite3.Connection,
    run_id: int,
) -> str:
    row = connection.execute(
        """
        SELECT project_id
        FROM narrative_analysis_runs
        WHERE id = ?
        """,
        (run_id,),
    ).fetchone()

    if row is None:
        raise LookupError(
            f"Narrative analysis run does not exist: {run_id}"
        )

    return str(row["project_id"])


def _get_pending_chunk_database_ids(
    *,
    run_id: int,
    database_path: DatabasePath | None,
) -> set[int]:
    with database_session(database_path=database_path) as connection:
        rows = connection.execute(
            """
            SELECT chunk_database_id
            FROM narrative_unit_analyses
            WHERE run_id = ?
              AND status = 'pending'
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()

    return {
        int(row["chunk_database_id"])
        for row in rows
    }


def _get_run_request(
    *,
    run_id: int,
    database_path: DatabasePath | None,
) -> tuple[str, dict[str, Any]]:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT project_id, request_json
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            raise LookupError(
                f"Narrative analysis run does not exist: {run_id}"
            )

        raw_request = row["request_json"]

    if raw_request:
        parsed_request = json.loads(raw_request)
        if isinstance(parsed_request, dict):
            return str(row["project_id"]), parsed_request

    return str(row["project_id"]), {}


def _precreate_pending_units(
    *,
    connection: sqlite3.Connection,
    run_id: int,
    project_id: str,
    chunks: Sequence[NarrativeChunk],
    provider: LLMProvider,
    previous_context_chars: int,
    next_context_chars: int,
) -> None:
    for index, chunk in enumerate(chunks):
        previous_context, next_context = _build_contexts(
            chunks=chunks,
            index=index,
            previous_context_chars=previous_context_chars,
            next_context_chars=next_context_chars,
        )
        analysis_input_hash = compute_analysis_input_hash(
            previous_context=previous_context,
            target_text=chunk.text,
            next_context=next_context,
        )

        connection.execute(
            """
            INSERT INTO narrative_unit_analyses (
                run_id,
                project_id,
                chunk_database_id,
                chunk_id,
                text_hash,
                analysis_input_hash,
                status,
                provider,
                model,
                prompt_version,
                schema_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                project_id,
                chunk.database_id,
                chunk.chunk_id,
                compute_text_hash(chunk.text),
                analysis_input_hash,
                "pending",
                provider.provider_name,
                provider.model_name,
                PROMPT_VERSION,
                SCHEMA_VERSION,
            ),
        )


def _create_analysis_run(
    *,
    project_id: str,
    provider: LLMProvider,
    database_path: DatabasePath | None,
    status: str = ANALYSIS_STATUS_RUNNING,
    total_chunks: int = 0,
    request_json: str | None = None,
) -> int:
    with database_session(database_path=database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO narrative_analysis_runs (
                project_id,
                provider,
                model,
                prompt_version,
                schema_version,
                status,
                total_chunks,
                request_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                provider.provider_name,
                provider.model_name,
                PROMPT_VERSION,
                SCHEMA_VERSION,
                status,
                total_chunks,
                request_json,
            ),
        )

        run_id = cursor.lastrowid

        if run_id is None:
            raise RuntimeError(
                "Could not create narrative analysis run."
            )

        return run_id


def create_narrative_analysis_job(
    *,
    project_id: str,
    provider: LLMProvider,
    database_path: DatabasePath | None = None,
    max_chunks: int | None = None,
    previous_context_chars: int = 500,
    next_context_chars: int = 0,
    force_reanalyze: bool = False,
) -> NarrativeAnalysisResult:
    normalized_project_id = project_id.strip()

    if not normalized_project_id:
        raise ValueError("project_id cannot be empty.")

    chunks = _load_project_chunks(
        project_id=normalized_project_id,
        database_path=database_path,
    )

    if max_chunks is not None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than 0.")

        chunks = chunks[:max_chunks]

    request_json = json.dumps(
        {
            "max_chunks": max_chunks,
            "previous_context_chars": previous_context_chars,
            "next_context_chars": next_context_chars,
            "force_reanalyze": force_reanalyze,
        },
        ensure_ascii=False,
    )

    with database_session(database_path=database_path) as connection:
        active_run_id = _get_active_run_id(
            connection=connection,
            project_id=normalized_project_id,
        )

        if active_run_id is not None:
            raise ActiveNarrativeAnalysisError(active_run_id)

        try:
            cursor = connection.execute(
                """
                INSERT INTO narrative_analysis_runs (
                    project_id,
                    provider,
                    model,
                    prompt_version,
                    schema_version,
                    status,
                    total_chunks,
                    request_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_project_id,
                    provider.provider_name,
                    provider.model_name,
                    PROMPT_VERSION,
                    SCHEMA_VERSION,
                    ANALYSIS_STATUS_QUEUED,
                    len(chunks),
                    request_json,
                ),
            )
        except sqlite3.IntegrityError as error:
            active_run_id = _get_active_run_id(
                connection=connection,
                project_id=normalized_project_id,
            )
            raise ActiveNarrativeAnalysisError(
                active_run_id or 0
            ) from error

        run_id = cursor.lastrowid

        if run_id is None:
            raise RuntimeError("Could not create narrative analysis run.")

        _precreate_pending_units(
            connection=connection,
            run_id=run_id,
            project_id=normalized_project_id,
            chunks=chunks,
            provider=provider,
            previous_context_chars=previous_context_chars,
            next_context_chars=next_context_chars,
        )

    return NarrativeAnalysisResult(
        run_id=int(run_id),
        project_id=normalized_project_id,
        status=ANALYSIS_STATUS_QUEUED,
        total_chunks=len(chunks),
        processed_chunks=0,
        successful_chunks=0,
        failed_chunks=0,
        partial_chunks=0,
        cached_chunks=0,
        cached_layers=0,
    )


def _update_analysis_run(
    *,
    run_id: int,
    status: str,
    database_path: DatabasePath | None,
    error_message: str | None = None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE narrative_analysis_runs
            SET
                status = ?,
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status,
                error_message,
                run_id,
            ),
        )


def _save_unit_analysis(
    *,
    run_id: int,
    project_id: str,
    chunk: NarrativeChunk,
    provider: LLMProvider,
    status: str,
    text_hash: str,
    analysis_input_hash: str,
    database_path: DatabasePath | None,
    result_json: str | None = None,
    validated_result_json: str | None = None,
    error_message: str | None = None,
    cache_hit: bool = False,
    cache_source_unit_id: int | None = None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            INSERT INTO narrative_unit_analyses (
                run_id,
                project_id,
                chunk_database_id,
                chunk_id,
                text_hash,
                analysis_input_hash,
                status,
                cache_hit,
                cache_source_unit_id,
                provider,
                model,
                prompt_version,
                schema_version,
                result_json,
                validated_result_json,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                run_id,
                chunk_database_id
            )
            DO UPDATE SET
                text_hash = excluded.text_hash,
                analysis_input_hash = excluded.analysis_input_hash,
                status = excluded.status,
                cache_hit = excluded.cache_hit,
                cache_source_unit_id = excluded.cache_source_unit_id,
                provider = excluded.provider,
                model = excluded.model,
                prompt_version = excluded.prompt_version,
                schema_version = excluded.schema_version,
                result_json = excluded.result_json,
                validated_result_json = excluded.validated_result_json,
                error_message = excluded.error_message,
                last_finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                run_id,
                project_id,
                chunk.database_id,
                chunk.chunk_id,
                text_hash,
                analysis_input_hash,
                status,
                int(cache_hit),
                cache_source_unit_id,
                provider.provider_name,
                provider.model_name,
                PROMPT_VERSION,
                SCHEMA_VERSION,
                result_json,
                validated_result_json,
                error_message,
            ),
        )

        _refresh_run_progress(
            connection=connection,
            run_id=run_id,
            current_chunk_id=None,
        )


def _mark_unit_running(
    *,
    run_id: int,
    chunk: NarrativeChunk,
    database_path: DatabasePath | None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE narrative_unit_analyses
            SET
                status = ?,
                attempt_count = attempt_count + 1,
                last_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND chunk_database_id = ?
            """,
            (
                ANALYSIS_STATUS_RUNNING,
                run_id,
                chunk.database_id,
            ),
        )
        _refresh_run_progress(
            connection=connection,
            run_id=run_id,
            current_chunk_id=chunk.chunk_id,
        )


def _refresh_run_progress(
    *,
    connection: sqlite3.Connection,
    run_id: int,
    current_chunk_id: str | None,
) -> None:
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS total_chunks,
            SUM(CASE WHEN status IN ('completed', 'partial', 'failed') THEN 1 ELSE 0 END) AS processed_chunks,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS successful_chunks,
            SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) AS partial_chunks,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_chunks,
            SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cached_chunks
        FROM narrative_unit_analyses
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()

    connection.execute(
        """
        UPDATE narrative_analysis_runs
        SET
            total_chunks = ?,
            processed_chunks = ?,
            successful_chunks = ?,
            partial_chunks = ?,
            failed_chunks = ?,
            cached_chunks = ?,
            current_chunk_id = COALESCE(?, current_chunk_id),
            heartbeat_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            row["total_chunks"] or 0,
            row["processed_chunks"] or 0,
            row["successful_chunks"] or 0,
            row["partial_chunks"] or 0,
            row["failed_chunks"] or 0,
            row["cached_chunks"] or 0,
            current_chunk_id,
            run_id,
        ),
    )


def _increment_run_cached_layers(
    *,
    run_id: int,
    database_path: DatabasePath | None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE narrative_analysis_runs
            SET
                cached_layers = cached_layers + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (run_id,),
        )


def _summarize_run_from_connection(
    *,
    connection: sqlite3.Connection,
    run_id: int,
    status: str | None = None,
) -> NarrativeAnalysisResult:
    run_row = connection.execute(
        """
        SELECT
            project_id,
            status,
            total_chunks,
            processed_chunks,
            successful_chunks,
            partial_chunks,
            failed_chunks,
            cached_chunks,
            cached_layers
        FROM narrative_analysis_runs
        WHERE id = ?
        """,
        (run_id,),
    ).fetchone()

    if run_row is None:
        raise LookupError(
            f"Narrative analysis run does not exist: {run_id}"
        )

    return NarrativeAnalysisResult(
        run_id=run_id,
        project_id=str(run_row["project_id"]),
        status=status or str(run_row["status"]),
        total_chunks=int(run_row["total_chunks"] or 0),
        processed_chunks=int(run_row["processed_chunks"] or 0),
        successful_chunks=int(run_row["successful_chunks"] or 0),
        failed_chunks=int(run_row["failed_chunks"] or 0),
        partial_chunks=int(run_row["partial_chunks"] or 0),
        cached_chunks=int(run_row["cached_chunks"] or 0),
        cached_layers=int(run_row["cached_layers"] or 0),
    )


def _finalize_analysis_run(
    *,
    run_id: int,
    database_path: DatabasePath | None,
    error_message: str | None = None,
) -> NarrativeAnalysisResult:
    with database_session(database_path=database_path) as connection:
        _refresh_run_progress(
            connection=connection,
            run_id=run_id,
            current_chunk_id=None,
        )
        row = connection.execute(
            """
            SELECT
                total_chunks,
                processed_chunks,
                successful_chunks,
                partial_chunks,
                failed_chunks
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            raise LookupError(
                f"Narrative analysis run does not exist: {run_id}"
            )

        total_chunks = int(row["total_chunks"] or 0)
        processed_chunks = int(row["processed_chunks"] or 0)
        successful_chunks = int(row["successful_chunks"] or 0)
        partial_chunks = int(row["partial_chunks"] or 0)
        failed_chunks = int(row["failed_chunks"] or 0)

        if processed_chunks < total_chunks:
            final_status = ANALYSIS_STATUS_PARTIAL
        elif failed_chunks == 0 and partial_chunks == 0:
            final_status = ANALYSIS_STATUS_COMPLETED
        elif successful_chunks == 0 and partial_chunks == 0:
            final_status = ANALYSIS_STATUS_FAILED
        else:
            final_status = ANALYSIS_STATUS_PARTIAL

        connection.execute(
            """
            UPDATE narrative_analysis_runs
            SET
                status = ?,
                error_message = ?,
                current_chunk_id = NULL,
                finished_at = CURRENT_TIMESTAMP,
                heartbeat_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                final_status,
                error_message,
                run_id,
            ),
        )

        return _summarize_run_from_connection(
            connection=connection,
            run_id=run_id,
            status=final_status,
        )


def _find_cached_unit_analysis(
    *,
    project_id: str,
    analysis_input_hash: str,
    provider: LLMProvider,
    database_path: DatabasePath | None,
) -> dict[str, Any] | None:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT
                id,
                result_json,
                validated_result_json
            FROM narrative_unit_analyses
            WHERE
                project_id = ?
                AND analysis_input_hash = ?
                AND provider = ?
                AND model = ?
                AND prompt_version = ?
                AND schema_version = ?
                AND status = ?
                AND result_json IS NOT NULL
                AND validated_result_json IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                project_id,
                analysis_input_hash,
                provider.provider_name,
                provider.model_name,
                PROMPT_VERSION,
                SCHEMA_VERSION,
                ANALYSIS_STATUS_COMPLETED,
            ),
        ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "result_json": row["result_json"],
            "validated_result_json": row["validated_result_json"],
        }


def _find_cached_layer_analysis(
    *,
    project_id: str,
    chunk: NarrativeChunk,
    layer_name: str,
    input_hash: str,
    provider: LLMProvider,
    prompt_version: str,
    schema_version: str,
    database_path: DatabasePath | None,
) -> dict[str, Any] | None:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT
                id,
                result_json,
                validated_result_json
            FROM narrative_layer_cache
            WHERE
                project_id = ?
                AND chunk_database_id = ?
                AND layer_name = ?
                AND input_hash = ?
                AND provider = ?
                AND model = ?
                AND prompt_version = ?
                AND schema_version = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                project_id,
                chunk.database_id,
                layer_name,
                input_hash,
                provider.provider_name,
                provider.model_name,
                prompt_version,
                schema_version,
            ),
        ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "result_json": row["result_json"],
            "validated_result_json": row["validated_result_json"],
        }


def _save_layer_analysis(
    *,
    project_id: str,
    chunk: NarrativeChunk,
    layer_name: str,
    input_hash: str,
    provider: LLMProvider,
    prompt_version: str,
    schema_version: str,
    result_json: str,
    validated_result_json: str,
    database_path: DatabasePath | None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            INSERT INTO narrative_layer_cache (
                project_id,
                chunk_database_id,
                chunk_id,
                layer_name,
                input_hash,
                provider,
                model,
                prompt_version,
                schema_version,
                result_json,
                validated_result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                project_id,
                chunk_database_id,
                layer_name,
                input_hash,
                provider,
                model,
                prompt_version,
                schema_version
            )
            DO UPDATE SET
                result_json = excluded.result_json,
                validated_result_json = excluded.validated_result_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                project_id,
                chunk.database_id,
                chunk.chunk_id,
                layer_name,
                input_hash,
                provider.provider_name,
                provider.model_name,
                prompt_version,
                schema_version,
                result_json,
                validated_result_json,
            ),
        )


def _build_contexts(
    *,
    chunks: Sequence[NarrativeChunk],
    index: int,
    previous_context_chars: int,
    next_context_chars: int,
) -> tuple[str, str]:
    previous_context = ""
    next_context = ""

    current_chunk = chunks[index]

    if index > 0 and previous_context_chars > 0:
        previous_chunk = chunks[index - 1]
        if previous_chunk.source_file_id == current_chunk.source_file_id:
            previous_context = previous_chunk.text[
                -previous_context_chars:
            ]

    if (
        index + 1 < len(chunks)
        and next_context_chars > 0
    ):
        next_chunk = chunks[index + 1]
        if next_chunk.source_file_id == current_chunk.source_file_id:
            next_context = next_chunk.text[
                :next_context_chars
            ]

    return previous_context, next_context


def _filter_validated_mentions(
    validated_result: dict[str, Any],
    chunk_id: str,
) -> dict[str, Any]:
    """Keep only evidence-located mentions; assign span-based stable IDs.

    IDs are of the form:  {chunk_id}_m_{type}_{mention_start}_{mention_end}
    e.g.  chunk_0001_m_character_125_129

    The span is computed as the mention_text's position within evidence_text,
    relative to the evidence's position in the chunk. This ensures that
    "韩立" in evidence "韩立和韩铸一起..." and "韩铸" in the same evidence
    produce different IDs even though evidence_text and evidence offsets are
    identical.
    """

    raw_mentions = validated_result.get("mentions")

    if not isinstance(raw_mentions, list):
        validated_result["mentions"] = []
        return validated_result

    warnings = validated_result.get("warnings")

    if not isinstance(warnings, list):
        warnings = []

    filtered_mentions: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()

    for index, mention in enumerate(raw_mentions):
        if not isinstance(mention, dict):
            warnings.append(
                f"mentions[{index}] 不是对象，已丢弃。"
            )
            continue

        mention_type = str(
            mention.get("mention_type") or ""
        ).strip()
        mention_text = str(
            mention.get("mention_text") or ""
        ).strip()
        evidence_text = str(
            mention.get("evidence_text") or ""
        ).strip()
        evidence_start = mention.get("start_offset")
        evidence_end = mention.get("end_offset")

        if mention_type != "character":
            warnings.append(
                f"mentions[{index}] 不是人物锚点，已丢弃："
                f"{mention_text or evidence_text or 'empty'}"
            )
            continue

        if not mention.get("evidence_validated") or evidence_start is None:
            warnings.append(
                f"mentions[{index}] 证据未能定位，已丢弃："
                f"{mention_text or evidence_text or 'empty'}"
            )
            continue

        if (
            mention_text
            and evidence_text
            and mention_text not in evidence_text
        ):
            warnings.append(
                f"mentions[{index}] mention_text 不在 evidence_text 中，"
                f"已丢弃：{mention_text}"
            )
            continue

        # Compute the mention span relative to evidence_text.
        # If mention_text appears multiple times in evidence_text, use the
        # occurrence_index to pick the right occurrence.
        occurrence_index = int(mention.get("occurrence_index") or 0)
        relative_pos = _find_nth_occurrence(
            evidence_text, mention_text, occurrence_index
        )

        if relative_pos is None:
            warnings.append(
                f"mentions[{index}] mention_text 在 evidence_text "
                f"第 {occurrence_index} 次出现不存在，已丢弃：{mention_text}"
            )
            continue

        relative_start, relative_end = relative_pos

        # Absolute span in the chunk.
        mention_start = evidence_start + relative_start
        mention_end = evidence_start + relative_end

        item_key = (
            mention_type.casefold(),
            mention_text.casefold(),
            occurrence_index,
        )

        if item_key in seen_keys:
            warnings.append(
                f"mentions[{index}] 与已有文本锚点重复，已丢弃："
                f"{mention_text}"
            )
            continue

        seen_keys.add(item_key)

        mention["mention_id"] = (
            f"{chunk_id}_m_{mention_type}_{mention_start}_{mention_end}"
        )
        mention["start_offset"] = mention_start
        mention["end_offset"] = mention_end
        filtered_mentions.append(mention)

    validated_result["mentions"] = filtered_mentions
    validated_result["warnings"] = warnings

    return validated_result


def _find_nth_occurrence(
    text: str,
    sub: str,
    n: int,
) -> tuple[int, int] | None:
    """Return (start, end) of the nth occurrence of sub in text, or None."""
    if not sub or n < 0:
        return None

    pos = 0
    count = 0
    while True:
        idx = text.find(sub, pos)
        if idx < 0:
            return None
        if count == n:
            return idx, idx + len(sub)
        pos = idx + 1
        count += 1


def _compute_stable_id(
    *,
    chunk_id: str,
    prefix: str,
    payload: dict[str, Any],
) -> str:
    """Build a stable, collision-resistant ID from a payload dict.

    Uses the first 12 hex chars of SHA-256 over the JSON-serialized payload.
    """
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:12]
    return f"{chunk_id}_{prefix}_{digest}"


def _build_mention_candidates(
    mentions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build compact mention candidates for downstream layers.

    Excludes evidence_text since TARGET_TEXT is already available to the model.
    """

    return [
        {
            "mention_id": mention.get("mention_id"),
            "mention_type": mention.get("mention_type"),
            "mention_text": mention.get("mention_text"),
            "start_offset": mention.get("start_offset"),
            "end_offset": mention.get("end_offset"),
        }
        for mention in mentions
    ]



def _filter_validated_relations(
    validated_result: dict[str, Any],
    *,
    mention_by_id: dict[str, dict[str, Any]],
    allowed_mention_ids: set[str],
    chunk_id: str,
) -> dict[str, Any]:
    """Validate relations against anchored mentions with ID-text correspondence."""
    raw_relations = validated_result.get("relations")

    if not isinstance(raw_relations, list):
        validated_result["relations"] = []
        return validated_result

    warnings = validated_result.get("warnings")

    if not isinstance(warnings, list):
        warnings = []

    filtered_relations: list[dict[str, Any]] = []
    seen_relations: set[tuple[str, str, str, str]] = set()

    for index, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            warnings.append(
                f"relations[{index}] 不是对象，已丢弃。"
            )
            continue

        source_mention = str(
            relation.get("source_mention") or ""
        ).strip()
        source_mention_id = str(
            relation.get("source_mention_id") or ""
        ).strip()
        relation_text = str(
            relation.get("relation") or ""
        ).strip()
        target_mention = str(
            relation.get("target_mention") or ""
        ).strip()
        target_mention_id = str(
            relation.get("target_mention_id") or ""
        ).strip()
        evidence_text = str(
            relation.get("evidence_text") or ""
        ).strip()

        if not relation.get("evidence_validated"):
            warnings.append(
                f"relations[{index}] 证据未能定位，已丢弃："
                f"{source_mention}->{target_mention}"
            )
            continue

        if (
            source_mention_id not in allowed_mention_ids
            or target_mention_id not in allowed_mention_ids
        ):
            warnings.append(
                f"relations[{index}] 引用了未定位的 mention，已丢弃："
                f"{source_mention}->{target_mention}"
            )
            continue

        # P1-5: Verify ID-text correspondence.
        source_info = mention_by_id.get(source_mention_id)
        target_info = mention_by_id.get(target_mention_id)

        source_text_match = (
            source_info is not None
            and source_info.get("mention_text", "").strip().casefold()
            == source_mention.casefold()
        )
        target_text_match = (
            target_info is not None
            and target_info.get("mention_text", "").strip().casefold()
            == target_mention.casefold()
        )

        if not source_text_match:
            warnings.append(
                f"relations[{index}] source_mention '{source_mention}' "
                f"与 ID {source_mention_id} 对应的文本不一致，已丢弃。"
            )
            continue

        if not target_text_match:
            warnings.append(
                f"relations[{index}] target_mention '{target_mention}' "
                f"与 ID {target_mention_id} 对应的文本不一致，已丢弃。"
            )
            continue

        item_key = (
            source_mention_id.casefold(),
            relation_text.casefold(),
            target_mention_id.casefold(),
            evidence_text,
        )

        if item_key in seen_relations:
            warnings.append(
                f"relations[{index}] 与已有关系重复，已丢弃。"
            )
            continue

        seen_relations.add(item_key)
        relation["_relation_payload"] = {
            "source": source_mention_id,
            "relation": relation_text,
            "target": target_mention_id,
            "start": relation.get("start_offset"),
            "end": relation.get("end_offset"),
        }
        filtered_relations.append(relation)

    for rel in filtered_relations:
        rel["relation_id"] = _compute_stable_id(
            chunk_id=chunk_id,
            prefix="r",
            payload=rel.pop("_relation_payload"),
        )

    validated_result["relations"] = filtered_relations
    validated_result["warnings"] = warnings

    return validated_result



async def analyze_project_narrative(
    *,
    project_id: str,
    database_path: DatabasePath | None = None,
    provider: LLMProvider | None = None,
    max_chunks: int | None = None,
    previous_context_chars: int = 500,
    next_context_chars: int = 0,
    force_reanalyze: bool = False,
    existing_run_id: int | None = None,
) -> NarrativeAnalysisResult:
    """
    Analyze saved text chunks and persist raw plus evidence-validated results.
    """

    normalized_project_id = project_id.strip()

    if not normalized_project_id:
        raise ValueError("project_id cannot be empty.")

    chunks = _load_project_chunks(
        project_id=normalized_project_id,
        database_path=database_path,
    )

    if max_chunks is not None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than 0.")

        chunks = chunks[:max_chunks]

    owns_provider = provider is None

    if provider is None:
        from backend.llm.factory import create_llm_provider

        resolved_provider = create_llm_provider()
    else:
        resolved_provider = provider

    all_chunks = list(chunks)

    if existing_run_id is None:
        request_json = json.dumps(
            {
                "max_chunks": max_chunks,
                "previous_context_chars": previous_context_chars,
                "next_context_chars": next_context_chars,
                "force_reanalyze": force_reanalyze,
            },
            ensure_ascii=False,
        )
        run_id = _create_analysis_run(
            project_id=normalized_project_id,
            provider=resolved_provider,
            database_path=database_path,
            status=ANALYSIS_STATUS_RUNNING,
            total_chunks=len(chunks),
            request_json=request_json,
        )
    else:
        run_id = existing_run_id
        pending_chunk_ids = _get_pending_chunk_database_ids(
            run_id=run_id,
            database_path=database_path,
        )
        chunks = [
            chunk
            for chunk in chunks
            if chunk.database_id in pending_chunk_ids
        ]

    processing_items = [
        (
            index,
            chunk,
        )
        for index, chunk in enumerate(all_chunks)
        if chunk in chunks
    ]

    successful_chunks = 0
    failed_chunks = 0
    partial_chunks = 0
    cached_chunks = 0
    cached_layers = 0
    run_error_message: str | None = None

    try:
        with database_session(database_path=database_path) as connection:
            connection.execute(
                """
                UPDATE narrative_analysis_runs
                SET
                    status = ?,
                    started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                    finished_at = NULL,
                    heartbeat_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    ANALYSIS_STATUS_RUNNING,
                    run_id,
                ),
            )

        for index, chunk in processing_items:
            text_hash = compute_text_hash(chunk.text)
            previous_context, next_context = _build_contexts(
                chunks=all_chunks,
                index=index,
                previous_context_chars=previous_context_chars,
                next_context_chars=next_context_chars,
            )
            analysis_input_hash = compute_analysis_input_hash(
                previous_context=previous_context,
                target_text=chunk.text,
                next_context=next_context,
            )

            try:
                _mark_unit_running(
                    run_id=run_id,
                    chunk=chunk,
                    database_path=database_path,
                )

                cached_unit = None

                if not force_reanalyze:
                    cached_unit = _find_cached_unit_analysis(
                        project_id=normalized_project_id,
                        analysis_input_hash=analysis_input_hash,
                        provider=resolved_provider,
                        database_path=database_path,
                    )

                if cached_unit is not None:
                    _save_unit_analysis(
                        run_id=run_id,
                        project_id=normalized_project_id,
                        chunk=chunk,
                        provider=resolved_provider,
                        status=ANALYSIS_STATUS_COMPLETED,
                        text_hash=text_hash,
                        analysis_input_hash=analysis_input_hash,
                        result_json=cached_unit["result_json"],
                        validated_result_json=(
                            cached_unit["validated_result_json"]
                        ),
                        cache_hit=True,
                        cache_source_unit_id=cached_unit["id"],
                        database_path=database_path,
                    )

                    successful_chunks += 1
                    cached_chunks += 1
                    continue

                # ── Layer 1: Mention Extraction ─────────────────────────────

                mention_input_hash = compute_json_hash(
                    {
                        "previous_context": previous_context,
                        "target_text": chunk.text,
                        "next_context": next_context,
                    }
                )
                cached_mentions = None

                if not force_reanalyze:
                    cached_mentions = _find_cached_layer_analysis(
                        project_id=normalized_project_id,
                        chunk=chunk,
                        layer_name="mentions",
                        input_hash=mention_input_hash,
                        provider=resolved_provider,
                        prompt_version=MENTION_PROMPT_VERSION,
                        schema_version=MENTION_SCHEMA_VERSION,
                        database_path=database_path,
                    )

                if cached_mentions is None:
                    mention_extraction = await _run_layer(
                        chunk_id=chunk.chunk_id,
                        layer_name="mentions",
                        provider=resolved_provider,
                        messages=build_mention_extraction_messages(
                            previous_context=previous_context,
                            target_text=chunk.text,
                            next_context=next_context,
                        ),
                        response_model=MentionExtractionOutput,
                    )
                    mention_raw_result = mention_extraction.model_dump(
                        mode="json"
                    )

                    validated_mentions = validate_evidence(
                        target_text=chunk.text,
                        extraction=mention_extraction,
                    )
                    validated_mentions = _filter_validated_mentions(
                        validated_mentions,
                        chunk_id=chunk.chunk_id,
                    )
                    mention_raw_result = _truncate_layer_items(
                        mention_raw_result,
                        field_name="mentions",
                        limit=MENTION_LIMIT,
                        layer_name="mentions",
                    )
                    validated_mentions = _truncate_layer_items(
                        validated_mentions,
                        field_name="mentions",
                        limit=MENTION_LIMIT,
                        layer_name="mentions",
                    )

                    _save_layer_analysis(
                        project_id=normalized_project_id,
                        chunk=chunk,
                        layer_name="mentions",
                        input_hash=mention_input_hash,
                        provider=resolved_provider,
                        prompt_version=MENTION_PROMPT_VERSION,
                        schema_version=MENTION_SCHEMA_VERSION,
                        result_json=json.dumps(
                            mention_raw_result,
                            ensure_ascii=False,
                        ),
                        validated_result_json=json.dumps(
                            validated_mentions,
                            ensure_ascii=False,
                        ),
                        database_path=database_path,
                    )
                else:
                    cached_layers += 1
                    _increment_run_cached_layers(
                        run_id=run_id,
                        database_path=database_path,
                    )
                    mention_raw_result = json.loads(
                        cached_mentions["result_json"]
                    )
                    validated_mentions = json.loads(
                        cached_mentions["validated_result_json"]
                    )

                mention_candidates = _build_mention_candidates(
                    validated_mentions["mentions"]
                )
                allowed_mention_ids = {
                    str(mention["mention_id"])
                    for mention in mention_candidates
                    if mention.get("mention_id")
                }
                chunk_warnings: list[str] = []
                chunk_status = ANALYSIS_STATUS_COMPLETED
                layer_statuses: dict[str, str] = {
                    "mentions": (
                        "cached"
                        if cached_mentions is not None
                        else ANALYSIS_STATUS_COMPLETED
                    ),
                    "relations": "pending",
                }

                # Build ID → mention map for downstream ID-text validation (P1-5).
                mention_by_id = {
                    str(mention["mention_id"]): mention
                    for mention in validated_mentions["mentions"]
                    if mention.get("mention_id")
                }

                # Skip subsequent layers if no mentions were extracted.
                if not allowed_mention_ids:
                    _logger.info(
                        "chunk=%s 无文本锚点，跳过关系/事件/人物候选层",
                        chunk.chunk_id,
                    )
                    layer_statuses["relations"] = "skipped"
                    validated_relations = {
                        "relations": [],
                        "warnings": [],
                    }
                    relation_raw_result = {
                        "relations": [],
                        "warnings": [],
                    }
                else:
                    # ── Layer 2: Relation Extraction ───────────────────────

                    relation_input_hash = compute_json_hash(
                        {
                            "target_text": chunk.text,
                            "mentions": mention_candidates,
                        }
                    )
                    cached_relations = None

                    if not force_reanalyze:
                        cached_relations = _find_cached_layer_analysis(
                            project_id=normalized_project_id,
                            chunk=chunk,
                            layer_name="relations",
                            input_hash=relation_input_hash,
                            provider=resolved_provider,
                            prompt_version=RELATION_PROMPT_VERSION,
                            schema_version=RELATION_SCHEMA_VERSION,
                            database_path=database_path,
                        )

                    if cached_relations is None:
                        try:
                            relation_extraction = await _run_layer(
                                chunk_id=chunk.chunk_id,
                                layer_name="relations",
                                provider=resolved_provider,
                                messages=build_relation_extraction_messages(
                                    target_text=chunk.text,
                                    mentions=mention_candidates,
                                ),
                                response_model=RelationExtractionOutput,
                            )
                            relation_raw_result = relation_extraction.model_dump(
                                mode="json"
                            )
                            validated_relations = validate_evidence(
                                target_text=chunk.text,
                                extraction=relation_extraction,
                            )
                            validated_relations = _filter_validated_relations(
                                validated_relations,
                                mention_by_id=mention_by_id,
                                allowed_mention_ids=allowed_mention_ids,
                                chunk_id=chunk.chunk_id,
                            )
                            relation_raw_result = _truncate_layer_items(
                                relation_raw_result,
                                field_name="relations",
                                limit=RELATION_LIMIT,
                                layer_name="relations",
                            )
                            validated_relations = _truncate_layer_items(
                                validated_relations,
                                field_name="relations",
                                limit=RELATION_LIMIT,
                                layer_name="relations",
                            )

                            _save_layer_analysis(
                                project_id=normalized_project_id,
                                chunk=chunk,
                                layer_name="relations",
                                input_hash=relation_input_hash,
                                provider=resolved_provider,
                                prompt_version=RELATION_PROMPT_VERSION,
                                schema_version=RELATION_SCHEMA_VERSION,
                                result_json=json.dumps(
                                    relation_raw_result,
                                    ensure_ascii=False,
                                ),
                                validated_result_json=json.dumps(
                                    validated_relations,
                                    ensure_ascii=False,
                                ),
                                database_path=database_path,
                            )
                            layer_statuses["relations"] = (
                                ANALYSIS_STATUS_COMPLETED
                            )
                        except Exception as error:
                            chunk_status = ANALYSIS_STATUS_PARTIAL
                            layer_statuses["relations"] = (
                                ANALYSIS_STATUS_FAILED
                            )
                            warning = _build_layer_timeout_warning(
                                chunk_id=chunk.chunk_id,
                                layer_name="relations",
                                error=error,
                            )
                            chunk_warnings.append(warning)
                            (
                                relation_raw_result,
                                validated_relations,
                            ) = _empty_layer_result(
                                field_name="relations",
                                warning=warning,
                            )
                    else:
                        cached_layers += 1
                        _increment_run_cached_layers(
                            run_id=run_id,
                            database_path=database_path,
                        )
                        layer_statuses["relations"] = "cached"
                        relation_raw_result = json.loads(
                            cached_relations["result_json"]
                        )
                        validated_relations = json.loads(
                            cached_relations["validated_result_json"]
                        )


                chunk_warnings.extend(
                    validated_mentions.get("warnings", [])
                    + validated_relations.get("warnings", [])
                )

                raw_result = {
                    "mentions": mention_raw_result.get("mentions", []),
                    "relations": relation_raw_result.get(
                        "relations",
                        [],
                    ),
                    "warnings": {
                        "mentions": mention_raw_result.get(
                            "warnings",
                            [],
                        ),
                        "relations": relation_raw_result.get(
                            "warnings",
                            [],
                        ),
                        "chunk_level": chunk_warnings,
                    },
                    "layer_versions": {
                        "mention_prompt_version": MENTION_PROMPT_VERSION,
                        "mention_schema_version": MENTION_SCHEMA_VERSION,
                        "relation_prompt_version": RELATION_PROMPT_VERSION,
                        "relation_schema_version": RELATION_SCHEMA_VERSION,
                    },
                    "layer_statuses": layer_statuses,
                }
                validated_result = {
                    "mentions": validated_mentions["mentions"],
                    "relations": validated_relations["relations"],
                    "warnings": {
                        "mentions": validated_mentions["warnings"],
                        "relations": validated_relations["warnings"],
                        "chunk_level": chunk_warnings,
                    },
                    "layer_versions": raw_result["layer_versions"],
                    "layer_statuses": layer_statuses,
                }

                _save_unit_analysis(
                    run_id=run_id,
                    project_id=normalized_project_id,
                    chunk=chunk,
                    provider=resolved_provider,
                    status=chunk_status,
                    text_hash=text_hash,
                    analysis_input_hash=analysis_input_hash,
                    result_json=json.dumps(
                        raw_result,
                        ensure_ascii=False,
                    ),
                    validated_result_json=json.dumps(
                        validated_result,
                        ensure_ascii=False,
                    ),
                    database_path=database_path,
                )

                if chunk_status == ANALYSIS_STATUS_COMPLETED:
                    successful_chunks += 1
                else:
                    successful_chunks += 1
                    partial_chunks += 1

            except Exception as error:
                failed_chunks += 1
                run_error_message = str(error)

                _save_unit_analysis(
                    run_id=run_id,
                    project_id=normalized_project_id,
                    chunk=chunk,
                    provider=resolved_provider,
                    status=ANALYSIS_STATUS_FAILED,
                    text_hash=text_hash,
                    analysis_input_hash=analysis_input_hash,
                    error_message=str(error),
                    database_path=database_path,
                )

        if existing_run_id is not None:
            return _finalize_analysis_run(
                run_id=run_id,
                database_path=database_path,
                error_message=run_error_message,
            )

        finalized_result = _finalize_analysis_run(
            run_id=run_id,
            database_path=database_path,
            error_message=run_error_message,
        )
        return NarrativeAnalysisResult(
            run_id=finalized_result.run_id,
            project_id=finalized_result.project_id,
            status=finalized_result.status,
            total_chunks=finalized_result.total_chunks,
            processed_chunks=finalized_result.processed_chunks,
            successful_chunks=successful_chunks,
            failed_chunks=failed_chunks,
            partial_chunks=partial_chunks,
            cached_chunks=cached_chunks,
            cached_layers=cached_layers,
        )

    finally:
        if owns_provider:
            await resolved_provider.close()


async def execute_narrative_analysis_job(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
    provider: LLMProvider | None = None,
) -> NarrativeAnalysisResult:
    project_id, request = _get_run_request(
        run_id=run_id,
        database_path=database_path,
    )

    with database_session(database_path=database_path) as connection:
        run_row = connection.execute(
            """
            SELECT status
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if run_row is None:
            raise LookupError(
                f"Narrative analysis run does not exist: {run_id}"
            )

        if run_row["status"] not in {
            ANALYSIS_STATUS_QUEUED,
            ANALYSIS_STATUS_RUNNING,
            ANALYSIS_STATUS_INTERRUPTED,
        }:
            return _summarize_run_from_connection(
                connection=connection,
                run_id=run_id,
            )

        connection.execute(
            """
            UPDATE narrative_unit_analyses
            SET
                status = 'pending',
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND status = 'running'
            """,
            (run_id,),
        )

    return await analyze_project_narrative(
        project_id=project_id,
        database_path=database_path,
        provider=provider,
        max_chunks=request.get("max_chunks"),
        previous_context_chars=int(
            request.get("previous_context_chars", 500)
        ),
        next_context_chars=int(request.get("next_context_chars", 0)),
        force_reanalyze=bool(request.get("force_reanalyze", False)),
        existing_run_id=run_id,
    )


def recover_analysis_jobs(
    *,
    database_path: DatabasePath | None = None,
) -> list[int]:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE narrative_unit_analyses
            SET
                status = 'pending',
                updated_at = CURRENT_TIMESTAMP
            WHERE status = 'running'
              AND run_id IN (
                  SELECT id
                  FROM narrative_analysis_runs
                  WHERE status = 'running'
              )
            """
        )
        connection.execute(
            """
            UPDATE narrative_analysis_runs
            SET
                status = ?,
                current_chunk_id = NULL,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE status = ?
            """,
            (
                ANALYSIS_STATUS_INTERRUPTED,
                ANALYSIS_STATUS_RUNNING,
            ),
        )
        rows = connection.execute(
            """
            SELECT id
            FROM narrative_analysis_runs
            WHERE status = ?
            ORDER BY id
            """,
            (ANALYSIS_STATUS_QUEUED,),
        ).fetchall()

    return [
        int(row["id"])
        for row in rows
    ]


def mark_narrative_analysis_job_failed(
    *,
    run_id: int,
    error_message: str,
    database_path: DatabasePath | None = None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE narrative_unit_analyses
            SET
                status = 'failed',
                error_message = COALESCE(error_message, ?),
                last_finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND status = 'running'
            """,
            (
                error_message,
                run_id,
            ),
        )
        _refresh_run_progress(
            connection=connection,
            run_id=run_id,
            current_chunk_id=None,
        )
        connection.execute(
            """
            UPDATE narrative_analysis_runs
            SET
                status = ?,
                error_message = ?,
                current_chunk_id = NULL,
                finished_at = CURRENT_TIMESTAMP,
                heartbeat_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND status IN (?, ?)
            """,
            (
                ANALYSIS_STATUS_FAILED,
                error_message,
                run_id,
                ANALYSIS_STATUS_QUEUED,
                ANALYSIS_STATUS_RUNNING,
            ),
        )


def resume_narrative_analysis_job(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
) -> NarrativeAnalysisResult:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT status
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            raise LookupError(
                f"Narrative analysis run does not exist: {run_id}"
            )

        if row["status"] != ANALYSIS_STATUS_INTERRUPTED:
            raise ValueError("Only interrupted analysis jobs can be resumed.")

        connection.execute(
            """
            UPDATE narrative_unit_analyses
            SET
                status = 'pending',
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND status = 'running'
            """,
            (run_id,),
        )
        connection.execute(
            """
            UPDATE narrative_analysis_runs
            SET
                status = ?,
                error_message = NULL,
                finished_at = NULL,
                current_chunk_id = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                ANALYSIS_STATUS_QUEUED,
                run_id,
            ),
        )
        _refresh_run_progress(
            connection=connection,
            run_id=run_id,
            current_chunk_id=None,
        )

        return _summarize_run_from_connection(
            connection=connection,
            run_id=run_id,
            status=ANALYSIS_STATUS_QUEUED,
        )


def retry_failed_narrative_analysis_job(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
) -> NarrativeAnalysisResult:
    with database_session(database_path=database_path) as connection:
        _get_run_project_id(
            connection=connection,
            run_id=run_id,
        )
        run_row = connection.execute(
            """
            SELECT status
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if run_row is None:
            raise LookupError(
                f"Narrative analysis run does not exist: {run_id}"
            )

        if run_row["status"] in {
            ANALYSIS_STATUS_QUEUED,
            ANALYSIS_STATUS_RUNNING,
        }:
            raise ValueError(
                "Running analysis jobs cannot retry failed chunks."
            )

        connection.execute(
            """
            UPDATE narrative_unit_analyses
            SET
                status = 'pending',
                result_json = NULL,
                validated_result_json = NULL,
                error_message = NULL,
                cache_hit = 0,
                cache_source_unit_id = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND status = 'failed'
            """,
            (run_id,),
        )
        connection.execute(
            """
            UPDATE narrative_analysis_runs
            SET
                status = ?,
                error_message = NULL,
                finished_at = NULL,
                current_chunk_id = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                ANALYSIS_STATUS_QUEUED,
                run_id,
            ),
        )
        _refresh_run_progress(
            connection=connection,
            run_id=run_id,
            current_chunk_id=None,
        )

        return _summarize_run_from_connection(
            connection=connection,
            run_id=run_id,
            status=ANALYSIS_STATUS_QUEUED,
        )


def get_active_narrative_analysis_run(
    *,
    project_id: str,
    database_path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    normalized_project_id = project_id.strip()

    if not normalized_project_id:
        raise ValueError("project_id cannot be empty.")

    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT id
            FROM narrative_analysis_runs
            WHERE project_id = ?
              AND status IN (?, ?, ?)
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                normalized_project_id,
                ANALYSIS_STATUS_QUEUED,
                ANALYSIS_STATUS_RUNNING,
                ANALYSIS_STATUS_INTERRUPTED,
            ),
        ).fetchone()

    if row is None:
        return None

    return get_narrative_analysis_run(
        run_id=int(row["id"]),
        database_path=database_path,
        include_units=False,
    )


def get_narrative_analysis_run(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
    include_units: bool = True,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        run_row = connection.execute(
            """
            SELECT
                id,
                project_id,
                provider,
                model,
                prompt_version,
                schema_version,
                status,
                error_message,
                total_chunks,
                processed_chunks,
                successful_chunks,
                partial_chunks,
                failed_chunks,
                cached_chunks,
                cached_layers,
                current_chunk_id,
                request_json,
                started_at,
                finished_at,
                heartbeat_at,
                created_at,
                updated_at
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if run_row is None:
            raise LookupError(
                f"Narrative analysis run does not exist: {run_id}"
            )

        unit_rows = []

        if include_units:
            unit_rows = connection.execute(
                """
                SELECT
                    id,
                    chunk_database_id,
                    chunk_id,
                    text_hash,
                    analysis_input_hash,
                    status,
                    cache_hit,
                    cache_source_unit_id,
                    result_json,
                    validated_result_json,
                    error_message,
                    attempt_count,
                    last_started_at,
                    last_finished_at,
                    created_at,
                    updated_at
                FROM narrative_unit_analyses
                WHERE run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()

        return {
            "id": run_row["id"],
            "project_id": run_row["project_id"],
            "provider": run_row["provider"],
            "model": run_row["model"],
            "prompt_version": run_row["prompt_version"],
            "schema_version": run_row["schema_version"],
            "status": run_row["status"],
            "error_message": run_row["error_message"],
            "total_chunks": run_row["total_chunks"],
            "processed_chunks": run_row["processed_chunks"],
            "successful_chunks": run_row["successful_chunks"],
            "partial_chunks": run_row["partial_chunks"],
            "failed_chunks": run_row["failed_chunks"],
            "cached_chunks": run_row["cached_chunks"],
            "cached_layers": run_row["cached_layers"],
            "current_chunk_id": run_row["current_chunk_id"],
            "request_json": run_row["request_json"],
            "started_at": run_row["started_at"],
            "finished_at": run_row["finished_at"],
            "heartbeat_at": run_row["heartbeat_at"],
            "created_at": run_row["created_at"],
            "updated_at": run_row["updated_at"],
            "units": [
                {
                    "id": row["id"],
                    "chunk_database_id": row["chunk_database_id"],
                    "chunk_id": row["chunk_id"],
                    "text_hash": row["text_hash"],
                    "analysis_input_hash": row["analysis_input_hash"],
                    "status": row["status"],
                    "cache_hit": bool(row["cache_hit"]),
                    "cache_source_unit_id": row[
                        "cache_source_unit_id"
                    ],
                    "result_json": row["result_json"],
                    "validated_result_json": row[
                        "validated_result_json"
                    ],
                    "error_message": row["error_message"],
                    "attempt_count": row["attempt_count"],
                    "last_started_at": row["last_started_at"],
                    "last_finished_at": row["last_finished_at"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in unit_rows
            ],
        }
