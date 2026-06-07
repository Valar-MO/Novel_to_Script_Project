from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from backend.llm.base import LLMCallMetadata, LLMProvider
from backend.llm.schemas import ScriptGenerationOutput
from backend.storage.database import DatabasePath, database_session


PROMPT_VERSION = "script_generation_v2"
SCHEMA_VERSION = "script_generation_schema_v1"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_CANCELLED = "cancelled"

UNIT_PENDING = "pending"
UNIT_RUNNING = "running"
UNIT_COMPLETED = "completed"
UNIT_PARTIAL = "partial"
UNIT_FAILED = "failed"
UNIT_CANCELLED = "cancelled"


class ActiveScriptGenerationError(RuntimeError):
    def __init__(self, active_run_id: int) -> None:
        super().__init__("该项目已有剧本生成任务正在执行。")
        self.active_run_id = active_run_id


@dataclass(frozen=True, slots=True)
class ScriptGenerationJobResult:
    run_id: int
    project_id: str
    status: str
    total_chunks: int
    processed_chunks: int = 0
    successful_chunks: int = 0
    partial_chunks: int = 0
    failed_chunks: int = 0
    scene_count: int = 0


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _hash_payload(value: Any) -> str:
    payload = _json_dumps(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _active_run_id(
    *,
    connection: sqlite3.Connection,
    project_id: str,
) -> int | None:
    row = connection.execute(
        """
        SELECT id
        FROM script_generation_runs
        WHERE project_id = ?
          AND status IN ('queued', 'running')
        ORDER BY id DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    return int(row["id"]) if row else None


def _latest_character_run_id(
    *,
    connection: sqlite3.Connection,
    project_id: str,
) -> int | None:
    row = connection.execute(
        """
        SELECT id
        FROM project_character_runs
        WHERE project_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    return int(row["id"]) if row else None


def _resolve_requested_chunk_ids(
    *,
    project_id: str,
    scope: str,
    chunk_ids: list[str] | None,
    database_path: DatabasePath | None,
) -> list[str] | None:
    normalized_scope = (scope or "all").strip().lower()

    if normalized_scope == "all":
        return None

    if normalized_scope == "selected":
        normalized_chunk_ids = [
            chunk_id.strip()
            for chunk_id in (chunk_ids or [])
            if chunk_id and chunk_id.strip()
        ]
        if not normalized_chunk_ids:
            raise ValueError(
                "chunk_ids is required when scope is selected."
            )
        return normalized_chunk_ids

    if normalized_scope == "pending":
        state = get_project_script_generation_state(
            project_id=project_id,
            database_path=database_path,
        )
        return list(state["pending_script_chunk_ids"])

    raise ValueError(f"Unsupported script generation scope: {scope}")


def _load_project_chunks(
    *,
    project_id: str,
    database_path: DatabasePath | None,
    chunk_ids: list[str] | None = None,
    max_chunks: int | None = None,
) -> list[sqlite3.Row]:
    if chunk_ids == []:
        return []

    with database_session(database_path=database_path) as connection:
        project = connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if project is None:
            raise LookupError(f"Project does not exist: {project_id}")

        params: list[Any] = [project_id]
        where_clause = "WHERE sf.project_id = ?"

        if chunk_ids is not None:
            placeholders = ",".join("?" for _ in chunk_ids)
            where_clause += f" AND tc.chunk_id IN ({placeholders})"
            params.extend(chunk_ids)

        limit_clause = ""
        if max_chunks is not None:
            limit_clause = "LIMIT ?"
            params.append(max_chunks)

        rows = connection.execute(
            f"""
            SELECT
                tc.id,
                tc.chunk_id,
                tc.global_order,
                tc.text,
                tc.start_character,
                tc.end_character,
                ch.full_title AS chapter_title
            FROM text_chunks tc
            JOIN source_files sf ON sf.id = tc.source_file_id
            JOIN chapters ch ON ch.id = tc.chapter_id
            {where_clause}
            ORDER BY tc.global_order
            {limit_clause}
            """,
            params,
        ).fetchall()

    return rows


def create_script_generation_job(
    *,
    project_id: str,
    provider: LLMProvider,
    database_path: DatabasePath | None = None,
    chunk_ids: list[str] | None = None,
    max_chunks: int | None = None,
    generation_style: str = "standard",
    adaptation_mode: str = "faithful",
    requested_provider: str | None = None,
    requested_model: str | None = None,
    thinking_enabled: bool = False,
    scope: str = "all",
) -> ScriptGenerationJobResult:
    normalized_project_id = project_id.strip()
    if not normalized_project_id:
        raise ValueError("project_id cannot be empty.")

    normalized_chunk_ids = _resolve_requested_chunk_ids(
        project_id=normalized_project_id,
        scope=scope,
        chunk_ids=chunk_ids,
        database_path=database_path,
    )
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0.")

    chunks = _load_project_chunks(
        project_id=normalized_project_id,
        database_path=database_path,
        chunk_ids=normalized_chunk_ids,
        max_chunks=max_chunks,
    )

    request_payload = {
        "chunk_ids": normalized_chunk_ids or [],
        "max_chunks": max_chunks,
        "generation_style": generation_style,
        "adaptation_mode": adaptation_mode,
        "provider": requested_provider or provider.provider_name,
        "model": requested_model or provider.model_name,
        "thinking_enabled": thinking_enabled,
        "scope": scope,
    }

    with database_session(database_path=database_path) as connection:
        active_run_id = _active_run_id(
            connection=connection,
            project_id=normalized_project_id,
        )
        if active_run_id is not None:
            raise ActiveScriptGenerationError(active_run_id)

        source_character_run_id = _latest_character_run_id(
            connection=connection,
            project_id=normalized_project_id,
        )

        try:
            cursor = connection.execute(
                """
                INSERT INTO script_generation_runs (
                    project_id,
                    source_character_run_id,
                    provider,
                    model,
                    prompt_version,
                    schema_version,
                    status,
                    total_chunks,
                    request_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_project_id,
                    source_character_run_id,
                    provider.provider_name,
                    provider.model_name,
                    PROMPT_VERSION,
                    SCHEMA_VERSION,
                    STATUS_QUEUED,
                    len(chunks),
                    _json_dumps(request_payload),
                ),
            )
        except sqlite3.IntegrityError as error:
            active_run_id = _active_run_id(
                connection=connection,
                project_id=normalized_project_id,
            )
            raise ActiveScriptGenerationError(
                active_run_id or 0
            ) from error

        run_id = int(cursor.lastrowid)

        for chunk in chunks:
            input_hash = _hash_payload(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source_character_run_id": source_character_run_id,
                    "provider": provider.provider_name,
                    "model": provider.model_name,
                    "prompt_version": PROMPT_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "generation_style": generation_style,
                    "adaptation_mode": adaptation_mode,
                    "thinking_enabled": thinking_enabled,
                }
            )
            connection.execute(
                """
                INSERT INTO script_generation_units (
                    run_id,
                    project_id,
                    chunk_database_id,
                    chunk_id,
                    chunk_order,
                    status,
                    input_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    normalized_project_id,
                    int(chunk["id"]),
                    chunk["chunk_id"],
                    int(chunk["global_order"]),
                    UNIT_PENDING,
                    input_hash,
                ),
            )

    return ScriptGenerationJobResult(
        run_id=run_id,
        project_id=normalized_project_id,
        status=STATUS_QUEUED,
        total_chunks=len(chunks),
    )


def _load_project_characters(
    *,
    connection: sqlite3.Connection,
    character_run_id: int | None,
) -> list[dict[str, Any]]:
    if character_run_id is None:
        return []

    rows = connection.execute(
        """
        SELECT character_id, canonical_name, aliases_json, references_json
        FROM project_characters
        WHERE character_run_id = ?
        ORDER BY id
        """,
        (character_run_id,),
    ).fetchall()

    return [
        {
            "id": row["character_id"],
            "canonical_name": row["canonical_name"],
            "aliases": _parse_json(row["aliases_json"], []),
            "references": _parse_json(row["references_json"], []),
        }
        for row in rows
    ]


def _load_previous_tail(
    *,
    connection: sqlite3.Connection,
    project_id: str,
    chunk_order: int,
    max_chars: int = 500,
) -> str:
    row = connection.execute(
        """
        SELECT tc.text
        FROM text_chunks tc
        JOIN source_files sf ON sf.id = tc.source_file_id
        WHERE sf.project_id = ?
          AND tc.global_order < ?
        ORDER BY tc.global_order DESC
        LIMIT 1
        """,
        (project_id, chunk_order),
    ).fetchone()

    if row is None:
        return ""

    text = str(row["text"])
    return text[-max_chars:]


def _load_previous_scene_context(
    *,
    connection: sqlite3.Connection,
    run_id: int,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT id, heading, characters_json, scene_summary
        FROM script_scenes
        WHERE run_id = ?
        ORDER BY scene_number DESC, id DESC
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()

    if row is None:
        return None

    return {
        "scene_id": row["id"],
        "heading": row["heading"],
        "characters": _parse_json(row["characters_json"], []),
        "ending_summary": row["scene_summary"],
    }


def _build_generation_messages(
    *,
    chunk_text: str,
    chunk_id: str,
    previous_tail: str,
    project_characters: list[dict[str, Any]],
    previous_scene_context: dict[str, Any] | None,
    generation_style: str,
    adaptation_mode: str,
) -> list[dict[str, str]]:
    system_content = (
        "你是可靠的中文小说剧本改编助手。"
        "请把当前小说文本改写成可拍摄、可编辑的剧本场景。"
        "忠实保留主要事实、人物身份、事件顺序和直接对白归属。"
        "不要添加改变剧情走向或人物性格的新对白。"
        "剧本正文只能包含观众能够直接看到或听到的内容。"
        "禁止直接描述人物的内心、认知和抽象判断，例如："
        "‘他知道’、‘他觉得’、‘他意识到’、‘他想起’、"
        "‘脑中闪过’、‘心里想着’。"
        "原文中的心理活动，只有在原文有充分依据时，"
        "才能转化为动作、表情、停顿、声音或对白；"
        "无法可靠转化时应压缩或省略，不得凭空增加行为。"
        "旁白可以压缩成动作、环境描写或必要的场面说明。"
        "动作描写与对白分行书写。"
        "对白使用明确的人物名称标识。"
        "不要使用小说式全知旁白。"
        "默认不要加入特写、主观镜头、切黑等导演指令。"
        "人物名称优先使用项目级人物表中的 canonical_name。"
        "前文上下文只用于连续性判断，不得重复改写。"
    )

    user_payload = {
        "chunk_id": chunk_id,
        "generation_style": generation_style,
        "adaptation_mode": adaptation_mode,
        "previous_context_for_continuity_only": previous_tail,
        "project_characters": project_characters,
        "previous_scene_context": previous_scene_context,
        "current_source_text_to_adapt": chunk_text,
        "output_rules": [
            "A chunk may produce zero, one, or multiple scenes.",
            "Use source_anchor.start_text and source_anchor.end_text copied from current_source_text_to_adapt.",
            "Do not return numeric source offsets.",
            "If unsure about a source range, choose the shortest clear source anchor and add a warning.",
            "script_text must contain only visible actions, audible sounds, dialogue, and filmable environment descriptions.",
            "Do not directly state thoughts, knowledge, memories, intentions, or emotions that cannot be observed.",
            "Convert internal narration into observable behavior only when supported by the source text.",
            "Write dialogue and action on separate lines.",
            "Do not use parenthetical narration such as （他渐渐入睡）.",
            "Return only JSON that matches the schema.",
        ],
    }

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": _json_dumps(user_payload),
        },
    ]


def _locate_source_anchor(
    *,
    chunk_text: str,
    start_text: str,
    end_text: str,
) -> tuple[int, int, str] | None:
    start_offset = chunk_text.find(start_text)
    if start_offset < 0:
        return None

    end_start = chunk_text.find(end_text, start_offset)
    if end_start < 0:
        return None

    end_offset = end_start + len(end_text)
    if end_offset <= start_offset:
        return None

    return (
        start_offset,
        end_offset,
        chunk_text[start_offset:end_offset],
    )


def _map_characters(
    *,
    generated_characters: list[dict[str, Any]],
    project_characters: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    lookup: dict[str, dict[str, Any]] = {}

    for character in project_characters:
        names = [
            character.get("canonical_name", ""),
            *character.get("aliases", []),
            *character.get("references", []),
        ]
        for name in names:
            normalized = str(name).strip().casefold()
            if normalized:
                lookup[normalized] = character

    mapped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for character in generated_characters:
        name = str(character.get("name") or "").strip()
        if not name:
            continue

        matched = lookup.get(name.casefold())
        if matched is not None:
            mapped_item = {
                "character_id": matched["id"],
                "name": matched["canonical_name"],
            }
        else:
            mapped_item = {
                "character_id": character.get("character_id"),
                "name": name,
            }
            warnings.append(f"unresolved_character:{name}")

        key = f"{mapped_item.get('character_id')}:{mapped_item['name']}"
        if key in seen:
            continue
        seen.add(key)
        mapped.append(mapped_item)

    return mapped, warnings


def _request_settings(run_row: sqlite3.Row) -> dict[str, Any]:
    request_json = _parse_json(run_row["request_json"], {})
    return {
        "generation_style": request_json.get("generation_style", "standard"),
        "adaptation_mode": request_json.get("adaptation_mode", "faithful"),
    }


def is_script_generation_cancel_requested(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
) -> bool:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            "SELECT cancel_requested_at FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    return bool(row and row["cancel_requested_at"])


def request_script_generation_cancel(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            "SELECT status, cancel_requested_at, cancelled_at FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"Script generation run does not exist: {run_id}")

        if row["status"] in {STATUS_COMPLETED, STATUS_PARTIAL, STATUS_FAILED, STATUS_CANCELLED}:
            return get_script_generation_run(
                run_id=run_id,
                database_path=database_path,
            )

        if row["cancel_requested_at"] is None:
            connection.execute(
                """
                UPDATE script_generation_runs
                SET cancel_requested_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (run_id,),
            )

    return get_script_generation_run(
        run_id=run_id,
        database_path=database_path,
    )


def finalize_script_generation_cancel(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    should_finish_run = False

    with database_session(database_path=database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE script_generation_units
            SET status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND status IN (?, ?)
            """,
            (UNIT_CANCELLED, run_id, UNIT_PENDING, UNIT_RUNNING),
        )

        if cursor.rowcount <= 0:
            should_finish_run = True
            connection.execute(
                """
                UPDATE script_generation_runs
                SET cancel_requested_at = NULL,
                    cancelled_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (run_id,),
            )
        else:
            connection.execute(
                """
                UPDATE script_generation_runs
                SET status = ?,
                    current_chunk_id = NULL,
                    finished_at = CURRENT_TIMESTAMP,
                    cancelled_at = COALESCE(cancelled_at, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (STATUS_CANCELLED, run_id),
            )

    _refresh_run_progress(
        run_id=run_id,
        database_path=database_path,
    )

    if should_finish_run:
        _finish_run(run_id=run_id, database_path=database_path)

    return get_script_generation_run(
        run_id=run_id,
        database_path=database_path,
    )


async def execute_script_generation_job(
    *,
    run_id: int,
    provider: LLMProvider,
    database_path: DatabasePath | None = None,
) -> None:
    with database_session(database_path=database_path) as connection:
        run_row = connection.execute(
            "SELECT * FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            raise LookupError(f"Script generation run does not exist: {run_id}")

        if run_row["status"] == STATUS_CANCELLED:
            return

        if run_row["status"] not in {
            STATUS_QUEUED,
            STATUS_RUNNING,
            STATUS_INTERRUPTED,
        }:
            return

        connection.execute(
            """
            UPDATE script_generation_runs
            SET status = ?,
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                heartbeat_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (STATUS_RUNNING, run_id),
        )

    while True:
        if is_script_generation_cancel_requested(
            run_id=run_id,
            database_path=database_path,
        ):
            finalize_script_generation_cancel(
                run_id=run_id,
                database_path=database_path,
            )
            return

        with database_session(database_path=database_path) as connection:
            unit = connection.execute(
                """
                SELECT *
                FROM script_generation_units
                WHERE run_id = ?
                  AND status IN (?, ?)
                ORDER BY chunk_order
                LIMIT 1
                """,
                (run_id, UNIT_PENDING, UNIT_RUNNING),
            ).fetchone()

            if unit is None:
                break

            run_row = connection.execute(
                "SELECT * FROM script_generation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            chunk = connection.execute(
                "SELECT * FROM text_chunks WHERE id = ?",
                (unit["chunk_database_id"],),
            ).fetchone()
            if chunk is None or run_row is None:
                raise LookupError("Script generation unit has invalid references.")

            project_characters = _load_project_characters(
                connection=connection,
                character_run_id=run_row["source_character_run_id"],
            )
            previous_tail = _load_previous_tail(
                connection=connection,
                project_id=run_row["project_id"],
                chunk_order=unit["chunk_order"],
            )
            previous_scene_context = _load_previous_scene_context(
                connection=connection,
                run_id=run_id,
            )
            request_settings = _request_settings(run_row)

            connection.execute(
                """
                UPDATE script_generation_units
                SET status = ?,
                    attempt_count = attempt_count + 1,
                    last_started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (UNIT_RUNNING, unit["id"]),
            )
            connection.execute(
                """
                UPDATE script_generation_runs
                SET current_chunk_id = ?,
                    heartbeat_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (unit["chunk_id"], run_id),
            )

        try:
            messages = _build_generation_messages(
                chunk_text=chunk["text"],
                chunk_id=unit["chunk_id"],
                previous_tail=previous_tail,
                project_characters=project_characters,
                previous_scene_context=previous_scene_context,
                generation_style=request_settings["generation_style"],
                adaptation_mode=request_settings["adaptation_mode"],
            )
            result = await provider.generate_structured(
                messages=messages,
                response_model=ScriptGenerationOutput,
                temperature=0.2,
                metadata=LLMCallMetadata(
                    chunk_id=unit["chunk_id"],
                    layer_name="script_generation",
                ),
            )
            _save_generation_result(
                run_id=run_id,
                unit_id=int(unit["id"]),
                project_id=run_row["project_id"],
                chunk_database_id=int(unit["chunk_database_id"]),
                chunk_id=unit["chunk_id"],
                chunk_text=chunk["text"],
                project_characters=project_characters,
                result=result,
                database_path=database_path,
            )
        except Exception as error:
            _mark_unit_failed(
                unit_id=int(unit["id"]),
                error_message=str(error),
                database_path=database_path,
            )

        _refresh_run_progress(
            run_id=run_id,
            database_path=database_path,
        )

        if is_script_generation_cancel_requested(
            run_id=run_id,
            database_path=database_path,
        ):
            finalize_script_generation_cancel(
                run_id=run_id,
                database_path=database_path,
            )
            return

    _finish_run(run_id=run_id, database_path=database_path)


def _save_generation_result(
    *,
    run_id: int,
    unit_id: int,
    project_id: str,
    chunk_database_id: int,
    chunk_id: str,
    chunk_text: str,
    project_characters: list[dict[str, Any]],
    result: ScriptGenerationOutput,
    database_path: DatabasePath | None,
) -> None:
    result_payload = result.model_dump(mode="json")
    unit_warnings = list(result.warnings)
    scene_statuses: list[str] = []

    with database_session(database_path=database_path) as connection:
        next_scene_number = int(
            connection.execute(
                """
                SELECT COALESCE(MAX(scene_number), 0) + 1 AS next_number
                FROM script_scenes
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()["next_number"]
        )

        if not result.scenes:
            unit_warnings.append("no_scenes_generated")

        for scene in result.scenes:
            scene_warnings = [
                *result.warnings,
                *scene.warnings,
            ]
            mapped_characters, character_warnings = _map_characters(
                generated_characters=[
                    item.model_dump(mode="json")
                    for item in scene.characters
                ],
                project_characters=project_characters,
            )
            scene_warnings.extend(character_warnings)

            source_span = _locate_source_anchor(
                chunk_text=chunk_text,
                start_text=scene.source_anchor.start_text,
                end_text=scene.source_anchor.end_text,
            )
            if source_span is None:
                scene_warnings.append("source_anchor_not_found")
                generation_status = UNIT_PARTIAL
            else:
                generation_status = "generated"

            cursor = connection.execute(
                """
                INSERT INTO script_scenes (
                    project_id,
                    run_id,
                    unit_id,
                    scene_number,
                    order_in_unit,
                    heading,
                    interior_exterior,
                    location,
                    time_of_day,
                    characters_json,
                    script_text,
                    scene_summary,
                    generation_status,
                    warnings_json,
                    adaptation_notes_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    run_id,
                    unit_id,
                    next_scene_number,
                    scene.order_in_unit,
                    scene.heading,
                    scene.interior_exterior,
                    scene.location,
                    scene.time_of_day,
                    _json_dumps(mapped_characters),
                    scene.script_text,
                    scene.scene_summary,
                    generation_status,
                    _json_dumps(scene_warnings),
                    _json_dumps(scene.adaptation_notes),
                ),
            )
            scene_id = int(cursor.lastrowid)
            next_scene_number += 1
            scene_statuses.append(generation_status)

            if source_span is not None:
                start_offset, end_offset, evidence_text = source_span
                connection.execute(
                    """
                    INSERT INTO script_scene_sources (
                        scene_id,
                        chunk_database_id,
                        chunk_id,
                        start_offset,
                        end_offset,
                        evidence_text
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scene_id,
                        chunk_database_id,
                        chunk_id,
                        start_offset,
                        end_offset,
                        evidence_text,
                    ),
                )

        if not result.scenes:
            unit_status = UNIT_PARTIAL
        elif any(status == UNIT_PARTIAL for status in scene_statuses):
            unit_status = UNIT_PARTIAL
        elif unit_warnings:
            unit_status = UNIT_PARTIAL
        else:
            unit_status = UNIT_COMPLETED

        connection.execute(
            """
            UPDATE script_generation_units
            SET status = ?,
                raw_response_json = ?,
                parsed_result_json = ?,
                warnings_json = ?,
                error_message = NULL,
                last_finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                unit_status,
                _json_dumps(result_payload),
                _json_dumps(result_payload),
                _json_dumps(unit_warnings),
                unit_id,
            ),
        )


def _mark_unit_failed(
    *,
    unit_id: int,
    error_message: str,
    database_path: DatabasePath | None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE script_generation_units
            SET status = ?,
                error_message = ?,
                last_finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (UNIT_FAILED, error_message, unit_id),
        )


def _refresh_run_progress(
    *,
    run_id: int,
    database_path: DatabasePath | None,
) -> None:
    with database_session(database_path=database_path) as connection:
        stats = connection.execute(
            """
            SELECT
                SUM(CASE WHEN status IN (?, ?, ?) THEN 1 ELSE 0 END)
                    AS processed_chunks,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END)
                    AS successful_chunks,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END)
                    AS partial_chunks,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END)
                    AS failed_chunks
            FROM script_generation_units
            WHERE run_id = ?
            """,
            (
                UNIT_COMPLETED,
                UNIT_PARTIAL,
                UNIT_FAILED,
                UNIT_COMPLETED,
                UNIT_PARTIAL,
                UNIT_FAILED,
                run_id,
            ),
        ).fetchone()
        scene_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM script_scenes WHERE run_id = ?",
                (run_id,),
            ).fetchone()["count"]
        )
        connection.execute(
            """
            UPDATE script_generation_runs
            SET processed_chunks = ?,
                successful_chunks = ?,
                partial_chunks = ?,
                failed_chunks = ?,
                scene_count = ?,
                heartbeat_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                int(stats["processed_chunks"] or 0),
                int(stats["successful_chunks"] or 0),
                int(stats["partial_chunks"] or 0),
                int(stats["failed_chunks"] or 0),
                scene_count,
                run_id,
            ),
        )


def _finish_run(
    *,
    run_id: int,
    database_path: DatabasePath | None,
) -> None:
    with database_session(database_path=database_path) as connection:
        run = connection.execute(
            "SELECT * FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            return

        total = int(run["total_chunks"])
        successful = int(run["successful_chunks"])
        partial = int(run["partial_chunks"])
        failed = int(run["failed_chunks"])
        cancelled = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM script_generation_units WHERE run_id = ? AND status = ?",
                (run_id, UNIT_CANCELLED),
            ).fetchone()["count"]
        )

        if total == 0:
            final_status = STATUS_COMPLETED
        elif successful == total:
            final_status = STATUS_COMPLETED
        elif cancelled > 0:
            final_status = STATUS_CANCELLED
        elif successful + partial > 0:
            final_status = STATUS_PARTIAL
        elif failed > 0:
            final_status = STATUS_FAILED
        else:
            final_status = STATUS_FAILED

        connection.execute(
            """
            UPDATE script_generation_runs
            SET status = ?,
                current_chunk_id = NULL,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (final_status, run_id),
        )


def mark_script_generation_job_failed(
    *,
    run_id: int,
    error_message: str,
    database_path: DatabasePath | None = None,
) -> None:
    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE script_generation_runs
            SET status = ?,
                error_message = ?,
                finished_at = CURRENT_TIMESTAMP,
                cancel_requested_at = NULL,
                cancelled_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (STATUS_FAILED, error_message, run_id),
        )


def recover_script_generation_jobs(
    *,
    database_path: DatabasePath | None = None,
) -> list[int]:
    with database_session(database_path=database_path) as connection:
        running_rows = connection.execute(
            """
            SELECT id
            FROM script_generation_runs
            WHERE status = ?
            """,
            (STATUS_RUNNING,),
        ).fetchall()
        connection.execute(
            """
            UPDATE script_generation_runs
            SET status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE status = ?
            """,
            (STATUS_INTERRUPTED, STATUS_RUNNING),
        )
        connection.execute(
            """
            UPDATE script_generation_units
            SET status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE status = ?
            """,
            (UNIT_PENDING, UNIT_RUNNING),
        )
        queued_rows = connection.execute(
            """
            SELECT id
            FROM script_generation_runs
            WHERE status = ?
               OR status = ?
            ORDER BY id
            """,
            (STATUS_QUEUED, STATUS_INTERRUPTED),
        ).fetchall()

    interrupted_ids = [int(row["id"]) for row in running_rows]
    del interrupted_ids
    return [int(row["id"]) for row in queued_rows]


def _row_to_unit(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "chunk_database_id": row["chunk_database_id"],
        "chunk_id": row["chunk_id"],
        "chunk_order": row["chunk_order"],
        "status": row["status"],
        "warnings": _parse_json(row["warnings_json"], []),
        "error_message": row["error_message"],
        "attempt_count": row["attempt_count"],
        "last_started_at": row["last_started_at"],
        "last_finished_at": row["last_finished_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_script_generation_run(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
    include_units: bool = True,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        run = connection.execute(
            "SELECT * FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise LookupError(f"Script generation run does not exist: {run_id}")

        units = []
        if include_units:
            units = [
                _row_to_unit(row)
                for row in connection.execute(
                    """
                    SELECT *
                    FROM script_generation_units
                    WHERE run_id = ?
                    ORDER BY chunk_order
                    """,
                    (run_id,),
                ).fetchall()
            ]

    return {
        "id": run["id"],
        "project_id": run["project_id"],
        "source_character_run_id": run["source_character_run_id"],
        "provider": run["provider"],
        "model": run["model"],
        "prompt_version": run["prompt_version"],
        "schema_version": run["schema_version"],
        "status": run["status"],
        "error_message": run["error_message"],
        "total_chunks": run["total_chunks"],
        "processed_chunks": run["processed_chunks"],
        "successful_chunks": run["successful_chunks"],
        "partial_chunks": run["partial_chunks"],
        "failed_chunks": run["failed_chunks"],
        "scene_count": run["scene_count"],
        "current_chunk_id": run["current_chunk_id"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
        "heartbeat_at": run["heartbeat_at"],
        "cancel_requested_at": run["cancel_requested_at"],
        "cancelled_at": run["cancelled_at"],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "units": units,
    }


def get_project_script_generation_state(
    *,
    project_id: str,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        project_row = connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if project_row is None:
            raise LookupError(f"Project does not exist: {project_id}")

        latest_run = get_latest_script_generation_run(
            project_id=project_id,
            database_path=database_path,
        )

        scene_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM script_scenes WHERE project_id = ?",
                (project_id,),
            ).fetchone()["count"]
        )

        pending_rows = connection.execute(
            """
            SELECT tc.chunk_id, tc.global_order
            FROM text_chunks tc
            JOIN source_files sf ON sf.id = tc.source_file_id
            WHERE sf.project_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM script_generation_units sgu
                  WHERE sgu.project_id = sf.project_id
                    AND sgu.chunk_database_id = tc.id
                    AND sgu.status IN (?, ?)
              )
            ORDER BY tc.global_order
            """,
            (project_id, UNIT_COMPLETED, UNIT_PARTIAL),
        ).fetchall()

        never_attempted_rows = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM text_chunks tc
            JOIN source_files sf ON sf.id = tc.source_file_id
            WHERE sf.project_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM script_generation_units sgu
                  WHERE sgu.project_id = sf.project_id
                    AND sgu.chunk_database_id = tc.id
              )
            """,
            (project_id,),
        ).fetchone()

        pending_chunk_ids = [row["chunk_id"] for row in pending_rows]
        never_attempted_count = int(never_attempted_rows["count"] or 0)
        retryable_count = max(
            len(pending_chunk_ids) - never_attempted_count,
            0,
        )

    suggested_action = "all_generated"
    if latest_run and latest_run["status"] in {STATUS_RUNNING, STATUS_QUEUED}:
        suggested_action = "running"
    elif latest_run and latest_run["cancel_requested_at"] and not latest_run["cancelled_at"]:
        suggested_action = "cancelling"
    elif pending_chunk_ids:
        if never_attempted_count and retryable_count:
            suggested_action = "continue_pending"
        elif never_attempted_count:
            suggested_action = (
                "generate_all" if scene_count == 0 else "continue_new"
            )
        else:
            suggested_action = "continue_remaining"
    elif scene_count == 0:
        suggested_action = "generate_all"

    return {
        "project_id": project_id,
        "has_generated_scenes": scene_count > 0,
        "latest_run": latest_run,
        "pending_script_chunk_count": len(pending_chunk_ids),
        "pending_script_chunk_ids": pending_chunk_ids,
        "never_attempted_chunk_count": never_attempted_count,
        "retryable_chunk_count": retryable_count,
        "suggested_action": suggested_action,
    }

def resume_script_generation_job(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
) -> ScriptGenerationJobResult:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            "SELECT * FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"Script generation run does not exist: {run_id}")

        if row["status"] != STATUS_CANCELLED:
            raise ValueError("Only cancelled script generation jobs can be resumed.")

        connection.execute(
            """
            UPDATE script_generation_units
            SET status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND status IN (?, ?)
            """,
            (UNIT_PENDING, run_id, UNIT_CANCELLED, UNIT_FAILED),
        )
        connection.execute(
            """
            UPDATE script_generation_runs
            SET status = ?,
                error_message = NULL,
                finished_at = NULL,
                current_chunk_id = NULL,
                cancel_requested_at = NULL,
                cancelled_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (STATUS_QUEUED, run_id),
        )

    refreshed = get_script_generation_run(
        run_id=run_id,
        database_path=database_path,
    )
    return ScriptGenerationJobResult(
        run_id=refreshed["id"],
        project_id=refreshed["project_id"],
        status=refreshed["status"],
        total_chunks=refreshed["total_chunks"],
        processed_chunks=refreshed["processed_chunks"],
        successful_chunks=refreshed["successful_chunks"],
        partial_chunks=refreshed["partial_chunks"],
        failed_chunks=refreshed["failed_chunks"],
        scene_count=refreshed["scene_count"],
    )


def get_latest_script_generation_run(
    *,
    project_id: str,
    database_path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT id
            FROM script_generation_runs
            WHERE project_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

    if row is None:
        return None

    return get_script_generation_run(
        run_id=int(row["id"]),
        database_path=database_path,
    )


def get_script_generation_scenes(
    *,
    run_id: int,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        run = connection.execute(
            "SELECT * FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise LookupError(f"Script generation run does not exist: {run_id}")

        scene_rows = connection.execute(
            """
            SELECT *
            FROM script_scenes
            WHERE run_id = ?
            ORDER BY scene_number, id
            """,
            (run_id,),
        ).fetchall()
        source_rows = connection.execute(
            """
            SELECT *
            FROM script_scene_sources
            WHERE scene_id IN (
                SELECT id FROM script_scenes WHERE run_id = ?
            )
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()

    sources_by_scene: dict[int, list[dict[str, Any]]] = {}
    for row in source_rows:
        sources_by_scene.setdefault(int(row["scene_id"]), []).append(
            {
                "id": row["id"],
                "chunk_database_id": row["chunk_database_id"],
                "chunk_id": row["chunk_id"],
                "start_offset": row["start_offset"],
                "end_offset": row["end_offset"],
                "evidence_text": row["evidence_text"],
            }
        )

    return {
        "run_id": run["id"],
        "project_id": run["project_id"],
        "status": run["status"],
        "scenes": [
            {
                "id": row["id"],
                "scene_number": row["scene_number"],
                "heading": row["heading"],
                "interior_exterior": row["interior_exterior"],
                "location": row["location"],
                "time_of_day": row["time_of_day"],
                "characters": _parse_json(row["characters_json"], []),
                "script_text": row["script_text"],
                "scene_summary": row["scene_summary"],
                "generation_status": row["generation_status"],
                "is_user_edited": bool(row["is_user_edited"]),
                "warnings": _parse_json(row["warnings_json"], []),
                "adaptation_notes": _parse_json(
                    row["adaptation_notes_json"],
                    [],
                ),
                "source_spans": sources_by_scene.get(int(row["id"]), []),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in scene_rows
        ],
    }


def _scene_to_dict(
    *,
    scene_row: sqlite3.Row,
    source_rows: list[sqlite3.Row],
) -> dict[str, Any]:
    return {
        "id": scene_row["id"],
        "scene_number": scene_row["scene_number"],
        "heading": scene_row["heading"],
        "interior_exterior": scene_row["interior_exterior"],
        "location": scene_row["location"],
        "time_of_day": scene_row["time_of_day"],
        "characters": _parse_json(scene_row["characters_json"], []),
        "script_text": scene_row["script_text"],
        "scene_summary": scene_row["scene_summary"],
        "generation_status": scene_row["generation_status"],
        "is_user_edited": bool(scene_row["is_user_edited"]),
        "warnings": _parse_json(scene_row["warnings_json"], []),
        "adaptation_notes": _parse_json(
            scene_row["adaptation_notes_json"],
            [],
        ),
        "source_spans": [
            {
                "id": row["id"],
                "chunk_database_id": row["chunk_database_id"],
                "chunk_id": row["chunk_id"],
                "start_offset": row["start_offset"],
                "end_offset": row["end_offset"],
                "evidence_text": row["evidence_text"],
            }
            for row in source_rows
        ],
        "created_at": scene_row["created_at"],
        "updated_at": scene_row["updated_at"],
    }


def get_script_scene(
    *,
    scene_id: int,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        scene_row = connection.execute(
            "SELECT * FROM script_scenes WHERE id = ?",
            (scene_id,),
        ).fetchone()
        if scene_row is None:
            raise LookupError(f"Script scene does not exist: {scene_id}")

        source_rows = connection.execute(
            """
            SELECT *
            FROM script_scene_sources
            WHERE scene_id = ?
            ORDER BY id
            """,
            (scene_id,),
        ).fetchall()

    return _scene_to_dict(
        scene_row=scene_row,
        source_rows=source_rows,
    )


def update_script_scene(
    *,
    scene_id: int,
    heading: str | None = None,
    interior_exterior: str | None = None,
    location: str | None = None,
    time_of_day: str | None = None,
    script_text: str | None = None,
    characters: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        existing = connection.execute(
            "SELECT id FROM script_scenes WHERE id = ?",
            (scene_id,),
        ).fetchone()
        if existing is None:
            raise LookupError(f"Script scene does not exist: {scene_id}")

        updates: list[str] = [
            "is_user_edited = 1",
            "updated_at = CURRENT_TIMESTAMP",
        ]
        params: list[Any] = []

        if heading is not None:
            normalized_heading = heading.strip()
            if not normalized_heading:
                raise ValueError("heading cannot be empty.")
            updates.append("heading = ?")
            params.append(normalized_heading)

        if interior_exterior is not None:
            normalized_value = interior_exterior.strip()
            if not normalized_value:
                raise ValueError("interior_exterior cannot be empty.")
            updates.append("interior_exterior = ?")
            params.append(normalized_value)

        if location is not None:
            normalized_value = location.strip()
            if not normalized_value:
                raise ValueError("location cannot be empty.")
            updates.append("location = ?")
            params.append(normalized_value)

        if time_of_day is not None:
            normalized_value = time_of_day.strip()
            if not normalized_value:
                raise ValueError("time_of_day cannot be empty.")
            updates.append("time_of_day = ?")
            params.append(normalized_value)

        if script_text is not None:
            normalized_script_text = script_text.strip()
            if not normalized_script_text:
                raise ValueError("script_text cannot be empty.")
            updates.append("script_text = ?")
            params.append(normalized_script_text)

        if characters is not None:
            updates.append("characters_json = ?")
            params.append(_json_dumps(characters))

        if warnings is not None:
            updates.append("warnings_json = ?")
            params.append(_json_dumps(warnings))

        params.append(scene_id)
        connection.execute(
            f"""
            UPDATE script_scenes
            SET {", ".join(updates)}
            WHERE id = ?
            """,
            params,
        )

    return get_script_scene(
        scene_id=scene_id,
        database_path=database_path,
    )


def _load_scene_regeneration_context(
    *,
    connection: sqlite3.Connection,
    scene_id: int,
) -> tuple[sqlite3.Row, sqlite3.Row, list[sqlite3.Row], str]:
    scene_row = connection.execute(
        "SELECT * FROM script_scenes WHERE id = ?",
        (scene_id,),
    ).fetchone()
    if scene_row is None:
        raise LookupError(f"Script scene does not exist: {scene_id}")

    run_row = connection.execute(
        "SELECT * FROM script_generation_runs WHERE id = ?",
        (scene_row["run_id"],),
    ).fetchone()
    if run_row is None:
        raise LookupError("Script scene has no generation run.")

    source_rows = connection.execute(
        """
        SELECT *
        FROM script_scene_sources
        WHERE scene_id = ?
        ORDER BY id
        """,
        (scene_id,),
    ).fetchall()
    if not source_rows:
        raise ValueError("Scene has no source span to regenerate from.")

    source_text_parts: list[str] = []
    for source in source_rows:
        chunk = connection.execute(
            "SELECT text FROM text_chunks WHERE id = ?",
            (source["chunk_database_id"],),
        ).fetchone()
        if chunk is None:
            continue
        start_offset = int(source["start_offset"])
        end_offset = int(source["end_offset"])
        source_text_parts.append(str(chunk["text"])[start_offset:end_offset])

    source_text = "\n\n".join(
        part
        for part in source_text_parts
        if part.strip()
    )
    if not source_text.strip():
        raise ValueError("Scene source text is empty.")

    return scene_row, run_row, source_rows, source_text


async def regenerate_script_scene(
    *,
    scene_id: int,
    provider: LLMProvider,
    instruction: str = "",
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        scene_row, run_row, source_rows, source_text = (
            _load_scene_regeneration_context(
                connection=connection,
                scene_id=scene_id,
            )
        )
        project_characters = _load_project_characters(
            connection=connection,
            character_run_id=run_row["source_character_run_id"],
        )

    system_content = (
        "你是可靠的中文小说剧本改编助手。"
        "请只根据提供的原文重新生成当前这一场。"
        "不要改变主要事实、人物身份、事件顺序或对白归属。"
        "剧本正文只能包含观众能够直接看到或听到的内容。"
        "禁止直接描述人物的内心、认知和抽象判断，例如："
        "‘他知道’、‘他觉得’、‘他意识到’、‘他想起’、"
        "‘脑中闪过’、‘心里想着’。"
        "原文中的心理活动，只有在原文有充分依据时，"
        "才能转化为动作、表情、停顿、声音或对白；"
        "无法可靠转化时应压缩或省略，不得凭空增加行为。"
        "动作描写与对白分行书写。"
        "对白使用明确的人物名称标识。"
        "不要使用小说式全知旁白。"
        "只返回 JSON。"
    )
    user_payload = {
        "original_scene": {
            "heading": scene_row["heading"],
            "script_text": scene_row["script_text"],
        },
        "source_text_to_regenerate": source_text,
        "project_characters": project_characters,
        "user_instruction": instruction.strip(),
        "output_rules": [
            "Return exactly one scene in scenes.",
            "source_anchor.start_text and end_text must be copied from source_text_to_regenerate.",
            "script_text must contain only visible actions, audible sounds, dialogue, and filmable environment descriptions.",
            "Do not directly state thoughts, knowledge, memories, intentions, or emotions that cannot be observed.",
            "Convert internal narration into observable behavior only when supported by the source text.",
            "Write dialogue and action on separate lines.",
            "Do not use parenthetical narration such as （他渐渐入睡）.",
        ],
    }

    result = await provider.generate_structured(
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": _json_dumps(user_payload),
            },
        ],
        response_model=ScriptGenerationOutput,
        temperature=0.2,
        metadata=LLMCallMetadata(
            chunk_id=str(source_rows[0]["chunk_id"]),
            layer_name="script_scene_regeneration",
        ),
    )

    if not result.scenes:
        raise ValueError("Regeneration returned no scene.")

    regenerated = result.scenes[0]
    mapped_characters, character_warnings = _map_characters(
        generated_characters=[
            item.model_dump(mode="json")
            for item in regenerated.characters
        ],
        project_characters=project_characters,
    )
    scene_warnings = [
        *result.warnings,
        *regenerated.warnings,
        *character_warnings,
    ]

    source_span = _locate_source_anchor(
        chunk_text=source_text,
        start_text=regenerated.source_anchor.start_text,
        end_text=regenerated.source_anchor.end_text,
    )
    generation_status = "generated"
    if source_span is None:
        scene_warnings.append("source_anchor_not_found")
        generation_status = UNIT_PARTIAL

    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE script_scenes
            SET heading = ?,
                interior_exterior = ?,
                location = ?,
                time_of_day = ?,
                characters_json = ?,
                script_text = ?,
                scene_summary = ?,
                generation_status = ?,
                is_user_edited = 0,
                warnings_json = ?,
                adaptation_notes_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                regenerated.heading,
                regenerated.interior_exterior,
                regenerated.location,
                regenerated.time_of_day,
                _json_dumps(mapped_characters),
                regenerated.script_text,
                regenerated.scene_summary,
                generation_status,
                _json_dumps(scene_warnings),
                _json_dumps(regenerated.adaptation_notes),
                scene_id,
            ),
        )

    return get_script_scene(
        scene_id=scene_id,
        database_path=database_path,
    )
