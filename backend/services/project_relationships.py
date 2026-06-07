from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from backend.services.project_characters import (
    DEFAULT_CORE_CHARACTER_LIMIT,
    build_project_characters,
    get_core_project_characters,
    get_latest_project_character_run,
    get_project_character_run,
)
from backend.storage.database import DatabasePath, database_session


CORE_CHARACTER_LIMIT = DEFAULT_CORE_CHARACTER_LIMIT
MIN_ONE_CORE_EVIDENCE_COUNT = 1


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _normalize_label(value: str) -> str:
    return " ".join(value.strip().split())


def _load_narrative_units(
    *,
    narrative_run_id: int,
    database_path: DatabasePath | None,
) -> tuple[str, list[dict[str, Any]]]:
    with database_session(database_path=database_path) as connection:
        run_row = connection.execute(
            """
            SELECT project_id, status
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (narrative_run_id,),
        ).fetchone()

        if run_row is None:
            raise LookupError(
                "Narrative analysis run does not exist: "
                f"{narrative_run_id}"
            )

        if run_row["status"] not in ("completed", "partial"):
            raise ValueError(
                "Narrative analysis run has not completed successfully."
            )

        rows = connection.execute(
            """
            SELECT id, chunk_id, validated_result_json
            FROM narrative_unit_analyses
            WHERE run_id = ?
              AND validated_result_json IS NOT NULL
            ORDER BY id
            """,
            (narrative_run_id,),
        ).fetchall()

    return (
        str(run_row["project_id"]),
        [dict(row) for row in rows],
    )


def _character_run_lookup(
    character_run: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    character_by_id: dict[str, dict[str, Any]] = {}
    mention_to_character_id: dict[str, str] = {}

    for character in character_run["characters"]:
        character_by_id[str(character["character_id"])] = character
        for mention_id in character.get("mention_ids") or []:
            mention_to_character_id[str(mention_id)] = str(
                character["character_id"]
            )

    return character_by_id, mention_to_character_id


def _relation_description(relation: dict[str, Any]) -> str:
    description = str(
        relation.get("relation_description")
        or relation.get("description")
        or ""
    ).strip()
    if description:
        return description
    return str(relation.get("evidence_text") or "").strip()


def _collect_ai_relationships(
    *,
    project_id: str,
    character_run: dict[str, Any],
    core_character_ids: set[str],
    database_path: DatabasePath | None,
) -> list[dict[str, Any]]:
    narrative_run_id = int(
        character_run["narrative_run_id"]
    )

    run_project_id, units = _load_narrative_units(
        narrative_run_id=narrative_run_id,
        database_path=database_path,
    )

    if run_project_id != project_id:
        raise ValueError(
            "The narrative analysis run does not belong to this project."
        )
    character_by_id, mention_to_character_id = _character_run_lookup(
        character_run
    )
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    chunk_ids_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for unit in units:
        result = _parse_json(unit.get("validated_result_json"), {})
        relations = result.get("relations")
        if not isinstance(relations, list):
            continue

        for relation in relations:
            if not isinstance(relation, dict):
                continue

            source_character_id = mention_to_character_id.get(
                str(relation.get("source_mention_id") or "")
            )
            target_character_id = mention_to_character_id.get(
                str(relation.get("target_mention_id") or "")
            )
            if (
                not source_character_id
                or not target_character_id
                or source_character_id == target_character_id
            ):
                continue

            source_character = character_by_id.get(source_character_id)
            target_character = character_by_id.get(target_character_id)
            if not source_character or not target_character:
                continue

            label = _normalize_label(
                str(
                    relation.get("relation_label")
                    or relation.get("relation")
                    or ""
                )
            )
            if not label:
                continue

            key = (
                source_character_id,
                target_character_id,
                label.casefold(),
            )
            chunk_ids_by_key[key].add(str(unit["chunk_id"]))

            current = grouped.get(key)
            if current is None:
                grouped[key] = {
                    "project_id": project_id,
                    "source_character_id": source_character_id,
                    "source_character_name": source_character["canonical_name"],
                    "target_character_id": target_character_id,
                    "target_character_name": target_character["canonical_name"],
                    "relation_label": label,
                    "relation_description": _relation_description(relation),
                    "source_type": "ai",
                    "is_user_edited": False,
                    "evidence_text": str(
                        relation.get("evidence_text") or ""
                    ).strip(),
                    "source_chunk_id": str(unit["chunk_id"]),
                    "start_offset": relation.get("start_offset"),
                    "end_offset": relation.get("end_offset"),
                    "evidence_count": 0,
                }

    kept: list[dict[str, Any]] = []
    for key, relationship in grouped.items():
        evidence_count = len(chunk_ids_by_key[key])
        relationship["evidence_count"] = evidence_count
        source_is_core = relationship["source_character_id"] in core_character_ids
        target_is_core = relationship["target_character_id"] in core_character_ids

        if source_is_core and target_is_core:
            kept.append(relationship)
        elif (
            (source_is_core or target_is_core)
            and evidence_count >= MIN_ONE_CORE_EVIDENCE_COUNT
        ):
            kept.append(relationship)

    return kept


def build_project_relationships(
    *,
    project_id: str,
    character_run_id: int | None = None,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    if character_run_id is not None:
        character_run = get_project_character_run(
            character_run_id=character_run_id,
            database_path=database_path,
        )
    else:
        character_run = get_latest_project_character_run(
            project_id=project_id,
            database_path=database_path,
        )

    if character_run is None:
        character_run = build_project_characters(
            project_id=project_id,
            database_path=database_path,
        )

    if str(character_run["project_id"]) != str(project_id):
        raise ValueError(
            "character_run_id does not belong to project_id."
        )

    core_characters = get_core_project_characters(
        project_id=project_id,
        character_run_id=int(character_run["id"]),
        limit=CORE_CHARACTER_LIMIT,
        database_path=database_path,
    )
    core_character_ids = {
        str(character["character_id"])
        for character in core_characters
    }
    ai_relationships = _collect_ai_relationships(
        project_id=project_id,
        character_run=character_run,
        core_character_ids=core_character_ids,
        database_path=database_path,
    )

    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            DELETE FROM project_relationships
            WHERE project_id = ?
              AND source_type = 'ai'
              AND is_user_edited = 0
            """,
            (project_id,),
        )

        for relationship in ai_relationships:
            connection.execute(
                """
                INSERT INTO project_relationships (
                    project_id,
                    source_character_id,
                    source_character_name,
                    target_character_id,
                    target_character_name,
                    relation_label,
                    relation_description,
                    source_type,
                    is_user_edited,
                    evidence_text,
                    source_chunk_id,
                    start_offset,
                    end_offset,
                    evidence_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relationship["project_id"],
                    relationship["source_character_id"],
                    relationship["source_character_name"],
                    relationship["target_character_id"],
                    relationship["target_character_name"],
                    relationship["relation_label"],
                    relationship["relation_description"],
                    relationship["source_type"],
                    1 if relationship["is_user_edited"] else 0,
                    relationship["evidence_text"],
                    relationship["source_chunk_id"],
                    relationship["start_offset"],
                    relationship["end_offset"],
                    relationship["evidence_count"],
                ),
            )

    return get_project_relationships(
        project_id=project_id,
        database_path=database_path,
    )


def _row_to_relationship(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "source_character_id": row["source_character_id"],
        "source_character_name": row["source_character_name"],
        "target_character_id": row["target_character_id"],
        "target_character_name": row["target_character_name"],
        "relation_label": row["relation_label"],
        "relation_description": row["relation_description"],
        "source_type": row["source_type"],
        "is_user_edited": bool(row["is_user_edited"]),
        "evidence_text": row["evidence_text"],
        "source_chunk_id": row["source_chunk_id"],
        "start_offset": row["start_offset"],
        "end_offset": row["end_offset"],
        "evidence_count": row["evidence_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_project_relationships(
    *,
    project_id: str,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    core_characters = get_core_project_characters(
        project_id=project_id,
        limit=CORE_CHARACTER_LIMIT,
        database_path=database_path,
    )
    core_character_ids = {
        str(character["character_id"])
        for character in core_characters
    }

    with database_session(database_path=database_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM project_relationships
            WHERE project_id = ?
            ORDER BY
                source_type DESC,
                evidence_count DESC,
                updated_at DESC,
                id DESC
            """,
            (project_id,),
        ).fetchall()

    relationships = [_row_to_relationship(row) for row in rows]
    core_relationships = [
        relationship
        for relationship in relationships
        if (
            relationship["source_type"] == "user"
            or (
                relationship["source_character_id"] in core_character_ids
                and relationship["target_character_id"] in core_character_ids
            )
            or (
                (
                    relationship["source_character_id"] in core_character_ids
                    or relationship["target_character_id"] in core_character_ids
                )
                and relationship["evidence_count"]
                >= MIN_ONE_CORE_EVIDENCE_COUNT
            )
        )
    ]

    return {
        "project_id": project_id,
        "core_characters": core_characters,
        "relationships": relationships,
        "core_relationships": core_relationships,
    }


def create_project_relationship(
    *,
    project_id: str,
    source_character_id: str,
    source_character_name: str,
    target_character_id: str,
    target_character_name: str,
    relation_label: str,
    relation_description: str = "",
    evidence_text: str = "",
    source_chunk_id: str | None = None,
    start_offset: int | None = None,
    end_offset: int | None = None,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    label = _normalize_label(relation_label)
    if not label:
        raise ValueError("relation_label cannot be empty.")

    with database_session(database_path=database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO project_relationships (
                project_id,
                source_character_id,
                source_character_name,
                target_character_id,
                target_character_name,
                relation_label,
                relation_description,
                source_type,
                is_user_edited,
                evidence_text,
                source_chunk_id,
                start_offset,
                end_offset,
                evidence_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'user', 1, ?, ?, ?, ?, 1)
            """,
            (
                project_id,
                source_character_id,
                source_character_name,
                target_character_id,
                target_character_name,
                label,
                relation_description.strip(),
                evidence_text.strip(),
                source_chunk_id,
                start_offset,
                end_offset,
            ),
        )
        relationship_id = int(cursor.lastrowid)

    return get_project_relationship(
        relationship_id=relationship_id,
        database_path=database_path,
    )


def get_project_relationship(
    *,
    relationship_id: int,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM project_relationships
            WHERE id = ?
            """,
            (relationship_id,),
        ).fetchone()

    if row is None:
        raise LookupError(
            f"Project relationship does not exist: {relationship_id}"
        )

    return _row_to_relationship(row)


def update_project_relationship(
    *,
    relationship_id: int,
    relation_label: str | None = None,
    relation_description: str | None = None,
    evidence_text: str | None = None,
    source_chunk_id: str | None = None,
    start_offset: int | None = None,
    end_offset: int | None = None,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    current = get_project_relationship(
        relationship_id=relationship_id,
        database_path=database_path,
    )

    label = (
        _normalize_label(relation_label)
        if relation_label is not None
        else current["relation_label"]
    )
    if not label:
        raise ValueError("relation_label cannot be empty.")

    with database_session(database_path=database_path) as connection:
        connection.execute(
            """
            UPDATE project_relationships
            SET relation_label = ?,
                relation_description = ?,
                evidence_text = ?,
                source_chunk_id = ?,
                start_offset = ?,
                end_offset = ?,
                source_type = 'user',
                is_user_edited = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                label,
                (
                    relation_description.strip()
                    if relation_description is not None
                    else current["relation_description"]
                ),
                (
                    evidence_text.strip()
                    if evidence_text is not None
                    else current["evidence_text"]
                ),
                source_chunk_id
                if source_chunk_id is not None
                else current["source_chunk_id"],
                start_offset
                if start_offset is not None
                else current["start_offset"],
                end_offset
                if end_offset is not None
                else current["end_offset"],
                relationship_id,
            ),
        )

    return get_project_relationship(
        relationship_id=relationship_id,
        database_path=database_path,
    )


def delete_project_relationship(
    *,
    relationship_id: int,
    database_path: DatabasePath | None = None,
) -> None:
    with database_session(database_path=database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM project_relationships WHERE id = ?",
            (relationship_id,),
        )

    if cursor.rowcount == 0:
        raise LookupError(
            f"Project relationship does not exist: {relationship_id}"
        )


def load_relevant_project_relationships(
    *,
    project_id: str,
    character_names: list[str],
    limit: int = 15,
    database_path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    if not character_names:
        return []

    normalized_names = {
        name.strip().casefold()
        for name in character_names
        if name and name.strip()
    }
    if not normalized_names:
        return []

    with database_session(database_path=database_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM project_relationships
            WHERE project_id = ?
            ORDER BY
                source_type DESC,
                is_user_edited DESC,
                evidence_count DESC,
                updated_at DESC
            """,
            (project_id,),
        ).fetchall()

    matched: list[dict[str, Any]] = []
    for row in rows:
        relationship = _row_to_relationship(row)
        names = {
            relationship["source_character_name"].strip().casefold(),
            relationship["target_character_name"].strip().casefold(),
        }
        if names & normalized_names:
            matched.append(relationship)
        if len(matched) >= limit:
            break

    return matched
