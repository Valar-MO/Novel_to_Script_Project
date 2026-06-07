import hashlib
import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from backend.storage.database import DatabasePath, database_session


GENERIC_REFERENCE_NAMES = {
    "他",
    "她",
    "它",
    "他们",
    "她们",
    "母亲",
    "父亲",
    "爹",
    "娘",
    "三叔",
    "叔叔",
    "老人",
    "老者",
    "少年",
    "少女",
    "掌柜",
    "医生",
    "先生",
    "夫人",
}

REFERENCE_ONLY_NAMES = {
    "他",
    "她",
    "它",
    "他们",
    "她们",
}

MUST_LINK_RELATIONS = {
    "alias",
    "nickname",
    "called",
    "别名",
    "小名",
    "称作",
    "叫作",
    "叫做",
    "又叫",
    "也就是",
    "化名",
}

CANNOT_LINK_RELATIONS = {
    "brother",
    "sibling",
    "father",
    "mother",
    "parent",
    "child",
    "mentor",
    "teacher",
    "student",
    "enemy",
    "spouse",
    "父子",
    "母子",
    "父女",
    "母女",
    "兄弟",
    "姐妹",
    "姐弟",
    "兄妹",
    "师徒",
    "夫妻",
    "敌对",
    "叔侄",
    "亲属",
    "主仆",
    "同伴",
}

AUTO_MERGE_THRESHOLD = 0.80
AMBIGUOUS_THRESHOLD = 0.40


@dataclass(slots=True)
class CharacterCandidate:
    candidate_id: str
    project_id: str
    narrative_run_id: int
    source_unit_id: int
    chunk_id: str
    chunk_order: int
    canonical_name: str
    aliases: list[str]
    references: list[str]
    mention_ids: list[str]
    mention_texts: list[str]
    source_type: str
    model_reported_confidence: float | None


@dataclass(slots=True)
class InputGap:
    source_unit_id: int | None
    chunk_id: str | None
    unit_status: str | None
    layer_name: str | None
    reason: str


@dataclass(slots=True)
class MergeDecision:
    left_candidate_id: str
    right_candidate_id: str
    decision: str
    merge_score: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {
            item: item
            for item in items
        }

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        self.parent[right_root] = left_root


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", name.strip()).casefold()


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    return result


def _json_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _stable_project_character_id(
    *,
    source_candidate_ids: list[str],
    canonical_name: str,
) -> str:
    return f"pc_{_json_hash({'candidates': source_candidate_ids, 'name': canonical_name})}"


def _is_generic_name(name: str) -> bool:
    return _normalize_name(name) in {
        _normalize_name(item)
        for item in GENERIC_REFERENCE_NAMES
    }


def _parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _latest_narrative_run_id(
    *,
    project_id: str,
    database_path: DatabasePath | None,
) -> int:
    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            """
            SELECT id
            FROM narrative_analysis_runs
            WHERE project_id = ?
              AND status IN ('completed', 'partial', 'failed')
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

    if row is None:
        raise LookupError(
            f"No finished narrative analysis run exists for project: {project_id}"
        )

    return int(row["id"])


def _load_narrative_units(
    *,
    narrative_run_id: int,
    database_path: DatabasePath | None,
) -> tuple[str, list[dict[str, Any]]]:
    with database_session(database_path=database_path) as connection:
        run_row = connection.execute(
            """
            SELECT project_id
            FROM narrative_analysis_runs
            WHERE id = ?
            """,
            (narrative_run_id,),
        ).fetchone()

        if run_row is None:
            raise LookupError(
                f"Narrative analysis run does not exist: {narrative_run_id}"
            )

        rows = connection.execute(
            """
            SELECT
                nua.id,
                nua.project_id,
                nua.chunk_database_id,
                nua.chunk_id,
                nua.status,
                nua.validated_result_json,
                nua.error_message,
                tc.global_order
            FROM narrative_unit_analyses AS nua
            LEFT JOIN text_chunks AS tc
                ON tc.id = nua.chunk_database_id
            WHERE nua.run_id = ?
            ORDER BY
                COALESCE(tc.global_order, nua.id),
                nua.id
            """,
            (narrative_run_id,),
        ).fetchall()

    return str(run_row["project_id"]), [
        dict(row)
        for row in rows
    ]


def _candidate_from_mention(
    *,
    project_id: str,
    narrative_run_id: int,
    unit: dict[str, Any],
    mention: dict[str, Any],
) -> CharacterCandidate | None:
    mention_id = str(mention.get("mention_id") or "").strip()
    mention_text = str(mention.get("mention_text") or "").strip()

    if not mention_id or not mention_text:
        return None

    return CharacterCandidate(
        candidate_id=f"{mention_id}_fallback_candidate",
        project_id=project_id,
        narrative_run_id=narrative_run_id,
        source_unit_id=int(unit["id"]),
        chunk_id=str(unit["chunk_id"]),
        chunk_order=int(unit.get("global_order") or 0),
        canonical_name=mention_text,
        aliases=[] if _is_generic_name(mention_text) else [mention_text],
        references=[mention_text] if _is_generic_name(mention_text) else [],
        mention_ids=[mention_id],
        mention_texts=[mention_text],
        source_type="mention_fallback",
        model_reported_confidence=None,
    )


def _collect_inputs(
    *,
    project_id: str,
    narrative_run_id: int,
    units: list[dict[str, Any]],
) -> tuple[
    list[CharacterCandidate],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[InputGap],
    int,
]:
    candidates: list[CharacterCandidate] = []
    relations: list[dict[str, Any]] = []
    event_frames: list[dict[str, Any]] = []
    gaps: list[InputGap] = []
    used_units = 0
    seen_candidate_ids: set[str] = set()

    for unit in units:
        unit_status = str(unit.get("status") or "")
        if unit_status == "failed":
            gaps.append(
                InputGap(
                    source_unit_id=int(unit["id"]),
                    chunk_id=str(unit["chunk_id"]),
                    unit_status=unit_status,
                    layer_name=None,
                    reason=unit.get("error_message") or "unit_failed",
                )
            )
            continue

        result = _parse_json_object(unit.get("validated_result_json"))
        if not result:
            gaps.append(
                InputGap(
                    source_unit_id=int(unit["id"]),
                    chunk_id=str(unit["chunk_id"]),
                    unit_status=unit_status,
                    layer_name=None,
                    reason="validated_result_unavailable",
                )
            )
            continue

        used_units += 1
        layer_statuses = result.get("layer_statuses") or {}

        if isinstance(layer_statuses, dict):
            for layer_name, layer_status in layer_statuses.items():
                if layer_status == "failed":
                    gaps.append(
                        InputGap(
                            source_unit_id=int(unit["id"]),
                            chunk_id=str(unit["chunk_id"]),
                            unit_status=unit_status,
                            layer_name=str(layer_name),
                            reason="layer_failed",
                        )
                    )

        mentions = [
            item
            for item in result.get("mentions", [])
            if (
                isinstance(item, dict)
                and item.get("mention_type") == "character"
                and item.get("evidence_validated", True)
            )
        ]
        mention_by_id = {
            str(item.get("mention_id")): item
            for item in mentions
            if item.get("mention_id")
        }

        local_candidates = result.get("character_candidates")
        if not isinstance(local_candidates, list):
            local_candidates = []

        for index, item in enumerate(local_candidates):
            if not isinstance(item, dict):
                continue

            candidate_id = str(
                item.get("character_candidate_id")
                or f"{unit['chunk_id']}_candidate_{index}"
            )
            if candidate_id in seen_candidate_ids:
                continue

            mention_ids = [
                str(mention_id)
                for mention_id in item.get("mention_ids", [])
                if str(mention_id) in mention_by_id
            ]
            if not mention_ids:
                gaps.append(
                    InputGap(
                        source_unit_id=int(unit["id"]),
                        chunk_id=str(unit["chunk_id"]),
                        unit_status=unit_status,
                        layer_name="character_candidates",
                        reason="candidate_without_valid_mentions",
                    )
                )
                continue

            canonical_name = str(
                item.get("canonical_name")
                or mention_by_id[mention_ids[0]].get("mention_text")
                or ""
            ).strip()
            if not canonical_name:
                continue

            aliases = _unique(
                [
                    str(value)
                    for value in item.get("aliases", [])
                ]
            )
            references = _unique(
                [
                    str(value)
                    for value in item.get("references", [])
                ]
            )

            for mention_id in mention_ids:
                mention_text = str(
                    mention_by_id[mention_id].get("mention_text") or ""
                )
                if _is_generic_name(mention_text):
                    references.append(mention_text)
                else:
                    aliases.append(mention_text)

            seen_candidate_ids.add(candidate_id)
            candidates.append(
                CharacterCandidate(
                    candidate_id=candidate_id,
                    project_id=project_id,
                    narrative_run_id=narrative_run_id,
                    source_unit_id=int(unit["id"]),
                    chunk_id=str(unit["chunk_id"]),
                    chunk_order=int(unit.get("global_order") or 0),
                    canonical_name=canonical_name,
                    aliases=_unique(aliases),
                    references=_unique(references),
                    mention_ids=_unique(mention_ids),
                    mention_texts=_unique(
                        [
                            str(mention_by_id[mid].get("mention_text") or "")
                            for mid in mention_ids
                        ]
                    ),
                    source_type="character_candidate",
                    model_reported_confidence=(
                        float(item["confidence"])
                        if item.get("confidence") is not None
                        else None
                    ),
                )
            )

        covered_mentions = {
            mention_id
            for candidate in candidates
            if candidate.source_unit_id == int(unit["id"])
            for mention_id in candidate.mention_ids
        }
        for mention in mentions:
            if str(mention.get("mention_id")) in covered_mentions:
                continue
            fallback = _candidate_from_mention(
                project_id=project_id,
                narrative_run_id=narrative_run_id,
                unit=unit,
                mention=mention,
            )
            if fallback and fallback.candidate_id not in seen_candidate_ids:
                seen_candidate_ids.add(fallback.candidate_id)
                candidates.append(fallback)

        if isinstance(result.get("relations"), list):
            relations.extend(
                [
                    {
                        **relation,
                        "_source_unit_id": int(unit["id"]),
                        "_chunk_id": str(unit["chunk_id"]),
                    }
                    for relation in result.get("relations", [])
                    if isinstance(relation, dict)
                ]
            )
        else:
            gaps.append(
                InputGap(
                    source_unit_id=int(unit["id"]),
                    chunk_id=str(unit["chunk_id"]),
                    unit_status=unit_status,
                    layer_name="relations",
                    reason="relations_unavailable",
                )
            )

        if isinstance(result.get("event_frames"), list):
            event_frames.extend(
                [
                    {
                        **event_frame,
                        "_source_unit_id": int(unit["id"]),
                        "_chunk_id": str(unit["chunk_id"]),
                    }
                    for event_frame in result.get("event_frames", [])
                    if isinstance(event_frame, dict)
                ]
            )
        else:
            gaps.append(
                InputGap(
                    source_unit_id=int(unit["id"]),
                    chunk_id=str(unit["chunk_id"]),
                    unit_status=unit_status,
                    layer_name="event_frames",
                    reason="event_frames_unavailable",
                )
            )

    return candidates, relations, event_frames, gaps, used_units


def _build_mention_to_candidate(
    candidates: list[CharacterCandidate],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in candidates:
        for mention_id in candidate.mention_ids:
            result.setdefault(mention_id, candidate.candidate_id)
    return result


def _candidate_names(candidate: CharacterCandidate) -> set[str]:
    return {
        _normalize_name(value)
        for value in [
            candidate.canonical_name,
            *candidate.aliases,
            *candidate.references,
            *candidate.mention_texts,
        ]
        if value
    }


def _valid_alias_names(candidate: CharacterCandidate) -> set[str]:
    return {
        _normalize_name(value)
        for value in [
            candidate.canonical_name,
            *candidate.aliases,
            *candidate.mention_texts,
        ]
        if value and not _is_generic_name(value)
    }


def _is_only_generic(candidate: CharacterCandidate) -> bool:
    names = _candidate_names(candidate)
    return bool(names) and all(
        name in {
            _normalize_name(item)
            for item in GENERIC_REFERENCE_NAMES
        }
        for name in names
    )


def _score_pair(
    left: CharacterCandidate,
    right: CharacterCandidate,
    *,
    same_local_candidate: bool = False,
) -> MergeDecision:
    evidence: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    score = 0.0

    left_name = _normalize_name(left.canonical_name)
    right_name = _normalize_name(right.canonical_name)

    if left_name and left_name == right_name:
        if _is_generic_name(left.canonical_name):
            conflicts.append(
                {
                    "type": "generic_role_name",
                    "value": left.canonical_name,
                }
            )
            score -= 0.40
        else:
            evidence.append(
                {
                    "type": "exact_normalized_name",
                    "value": left.canonical_name,
                    "weight": 0.60,
                }
            )
            score += 0.60

    left_aliases = {
        _normalize_name(alias)
        for alias in left.aliases
        if alias and not _is_generic_name(alias)
    }
    right_aliases = {
        _normalize_name(alias)
        for alias in right.aliases
        if alias and not _is_generic_name(alias)
    }
    shared_aliases = sorted(left_aliases & right_aliases)
    if shared_aliases:
        evidence.append(
            {
                "type": "shared_alias",
                "value": shared_aliases[0],
                "weight": 0.45,
            }
        )
        score += 0.45

    left_valid_names = _valid_alias_names(left)
    right_valid_names = _valid_alias_names(right)
    if (
        left_name
        and right_name
        and left_name in right_valid_names
        and right_name in left_valid_names
        and not _is_generic_name(left.canonical_name)
        and not _is_generic_name(right.canonical_name)
    ):
        evidence.append(
            {
                "type": "reciprocal_alias",
                "weight": 0.90,
            }
        )
        score = max(score, 0.90)

    if same_local_candidate:
        evidence.append(
            {
                "type": "same_local_candidate",
                "weight": 0.25,
            }
        )
        score += 0.25

    if abs(left.chunk_order - right.chunk_order) == 1 and left_name == right_name:
        if not _is_generic_name(left.canonical_name):
            evidence.append(
                {
                    "type": "adjacent_chunk_same_name",
                    "weight": 0.10,
                }
            )
            score += 0.10

    if _is_only_generic(left) or _is_only_generic(right):
        conflicts.append(
            {
                "type": "only_generic_reference",
                "weight": -0.40,
            }
        )
        score -= 0.40

    score = max(0.0, min(score, 1.0))

    if score >= AUTO_MERGE_THRESHOLD:
        decision = "merged"
    elif score >= AMBIGUOUS_THRESHOLD:
        decision = "ambiguous"
    else:
        decision = "separate"

    return MergeDecision(
        left_candidate_id=left.candidate_id,
        right_candidate_id=right.candidate_id,
        decision=decision,
        merge_score=round(score, 3),
        evidence=evidence,
        conflicts=conflicts,
    )


def _should_keep_decision(
    decision: MergeDecision,
) -> bool:
    if decision.decision in {
        "merged",
        "must_link",
        "cannot_link",
        "ambiguous",
    }:
        return True

    return bool(decision.evidence or decision.conflicts)


def _constraint_decisions(
    *,
    candidates: list[CharacterCandidate],
    relations: list[dict[str, Any]],
    event_frames: list[dict[str, Any]],
) -> tuple[list[MergeDecision], set[tuple[str, str]], set[tuple[str, str]]]:
    mention_to_candidate = _build_mention_to_candidate(candidates)
    must_links: set[tuple[str, str]] = set()
    cannot_links: set[tuple[str, str]] = set()
    decisions: list[MergeDecision] = []

    def ordered(left: str, right: str) -> tuple[str, str] | None:
        if not left or not right or left == right:
            return None
        return tuple(sorted((left, right)))

    for relation in relations:
        left = mention_to_candidate.get(
            str(relation.get("source_mention_id") or "")
        )
        right = mention_to_candidate.get(
            str(relation.get("target_mention_id") or "")
        )
        pair = ordered(left or "", right or "")
        if pair is None:
            continue

        relation_text = str(relation.get("relation") or "").strip()
        normalized_relation = _normalize_name(relation_text)
        evidence_item = {
            "type": "relation_constraint",
            "relation": relation_text,
            "evidence_text": relation.get("evidence_text"),
            "source_unit_id": relation.get("_source_unit_id"),
        }

        if normalized_relation in {
            _normalize_name(item)
            for item in MUST_LINK_RELATIONS
        }:
            must_links.add(pair)
            decisions.append(
                MergeDecision(
                    left_candidate_id=pair[0],
                    right_candidate_id=pair[1],
                    decision="must_link",
                    merge_score=1.0,
                    evidence=[evidence_item],
                    conflicts=[],
                )
            )
        elif normalized_relation in {
            _normalize_name(item)
            for item in CANNOT_LINK_RELATIONS
        }:
            cannot_links.add(pair)
            decisions.append(
                MergeDecision(
                    left_candidate_id=pair[0],
                    right_candidate_id=pair[1],
                    decision="cannot_link",
                    merge_score=0.0,
                    evidence=[],
                    conflicts=[evidence_item],
                )
            )

    for event_frame in event_frames:
        argument_candidate_ids = _unique(
            [
                mention_to_candidate.get(
                    str(argument.get("mention_id") or "")
                )
                or ""
                for argument in event_frame.get("arguments", [])
                if isinstance(argument, dict)
            ]
        )
        for left, right in combinations(argument_candidate_ids, 2):
            pair = ordered(left, right)
            if pair is None:
                continue
            cannot_links.add(pair)
            decisions.append(
                MergeDecision(
                    left_candidate_id=pair[0],
                    right_candidate_id=pair[1],
                    decision="cannot_link",
                    merge_score=0.0,
                    evidence=[],
                    conflicts=[
                        {
                            "type": "same_event_distinct_arguments",
                            "event_frame_id": event_frame.get(
                                "event_frame_id"
                            ),
                            "trigger_text": event_frame.get("trigger_text"),
                        }
                    ],
                )
            )

    return decisions, must_links, cannot_links


def _build_characters(
    *,
    candidates: list[CharacterCandidate],
    decisions: list[MergeDecision],
    must_links: set[tuple[str, str]],
    cannot_links: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[MergeDecision]]:
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }
    uf = _UnionFind(list(candidate_by_id))
    extra_decisions: list[MergeDecision] = []
    decided_pairs = {
        tuple(sorted((decision.left_candidate_id, decision.right_candidate_id)))
        for decision in decisions
    }

    for left_id, right_id in must_links:
        if (left_id, right_id) not in cannot_links:
            uf.union(left_id, right_id)

    for left, right in combinations(candidates, 2):
        pair = tuple(sorted((left.candidate_id, right.candidate_id)))
        if pair in decided_pairs:
            continue
        decision = _score_pair(left, right)
        if _should_keep_decision(decision):
            extra_decisions.append(decision)
        if decision.decision == "merged" and pair not in cannot_links:
            uf.union(left.candidate_id, right.candidate_id)

    groups: dict[str, list[CharacterCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(uf.find(candidate.candidate_id), []).append(
            candidate
        )

    characters: list[dict[str, Any]] = []
    for group_candidates in groups.values():
        group_candidates.sort(
            key=lambda item: (
                _is_generic_name(item.canonical_name),
                item.chunk_order,
            )
        )
        canonical_name = group_candidates[0].canonical_name
        aliases = _unique(
            [
                alias
                for candidate in group_candidates
                for alias in candidate.aliases
                if not _is_generic_name(alias)
            ]
        )
        references = _unique(
            [
                reference
                for candidate in group_candidates
                for reference in candidate.references
            ]
            + [
                candidate.canonical_name
                for candidate in group_candidates
                if _is_generic_name(candidate.canonical_name)
            ]
        )
        mention_ids = _unique(
            [
                mention_id
                for candidate in group_candidates
                for mention_id in candidate.mention_ids
            ]
        )
        source_candidate_ids = sorted(
            candidate.candidate_id
            for candidate in group_candidates
        )
        characters.append(
            {
                "character_id": _stable_project_character_id(
                    source_candidate_ids=source_candidate_ids,
                    canonical_name=canonical_name,
                ),
                "canonical_name": canonical_name,
                "aliases": aliases,
                "references": references,
                "mention_ids": mention_ids,
                "source_candidate_ids": source_candidate_ids,
                "evidence_count": len(mention_ids),
                "input_quality": {
                    "source_candidate_count": len(group_candidates),
                    "model_confidences": [
                        candidate.model_reported_confidence
                        for candidate in group_candidates
                        if candidate.model_reported_confidence is not None
                    ],
                    "source_types": sorted(
                        {
                            candidate.source_type
                            for candidate in group_candidates
                        }
                    ),
                },
            }
        )

    characters.sort(
        key=lambda item: (
            _is_generic_name(str(item["canonical_name"])),
            str(item["canonical_name"]),
        )
    )
    return characters, extra_decisions


def _create_character_run(
    *,
    project_id: str,
    narrative_run_id: int,
    database_path: DatabasePath | None,
) -> int:
    with database_session(database_path=database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO project_character_runs (
                project_id,
                narrative_run_id,
                status
            )
            VALUES (?, ?, 'running')
            """,
            (
                project_id,
                narrative_run_id,
            ),
        )
        run_id = cursor.lastrowid

    if run_id is None:
        raise RuntimeError("Could not create project character run.")

    return int(run_id)


def _save_character_run_result(
    *,
    character_run_id: int,
    project_id: str,
    status: str,
    total_units: int,
    used_units: int,
    skipped_units: int,
    candidates: list[CharacterCandidate],
    characters: list[dict[str, Any]],
    decisions: list[MergeDecision],
    gaps: list[InputGap],
    database_path: DatabasePath | None,
    error_message: str | None = None,
) -> None:
    ambiguous_count = sum(
        1
        for decision in decisions
        if decision.decision == "ambiguous"
    )
    with database_session(database_path=database_path) as connection:
        for character in characters:
            connection.execute(
                """
                INSERT INTO project_characters (
                    character_run_id,
                    project_id,
                    character_id,
                    canonical_name,
                    aliases_json,
                    references_json,
                    mention_ids_json,
                    source_candidate_ids_json,
                    evidence_count,
                    input_quality_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_run_id,
                    project_id,
                    character["character_id"],
                    character["canonical_name"],
                    json.dumps(character["aliases"], ensure_ascii=False),
                    json.dumps(character["references"], ensure_ascii=False),
                    json.dumps(character["mention_ids"], ensure_ascii=False),
                    json.dumps(
                        character["source_candidate_ids"],
                        ensure_ascii=False,
                    ),
                    character["evidence_count"],
                    json.dumps(
                        character["input_quality"],
                        ensure_ascii=False,
                    ),
                ),
            )

        for decision in decisions:
            connection.execute(
                """
                INSERT INTO project_character_merge_decisions (
                    character_run_id,
                    project_id,
                    left_candidate_id,
                    right_candidate_id,
                    decision,
                    merge_score,
                    evidence_json,
                    conflicts_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_run_id,
                    project_id,
                    decision.left_candidate_id,
                    decision.right_candidate_id,
                    decision.decision,
                    decision.merge_score,
                    json.dumps(decision.evidence, ensure_ascii=False),
                    json.dumps(decision.conflicts, ensure_ascii=False),
                ),
            )

        for gap in gaps:
            connection.execute(
                """
                INSERT INTO project_character_input_gaps (
                    character_run_id,
                    project_id,
                    source_unit_id,
                    chunk_id,
                    unit_status,
                    layer_name,
                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_run_id,
                    project_id,
                    gap.source_unit_id,
                    gap.chunk_id,
                    gap.unit_status,
                    gap.layer_name,
                    gap.reason,
                ),
            )

        connection.execute(
            """
            UPDATE project_character_runs
            SET
                status = ?,
                error_message = ?,
                total_units = ?,
                used_units = ?,
                skipped_units = ?,
                total_candidates = ?,
                merged_characters = ?,
                ambiguous_pairs = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status,
                error_message,
                total_units,
                used_units,
                skipped_units,
                len(candidates),
                len(characters),
                ambiguous_count,
                character_run_id,
            ),
        )


def build_project_characters(
    *,
    project_id: str,
    narrative_run_id: int | None = None,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    normalized_project_id = project_id.strip()

    if not normalized_project_id:
        raise ValueError("project_id cannot be empty.")

    resolved_narrative_run_id = (
        narrative_run_id
        if narrative_run_id is not None
        else _latest_narrative_run_id(
            project_id=normalized_project_id,
            database_path=database_path,
        )
    )
    run_project_id, units = _load_narrative_units(
        narrative_run_id=resolved_narrative_run_id,
        database_path=database_path,
    )

    if run_project_id != normalized_project_id:
        raise ValueError("narrative_run_id does not belong to project_id.")

    character_run_id = _create_character_run(
        project_id=normalized_project_id,
        narrative_run_id=resolved_narrative_run_id,
        database_path=database_path,
    )

    candidates, relations, event_frames, gaps, used_units = _collect_inputs(
        project_id=normalized_project_id,
        narrative_run_id=resolved_narrative_run_id,
        units=units,
    )
    skipped_units = max(0, len(units) - used_units)

    if not candidates:
        status = "partial" if gaps else "failed"
        _save_character_run_result(
            character_run_id=character_run_id,
            project_id=normalized_project_id,
            status=status,
            total_units=len(units),
            used_units=used_units,
            skipped_units=skipped_units,
            candidates=candidates,
            characters=[],
            decisions=[],
            gaps=gaps,
            database_path=database_path,
            error_message="no_available_character_candidates",
        )
        return get_project_character_run(
            character_run_id=character_run_id,
            database_path=database_path,
        )

    constraint_decisions, must_links, cannot_links = _constraint_decisions(
        candidates=candidates,
        relations=relations,
        event_frames=event_frames,
    )
    characters, scoring_decisions = _build_characters(
        candidates=candidates,
        decisions=constraint_decisions,
        must_links=must_links,
        cannot_links=cannot_links,
    )
    decisions = [
        *constraint_decisions,
        *scoring_decisions,
    ]
    status = "partial" if gaps else "completed"

    _save_character_run_result(
        character_run_id=character_run_id,
        project_id=normalized_project_id,
        status=status,
        total_units=len(units),
        used_units=used_units,
        skipped_units=skipped_units,
        candidates=candidates,
        characters=characters,
        decisions=decisions,
        gaps=gaps,
        database_path=database_path,
    )

    return get_project_character_run(
        character_run_id=character_run_id,
        database_path=database_path,
    )


def get_project_character_run(
    *,
    character_run_id: int,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    with database_session(database_path=database_path) as connection:
        run_row = connection.execute(
            """
            SELECT *
            FROM project_character_runs
            WHERE id = ?
            """,
            (character_run_id,),
        ).fetchone()

        if run_row is None:
            raise LookupError(
                f"Project character run does not exist: {character_run_id}"
            )

        character_rows = connection.execute(
            """
            SELECT *
            FROM project_characters
            WHERE character_run_id = ?
            ORDER BY id
            """,
            (character_run_id,),
        ).fetchall()
        decision_rows = connection.execute(
            """
            SELECT *
            FROM project_character_merge_decisions
            WHERE character_run_id = ?
            ORDER BY id
            """,
            (character_run_id,),
        ).fetchall()
        gap_rows = connection.execute(
            """
            SELECT *
            FROM project_character_input_gaps
            WHERE character_run_id = ?
            ORDER BY id
            """,
            (character_run_id,),
        ).fetchall()

    return {
        "id": run_row["id"],
        "project_id": run_row["project_id"],
        "narrative_run_id": run_row["narrative_run_id"],
        "status": run_row["status"],
        "error_message": run_row["error_message"],
        "total_units": run_row["total_units"],
        "used_units": run_row["used_units"],
        "skipped_units": run_row["skipped_units"],
        "total_candidates": run_row["total_candidates"],
        "merged_characters": run_row["merged_characters"],
        "ambiguous_pairs": run_row["ambiguous_pairs"],
        "created_at": run_row["created_at"],
        "updated_at": run_row["updated_at"],
        "characters": [
            {
                "id": row["id"],
                "character_id": row["character_id"],
                "canonical_name": row["canonical_name"],
                "aliases": json.loads(row["aliases_json"]),
                "references": json.loads(row["references_json"]),
                "mention_ids": json.loads(row["mention_ids_json"]),
                "source_candidate_ids": json.loads(
                    row["source_candidate_ids_json"]
                ),
                "evidence_count": row["evidence_count"],
                "input_quality": json.loads(row["input_quality_json"]),
            }
            for row in character_rows
        ],
        "merge_decisions": [
            {
                "id": row["id"],
                "left_candidate_id": row["left_candidate_id"],
                "right_candidate_id": row["right_candidate_id"],
                "decision": row["decision"],
                "merge_score": row["merge_score"],
                "evidence": json.loads(row["evidence_json"]),
                "conflicts": json.loads(row["conflicts_json"]),
            }
            for row in decision_rows
        ],
        "input_gaps": [
            {
                "id": row["id"],
                "source_unit_id": row["source_unit_id"],
                "chunk_id": row["chunk_id"],
                "unit_status": row["unit_status"],
                "layer_name": row["layer_name"],
                "reason": row["reason"],
            }
            for row in gap_rows
        ],
    }


def get_latest_project_character_run(
    *,
    project_id: str,
    database_path: DatabasePath | None = None,
) -> dict[str, Any] | None:
    with database_session(database_path=database_path) as connection:
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

    if row is None:
        return None

    return get_project_character_run(
        character_run_id=int(row["id"]),
        database_path=database_path,
    )
