from typing import Any

from pydantic import BaseModel


EVIDENCE_COLLECTION_FIELDS = (
    "mentions",
    "relations",
    "event_frames",
    "character_candidates",
)


def _find_nth_occurrence(
    text: str,
    evidence_text: str,
    occurrence_index: int,
) -> tuple[int | None, int | None]:
    """Locate the nth occurrence of evidence_text in text.

    Returns (start_offset, end_offset) or (None, None) if not found.
    Uses str.find() repeatedly to find the Nth occurrence.
    """
    if not evidence_text:
        return None, None

    position = 0
    current_occurrence = 0

    while True:
        found_at = text.find(evidence_text, position)
        if found_at < 0:
            return None, None

        if current_occurrence == occurrence_index:
            return found_at, found_at + len(evidence_text)

        position = found_at + 1
        current_occurrence += 1


def _find_evidence_span(
    text: str,
    evidence_text: str,
    occurrence_index: int = 0,
) -> tuple[int | None, int | None]:
    """Find the span of evidence_text in text at the given occurrence index."""
    return _find_nth_occurrence(text, evidence_text, occurrence_index)


def _validate_evidence_item(
    target_text: str,
    item: dict[str, Any],
) -> None:
    evidence_text = str(item.get("evidence_text") or "").strip()
    occurrence_index = int(item.get("occurrence_index") or 0)

    start_offset, end_offset = _find_evidence_span(
        target_text,
        evidence_text,
        occurrence_index,
    )

    item["start_offset"] = start_offset
    item["end_offset"] = end_offset
    item["evidence_validated"] = (
        start_offset is not None and end_offset is not None
    )


def validate_evidence(
    *,
    target_text: str,
    extraction: BaseModel | dict[str, Any],
) -> dict[str, Any]:
    """Validate all evidence_text fields against the current target text.

    The returned dict keeps the original model structure and adds
    start_offset, end_offset, and evidence_validated to evidence-bearing
    records. Offsets are local to target_text.
    """

    if isinstance(extraction, BaseModel):
        validated_result = extraction.model_dump(mode="json")
    else:
        validated_result = dict(extraction)

    for field_name in EVIDENCE_COLLECTION_FIELDS:
        items = validated_result.get(field_name)

        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            _validate_evidence_item(
                target_text,
                item,
            )

    return validated_result
