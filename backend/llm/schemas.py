import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _normalize_text(
    value: str,
) -> str:
    """规范化：折叠所有连续空白为空格，首尾去空白。"""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_label_text(
    value: str,
) -> str:
    """用于标签字段（mention_type、relation、event_type 等）：折叠空白。"""
    return _normalize_text(value)


def _strip_text(
    value: str,
) -> str:
    """用于 evidence_text：只去除首尾空白，保留内部所有空白和换行。"""
    return value.strip()


def _deduplicate_strings(
    values: list[str],
) -> list[str]:
    """规范化并去重字符串列表，保留首次出现的顺序。"""
    normalized_values: list[str] = []
    seen_values: set[str] = set()

    for value in values:
        normalized_value = _normalize_text(value)

        if not normalized_value:
            continue

        comparison_key = normalized_value.casefold()

        if comparison_key in seen_values:
            continue

        seen_values.add(comparison_key)
        normalized_values.append(normalized_value)

    return normalized_values


class MentionOutput(BaseModel):
    """A directly anchored text mention from the current chunk."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    mention_type: Literal[
        "character",
        "location",
        "time",
        "organization",
        "object",
    ] = Field(
        description="The coarse type of the anchored mention.",
    )

    mention_text: str = Field(
        min_length=1,
        max_length=200,
        description="The entity or expression text as it appears.",
    )

    evidence_text: str = Field(
        min_length=1,
        max_length=1000,
        description="Shortest exact source text that anchors this mention.",
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Model confidence for this mention.",
    )

    @field_validator("mention_type")
    @classmethod
    def normalize_mention_type(cls, value: str) -> str:
        return _normalize_label_text(value)

    @field_validator("mention_text")
    @classmethod
    def normalize_mention_text(cls, value: str) -> str:
        normalized_value = _normalize_text(value)
        if not normalized_value:
            raise ValueError("mention_text cannot be empty.")
        return normalized_value

    @field_validator("evidence_text")
    @classmethod
    def normalize_evidence_text(cls, value: str) -> str:
        """evidence_text 只去除首尾空白，保留内部所有空白。"""
        stripped = _strip_text(value)
        if not stripped:
            raise ValueError("evidence_text cannot be empty.")
        return stripped


class MentionExtractionOutput(BaseModel):
    """Text-anchor mentions extracted from one text chunk."""

    model_config = ConfigDict(
        extra="forbid",
    )

    mentions: list[MentionOutput] = Field(
        default_factory=list,
        description="Directly anchored mentions in the current text chunk.",
    )
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("warnings must be a string list.")
        return _deduplicate_strings([
            str(item)
            for item in value
            if item is not None
        ])


class RelationOutput(BaseModel):
    """A relation directly supported by the current text chunk."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_mention: str = Field(
        min_length=1,
        max_length=200,
        description="The source mention text from the mention layer.",
    )
    source_mention_id: str = Field(
        min_length=1,
        max_length=80,
        description="Stable local ID of the source mention.",
    )
    relation: str = Field(
        min_length=1,
        max_length=100,
        description="The relation expressed by the text.",
    )
    target_mention: str = Field(
        min_length=1,
        max_length=200,
        description="The target mention text from the mention layer.",
    )
    target_mention_id: str = Field(
        min_length=1,
        max_length=80,
        description="Stable local ID of the target mention.",
    )
    evidence_text: str = Field(
        min_length=1,
        max_length=1500,
        description="Exact source text supporting this relation.",
    )
    occurrence_index: int = Field(
        default=0,
        ge=0,
        description="Which occurrence of evidence_text in the chunk (0 = first).",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Model confidence for this relation.",
    )

    @field_validator(
        "source_mention",
        "source_mention_id",
        "relation",
        "target_mention",
        "target_mention_id",
    )
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized_value = _normalize_text(value)
        if not normalized_value:
            raise ValueError("relation fields cannot be empty.")
        return normalized_value

    @field_validator("evidence_text")
    @classmethod
    def normalize_evidence_text(cls, value: str) -> str:
        """evidence_text 只去除首尾空白，保留内部所有空白。"""
        stripped = _strip_text(value)
        if not stripped:
            raise ValueError("evidence_text cannot be empty.")
        return stripped


class RelationExtractionOutput(BaseModel):
    """Relations extracted using anchored mentions as arguments."""

    model_config = ConfigDict(
        extra="forbid",
    )

    relations: list[RelationOutput] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("warnings must be a string list.")
        return _deduplicate_strings([
            str(item)
            for item in value
            if item is not None
        ])


class EventArgumentOutput(BaseModel):
    """A mention playing a role in an event frame."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    role: str = Field(
        min_length=1,
        max_length=80,
        description="Semantic role such as actor, patient, destination.",
    )
    mention_id: str = Field(
        default="",
        max_length=80,
        description=(
            "Stable local ID of the mention filling this role. "
            "The service drops arguments that cannot be resolved."
        ),
    )
    mention_text: str = Field(
        min_length=1,
        max_length=200,
        description="Mention text from the mention layer.",
    )

    @field_validator("role", "mention_text")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized_value = _normalize_text(value)
        if not normalized_value:
            raise ValueError("event argument fields cannot be empty.")
        return normalized_value

    @field_validator("mention_id")
    @classmethod
    def normalize_optional_mention_id(cls, value: str) -> str:
        return _normalize_text(value)


class EventFrameOutput(BaseModel):
    """A trigger-argument event frame anchored in the current text chunk."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    trigger_text: str = Field(
        min_length=1,
        max_length=100,
        description="The event trigger text as it appears in source text.",
    )
    event_type: str = Field(
        min_length=1,
        max_length=80,
        description="Coarse event type.",
    )
    arguments: list[EventArgumentOutput] = Field(default_factory=list)
    evidence_text: str = Field(
        min_length=1,
        max_length=2000,
        description="Exact source text supporting this event frame.",
    )
    occurrence_index: int = Field(
        default=0,
        ge=0,
        description="Which occurrence of evidence_text in the chunk (0 = first).",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Model confidence for this event frame.",
    )

    @field_validator("trigger_text", "event_type")
    @classmethod
    def normalize_label_text(cls, value: str) -> str:
        normalized_value = _normalize_label_text(value)
        if not normalized_value:
            raise ValueError("event frame fields cannot be empty.")
        return normalized_value

    @field_validator("evidence_text")
    @classmethod
    def normalize_evidence_text(cls, value: str) -> str:
        """evidence_text 只去除首尾空白，保留内部所有空白。"""
        stripped = _strip_text(value)
        if not stripped:
            raise ValueError("evidence_text cannot be empty.")
        return stripped


class EventFrameExtractionOutput(BaseModel):
    """Event frames extracted using anchored mentions as arguments."""

    model_config = ConfigDict(
        extra="forbid",
    )

    event_frames: list[EventFrameOutput] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_event_frame_root_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        if "event_frames" in value:
            return value

        alias_keys = (
            "events",
            "event",
            "event_frame",
            "eventFrame",
            "eventFrames",
            "event_frame_extractions",
            "eventFrameExtractionOutput",
            "event_frame_extraction_output",
            "frames",
        )

        for alias_key in alias_keys:
            if alias_key not in value:
                continue

            normalized_value = dict(value)
            alias_value = normalized_value.pop(alias_key)

            if isinstance(alias_value, list):
                normalized_value["event_frames"] = alias_value
            elif isinstance(alias_value, dict):
                normalized_value["event_frames"] = [alias_value]
            else:
                normalized_value["event_frames"] = []

            return normalized_value

        return value

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("warnings must be a string list.")
        return _deduplicate_strings([
            str(item)
            for item in value
            if item is not None
        ])


class CharacterCandidateOutput(BaseModel):
    """A local canonical character candidate within one text chunk."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    canonical_name: str = Field(
        min_length=1,
        max_length=200,
        description="Best local display name for the character candidate.",
    )
    mention_ids: list[str] = Field(
        default_factory=list,
        description="Validated character mention IDs grouped together.",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Stable names or nicknames for this character.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Contextual references such as pronouns or roles.",
    )
    evidence_text: str = Field(
        min_length=1,
        max_length=2000,
        description="Exact source text supporting this grouping.",
    )
    occurrence_index: int = Field(
        default=0,
        ge=0,
        description="Which occurrence of evidence_text in the chunk (0 = first).",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Model confidence for this candidate grouping.",
    )

    @field_validator("canonical_name")
    @classmethod
    def normalize_canonical_name(cls, value: str) -> str:
        normalized_value = _normalize_text(value)
        if not normalized_value:
            raise ValueError("canonical_name cannot be empty.")
        return normalized_value

    @field_validator("evidence_text")
    @classmethod
    def normalize_evidence_text(cls, value: str) -> str:
        """evidence_text 只去除首尾空白，保留内部所有空白。"""
        stripped = _strip_text(value)
        if not stripped:
            raise ValueError("evidence_text cannot be empty.")
        return stripped

    @field_validator("mention_ids", "aliases", "references", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("field must be a string list.")
        if not all(
            isinstance(item, str)
            for item in value
            if item is not None
        ):
            raise ValueError("Every item must be a string.")
        return _deduplicate_strings([
            item
            for item in value
            if item is not None
        ])


class CharacterCandidateExtractionOutput(BaseModel):
    """Local character candidates merged from mentions, relations, and events."""

    model_config = ConfigDict(
        extra="forbid",
    )

    character_candidates: list[CharacterCandidateOutput] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("warnings must be a string list.")
        return _deduplicate_strings([
            str(item)
            for item in value
            if item is not None
        ])


class GeneratedCharacterRef(BaseModel):
    """A character used in a generated script scene."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    character_id: str | None = Field(
        default=None,
        max_length=120,
    )
    name: str = Field(
        min_length=1,
        max_length=200,
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized_value = _normalize_text(value)
        if not normalized_value:
            raise ValueError("name cannot be empty.")
        return normalized_value

    @field_validator("character_id")
    @classmethod
    def normalize_character_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_value = _normalize_text(value)
        return normalized_value or None


class GeneratedSourceAnchor(BaseModel):
    """Text anchors used by the backend to locate source offsets."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    start_text: str = Field(
        min_length=1,
        max_length=500,
    )
    end_text: str = Field(
        min_length=1,
        max_length=500,
    )

    @field_validator("start_text", "end_text")
    @classmethod
    def normalize_anchor_text(cls, value: str) -> str:
        stripped = _strip_text(value)
        if not stripped:
            raise ValueError("source anchors cannot be empty.")
        return stripped


class GeneratedScene(BaseModel):
    """One scene generated from the current source chunk."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    order_in_unit: int = Field(
        ge=1,
    )
    continue_previous_scene: bool = False
    interior_exterior: str = Field(
        min_length=1,
        max_length=40,
    )
    location: str = Field(
        min_length=1,
        max_length=200,
    )
    time_of_day: str = Field(
        min_length=1,
        max_length=80,
    )
    heading: str = Field(
        min_length=1,
        max_length=300,
    )
    characters: list[GeneratedCharacterRef] = Field(
        default_factory=list,
    )
    script_text: str = Field(
        min_length=1,
        max_length=20000,
    )
    scene_summary: str = Field(
        default="",
        max_length=2000,
    )
    source_anchor: GeneratedSourceAnchor
    adaptation_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "interior_exterior",
        "location",
        "time_of_day",
        "heading",
        "scene_summary",
    )
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        return _normalize_text(value)

    @field_validator("script_text")
    @classmethod
    def normalize_script_text(cls, value: str) -> str:
        stripped = _strip_text(value)
        if not stripped:
            raise ValueError("script_text cannot be empty.")
        return stripped

    @field_validator("adaptation_notes", "warnings", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("field must be a string list.")
        return _deduplicate_strings([
            str(item)
            for item in value
            if item is not None
        ])


class ScriptGenerationOutput(BaseModel):
    """Script scenes generated from one source chunk."""

    model_config = ConfigDict(
        extra="forbid",
    )

    scenes: list[GeneratedScene] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("warnings must be a string list.")
        return _deduplicate_strings([
            str(item)
            for item in value
            if item is not None
        ])
