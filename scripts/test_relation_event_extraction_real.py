import asyncio
import json

from backend.llm.factory import create_llm_provider
from backend.llm.schemas import (
    CharacterCandidateExtractionOutput,
    EventFrameExtractionOutput,
    MentionExtractionOutput,
    RelationExtractionOutput,
)
from backend.prompts.character_candidate_extraction import (
    build_character_candidate_extraction_messages,
)
from backend.prompts.event_frame_extraction import (
    build_event_frame_extraction_messages,
)
from backend.prompts.mention_extraction import (
    build_mention_extraction_messages,
)
from backend.prompts.relation_extraction import (
    build_relation_extraction_messages,
)
from backend.services.evidence_validator import validate_evidence
from backend.services.narrative_analysis import (
    _build_mention_candidates,
    _filter_validated_character_candidates,
    _filter_validated_event_frames,
    _filter_validated_mentions,
    _filter_validated_relations,
)


async def main() -> None:
    provider = create_llm_provider()

    try:
        target_text = (
            "韩立只在很小的时候，见过这位三叔几次。"
            "他大哥在城里给一位老铁匠当学徒的工作，"
            "就是这位三叔给介绍的。"
            "这位三叔还经常托人给他父母捎带一些吃的用的东西，"
            "很是照顾他们一家。"
        )

        mention_output = await provider.generate_structured(
            messages=build_mention_extraction_messages(
                target_text=target_text,
            ),
            response_model=MentionExtractionOutput,
            temperature=0,
        )
        validated_mentions = _filter_validated_mentions(
            validate_evidence(
                target_text=target_text,
                extraction=mention_output,
            )
        )
        for index, mention in enumerate(
            validated_mentions["mentions"],
            start=1,
        ):
            mention["mention_id"] = f"chunk_0001_m_{index:03d}"

        mention_candidates = _build_mention_candidates(
            validated_mentions["mentions"]
        )
        allowed_mention_ids = {
            str(mention["mention_id"])
            for mention in mention_candidates
            if mention.get("mention_id")
        }

        relation_output = await provider.generate_structured(
            messages=build_relation_extraction_messages(
                target_text=target_text,
                mentions=mention_candidates,
            ),
            response_model=RelationExtractionOutput,
            temperature=0,
        )
        validated_relations = _filter_validated_relations(
            validate_evidence(
                target_text=target_text,
                extraction=relation_output,
            ),
            allowed_mention_ids=allowed_mention_ids,
            chunk_id="chunk_0001",
        )

        event_output = await provider.generate_structured(
            messages=build_event_frame_extraction_messages(
                target_text=target_text,
                mentions=mention_candidates,
            ),
            response_model=EventFrameExtractionOutput,
            temperature=0,
        )
        validated_events = _filter_validated_event_frames(
            validate_evidence(
                target_text=target_text,
                extraction=event_output,
            ),
            allowed_mention_ids=allowed_mention_ids,
            chunk_id="chunk_0001",
        )

        allowed_character_mention_ids = {
            str(mention.get("mention_id"))
            for mention in validated_mentions["mentions"]
            if (
                mention.get("mention_type") == "character"
                and mention.get("mention_id")
            )
        }

        character_output = await provider.generate_structured(
            messages=build_character_candidate_extraction_messages(
                target_text=target_text,
                mentions=validated_mentions["mentions"],
                relations=validated_relations["relations"],
                event_frames=validated_events["event_frames"],
            ),
            response_model=CharacterCandidateExtractionOutput,
            temperature=0,
        )
        validated_characters = _filter_validated_character_candidates(
            validate_evidence(
                target_text=target_text,
                extraction=character_output,
            ),
            allowed_character_mention_ids=allowed_character_mention_ids,
            chunk_id="chunk_0001",
        )

        print(
            json.dumps(
                {
                    "character_candidates": (
                        validated_characters["character_candidates"]
                    ),
                    "mentions": validated_mentions["mentions"],
                    "relations": validated_relations["relations"],
                    "event_frames": validated_events["event_frames"],
                    "warnings": {
                        "character_candidates": (
                            validated_characters["warnings"]
                        ),
                        "mentions": validated_mentions["warnings"],
                        "relations": validated_relations["warnings"],
                        "event_frames": validated_events["warnings"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
