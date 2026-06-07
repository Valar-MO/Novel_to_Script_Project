import asyncio
import json

from backend.llm.factory import create_llm_provider
from backend.llm.schemas import (
    MentionExtractionOutput,
    RelationExtractionOutput,
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
    _filter_validated_mentions,
    _filter_validated_relations,
)


async def main() -> None:
    provider = create_llm_provider()

    try:
        chunk_id = "chunk_0001"
        target_text = (
            "Han Li met Doctor Mo outside the gate. "
            "Doctor Mo taught Han Li basic cultivation methods, "
            "but Han Li quietly doubted his true purpose."
        )

        mention_output = await provider.generate_structured(
            messages=build_mention_extraction_messages(
                previous_context="",
                target_text=target_text,
                next_context="",
            ),
            response_model=MentionExtractionOutput,
            temperature=0,
        )
        validated_mentions = _filter_validated_mentions(
            validate_evidence(
                target_text=target_text,
                extraction=mention_output,
            ),
            chunk_id=chunk_id,
        )
        mention_candidates = _build_mention_candidates(
            validated_mentions["mentions"]
        )
        mention_by_id = {
            str(mention["mention_id"]): mention
            for mention in validated_mentions["mentions"]
            if mention.get("mention_id")
        }
        allowed_mention_ids = set(mention_by_id)

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
            mention_by_id=mention_by_id,
            allowed_mention_ids=allowed_mention_ids,
            chunk_id=chunk_id,
        )

        print(
            json.dumps(
                {
                    "mentions": validated_mentions["mentions"],
                    "relations": validated_relations["relations"],
                    "warnings": {
                        "mentions": validated_mentions["warnings"],
                        "relations": validated_relations["warnings"],
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
