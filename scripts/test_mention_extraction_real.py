import asyncio

from backend.llm.factory import create_llm_provider
from backend.llm.schemas import MentionExtractionOutput
from backend.prompts.mention_extraction import (
    PROMPT_VERSION,
    build_mention_extraction_messages,
)


async def main() -> None:
    provider = create_llm_provider()

    try:
        health = await provider.health_check()

        print("LLM health:")
        print(health.to_dict())

        if not health.available:
            raise RuntimeError(
                f"LLM is not available: {health.detail}"
            )

        target_text = (
            "二愣子睁大着双眼，直直望着茅草和烂泥糊成的黑屋顶。"
            "在他身边紧挨着的另一人，是二哥韩铸。"
            "从这些裂纹中，隐隐约约的传来韩母唠唠叨叨的埋怨声，"
            "偶尔还掺杂着韩父抽旱烟杆的声音。"
        )

        result = await provider.generate_structured(
            messages=build_mention_extraction_messages(
                previous_context="",
                target_text=target_text,
                next_context="",
            ),
            response_model=MentionExtractionOutput,
            temperature=0,
        )

        print("Prompt version:", PROMPT_VERSION)
        print("\nMention extraction result:")
        print(
            result.model_dump_json(
                indent=2,
            )
        )

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
