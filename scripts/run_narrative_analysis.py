import argparse
import asyncio

from backend.services.narrative_analysis import analyze_project_narrative


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run narrative analysis for a saved project.",
    )
    parser.add_argument("project_id")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit chunks for smoke testing.",
    )
    parser.add_argument(
        "--next-context-chars",
        type=int,
        default=0,
        help="Optional next chunk context size.",
    )

    args = parser.parse_args()

    result = await analyze_project_narrative(
        project_id=args.project_id,
        max_chunks=args.max_chunks,
        next_context_chars=args.next_context_chars,
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
