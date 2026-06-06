import json
from typing import Any

from backend.llm.schemas import CharacterCandidateExtractionOutput


PROMPT_VERSION = "character_candidate_extraction_v4"


def build_character_candidate_extraction_messages(
    *,
    target_text: str,
    mentions: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    event_frames: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build messages for local character candidate merging."""

    context_payload = {
        "mentions": mentions,
        "relations": relations,
        "event_frames": event_frames,
    }
    context_text = json.dumps(
        context_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    system_prompt = (
        "你是 Novel2Script 的本地人物候选合并器。"
        "你的任务是基于 TARGET_TEXT、已经验证的 mentions、relations 和 event_frames，"
        "把同一人物的不同称呼合并成当前文本块内的 character_candidates。"
        "只处理 mention_type 为 character 的 mention。"
        "每个候选必须返回 canonical_name、mention_ids、aliases、references、"
        "evidence_text、confidence。"
        "mention_ids 必须全部来自已验证 character mentions 的 mention_id。"
        "aliases 只放稳定名字、外号或明确别名，例如 韩立、二愣子。"
        "references 放上下文称谓或指代，例如 他、她、母亲、父亲、三叔、老人、少年。"
        "不要把 references 直接当作正式 aliases。"
        "如果只有称谓或代词，宁可保守地放入 references，并在 canonical_name 中选择"
        "当前文本块最适合展示的原文称呼。"
        "不要合并证据不足的人物；不确定就分成不同候选或写入 warnings。"
        "最多返回 20 个最重要的人物候选。"
        "evidence_text 必须是 TARGET_TEXT 中支持该合并的精确原文片段。"
        "不要写人物传记、性格分析、角色弧光或剧情推断。"
        "不得输出 Markdown、解释、表格或代码围栏。"
        "JSON 对象结束后不得追加任何文字。"
    )

    user_prompt = (
        "TARGET_TEXT:\n"
        f"{target_text}\n\n"
        "已验证的输入:\n"
        f"{context_text}\n\n"
        "请输出当前文本块内的人物候选合并结果。"
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
