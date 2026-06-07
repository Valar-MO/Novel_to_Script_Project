import json
from typing import Any


PROMPT_VERSION = "free_character_relation_extraction_v1"


def build_relation_extraction_messages(
    *,
    target_text: str,
    mentions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build messages for free-label character relation extraction."""

    character_mentions = [
        mention
        for mention in mentions
        if mention.get("mention_type") == "character"
    ]
    mentions_text = json.dumps(
        character_mentions,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    system_prompt = (
        "你是 Novel2Script 的人物关系抽取器。"
        "你的任务是基于 TARGET_TEXT 和已验证定位的 CHARACTER_MENTIONS，"
        "抽取原文明确支持的人物之间关系。"
        "只允许使用 CHARACTER_MENTIONS 中的 mention_id 作为 source_mention_id 和 target_mention_id。"
        "source_mention 和 target_mention 必须与对应 mention_id 的 mention_text 一致。"
        "关系标签不受固定枚举限制，请根据原文自由概括。"
        "允许复杂、模糊、单向和复合关系，例如：兄弟、母子、名义师徒，彼此提防、"
        "表面合作，暗中敌对、单方面依赖、互相利用、关系暧昧。"
        "relation 字段用于关系图边上的简短文字，应尽量短。"
        "如需解释，把细节放入 evidence_text 支持的原文中；不要脱离原文推测。"
        "不要把一次性动作抽成关系，例如携带、乘坐、闭眼、睡觉、看见、说话、进入、离开。"
        "evidence_text 必须是 TARGET_TEXT 中的精确连续原文片段。"
        "最多返回 15 条最重要的人物关系。"
        "不确定的关系不要返回，可以写入 warnings。"
        "不得输出 Markdown、解释或代码围栏。"
    )

    user_prompt = (
        "TARGET_TEXT:\n"
        f"{target_text}\n\n"
        "CHARACTER_MENTIONS，已经由程序验证可在 TARGET_TEXT 中定位：\n"
        f"{mentions_text}\n\n"
        "请只抽取这些人物 mentions 之间在 TARGET_TEXT 中明确表达的人物关系。"
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
