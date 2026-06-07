from backend.llm.schemas import MentionExtractionOutput


PROMPT_VERSION = "character_mention_extraction_v1"


def build_mention_extraction_messages(
    *,
    previous_context: str = "",
    target_text: str,
    next_context: str = "",
) -> list[dict[str, str]]:
    """Build messages for character-only text-anchor extraction."""

    system_prompt = (
        "你是 Novel2Script 的人物文本锚点识别器。"
        "你的任务只是在 TARGET_TEXT 中找出明确出现的人物相关文本片段。"
        "不要抽取地点、时间、组织、物品、事件、动作、对白、目标、冲突或叙事解释。"
        "每条记录必须使用统一字段：mention_type, mention_text, evidence_text, confidence。"
        "mention_type 必须始终为 character。"
        "mention_text 必须是 TARGET_TEXT 中出现的人物姓名、别名、小名、称谓、稳定身份称呼或代称。"
        "可以抽取如 韩立、二愣子、三叔、韩父、韩母、墨大夫、少年、老人。"
        "不要抽取纯地点、门派组织、物件或抽象群体。"
        "同一文本块内，相同 mention_text 只输出一次，不要用 occurrence_index 区分多次出现。"
        "evidence_text 只保留最短且足以证明该人物锚点的连续原文，尽量与 mention_text 相同。"
        "evidence_text 必须来自 TARGET_TEXT，不能来自上下文。"
        "所有字段值必须使用 TARGET_TEXT 的原文语言，不得翻译。"
        "不得新增 schema 外字段，不得输出 Markdown、解释或代码围栏。"
        "无法确认的内容不要返回。"
    )

    user_prompt = (
        "PREVIOUS_CONTEXT，仅用于消歧，不能作为 evidence_text 来源：\n"
        f"{previous_context or '[empty]'}\n\n"
        "TARGET_TEXT，唯一允许作为 evidence_text 来源的正文：\n"
        f"{target_text}\n\n"
        "NEXT_CONTEXT，仅用于消歧，不能作为 evidence_text 来源：\n"
        f"{next_context or '[empty]'}\n\n"
        "请只识别 TARGET_TEXT 中明确出现的人物锚点，所有 mention_type 都必须是 character。"
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


__all__ = [
    "MentionExtractionOutput",
    "PROMPT_VERSION",
    "build_mention_extraction_messages",
]
