from backend.llm.schemas import MentionExtractionOutput


PROMPT_VERSION = "mention_extraction_v4"


def build_mention_extraction_messages(
    *,
    previous_context: str = "",
    target_text: str,
    next_context: str = "",
) -> list[dict[str, str]]:
    """Build messages for direct text-anchor mention extraction."""

    system_prompt = (
        "你是 Novel2Script 的文本锚点识别器。"
        "你的任务只是在 TARGET_TEXT 中找出明确出现的文本片段，"
        "不要抽取事件、关系、动作、对白、目标、冲突、场景边界"
        "或叙事解释。"
        "每条记录必须使用统一字段：mention_type, mention_text, "
        "evidence_text, confidence。"
        "mention_type 只能是 character, location, time, "
        "organization, object 之一。"
        "mention_text 必须是 TARGET_TEXT 中出现的人物称呼、地点、"
        "时间表达、组织名称或重要物件名称。"
        "仅抽取会参与关键事件或剧情推进的重要物件，"
        "普通物件不要穷举。"
        "同一文本块内，相同 mention_type 和 mention_text 只输出一次，"
        "不要用 occurrence_index 区分多次出现。"
        "evidence_text 只保留最短且足以证明该 mention 的连续原文，"
        "尽量与 mention_text 相同或只增加极少量上下文；"
        "必须保留原文中所有字符不变，包括换行、空格和标点。"
        "evidence_text 不得超过 200 个字符。"
        "最多返回 30 条最重要的文本锚点。"
        "上下文只能用于消解代词或称呼，不能作为 evidence_text 来源。"
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
        "请只识别 TARGET_TEXT 中明确出现的文本锚点。"
        "不要生成事件摘要、人物关系、动作分析或叙事推理。"
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
