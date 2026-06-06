import json
from typing import Any

from backend.llm.schemas import EventFrameExtractionOutput


PROMPT_VERSION = "event_frame_extraction_v6"


def build_event_frame_extraction_messages(
    *,
    target_text: str,
    mentions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build messages for trigger-argument event frame extraction."""

    mentions_text = json.dumps(
        mentions,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    allowed_event_types = (
        "movement, communication, perception, cognition, state, "
        "possession, social, creation, conflict, other"
    )

    system_prompt = (
        "你是 Novel2Script 的事件框架抽取器。"
        "你的任务是基于 TARGET_TEXT 和已经验证定位的 MENTIONS，"
        "抽取原文明确表达的低层事件。"
        "每个事件必须使用 trigger_text、event_type、arguments、"
        "evidence_text、confidence。"
        f"event_type 只能从这些英文枚举中选择：{allowed_event_types}。"
        "如果拿不准类型，必须使用 other，不得自创新类型。"
        "trigger_text 必须是 TARGET_TEXT 中出现的触发词或短语。"
        "arguments 中每个论元必须同时返回 role、mention_id、mention_text。"
        "mention_id 必须来自 MENTIONS 中的 mention_id；"
        "mention_text 必须与对应 mention_id 的 mention_text 一致。"
        "不得编造 MENTIONS 之外的新实体。"
        "evidence_text 必须是 TARGET_TEXT 中的精确原文片段。"
        "最多返回 8 个最重要、最明确的事件框架。"
        "不要为纯静态描写、背景环境描写、睡眠状态、细小生理动作、"
        "声音细节或不改变故事状态的动作单独创建事件，例如"
        "睁大眼睛、酣睡、打呼、听到声音、眨眼、眨眼皮、"
        "身体微微一动、嘴角微微一翘、深吸一口气、转过头来。"
        "不要写 event_summary，不要做目标、冲突、动机或主题分析。"
        "不确定的事件不要返回，可以写入 warnings。"
        "不得输出 Markdown、解释、表格、分析建议或代码围栏。"
        "JSON 对象结束后不得追加任何文字。"
    )

    user_prompt = (
        "TARGET_TEXT:\n"
        f"{target_text}\n\n"
        "MENTIONS，已经由程序验证可在 TARGET_TEXT 中定位:\n"
        f"{mentions_text}\n\n"
        "请只抽取 TARGET_TEXT 中明确表达的事件框架。"
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
