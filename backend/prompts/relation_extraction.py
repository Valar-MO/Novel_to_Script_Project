import json
from typing import Any


PROMPT_VERSION = "relation_extraction_v5"


def build_relation_extraction_messages(
    *,
    target_text: str,
    mentions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build messages for relation extraction over anchored mentions."""

    mentions_text = json.dumps(
        mentions,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    system_prompt = (
        "你是 Novel2Script 的关系抽取器。"
        "你的任务是基于 TARGET_TEXT 和已经验证定位的 MENTIONS，"
        "只抽取原文明确确认的、相对稳定的 mention 之间关系。"
        "每条 relation 必须同时返回 source_mention、source_mention_id、"
        "relation、target_mention、target_mention_id、evidence_text、confidence。"
        "source_mention_id 和 target_mention_id 必须来自 MENTIONS 中的 mention_id；"
        "source_mention 和 target_mention 必须与对应 mention_id 的 mention_text 一致。"
        "优先抽取人物、组织、地点之间的稳定关系，例如亲属、师徒、同伴、"
        "所属、居住地、称呼、别名、任职、隶属、照顾关系、介绍关系。"
        "不要把一次性动作或事件论元抽成关系，例如携带、乘坐、闭眼、睡觉、"
        "走向、拿起、放下、看见、听见、说话、叫喊、进入、离开等。"
        "这类动作应留给事件框架抽取。"
        "relation 必须是原文可直接支持的简短稳定关系。"
        "evidence_text 必须是 TARGET_TEXT 中的精确原文片段。"
        "最多返回 15 条最重要的关系。"
        "不确定的关系不要返回，可以写入 warnings。"
        "不得输出 Markdown、解释或代码围栏。"
    )

    user_prompt = (
        "TARGET_TEXT:\n"
        f"{target_text}\n\n"
        "MENTIONS，已经由程序验证可在 TARGET_TEXT 中定位:\n"
        f"{mentions_text}\n\n"
        "请只抽取这些 mentions 之间在 TARGET_TEXT 中明确表达的稳定关系。"
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
