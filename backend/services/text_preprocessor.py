import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessedText:
    """单个文本文件的预处理结果。"""

    text: str
    original_character_count: int
    processed_character_count: int
    processed_line_count: int


def preprocess_text(text: str) -> PreprocessedText:
    """
    对小说正文进行基础规范化。

    当前只处理格式问题，不进行章节识别、内容改写或文本分块。
    """

    original_character_count = len(text)

    # 去除文件开头可能存在的 UTF-8 BOM。
    normalized_text = text.lstrip("\ufeff")

    # 统一 Windows、旧版 Mac 和 Linux 换行符。
    normalized_text = normalized_text.replace("\r\n", "\n")
    normalized_text = normalized_text.replace("\r", "\n")

    # 将制表符统一为四个空格，保留段落缩进结构。
    normalized_text = normalized_text.replace("\t", "    ")

    # 删除每行末尾的空格，但保留行首缩进。
    lines = [
        line.rstrip()
        for line in normalized_text.split("\n")
    ]
    normalized_text = "\n".join(lines)

    # 三个及以上连续换行统一为两个换行，即最多保留一个空行。
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)

    # 删除整篇文本首尾的空行，同时保留首行可能存在的段落缩进。
    normalized_text = normalized_text.strip("\n")

    processed_line_count = (
        normalized_text.count("\n") + 1
        if normalized_text
        else 0
    )

    return PreprocessedText(
        text=normalized_text,
        original_character_count=original_character_count,
        processed_character_count=len(normalized_text),
        processed_line_count=processed_line_count,
    )