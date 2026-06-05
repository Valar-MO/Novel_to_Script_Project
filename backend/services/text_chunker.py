import re
from dataclasses import dataclass

from backend.services.chapter_detector import ChapterSpan, detect_chapters


DEFAULT_MIN_CHARS = 2000
DEFAULT_TARGET_CHARS = 3000
DEFAULT_MAX_CHARS = 4000

_PARAGRAPH_SEPARATOR_PATTERN = re.compile(r"\n{2,}")
_ENGLISH_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "eg",
    "ie",
}
_CLOSING_PUNCTUATION = '”’」』】》）)]\"\''


@dataclass(frozen=True)
class TextChunk:
    """一个可追溯的小说文本块。"""

    chunk_id: str
    global_order: int
    source_file_name: str
    source_file_order: int
    start_character: int
    end_character: int
    character_count: int
    paragraph_start: int
    paragraph_end: int
    text: str
    chapter_order: int | None = None
    chapter_number: int | None = None
    chapter_title: str | None = None
    chapter_full_title: str | None = None
    chunk_order_in_chapter: int | None = None
    is_chapter_start: bool = False
    is_chapter_end: bool = False


@dataclass(frozen=True)
class ChapterChunkingResult:
    """单个文件的章节识别和文本分块结果。"""

    chapters: list[ChapterSpan]
    chunks: list[TextChunk]


@dataclass(frozen=True)
class _TextSpan:
    start: int
    end: int
    paragraph_index: int


@dataclass(frozen=True)
class _ChunkSpan:
    start: int
    end: int
    paragraph_start: int
    paragraph_end: int


def _validate_chunk_sizes(
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> None:
    if min_chars <= 0:
        raise ValueError("min_chars 必须大于 0。")
    if target_chars <= 0:
        raise ValueError("target_chars 必须大于 0。")
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0。")
    if not min_chars <= target_chars <= max_chars:
        raise ValueError(
            "分块长度必须满足 "
            "min_chars <= target_chars <= max_chars。"
        )


def _extract_paragraph_spans(text: str) -> list[_TextSpan]:
    """识别段落，并把段落后的空行归入前一段。"""

    paragraph_spans: list[_TextSpan] = []
    cursor = 0
    paragraph_index = 1

    for separator in _PARAGRAPH_SEPARATOR_PATTERN.finditer(text):
        if separator.start() > cursor:
            paragraph_spans.append(
                _TextSpan(
                    start=cursor,
                    end=separator.end(),
                    paragraph_index=paragraph_index,
                )
            )
            paragraph_index += 1
        cursor = separator.end()

    if cursor < len(text):
        paragraph_spans.append(
            _TextSpan(
                start=cursor,
                end=len(text),
                paragraph_index=paragraph_index,
            )
        )

    if not paragraph_spans and text:
        paragraph_spans.append(
            _TextSpan(
                start=0,
                end=len(text),
                paragraph_index=1,
            )
        )

    return paragraph_spans


def _is_english_period_boundary(
    text: str,
    index: int,
    span_start: int,
    span_end: int,
) -> bool:
    previous_character = text[index - 1] if index > span_start else ""
    next_character = text[index + 1] if index + 1 < span_end else ""

    if previous_character.isdigit() and next_character.isdigit():
        return False

    prefix = text[span_start:index]
    word_match = re.search(r"([A-Za-z]+)$", prefix)
    previous_word = word_match.group(1).lower() if word_match else ""

    if previous_word in _ENGLISH_ABBREVIATIONS:
        return False

    if len(previous_word) == 1 and previous_word.isalpha():
        return False

    lookahead = index + 1
    while (
        lookahead < span_end
        and text[lookahead] in _CLOSING_PUNCTUATION
    ):
        lookahead += 1

    if lookahead >= span_end:
        return True

    return text[lookahead].isspace()


def _find_sentence_end_positions(
    text: str,
    span_start: int,
    span_end: int,
) -> list[int]:
    end_positions: list[int] = []
    index = span_start

    while index < span_end:
        character = text[index]
        is_boundary = character in "。！？；!?;"

        if character == ".":
            is_boundary = _is_english_period_boundary(
                text=text,
                index=index,
                span_start=span_start,
                span_end=span_end,
            )

        if not is_boundary:
            index += 1
            continue

        end = index + 1

        while end < span_end and text[end] == character:
            end += 1

        while (
            end < span_end
            and text[end] in _CLOSING_PUNCTUATION
        ):
            end += 1

        end_positions.append(end)
        index = end

    return end_positions


def _extract_sentence_spans(
    text: str,
    paragraph_span: _TextSpan,
) -> list[_TextSpan]:
    """在一个段落中识别中英文句子边界。"""

    sentence_spans: list[_TextSpan] = []
    cursor = paragraph_span.start

    for sentence_end in _find_sentence_end_positions(
        text=text,
        span_start=paragraph_span.start,
        span_end=paragraph_span.end,
    ):
        if sentence_end <= cursor:
            continue

        sentence_spans.append(
            _TextSpan(
                start=cursor,
                end=sentence_end,
                paragraph_index=paragraph_span.paragraph_index,
            )
        )
        cursor = sentence_end

    if cursor < paragraph_span.end:
        remaining_text = text[cursor:paragraph_span.end]

        if remaining_text.isspace() and sentence_spans:
            last_sentence = sentence_spans[-1]
            sentence_spans[-1] = _TextSpan(
                start=last_sentence.start,
                end=paragraph_span.end,
                paragraph_index=paragraph_span.paragraph_index,
            )
        else:
            sentence_spans.append(
                _TextSpan(
                    start=cursor,
                    end=paragraph_span.end,
                    paragraph_index=paragraph_span.paragraph_index,
                )
            )

    return sentence_spans


def _hard_split_span(
    span: _TextSpan,
    min_chars: int,
    max_chars: int,
) -> list[_TextSpan]:
    """将超长单句硬切，并平衡过短尾段。"""

    split_spans: list[_TextSpan] = []
    start = span.start

    while start < span.end:
        end = min(start + max_chars, span.end)
        split_spans.append(
            _TextSpan(
                start=start,
                end=end,
                paragraph_index=span.paragraph_index,
            )
        )
        start = end

    if len(split_spans) >= 2:
        previous_span = split_spans[-2]
        last_span = split_spans[-1]
        last_length = last_span.end - last_span.start

        if last_length < min_chars:
            combined_start = previous_span.start
            combined_end = last_span.end
            split_position = (
                combined_start
                + (combined_end - combined_start) // 2
            )

            split_spans[-2:] = [
                _TextSpan(
                    start=combined_start,
                    end=split_position,
                    paragraph_index=span.paragraph_index,
                ),
                _TextSpan(
                    start=split_position,
                    end=combined_end,
                    paragraph_index=span.paragraph_index,
                ),
            ]

    return split_spans


def _split_oversized_paragraph(
    text: str,
    paragraph_span: _TextSpan,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[_TextSpan]:
    sentence_spans = _extract_sentence_spans(
        text=text,
        paragraph_span=paragraph_span,
    )

    if not sentence_spans:
        return _hard_split_span(
            span=paragraph_span,
            min_chars=min_chars,
            max_chars=max_chars,
        )

    atomic_spans: list[_TextSpan] = []

    for sentence_span in sentence_spans:
        sentence_length = sentence_span.end - sentence_span.start

        if sentence_length <= max_chars:
            atomic_spans.append(sentence_span)
        else:
            atomic_spans.extend(
                _hard_split_span(
                    span=sentence_span,
                    min_chars=min_chars,
                    max_chars=max_chars,
                )
            )

    combined_spans: list[_TextSpan] = []
    current_start: int | None = None
    current_end: int | None = None

    for span in atomic_spans:
        if current_start is None or current_end is None:
            current_start = span.start
            current_end = span.end
            continue

        current_length = current_end - current_start
        candidate_length = span.end - current_start

        should_extend = (
            candidate_length <= target_chars
            or (
                current_length < min_chars
                and candidate_length <= max_chars
            )
        )

        if should_extend:
            current_end = span.end
        else:
            combined_spans.append(
                _TextSpan(
                    start=current_start,
                    end=current_end,
                    paragraph_index=paragraph_span.paragraph_index,
                )
            )
            current_start = span.start
            current_end = span.end

    if current_start is not None and current_end is not None:
        combined_spans.append(
            _TextSpan(
                start=current_start,
                end=current_end,
                paragraph_index=paragraph_span.paragraph_index,
            )
        )

    return combined_spans


def _prepare_text_spans(
    text: str,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[_TextSpan]:
    prepared_spans: list[_TextSpan] = []

    for paragraph_span in _extract_paragraph_spans(text):
        paragraph_length = paragraph_span.end - paragraph_span.start

        if paragraph_length <= max_chars:
            prepared_spans.append(paragraph_span)
        else:
            prepared_spans.extend(
                _split_oversized_paragraph(
                    text=text,
                    paragraph_span=paragraph_span,
                    min_chars=min_chars,
                    target_chars=target_chars,
                    max_chars=max_chars,
                )
            )

    return prepared_spans


def _build_chunk_spans(
    spans: list[_TextSpan],
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[_ChunkSpan]:
    if not spans:
        return []

    chunk_spans: list[_ChunkSpan] = []
    current_start = spans[0].start
    current_end = spans[0].end
    paragraph_start = spans[0].paragraph_index
    paragraph_end = spans[0].paragraph_index

    for span in spans[1:]:
        current_length = current_end - current_start
        candidate_length = span.end - current_start

        should_extend = (
            candidate_length <= target_chars
            or (
                current_length < min_chars
                and candidate_length <= max_chars
            )
        )

        if should_extend:
            current_end = span.end
            paragraph_end = span.paragraph_index
        else:
            chunk_spans.append(
                _ChunkSpan(
                    start=current_start,
                    end=current_end,
                    paragraph_start=paragraph_start,
                    paragraph_end=paragraph_end,
                )
            )
            current_start = span.start
            current_end = span.end
            paragraph_start = span.paragraph_index
            paragraph_end = span.paragraph_index

    chunk_spans.append(
        _ChunkSpan(
            start=current_start,
            end=current_end,
            paragraph_start=paragraph_start,
            paragraph_end=paragraph_end,
        )
    )

    if len(chunk_spans) >= 2:
        last_chunk = chunk_spans[-1]
        previous_chunk = chunk_spans[-2]
        last_length = last_chunk.end - last_chunk.start
        merged_length = last_chunk.end - previous_chunk.start

        if last_length < min_chars and merged_length <= max_chars:
            chunk_spans[-2] = _ChunkSpan(
                start=previous_chunk.start,
                end=last_chunk.end,
                paragraph_start=previous_chunk.paragraph_start,
                paragraph_end=last_chunk.paragraph_end,
            )
            chunk_spans.pop()

    return chunk_spans


def chunk_text(
    text: str,
    source_file_name: str,
    source_file_order: int,
    global_order_start: int = 1,
    min_chars: int = DEFAULT_MIN_CHARS,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[TextChunk]:
    """对一段文本进行语言无关的长度分块。"""

    _validate_chunk_sizes(
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
    )

    if global_order_start <= 0:
        raise ValueError("global_order_start 必须大于 0。")
    if source_file_order <= 0:
        raise ValueError("source_file_order 必须大于 0。")
    if not text:
        return []

    prepared_spans = _prepare_text_spans(
        text=text,
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
    )
    chunk_spans = _build_chunk_spans(
        spans=prepared_spans,
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
    )

    chunks: list[TextChunk] = []

    for offset, chunk_span in enumerate(chunk_spans):
        global_order = global_order_start + offset
        chunk_content = text[chunk_span.start:chunk_span.end]

        chunks.append(
            TextChunk(
                chunk_id=f"chunk_{global_order:04d}",
                global_order=global_order,
                source_file_name=source_file_name,
                source_file_order=source_file_order,
                start_character=chunk_span.start,
                end_character=chunk_span.end,
                character_count=len(chunk_content),
                paragraph_start=chunk_span.paragraph_start,
                paragraph_end=chunk_span.paragraph_end,
                text=chunk_content,
            )
        )

    return chunks


def chunk_text_by_chapters(
    text: str,
    source_file_name: str,
    source_file_order: int,
    global_order_start: int = 1,
    min_chars: int = DEFAULT_MIN_CHARS,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ChapterChunkingResult:
    """先识别章节，再保证每个文本块只属于一个章节。"""

    _validate_chunk_sizes(
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
    )

    if global_order_start <= 0:
        raise ValueError("global_order_start 必须大于 0。")
    if source_file_order <= 0:
        raise ValueError("source_file_order 必须大于 0。")
    if not text:
        return ChapterChunkingResult(chapters=[], chunks=[])

    chapters = detect_chapters(text)
    all_chunks: list[TextChunk] = []
    next_global_order = global_order_start

    for chapter in chapters:
        chapter_text = text[
            chapter.start_character:chapter.end_character
        ]
        local_chunks = chunk_text(
            text=chapter_text,
            source_file_name=source_file_name,
            source_file_order=source_file_order,
            global_order_start=next_global_order,
            min_chars=min_chars,
            target_chars=target_chars,
            max_chars=max_chars,
        )

        for local_order, local_chunk in enumerate(
            local_chunks,
            start=1,
        ):
            global_start = (
                chapter.start_character
                + local_chunk.start_character
            )
            global_end = (
                chapter.start_character
                + local_chunk.end_character
            )
            chunk_content = text[global_start:global_end]

            all_chunks.append(
                TextChunk(
                    chunk_id=local_chunk.chunk_id,
                    global_order=local_chunk.global_order,
                    source_file_name=source_file_name,
                    source_file_order=source_file_order,
                    start_character=global_start,
                    end_character=global_end,
                    character_count=len(chunk_content),
                    paragraph_start=local_chunk.paragraph_start,
                    paragraph_end=local_chunk.paragraph_end,
                    text=chunk_content,
                    chapter_order=chapter.chapter_order,
                    chapter_number=chapter.chapter_number,
                    chapter_title=chapter.chapter_title,
                    chapter_full_title=chapter.full_title,
                    chunk_order_in_chapter=local_order,
                    is_chapter_start=local_order == 1,
                    is_chapter_end=local_order == len(local_chunks),
                )
            )

        next_global_order += len(local_chunks)

    return ChapterChunkingResult(
        chapters=chapters,
        chunks=all_chunks,
    )
