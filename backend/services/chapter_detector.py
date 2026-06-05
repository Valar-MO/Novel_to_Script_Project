import re
from dataclasses import dataclass


MAX_TITLE_CHARS = 100
MIN_BODY_CHARS_BETWEEN_BARE_CHAPTERS = 120
MIN_BODY_LINES_BETWEEN_BARE_CHAPTERS = 2
ADJACENT_LIST_MAX_CHARS = 50
BARE_CHAPTER_ACCEPT_SCORE = 4

_CHINESE_NUMBER_CHARS = "零〇一二三四五六七八九十百千万两"
_NUMBER_TOKEN_PATTERN = rf"[{_CHINESE_NUMBER_CHARS}\d]+"

_CHINESE_HIERARCHICAL_PATTERN = re.compile(
    rf"^第(?P<part_number>{_NUMBER_TOKEN_PATTERN})部\s*"
    rf"(?P<part_title>.+?)\s+"
    rf"第(?P<volume_number>{_NUMBER_TOKEN_PATTERN})卷\s*"
    rf"(?P<volume_title>.+?)\s+"
    rf"(?P<chapter_number>{_NUMBER_TOKEN_PATTERN})"
    rf"[、.．\s]+(?P<chapter_title>.+)$"
)

_ENGLISH_HIERARCHICAL_PATTERN = re.compile(
    r"^PART\s+(?P<part_number>[A-Z0-9-]+)\s*[:\-]\s*"
    r"(?P<part_title>.+?)\s+"
    r"BOOK\s+(?P<volume_number>[A-Z0-9-]+)\s*[:\-]\s*"
    r"(?P<volume_title>.+?)\s+"
    r"CHAPTER\s+(?P<chapter_number>[A-Z0-9-]+)\s*[:\-]\s*"
    r"(?P<chapter_title>.+)$",
    re.IGNORECASE,
)

_CHINESE_EXPLICIT_CHAPTER_PATTERN = re.compile(
    rf"^第(?P<number>{_NUMBER_TOKEN_PATTERN})"
    r"(?P<kind>章|回|节|幕)"
    r"(?:(?:\s+|[:：\-—]\s*)(?P<title>.*))?$"
)

_ENGLISH_EXPLICIT_CHAPTER_PATTERN = re.compile(
    r"^CHAPTER\s+(?P<number>[A-Z0-9-]+)"
    r"(?:\s*[:.\-]\s*|\s+)?(?P<title>.*)$",
    re.IGNORECASE,
)

_CHINESE_STRUCTURE_PATTERN = re.compile(
    rf"^第(?P<number>{_NUMBER_TOKEN_PATTERN})"
    r"(?P<kind>部|卷|篇|集)\s*(?P<title>.*)$"
)

_ENGLISH_STRUCTURE_PATTERN = re.compile(
    r"^(?P<kind>PART|BOOK)\s+(?P<number>[A-Z0-9-]+)"
    r"(?:\s*[:.\-]\s*|\s+)?(?P<title>.*)$",
    re.IGNORECASE,
)

_CHINESE_SPECIAL_PATTERN = re.compile(
    rf"^(?P<label>序章|楔子|引子|前言|序言|尾声|终章|后记|番外(?:{_NUMBER_TOKEN_PATTERN})?)"
    r"(?:\s*[:：\-—]?\s*(?P<title>.*))?$"
)

_ENGLISH_SPECIAL_PATTERN = re.compile(
    r"^(?P<label>PROLOGUE|EPILOGUE|INTRODUCTION|PRELUDE|AFTERWORD|APPENDIX)"
    r"(?:\s*[:\-—]?\s*(?P<title>.*))?$",
    re.IGNORECASE,
)

_CHINESE_BARE_NUMBERED_PATTERN = re.compile(
    rf"^(?P<number>[{_CHINESE_NUMBER_CHARS}]+)"
    r"[、.．\s]+(?P<title>\S.*)$"
)

_ENGLISH_BARE_NUMBERED_PATTERN = re.compile(
    r"^(?P<number>(?:[IVXLCDM]+|[A-Z-]+|\d+))"
    r"[.)、.．\s]+(?P<title>\S.*)$",
    re.IGNORECASE,
)

_QUANTITY_PATTERN = re.compile(
    rf"[{_CHINESE_NUMBER_CHARS}\d]+\s*"
    r"(斤|两|件|个|只|匹|袋|箱|套|枚|元|块|份|米|升|瓶|包|磅|英里|美元|欧元|公斤|克)"
)

_ROMAN_PATTERN = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)

_ENGLISH_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}


@dataclass(frozen=True)
class TextLine:
    line_index: int
    start: int
    end: int
    content: str
    stripped_content: str
    is_blank: bool


@dataclass(frozen=True)
class HeadingCandidate:
    line_index: int
    heading_start: int
    heading_end: int
    boundary_start: int
    full_title: str
    candidate_type: str
    chapter_number: int | None
    chapter_title: str | None
    part_order: int | None
    part_title: str | None
    volume_order: int | None
    volume_title: str | None
    is_direct: bool


@dataclass(frozen=True)
class ChapterSpan:
    chapter_order: int
    full_title: str | None
    chapter_title: str | None
    part_order: int | None
    part_title: str | None
    volume_order: int | None
    volume_title: str | None
    chapter_number: int | None
    start_character: int
    end_character: int
    character_count: int
    is_detected: bool


def _extract_lines(text: str) -> list[TextLine]:
    lines: list[TextLine] = []
    cursor = 0

    for line_index, raw_line in enumerate(
        text.splitlines(keepends=True),
        start=1,
    ):
        end = cursor + len(raw_line)
        content = raw_line.rstrip("\r\n")
        stripped_content = content.strip()

        lines.append(
            TextLine(
                line_index=line_index,
                start=cursor,
                end=end,
                content=content,
                stripped_content=stripped_content,
                is_blank=not stripped_content,
            )
        )
        cursor = end

    if cursor < len(text):
        content = text[cursor:]
        stripped_content = content.strip()
        lines.append(
            TextLine(
                line_index=len(lines) + 1,
                start=cursor,
                end=len(text),
                content=content,
                stripped_content=stripped_content,
                is_blank=not stripped_content,
            )
        )

    if not lines and text:
        stripped_content = text.strip()
        lines.append(
            TextLine(
                line_index=1,
                start=0,
                end=len(text),
                content=text,
                stripped_content=stripped_content,
                is_blank=not stripped_content,
            )
        )

    return lines


def _parse_chinese_number(token: str) -> int | None:
    if token.isdigit():
        return int(token)

    digit_map = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    unit_map = {
        "十": 10,
        "百": 100,
        "千": 1000,
        "万": 10000,
    }

    if all(character in digit_map for character in token):
        digits = "".join(str(digit_map[character]) for character in token)
        return int(digits) if digits else None

    total = 0
    section = 0
    current_number = 0

    for character in token:
        if character in digit_map:
            current_number = digit_map[character]
            continue

        unit = unit_map.get(character)
        if unit is None:
            return None

        if unit == 10000:
            section += current_number
            total += section * unit
            section = 0
            current_number = 0
        else:
            if current_number == 0:
                current_number = 1
            section += current_number * unit
            current_number = 0

    return total + section + current_number


def _parse_roman_number(token: str) -> int | None:
    if not _ROMAN_PATTERN.fullmatch(token):
        return None

    values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    total = 0
    previous = 0

    for character in reversed(token.upper()):
        value = values[character]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value

    return total if total > 0 else None


def _parse_english_number(token: str) -> int | None:
    normalized = token.strip().lower().replace("-", " ")

    if normalized.isdigit():
        return int(normalized)

    roman_number = _parse_roman_number(normalized)
    if roman_number is not None:
        return roman_number

    words = normalized.split()
    if not words:
        return None

    if len(words) == 1:
        return _ENGLISH_NUMBER_WORDS.get(words[0])

    if len(words) == 2:
        first = _ENGLISH_NUMBER_WORDS.get(words[0])
        second = _ENGLISH_NUMBER_WORDS.get(words[1])
        if first is not None and second is not None and first >= 20 and second < 10:
            return first + second

    return None


def _parse_number_token(token: str) -> int | None:
    normalized = token.strip()

    if not normalized:
        return None

    chinese_number = _parse_chinese_number(normalized)
    if chinese_number is not None:
        return chinese_number

    return _parse_english_number(normalized)


def _clean_optional_title(title: str | None) -> str | None:
    if title is None:
        return None

    cleaned = title.strip(" \t:：.-—")
    return cleaned or None


def _is_short_heading_line(line: TextLine) -> bool:
    return (
        bool(line.stripped_content)
        and len(line.stripped_content) <= MAX_TITLE_CHARS
    )


def _extract_heading_candidates(
    lines: list[TextLine],
) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []

    current_part_order: int | None = None
    current_part_title: str | None = None
    current_volume_order: int | None = None
    current_volume_title: str | None = None
    pending_structure_start: int | None = None

    for line in lines:
        title = line.stripped_content

        if line.is_blank or not _is_short_heading_line(line):
            continue

        hierarchical_match = _CHINESE_HIERARCHICAL_PATTERN.fullmatch(title)
        if hierarchical_match:
            part_order = _parse_number_token(
                hierarchical_match.group("part_number")
            )
            volume_order = _parse_number_token(
                hierarchical_match.group("volume_number")
            )
            chapter_number = _parse_number_token(
                hierarchical_match.group("chapter_number")
            )

            candidates.append(
                HeadingCandidate(
                    line_index=line.line_index,
                    heading_start=line.start,
                    heading_end=line.end,
                    boundary_start=(
                        pending_structure_start
                        if pending_structure_start is not None
                        else line.start
                    ),
                    full_title=title,
                    candidate_type="hierarchical_chapter",
                    chapter_number=chapter_number,
                    chapter_title=_clean_optional_title(
                        hierarchical_match.group("chapter_title")
                    ),
                    part_order=part_order,
                    part_title=_clean_optional_title(
                        hierarchical_match.group("part_title")
                    ),
                    volume_order=volume_order,
                    volume_title=_clean_optional_title(
                        hierarchical_match.group("volume_title")
                    ),
                    is_direct=True,
                )
            )

            current_part_order = part_order
            current_part_title = _clean_optional_title(
                hierarchical_match.group("part_title")
            )
            current_volume_order = volume_order
            current_volume_title = _clean_optional_title(
                hierarchical_match.group("volume_title")
            )
            pending_structure_start = None
            continue

        english_hierarchical_match = _ENGLISH_HIERARCHICAL_PATTERN.fullmatch(
            title
        )
        if english_hierarchical_match:
            part_order = _parse_number_token(
                english_hierarchical_match.group("part_number")
            )
            volume_order = _parse_number_token(
                english_hierarchical_match.group("volume_number")
            )
            chapter_number = _parse_number_token(
                english_hierarchical_match.group("chapter_number")
            )

            part_title = _clean_optional_title(
                english_hierarchical_match.group("part_title")
            )
            volume_title = _clean_optional_title(
                english_hierarchical_match.group("volume_title")
            )

            candidates.append(
                HeadingCandidate(
                    line_index=line.line_index,
                    heading_start=line.start,
                    heading_end=line.end,
                    boundary_start=(
                        pending_structure_start
                        if pending_structure_start is not None
                        else line.start
                    ),
                    full_title=title,
                    candidate_type="hierarchical_chapter",
                    chapter_number=chapter_number,
                    chapter_title=_clean_optional_title(
                        english_hierarchical_match.group("chapter_title")
                    ),
                    part_order=part_order,
                    part_title=part_title,
                    volume_order=volume_order,
                    volume_title=volume_title,
                    is_direct=True,
                )
            )

            current_part_order = part_order
            current_part_title = part_title
            current_volume_order = volume_order
            current_volume_title = volume_title
            pending_structure_start = None
            continue

        chinese_structure_match = _CHINESE_STRUCTURE_PATTERN.fullmatch(title)
        if chinese_structure_match:
            structure_order = _parse_number_token(
                chinese_structure_match.group("number")
            )
            structure_title = _clean_optional_title(
                chinese_structure_match.group("title")
            )
            structure_kind = chinese_structure_match.group("kind")

            if structure_kind == "部":
                current_part_order = structure_order
                current_part_title = structure_title
                current_volume_order = None
                current_volume_title = None
                pending_structure_start = line.start
            else:
                current_volume_order = structure_order
                current_volume_title = structure_title
                if pending_structure_start is None:
                    pending_structure_start = line.start
            continue

        english_structure_match = _ENGLISH_STRUCTURE_PATTERN.fullmatch(title)
        if english_structure_match:
            structure_order = _parse_number_token(
                english_structure_match.group("number")
            )
            structure_title = _clean_optional_title(
                english_structure_match.group("title")
            )
            structure_kind = english_structure_match.group("kind").upper()

            if structure_kind == "PART":
                current_part_order = structure_order
                current_part_title = structure_title
                current_volume_order = None
                current_volume_title = None
                pending_structure_start = line.start
            else:
                current_volume_order = structure_order
                current_volume_title = structure_title
                if pending_structure_start is None:
                    pending_structure_start = line.start
            continue

        explicit_chinese_match = _CHINESE_EXPLICIT_CHAPTER_PATTERN.fullmatch(
            title
        )
        if explicit_chinese_match:
            candidates.append(
                HeadingCandidate(
                    line_index=line.line_index,
                    heading_start=line.start,
                    heading_end=line.end,
                    boundary_start=(
                        pending_structure_start
                        if pending_structure_start is not None
                        else line.start
                    ),
                    full_title=title,
                    candidate_type="explicit_chapter",
                    chapter_number=_parse_number_token(
                        explicit_chinese_match.group("number")
                    ),
                    chapter_title=_clean_optional_title(
                        explicit_chinese_match.group("title")
                    ),
                    part_order=current_part_order,
                    part_title=current_part_title,
                    volume_order=current_volume_order,
                    volume_title=current_volume_title,
                    is_direct=True,
                )
            )
            pending_structure_start = None
            continue

        explicit_english_match = _ENGLISH_EXPLICIT_CHAPTER_PATTERN.fullmatch(
            title
        )
        if explicit_english_match:
            candidates.append(
                HeadingCandidate(
                    line_index=line.line_index,
                    heading_start=line.start,
                    heading_end=line.end,
                    boundary_start=(
                        pending_structure_start
                        if pending_structure_start is not None
                        else line.start
                    ),
                    full_title=title,
                    candidate_type="explicit_chapter",
                    chapter_number=_parse_number_token(
                        explicit_english_match.group("number")
                    ),
                    chapter_title=_clean_optional_title(
                        explicit_english_match.group("title")
                    ),
                    part_order=current_part_order,
                    part_title=current_part_title,
                    volume_order=current_volume_order,
                    volume_title=current_volume_title,
                    is_direct=True,
                )
            )
            pending_structure_start = None
            continue

        chinese_special_match = _CHINESE_SPECIAL_PATTERN.fullmatch(title)
        if chinese_special_match:
            chapter_title = _clean_optional_title(
                chinese_special_match.group("title")
            )
            if chapter_title is None:
                chapter_title = chinese_special_match.group("label")

            candidates.append(
                HeadingCandidate(
                    line_index=line.line_index,
                    heading_start=line.start,
                    heading_end=line.end,
                    boundary_start=(
                        pending_structure_start
                        if pending_structure_start is not None
                        else line.start
                    ),
                    full_title=title,
                    candidate_type="special_chapter",
                    chapter_number=None,
                    chapter_title=chapter_title,
                    part_order=current_part_order,
                    part_title=current_part_title,
                    volume_order=current_volume_order,
                    volume_title=current_volume_title,
                    is_direct=True,
                )
            )
            pending_structure_start = None
            continue

        english_special_match = _ENGLISH_SPECIAL_PATTERN.fullmatch(title)
        if english_special_match:
            chapter_title = _clean_optional_title(
                english_special_match.group("title")
            )
            if chapter_title is None:
                chapter_title = english_special_match.group("label").title()

            candidates.append(
                HeadingCandidate(
                    line_index=line.line_index,
                    heading_start=line.start,
                    heading_end=line.end,
                    boundary_start=(
                        pending_structure_start
                        if pending_structure_start is not None
                        else line.start
                    ),
                    full_title=title,
                    candidate_type="special_chapter",
                    chapter_number=None,
                    chapter_title=chapter_title,
                    part_order=current_part_order,
                    part_title=current_part_title,
                    volume_order=current_volume_order,
                    volume_title=current_volume_title,
                    is_direct=True,
                )
            )
            pending_structure_start = None
            continue

        bare_match = _CHINESE_BARE_NUMBERED_PATTERN.fullmatch(title)
        if bare_match is None:
            bare_match = _ENGLISH_BARE_NUMBERED_PATTERN.fullmatch(title)

        if bare_match:
            chapter_number = _parse_number_token(
                bare_match.group("number")
            )
            if chapter_number is None:
                continue

            candidates.append(
                HeadingCandidate(
                    line_index=line.line_index,
                    heading_start=line.start,
                    heading_end=line.end,
                    boundary_start=(
                        pending_structure_start
                        if pending_structure_start is not None
                        else line.start
                    ),
                    full_title=title,
                    candidate_type="bare_numbered",
                    chapter_number=chapter_number,
                    chapter_title=_clean_optional_title(
                        bare_match.group("title")
                    ),
                    part_order=current_part_order,
                    part_title=current_part_title,
                    volume_order=current_volume_order,
                    volume_title=current_volume_title,
                    is_direct=False,
                )
            )
            pending_structure_start = None

    return candidates


def _same_structure(
    first: HeadingCandidate,
    second: HeadingCandidate,
) -> bool:
    return (
        first.part_order == second.part_order
        and first.volume_order == second.volume_order
        and first.part_title == second.part_title
        and first.volume_title == second.volume_title
    )


def _body_metrics_between(
    text: str,
    lines: list[TextLine],
    first: HeadingCandidate,
    second: HeadingCandidate,
) -> tuple[int, int]:
    body_text = text[first.heading_end:second.boundary_start]
    body_character_count = len(body_text.strip())

    body_line_count = sum(
        1
        for line in lines
        if first.line_index < line.line_index < second.line_index
        and not line.is_blank
    )

    return body_character_count, body_line_count


def _has_blank_around(
    lines: list[TextLine],
    candidate: HeadingCandidate,
) -> bool:
    line_position = candidate.line_index - 1

    previous_blank = (
        line_position == 0
        or lines[line_position - 1].is_blank
    )
    next_blank = (
        line_position == len(lines) - 1
        or lines[line_position + 1].is_blank
    )

    return previous_blank and next_blank


def _validate_bare_candidate(
    text: str,
    lines: list[TextLine],
    candidate: HeadingCandidate,
    previous_candidate: HeadingCandidate | None,
    next_candidate: HeadingCandidate | None,
) -> bool:
    score = 0

    if (
        candidate.part_order is not None
        or candidate.volume_order is not None
    ):
        score += 3

    has_sequential_neighbor = False
    has_sufficient_body = False
    has_sufficient_body_lines = False
    is_adjacent_list = False

    for neighbor, direction in (
        (previous_candidate, "previous"),
        (next_candidate, "next"),
    ):
        if neighbor is None or not _same_structure(candidate, neighbor):
            continue

        if candidate.chapter_number is None or neighbor.chapter_number is None:
            continue

        expected_difference = 1 if direction == "next" else -1
        number_difference = neighbor.chapter_number - candidate.chapter_number

        if number_difference == expected_difference:
            has_sequential_neighbor = True

        if direction == "next":
            body_chars, body_lines = _body_metrics_between(
                text,
                lines,
                candidate,
                neighbor,
            )
        else:
            body_chars, body_lines = _body_metrics_between(
                text,
                lines,
                neighbor,
                candidate,
            )

        if body_chars >= MIN_BODY_CHARS_BETWEEN_BARE_CHAPTERS:
            has_sufficient_body = True

        if body_lines >= MIN_BODY_LINES_BETWEEN_BARE_CHAPTERS:
            has_sufficient_body_lines = True

        if (
            body_chars <= ADJACENT_LIST_MAX_CHARS
            and body_lines == 0
        ):
            is_adjacent_list = True

    if has_sequential_neighbor:
        score += 2

    if has_sufficient_body:
        score += 2

    if has_sufficient_body_lines:
        score += 1

    if _has_blank_around(lines, candidate):
        score += 1

    if is_adjacent_list:
        score -= 4

    if candidate.chapter_title and _QUANTITY_PATTERN.search(
        candidate.chapter_title
    ):
        score -= 2

    if len(candidate.full_title) > 70:
        score -= 3

    return score >= BARE_CHAPTER_ACCEPT_SCORE


def _validate_heading_candidates(
    text: str,
    lines: list[TextLine],
    candidates: list[HeadingCandidate],
) -> list[HeadingCandidate]:
    accepted: list[HeadingCandidate] = []
    bare_candidates = [
        candidate
        for candidate in candidates
        if candidate.candidate_type == "bare_numbered"
    ]

    bare_index_map = {
        id(candidate): index
        for index, candidate in enumerate(bare_candidates)
    }

    for candidate in candidates:
        if candidate.is_direct:
            accepted.append(candidate)
            continue

        bare_index = bare_index_map[id(candidate)]
        previous_candidate = (
            bare_candidates[bare_index - 1]
            if bare_index > 0
            else None
        )
        next_candidate = (
            bare_candidates[bare_index + 1]
            if bare_index + 1 < len(bare_candidates)
            else None
        )

        if _validate_bare_candidate(
            text=text,
            lines=lines,
            candidate=candidate,
            previous_candidate=previous_candidate,
            next_candidate=next_candidate,
        ):
            accepted.append(candidate)

    accepted.sort(key=lambda candidate: candidate.boundary_start)

    deduplicated: list[HeadingCandidate] = []
    seen_starts: set[int] = set()

    for candidate in accepted:
        if candidate.boundary_start in seen_starts:
            continue
        seen_starts.add(candidate.boundary_start)
        deduplicated.append(candidate)

    return deduplicated


def _build_chapter_spans(
    text: str,
    accepted_headings: list[HeadingCandidate],
) -> list[ChapterSpan]:
    if not text:
        return []

    if not accepted_headings:
        return [
            ChapterSpan(
                chapter_order=1,
                full_title=None,
                chapter_title=None,
                part_order=None,
                part_title=None,
                volume_order=None,
                volume_title=None,
                chapter_number=None,
                start_character=0,
                end_character=len(text),
                character_count=len(text),
                is_detected=False,
            )
        ]

    spans: list[ChapterSpan] = []
    chapter_order = 1
    first_start = accepted_headings[0].boundary_start

    if first_start > 0:
        spans.append(
            ChapterSpan(
                chapter_order=chapter_order,
                full_title=None,
                chapter_title=None,
                part_order=None,
                part_title=None,
                volume_order=None,
                volume_title=None,
                chapter_number=None,
                start_character=0,
                end_character=first_start,
                character_count=first_start,
                is_detected=False,
            )
        )
        chapter_order += 1

    for index, heading in enumerate(accepted_headings):
        start = heading.boundary_start
        end = (
            accepted_headings[index + 1].boundary_start
            if index + 1 < len(accepted_headings)
            else len(text)
        )

        if end <= start:
            continue

        spans.append(
            ChapterSpan(
                chapter_order=chapter_order,
                full_title=heading.full_title,
                chapter_title=heading.chapter_title,
                part_order=heading.part_order,
                part_title=heading.part_title,
                volume_order=heading.volume_order,
                volume_title=heading.volume_title,
                chapter_number=heading.chapter_number,
                start_character=start,
                end_character=end,
                character_count=end - start,
                is_detected=True,
            )
        )
        chapter_order += 1

    return spans


def detect_chapters(text: str) -> list[ChapterSpan]:
    """识别中英文小说中的高置信度章节范围。"""

    if not text:
        return []

    lines = _extract_lines(text)
    candidates = _extract_heading_candidates(lines)
    accepted_headings = _validate_heading_candidates(
        text=text,
        lines=lines,
        candidates=candidates,
    )

    return _build_chapter_spans(
        text=text,
        accepted_headings=accepted_headings,
    )
