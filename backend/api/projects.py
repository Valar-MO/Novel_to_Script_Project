from collections import Counter
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile as FastAPIUploadFile,
    status,
)
from pydantic import BaseModel, WithJsonSchema

from backend.services.text_chunker import chunk_text_by_chapters
from backend.services.text_preprocessor import preprocess_text


router = APIRouter(
    prefix="/api/projects",
    tags=["projects"],
)

MAX_FILE_COUNT = 50
MAX_SINGLE_FILE_SIZE = 10 * 1024 * 1024
MAX_TOTAL_FILE_SIZE = 20 * 1024 * 1024
PREVIEW_CHARACTER_LIMIT = 300
CHUNK_PREVIEW_CHARACTER_LIMIT = 200


BinaryUploadFile = Annotated[
    FastAPIUploadFile,
    WithJsonSchema(
        {
            "type": "string",
            "format": "binary",
        }
    ),
]


class TextPreprocessingSummary(BaseModel):
    original_character_count: int
    processed_character_count: int
    processed_line_count: int
    preview: str


class ChapterSummary(BaseModel):
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
    chunk_count: int
    is_detected: bool


class TextChunkSummary(BaseModel):
    chunk_id: str
    global_order: int
    source_file_name: str
    source_file_order: int
    start_character: int
    end_character: int
    character_count: int
    paragraph_start: int
    paragraph_end: int
    chapter_order: int | None
    chapter_number: int | None
    chapter_title: str | None
    chapter_full_title: str | None
    chunk_order_in_chapter: int | None
    is_chapter_start: bool
    is_chapter_end: bool
    preview: str


class UploadedFileSummary(BaseModel):
    order: int
    file_name: str
    size_bytes: int
    character_count: int
    line_count: int
    preprocessing: TextPreprocessingSummary
    chapter_count: int
    chapters: list[ChapterSummary]
    chunk_count: int
    chunks: list[TextChunkSummary]


class ProjectUploadResponse(BaseModel):
    project_name: str
    file_count: int
    total_size_bytes: int
    total_characters: int
    total_lines: int
    total_processed_characters: int
    total_processed_lines: int
    total_chapters: int
    total_chunks: int
    files: list[UploadedFileSummary]


@router.post(
    "/upload",
    response_model=ProjectUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="上传、识别章节并分块小说 TXT 文件",
)
async def upload_project_files(
    project_name: Annotated[
        str,
        Form(
            min_length=1,
            max_length=100,
            description="小说改编项目名称",
        ),
    ],
    files: Annotated[
        list[BinaryUploadFile],
        File(description="按照处理顺序上传的 TXT 文件"),
    ],
) -> ProjectUploadResponse:
    """校验、预处理、识别章节并对小说文本进行可追溯分块。"""

    normalized_project_name = project_name.strip()

    if not normalized_project_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="项目名称不能为空。",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请至少上传一个 TXT 文件。",
        )

    if len(files) > MAX_FILE_COUNT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"每个项目最多上传 {MAX_FILE_COUNT} 个文件。",
        )

    file_summaries: list[UploadedFileSummary] = []
    total_size_bytes = 0
    total_characters = 0
    total_lines = 0
    total_processed_characters = 0
    total_processed_lines = 0
    total_chapters = 0
    total_chunks = 0
    next_global_chunk_order = 1

    for order, uploaded_file in enumerate(files, start=1):
        file_name = uploaded_file.filename or f"file_{order}.txt"
        file_suffix = Path(file_name).suffix.lower()

        if file_suffix != ".txt":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件“{file_name}”不是 TXT 文件。",
            )

        try:
            raw_content = await uploaded_file.read()
        finally:
            await uploaded_file.close()

        file_size = len(raw_content)

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件“{file_name}”为空。",
            )

        if file_size > MAX_SINGLE_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"文件“{file_name}”超过单文件 "
                    "10 MB 的大小限制。"
                ),
            )

        total_size_bytes += file_size

        if total_size_bytes > MAX_TOTAL_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="所有文件的总大小不能超过 20 MB。",
            )

        try:
            text = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"文件“{file_name}”不是有效的 UTF-8 编码，"
                    "请转换编码后重新上传。"
                ),
            ) from error

        preprocessed = preprocess_text(text)

        if not preprocessed.text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件“{file_name}”不包含有效文本。",
            )

        chunking_result = chunk_text_by_chapters(
            text=preprocessed.text,
            source_file_name=file_name,
            source_file_order=order,
            global_order_start=next_global_chunk_order,
        )

        chapters = chunking_result.chapters
        text_chunks = chunking_result.chunks
        chunks_per_chapter = Counter(
            chunk.chapter_order
            for chunk in text_chunks
        )

        chapter_summaries = [
            ChapterSummary(
                chapter_order=chapter.chapter_order,
                full_title=chapter.full_title,
                chapter_title=chapter.chapter_title,
                part_order=chapter.part_order,
                part_title=chapter.part_title,
                volume_order=chapter.volume_order,
                volume_title=chapter.volume_title,
                chapter_number=chapter.chapter_number,
                start_character=chapter.start_character,
                end_character=chapter.end_character,
                character_count=chapter.character_count,
                chunk_count=chunks_per_chapter[chapter.chapter_order],
                is_detected=chapter.is_detected,
            )
            for chapter in chapters
        ]

        chunk_summaries = [
            TextChunkSummary(
                chunk_id=chunk.chunk_id,
                global_order=chunk.global_order,
                source_file_name=chunk.source_file_name,
                source_file_order=chunk.source_file_order,
                start_character=chunk.start_character,
                end_character=chunk.end_character,
                character_count=chunk.character_count,
                paragraph_start=chunk.paragraph_start,
                paragraph_end=chunk.paragraph_end,
                chapter_order=chunk.chapter_order,
                chapter_number=chunk.chapter_number,
                chapter_title=chunk.chapter_title,
                chapter_full_title=chunk.chapter_full_title,
                chunk_order_in_chapter=chunk.chunk_order_in_chapter,
                is_chapter_start=chunk.is_chapter_start,
                is_chapter_end=chunk.is_chapter_end,
                preview=chunk.text[:CHUNK_PREVIEW_CHARACTER_LIMIT],
            )
            for chunk in text_chunks
        ]

        original_character_count = len(text)
        original_line_count = text.count("\n") + 1

        file_summaries.append(
            UploadedFileSummary(
                order=order,
                file_name=file_name,
                size_bytes=file_size,
                character_count=original_character_count,
                line_count=original_line_count,
                preprocessing=TextPreprocessingSummary(
                    original_character_count=(
                        preprocessed.original_character_count
                    ),
                    processed_character_count=(
                        preprocessed.processed_character_count
                    ),
                    processed_line_count=(
                        preprocessed.processed_line_count
                    ),
                    preview=preprocessed.text[:PREVIEW_CHARACTER_LIMIT],
                ),
                chapter_count=len(chapters),
                chapters=chapter_summaries,
                chunk_count=len(text_chunks),
                chunks=chunk_summaries,
            )
        )

        next_global_chunk_order += len(text_chunks)
        total_characters += original_character_count
        total_lines += original_line_count
        total_processed_characters += (
            preprocessed.processed_character_count
        )
        total_processed_lines += preprocessed.processed_line_count
        total_chapters += len(chapters)
        total_chunks += len(text_chunks)

    return ProjectUploadResponse(
        project_name=normalized_project_name,
        file_count=len(file_summaries),
        total_size_bytes=total_size_bytes,
        total_characters=total_characters,
        total_lines=total_lines,
        total_processed_characters=total_processed_characters,
        total_processed_lines=total_processed_lines,
        total_chapters=total_chapters,
        total_chunks=total_chunks,
        files=file_summaries,
    )
