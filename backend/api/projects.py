
import sqlite3
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

from backend.services.project_storage import (
    ProjectChunkNotFoundError,
    ProjectFileData,
    ProjectNotFoundError,
    get_project_chunk,
    get_project_summary,
    save_project,
)
from backend.services.text_chunker import (
    chunk_text_by_chapters,
)
from backend.services.text_preprocessor import (
    preprocess_text,
)


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
    """单个文件的文本预处理摘要。"""

    original_character_count: int
    processed_character_count: int
    processed_line_count: int
    preview: str


class ChapterSummary(BaseModel):
    """单个章节的识别和分块摘要。"""

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
    """上传接口返回的文本块摘要。"""

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
    """单个上传文件的处理结果。"""

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
    """项目创建、文本处理和持久化结果。"""

    project_id: str
    project_name: str
    status: str

    file_count: int
    total_size_bytes: int

    total_characters: int
    total_lines: int

    total_processed_characters: int
    total_processed_lines: int

    total_chapters: int
    total_chunks: int

    files: list[UploadedFileSummary]


class StoredTextChunkSummary(BaseModel):
    """已保存文本块的摘要，不包含完整正文。"""

    id: int
    chunk_id: str

    global_order: int
    chunk_order_in_chapter: int

    start_character: int
    end_character: int
    character_count: int

    paragraph_start: int
    paragraph_end: int

    preview: str

    is_chapter_start: bool
    is_chapter_end: bool

    created_at: str


class StoredChapterSummary(BaseModel):
    """已保存章节及其文本块摘要。"""

    id: int

    chapter_order: int
    chapter_number: int | None
    chapter_title: str | None
    full_title: str | None

    part_order: int | None
    part_title: str | None

    volume_order: int | None
    volume_title: str | None

    start_character: int
    end_character: int
    character_count: int

    is_detected: bool
    chunk_count: int

    chunks: list[StoredTextChunkSummary]

    created_at: str


class StoredSourceFileSummary(BaseModel):
    """已保存源文件的摘要。"""

    id: int
    file_order: int
    file_name: str

    size_bytes: int

    original_character_count: int
    original_line_count: int

    processed_character_count: int
    processed_line_count: int

    chapter_count: int
    chunk_count: int

    chapters: list[StoredChapterSummary]

    created_at: str


class ProjectDetailResponse(BaseModel):
    """项目、文件、章节和文本块摘要。"""

    project_id: str
    project_name: str
    status: str

    created_at: str
    updated_at: str

    file_count: int
    chapter_count: int
    chunk_count: int

    files: list[StoredSourceFileSummary]


class ProjectChunkDetailResponse(BaseModel):
    """单个文本块的完整正文和来源信息。"""

    project_id: str
    project_name: str

    source_file_id: int
    source_file_order: int
    source_file_name: str

    chapter_id: int
    chapter_order: int
    chapter_number: int | None
    chapter_title: str | None
    chapter_full_title: str | None

    id: int
    chunk_id: str
    global_order: int
    chunk_order_in_chapter: int

    start_character: int
    end_character: int
    character_count: int

    paragraph_start: int
    paragraph_end: int

    text: str

    is_chapter_start: bool
    is_chapter_end: bool

    created_at: str


@router.post(
    "/upload",
    response_model=ProjectUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传、处理并保存小说项目",
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
        File(
            description=(
                "按照处理顺序上传的 TXT 文件"
            )
        ),
    ],
) -> ProjectUploadResponse:
    """
    创建一个新的小说项目。

    处理过程包括文件校验、文本预处理、章节识别、
    长文本分块以及项目持久化。
    """

    normalized_project_name = (
        project_name.strip()
    )

    if not normalized_project_name:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="项目名称不能为空。",
        )

    if not files:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="请至少上传一个 TXT 文件。",
        )

    if len(files) > MAX_FILE_COUNT:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"每个项目最多上传 "
                f"{MAX_FILE_COUNT} 个文件。"
            ),
        )

    file_summaries: list[
        UploadedFileSummary
    ] = []

    storage_files: list[
        ProjectFileData
    ] = []

    total_size_bytes = 0
    total_characters = 0
    total_lines = 0

    total_processed_characters = 0
    total_processed_lines = 0

    next_global_chunk_order = 1

    for order, uploaded_file in enumerate(
        files,
        start=1,
    ):
        file_name = (
            uploaded_file.filename
            or f"file_{order}.txt"
        )

        file_suffix = (
            Path(file_name).suffix.lower()
        )

        if file_suffix != ".txt":
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    f"文件“{file_name}”"
                    "不是 TXT 文件。"
                ),
            )

        try:
            raw_content = (
                await uploaded_file.read()
            )
        finally:
            await uploaded_file.close()

        file_size = len(raw_content)

        if file_size == 0:
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    f"文件“{file_name}”为空。"
                ),
            )

        if file_size > MAX_SINGLE_FILE_SIZE:
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    f"文件“{file_name}”超过"
                    "单文件 10 MB 的大小限制。"
                ),
            )

        total_size_bytes += file_size

        if (
            total_size_bytes
            > MAX_TOTAL_FILE_SIZE
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "所有文件的总大小不能"
                    "超过 20 MB。"
                ),
            )

        try:
            text = raw_content.decode(
                "utf-8-sig"
            )
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    f"文件“{file_name}”不是"
                    "有效的 UTF-8 编码，请转换"
                    "编码后重新上传。"
                ),
            ) from error

        preprocessed = preprocess_text(text)

        if not preprocessed.text:
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    f"文件“{file_name}”"
                    "不包含有效文本。"
                ),
            )

        chunking_result = (
            chunk_text_by_chapters(
                text=preprocessed.text,
                source_file_name=file_name,
                source_file_order=order,
                global_order_start=(
                    next_global_chunk_order
                ),
            )
        )

        chapters = chunking_result.chapters
        text_chunks = chunking_result.chunks

        if not chapters:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"文件“{file_name}”"
                    "未生成有效章节范围。"
                ),
            )

        if not text_chunks:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"文件“{file_name}”"
                    "未生成有效文本块。"
                ),
            )

        chunks_per_chapter = Counter(
            chunk.chapter_order
            for chunk in text_chunks
        )

        chapter_summaries = [
            ChapterSummary(
                chapter_order=(
                    chapter.chapter_order
                ),
                full_title=chapter.full_title,
                chapter_title=(
                    chapter.chapter_title
                ),
                part_order=chapter.part_order,
                part_title=chapter.part_title,
                volume_order=(
                    chapter.volume_order
                ),
                volume_title=(
                    chapter.volume_title
                ),
                chapter_number=(
                    chapter.chapter_number
                ),
                start_character=(
                    chapter.start_character
                ),
                end_character=(
                    chapter.end_character
                ),
                character_count=(
                    chapter.character_count
                ),
                chunk_count=(
                    chunks_per_chapter[
                        chapter.chapter_order
                    ]
                ),
                is_detected=(
                    chapter.is_detected
                ),
            )
            for chapter in chapters
        ]

        chunk_summaries = [
            TextChunkSummary(
                chunk_id=chunk.chunk_id,
                global_order=(
                    chunk.global_order
                ),
                source_file_name=(
                    chunk.source_file_name
                ),
                source_file_order=(
                    chunk.source_file_order
                ),
                start_character=(
                    chunk.start_character
                ),
                end_character=(
                    chunk.end_character
                ),
                character_count=(
                    chunk.character_count
                ),
                paragraph_start=(
                    chunk.paragraph_start
                ),
                paragraph_end=(
                    chunk.paragraph_end
                ),
                chapter_order=(
                    chunk.chapter_order
                ),
                chapter_number=(
                    chunk.chapter_number
                ),
                chapter_title=(
                    chunk.chapter_title
                ),
                chapter_full_title=(
                    chunk.chapter_full_title
                ),
                chunk_order_in_chapter=(
                    chunk.chunk_order_in_chapter
                ),
                is_chapter_start=(
                    chunk.is_chapter_start
                ),
                is_chapter_end=(
                    chunk.is_chapter_end
                ),
                preview=chunk.text[
                    :CHUNK_PREVIEW_CHARACTER_LIMIT
                ],
            )
            for chunk in text_chunks
        ]

        original_character_count = len(text)

        original_line_count = (
            text.count("\n") + 1
            if text
            else 0
        )

        file_summaries.append(
            UploadedFileSummary(
                order=order,
                file_name=file_name,
                size_bytes=file_size,
                character_count=(
                    original_character_count
                ),
                line_count=(
                    original_line_count
                ),
                preprocessing=(
                    TextPreprocessingSummary(
                        original_character_count=(
                            preprocessed
                            .original_character_count
                        ),
                        processed_character_count=(
                            preprocessed
                            .processed_character_count
                        ),
                        processed_line_count=(
                            preprocessed
                            .processed_line_count
                        ),
                        preview=(
                            preprocessed.text[
                                :PREVIEW_CHARACTER_LIMIT
                            ]
                        ),
                    )
                ),
                chapter_count=len(chapters),
                chapters=chapter_summaries,
                chunk_count=len(text_chunks),
                chunks=chunk_summaries,
            )
        )

        storage_files.append(
            ProjectFileData(
                file_order=order,
                file_name=file_name,
                raw_content=raw_content,
                original_text=text,
                processed_text=(
                    preprocessed.text
                ),
                chapters=chapters,
                chunks=text_chunks,
            )
        )

        next_global_chunk_order += len(
            text_chunks
        )

        total_characters += (
            original_character_count
        )

        total_lines += original_line_count

        total_processed_characters += (
            preprocessed
            .processed_character_count
        )

        total_processed_lines += (
            preprocessed.processed_line_count
        )

    try:
        saved_project = save_project(
            project_name=(
                normalized_project_name
            ),
            files=storage_files,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "项目数据校验失败，"
                "未保存项目。"
            ),
        ) from error
    except (
        sqlite3.Error,
        OSError,
    ) as error:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "项目保存失败，"
                "请稍后重新尝试。"
            ),
        ) from error

    return ProjectUploadResponse(
        project_id=saved_project.project_id,
        project_name=(
            saved_project.project_name
        ),
        status=saved_project.status,
        file_count=saved_project.file_count,
        total_size_bytes=total_size_bytes,
        total_characters=total_characters,
        total_lines=total_lines,
        total_processed_characters=(
            total_processed_characters
        ),
        total_processed_lines=(
            total_processed_lines
        ),
        total_chapters=(
            saved_project.chapter_count
        ),
        total_chunks=(
            saved_project.chunk_count
        ),
        files=file_summaries,
    )


@router.get(
    "/{project_id}",
    response_model=ProjectDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="查询小说项目摘要",
)
def read_project(
    project_id: str,
) -> ProjectDetailResponse:
    """
    查询已保存项目的摘要。

    返回项目、文件、章节和文本块摘要，
    不返回文本块完整正文。
    """

    normalized_project_id = (
        project_id.strip()
    )

    if not normalized_project_id:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="project_id 不能为空。",
        )

    try:
        project_data = get_project_summary(
            normalized_project_id
        )
    except ProjectNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="项目不存在。",
        ) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "读取项目失败，"
                "请稍后重新尝试。"
            ),
        ) from error

    return ProjectDetailResponse.model_validate(
        project_data
    )


@router.get(
    "/{project_id}/chunks/{chunk_id}",
    response_model=ProjectChunkDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="查询指定文本块详情",
)
def read_project_chunk(
    project_id: str,
    chunk_id: str,
) -> ProjectChunkDetailResponse:
    """读取指定项目中某个文本块的完整正文。"""

    normalized_project_id = (
        project_id.strip()
    )

    normalized_chunk_id = (
        chunk_id.strip()
    )

    if not normalized_project_id:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="project_id 不能为空。",
        )

    if not normalized_chunk_id:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="chunk_id 不能为空。",
        )

    try:
        chunk_data = get_project_chunk(
            project_id=(
                normalized_project_id
            ),
            chunk_id=normalized_chunk_id,
        )
    except ProjectNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="项目不存在。",
        ) from error
    except (
        ProjectChunkNotFoundError
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="文本块不存在。",
        ) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "读取文本块失败，"
                "请稍后重新尝试。"
            ),
        ) from error

    return (
        ProjectChunkDetailResponse
        .model_validate(chunk_data)
    )

