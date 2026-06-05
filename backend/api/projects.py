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


router = APIRouter(
    prefix="/api/projects",
    tags=["projects"],
)

MAX_FILE_COUNT = 50
MAX_SINGLE_FILE_SIZE = 10 * 1024 * 1024
MAX_TOTAL_FILE_SIZE = 20 * 1024 * 1024


# 兼容新版 FastAPI 与 Swagger UI 的文件选择器显示问题。
BinaryUploadFile = Annotated[
    FastAPIUploadFile,
    WithJsonSchema(
        {
            "type": "string",
            "format": "binary",
        }
    ),
]


class UploadedFileSummary(BaseModel):
    """单个上传文件的统计信息。"""

    order: int
    file_name: str
    size_bytes: int
    character_count: int
    line_count: int


class ProjectUploadResponse(BaseModel):
    """小说项目上传结果。"""

    project_name: str
    file_count: int
    total_size_bytes: int
    total_characters: int
    total_lines: int
    files: list[UploadedFileSummary]


@router.post(
    "/upload",
    response_model=ProjectUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="上传小说 TXT 文件",
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
    """
    接收一个小说项目及其一个或多个 TXT 文件。

    当前接口只完成文件校验、文本读取和统计，
    不会永久保存文件，也不会执行 AI 分析。
    """

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
            # 同时兼容普通 UTF-8 和带 BOM 的 UTF-8 文本。
            text = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"文件“{file_name}”不是有效的 UTF-8 编码，"
                    "请转换编码后重新上传。"
                ),
            ) from error

        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件“{file_name}”不包含有效文本。",
            )

        character_count = len(text)
        line_count = text.count("\n") + 1

        file_summaries.append(
            UploadedFileSummary(
                order=order,
                file_name=file_name,
                size_bytes=file_size,
                character_count=character_count,
                line_count=line_count,
            )
        )

        total_characters += character_count
        total_lines += line_count

    return ProjectUploadResponse(
        project_name=normalized_project_name,
        file_count=len(file_summaries),
        total_size_bytes=total_size_bytes,
        total_characters=total_characters,
        total_lines=total_lines,
        files=file_summaries,
    )