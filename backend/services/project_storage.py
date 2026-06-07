import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from backend.services.chapter_detector import ChapterSpan
from backend.services.text_chunker import TextChunk
from backend.storage.database import (
    PROJECT_ROOT,
    DatabasePath,
    database_session,
)
from backend.storage.schema import create_schema


DEFAULT_PROJECTS_DIRECTORY = PROJECT_ROOT / "data" / "projects"
PROJECT_STATUS_PREPROCESSED = "preprocessed"


class ProjectNotFoundError(LookupError):
    """请求的项目不存在。"""


class ProjectBusyError(RuntimeError):
    """项目当前存在运行中的任务，暂时不能追加文件。"""


class ProjectChunkNotFoundError(LookupError):
    """请求的项目文本块不存在。"""


class ProjectChapterNotFoundError(LookupError):
    """请求的项目章节不存在。"""


@dataclass(frozen=True)
class ProjectFileData:
    """一个已经完成预处理、章节识别和分块的上传文件。"""

    file_order: int
    file_name: str

    raw_content: bytes
    original_text: str
    processed_text: str

    chapters: Sequence[ChapterSpan]
    chunks: Sequence[TextChunk]


@dataclass(frozen=True)
class SavedProject:
    """项目保存完成后的摘要信息。"""

    project_id: str
    project_name: str
    status: str
    project_directory: Path

    file_count: int
    chapter_count: int
    chunk_count: int


def _count_lines(text: str) -> int:
    """统计文本行数。"""

    if not text:
        return 0

    return text.count("\n") + 1


def _serialize_storage_path(path: Path) -> str:
    """
    将存储路径转换为适合写入数据库的字符串。

    位于项目根目录中的路径优先保存为相对路径；
    测试临时目录等外部路径保存为绝对路径。
    """

    resolved_path = path.resolve()

    try:
        relative_path = resolved_path.relative_to(PROJECT_ROOT.resolve())
        return relative_path.as_posix()
    except ValueError:
        return str(resolved_path)


def _validate_chapters(
    processed_text: str,
    chapters: Sequence[ChapterSpan],
) -> dict[int, ChapterSpan]:
    """检查章节是否连续覆盖完整的预处理文本。"""

    ordered_chapters = sorted(
        chapters,
        key=lambda chapter: chapter.chapter_order,
    )

    if not ordered_chapters:
        raise ValueError("每个文件必须至少包含一个章节范围。")

    expected_orders = list(range(1, len(ordered_chapters) + 1))
    actual_orders = [chapter.chapter_order for chapter in ordered_chapters]

    if actual_orders != expected_orders:
        raise ValueError("章节顺序必须从 1 开始并连续递增。")

    chapter_by_order: dict[int, ChapterSpan] = {}
    expected_start = 0

    for chapter in ordered_chapters:
        if chapter.start_character != expected_start:
            raise ValueError("章节范围之间存在空隙或重叠。")

        if chapter.end_character < chapter.start_character:
            raise ValueError("章节结束位置不能小于开始位置。")

        expected_character_count = (
            chapter.end_character - chapter.start_character
        )

        if chapter.character_count != expected_character_count:
            raise ValueError("章节字符数与起止字符位置不一致。")

        chapter_text = processed_text[
            chapter.start_character:chapter.end_character
        ]

        if len(chapter_text) != chapter.character_count:
            raise ValueError("章节范围无法正确对应预处理文本。")

        chapter_by_order[chapter.chapter_order] = chapter
        expected_start = chapter.end_character

    if expected_start != len(processed_text):
        raise ValueError("章节范围没有完整覆盖预处理文本。")

    return chapter_by_order


def _validate_chunks(
    file_data: ProjectFileData,
    chapter_by_order: dict[int, ChapterSpan],
) -> list[TextChunk]:
    """检查文本块的位置、正文和章节归属。"""

    ordered_chunks = sorted(
        file_data.chunks,
        key=lambda chunk: chunk.global_order,
    )

    if not ordered_chunks:
        raise ValueError("每个文件必须至少包含一个文本块。")

    expected_start = 0
    seen_chunk_ids: set[str] = set()

    chunks_by_chapter: dict[int, list[TextChunk]] = {
        chapter_order: []
        for chapter_order in chapter_by_order
    }

    for chunk in ordered_chunks:
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"文本块编号重复：{chunk.chunk_id}")

        seen_chunk_ids.add(chunk.chunk_id)

        if chunk.source_file_order != file_data.file_order:
            raise ValueError("文本块记录的文件顺序与当前文件不一致。")

        if chunk.source_file_name != file_data.file_name:
            raise ValueError("文本块记录的文件名称与当前文件不一致。")

        if chunk.start_character != expected_start:
            raise ValueError("文本块范围之间存在空隙或重叠。")

        if chunk.end_character < chunk.start_character:
            raise ValueError("文本块结束位置不能小于开始位置。")

        expected_character_count = (
            chunk.end_character - chunk.start_character
        )

        if chunk.character_count != expected_character_count:
            raise ValueError("文本块字符数与起止位置不一致。")

        expected_text = file_data.processed_text[
            chunk.start_character:chunk.end_character
        ]

        if chunk.text != expected_text:
            raise ValueError("文本块正文与其字符范围不一致。")

        if chunk.chapter_order is None:
            raise ValueError("文本块必须记录所属章节顺序。")

        chapter = chapter_by_order.get(chunk.chapter_order)

        if chapter is None:
            raise ValueError("文本块引用了不存在的章节。")

        if (
            chunk.start_character < chapter.start_character
            or chunk.end_character > chapter.end_character
        ):
            raise ValueError("文本块范围超出了所属章节范围。")

        chunks_by_chapter[chunk.chapter_order].append(chunk)
        expected_start = chunk.end_character

    if expected_start != len(file_data.processed_text):
        raise ValueError("文本块没有完整覆盖预处理文本。")

    for chapter_order, chapter_chunks in chunks_by_chapter.items():
        if not chapter_chunks:
            raise ValueError(f"第 {chapter_order} 个章节没有文本块。")

        chapter_chunks.sort(
            key=lambda chunk: (
                chunk.chunk_order_in_chapter
                if chunk.chunk_order_in_chapter is not None
                else 0
            )
        )

        expected_chunk_orders = list(range(1, len(chapter_chunks) + 1))
        actual_chunk_orders = [
            chunk.chunk_order_in_chapter
            for chunk in chapter_chunks
        ]

        if actual_chunk_orders != expected_chunk_orders:
            raise ValueError("章内文本块顺序必须从 1 开始连续递增。")

        first_chunk = chapter_chunks[0]
        last_chunk = chapter_chunks[-1]

        if not first_chunk.is_chapter_start:
            raise ValueError("章节的第一个文本块必须标记为章节开头。")

        if not last_chunk.is_chapter_end:
            raise ValueError("章节的最后一个文本块必须标记为章节结尾。")

        for middle_chunk in chapter_chunks[1:-1]:
            if middle_chunk.is_chapter_start or middle_chunk.is_chapter_end:
                raise ValueError("章节中间文本块不能标记为章节首尾。")

    return ordered_chunks


def _validate_project_files(
    files: Sequence[ProjectFileData],
    *,
    file_order_start: int = 1,
    global_order_start: int = 1,
) -> list[
    tuple[
        ProjectFileData,
        list[ChapterSpan],
        list[TextChunk],
    ]
]:
    """检查整个项目的文件、章节和文本块数据。"""

    if not files:
        raise ValueError("项目必须至少包含一个文件。")

    ordered_files = sorted(
        files,
        key=lambda file_data: file_data.file_order,
    )

    expected_file_orders = list(
        range(file_order_start, file_order_start + len(ordered_files))
    )
    actual_file_orders = [file_data.file_order for file_data in ordered_files]

    if actual_file_orders != expected_file_orders:
        raise ValueError("文件顺序必须连续递增。")

    validated_files: list[
        tuple[
            ProjectFileData,
            list[ChapterSpan],
            list[TextChunk],
        ]
    ] = []

    all_global_orders: list[int] = []

    for file_data in ordered_files:
        if not file_data.file_name.strip():
            raise ValueError("文件名称不能为空。")

        if not file_data.raw_content:
            raise ValueError(f"文件“{file_data.file_name}”没有原始内容。")

        if not file_data.original_text:
            raise ValueError(f"文件“{file_data.file_name}”没有原始文本。")

        if not file_data.processed_text:
            raise ValueError(f"文件“{file_data.file_name}”没有处理后文本。")

        chapter_by_order = _validate_chapters(
            processed_text=file_data.processed_text,
            chapters=file_data.chapters,
        )

        ordered_chunks = _validate_chunks(
            file_data=file_data,
            chapter_by_order=chapter_by_order,
        )

        ordered_chapters = sorted(
            file_data.chapters,
            key=lambda chapter: chapter.chapter_order,
        )

        all_global_orders.extend(
            chunk.global_order
            for chunk in ordered_chunks
        )

        validated_files.append(
            (
                file_data,
                ordered_chapters,
                ordered_chunks,
            )
        )

    expected_global_orders = list(
        range(global_order_start, global_order_start + len(all_global_orders))
    )

    if all_global_orders != expected_global_orders:
        raise ValueError("项目文本块的全局顺序必须连续递增。")

    return validated_files


def _project_directory(
    *,
    projects_directory: str | Path | None,
    project_id: str,
) -> Path:
    storage_root = Path(
        projects_directory
        if projects_directory is not None
        else DEFAULT_PROJECTS_DIRECTORY
    )
    storage_root.mkdir(parents=True, exist_ok=True)
    return storage_root / project_id


def _write_project_files(
    *,
    connection,
    project_id: str,
    project_directory: Path,
    validated_files: list[
        tuple[
            ProjectFileData,
            list[ChapterSpan],
            list[TextChunk],
        ]
    ],
) -> tuple[int, int]:
    chapter_count = 0
    chunk_count = 0

    for file_data, ordered_chapters, ordered_chunks in validated_files:
        stored_file_name = f"{file_data.file_order:04d}.txt"
        final_source_path = (
            project_directory / "source" / stored_file_name
        )
        final_processed_path = (
            project_directory / "processed" / stored_file_name
        )

        final_source_path.write_bytes(file_data.raw_content)
        final_processed_path.write_text(
            file_data.processed_text,
            encoding="utf-8",
            newline="\n",
        )

        source_file_cursor = connection.execute(
            """
            INSERT INTO source_files (
                project_id,
                file_order,
                file_name,
                source_path,
                processed_path,
                size_bytes,
                original_character_count,
                original_line_count,
                processed_character_count,
                processed_line_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                file_data.file_order,
                file_data.file_name,
                _serialize_storage_path(final_source_path),
                _serialize_storage_path(final_processed_path),
                len(file_data.raw_content),
                len(file_data.original_text),
                _count_lines(file_data.original_text),
                len(file_data.processed_text),
                _count_lines(file_data.processed_text),
            ),
        )

        source_file_id = source_file_cursor.lastrowid

        if source_file_id is None:
            raise RuntimeError("无法取得源文件数据库 ID。")

        chapter_id_by_order: dict[int, int] = {}

        for chapter in ordered_chapters:
            chapter_cursor = connection.execute(
                """
                INSERT INTO chapters (
                    source_file_id,
                    chapter_order,
                    chapter_number,
                    chapter_title,
                    full_title,
                    part_order,
                    part_title,
                    volume_order,
                    volume_title,
                    start_character,
                    end_character,
                    character_count,
                    is_detected
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_file_id,
                    chapter.chapter_order,
                    chapter.chapter_number,
                    chapter.chapter_title,
                    chapter.full_title,
                    chapter.part_order,
                    chapter.part_title,
                    chapter.volume_order,
                    chapter.volume_title,
                    chapter.start_character,
                    chapter.end_character,
                    chapter.character_count,
                    int(chapter.is_detected),
                ),
            )

            chapter_database_id = chapter_cursor.lastrowid

            if chapter_database_id is None:
                raise RuntimeError("无法取得章节数据库 ID。")

            chapter_id_by_order[chapter.chapter_order] = chapter_database_id
            chapter_count += 1

        for chunk in ordered_chunks:
            if chunk.chapter_order is None:
                raise ValueError("文本块缺少章节顺序。")

            chapter_database_id = chapter_id_by_order[
                chunk.chapter_order
            ]

            connection.execute(
                """
                INSERT INTO text_chunks (
                    source_file_id,
                    chapter_id,
                    chunk_id,
                    global_order,
                    chunk_order_in_chapter,
                    start_character,
                    end_character,
                    character_count,
                    paragraph_start,
                    paragraph_end,
                    text,
                    is_chapter_start,
                    is_chapter_end
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_file_id,
                    chapter_database_id,
                    chunk.chunk_id,
                    chunk.global_order,
                    chunk.chunk_order_in_chapter,
                    chunk.start_character,
                    chunk.end_character,
                    chunk.character_count,
                    chunk.paragraph_start,
                    chunk.paragraph_end,
                    chunk.text,
                    int(chunk.is_chapter_start),
                    int(chunk.is_chapter_end),
                ),
            )
            chunk_count += 1

    return chapter_count, chunk_count


def save_project(
    project_name: str,
    files: Sequence[ProjectFileData],
    *,
    database_path: DatabasePath | None = None,
    projects_directory: str | Path | None = None,
    project_id: str | None = None,
) -> SavedProject:
    """
    保存一个已经完成预处理和分块的小说项目。

    任何步骤失败时，回滚数据库事务，并清理本次创建
    的临时目录或正式项目目录。
    """

    normalized_name = project_name.strip()

    if not normalized_name:
        raise ValueError("项目名称不能为空。")

    validated_files = _validate_project_files(
        files,
        file_order_start=1,
        global_order_start=1,
    )
    resolved_project_id = project_id or str(uuid.uuid4())

    if not resolved_project_id.strip():
        raise ValueError("project_id 不能为空。")

    final_project_directory = _project_directory(
        projects_directory=projects_directory,
        project_id=resolved_project_id,
    )

    if final_project_directory.exists():
        raise FileExistsError(
            f"项目目录已经存在：{final_project_directory}"
        )

    temporary_project_directory = (
        final_project_directory.parent
        / f".tmp-{resolved_project_id}-{uuid.uuid4().hex}"
    )

    source_directory = temporary_project_directory / "source"
    processed_directory = temporary_project_directory / "processed"

    chapter_count = sum(
        len(chapters)
        for _, chapters, _ in validated_files
    )
    chunk_count = sum(
        len(chunks)
        for _, _, chunks in validated_files
    )

    try:
        source_directory.mkdir(parents=True, exist_ok=False)
        processed_directory.mkdir(parents=True, exist_ok=False)

        for file_data, _, _ in validated_files:
            stored_file_name = f"{file_data.file_order:04d}.txt"
            temporary_source_path = source_directory / stored_file_name
            temporary_processed_path = processed_directory / stored_file_name

            temporary_source_path.write_bytes(file_data.raw_content)
            temporary_processed_path.write_text(
                file_data.processed_text,
                encoding="utf-8",
                newline="\n",
            )

        with database_session(database_path=database_path) as connection:
            create_schema(connection)

            connection.execute(
                """
                INSERT INTO projects (
                    id,
                    name,
                    status,
                    file_count,
                    chapter_count,
                    chunk_count
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_project_id,
                    normalized_name,
                    PROJECT_STATUS_PREPROCESSED,
                    len(validated_files),
                    chapter_count,
                    chunk_count,
                ),
            )

            _write_project_files(
                connection=connection,
                project_id=resolved_project_id,
                project_directory=temporary_project_directory,
                validated_files=validated_files,
            )

            temporary_project_directory.replace(final_project_directory)

        return SavedProject(
            project_id=resolved_project_id,
            project_name=normalized_name,
            status=PROJECT_STATUS_PREPROCESSED,
            project_directory=final_project_directory,
            file_count=len(validated_files),
            chapter_count=chapter_count,
            chunk_count=chunk_count,
        )

    except Exception:
        shutil.rmtree(
            temporary_project_directory,
            ignore_errors=True,
        )
        shutil.rmtree(
            final_project_directory,
            ignore_errors=True,
        )
        raise

def append_project_files(
    project_id: str,
    files: Sequence[ProjectFileData],
    *,
    database_path: DatabasePath | None = None,
    projects_directory: str | Path | None = None,
) -> dict[str, Any]:
    if not project_id.strip():
        raise ValueError("project_id 不能为空。")

    if not files:
        raise ValueError("至少需要一个待追加文件。")

    with database_session(database_path=database_path) as connection:
        create_schema(connection)

        project_row = connection.execute(
            """
            SELECT id, file_count, chapter_count, chunk_count
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()

        if project_row is None:
            raise ProjectNotFoundError(f"项目不存在：{project_id}")

        busy_row = connection.execute(
            """
            SELECT 1
            FROM narrative_analysis_runs
            WHERE project_id = ?
              AND status IN ('queued', 'running')
            UNION ALL
            SELECT 1
            FROM script_generation_runs
            WHERE project_id = ?
              AND status IN ('queued', 'running')
            LIMIT 1
            """,
            (project_id, project_id),
        ).fetchone()

        if busy_row is not None:
            raise ProjectBusyError("项目当前存在运行中的任务，暂时不能追加文件。")

        next_file_order = int(project_row["file_count"]) + 1
        next_global_order = int(project_row["chunk_count"]) + 1

    validated_files = _validate_project_files(
        files,
        file_order_start=next_file_order,
        global_order_start=next_global_order,
    )

    project_directory = _project_directory(
        projects_directory=projects_directory,
        project_id=project_id,
    )
    source_directory = project_directory / "source"
    processed_directory = project_directory / "processed"

    if not source_directory.exists() or not processed_directory.exists():
        raise FileNotFoundError(f"项目目录不完整：{project_directory}")

    added_chunk_ids = [
        chunk.chunk_id
        for _, _, ordered_chunks in validated_files
        for chunk in ordered_chunks
    ]

    try:
        with database_session(database_path=database_path) as connection:
            create_schema(connection)
            added_chapter_count, added_chunk_count = _write_project_files(
                connection=connection,
                project_id=project_id,
                project_directory=project_directory,
                validated_files=validated_files,
            )
            connection.execute(
                """
                UPDATE projects
                SET file_count = file_count + ?,
                    chapter_count = chapter_count + ?,
                    chunk_count = chunk_count + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    len(validated_files),
                    added_chapter_count,
                    added_chunk_count,
                    project_id,
                ),
            )
    except Exception:
        for file_data, _, _ in validated_files:
            stored_file_name = f"{file_data.file_order:04d}.txt"
            (source_directory / stored_file_name).unlink(missing_ok=True)
            (processed_directory / stored_file_name).unlink(missing_ok=True)
        raise

    return {
        "project_id": project_id,
        "added_file_count": len(validated_files),
        "added_chapter_count": sum(
            len(ordered_chapters)
            for _, ordered_chapters, _ in validated_files
        ),
        "added_chunk_count": len(added_chunk_ids),
        "added_chunk_ids": added_chunk_ids,
    }



def delete_project(
    project_id: str,
    *,
    database_path: DatabasePath | None = None,
    projects_directory: str | Path | None = None,
) -> None:
    normalized_project_id = project_id.strip()

    if not normalized_project_id:
        raise ValueError("project_id 不能为空。")

    project_directory = _project_directory(
        projects_directory=projects_directory,
        project_id=normalized_project_id,
    )

    deleting_directory = (
        project_directory.parent
        / (
            f".deleting-{normalized_project_id}-"
            f"{uuid.uuid4().hex}"
        )
    )

    with database_session(
        database_path=database_path,
    ) as connection:
        create_schema(connection)

        project_row = connection.execute(
            """
            SELECT id
            FROM projects
            WHERE id = ?
            """,
            (normalized_project_id,),
        ).fetchone()

        if project_row is None:
            raise ProjectNotFoundError(
                f"项目不存在：{normalized_project_id}"
            )

        busy_row = connection.execute(
            """
            SELECT 1
            FROM narrative_analysis_runs
            WHERE project_id = ?
              AND status IN ('queued', 'running')
            UNION ALL
            SELECT 1
            FROM script_generation_runs
            WHERE project_id = ?
              AND status IN ('queued', 'running')
            LIMIT 1
            """,
            (
                normalized_project_id,
                normalized_project_id,
            ),
        ).fetchone()

        if busy_row is not None:
            raise ProjectBusyError(
                "项目当前存在运行中的任务，暂时不能删除。"
            )

    if project_directory.exists():
        project_directory.replace(deleting_directory)

    try:
        with database_session(
            database_path=database_path,
        ) as connection:
            create_schema(connection)

            cursor = connection.execute(
                """
                DELETE FROM projects
                WHERE id = ?
                """,
                (normalized_project_id,),
            )

            if cursor.rowcount == 0:
                raise ProjectNotFoundError(
                    f"项目不存在：{normalized_project_id}"
                )

    except Exception:
        if (
            deleting_directory.exists()
            and not project_directory.exists()
        ):
            deleting_directory.replace(project_directory)

        raise

    shutil.rmtree(
        deleting_directory,
        ignore_errors=True,
    )



def list_projects(
    *,
    database_path: DatabasePath | None = None,
) -> list[dict[str, Any]]:
    """
    按创建时间倒序读取所有项目摘要。

    项目列表只返回项目级统计信息，不加载源文件、章节或正文。
    """

    with database_session(
        database_path=database_path
    ) as connection:
        create_schema(connection)

        project_rows = connection.execute(
            """
            SELECT
                id,
                name,
                status,
                created_at,
                updated_at,
                file_count,
                chapter_count,
                chunk_count
            FROM projects
            ORDER BY
                datetime(created_at) DESC,
                rowid DESC
            """
        ).fetchall()

        return [
            {
                "project_id": row["id"],
                "project_name": row["name"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "file_count": row["file_count"],
                "chapter_count": row["chapter_count"],
                "chunk_count": row["chunk_count"],
            }
            for row in project_rows
        ]

def get_project_summary(
    project_id: str,
    *,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    """
    读取项目摘要。

    返回项目、文件和章节摘要，不返回文本块列表或完整正文。
    文本块仍保存在数据库中，供内部处理和 AI 分析使用。
    """

    with database_session(database_path=database_path) as connection:
        create_schema(connection)

        project_row = connection.execute(
            """
            SELECT
                id,
                name,
                status,
                created_at,
                updated_at,
                file_count,
                chapter_count,
                chunk_count
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()

        if project_row is None:
            raise ProjectNotFoundError(f"项目不存在：{project_id}")

        file_rows = connection.execute(
            """
            SELECT
                id,
                file_order,
                file_name,
                size_bytes,
                original_character_count,
                original_line_count,
                processed_character_count,
                processed_line_count,
                created_at
            FROM source_files
            WHERE project_id = ?
            ORDER BY file_order
            """,
            (project_id,),
        ).fetchall()

        files: list[dict[str, Any]] = []

        for file_row in file_rows:
            source_file_id = file_row["id"]

            chapter_rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.chapter_order,
                    c.chapter_number,
                    c.chapter_title,
                    c.full_title,
                    c.part_order,
                    c.part_title,
                    c.volume_order,
                    c.volume_title,
                    c.start_character,
                    c.end_character,
                    c.character_count,
                    c.is_detected,
                    c.created_at,
                    COUNT(tc.id) AS chunk_count
                FROM chapters AS c
                LEFT JOIN text_chunks AS tc
                    ON tc.chapter_id = c.id
                WHERE c.source_file_id = ?
                GROUP BY c.id
                ORDER BY c.chapter_order
                """,
                (source_file_id,),
            ).fetchall()

            chapters: list[dict[str, Any]] = []

            for chapter_row in chapter_rows:
                chapters.append(
                    {
                        "id": chapter_row["id"],
                        "chapter_order": chapter_row["chapter_order"],
                        "chapter_number": chapter_row["chapter_number"],
                        "chapter_title": chapter_row["chapter_title"],
                        "full_title": chapter_row["full_title"],
                        "part_order": chapter_row["part_order"],
                        "part_title": chapter_row["part_title"],
                        "volume_order": chapter_row["volume_order"],
                        "volume_title": chapter_row["volume_title"],
                        "start_character": chapter_row["start_character"],
                        "end_character": chapter_row["end_character"],
                        "character_count": chapter_row["character_count"],
                        "is_detected": bool(chapter_row["is_detected"]),
                        "chunk_count": chapter_row["chunk_count"],
                        "created_at": chapter_row["created_at"],
                    }
                )

            file_chunk_count = sum(
                chapter["chunk_count"]
                for chapter in chapters
            )

            files.append(
                {
                    "id": source_file_id,
                    "file_order": file_row["file_order"],
                    "file_name": file_row["file_name"],
                    "size_bytes": file_row["size_bytes"],
                    "original_character_count": file_row[
                        "original_character_count"
                    ],
                    "original_line_count": file_row["original_line_count"],
                    "processed_character_count": file_row[
                        "processed_character_count"
                    ],
                    "processed_line_count": file_row[
                        "processed_line_count"
                    ],
                    "chapter_count": len(chapters),
                    "chunk_count": file_chunk_count,
                    "chapters": chapters,
                    "created_at": file_row["created_at"],
                }
            )

        return {
            "project_id": project_row["id"],
            "project_name": project_row["name"],
            "status": project_row["status"],
            "created_at": project_row["created_at"],
            "updated_at": project_row["updated_at"],
            "file_count": project_row["file_count"],
            "chapter_count": project_row["chapter_count"],
            "chunk_count": project_row["chunk_count"],
            "files": files,
        }


def get_project_chapter(
    project_id: str,
    chapter_id: int,
    *,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    """
    读取指定章节的完整正文。

    章节正文由该章节所属文本块按章内顺序拼接得到。
    文本块仍作为后端内部处理单元，但不暴露给普通前端界面。
    """

    with database_session(database_path=database_path) as connection:
        create_schema(connection)

        project_row = connection.execute(
            """
            SELECT
                id,
                name
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()

        if project_row is None:
            raise ProjectNotFoundError(f"项目不存在：{project_id}")

        chapter_row = connection.execute(
            """
            SELECT
                c.id,
                c.chapter_order,
                c.chapter_number,
                c.chapter_title,
                c.full_title,
                c.part_order,
                c.part_title,
                c.volume_order,
                c.volume_title,
                c.start_character,
                c.end_character,
                c.character_count,
                c.is_detected,
                c.created_at,

                sf.id AS source_file_id,
                sf.file_order AS source_file_order,
                sf.file_name AS source_file_name

            FROM chapters AS c

            INNER JOIN source_files AS sf
                ON sf.id = c.source_file_id

            WHERE
                sf.project_id = ?
                AND c.id = ?
            """,
            (
                project_id,
                chapter_id,
            ),
        ).fetchone()

        if chapter_row is None:
            raise ProjectChapterNotFoundError(
                f"项目中不存在章节：{chapter_id}"
            )

        chunk_rows = connection.execute(
            """
            SELECT
                text
            FROM text_chunks
            WHERE chapter_id = ?
            ORDER BY chunk_order_in_chapter
            """,
            (chapter_id,),
        ).fetchall()

        if not chunk_rows:
            raise RuntimeError(f"章节没有关联文本块：{chapter_id}")

        chapter_text = "".join(
            chunk_row["text"]
            for chunk_row in chunk_rows
        )

        if len(chapter_text) != chapter_row["character_count"]:
            raise RuntimeError("章节正文长度与数据库记录不一致。")

        return {
            "project_id": project_row["id"],
            "project_name": project_row["name"],
            "source_file_id": chapter_row["source_file_id"],
            "source_file_order": chapter_row["source_file_order"],
            "source_file_name": chapter_row["source_file_name"],
            "chapter_id": chapter_row["id"],
            "chapter_order": chapter_row["chapter_order"],
            "chapter_number": chapter_row["chapter_number"],
            "chapter_title": chapter_row["chapter_title"],
            "full_title": chapter_row["full_title"],
            "part_order": chapter_row["part_order"],
            "part_title": chapter_row["part_title"],
            "volume_order": chapter_row["volume_order"],
            "volume_title": chapter_row["volume_title"],
            "start_character": chapter_row["start_character"],
            "end_character": chapter_row["end_character"],
            "character_count": chapter_row["character_count"],
            "is_detected": bool(chapter_row["is_detected"]),
            "internal_chunk_count": len(chunk_rows),
            "text": chapter_text,
            "created_at": chapter_row["created_at"],
        }


def get_project_chunk(
    project_id: str,
    chunk_id: str,
    *,
    database_path: DatabasePath | None = None,
) -> dict[str, Any]:
    """读取指定项目中某个文本块的完整正文和来源信息。"""

    with database_session(database_path=database_path) as connection:
        create_schema(connection)

        project_row = connection.execute(
            """
            SELECT
                id,
                name
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()

        if project_row is None:
            raise ProjectNotFoundError(f"项目不存在：{project_id}")

        chunk_row = connection.execute(
            """
            SELECT
                tc.id,
                tc.chunk_id,
                tc.global_order,
                tc.chunk_order_in_chapter,
                tc.start_character,
                tc.end_character,
                tc.character_count,
                tc.paragraph_start,
                tc.paragraph_end,
                tc.text,
                tc.is_chapter_start,
                tc.is_chapter_end,
                tc.created_at,

                sf.id AS source_file_id,
                sf.file_order AS source_file_order,
                sf.file_name AS source_file_name,

                c.id AS chapter_id,
                c.chapter_order,
                c.chapter_number,
                c.chapter_title,
                c.full_title AS chapter_full_title

            FROM text_chunks AS tc

            INNER JOIN source_files AS sf
                ON sf.id = tc.source_file_id

            INNER JOIN chapters AS c
                ON c.id = tc.chapter_id

            WHERE
                sf.project_id = ?
                AND tc.chunk_id = ?
            """,
            (
                project_id,
                chunk_id,
            ),
        ).fetchone()

        if chunk_row is None:
            raise ProjectChunkNotFoundError(
                f"项目中不存在文本块：{chunk_id}"
            )

        return {
            "project_id": project_row["id"],
            "project_name": project_row["name"],
            "source_file_id": chunk_row["source_file_id"],
            "source_file_order": chunk_row["source_file_order"],
            "source_file_name": chunk_row["source_file_name"],
            "chapter_id": chunk_row["chapter_id"],
            "chapter_order": chunk_row["chapter_order"],
            "chapter_number": chunk_row["chapter_number"],
            "chapter_title": chunk_row["chapter_title"],
            "chapter_full_title": chunk_row["chapter_full_title"],
            "id": chunk_row["id"],
            "chunk_id": chunk_row["chunk_id"],
            "global_order": chunk_row["global_order"],
            "chunk_order_in_chapter": chunk_row[
                "chunk_order_in_chapter"
            ],
            "start_character": chunk_row["start_character"],
            "end_character": chunk_row["end_character"],
            "character_count": chunk_row["character_count"],
            "paragraph_start": chunk_row["paragraph_start"],
            "paragraph_end": chunk_row["paragraph_end"],
            "text": chunk_row["text"],
            "is_chapter_start": bool(chunk_row["is_chapter_start"]),
            "is_chapter_end": bool(chunk_row["is_chapter_end"]),
            "created_at": chunk_row["created_at"],
        }
