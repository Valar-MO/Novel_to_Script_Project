import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from backend.services.chapter_detector import ChapterSpan
from backend.services.project_storage import (
    ProjectChapterNotFoundError,
    ProjectChunkNotFoundError,
    ProjectFileData,
    ProjectNotFoundError,
    get_project_chapter,
    get_project_chunk,
    get_project_summary,
    list_projects,
    save_project,
)
from backend.services.text_chunker import TextChunk
from backend.storage.database import database_session
from backend.storage.schema import create_schema


class TestProjectStorage(unittest.TestCase):
    """测试项目文件和 SQLite 数据的持久化。"""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary_directory.name)
        self.database_path = self.test_root / "test.db"
        self.projects_directory = self.test_root / "projects"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _build_valid_file_data(
        self,
        *,
        file_order: int = 1,
        file_name: str = "第一章.txt",
        global_order_start: int = 1,
    ) -> ProjectFileData:
        processed_text = (
            "第一章 测试\n\n"
            "第一段内容。\n\n"
            "第二段内容。"
        )

        original_text = processed_text
        raw_content = original_text.encode("utf-8")

        chapter = ChapterSpan(
            chapter_order=1,
            full_title="第一章 测试",
            chapter_title="测试",
            part_order=None,
            part_title=None,
            volume_order=None,
            volume_title=None,
            chapter_number=1,
            start_character=0,
            end_character=len(processed_text),
            character_count=len(processed_text),
            is_detected=True,
        )

        second_chunk_start = processed_text.index("第二段内容。")

        first_chunk_text = processed_text[:second_chunk_start]
        second_chunk_text = processed_text[second_chunk_start:]

        first_chunk = TextChunk(
            chunk_id=f"chunk_{global_order_start:04d}",
            global_order=global_order_start,
            source_file_name=file_name,
            source_file_order=file_order,
            start_character=0,
            end_character=second_chunk_start,
            character_count=len(first_chunk_text),
            paragraph_start=1,
            paragraph_end=2,
            text=first_chunk_text,
            chapter_order=1,
            chapter_number=1,
            chapter_title="测试",
            chapter_full_title="第一章 测试",
            chunk_order_in_chapter=1,
            is_chapter_start=True,
            is_chapter_end=False,
        )

        second_chunk = TextChunk(
            chunk_id=f"chunk_{global_order_start + 1:04d}",
            global_order=global_order_start + 1,
            source_file_name=file_name,
            source_file_order=file_order,
            start_character=second_chunk_start,
            end_character=len(processed_text),
            character_count=len(second_chunk_text),
            paragraph_start=3,
            paragraph_end=3,
            text=second_chunk_text,
            chapter_order=1,
            chapter_number=1,
            chapter_title="测试",
            chapter_full_title="第一章 测试",
            chunk_order_in_chapter=2,
            is_chapter_start=False,
            is_chapter_end=True,
        )

        return ProjectFileData(
            file_order=file_order,
            file_name=file_name,
            raw_content=raw_content,
            original_text=original_text,
            processed_text=processed_text,
            chapters=[chapter],
            chunks=[
                first_chunk,
                second_chunk,
            ],
        )

    def test_save_project_creates_files_and_database_rows(self):
        """保存项目后应创建文件目录和数据库记录。"""

        file_data = self._build_valid_file_data()

        saved_project = save_project(
            project_name="测试项目",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-test-001",
        )

        self.assertEqual(saved_project.project_id, "project-test-001")
        self.assertEqual(saved_project.project_name, "测试项目")
        self.assertEqual(saved_project.status, "preprocessed")
        self.assertEqual(saved_project.file_count, 1)
        self.assertEqual(saved_project.chapter_count, 1)
        self.assertEqual(saved_project.chunk_count, 2)

        project_directory = self.projects_directory / "project-test-001"
        source_path = project_directory / "source" / "0001.txt"
        processed_path = project_directory / "processed" / "0001.txt"

        self.assertTrue(source_path.exists())
        self.assertTrue(processed_path.exists())
        self.assertEqual(source_path.read_bytes(), file_data.raw_content)
        self.assertEqual(
            processed_path.read_text(encoding="utf-8"),
            file_data.processed_text,
        )

        with database_session(database_path=self.database_path) as connection:
            create_schema(connection)

            project_count = connection.execute(
                "SELECT COUNT(*) FROM projects"
            ).fetchone()[0]
            source_file_count = connection.execute(
                "SELECT COUNT(*) FROM source_files"
            ).fetchone()[0]
            chapter_count = connection.execute(
                "SELECT COUNT(*) FROM chapters"
            ).fetchone()[0]
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM text_chunks"
            ).fetchone()[0]

        self.assertEqual(project_count, 1)
        self.assertEqual(source_file_count, 1)
        self.assertEqual(chapter_count, 1)
        self.assertEqual(chunk_count, 2)

    def test_list_projects_returns_empty_list(self):
        """没有项目时应返回空列表。"""

        projects = list_projects(
            database_path=self.database_path,
        )

        self.assertEqual(projects, [])

    def test_list_projects_returns_newest_first(self):
        """项目列表应按创建时间倒序返回项目摘要。"""

        file_data = self._build_valid_file_data()

        save_project(
            project_name="较早项目",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-older",
        )

        save_project(
            project_name="较新项目",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-newer",
        )

        with database_session(
            database_path=self.database_path
        ) as connection:
            connection.execute(
                """
                UPDATE projects
                SET created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "2026-06-05 10:00:00",
                    "2026-06-05 10:00:00",
                    "project-older",
                ),
            )

            connection.execute(
                """
                UPDATE projects
                SET created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "2026-06-06 10:00:00",
                    "2026-06-06 10:00:00",
                    "project-newer",
                ),
            )

        projects = list_projects(
            database_path=self.database_path,
        )

        self.assertEqual(
            [project["project_id"] for project in projects],
            ["project-newer", "project-older"],
        )

        newest_project = projects[0]

        self.assertEqual(
            newest_project["project_name"],
            "较新项目",
        )
        self.assertEqual(newest_project["status"], "preprocessed")
        self.assertEqual(newest_project["file_count"], 1)
        self.assertEqual(newest_project["chapter_count"], 1)
        self.assertEqual(newest_project["chunk_count"], 2)
        self.assertNotIn("files", newest_project)

    def test_get_project_summary_excludes_internal_chunks(self):
        """项目摘要应返回章节，但不暴露内部文本块。"""

        file_data = self._build_valid_file_data()

        save_project(
            project_name="查询测试",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-query-001",
        )

        project = get_project_summary(
            "project-query-001",
            database_path=self.database_path,
        )

        self.assertEqual(project["project_id"], "project-query-001")
        self.assertEqual(project["project_name"], "查询测试")
        self.assertEqual(project["file_count"], 1)
        self.assertEqual(project["chapter_count"], 1)
        self.assertEqual(project["chunk_count"], 2)

        stored_file = project["files"][0]

        self.assertEqual(stored_file["file_name"], "第一章.txt")
        self.assertEqual(stored_file["chapter_count"], 1)
        self.assertEqual(stored_file["chunk_count"], 2)
        self.assertNotIn("chunks", stored_file)

        stored_chapter = stored_file["chapters"][0]

        self.assertEqual(stored_chapter["chapter_title"], "测试")
        self.assertEqual(stored_chapter["chunk_count"], 2)
        self.assertNotIn("chunks", stored_chapter)
        self.assertNotIn("text", stored_chapter)

    def test_get_project_chapter_returns_reconstructed_text(self):
        """章节详情应按顺序拼接该章全部文本块。"""

        file_data = self._build_valid_file_data()

        save_project(
            project_name="章节查询测试",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-chapter-001",
        )

        project = get_project_summary(
            "project-chapter-001",
            database_path=self.database_path,
        )

        chapter_id = project["files"][0]["chapters"][0]["id"]

        chapter = get_project_chapter(
            project_id="project-chapter-001",
            chapter_id=chapter_id,
            database_path=self.database_path,
        )

        self.assertEqual(chapter["project_id"], "project-chapter-001")
        self.assertEqual(chapter["source_file_name"], "第一章.txt")
        self.assertEqual(chapter["chapter_title"], "测试")
        self.assertEqual(chapter["internal_chunk_count"], 2)
        self.assertEqual(chapter["text"], file_data.processed_text)
        self.assertEqual(
            chapter["character_count"],
            len(file_data.processed_text),
        )

    def test_get_project_chunk_returns_full_text(self):
        """单块详情查询应继续返回完整正文和来源信息。"""

        file_data = self._build_valid_file_data()

        save_project(
            project_name="文本块查询",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-chunk-001",
        )

        chunk = get_project_chunk(
            project_id="project-chunk-001",
            chunk_id="chunk_0002",
            database_path=self.database_path,
        )

        self.assertEqual(chunk["project_id"], "project-chunk-001")
        self.assertEqual(chunk["project_name"], "文本块查询")
        self.assertEqual(chunk["source_file_name"], "第一章.txt")
        self.assertEqual(chunk["chapter_order"], 1)
        self.assertEqual(chunk["chapter_title"], "测试")
        self.assertEqual(chunk["chunk_id"], "chunk_0002")
        self.assertEqual(chunk["text"], file_data.chunks[1].text)
        self.assertFalse(chunk["is_chapter_start"])
        self.assertTrue(chunk["is_chapter_end"])

    def test_invalid_chunk_data_is_not_saved(self):
        """无效文本块不应创建目录或项目记录。"""

        valid_file_data = self._build_valid_file_data()

        invalid_first_chunk = replace(
            valid_file_data.chunks[0],
            text="错误文本",
        )

        invalid_file_data = replace(
            valid_file_data,
            chunks=[
                invalid_first_chunk,
                valid_file_data.chunks[1],
            ],
        )

        with self.assertRaises(ValueError):
            save_project(
                project_name="无效项目",
                files=[invalid_file_data],
                database_path=self.database_path,
                projects_directory=self.projects_directory,
                project_id="invalid-project",
            )

        self.assertFalse(
            (self.projects_directory / "invalid-project").exists()
        )

        with self.assertRaises(ProjectNotFoundError):
            get_project_summary(
                "invalid-project",
                database_path=self.database_path,
            )

    def test_database_failure_rolls_back_and_cleans_files(self):
        """数据库写入失败时应清理临时项目文件。"""

        duplicate_project_id = "duplicate-project"

        with database_session(database_path=self.database_path) as connection:
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
                    duplicate_project_id,
                    "已有项目",
                    "created",
                    0,
                    0,
                    0,
                ),
            )

        file_data = self._build_valid_file_data()

        with self.assertRaises(sqlite3.IntegrityError):
            save_project(
                project_name="重复项目",
                files=[file_data],
                database_path=self.database_path,
                projects_directory=self.projects_directory,
                project_id=duplicate_project_id,
            )

        self.assertFalse(
            (self.projects_directory / duplicate_project_id).exists()
        )

        temporary_directories = list(
            self.projects_directory.glob(
                f".tmp-{duplicate_project_id}-*"
            )
        )

        self.assertEqual(temporary_directories, [])

        project = get_project_summary(
            duplicate_project_id,
            database_path=self.database_path,
        )

        self.assertEqual(project["project_name"], "已有项目")
        self.assertEqual(project["file_count"], 0)

    def test_missing_project_raises_error(self):
        """查询不存在的项目应抛出明确异常。"""

        with self.assertRaises(ProjectNotFoundError):
            get_project_summary(
                "missing-project",
                database_path=self.database_path,
            )

    def test_missing_chapter_raises_error(self):
        """查询项目中不存在的章节应抛出明确异常。"""

        file_data = self._build_valid_file_data()

        save_project(
            project_name="缺失章节测试",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-missing-chapter",
        )

        with self.assertRaises(ProjectChapterNotFoundError):
            get_project_chapter(
                project_id="project-missing-chapter",
                chapter_id=999999,
                database_path=self.database_path,
            )

    def test_missing_chunk_raises_error(self):
        """查询不存在的文本块应抛出明确异常。"""

        file_data = self._build_valid_file_data()

        save_project(
            project_name="缺失文本块测试",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-missing-chunk",
        )

        with self.assertRaises(ProjectChunkNotFoundError):
            get_project_chunk(
                project_id="project-missing-chunk",
                chunk_id="chunk_9999",
                database_path=self.database_path,
            )


if __name__ == "__main__":
    unittest.main()
