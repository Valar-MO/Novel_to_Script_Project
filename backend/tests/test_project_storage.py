import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from backend.services.chapter_detector import ChapterSpan
from backend.services.project_storage import (
    ProjectBusyError,
    ProjectChapterNotFoundError,
    ProjectChunkNotFoundError,
    ProjectFileData,
    ProjectNotFoundError,
    append_project_files,
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

    def test_append_project_files_continues_orders(self):
        first_file = self._build_valid_file_data(
            file_order=1,
            file_name="第一章.txt",
            global_order_start=1,
        )
        second_file = self._build_valid_file_data(
            file_order=2,
            file_name="第二章.txt",
            global_order_start=3,
        )

        saved_project = save_project(
            project_name="追加测试",
            files=[first_file],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-append-001",
        )

        result = append_project_files(
            saved_project.project_id,
            [second_file],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
        )

        self.assertEqual(result["added_file_count"], 1)
        self.assertEqual(result["added_chapter_count"], 1)
        self.assertEqual(result["added_chunk_count"], 2)
        self.assertEqual(result["added_chunk_ids"], ["chunk_0003", "chunk_0004"])

        project = get_project_summary(
            saved_project.project_id,
            database_path=self.database_path,
        )
        self.assertEqual(project["file_count"], 2)
        self.assertEqual(project["chapter_count"], 2)
        self.assertEqual(project["chunk_count"], 4)
        self.assertEqual(
            [item["file_order"] for item in project["files"]],
            [1, 2],
        )

    def test_append_project_files_rejects_busy_project(self):
        file_data = self._build_valid_file_data()
        saved_project = save_project(
            project_name="忙碌项目",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-busy-001",
        )

        with database_session(database_path=self.database_path) as connection:
            create_schema(connection)
            connection.execute(
                """
                INSERT INTO script_generation_runs (
                    project_id,
                    provider,
                    model,
                    prompt_version,
                    schema_version,
                    status,
                    total_chunks
                )
                VALUES (?, 'mock', 'mock-model', 'v1', 's1', 'running', 1)
                """,
                (saved_project.project_id,),
            )

        with self.assertRaises(ProjectBusyError):
            append_project_files(
                saved_project.project_id,
                [
                    self._build_valid_file_data(
                        file_order=2,
                        file_name="第二章.txt",
                        global_order_start=3,
                    )
                ],
                database_path=self.database_path,
                projects_directory=self.projects_directory,
            )

    def test_append_project_files_rolls_back_written_files(self):
        file_data = self._build_valid_file_data()
        saved_project = save_project(
            project_name="回滚项目",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-rollback-001",
        )

        invalid_file = replace(
            self._build_valid_file_data(
                file_order=2,
                file_name="第二章.txt",
                global_order_start=3,
            ),
            chunks=[],
        )

        with self.assertRaises(ValueError):
            append_project_files(
                saved_project.project_id,
                [invalid_file],
                database_path=self.database_path,
                projects_directory=self.projects_directory,
            )

        project = get_project_summary(
            saved_project.project_id,
            database_path=self.database_path,
        )
        self.assertEqual(project["file_count"], 1)
        self.assertFalse(
            (
                self.projects_directory
                / saved_project.project_id
                / "source"
                / "0002.txt"
            ).exists()
        )

    def test_list_projects_returns_empty_list(self):
        projects = list_projects(
            database_path=self.database_path,
        )
        self.assertEqual(projects, [])

    def test_list_projects_returns_newest_first(self):
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

    def test_get_project_summary_excludes_internal_chunks(self):
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

    def test_get_project_chapter_returns_reconstructed_text(self):
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

    def test_get_project_chunk_returns_full_text(self):
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
        self.assertEqual(chunk["chunk_id"], "chunk_0002")
        self.assertEqual(chunk["text"], file_data.chunks[1].text)

    def test_invalid_chunk_data_is_not_saved(self):
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

    def test_database_failure_rolls_back_and_cleans_files(self):
        with database_session(database_path=self.database_path) as connection:
            create_schema(connection)
            connection.execute(
                "INSERT INTO projects (id, name, status) VALUES (?, ?, ?)",
                ("duplicate-id", "已有项目", "created"),
            )

        with self.assertRaises((sqlite3.IntegrityError, FileExistsError)):
            save_project(
                project_name="冲突项目",
                files=[self._build_valid_file_data()],
                database_path=self.database_path,
                projects_directory=self.projects_directory,
                project_id="duplicate-id",
            )

    def test_missing_project_raises_error(self):
        with self.assertRaises(ProjectNotFoundError):
            get_project_summary(
                "missing-project",
                database_path=self.database_path,
            )

    def test_missing_chapter_raises_error(self):
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
                chapter_id=999,
                database_path=self.database_path,
            )

    def test_missing_chunk_raises_error(self):
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
                chunk_id="missing_chunk_id",
                database_path=self.database_path,
            )


if __name__ == "__main__":
    unittest.main()
