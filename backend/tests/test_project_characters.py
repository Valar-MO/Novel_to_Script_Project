import json
import tempfile
import unittest
from pathlib import Path

from backend.services.chapter_detector import ChapterSpan
from backend.services.project_characters import (
    get_core_project_characters,
    get_project_character_run,
    suppress_ordinary_character,
)
from backend.services.project_storage import ProjectFileData, save_project
from backend.services.text_chunker import TextChunk
from backend.storage.database import database_session


class TestProjectCharacters(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary_directory.name)
        self.database_path = self.test_root / "test.db"
        self.projects_directory = self.test_root / "projects"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _save_project(self) -> str:
        text = "Han Li met Doctor Mo.\n\nThird Uncle passed by."
        split_at = text.index("Third Uncle")
        chapter = ChapterSpan(
            chapter_order=1,
            full_title="Chapter 1",
            chapter_title="Chapter 1",
            part_order=None,
            part_title=None,
            volume_order=None,
            volume_title=None,
            chapter_number=1,
            start_character=0,
            end_character=len(text),
            character_count=len(text),
            is_detected=True,
        )
        chunks = [
            TextChunk(
                chunk_id="chunk_0001",
                global_order=1,
                source_file_name="novel.txt",
                source_file_order=1,
                start_character=0,
                end_character=split_at,
                character_count=split_at,
                paragraph_start=1,
                paragraph_end=1,
                text=text[:split_at],
                chapter_order=1,
                chapter_number=1,
                chapter_title="Chapter 1",
                chapter_full_title="Chapter 1",
                chunk_order_in_chapter=1,
                is_chapter_start=True,
                is_chapter_end=False,
            ),
            TextChunk(
                chunk_id="chunk_0002",
                global_order=2,
                source_file_name="novel.txt",
                source_file_order=1,
                start_character=split_at,
                end_character=len(text),
                character_count=len(text) - split_at,
                paragraph_start=2,
                paragraph_end=2,
                text=text[split_at:],
                chapter_order=1,
                chapter_number=1,
                chapter_title="Chapter 1",
                chapter_full_title="Chapter 1",
                chunk_order_in_chapter=2,
                is_chapter_start=False,
                is_chapter_end=True,
            ),
        ]
        project = save_project(
            project_name="Character Test",
            files=[
                ProjectFileData(
                    file_order=1,
                    file_name="novel.txt",
                    raw_content=text.encode("utf-8"),
                    original_text=text,
                    processed_text=text,
                    chapters=[chapter],
                    chunks=chunks,
                )
            ],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
        )
        return project.project_id

    def _insert_character_run(self, project_id: str) -> int:
        with database_session(database_path=self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO narrative_analysis_runs (
                    project_id,
                    provider,
                    model,
                    prompt_version,
                    schema_version,
                    status,
                    total_chunks,
                    processed_chunks,
                    successful_chunks
                )
                VALUES (?, 'mock', 'mock', 'p', 's', 'completed', 2, 2, 2)
                """,
                (project_id,),
            )
            narrative_run_id = int(cursor.lastrowid)

            cursor = connection.execute(
                """
                INSERT INTO project_character_runs (
                    project_id,
                    narrative_run_id,
                    status,
                    total_units,
                    used_units,
                    total_candidates,
                    merged_characters
                )
                VALUES (?, ?, 'completed', 2, 2, 3, 3)
                """,
                (project_id, narrative_run_id),
            )
            character_run_id = int(cursor.lastrowid)

            characters = [
                {
                    "character_id": "hanli",
                    "canonical_name": "Han Li",
                    "mention_ids": ["m_hanli_1", "m_hanli_2"],
                    "evidence_count": 2,
                    "input_quality": {
                        "chunk_count": 2,
                        "mention_count": 2,
                    },
                    "is_user_pinned": 0,
                },
                {
                    "character_id": "doctormo",
                    "canonical_name": "Doctor Mo",
                    "mention_ids": ["m_doctor_1", "m_doctor_2"],
                    "evidence_count": 2,
                    "input_quality": {
                        "chunk_count": 2,
                        "mention_count": 2,
                    },
                    "is_user_pinned": 0,
                },
                {
                    "character_id": "ordinary-uncle",
                    "canonical_name": "Third Uncle",
                    "mention_ids": ["m_uncle_1"],
                    "evidence_count": 1,
                    "input_quality": {
                        "chunk_count": 1,
                        "mention_count": 1,
                    },
                    "is_user_pinned": 0,
                },
            ]

            for character in characters:
                connection.execute(
                    """
                    INSERT INTO project_characters (
                        character_run_id,
                        project_id,
                        character_id,
                        canonical_name,
                        aliases_json,
                        references_json,
                        mention_ids_json,
                        source_candidate_ids_json,
                        evidence_count,
                        is_user_pinned,
                        input_quality_json
                    )
                    VALUES (?, ?, ?, ?, '[]', '[]', ?, '[]', ?, ?, ?)
                    """,
                    (
                        character_run_id,
                        project_id,
                        character["character_id"],
                        character["canonical_name"],
                        json.dumps(character["mention_ids"]),
                        character["evidence_count"],
                        character["is_user_pinned"],
                        json.dumps(character["input_quality"]),
                    ),
                )

        return character_run_id

    def test_suppress_ordinary_character_hides_it_from_run(self):
        project_id = self._save_project()
        character_run_id = self._insert_character_run(project_id)

        with database_session(database_path=self.database_path) as connection:
            row = connection.execute(
                """
                SELECT id, character_id
                FROM project_characters
                WHERE project_id = ?
                  AND character_id = 'ordinary-uncle'
                """,
                (project_id,),
            ).fetchone()

        updated_run = suppress_ordinary_character(
            character_row_id=int(row["id"]),
            database_path=self.database_path,
        )

        self.assertTrue(
            all(
                character["character_id"] != row["character_id"]
                for character in updated_run["characters"]
            )
        )

        latest_run = get_project_character_run(
            character_run_id=character_run_id,
            database_path=self.database_path,
        )
        self.assertTrue(
            all(
                character["character_id"] != row["character_id"]
                for character in latest_run["characters"]
            )
        )

    def test_auto_core_character_cannot_be_deleted(self):
        project_id = self._save_project()
        character_run_id = self._insert_character_run(project_id)
        core_characters = get_core_project_characters(
            project_id=project_id,
            character_run_id=character_run_id,
            database_path=self.database_path,
        )
        auto_core_id = core_characters[0]["character_id"]

        with database_session(database_path=self.database_path) as connection:
            row = connection.execute(
                """
                SELECT id
                FROM project_characters
                WHERE project_id = ?
                  AND character_id = ?
                """,
                (project_id, auto_core_id),
            ).fetchone()

        with self.assertRaisesRegex(ValueError, "自动核心人物不能删除"):
            suppress_ordinary_character(
                character_row_id=int(row["id"]),
                database_path=self.database_path,
            )


if __name__ == "__main__":
    unittest.main()
