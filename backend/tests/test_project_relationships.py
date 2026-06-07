import json
import tempfile
import unittest
from pathlib import Path

from backend.services.chapter_detector import ChapterSpan
from backend.services.project_characters import (
    build_project_characters,
    get_core_project_characters,
    suppress_ordinary_character,
)
from backend.services.project_relationships import (
    build_project_relationships,
    create_project_relationship,
    get_project_relationships,
)
from backend.services.project_storage import ProjectFileData, save_project
from backend.services.text_chunker import TextChunk
from backend.storage.database import database_session


class TestProjectRelationships(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary_directory.name)
        self.database_path = self.test_root / "test.db"
        self.projects_directory = self.test_root / "projects"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _save_project(self) -> str:
        text = (
            "Han Li met Doctor Mo.\n\n"
            "Doctor Mo taught Han Li."
        )
        split_at = text.index("Doctor Mo taught")
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
            project_name="Relationship Test",
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

    def _insert_narrative_run(self, project_id: str) -> int:
        with database_session(database_path=self.database_path) as connection:
            chunks = connection.execute(
                """
                SELECT id, chunk_id
                FROM text_chunks
                ORDER BY id
                """
            ).fetchall()
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
            run_id = int(cursor.lastrowid)

            results = [
                {
                    "mentions": [
                        {
                            "mention_id": "m_hanli_1",
                            "mention_type": "character",
                            "mention_text": "Han Li",
                            "evidence_validated": True,
                        },
                        {
                            "mention_id": "m_doctor_1",
                            "mention_type": "character",
                            "mention_text": "Doctor Mo",
                            "evidence_validated": True,
                        },
                    ],
                    "relations": [
                        {
                            "source_mention_id": "m_doctor_1",
                            "target_mention_id": "m_hanli_1",
                            "source_mention": "Doctor Mo",
                            "target_mention": "Han Li",
                            "relation": "teacher",
                            "evidence_text": "Han Li met Doctor Mo",
                            "start_offset": 0,
                            "end_offset": 20,
                        }
                    ],
                },
                {
                    "mentions": [
                        {
                            "mention_id": "m_doctor_2",
                            "mention_type": "character",
                            "mention_text": "Doctor Mo",
                            "evidence_validated": True,
                        },
                        {
                            "mention_id": "m_hanli_2",
                            "mention_type": "character",
                            "mention_text": "Han Li",
                            "evidence_validated": True,
                        },
                    ],
                    "relations": [
                        {
                            "source_mention_id": "m_doctor_2",
                            "target_mention_id": "m_hanli_2",
                            "source_mention": "Doctor Mo",
                            "target_mention": "Han Li",
                            "relation": "teacher",
                            "evidence_text": "Doctor Mo taught Han Li",
                            "start_offset": 0,
                            "end_offset": 23,
                        }
                    ],
                },
            ]

            for chunk, result in zip(chunks, results, strict=True):
                connection.execute(
                    """
                    INSERT INTO narrative_unit_analyses (
                        run_id,
                        project_id,
                        chunk_database_id,
                        chunk_id,
                        text_hash,
                        analysis_input_hash,
                        status,
                        provider,
                        model,
                        prompt_version,
                        schema_version,
                        validated_result_json
                    )
                    VALUES (?, ?, ?, ?, 't', 'a', 'completed', 'mock', 'mock', 'p', 's', ?)
                    """,
                    (
                        run_id,
                        project_id,
                        chunk["id"],
                        chunk["chunk_id"],
                        json.dumps(result, ensure_ascii=False),
                    ),
                )

        return run_id

    def _insert_character_run(self, project_id: str, narrative_run_id: int) -> int:
        with database_session(database_path=self.database_path) as connection:
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
            run_id = int(cursor.lastrowid)

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
                        run_id,
                        project_id,
                        character["character_id"],
                        character["canonical_name"],
                        json.dumps(character["mention_ids"]),
                        character["evidence_count"],
                        character["is_user_pinned"],
                        json.dumps(character["input_quality"]),
                    ),
                )

        return run_id

    def test_build_relationships_keeps_single_core_evidence_relation(self):
        project_id = self._save_project()
        narrative_run_id = self._insert_narrative_run(project_id)
        character_run = build_project_characters(
            project_id=project_id,
            narrative_run_id=narrative_run_id,
            database_path=self.database_path,
        )

        relationships = build_project_relationships(
            project_id=project_id,
            character_run_id=character_run["id"],
            database_path=self.database_path,
        )

        self.assertEqual(len(relationships["core_relationships"]), 1)
        self.assertGreaterEqual(
            relationships["core_relationships"][0]["evidence_count"],
            1,
        )

    def test_get_relationships_returns_user_relationships(self):
        project_id = self._save_project()
        narrative_run_id = self._insert_narrative_run(project_id)
        build_project_characters(
            project_id=project_id,
            narrative_run_id=narrative_run_id,
            database_path=self.database_path,
        )
        build_project_relationships(
            project_id=project_id,
            database_path=self.database_path,
        )

        relationships = get_project_relationships(
            project_id=project_id,
            database_path=self.database_path,
        )
        ai_count = len(relationships["relationships"])

        created_relationship = create_project_relationship(
            project_id=project_id,
            source_character_id="hanli",
            source_character_name="Han Li",
            target_character_id="doctormo",
            target_character_name="Doctor Mo",
            relation_label="ally",
            relation_description="User curated",
            evidence_text="",
            database_path=self.database_path,
        )

        created = get_project_relationships(
            project_id=project_id,
            database_path=self.database_path,
        )

        self.assertEqual(created_relationship["relation_label"], "ally")
        self.assertEqual(created["core_relationships"][0]["relation_label"], "ally")
        self.assertEqual(
            len(created["relationships"]),
            ai_count + 1,
        )
        self.assertEqual(
            created["relationships"][0]["source_type"],
            "user",
        )


class TestProjectCharacterSuppressions(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary_directory.name)
        self.database_path = self.test_root / "test.db"
        self.projects_directory = self.test_root / "projects"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _prepare_project_with_character_run(self):
        helper = TestProjectRelationships()
        helper.temporary_directory = self.temporary_directory
        helper.test_root = self.test_root
        helper.database_path = self.database_path
        helper.projects_directory = self.projects_directory

        project_id = helper._save_project()
        narrative_run_id = helper._insert_narrative_run(project_id)
        character_run_id = helper._insert_character_run(project_id, narrative_run_id)
        return project_id, character_run_id

    def test_ordinary_character_can_be_suppressed(self):
        project_id, character_run_id = self._prepare_project_with_character_run()

        with database_session(database_path=self.database_path) as connection:
            ordinary_character = connection.execute(
                """
                SELECT id, character_id
                FROM project_characters
                WHERE project_id = ?
                  AND canonical_name = 'Third Uncle'
                """,
                (project_id,),
            ).fetchone()

        updated_run = suppress_ordinary_character(
            character_row_id=int(ordinary_character["id"]),
            database_path=self.database_path,
        )

        self.assertTrue(
            all(
                character["character_id"] != ordinary_character["character_id"]
                for character in updated_run["characters"]
            )
        )

    def test_auto_core_character_cannot_be_suppressed(self):
        project_id, character_run_id = self._prepare_project_with_character_run()
        core_characters = get_core_project_characters(
            project_id=project_id,
            character_run_id=character_run_id,
            database_path=self.database_path,
        )
        auto_core_id = core_characters[0]["character_id"]

        with database_session(database_path=self.database_path) as connection:
            core_row = connection.execute(
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
                character_row_id=int(core_row["id"]),
                database_path=self.database_path,
            )

    def test_suppressing_ordinary_character_removes_related_relationships(self):
        project_id, character_run_id = self._prepare_project_with_character_run()

        with database_session(database_path=self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO project_relationships (
                    project_id,
                    source_character_id,
                    source_character_name,
                    target_character_id,
                    target_character_name,
                    relation_label,
                    relation_description,
                    source_type,
                    evidence_text,
                    evidence_count
                )
                VALUES (?, 'hanli', 'Han Li', 'ordinary-uncle', 'Third Uncle', '叔侄', '', 'ai', '', 1)
                """,
                (project_id,),
            )
            ordinary_row = connection.execute(
                """
                SELECT id, character_id
                FROM project_characters
                WHERE project_id = ?
                  AND character_id = 'ordinary-uncle'
                """,
                (project_id,),
            ).fetchone()

        suppress_ordinary_character(
            character_row_id=int(ordinary_row["id"]),
            database_path=self.database_path,
        )

        relationships = get_project_relationships(
            project_id=project_id,
            database_path=self.database_path,
        )

        self.assertTrue(
            all(
                relationship["source_character_id"] != ordinary_row["character_id"]
                and relationship["target_character_id"] != ordinary_row["character_id"]
                for relationship in relationships["relationships"]
            )
        )


if __name__ == "__main__":
    unittest.main()
