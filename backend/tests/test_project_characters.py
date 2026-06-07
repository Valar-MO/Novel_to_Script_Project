import json
import tempfile
import unittest
from pathlib import Path

from backend.services.chapter_detector import ChapterSpan
from backend.services.project_characters import build_project_characters
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
        text = "Han Li appears.\n\nHan Zhu appears."
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
        split_at = text.index("Han Zhu")
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
            project_name="Character Table Test",
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

    def _insert_narrative_run(
        self,
        *,
        project_id: str,
        first_result: dict,
        second_status: str = "completed",
        second_result: dict | None = None,
    ) -> int:
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
                    successful_chunks,
                    failed_chunks
                )
                VALUES (?, 'mock', 'mock', 'p', 's', 'partial', 2, 2, 1, ?)
                """,
                (
                    project_id,
                    1 if second_status == "failed" else 0,
                ),
            )
            run_id = int(cursor.lastrowid)
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
                VALUES (?, ?, ?, ?, 't1', 'a1', 'completed', 'mock', 'mock', 'p', 's', ?)
                """,
                (
                    run_id,
                    project_id,
                    chunks[0]["id"],
                    chunks[0]["chunk_id"],
                    json.dumps(first_result, ensure_ascii=False),
                ),
            )
            if second_status == "failed":
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
                        error_message
                    )
                    VALUES (?, ?, ?, ?, 't2', 'a2', 'failed', 'mock', 'mock', 'p', 's', 'model failed')
                    """,
                    (
                        run_id,
                        project_id,
                        chunks[1]["id"],
                        chunks[1]["chunk_id"],
                    ),
                )
            else:
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
                    VALUES (?, ?, ?, ?, 't2', 'a2', 'completed', 'mock', 'mock', 'p', 's', ?)
                    """,
                    (
                        run_id,
                        project_id,
                        chunks[1]["id"],
                        chunks[1]["chunk_id"],
                        json.dumps(second_result or {}, ensure_ascii=False),
                    ),
                )
        return run_id

    def test_build_merges_must_link_and_blocks_cannot_link(self):
        project_id = self._save_project()
        result = {
            "mentions": [
                {
                    "mention_id": "m_hanli",
                    "mention_type": "character",
                    "mention_text": "Han Li",
                    "evidence_validated": True,
                },
                {
                    "mention_id": "m_er",
                    "mention_type": "character",
                    "mention_text": "Er Leng",
                    "evidence_validated": True,
                },
                {
                    "mention_id": "m_hanzhu",
                    "mention_type": "character",
                    "mention_text": "Han Zhu",
                    "evidence_validated": True,
                },
            ],
            "relations": [
                {
                    "source_mention_id": "m_hanli",
                    "relation": "alias",
                    "target_mention_id": "m_er",
                    "evidence_text": "Han Li, alias Er Leng",
                },
                {
                    "source_mention_id": "m_hanli",
                    "relation": "brother",
                    "target_mention_id": "m_hanzhu",
                    "evidence_text": "brother Han Zhu",
                },
            ],
            "event_frames": [],
            "character_candidates": [
                {
                    "character_candidate_id": "c_hanli",
                    "canonical_name": "Han Li",
                    "aliases": ["Han Li"],
                    "references": [],
                    "mention_ids": ["m_hanli"],
                    "confidence": 0.95,
                },
                {
                    "character_candidate_id": "c_er",
                    "canonical_name": "Er Leng",
                    "aliases": ["Er Leng"],
                    "references": [],
                    "mention_ids": ["m_er"],
                    "confidence": 0.99,
                },
                {
                    "character_candidate_id": "c_hanzhu",
                    "canonical_name": "Han Zhu",
                    "aliases": ["Han Zhu"],
                    "references": [],
                    "mention_ids": ["m_hanzhu"],
                    "confidence": 0.99,
                },
            ],
            "layer_statuses": {
                "mentions": "completed",
                "relations": "completed",
                "event_frames": "completed",
                "character_candidates": "completed",
            },
        }
        run_id = self._insert_narrative_run(
            project_id=project_id,
            first_result=result,
            second_result={
                "mentions": [],
                "relations": [],
                "event_frames": [],
                "character_candidates": [],
            },
        )

        output = build_project_characters(
            project_id=project_id,
            narrative_run_id=run_id,
            database_path=self.database_path,
        )

        character_names = {
            character["canonical_name"]: character
            for character in output["characters"]
        }
        self.assertIn("Han Li", character_names)
        self.assertIn("Er Leng", character_names["Han Li"]["aliases"])
        self.assertIn("Han Zhu", character_names)
        self.assertTrue(
            any(
                decision["decision"] == "must_link"
                for decision in output["merge_decisions"]
            )
        )
        self.assertTrue(
            any(
                decision["decision"] == "cannot_link"
                for decision in output["merge_decisions"]
            )
        )
        self.assertFalse(
            any(
                decision["decision"] == "separate"
                and decision["merge_score"] == 0
                and not decision["evidence"]
                and not decision["conflicts"]
                for decision in output["merge_decisions"]
            )
        )

    def test_build_merges_reciprocal_alias_without_model_confidence(self):
        project_id = self._save_project()
        result = {
            "mentions": [
                {
                    "mention_id": "m_hanli",
                    "mention_type": "character",
                    "mention_text": "Han Li",
                    "evidence_validated": True,
                },
                {
                    "mention_id": "m_er",
                    "mention_type": "character",
                    "mention_text": "Er Leng",
                    "evidence_validated": True,
                },
            ],
            "relations": [],
            "event_frames": [],
            "character_candidates": [
                {
                    "character_candidate_id": "c_hanli",
                    "canonical_name": "Han Li",
                    "aliases": ["Han Li", "Er Leng"],
                    "references": [],
                    "mention_ids": ["m_hanli"],
                    "confidence": 0.1,
                },
                {
                    "character_candidate_id": "c_er",
                    "canonical_name": "Er Leng",
                    "aliases": ["Er Leng", "Han Li"],
                    "references": [],
                    "mention_ids": ["m_er"],
                    "confidence": 0.1,
                },
            ],
            "layer_statuses": {
                "mentions": "completed",
                "relations": "completed",
                "event_frames": "completed",
                "character_candidates": "completed",
            },
        }
        run_id = self._insert_narrative_run(
            project_id=project_id,
            first_result=result,
            second_result={
                "mentions": [],
                "relations": [],
                "event_frames": [],
                "character_candidates": [],
            },
        )

        output = build_project_characters(
            project_id=project_id,
            narrative_run_id=run_id,
            database_path=self.database_path,
        )

        self.assertEqual(len(output["characters"]), 1)
        self.assertEqual(output["characters"][0]["canonical_name"], "Han Li")
        self.assertIn("Er Leng", output["characters"][0]["aliases"])
        self.assertTrue(
            any(
                decision["decision"] == "merged"
                and decision["merge_score"] >= 0.8
                and any(
                    evidence["type"] == "reciprocal_alias"
                    for evidence in decision["evidence"]
                )
                for decision in output["merge_decisions"]
            )
        )

    def test_build_keeps_weak_shared_alias_as_ambiguous(self):
        project_id = self._save_project()
        result = {
            "mentions": [
                {
                    "mention_id": "m_left",
                    "mention_type": "character",
                    "mention_text": "Left Name",
                    "evidence_validated": True,
                },
                {
                    "mention_id": "m_right",
                    "mention_type": "character",
                    "mention_text": "Right Name",
                    "evidence_validated": True,
                },
            ],
            "relations": [],
            "event_frames": [],
            "character_candidates": [
                {
                    "character_candidate_id": "c_left",
                    "canonical_name": "Left Name",
                    "aliases": ["Shared Alias"],
                    "references": [],
                    "mention_ids": ["m_left"],
                    "confidence": 1.0,
                },
                {
                    "character_candidate_id": "c_right",
                    "canonical_name": "Right Name",
                    "aliases": ["Shared Alias"],
                    "references": [],
                    "mention_ids": ["m_right"],
                    "confidence": 1.0,
                },
            ],
            "layer_statuses": {
                "mentions": "completed",
                "relations": "completed",
                "event_frames": "completed",
                "character_candidates": "completed",
            },
        }
        run_id = self._insert_narrative_run(
            project_id=project_id,
            first_result=result,
            second_result={
                "mentions": [],
                "relations": [],
                "event_frames": [],
                "character_candidates": [],
            },
        )

        output = build_project_characters(
            project_id=project_id,
            narrative_run_id=run_id,
            database_path=self.database_path,
        )

        self.assertEqual(len(output["characters"]), 2)
        self.assertTrue(
            any(
                decision["decision"] == "ambiguous"
                and decision["merge_score"] >= 0.4
                for decision in output["merge_decisions"]
            )
        )

    def test_build_is_partial_when_some_units_failed(self):
        project_id = self._save_project()
        result = {
            "mentions": [
                {
                    "mention_id": "m_hanli",
                    "mention_type": "character",
                    "mention_text": "Han Li",
                    "evidence_validated": True,
                }
            ],
            "relations": [],
            "event_frames": [],
            "character_candidates": [],
            "layer_statuses": {
                "mentions": "completed",
                "relations": "skipped",
                "event_frames": "skipped",
                "character_candidates": "skipped",
            },
        }
        run_id = self._insert_narrative_run(
            project_id=project_id,
            first_result=result,
            second_status="failed",
        )

        output = build_project_characters(
            project_id=project_id,
            narrative_run_id=run_id,
            database_path=self.database_path,
        )

        self.assertEqual(output["status"], "partial")
        self.assertEqual(len(output["characters"]), 1)
        self.assertGreater(len(output["input_gaps"]), 0)


if __name__ == "__main__":
    unittest.main()
