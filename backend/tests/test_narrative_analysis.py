import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.llm.mock_provider import MockProvider
from backend.llm.schemas import (
    MentionExtractionOutput,
    RelationExtractionOutput,
)
from backend.services.chapter_detector import ChapterSpan
from backend.services.narrative_analysis import (
    ANALYSIS_STATUS_COMPLETED,
    _filter_validated_relations,
    analyze_project_narrative,
    get_narrative_analysis_run,
)
from backend.services.project_storage import ProjectFileData, save_project
from backend.services.text_chunker import TextChunk


class TestNarrativeAnalysis(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary_directory.name)
        self.database_path = self.test_root / "test.db"
        self.projects_directory = self.test_root / "projects"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_relation_filter_requires_valid_mention_ids(self):
        mention_by_id = {
            "chunk_0001_m_001": {
                "mention_id": "chunk_0001_m_001",
                "mention_text": "Han Li",
            },
            "chunk_0001_m_002": {
                "mention_id": "chunk_0001_m_002",
                "mention_text": "Doctor Mo",
            },
        }

        result = _filter_validated_relations(
            {
                "relations": [
                    {
                        "source_mention": "Han Li",
                        "source_mention_id": "chunk_0001_m_001",
                        "relation": "teacher",
                        "target_mention": "Doctor Mo",
                        "target_mention_id": "chunk_0001_m_002",
                        "evidence_text": "Han Li met Doctor Mo",
                        "evidence_validated": True,
                        "confidence": 0.9,
                    },
                    {
                        "source_mention": "Han Li",
                        "source_mention_id": "chunk_0001_m_001",
                        "relation": "teacher",
                        "target_mention": "Missing",
                        "target_mention_id": "chunk_0001_m_999",
                        "evidence_text": "Han Li met Doctor Mo",
                        "evidence_validated": True,
                        "confidence": 0.9,
                    },
                ],
                "warnings": [],
            },
            mention_by_id=mention_by_id,
            allowed_mention_ids={
                "chunk_0001_m_001",
                "chunk_0001_m_002",
            },
            chunk_id="chunk_0001",
        )

        self.assertEqual(len(result["relations"]), 1)
        self.assertTrue(
            result["relations"][0]["relation_id"].startswith(
                "chunk_0001_r_"
            )
        )

    def _save_project(self) -> str:
        processed_text = (
            "Han Li entered the valley with Doctor Mo.\n\n"
            "Doctor Mo waited beside the gate."
        )
        split_at = processed_text.rindex("Doctor Mo")

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
            end_character=len(processed_text),
            character_count=len(processed_text),
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
                text=processed_text[:split_at],
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
                end_character=len(processed_text),
                character_count=len(processed_text) - split_at,
                paragraph_start=2,
                paragraph_end=2,
                text=processed_text[split_at:],
                chapter_order=1,
                chapter_number=1,
                chapter_title="Chapter 1",
                chapter_full_title="Chapter 1",
                chunk_order_in_chapter=2,
                is_chapter_start=False,
                is_chapter_end=True,
            ),
        ]

        file_data = ProjectFileData(
            file_order=1,
            file_name="novel.txt",
            raw_content=processed_text.encode("utf-8"),
            original_text=processed_text,
            processed_text=processed_text,
            chapters=[chapter],
            chunks=chunks,
        )

        saved_project = save_project(
            project_name="Narrative Test",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-narrative-test",
        )

        return saved_project.project_id

    async def test_analyze_project_persists_character_relations(self):
        project_id = self._save_project()
        provider = MockProvider()

        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li",
                        "confidence": 0.95,
                    },
                    {
                        "mention_type": "character",
                        "mention_text": "Doctor Mo",
                        "evidence_text": "Doctor Mo",
                        "confidence": 0.9,
                    },
                ],
            },
        )
        provider.register_response(
            RelationExtractionOutput,
            {
                "relations": [
                    {
                        "source_mention": "Han Li",
                        "source_mention_id": (
                            "chunk_0001_m_character_0_6"
                        ),
                        "relation": "companion",
                        "target_mention": "Doctor Mo",
                        "target_mention_id": (
                            "chunk_0001_m_character_31_40"
                        ),
                        "evidence_text": (
                            "Han Li entered the valley with Doctor Mo"
                        ),
                        "confidence": 0.9,
                    }
                ],
            },
        )
        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Doctor Mo",
                        "evidence_text": "Doctor Mo",
                        "confidence": 0.93,
                    }
                ],
            },
        )

        result = await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            previous_context_chars=20,
        )

        self.assertEqual(result.status, ANALYSIS_STATUS_COMPLETED)
        self.assertEqual(result.total_chunks, 2)
        self.assertEqual(result.successful_chunks, 2)
        self.assertEqual(result.failed_chunks, 0)
        self.assertEqual(result.cached_chunks, 0)
        self.assertEqual(provider.call_count, 4)

        connection = sqlite3.connect(self.database_path)
        try:
            layer_cache_count = connection.execute(
                "SELECT COUNT(*) FROM narrative_layer_cache"
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(layer_cache_count, 4)

        saved_run = get_narrative_analysis_run(
            run_id=result.run_id,
            database_path=self.database_path,
        )

        self.assertEqual(saved_run["status"], ANALYSIS_STATUS_COMPLETED)
        self.assertEqual(len(saved_run["units"]), 2)

        first_validated = json.loads(
            saved_run["units"][0]["validated_result_json"]
        )

        self.assertEqual(len(first_validated["mentions"]), 2)
        self.assertEqual(
            first_validated["relations"][0]["source_mention"],
            "Han Li",
        )
        self.assertEqual(
            first_validated["relations"][0]["target_mention"],
            "Doctor Mo",
        )
        self.assertNotIn("event_frames", first_validated)
        self.assertNotIn("character_candidates", first_validated)
        self.assertNotIn(
            "event_frames",
            first_validated["layer_statuses"],
        )
        self.assertNotIn(
            "character_candidates",
            first_validated["layer_statuses"],
        )

    async def test_analyze_project_can_limit_chunks(self):
        project_id = self._save_project()

        result = await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=MockProvider(),
            max_chunks=1,
        )

        saved_run = get_narrative_analysis_run(
            run_id=result.run_id,
            database_path=self.database_path,
        )

        self.assertEqual(result.total_chunks, 1)
        self.assertEqual(len(saved_run["units"]), 1)

    async def test_second_identical_analysis_uses_cache(self):
        project_id = self._save_project()
        provider = MockProvider()

        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li",
                        "confidence": 0.95,
                    }
                ],
            },
        )

        first_result = await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=1,
        )
        second_result = await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=1,
        )

        self.assertEqual(provider.call_count, 2)
        self.assertEqual(first_result.cached_chunks, 0)
        self.assertEqual(second_result.cached_chunks, 1)

        second_run = get_narrative_analysis_run(
            run_id=second_result.run_id,
            database_path=self.database_path,
        )

        self.assertTrue(second_run["units"][0]["cache_hit"])
        self.assertIsNotNone(
            second_run["units"][0]["cache_source_unit_id"]
        )

    async def test_changed_context_does_not_use_cache(self):
        project_id = self._save_project()
        provider = MockProvider()

        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Doctor Mo",
                        "evidence_text": "Doctor Mo",
                        "confidence": 0.93,
                    }
                ],
            },
        )
        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Doctor Mo",
                        "evidence_text": "Doctor Mo",
                        "confidence": 0.94,
                    }
                ],
            },
        )

        await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=2,
            previous_context_chars=20,
        )
        await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=2,
            previous_context_chars=10,
        )

        self.assertEqual(provider.call_count, 5)

    async def test_force_reanalyze_bypasses_cache(self):
        project_id = self._save_project()
        provider = MockProvider()

        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li",
                        "confidence": 0.95,
                    }
                ],
            },
        )
        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li",
                        "confidence": 0.96,
                    }
                ],
            },
        )

        await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=1,
        )
        second_result = await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=1,
            force_reanalyze=True,
        )

        self.assertEqual(provider.call_count, 4)
        self.assertEqual(second_result.cached_chunks, 0)

    async def test_cached_result_is_saved_into_new_run(self):
        project_id = self._save_project()
        provider = MockProvider()

        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li",
                        "confidence": 0.95,
                    }
                ],
            },
        )

        first_result = await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=1,
        )
        second_result = await analyze_project_narrative(
            project_id=project_id,
            database_path=self.database_path,
            provider=provider,
            max_chunks=1,
        )

        first_run = get_narrative_analysis_run(
            run_id=first_result.run_id,
            database_path=self.database_path,
        )
        second_run = get_narrative_analysis_run(
            run_id=second_result.run_id,
            database_path=self.database_path,
        )

        self.assertNotEqual(first_result.run_id, second_result.run_id)
        self.assertEqual(len(second_run["units"]), 1)
        self.assertEqual(
            second_run["units"][0]["cache_source_unit_id"],
            first_run["units"][0]["id"],
        )
        self.assertEqual(
            second_run["units"][0]["validated_result_json"],
            first_run["units"][0]["validated_result_json"],
        )


if __name__ == "__main__":
    unittest.main()
