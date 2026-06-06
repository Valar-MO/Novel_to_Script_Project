import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.narrative_analysis import (
    get_analysis_database_path,
    get_analysis_provider,
)
from backend.llm.mock_provider import MockProvider
from backend.llm.schemas import MentionExtractionOutput
from backend.main import app
from backend.services.chapter_detector import ChapterSpan
from backend.services.project_storage import ProjectFileData, save_project
from backend.services.text_chunker import TextChunk


class TestNarrativeAnalysisApi(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary_directory.name)
        self.database_path = self.test_root / "test.db"
        self.projects_directory = self.test_root / "projects"
        self.provider = MockProvider()

        app.dependency_overrides[get_analysis_provider] = (
            lambda: self.provider
        )
        app.dependency_overrides[get_analysis_database_path] = (
            lambda: self.database_path
        )

        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.temporary_directory.cleanup()

    def _save_project(self) -> str:
        processed_text = "Han Li entered the valley."

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

        chunk = TextChunk(
            chunk_id="chunk_0001",
            global_order=1,
            source_file_name="novel.txt",
            source_file_order=1,
            start_character=0,
            end_character=len(processed_text),
            character_count=len(processed_text),
            paragraph_start=1,
            paragraph_end=1,
            text=processed_text,
            chapter_order=1,
            chapter_number=1,
            chapter_title="Chapter 1",
            chapter_full_title="Chapter 1",
            chunk_order_in_chapter=1,
            is_chapter_start=True,
            is_chapter_end=True,
        )

        file_data = ProjectFileData(
            file_order=1,
            file_name="novel.txt",
            raw_content=processed_text.encode("utf-8"),
            original_text=processed_text,
            processed_text=processed_text,
            chapters=[chapter],
            chunks=[chunk],
        )

        saved_project = save_project(
            project_name="API Narrative Test",
            files=[file_data],
            database_path=self.database_path,
            projects_directory=self.projects_directory,
            project_id="project-api-test",
        )

        return saved_project.project_id

    def test_start_analysis_returns_404_for_missing_project(self):
        response = self.client.post(
            "/api/projects/missing/narrative-analysis",
            json={
                "max_chunks": 1,
            },
        )

        self.assertEqual(response.status_code, 404)

    def test_start_analysis_rejects_invalid_parameters(self):
        project_id = self._save_project()

        response = self.client.post(
            f"/api/projects/{project_id}/narrative-analysis",
            json={
                "max_chunks": 0,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_start_analysis_returns_summary(self):
        project_id = self._save_project()

        self.provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li entered",
                        "confidence": 0.95,
                    }
                ],
            },
        )

        response = self.client.post(
            f"/api/projects/{project_id}/narrative-analysis",
            json={
                "max_chunks": 1,
            },
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(body["project_id"], project_id)
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["total_chunks"], 1)
        self.assertEqual(body["successful_chunks"], 1)
        self.assertEqual(body["failed_chunks"], 0)
        self.assertEqual(body["cached_chunks"], 0)
        self.assertIsInstance(body["run_id"], int)

    def test_get_analysis_returns_404_for_missing_run(self):
        response = self.client.get(
            "/api/narrative-analysis/9999",
        )

        self.assertEqual(response.status_code, 404)

    def test_get_analysis_returns_saved_units(self):
        project_id = self._save_project()

        self.provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li",
                        "confidence": 0.9,
                    }
                ],
            },
        )

        start_response = self.client.post(
            f"/api/projects/{project_id}/narrative-analysis",
            json={
                "max_chunks": 1,
            },
        )
        run_id = start_response.json()["run_id"]

        response = self.client.get(
            f"/api/narrative-analysis/{run_id}",
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(body["id"], run_id)
        self.assertEqual(body["project_id"], project_id)
        self.assertEqual(len(body["units"]), 1)
        self.assertIsInstance(body["units"][0]["result"], dict)
        self.assertIsInstance(
            body["units"][0]["validated_result"],
            dict,
        )
        self.assertEqual(
            body["units"][0]["result"]["mentions"][0]["mention_text"],
            "Han Li",
        )
        self.assertTrue(
            body["units"][0]["validated_result"]["mentions"][0][
                "evidence_validated"
            ]
        )


if __name__ == "__main__":
    unittest.main()
