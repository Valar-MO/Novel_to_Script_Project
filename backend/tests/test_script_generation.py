import tempfile
import unittest
from pathlib import Path

from backend.llm.mock_provider import MockProvider
from backend.llm.schemas import ScriptGenerationOutput
from backend.services.chapter_detector import ChapterSpan
from backend.services.project_storage import ProjectFileData, save_project
from backend.services.script_generation import (
    create_script_generation_job,
    execute_script_generation_job,
    get_script_generation_run,
    get_script_generation_scenes,
    regenerate_script_scene,
    update_script_scene,
)
from backend.services.text_chunker import TextChunk


class TestScriptGeneration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary_directory.name)
        self.database_path = self.test_root / "test.db"
        self.projects_directory = self.test_root / "projects"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _save_project(self) -> str:
        first_chunk = "Han Li opens the door. Han Li walks inside.\n\n"
        second_chunk = "Han Zhu waits outside."
        text = first_chunk + second_chunk
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
                end_character=len(first_chunk),
                character_count=len(first_chunk),
                paragraph_start=1,
                paragraph_end=1,
                text=first_chunk,
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
                start_character=len(first_chunk),
                end_character=len(text),
                character_count=len(second_chunk),
                paragraph_start=2,
                paragraph_end=2,
                text=second_chunk,
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
            project_name="Script Generation Test",
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

    async def test_generates_scene_and_source_span_without_extraction(self):
        project_id = self._save_project()
        provider = MockProvider(model_name="mock-script-model")
        provider.register_response(
            ScriptGenerationOutput,
            {
                "scenes": [
                    {
                        "order_in_unit": 1,
                        "continue_previous_scene": False,
                        "interior_exterior": "INT",
                        "location": "House",
                        "time_of_day": "Day",
                        "heading": "INT. House - Day",
                        "characters": [
                            {
                                "character_id": None,
                                "name": "Han Li",
                            }
                        ],
                        "script_text": "Han Li opens the door and steps in.",
                        "scene_summary": "Han Li enters the house.",
                        "source_anchor": {
                            "start_text": "Han Li opens",
                            "end_text": "walks inside.",
                        },
                        "adaptation_notes": [],
                        "warnings": [],
                    }
                ],
                "warnings": [],
            },
        )

        result = create_script_generation_job(
            project_id=project_id,
            provider=provider,
            database_path=self.database_path,
            max_chunks=1,
        )

        self.assertEqual(result.status, "queued")
        self.assertEqual(result.total_chunks, 1)

        await execute_script_generation_job(
            run_id=result.run_id,
            provider=provider,
            database_path=self.database_path,
        )

        run = get_script_generation_run(
            run_id=result.run_id,
            database_path=self.database_path,
        )
        scenes = get_script_generation_scenes(
            run_id=result.run_id,
            database_path=self.database_path,
        )

        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["processed_chunks"], 1)
        self.assertEqual(run["scene_count"], 1)
        self.assertEqual(len(scenes["scenes"]), 1)
        self.assertEqual(scenes["scenes"][0]["heading"], "INT. House - Day")
        self.assertEqual(len(scenes["scenes"][0]["source_spans"]), 1)
        self.assertEqual(
            scenes["scenes"][0]["source_spans"][0]["evidence_text"],
            "Han Li opens the door. Han Li walks inside.",
        )
        self.assertEqual(provider.call_count, 1)

    async def test_edits_and_regenerates_single_scene(self):
        project_id = self._save_project()
        provider = MockProvider(model_name="mock-script-model")
        provider.register_response(
            ScriptGenerationOutput,
            {
                "scenes": [
                    {
                        "order_in_unit": 1,
                        "continue_previous_scene": False,
                        "interior_exterior": "INT",
                        "location": "House",
                        "time_of_day": "Day",
                        "heading": "INT. House - Day",
                        "characters": [],
                        "script_text": "Han Li opens the door.",
                        "scene_summary": "Han Li enters.",
                        "source_anchor": {
                            "start_text": "Han Li opens",
                            "end_text": "walks inside.",
                        },
                        "adaptation_notes": [],
                        "warnings": [],
                    }
                ],
                "warnings": [],
            },
        )
        provider.register_response(
            ScriptGenerationOutput,
            {
                "scenes": [
                    {
                        "order_in_unit": 1,
                        "continue_previous_scene": False,
                        "interior_exterior": "INT",
                        "location": "House",
                        "time_of_day": "Day",
                        "heading": "INT. House - Day - Revised",
                        "characters": [],
                        "script_text": "Han Li pushes the door and enters.",
                        "scene_summary": "Han Li enters after opening the door.",
                        "source_anchor": {
                            "start_text": "Han Li opens",
                            "end_text": "walks inside.",
                        },
                        "adaptation_notes": [],
                        "warnings": [],
                    }
                ],
                "warnings": [],
            },
        )
        result = create_script_generation_job(
            project_id=project_id,
            provider=provider,
            database_path=self.database_path,
            max_chunks=1,
        )
        await execute_script_generation_job(
            run_id=result.run_id,
            provider=provider,
            database_path=self.database_path,
        )
        scenes = get_script_generation_scenes(
            run_id=result.run_id,
            database_path=self.database_path,
        )
        scene_id = scenes["scenes"][0]["id"]

        edited = update_script_scene(
            scene_id=scene_id,
            heading="Edited Heading",
            script_text="Edited scene text.",
            database_path=self.database_path,
        )

        self.assertEqual(edited["heading"], "Edited Heading")
        self.assertTrue(edited["is_user_edited"])

        regenerated = await regenerate_script_scene(
            scene_id=scene_id,
            provider=provider,
            instruction="Make it more faithful.",
            database_path=self.database_path,
        )

        self.assertEqual(
            regenerated["heading"],
            "INT. House - Day - Revised",
        )
        self.assertFalse(regenerated["is_user_edited"])
        self.assertEqual(provider.call_count, 2)


if __name__ == "__main__":
    unittest.main()
