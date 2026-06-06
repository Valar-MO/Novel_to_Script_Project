import unittest

from backend.llm.schemas import (
    EventFrameExtractionOutput,
    MentionExtractionOutput,
    RelationExtractionOutput,
)
from backend.services.evidence_validator import validate_evidence


class TestEvidenceValidator(unittest.TestCase):
    def test_event_frame_output_accepts_common_root_aliases(self):
        payload = {
            "events": [
                {
                    "trigger_text": "walked",
                    "event_type": "movement",
                    "arguments": [
                        {
                            "role": "actor",
                            "mention_id": "chunk_0001_m_001",
                            "mention_text": "Han Li",
                        }
                    ],
                    "evidence_text": "Han Li walked",
                    "confidence": 0.9,
                }
            ]
        }

        result = EventFrameExtractionOutput.model_validate(payload)

        self.assertEqual(len(result.event_frames), 1)
        self.assertEqual(result.event_frames[0].trigger_text, "walked")

    def test_event_frame_output_accepts_single_event_alias(self):
        payload = {
            "event": {
                "trigger_text": "walked",
                "event_type": "movement",
                "arguments": [],
                "evidence_text": "Han Li walked",
                "confidence": 0.9,
            }
        }

        result = EventFrameExtractionOutput.model_validate(payload)

        self.assertEqual(len(result.event_frames), 1)

    def test_marks_mention_evidence_with_offsets(self):
        text = "Han Li walked into Seven Mysteries Sect."
        extraction = MentionExtractionOutput(
            mentions=[
                {
                    "mention_type": "character",
                    "mention_text": "Han Li",
                    "evidence_text": "Han Li",
                    "confidence": 0.9,
                },
                {
                    "mention_type": "organization",
                    "mention_text": "Seven Mysteries Sect",
                    "evidence_text": "Seven Mysteries Sect",
                    "confidence": 0.8,
                },
            ],
        )

        result = validate_evidence(
            target_text=text,
            extraction=extraction,
        )

        first_mention = result["mentions"][0]
        second_mention = result["mentions"][1]

        self.assertTrue(first_mention["evidence_validated"])
        self.assertEqual(first_mention["start_offset"], 0)
        self.assertEqual(first_mention["end_offset"], len("Han Li"))

        self.assertTrue(second_mention["evidence_validated"])
        self.assertEqual(
            text[
                second_mention["start_offset"]:
                second_mention["end_offset"]
            ],
            "Seven Mysteries Sect",
        )

    def test_marks_missing_mention_evidence_as_invalid(self):
        extraction = MentionExtractionOutput(
            mentions=[
                {
                    "mention_type": "character",
                    "mention_text": "Han Li",
                    "evidence_text": "Han Li leaves",
                    "confidence": 0.7,
                }
            ],
        )

        result = validate_evidence(
            target_text="Han Li stayed in the room.",
            extraction=extraction,
        )

        mention = result["mentions"][0]

        self.assertFalse(mention["evidence_validated"])
        self.assertIsNone(mention["start_offset"])
        self.assertIsNone(mention["end_offset"])

    def test_validates_relation_evidence(self):
        text = "Doctor Mo was Han Li's teacher."
        extraction = RelationExtractionOutput(
            relations=[
                {
                    "source_mention": "Doctor Mo",
                    "source_mention_id": "chunk_0001_m_001",
                    "relation": "teacher",
                    "target_mention": "Han Li",
                    "target_mention_id": "chunk_0001_m_002",
                    "evidence_text": "Doctor Mo was Han Li's teacher",
                    "confidence": 0.88,
                }
            ],
        )

        result = validate_evidence(
            target_text=text,
            extraction=extraction,
        )

        relation = result["relations"][0]

        self.assertTrue(relation["evidence_validated"])
        self.assertEqual(relation["start_offset"], 0)
        self.assertEqual(
            relation["end_offset"],
            len("Doctor Mo was Han Li's teacher"),
        )

    def test_validates_event_frame_evidence(self):
        text = "Han Li walked into Seven Mysteries Sect."
        extraction = EventFrameExtractionOutput(
            event_frames=[
                {
                    "trigger_text": "walked",
                    "event_type": "movement",
                    "arguments": [
                        {
                            "role": "actor",
                            "mention_id": "chunk_0001_m_001",
                            "mention_text": "Han Li",
                        },
                        {
                            "role": "destination",
                            "mention_id": "chunk_0001_m_002",
                            "mention_text": "Seven Mysteries Sect",
                        },
                    ],
                    "evidence_text": (
                        "Han Li walked into Seven Mysteries Sect"
                    ),
                    "confidence": 0.9,
                }
            ],
        )

        result = validate_evidence(
            target_text=text,
            extraction=extraction,
        )

        event_frame = result["event_frames"][0]

        self.assertTrue(event_frame["evidence_validated"])
        self.assertEqual(event_frame["start_offset"], 0)

    def test_accepts_plain_dict_pipeline_result(self):
        result = validate_evidence(
            target_text="Han Li entered the valley.",
            extraction={
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "Han Li",
                        "evidence_text": "Han Li",
                        "confidence": 0.9,
                    }
                ]
            },
        )

        self.assertTrue(
            result["mentions"][0]["evidence_validated"]
        )


if __name__ == "__main__":
    unittest.main()
