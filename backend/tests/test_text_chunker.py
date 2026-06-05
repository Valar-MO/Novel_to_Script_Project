import unittest

from backend.services.text_chunker import (
    chunk_text,
    chunk_text_by_chapters,
)


class TestTextChunker(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        chunks = chunk_text(
            text="",
            source_file_name="empty.txt",
            source_file_order=1,
        )
        self.assertEqual(chunks, [])

    def test_short_text_creates_one_traceable_chunk(self):
        text = "第一段。"
        chunks = chunk_text(
            text=text,
            source_file_name="01_第一章.txt",
            source_file_order=1,
            global_order_start=7,
            min_chars=5,
            target_chars=10,
            max_chars=15,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_id, "chunk_0007")
        self.assertEqual(chunk.global_order, 7)
        self.assertEqual(chunk.start_character, 0)
        self.assertEqual(chunk.end_character, len(text))
        self.assertEqual(chunk.text, text)

    def test_greedy_chunking_preserves_paragraph_separators(self):
        text = "AAAAA\n\nBBBBB\n\nCCCCC"
        chunks = chunk_text(
            text=text,
            source_file_name="chapter.txt",
            source_file_order=1,
            min_chars=6,
            target_chars=12,
            max_chars=16,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "AAAAA\n\n")
        self.assertEqual(chunks[1].text, "BBBBB\n\nCCCCC")
        self.assertEqual(
            "".join(chunk.text for chunk in chunks),
            text,
        )

    def test_oversized_paragraph_splits_on_chinese_sentences(self):
        text = "AAAAA。BBBBB。CCCCC。"
        chunks = chunk_text(
            text=text,
            source_file_name="chapter.txt",
            source_file_order=1,
            min_chars=5,
            target_chars=10,
            max_chars=12,
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["AAAAA。", "BBBBB。", "CCCCC。"],
        )

    def test_english_sentence_boundaries_avoid_common_abbreviations(self):
        text = (
            "Mr. Myriel entered the room. "
            "He greeted Dr. Smith. "
            "The meeting ended."
        )
        chunks = chunk_text(
            text=text,
            source_file_name="english.txt",
            source_file_order=1,
            min_chars=10,
            target_chars=32,
            max_chars=38,
        )

        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertTrue(all(chunk.character_count <= 38 for chunk in chunks))
        self.assertIn("Mr. Myriel", chunks[0].text)

    def test_oversized_sentence_falls_back_to_hard_split(self):
        text = "A" * 25
        chunks = chunk_text(
            text=text,
            source_file_name="chapter.txt",
            source_file_order=1,
            min_chars=5,
            target_chars=10,
            max_chars=10,
        )

        self.assertEqual(
            [chunk.character_count for chunk in chunks],
            [10, 10, 5],
        )
        self.assertEqual(
            [(chunk.start_character, chunk.end_character) for chunk in chunks],
            [(0, 10), (10, 20), (20, 25)],
        )

    def test_short_final_chunk_merges_with_previous_chunk(self):
        text = "AAAAA\n\nBBBBB\n\nCCC"
        chunks = chunk_text(
            text=text,
            source_file_name="chapter.txt",
            source_file_order=1,
            min_chars=6,
            target_chars=14,
            max_chars=20,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_chunks_cover_complete_text_without_gaps(self):
        text = (
            "第一段内容。\n\n"
            "第二段内容较长一些。\n\n"
            "第三段内容。\n\n"
            "第四段内容。"
        )
        chunks = chunk_text(
            text=text,
            source_file_name="chapter.txt",
            source_file_order=1,
            min_chars=8,
            target_chars=18,
            max_chars=26,
        )

        self.assertEqual(chunks[0].start_character, 0)
        self.assertEqual(chunks[-1].end_character, len(text))
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(
                previous.end_character,
                following.start_character,
            )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_invalid_size_configuration_raises_error(self):
        invalid_configurations = [
            (0, 10, 20),
            (10, 5, 20),
            (10, 20, 15),
        ]

        for min_chars, target_chars, max_chars in invalid_configurations:
            with self.subTest(
                min_chars=min_chars,
                target_chars=target_chars,
                max_chars=max_chars,
            ):
                with self.assertRaises(ValueError):
                    chunk_text(
                        text="测试文本。",
                        source_file_name="chapter.txt",
                        source_file_order=1,
                        min_chars=min_chars,
                        target_chars=target_chars,
                        max_chars=max_chars,
                    )

    def test_invalid_order_raises_error(self):
        with self.assertRaises(ValueError):
            chunk_text(
                text="测试文本。",
                source_file_name="chapter.txt",
                source_file_order=0,
                min_chars=5,
                target_chars=10,
                max_chars=20,
            )

        with self.assertRaises(ValueError):
            chunk_text(
                text="测试文本。",
                source_file_name="chapter.txt",
                source_file_order=1,
                global_order_start=0,
                min_chars=5,
                target_chars=10,
                max_chars=20,
            )

    def test_chapter_aware_chunking_does_not_cross_chapters(self):
        first_body = "第一章正文内容。" * 10
        second_body = "第二章正文内容。" * 10
        text = (
            "第一章 开始\n\n"
            f"{first_body}\n\n"
            "第二章 继续\n\n"
            f"{second_body}"
        )

        result = chunk_text_by_chapters(
            text=text,
            source_file_name="novel.txt",
            source_file_order=1,
            min_chars=30,
            target_chars=60,
            max_chars=90,
        )

        self.assertEqual(len(result.chapters), 2)
        self.assertGreaterEqual(len(result.chunks), 2)

        first_chapter_end = result.chapters[0].end_character
        for chunk in result.chunks:
            if chunk.chapter_order == 1:
                self.assertLessEqual(chunk.end_character, first_chapter_end)
            if chunk.chapter_order == 2:
                self.assertGreaterEqual(chunk.start_character, first_chapter_end)

    def test_chapter_metadata_and_orders_are_correct(self):
        body = "正文内容。" * 30
        text = "第一章 开始\n\n" + body

        result = chunk_text_by_chapters(
            text=text,
            source_file_name="novel.txt",
            source_file_order=1,
            global_order_start=5,
            min_chars=30,
            target_chars=60,
            max_chars=80,
        )

        self.assertGreater(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].chunk_id, "chunk_0005")
        self.assertTrue(result.chunks[0].is_chapter_start)
        self.assertTrue(result.chunks[-1].is_chapter_end)
        self.assertEqual(
            [chunk.chunk_order_in_chapter for chunk in result.chunks],
            list(range(1, len(result.chunks) + 1)),
        )
        self.assertTrue(
            all(chunk.chapter_title == "开始" for chunk in result.chunks)
        )

    def test_chapter_aware_chunks_reconstruct_complete_text(self):
        text = (
            "序章\n\n"
            "序章正文。\n\n"
            "Chapter 1 Beginning\n\n"
            "English body text. More body text.\n\n"
            "第二章 结束\n\n"
            "中文正文。"
        )

        result = chunk_text_by_chapters(
            text=text,
            source_file_name="mixed.txt",
            source_file_order=1,
            min_chars=10,
            target_chars=24,
            max_chars=36,
        )

        self.assertEqual(
            "".join(chunk.text for chunk in result.chunks),
            text,
        )
        self.assertEqual(result.chunks[0].start_character, 0)
        self.assertEqual(result.chunks[-1].end_character, len(text))
        for previous, following in zip(result.chunks, result.chunks[1:]):
            self.assertEqual(
                previous.end_character,
                following.start_character,
            )


if __name__ == "__main__":
    unittest.main()
