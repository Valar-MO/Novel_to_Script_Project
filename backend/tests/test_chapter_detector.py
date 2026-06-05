import unittest

from backend.services.chapter_detector import detect_chapters


class TestChapterDetector(unittest.TestCase):
    def test_detects_explicit_chinese_chapters(self):
        text = (
            "第一章 山边小村\n\n"
            "第一章正文内容。\n\n"
            "第二章 青牛镇\n\n"
            "第二章正文内容。"
        )

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].chapter_number, 1)
        self.assertEqual(chapters[0].chapter_title, "山边小村")
        self.assertEqual(chapters[1].chapter_number, 2)
        self.assertEqual(chapters[1].chapter_title, "青牛镇")
        self.assertTrue(all(chapter.is_detected for chapter in chapters))

    def test_does_not_detect_chapter_words_inside_body_sentence(self):
        text = "他翻开第二章，继续读了下去。\n\n故事仍在继续。"

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 1)
        self.assertFalse(chapters[0].is_detected)
        self.assertEqual(chapters[0].start_character, 0)
        self.assertEqual(chapters[0].end_character, len(text))

    def test_detects_full_chinese_hierarchical_headings(self):
        text = (
            "第一部 芳汀 第一卷 一个正直的人 一 米里哀先生\n\n"
            "第一章正文。\n\n"
            "第一部 芳汀 第一卷 一个正直的人 二 米里哀先生改称卞福汝主教\n\n"
            "第二章正文。"
        )

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].part_order, 1)
        self.assertEqual(chapters[0].part_title, "芳汀")
        self.assertEqual(chapters[0].volume_order, 1)
        self.assertEqual(chapters[0].volume_title, "一个正直的人")
        self.assertEqual(chapters[0].chapter_number, 1)
        self.assertEqual(chapters[0].chapter_title, "米里哀先生")
        self.assertEqual(chapters[1].chapter_number, 2)
        self.assertEqual(
            chapters[1].chapter_title,
            "米里哀先生改称卞福汝主教",
        )

    def test_detects_bare_chapters_inside_part_and_volume(self):
        body_one = "这是第一节正文内容。" * 20
        body_two = "这是第二节正文内容。" * 20
        text = (
            "第一部 芳汀\n\n"
            "第一卷 一个正直的人\n\n"
            "一 米里哀先生\n\n"
            f"{body_one}\n\n"
            "二 米里哀先生改称卞福汝主教\n\n"
            f"{body_two}"
        )

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].start_character, 0)
        self.assertEqual(chapters[0].chapter_number, 1)
        self.assertEqual(chapters[0].part_title, "芳汀")
        self.assertEqual(chapters[0].volume_title, "一个正直的人")
        self.assertEqual(chapters[1].chapter_number, 2)

    def test_numbered_inventory_is_not_detected_as_chapters(self):
        text = "一 米面十斤\n二 棉衣两件\n三 白糖五斤"

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 1)
        self.assertFalse(chapters[0].is_detected)
        self.assertEqual(chapters[0].start_character, 0)
        self.assertEqual(chapters[0].end_character, len(text))

    def test_detects_english_chapters(self):
        text = (
            "Chapter 1 The Visitor\n\n"
            "The first chapter body.\n\n"
            "CHAPTER II: The Return\n\n"
            "The second chapter body."
        )

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].chapter_number, 1)
        self.assertEqual(chapters[0].chapter_title, "The Visitor")
        self.assertEqual(chapters[1].chapter_number, 2)
        self.assertEqual(chapters[1].chapter_title, "The Return")

    def test_detects_english_book_with_roman_bare_chapters(self):
        body_one = "This is narrative body text. " * 10
        body_two = "This is more narrative body text. " * 10
        text = (
            "BOOK ONE: A JUST MAN\n\n"
            "I. Monsieur Myriel\n\n"
            f"{body_one}\n\n"
            "II. Monsieur Myriel Becomes Monseigneur Bienvenu\n\n"
            f"{body_two}"
        )

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].volume_order, 1)
        self.assertEqual(chapters[0].volume_title, "A JUST MAN")
        self.assertEqual(chapters[0].chapter_number, 1)
        self.assertEqual(chapters[1].chapter_number, 2)

    def test_preserves_preface_before_first_detected_chapter(self):
        text = (
            "作品简介。\n\n"
            "作者说明。\n\n"
            "第一章 开始\n\n"
            "正文。"
        )

        chapters = detect_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertFalse(chapters[0].is_detected)
        self.assertTrue(chapters[1].is_detected)
        self.assertEqual(chapters[0].start_character, 0)
        self.assertEqual(
            chapters[0].end_character,
            chapters[1].start_character,
        )

    def test_chapter_spans_cover_complete_text_without_gaps(self):
        text = (
            "Prologue\n\n"
            "Opening text.\n\n"
            "Chapter 1 Beginning\n\n"
            "First body.\n\n"
            "Chapter 2 Ending\n\n"
            "Second body."
        )

        chapters = detect_chapters(text)

        self.assertEqual(chapters[0].start_character, 0)
        self.assertEqual(chapters[-1].end_character, len(text))

        for previous, following in zip(chapters, chapters[1:]):
            self.assertEqual(
                previous.end_character,
                following.start_character,
            )

        reconstructed = "".join(
            text[chapter.start_character:chapter.end_character]
            for chapter in chapters
        )
        self.assertEqual(reconstructed, text)


if __name__ == "__main__":
    unittest.main()
