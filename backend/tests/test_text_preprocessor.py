import unittest

from backend.services.text_preprocessor import preprocess_text


class TestTextPreprocessor(unittest.TestCase):

    def test_normalize_bom_and_windows_line_breaks(self):
        source_text = "\ufeff第一章\r\n第一段\r\n第二段\r\n"

        result = preprocess_text(source_text)

        self.assertEqual(
            result.text,
            "第一章\n第一段\n第二段",
        )
        self.assertNotIn("\r", result.text)
        self.assertFalse(result.text.startswith("\ufeff"))

    def test_replace_tabs_and_remove_trailing_spaces(self):
        source_text = "\t第一段   \n\t第二段\t\n"

        result = preprocess_text(source_text)

        self.assertEqual(
            result.text,
            "    第一段\n    第二段",
        )

    def test_collapse_excessive_blank_lines(self):
        source_text = "第一段\n\n\n\n第二段"

        result = preprocess_text(source_text)

        self.assertEqual(
            result.text,
            "第一段\n\n第二段",
        )

    def test_preserve_internal_spaces(self):
        source_text = "他说：这是  一个测试。"

        result = preprocess_text(source_text)

        self.assertEqual(
            result.text,
            "他说：这是  一个测试。",
        )

    def test_empty_text(self):
        result = preprocess_text("   \n\n")

        self.assertEqual(result.text, "")
        self.assertEqual(result.processed_character_count, 0)
        self.assertEqual(result.processed_line_count, 0)


if __name__ == "__main__":
    unittest.main()