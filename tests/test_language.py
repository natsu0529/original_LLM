from __future__ import annotations

import unittest

from original_llm.language import (
    has_japanese_char,
    is_non_japanese_input,
    looks_like_single_word,
    looks_like_typo_correction,
)


class JapaneseDetectionTests(unittest.TestCase):
    def test_recognises_hiragana_katakana_kanji(self) -> None:
        self.assertTrue(has_japanese_char("こんにちは"))
        self.assertTrue(has_japanese_char("カタカナ"))
        self.assertTrue(has_japanese_char("漢字"))
        self.assertTrue(has_japanese_char("Hello こ"))

    def test_pure_latin_is_non_japanese(self) -> None:
        self.assertTrue(is_non_japanese_input("Hello"))
        self.assertTrue(is_non_japanese_input("good morning"))
        self.assertTrue(is_non_japanese_input("привет"))

    def test_japanese_or_punctuation_is_not_flagged(self) -> None:
        self.assertFalse(is_non_japanese_input("こんにちは"))
        self.assertFalse(is_non_japanese_input("Hi こん"))
        self.assertFalse(is_non_japanese_input("?!?"))
        self.assertFalse(is_non_japanese_input(""))

    def test_pure_digits_not_flagged_as_language(self) -> None:
        self.assertFalse(is_non_japanese_input("12345"))


class SingleWordDetectionTests(unittest.TestCase):
    def test_short_one_word_is_single_word(self) -> None:
        self.assertTrue(looks_like_single_word("ぴえん"))
        self.assertTrue(looks_like_single_word("ヤバみ"))

    def test_punctuated_or_long_is_not(self) -> None:
        self.assertFalse(looks_like_single_word("こんにちは。"))
        self.assertFalse(looks_like_single_word("今日は天気がいいね"))
        self.assertFalse(looks_like_single_word("hello world"))
        self.assertFalse(looks_like_single_word(""))

    def test_max_chars_is_configurable(self) -> None:
        self.assertFalse(looks_like_single_word("あいうえおかきくけこさし", max_chars=10))
        self.assertTrue(looks_like_single_word("あいうえおかきくけ", max_chars=10))


class TypoRetractionTests(unittest.TestCase):
    def test_recognises_common_phrases(self) -> None:
        self.assertTrue(looks_like_typo_correction("間違えた、ごめん"))
        self.assertTrue(looks_like_typo_correction("打ち間違いだった"))
        self.assertTrue(looks_like_typo_correction("ごめん、間違えてた"))
        self.assertTrue(looks_like_typo_correction("typoだ"))
        self.assertTrue(looks_like_typo_correction("タイポ"))

    def test_normal_input_is_not_a_retraction(self) -> None:
        self.assertFalse(looks_like_typo_correction("猫が好き"))
        self.assertFalse(looks_like_typo_correction("普通のメッセージ"))
        self.assertFalse(looks_like_typo_correction(""))


class UpdateQuestionTests(unittest.TestCase):
    def test_recognises_japanese_phrasings(self) -> None:
        from original_llm.generate import looks_like_update_question
        for phrase in (
            "アップデートのやり方は？",
            "更新ってどうやるの",
            "最新版にしたい",
            "バージョンアップしたい",
            "新しいバージョンが出たら？",
            "アップグレードのコマンドおしえて",
        ):
            self.assertTrue(looks_like_update_question(phrase), phrase)

    def test_recognises_command_phrasings(self) -> None:
        from original_llm.generate import looks_like_update_question
        for phrase in (
            "uv tool upgrade って何？",
            "pip install -U の正しい打ち方",
        ):
            self.assertTrue(looks_like_update_question(phrase), phrase)

    def test_irrelevant_input_is_false(self) -> None:
        from original_llm.generate import looks_like_update_question
        for phrase in (
            "今日は天気がいい",
            "ぴえん",
            "猫を飼いたい",
            "",
        ):
            self.assertFalse(looks_like_update_question(phrase), phrase)


if __name__ == "__main__":
    unittest.main()
