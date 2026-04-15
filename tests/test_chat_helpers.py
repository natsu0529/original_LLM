from __future__ import annotations

import argparse
import unittest

from original_llm.cli import resolve_interactive_mode
from original_llm.data import CharTokenizer
from original_llm.generate import append_chat_history, extract_chat_reply


class ResolveInteractiveModeTests(unittest.TestCase):
    def test_prompt_defaults_to_one_shot(self) -> None:
        args = argparse.Namespace(prompt="私: こんにちは\n相手: ", interactive=True)
        self.assertFalse(resolve_interactive_mode(args, []))

    def test_explicit_interactive_flag_wins(self) -> None:
        args = argparse.Namespace(prompt="私: こんにちは\n相手: ", interactive=True)
        self.assertTrue(resolve_interactive_mode(args, ["--interactive"]))

    def test_explicit_no_interactive_flag_wins(self) -> None:
        args = argparse.Namespace(prompt=None, interactive=True)
        self.assertFalse(resolve_interactive_mode(args, ["--no-interactive"]))


class ChatReplyTests(unittest.TestCase):
    def test_extract_chat_reply_strips_reply_label_and_next_turn(self) -> None:
        text = "相手: いいよ。\n私: じゃあまたあとで"
        self.assertEqual(extract_chat_reply(text, "私", "相手"), "いいよ。")

    def test_append_chat_history_normalizes_turn(self) -> None:
        tokenizer = CharTokenizer.build(["私: こんにちは\n相手: いいよ。"])
        history = append_chat_history(
            history="",
            user_input="こんにちは",
            reply_text="いいよ。",
            user_label="私",
            reply_label="相手",
            tokenizer=tokenizer,
            context_length=256,
        )
        self.assertEqual(history, "私: こんにちは\n相手: いいよ。")


if __name__ == "__main__":
    unittest.main()
