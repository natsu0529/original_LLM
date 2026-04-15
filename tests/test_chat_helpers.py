from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from original_llm.cli import resolve_interactive_mode
from original_llm.data import CharTokenizer
from original_llm.generate import (
    append_chat_history,
    extract_chat_reply,
    extract_pending_chat_user_input,
    prepare_chat_user_input,
    select_direct_chat_reply,
    select_chat_retrieval_examples,
)


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

    def test_append_chat_history_respects_max_turns(self) -> None:
        tokenizer = CharTokenizer.build(
            [
                "私: こんにちは\n相手: いいよ。\n私: ねむい\n相手: もう寝たほうがいい。",
            ]
        )
        history = append_chat_history(
            history="",
            user_input="こんにちは",
            reply_text="いいよ。",
            user_label="私",
            reply_label="相手",
            tokenizer=tokenizer,
            context_length=256,
            max_turns=1,
        )
        history = append_chat_history(
            history=history,
            user_input="ねむい",
            reply_text="もう寝たほうがいい。",
            user_label="私",
            reply_label="相手",
            tokenizer=tokenizer,
            context_length=256,
            max_turns=1,
        )
        self.assertEqual(history, "私: ねむい\n相手: もう寝たほうがいい。")


class ChatPromptSupportTests(unittest.TestCase):
    def test_prepare_chat_user_input_normalizes_short_casual_text(self) -> None:
        args = argparse.Namespace(
            user_label="私",
            reply_label="相手",
            normalize_chat_input=True,
        )
        self.assertEqual(prepare_chat_user_input("どゆこと？", args), "どういうこと")

    def test_extract_pending_chat_user_input_reads_last_open_turn(self) -> None:
        prompt = "私: よお\n相手: おはよう。\n私: 何してる？\n相手: "
        self.assertEqual(
            extract_pending_chat_user_input(prompt, "私", "相手"),
            "何してる？",
        )

    def test_select_chat_retrieval_examples_prefers_close_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir)
            (corpus_dir / "seed.txt").write_text(
                "\n\n".join(
                    [
                        "私: いま何してる\n相手: べつに大した事はしていない。ただ、ぼんやりしていた。",
                        "私: 好き\n相手: 好きと言われると、胸のあたりが少し静かでなくなる。",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            examples = select_chat_retrieval_examples(
                user_input="何してる？",
                corpus_dir=corpus_dir,
                user_label="私",
                reply_label="相手",
                limit=1,
            )

        self.assertEqual(
            examples,
            [
                "私: いま何してる\n相手: べつに大した事はしていない。ただ、ぼんやりしていた。"
            ],
        )

    def test_select_chat_retrieval_examples_rejects_weak_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir)
            (corpus_dir / "seed.txt").write_text(
                "私: どう思う\n相手: そうだね。悪くはないが、少し頼りない。\n",
                encoding="utf-8",
            )
            examples = select_chat_retrieval_examples(
                user_input="どゆこと？",
                corpus_dir=corpus_dir,
                user_label="私",
                reply_label="相手",
                limit=1,
            )

        self.assertEqual(examples, [])

    def test_select_direct_chat_reply_uses_exact_short_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir)
            (corpus_dir / "seed.txt").write_text(
                "私: パンはパンでも食べられないパンは？\n相手: フライパンだろう。そういう顔をしている。\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                user_label="私",
                reply_label="相手",
                retrieval_corpus_dir=str(corpus_dir),
            )
            reply = select_direct_chat_reply("パンはパンでも食べられないパンは？", args)

        self.assertEqual(reply, "フライパンだろう。そういう顔をしている。")


if __name__ == "__main__":
    unittest.main()
