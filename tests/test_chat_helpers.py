from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from original_llm.cli import resolve_interactive_mode
from original_llm.config import CHAT_TURN_END_MARKER
from original_llm.data import CharTokenizer
from original_llm.generate import (
    append_chat_history,
    chat_stop_sequences,
    curated_short_reply,
    extract_chat_reply,
    extract_pending_chat_user_input,
    fallback_friendly_reply,
    generated_suffix,
    has_dangling_particle_tail,
    is_echo_reply,
    is_low_quality_reply,
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

    def test_extract_chat_reply_strips_turn_end_marker(self) -> None:
        text = f"相手: いいよ。\n{CHAT_TURN_END_MARKER}\n私: じゃあまたあとで"
        self.assertEqual(extract_chat_reply(text, "私", "相手"), "いいよ。")

    def test_extract_chat_reply_strips_decoded_turn_end_marker(self) -> None:
        tokenizer = CharTokenizer.build(["相手: いいよ。\neot\n私: じゃあまたあとで"])
        decoded_marker = tokenizer.decode(tokenizer.encode(CHAT_TURN_END_MARKER))
        text = f"相手: いいよ。\n{decoded_marker}\n私: じゃあまたあとで"
        self.assertEqual(
            extract_chat_reply(text, "私", "相手", tokenizer=tokenizer),
            "いいよ。",
        )

    def test_chat_stop_sequences_include_decoded_turn_end_marker(self) -> None:
        tokenizer = CharTokenizer.build(["相手: いいよ。\neot"])
        decoded_marker = tokenizer.decode(tokenizer.encode(CHAT_TURN_END_MARKER))
        sequences = chat_stop_sequences("私", "相手", tokenizer=tokenizer)
        self.assertIn(decoded_marker, sequences)
        self.assertIn(f"\n{decoded_marker}", sequences)

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
        self.assertEqual(
            history,
            f"私: こんにちは\n相手: いいよ。\n{CHAT_TURN_END_MARKER}",
        )

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
        self.assertEqual(
            history,
            f"私: ねむい\n相手: もう寝たほうがいい。\n{CHAT_TURN_END_MARKER}",
        )


class ChatPromptSupportTests(unittest.TestCase):
    def test_generated_suffix_accepts_tokenizer_roundtrip_prompt(self) -> None:
        tokenizer = CharTokenizer.build(["私: こんにちは\neot\n相手: いいよ。"])
        prompt = f"私: こんにちは\n{CHAT_TURN_END_MARKER}\n相手: "
        generated = (
            tokenizer.decode(tokenizer.encode(prompt))
            + "いいよ。"
        )
        self.assertEqual(
            generated_suffix(prompt, generated, tokenizer=tokenizer),
            "いいよ。",
        )

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
                (
                    "私: いま何してる\n"
                    "相手: べつに大した事はしていない。ただ、ぼんやりしていた。\n"
                    f"{CHAT_TURN_END_MARKER}"
                )
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


class NaturalnessHelperTests(unittest.TestCase):
    def test_is_echo_reply_catches_normalized_echo(self) -> None:
        self.assertTrue(is_echo_reply("こんにちは", "こんにちは"))
        self.assertTrue(is_echo_reply("こんにちは", "こんにちは。"))
        self.assertTrue(is_echo_reply("こんにちは", "こーんにーちはー。"))
        self.assertFalse(is_echo_reply("こんにちは", "こんにちは。今日はどうしてた？"))

    def test_is_low_quality_reply_flags_known_artifacts(self) -> None:
        self.assertTrue(is_low_quality_reply("お腹すいた", "食べたいないと、言葉までやせる。"))
        self.assertTrue(is_low_quality_reply("つらい", "もどすのが。"))
        self.assertTrue(is_low_quality_reply("test", "実在する場所ではないので、お金も？"))
        self.assertTrue(is_low_quality_reply("test", "うん"))

    def test_is_low_quality_reply_keeps_natural_replies(self) -> None:
        self.assertFalse(is_low_quality_reply("元気？", "まあまあ元気。そっちは？"))
        self.assertFalse(is_low_quality_reply("test", "本当にそうかな？"))
        self.assertFalse(is_low_quality_reply("test", "なにか軽く食べちゃおう。"))

    def test_has_dangling_particle_tail_only_flags_comma_fragments(self) -> None:
        self.assertTrue(has_dangling_particle_tail("実在する場所ではないので、お金も？"))
        self.assertFalse(has_dangling_particle_tail("まあまあ元気。そっちは？"))

    def test_curated_short_reply_returns_natural_greeting(self) -> None:
        reply = curated_short_reply("こんにちは")
        self.assertIsNotNone(reply)
        self.assertIn("こんにちは", reply)

    def test_curated_short_reply_avoids_recent_replies(self) -> None:
        first = curated_short_reply("ありがとう", rotation_index=0)
        second = curated_short_reply(
            "ありがとう",
            avoid_replies=(first,) if first is not None else (),
            rotation_index=1,
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_curated_short_reply_returns_none_for_unknown_input(self) -> None:
        self.assertIsNone(curated_short_reply("ピクニック行きたいな"))

    def test_fallback_friendly_reply_picks_safe_response(self) -> None:
        first = fallback_friendly_reply(rotation_index=0)
        self.assertTrue(first)
        second = fallback_friendly_reply(avoid_replies=(first,), rotation_index=1)
        self.assertNotEqual(first, second)


class DirectChatReplyQualityTests(unittest.TestCase):
    def test_skips_echo_seed_in_favor_of_richer_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir)
            (corpus_dir / "seed.txt").write_text(
                "\n\n".join(
                    [
                        "私: こんにちは\n相手: こんにちは",
                        "私: こんにちは\n相手: こんにちは。今日はどうしてた？",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                user_label="私",
                reply_label="相手",
                retrieval_corpus_dir=str(corpus_dir),
            )
            reply = select_direct_chat_reply("こんにちは", args)
        self.assertEqual(reply, "こんにちは。今日はどうしてた？")

    def test_avoid_replies_skips_recent_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir)
            (corpus_dir / "seed.txt").write_text(
                "\n\n".join(
                    [
                        "私: ありがとう\n相手: どういたしまして。",
                        "私: ありがとう\n相手: 気にしないで。こちらこそありがとう。",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                user_label="私",
                reply_label="相手",
                retrieval_corpus_dir=str(corpus_dir),
            )
            first = select_direct_chat_reply("ありがとう", args)
            second = select_direct_chat_reply(
                "ありがとう",
                args,
                avoid_replies=(first,) if first is not None else (),
            )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
