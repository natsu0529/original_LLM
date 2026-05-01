from __future__ import annotations

import unittest

from original_llm.data import (
    ByteTokenizer,
    CharTokenizer,
    SentencePieceTokenizer,
    build_reply_loss_mask,
    tokenizer_from_state_dict,
)


CORPUS_TEXTS = [
    "私: おはよ\n相手: いいね。\n<eot>",
    "私: 何してる？\n相手: ぼんやりしてる。\n<eot>",
    "私: 眠い\n相手: じゃあ少し話してから寝よう。\n<eot>",
    "私: まだ眠い\n相手: じゃあ少し話そう。\n<eot>",
] * 6


class SentencePieceTokenizerTests(unittest.TestCase):
    def test_sentencepiece_roundtrip_preserves_chat_text(self) -> None:
        tokenizer = SentencePieceTokenizer.build(
            CORPUS_TEXTS,
            vocab_size=128,
            model_type="unigram",
            character_coverage=1.0,
        )

        text = "私: おはよ\n相手: じゃあ少し話そう。\n<eot>"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_sentencepiece_state_dict_roundtrip(self) -> None:
        tokenizer = SentencePieceTokenizer.build(
            CORPUS_TEXTS,
            vocab_size=128,
            model_type="unigram",
            character_coverage=1.0,
        )

        restored = tokenizer_from_state_dict(tokenizer.state_dict())
        text = "私: 何してる？\n相手: ぼんやりしてる。\n<eot>"

        self.assertEqual(restored.tokenizer_type, "sentencepiece")
        self.assertEqual(restored.decode(restored.encode(text)), text)


class ReplyLossMaskTests(unittest.TestCase):
    def test_reply_loss_mask_marks_reply_text_only_across_tokenizers(self) -> None:
        text = "私: おはよ\n相手: いいね。\nまだ眠い。\n私: 何してる？\n"
        expected_reply_text = "いいね。\nまだ眠い。\n"
        tokenizers = [
            ByteTokenizer.build(CORPUS_TEXTS),
            CharTokenizer.build(CORPUS_TEXTS),
            SentencePieceTokenizer.build(
                CORPUS_TEXTS,
                vocab_size=128,
                model_type="unigram",
                character_coverage=1.0,
            ),
        ]

        for tokenizer in tokenizers:
            with self.subTest(tokenizer=tokenizer.tokenizer_type):
                token_ids = tokenizer.encode(text)
                mask = build_reply_loss_mask(text, tokenizer, "相手")
                reply_token_ids = [
                    token_id
                    for token_id, active in zip(token_ids, mask, strict=True)
                    if active
                ]

                self.assertEqual(len(mask), len(token_ids))
                self.assertEqual(
                    tokenizer.decode(reply_token_ids),
                    expected_reply_text,
                )


if __name__ == "__main__":
    unittest.main()
