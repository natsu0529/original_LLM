"""Probe a chat checkpoint with the user's complaint prompts.

Bypasses curated/retrieval/filters: just runs the raw model generate.
Useful to compare two checkpoints' base reply quality side by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from original_llm.generate import generate_text, load_generator


PROBE_PROMPTS = [
    "こんにちは",
    "家にいるよ、あなたは？",
    "何してる？",
    "そっか、元気？",
    "大丈夫だよ、少し遊ぼうよ",
    "何を食べたの？",
    "えっと、どゆこと？",
    "どこに引っ越したの？",
    "え？",
    "今日は何してた？",
    "私はもう起きたよ",
    "映画でも見ようかな",
]


def chat_prompt(user_text: str) -> str:
    return f"私: {user_text}\n相手: "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model, tokenizer, _ = load_generator(args.checkpoint, device)

    print(f"=== {args.checkpoint} ===")
    for p in PROBE_PROMPTS:
        full = chat_prompt(p)
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=full,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=1.1,
            repetition_window=64,
            stop_at_period=True,
            stop_at_blank_line=True,
            min_new_chars_before_stop=2,
            device=device,
            stop_sequences=("<eot>", "\n私:"),
        )
        reply = text[len(full):].strip().splitlines()[0] if text.startswith(full) else text.strip().splitlines()[0]
        print(f"私: {p}")
        print(f"相手: {reply}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
