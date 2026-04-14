#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import CharTokenizer, load_manifest, load_work_texts  # noqa: E402
from train import load_checkpoint_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how well a checkpoint tokenizer covers a text corpus."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint whose tokenizer_state will be inspected.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing UTF-8 text files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest.jsonl for author/title metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only inspect the first N text files.",
    )
    parser.add_argument(
        "--top-unknown",
        type=int,
        default=32,
        help="How many unknown characters to print.",
    )
    return parser.parse_args()


def load_char_tokenizer(checkpoint_path: Path) -> CharTokenizer:
    checkpoint = load_checkpoint_metadata(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer_state = checkpoint.get("tokenizer_state")
    if not isinstance(tokenizer_state, dict):
        raise ValueError("Checkpoint does not contain tokenizer_state")
    if tokenizer_state.get("tokenizer_type") != "char":
        raise ValueError("This script currently supports only char tokenizers")
    return CharTokenizer.from_state_dict(tokenizer_state)


def safe_char_display(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == "\t":
        return "\\t"
    if char == " ":
        return "<space>"
    return char


def main() -> int:
    args = parse_args()
    tokenizer = load_char_tokenizer(args.checkpoint)
    manifest_by_id = load_manifest(args.manifest_path) if args.manifest_path else {}
    works = load_work_texts(
        data_dir=args.data_dir,
        manifest_by_id=manifest_by_id,
        limit=args.limit,
    )

    known_chars = set(tokenizer.id_to_token)
    known_chars.discard(tokenizer.unk_token)

    total_chars = 0
    unknown_chars = 0
    unknown_counter: Counter[str] = Counter()
    author_unknown_counter: Counter[str] = Counter()
    author_total_counter: Counter[str] = Counter()

    for work in works:
        author_name = manifest_by_id.get(work.work_id, {}).get("author_name", "unknown")
        for char in work.cleaned_text:
            total_chars += 1
            author_total_counter[author_name] += 1
            if char in known_chars:
                continue
            unknown_chars += 1
            unknown_counter[char] += 1
            author_unknown_counter[author_name] += 1

    unique_chars = {char for work in works for char in work.cleaned_text}
    unknown_unique_chars = sorted(char for char in unique_chars if char not in known_chars)
    known_unique_chars = len(unique_chars) - len(unknown_unique_chars)
    unknown_rate = (unknown_chars / total_chars * 100.0) if total_chars else 0.0

    print(f"checkpoint={args.checkpoint}")
    print(f"data_dir={args.data_dir}")
    if args.manifest_path is not None:
        print(f"manifest_path={args.manifest_path}")
    print(f"work_count={len(works)}")
    print(f"tokenizer_vocab_size={tokenizer.vocab_size}")
    print(f"total_chars={total_chars}")
    print(f"unique_chars={len(unique_chars)}")
    print(f"known_unique_chars={known_unique_chars}")
    print(f"unknown_unique_chars={len(unknown_unique_chars)}")
    print(f"unknown_char_occurrences={unknown_chars}")
    print(f"unknown_char_rate={unknown_rate:.6f}%")

    print("top_unknown_chars=")
    for char, count in unknown_counter.most_common(args.top_unknown):
        rate = count / total_chars * 100.0 if total_chars else 0.0
        print(f"  {safe_char_display(char)}\tcount={count}\trate={rate:.6f}%")

    print("per_author_unknown_rate=")
    for author_name, total in author_total_counter.most_common():
        unknown = author_unknown_counter[author_name]
        rate = unknown / total * 100.0 if total else 0.0
        print(f"  {author_name}\tunknown={unknown}\ttotal={total}\trate={rate:.6f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
