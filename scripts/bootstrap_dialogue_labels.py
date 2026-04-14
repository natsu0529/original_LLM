#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "dialogue_batches"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "labeled_batches_bootstrap"
)

ALCOHOL_RE = re.compile(r"(酒|ビール|麦酒|ウイス|焼酎|日本酒|熱燗|冷酒|ワイン|酔|ハイボール|アブサン|サワー|ラム|ジン|テキーラ)")
LITERATURE_RE = re.compile(r"(小説|詩|文学|本|作家|太宰|漱石|芥川|読|書|文章|作品|言葉)")
SENSUAL_RE = re.compile(r"(肌|胸|脚|唇|くちびる|抱|接吻|キス|色気|艶|乳房|乳首|いやらし|欲情)")
AFFECTION_RE = re.compile(r"(好き|愛|恋|会いた|そば|抱きしめ|かわいい|可愛い)")
MOOD_RE = re.compile(r"(悲|哀|寂|さびし|かなし|苦し|つら|泣|不安|憂鬱|いや)")
HUMOR_RE = re.compile(r"(笑|冗談|おかしい|滑稽)")
GREETING_RE = re.compile(r"(おはよう|こんにちは|こんばんは|さよなら|どうも)")
QUESTION_RE = re.compile(r"(？|\?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate first-pass labels for dialogue candidate batches."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--batches",
        nargs="*",
        default=[],
        help="Optional batch numbers like 0001 0002. If omitted, process all.",
    )
    return parser.parse_args()


def normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = " ".join(text.replace("\n", " ").split())
    return normalized or None


def detect_tags(text: str) -> list[str]:
    tags: list[str] = []
    if ALCOHOL_RE.search(text):
        tags.append("alcohol")
    if LITERATURE_RE.search(text):
        tags.append("literature")
    if SENSUAL_RE.search(text):
        tags.append("sensual")
    if AFFECTION_RE.search(text):
        tags.append("affection")
    if MOOD_RE.search(text):
        tags.append("mood")
    if HUMOR_RE.search(text):
        tags.append("humor")
    if GREETING_RE.search(text):
        tags.append("greeting")
    if QUESTION_RE.search(text):
        tags.append("question")

    if not tags:
        tags.append("daily")
    elif "daily" not in tags and any(tag in tags for tag in ("greeting", "question")):
        tags.append("daily")
    return tags


def bootstrap_row(row: dict) -> dict:
    quote = normalize_text(row.get("quote_text"))
    prev_quote = normalize_text(row.get("prev_quote_text"))
    next_quote = normalize_text(row.get("next_quote_text"))
    cluster_size = int(row.get("cluster_size", 0))
    heuristic_score = int(row.get("heuristic_score", 0))
    flags = set(row.get("heuristic_flags") or [])

    label = "noise"
    pair_with = None
    user_text = None
    reply_text = None
    formatted_text = None
    notes = ""

    quote_len = len(quote or "")
    if not quote or quote_len <= 2:
        label = "noise"
        notes = "too short"
    elif quote_len > 320 or ("multiline_quote" in flags and quote_len > 200):
        label = "monologue"
        notes = "too long for chat"
    elif cluster_size >= 2 and prev_quote and len(prev_quote) >= 2 and quote_len <= 220:
        label = "pair"
        pair_with = "prev"
        user_text = prev_quote
        reply_text = quote
        formatted_text = f"私: {user_text}\n相手: {reply_text}"
        notes = "bootstrap from previous quote"
    elif cluster_size >= 2 and next_quote and quote_len <= 120 and len(next_quote) <= 220:
        label = "pair"
        pair_with = "next"
        user_text = quote
        reply_text = next_quote
        formatted_text = f"私: {user_text}\n相手: {reply_text}"
        notes = "bootstrap from next quote"
    elif heuristic_score >= 2 and quote_len <= 140:
        label = "single"
        reply_text = quote
        formatted_text = f"相手: {reply_text}"
        notes = "standalone reply candidate"
    else:
        label = "monologue"
        notes = "default conservative fallback"

    if label == "noise":
        tags: list[str] = []
    elif label in {"pair", "single"}:
        tags = detect_tags((user_text or "") + "\n" + (reply_text or ""))
    else:
        tags = detect_tags(quote)[:1] if quote else []

    enriched = dict(row)
    enriched.update(
        {
            "label": label,
            "pair_with": pair_with,
            "user_text": user_text,
            "reply_text": reply_text,
            "formatted_text": formatted_text,
            "tags": tags,
            "notes": notes,
        }
    )
    return enriched


def process_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    row_count = 0
    pair_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            labeled = bootstrap_row(row)
            if labeled["label"] == "pair":
                pair_count += 1
            dst.write(json.dumps(labeled, ensure_ascii=False))
            dst.write("\n")
            row_count += 1
    return row_count, pair_count


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.batches:
        input_files = [
            args.input_dir / f"dialogue_batch_{batch}.jsonl" for batch in args.batches
        ]
    else:
        input_files = sorted(args.input_dir.glob("dialogue_batch_*.jsonl"))

    total_rows = 0
    total_pairs = 0
    for input_path in input_files:
        if not input_path.exists():
            raise FileNotFoundError(f"Input batch not found: {input_path}")
        output_path = args.output_dir / input_path.name.replace(".jsonl", ".labeled.jsonl")
        row_count, pair_count = process_file(input_path, output_path)
        total_rows += row_count
        total_pairs += pair_count

    print(f"file_count={len(input_files)}")
    print(f"row_count={total_rows}")
    print(f"pair_count={total_pairs}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
