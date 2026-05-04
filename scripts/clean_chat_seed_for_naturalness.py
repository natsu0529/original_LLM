"""Filter chat seed `.txt` files using the runtime low-quality detector.

This script reads every chat-seed-style `.txt` file under ``--source-dir``,
parses its turns, drops blocks whose final reply fails the same
``is_low_quality_reply`` heuristic the runtime uses, and writes the surviving
blocks back into ``--out-dir`` (one file per source file, preserving the
filename).

Use this to build a curated copy of `chat_seed_friend_clean_v1` (or any other
seed dir) before retraining or before pointing ``--retrieval-corpus-dir``.

Example:

  uv run python scripts/clean_chat_seed_for_naturalness.py \
      --source-dir data/chat_seed_friend_clean_v1 \
      --out-dir data/chat_seed_friend_clean_v2 \
      --report data/chat_seed_friend_clean_v2/_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from original_llm.config import CHAT_TURN_END_MARKER
from original_llm.generate import (
    format_chat_turns,
    is_low_quality_reply,
    parse_chat_turns,
    split_chat_blocks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--user-label", type=str, default="私")
    parser.add_argument("--reply-label", type=str, default="相手")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path with summary stats and dropped samples.",
    )
    parser.add_argument(
        "--max-dropped-samples",
        type=int,
        default=200,
        help="How many dropped (user, reply) pairs to retain in the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir: Path = args.source_dir
    out_dir: Path = args.out_dir

    if not source_dir.is_dir():
        raise SystemExit(f"--source-dir not found: {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    total_blocks = 0
    kept_blocks = 0
    dropped_blocks = 0
    dropped_samples: list[dict[str, str]] = []
    dropped_by_file: Counter[str] = Counter()

    for path in sorted(source_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        kept_chunks: list[str] = []
        for block in split_chat_blocks(text):
            total_blocks += 1
            turns = parse_chat_turns(block, args.user_label, args.reply_label)
            if not turns:
                dropped_blocks += 1
                dropped_by_file[path.name] += 1
                continue
            last_user, last_reply = turns[-1]
            if is_low_quality_reply(last_user, last_reply):
                dropped_blocks += 1
                dropped_by_file[path.name] += 1
                if len(dropped_samples) < args.max_dropped_samples:
                    dropped_samples.append(
                        {
                            "file": path.name,
                            "user": last_user,
                            "reply": last_reply,
                        }
                    )
                continue
            kept_chunks.append(
                format_chat_turns(turns, args.user_label, args.reply_label)
            )
            kept_blocks += 1

        if kept_chunks:
            out_path = out_dir / path.name
            out_path.write_text(
                "\n\n".join(kept_chunks) + "\n",
                encoding="utf-8",
            )

    summary = {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "total_blocks": total_blocks,
        "kept_blocks": kept_blocks,
        "dropped_blocks": dropped_blocks,
        "kept_files": sum(1 for _ in out_dir.glob("*.txt")),
        "top_dropped_files": dropped_by_file.most_common(20),
        "turn_end_marker": CHAT_TURN_END_MARKER,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "top_dropped_files"}, ensure_ascii=False, indent=2))
    print(f"top dropped files: {summary['top_dropped_files']}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {**summary, "dropped_samples": dropped_samples},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report written to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
