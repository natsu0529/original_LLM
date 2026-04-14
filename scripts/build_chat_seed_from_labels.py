#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "labeled_batches"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "chat_seed_auto"
DEFAULT_WORK_ID_START = 4001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chat seed txt files from labeled dialogue batches."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing dialogue_batch_*.labeled.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where chat seed txt files and manifest are written.",
    )
    parser.add_argument(
        "--work-id-start",
        type=int,
        default=DEFAULT_WORK_ID_START,
        help="Starting synthetic work id for exported files.",
    )
    parser.add_argument(
        "--include-single",
        action="store_true",
        help="Also export single replies into separate txt files.",
    )
    parser.add_argument(
        "--max-user-chars",
        type=int,
        default=None,
        help="Skip pair rows whose user_text exceeds this character length.",
    )
    parser.add_argument(
        "--max-reply-chars",
        type=int,
        default=None,
        help="Skip pair/single rows whose reply_text exceeds this character length.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove previously exported txt files in output-dir before writing new ones.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")
    if args.max_user_chars is not None and args.max_user_chars <= 0:
        raise ValueError(f"max_user_chars must be positive, got {args.max_user_chars}")
    if args.max_reply_chars is not None and args.max_reply_chars <= 0:
        raise ValueError(f"max_reply_chars must be positive, got {args.max_reply_chars}")

    files = sorted(args.input_dir.glob("dialogue_batch_*.labeled.jsonl"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for stale_path in args.output_dir.glob("*.txt"):
            stale_path.unlink()

    manifest_rows: list[dict] = []
    pair_export_count = 0
    single_export_count = 0
    skipped_long_pair_count = 0
    skipped_long_single_count = 0
    current_work_id = args.work_id_start

    for path in files:
        rows = load_rows(path)
        pair_blocks: list[str] = []
        single_blocks: list[str] = []
        tag_counter: defaultdict[str, int] = defaultdict(int)

        for row in rows:
            label = row.get("label")
            tags = row.get("tags") or []
            for tag in tags:
                tag_counter[tag] += 1

            if label == "pair":
                formatted_text = row.get("formatted_text")
                user_text = row.get("user_text")
                reply_text = row.get("reply_text")
                if args.max_user_chars is not None and isinstance(user_text, str):
                    if len(user_text.strip()) > args.max_user_chars:
                        skipped_long_pair_count += 1
                        continue
                if args.max_reply_chars is not None and isinstance(reply_text, str):
                    if len(reply_text.strip()) > args.max_reply_chars:
                        skipped_long_pair_count += 1
                        continue
                if isinstance(formatted_text, str) and formatted_text.strip():
                    pair_blocks.append(formatted_text.strip())
                    pair_export_count += 1
            elif args.include_single and label == "single":
                formatted_text = row.get("formatted_text")
                reply_text = row.get("reply_text")
                if args.max_reply_chars is not None and isinstance(reply_text, str):
                    if len(reply_text.strip()) > args.max_reply_chars:
                        skipped_long_single_count += 1
                        continue
                if isinstance(formatted_text, str) and formatted_text.strip():
                    single_blocks.append(formatted_text.strip())
                    single_export_count += 1

        if pair_blocks:
            output_name = f"{current_work_id}_auto_pairs_{path.stem}.txt"
            output_path = args.output_dir / output_name
            write_text(output_path, "\n\n".join(pair_blocks) + "\n",)
            manifest_rows.append(
                {
                    "work_id": current_work_id,
                    "title": f"auto_pairs_{path.stem}",
                    "source_batch": str(path),
                    "kind": "pair",
                    "pair_count": len(pair_blocks),
                    "single_count": len(single_blocks),
                    "top_tags": sorted(tag_counter.items(), key=lambda item: item[1], reverse=True)[:8],
                }
            )
            current_work_id += 1

        if args.include_single and single_blocks:
            output_name = f"{current_work_id}_auto_single_{path.stem}.txt"
            output_path = args.output_dir / output_name
            write_text(output_path, "\n\n".join(single_blocks) + "\n",)
            manifest_rows.append(
                {
                    "work_id": current_work_id,
                    "title": f"auto_single_{path.stem}",
                    "source_batch": str(path),
                    "kind": "single",
                    "pair_count": len(pair_blocks),
                    "single_count": len(single_blocks),
                    "top_tags": sorted(tag_counter.items(), key=lambda item: item[1], reverse=True)[:8],
                }
            )
            current_work_id += 1

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(files),
        "manifest_count": len(manifest_rows),
        "pair_export_count": pair_export_count,
        "single_export_count": single_export_count,
        "skipped_long_pair_count": skipped_long_pair_count,
        "skipped_long_single_count": skipped_long_single_count,
        "max_user_chars": args.max_user_chars,
        "max_reply_chars": args.max_reply_chars,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"file_count={len(files)}")
    print(f"manifest_count={len(manifest_rows)}")
    print(f"pair_export_count={pair_export_count}")
    print(f"single_export_count={single_export_count}")
    print(f"skipped_long_pair_count={skipped_long_pair_count}")
    print(f"skipped_long_single_count={skipped_long_single_count}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
