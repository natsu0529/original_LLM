#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "labeled_batches"
)
VALID_LABELS = {"pair", "single", "monologue", "noise"}
VALID_PAIR_WITH = {"prev", "next", None}
VALID_TAGS = {
    "daily",
    "banter",
    "mood",
    "confession",
    "argument",
    "affection",
    "sensual",
    "alcohol",
    "literature",
    "philosophy",
    "humor",
    "greeting",
    "question",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate labeled dialogue batch JSONL files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing dialogue_batch_*.labeled.jsonl files.",
    )
    return parser.parse_args()


def validate_row(path: Path, row_number: int, row: dict) -> list[str]:
    errors: list[str] = []
    label = row.get("label")
    pair_with = row.get("pair_with")
    user_text = row.get("user_text")
    reply_text = row.get("reply_text")
    formatted_text = row.get("formatted_text")
    tags = row.get("tags")

    if label not in VALID_LABELS:
        errors.append(f"{path}:{row_number}: invalid label {label!r}")
    if pair_with not in VALID_PAIR_WITH:
        errors.append(f"{path}:{row_number}: invalid pair_with {pair_with!r}")
    if not isinstance(tags, list):
        errors.append(f"{path}:{row_number}: tags must be a list")
    else:
        unknown_tags = [tag for tag in tags if tag not in VALID_TAGS]
        if unknown_tags:
            errors.append(f"{path}:{row_number}: unknown tags {unknown_tags!r}")

    if label == "pair":
        if pair_with not in {"prev", "next"}:
            errors.append(f"{path}:{row_number}: pair must set pair_with")
        if not user_text or not reply_text:
            errors.append(f"{path}:{row_number}: pair must set user_text and reply_text")
        expected_prefix = "私: "
        expected_reply = "\n相手: "
        if not isinstance(formatted_text, str) or expected_prefix not in formatted_text or expected_reply not in formatted_text:
            errors.append(f"{path}:{row_number}: pair formatted_text must contain 私:/相手:")
    elif label == "single":
        if pair_with is not None:
            errors.append(f"{path}:{row_number}: single must not set pair_with")
        if user_text is not None:
            errors.append(f"{path}:{row_number}: single must not set user_text")
        if not reply_text:
            errors.append(f"{path}:{row_number}: single must set reply_text")
        if not isinstance(formatted_text, str) or not formatted_text.startswith("相手: "):
            errors.append(f"{path}:{row_number}: single formatted_text must start with 相手:")
    else:
        if pair_with is not None:
            errors.append(f"{path}:{row_number}: {label} must not set pair_with")
        if user_text is not None or reply_text is not None or formatted_text is not None:
            errors.append(f"{path}:{row_number}: {label} must not set user_text/reply_text/formatted_text")

    return errors


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    files = sorted(args.input_dir.glob("dialogue_batch_*.labeled.jsonl"))
    total_rows = 0
    all_errors: list[str] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                total_rows += 1
                row = json.loads(line)
                all_errors.extend(validate_row(path, row_number, row))

    print(f"file_count={len(files)}")
    print(f"row_count={total_rows}")
    print(f"error_count={len(all_errors)}")
    for error in all_errors[:100]:
        print(error)
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
