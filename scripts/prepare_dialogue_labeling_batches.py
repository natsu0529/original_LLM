#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "dialogue_candidates.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "dialogue_batches"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare score-sorted JSONL batches for AI-assisted dialogue labeling."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Dialogue candidate JSONL produced by extract_aozora_dialogue_candidates.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where score-sorted labeling batches are written.",
    )
    parser.add_argument(
        "--min-heuristic-score",
        type=int,
        default=4,
        help="Only keep candidates whose heuristic_score is at least this value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of candidates per batch file.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of batches to write.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate JSONL not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sort_key(row: dict) -> tuple:
    return (
        -int(row.get("heuristic_score", 0)),
        -int(row.get("cluster_size", 0)),
        row.get("work_id", ""),
        int(row.get("quote_index", 0)),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input_path)
    filtered_rows = [
        row for row in rows if int(row.get("heuristic_score", 0)) >= args.min_heuristic_score
    ]
    filtered_rows.sort(key=sort_key)

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_count = 0
    written_rows = 0
    for offset in range(0, len(filtered_rows), args.batch_size):
        if args.max_batches is not None and batch_count >= args.max_batches:
            break

        batch_rows = filtered_rows[offset : offset + args.batch_size]
        if not batch_rows:
            continue

        batch_count += 1
        batch_path = args.output_dir / f"dialogue_batch_{batch_count:04d}.jsonl"
        annotated_rows: list[dict] = []
        for index, row in enumerate(batch_rows, start=1):
            item = dict(row)
            item["batch_id"] = f"batch-{batch_count:04d}"
            item["batch_index"] = index
            annotated_rows.append(item)
        write_jsonl(batch_path, annotated_rows)
        written_rows += len(annotated_rows)

    summary = {
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "input_count": len(rows),
        "filtered_count": len(filtered_rows),
        "written_count": written_rows,
        "batch_count": batch_count,
        "min_heuristic_score": args.min_heuristic_score,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
    }
    write_summary(args.output_dir / "summary.json", summary)

    print(f"input_count={len(rows)}")
    print(f"filtered_count={len(filtered_rows)}")
    print(f"written_count={written_rows}")
    print(f"batch_count={batch_count}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
