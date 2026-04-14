#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "chat_seed_mix"
DEFAULT_WORK_ID_START = 6001


@dataclass(slots=True)
class SourceSpec:
    path: Path
    repeat: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple chat seed directories into one weighted dataset."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source spec in the form path[:repeat]. Repeat defaults to 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Combined output directory.",
    )
    parser.add_argument(
        "--work-id-start",
        type=int,
        default=DEFAULT_WORK_ID_START,
        help="Starting synthetic work id for copied files.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove previously exported txt files and manifest before writing.",
    )
    return parser.parse_args()


def parse_source_spec(raw: str) -> SourceSpec:
    value = raw.strip()
    if not value:
        raise ValueError("Empty --source value")

    if ":" in value:
        path_text, repeat_text = value.rsplit(":", 1)
        repeat = int(repeat_text)
    else:
        path_text = value
        repeat = 1

    path = Path(path_text)
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}")
    if not path.exists():
        raise FileNotFoundError(f"Source directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {path}")
    return SourceSpec(path=path, repeat=repeat, name=path.name)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "seed"


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records[Path(str(row.get("title", ""))).stem] = row
            source_path = row.get("source_path")
            if isinstance(source_path, str):
                records[Path(source_path).stem] = row
    return records


def main() -> int:
    args = parse_args()
    if not args.source:
        raise ValueError("At least one --source is required")

    specs = [parse_source_spec(raw) for raw in args.source]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for stale_path in args.output_dir.glob("*.txt"):
            stale_path.unlink()
        for stale_name in ("manifest.jsonl", "summary.json"):
            stale_path = args.output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    manifest_rows: list[dict] = []
    current_work_id = args.work_id_start
    total_text_files = 0

    for spec in specs:
        manifest_by_stem = load_manifest(spec.path / "manifest.jsonl")
        text_paths = sorted(spec.path.glob("*.txt"))
        if not text_paths:
            raise ValueError(f"No .txt files found in {spec.path}")

        for repeat_index in range(1, spec.repeat + 1):
            for text_path in text_paths:
                source_stem = text_path.stem
                source_manifest = manifest_by_stem.get(source_stem, {})
                title = source_manifest.get("title", source_stem)
                dataset_slug = slugify(spec.name)
                output_name = (
                    f"{current_work_id}_{dataset_slug}_r{repeat_index:02d}_{text_path.name}"
                )
                output_path = args.output_dir / output_name
                shutil.copy2(text_path, output_path)

                manifest_rows.append(
                    {
                        "work_id": current_work_id,
                        "title": title,
                        "source_dataset": spec.name,
                        "source_path": str(text_path),
                        "source_title": source_manifest.get("title", title),
                        "source_work_id": source_manifest.get("work_id"),
                        "repeat_index": repeat_index,
                        "repeat_total": spec.repeat,
                    }
                )
                current_work_id += 1
                total_text_files += 1

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "output_dir": str(args.output_dir),
        "file_count": total_text_files,
        "manifest_count": len(manifest_rows),
        "sources": [
            {
                "path": str(spec.path),
                "name": spec.name,
                "repeat": spec.repeat,
                "text_file_count": len(list(spec.path.glob("*.txt"))),
            }
            for spec in specs
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"file_count={total_text_files}")
    print(f"manifest_count={len(manifest_rows)}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
