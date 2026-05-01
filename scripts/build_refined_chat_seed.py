#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from original_llm.config import CHAT_TURN_END_MARKER, REPO_ROOT


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "chat_seed_refined_v1"
DEFAULT_WORK_ID_START = 7001
DEFAULT_SOURCE_SPECS = (
    f"{REPO_ROOT / 'data' / 'chat_seed_simple'}:8",
    f"{REPO_ROOT / 'data' / 'chat_seed_auto_short'}:2",
    f"{REPO_ROOT / 'data' / 'chat_seed_auto_peers_short'}:1",
)
ROLE_LINE_RE = re.compile(r"^(私|相手):\s*(.*)$")
SPACE_RE = re.compile(r"[ \t\u3000]+")
NOISE_SUBSTRINGS = (
    "底本：",
    "青空文庫",
    "青空文庫作成ファイル",
    "［＃",
    "※",
)


@dataclass(slots=True)
class SourceSpec:
    path: Path
    repeat: int
    name: str


@dataclass(frozen=True, slots=True)
class Turn:
    user_text: str
    reply_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cleaned chat dataset with short reply turns, "
            "2-4 turn buckets, and explicit turn-end markers."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source spec in the form path[:repeat]. Defaults to the built-in refined mix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where refined txt files and manifest are written.",
    )
    parser.add_argument(
        "--work-id-start",
        type=int,
        default=DEFAULT_WORK_ID_START,
        help="Starting synthetic work id for exported files.",
    )
    parser.add_argument(
        "--max-user-chars",
        type=int,
        default=56,
        help="Drop turns whose user side exceeds this many characters.",
    )
    parser.add_argument(
        "--max-reply-chars",
        type=int,
        default=72,
        help="Drop turns whose reply side exceeds this many characters.",
    )
    parser.add_argument(
        "--max-reply-sentences",
        type=int,
        default=2,
        help="Drop replies with too many sentence endings.",
    )
    parser.add_argument(
        "--max-turns-per-example",
        type=int,
        default=4,
        help="Maximum turns kept in a multi-turn training example.",
    )
    parser.add_argument(
        "--min-multi-turns",
        type=int,
        default=2,
        help="Minimum turns required for a multi-turn training example.",
    )
    parser.add_argument(
        "--max-example-chars",
        type=int,
        default=320,
        help="Maximum character length of a formatted example including markers.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove previously exported txt files and metadata before writing.",
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


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            title = row.get("title")
            if isinstance(title, str) and title.strip():
                rows[Path(title).stem] = row
            source_path = row.get("source_path")
            if isinstance(source_path, str) and source_path.strip():
                rows[Path(source_path).stem] = row
    return rows


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "seed"


def split_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]


def normalize_text(text: str) -> str:
    normalized = text.replace(CHAT_TURN_END_MARKER, "").strip()
    normalized = SPACE_RE.sub(" ", normalized)
    return normalized.strip()


def count_sentence_endings(text: str) -> int:
    return sum(text.count(mark) for mark in ("。", "！", "？", "!", "?"))


def is_clean_text(text: str) -> bool:
    if not text:
        return False
    if any(marker in text for marker in NOISE_SUBSTRINGS):
        return False
    if "私:" in text or "相手:" in text or CHAT_TURN_END_MARKER in text:
        return False
    if text.count("\n") > 0:
        return False
    if len(set(text)) == 1 and len(text) > 4:
        return False
    return True


def parse_turns(block: str) -> list[Turn]:
    turns: list[Turn] = []
    pending_user: str | None = None

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == CHAT_TURN_END_MARKER:
            continue
        match = ROLE_LINE_RE.match(line)
        if match is None:
            return []
        role, body = match.groups()
        text = normalize_text(body)
        if role == "私":
            if pending_user is not None:
                return []
            pending_user = text
            continue
        if pending_user is None:
            return []
        turns.append(Turn(user_text=pending_user, reply_text=text))
        pending_user = None

    if pending_user is not None:
        return []
    return turns


def is_valid_turn(
    turn: Turn,
    *,
    max_user_chars: int,
    max_reply_chars: int,
    max_reply_sentences: int,
) -> bool:
    if not is_clean_text(turn.user_text) or not is_clean_text(turn.reply_text):
        return False
    if len(turn.user_text) > max_user_chars or len(turn.reply_text) > max_reply_chars:
        return False
    if count_sentence_endings(turn.reply_text) > max_reply_sentences:
        return False
    if turn.user_text == turn.reply_text and len(turn.user_text) > 8:
        return False
    return True


def split_valid_segments(turns: list[Turn], **kwargs: int) -> list[list[Turn]]:
    segments: list[list[Turn]] = []
    current: list[Turn] = []

    for turn in turns:
        if is_valid_turn(turn, **kwargs):
            current.append(turn)
            continue
        if current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def format_turns(turns: list[Turn]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"私: {turn.user_text}".rstrip())
        lines.append(f"相手: {turn.reply_text}".rstrip())
        lines.append(CHAT_TURN_END_MARKER)
    return "\n".join(lines).rstrip()


def build_multi_turn_chunks(
    turns: list[Turn],
    *,
    min_multi_turns: int,
    max_turns_per_example: int,
    max_example_chars: int,
) -> list[list[Turn]]:
    if len(turns) < min_multi_turns:
        return []

    chunks: list[list[Turn]] = []
    start = 0
    while start < len(turns):
        remaining = len(turns) - start
        if remaining < min_multi_turns:
            break
        if remaining == max_turns_per_example + 1:
            chunk_size = max_turns_per_example - 1
        else:
            chunk_size = min(max_turns_per_example, remaining)
        chunk = turns[start : start + chunk_size]
        if len(chunk) < min_multi_turns:
            break
        if len(format_turns(chunk)) <= max_example_chars:
            chunks.append(chunk)
        start += chunk_size
    return chunks


def main() -> int:
    args = parse_args()
    raw_sources = args.source or list(DEFAULT_SOURCE_SPECS)
    specs = [parse_source_spec(raw) for raw in raw_sources]

    if args.max_user_chars <= 0:
        raise ValueError("--max-user-chars must be positive")
    if args.max_reply_chars <= 0:
        raise ValueError("--max-reply-chars must be positive")
    if args.max_reply_sentences <= 0:
        raise ValueError("--max-reply-sentences must be positive")
    if args.min_multi_turns < 2:
        raise ValueError("--min-multi-turns must be at least 2")
    if args.max_turns_per_example < args.min_multi_turns:
        raise ValueError("--max-turns-per-example must be >= --min-multi-turns")
    if args.max_example_chars <= 0:
        raise ValueError("--max-example-chars must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for stale_path in args.output_dir.glob("*.txt"):
            stale_path.unlink()
        for stale_name in ("manifest.jsonl", "summary.json"):
            stale_path = args.output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    manifest_rows: list[dict] = []
    summary_rows: list[dict] = []
    current_work_id = args.work_id_start
    total_turn1_examples = 0
    total_multi_examples = 0
    total_rejected_blocks = 0

    filter_kwargs = {
        "max_user_chars": args.max_user_chars,
        "max_reply_chars": args.max_reply_chars,
        "max_reply_sentences": args.max_reply_sentences,
    }

    for spec in specs:
        manifest_by_stem = load_manifest(spec.path / "manifest.jsonl")
        text_paths = sorted(spec.path.glob("*.txt"))
        if not text_paths:
            raise ValueError(f"No .txt files found in {spec.path}")

        dataset_slug = slugify(spec.name)
        for repeat_index in range(1, spec.repeat + 1):
            for text_path in text_paths:
                source_manifest = manifest_by_stem.get(text_path.stem, {})
                turn1_blocks: list[str] = []
                multi_blocks: list[str] = []
                rejected_blocks = 0

                for block in split_blocks(text_path.read_text(encoding="utf-8")):
                    turns = parse_turns(block)
                    if not turns:
                        rejected_blocks += 1
                        continue
                    valid_segments = split_valid_segments(turns, **filter_kwargs)
                    if not valid_segments:
                        rejected_blocks += 1
                        continue

                    for segment in valid_segments:
                        for turn in segment:
                            formatted = format_turns([turn])
                            if len(formatted) <= args.max_example_chars:
                                turn1_blocks.append(formatted)
                                total_turn1_examples += 1
                        for chunk in build_multi_turn_chunks(
                            segment,
                            min_multi_turns=args.min_multi_turns,
                            max_turns_per_example=args.max_turns_per_example,
                            max_example_chars=args.max_example_chars,
                        ):
                            multi_blocks.append(format_turns(chunk))
                            total_multi_examples += 1

                total_rejected_blocks += rejected_blocks

                for bucket_name, blocks in (
                    ("turn1", turn1_blocks),
                    ("turn2to4", multi_blocks),
                ):
                    if not blocks:
                        continue
                    output_name = (
                        f"{current_work_id}_{dataset_slug}_r{repeat_index:02d}_"
                        f"{text_path.stem}_{bucket_name}.txt"
                    )
                    output_path = args.output_dir / output_name
                    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
                    manifest_rows.append(
                        {
                            "work_id": current_work_id,
                            "title": output_name[:-4],
                            "source_dataset": spec.name,
                            "source_path": str(text_path),
                            "source_title": source_manifest.get("title", text_path.stem),
                            "source_work_id": source_manifest.get("work_id"),
                            "repeat_index": repeat_index,
                            "repeat_total": spec.repeat,
                            "bucket": bucket_name,
                            "example_count": len(blocks),
                            "turn_end_marker": CHAT_TURN_END_MARKER,
                        }
                    )
                    summary_rows.append(
                        {
                            "output_name": output_name,
                            "bucket": bucket_name,
                            "example_count": len(blocks),
                            "source_dataset": spec.name,
                            "source_file": text_path.name,
                            "repeat_index": repeat_index,
                        }
                    )
                    current_work_id += 1

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "output_dir": str(args.output_dir),
        "manifest_count": len(manifest_rows),
        "turn1_example_count": total_turn1_examples,
        "multi_example_count": total_multi_examples,
        "rejected_block_count": total_rejected_blocks,
        "turn_end_marker": CHAT_TURN_END_MARKER,
        "sources": [
            {"path": str(spec.path), "name": spec.name, "repeat": spec.repeat}
            for spec in specs
        ],
        "files": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"manifest_count={len(manifest_rows)}")
    print(f"turn1_example_count={total_turn1_examples}")
    print(f"multi_example_count={total_multi_examples}")
    print(f"rejected_block_count={total_rejected_blocks}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
