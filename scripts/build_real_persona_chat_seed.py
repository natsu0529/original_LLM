#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from original_llm.config import CHAT_TURN_END_MARKER, REPO_ROOT


DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "raw" / "real_persona_chat" / "source"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "chat_seed_real_persona_v1"
DEFAULT_WORK_ID_START = 8001
DEFAULT_DATASET_SLUG = "real_persona_chat"
DEFAULT_EVAL_METRICS = ("familiarity", "comprehension", "satisfaction")
ROLE_LINE_RE = re.compile(r"^(私|相手):\s*(.*)$")
SPACE_RE = re.compile(r"[ \t\u3000]+")
NAME_PLACEHOLDER_RE = re.compile(r"<[A-Z]{2,}>")
POLITE_RE = re.compile(
    r"(です|ます|でした|ました|でしょう|でしょうか|ません|ませんか|ございます|"
    r"いたします|お願い(?:します|いたします)|ください)"
)


@dataclass(frozen=True, slots=True)
class Turn:
    user_text: str
    reply_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean Japanese chat seed from RealPersonaChat with short "
            "turns, explicit turn-end markers, and 2-4 turn buckets."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Path to the RealPersonaChat repository root or the nested "
            "real_persona_chat directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the converted chat seed is written.",
    )
    parser.add_argument(
        "--work-id-start",
        type=int,
        default=DEFAULT_WORK_ID_START,
        help="Starting synthetic work id for exported shard files.",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=DEFAULT_DATASET_SLUG,
        help="Dataset slug written into exported filenames and manifest rows.",
    )
    parser.add_argument(
        "--max-user-chars",
        type=int,
        default=72,
        help="Drop turns whose user side exceeds this many characters.",
    )
    parser.add_argument(
        "--max-reply-chars",
        type=int,
        default=88,
        help="Drop turns whose reply side exceeds this many characters.",
    )
    parser.add_argument(
        "--max-reply-sentences",
        type=int,
        default=3,
        help="Drop replies with too many sentence endings.",
    )
    parser.add_argument(
        "--min-eval-score",
        type=float,
        default=4.0,
        help=(
            "Minimum average score required for each selected evaluation metric. "
            "Set below 0 to disable dialogue-level filtering."
        ),
    )
    parser.add_argument(
        "--eval-metric",
        action="append",
        default=[],
        help=(
            "Evaluation metric to gate on. Repeatable. Defaults to "
            f"{', '.join(DEFAULT_EVAL_METRICS)}."
        ),
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
        default=420,
        help="Maximum character length of a formatted example including markers.",
    )
    parser.add_argument(
        "--max-examples-per-file",
        type=int,
        default=2048,
        help="Maximum number of examples written to each output shard.",
    )
    parser.add_argument(
        "--both-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export each dialogue from both speaker directions when possible.",
    )
    parser.add_argument(
        "--forbid-polite-user",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop turns whose user side contains explicit polite-form markers.",
    )
    parser.add_argument(
        "--forbid-polite-reply",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop turns whose reply side contains explicit polite-form markers.",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip exact duplicate formatted examples.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove previously exported txt files and metadata before writing.",
    )
    return parser.parse_args()


def resolve_dataset_dir(source_dir: Path) -> Path:
    candidates = (
        source_dir,
        source_dir / "real_persona_chat",
    )
    for candidate in candidates:
        if (candidate / "dialogues").is_dir() and (candidate / "interlocutors.json").is_file():
            return candidate
    raise FileNotFoundError(
        "RealPersonaChat dataset not found. Expected either:\n"
        f"  - {source_dir / 'dialogues'}\n"
        f"  - {source_dir / 'real_persona_chat' / 'dialogues'}"
    )


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace(CHAT_TURN_END_MARKER, " ")
    normalized = normalized.replace("\n", " ")
    normalized = SPACE_RE.sub(" ", normalized)
    return normalized.strip()


def count_sentence_endings(text: str) -> int:
    return sum(text.count(mark) for mark in ("。", "！", "？", "!", "?"))


def is_clean_text(text: str) -> bool:
    if not text:
        return False
    if ROLE_LINE_RE.match(text):
        return False
    if "私:" in text or "相手:" in text or CHAT_TURN_END_MARKER in text:
        return False
    if NAME_PLACEHOLDER_RE.search(text):
        return False
    if len(set(text)) == 1 and len(text) > 4:
        return False
    return True


def is_valid_turn(
    turn: Turn,
    *,
    max_user_chars: int,
    max_reply_chars: int,
    max_reply_sentences: int,
    forbid_polite_user: bool,
    forbid_polite_reply: bool,
) -> bool:
    if not is_clean_text(turn.user_text) or not is_clean_text(turn.reply_text):
        return False
    if len(turn.user_text) > max_user_chars or len(turn.reply_text) > max_reply_chars:
        return False
    if count_sentence_endings(turn.reply_text) > max_reply_sentences:
        return False
    if forbid_polite_user and POLITE_RE.search(turn.user_text):
        return False
    if forbid_polite_reply and POLITE_RE.search(turn.reply_text):
        return False
    if turn.user_text == turn.reply_text and len(turn.user_text) > 8:
        return False
    return True


def format_turns(turns: list[Turn]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"私: {turn.user_text}".rstrip())
        lines.append(f"相手: {turn.reply_text}".rstrip())
        lines.append(CHAT_TURN_END_MARKER)
    return "\n".join(lines).rstrip()


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


def dialogue_passes_quality_gate(
    dialogue: dict,
    *,
    eval_metrics: tuple[str, ...],
    min_eval_score: float,
) -> tuple[bool, dict[str, float]]:
    evaluations = dialogue.get("evaluations", [])
    metric_scores: dict[str, float] = {}
    if not evaluations or min_eval_score < 0:
        return True, metric_scores

    for metric in eval_metrics:
        values = [row.get(metric) for row in evaluations if isinstance(row.get(metric), int)]
        if not values:
            return False, metric_scores
        metric_scores[metric] = mean(values)
        if metric_scores[metric] < min_eval_score:
            return False, metric_scores
    return True, metric_scores


def build_turns_from_utterances(utterances: list[dict], start_index: int) -> list[Turn]:
    turns: list[Turn] = []
    for index in range(start_index, len(utterances) - 1, 2):
        user_text = normalize_text(str(utterances[index].get("text", "")))
        reply_text = normalize_text(str(utterances[index + 1].get("text", "")))
        turns.append(Turn(user_text=user_text, reply_text=reply_text))
    return turns


def shard_blocks(
    *,
    output_dir: Path,
    dataset_slug: str,
    bucket: str,
    blocks: list[str],
    dialogue_ids: set[int],
    work_id: int,
    shard_index: int,
    manifest_rows: list[dict],
    summary_rows: list[dict],
    source_dir: Path,
    eval_metrics: tuple[str, ...],
    min_eval_score: float,
) -> tuple[int, int]:
    output_name = f"{work_id}_{dataset_slug}_{bucket}_s{shard_index:04d}.txt"
    output_path = output_dir / output_name
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    manifest_rows.append(
        {
            "work_id": work_id,
            "title": output_name[:-4],
            "source_dataset": dataset_slug,
            "source_path": str(source_dir),
            "bucket": bucket,
            "shard_index": shard_index,
            "example_count": len(blocks),
            "dialogue_count": len(dialogue_ids),
            "turn_end_marker": CHAT_TURN_END_MARKER,
            "eval_metrics": list(eval_metrics),
            "min_eval_score": min_eval_score,
        }
    )
    summary_rows.append(
        {
            "output_name": output_name,
            "bucket": bucket,
            "example_count": len(blocks),
            "dialogue_count": len(dialogue_ids),
            "shard_index": shard_index,
        }
    )
    return work_id + 1, shard_index + 1


def main() -> int:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.source_dir)
    dialogue_dir = dataset_dir / "dialogues"
    dialogue_paths = sorted(dialogue_dir.glob("*.json"))
    if not dialogue_paths:
        raise ValueError(f"No dialogue JSON files found in {dialogue_dir}")

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
    if args.max_examples_per_file <= 0:
        raise ValueError("--max-examples-per-file must be positive")

    eval_metrics = tuple(args.eval_metric) if args.eval_metric else DEFAULT_EVAL_METRICS
    if not eval_metrics:
        raise ValueError("At least one eval metric is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for stale_path in args.output_dir.glob("*.txt"):
            stale_path.unlink()
        for stale_name in ("manifest.jsonl", "summary.json"):
            stale_path = args.output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    filter_kwargs = {
        "max_user_chars": args.max_user_chars,
        "max_reply_chars": args.max_reply_chars,
        "max_reply_sentences": args.max_reply_sentences,
        "forbid_polite_user": args.forbid_polite_user,
        "forbid_polite_reply": args.forbid_polite_reply,
    }
    manifest_rows: list[dict] = []
    summary_rows: list[dict] = []
    current_work_id = args.work_id_start
    turn1_shard_index = 1
    multi_shard_index = 1
    turn1_blocks: list[str] = []
    multi_blocks: list[str] = []
    turn1_dialogue_ids: set[int] = set()
    multi_dialogue_ids: set[int] = set()
    seen_turn1: set[str] = set()
    seen_multi: set[str] = set()

    total_dialogues = 0
    kept_dialogues = 0
    filtered_dialogues = 0
    total_turn1_examples = 0
    total_multi_examples = 0
    duplicate_turn1_examples = 0
    duplicate_multi_examples = 0
    rejected_turns = 0

    direction_starts = (0, 1) if args.both_directions else (0,)

    for dialogue_path in dialogue_paths:
        dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
        total_dialogues += 1
        passes_quality, metric_scores = dialogue_passes_quality_gate(
            dialogue,
            eval_metrics=eval_metrics,
            min_eval_score=args.min_eval_score,
        )
        if not passes_quality:
            filtered_dialogues += 1
            continue

        utterances = dialogue.get("utterances", [])
        if len(utterances) < 2:
            filtered_dialogues += 1
            continue

        dialogue_id = int(dialogue.get("dialogue_id", total_dialogues))
        dialogue_contributed = False

        for start_index in direction_starts:
            turns = build_turns_from_utterances(utterances, start_index)
            if not turns:
                continue

            valid_segments = split_valid_segments(turns, **filter_kwargs)
            if not valid_segments:
                rejected_turns += len(turns)
                continue

            invalid_turn_count = len(turns) - sum(len(segment) for segment in valid_segments)
            rejected_turns += max(invalid_turn_count, 0)

            for segment in valid_segments:
                for turn in segment:
                    formatted = format_turns([turn])
                    if len(formatted) > args.max_example_chars:
                        rejected_turns += 1
                        continue
                    if args.dedupe and formatted in seen_turn1:
                        duplicate_turn1_examples += 1
                        continue
                    if args.dedupe:
                        seen_turn1.add(formatted)
                    turn1_blocks.append(formatted)
                    turn1_dialogue_ids.add(dialogue_id)
                    total_turn1_examples += 1
                    dialogue_contributed = True
                    if len(turn1_blocks) >= args.max_examples_per_file:
                        current_work_id, turn1_shard_index = shard_blocks(
                            output_dir=args.output_dir,
                            dataset_slug=args.dataset_slug,
                            bucket="turn1",
                            blocks=turn1_blocks,
                            dialogue_ids=turn1_dialogue_ids,
                            work_id=current_work_id,
                            shard_index=turn1_shard_index,
                            manifest_rows=manifest_rows,
                            summary_rows=summary_rows,
                            source_dir=dataset_dir,
                            eval_metrics=eval_metrics,
                            min_eval_score=args.min_eval_score,
                        )
                        turn1_blocks = []
                        turn1_dialogue_ids = set()

                for chunk in build_multi_turn_chunks(
                    segment,
                    min_multi_turns=args.min_multi_turns,
                    max_turns_per_example=args.max_turns_per_example,
                    max_example_chars=args.max_example_chars,
                ):
                    formatted = format_turns(chunk)
                    if args.dedupe and formatted in seen_multi:
                        duplicate_multi_examples += 1
                        continue
                    if args.dedupe:
                        seen_multi.add(formatted)
                    multi_blocks.append(formatted)
                    multi_dialogue_ids.add(dialogue_id)
                    total_multi_examples += 1
                    dialogue_contributed = True
                    if len(multi_blocks) >= args.max_examples_per_file:
                        current_work_id, multi_shard_index = shard_blocks(
                            output_dir=args.output_dir,
                            dataset_slug=args.dataset_slug,
                            bucket="turn2to4",
                            blocks=multi_blocks,
                            dialogue_ids=multi_dialogue_ids,
                            work_id=current_work_id,
                            shard_index=multi_shard_index,
                            manifest_rows=manifest_rows,
                            summary_rows=summary_rows,
                            source_dir=dataset_dir,
                            eval_metrics=eval_metrics,
                            min_eval_score=args.min_eval_score,
                        )
                        multi_blocks = []
                        multi_dialogue_ids = set()

        if dialogue_contributed:
            kept_dialogues += 1
        else:
            filtered_dialogues += 1

    if turn1_blocks:
        current_work_id, turn1_shard_index = shard_blocks(
            output_dir=args.output_dir,
            dataset_slug=args.dataset_slug,
            bucket="turn1",
            blocks=turn1_blocks,
            dialogue_ids=turn1_dialogue_ids,
            work_id=current_work_id,
            shard_index=turn1_shard_index,
            manifest_rows=manifest_rows,
            summary_rows=summary_rows,
            source_dir=dataset_dir,
            eval_metrics=eval_metrics,
            min_eval_score=args.min_eval_score,
        )
    if multi_blocks:
        current_work_id, multi_shard_index = shard_blocks(
            output_dir=args.output_dir,
            dataset_slug=args.dataset_slug,
            bucket="turn2to4",
            blocks=multi_blocks,
            dialogue_ids=multi_dialogue_ids,
            work_id=current_work_id,
            shard_index=multi_shard_index,
            manifest_rows=manifest_rows,
            summary_rows=summary_rows,
            source_dir=dataset_dir,
            eval_metrics=eval_metrics,
            min_eval_score=args.min_eval_score,
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "source_dir": str(dataset_dir),
        "output_dir": str(args.output_dir),
        "dialogue_count": total_dialogues,
        "kept_dialogue_count": kept_dialogues,
        "filtered_dialogue_count": filtered_dialogues,
        "manifest_count": len(manifest_rows),
        "turn1_example_count": total_turn1_examples,
        "multi_example_count": total_multi_examples,
        "duplicate_turn1_example_count": duplicate_turn1_examples,
        "duplicate_multi_example_count": duplicate_multi_examples,
        "rejected_turn_count": rejected_turns,
        "turn_end_marker": CHAT_TURN_END_MARKER,
        "both_directions": args.both_directions,
        "dedupe": args.dedupe,
        "eval_metrics": list(eval_metrics),
        "min_eval_score": args.min_eval_score,
        "forbid_polite_user": args.forbid_polite_user,
        "forbid_polite_reply": args.forbid_polite_reply,
        "files": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"dialogue_count={total_dialogues}")
    print(f"kept_dialogue_count={kept_dialogues}")
    print(f"filtered_dialogue_count={filtered_dialogues}")
    print(f"manifest_count={len(manifest_rows)}")
    print(f"turn1_example_count={total_turn1_examples}")
    print(f"multi_example_count={total_multi_examples}")
    print(f"duplicate_turn1_example_count={duplicate_turn1_examples}")
    print(f"duplicate_multi_example_count={duplicate_multi_examples}")
    print(f"rejected_turn_count={rejected_turns}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
