#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from original_llm.config import CHAT_TURN_END_MARKER, REPO_ROOT


DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "chat_seed_friend_casual_mix_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "chat_seed_friend_clean_v1"
DEFAULT_WORK_ID_START = 11001
ROLE_LINE_RE = re.compile(r"^(私|相手):\s*(.*)$")
SPACE_RE = re.compile(r"[ \t\u3000]+")
SENTENCE_END_RE = re.compile(r"[。！？!?]")
POLITE_RE = re.compile(
    r"(です|ます|でした|ました|でしょう|でしょうか|ません|ませんか|ございます|"
    r"いたします|お願い(?:します|いたします)|ください|下さい)(?:[。！？!?\s]|$)"
)
FORMAL_ACK_RE = re.compile(
    r"(^|[、。！？!?\s])(?:あ、|ほう、|うん、)?"
    r"(はい|そうですね|そうなんですね|なるほどですね)(?:[、。！？!?\s]|$)"
)
META_REPLY_RE = re.compile(
    r"(会話らしく|話が進まない|もう少しだけ言って|言い直す|突っ込むところ|"
    r"少しだらだら話そう|では、少しだらだら)"
)
INCOMPLETE_REPLY_SUFFIX_RE = re.compile(
    r"(?:系の|発売の|ような|みたいな|という|とか|って|けど|けれど|から|ので|"
    r"ために|ものを|ことを|予定は|つづきが|準備が|偏向報道が|ポジションは|"
    r"オムライスを|食欲のために|カフェに|砂漠の|ふわとろの)$"
)
COMPLETE_REPLY_SUFFIX_RE = re.compile(
    r"(?:[。！？!?]|よ|ね|な|わ|ぞ|ぜ|か|かな|かも|だ|だよ|だね|だな|る|た|"
    r"ない|たい|いい|する|した|してる|いる|ある|思う|わかる|なるほど|"
    r"ありがとう|おはよう|こんにちは|こんばんは|おやすみ|うん|うんうん|"
    r"あはは|ははは|へえ|ふふふ|たしかに|ほんとに|笑)$"
)
COMPLETE_REPLY_EXACT = {
    "いいよ",
    "いいね",
    "そうだね",
    "そうだよ",
    "わかる",
    "わかるよ",
    "なるほど",
    "たしかに",
    "ほんとに",
    "うん",
    "うんうん",
    "あはは",
    "ははは",
    "へえ",
    "こんにちは",
    "こんばんは",
    "おはよう",
}
DEFAULT_DENY_SUBSTRINGS = (
    "この代数の問題",
    "みんな新幹線",
    "交通手段新幹線",
    "新幹線か電車",
    "昆虫系",
    "昆虫食",
    "犬も猫",
    "猫かな",
    "きめれない",
    "怖い顔をしているね",
    "わかるよ、と言い切るほどでもない",
    "アイスも、それを",
    "いえ、味を",
    "味を。",
    "そこまで？",
    "口の中まで貧しくなる",
    "手の先から言葉まで冷える",
    "吐き出してしまえば",
    "何か吐き出したい",
    "焼きそばにキムチを入れたものを",
    "なんとなく。でもどちらもあまり飲まないけど笑",
    "あと、おいしい",
    "わー！！",
    "まだは、まだ続きましょう",
    "まだ続きましょう",
    "ベンアフレック",
    "どんなところがお好き",
    "遠いとね",
    "愛媛かな",
    "夢をはっきり記憶しているというか",
)
NOISE_SUBSTRINGS = (
    "底本：",
    "青空文庫",
    "青空文庫作成ファイル",
    "［＃",
    "※",
)


@dataclass(frozen=True, slots=True)
class Turn:
    user_text: str
    reply_text: str


@dataclass(frozen=True, slots=True)
class Rejection:
    reason: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean formatted 私/相手 chat seed files with block-level filters."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-id-start", type=int, default=DEFAULT_WORK_ID_START)
    parser.add_argument("--max-user-chars", type=int, default=56)
    parser.add_argument("--max-reply-chars", type=int, default=64)
    parser.add_argument("--max-reply-sentences", type=int, default=2)
    parser.add_argument("--max-example-chars", type=int, default=320)
    parser.add_argument("--max-turns-per-example", type=int, default=4)
    parser.add_argument("--max-examples-per-file", type=int, default=512)
    parser.add_argument(
        "--reject-polite-reply",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reject-formal-ack",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reject-meta-reply",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deny-substring",
        action="append",
        default=[],
        help="Additional substring that rejects a turn when present on either side.",
    )
    parser.add_argument(
        "--use-default-deny-substrings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def split_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]


def normalize_text(text: str) -> str:
    normalized = text.replace(CHAT_TURN_END_MARKER, "").strip()
    normalized = SPACE_RE.sub(" ", normalized)
    return normalized


def parse_turns(block: str) -> list[Turn]:
    turns: list[Turn] = []
    pending_user: str | None = None

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line == CHAT_TURN_END_MARKER:
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


def count_sentence_endings(text: str) -> int:
    return len(SENTENCE_END_RE.findall(text))


def is_noise_text(text: str) -> bool:
    if not text:
        return True
    if any(marker in text for marker in NOISE_SUBSTRINGS):
        return True
    if "私:" in text or "相手:" in text or CHAT_TURN_END_MARKER in text:
        return True
    if "\n" in text:
        return True
    if len(set(text)) == 1 and len(text) > 4:
        return True
    return False


def is_incomplete_reply(text: str) -> bool:
    compact = text.strip(" 　「」『』（）()[]")
    if not compact:
        return True
    if compact in COMPLETE_REPLY_EXACT:
        return False
    if INCOMPLETE_REPLY_SUFFIX_RE.search(compact):
        return True
    return COMPLETE_REPLY_SUFFIX_RE.search(compact) is None


def reject_turn(
    turn: Turn,
    *,
    max_user_chars: int,
    max_reply_chars: int,
    max_reply_sentences: int,
    reject_polite_reply: bool,
    reject_formal_ack: bool,
    reject_meta_reply: bool,
    deny_substrings: tuple[str, ...],
) -> Rejection | None:
    if is_noise_text(turn.user_text) or is_noise_text(turn.reply_text):
        return Rejection("noise", f"{turn.user_text} / {turn.reply_text}")
    if len(turn.user_text) > max_user_chars:
        return Rejection("user_too_long", turn.user_text)
    if len(turn.reply_text) > max_reply_chars:
        return Rejection("reply_too_long", turn.reply_text)
    if count_sentence_endings(turn.reply_text) > max_reply_sentences:
        return Rejection("reply_too_many_sentences", turn.reply_text)
    if turn.user_text == turn.reply_text and len(turn.user_text) > 8:
        return Rejection("echo", turn.reply_text)
    if is_incomplete_reply(turn.reply_text):
        return Rejection("incomplete_reply", turn.reply_text)
    if reject_polite_reply and POLITE_RE.search(turn.reply_text):
        return Rejection("polite_reply", turn.reply_text)
    if reject_formal_ack and FORMAL_ACK_RE.search(turn.reply_text):
        return Rejection("formal_ack_reply", turn.reply_text)
    if reject_meta_reply and META_REPLY_RE.search(turn.reply_text):
        return Rejection("meta_reply", turn.reply_text)
    for value in deny_substrings:
        if value and (value in turn.user_text or value in turn.reply_text):
            return Rejection("deny_substring", value)
    return None


def split_valid_segments(
    turns: list[Turn],
    *,
    rejection_counter: Counter[str],
    rejection_examples: dict[str, list[str]],
    **kwargs: object,
) -> list[list[Turn]]:
    segments: list[list[Turn]] = []
    current: list[Turn] = []

    for turn in turns:
        rejection = reject_turn(turn, **kwargs)
        if rejection is None:
            current.append(turn)
            continue
        rejection_counter[rejection.reason] += 1
        rejection_examples.setdefault(rejection.reason, [])
        if len(rejection_examples[rejection.reason]) < 5:
            rejection_examples[rejection.reason].append(rejection.text)
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


def chunk_segment(
    segment: list[Turn],
    *,
    max_turns_per_example: int,
    max_example_chars: int,
) -> list[str]:
    if len(segment) == 1:
        formatted = format_turns(segment)
        return [formatted] if len(formatted) <= max_example_chars else []

    blocks: list[str] = []
    start = 0
    while start < len(segment):
        chunk = segment[start : start + max_turns_per_example]
        if len(chunk) >= 2:
            formatted = format_turns(chunk)
            if len(formatted) <= max_example_chars:
                blocks.append(formatted)
        start += max_turns_per_example
    return blocks


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


def clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.txt"):
        stale_path.unlink()
    for stale_name in ("manifest.jsonl", "summary.json"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()


def shard_blocks(blocks: list[str], max_examples_per_file: int) -> list[list[str]]:
    return [
        blocks[index : index + max_examples_per_file]
        for index in range(0, len(blocks), max_examples_per_file)
    ]


def main() -> int:
    args = parse_args()
    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"source dir not found: {args.source_dir}")
    if args.max_turns_per_example < 1:
        raise ValueError("--max-turns-per-example must be positive")
    if args.max_example_chars <= 0:
        raise ValueError("--max-example-chars must be positive")
    if args.max_examples_per_file <= 0:
        raise ValueError("--max-examples-per-file must be positive")

    if args.clean:
        clear_output_dir(args.output_dir)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    deny_substrings = tuple(
        [
            *(DEFAULT_DENY_SUBSTRINGS if args.use_default_deny_substrings else ()),
            *args.deny_substring,
        ]
    )
    manifest_by_stem = load_manifest(args.source_dir / "manifest.jsonl")
    text_paths = sorted(args.source_dir.glob("*.txt"))
    if not text_paths:
        raise ValueError(f"No .txt files found in {args.source_dir}")

    manifest_rows: list[dict] = []
    summary_rows: list[dict] = []
    rejection_counter: Counter[str] = Counter()
    rejection_examples: dict[str, list[str]] = {}
    seen_blocks: set[str] = set()
    current_work_id = args.work_id_start
    input_block_count = 0
    output_block_count = 0

    filter_kwargs = {
        "max_user_chars": args.max_user_chars,
        "max_reply_chars": args.max_reply_chars,
        "max_reply_sentences": args.max_reply_sentences,
        "reject_polite_reply": args.reject_polite_reply,
        "reject_formal_ack": args.reject_formal_ack,
        "reject_meta_reply": args.reject_meta_reply,
        "deny_substrings": deny_substrings,
    }

    for text_path in text_paths:
        source_manifest = manifest_by_stem.get(text_path.stem, {})
        output_blocks: list[str] = []
        source_blocks = split_blocks(text_path.read_text(encoding="utf-8"))
        input_block_count += len(source_blocks)

        for block in source_blocks:
            turns = parse_turns(block)
            if not turns:
                rejection_counter["parse_failed"] += 1
                continue
            segments = split_valid_segments(
                turns,
                rejection_counter=rejection_counter,
                rejection_examples=rejection_examples,
                **filter_kwargs,
            )
            if not segments:
                continue
            for segment in segments:
                for formatted in chunk_segment(
                    segment,
                    max_turns_per_example=args.max_turns_per_example,
                    max_example_chars=args.max_example_chars,
                ):
                    if args.dedupe and formatted in seen_blocks:
                        rejection_counter["duplicate"] += 1
                        continue
                    seen_blocks.add(formatted)
                    output_blocks.append(formatted)

        if not output_blocks:
            continue

        shards = shard_blocks(output_blocks, args.max_examples_per_file)
        for shard_index, shard in enumerate(shards, start=1):
            output_name = (
                f"{current_work_id}_{text_path.stem}_clean_s{shard_index:04d}.txt"
            )
            output_path = args.output_dir / output_name
            output_path.write_text("\n\n".join(shard) + "\n", encoding="utf-8")
            output_block_count += len(shard)
            manifest_rows.append(
                {
                    "work_id": current_work_id,
                    "title": output_name[:-4],
                    "source_dataset": source_manifest.get(
                        "source_dataset",
                        args.source_dir.name,
                    ),
                    "source_path": str(text_path),
                    "source_title": source_manifest.get("title", text_path.stem),
                    "source_work_id": source_manifest.get("work_id"),
                    "source_shard_index": shard_index,
                    "source_shard_total": len(shards),
                    "example_count": len(shard),
                    "turn_end_marker": CHAT_TURN_END_MARKER,
                }
            )
            summary_rows.append(
                {
                    "output_name": output_name,
                    "source_file": text_path.name,
                    "source_shard_index": shard_index,
                    "source_shard_total": len(shards),
                    "example_count": len(shard),
                }
            )
            current_work_id += 1

    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "input_file_count": len(text_paths),
        "output_file_count": len(manifest_rows),
        "input_block_count": input_block_count,
        "output_block_count": output_block_count,
        "rejection_counts": dict(rejection_counter),
        "rejection_examples": rejection_examples,
        "deny_substrings": list(deny_substrings),
        "files": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"input_file_count={len(text_paths)}")
    print(f"output_file_count={len(manifest_rows)}")
    print(f"input_block_count={input_block_count}")
    print(f"output_block_count={output_block_count}")
    print(f"rejection_counts={json.dumps(dict(rejection_counter), ensure_ascii=False)}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
