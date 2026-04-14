#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "aozora" / "dazai" / "manifest.jsonl"
DEFAULT_TEXT_DIR = REPO_ROOT / "data" / "raw" / "aozora" / "dazai" / "txt"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "dialogue_candidates.jsonl"
)
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "data" / "intermediate" / "aozora" / "dazai" / "dialogue_summary.json"
)
HEADER_DELIMITER_RE = re.compile(r"(?m)^-+\s*$")
RUBY_RE = re.compile(r"《[^》]*》")
NOTE_RE = re.compile(r"［＃[^］]*］")
FOOTER_MARKERS = ("\n底本：", "\n青空文庫作成ファイル：")
SPEECH_VERB_HINT_RE = re.compile(
    r"(?:と(?:言|い|答|叫|笑|問|聞|教|囁|つぶや|呟|呼|どな|諭|命じ)|に向い|に向かい)"
)


@dataclass(slots=True)
class WorkRecord:
    work_id: str
    title: str
    orthography: str | None
    card_url: str | None
    text_path: Path


@dataclass(slots=True)
class QuoteSpan:
    start: int
    end: int
    text: str


@dataclass(slots=True)
class DialogueCandidate:
    candidate_id: str
    work_id: str
    title: str
    orthography: str | None
    card_url: str | None
    source_text_path: str
    quote_index: int
    cluster_id: str
    cluster_size: int
    heuristic_score: int
    heuristic_flags: list[str]
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    quote_text: str
    quote_text_normalized: str
    prev_context: str
    next_context: str
    line_window_text: str
    gap_from_prev: str | None
    gap_to_next: str | None
    prev_quote_text: str | None
    next_quote_text: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dialogue candidates wrapped in Japanese quote marks from Aozora Bunko texts."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Aozora manifest JSONL created by download_aozora_dazai.py.",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=DEFAULT_TEXT_DIR,
        help="Directory containing extracted Aozora text files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for dialogue candidates.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output JSON path for aggregate statistics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N works after manifest order.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=120,
        help="Characters to keep on each side of the quote for local context.",
    )
    parser.add_argument(
        "--line-window",
        type=int,
        default=2,
        help="How many lines before/after to keep in the line-level excerpt.",
    )
    parser.add_argument(
        "--max-gap-chars",
        type=int,
        default=80,
        help="Maximum non-whitespace gap between quotes to group them into the same cluster.",
    )
    parser.add_argument(
        "--min-quote-chars",
        type=int,
        default=1,
        help="Drop quotes shorter than this after normalization.",
    )
    parser.add_argument(
        "--keep-ruby",
        action="store_true",
        help="Keep Aozora ruby markers such as 《...》 and ｜ in the cleaned text.",
    )
    parser.add_argument(
        "--keep-notes",
        action="store_true",
        help="Keep Aozora input notes such as ［＃...］ in the cleaned text.",
    )
    parser.add_argument(
        "--print-sample-count",
        type=int,
        default=3,
        help="Print the first N extracted candidates after writing output.",
    )
    return parser.parse_args()


def load_manifest(path: Path, text_dir: Path, limit: int | None) -> list[WorkRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")

    records: list[WorkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            text_path_value = payload.get("text_path")
            if not text_path_value:
                continue
            text_path = Path(text_path_value)
            if not text_path.exists():
                candidate = text_dir / text_path.name
                if not candidate.exists():
                    continue
                text_path = candidate

            records.append(
                WorkRecord(
                    work_id=str(payload.get("work_id", "")),
                    title=str(payload.get("title", "")),
                    orthography=payload.get("orthography"),
                    card_url=payload.get("card_url"),
                    text_path=text_path,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def strip_aozora_boilerplate(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    header_delimiters = list(HEADER_DELIMITER_RE.finditer(normalized))
    if len(header_delimiters) >= 2:
        normalized = normalized[header_delimiters[1].end() :]

    footer_positions = [
        normalized.find(marker) for marker in FOOTER_MARKERS if normalized.find(marker) != -1
    ]
    if footer_positions:
        normalized = normalized[: min(footer_positions)]

    return normalized.strip()


def clean_aozora_text(text: str, keep_ruby: bool, keep_notes: bool) -> str:
    cleaned = strip_aozora_boilerplate(text)
    if not keep_ruby:
        cleaned = cleaned.replace("｜", "")
        cleaned = RUBY_RE.sub("", cleaned)
    if not keep_notes:
        cleaned = NOTE_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_inline_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def trim_text(text: str, max_chars: int = 180) -> str:
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def extract_quote_spans(text: str) -> list[QuoteSpan]:
    spans: list[QuoteSpan] = []
    stack: list[int] = []
    current_start: int | None = None

    for index, char in enumerate(text):
        if char == "「":
            if not stack:
                current_start = index
            stack.append(index)
            continue

        if char == "」" and stack:
            stack.pop()
            if not stack and current_start is not None:
                spans.append(
                    QuoteSpan(
                        start=current_start,
                        end=index + 1,
                        text=text[current_start + 1 : index],
                    )
                )
                current_start = None

    return spans


def build_line_starts(text: str) -> list[int]:
    line_starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            line_starts.append(index + 1)
    return line_starts


def line_number_for_index(line_starts: list[int], index: int) -> int:
    return bisect_right(line_starts, index)


def extract_line_window(
    lines: list[str],
    line_start: int,
    line_end: int,
    line_window: int,
) -> str:
    start_index = max(0, line_start - 1 - line_window)
    end_index = min(len(lines), line_end + line_window)
    return "\n".join(lines[start_index:end_index]).strip()


def build_clusters(text: str, spans: list[QuoteSpan], max_gap_chars: int) -> tuple[list[int], Counter[int]]:
    cluster_numbers: list[int] = []
    current_cluster = 0

    for index, span in enumerate(spans):
        if index == 0:
            current_cluster = 1
            cluster_numbers.append(current_cluster)
            continue

        previous = spans[index - 1]
        gap_text = text[previous.end : span.start]
        normalized_gap = normalize_inline_text(gap_text)
        same_cluster = len(normalized_gap) <= max_gap_chars and "\n\n" not in gap_text
        if not same_cluster:
            current_cluster += 1
        cluster_numbers.append(current_cluster)

    counts = Counter(cluster_numbers)
    return cluster_numbers, counts


def dialogue_heuristics(
    quote_text: str,
    prev_context: str,
    next_context: str,
    cluster_size: int,
) -> tuple[int, list[str]]:
    flags: list[str] = []
    score = 0
    normalized_quote = normalize_inline_text(quote_text)
    nearby_context = prev_context[-60:] + next_context[:60]

    if cluster_size > 1:
        flags.append("clustered_quotes")
        score += 1
    if any(char in normalized_quote for char in ("。", "！", "？")):
        flags.append("sentence_punctuation")
        score += 1
    if SPEECH_VERB_HINT_RE.search(nearby_context):
        flags.append("speech_verb_nearby")
        score += 2
    if len(normalized_quote) <= 3:
        flags.append("very_short")
        score -= 1
    if "\n" in quote_text:
        flags.append("multiline_quote")
        score += 1

    return score, flags


def candidate_from_span(
    work: WorkRecord,
    text: str,
    lines: list[str],
    line_starts: list[int],
    spans: list[QuoteSpan],
    cluster_numbers: list[int],
    cluster_sizes: Counter[int],
    span_index: int,
    context_chars: int,
    line_window: int,
) -> DialogueCandidate:
    span = spans[span_index]
    previous_span = spans[span_index - 1] if span_index > 0 else None
    next_span = spans[span_index + 1] if span_index + 1 < len(spans) else None

    char_start = span.start
    char_end = span.end
    line_start = line_number_for_index(line_starts, char_start)
    line_end = line_number_for_index(line_starts, max(char_start, char_end - 1))

    prev_context = trim_text(text[max(0, char_start - context_chars) : char_start])
    next_context = trim_text(text[char_end : min(len(text), char_end + context_chars)])
    gap_from_prev = (
        trim_text(text[previous_span.end : char_start]) if previous_span is not None else None
    )
    gap_to_next = trim_text(text[char_end : next_span.start]) if next_span is not None else None

    cluster_number = cluster_numbers[span_index]
    cluster_id = f"{work.work_id}-cluster-{cluster_number:04d}"
    heuristic_score, heuristic_flags = dialogue_heuristics(
        quote_text=span.text,
        prev_context=prev_context,
        next_context=next_context,
        cluster_size=cluster_sizes[cluster_number],
    )

    return DialogueCandidate(
        candidate_id=f"{work.work_id}-quote-{span_index + 1:05d}",
        work_id=work.work_id,
        title=work.title,
        orthography=work.orthography,
        card_url=work.card_url,
        source_text_path=str(work.text_path),
        quote_index=span_index + 1,
        cluster_id=cluster_id,
        cluster_size=cluster_sizes[cluster_number],
        heuristic_score=heuristic_score,
        heuristic_flags=heuristic_flags,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_end,
        quote_text=span.text,
        quote_text_normalized=normalize_inline_text(span.text),
        prev_context=prev_context,
        next_context=trim_text(next_context),
        line_window_text=extract_line_window(lines, line_start, line_end, line_window),
        gap_from_prev=gap_from_prev,
        gap_to_next=gap_to_next,
        prev_quote_text=previous_span.text if previous_span is not None else None,
        next_quote_text=next_span.text if next_span is not None else None,
    )


def write_jsonl(path: Path, rows: list[DialogueCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    works = load_manifest(args.manifest_path, args.text_dir, args.limit)

    candidates: list[DialogueCandidate] = []
    quote_count_by_work: dict[str, int] = {}
    cluster_count_by_work: dict[str, int] = {}

    for work in works:
        raw_text = work.text_path.read_text(encoding="utf-8")
        cleaned_text = clean_aozora_text(
            raw_text,
            keep_ruby=args.keep_ruby,
            keep_notes=args.keep_notes,
        )
        spans = [
            span
            for span in extract_quote_spans(cleaned_text)
            if len(normalize_inline_text(span.text)) >= args.min_quote_chars
        ]
        if not spans:
            continue

        lines = cleaned_text.splitlines()
        line_starts = build_line_starts(cleaned_text)
        cluster_numbers, cluster_sizes = build_clusters(
            cleaned_text,
            spans,
            max_gap_chars=args.max_gap_chars,
        )
        quote_count_by_work[work.work_id] = len(spans)
        cluster_count_by_work[work.work_id] = len(cluster_sizes)

        for span_index in range(len(spans)):
            candidates.append(
                candidate_from_span(
                    work=work,
                    text=cleaned_text,
                    lines=lines,
                    line_starts=line_starts,
                    spans=spans,
                    cluster_numbers=cluster_numbers,
                    cluster_sizes=cluster_sizes,
                    span_index=span_index,
                    context_chars=args.context_chars,
                    line_window=args.line_window,
                )
            )

    write_jsonl(args.output_path, candidates)

    works_with_quotes = len(quote_count_by_work)
    cluster_total = sum(cluster_count_by_work.values())
    quote_lengths = [len(candidate.quote_text_normalized) for candidate in candidates]
    summary = {
        "manifest_path": str(args.manifest_path),
        "text_dir": str(args.text_dir),
        "output_path": str(args.output_path),
        "summary_path": str(args.summary_path),
        "work_count_processed": len(works),
        "works_with_quotes": works_with_quotes,
        "candidate_count": len(candidates),
        "cluster_count": cluster_total,
        "average_quote_length": mean(quote_lengths) if quote_lengths else 0.0,
        "top_works_by_quote_count": [
            {"work_id": work_id, "quote_count": count}
            for work_id, count in sorted(
                quote_count_by_work.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        ],
    }
    write_json(args.summary_path, summary)

    print(f"work_count_processed={len(works)}")
    print(f"works_with_quotes={works_with_quotes}")
    print(f"candidate_count={len(candidates)}")
    print(f"cluster_count={cluster_total}")
    print(f"output_path={args.output_path}")
    print(f"summary_path={args.summary_path}")

    sample_count = min(args.print_sample_count, len(candidates))
    for index in range(sample_count):
        candidate = candidates[index]
        print()
        print(f"[sample {index + 1}] {candidate.candidate_id} {candidate.title}")
        print(candidate.quote_text_normalized)
        if candidate.prev_context:
            print(f"prev={candidate.prev_context}")
        if candidate.next_context:
            print(f"next={candidate.next_context}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
