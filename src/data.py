from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from config import DataConfig


HEADER_GUIDE_RE = re.compile(
    r"\A.*?^-{20,}\n.*?テキスト中に現れる記号について.*?^-{20,}\n+",
    flags=re.DOTALL | re.MULTILINE,
)
FOOTER_START_RE = re.compile(
    r"(?m)^(底本：|青空文庫作成ファイル：)"
)
AOZORA_NOTE_RE = re.compile(r"［＃.*?］")
RUBY_RE = re.compile(r"《.*?》")
RUBY_PIPE_RE = re.compile(r"｜")
MULTI_BLANK_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class WorkText:
    work_id: str
    title: str
    path: Path
    original_text: str
    cleaned_text: str

    @property
    def byte_length(self) -> int:
        return len(self.cleaned_text.encode("utf-8"))


@dataclass(slots=True)
class SplitSummary:
    work_count: int
    byte_count: int


class ByteDataset:
    vocab_size = 256

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self.generator = torch.Generator().manual_seed(config.seed)
        self._manifest_by_id = load_manifest(config.manifest_path)
        loaded_works = load_work_texts(
            data_dir=config.data_dir,
            manifest_by_id=self._manifest_by_id,
            limit=config.limit,
        )
        self.works = shuffle_works(loaded_works, config.seed)
        self.train_works, self.valid_works = split_works(self.works, config.train_split)
        self.train_data = encode_works(self.train_works)
        self.valid_data = encode_works(self.valid_works)
        ensure_split_size("train", self.train_data, config.context_length, config.min_bytes_per_split)
        ensure_split_size("valid", self.valid_data, config.context_length, config.min_bytes_per_split)

    def train_summary(self) -> SplitSummary:
        return summarize(self.train_works)

    def valid_summary(self) -> SplitSummary:
        return summarize(self.valid_works)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_data
        elif split == "valid":
            data = self.valid_data
        else:
            raise ValueError(f"Unknown split: {split}")

        max_start = len(data) - self.config.context_length - 1
        starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(self.config.batch_size,),
            generator=self.generator,
        )
        x = torch.stack(
            [data[start : start + self.config.context_length] for start in starts]
        )
        y = torch.stack(
            [data[start + 1 : start + self.config.context_length + 1] for start in starts]
        )
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and batch byte-level training data for the Dazai corpus."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DataConfig().data_dir,
        help="Directory containing UTF-8 Aozora text files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DataConfig().manifest_path,
        help="Optional manifest produced by the downloader.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only load the first N works.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=DataConfig().train_split,
        help="Fraction of works assigned to train.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DataConfig().context_length,
        help="Number of bytes per sequence.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DataConfig().batch_size,
        help="Number of sequences per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DataConfig().seed,
        help="Random seed used for batch sampling.",
    )
    return parser.parse_args()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_header(text: str) -> str:
    return HEADER_GUIDE_RE.sub("", text, count=1)


def strip_footer(text: str) -> str:
    match = FOOTER_START_RE.search(text)
    if match is None:
        return text
    return text[: match.start()]


def clean_aozora_text(text: str) -> str:
    text = normalize_newlines(text)
    text = strip_header(text)
    text = strip_footer(text)
    text = AOZORA_NOTE_RE.sub("", text)
    text = RUBY_RE.sub("", text)
    text = RUBY_PIPE_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def extract_work_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not infer work_id from path: {path}")
    return str(int(match.group(1)))


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}

    manifest_by_id: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            manifest_by_id[str(int(record["work_id"]))] = record
    return manifest_by_id


def load_work_texts(
    data_dir: Path,
    manifest_by_id: dict[str, dict[str, str]],
    limit: int | None = None,
) -> list[WorkText]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    paths = sorted(data_dir.glob("*.txt"))
    if limit is not None:
        paths = paths[:limit]

    works: list[WorkText] = []
    for path in paths:
        raw_text = path.read_text(encoding="utf-8")
        work_id = extract_work_id(path)
        manifest_record = manifest_by_id.get(work_id, {})
        title = manifest_record.get("title", path.stem)
        cleaned_text = clean_aozora_text(raw_text)
        if not cleaned_text:
            continue
        works.append(
            WorkText(
                work_id=work_id,
                title=title,
                path=path,
                original_text=raw_text,
                cleaned_text=cleaned_text,
            )
        )

    if not works:
        raise ValueError(f"No usable text files found in {data_dir}")

    return works


def split_works(works: list[WorkText], train_split: float) -> tuple[list[WorkText], list[WorkText]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
    if len(works) < 2:
        raise ValueError("Need at least 2 works to create train/valid splits")

    train_count = max(1, int(len(works) * train_split))
    train_count = min(train_count, len(works) - 1)
    return works[:train_count], works[train_count:]


def shuffle_works(works: list[WorkText], seed: int) -> list[WorkText]:
    shuffled = list(works)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def encode_works(works: list[WorkText]) -> torch.Tensor:
    separator = "\n\n"
    merged_text = separator.join(work.cleaned_text for work in works)
    byte_values = list(merged_text.encode("utf-8"))
    return torch.tensor(byte_values, dtype=torch.long)


def summarize(works: list[WorkText]) -> SplitSummary:
    return SplitSummary(
        work_count=len(works),
        byte_count=sum(work.byte_length for work in works),
    )


def ensure_split_size(
    split_name: str,
    data: torch.Tensor,
    context_length: int,
    min_bytes_per_split: int,
) -> None:
    required = max(context_length + 1, min_bytes_per_split)
    if len(data) < required:
        raise ValueError(
            f"{split_name} split is too small: {len(data)} bytes, need at least {required}"
        )


def build_config_from_args(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        train_split=args.train_split,
        context_length=args.context_length,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
    )


def preview_tensor(tensor: torch.Tensor, length: int = 24) -> str:
    values = tensor[0, :length].tolist()
    return " ".join(f"{value:03d}" for value in values)


def main() -> int:
    args = parse_args()
    config = build_config_from_args(args)
    dataset = ByteDataset(config)
    train_x, train_y = dataset.get_batch("train")
    valid_x, valid_y = dataset.get_batch("valid")

    print(f"loaded_works={len(dataset.works)}")
    print(
        f"train works={dataset.train_summary().work_count} bytes={dataset.train_summary().byte_count}"
    )
    print(
        f"valid works={dataset.valid_summary().work_count} bytes={dataset.valid_summary().byte_count}"
    )
    print(f"train_batch_x_shape={tuple(train_x.shape)}")
    print(f"train_batch_y_shape={tuple(train_y.shape)}")
    print(f"valid_batch_x_shape={tuple(valid_x.shape)}")
    print(f"valid_batch_y_shape={tuple(valid_y.shape)}")
    print(f"train_batch_x_preview={preview_tensor(train_x)}")
    print(f"train_batch_y_preview={preview_tensor(train_y)}")
    print(f"first_train_title={dataset.train_works[0].title}")
    print(f"first_valid_title={dataset.valid_works[0].title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
