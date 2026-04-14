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
FOOTER_START_RE = re.compile(r"(?m)^(底本：|青空文庫作成ファイル：)")
AOZORA_NOTE_RE = re.compile(r"［＃.*?］")
RUBY_RE = re.compile(r"《.*?》")
RUBY_PIPE_RE = re.compile(r"｜")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
ROLE_LINE_RE = re.compile(r"^[^:\n]{1,32}:")
IGNORE_INDEX = -100


@dataclass(slots=True)
class WorkText:
    work_id: str
    title: str
    path: Path
    original_text: str
    cleaned_text: str


@dataclass(slots=True)
class SplitSummary:
    work_count: int
    token_count: int


class ByteTokenizer:
    tokenizer_type = "byte"
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="ignore")

    def state_dict(self) -> dict[str, object]:
        return {"tokenizer_type": self.tokenizer_type}

    @classmethod
    def build(cls, _: list[str]) -> ByteTokenizer:
        return cls()

    @classmethod
    def from_state_dict(cls, _: dict[str, object] | None = None) -> ByteTokenizer:
        return cls()


class CharTokenizer:
    tokenizer_type = "char"
    unk_token = "<unk>"
    unk_display = "\uFFFD"

    def __init__(self, id_to_token: list[str], unk_token: str = "<unk>") -> None:
        self.id_to_token = id_to_token
        self.token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        self.unk_token = unk_token
        self.unk_id = self.token_to_id[unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(char, self.unk_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            if 0 <= token_id < len(self.id_to_token):
                token = self.id_to_token[token_id]
                chars.append(self.unk_display if token == self.unk_token else token)
            else:
                chars.append(self.unk_display)
        return "".join(chars)

    def state_dict(self) -> dict[str, object]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "unk_token": self.unk_token,
            "id_to_token": self.id_to_token,
        }

    @classmethod
    def build(cls, texts: list[str]) -> CharTokenizer:
        unique_chars = sorted({char for text in texts for char in text})
        id_to_token = [cls.unk_token, *unique_chars]
        return cls(id_to_token=id_to_token, unk_token=cls.unk_token)

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> CharTokenizer:
        id_to_token = state.get("id_to_token")
        unk_token = state.get("unk_token", cls.unk_token)
        if not isinstance(id_to_token, list) or not id_to_token:
            raise ValueError("Invalid char tokenizer state")
        return cls(id_to_token=id_to_token, unk_token=str(unk_token))


Tokenizer = ByteTokenizer | CharTokenizer


class TokenDataset:
    def __init__(
        self,
        config: DataConfig,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = config
        self.generator = torch.Generator().manual_seed(config.seed)
        self._manifest_by_id = load_manifest(config.manifest_path)
        loaded_works = load_work_texts(
            data_dir=config.data_dir,
            manifest_by_id=self._manifest_by_id,
            limit=config.limit,
        )
        self.works = shuffle_works(loaded_works, config.seed)
        self.tokenizer = tokenizer or build_tokenizer(
            tokenizer_type=config.tokenizer_type,
            texts=[work.cleaned_text for work in self.works],
        )
        self.vocab_size = self.tokenizer.vocab_size
        self.train_works, self.valid_works = split_works(self.works, config.train_split)
        self.train_data, self.train_loss_mask = encode_works(
            self.train_works,
            self.tokenizer,
            reply_loss_label=config.reply_loss_label,
        )
        self.valid_data, self.valid_loss_mask = encode_works(
            self.valid_works,
            self.tokenizer,
            reply_loss_label=config.reply_loss_label,
        )
        self.train_valid_starts = compute_valid_starts(
            self.train_data,
            self.train_loss_mask,
            config.context_length,
        )
        self.valid_valid_starts = compute_valid_starts(
            self.valid_data,
            self.valid_loss_mask,
            config.context_length,
        )
        ensure_split_size(
            "train",
            self.train_data,
            config.context_length,
            config.min_tokens_per_split,
        )
        ensure_split_size(
            "valid",
            self.valid_data,
            config.context_length,
            config.min_tokens_per_split,
        )

    def train_summary(self) -> SplitSummary:
        return summarize(self.train_works, self.tokenizer)

    def valid_summary(self) -> SplitSummary:
        return summarize(self.valid_works, self.tokenizer)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_data
            loss_mask = self.train_loss_mask
            valid_starts = self.train_valid_starts
        elif split == "valid":
            data = self.valid_data
            loss_mask = self.valid_loss_mask
            valid_starts = self.valid_valid_starts
        else:
            raise ValueError(f"Unknown split: {split}")

        if valid_starts is None:
            max_start = len(data) - self.config.context_length - 1
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(self.config.batch_size,),
                generator=self.generator,
            )
        else:
            start_positions = torch.randint(
                low=0,
                high=valid_starts.numel(),
                size=(self.config.batch_size,),
                generator=self.generator,
            )
            starts = valid_starts[start_positions]
        x = torch.stack(
            [data[start : start + self.config.context_length] for start in starts]
        )
        y = torch.stack(
            [data[start + 1 : start + self.config.context_length + 1] for start in starts]
        )
        if loss_mask is not None:
            y_loss_mask = torch.stack(
                [
                    loss_mask[start + 1 : start + self.config.context_length + 1]
                    for start in starts
                ]
            )
            y = y.masked_fill(~y_loss_mask, IGNORE_INDEX)
        return x, y

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


def build_tokenizer(tokenizer_type: str, texts: list[str]) -> Tokenizer:
    if tokenizer_type == "char":
        return CharTokenizer.build(texts)
    if tokenizer_type == "byte":
        return ByteTokenizer.build(texts)
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")


def tokenizer_from_state_dict(state: dict[str, object] | None) -> Tokenizer:
    if state is None:
        return ByteTokenizer.from_state_dict()

    tokenizer_type = state.get("tokenizer_type", "byte")
    if tokenizer_type == "char":
        return CharTokenizer.from_state_dict(state)
    if tokenizer_type == "byte":
        return ByteTokenizer.from_state_dict(state)
    raise ValueError(f"Unsupported tokenizer_type in checkpoint: {tokenizer_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and batch tokenized training data for the Dazai corpus."
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
        "--tokenizer-type",
        choices=["char", "byte"],
        default=DataConfig().tokenizer_type,
        help="Tokenization mode used for training.",
    )
    parser.add_argument(
        "--reply-loss-label",
        type=str,
        default=DataConfig().reply_loss_label,
        help="Only tokens in lines starting with this label contribute to loss.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=DataConfig().train_split)
    parser.add_argument("--context-length", type=int, default=DataConfig().context_length)
    parser.add_argument("--batch-size", type=int, default=DataConfig().batch_size)
    parser.add_argument("--seed", type=int, default=DataConfig().seed)
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


def encode_works(
    works: list[WorkText],
    tokenizer: Tokenizer,
    reply_loss_label: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    separator = "\n\n"
    merged_token_ids: list[int] = []
    merged_loss_mask: list[bool] = []
    for index, work in enumerate(works):
        token_ids = tokenizer.encode(work.cleaned_text)
        merged_token_ids.extend(token_ids)
        if reply_loss_label is not None:
            merged_loss_mask.extend(
                build_reply_loss_mask(
                    work.cleaned_text,
                    tokenizer,
                    reply_loss_label,
                )
            )
        if index != len(works) - 1:
            separator_ids = tokenizer.encode(separator)
            merged_token_ids.extend(separator_ids)
            if reply_loss_label is not None:
                merged_loss_mask.extend([False] * len(separator_ids))

    data = torch.tensor(merged_token_ids, dtype=torch.long)
    if reply_loss_label is None:
        return data, None
    return data, torch.tensor(merged_loss_mask, dtype=torch.bool)


def build_reply_loss_mask(
    text: str,
    tokenizer: Tokenizer,
    reply_loss_label: str,
) -> list[bool]:
    reply_prefix = f"{reply_loss_label}:"
    token_ids = tokenizer.encode(text)
    mask: list[bool] = []
    reply_active = False

    for line in text.splitlines(keepends=True):
        newline_count = len(line) - len(line.rstrip("\n"))
        line_body = line[:-newline_count] if newline_count > 0 else line

        if line_body.startswith(reply_prefix):
            prefix_text = reply_prefix
            remainder = line_body[len(reply_prefix) :]
            if remainder.startswith(" "):
                prefix_text += " "
                remainder = remainder[1:]
            mask.extend([False] * len(tokenizer.encode(prefix_text)))
            mask.extend([True] * len(tokenizer.encode(remainder)))
            if newline_count > 0:
                mask.extend([True] * len(tokenizer.encode("\n" * newline_count)))
            reply_active = True
            continue

        if line_body and ROLE_LINE_RE.match(line_body):
            mask.extend([False] * len(tokenizer.encode(line)))
            reply_active = False
            continue

        mask.extend([reply_active] * len(tokenizer.encode(line)))

    if len(mask) != len(token_ids):
        raise ValueError(
            "Reply loss mask length mismatch: "
            f"tokens={len(token_ids)} mask={len(mask)}"
        )
    return mask


def compute_valid_starts(
    data: torch.Tensor,
    loss_mask: torch.Tensor | None,
    context_length: int,
) -> torch.Tensor | None:
    if loss_mask is None:
        return None

    max_start = len(data) - context_length - 1
    if max_start < 0:
        return None

    target_mask = loss_mask[1:]
    window_sums = target_mask.unfold(0, context_length, 1).sum(dim=1)
    valid_starts = torch.nonzero(window_sums > 0, as_tuple=False).flatten()
    if valid_starts.numel() == 0:
        raise ValueError("No valid batch windows contain reply-loss tokens")
    return valid_starts


def summarize(works: list[WorkText], tokenizer: Tokenizer) -> SplitSummary:
    return SplitSummary(
        work_count=len(works),
        token_count=sum(len(tokenizer.encode(work.cleaned_text)) for work in works),
    )


def ensure_split_size(
    split_name: str,
    data: torch.Tensor,
    context_length: int,
    min_tokens_per_split: int,
) -> None:
    required = max(context_length + 1, min_tokens_per_split)
    if len(data) < required:
        raise ValueError(
            f"{split_name} split is too small: {len(data)} tokens, need at least {required}"
        )


def build_config_from_args(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        tokenizer_type=args.tokenizer_type,
        reply_loss_label=args.reply_loss_label,
        train_split=args.train_split,
        context_length=args.context_length,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
    )


def preview_tensor(tensor: torch.Tensor, length: int = 24) -> str:
    values = tensor[0, :length].tolist()
    return " ".join(str(value) for value in values)


def main() -> int:
    args = parse_args()
    config = build_config_from_args(args)
    dataset = TokenDataset(config)
    train_x, train_y = dataset.get_batch("train")
    valid_x, valid_y = dataset.get_batch("valid")
    preview_text = dataset.decode_tokens(train_x[0, : min(32, train_x.size(1))].tolist())

    print(f"loaded_works={len(dataset.works)}")
    print(f"tokenizer_type={dataset.tokenizer.tokenizer_type}")
    print(f"vocab_size={dataset.vocab_size}")
    print(f"reply_loss_label={config.reply_loss_label}")
    print(
        f"train works={dataset.train_summary().work_count} tokens={dataset.train_summary().token_count}"
    )
    print(
        f"valid works={dataset.valid_summary().work_count} tokens={dataset.valid_summary().token_count}"
    )
    print(f"train_batch_x_shape={tuple(train_x.shape)}")
    print(f"train_batch_y_shape={tuple(train_y.shape)}")
    print(f"valid_batch_x_shape={tuple(valid_x.shape)}")
    print(f"valid_batch_y_shape={tuple(valid_y.shape)}")
    print(f"train_batch_x_preview={preview_tensor(train_x)}")
    print(f"train_batch_y_preview={preview_tensor(train_y)}")
    print(f"train_batch_text_preview={preview_text}")
    print(f"first_train_title={dataset.train_works[0].title}")
    print(f"first_valid_title={dataset.valid_works[0].title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
