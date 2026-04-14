from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class DataConfig:
    data_dir: Path = REPO_ROOT / "data" / "raw" / "aozora" / "dazai" / "txt"
    manifest_path: Path = REPO_ROOT / "data" / "raw" / "aozora" / "dazai" / "manifest.jsonl"
    tokenizer_type: str = "char"
    reply_loss_label: str | None = None
    train_split: float = 0.9
    context_length: int = 256
    batch_size: int = 8
    seed: int = 42
    limit: int | None = None
    min_tokens_per_split: int = 257


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 256
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 4
    ffn_hidden: int = 1024
    context_length: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.ffn_hidden <= 0:
            raise ValueError(f"ffn_hidden must be positive, got {self.ffn_hidden}")
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(slots=True)
class RunConfig:
    run_name: str = "dazai-debug"
    output_root: Path = REPO_ROOT

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_root / "checkpoints" / self.run_name

    @property
    def sample_dir(self) -> Path:
        return self.output_root / "samples" / self.run_name

    @property
    def log_dir(self) -> Path:
        return self.output_root / "logs" / self.run_name
