from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class DataConfig:
    data_dir: Path = REPO_ROOT / "data" / "raw" / "aozora" / "dazai" / "txt"
    manifest_path: Path = REPO_ROOT / "data" / "raw" / "aozora" / "dazai" / "manifest.jsonl"
    train_split: float = 0.9
    context_length: int = 256
    batch_size: int = 8
    seed: int = 42
    limit: int | None = None
    min_bytes_per_split: int = 257


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
