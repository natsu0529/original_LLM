from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from config import DataConfig, ModelConfig, RunConfig
from data import TokenDataset, Tokenizer, tokenizer_from_state_dict
from model import DecoderOnlyTransformer, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal decoder-only model.")

    parser.add_argument("--run-name", type=str, default="dazai-debug")
    parser.add_argument("--out-dir", type=Path, default=RunConfig().output_root)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-best-val-loss", action="store_true")

    parser.add_argument("--data-dir", type=Path, default=DataConfig().data_dir)
    parser.add_argument("--manifest-path", type=Path, default=DataConfig().manifest_path)
    parser.add_argument(
        "--tokenizer-type",
        choices=["char", "byte"],
        default=DataConfig().tokenizer_type,
    )
    parser.add_argument("--reply-loss-label", type=str, default=DataConfig().reply_loss_label)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=DataConfig().train_split)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--sample-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", type=str, default="太宰治")

    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_config(args: argparse.Namespace) -> DataConfig:
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


def build_model_config(args: argparse.Namespace, dataset_vocab_size: int) -> ModelConfig:
    if args.vocab_size is not None and args.vocab_size != dataset_vocab_size:
        raise ValueError(
            f"vocab_size mismatch: args={args.vocab_size}, dataset={dataset_vocab_size}"
        )
    return ModelConfig(
        vocab_size=dataset_vocab_size,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        ffn_hidden=args.ffn_hidden,
        context_length=args.context_length,
        dropout=args.dropout,
    )


def load_checkpoint_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(item) for item in value)
    return value


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def apply_optimizer_overrides(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    weight_decay: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["weight_decay"] = weight_decay


def checkpoint_payload(
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    step: int,
    best_val_loss: float | None,
    run_config: RunConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "step": step,
        "best_val_loss": best_val_loss,
        "run_name": run_config.run_name,
        "data_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(data_config).items()},
        "model_config": asdict(model_config),
        "args": serialize_args(args),
        "tokenizer_state": tokenizer.state_dict(),
        "model_state": to_cpu_tree(model.state_dict()),
        "optimizer_state": to_cpu_tree(optimizer.state_dict()),
    }


def save_checkpoint(
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    step: int,
    best_val_loss: float | None,
    run_config: RunConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    run_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        step=step,
        best_val_loss=best_val_loss,
        run_config=run_config,
        data_config=data_config,
        model_config=model_config,
        args=args,
    )

    latest_path = run_config.checkpoint_dir / "latest.pt"
    step_path = run_config.checkpoint_dir / f"step_{step:07d}.pt"
    torch.save(payload, latest_path)
    torch.save(payload, step_path)
    return latest_path, step_path


def save_best_checkpoint(
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    step: int,
    best_val_loss: float,
    run_config: RunConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    args: argparse.Namespace,
) -> Path:
    run_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        step=step,
        best_val_loss=best_val_loss,
        run_config=run_config,
        data_config=data_config,
        model_config=model_config,
        args=args,
    )
    best_path = run_config.checkpoint_dir / "best.pt"
    torch.save(payload, best_path)
    return best_path


def load_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    load_optimizer: bool = True,
) -> tuple[int, float | None]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to_device(optimizer, device)
    step = int(checkpoint["step"])
    best_val_loss = checkpoint.get("best_val_loss")
    return step, best_val_loss


@torch.no_grad()
def estimate_loss(
    model: DecoderOnlyTransformer,
    dataset: TokenDataset,
    device: torch.device,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split in ("train", "valid"):
        split_losses = []
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split)
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            assert loss is not None
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


@torch.no_grad()
def generate_sample(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = tokenizer.encode(" ")
    if not tokens:
        raise ValueError("Tokenizer failed to encode fallback prompt")

    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_length :]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k=k)
                threshold = values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf"),
                )
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    generated = tokenizer.decode(idx[0].tolist())
    model.train()
    return generated


def write_sample(path: Path, prompt: str, text: str, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"step={step}\nprompt={prompt}\n\n{text}\n"
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {args.max_steps}")
    if args.eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {args.eval_every}")
    if args.log_every <= 0:
        raise ValueError(f"log_every must be positive, got {args.log_every}")
    if args.save_every <= 0:
        raise ValueError(f"save_every must be positive, got {args.save_every}")
    if args.sample_every <= 0:
        raise ValueError(f"sample_every must be positive, got {args.sample_every}")
    if args.eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {args.eval_iters}")

    device = choose_device(args.device)
    set_seed(args.seed)

    if device.type == "mps":
        torch.set_float32_matmul_precision("high")

    run_config = RunConfig(run_name=args.run_name, output_root=args.out_dir)
    resume_checkpoint = load_checkpoint_metadata(args.resume)
    data_config = build_data_config(args)
    resume_tokenizer = (
        tokenizer_from_state_dict(resume_checkpoint.get("tokenizer_state"))
        if resume_checkpoint is not None
        else None
    )
    if resume_tokenizer is not None:
        data_config.tokenizer_type = resume_tokenizer.tokenizer_type

    dataset = TokenDataset(data_config, tokenizer=resume_tokenizer)
    if resume_checkpoint is not None:
        model_config = ModelConfig(**resume_checkpoint["model_config"])
        if model_config.vocab_size != dataset.vocab_size:
            raise ValueError(
                "resume checkpoint vocab_size does not match dataset tokenizer: "
                f"checkpoint={model_config.vocab_size}, dataset={dataset.vocab_size}"
            )
    else:
        model_config = build_model_config(args, dataset.vocab_size)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_meta = {
        "run_name": run_config.run_name,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "tokenizer_type": dataset.tokenizer.tokenizer_type,
        "vocab_size": dataset.vocab_size,
        "data_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(data_config).items()},
        "model_config": asdict(model_config),
        "args": serialize_args(args),
    }
    write_json(run_config.log_dir / "run_config.json", run_meta)
    write_json(run_config.log_dir / "tokenizer.json", dataset.tokenizer.state_dict())

    start_step = 1
    best_val_loss: float | None = None
    if args.resume is not None:
        resumed_step, best_val_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
            load_optimizer=not args.reset_optimizer,
        )
        apply_optimizer_overrides(
            optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if args.reset_best_val_loss:
            best_val_loss = None
        start_step = resumed_step + 1
        print(f"resumed_from={args.resume} step={resumed_step}")
        if args.reset_optimizer:
            print("optimizer_state=reset")
        if args.reset_best_val_loss:
            print("best_val_loss=reset")

    print(f"run_name={run_config.run_name}")
    print(f"device={device}")
    print(f"parameter_count={count_parameters(model)}")
    print(f"tokenizer_type={dataset.tokenizer.tokenizer_type}")
    print(f"vocab_size={dataset.vocab_size}")
    print(
        f"train works={dataset.train_summary().work_count} tokens={dataset.train_summary().token_count}"
    )
    print(
        f"valid works={dataset.valid_summary().work_count} tokens={dataset.valid_summary().token_count}"
    )
    print(f"checkpoints={run_config.checkpoint_dir}")

    last_log_time = time.time()

    for step in range(start_step, args.max_steps + 1):
        x, y = dataset.get_batch("train")
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == start_step or step % args.log_every == 0:
            now = time.time()
            dt = now - last_log_time
            last_log_time = now
            print(f"step={step} train_loss={loss.item():.6f} dt={dt:.2f}s")

        if step == start_step or step % args.eval_every == 0 or step == args.max_steps:
            losses = estimate_loss(model, dataset, device, args.eval_iters)
            val_loss = losses["valid"]
            is_best = best_val_loss is None or val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            metrics = {
                "step": step,
                "train_loss": losses["train"],
                "valid_loss": losses["valid"],
                "best_valid_loss": best_val_loss,
                "timestamp": time.time(),
            }
            append_jsonl(run_config.log_dir / "metrics.jsonl", metrics)
            print(
                f"eval step={step} train_loss={losses['train']:.6f} valid_loss={losses['valid']:.6f} best_valid_loss={best_val_loss:.6f}"
            )
            if is_best:
                best_path = save_best_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    tokenizer=dataset.tokenizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    run_config=run_config,
                    data_config=data_config,
                    model_config=model_config,
                    args=args,
                )
                print(f"checkpoint_best={best_path}")

        if step == start_step or step % args.sample_every == 0 or step == args.max_steps:
            sample_text = generate_sample(
                model=model,
                tokenizer=dataset.tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.sample_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            sample_path = run_config.sample_dir / f"step_{step:07d}.txt"
            write_sample(sample_path, args.prompt, sample_text, step)
            print(f"sample_written={sample_path}")

        if step == start_step or step % args.save_every == 0 or step == args.max_steps:
            latest_path, step_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                tokenizer=dataset.tokenizer,
                step=step,
                best_val_loss=best_val_loss,
                run_config=run_config,
                data_config=data_config,
                model_config=model_config,
                args=args,
            )
            print(f"checkpoint_latest={latest_path}")
            print(f"checkpoint_step={step_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
