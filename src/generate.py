from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import REPO_ROOT, ModelConfig
from data import Tokenizer, tokenizer_from_state_dict
from model import DecoderOnlyTransformer, count_parameters


DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "dazai-long" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path, for example checkpoints/dazai-long/best.pt",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="太宰治")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--carry-context", action="store_true")
    parser.add_argument("--show-meta", action="store_true")
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


def load_generator(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DecoderOnlyTransformer, Tokenizer, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"])
    tokenizer = tokenizer_from_state_dict(checkpoint.get("tokenizer_state"))
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
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

    return tokenizer.decode(idx[0].tolist())


def generated_suffix(prompt: str, generated: str) -> str:
    if generated.startswith(prompt):
        return generated[len(prompt) :]
    return generated


def print_meta(
    checkpoint_path: Path,
    checkpoint: dict,
    tokenizer: Tokenizer,
    model: DecoderOnlyTransformer,
    device: torch.device,
) -> None:
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"step={checkpoint.get('step')}")
    print(f"best_val_loss={checkpoint.get('best_val_loss')}")
    print(f"parameter_count={count_parameters(model)}")
    print(f"tokenizer_type={tokenizer.tokenizer_type}")
    print(f"vocab_size={model.config.vocab_size}")
    print(f"model_config={checkpoint.get('model_config')}")


def interactive_loop(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print("interactive mode")
    print(":quit or :exit で終了")
    print(":reset で文脈をリセット")
    print(":help でヘルプ")

    history = ""
    while True:
        try:
            user_input = input("> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        command = user_input.strip()
        if not command:
            continue
        if command in {":quit", ":exit"}:
            break
        if command == ":reset":
            history = ""
            print("(reset)")
            continue
        if command == ":help":
            print("input text to generate continuation")
            print(":reset -> clear history")
            print(":quit  -> exit")
            continue

        prompt = user_input
        if args.carry_context and history:
            prompt = f"{history}\n{user_input}"

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        suffix = generated_suffix(prompt, generated)
        print(suffix.strip() or generated.strip())

        if args.carry_context:
            history = generated


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    model, tokenizer, checkpoint = load_generator(args.checkpoint, device)

    if args.show_meta:
        print_meta(args.checkpoint, checkpoint, tokenizer, model, device)

    if args.interactive:
        interactive_loop(model, tokenizer, args, device)
        return 0

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
