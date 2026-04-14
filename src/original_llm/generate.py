from __future__ import annotations

import argparse
from pathlib import Path

import torch

from original_llm.config import ModelConfig
from original_llm.data import Tokenizer, tokenizer_from_state_dict
from original_llm.model import DecoderOnlyTransformer, count_parameters


DEFAULT_CHECKPOINT = Path("checkpoints") / "dazai-long" / "best.pt"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_MIN_NEW_CHARS_BEFORE_STOP = 24
DEFAULT_STOP_CHARS = ("。", "！", "？", "」")


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
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument(
        "--stop-at-period",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stop-at-blank-line",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--min-new-chars-before-stop",
        type=int,
        default=DEFAULT_MIN_NEW_CHARS_BEFORE_STOP,
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--carry-context", action="store_true")
    parser.add_argument("--user-label", type=str, default=None)
    parser.add_argument("--reply-label", type=str, default=None)
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


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")
    if args.top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {args.top_k}")
    if args.repetition_penalty < 1.0:
        raise ValueError(
            f"repetition_penalty must be >= 1.0, got {args.repetition_penalty}"
        )
    if args.repetition_window < 0:
        raise ValueError(
            f"repetition_window must be non-negative, got {args.repetition_window}"
        )
    if args.min_new_chars_before_stop < 0:
        raise ValueError(
            "min_new_chars_before_stop must be non-negative, "
            f"got {args.min_new_chars_before_stop}"
        )
    if (args.user_label is None) != (args.reply_label is None):
        raise ValueError("user_label and reply_label must be provided together")


def trim_text_to_context(
    text: str,
    tokenizer: Tokenizer,
    context_length: int,
) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= context_length:
        return text
    return tokenizer.decode(tokens[-context_length:]).lstrip()


def apply_repetition_penalty(
    next_token_logits: torch.Tensor,
    recent_token_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty <= 1.0 or not recent_token_ids:
        return next_token_logits

    adjusted_logits = next_token_logits.clone()
    for token_id in set(recent_token_ids):
        token_logits = adjusted_logits[:, token_id]
        adjusted_logits[:, token_id] = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )
    return adjusted_logits


def should_stop_early(
    generated_suffix_text: str,
    args: argparse.Namespace,
) -> bool:
    if args.stop_at_blank_line and "\n\n" in generated_suffix_text:
        return True

    if not args.stop_at_period:
        return False

    visible_text = generated_suffix_text.strip()
    if len(visible_text) < args.min_new_chars_before_stop:
        return False
    return visible_text.endswith(DEFAULT_STOP_CHARS)


def print_block(label: str, text: str) -> None:
    print(f"[{label}]")
    print(text.rstrip() or "(empty)")


def role_prefix(label: str) -> str:
    return f"{label}:"


def role_prompt(label: str) -> str:
    return f"{label}: "


def build_interactive_prompt(
    user_input: str,
    history: str,
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    context_length: int,
) -> str:
    if args.user_label is None or args.reply_label is None:
        prompt = user_input
        if args.carry_context and history:
            prompt = f"{history}\n{user_input}"
        return trim_text_to_context(prompt, tokenizer, context_length)

    turn_prompt = (
        f"{role_prompt(args.user_label)}{user_input}\n"
        f"{role_prompt(args.reply_label)}"
    )
    if args.carry_context and history:
        return trim_text_to_context(
            f"{history}\n{turn_prompt}",
            tokenizer,
            context_length,
        )
    return trim_text_to_context(turn_prompt, tokenizer, context_length)


@torch.no_grad()
def generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    stop_at_period: bool,
    stop_at_blank_line: bool,
    min_new_chars_before_stop: int,
    device: torch.device,
) -> str:
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = tokenizer.encode(" ")
    if not tokens:
        raise ValueError("Tokenizer failed to encode fallback prompt")

    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    stop_args = argparse.Namespace(
        stop_at_period=stop_at_period,
        stop_at_blank_line=stop_at_blank_line,
        min_new_chars_before_stop=min_new_chars_before_stop,
    )

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_length :]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        recent_tokens = (
            idx[0, -repetition_window:].tolist() if repetition_window > 0 else []
        )
        next_token_logits = apply_repetition_penalty(
            next_token_logits,
            recent_tokens,
            repetition_penalty,
        )

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
        suffix = generated_suffix(prompt, generated)
        if should_stop_early(suffix, stop_args):
            break

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
    if args.carry_context:
        print("carry-context: on")
    if args.user_label is not None and args.reply_label is not None:
        print(f"chat-format: {args.user_label}/{args.reply_label}")

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
            print(f"carry-context -> {'on' if args.carry_context else 'off'}")
            if args.user_label is not None and args.reply_label is not None:
                print(f"labels -> {args.user_label}/{args.reply_label}")
            continue

        prompt = build_interactive_prompt(
            user_input=user_input,
            history=history,
            args=args,
            tokenizer=tokenizer,
            context_length=model.config.context_length,
        )

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            stop_at_period=args.stop_at_period,
            stop_at_blank_line=args.stop_at_blank_line,
            min_new_chars_before_stop=args.min_new_chars_before_stop,
            device=device,
        )
        suffix = generated_suffix(prompt, generated)
        print()
        print_block("prompt", prompt)
        print()
        print_block("output", suffix or generated)
        print()

        if args.carry_context:
            history = trim_text_to_context(
                generated,
                tokenizer,
                model.config.context_length,
            )


def main() -> int:
    args = parse_args()
    validate_args(args)
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
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        stop_at_period=args.stop_at_period,
        stop_at_blank_line=args.stop_at_blank_line,
        min_new_chars_before_stop=args.min_new_chars_before_stop,
        device=device,
    )
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
