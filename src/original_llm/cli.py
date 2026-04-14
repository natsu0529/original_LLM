from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from original_llm.config import REPO_ROOT
from original_llm.generate import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_CHARS_BEFORE_STOP,
    choose_device,
    generate_text,
    interactive_loop,
    load_generator,
    set_seed,
    validate_args,
)

CACHE_DIR = Path.home() / ".cache" / "original-llm"
DEFAULT_CHECKPOINT_NAME = "best.pt"
DEFAULT_CHAT_MAX_NEW_TOKENS = 48
DEFAULT_CHAT_TEMPERATURE = 0.2
DEFAULT_CHAT_TOP_K = 8
DEFAULT_CHAT_REPETITION_PENALTY = 1.1


def download_checkpoint(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"Downloading checkpoint from {url} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"Saved to {dest}")


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        path = Path(args.checkpoint)
        if not path.exists():
            print(f"Error: checkpoint not found: {path}", file=sys.stderr)
            raise SystemExit(1)
        return path

    cached = CACHE_DIR / DEFAULT_CHECKPOINT_NAME
    if cached.exists():
        return cached

    if args.download_url is None:
        print(
            "Error: no checkpoint found.\n"
            "Specify --checkpoint <path> or --download-url <url>",
            file=sys.stderr,
        )
        raise SystemExit(1)

    download_checkpoint(args.download_url, cached)
    return cached


def preferred_chat_checkpoint() -> Path | None:
    checkpoint_root = REPO_ROOT / "checkpoints"
    patterns = (
        "dazai-friend-peers-512x8*/best.pt",
        "dazai-friend-reply*/best.pt",
        "dazai-friend-auto*/best.pt",
        "dazai-friend-simple*/best.pt",
    )
    for pattern in patterns:
        candidates = sorted(
            checkpoint_root.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def parse_args(
    *,
    prog: str,
    description: str,
    default_checkpoint: str | None = None,
    default_interactive: bool = False,
    default_carry_context: bool = False,
    default_user_label: str | None = None,
    default_reply_label: str | None = None,
    default_temperature: float = 0.8,
    default_top_k: int = 40,
    default_repetition_penalty: float = 1.0,
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Input prompt text")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--temperature", type=float, default=default_temperature)
    parser.add_argument("--top-k", type=int, default=default_top_k)
    parser.add_argument("--repetition-penalty", type=float, default=default_repetition_penalty)
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
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=default_interactive,
    )
    parser.add_argument(
        "--carry-context",
        action=argparse.BooleanOptionalAction,
        default=default_carry_context,
    )
    parser.add_argument("--user-label", type=str, default=default_user_label)
    parser.add_argument("--reply-label", type=str, default=default_reply_label)
    parser.add_argument("--show-meta", action="store_true")
    return parser.parse_args()


def run_cli(
    *,
    prog: str,
    description: str,
    default_checkpoint: str | None = None,
    default_interactive: bool = False,
    default_carry_context: bool = False,
    default_user_label: str | None = None,
    default_reply_label: str | None = None,
    default_temperature: float = 0.8,
    default_top_k: int = 40,
    default_repetition_penalty: float = 1.0,
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> int:
    args = parse_args(
        prog=prog,
        description=description,
        default_checkpoint=default_checkpoint,
        default_interactive=default_interactive,
        default_carry_context=default_carry_context,
        default_user_label=default_user_label,
        default_reply_label=default_reply_label,
        default_temperature=default_temperature,
        default_top_k=default_top_k,
        default_repetition_penalty=default_repetition_penalty,
        default_max_new_tokens=default_max_new_tokens,
    )
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_path = resolve_checkpoint(args)
    model, tokenizer, checkpoint = load_generator(checkpoint_path, device)

    if args.show_meta:
        from original_llm.generate import print_meta
        print_meta(checkpoint_path, checkpoint, tokenizer, model, device)

    if args.interactive:
        interactive_loop(model, tokenizer, args, device)
        return 0

    prompt = args.prompt or "むかしむかし"
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
    print(generated)
    return 0


def main() -> int:
    return run_cli(
        prog="original-llm",
        description="Generate text with a small LLM trained from scratch.",
    )


def main_chat() -> int:
    chat_checkpoint = preferred_chat_checkpoint()
    return run_cli(
        prog="dazai-chat",
        description="Chat with the current Dazai-style checkpoint using conversation defaults.",
        default_checkpoint=str(chat_checkpoint) if chat_checkpoint is not None else None,
        default_interactive=True,
        default_carry_context=True,
        default_user_label="私",
        default_reply_label="相手",
        default_temperature=DEFAULT_CHAT_TEMPERATURE,
        default_top_k=DEFAULT_CHAT_TOP_K,
        default_repetition_penalty=DEFAULT_CHAT_REPETITION_PENALTY,
        default_max_new_tokens=DEFAULT_CHAT_MAX_NEW_TOKENS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
