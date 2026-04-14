from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="original-llm",
        description="Generate text with a small LLM trained from scratch.",
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Input prompt text")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
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


def main() -> int:
    args = parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
