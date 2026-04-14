from __future__ import annotations

import argparse
import sys
import textwrap
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
DEFAULT_CHAT_DOWNLOAD_URL = (
    "https://github.com/natsu0529/original_LLM/releases/download/v0.1.0/best.pt"
)


def download_checkpoint(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"Downloading checkpoint from {url} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"Saved to {dest}")


def resolve_checkpoint(
    args: argparse.Namespace,
    *,
    default_download_url: str | None = None,
) -> Path:
    if args.checkpoint is not None:
        path = Path(args.checkpoint)
        if not path.exists():
            print(f"Error: checkpoint not found: {path}", file=sys.stderr)
            raise SystemExit(1)
        return path

    cached = CACHE_DIR / DEFAULT_CHECKPOINT_NAME
    if cached.exists():
        return cached

    url = args.download_url or default_download_url
    if url is None:
        print(
            "Error: no checkpoint found.\n"
            "Specify --checkpoint <path> or --download-url <url>",
            file=sys.stderr,
        )
        raise SystemExit(1)

    download_checkpoint(url, cached)
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
    epilog: str | None = None,
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
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="One-shot input prompt. Omit this in interactive chat mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help="Local checkpoint path. If omitted, use the cached or auto-detected model.",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=None,
        help="Checkpoint URL to download on first run when no local cache exists.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Inference device. 'auto' prefers MPS, then CUDA, then CPU.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_max_new_tokens,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_temperature,
        help="Sampling temperature. Lower is steadier.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=default_repetition_penalty,
        help="Penalty applied to recently used tokens. 1.0 disables it.",
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=128,
        help="How many recent tokens are considered by repetition penalty.",
    )
    parser.add_argument(
        "--stop-at-period",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after sentence-ending punctuation once enough text was produced.",
    )
    parser.add_argument(
        "--stop-at-blank-line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when a blank line is generated.",
    )
    parser.add_argument(
        "--min-new-chars-before-stop",
        type=int,
        default=DEFAULT_MIN_NEW_CHARS_BEFORE_STOP,
        help="Minimum visible characters before stop conditions can trigger.",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=default_interactive,
        help="Start a chat REPL instead of one-shot generation.",
    )
    parser.add_argument(
        "--carry-context",
        action=argparse.BooleanOptionalAction,
        default=default_carry_context,
        help="Keep session history within the current interactive run.",
    )
    parser.add_argument(
        "--user-label",
        type=str,
        default=default_user_label,
        help="Role label for user turns, for example '私'.",
    )
    parser.add_argument(
        "--reply-label",
        type=str,
        default=default_reply_label,
        help="Role label for model replies, for example '相手'.",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="Print checkpoint metadata before generation.",
    )
    return parser.parse_args()


def run_cli(
    *,
    prog: str,
    description: str,
    epilog: str | None = None,
    default_checkpoint: str | None = None,
    default_download_url: str | None = None,
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
        epilog=epilog,
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
    checkpoint_path = resolve_checkpoint(args, default_download_url=default_download_url)
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
        epilog=textwrap.dedent(
            """
            Examples:
              original-llm "私は"
              original-llm --interactive --checkpoint checkpoints/dazai-long/best.pt
              original-llm --show-meta --checkpoint checkpoints/dazai-friend-peers-512x8-v1/best.pt
            """
        ).strip(),
    )


def main_chat() -> int:
    chat_checkpoint = preferred_chat_checkpoint()
    return run_cli(
        prog="dazai-chat",
        description=(
            "Chat with the current Dazai-style checkpoint using conversation defaults.\n"
            "If no local checkpoint is found, the first run downloads one into "
            "~/.cache/original-llm/best.pt ."
        ),
        epilog=textwrap.dedent(
            f"""
            Quick Start:
              dazai-chat
              dazai-chat --show-meta
              dazai-chat --no-carry-context

            One-shot example:
              dazai-chat $'私: 酒飲もうぜ\\n相手: '

            Interactive commands:
              :help   show in-chat help
              :reset  clear session history
              :quit   exit

            Defaults:
              interactive=True
              carry-context=True
              user-label=私
              reply-label=相手
              temperature={DEFAULT_CHAT_TEMPERATURE}
              top-k={DEFAULT_CHAT_TOP_K}
              repetition-penalty={DEFAULT_CHAT_REPETITION_PENALTY}
              max-new-tokens={DEFAULT_CHAT_MAX_NEW_TOKENS}
            """
        ).strip(),
        default_checkpoint=str(chat_checkpoint) if chat_checkpoint is not None else None,
        default_download_url=DEFAULT_CHAT_DOWNLOAD_URL,
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
