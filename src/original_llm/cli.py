from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
import tomllib

from original_llm.config import REPO_ROOT
from original_llm.generate import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_CHARS_BEFORE_STOP,
    chat_stop_sequences,
    choose_device,
    curated_short_reply,
    extract_chat_reply,
    extract_pending_chat_user_input,
    generate_text,
    generated_suffix,
    interactive_loop,
    load_generator,
    prepare_chat_user_input,
    prepend_chat_retrieval_examples,
    select_direct_chat_reply,
    set_seed,
    validate_args,
)

CACHE_DIR = Path.home() / ".cache" / "original-llm"
PACKAGE_NAME = "original-llm"
DEFAULT_CHECKPOINT_NAME = "best.pt"
UPDATE_CHECK_CACHE_NAME = "update-check.json"
UPDATE_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
UPDATE_CHECK_TIMEOUT_SECONDS = 1.0
DISABLE_UPDATE_CHECK_ENV = "ORIGINAL_LLM_DISABLE_UPDATE_CHECK"
DEFAULT_CHAT_MAX_NEW_TOKENS = 48
DEFAULT_CHAT_TEMPERATURE = 0.2
DEFAULT_CHAT_TOP_K = 8
DEFAULT_CHAT_REPETITION_PENALTY = 1.1
DEFAULT_CHAT_MAX_HISTORY_TURNS = 2
DEFAULT_CHAT_RETRIEVAL_EXAMPLES = 0
RELEASE_DOWNLOAD_BASE_URL = (
    "https://github.com/natsu0529/original_LLM/releases/download"
)
RELEASE_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(slots=True)
class UpdateCheckResult:
    installed_version: str
    latest_version: str | None
    cache: dict[str, object]

    @property
    def update_available(self) -> bool:
        if self.latest_version is None:
            return False
        return is_newer_version(self.latest_version, self.installed_version)


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
    cache_name: str = DEFAULT_CHECKPOINT_NAME,
) -> Path:
    if args.checkpoint is not None:
        path = Path(args.checkpoint)
        if not path.exists():
            print(f"Error: checkpoint not found: {path}", file=sys.stderr)
            raise SystemExit(1)
        return path

    cached = CACHE_DIR / cache_name
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
        "dazai-friend-topic-clean-spm-*/best.pt",
        "dazai-friend-clean-actions-dream-spm-*/best.pt",
        "dazai-friend-clean-actions-spm-*/best.pt",
        "dazai-friend-clean-spm-*/best.pt",
        "dazai-friend-support-animals-spm-*/best.pt",
        "dazai-friend-support-spm-*/best.pt",
        "dazai-friend-casual-spm-*/best.pt",
        "dazai-friend-real-persona-casual*/best.pt",
        "dazai-friend-real-persona*/best.pt",
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


def resolve_existing_dir(path_text: str | os.PathLike[str] | None) -> Path | None:
    if path_text is None:
        return None

    raw_path = Path(path_text).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(REPO_ROOT / raw_path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_retrieval_corpus_dir(
    args: argparse.Namespace,
    checkpoint: dict,
) -> str | None:
    explicit_dir = getattr(args, "retrieval_corpus_dir", None)
    resolved = resolve_existing_dir(explicit_dir)
    if resolved is not None:
        return str(resolved)

    checkpoint_args = checkpoint.get("args", {})
    checkpoint_reply_loss_label = checkpoint_args.get("reply_loss_label")
    checkpoint_data_dir = checkpoint_args.get("data_dir")
    if isinstance(checkpoint_data_dir, str) and checkpoint_data_dir.strip():
        resolved = resolve_existing_dir(checkpoint_data_dir)
        if resolved is not None and (
            (isinstance(checkpoint_reply_loss_label, str) and checkpoint_reply_loss_label.strip())
            or "chat_seed" in resolved.name
        ):
            return str(resolved)

    preferred_real_persona_dir = resolve_existing_dir(
        REPO_ROOT / "data" / "chat_seed_friend_casual_mix_v1"
    )
    if preferred_real_persona_dir is not None:
        return str(preferred_real_persona_dir)

    preferred_real_persona_dir = resolve_existing_dir(
        REPO_ROOT / "data" / "chat_seed_real_persona_casual_v1"
    )
    if preferred_real_persona_dir is not None:
        return str(preferred_real_persona_dir)

    preferred_real_persona_dir = resolve_existing_dir(
        REPO_ROOT / "data" / "chat_seed_real_persona_v1"
    )
    if preferred_real_persona_dir is not None:
        return str(preferred_real_persona_dir)

    preferred_refined_dir = resolve_existing_dir(
        REPO_ROOT / "data" / "chat_seed_refined_v1"
    )
    if preferred_refined_dir is not None:
        return str(preferred_refined_dir)

    preferred_simple_dir = resolve_existing_dir(REPO_ROOT / "data" / "chat_seed_simple")
    if preferred_simple_dir is not None:
        return str(preferred_simple_dir)

    if isinstance(checkpoint_data_dir, str) and checkpoint_data_dir.strip():
        resolved = resolve_existing_dir(checkpoint_data_dir)
        if resolved is not None:
            return str(resolved)
    return None


@lru_cache(maxsize=1)
def package_version_string() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        pyproject_path = REPO_ROOT / "pyproject.toml"
        try:
            with pyproject_path.open("rb") as handle:
                project = tomllib.load(handle).get("project", {})
        except (OSError, tomllib.TOMLDecodeError):
            return "0.0.0"
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
        return "0.0.0"


def normalize_release_version(version_text: str | None) -> str | None:
    if version_text is None:
        return None
    normalized = version_text.strip()
    if not RELEASE_VERSION_RE.fullmatch(normalized):
        return None
    return normalized


def checkpoint_cache_name(
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    *,
    version_text: str | None = None,
) -> str:
    normalized_version = normalize_release_version(version_text)
    if normalized_version is None:
        return checkpoint_name

    checkpoint_path = Path(checkpoint_name)
    suffix = "".join(checkpoint_path.suffixes)
    stem = (
        checkpoint_path.name[: -len(suffix)]
        if suffix
        else checkpoint_path.name
    )
    return f"{stem}-v{normalized_version}{suffix}"


def release_asset_download_url(
    version_text: str | None,
    *,
    asset_name: str = DEFAULT_CHECKPOINT_NAME,
) -> str | None:
    normalized_version = normalize_release_version(version_text)
    if normalized_version is None:
        return None
    return f"{RELEASE_DOWNLOAD_BASE_URL}/v{normalized_version}/{asset_name}"


def default_chat_download_url(version_text: str | None = None) -> str | None:
    return release_asset_download_url(
        version_text or package_version_string(),
        asset_name=DEFAULT_CHECKPOINT_NAME,
    )


def update_check_cache_path() -> Path:
    return CACHE_DIR / UPDATE_CHECK_CACHE_NAME


def version_parts(version_text: str) -> tuple[int, ...] | None:
    normalized = normalize_release_version(version_text)
    if normalized is None:
        return None
    parts = tuple(int(part) for part in normalized.split("."))
    trimmed = list(parts)
    while trimmed and trimmed[-1] == 0:
        trimmed.pop()
    return tuple(trimmed)


def is_newer_version(candidate: str, current: str) -> bool:
    candidate_parts = version_parts(candidate)
    current_parts = version_parts(current)
    if candidate_parts is None or current_parts is None:
        return candidate.strip() != current.strip()

    width = max(len(candidate_parts), len(current_parts))
    padded_candidate = candidate_parts + (0,) * (width - len(candidate_parts))
    padded_current = current_parts + (0,) * (width - len(current_parts))
    return padded_candidate > padded_current


def update_check_disabled() -> bool:
    value = os.getenv(DISABLE_UPDATE_CHECK_ENV)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def load_update_check_cache(
    cache_path: Path | None = None,
) -> dict[str, object]:
    path = cache_path or update_check_cache_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_update_check_cache(
    cache: dict[str, object],
    cache_path: Path | None = None,
) -> None:
    path = cache_path or update_check_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle)
        tmp.replace(path)
    except OSError:
        return


def cached_latest_version(
    cache: dict[str, object],
    *,
    now: float,
) -> str | None:
    checked_at = cache.get("checked_at")
    latest_version = cache.get("latest_version")
    if not isinstance(checked_at, (int, float)):
        return None
    if not isinstance(latest_version, str) or not latest_version.strip():
        return None
    if now - float(checked_at) > UPDATE_CHECK_INTERVAL_SECONDS:
        return None
    return latest_version.strip()


def fetch_latest_pypi_version(
    package_name: str = PACKAGE_NAME,
    *,
    timeout_seconds: float = UPDATE_CHECK_TIMEOUT_SECONDS,
) -> str | None:
    url = f"https://pypi.org/pypi/{package_name}/json"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    info = payload.get("info", {})
    version = info.get("version")
    if not isinstance(version, str) or not version.strip():
        return None
    return version.strip()


def check_for_updates(
    installed_version: str,
    *,
    cache_path: Path | None = None,
    now: float | None = None,
    force: bool = False,
    fetch_latest_version: Callable[[], str | None] | None = None,
) -> UpdateCheckResult:
    path = cache_path or update_check_cache_path()
    fetcher = fetch_latest_version or fetch_latest_pypi_version
    current_time = time.time() if now is None else now
    cache = load_update_check_cache(path)

    latest_version = None if force else cached_latest_version(cache, now=current_time)
    if latest_version is None:
        try:
            latest_version = fetcher()
        except Exception:
            cached_version = cache.get("latest_version")
            latest_version = (
                cached_version.strip()
                if isinstance(cached_version, str) and cached_version.strip()
                else None
            )
        else:
            cache["checked_at"] = current_time
            cache["latest_version"] = latest_version
            save_update_check_cache(cache, path)

    return UpdateCheckResult(
        installed_version=installed_version,
        latest_version=latest_version,
        cache=cache,
    )


def should_notify_about_update(result: UpdateCheckResult) -> bool:
    if not result.update_available or result.latest_version is None:
        return False
    notified_version = result.cache.get("notified_version")
    return notified_version != result.latest_version


def mark_update_notified(
    cache: dict[str, object],
    latest_version: str,
    *,
    cache_path: Path | None = None,
) -> None:
    cache["notified_version"] = latest_version
    save_update_check_cache(cache, cache_path)


def print_update_notice(
    latest_version: str,
    *,
    stream: object = sys.stderr,
) -> None:
    print(
        f"A newer version of {PACKAGE_NAME} is available: {latest_version}",
        file=stream,
    )
    print("Run: uv tool upgrade original-llm", file=stream)


def maybe_notify_about_update(
    *,
    force: bool = False,
    stream: object = sys.stderr,
    cache_path: Path | None = None,
    fetch_latest_version: Callable[[], str | None] | None = None,
) -> bool:
    installed_version = package_version_string()
    result = check_for_updates(
        installed_version,
        cache_path=cache_path,
        force=force,
        fetch_latest_version=fetch_latest_version,
    )
    if result.latest_version is None:
        if force:
            print(f"Could not check for updates for {PACKAGE_NAME}.", file=stream)
        return False

    if result.update_available:
        if force or should_notify_about_update(result):
            print_update_notice(result.latest_version, stream=stream)
            mark_update_notified(
                result.cache,
                result.latest_version,
                cache_path=cache_path,
            )
        return True

    if force:
        print(f"{PACKAGE_NAME} is up to date ({installed_version}).", file=stream)
    return False


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
    default_show_prompt_output: bool = True,
    default_max_history_turns: int | None = None,
    default_retrieval_examples: int = 0,
    default_normalize_chat_input: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version_string()}",
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
        "--max-history-turns",
        type=int,
        default=default_max_history_turns,
        help="Maximum retained chat turns when carry-context is on. 0 disables history retention.",
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
        "--retrieval-examples",
        type=int,
        default=default_retrieval_examples,
        help="How many similar chat examples to prepend before answering.",
    )
    parser.add_argument(
        "--retrieval-corpus-dir",
        type=str,
        default=None,
        help="Directory of .txt chat seed files used for retrieval. Defaults to the checkpoint chat data dir when available, otherwise data/chat_seed_real_persona_v1, data/chat_seed_refined_v1, data/chat_seed_simple, then finally the checkpoint training data dir.",
    )
    parser.add_argument(
        "--normalize-chat-input",
        action=argparse.BooleanOptionalAction,
        default=default_normalize_chat_input,
        help="Lightly normalize short chat input before interactive generation and retrieval lookup.",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="Print checkpoint metadata before generation.",
    )
    parser.add_argument(
        "--show-prompt-output",
        action=argparse.BooleanOptionalAction,
        default=default_show_prompt_output,
        help="Print prompt/output debug blocks in interactive mode.",
    )
    parser.add_argument(
        "--check-update",
        action="store_true",
        help="Check PyPI for a newer package version and exit.",
    )
    return parser.parse_args()


def resolve_interactive_mode(
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> bool:
    argv = argv if argv is not None else sys.argv[1:]
    if "--interactive" in argv:
        return True
    if "--no-interactive" in argv:
        return False
    if args.prompt is not None:
        return False
    return bool(args.interactive)


def run_cli(
    *,
    prog: str,
    description: str,
    epilog: str | None = None,
    default_checkpoint: str | None = None,
    default_download_url: str | None = None,
    default_cache_name: str = DEFAULT_CHECKPOINT_NAME,
    default_interactive: bool = False,
    default_carry_context: bool = False,
    default_user_label: str | None = None,
    default_reply_label: str | None = None,
    default_temperature: float = 0.8,
    default_top_k: int = 40,
    default_repetition_penalty: float = 1.0,
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    default_show_prompt_output: bool = True,
    default_max_history_turns: int | None = None,
    default_retrieval_examples: int = 0,
    default_normalize_chat_input: bool = False,
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
        default_show_prompt_output=default_show_prompt_output,
        default_max_history_turns=default_max_history_turns,
        default_retrieval_examples=default_retrieval_examples,
        default_normalize_chat_input=default_normalize_chat_input,
    )
    if args.check_update:
        maybe_notify_about_update(force=True, stream=sys.stdout)
        return 0

    if not update_check_disabled():
        maybe_notify_about_update()

    args.interactive = resolve_interactive_mode(args)
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_path = resolve_checkpoint(
        args,
        default_download_url=default_download_url,
        cache_name=default_cache_name,
    )
    model, tokenizer, checkpoint = load_generator(checkpoint_path, device)
    args.retrieval_corpus_dir = resolve_retrieval_corpus_dir(args, checkpoint)

    if args.show_meta:
        from original_llm.generate import print_meta
        print_meta(checkpoint_path, checkpoint, tokenizer, model, device)

    if args.interactive:
        interactive_loop(model, tokenizer, args, device)
        return 0

    prompt = args.prompt or "むかしむかし"
    effective_prompt = prompt
    direct_reply: str | None = None
    pending_user_input: str | None = None
    if args.user_label is not None and args.reply_label is not None:
        pending_user_input = extract_pending_chat_user_input(
            prompt,
            args.user_label,
            args.reply_label,
        )
        if pending_user_input is not None:
            prepared_user_input = prepare_chat_user_input(pending_user_input, args)
            direct_reply = curated_short_reply(prepared_user_input)
            if direct_reply is None:
                direct_reply = select_direct_chat_reply(prepared_user_input, args)
            if direct_reply is None:
                effective_prompt = prepend_chat_retrieval_examples(
                    prompt=prompt,
                    user_input=prepared_user_input,
                    args=args,
                    tokenizer=tokenizer,
                    context_length=model.config.context_length,
                )
            else:
                generated = f"{prompt}{direct_reply}"
                print(generated)
                return 0
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=effective_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        stop_at_period=args.stop_at_period,
        stop_at_blank_line=args.stop_at_blank_line,
        min_new_chars_before_stop=args.min_new_chars_before_stop,
        device=device,
        stop_sequences=chat_stop_sequences(
            args.user_label,
            args.reply_label,
            tokenizer=tokenizer,
        ),
    )
    continuation = generated_suffix(
        effective_prompt,
        generated,
        tokenizer=tokenizer,
    )
    if (
        pending_user_input is not None
        and args.user_label is not None
        and args.reply_label is not None
    ):
        continuation = extract_chat_reply(
            continuation,
            args.user_label,
            args.reply_label,
            tokenizer=tokenizer,
        )
    generated = f"{prompt}{continuation}"
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

            Update:
              uv tool upgrade original-llm        # if installed via `uv tool install`
              uv pip install -U original-llm      # if installed into a uv-managed venv
              pipx upgrade original-llm           # if installed via pipx
              original-llm --check-update         # check PyPI for a newer version
            """
        ).strip(),
    )


def main_chat() -> int:
    chat_checkpoint = preferred_chat_checkpoint()
    installed_version = package_version_string()
    return run_cli(
        prog="dazai-chat",
        description=(
            "Chat with the current Dazai-style checkpoint using conversation defaults.\n"
            "If no local checkpoint is found, the first run downloads one into "
            "~/.cache/original-llm/ using the installed package version."
        ),
        epilog=textwrap.dedent(
            f"""
            Quick Start:
              dazai-chat
              dazai-chat --show-meta
              dazai-chat --no-carry-context

            One-shot example:
              dazai-chat --no-interactive $'私: 酒飲もうぜ\\n相手: '

            Interactive commands:
              :help   show in-chat help
              :reset  clear session history
              :quit   exit

            Update:
              uv tool upgrade original-llm        # if installed via `uv tool install`
              uv pip install -U original-llm      # if installed into a uv-managed venv
              pipx upgrade original-llm           # if installed via pipx
              dazai-chat --check-update           # check PyPI for a newer version

            Defaults:
              interactive=True
              carry-context=True
              max-history-turns={DEFAULT_CHAT_MAX_HISTORY_TURNS}
              user-label=私
              reply-label=相手
              retrieval-examples={DEFAULT_CHAT_RETRIEVAL_EXAMPLES}
              normalize-chat-input=True
              temperature={DEFAULT_CHAT_TEMPERATURE}
              top-k={DEFAULT_CHAT_TOP_K}
              repetition-penalty={DEFAULT_CHAT_REPETITION_PENALTY}
              max-new-tokens={DEFAULT_CHAT_MAX_NEW_TOKENS}
            """
        ).strip(),
        default_checkpoint=str(chat_checkpoint) if chat_checkpoint is not None else None,
        default_download_url=default_chat_download_url(installed_version),
        default_cache_name=checkpoint_cache_name(
            version_text=installed_version,
        ),
        default_interactive=True,
        default_carry_context=True,
        default_user_label="私",
        default_reply_label="相手",
        default_temperature=DEFAULT_CHAT_TEMPERATURE,
        default_top_k=DEFAULT_CHAT_TOP_K,
        default_repetition_penalty=DEFAULT_CHAT_REPETITION_PENALTY,
        default_max_new_tokens=DEFAULT_CHAT_MAX_NEW_TOKENS,
        default_show_prompt_output=False,
        default_max_history_turns=DEFAULT_CHAT_MAX_HISTORY_TURNS,
        default_retrieval_examples=DEFAULT_CHAT_RETRIEVAL_EXAMPLES,
        default_normalize_chat_input=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
