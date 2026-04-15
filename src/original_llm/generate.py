from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import torch

from original_llm.config import ModelConfig
from original_llm.data import Tokenizer, tokenizer_from_state_dict
from original_llm.model import DecoderOnlyTransformer, count_parameters


DEFAULT_CHECKPOINT = Path("checkpoints") / "dazai-long" / "best.pt"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_MIN_NEW_CHARS_BEFORE_STOP = 24
DEFAULT_STOP_CHARS = ("。", "！", "？", "」")
DEFAULT_RETRIEVAL_SCORE_THRESHOLD = 55.0
DEFAULT_SHORT_CHAT_LOOKUP_LENGTH = 12
CHAT_LOOKUP_PUNCT_RE = re.compile(r"[。、！？!?…「」『』（）()\[\]{}<>:：,，./\\\-]+")
CHAT_LOOKUP_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ChatExample:
    text: str
    last_user_text: str
    last_reply_text: str
    lookup_text: str


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
    parser.add_argument(
        "--show-prompt-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print prompt/output debug blocks in interactive mode.",
    )
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
    retrieval_examples = getattr(args, "retrieval_examples", 0)
    if retrieval_examples < 0:
        raise ValueError(
            f"retrieval_examples must be non-negative, got {retrieval_examples}"
        )
    max_history_turns = getattr(args, "max_history_turns", None)
    if max_history_turns is not None and max_history_turns < 0:
        raise ValueError(
            f"max_history_turns must be non-negative, got {max_history_turns}"
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


def strip_role_label(
    line: str,
    label: str,
) -> str | None:
    for prefix in (role_prompt(label), role_prefix(label)):
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def normalize_chat_user_input(text: str) -> str:
    normalized = text.replace("\u3000", " ").strip()
    normalized = CHAT_LOOKUP_SPACE_RE.sub(" ", normalized)
    normalized = normalized.rstrip("!?！？").strip()
    replacements = (
        ("どゆこと", "どういうこと"),
        ("なにしてる", "何してる"),
        ("何してるの", "何してる"),
        ("今何してる", "いま何してる"),
        ("ご飯", "ごはん"),
    )
    for before, after in replacements:
        normalized = normalized.replace(before, after)
    return normalized


def normalize_chat_lookup_text(text: str) -> str:
    normalized = normalize_chat_user_input(text)
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = CHAT_LOOKUP_SPACE_RE.sub("", normalized)
    normalized = CHAT_LOOKUP_PUNCT_RE.sub("", normalized)
    replacements = (
        ("今日何するの", "今日は何する"),
        ("今日は何するの", "今日は何する"),
        ("今日何する", "今日は何する"),
    )
    for before, after in replacements:
        normalized = normalized.replace(before, after)
    return normalized


def prepare_chat_user_input(
    user_input: str,
    args: argparse.Namespace,
) -> str:
    if args.user_label is None or args.reply_label is None:
        return user_input
    if not getattr(args, "normalize_chat_input", False):
        return user_input
    return normalize_chat_user_input(user_input)


def parse_chat_turns(
    text: str,
    user_label: str,
    reply_label: str,
) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    pending_user: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        user_text = strip_role_label(line, user_label)
        if user_text is not None:
            if pending_user is not None:
                return []
            pending_user = user_text
            continue
        reply_text = strip_role_label(line, reply_label)
        if reply_text is not None:
            if pending_user is None:
                return []
            turns.append((pending_user, reply_text))
            pending_user = None
            continue
        return []

    if pending_user is not None or not turns:
        return []
    return turns


def format_chat_turns(
    turns: list[tuple[str, str]],
    user_label: str,
    reply_label: str,
) -> str:
    lines: list[str] = []
    for user_text, reply_text in turns:
        lines.append(f"{role_prompt(user_label)}{user_text}".rstrip())
        lines.append(f"{role_prompt(reply_label)}{reply_text}".rstrip())
    return "\n".join(lines).rstrip()


def extract_pending_chat_user_input(
    prompt: str,
    user_label: str,
    reply_label: str,
) -> str | None:
    pending_user: str | None = None

    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        user_text = strip_role_label(line, user_label)
        if user_text is not None:
            pending_user = user_text
            continue
        reply_text = strip_role_label(line, reply_label)
        if reply_text is not None:
            if pending_user is None:
                continue
            if not reply_text:
                return pending_user
            pending_user = None
            continue
        pending_user = None

    return None


def split_chat_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]


@lru_cache(maxsize=16)
def load_chat_examples(
    corpus_dir_text: str,
    user_label: str,
    reply_label: str,
) -> tuple[ChatExample, ...]:
    corpus_dir = Path(corpus_dir_text)
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        return ()

    examples: list[ChatExample] = []
    seen_texts: set[str] = set()
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for block in split_chat_blocks(text):
            turns = parse_chat_turns(block, user_label, reply_label)
            if not turns:
                continue
            formatted = format_chat_turns(turns, user_label, reply_label)
            if formatted in seen_texts:
                continue
            last_user_text = turns[-1][0].strip()
            lookup_text = normalize_chat_lookup_text(last_user_text)
            if not lookup_text:
                continue
            examples.append(
                ChatExample(
                    text=formatted,
                    last_user_text=last_user_text,
                    last_reply_text=turns[-1][1].strip(),
                    lookup_text=lookup_text,
                )
            )
            seen_texts.add(formatted)
    return tuple(examples)


def chat_example_score(
    query_lookup_text: str,
    example_lookup_text: str,
) -> float:
    if not query_lookup_text or not example_lookup_text:
        return 0.0

    score = SequenceMatcher(None, query_lookup_text, example_lookup_text).ratio() * 100.0
    if query_lookup_text == example_lookup_text:
        score += 120.0
    elif (
        query_lookup_text in example_lookup_text
        or example_lookup_text in query_lookup_text
    ):
        score += 45.0

    query_chars = set(query_lookup_text)
    example_chars = set(example_lookup_text)
    if query_chars and example_chars:
        score += 10.0 * len(query_chars & example_chars) / len(query_chars | example_chars)
    return score


def longest_common_substring_length(
    left: str,
    right: str,
) -> int:
    if not left or not right:
        return 0
    return SequenceMatcher(None, left, right).find_longest_match(
        0,
        len(left),
        0,
        len(right),
    ).size


def minimum_retrieval_common_length(
    query_lookup_text: str,
    example_lookup_text: str,
) -> int:
    min_length = min(len(query_lookup_text), len(example_lookup_text))
    if min_length <= 1:
        return min_length
    return max(2, (min_length * 3 + 4) // 5)


def select_chat_retrieval_candidates(
    user_input: str,
    corpus_dir: str | Path,
    user_label: str,
    reply_label: str,
    limit: int,
    *,
    min_score: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
) -> list[ChatExample]:
    if limit <= 0:
        return []

    query_lookup_text = normalize_chat_lookup_text(user_input)
    if not query_lookup_text:
        return []

    examples = load_chat_examples(str(Path(corpus_dir)), user_label, reply_label)
    scored_examples: list[tuple[float, int, float, int, int, ChatExample]] = []
    for example in examples:
        common_length = longest_common_substring_length(
            query_lookup_text,
            example.lookup_text,
        )
        required_common_length = minimum_retrieval_common_length(
            query_lookup_text,
            example.lookup_text,
        )
        if common_length < required_common_length:
            continue
        score = chat_example_score(query_lookup_text, example.lookup_text)
        if score < min_score:
            continue
        scored_examples.append(
            (
                common_length,
                required_common_length,
                score,
                abs(len(example.lookup_text) - len(query_lookup_text)),
                len(example.text),
                example,
            )
        )

    scored_examples.sort(
        key=lambda item: (
            -item[0],
            item[1],
            -item[2],
            item[3],
            item[4],
            item[5].text,
        )
    )
    return [example for _, _, _, _, _, example in scored_examples[:limit]]


def select_chat_retrieval_examples(
    user_input: str,
    corpus_dir: str | Path,
    user_label: str,
    reply_label: str,
    limit: int,
    *,
    min_score: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
) -> list[str]:
    return [
        example.text
        for example in select_chat_retrieval_candidates(
            user_input=user_input,
            corpus_dir=corpus_dir,
            user_label=user_label,
            reply_label=reply_label,
            limit=limit,
            min_score=min_score,
        )
    ]


def select_direct_chat_reply(
    user_input: str,
    args: argparse.Namespace,
) -> str | None:
    if args.user_label is None or args.reply_label is None:
        return None

    retrieval_corpus_dir = getattr(args, "retrieval_corpus_dir", None)
    if retrieval_corpus_dir in {None, ""}:
        return None

    query_lookup_text = normalize_chat_lookup_text(user_input)
    if not query_lookup_text or len(query_lookup_text) > 40:
        return None

    candidates = select_chat_retrieval_candidates(
        user_input=user_input,
        corpus_dir=retrieval_corpus_dir,
        user_label=args.user_label,
        reply_label=args.reply_label,
        limit=1,
        min_score=DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    )
    if not candidates:
        return None

    best = candidates[0]
    common_length = longest_common_substring_length(query_lookup_text, best.lookup_text)
    min_length = min(len(query_lookup_text), len(best.lookup_text))
    if common_length != min_length:
        return None
    return best.last_reply_text or None


def build_chat_retrieval_block(
    user_input: str,
    args: argparse.Namespace,
) -> str | None:
    if args.user_label is None or args.reply_label is None:
        return None

    retrieval_examples = getattr(args, "retrieval_examples", 0)
    retrieval_corpus_dir = getattr(args, "retrieval_corpus_dir", None)
    if retrieval_examples <= 0 or retrieval_corpus_dir in {None, ""}:
        return None

    examples = select_chat_retrieval_examples(
        user_input=user_input,
        corpus_dir=retrieval_corpus_dir,
        user_label=args.user_label,
        reply_label=args.reply_label,
        limit=retrieval_examples,
    )
    if not examples:
        return None

    return "\n\n".join(examples)


def prepend_chat_retrieval_examples(
    prompt: str,
    user_input: str,
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    context_length: int,
) -> str:
    retrieval_block = build_chat_retrieval_block(user_input, args)
    if retrieval_block is None:
        return prompt

    return trim_text_to_context(
        f"{retrieval_block}\n\n{prompt}",
        tokenizer,
        context_length,
    )


def chat_stop_sequences(
    user_label: str | None,
    reply_label: str | None,
) -> tuple[str, ...]:
    if user_label is None or reply_label is None:
        return ()
    return (f"\n{role_prefix(user_label)}", f"\n{role_prefix(reply_label)}")


def contains_stop_sequence(text: str, stop_sequences: tuple[str, ...]) -> bool:
    return any(sequence and sequence in text for sequence in stop_sequences)


def extract_chat_reply(
    text: str,
    user_label: str,
    reply_label: str,
) -> str:
    reply = text.lstrip()
    for prefix in (role_prompt(reply_label), role_prefix(reply_label)):
        if reply.startswith(prefix):
            reply = reply[len(prefix) :].lstrip()
            break

    stop_positions: list[int] = []
    for sequence in chat_stop_sequences(user_label, reply_label):
        position = reply.find(sequence)
        if position != -1:
            stop_positions.append(position)
    if stop_positions:
        reply = reply[: min(stop_positions)]

    return reply.strip()


def append_chat_history(
    history: str,
    user_input: str,
    reply_text: str,
    user_label: str,
    reply_label: str,
    tokenizer: Tokenizer,
    context_length: int,
    max_turns: int | None = None,
) -> str:
    turn = (
        f"{role_prompt(user_label)}{user_input}\n"
        f"{role_prompt(reply_label)}{reply_text}"
    ).rstrip()
    if history:
        turn = f"{history}\n{turn}"
    if max_turns is not None:
        if max_turns <= 0:
            return ""
        turns = parse_chat_turns(turn, user_label, reply_label)
        if turns:
            turn = format_chat_turns(turns[-max_turns:], user_label, reply_label)
    return trim_text_to_context(turn, tokenizer, context_length)


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
    stop_sequences: tuple[str, ...] = (),
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
        if contains_stop_sequence(suffix, stop_sequences):
            break
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
    show_prompt_output = getattr(args, "show_prompt_output", True)
    print("interactive mode")
    print(":quit or :exit で終了")
    print(":reset で文脈をリセット")
    print(":help でヘルプ")
    if args.carry_context:
        max_history_turns = getattr(args, "max_history_turns", None)
        if max_history_turns is None:
            print("carry-context: on")
        else:
            print(f"carry-context: on (max-history-turns={max_history_turns})")
    if args.user_label is not None and args.reply_label is not None:
        print(f"chat-format: {args.user_label}/{args.reply_label}")
    retrieval_examples = getattr(args, "retrieval_examples", 0)
    if args.user_label is not None and args.reply_label is not None and retrieval_examples > 0:
        retrieval_corpus_dir = getattr(args, "retrieval_corpus_dir", None)
        if retrieval_corpus_dir in {None, ""}:
            print("retrieval: off (corpus not found)")
        else:
            print(f"retrieval: {retrieval_examples} example(s)")
    if args.user_label is not None and args.reply_label is not None:
        normalize_chat_input = getattr(args, "normalize_chat_input", False)
        print(f"normalize-chat-input: {'on' if normalize_chat_input else 'off'}")

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
            print("テキストを入力すると返答を生成")
            print(":reset -> この起動中の会話履歴を消す")
            print(":quit  -> 終了")
            print(f"carry-context -> {'on' if args.carry_context else 'off'}")
            if args.carry_context:
                print(
                    "max-history-turns -> "
                    f"{getattr(args, 'max_history_turns', None)}"
                )
            print(f"show-prompt-output -> {'on' if show_prompt_output else 'off'}")
            if args.user_label is not None and args.reply_label is not None:
                print(f"labels -> {args.user_label}/{args.reply_label}")
                print(
                    "retrieval-examples -> "
                    f"{getattr(args, 'retrieval_examples', 0)}"
                )
                print(
                    "normalize-chat-input -> "
                    f"{'on' if getattr(args, 'normalize_chat_input', False) else 'off'}"
                )
            print(f"max-new-tokens -> {args.max_new_tokens}")
            continue

        prepared_user_input = prepare_chat_user_input(user_input, args)
        lookup_text = normalize_chat_lookup_text(prepared_user_input)
        direct_reply = select_direct_chat_reply(prepared_user_input, args)
        if direct_reply is not None:
            reply_text = direct_reply
            print()
            if show_prompt_output or args.user_label is None or args.reply_label is None:
                direct_prompt = build_interactive_prompt(
                    user_input=prepared_user_input,
                    history=history,
                    args=args,
                    tokenizer=tokenizer,
                    context_length=model.config.context_length,
                )
                print_block("prompt", direct_prompt)
                print()
                print_block("output", reply_text)
            else:
                print(f"{role_prompt(args.reply_label)}{reply_text}".rstrip())
            print()

            if args.carry_context and args.user_label is not None and args.reply_label is not None:
                history = append_chat_history(
                    history=history,
                    user_input=prepared_user_input,
                    reply_text=reply_text,
                    user_label=args.user_label,
                    reply_label=args.reply_label,
                    tokenizer=tokenizer,
                    context_length=model.config.context_length,
                    max_turns=getattr(args, "max_history_turns", None),
                )
            continue

        prompt = build_interactive_prompt(
            user_input=prepared_user_input,
            history=(
                ""
                if history and len(lookup_text) <= DEFAULT_SHORT_CHAT_LOOKUP_LENGTH
                else history
            ),
            args=args,
            tokenizer=tokenizer,
            context_length=model.config.context_length,
        )
        retrieval_block = build_chat_retrieval_block(prepared_user_input, args)
        if retrieval_block is not None and history and args.user_label is not None:
            current_turn_prompt = build_interactive_prompt(
                user_input=prepared_user_input,
                history="",
                args=args,
                tokenizer=tokenizer,
                context_length=model.config.context_length,
            )
            if len(lookup_text) <= DEFAULT_SHORT_CHAT_LOOKUP_LENGTH:
                prompt = trim_text_to_context(
                    f"{retrieval_block}\n\n{current_turn_prompt}",
                    tokenizer,
                    model.config.context_length,
                )
            else:
                prompt = trim_text_to_context(
                    f"{history}\n\n{retrieval_block}\n\n{current_turn_prompt}",
                    tokenizer,
                    model.config.context_length,
                )
        elif retrieval_block is not None:
            prompt = trim_text_to_context(
                f"{retrieval_block}\n\n{prompt}",
                tokenizer,
                model.config.context_length,
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
            stop_sequences=chat_stop_sequences(args.user_label, args.reply_label),
        )
        suffix = generated_suffix(prompt, generated)
        reply_text = suffix
        if args.user_label is not None and args.reply_label is not None:
            reply_text = extract_chat_reply(
                suffix,
                user_label=args.user_label,
                reply_label=args.reply_label,
            )
        print()
        if show_prompt_output or args.user_label is None or args.reply_label is None:
            print_block("prompt", prompt)
            print()
            print_block("output", reply_text or suffix or generated)
        else:
            print(f"{role_prompt(args.reply_label)}{reply_text}".rstrip())
        print()

        if args.carry_context:
            if args.user_label is not None and args.reply_label is not None:
                history = append_chat_history(
                    history=history,
                    user_input=prepared_user_input,
                    reply_text=reply_text,
                    user_label=args.user_label,
                    reply_label=args.reply_label,
                    tokenizer=tokenizer,
                    context_length=model.config.context_length,
                    max_turns=getattr(args, "max_history_turns", None),
                )
            else:
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
        stop_sequences=chat_stop_sequences(args.user_label, args.reply_label),
    )
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
