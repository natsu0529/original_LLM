from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import torch

from original_llm.config import CHAT_TURN_END_MARKER, ModelConfig
from original_llm.data import (
    Tokenizer,
    blocked_generation_token_ids,
    tokenizer_from_state_dict,
)
from original_llm.language import (
    is_non_japanese_input,
    looks_like_single_word,
    looks_like_typo_correction,
)
from original_llm.memory import (
    DEFAULT_IMPORTANCE,
    DEFAULT_INJECT_LIMIT,
    MAX_IMPORTANCE,
    MIN_IMPORTANCE,
    MemoryEntry,
    MemoryStore,
    default_memory_path,
)
from original_llm.model import DecoderOnlyTransformer, count_parameters


DEFAULT_CHECKPOINT = Path("checkpoints") / "dazai-long" / "best.pt"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_MIN_NEW_CHARS_BEFORE_STOP = 24
DEFAULT_STOP_CHARS = ("。", "！", "？", "」")
DEFAULT_RETRIEVAL_SCORE_THRESHOLD = 55.0
DEFAULT_SHORT_CHAT_LOOKUP_LENGTH = 12

# Reply text used when the user types something that contains zero Japanese
# letters. Hard-coded so the model is never asked to invent it.
NON_JAPANESE_GUARD_REPLY = "私は日本語しか喋れないんだ。"

# Reply when the model's confidence is low and the input doesn't match any
# stored memory. The user is invited to teach us.
UNKNOWN_WORD_QUESTION_TEMPLATES: tuple[str, ...] = (
    "知らない言葉だね、それは何？",
    "{word}って初めて聞くな、何のこと？",
    "{word}か、教えてもらってもいい？",
)

# Reply we emit after the user explains a previously-unknown word and we have
# successfully stored their explanation.
UNKNOWN_WORD_LEARNED_TEMPLATES: tuple[str, ...] = (
    "覚えた、{word}ってそういうことなんだね。",
    "なるほど、{word}メモしておくね。",
    "ありがとう、{word}覚えた。",
)

# Confidence threshold below which a single-word reply is treated as a sign
# the model didn't really understand the input. Chosen empirically — typical
# coherent replies for known phrases sit well above 0.4 mean prob.
DEFAULT_UNKNOWN_CONFIDENCE_THRESHOLD = 0.55

# Memory key prefix used when storing words the user introduced through the
# unknown-word teaching flow. Keeps them visually distinct from explicit
# ``:remember`` entries.
UNKNOWN_MEMORY_KEY_PREFIX = "word:"

# Triggers + reply for "how do I update?" type questions. We keep this as a
# curated answer because the model itself shouldn't try to invent install
# commands. Patterns are checked against the lower-cased input AND the
# original input (Japanese strings are case-stable; the lower-case pass is
# only there so "uv" / "PyPI" / etc. don't slip through).
UPDATE_QUESTION_PATTERNS: tuple[str, ...] = (
    "アップデート",
    "アップグレード",
    "更新",
    "バージョンアップ",
    "最新版",
    "新しいバージョン",
    "uv tool upgrade",
    "pip install -u",
    "upgrade",
)
UPDATE_GUIDE_REPLY = (
    "アップデートは `uv tool upgrade original-llm` を打つだけだよ。"
    "uv じゃない人は `pip install -U original-llm` でも OK。"
)


def looks_like_update_question(text: str) -> bool:
    """True if ``text`` looks like a "how do I update?" question.

    Keyword based, on purpose: the model is too small to identify intent on
    its own, so the curated answer is gated by simple substring matches.
    """
    if not text:
        return False
    target = text.lower()
    if any(pattern in target for pattern in UPDATE_QUESTION_PATTERNS):
        return True
    return any(pattern in text for pattern in UPDATE_QUESTION_PATTERNS)
CHAT_LOOKUP_PUNCT_RE = re.compile(r"[。、！？!?…「」『』（）()\[\]{}<>:：,，./\\\-]+")
CHAT_LOOKUP_SPACE_RE = re.compile(r"\s+")

# Patterns that mark broken / low-quality model output we don't want to surface
# from the retrieval corpus as canned answers.
LOW_QUALITY_REPLY_SUBSTRINGS: tuple[str, ...] = (
    "、、、",
    "、、",
    "。。",
    "ーー",
    "たいない",
    "ているない",
    "るのいる",
    "なのなの",
)
LOW_QUALITY_SHORT_REPLIES: frozenset[str] = frozenset(
    {
        "はい",
        "はい。",
        "うん",
        "うん。",
        "ええ",
        "ええ。",
        "ああ",
        "ああ。",
        "そう",
        "そう。",
        "ね",
        "ねえ",
        "へえ",
        "へえー",
        "ふうん",
        "なるほど",
    }
)
MIN_QUALITY_REPLY_CHAR_LENGTH = 4

# Replies that begin with these stems are typically dangling fragments cut
# from a longer narrative passage (e.g. "もどすのが。" / "ところが。").
# Filtering these out helps even when individual chars are kanji.
LOW_QUALITY_REPLY_PREFIXES: tuple[str, ...] = (
    "もどすのが",
    "ところが",
    "けれども",
    "それでも",
    "なるほどね",
)

# Replies that end with these endings on their own are usually incomplete.
LOW_QUALITY_REPLY_ENDINGS: tuple[str, ...] = (
    "のが",
    "のが。",
    "けれど",
    "けれど。",
    "ですが",
    "ですが。",
)

# Sentence-final patterns that signal a truncated / dangling utterance.
# We deliberately keep the trigger set narrow: "は？" alone is a perfectly
# natural friend-style follow-up ("そっちは？"), while comma-prefixed short
# fragments like "、お金も？" or "、おも？" are almost always model artifacts.
DANGLING_FRAGMENT_TAIL_RE = re.compile(
    r"、[ぁ-んァ-ン一-龥ー]{1,5}(?:が|を|に|で|も|へ)[?？！!]?$"
)
SOLO_PARTICLE_TAIL_RE = re.compile(
    r"^(?:.{0,4}?)(?:が|を|に|で|も)[?？！!]?$"
)

# Generic supportive fallbacks for when the model returns broken or
# off-topic text and no curated/retrieval reply is available. These are
# intentionally vague so they don't lie about what the user said.
GENERIC_FRIENDLY_FALLBACKS: tuple[str, ...] = (
    "うん、もう少し聞かせて。",
    "そっか。話してくれてありがとう。",
    "なるほど、そういう感じか。",
    "ちょっと考えるね。続けて？",
    "うん、それでそれで？",
)


# Curated short replies for very common everyday utterances. These take
# precedence over the noisy auto-built seed corpus when the user input
# closely matches one of the keys. Replies are intentionally varied so
# repeated identical prompts don't always emit the same answer.
CURATED_SHORT_REPLIES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("こんにちは", "こんにちわ"): (
        "こんにちは。今日はどうしてた？",
        "こんにちは。元気にしてた？",
        "やあ、こんにちは。",
    ),
    ("こんばんは", "こんばんわ"): (
        "こんばんは。今日はもう落ち着いた？",
        "こんばんは。無事に1日終わったね。",
    ),
    ("おはよう", "おはよ"): (
        "おはよう。よく眠れた？",
        "おはよう。今日はゆっくりやろう。",
    ),
    ("ただいま",): (
        "おかえり。お疲れさま。",
        "おかえり。少し休もう。",
    ),
    ("おやすみ",): (
        "おやすみ。明日はまた話そう。",
        "おやすみ。ゆっくり休んでね。",
    ),
    ("ありがとう", "ありがと", "サンキュー", "さんきゅう"): (
        "どういたしまして。",
        "気にしないで。こちらこそありがとう。",
        "うん、いつでも。",
    ),
    ("ごめん", "ごめんね", "すまない", "すまん"): (
        "気にしないで。大丈夫だよ。",
        "うん、もう平気。",
    ),
    ("じゃあね", "またね", "バイバイ", "ばいばい"): (
        "またね。気をつけて。",
        "うん、またね。",
    ),
    ("元気", "元気？"): (
        "まあまあ元気。そっちは？",
        "うん、なんとかやってる。そっちはどう？",
    ),
    ("つかれた", "疲れた", "しんどい", "だるい"): (
        "お疲れさま。少し休もう。",
        "そっか、無理しないで。",
        "今日はもうゆっくりしよう。",
    ),
    ("ねむい", "眠い"): (
        "そろそろ寝てもいいかもね。",
        "うん、無理しないで休んで。",
    ),
    ("眠れない", "ねむれない"): (
        "そういう日はあるよ。深呼吸でもしてみる？",
        "うん、ゆっくり考え事しないで横になろう。",
    ),
    ("お腹すいた", "おなかすいた", "腹減った", "はらへった"): (
        "なにか軽く食べちゃおう。",
        "うん、ちゃんと食べたほうがいい。",
    ),
    ("うれしい", "嬉しい"): (
        "よかったね。それは何より。",
        "おお、いいね。話聞きたい。",
    ),
    ("かなしい", "悲しい", "つらい", "辛い"): (
        "そっか、つらいね。少しだけ話そうか。",
        "無理しなくていいよ。ここにいるから。",
    ),
    ("さびしい", "寂しい"): (
        "うん、そういう夜あるよね。",
        "そばにいるよ、ゆっくりでいい。",
    ),
    ("好き", "大好き"): (
        "うん、それを聞けてうれしい。",
        "ありがとう。なんかこそばゆいな。",
    ),
    ("楽しい", "たのしい"): (
        "それはいいね。続きが聞きたい。",
        "うん、いい一日になったね。",
    ),
    ("今日は何してた", "今日何してた", "今日何してたの", "今日は何してたの"): (
        "本読んでぼんやりしてた。そっちは？",
        "家にいたよ、ゆっくりしてた。",
        "散歩してた。気持ちよかったよ。",
    ),
    ("どういうこと", "どゆこと", "えっとどういうこと", "えっとどゆこと"): (
        "あ、ごめん。わかりにくかったね。もう一回言うね。",
        "ごめん、変な言い方だった。要するに少し休もうって話。",
    ),
    ("遊ぼう", "遊ぼうよ", "少し遊ぼう", "少し遊ぼうよ", "ちょっと遊ぼう"): (
        "いいね、何して遊ぼうか？",
        "うん、いいよ。なに話そっか。",
    ),
}

# Greetings vs farewells live in disjoint "category" buckets — replying with
# the wrong category is a frequent failure mode in the auto-built seed data
# (e.g. "こんにちは" -> "こんばんわ"). We treat such cross-category replies as
# low quality so they don't get surfaced as direct retrieval answers.
GREETING_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "morning": ("おはよう",),
    "afternoon": ("こんにちは", "こんにちわ"),
    "evening": ("こんばんは", "こんばんわ"),
    "farewell": ("さよなら", "さようなら", "じゃあね", "またね", "バイバイ", "ばいばい"),
    "welcome_home": ("ただいま", "おかえり"),
}


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
    memory_inject = getattr(args, "memory_inject", None)
    if memory_inject is not None and memory_inject < 0:
        raise ValueError(
            f"memory_inject must be non-negative, got {memory_inject}"
        )


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


def suppress_token_ids(
    next_token_logits: torch.Tensor,
    token_ids: tuple[int, ...],
) -> torch.Tensor:
    if not token_ids:
        return next_token_logits

    adjusted_logits = next_token_logits.clone()
    for token_id in token_ids:
        if 0 <= token_id < adjusted_logits.size(-1):
            adjusted_logits[:, token_id] = float("-inf")
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


def dedupe_texts(values: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return tuple(ordered)


def turn_end_markers(
    tokenizer: Tokenizer | None = None,
) -> tuple[str, ...]:
    markers = [CHAT_TURN_END_MARKER]
    if tokenizer is not None:
        decoded_marker = tokenizer.decode(
            tokenizer.encode(CHAT_TURN_END_MARKER)
        ).strip()
        if decoded_marker:
            markers.append(decoded_marker)
    return dedupe_texts(markers)


def is_turn_end_marker(
    line: str,
    tokenizer: Tokenizer | None = None,
) -> bool:
    return line.strip() in turn_end_markers(tokenizer)


def strip_turn_end_marker(
    text: str,
    tokenizer: Tokenizer | None = None,
) -> str:
    stripped = text.strip()
    marker_positions = [
        position
        for marker in turn_end_markers(tokenizer)
        if (position := stripped.find(marker)) != -1
    ]
    if marker_positions:
        stripped = stripped[: min(marker_positions)].rstrip()
    return stripped


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


PROBE_SUFFIX_NORMS = (
    "あなたは",
    "そっちは",
    "そちらは",
    "きみは",
    "君は",
)


SELF_STATE_REPLY_MARKERS = (
    "私は",
    "私も",
    "私？",
    "わたしは",
    "わたしも",
    "ぼちぼち",
    "ぼんやり",
    "本読ん",
    "ぼーっと",
    "家にい",
    "家でゆっくり",
    "ふつう",
    "まあまあ",
    "ゆっくりしてる",
    "出かけて",
)


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
        if is_turn_end_marker(line):
            if pending_user is not None:
                return []
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
            turns.append((pending_user, strip_turn_end_marker(reply_text)))
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
        lines.append(
            f"{role_prompt(reply_label)}{strip_turn_end_marker(reply_text)}".rstrip()
        )
        lines.append(CHAT_TURN_END_MARKER)
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
        if is_turn_end_marker(line):
            pending_user = None
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


def greeting_category(lookup_text: str) -> str | None:
    if not lookup_text:
        return None
    for category, keywords in GREETING_CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lookup_text:
                return category
    return None


def _echo_compact(text: str) -> str:
    compact = normalize_chat_lookup_text(text)
    compact = compact.replace("ー", "").replace("〜", "").replace("～", "")
    return compact


def is_echo_reply(user_text: str, reply_text: str) -> bool:
    user_norm = normalize_chat_lookup_text(user_text)
    reply_norm = normalize_chat_lookup_text(reply_text)
    if not user_norm or not reply_norm:
        return False
    if user_norm == reply_norm:
        return True
    user_compact = _echo_compact(user_text)
    reply_compact = _echo_compact(reply_text)
    if user_compact and user_compact == reply_compact:
        return True
    return False


def has_dangling_particle_tail(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return DANGLING_FRAGMENT_TAIL_RE.search(stripped) is not None


def is_low_quality_reply(
    user_text: str,
    reply_text: str,
) -> bool:
    cleaned = strip_turn_end_marker(reply_text).strip()
    if not cleaned:
        return True
    if cleaned in LOW_QUALITY_SHORT_REPLIES:
        return True
    if any(token in cleaned for token in LOW_QUALITY_REPLY_SUBSTRINGS):
        return True
    if any(cleaned.startswith(prefix) for prefix in LOW_QUALITY_REPLY_PREFIXES):
        return True
    cleaned_for_ending = cleaned.rstrip("。！？!?")
    if cleaned_for_ending and any(
        cleaned_for_ending.endswith(ending.rstrip("。！？!?"))
        and len(cleaned_for_ending) <= len(ending) + 2
        for ending in LOW_QUALITY_REPLY_ENDINGS
    ):
        return True
    if has_dangling_particle_tail(cleaned):
        return True
    if is_echo_reply(user_text, cleaned):
        return True

    user_lookup = normalize_chat_lookup_text(user_text)
    reply_lookup = normalize_chat_lookup_text(cleaned)
    if not reply_lookup:
        return True
    if len(reply_lookup) < MIN_QUALITY_REPLY_CHAR_LENGTH:
        return True

    user_category = greeting_category(user_lookup)
    reply_category = greeting_category(reply_lookup)
    if (
        user_category is not None
        and reply_category is not None
        and user_category != reply_category
    ):
        return True

    return False


def curated_short_reply(
    user_input: str,
    *,
    avoid_replies: tuple[str, ...] = (),
    rotation_index: int = 0,
) -> str | None:
    primary = normalize_chat_lookup_text(user_input)
    if not primary:
        return None

    # Compound inputs like "大丈夫だよ、少し遊ぼうよ" carry their intent in the
    # last clause; try it as a secondary lookup so curated still fires.
    last_clause = ""
    for sep in ("、", ","):
        if sep in user_input:
            last_clause = user_input.rsplit(sep, 1)[-1]
            break
    last_lookup = normalize_chat_lookup_text(last_clause) if last_clause else ""

    lookups = [primary]
    if last_lookup and last_lookup != primary:
        lookups.append(last_lookup)

    avoid_norms = {
        normalize_chat_lookup_text(r) for r in avoid_replies if r
    }
    matched_replies: tuple[str, ...] | None = None
    for lookup in lookups:
        if not lookup or len(lookup) > 12:
            continue
        for keys, replies in CURATED_SHORT_REPLIES.items():
            for key in keys:
                key_norm = normalize_chat_lookup_text(key)
                if not key_norm:
                    continue
                if lookup == key_norm or (
                    len(lookup) <= 6 and (key_norm in lookup or lookup in key_norm)
                ):
                    matched_replies = replies
                    break
            if matched_replies is not None:
                break
        if matched_replies is not None:
            break
    if matched_replies is None:
        return None

    ordered = list(matched_replies)
    fresh = [r for r in ordered if normalize_chat_lookup_text(r) not in avoid_norms]
    pool = fresh or ordered
    return pool[rotation_index % len(pool)]


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
            last_reply_text = turns[-1][1].strip()
            if is_low_quality_reply(last_user_text, last_reply_text):
                continue
            lookup_text = normalize_chat_lookup_text(last_user_text)
            if not lookup_text:
                continue
            examples.append(
                ChatExample(
                    text=formatted,
                    last_user_text=last_user_text,
                    last_reply_text=last_reply_text,
                    lookup_text=lookup_text,
                )
            )
            seen_texts.add(formatted)
    return tuple(examples)


def chat_example_score(
    query_lookup_text: str,
    example_lookup_text: str,
    *,
    reply_text: str | None = None,
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

    if reply_text is not None:
        reply_clean = strip_turn_end_marker(reply_text).strip()
        reply_lookup = normalize_chat_lookup_text(reply_clean)
        reply_length = len(reply_lookup)
        if 8 <= reply_length <= 30:
            score += 8.0
        elif 31 <= reply_length <= 50:
            score += 4.0
        elif reply_length < 4:
            score -= 10.0
        elif reply_length > 80:
            score -= 6.0
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
        score = chat_example_score(
            query_lookup_text,
            example.lookup_text,
            reply_text=example.last_reply_text,
        )
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


def extract_probe_suffix(user_input: str) -> str | None:
    """Return the probe portion ("あなたは" etc.) when input ends with one.

    The compound case is "{statement}, {probe}?" — without splitting we'd match
    the statement against unrelated weather/closing templates and miss the
    self-state reply that the probe expects. Returning the probe lets a
    secondary retrieval target probe → 私は… templates explicitly.
    """
    norm = normalize_chat_lookup_text(user_input)
    for probe in PROBE_SUFFIX_NORMS:
        if norm.endswith(probe) and len(norm) > len(probe):
            return probe
    return None


def reply_is_self_state(reply: str) -> bool:
    """True if the reply describes the speaker's own state (suitable for a probe)."""
    norm = normalize_chat_lookup_text(reply)
    return any(marker in norm for marker in SELF_STATE_REPLY_MARKERS)


def select_direct_chat_reply(
    user_input: str,
    args: argparse.Namespace,
    *,
    avoid_replies: tuple[str, ...] = (),
) -> str | None:
    if args.user_label is None or args.reply_label is None:
        return None

    retrieval_corpus_dir = getattr(args, "retrieval_corpus_dir", None)
    if retrieval_corpus_dir in {None, ""}:
        return None

    query_lookup_text = normalize_chat_lookup_text(user_input)
    if not query_lookup_text or len(query_lookup_text) > 40:
        return None

    avoid_norms = {
        normalize_chat_lookup_text(reply)
        for reply in avoid_replies
        if reply
    }

    probe = extract_probe_suffix(user_input)
    if probe is not None:
        probe_candidates = select_chat_retrieval_candidates(
            user_input=probe,
            corpus_dir=retrieval_corpus_dir,
            user_label=args.user_label,
            reply_label=args.reply_label,
            limit=8,
            min_score=DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
        )
        for candidate in probe_candidates:
            reply = candidate.last_reply_text
            if not reply:
                continue
            if not reply_is_self_state(reply):
                continue
            if is_low_quality_reply(candidate.last_user_text, reply):
                continue
            if normalize_chat_lookup_text(reply) in avoid_norms:
                continue
            return reply

    candidates = select_chat_retrieval_candidates(
        user_input=user_input,
        corpus_dir=retrieval_corpus_dir,
        user_label=args.user_label,
        reply_label=args.reply_label,
        limit=8,
        min_score=DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    )
    if not candidates:
        return None

    for candidate in candidates:
        common_length = longest_common_substring_length(
            query_lookup_text,
            candidate.lookup_text,
        )
        min_length = min(len(query_lookup_text), len(candidate.lookup_text))
        if common_length != min_length:
            continue
        reply = candidate.last_reply_text
        if not reply:
            continue
        if is_low_quality_reply(candidate.last_user_text, reply):
            continue
        if normalize_chat_lookup_text(reply) in avoid_norms:
            continue
        return reply
    return None


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
    tokenizer: Tokenizer | None = None,
) -> tuple[str, ...]:
    if user_label is None or reply_label is None:
        return ()
    sequences = [
        f"\n{marker}"
        for marker in turn_end_markers(tokenizer)
    ]
    sequences.extend(
        [
            *turn_end_markers(tokenizer),
            f"\n{role_prefix(user_label)}",
            f"\n{role_prefix(reply_label)}",
        ]
    )
    return dedupe_texts(sequences)


def contains_stop_sequence(text: str, stop_sequences: tuple[str, ...]) -> bool:
    return any(sequence and sequence in text for sequence in stop_sequences)


def extract_chat_reply(
    text: str,
    user_label: str,
    reply_label: str,
    tokenizer: Tokenizer | None = None,
) -> str:
    reply = text.lstrip()
    for prefix in (role_prompt(reply_label), role_prefix(reply_label)):
        if reply.startswith(prefix):
            reply = reply[len(prefix) :].lstrip()
            break

    stop_positions: list[int] = []
    for sequence in chat_stop_sequences(
        user_label,
        reply_label,
        tokenizer=tokenizer,
    ):
        position = reply.find(sequence)
        if position != -1:
            stop_positions.append(position)
    if stop_positions:
        reply = reply[: min(stop_positions)]

    return strip_turn_end_marker(reply, tokenizer=tokenizer)


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
    turn = format_chat_turns(
        [(user_input, reply_text)],
        user_label,
        reply_label,
    )
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


def is_unsatisfactory_reply(
    reply_text: str,
    *,
    user_input: str,
    avoid_replies: tuple[str, ...] = (),
) -> bool:
    cleaned = strip_turn_end_marker(reply_text).strip()
    if not cleaned:
        return True
    if is_low_quality_reply(user_input, cleaned):
        return True
    reply_norm = normalize_chat_lookup_text(cleaned)
    if not reply_norm:
        return True
    for avoid in avoid_replies:
        avoid_norm = normalize_chat_lookup_text(avoid)
        if avoid_norm and avoid_norm == reply_norm:
            return True
    return False


def format_memory_block(
    entries: list[MemoryEntry],
    user_label: str,
    reply_label: str,
) -> str:
    """Render memory entries as a single chat turn the model can attend to."""
    if not entries:
        return ""
    facts: list[str] = []
    for entry in entries:
        key = entry.key.strip()
        value = entry.value.strip()
        if not key or not value:
            continue
        facts.append(f"{key}は{value}")
    if not facts:
        return ""
    user_line = "覚えておいてね: " + "、".join(facts) + "。"
    reply_line = "うん、覚えた。"
    return (
        f"{role_prompt(user_label)}{user_line}\n"
        f"{role_prompt(reply_label)}{reply_line}\n"
        f"{CHAT_TURN_END_MARKER}"
    )


def parse_remember_command(
    command: str,
) -> tuple[str, str, int] | None:
    """Parse ``:remember <key> <value> [importance]`` style command lines.

    Returns ``(key, value, importance)`` or ``None`` if it does not match.
    Accepts ``:remember key=value`` and trailing 1-5 importance tokens.
    """
    if not command.startswith(":remember"):
        return None
    rest = command[len(":remember") :].strip()
    if not rest:
        return None

    # Optional trailing importance number (single digit 1..5).
    importance = DEFAULT_IMPORTANCE
    parts = rest.rsplit(None, 1)
    if len(parts) == 2 and parts[1].isdigit():
        candidate = int(parts[1])
        if MIN_IMPORTANCE <= candidate <= MAX_IMPORTANCE:
            importance = candidate
            rest = parts[0].strip()
            if not rest:
                return None

    if "=" in rest:
        key, _, value = rest.partition("=")
    else:
        head_parts = rest.split(None, 1)
        if len(head_parts) != 2:
            return None
        key, value = head_parts
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return None
    return key, value, importance


def parse_forget_command(command: str) -> tuple[str, str] | None:
    """Parse ``:forget <id>`` or ``:forget-key <key>``.

    Returns ``("id", "<id>")`` or ``("key", "<key>")`` or ``None``.
    """
    if command.startswith(":forget-key"):
        rest = command[len(":forget-key") :].strip()
        return ("key", rest) if rest else None
    if command.startswith(":forget"):
        rest = command[len(":forget") :].strip()
        return ("id", rest) if rest else None
    return None


def fallback_friendly_reply(
    *,
    avoid_replies: tuple[str, ...] = (),
    rotation_index: int = 0,
) -> str:
    avoid_norms = {
        normalize_chat_lookup_text(r) for r in avoid_replies if r
    }
    fresh = [
        r
        for r in GENERIC_FRIENDLY_FALLBACKS
        if normalize_chat_lookup_text(r) not in avoid_norms
    ]
    pool = fresh or list(GENERIC_FRIENDLY_FALLBACKS)
    return pool[rotation_index % len(pool)]


def generate_chat_reply_with_resample(
    *,
    model: "DecoderOnlyTransformer",
    tokenizer: Tokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    user_input: str,
    avoid_replies: tuple[str, ...] = (),
    max_resamples: int = 2,
) -> tuple[str, str, str, float]:
    base_temperature = max(args.temperature, 1e-6)
    base_top_k = args.top_k
    base_seed = getattr(args, "seed", 0)
    chosen_reply: str = ""
    chosen_suffix: str = ""
    chosen_generated: str = ""
    chosen_confidence: float = 0.0
    last_generated: str = ""
    last_suffix: str = ""
    last_reply: str = ""
    last_confidence: float = 0.0
    for attempt in range(max_resamples + 1):
        if attempt > 0:
            torch.manual_seed(base_seed + 100 * attempt + 7)
        temperature = base_temperature
        top_k = base_top_k
        if attempt == 1:
            temperature = max(base_temperature * 1.6, 0.5)
            top_k = max(base_top_k, 16)
        elif attempt >= 2:
            temperature = max(base_temperature * 2.2, 0.7)
            top_k = max(base_top_k, 24)

        generated, confidence = generate_text_tracked(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
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
        suffix = generated_suffix(prompt, generated, tokenizer=tokenizer)
        if args.user_label is not None and args.reply_label is not None:
            reply_text = extract_chat_reply(
                suffix,
                user_label=args.user_label,
                reply_label=args.reply_label,
                tokenizer=tokenizer,
            )
        else:
            reply_text = suffix
        last_generated = generated
        last_suffix = suffix
        last_reply = reply_text
        last_confidence = confidence
        if not is_unsatisfactory_reply(
            reply_text,
            user_input=user_input,
            avoid_replies=avoid_replies,
        ):
            chosen_reply = reply_text
            chosen_suffix = suffix
            chosen_generated = generated
            chosen_confidence = confidence
            break
    if not chosen_generated:
        if is_unsatisfactory_reply(
            last_reply,
            user_input=user_input,
            avoid_replies=avoid_replies,
        ):
            chosen_reply = fallback_friendly_reply(
                avoid_replies=avoid_replies,
                rotation_index=len(avoid_replies),
            )
            chosen_suffix = chosen_reply
            chosen_generated = chosen_reply
            chosen_confidence = 0.0
        else:
            chosen_reply = last_reply
            chosen_suffix = last_suffix
            chosen_generated = last_generated
            chosen_confidence = last_confidence
    if base_seed:
        torch.manual_seed(base_seed)
    return chosen_reply, chosen_suffix, chosen_generated, chosen_confidence


@torch.no_grad()
def generate_text_tracked(
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
) -> tuple[str, float]:
    """Like ``generate_text``, but also returns the mean probability the model
    assigned to the tokens it chose, as a rough confidence signal.

    The probability is computed under the *raw* (no temperature, no
    repetition penalty, no blocked-id mask) softmax of the next-token
    logits. This way a high temperature does not artificially deflate the
    confidence value, and the score reflects the model's own beliefs.
    """
    text, probs = _run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        stop_at_period=stop_at_period,
        stop_at_blank_line=stop_at_blank_line,
        min_new_chars_before_stop=min_new_chars_before_stop,
        device=device,
        stop_sequences=stop_sequences,
        track_probs=True,
    )
    if probs:
        mean_conf = sum(probs) / len(probs)
    else:
        mean_conf = 0.0
    return text, mean_conf


@torch.no_grad()
def _run_generation(
    *,
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
    track_probs: bool = False,
) -> tuple[str, list[float]]:
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = tokenizer.encode(" ")
    if not tokens:
        raise ValueError("Tokenizer failed to encode fallback prompt")

    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    blocked_token_ids = blocked_generation_token_ids(tokenizer)
    stop_args = argparse.Namespace(
        stop_at_period=stop_at_period,
        stop_at_blank_line=stop_at_blank_line,
        min_new_chars_before_stop=min_new_chars_before_stop,
    )

    chosen_probs: list[float] = []

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_length :]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]

        # Snapshot the raw distribution before temperature / penalties /
        # blocking, so the confidence reflects the model's own beliefs.
        if track_probs:
            raw_probs = torch.softmax(next_token_logits, dim=-1)
        else:
            raw_probs = None

        recent_tokens = (
            idx[0, -repetition_window:].tolist() if repetition_window > 0 else []
        )
        next_token_logits = apply_repetition_penalty(
            next_token_logits,
            recent_tokens,
            repetition_penalty,
        )
        next_token_logits = suppress_token_ids(next_token_logits, blocked_token_ids)

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

        if raw_probs is not None:
            chosen_probs.append(float(raw_probs[0, int(next_token.item())].item()))

        idx = torch.cat((idx, next_token), dim=1)
        generated = tokenizer.decode(idx[0].tolist())
        suffix = generated_suffix(prompt, generated, tokenizer=tokenizer)
        if contains_stop_sequence(suffix, stop_sequences):
            break
        if should_stop_early(suffix, stop_args):
            break

    return tokenizer.decode(idx[0].tolist()), chosen_probs


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
    text, _ = _run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        stop_at_period=stop_at_period,
        stop_at_blank_line=stop_at_blank_line,
        min_new_chars_before_stop=min_new_chars_before_stop,
        device=device,
        stop_sequences=stop_sequences,
        track_probs=False,
    )
    return text


def generated_suffix(
    prompt: str,
    generated: str,
    tokenizer: Tokenizer | None = None,
) -> str:
    prompt_variants = [prompt]
    if tokenizer is not None:
        prompt_variants.append(tokenizer.decode(tokenizer.encode(prompt)))
    for prompt_variant in dedupe_texts(prompt_variants):
        if generated.startswith(prompt_variant):
            return generated[len(prompt_variant) :]
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

    memory_store: MemoryStore | None = None
    memory_enabled = (
        getattr(args, "use_memory", True)
        and args.user_label is not None
        and args.reply_label is not None
    )
    if memory_enabled:
        memory_path = getattr(args, "memory_db", None) or str(default_memory_path())
        try:
            memory_store = MemoryStore(memory_path)
            entry_count = len(memory_store.list_all())
            print(f"memory-db: {memory_path} ({entry_count} entries)")
        except Exception as exc:
            print(f"memory-db: disabled (could not open {memory_path}: {exc})")
            memory_store = None

    memory_inject_limit = max(0, int(getattr(args, "memory_inject", DEFAULT_INJECT_LIMIT)))
    language_guard = bool(getattr(args, "language_guard", True))
    unknown_threshold = float(
        getattr(
            args,
            "unknown_confidence_threshold",
            DEFAULT_UNKNOWN_CONFIDENCE_THRESHOLD,
        )
    )

    history = ""
    recent_replies: list[str] = []
    # Word the user wrote on a previous turn that we asked them to explain.
    pending_unknown_word: str | None = None
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
            recent_replies = []
            print("(reset)")
            continue
        if memory_store is not None:
            if command == ":memory":
                entries = memory_store.list_all()
                if not entries:
                    print("(memory is empty)")
                else:
                    for entry in entries:
                        print(
                            f"  #{entry.id} [{entry.importance}] "
                            f"{entry.key} = {entry.value} "
                            f"(updated {entry.updated_at})"
                        )
                continue
            if command == ":memory clear":
                deleted = memory_store.clear()
                print(f"(cleared {deleted} entries)")
                continue
            remember_parsed = parse_remember_command(command)
            if remember_parsed is not None:
                key, value, importance = remember_parsed
                entry = memory_store.add(key, value, importance)
                print(f"(remembered #{entry.id}: {entry.key} = {entry.value}, importance={entry.importance})")
                continue
            forget_parsed = parse_forget_command(command)
            if forget_parsed is not None:
                kind, target = forget_parsed
                if kind == "id":
                    if not target.isdigit():
                        print(f"(usage: :forget <id>; got {target!r})")
                        continue
                    if memory_store.delete(int(target)):
                        print(f"(forgot #{target})")
                    else:
                        print(f"(no entry with id {target})")
                else:
                    deleted = memory_store.delete_by_key(target)
                    if deleted:
                        print(f"(forgot {deleted} entries with key {target!r})")
                    else:
                        print(f"(no entries with key {target!r})")
                continue
        if command == ":help":
            print("テキストを入力すると返答を生成")
            print(":reset -> この起動中の会話履歴を消す")
            print(":quit  -> 終了")
            if memory_store is not None:
                print(":memory               -> 記憶している項目を一覧表示")
                print(":memory clear         -> 記憶を全消去")
                print(":remember <key> <value> [1-5] -> 記憶を追加（1-5は重要度、既定3）")
                print(":forget <id>          -> id 指定で記憶を消す")
                print(":forget-key <key>     -> key 指定で記憶を消す")
            print("upgrade -> uv tool upgrade original-llm")
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
        avoid_replies = tuple(recent_replies[-3:])

        # 1) Hard-coded guard: pure non-Japanese input gets the "Japanese only"
        #    reply, no matter what the model would have said.
        if (
            language_guard
            and args.user_label is not None
            and args.reply_label is not None
            and is_non_japanese_input(prepared_user_input)
        ):
            reply_text = NON_JAPANESE_GUARD_REPLY
            pending_unknown_word = None
            print()
            if show_prompt_output:
                guard_prompt = build_interactive_prompt(
                    user_input=prepared_user_input,
                    history=history,
                    args=args,
                    tokenizer=tokenizer,
                    context_length=model.config.context_length,
                )
                print_block("prompt", guard_prompt)
                print()
                print_block("output", reply_text)
            else:
                print(f"{role_prompt(args.reply_label)}{reply_text}".rstrip())
            print()
            recent_replies.append(reply_text)
            recent_replies = recent_replies[-5:]
            continue

        # 2) If the previous turn ended with us asking "what's that?", treat
        #    this turn as the user's explanation (or retraction).
        if (
            memory_store is not None
            and pending_unknown_word is not None
            and args.user_label is not None
            and args.reply_label is not None
        ):
            if looks_like_typo_correction(prepared_user_input):
                # Drop the pending word. Fall through to normal handling so
                # the user can keep talking right away.
                pending_unknown_word = None
            else:
                explanation = prepared_user_input.strip()
                key = f"{UNKNOWN_MEMORY_KEY_PREFIX}{pending_unknown_word}"
                stored = memory_store.bump_or_add(key, explanation)
                rotation = len(recent_replies)
                template = UNKNOWN_WORD_LEARNED_TEMPLATES[
                    rotation % len(UNKNOWN_WORD_LEARNED_TEMPLATES)
                ]
                reply_text = template.format(word=pending_unknown_word)
                print()
                if show_prompt_output:
                    print_block(
                        "prompt",
                        f"(unknown-word follow-up: stored '{stored.key}'='{stored.value}',"
                        f" importance={stored.importance})",
                    )
                    print()
                    print_block("output", reply_text)
                else:
                    print(f"{role_prompt(args.reply_label)}{reply_text}".rstrip())
                print()
                pending_unknown_word = None
                if args.carry_context:
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
                recent_replies.append(reply_text)
                recent_replies = recent_replies[-5:]
                continue

        # 3) "How do I update?" question gets a curated answer with the
        #    actual upgrade command. The model shouldn't be inventing CLI.
        if (
            args.user_label is not None
            and args.reply_label is not None
            and looks_like_update_question(prepared_user_input)
        ):
            reply_text = UPDATE_GUIDE_REPLY
            print()
            if show_prompt_output:
                upgrade_prompt = build_interactive_prompt(
                    user_input=prepared_user_input,
                    history=history,
                    args=args,
                    tokenizer=tokenizer,
                    context_length=model.config.context_length,
                )
                print_block("prompt", upgrade_prompt)
                print()
                print_block("output", reply_text)
            else:
                print(f"{role_prompt(args.reply_label)}{reply_text}".rstrip())
            print()
            if args.carry_context:
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
            recent_replies.append(reply_text)
            recent_replies = recent_replies[-5:]
            continue

        direct_reply = curated_short_reply(
            prepared_user_input,
            avoid_replies=avoid_replies,
            rotation_index=len(recent_replies),
        )
        if direct_reply is None:
            direct_reply = select_direct_chat_reply(
                prepared_user_input,
                args,
                avoid_replies=avoid_replies,
            )
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
            recent_replies.append(reply_text)
            recent_replies = recent_replies[-5:]
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
        memory_block: str = ""
        if memory_store is not None and memory_inject_limit > 0:
            relevant = memory_store.select_relevant(
                prepared_user_input,
                limit=memory_inject_limit,
            )
            memory_block = format_memory_block(
                relevant,
                args.user_label,
                args.reply_label,
            )
        if memory_block:
            prompt = trim_text_to_context(
                f"{memory_block}\n\n{prompt}",
                tokenizer,
                model.config.context_length,
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

        reply_text, suffix, generated, confidence = generate_chat_reply_with_resample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            args=args,
            device=device,
            user_input=prepared_user_input,
            avoid_replies=avoid_replies,
        )

        # If the user typed a short term that doesn't match anything in
        # memory and the model couldn't muster real confidence, ask them.
        memory_known = (
            memory_store is not None
            and memory_store.contains_word(prepared_user_input)
        )
        single_word = looks_like_single_word(prepared_user_input)
        if show_prompt_output:
            print(
                f"(confidence={confidence:.3f} "
                f"single_word={single_word} memory_known={memory_known})"
            )
        if (
            memory_store is not None
            and single_word
            and not memory_known
            and confidence < unknown_threshold
            and args.reply_label is not None
        ):
            rotation = len(recent_replies)
            template = UNKNOWN_WORD_QUESTION_TEMPLATES[
                rotation % len(UNKNOWN_WORD_QUESTION_TEMPLATES)
            ]
            reply_text = template.format(word=prepared_user_input)
            pending_unknown_word = prepared_user_input
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
        if reply_text:
            recent_replies.append(reply_text)
            recent_replies = recent_replies[-5:]

    if memory_store is not None:
        memory_store.close()


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
        stop_sequences=chat_stop_sequences(
            args.user_label,
            args.reply_label,
            tokenizer=tokenizer,
        ),
    )
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
