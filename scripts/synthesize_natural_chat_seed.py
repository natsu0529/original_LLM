"""Generate a curated synthetic chat seed corpus covering everyday topics.

Each output file contains one or more `私: ... / 相手: ... / <eot>` blocks. The
output is purely rule based: a small set of topic-specific user/reply pairs is
expanded over slot fillers (food items, places, weather words, etc.) plus a
handful of free-form one-shot pairs that the auto-built corpus tends to miss.

The aim is to add coverage for inputs the current model breaks on:

  - 食事の予定 ("ラーメン食べたい" / "今日ラーメン食べた")
  - 出かける予定 ("映画でも見ようかな" / "ピクニック行きたい")
  - 軽い感想 ("天気いいね" / "雨降ってきた")

so that we can retrain on a richer, less noisy corpus.

Usage:

  uv run python scripts/synthesize_natural_chat_seed.py \
      --out-dir data/chat_seed_friend_synth_v1 \
      --target-blocks 8000

The generator deterministically produces the same set per --seed.
"""

from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from original_llm.config import CHAT_TURN_END_MARKER

# ---------------------------------------------------------------------------
# slot fillers
# ---------------------------------------------------------------------------

FOODS_SAVORY = (
    "ラーメン",
    "うどん",
    "そば",
    "カレー",
    "牛丼",
    "親子丼",
    "オムライス",
    "ハンバーグ",
    "唐揚げ",
    "餃子",
    "焼きそば",
    "お好み焼き",
    "天ぷら",
    "寿司",
    "刺身",
    "焼き魚",
    "鍋",
    "おでん",
    "サンドイッチ",
    "ピザ",
    "パスタ",
    "サラダ",
    "味噌汁",
    "おにぎり",
)
FOODS_SWEET = (
    "ケーキ",
    "プリン",
    "アイス",
    "クッキー",
    "ドーナツ",
    "チョコ",
    "あんみつ",
    "わらび餅",
    "シュークリーム",
    "メロンパン",
    "おまんじゅう",
    "ぜんざい",
)
DRINKS = (
    "コーヒー",
    "紅茶",
    "緑茶",
    "ほうじ茶",
    "ココア",
    "ミルク",
    "ジュース",
    "炭酸水",
    "白湯",
    "麦茶",
)
PLACES = (
    "公園",
    "海",
    "山",
    "川沿い",
    "商店街",
    "駅前",
    "図書館",
    "本屋",
    "カフェ",
    "コンビニ",
    "スーパー",
    "銭湯",
    "美術館",
    "映画館",
    "古着屋",
)
WEATHER_GOOD = (
    "晴れ",
    "ぽかぽか",
    "気持ちよく晴れ",
    "雲ひとつなく晴れ",
    "風が気持ちいい",
)
WEATHER_BAD = (
    "雨",
    "土砂降り",
    "ぐずついた天気",
    "風が強い",
    "じめじめした天気",
    "曇り",
)
HOBBIES = (
    "読書",
    "散歩",
    "映画鑑賞",
    "音楽を聴くこと",
    "絵を描くこと",
    "写真を撮ること",
    "料理",
    "ゲーム",
    "観葉植物のお世話",
    "ストレッチ",
    "カフェ巡り",
)
ANIMALS = (
    "犬",
    "猫",
    "うさぎ",
    "ハムスター",
    "金魚",
    "鳥",
    "カピバラ",
    "ペンギン",
    "ねこ",
)
TIME_OF_DAY = (
    ("朝", "おはよう"),
    ("昼", "こんにちは"),
    ("夕方", "おつかれさま"),
    ("夜", "こんばんは"),
    ("深夜", "まだ起きてたんだ"),
)


@dataclass(frozen=True, slots=True)
class TurnPair:
    user: str
    reply: str


def _format_block(turns: Sequence[TurnPair]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"私: {turn.user}")
        lines.append(f"相手: {turn.reply}")
        lines.append(CHAT_TURN_END_MARKER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# pattern generators
# ---------------------------------------------------------------------------


def _product_pairs(
    user_templates: Sequence[str],
    reply_templates: Sequence[str],
    items: Sequence[str],
    item_key: str,
) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    for item in items:
        for u_t in user_templates:
            for r_t in reply_templates:
                user = u_t.format(**{item_key: item})
                reply = r_t.format(**{item_key: item})
                pairs.append(TurnPair(user, reply))
    return pairs


def gen_food_eaten(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    user_templates_savory = (
        "今日{food}食べた",
        "{food}食べちゃった",
        "お昼に{food}食べたよ",
        "{food}食べてきた",
        "晩ごはん{food}にした",
    )
    reply_templates_savory = (
        "いいね、{food}おいしいよね。",
        "おっ、{food}いい選択だ。元気でた？",
        "{food}か、安心する味だよね。",
        "うらやましい。今度連れてって。",
        "おいしそう。お腹すいてきた。",
    )
    pairs.extend(
        _product_pairs(
            user_templates_savory,
            reply_templates_savory,
            FOODS_SAVORY,
            "food",
        )
    )

    user_templates_sweet = (
        "{food}食べちゃった",
        "おやつに{food}食べた",
        "{food}買ってきたよ",
    )
    reply_templates_sweet = (
        "いいね、{food}は元気でる。",
        "それは幸せ時間だ。",
        "ちょっとうらやましい。",
        "甘いものは正義だよ。",
    )
    pairs.extend(
        _product_pairs(
            user_templates_sweet,
            reply_templates_sweet,
            FOODS_SWEET,
            "food",
        )
    )
    return pairs


def gen_food_want(rng: random.Random) -> list[TurnPair]:
    foods = FOODS_SAVORY + FOODS_SWEET
    user_templates = (
        "{food}食べたい",
        "{food}食べたいなー",
        "なんか{food}の気分",
        "今日{food}にしようかな",
    )
    reply_templates = (
        "いいね、食べちゃおうよ。",
        "{food}か、おいしそう。",
        "わかる、{food}は安心する。",
        "じゃあ買いに行こうか。",
        "気分出てきたなら食べた方がいいよ。",
    )
    return _product_pairs(user_templates, reply_templates, foods, "food")


def gen_drinks(rng: random.Random) -> list[TurnPair]:
    user_templates = (
        "{drink}飲みたい",
        "{drink}いれたよ",
        "今日は{drink}の気分",
    )
    reply_templates = (
        "いいね、{drink}落ち着くよね。",
        "おっ、ちょうど私も飲みたかった。",
        "{drink}か、ほっとするよね。",
        "あったかいの飲むと一息つける。",
    )
    return _product_pairs(user_templates, reply_templates, DRINKS, "drink")


def gen_places(rng: random.Random) -> list[TurnPair]:
    user_templates = (
        "{place}行きたいな",
        "今日{place}行ってきた",
        "{place}でぼーっとしたい",
        "{place}いいところだよ",
    )
    reply_templates = (
        "いいね、{place}気持ちよさそう。",
        "{place}か、たまに行くといいよね。",
        "うらやましい。私も行きたい。",
        "今度一緒に行ってみよっか。",
        "天気のいい日に行くと最高だよ。",
    )
    return _product_pairs(user_templates, reply_templates, PLACES, "place")


def gen_weather(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    good_user_templates = (
        "今日{weather}だね",
        "{weather}で気持ちいい",
        "外{weather}だよ",
    )
    good_replies = (
        "ほんと、こんな日は出かけたくなる。",
        "いいね、洗濯物よく乾きそう。",
        "気持ちいいよね。少し散歩したい。",
        "こういう日は何しても気分いい。",
    )
    for w in WEATHER_GOOD:
        for u_t in good_user_templates:
            r = rng.choice(good_replies)
            pairs.append(TurnPair(u_t.format(weather=w), r))

    bad_user_templates = (
        "{weather}だね",
        "外{weather}",
        "今日は{weather}",
        "{weather}でだるい",
    )
    bad_replies = (
        "うん、こういう日は無理しないでいいよ。",
        "ね、出かけるの億劫だよね。",
        "{weather}か、家でゆっくりしよ。",
        "あったかくしてね。",
        "おうち時間にしようか。",
    )
    for w in WEATHER_BAD:
        for u_t in bad_user_templates:
            r_t = rng.choice(bad_replies)
            pairs.append(TurnPair(u_t.format(weather=w), r_t.format(weather=w)))

    pairs.append(TurnPair("雨降ってきた", "うん、傘持って行ってね。"))
    pairs.append(TurnPair("雨降ってきた", "あー、洗濯物気になるね。"))
    pairs.append(TurnPair("雪降ってきた", "おお、寒いから気をつけてね。"))
    pairs.append(TurnPair("急に寒くなったね", "ね、体調崩さないようにしよ。"))
    pairs.append(TurnPair("急に暑くなったね", "ね、水分とってよ。"))
    return pairs


def gen_movie_picnic_outings(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    movie_users = (
        "映画でも見ようかな",
        "映画見たい気分",
        "今日映画見るか迷ってる",
        "映画館行きたい",
    )
    movie_replies = (
        "いいね、なに見るか決めた？",
        "映画館で見るとちがうよね。",
        "おすすめあるなら教えて。",
        "気分転換にちょうどいいかも。",
        "ポップコーン買って行きなよ。",
    )
    for u in movie_users:
        for r in movie_replies:
            pairs.append(TurnPair(u, r))

    picnic_users = (
        "ピクニック行きたいな",
        "今度ピクニックしようよ",
        "天気いいしピクニックしたい",
    )
    picnic_replies = (
        "いいね、お弁当作っていこ。",
        "公園でやろうか、気持ちよさそう。",
        "天気いい日に行きたいね。",
        "サンドイッチ作っていく？",
    )
    for u in picnic_users:
        for r in picnic_replies:
            pairs.append(TurnPair(u, r))

    drive_users = (
        "ドライブ行きたい",
        "ドライブしようよ",
    )
    drive_replies = (
        "いいね、海沿い走ろっか。",
        "音楽流してのんびり行きたい。",
        "気分転換になりそう。",
    )
    for u in drive_users:
        for r in drive_replies:
            pairs.append(TurnPair(u, r))
    return pairs


def gen_hobbies(rng: random.Random) -> list[TurnPair]:
    user_templates = (
        "趣味は{hobby}",
        "最近{hobby}にハマってる",
        "{hobby}って楽しいよ",
    )
    reply_templates = (
        "いいね、{hobby}は心が落ち着くよね。",
        "{hobby}か、私もちょっと興味ある。",
        "おすすめあったら教えて。",
        "続けてるのえらいなあ。",
    )
    pairs = _product_pairs(user_templates, reply_templates, HOBBIES, "hobby")
    pairs += [
        TurnPair("最近何にハマってる？", "最近は本ばっかり読んでるよ。"),
        TurnPair("休みの日何してる？", "だいたい家でぼーっとしてる。"),
        TurnPair("休みの日何してる？", "近所散歩したり、本読んだり。"),
    ]
    return pairs


def gen_animals(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    for animal in ANIMALS:
        pairs.append(TurnPair(f"{animal}かわいい", f"わかる、{animal}は癒し。"))
        pairs.append(TurnPair(f"{animal}飼いたい", f"いいね、{animal}と暮らせたら楽しそう。"))
        pairs.append(TurnPair(f"{animal}好き", f"私も好き、{animal}いいよね。"))
    return pairs


def gen_mood_supportive(rng: random.Random) -> list[TurnPair]:
    bad_inputs = (
        "なんかつらい",
        "今日しんどい",
        "落ち込んでる",
        "やる気でない",
        "もうだめかも",
        "なんもしたくない",
        "気分が重い",
        "心折れそう",
        "ちょっと泣きそう",
        "イライラする",
        "ムカつくことあった",
    )
    bad_replies = (
        "うん、無理しなくていいよ。",
        "そっか、話せる範囲で聞くよ。",
        "ここにいるから、ゆっくりしてて。",
        "深呼吸しよ、まずひと息ついて。",
        "頑張ってるの伝わってるからね。",
        "そういう日もあるよ、責めないで。",
        "ちょっと水でも飲んで休もう。",
    )
    good_inputs = (
        "今日いい日だった",
        "ちょっと嬉しいことあった",
        "うまくいった",
        "ほめられちゃった",
        "目標達成した",
    )
    good_replies = (
        "おお、よかったね。",
        "それは嬉しい、聞かせて。",
        "がんばった甲斐あったね。",
        "よかった、こっちも嬉しい。",
    )
    pairs: list[TurnPair] = []
    for u in bad_inputs:
        for r in bad_replies:
            pairs.append(TurnPair(u, r))
    for u in good_inputs:
        for r in good_replies:
            pairs.append(TurnPair(u, r))
    return pairs


def gen_sleep(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    pairs += [
        TurnPair("眠れない", "深呼吸ゆっくりしてみよ。"),
        TurnPair("眠れない", "考え事おやすみして、横になってみて。"),
        TurnPair("眠れない", "白湯でも飲んでみる？"),
        TurnPair("ねむい", "そろそろ寝た方がいいよ。"),
        TurnPair("ねむい", "おふとんでぬくぬくしよ。"),
        TurnPair("もう寝る", "おやすみ、ゆっくりね。"),
        TurnPair("もう寝るわ", "おやすみ、また明日。"),
        TurnPair("おやすみ", "おやすみ、いい夢見てね。"),
        TurnPair("おやすみ", "ゆっくり休んでね。"),
        TurnPair("夢見た", "へえ、どんな夢？"),
        TurnPair("変な夢見た", "それはちょっと聞きたい。"),
        TurnPair("寝坊した", "あらら、急いでね。"),
        TurnPair("寝過ごした", "そういう日もあるよ。"),
    ]
    return pairs


def gen_work_study(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    inputs = (
        "仕事だるい",
        "仕事終わった",
        "残業つらい",
        "勉強したくない",
        "勉強がんばった",
        "テスト終わった",
        "ミーティング多い",
        "在宅つらい",
        "会議疲れた",
    )
    pos_replies = (
        "おつかれさま、ゆっくりしてね。",
        "がんばったね、すごい。",
        "今日はもう休もう。",
    )
    neg_replies = (
        "わかる、ちょっと休憩しよ。",
        "おつかれ、無理しないで。",
        "息抜き大事だよ。",
        "深呼吸しよ。",
    )
    for u in inputs:
        replies = pos_replies if "終わった" in u or "がんばった" in u else neg_replies
        for r in replies:
            pairs.append(TurnPair(u, r))
    return pairs


def gen_greetings_and_smalltalk(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    for _, greeting in TIME_OF_DAY:
        pairs.append(TurnPair(greeting, f"{greeting}、今日はどう？"))
        pairs.append(TurnPair(greeting, f"{greeting}、調子どう？"))
    pairs += [
        TurnPair("元気？", "うん、まあまあ。そっちは？"),
        TurnPair("元気？", "ぼちぼちかな。そっちは？"),
        TurnPair("最近どう？", "いつも通りだよ、そっちは？"),
        TurnPair("最近どう？", "そこそこかな。なんか変わったことあった？"),
        TurnPair("ひさしぶり", "ね、元気にしてた？"),
        TurnPair("ひさしぶり", "ほんと、会えて嬉しい。"),
        TurnPair("ありがとう", "どういたしまして。"),
        TurnPair("ありがとう", "うん、いつでも頼ってね。"),
        TurnPair("ありがとう", "気にしないで、こっちこそ。"),
        TurnPair("ごめん", "ううん、気にしないで。"),
        TurnPair("ごめんね", "うん、もう大丈夫。"),
        TurnPair("じゃあね", "うん、またね。気をつけて。"),
        TurnPair("またね", "うん、またあとで。"),
        TurnPair("ただいま", "おかえり、おつかれさま。"),
        TurnPair("ただいま", "おかえり、ゆっくりして。"),
    ]
    return pairs


def gen_yes_no_short(rng: random.Random) -> list[TurnPair]:
    pairs: list[TurnPair] = []
    pairs += [
        TurnPair("そうかな", "うん、そう思うよ。"),
        TurnPair("ほんとに？", "うん、ほんと。"),
        TurnPair("マジで？", "うん、マジで。"),
        TurnPair("やばい", "どうした？大丈夫？"),
        TurnPair("やばい", "なになに、聞くよ。"),
        TurnPair("わかる", "ね、わかってくれて嬉しい。"),
        TurnPair("わかった", "うん、よろしく。"),
        TurnPair("わからない", "そっか、いっしょに考えよ。"),
        TurnPair("どうしよう", "ひと息ついて、ゆっくり考えよ。"),
        TurnPair("助けて", "うん、なに困ってる？"),
        TurnPair("聞いて", "うん、聞くよ。"),
        TurnPair("見て見て", "おお、なになに？"),
    ]
    return pairs


def gen_two_turn_followups(rng: random.Random) -> list[Sequence[TurnPair]]:
    """Multi-turn conversational blocks (2 turns each)."""
    blocks: list[Sequence[TurnPair]] = []
    blocks.append(
        (
            TurnPair("今日疲れた", "おつかれさま、なにかあった？"),
            TurnPair("仕事忙しかった", "そっか、ゆっくり休んでね。"),
        )
    )
    blocks.append(
        (
            TurnPair("お腹すいた", "なにか食べよっか。"),
            TurnPair("ラーメン食べたいな", "いいね、行こう。"),
        )
    )
    blocks.append(
        (
            TurnPair("眠れない", "考え事しちゃってる？"),
            TurnPair("うん、ちょっと不安で", "そっか、ここにいるよ。深呼吸しよ。"),
        )
    )
    blocks.append(
        (
            TurnPair("映画見ようかな", "いいね、なに見るの？"),
            TurnPair("まだ決めてない", "じゃあおすすめ調べよっか。"),
        )
    )
    blocks.append(
        (
            TurnPair("ピクニック行きたい", "いいね、どこ行く？"),
            TurnPair("近くの公園", "じゃあお弁当作って行こ。"),
        )
    )
    blocks.append(
        (
            TurnPair("散歩してきた", "気持ちよかった？"),
            TurnPair("うん、晴れてた", "よかったね、リフレッシュできたね。"),
        )
    )
    blocks.append(
        (
            TurnPair("元気？", "うん、まあまあ。そっちは？"),
            TurnPair("ぼちぼち", "そっか、お互いほどほどに。"),
        )
    )
    blocks.append(
        (
            TurnPair("こんにちは", "こんにちは、今日はどう？"),
            TurnPair("ふつうかな", "いいね、ふつうが一番だよ。"),
        )
    )
    blocks.append(
        (
            TurnPair("おはよう", "おはよう、よく眠れた？"),
            TurnPair("まあまあ", "そっか、今日はゆっくりやろう。"),
        )
    )
    blocks.append(
        (
            TurnPair("おやすみ", "おやすみ、いい夢みてね。"),
            TurnPair("ありがとう", "うん、また明日ね。"),
        )
    )
    blocks.append(
        (
            TurnPair("つらい", "うん、無理しないで。話せる？"),
            TurnPair("仕事で怒られた", "それはきついね、よくがんばってる。"),
        )
    )
    blocks.append(
        (
            TurnPair("嬉しいことあった", "おお、なに？聞きたい。"),
            TurnPair("褒められた", "それは嬉しいね、こっちも嬉しい。"),
        )
    )
    blocks.append(
        (
            TurnPair("カフェ行きたい", "いいね、どんな気分？"),
            TurnPair("コーヒー飲みたい", "じゃあのんびりできるところ行こ。"),
        )
    )
    blocks.append(
        (
            TurnPair("買い物行く", "なに買うの？"),
            TurnPair("食材", "いいね、なに作るの？"),
        )
    )
    blocks.append(
        (
            TurnPair("ねえ、最近何してる？", "最近は本ばっかり読んでる。そっちは？"),
            TurnPair("私はゲームばっかり", "いいね、なんのゲーム？"),
        )
    )
    return blocks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-blocks", type=int, default=8000)
    parser.add_argument("--blocks-per-file", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260503)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    single_pairs: list[TurnPair] = []
    for builder in (
        gen_food_eaten,
        gen_food_want,
        gen_drinks,
        gen_places,
        gen_weather,
        gen_movie_picnic_outings,
        gen_hobbies,
        gen_animals,
        gen_mood_supportive,
        gen_sleep,
        gen_work_study,
        gen_greetings_and_smalltalk,
        gen_yes_no_short,
    ):
        single_pairs.extend(builder(rng))

    # de-duplicate by exact (user, reply) pair
    seen: set[tuple[str, str]] = set()
    unique_pairs: list[TurnPair] = []
    for pair in single_pairs:
        key = (pair.user, pair.reply)
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)

    rng.shuffle(unique_pairs)

    multi_blocks = gen_two_turn_followups(rng)

    # Build blocks: most are single-turn, every ~10th is multi-turn.
    blocks_text: list[str] = []
    multi_iter = itertools.cycle(multi_blocks)
    for index, pair in enumerate(unique_pairs):
        blocks_text.append(_format_block([pair]))
        if index % 10 == 9:
            blocks_text.append(_format_block(next(multi_iter)))
        if len(blocks_text) >= args.target_blocks:
            break

    # If we still have headroom, append the remaining multi-turn blocks once.
    if len(blocks_text) < args.target_blocks:
        for block in multi_blocks:
            blocks_text.append(_format_block(block))
            if len(blocks_text) >= args.target_blocks:
                break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0
    for chunk_idx, start in enumerate(range(0, len(blocks_text), args.blocks_per_file)):
        chunk = blocks_text[start : start + args.blocks_per_file]
        out_path = args.out_dir / f"synth_{chunk_idx:05d}.txt"
        out_path.write_text("\n\n".join(chunk) + "\n", encoding="utf-8")
        files_written += 1

    print(
        "wrote",
        files_written,
        "files,",
        len(blocks_text),
        "blocks ->",
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
