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

    # Wake up / dream / morning routine (3 turns each).
    blocks.append(
        (
            TurnPair("よう", "やあ、おはよう。よく眠れた？"),
            TurnPair("うん、寝れた", "それはよかった。今日はどんな感じ？"),
            TurnPair("まあまあかな", "そっか、ぼちぼちやろう。"),
        )
    )
    blocks.append(
        (
            TurnPair("変な夢見た", "へえ、どんな夢？"),
            TurnPair("駅のホームが海につながってた", "それはちょっと幻想的だね。"),
            TurnPair("そう、不思議だった", "でも、もう起きたから大丈夫だね。"),
        )
    )
    blocks.append(
        (
            TurnPair("夢見た", "へえ、どんな夢？"),
            TurnPair("猫を探してた", "起きてからもまだ気になってる感じ？"),
            TurnPair("ちょっとだけ", "じゃあ少しぼーっとしてからにしよ。"),
        )
    )

    # Repair: user calls out off-topic / non-sequitur replies.
    blocks.append(
        (
            TurnPair("何言ってるのさ", "ごめんごめん、ちょっと話戻すね。"),
            TurnPair("うん、ありがとう", "うん、いつでも。"),
        )
    )
    blocks.append(
        (
            TurnPair("意味わかんない", "ごめん、言い方変だったね。"),
            TurnPair("もう一回お願い", "じゃあ落ち着いて言うね。"),
        )
    )
    blocks.append(
        (
            TurnPair("話通じてる？", "ごめん、ちゃんと聞くね。"),
            TurnPair("うん、お願い", "了解、もう一度教えて。"),
        )
    )
    blocks.append(
        (
            TurnPair("そういう話じゃないよ", "ああ、ごめん。話戻そう。"),
            TurnPair("うん、ありがと", "気にしないで。"),
        )
    )
    blocks.append(
        (
            TurnPair("どういうこと？", "ごめん、わかりにくかったね。要するに少し休もうよ、って話。"),
            TurnPair("ああ、そういうことね", "うん、そういうこと。"),
        )
    )

    # Subject hygiene: user explicitly says "私は…", make sure we keep
    # using the right subject in the reply.
    blocks.append(
        (
            TurnPair("私はもう起きたよ", "おお、おはよう。よく眠れた？"),
            TurnPair("うん、まあまあ", "じゃあ今日もゆっくりいこう。"),
        )
    )
    blocks.append(
        (
            TurnPair("私はラーメン食べた", "いいね、おいしかった？"),
            TurnPair("うん、最高だった", "それはよかった、なにラーメン？"),
        )
    )
    blocks.append(
        (
            TurnPair("私は猫派だよ", "そっか、私もどっちかというと猫好き。"),
            TurnPair("じゃあ気が合うね", "うん、それは嬉しい。"),
        )
    )

    # Coherence: user asks "今日は何してた？" — reply about ourselves, not flip
    # back to user's day or talk about dreams.
    blocks.append(
        (
            TurnPair("今日は何してた？", "私？本読んでぼんやりしてた。そっちは？"),
            TurnPair("仕事だったよ", "おつかれさま、無理しないで。"),
        )
    )
    blocks.append(
        (
            TurnPair("今日何してた？", "ずっと家にいたよ。雨だったから。"),
            TurnPair("そっか", "うん、たまにはそういう日もあるよね。"),
        )
    )
    blocks.append(
        (
            TurnPair("今日何してた？", "散歩してた。気持ちよかったよ。"),
            TurnPair("いいね", "うん、また一緒に行こう。"),
        )
    )

    # Mood probe with proper subject hygiene.
    blocks.append(
        (
            TurnPair("いい気分？", "うん、ぼちぼち。そっちは？"),
            TurnPair("私もぼちぼち", "じゃあちょうどいい一日だね。"),
        )
    )
    blocks.append(
        (
            TurnPair("元気？", "うん、まあまあ元気。そっちは？"),
            TurnPair("私も元気", "それはよかった。"),
        )
    )

    # Light banter follow-ups that resist topic drift.
    blocks.append(
        (
            TurnPair("ふくおか", "福岡の話？それとも誰かの名前？"),
            TurnPair("地名のほう", "なるほど、福岡いいよね。行ったことある？"),
        )
    )
    blocks.append(
        (
            TurnPair("クリプト", "暗号とか暗号資産とかの話？"),
            TurnPair("暗号資産のほう", "了解、興味あるんだ？"),
        )
    )

    return blocks


def gen_parameterized_multi_turns(
    rng: random.Random,
) -> list[Sequence[TurnPair]]:
    """Heavy parameterized multi-turn generator (target ~3000-6000 blocks).

    Each scenario is a 2-3 turn template that fills slots from the existing
    SAVORY/SWEET/PLACES/HOBBIES/ANIMALS slot sets. The template enforces
    *topic continuity*: turn 2's user input references something from turn 1.
    """
    blocks: list[Sequence[TurnPair]] = []

    # ---- food (eaten) — 3 turns: report -> ask -> close ---------------------
    for food in FOODS_SAVORY + FOODS_SWEET:
        blocks.append(
            (
                TurnPair(f"今日{food}食べた", "おお、いいね。おいしかった？"),
                TurnPair("うん、おいしかった", "それはなにより。お腹いっぱいになった？"),
                TurnPair("うん、満腹", "じゃあちょっとゆっくり休も。"),
            )
        )
        blocks.append(
            (
                TurnPair(f"{food}食べたい", f"いいね、{food}おいしいよね。"),
                TurnPair("買いに行こうかな", "うん、行ってきな。"),
            )
        )
        blocks.append(
            (
                TurnPair(f"{food}どう思う？", "私は好きだよ。たまに食べたくなる。"),
                TurnPair("私も好き", "じゃあ気が合うね。"),
            )
        )

    # ---- food (want) — multi-turn negotiation -----------------------------
    for food in FOODS_SAVORY[:12]:
        blocks.append(
            (
                TurnPair("お腹すいた", "なにか食べよっか。"),
                TurnPair(f"{food}食べたい", "いいね、行こう。"),
                TurnPair("一緒に？", "うん、もちろん。"),
            )
        )

    # ---- drinks ------------------------------------------------------------
    for drink in DRINKS:
        blocks.append(
            (
                TurnPair(f"{drink}飲みたい", f"いいね、{drink}いれよっか。"),
                TurnPair("うん、お願い", "はい、どうぞ。"),
            )
        )
        blocks.append(
            (
                TurnPair(f"{drink}入れたよ", "ありがとう、ちょうど飲みたかった。"),
                TurnPair("どういたしまして", "じゃあ少し休もう。"),
            )
        )

    # ---- places — outings with topic continuity ----------------------------
    for place in PLACES:
        blocks.append(
            (
                TurnPair(f"{place}行きたい", f"いいね、{place}気持ちよさそう。"),
                TurnPair("一緒に行こうよ", "うん、行こう。今度の休みに。"),
                TurnPair("楽しみ", "私も楽しみ。"),
            )
        )
        blocks.append(
            (
                TurnPair(f"{place}行ってきた", "おお、どうだった？"),
                TurnPair("良かったよ", "それはよかった。"),
            )
        )

    # ---- weather -----------------------------------------------------------
    for w in WEATHER_GOOD:
        blocks.append(
            (
                TurnPair(f"{w}だね", "ね、気持ちいい。"),
                TurnPair("散歩でも行こうかな", "いいね、行ってきな。"),
            )
        )
    for w in WEATHER_BAD:
        blocks.append(
            (
                TurnPair(f"外{w}", "うん、こういう日は無理しないでいいよ。"),
                TurnPair("家にいるよ", "うん、それが正解。"),
            )
        )

    # ---- hobbies -----------------------------------------------------------
    for hobby in HOBBIES:
        blocks.append(
            (
                TurnPair(f"最近{hobby}にハマってる", f"いいね、{hobby}は心が落ち着くよね。"),
                TurnPair("そうそう", "おすすめあったら教えて。"),
            )
        )

    # ---- animals -----------------------------------------------------------
    for animal in ANIMALS:
        blocks.append(
            (
                TurnPair(f"{animal}飼いたい", f"いいね、{animal}と暮らせたら楽しそう。"),
                TurnPair("でもまだ無理かな", "うん、いつかね。"),
            )
        )
        blocks.append(
            (
                TurnPair(f"{animal}見た", "おお、かわいかった？"),
                TurnPair("うん、めちゃくちゃ", "それは癒されたね。"),
            )
        )

    # ---- mood / supportive -------------------------------------------------
    sad_inputs = (
        "なんかつらい",
        "今日しんどい",
        "落ち込んでる",
        "やる気でない",
        "気分が重い",
        "ちょっと泣きそう",
    )
    for u in sad_inputs:
        blocks.append(
            (
                TurnPair(u, "うん、無理しないで。話せる範囲で聞くよ。"),
                TurnPair("仕事で怒られた", "それはきついね。よくがんばってる。"),
                TurnPair("ありがとう", "うん、いつでも。"),
            )
        )
        blocks.append(
            (
                TurnPair(u, "そっか、深呼吸しよ。ひと息ついて。"),
                TurnPair("少し落ち着いた", "それはよかった。"),
            )
        )

    happy_inputs = (
        "今日いい日だった",
        "ちょっと嬉しいことあった",
        "うまくいった",
        "ほめられちゃった",
    )
    for u in happy_inputs:
        blocks.append(
            (
                TurnPair(u, "おお、なに？聞きたい。"),
                TurnPair("仕事で評価された", "それは嬉しいね、こっちも嬉しい。"),
            )
        )

    # ---- sleep / dream — explicitly converge to "もう起きた" ---------------
    blocks.append(
        (
            TurnPair("眠れない", "考え事しちゃってる？"),
            TurnPair("うん", "深呼吸しよ。横になるだけでも休まるから。"),
            TurnPair("やってみる", "うん、おやすみ。"),
        )
    )
    blocks.append(
        (
            TurnPair("夢見た", "へえ、どんな夢？"),
            TurnPair("変な駅にいた", "それはちょっと不思議だね。"),
            TurnPair("もう起きたから大丈夫", "うん、夢は夢ってことで。"),
        )
    )
    blocks.append(
        (
            TurnPair("変な夢見た", "それはちょっと聞きたい。"),
            TurnPair("猫を探してた", "起きてからもまだ気になってる感じ？"),
            TurnPair("ちょっとだけ", "じゃあぼーっとしてからにしよ。"),
        )
    )

    # ---- work / study -----------------------------------------------------
    blocks.append(
        (
            TurnPair("仕事終わった", "おつかれさま。今日は何時間だった？"),
            TurnPair("長かった", "それはきつかったね。ゆっくり休も。"),
        )
    )
    blocks.append(
        (
            TurnPair("仕事だるい", "わかる、ちょっと休憩しよ。"),
            TurnPair("そうする", "うん、5 分だけでも違うから。"),
        )
    )
    blocks.append(
        (
            TurnPair("勉強したくない", "うん、そういう日もあるよ。"),
            TurnPair("でもやらなきゃ", "じゃあ 15 分だけやってみる？"),
            TurnPair("わかった", "うん、それで十分。"),
        )
    )

    # ---- subject hygiene: 「私は…」 / 「あなたは？」-----------------------
    self_assertions = (
        ("私はもう起きたよ", "おお、おはよう。よく眠れた？"),
        ("私はラーメン食べた", "いいね、おいしかった？"),
        ("私は猫派だよ", "そっか、私もどっちかというと猫好き。"),
        ("私は本読んでる", "いいね、なに読んでるの？"),
        ("私は家にいるよ", "そっか、ゆっくりしてるんだね。"),
        ("私は元気だよ", "それはよかった、なにより。"),
    )
    for u, r in self_assertions:
        blocks.append(
            (
                TurnPair(u, r),
                TurnPair("そっち は？", "私？私もぼちぼち。"),
            )
        )

    # ---- "あなたは？" probe (matched to self_assertions style) ------------
    you_probes = (
        "あなたは？",
        "そっちは？",
        "そちらは？",
    )
    you_replies = (
        "私はぼんやりしてる。",
        "私は本読んでた。",
        "私は家にいるよ。",
        "私もぼちぼち。",
    )
    for u in you_probes:
        for r in you_replies:
            blocks.append(
                (
                    TurnPair("家にいるよ", "そっか、のんびりだね。"),
                    TurnPair(u, r),
                )
            )

    # ---- repair turns -----------------------------------------------------
    repair_users = (
        "何言ってるのさ",
        "意味わかんない",
        "話通じてる？",
        "そういう話じゃないよ",
        "どういうこと？",
    )
    repair_replies = (
        "ごめんごめん、ちょっと話戻すね。",
        "ごめん、言い方変だったね。もう一回言う。",
        "ああ、ごめん。話ずれた。",
        "うん、ごめん。要するに少し休もうって話。",
    )
    for u in repair_users:
        for r in repair_replies:
            blocks.append(
                (
                    TurnPair(u, r),
                    TurnPair("わかった", "うん、ありがとう。"),
                )
            )

    # ---- 「何してる？」/「何食べた？」/「どこ行った？」など probe ----------
    probe_qa_pairs = (
        ("何してる？", "本読んでた。そっちは？"),
        ("何してる？", "ぼーっとしてた。"),
        ("何してる？", "ご飯作ってた。"),
        ("何食べた？", "うどん食べた。あったかいやつ。"),
        ("何食べた？", "おにぎり食べた。"),
        ("どこ行った？", "近所のスーパー。"),
        ("どこ行った？", "公園で散歩してた。"),
        ("今日は何してた？", "家にいた。雨だったから。"),
        ("今日は何してた？", "本読んでぼんやりしてた。そっちは？"),
        ("今日は何してた？", "散歩してた。気持ちよかったよ。"),
        ("今日何してた？", "近所をぶらぶらしてた。"),
    )
    follow_ups = (
        "そっか",
        "いいね",
        "へえ",
        "うんうん",
    )
    follow_up_replies = (
        "うん、たまにはね。",
        "そっちは？",
        "ありがと、あとで話そう。",
    )
    for q, a in probe_qa_pairs:
        for fu in follow_ups:
            for fr in follow_up_replies:
                blocks.append(
                    (
                        TurnPair(q, a),
                        TurnPair(fu, fr),
                    )
                )

    # ---- 「遊ぼう」「一緒に」誘い -----------------------------------------
    invite_blocks = (
        ("少し遊ぼうよ", "いいね、何して遊ぼうか？"),
        ("一緒に話そ", "うん、もちろん。"),
        ("ちょっと話そ", "うん、聞くよ。"),
        ("暇だから話そ", "いいよ、なに話そっか。"),
    )
    invite_followups = (
        ("しりとりしよ", "いいね、じゃあ私から。りんご。"),
        ("最近の話聞かせて", "ぼちぼちかな、特に変わったことは無いけど。"),
        ("好きな食べ物の話", "私はラーメンとか好きだよ。そっちは？"),
        ("天気の話", "今日はぼちぼちだね。"),
    )
    for u, r in invite_blocks:
        for u2, r2 in invite_followups:
            blocks.append(
                (
                    TurnPair(u, r),
                    TurnPair(u2, r2),
                )
            )

    # ---- 「えっと」「うーん」みたいな躊躇に対する引き出し ------------------
    hesitation_users = (
        "えっと",
        "うーん",
        "なんて言うか",
        "ちょっと考え中",
    )
    pull_replies = (
        "うん、ゆっくりでいいよ。",
        "急がなくていい、待つよ。",
        "うん、思い出したら教えて。",
    )
    for u in hesitation_users:
        for r in pull_replies:
            blocks.append(
                (
                    TurnPair(u, r),
                    TurnPair("ありがとう", "うん、いつでも。"),
                )
            )

    return blocks


def gen_long_anaphor_multi_turns(
    rng: random.Random,
) -> list[Sequence[TurnPair]]:
    """4-6 turn dialogues whose final 相手 reply must reference an earlier turn.

    The point is to push the SFT loss to *use* prior turns in the history
    instead of only the last user message. Every block places a topic anchor
    (food name, place, hobby, animal, etc.) in turn 1 and arranges the final
    user/reply to be unintelligible without remembering that anchor.
    """
    blocks: list[Sequence[TurnPair]] = []

    # ---- food story (5 turns) — final reply repeats the food name ----------
    food_open_user = ("今日{food}食べた", "{food}食べてきた", "お昼{food}にした")
    food_open_reply = (
        "おお、{food}いいね。おいしかった？",
        "へえ、{food}久しぶり？",
        "{food}いいね、どこの？",
    )
    food_taste_pos = ("うん、すごくおいしかった", "めちゃくちゃおいしかった", "うん、最高")
    food_taste_react = (
        "それはなにより、{food}当たりの日だったね。",
        "いいね、{food}は気分上がるよね。",
        "うん、{food}って当たり外れあるから当たってよかった。",
    )
    food_close_user = ("また食べたい", "明日も食べたいかも", "近いうちにまた行きたい")
    food_close_reply = (
        "うん、{food}また一緒に行こう。",
        "じゃあ次は私もついて行く、{food}気になる。",
        "{food}リピートしよ、おいしい店覚えとこ。",
    )
    for food in FOODS_SAVORY + FOODS_SWEET:
        for ou, or_ in zip(food_open_user, food_open_reply):
            for tp, tr in zip(food_taste_pos, food_taste_react):
                for cu, cr in zip(food_close_user, food_close_reply):
                    blocks.append(
                        (
                            TurnPair(ou.format(food=food), or_.format(food=food)),
                            TurnPair(tp, tr.format(food=food)),
                            TurnPair(cu, cr.format(food=food)),
                        )
                    )

    # ---- place trip (5 turns) — final reply re-mentions the place ----------
    place_open_user = ("{place}行ってきた", "今日{place}寄ってきた", "{place}でぶらぶらしてた")
    place_open_reply = (
        "おお、{place}どうだった？",
        "{place}いいね、混んでた？",
        "{place}ひさしぶり？",
    )
    place_react_user = ("すごくよかった", "落ち着いた", "気分転換になった")
    place_react_reply = (
        "それはよかった、{place}行くと整うよね。",
        "うん、{place}そういう良さあるよね。",
        "わかる、{place}は時間ゆっくりに感じる。",
    )
    place_close_user = ("また行きたい", "次は一緒に行こうよ", "天気いい日にまた行く")
    place_close_reply = (
        "うん、{place}また行こう。連れてって。",
        "うん、{place}今度こそ一緒に。",
        "{place}いい天気の日が一番だね。",
    )
    for place in PLACES:
        for ou, or_ in zip(place_open_user, place_open_reply):
            for ru, rr in zip(place_react_user, place_react_reply):
                for cu, cr in zip(place_close_user, place_close_reply):
                    blocks.append(
                        (
                            TurnPair(ou.format(place=place), or_.format(place=place)),
                            TurnPair(ru, rr.format(place=place)),
                            TurnPair(cu, cr.format(place=place)),
                        )
                    )

    # ---- hobby ramp (4 turns) — final reply names the hobby ----------------
    hobby_open_user = ("最近{hobby}にハマってる", "{hobby}始めた", "{hobby}が楽しい")
    hobby_open_reply = (
        "いいね、{hobby}って心が落ち着くよね。",
        "へえ、{hobby}どこでやってるの？",
        "{hobby}いいよね、続いてる？",
    )
    hobby_ask_user = ("おすすめある？", "コツとかある？", "なに揃えればいい？")
    hobby_ask_reply = (
        "{hobby}なら、最初は気軽な道具で十分だよ。",
        "{hobby}は無理せず短時間からが続くコツ。",
        "{hobby}は好きな時間にやるのが一番。",
    )
    for hobby in HOBBIES:
        for ou, or_ in zip(hobby_open_user, hobby_open_reply):
            for au, ar in zip(hobby_ask_user, hobby_ask_reply):
                blocks.append(
                    (
                        TurnPair(ou.format(hobby=hobby), or_.format(hobby=hobby)),
                        TurnPair(au, ar.format(hobby=hobby)),
                    )
                )

    # ---- animal (5 turns) — final reply uses animal name ------------------
    animal_open_user = ("{animal}飼いたい", "{animal}見た", "{animal}好き")
    animal_open_reply = (
        "いいね、{animal}と暮らせたら楽しそう。",
        "へえ、{animal}どこで？",
        "{animal}いいよね、私も好き。",
    )
    animal_concern_user = ("でもまだ無理かな", "準備が必要だよね", "今は時期じゃない")
    animal_concern_reply = (
        "うん、{animal}迎えるなら準備大事だもんね。",
        "{animal}は環境作ってからの方がお互い幸せ。",
        "うん、{animal}にとっても準備された家がいいよね。",
    )
    animal_close_user = ("いつかね", "ちゃんと準備する", "夢としてとっておく")
    animal_close_reply = (
        "うん、{animal}に出会える日を楽しみに。",
        "うん、いつか{animal}との生活、応援する。",
        "{animal}との未来、いいね。",
    )
    for animal in ANIMALS:
        for ou, or_ in zip(animal_open_user, animal_open_reply):
            for cu, cr in zip(animal_concern_user, animal_concern_reply):
                for clu, clr in zip(animal_close_user, animal_close_reply):
                    blocks.append(
                        (
                            TurnPair(ou.format(animal=animal), or_.format(animal=animal)),
                            TurnPair(cu, cr.format(animal=animal)),
                            TurnPair(clu, clr.format(animal=animal)),
                        )
                    )

    # ---- weather plan (4 turns) — final reply references weather ----------
    weather_pairs = (
        ("雨", "晴れ"),
        ("土砂降り", "晴れ"),
        ("曇り", "晴れ"),
        ("じめじめした天気", "風が気持ちいい日"),
        ("風が強い", "穏やかな日"),
    )
    for w_now, w_hope in weather_pairs:
        for opener, opener_reply in (
            (f"外{w_now}", "うん、こういう日は無理しないでいいよ。"),
            (f"今日{w_now}", "そっか、{w_now}だと気分も沈むよね。"),
        ):
            for plan_user, plan_reply in (
                ("家にいるよ", "うん、それが正解。"),
                ("散歩は明日にする", "うん、{w_now}だしね。"),
                ("出かけるのやめた", "うん、{w_now}だと気が乗らないもんね。"),
            ):
                for closer_user, closer_reply in (
                    (f"明日は{w_hope}になるといいね", f"うん、{w_hope}だったら気持ちよく出かけられる。"),
                    ("天気回復したら出かけよ", f"うん、{w_hope}になったら一緒に。"),
                ):
                    blocks.append(
                        (
                            TurnPair(opener, opener_reply.format(w_now=w_now)),
                            TurnPair(plan_user, plan_reply.format(w_now=w_now)),
                            TurnPair(closer_user, closer_reply.format(w_hope=w_hope)),
                        )
                    )

    # ---- mood support deep (5 turns) — final reply ties back to cause -----
    mood_open_user = (
        "なんかつらい",
        "今日しんどい",
        "落ち込んでる",
        "やる気でない",
        "気分が重い",
    )
    causes = (
        ("仕事で怒られた", "仕事"),
        ("人間関係つかれた", "人間関係"),
        ("勉強うまくいかない", "勉強"),
        ("体調いまいち", "体調"),
    )
    for u in mood_open_user:
        for cause_text, cause_topic in causes:
            blocks.append(
                (
                    TurnPair(u, "うん、無理しないで。話せる範囲で聞くよ。"),
                    TurnPair(cause_text, "それはきついね、よくがんばってる。"),
                    TurnPair(
                        "ちょっと話聞いてくれてありがと",
                        f"うん、いつでも。{cause_topic}は無理しないで、ちょっとずつね。",
                    ),
                )
            )

    # ---- movie/picnic plan (4 turns) — final reply ties to genre ----------
    movie_kinds = ("アニメ映画", "邦画", "洋画", "ドキュメンタリー", "コメディ")
    for kind in movie_kinds:
        for opener_user, opener_reply in (
            ("映画見たい", "いいね、何の？"),
            ("映画でも見ようかな", "なに見るの？"),
        ):
            blocks.append(
                (
                    TurnPair(opener_user, opener_reply),
                    TurnPair(kind, f"{kind}いいね、最近よさそうなの出てる？"),
                    TurnPair("一緒に行こうよ", f"うん、映画館で会おう。{kind}楽しみ。"),
                )
            )

    # ---- subject hygiene long (4 turns) — final reply contrasts X and Y ---
    self_objects = (
        ("本読んでる", "本"),
        ("ゲームしてる", "ゲーム"),
        ("散歩してきた", "散歩"),
        ("コーヒー飲んでる", "コーヒー"),
        ("音楽聴いてる", "音楽"),
        ("家にいる", "家"),
    )
    you_objects = (
        ("ぼんやりしてる", "ぼーっと"),
        ("仕事してた", "仕事"),
        ("料理してた", "料理"),
        ("出かけてた", "外"),
    )
    for su, sx in self_objects:
        for yu, yx in you_objects:
            blocks.append(
                (
                    TurnPair(f"私は{su}", f"そっか、{sx}いいね。"),
                    TurnPair("あなたは？", f"私は{yu}。"),
                    TurnPair("お互いゆっくりだね", f"うん、{sx}と{yx}、それぞれの時間ね。"),
                )
            )

    # ---- typo / repair recovery (4 turns) ---------------------------------
    swap_pairs = (
        ("ラーメン", "うどん"),
        ("コーヒー", "紅茶"),
        ("公園", "図書館"),
        ("映画", "本"),
    )
    for original, corrected in swap_pairs:
        blocks.append(
            (
                TurnPair(f"{original}食べたい", f"いいね、{original}行こう。"),
                TurnPair(f"あ、違う、{corrected}だった", f"あ、ごめん。じゃあ{corrected}にしよ。"),
                TurnPair("うん、お願い", f"うん、{corrected}気分だね、わかる。"),
            )
        )

    # ---- probe → anchor → ambiguous → resolve (5 turns) -------------------
    # Mirrors the user's failure case: "家にいるよ、あなたは？" → ambiguous probe.
    self_states_short = (
        ("家にいる", "家"),
        ("ちょっと出かけてる", "外"),
        ("カフェにいる", "カフェ"),
        ("仕事中", "仕事"),
    )
    for state, anchor in self_states_short:
        blocks.append(
            (
                TurnPair(f"今{state}", f"そっか、{anchor}でゆっくり？"),
                TurnPair("あなたは？", "私は本読んでた。"),
                TurnPair("読書いいね", f"うん、{anchor}で過ごしてるあなたとは別の静けさかも。"),
            )
        )

    # ---- "え？" / "どゆこと？" — repair turn that references prior topic --
    prior_topics = (
        ("ラーメン食べた", "ラーメンの話"),
        ("公園行ってきた", "公園の話"),
        ("猫見たよ", "猫の話"),
        ("仕事終わった", "仕事終わった話"),
    )
    for prior_user, prior_topic in prior_topics:
        blocks.append(
            (
                TurnPair(prior_user, "おお、いいね。"),
                TurnPair("え？", f"あ、ごめん。さっきの{prior_topic}の続きね。"),
                TurnPair("ああ、そういうこと", "うん、わかりにくくてごめん。"),
            )
        )
        blocks.append(
            (
                TurnPair(prior_user, "へえ、それでそれで？"),
                TurnPair("どゆこと？", f"あ、ごめん。{prior_topic}のほう、もっと聞きたかっただけ。"),
                TurnPair("なるほど", "うん、続き聞かせて。"),
            )
        )

    # ---- 6-turn deep food story (extends 5-turn pattern with one more pair)
    for food in FOODS_SAVORY[:8]:
        blocks.append(
            (
                TurnPair(f"{food}食べたい", f"いいね、{food}おいしいよね。"),
                TurnPair("一緒に行こうよ", "うん、行こう。今日空いてる？"),
                TurnPair("夜なら大丈夫", f"じゃあ夜、{food}食べに行こう。"),
                TurnPair("楽しみ", f"うん、私も。{food}久しぶり。"),
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

    # Big parameterized multi-turn pool first (target dominance).
    multi_blocks: list[Sequence[TurnPair]] = []
    multi_blocks.extend(gen_parameterized_multi_turns(rng))
    multi_blocks.extend(gen_two_turn_followups(rng))
    multi_blocks.extend(gen_long_anaphor_multi_turns(rng))

    # de-duplicate multi-turn blocks by their stringified form
    seen_blocks: set[str] = set()
    deduped_multi: list[Sequence[TurnPair]] = []
    for block in multi_blocks:
        key = _format_block(block)
        if key in seen_blocks:
            continue
        seen_blocks.add(key)
        deduped_multi.append(block)
    multi_blocks = deduped_multi
    rng.shuffle(multi_blocks)

    blocks_text: list[str] = []

    # Phase 1 — emit every multi-turn block at least once (the heart of the
    # corpus now: we want context-following to dominate).
    for block in multi_blocks:
        blocks_text.append(_format_block(block))
        if len(blocks_text) >= args.target_blocks:
            break

    # Phase 2 — interleave single-turn pairs and additional multi-turn
    # repeats so the ratio stays roughly 1 single : 1 multi.
    multi_iter = itertools.cycle(multi_blocks)
    for index, pair in enumerate(unique_pairs):
        if len(blocks_text) >= args.target_blocks:
            break
        blocks_text.append(_format_block([pair]))
        if index % 2 == 1:
            blocks_text.append(_format_block(next(multi_iter)))

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
