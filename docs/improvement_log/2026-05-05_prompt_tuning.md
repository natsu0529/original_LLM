# 2026-05-05 プロンプトチューニング: 言語ガード / 未知語フロー / アップデート案内

## 0.1.6 までのおさらい

`docs/improvement_log/2026-05-05_memory.md` で 1 テーブル SQLite の永続記憶を導入した。
ユーザの追加要件はこの上に乗る:

1. 日本語以外の入力 → 「私は日本語しか喋れないんだ。」
2. 知らない単語 → 「知らない言葉だね、それは何？」と聞き、ユーザが答えたら memory に保存
3. ユーザが「間違えた」「ごめん打ち間違い」のような retraction を出したら覚えない
4. アップデートのやり方を聞かれたら `uv tool upgrade original-llm` を答える

## 設計

### 1. 非日本語ガード

`src/original_llm/language.py` を新規追加。

```python
def is_non_japanese_input(text: str) -> bool:
    if not text: return False
    if has_japanese_char(text): return False        # ひらがな/カタカナ/漢字を1つでも含めば日本語
    stripped = strip_punctuation_only(text)
    if not stripped: return False                    # 句読点だけは無視
    return any(ch.isalpha() for ch in stripped)      # 記号と数字だけも非言語扱いしない
```

interactive_loop の冒頭で chec し、True なら定型 `NON_JAPANESE_GUARD_REPLY = "私は日本語しか喋れないんだ。"` を返す。
モデル経路は通らないので `--seed` などの sampling 設定の影響を受けない。

### 2. 未知語判定（DB に無い × 確信度低）

「DB に無い」は `MemoryStore.contains_word(input)`:

```python
# 完全一致 / どちらかが substring / どちらかの key/value が input を含む
def contains_word(self, word: str) -> bool: ...
```

「確信度」は **生成した相手返答の各トークンに対する model の素の確率の平均** を使う。
一旦 logits を取った段階で raw softmax して、選ばれた token の確率を記録。
```python
# generate_text_tracked: text, mean_chosen_prob を返す
# generate_chat_reply_with_resample: chosen_reply, suffix, generated, confidence を返す
```

判定:

```python
single_word = looks_like_single_word(prepared_user_input)
if (memory_store and single_word
    and not memory_store.contains_word(prepared_user_input)
    and confidence < UNKNOWN_THRESHOLD):
    reply = UNKNOWN_WORD_QUESTION_TEMPLATES[rotation % N].format(word=...)
    pending_unknown_word = prepared_user_input
```

`looks_like_single_word` は文字数 ≤ 12、空白・句読点なし、文末助詞 (`です` `ね` `よ` 等) で終わらない、内側に主要助詞 (`は` `が` `を` `に` `へ` `で` `と` 等) を含まない、を満たす入力だけ true。

#### 閾値の決め方

学習済みモデルに 16 種の入力をぶつけて mean conf を測った:

| 入力 | conf | 状態 |
| --- | --- | --- |
| ぴえん | 0.37 | unknown |
| ふくおか | 0.55 | unknown |
| クリプトコイン | 0.27 | unknown |
| シャインマスカット | 0.25 | unknown |
| ぽよぽよ | 0.92 | unknown だが model が hallucinate |
| ガチホ | 0.38 | unknown |
| リスケ | 0.87 | unknown だが hallucinate |
| AIエージェント | 0.29 | unknown |
| 猫 | 0.88 | known |
| 犬 | 0.91 | known |
| 本 | 0.62 | known (曖昧) |
| コーヒー | 0.94 | known |
| ラーメン | 0.93 | known |
| ありがとう | 0.59 | known (curated 経由なので影響なし) |
| こんにちは | 0.86 | known (curated 経由) |
| 今日疲れた | 0.83 | known (curated 経由) |

クリーンな分離点は 0.55-0.59 の間にあり、`DEFAULT_UNKNOWN_CONFIDENCE_THRESHOLD = 0.55` で固定。
- 取れる: ぴえん, ふくおか, クリプトコイン, シャインマスカット, ガチホ, AIエージェント
- 取れない: ぽよぽよ, リスケ（mode collapse 気味の hallucinate）
- 誤検出ない: 本 (0.62) は閾値超え

ぽよぽよ / リスケは閾値を上げると `本` のような曖昧 known まで誤検出するので断念。ユーザは `--unknown-confidence-threshold 0.65` で締められる。

### 3. retraction による棄却

`looks_like_typo_correction(text)` が `間違えた` `打ち間違` `タイポ` `typo` `ごめん間違` 等を含むかを substring で判定。
pending_unknown_word が立っているターンで retraction 検出 → pending を破棄するだけ（fall through して通常の chat 経路へ）。

### 4. 学習動作

pending_unknown_word が立っている次のターンで retraction でない普通の入力が来たら、それを「説明」として扱う:

```python
key = f"word:{pending_unknown_word}"
stored = memory_store.bump_or_add(key, prepared_user_input)
# bump_or_add: 既に key があれば importance += 1, value 上書き / 無ければ MIN_IMPORTANCE で新規
reply = UNKNOWN_WORD_LEARNED_TEMPLATES[rotation % N].format(word=...)
```

同じ未知語が再登場 → 再度説明されると importance が上がる（タイポでない確信度に相当）。`MAX_IMPORTANCE = 5` でキャップ。

### 5. アップデート案内

`looks_like_update_question(text)`: `アップデート` `アップグレード` `更新` `バージョンアップ` `最新版` `新しいバージョン` `uv tool upgrade` `pip install -u` `upgrade` の substring を素朴に判定。
ヒットしたら、モデルを通さずに

```
アップデートは `uv tool upgrade original-llm` を打つだけだよ。
uv じゃない人は `pip install -U original-llm` でも OK。
```

を返す。モデルにコマンドを発明させない。

## 実装したフロー（interactive_loop の優先順位）

1. 言語ガード（非日本語）
2. pending_unknown_word が立ってる: retraction なら破棄、それ以外は memory に保存して `覚えた` 系返答
3. アップデート質問
4. curated_short_reply（既存）
5. select_direct_chat_reply（既存）
6. モデル生成 → 信頼度 + DB 不在チェック → 未知語ならば `知らない言葉だね、それは何？` で pending を立てる

## CLI

新オプション:
```
--language-guard / --no-language-guard         # 1) のオン/オフ。既定 on
--unknown-confidence-threshold 0.55             # 6) の閾値。既定 0.55
```

## テスト

`tests/test_language.py` 13 件:
- has_japanese_char, is_non_japanese_input（latin / cyrillic / 日本語 / 句読点のみ / 数字のみ）
- looks_like_single_word（短い名詞 / 文 / max_chars 設定）
- looks_like_typo_correction（間違えた系 / typo / 通常入力）
- looks_like_update_question（日本語の言い回し / コマンド名 / 無関係入力）

`tests/test_memory.py` には `bump_or_add` と `contains_word` の 7 件を追加。

`uv run python -m pytest tests/` 全 87 件パス。

## 検証

```bash
TMP_DB=$(mktemp); rm -f "$TMP_DB"
printf '%s\n' \
  'Hello' \
  'こんにちは' \
  'アップデートのやり方' \
  'クリプトコイン' \
  'ビットコインの一種' \
  'シャインマスカット' \
  '高級ぶどう' \
  ':memory' \
  ':quit' \
  | uv run dazai-chat --memory-db "$TMP_DB"
```

実測:
- Hello → `私は日本語しか喋れないんだ。` ✓
- こんにちは → `こんにちは。元気にしてた？` ✓ (curated)
- アップデートのやり方 → `アップデートは `uv tool upgrade original-llm` を打つだけだよ。…` ✓
- クリプトコイン → `クリプトコインか、教えてもらってもいい？` ✓ (unknown)
- ビットコインの一種 → `ありがとう、クリプトコイン覚えた。` ✓ (pending answer 学習)
- シャインマスカット → 高 conf hallucinate でスルー（false negative）
- 高級ぶどう → `高級ぶどうか、教えてもらってもいい？` ✓
- :memory → `#1 [1] word:クリプトコイン = ビットコインの一種`

## 残課題

- ぽよぽよ / リスケ / シャインマスカット のような **hallucinate confident** ケースを取れない
  - 次に試すなら: 入力 sentencepiece の token rarity を併用（unknown は piece が多く分割される傾向）
- ユーザが教えてくれた説明そのものが汚い時のフィルタなし（`?` だけ等）
  - 暫定: 短すぎる explanation はその場で再質問する layer を入れる余地

これらは次バージョンで対応。
