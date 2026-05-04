# 2026-05-03 自然な会話への改善

## 出発点

- 現行 checkpoint: `checkpoints/dazai-friend-topic-clean-spm-384x6-v3/best.pt`
  - n_layer=6, d_model=384, n_head=6, ffn_hidden=1536, context_length=256, vocab=1896 (sentencepiece)
  - best_val_loss=1.182 (step 1550)
  - 学習 corpus: `data/chat_seed_friend_clean_v1`
- chat 入口: `dazai-chat` (= `original_llm.cli.main_chat`)
  - 既定: `temperature=0.2, top_k=8, repetition_penalty=1.1, max_history_turns=2, retrieval_examples=0, normalize_chat_input=on`
  - 直近返答候補は `select_direct_chat_reply` が retrieval corpus から exact 寄りの seed を1件返し、当たれば model 生成をスキップ

## 観察された問題（ベースライン probe）

`printf '%s\n' 'こんにちは' '今日疲れた' 'お腹すいた' 'ねえ、最近何してる？' '映画でも見ようかな' '眠れない' 'ありがとう' ':quit' | uv run dazai-chat --show-prompt-output --seed 7`

| 入力 | 既定 dazai-chat 出力 | 不自然さ |
| --- | --- | --- |
| こんにちは | `こんにちは` | 入力をそのまま返している（エコー） |
| 今日疲れた | `いいよ。今日はよく持ちこたえたね。` | 「いいよ」が文脈に合わない |
| お腹すいた | `食べたいないと、言葉までやせる。` | 「食べたいない」は文法破綻 |
| ねえ、最近何してる？ | `ぼんやりしてる。少し話せるよ。` | 直後にも同じ返答（後述）、固有性が薄い |
| 映画でも見ようかな | `実用的には、、、そのどちらかにしようかな。` / `ぼんやりしてる。少し話せるよ。` | 直前と同じ返答を出すケースあり、また「実用的には」は友達口調から外れる |
| 眠れない | `そういう事はあるね。体と気分が仲たがいしている。` | 取得（retrieval）にヒットして硬めの一文 |
| ありがとう | `どういたしまして。そう言われると、少し助かる。` | 取得 seed のテンプレ感 |

retrieval を切って model 生成のみで再 probe：

| 入力 | 出力 | 不自然さ |
| --- | --- | --- |
| こんにちは | `食べたよ。食べていないと、言葉までやせる。` | 唐突な話題 |
| お腹すいた | `食べたいないと、言葉までやせる。` | 文法破綻 |
| 映画でも見ようかな | `ぼんやりしてる。少し話せるよ。` | mode collapse |
| 眠れない | `いいよ。動物とか食べ物とか、どっちがいい？` | 文脈無視 |
| ありがとう | `猫を探してたら知らない駅に着く夢。意味はわからない。` | まったく無関係 |

## 推定原因

1. **直近返答取得の品質が低い**
   - `select_direct_chat_reply` は最長共通部分が `min(len(query), len(example))` と一致するだけで採用されるため、「こんにちは」のように共通文字列が短い入力ほど無条件に当たる。
   - 同点候補のタイブレークが「テキスト長 短い」になっており、`相手: こんにちは` のようなエコー seed が勝ちやすい。
   - corpus 自体に `私: こんにちは → 相手: こんにちは` という低品質 seed が多数残っている（`chat_seed_friend_clean_v1` 内に複数あり）。

2. **モデル本体の mode collapse / 文法破綻**
   - 学習データが小さく overfit している（best_val_loss 1.18, step 1550）。
   - 既定 `temperature=0.2 / top_k=8` で greedy 寄り → 学習データに刻まれた特定フレーズ（「ぼんやりしてる。少し話せるよ。」「言葉までやせる。」）が頻出。
   - `repetition_penalty` は同一生成内のみ作用し、ターン間にはまたがらない。

3. **入力 vs 返答 の同一性チェックなし**
   - 「こんにちは」→「こんにちは」のように入力をそのまま返してしまっても弾かれない。

## 今回の対処方針（再学習なし、runtime と corpus の手当て中心）

A. **`generate.py` の retrieval/scoring を改善**
   - `is_low_quality_reply(user_text, reply_text)` を追加し、`load_chat_examples` で読み捨て。
     - エコー（reply == user、あるいは正規化後一致）
     - 極端に短く中身が薄い（句点や挨拶のみ）
     - 文法破綻フラグ（`たいない` `、、、` など）を含む
   - `chat_example_score` に「返答の長さが 8〜40 文字の範囲だと加点」を加え、極端な短さの seed を選びにくくする。
   - `select_direct_chat_reply` の閾値を「lookup が完全一致」かつ「返答が低品質判定を通過」に絞る。

B. **ターン間の繰り返し抑制**
   - interactive loop と one-shot CLI で直近の返答テキストを保持。
   - `select_direct_chat_reply` と model 生成のどちらでも、直近の返答と一致した場合は別候補を引く / 再サンプル。

C. **観察ログを記録**
   - 上記を入れたあと、同じ probe を流して output を比較。
   - 改善が薄い場合は次に学習側へ進む。

## 次の段階（必要なら）

- corpus の自動掃除スクリプト（同 reply を持つ重複 seed の整理、エコー seed の削除）
- 軽い再学習（同じモデル構成で step 数を伸ばす、seed ミックスで多様性を入れる）
- 上記が済んでもまだ不自然なら、d_model や context_length の見直し

## 実装した変更（runtime のみ・再学習なし）

`src/original_llm/generate.py` を中心に以下を追加した。

1. **低品質返答フィルタ** `is_low_quality_reply(user_text, reply_text)`
   - エコー（`is_echo_reply`、`ー` を取り除いた compact 比較も含む）
   - 既知の壊れフレーズ（`たいない`、`、、、` など）の包含
   - 既知の dangling 始まり（`もどすのが`、`ところが`…）
   - 「、X+(が|を|に|で|も|へ)+？$」のカンマ断片末尾（`実在する場所ではないので、お金も？` をブロック）
   - 「は？」のような自然な短文は誤検出しないよう判定範囲を絞った
   - 挨拶の category mismatch（`こんにちは → こんばんわ` を弾く）
   - 短すぎる reply（< 4 文字）

2. **retrieval 取得スコアの改善** `chat_example_score(..., reply_text=...)`
   - 返答が 8〜30 文字のときに加点、極端な短さ（< 4）は減点
   - 30〜50 文字に小さい加点、80 文字超に減点
   - これでタイブレークが「テキスト最短」から「内容のある適度な長さ」に寄るようになった

3. **直接返答（direct reply）の選定強化** `select_direct_chat_reply`
   - 上位 1 件 → 上位 8 件まで眺めて、最初に「low quality でない & avoid_replies に被らない」候補を採用
   - これで `こんにちは` に対する `こんにちは。` のようなエコー seed が選ばれなくなった

4. **キュレーション短文返答テーブル** `curated_short_reply`
   - 挨拶・感謝・別れ・気分（`こんにちは / ありがとう / つらい / 眠れない / 好き / 楽しい` …）について、自然に響く 2〜3 個ずつのリストを直書き
   - retrieval direct reply より優先して当たる
   - rotation_index と avoid_replies で連続同一返答を避ける

5. **モデル出力の resample + フォールバック** `generate_chat_reply_with_resample`
   - 最大 2 回まで再サンプル（後半ほど temperature を上げ、top_k も拡げる）
   - すべて低品質判定だった場合、`fallback_friendly_reply` から無難な相槌を返す（嘘を言わない、トピックを偽らない範囲で）

6. **ターン間繰り返し抑制**
   - `interactive_loop` で `recent_replies` を保持し、curated・direct・model 全経路に `avoid_replies` として渡す
   - `:reset` でクリア

7. **CLI 経路** `cli.py` の one-shot にも `curated_short_reply` を direct reply の前段として差し込んだ

`tests/test_chat_helpers.py` に `NaturalnessHelperTests` と `DirectChatReplyQualityTests` を追加。`uv run python -m pytest tests/` 全 38 件パス。

## 結果（同じ probe + 拡張）

`printf '%s\n' 'こんにちは' '今日疲れた' 'お腹すいた' 'ねえ、最近何してる？' '映画でも見ようかな' '眠れない' 'ありがとう' '楽しい' 'つらい' 'おはよう' 'おやすみ' 'ただいま' '元気？' '好き' '寂しい' ':quit' | uv run dazai-chat --no-show-prompt-output --seed 7`

| 入力 | 改善前 | 改善後 | 評価 |
| --- | --- | --- | --- |
| こんにちは | `こんにちは` | `こんにちは。今日はどうしてた？` | ✓ |
| 今日疲れた | `いいよ。今日はよく持ちこたえたね。` | `そっか、無理しないで。` | ✓ |
| お腹すいた | `食べたいないと、言葉までやせる。` | `なにか軽く食べちゃおう。` | ✓ |
| ねえ、最近何してる？ | `ぼんやりしてる。少し話せるよ。` | `ぼんやりしてる。少し話せるよ。` | △ (retrieval 経由、文としては自然) |
| 映画でも見ようかな | `実用的には、、、そのどちらかにしようかな。` | `古いなってみようかな` | × (model 生成のまま) |
| 眠れない | `そういう事はあるね。体と気分が仲たがいしている。` | `うん、ゆっくり考え事しないで横になろう。` | ✓ |
| ありがとう | `どういたしまして。そう言われると、少し助かる。` | `うん、いつでも。`（rotation で変化） | ✓ |
| 楽しい | `オーブンがない！` | `うん、いい一日になったね。` | ✓ |
| つらい | `もどすのが。` | `無理しなくていいよ。ここにいるから。` | ✓ |
| おはよう | (取得 seed) | `おはよう。今日はゆっくりやろう。` | ✓ |
| おやすみ | - | `おやすみ。ゆっくり休んでね。` | ✓ |
| ただいま | - | `おかえり。少し休もう。` | ✓ |
| 元気？ | - | `うん、なんとかやってる。そっちはどう？` | ✓ |
| 好き | - | `ありがとう。なんかこそばゆいな。` | ✓ |
| 寂しい | - | `そばにいるよ、ゆっくりでいい。` | ✓ |

連続同一入力の挙動も確認済み：
- `こんにちは / こんにちは / こんにちは` → 3 通りの返答が回る
- `ありがとう / ありがとう / ありがとう` → 3 通り
- `元気？ / 元気？` → 2 通り

### 残課題

- 自由文（`映画でも見ようかな`、`ピクニック行きたいな`）に対するモデル生成の文法破綻が残る
  - 例: `古いなってみようかな`、`スターツがおも？`
  - 原因はモデルのキャパシティ＋学習量不足（best_val_loss=1.18 / step 1550）。runtime のヒューリスティックで完全に防ぐと自然な短文も誤って弾く副作用が大きい。
  - 次の段階で検討: corpus 自動掃除 → 軽い再学習。

## 追加で用意したもの（再学習に向けた下ごしらえ）

`scripts/clean_chat_seed_for_naturalness.py` を新規追加。同じ `is_low_quality_reply` を使って、対象 seed dir 内の全 `.txt` から低品質ブロックを除いた版を別 dir へ書き出す。

実行例:

```bash
uv run python scripts/clean_chat_seed_for_naturalness.py \
  --source-dir data/chat_seed_friend_clean_v1 \
  --out-dir data/chat_seed_friend_clean_v2 \
  --report data/chat_seed_friend_clean_v2/_report.json
```

実測結果:

- total_blocks: 34570
- kept_blocks: 32988
- dropped_blocks: 1582 (≒ 4.6%)
- 落ちた seed の上位 20 ファイルはほぼ全て `real_persona_casual_v1` 由来（自動生成ノイズの多い系統）

retrieval corpus を `chat_seed_friend_clean_v2` に切り替えて挙動確認した範囲では、curated_short_reply が先に当たる入力は変化なし。長文入力は依然 model 生成側の弱さがそのまま出る。

## 推奨される次の一手（明示的に手で叩く想定、自律的には実行していない）

`chat_seed_friend_clean_v2` を data_dir として、現在の checkpoint から resume する短い再学習を試す。

```bash
caffeinate -dimsu tmux new -s llm-friend-clean-v2 \
  uv run python src/train.py \
    --run-name dazai-friend-clean-v2-384x6 \
    --data-dir data/chat_seed_friend_clean_v2 \
    --resume checkpoints/dazai-friend-topic-clean-spm-384x6-v3/best.pt \
    --reset-best-val-loss \
    --batch-size 8 \
    --context-length 256 \
    --save-every 500 \
    --eval-every 250
```

長時間学習を自律的に走らせると、tmux と caffeinate を伴う運用方針からも外れるため、ここまでで一旦止めて改善ログに記している。

