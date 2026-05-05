# 2026-05-06 v6: 2 段階学習 (aozora 事前学習 → chat SFT)

## 背景

v0.2.1 (`dazai-friend-natural-v5-768x12`) を実走で確認したところ、

- 多ターン訴追では一部改善 (`v5` log 参照)
- しかし依然として「家にいるよ、あなたは？」→「口あたりはやわらかいくせに、あとで急に効いて来る。」のような **言語事前確率の弱さ** が残る

原因仮説: 学習 corpus が chat seed (1.9 MB) のみで、十分な日本語の言語モデル化ができていない。
v5 までは小さい corpus に対して overfit するまで学習し、自然な日本語生成より「テンプレ反復」になっていた。

## 対処

### Stage 1: aozora + chat の事前学習

`data/pretrain_natural_v6/` を新設:

- `aozora_peers/` 13.9 MB (太宰, 芥川, 漱石, 鴎外, 寺田寅彦 などの clean 済み青空文庫テキスト)
- `chat_seed_friend_natural_v5/` 1.9 MB

合計 15.8 MB。tokenizer は SentencePiece unigram, vocab=5000 (chat-only より広く取る)。

87M params (n_layer=12, d_model=768, n_head=12, ffn_hidden=3072, context_length=512) を **full LM loss** で 8000 step 学習:

```
batch_size=8 max_steps=8000 lr=4e-4 → 4e-5 (cosine)
warmup_steps=400 weight_decay=0.05 dropout=0.10
```

結果: best val_loss 3.699 @ step 6750 (full LM loss なので絶対値は v5 と直接比較不可)。

### Stage 2: chat SFT (reply-only loss)

stage 1 の best.pt を init として `chat_seed_friend_natural_v5/` のみで SFT:

- `--reset-optimizer --reset-best-val-loss --reset-step` で fresh 学習扱い
- `reply_loss_label=相手` で「相手:」行のみ loss を計算
- max_steps=3000 lr=1e-4 → 1e-5 cosine, warmup=100, batch_size=8, dropout=0.10, weight_decay=0.05

### `train.py` の修正

`--reset-step` フラグを追加。これがないと `start_step = resumed_step + 1`（つまり 6751）に対し
`max_steps=3000` を渡すと for-loop が空イテレーションになり、エラーも吐かず即終了する silent failure を起こしていた。

## 結果

eval 履歴抜粋:

| step | train | valid | best |
| --- | --- | --- | --- |
| 1 | 1.624 | 1.922 | 1.922 |
| 200 | 1.316 | 1.614 | 1.614 |
| 400 | 1.470 | 1.755 | 1.614 |
| 1000 | 0.767 | 2.051 | 1.614 |
| **1400** | **0.789** | **1.608** | **1.608** ← best.pt |
| 2000 | 0.560 | 1.791 | 1.608 |
| 3000 | 0.435 | 1.669 | 1.608 |

best step = 1400, **valid_loss = 1.608**。

数値的には v5 (1.441) より高い (= 悪い)。これは reply-only loss の値だが、stage 1 の事前学習で陳述文/会話以外の言語分布も学んだため、chat-only の hold-out reply 予測には不利になる。

しかし **質的な振る舞いは改善している** (下記 probe)。

## probe 比較 (温度 0.8, top-k 40, seed 42, 全フィルタ off)

| 入力 | v0.2.1 (v5) | v0.3.0 (v6 SFT) |
| --- | --- | --- |
| 家にいるよ、あなたは？ | 口あたりはやわらかいくせに、あとで急に効いて来る。 | 私もぼちぼち。 ✅ |
| 何を食べたの？ | おいしそう。 | 食べてないよ。 ✅ |
| 私はもう起きたよ | 今。 | そっちは？ ✅ |
| え？ | どうもー。 | たぶん、やり方がまだ噛み合っていないだけだね。 ✅ |
| どこに引っ越したの？ | 口あたりに越した事はしをしている。（broken） | ずっと誰か一人暮らしに挑戦してみようかな。 ✅ |
| こんにちは | どうお、こんにちは。 | 最初はグーね。 △ |
| 何してる？ | ぼんやりしてる。 | ぼんやりしてる。 = |
| そっか、元気？ | そっちは？ | そっちは？ = |
| えっと、どゆこと？ | ついているわけでもないよ。 | 確かに。 = |
| 大丈夫だよ、少し遊ぼうよ | ぼーっとしてから。 | 私も飲みたかった。 = |
| 今日は何してた？ | 家にいた。 | 家にいた。 = |
| 映画でも見ようかな | まずひと息ついて。 | 映画館で見てしまったのが…（文法弱） |

5 wins / 1 loss / 残り tied。subject-aware reply (家にいる→私もぼちぼち, 私はもう起きた→そっちは？), broken 日本語の解消, repair turn が動くようになった。

## 配布

`v0.3.0` として:

- `dazai-friend-sft-v6-768x12/best.pt` を slim (358 MiB)
- `cli.py` の `preferred_chat_checkpoint()` 先頭に `dazai-friend-sft-v6-*/best.pt` を追加
- retrieval corpus は v5 と同じ (`chat_seed_friend_natural_v5/`) なので変更なし
- 87 テスト pass

## 残課題

- 言語事前確率は底上げされたが、chat 文脈の coherence (多ターン記憶) はまだ弱い。
  - 例: 「映画でも見ようかな」→「映画館で…映画館は…」と単語反復が出る
- corpus 量で殴る方向と並行して、context 拡張 (768 へ) や ETM (early stopping) を試す価値あり
