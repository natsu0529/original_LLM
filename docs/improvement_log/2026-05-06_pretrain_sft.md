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

## v0.2.2 として最終リリース

最初の試行（このログ前半）は v5 corpus のままで SFT して見送ったが、その後:

1. multi-turn corpus を `chat_seed_friend_natural_v6` に拡張（4-6 ターン anaphor block を ~1857 追加）
2. SFT 再走 → best valid 0.868 @ step 2200（v0.2.1 1.441 から大幅改善）
3. 対話デバッグで「家にいるよ、あなたは？」など主訴ケースを検証
4. retrieval / curated 側を以下のとおり改善:
   - `extract_probe_suffix()` で compound input の probe 部分を分離 → secondary retrieval で 私は〜 reply を選ぶ
   - `curated_short_reply` が compound input の last clause も検査
   - direct retrieval で query 長の 60% 以上の overlap を要求（短すぎテンプレ衝突防止）
   - tokenizer の token/char 比 ≥0.7 で unknown-word 判定（confidence 単独より頑健）
   - `looks_like_single_word` を `の/さ/わ/ぞ/ぜ/ー` particle/tail と verb-phrase ending で強化

### 主訴ケース before/after

| 入力 | v0.2.1 | v0.2.2 |
| --- | --- | --- |
| 家にいるよ、あなたは？ | うん、それが正解。 | 私はぼんやりしてる。 |
| 大丈夫だよ、少し遊ぼうよ | ぼーっとしてから。 | いいね、何して遊ぼうか？ |
| えっと、どゆこと？ | ついているわけでもないよ。 | ごめん、変な言い方だった… |
| え？ | どうもー。 | あ、ごめん。わかりにくかったね… |
| 今日は何してた？ | 家にいた。 | 散歩してた。気持ちよかったよ。 |

### 残課題

- 多ターン context 引きずりは依然として完全解決ではない（87M model の限界）
- 例:「本いいね、なに読んでる？」(compound 質問) で model gen が脱線
- corpus 大規模化 / context_length 拡張 / model 拡大は次の段階

## v0.2.3 追記: empathic 多ターン block で friend-chat 自然さを底上げ

v0.2.2 を友達的な実会話で評価したところ、主訴ケースは fix されているが、
自由会話（仕事愚痴、嬉しいシェア、深夜とりとめない）で round-2 corpus に追加した
long_anaphor テンプレ（「{place}いい天気の日が一番だね」「{place}ひさしぶり？」）が
**無関係な context で漏れ出してた**。

```
私: もうへとへと
相手: スーパーで？               ← 場所テンプレが反応
私: 今日上司に怒られた
相手: 美術館いい天気の日が一番だね   ← 場所 anaphor template
```

対処:
1. `gen_long_anaphor_multi_turns` のテンプレから「{slot}+専用suffix」型を generic 化
   - place: 「{place}ひさしぶり？」→「ひさしぶりだったの？」
   - place: 「{place}いい天気の日が一番だね」→「いいね、天気いい日に。」
   - food: 「{food}リピートしよ、おいしい店覚えとこ」→「うん、いいね。覚えとこ。」
2. `gen_friend_empathic_multi_turns` 新設（22321 block）:
   - vent: 「もうへとへと」→「うん、それは疲れるね。」
   - happy: 「やった！」→「おお、なに？聞きたい。」
   - casual: 「休みだー」→「いいね、たまにはね。」
   - banter: 「わかる」「だよね」「なるほど」… の短い相槌返し
   - vague mood: 「もやもやしてる」→「うん、聞くよ。少しずつでいい。」
3. distribution: empathic 22k : long_anaphor 1.8k = 12:1 → 自由会話の safe default が
   short empathic ack に変わる

### round 3 結果

| step | train | valid | best |
| --- | --- | --- | --- |
| 200 | 0.78 | 0.679 | 0.679 |
| 800 | 0.59 | 0.570 | 0.570 |
| 1000 | 0.55 | 0.554 | 0.554 |
| **1400** | **0.57** | **0.510** | **0.510** ← best.pt |
| 3000 | 0.50 | 0.599 | 0.510 |

### 友達会話 before/after (v0.2.2 → v0.2.3)

シナリオ A (仕事愚痴 7 ターン):

| 入力 | v0.2.2 | v0.2.3 |
| --- | --- | --- |
| もうへとへと | スーパーで？ | おっ、ちょうど私も飲みたかった。 |
| 今日上司に怒られた | 美術館いい天気の日が一番だね | そういう日もあるよ。 ✅ |
| 頑張ってんのに評価されない | うんうん、こっちも嬉しい。 | しんどいよね。 ✅ |
| もう寝るわ | シュークリームリピートしよ | おやすみ、また明日。 ✅ |

シナリオ A スコア: 5/7 ✅ (v0.2.2 は 2/7)。
場所/食べ物テンプレ侵入は完全に消失、empathic ack が defaults に。

## (旧記録) round 1 の見送り判断

slim 化と build までは行ったが、`carry-context` on / max-history-turns=2 の本番設定で
本物の対話を駆動したところ、probe 単発では出ていなかった崩れが多数再発した:

```
私: 家にいるよ、あなたは？
相手: うん、それが正解。           ← 質問に答えていない
私: 大丈夫だよ、少し遊ぼうよ
相手: 食べてないよ。変な話にそれただけ。  ← off-topic
私: えっと、どゆこと？
相手: 私もやってた？               ← repair 不発
私: え？
相手: 引っ越したばっかで。早く買わないと。 ← 前ターン引きずり (v0.2.0 の元バグ)
私: 今日は何してた？
相手: ごめん、変な言い方だった…   ← repair 誤発火
```

probe は単発の 1 ターンしか見ないので「家にいるよ、あなたは？」→「私もぼちぼち。」と良い応答が出るが、
履歴に直前 2 ターンを足した瞬間に context が混乱して崩れる。

仮説: stage 1 で言語事前確率は底上げされたが、SFT corpus (1.9MB) の中で
本当の意味で「直前 2 ターン履歴 + 新規入力 → 文脈整合的な相手返答」を学ぶ multi-turn block の量が足りていない。
3925 ブロックあるが、ほとんどは 2 ターンで完結し、3 ターン目以降の文脈整合は学習されていない。

## 残課題と次の一手

- multi-turn corpus を **3 ターン以上の対話** で大幅増やす（現状はほぼ 2 ターン pair）
- もしくは context_length を 768 へ拡張して履歴ウィンドウを増やす
- どちらをしても overfit 速度は早いので、stage 1 の事前学習量も並行して増やす必要がある

ロールバック内容:
- pyproject.toml を 0.2.1 に戻す
- `cli.py preferred_chat_checkpoint()` から `dazai-friend-sft-v6-*` を削除
- `src/train.py --reset-step` フラグは保持（resume + 短い max_steps の silent failure を直すバグ修正）
- `scripts/probe_chat_checkpoint.py` は保持（今後のデバッグに使う）
- `docs/improvement_log/2026-05-06_pretrain_sft.md` はこのファイルとして保持（学びを残す）
