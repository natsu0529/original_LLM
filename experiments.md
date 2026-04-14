# Experiment Log

設定を変えて試した内容と結果を、雑でもいいのでここに残す。

大事なのは以下の4点だけ。

- 何を変えたか
- 数値はどうだったか
- 生成の使用感はどうだったか
- 次に何を試すか

注意:

- `byte` と `char` では token 定義が違うので、`valid_loss` の絶対値は横比較しにくい
- tokenizer が違う run 同士は、最終判断を生成品質で行う

## Entry Template

```text
## YYYY-MM-DD run-name

- status:
- purpose:
- command:
- config:
- best:
- last:
- notes:
- next:
```

## Runs

## 2026-04-14 dazai-long

- status: done
- purpose: byte-level のベースラインを長めに回して、最低限の日本語生成を作る
- command: `uv run python src/train.py --run-name dazai-long --resume checkpoints/dazai-long/latest.pt ... --max-steps 20000`
- config: `tokenizer=byte`, `vocab_size=256`, `n_layer=4`, `d_model=256`, `n_head=4`, `ffn_hidden=1024`, `context_length=256`, `batch_size=8`, `params=3,290,624`
- best: `step=19900`, `valid_loss=1.060665`
- last: `step=20000`, `valid_loss=1.067642`
- notes: 日本語らしい見た目までは行くが、UTF-8 byte 断片を学習しているので単語や文意が崩れやすい
- next: byte は比較対象として残し、以後の本命は char に寄せる

## 2026-04-14 dazai-char

- status: done
- purpose: byte-level をやめて、文字単位でどれだけ使用感が改善するかを見る
- command: `uv run python src/train.py --run-name dazai-char --tokenizer-type char --device mps --context-length 256 --batch-size 8 --max-steps 10000 ...`
- config: `tokenizer=char`, `vocab_size=4265`, `n_layer=4`, `d_model=256`, `n_head=4`, `ffn_hidden=1024`, `context_length=256`, `batch_size=8`, `params=4,316,928`
- best: `step=8400`, `valid_loss=2.798041`
- last: `step=10000`, `valid_loss=2.842097`
- notes: user 所感は「使用感は圧倒的に char の方がいい」。loss は byte と直接比較しない。`step=8400` 以降は少し悪化
- next: char を本線にして、次はモデル容量を増やす

## 2026-04-14 dazai-384x6

- status: done
- purpose: モデルを一段大きくしたときの重さと改善幅を見る
- command: `uv run python src/train.py --run-name dazai-384x6 --device mps --context-length 256 --batch-size 6 --n-layer 6 --d-model 384 --n-head 6 --ffn-hidden 1536 --dropout 0.1 --max-steps 5000 ...`
- config: `tokenizer=byte`, `vocab_size=256`, `n_layer=6`, `d_model=384`, `n_head=6`, `ffn_hidden=1536`, `context_length=256`, `batch_size=6`, `params=10,844,160`
- best: `step=5000`, `valid_loss=1.254744`
- last: `step=5000`, `valid_loss=1.254744`
- notes: 容量増加は確認できたが、tokenizer が byte のままなので日本語生成の根本改善としては弱い
- next: 同じサイズで char 化した run を回す

## 2026-04-14 dazai-char-384x6

- status: running
- purpose: char-level の良さを維持したまま、モデル容量を一段増やす
- command: `uv run python src/train.py --run-name dazai-char-384x6 --tokenizer-type char --device mps --context-length 256 --batch-size 6 --n-layer 6 --d-model 384 --n-head 6 --ffn-hidden 1536 --dropout 0.1 --max-steps 10000 ...`
- config: `tokenizer=char`, `vocab_size=4265`, `n_layer=6`, `d_model=384`, `n_head=6`, `ffn_hidden=1536`, `context_length=256`, `batch_size=6`, `params=12,383,616`
- best: `step=6600`, `valid_loss=2.842542` まで確認
- last: `step=6800`, `valid_loss=2.889892` まで確認
- notes: 進行中。初期の速度感は問題なし。char 256x4 より表現力の余地を見たい run
- next: 完走後に `checkpoints/dazai-char-384x6/best.pt` で生成比較する
