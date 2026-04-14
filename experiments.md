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

- status: done (`latest.pt` から `15000` まで低LR延長まで完了)
- purpose: char-level の良さを維持したまま、モデル容量を一段増やす
- command: `uv run python src/train.py --run-name dazai-char-384x6 --tokenizer-type char --device mps --context-length 256 --batch-size 6 --n-layer 6 --d-model 384 --n-head 6 --ffn-hidden 1536 --dropout 0.1 --max-steps 10000 ...` -> その後 `--resume checkpoints/dazai-char-384x6/latest.pt --learning-rate 1e-4 --max-steps 15000`
- config: `tokenizer=char`, `vocab_size=4265`, `n_layer=6`, `d_model=384`, `n_head=6`, `ffn_hidden=1536`, `context_length=256`, `batch_size=6`, `params=12,383,616`
- best: `step=14700`, `valid_loss=2.665377`
- last: `step=15000`, `valid_loss=2.702346`
- notes: `10000 -> 15000` の低LR延長でさらに改善した。`step=14700` で best 更新。ただし `15000` の最後は best ではなく、前に決めた基準ならここで一旦止めるのが妥当
- next: `checkpoints/dazai-char-384x6/best.pt` を使って生成比較し、必要なら次は会話データ追加か 512x8 を検討する

## 2026-04-14 dazai-chat-simple

- status: running
- purpose: 太宰寄りのベースを残したまま、続きを書く癖から短い返答へ少し寄せる
- command: `uv run python src/train.py --run-name dazai-chat-simple --resume checkpoints/dazai-char-384x6/best.pt --reset-optimizer --reset-best-val-loss --data-dir data/chat_seed_simple --manifest-path data/chat_seed_simple/manifest.jsonl --tokenizer-type char --device mps --context-length 256 --batch-size 6 --n-layer 6 --d-model 384 --n-head 6 --ffn-hidden 1536 --dropout 0.1 --learning-rate 1e-4 --max-steps 17700 ...`
- config: `tokenizer=char`, `base_checkpoint=dazai-char-384x6/best.pt`, `n_layer=6`, `d_model=384`, `n_head=6`, `ffn_hidden=1536`, `context_length=256`, `batch_size=6`, `params=12,383,616`
- best: `step=14701`, `valid_loss=3.825891` から開始
- last: `step=14701`, `valid_loss=3.825891`
- notes: base の optimizer state と best loss は引き継がず、会話 seed 用の run として切り分けた
- next: まず `17000` 台まで回し、生成が返答に寄るか確認する

## 2026-04-14 dazai-friend-simple

- status: running
- purpose: `アシスタント` ではなく、友達寄りの `私:` / `相手:` 会話へ寄せる
- command: `uv run python src/train.py --run-name dazai-friend-simple --resume checkpoints/dazai-char-384x6/best.pt --reset-optimizer --reset-best-val-loss --data-dir data/chat_seed_simple --manifest-path data/chat_seed_simple/manifest.jsonl --tokenizer-type char --device mps --context-length 256 --batch-size 6 --n-layer 6 --d-model 384 --n-head 6 --ffn-hidden 1536 --dropout 0.1 --learning-rate 5e-5 --max-steps 15100 ...`
- config: `tokenizer=char`, `base_checkpoint=dazai-char-384x6/best.pt`, `role_format=私/相手`, `n_layer=6`, `d_model=384`, `n_head=6`, `ffn_hidden=1536`, `batch_size=6`
- best: `step=14701`, `valid_loss=4.241915` から開始
- last: `step=14701`, `valid_loss=4.241915`
- notes: 旧 `dazai-chat-simple` は `ユーザー/アシスタント` 形式でズレていたため、新ラベル版を別 run で切り直した。過学習を抑えるため step は短め、learning rate も `5e-5`
- next: `best.pt` で `私: こんにちは\n相手:` の返答を見て、口調の向きが合うか確認する
