# Claude Notes

このリポジトリの目的は、PyTorchのみを使って、外部の事前学習済みLLMに依存せず、小さな独自LLMをゼロから作ることです。

## Hard Constraints

- 外部モデルのfine-tuningは禁止
- 外部モデルの重み流用は禁止
- まずはPyTorchと標準ライブラリ中心で進める
- 性能改善より、学習が通ることを優先する
- Python バージョンと依存管理は `uv` を使う

## Technical Direction

- モデル種別: decoder-only Transformer
- tokenization: corpus 由来の char-level
- 学習目標: next-token prediction
- 実行環境: Mac mini M4, 32GB RAM, MPS

## Priority Order

1. 単純で壊れにくい実装
2. 小さなデータでlossが下がること
3. checkpointと生成まで通ること
4. その後に整理や高速化

## Implementation Rules

- 既存ファイルが少ない間は、過剰な分割を避ける
- 最初から汎用フレームワーク化しない
- 学習コードは追跡しやすく、読める形を優先する
- 数値安定性に関わる箇所だけは雑にしない

## Training Operation Rules

長時間学習は、以下の運用を前提に実装と手順を書くこと。

- `caffeinate` を使ってスリープ停止を防ぐ
- `tmux` 上で学習を動かす
- checkpoint を定期保存する
- `--resume` で途中再開できるようにする
- 複数 run を並行実行できるよう、出力先を run 単位で分離する

学習スクリプトを作るときは、最低でも以下を持たせる。

- `--run-name`
- `--resume`
- `--save-every`
- `--out-dir` または `run_name` から決まる固有出力先

同時実行時に以下が衝突しないようにする。

- checkpoint
- sample 出力
- log
- 一時ファイル

M4 Mac mini 32GB では MPS 資源が限られるため、重い学習の完全並行は前提にしない。
ただし、軽量実験を別 run として並行に回せる設計は維持する。

## Environment Rules

- 実行例は `python3` 直打ちではなく `uv run python ...` を優先する
- 依存追加は `uv add`
- 依存削除は `uv remove`
- lock 更新は `uv lock`
- Python バージョンは `.python-version` と `pyproject.toml` の両方で揃える

## First Deliverable

最初の完成条件は以下です。

- 生テキストを読み込む
- 文字 token列へ変換する
- 小型Transformerで学習できる
- checkpointを保存できる
- 短いテキストを生成できる

## Model Size Guidance

最初は以下を標準候補とします。

- `n_layer=4`
- `d_model=256`
- `n_head=4`
- `ffn_hidden=1024`
- `context_length=256`
- `batch_size` はMPS使用量を見ながら調整

## What To Avoid Early

- tokenizer自作の複雑化
- BPE や SentencePiece 相当の早すぎる導入
- MoEやRAGのような横道
- データパイプラインの過剰最適化
- configシステムの作り込み

## Working Style

- まず最小実装を置く
- 動いたあとで整理する
- 変更時はREADMEとの整合を保つ
