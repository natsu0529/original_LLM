# Claude Notes

このリポジトリの目的は、PyTorchのみを使って、外部の事前学習済みLLMに依存せず、小さな独自LLMをゼロから作ることです。

## Hard Constraints

- 外部モデルのfine-tuningは禁止
- 外部モデルの重み流用は禁止
- まずはPyTorchと標準ライブラリ中心で進める
- 性能改善より、学習が通ることを優先する

## Technical Direction

- モデル種別: decoder-only Transformer
- tokenization: UTF-8 byte-level
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

## First Deliverable

最初の完成条件は以下です。

- 生テキストを読み込む
- byte token列へ変換する
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
- MoEやRAGのような横道
- データパイプラインの過剰最適化
- configシステムの作り込み

## Working Style

- まず最小実装を置く
- 動いたあとで整理する
- 変更時はREADMEとの整合を保つ
