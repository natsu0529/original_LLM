# Agent Guide

このプロジェクトで作業するエージェントは、以下の前提を守って進めてください。

## Mission

PyTorchのみで、最小の独自LLMをゼロから実装し、ローカルで学習と生成を成立させること。

## Project Constraints

- 外部の事前学習済みモデルを使わない
- fine-tuningをしない
- まずは小さく動くものを作る
- 性能よりも実装の一貫性と再現性を優先する

## v0 Scope

- byte-level tokenization
- decoder-only Transformer
- causal self-attention
- next-token prediction training
- checkpoint save/load
- text generation

## Suggested File Ownership

初期段階では、以下のような責務分割を想定します。

- `src/data.py`: テキスト読込、byte変換、データセット、バッチ作成
- `src/model.py`: Transformer本体
- `src/train.py`: 学習ループ
- `src/generate.py`: 推論とサンプリング
- `src/config.py`: 最小設定
- `src/utils.py`: seed、device、checkpoint補助

## Engineering Rules

- 小さく始める
- 既存設計にない抽象化を急いで入れない
- デバッグしやすいコードを優先する
- まず1つの成功経路を通す

## Success Criteria For Early Work

- 学習がクラッシュせず回る
- train lossが下がる
- 保存したcheckpointから再開できる
- 学習データの癖を少し反映した生成が出る

## Early Failure Cases

以下は早めに潰すべきです。

- shape mismatch
- attention maskの誤り
- lossがNaNになる
- MPSでのOOM
- save/load不整合

## Practical Guidance

- 最初のデータ量は小さくてよい
- 過学習できることをまず確認する
- batch sizeやcontext lengthは環境に合わせて下げてよい
- 改善案より、まず実行可能な形を優先する

## Definition Of Done For v0

以下を満たせば、v0は成立です。

- 学習スクリプトが動く
- checkpointを書ける
- 生成スクリプトが動く
- 生成結果が完全ランダムよりはマシになっている
