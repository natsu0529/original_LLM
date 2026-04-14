# Agent Guide

このプロジェクトで作業するエージェントは、以下の前提を守って進めてください。

## Mission

PyTorchのみで、最小の独自LLMをゼロから実装し、ローカルで学習と生成を成立させること。

## Project Constraints

- 外部の事前学習済みモデルを使わない
- fine-tuningをしない
- まずは小さく動くものを作る
- 性能よりも実装の一貫性と再現性を優先する
- Python バージョンと依存管理は `uv` を標準にする

## v0 Scope

- char-level tokenization
- decoder-only Transformer
- causal self-attention
- next-token prediction training
- checkpoint save/load
- text generation

## Suggested File Ownership

初期段階では、以下のような責務分割を想定します。

- `src/data.py`: テキスト読込、文字token化、データセット、バッチ作成
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

## Training Runtime Policy

このプロジェクトの学習ジョブは、以下を標準運用とする。

- `caffeinate` 付きで起動する
- `tmux` セッション内で動かす
- checkpoint を保存しながら走らせる
- `resume` で中断後に再開できる
- 複数 run を並行で回せるよう、出力先を分ける

エージェントが学習系コードを追加するときは、次を満たすこと。

- `--run-name` を受け取れる
- `--resume` で checkpoint から再開できる
- `checkpoints/<run_name>/` のような専用出力先を使う
- `samples/<run_name>/` や `logs/<run_name>/` のように run ごとに分離する

並行実験を行う際は、別の tmux window または session を使う。
同じ checkpoint に複数プロセスで書き込む構成は禁止。

M4 Mac mini 32GB では、重い MPS 学習を同時に複数本回すと詰まりやすい。
そのため、本学習は1本を基本とし、並行は軽量デバッグ run から始める。

## Environment Policy

- 実行コマンドは `uv run python ...` を基本にする
- 依存関係の変更は `uv add` / `uv remove` で行う
- lockfile は更新して維持する
- Python バージョンは `.python-version` と `pyproject.toml` に合わせる

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
