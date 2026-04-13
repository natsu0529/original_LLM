# original_LLM

PyTorchのみで、外部の事前学習モデルやFTに頼らず、最小の独自LLMをゼロから作るための実験リポジトリです。

現時点では性能を追いません。まずは「自分で実装した学習器が動き、テキストを少しでも生成できる」状態を最優先にします。

## Goal

- PyTorchだけで学習と推論を実装する
- 外部の事前学習モデルは使わない
- Fine-tuningではなく、ゼロから学習する
- ローカル環境で完結する
- まずは小さくても動くものを作る

## Non-Goal

- 高性能な日本語LLMを作ること
- 商用レベルの品質を狙うこと
- 大規模分散学習
- 重い学習基盤の導入
- 複雑な最適化を最初から盛り込むこと

## Current Assumptions

- 開発マシン: Mac mini M4
- メモリ: 32GB
- フレームワーク: PyTorch
- まずはMPSで動かす

## v0 Plan

最初のバージョンでは、以下のような最小構成を目指します。

1. byte-level tokenizer
2. decoder-only Transformer
3. next-token predictionで学習
4. checkpoint保存
5. 文章生成スクリプト

## Why Byte-Level

最初にBPEやSentencePiece相当を自作すると、実験の本題よりトークナイザ実装に時間を取られます。

このリポジトリでは、まずUTF-8 bytesをそのままトークンとして扱います。

利点:

- 実装が単純
- 日本語と英語を同じ仕組みで扱える
- tokenizer学習が不要

欠点:

- 効率が悪い
- コンテキスト長を食いやすい

今回はこの欠点は許容します。

## First Model Target

まずは以下のような小型モデルから始めます。

- `n_layer=4`
- `d_model=256`
- `n_head=4`
- `ffn_hidden=1024`
- `context_length=256`
- `vocab_size=256`

この段階では、性能よりも以下を重視します。

- 学習が最後まで走る
- lossが下がる
- 生成が完全なランダム文字列ではなくなる

## Planned Layout

```text
original_LLM/
  README.md
  claude.md
  agent.md
  src/
    config.py
    data.py
    model.py
    train.py
    generate.py
    utils.py
  data/
    raw/
  checkpoints/
  samples/
```

## Milestones

### Milestone 1

- 学習用テキストを1ファイル読み込める
- byte列に変換できる
- ミニバッチを作れる

### Milestone 2

- Transformer本体を実装する
- 学習ループを回す
- 小さなデータで過学習を確認する

### Milestone 3

- checkpoint save/load
- 推論スクリプト
- temperature / top-k sampling

### Milestone 4

- 複数テキスト対応
- validation loss
- 学習再開

## Development Principles

- まずは小さく作る
- 動くものを優先する
- 余計な抽象化は後回しにする
- 1ファイル過学習を最初の成功条件にする
- 依存関係は増やしすぎない

## Status

初期設計段階。これから最小実装に入る。
