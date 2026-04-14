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
- Python と依存管理は `uv` を使う

## v0 Plan

最初のバージョンでは、以下のような最小構成を目指します。

1. char-level tokenizer
2. decoder-only Transformer
3. next-token predictionで学習
4. checkpoint保存
5. 文章生成スクリプト

## Why Char-Level

byte-level は実装が単純だが、日本語では UTF-8 の断片を追うことになるため、自然な文章生成がかなり弱くなりやすい。

このリポジトリでは、まず corpus から登場文字を集めて、1文字を1トークンとして扱う。

利点:

- 日本語の自然さを出しやすい
- tokenizer 実装がまだ単純
- 外部 tokenizer に依存しない

欠点:

- 語彙サイズが corpus 依存で増える
- 未知文字対応が必要
- byte-level より実装は少し増える

## First Model Target

まずは以下のような小型モデルから始めます。

- `n_layer=4`
- `d_model=256`
- `n_head=4`
- `ffn_hidden=1024`
- `context_length=256`
- `vocab_size` は corpus から動的に決まる

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
  experiments.md
  data_policy.md
  scripts/
    download_aozora_dazai.py
  src/
    config.py
    data.py
    generate.py
    model.py
    train.py
    utils.py
  data/
    raw/
  checkpoints/
  logs/
  samples/
```

## Milestones

### Milestone 1

- 学習用テキストを1ファイル読み込める
- 文字列を token 列に変換できる
- ミニバッチを作れる
- 青空文庫の太宰作品を取得できる

### Milestone 2

- Transformer本体を実装する
- 学習ループを回す
- 小さなデータで過学習を確認する

### Milestone 3

- checkpoint save/load
- 推論スクリプト
- temperature / top-k sampling
- `train.py` で `run_name` ごとの checkpoint / sample / log を分離

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

## Experiment Log

試した設定と結果は [experiments.md](/Users/natsuhirosuzuki/original_LLM/experiments.md) に残す。

最低限、以下を記録する。

- 何を変えたか
- best / last の数値
- 生成の使用感
- 次に何を試すか

## Environment Setup

このリポジトリは `uv` で Python バージョンと依存関係を管理する。

初回セットアップ:

```bash
uv sync
```

コマンド実行例:

```bash
uv run python scripts/download_aozora_dazai.py --limit 5
uv run python src/data.py --limit 2
uv run python src/model.py --context-length 256
```

Python バージョンは [.python-version](/Users/natsuhirosuzuki/original_LLM/.python-version) と [pyproject.toml](/Users/natsuhirosuzuki/original_LLM/pyproject.toml) で管理する。
依存更新時は `uv add` または `uv remove` を使い、`uv lock` を更新する。

## Training Operations

このリポジトリの学習運用は、最初から以下を前提にする。

- `caffeinate` を付けてスリープで学習を止めない
- `tmux` 上で学習を実行し、切断や端末終了に備える
- checkpoint を定期保存し、`resume` で再開できるようにする
- 複数実験を並行で回せるように、run ごとに出力先を分離する

想定コマンド例:

```bash
caffeinate -dimsu tmux new -s llm-train
```

`tmux` の中で学習:

```bash
uv run python src/train.py --run-name dazai-debug
```

再開:

```bash
uv run python src/train.py --run-name dazai-debug --resume checkpoints/dazai-debug/latest.pt
```

並行実験:

```bash
uv run python src/train.py --run-name dazai-a --limit 8
uv run python src/train.py --run-name dazai-b --limit 32
```

並行で回す場合は、checkpoint、sample、log の出力先を `run_name` 単位で必ず分ける。
同一ファイルへ複数プロセスが書く構成は避ける。

M4 Mac mini 32GB では、重い MPS 学習を同時に複数本走らせると競合しやすい。
そのため、実運用では「重い本学習を1本 + 軽い検証を1本」程度から始める。

現在の `train.py` は以下を持つ。

- `--run-name`
- `--resume`
- `--save-every`
- `--sample-every`
- `--eval-every`
- `checkpoints/<run_name>/`
- `samples/<run_name>/`
- `logs/<run_name>/`

## Debug Generation

学習済み checkpoint を手で試すには `generate.py` を使う。

単発生成:

```bash
uv run python src/generate.py --checkpoint checkpoints/dazai-long/best.pt --prompt "私は"
```

対話モード:

```bash
uv run python src/generate.py --checkpoint checkpoints/dazai-long/best.pt --interactive
```

`私:` / `相手:` 形式の会話モデルを対話モードで使う例:

```bash
uv run python src/generate.py \
  --checkpoint checkpoints/dazai-friend-simple/best.pt \
  --interactive \
  --user-label 私 \
  --reply-label 相手
```

疑似的に文脈を持たせたい場合:

```bash
uv run python src/generate.py --checkpoint checkpoints/dazai-long/best.pt --interactive --carry-context
```

現在の `generate.py` は、デバッグ用途を優先して以下を既定で持つ。

- `max_new_tokens=64`
- `。`, `！`, `？`, `」` または空行での簡易 stop
- interactive 時の `prompt` / `output` 表示
- 必要なら `--repetition-penalty 1.1` のような軽い繰り返し抑制

短めに試す例:

```bash
uv run python src/generate.py --checkpoint checkpoints/dazai-char/best.pt --interactive --max-new-tokens 40
```

## Data Acquisition

学習データは当面、青空文庫の太宰治作品を使います。

取得スクリプト:

```bash
uv run python scripts/download_aozora_dazai.py --limit 5
```

全件取得:

```bash
uv run python scripts/download_aozora_dazai.py
```

出力先:

- `data/raw/aozora/dazai/zips/`
- `data/raw/aozora/dazai/txt/`
- `data/raw/aozora/dazai/manifest.jsonl`

最初の実験では `--limit` 付きで数作品だけ取得し、導線が固まったら全件に切り替える。

太宰に近い作家をまとめて取得する mixed corpus 用スクリプト:

```bash
uv run python scripts/download_aozora_authors.py
```

既定の対象:

- 太宰治
- 芥川龍之介
- 坂口安吾
- 織田作之助
- 梶井基次郎
- 中島敦

出力先:

- `data/raw/aozora/dazai_peers/zips/`
- `data/raw/aozora/dazai_peers/txt/`
- `data/raw/aozora/dazai_peers/manifest.jsonl`
- `data/raw/aozora/dazai_peers/summary.json`

既存チェックポイントのトークナイザで未知文字率を確認する例:

```bash
uv run python scripts/check_tokenizer_coverage.py \
  --checkpoint checkpoints/dazai-char-512x8/best.pt \
  --data-dir data/raw/aozora/dazai_peers/txt \
  --manifest-path data/raw/aozora/dazai_peers/manifest.jsonl
```

## Dialogue Candidate Extraction

青空文庫の太宰テキストから、`「...」` に包まれた発話候補を JSONL に抜き出すスクリプトを用意している。

これはそのまま学習に入れるためではなく、後段で AI エージェントに

- 会話ペアかどうかの判定
- 酒 / 文学 / 官能 / 日常などの分類
- `私:` / `相手:` 形式への軽整形

をさせるための中間データである。

実行例:

```bash
uv run python scripts/extract_aozora_dialogue_candidates.py
```

まず一部だけ試す例:

```bash
uv run python scripts/extract_aozora_dialogue_candidates.py --limit 5
```

主な出力先:

- `data/intermediate/aozora/dazai/dialogue_candidates.jsonl`
- `data/intermediate/aozora/dazai/dialogue_summary.json`

各候補には、作品ID、タイトル、発話本文、前後文脈、前後の鉤括弧発話、近接する発話クラスタIDを入れている。

AI に渡しやすいように、高スコア候補だけを score 順でバッチ化する例:

```bash
uv run python scripts/prepare_dialogue_labeling_batches.py --min-heuristic-score 4 --batch-size 200
```

出力先:

- `data/intermediate/aozora/dazai/dialogue_batches/`

分類ルールは [dialogue_labeling.md](/Users/natsuhirosuzuki/original_LLM/dialogue_labeling.md) にまとめている。

ラベル済み batch の検証:

```bash
uv run python scripts/validate_dialogue_labels.py
```

ラベル済み batch から自動 seed を書き出す例:

```bash
uv run python scripts/build_chat_seed_from_labels.py
```

first-pass の自動ラベルを作る例:

```bash
uv run python scripts/bootstrap_dialogue_labels.py --batches 0001 0002
```

## Status

データ収集導線の実装を開始。学習系の実装は `caffeinate + tmux + resume + run_name 分離` を前提に進める。
