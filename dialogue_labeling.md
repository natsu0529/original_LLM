# Dialogue Labeling

青空文庫の鉤括弧候補を AI で分類・整形する時の共通ルール。

対象入力:

- `data/intermediate/aozora/dazai/dialogue_batches/dialogue_batch_*.jsonl`

対象出力:

- `data/intermediate/aozora/dazai/labeled_batches/dialogue_batch_*.labeled.jsonl`

## Goal

- 会話学習に使える候補を `pair` / `single` として抽出する
- 会話に向かないものを `monologue` / `noise` に落とす
- 原文の太宰らしさはなるべく残す
- 無理な現代化や言い換えをしない

## Labels

- `pair`
  - 前後の発話と自然につながる会話ペアとして使える
  - 原則として `prev_quote_text -> quote_text` を優先して使う
  - どうしても `next_quote_text` 側のほうが自然なら `pair_with=next`
- `single`
  - 相手発話として単独で使える
  - 返答としての形や声色はあるが、自然な相手文が取りにくい
- `monologue`
  - 鉤括弧内だが会話より独白・演説・引用・作文に近い
- `noise`
  - 会話 seed に使う価値が薄い
  - 題名、短すぎる単語、メタ参照、崩れた引用など

## Tags

必要なものだけ付ける。0 個でもよい。

- `daily`
- `banter`
- `mood`
- `confession`
- `argument`
- `affection`
- `sensual`
- `alcohol`
- `literature`
- `philosophy`
- `humor`
- `greeting`
- `question`

## Editing Rule

- 基本は最小修正
- 旧仮名や文体はむやみに現代化しない
- ルビや注記は抽出側で概ね除いてあるので、そのままでよい
- 明らかな地の文混入が少しある場合だけ軽く削る
- 意味を変える書き換えはしない

## Output Schema

各行は入力 JSON を引き継ぎ、以下を追加する。

- `label`
  - `pair` / `single` / `monologue` / `noise`
- `pair_with`
  - `prev` / `next` / `null`
- `user_text`
  - `pair` の時だけ埋める
- `reply_text`
  - `pair` または `single` の時だけ埋める
- `formatted_text`
  - `pair`: `私: ...\n相手: ...`
  - `single`: `相手: ...`
  - それ以外は `null`
- `tags`
  - 上の tag 一覧から選ぶ
- `notes`
  - 任意。短く一言だけ

## Pair Judgment

`pair` にする条件:

1. 前後どちらかの発話と意味がつながる
2. 応答のターンとして読める
3. 地の文や説明文としての比率が高すぎない

落とす例:

- 長大な演説がほぼ一人で続いている
- 引用文中の引用が中心
- 題名や見出し
- 詩句や文章見本

## Example

入力:

```json
{
  "prev_quote_text": "おはよう。",
  "quote_text": "おそいじゃないか。"
}
```

出力:

```json
{
  "label": "pair",
  "pair_with": "prev",
  "user_text": "おはよう。",
  "reply_text": "おそいじゃないか。",
  "formatted_text": "私: おはよう。\n相手: おそいじゃないか。",
  "tags": ["daily", "banter"],
  "notes": "軽い応答"
}
```

## Worker Rule

- batch ごとに独立して処理する
- 他 worker の出力は触らない
- 判定に迷ったら `monologue` または `noise` に倒す
- 量よりも一貫性を優先する
