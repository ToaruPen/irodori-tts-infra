# Irodori-TTS v4.1 固定プリセット・自由記述 delivery caption 設計

## 目的

Irodori-TTS v4.1 Small VoiceDesign と Speaker Inversion を使う標準合成経路に、
声の演技を指定する二つの排他的な入力方法を設ける。

1. `style` は、サーバーが所有する固定プリセットを選ぶ。
2. `delivery_caption` は、エージェントが場面に応じて生成した自由記述を渡す。

二つの指示は合成しない。固定プリセットは再現性のある定型表現を担当し、自由記述は
状況、感情、距離感、話速、息遣いなど、そのリクエスト固有の演技を担当する。

## 前提

標準 checkpoint は `Aratako/Irodori-TTS-v4.1-Small` とする。v4.1 の
VoiceDesign は推論ごとの caption と Speaker Inversion embedding を同時に受け取れるため、
話者同一性を embedding に任せたまま演技だけを動的に変更できる。

自由記述 caption は声質そのものを再定義する用途には使わない。参照話者と矛盾する
年齢・声域・性別などの指定は出力を不安定にし得るため、エージェントは演技と場面を
中心に、短く矛盾のない日本語で記述する。

## 公開 API 契約

### 固定プリセットモード

`style` に既存または追加のプリセット名を指定する。

```json
{
  "text": "もう少し、そばに来て。",
  "style": "alluring"
}
```

公開する `IrodoriStyle` は次の六種類とする。

| style | サーバー側の固定 caption |
| --- | --- |
| `neutral` | caption なし |
| `calm` | `穏やかで優しい女性の声で、自然に話す。` |
| `cheerful` | `明るく親しみやすい女性の声で、自然に話す。` |
| `clear` | `子どもに伝わるように、ゆっくり明瞭な女性の声で話す。` |
| `alluring` | `落ち着いた大人の女性の、艶のある色っぽい声で、自然に話す。` |
| `lewd` | `吐息を交えた大人の女性の、卑猥で挑発的な声で、艶っぽく話す。` |

プリセットは入力本文を変更しない。`lewd` を含め、固定 caption が制御するのは演技だけである。

### 自由記述モード

`delivery_caption` にエージェントが生成した日本語の演技指示を指定する。

```json
{
  "text": "もう少し、そばに来て。",
  "delivery_caption": "雨の夜、親しい相手の耳元で囁くように、吐息を少し交えて話す。"
}
```

`delivery_caption` が指定された場合、その値だけを upstream の `caption` に渡す。
固定プリセットの caption は前置・追記しない。`cfg_scale_caption` は固定プリセットと
自由記述の両方に従来どおり適用する。

### 排他規則と既定動作

- `delivery_caption` がない場合は `style` を固定プリセットとして解決する。
- `delivery_caption` がある場合、`style` は `neutral` だけを許可する。`neutral` は caption
  を生成しないため、自由記述との混合は起きない。
- `delivery_caption` と `calm`、`cheerful`、`clear`、`alluring`、`lewd` のいずれかを
  同時に指定した要求は、曖昧な優先順位を設けず validation error にする。
- どちらも指定されなければ、従来どおり `neutral` として caption なしで合成する。
- upstream の生のフィールド名である `caption` は、引き続き unknown field として拒否する。

`style` の既定値は `neutral` のまま維持する。このため既存クライアントの wire format と
既定動作は変わらず、自由記述モードだけを追加できる。

## 自由記述の検証

`delivery_caption` は公開 API 境界で次の規則を満たさなければならない。

- 前後の空白を除去した後に一文字以上ある。
- 長さは Unicode code point で 300 文字以下とする。
- 改行、タブ、NUL を含む Unicode control character と、行区切り U+2028・段落区切り U+2029 を許可しない。
- 日本語の意味内容はフィルタリングしない。卑猥な演技指定も構造検証を通れば受理する。

300 文字の上限は、エージェントが一つから三つ程度の短い日本語文で演技を指定する用途に
十分な余地を残しつつ、長文の投入と upstream による黙示的な切り詰めを抑えるための公開契約である。
checkpoint 内部の token 上限を API の文字数上限として露出させない。

## Capabilities

`GET /capabilities` の `conditioning.delivery_caption` を、標準 v4.1 runtime では次のように返す。

```json
{
  "supported": true,
  "max_chars": 300
}
```

`DeliveryCaptionCapability` は対応 runtime と非対応 runtime の両方を表現できる契約にする。
`supported=false` なら `max_chars` は `null`、`supported=true` なら `max_chars` は正の整数を
必須とする。フィールド自体は contract version 1 に既に存在するため、version は変更しない。

## データフロー

HTTP の単発要求と batch segment は、同じ `SynthesisRequest` の検証規則を使う。
検証済みの `delivery_caption` は `SynthesisJob`、`ResolvedSynthesisRequest` を経由して backend
まで運ぶ。backend は次の排他的な解決だけを行う。

```text
delivery_caption がある → その文字列を SamplingRequest.caption へ渡す
delivery_caption がない → style_caption(style) の結果を SamplingRequest.caption へ渡す
```

server router は値を解釈せず、contract から job へ転送する。pipeline は話者解決と backpressure
に専念し、caption の合成や補正を行わない。warm-up は外部入力を受けないため、既存の
`warmup_style` による固定プリセット経路を維持する。

## エラー処理

次の要求は Pydantic の validation error とし、HTTP 境界では既存どおり `422` を返す。

- 空または空白だけの `delivery_caption`
- 300 文字を超える `delivery_caption`
- control character を含む `delivery_caption`
- 非 `neutral` の `style` と `delivery_caption` の同時指定
- 生の `caption` フィールド

有効な caption を upstream が受理した後の生成失敗は、既存の backend error 変換を使う。
自由記述専用の再試行、caption 書き換え、プリセットへの暗黙 fallback は追加しない。

## テスト

実装は Red → Green → Refactor で進める。少なくとも次の振る舞いを自動テストする。

- `alluring` と `lewd` が公開 style として受理され、正しい固定 caption に解決される。
- 有効な `delivery_caption` が正規化され、JSON round-trip と batch segment で保持される。
- 空白、301 文字、control character を含む自由記述が拒否される。
- 非 `neutral` style と自由記述の同時指定が拒否される。
- `neutral` または省略時の style と自由記述では、自由記述だけが backend に渡る。
- preset mode では固定 caption だけが backend に渡る。
- `SynthesisJob` と router が自由記述を欠落させずに転送する。
- `/capabilities` が `supported=true` と `max_chars=300` を返し、対応状態と上限の矛盾を拒否する。
- 生の `caption` が引き続き拒否される。

実モデルによる音声品質は、任意の caption 全体に対して単体テストで保証できない。既存の GPU
smoke test では Speaker Inversion と自由記述 caption を一件組み合わせ、upstream interface の
互換性だけを確認する。GPU、model weight、SSH がない既定テストは fake runtime で境界を検証する。

## ドキュメントと運用境界

`AGENTS.md`、`README.md`、`docs/connection.md`、`docs/irodori-rvc-architecture.md` を更新し、
固定プリセットと自由記述の二モードを標準経路として記載する。利用例には `alluring`、`lewd`、
`delivery_caption` を一例ずつ示し、自由記述がプリセットと合成されないことを明記する。

この変更は repository 内の契約と実装に限る。Windows サービスの再起動、runtime generation の
変更、voice bank の交換、checkpoint の更新は行わない。v4.1 checkpoint への更新は既に
repository の設定とドキュメントへ反映済みであり、本変更はその更新を再実施しない。

## Test クライアント

`/Users/sankenbisha/Dev/Test/tts` のクライアントも、infra と同じ排他契約を実装する。
`TTSEngine` は単発合成と batch segment の両方で `delivery_caption` を転送し、非 `neutral`
の `style` との同時指定を送信前に拒否する。`say.py` と `remote_synth.py` は
`--delivery-caption` を公開し、既存の `--style` は固定プリセット専用として残す。

`read_aloud.py` は既存の話者タグ parser を維持し、コロン以降の指示だけを次の規則で解釈する。

```text
【名前】                              → neutral preset
【名前:style=alluring】               → alluring preset
【名前:親しい相手へ耳元で語りかける場面。…】 → delivery_caption
```

`style=` には公開された六つの preset だけを許可する。それ以外の空でない自然文は
`delivery_caption` として加工せずに送る。自然文から最寄りの preset を推測する旧 heuristic は
削除し、入力した自由記述が別の指示へ暗黙に変わらないようにする。

Test リポジトリの root `AGENTS.md`、`tts/AGENTS.md`、`tts/CLAUDE.md`、writing guideline は、
v4.1、六つの preset、自由記述モード、排他規則、タグ構文を同じ契約として記載する。

## エージェントスキルの分離

固定 preset と自由記述は、選択条件も作業内容も異なる。一つの skill にまとめず、名前だけで
入力方式が判別できる二つの skill に分ける。

### `irodori-preset-speech`

既存の `novel_dialogue_playback` を `irodori-preset-speech` へ改名する。この skill は
`neutral`、`calm`、`cheerful`、`clear`、`alluring`、`lewd` の選択と、`--style` を使う
再現性重視の読み上げだけを扱う。自由記述を生成せず、`delivery_caption` も送らない。

### `irodori-freeform-speech`

新しい `irodori-freeform-speech` は、場面固有の演技が必要なときだけ選択する。
Speaker Inversion が話者 identity を供給するため、caption には年齢、性別、別人の声質を
書かず、場面・相手、主感情と強度、速度・声量・距離感・口調を日本語一〜三文、300文字以内で
記述する。目安は一場面、一つの主感情、二〜四個の演技属性とする。

自由記述は肯定形の自然な描写にし、`速い／遅い`、`静か／叫ぶ`、`耳元／遠く` のような
同時成立しない指示を残さない。吐息、囁き、喘ぎ、リップノイズなどの発生位置は caption に
列挙せず、公式 emoji annotation を発話本文の該当位置へ置く。skill には v4.1 model card、
parameter guide、emoji annotation への出典を記載する。

`read_aloud` skill はファイル読み上げの orchestration に専念する。`style=...` タグでは
`irodori-preset-speech`、自然文タグでは `irodori-freeform-speech` の契約を参照し、caption
作成規則を重複して持たない。`start_tts_server` skill は運用手順を変えず、モデル表記だけを
v4.1 に合わせる。

## スキル評価とチューニング

skill 編集にも test-first を適用する。現行 skill を使う baseline で、自由記述要求を固定 preset
へ丸める、話者 identity を caption に重ねる、絵文字イベントを caption に書く、といった失敗を
先に記録する。その後、fresh executor を使って次の三種類を評価する。

- median: 固定 preset で十分な短い台詞を `irodori-preset-speech` が選ぶ。
- edge: 場面固有の親密さと息遣いが必要な台詞を `irodori-freeform-speech` が扱う。
- holdout: 話者属性と演技指示が混在した要求から、identity の再指定と矛盾を除く。

各 scenario は、skill 選択、preset と自由記述の排他、caption の構造、emoji の配置、実行コマンドを
critical checklist で採点する。修正後の median と edge は fresh executor で二回連続して
新しい不明点なしとなるまで反復し、最後に未使用の holdout を実行する。リポジトリに既存の
skill eval manifest はないため、評価結果は最終報告に構造化して残し、新しい永続形式は追加しない。

## 対象外

- preset caption と `delivery_caption` の連結・要約・優先順位付け
- runtime または Test クライアントからの別の LLM 呼び出し
- caption の意味内容に対する moderation
- 話者ごとの caption 許可リスト
- 自由記述に応じた `cfg_scale_caption` の自動調整
- runtime 配備、voice bank promotion、サービス再起動
