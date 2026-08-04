# Irodori-TTS v4 隔離移行設計

## 目的

稼働中の標準経路を Irodori-TTS v3 600M VoiceDesign と既存の Speaker
Inversion voice bank に固定したまま、v4 対応ランタイム、v4-Small
checkpoint、v4 用 Speaker Inversion 埋め込みの順に独立検証する。

本設計は互換性を「読み込めるか」と「品質を維持できるか」に分ける。v4 の
推論 API が v3 と同じ `caption`、`ref_embed`、caption/speaker CFG を受け取れても、
v3 の Speaker Inversion 埋め込みが v4-Small 上で同じ話者を再現できるとは扱わない。

## 確認済みの基準点

- 現行標準 checkpoint は
  `Aratako/Irodori-TTS-600M-v3-VoiceDesign`、revision
  `e863a3a93e652e09afeff3e84823a206a0a60314` である。
- 現行 upstream は v3 tag の commit
  `eaf74d6a19138f743acb5b71a445fd25a57db987` に固定されている。
- v4 対応 upstream の基準 commit は
  `8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71` とする。この commit の README は
  v2/v3 base・VoiceDesign checkpoint との推論互換を明記している。
- v4 checkpoint は `Aratako/Irodori-TTS-v4-Small`、revision
  `e4aaac4df355ff560dcd35e0dae272c3a759317b`、`model.safetensors` の SHA-256 は
  `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593` である。
- v4 checkpoint は `tokenizer/tokenizer.json` と
  `tokenizer/tokenizer_config.json` を同じ snapshot に含む。v4 upstream は checkpoint
  に隣接する tokenizer を優先してローカル読込する。
- v3 と v4 の Speaker Inversion token 形状はともに 16 token、speaker dimension
  768 である。ただし upstream は、埋め込みを学習時と同じ base model で使うことを
  要求している。したがって形状一致は品質互換の証拠ではない。

## 安全境界

- 稼働中サービスを停止、置換、再起動しない。
- 配備済み voice bank と v3 checkpoint 設定を変更しない。
- 現在の v3 upstream checkout を更新しない。v4 は別 checkout と別 runtime で扱う。
- v4 診断が通るまで、AGENTS.md が定める標準経路を v4 に変更しない。
- v3 埋め込みを v4 上で使う結果は診断値としてだけ保存し、配備候補にしない。
- checkpoint、tokenizer、埋め込み、生成音声は runtime asset とし、Git に追加しない。
- 既存の create-only 証跡を上書きしない。失敗証跡も履歴として保存する。

## 互換性マトリクス

| ランタイム | checkpoint | 埋め込み | 位置づけ |
|---|---|---|---|
| v4 commit | v3 600M VoiceDesign | v3 埋め込み | v4 ランタイムの後方互換回帰 |
| v4 commit | v4-Small | v3 埋め込み | 形式・品質の診断のみ |
| v4 commit | v4-Small | v4 で新規学習 | v4 配備候補の評価対象 |
| v3 commit | v3 600M VoiceDesign | v3 埋め込み | 現行比較基準と切戻し先 |

最初の行が通らなければ v4 ランタイム導入を中止する。2 行目が良好でも、同一 base
model 契約を満たさないため配備判断には使わない。配備候補になり得るのは、3 行目が
現行比較基準を品質ゲートで上回るか同等になった場合だけである。

## checkpoint snapshot 取得契約

### 共通取得

backend factory は単一ファイル取得ではなく、commit revision を指定した限定 snapshot
取得を使う。許可する asset は次のものだけとする。

- `model.safetensors`
- `tokenizer/*`

取得後は `model.safetensors` の SHA-256 を必ず検証してから runtime factory を呼ぶ。
snapshot 内の checkpoint path を `RuntimeKey.checkpoint` に渡すことで、v4 upstream が
隣接 tokenizer を発見できる配置を維持する。

### v3 と v4 の差

v3 checkpoint では bundled tokenizer を要求しない。tokenizer directory がなければ、
checkpoint metadata に記録された tokenizer repository へ upstream がフォールバックする
現行挙動を保つ。

v4 診断設定では次の 2 asset の SHA-256 pin を両方指定する。

- `tokenizer/tokenizer.json`:
  `6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d`
- `tokenizer/tokenizer_config.json`:
  `d229a271c64de1a7939d20d3665498e873fa91d5ee2edf135d73ec752cb9c9d3`

pin は両方未指定、または両方指定だけを許す。片方だけの設定、file 不在、hash 不一致は
runtime 構築前に `BackendUnavailableError` とする。これにより v3 の既定値を変えず、v4
設定だけを bundled-tokenizer 必須にできる。

## ランタイム隔離

v4 upstream は現行の `C:\Users\takut\Dev\Irodori-TTS` を更新せず、versioned な別
checkout に置く。checkout は commit を detached HEAD で固定し、実行証跡に commit、
checkpoint revision、model/tokenizer hash、Python executable を記録する。

v4 ランタイムを使う最初の GPU 検査では v3 checkpoint と v3 埋め込みを読み込み、
`neutral` と caption 付き style を各 1 件以上生成する。本文、seed、埋め込み、sampling
parameter は現行 v3 比較音声と一致させる。API 互換だけでなく、非空 WAV、有限値、
sample rate、例外なしを確認する。

この検査は別 process・別出力 root で行い、標準 HTTP サービスの環境変数や voice bank
manifest を更新しない。

## v4 Speaker Inversion 学習

v4 用埋め込みは v4-Small を base model とし、既存の clean manifest を入力として新規に
初期化する。v3 埋め込みからの継続学習は、base model 契約が異なるため既定経路にしない。
upstream の `configs/train_v4_small_speaker_inversion.yaml` を基準に、16 token、step 3000、
250 step 間隔の checkpoint、caption/speaker dropout 0 を証跡に固定する。

最初は 1 話者だけを pilot とする。pilot の loss、checkpoint inventory、推論互換、GPU
終了状態が通った後にだけ残りの話者へ広げる。v3 学習成果物と同じディレクトリへは出力しない。

## 評価ゲート

v4 用埋め込みは、少なくとも次を現行 v3 と同一 case で比較する。

- neutral と固定 style preset
- 複数 seed
- 話者類似度
- 文字誤り率または同等の書き起こし指標
- style 遵守
- beep・非有限値・無音・異常 duration の hard gate
- RTF、最大 VRAM、終了後 GPU 解放
- 人手確認用の WAV と review packet

caption が話者 identity と矛盾すると v4 でも品質が不安定になり得る。現行 caption に
性別や年齢の指定が含まれる case は個別に記録し、v4 学習や runtime の失敗と混同しない。
caption 文言の変更は公開 style 契約に関わるため、本移行では行わず、評価結果を受けた別変更とする。

v4 を配備可能と判定するには、全 hard gate を通り、現行 v3 より話者同一性・明瞭性を
悪化させず、style と性能が許容範囲内であることが必要である。自動で標準設定へ昇格しない。

## 監視ラッパーの再発防止

今回の v3 quality retraining では学習本体が成功した一方、所有する子 Python が生成した
`irodori_tts/__pycache__/*.pyc` を終了時の厳格検査が拒否し、terminal evidence が
`FAILED` になった。未追跡・ignored importable source を拒否する検査自体は維持する。

将来の versioned supervisor は、起動する queue process の環境だけに
`PYTHONDONTWRITEBYTECODE=1` を設定する。親 process やホスト全体の環境は変更しない。
テストでは、既存環境を保持しつつこの値だけを子 process に追加すること、および厳格検査が
既存の ignored bytecode を引き続き拒否することを確認する。既存の v004 remote bundle と
失敗証跡は変更せず、修正版は新しい bundle version としてのみ配置する。

## 実装単位

本設計は次の順に分割して実装する。

1. 監視ラッパーの bytecode 自己生成を TDD で防止する。
2. v3 既定値を保持した pinned snapshot 取得と optional tokenizer hash 検証を TDD で追加する。
3. v4 upstream の隔離 checkout と v3 checkpoint 後方互換 smoke を実行する。
4. v4 checkpoint と既存 v3 埋め込みの診断を 1 話者で実行する。
5. v4 用 Speaker Inversion pilot を新規学習し、同一評価 case で比較する。
6. pilot が通った場合だけ全話者学習と最終評価を別計画で進める。

この段階では 1 と 2 を repository 実装対象とし、3 以降は非配備の GPU operational gate
として扱う。
