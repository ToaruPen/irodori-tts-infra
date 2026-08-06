# Irodori VoiceDesign サービス設計

> 本仕様は
> `2026-05-21-v3-base-speaker-inversion-design.md`
> の「VoiceDesignを標準経路に含めない」という判断を更新する。話者同一性を
> Speaker Inversionが所有する境界は維持する。

## 目的

Windows GPU上の Irodori TTS を `Aratako/Irodori-TTS-600M-v3-VoiceDesign`
へ移行し、既存の Speaker Inversion 話者を保ったまま、Pico が安全で
再現可能な話し方プリセットを指定できるようにする。

## 境界

- 公開HTTP APIは、任意の VoiceDesign caption や話者埋め込みを受け取らない。
- クライアントは `style` として `neutral`、`calm`、`cheerful`、`clear`
  のいずれかだけを指定する。
- サービスが `style` を日本語の固定captionへ決定論的に変換する。
- `neutral` はcaptionを付与しない。
- 話者は既存の Speaker Inversion voice bankから名前で選ぶ。
- VoiceDesign captionとSpeaker Inversion埋め込みは同時に推論へ渡す。
- 自動フォールバックは設けない。起動失敗や推論失敗は明示的な失敗として返す。
- Windows側のHTTPポートはloopbackに限定し、Picoからは既存の保護された
  SSHトンネルを通して接続する。

## 話し方プリセット

| style | VoiceDesign caption |
| --- | --- |
| `neutral` | captionなし |
| `calm` | 穏やかで優しい女性の声で、自然に話す。 |
| `cheerful` | 明るく親しみやすい女性の声で、自然に話す。 |
| `clear` | 子どもに伝わるように、ゆっくり明瞭な女性の声で話す。 |

captionの追加・変更はAPI契約の変更として、音質と推論時間を再評価する。

## HTTP契約

`POST /synthesize` の既存リクエストに次を追加する。

- `style`: 上記4値。既定値は `neutral`。
- `cfg_scale_caption`: 正の数。既定値は公式実装と同じ `3.0`。

ジョブキューおよび分割合成でも両値を失わず、各セグメントへ同一値を渡す。
レスポンス形式、WAV形式、ジョブ状態契約は変更しない。

## ランタイム

- 既定checkpointを `Aratako/Irodori-TTS-600M-v3-VoiceDesign` とする。
- warmupは `calm` を使い、caption経路とSpeaker Inversion経路の両方を
  起動時に検証する。
- 推論では公式 `SamplingRequest` に `caption`、
  `cfg_scale_caption`、`cfg_guidance_mode="independent"` を渡す。
- テキスト、caption、speakerのCFGは独立ガイダンスを使う。

## Speaker Inversion互換性ゲート

既存のSpeaker Inversion埋め込みは500M v3を基盤として学習されている。
upstreamは埋め込みをinversion学習時と同じ基盤モデルで使うよう案内しているため、
600M VoiceDesignで読み込めることだけでは互換性を確認したことにならない。

既定checkpointの切り替えは、Windows GPU上で既存voice bankの埋め込みを使い、
captionなしとcaptionありの両方で合成成功、音声の非破損、話者同一性を確認してから
行う。互換性または品質を確認できない場合は500M v3を標準のまま維持し、
600M VoiceDesign用のSpeaker Inversion再学習を別変更として扱う。

## 検証

- 契約、設定、キュー伝播、バックエンド引数を単体テストする。
- Windows RTX 4070で短文、石垣市の天気相当文、寿限無相当長文を合成する。
- 初回とwarm時の応答時間、音声長、RTF、GPUメモリピークを記録する。
- 同一話者・同一条件でcaptionなしと `calm` を比較し、追加オーバーヘッドを確認する。
- `/health` が正常で、外部LANへ直接ポートを公開していないことを確認する。

## 切り戻し

切り戻しはWindowsの環境変数でcheckpointを従来の
`Aratako/Irodori-TTS-500M-v3` に戻し、VoiceDesignを必要としない
`neutral` リクエストだけを送る明示操作とする。実行時の自動切り替えは行わない。
