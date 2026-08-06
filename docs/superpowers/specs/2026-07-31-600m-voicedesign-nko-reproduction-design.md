# 600M VoiceDesign移行と「んこ」系ピー音再現設計

## 目的

Windows GPU上にある13個のSpeaker Inversion埋め込みを
`Aratako/Irodori-TTS-600M-v3-VoiceDesign`と組み合わせ、「うんこ」、
「ちんこ」、「まんこ」を含む読み上げで狭帯域のピー音が発生する埋め込みを
特定する。再学習は行わず、再現に必要な埋め込み、文章、seed、caption条件と
確認用WAVを残す。

同時に、`irodori-tts-infra`の標準推論経路を600M VoiceDesignの公式契約へ
合わせる。話者同一性は既存のSpeaker Inversion埋め込みが担い、captionは
話し方だけを制御する。

## 確認済みの前提

- GPUホストの`C:\Users\takut\Dev\Irodori-TTS\speakers`には、
  `.speaker.safetensors`が13個ある。
- voice bank manifestは12個の異なる埋め込みを参照している。
  `narrator_sayoko.speaker.safetensors`だけはmanifest未登録であり、
  全件検査では直接指定する必要がある。
- `絢音`と`チヅル`は同じ埋め込みを共有しているため、表示名ではなく
  埋め込みファイル単位で検査する。
- 公式の600M VoiceDesignは、本文、参照音声またはSpeaker Inversion埋め込み、
  captionを同時に条件として与えられる。
- 既存のSpeaker Inversion埋め込みは500M v3を使って学習されている。
  upstreamはSpeaker Inversion埋め込みを学習時と同じ基盤モデルで使うよう
  案内しているため、600M VoiceDesignでの互換性は実機検査で確認すべき条件である。
  読み込み失敗や話者条件の破綻は、ピー音とは分けて記録する。
- GPUホストのinfra設定は600M VoiceDesignを指しているが、過去の起動ログには
  upstreamランタイムがモデル設定の`use_speaker_condition`を解釈できず
  起動に失敗した記録がある。別プロセスのloopbackサーバーは現在
  `127.0.0.1:8924`で正常稼働している。デプロイの再現性は未確認である。

## ピー音の検査範囲

### 対象文章

各埋め込みについて、次の7文を生成する。

1. `うんこ。`
2. `ちんこ。`
3. `まんこ。`
4. `「うんこ」という言葉を読み上げます。`
5. `「ちんこ」という言葉を読み上げます。`
6. `「まんこ」という言葉を読み上げます。`
7. `こんにちは。今日はいい天気ですね。`

7番は、同じ埋め込みとseedで通常音声にも定常音が現れるかを確認する対照文とする。
13埋め込み、7文、2seedの組み合わせにより、一次検査は182音声となる。

### 推論条件

- checkpoint:
  `Aratako/Irodori-TTS-600M-v3-VoiceDesign`
- style: `neutral`
- caption: なし
- `cfg_scale_text`: `3.0`
- `cfg_scale_caption`: `3.0`
- `cfg_scale_speaker`: `5.0`
- `cfg_guidance_mode`: `independent`
- `num_steps`: runtime設定値（今回のWindows `.env` では`30`）
- seed: `1234`と`5678`
- その他の値: infraの標準値

captionを外した一次検査は、Speaker Inversion埋め込みと本文の組み合わせを
切り分けるために必要である。候補が見つかった場合に限り、同一の本文とseedを
`calm` captionで再生成し、caption条件によって発生状況が変わるかを確認する。

## 判定方法

全182音声を耳だけで判定すると、確認時間が長くなり、短いピー音を見落としやすい。
そこで、機械検出とスペクトログラム確認を組み合わせる。

WAVをmonoの浮動小数点波形へ変換し、Hann窓2,048 sample、hop 480 sampleで
短時間フーリエ変換する。48 kHz音声では約42.7 msの窓と10 msのhopになる。
最初の全件走査では300 Hz以上を解析帯域にしたため、声の基音を90件検出した。
対照文とスペクトログラムで過検出を確認した後、低域を除外するのではなく、
80–5,000 Hz全体を純度計算の分母に含めて有声音の倍音を識別するよう校正した。
次の条件を満たす区間を候補とする。

- 500–5,000 Hzに主周波数がある
- 主周波数±40 Hzのエネルギーが、80–5,000 Hzのエネルギーの95%以上を占める
- 正規化spectral entropyが0.20以下である
- 候補frameが8個以上あり、最初と最後が10 hop以上離れている
- 3 frameまでの短い途切れを同一区間として扱う
- 区間内の主周波数の標準偏差が80 Hz以下である
- frame RMSが-45 dBFS以上である

しきい値は検査スクリプト内の定数として固定し、結果ファイルへ記録する。
母音や有声音も強い基音を持つため、検出だけではピー音と断定しない。候補区間の
スペクトログラムを作り、同じ埋め込みとseedの対照文と比較する。

判定は次の4段階とする。

- `CLEAR`: 検出条件を満たす区間がない
- `CANDIDATE`: 狭帯域の定常音を検出したが、対照との差を確定できない
- `REPRODUCED`: 対象文だけに狭帯域の定常音が現れ、スペクトログラム上でも
  同一区間を確認できる
- `ERROR`: モデルロード、埋め込み読込、推論、WAV保存のいずれかに失敗した

`CANDIDATE`と`REPRODUCED`にはWAV、スペクトログラム、埋め込みファイル名、
文章、seed、推論パラメーター、候補区間の開始・終了時刻、主周波数を残す。
`ERROR`には例外型とメッセージを残し、600M VoiceDesignとの互換性問題を
ピー音判定へ混ぜない。
生成物はGit管理外の作業用artifactディレクトリへ保存する。

## 600M VoiceDesign対応

公開HTTP APIは任意captionを受け取らず、`neutral`、`calm`、`cheerful`、
`clear`の固定styleだけを受け取る。サーバーはstyleを固定captionへ変換する。
`neutral`はcaptionを付与しない。

標準推論経路は、公式`SamplingRequest`へ次の値を渡す。

- `caption`
- `ref_embed`
- `cfg_scale_caption`
- `cfg_scale_speaker`
- `cfg_guidance_mode="independent"`

既定checkpointを600M VoiceDesignへ変更し、warmupでは`calm`とnarratorの
Speaker Inversion埋め込みを同時に使う。これにより、caption分岐とspeaker分岐の
両方を起動時に検証する。

デプロイ後のランタイムは、600Mモデル設定を読み込めるupstream Irodori-TTSと
公式`SamplingRequest`の必須引数を備えていなければならない。ローカル単体テストが
通っても、古いupstream checkoutから構築したWindowsランタイムでは起動できない。
GPU検証では、稼働中サーバーと分離した一時デプロイ先でbootstrapからserver起動までを
再実行し、この差を解消する。検証成功前に現在のruntimeやプロセスを置き換えない。

## テストと検証

既存の未コミット差分は保持し、足りない振る舞いをテストで先に固定する。

- API契約: styleと`cfg_scale_caption`の受理、任意captionの拒否
- ジョブ伝播: styleと`cfg_scale_caption`が分割・バッチ経路で失われない
- backend: caption、Speaker Inversion、独立CFGが公式requestへ渡る
- warmup: `calm` captionとnarrator埋め込みを同時に使う
- 設定: 既定checkpointとVoiceDesign向け既定値
- GPU: 600Mモデルのロード、全13埋め込みの生成、ピー音検査

Python変更後は対象テストから始め、ruff、format、mypy、vulture、default pytestを
実行する。GPU依存の検査はWindowsで実行し、ローカル検証と分けて結果を記録する。

## 完了条件

- 13個すべての埋め込みについて、成功または`ERROR`を含む182件の結果行がある
- `CANDIDATE`または`REPRODUCED`には再確認できるWAVと条件がある
- 候補は`calm` captionでも再生成され、条件差が記録されている
- infraの単体・静的検証が通る
- Windowsで600M VoiceDesignを読み込み、Speaker Inversionとcaptionを
  同時指定した推論が成功する
- 再学習やモデルファイルの変更は行わない
