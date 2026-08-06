# 600M VoiceDesign向け全話者再学習・クリーンデータ設計

## 目的

GPUホストにあるSpeaker Inversion embeddingを、
`Aratako/Irodori-TTS-600M-v3-VoiceDesign`を初期checkpointとして再学習する。
同時に、元音声を変更せず、人工的なピー音を除外しながら有効な発声を最大限残した
再現可能な学習データを作る。

アイとミウは同じOOP55音声集合から作られたembeddingであるため統合する。
OOP55の採用可能ファイル数を最大化したデータで学習したembeddingを
`miu.speaker.safetensors`とする。再学習対象は12モデル、固有の音声集合も12個とする。

## 対象

- OOPPEENN由来の既存10音声集合のうち、OOP55はミウとして扱う。
- カスミは過去の加工済みmanifestではなく、772件の元manifestから再監査する。
- `narrator_sayoko`は`narrator_toshiue_ama_20260530`の2,469件の元データから再監査する。
- 現行OOPPEENN 9話者、ミウ、カスミ、Sayokoの計12 embeddingを生成する。
- 旧embedding、旧manifest、元音声は上書きしない。検証済みの新成果物だけを別の
  stagingディレクトリへ置く。

## 基本方針

### 品質の定義

品質不良として除外するのは次に限る。

- 人工的なピー音を含む音声
- デコード不能、空、全ゼロなど技術的に無効な音声
- 極端な破損や、音声とキャプションの対応が復元できない音声
- 同一音声のbyte単位またはPCM単位の完全重複

喘ぎ声、吐息、囁き、うめき声、笑い、泣き、短い非言語発声、低音量の発声は
有効な話者表現として保持する。音量、ASR成功率、短さ、成人向け語彙、
スペクトルのノイズ性だけを理由に除外しない。

### 元データ不変

元音声へnotch filter、無音置換、区間補間などの加工は行わない。
除外はclean manifestから参照を外すことで行い、採用音声は元波形のまま使う。
全判断には元ファイル、理由、特徴量、ラベル履歴を残す。

## データ監査

### 状態

全レコードを次のいずれかへ分類する。

- `KEEP`: 学習へ採用する。
- `KEEP_RECAPTIONED`: 音声にピー音がなく、伏字を実発声へ復元したキャプションで採用する。
- `REVIEW`: 自動判定ではピー音と有効発声を区別できない、またはキャプション復元が曖昧である。
- `EXCLUDE_CONFIRMED_TONE`: 確認済みの人工的なピー音を含む。
- `EXCLUDE_INVALID_AUDIO`: デコード不能、空、全ゼロ、明白な破損である。
- `EXCLUDE_TRANSCRIPT_MISMATCH`: 音声とキャプションの対応を信頼できず、復元もできない。
- `EXCLUDE_DUPLICATE`: 同じPCMを持つ重複レコードの2件目以降である。

`REVIEW`が残っているデータ集合からは学習を開始しない。

### 広帯域監査

44.1kHzまたは48kHz音声の80Hz–20kHzを走査する。単一の周波数帯だけでは
除外しない。初期候補は次の特徴を組み合わせて抽出する。

- 狭帯域ピークの継続時間
- 区間内周波数の標準偏差
- 周辺帯域に対するピーク比
- spectral entropy
- 倍音列の有無
- 振幅包絡の急な開始・終了
- 既知の703Hzおよび公称1kHz固定音との一致
- 同じ収録作品内で隣接する音声の共通パターン

声の基音や喘ぎ声は倍音列または自然な周波数変動を持つため、狭帯域候補という
理由だけで自動除外しない。既に人がピー音と確認した特徴へ十分近いものだけを
高信頼候補とし、それ以外は`REVIEW`へ送る。

### 人手ラベルによる先鋭化

`REVIEW`を周波数、純度、継続時間、包絡、音源、キャプション記号でcluster化し、
各clusterの代表、境界、外れ値を確認用WAVとして提示する。ユーザーの
`TONE`、`VOICE`、`UNSURE`ラベルを永続化し、次の規則で再走査する。

- 人が直接`TONE`と確認した有効な一意音声は、自動判定が`KEEP`でも
  `EXCLUDE_CONFIRMED_TONE`を優先する。無効音声と重複除外はそれより強い。
- `TONE`と同質な保守的clusterだけを`EXCLUDE_CONFIRMED_TONE`へ伝播する。
- `VOICE`は自動判定が`REVIEW`の候補だけを`KEEP`へ戻し、未解決の伏字修復は
  回避しない。
- decision boundaryと`UNSURE`は再度`REVIEW`に残す。
- 新しいラベルごとに全12データ集合を再走査し、未解決候補がなくなるまで反復する。

周波数だけを学習境界にせず、既知帯域外のピー音も候補化できる状態を維持する。

### 伏字キャプション

`◯`、`○`、`〇`などの伏字記号だけを理由に除外しない。

1. 音声に確認済みピー音があれば除外する。
2. 音声が実際の発声で、語が文脈と音声から一意に復元できればキャプションを修復する。
3. 複数の復元候補がある、または発声と一致するか不明な場合は`REVIEW`へ送る。

修復前後の文字列、適用規則、ユーザーラベルを監査ログへ残す。推測だけで
キャプションを生成しない。

### 採用件数の最大化

既存のfiltered manifestから始めず、利用可能な元indexまたは元manifestを
source of truthとする。過去に伏字、成人向け語彙、広すぎるtone scoreだけで
除外されたレコードも再評価し、ピー音がなくキャプションが正しければ復帰させる。

データ集合ごとに元件数、採用件数、状態別除外件数、復帰件数、採用率を出力する。
ミウはOOP55集合の採用率を最大化した唯一のembeddingとし、アイ用の再学習は行わない。

## 再現可能なリモート実行環境

Windows上の監査、manifest生成、Speaker Inversion学習はPowerShellの`python`解決へ
依存しない。`Irodori-TTS`プロジェクトと既存`.venv`を明示し、
`uv run --project ... --no-sync --python ...`で起動する。実行中にlockfileや依存環境を
同期・変更しない。

リポジトリの`justfile`は`.env`の`IRODORI_REMOTE_HOST`だけを接続先として使い、
資格情報や実ホスト値を保持しない。任意スクリプトを固定環境で実行する
`remote-python`と、監査入力を固定して指定roundへ出力する`remote-audit`を公開する。
引数はシェル展開から保護し、`just --list`と`just --dry-run`で実行内容を確認できる。

## clean manifest

clean datasetは実体音声を複製せず、元音声への参照と監査結果を持つ。
既存latentsと元音声の対応がhashで確認できた行はlatentsを再利用する。
新たに復帰した行だけ、現在のupstream `prepare_manifest.py`でDACVAE latentを生成する。
全manifest行について、音声hash、latentの存在、text、frame数を検証する。

Speaker Inversion学習では1話者を1 runとして扱うため`caption`は空にする。
600M VoiceDesignのcaption branchと基盤モデルは凍結し、Speaker Inversion tokensだけを
学習する。推論時の固定style captionが表現を担当し、embeddingは話者同一性を担当する。

## 600M VoiceDesign再学習

### 初期checkpointと設定

- checkpoint: `Aratako/Irodori-TTS-600M-v3-VoiceDesign`
- snapshot: 実行時にhashとrevisionを記録する。
- upstream: 実行時のGit commitを記録する。
- model config: checkpointのsafetensors metadataから取得し、caption、speaker、durationの
  各branchを含む完全な600M設定と一致させる。
- `speaker_inversion_enabled: true`
- `speaker_inversion_tokens: 16`
- 基盤モデルとcaption branchは凍結する。
- 定期checkpointを250 stepごとに保存し、最終stepだけを自動採用しない。

既存の500M embeddingを初期値として使わず、600M checkpointに対して新規に
Speaker Inversion tokensを初期化する。これにより世代をまたぐembedding不整合を除く。

### pilot gate

12件を連続学習する前に1話者でpilotを実施する。

- 600M model configで学習が開始できる。
- trainable parameterがSpeaker Inversion tokensだけである。
- 250 step checkpointを600M推論で読み込める。
- 通常文と「んこ」系文でWAVを生成できる。
- captionなしと固定style captionの両方で破綻しない。

pilotが通らない場合は全話者学習を開始しない。

### checkpoint選択

全採用ファイルを学習へ使い、validation splitによって件数を減らさない。
250 stepごとのcheckpointを固定評価文、固定seed、複数styleで比較する。
選択には少なくとも次を使う。

- ピー音を含む狭帯域異常がない。
- 音声が空、破損、極端な途切れにならない。
- 既存の話者同一性指標が許容範囲にある。
- text品質指標が悪化しない。
- styleを変えたときに表現が変わり、話者同一性が維持される。

最終checkpointを無条件に採用せず、条件を満たすcheckpointの中から品質が最良のものを
staging embeddingとする。

## 生成音声検証

12 embeddingについて、少なくとも次を固定seedで生成する。

- `うんこ。`、`ちんこ。`、`まんこ。`
- 上記3語を文中に含む文章
- 通常の対照文
- `neutral`、`calm`、`cheerful`、`clear`のstyle

既存の182件再現条件を、12 embedding向けのマトリクスへ更新する。
旧embeddingで再現したMio、Ayaki、Anabel条件は必須回帰条件とする。
機械検出候補にはWAV、スペクトログラム、条件、checkpointを残し、ユーザーが
最終確認できるようにする。

## 完了条件

- 12データ集合すべてに未解決`REVIEW`がない。
- 採用manifestに、確認済みピー音と同じ特徴を持つ音声がない。
- 全広帯域候補がユーザーラベルまたは保守的な同質cluster判定で説明できる。
- 伏字行は、確認済みピー音として除外されるか、根拠付きで修復・採用されている。
- 喘ぎ声、吐息、囁き、短い非言語発声が品質不良として一律除外されていない。
- 状態別件数と採用率を12集合すべてで報告できる。
- 12個の600M向けembeddingが生成され、各checkpointの初期モデルrevisionを追跡できる。
- 「んこ」系回帰マトリクスと通常文対照で機械検出上のピー音候補が解決済みである。
- 旧embeddingと現行サービスは、新成果物の検証完了まで変更されていない。

## 非目標

- 元音声の修復加工
- Irodori-TTS基盤モデル自体のfine-tuning
- 音声からの推測だけによる大量の自動キャプション生成
- 検証前の現行voice bankへの自動配備
