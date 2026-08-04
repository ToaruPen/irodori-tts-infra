# Irodori-TTS v4 複数話者 Speaker Inversion 作成設計

## 目的

`oop53_aibeya_sp_f7269f5ffc` で成立した v4-Small の Speaker Inversion と実用判定を、
既存のclean-data契約に含まれる残り11話者へ適用する。生成物はすべて既存v3成果物・現行
voice bankと分離したcreate-only領域に保存し、自動配備しない。

## 対象

既存12話者のうち、pilot完了済みの `oop53_aibeya_sp_f7269f5ffc` を除く次の11話者を対象とする。

- `oop176_natsu_no_owari_sp_dcec9a11d3`
- `oop52_aibeya_2_sp_5d544fe890`
- `oop54_aikagi_2_sp_85dded42a7`
- `miu`
- `oop68_maid_san_no_iru_kurashi_s_sp_e4da3225a4`
- `oop69_maid_san_no_iru_kurashi_sp_07497b0fbd`
- `oop70_osananajimi_no_iru_kurashi_sp_7195504dbb`
- `oop73_toshishita_kanojo_sp_6b50dbf844`
- `oop77_anabel_maidgarden_sp_451488a7c1`
- `narrator_sayoko`
- `kasumi`

`miu` はclean-data契約どおり `oop55_aikagi_3_sp_683c9895cc` のデータを使う。
`narrator_sayoko` は `narrator_toshiue_ama` のデータを使う。

## 固定入力

- base model: `Aratako/Irodori-TTS-v4-Small`
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- upstream commit: `8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71`
- base model SHA-256:
  `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- 各話者の既存immutable `clean-manifest.jsonl` とそのSHA-256
- 各話者の既存140-case評価manifest、参照音声、ECAPA/Whisper cache
- 7 text IDs、2 synthesis seeds、2 styles、5 checkpointの評価契約
- Speaker Inversion shape `F32[16,768]`

棚卸し時点で11話者すべてのclean manifestが存在し、660〜5,202 rowsで読み取り可能である。
旧v3評価で選定が空の話者があるため、v3選定embeddingを共通初期値にはしない。

## 作成方式

### Phase 1: 共通scratch訓練

各話者を次の共通条件でゼロから直列訓練する。

- token数16
- learning rate `0.01`
- seed `2`
- effective batch size16（batch4、gradient accumulation4）
- gradient checkpointing無効
- bf16、TF32
- 3,000 optimizer steps
- 250 stepsごとに保存
- `rf_loss_mode: utterance_mean`
- condition dropoutはすべて0

これはoop53の既存baselineと同じ比較可能な初期経路である。話者ごとの同時訓練は行わず、
1 GPU上で1 processずつ実行する。途中失敗はその話者のFAIL証跡を残して次へ進む。

### Phase 2: 共通評価と実用判定

各話者のstep 1000、1500、2000、2500、3000を140-caseで評価する。話者ごとの既存
reference WAV集合を使用し、閾値・text・style・seedは変更しない。

checkpointは次のすべてを満たした場合に `PRAGMATICALLY_ELIGIBLE` とする。

1. 140/140 caseが成功する。
2. 16 speaker hard-gate case中14件以上がnormalized ECAPA `0.75`以上である。
3. 未達は最大2件で、各値が`0.72`以上である。
4. 16件平均が`0.765`以上である。
5. CER、duration、RMS、silence、clipping、styleを含む他のhard-gate失敗が0件である。
6. 学習・評価・embedding・参照・scriptのhash bindingが再検証できる。

旧strict判定は変更せず、16/16が`0.75`未満ならstrict上は不合格のまま保持する。同一話者に
複数の実用合格checkpointがある場合は、strict合格、未達件数、最小値、平均値、早いstepの順に
決定的に選ぶ。

### Phase 3: 不合格話者だけのrefinement

Phase 2で実用合格しなかった話者だけを追加対象とする。全11話者にoop53の26段階探索を
機械的に再実行しない。最初にcheckpoint別の未達件数・最小値・平均値を比較し、最良のexact
checkpointからlearning rate `0.0001`、最大1,500 stepsの局所継続を行う。

full-clean継続で不合格の場合のみ、固定ECAPA参照centroidに対する学習音声のmedian以上を保持した
話者別 `central_q50` をcreate-onlyで作り、pilotで最終的に改善を示した `echo` RF lossを1回だけ
検証する。cutoffとretained source IDは学習前に固定する。同じ条件のseed・rate列挙は行わない。

Phase 3終了時に未達話者が残る場合は、低品質モデルを合格扱いせず、利用可能な最良checkpointと
失敗caseを報告して区切る。

## 証跡と再開

remote rootは
`v4_compatibility/v4_speaker_inversion_multi_v001` とし、話者ごとに `training`、`evaluation`、
`acceptance` を分離する。queueは状態をJSONLで追記し、入力hashが一致する成功済み話者だけを
skipできる。各run root、設定、checkpoint、評価結果、選定decisionは上書きしない。

detached supervisorは自分が起動したprocessだけを所有し、既存serviceや他の訓練processを停止しない。
GPU空き、競合process、upstream commit、model/tokenizer、manifest、設定をfail-closedで確認する。

## 安全境界

- 現行v3 serviceとvoice bankを変更しない。
- v4 base model、既存clean manifest、既存評価成果物、oop53 pilot成果物を変更しない。
- 自動配備、active設定変更、モデル名の置換をしない。
- dataset、checkpoint、生成音声をGitへ追加しない。
- operational failureの再実行は新しいversioned rootに限定する。

## 完了条件

- 11話者すべてに、少なくとも1つの正常なv4 `F32[16,768]` embeddingが作成されている。
- 各話者について140-case評価と実用判定が完了している。
- 実用合格話者は決定的に選定され、不合格話者は追加refinementまたは明示的な停止理由を持つ。
- 全入力・設定・script・checkpoint・decisionのhashが再検証できる。
- 現行serviceとvoice bankが開始前から不変である。

## 実行確定事項（2026-08-04）

設計どおりのbounded refinementまで完了し、最終verifierは `PASS` した。oop53 pilotを含む
12話者のうち8話者が実用合格、4話者はfull-cleanとcentral-q50 + echoの両方を実施しても
基準未達だったため `REJECTED_EXHAUSTED` とした。探索範囲は追加していない。

Phase 1で合格した7話者はrefinement対象から除外した。不合格4話者のfull-cleanとechoは、
それぞれ5 checkpoint × 28 caseを140/140生成・評価し、運用失敗0で最終decisionを閉じた。
実行途中に見つかったgenerator、bundle dependency、directory binding、evaluation rootの暗黙前提は、
versioned recovery rootで明示契約へ置き換えた。失敗rootは監査証跡として保持し、成功成果物へ
上書きしていない。

最終成果物は
`v4_compatibility/v4_speaker_inversion_multi_v001/final-verification-v001.json`を正本とする。
そこには12話者の最終段階、選定または最良checkpoint、未達case、全hash binding、queue証跡、
v3/voice bank不変、v4 upstream commit、終了時runtime idleが含まれる。配備は別作業であり、
本実行では行っていない。
