# Irodori-TTS v4 手動実用合格とoop70再学習設計

## 目的

既存の自動品質判定を改変せず、ユーザーの聴感判断に基づいて
`oop176_natsu_no_owari_sp_dcec9a11d3`、`oop52_aibeya_2_sp_5d544fe890`、`miu`を
手動実用合格として記録する。残る`oop70_osananajimi_no_iru_kurashi_sp_7195504dbb`は、
音声長と文字数によるECAPA類似度の偏りを補正した学習manifestを作り、Speaker Inversionを
最初から学習し直す。

本作業はcreate-onlyの評価・学習作業であり、配備、現行v3 service、voice bank、既存decisionを
変更しない。

## 先行結果の扱い

`v4_speaker_inversion_multi_v001/final-verification-v001.json`は、次の自動判定を記録した
immutableな監査証跡として保持する。

- 自動実用合格: 8話者
- bounded refinement終了後の自動不合格: oop176、oop52、miu、oop70
- 自動判定policy: hard 16件中14件以上が0.75以上、最大2外れ値は各0.72以上、
  平均0.765以上、その他hard gate失敗0

後続の手動実用合格はこのファイルを書き換えず、別のoperator decisionとして表現する。

## 手動実用合格override

### 対象と選択checkpoint

| 話者 | 選択段階 | step | 自動判定上の注意点 |
|---|---|---:|---|
| `oop176_natsu_no_owari_sp_dcec9a11d3` | central-q50 + echo | 750 | similarityは16/16合格、tone候補2件 |
| `oop52_aibeya_2_sp_5d544fe890` | central-q50 + echo | 750 | similarity未達5件、平均0.759852 |
| `miu` | central-q50 + echo | 1500 | similarity未達2件、平均0.760725 |

### 証跡契約

手動overrideは`manual-acceptance-override-v001.json`としてcreate-only保存する。各話者について
次を必須とする。

- 元の`NO_PRAGMATIC_CANDIDATE` decisionへのpathとSHA-256
- 選択したembeddingへのpath、SHA-256、`F32[16,768]`、全値finiteの検証結果
- 対応するtraining terminal、evaluation terminal、140-case結果へのhash binding
- 自動判定値と、不合格理由を保持したままのoperator acceptance
- 承認根拠がユーザーの聴感判断であること
- `deployment_performed: false`と`active_voice_bank_unchanged: true`

自動policy、既存acceptance decision、`final-verification-v001.json`は変更しない。後続の集約では
`automated_status`と`operator_status`を別フィールドで保持し、自動合格と手動合格を混同しない。

## oop70の原因仮説

oop70の2209学習音声は、25 referenceに対するECAPA類似度が平均0.782964、中央値0.786982である。
一方、生成音声の最良平均は0.718988付近で、Phase 1、full-clean、central-q50 + echoのいずれでも
大きく変化しなかった。

学習音声の類似度はdurationと0.645030、text lengthと0.399257の相関を持つ。前回の
`central_q50`は生の類似度上位50%を選んだため、話者らしさだけでなく長い音声を優先した可能性が
ある。次の再学習では、生の類似度順位を使わず、durationとtext lengthで説明される成分を
差し引いた残差を選択指標にする。

## duration-balanced reference-residual manifest

### 固定入力

- 元manifest: oop70の既存2209件`clean-manifest.jsonl`
- ECAPA結果: `central-q50-audit-v001/training-ecapa-results.jsonl`
- reference: 既存の固定25 WAV
- ECAPA model tree: 既存hash-bound snapshot
- 元manifest順序、latent path、text、duration、source ID

### 選択アルゴリズム

1. 2209件すべてについて、speaker similarity、duration、text length、latent assetを再検証する。
2. `log1p(duration_seconds)`と`log1p(text_length)`を説明変数、speaker similarityを目的変数とする
   OLSを決定的に計算する。
3. 各音声について`observed_similarity - predicted_similarity`をreference residualとする。
4. durationの五分位で5 binを固定し、各bin内でresidual降順、同値時はsource ID順に並べる。
5. speaker similarityが0.72以上の音声から各bin160件、合計800件を選ぶ。1 binでも160件を
   満たせない場合はmanifestを作らずfail-closedで停止する。
6. 選択後は元manifest順へ戻し、latent asset 800件の存在とidentity hashを検証する。

作成物にはOLS係数、五分位境界、各binの候補数・選択数、selected/excluded source ID、元manifestと
ECAPA結果のSHA-256、作成scriptのSHA-256を保存する。選択集合の平均speaker similarityが元2209件の
平均0.782964未満、または平均residualが0以下の場合も学習前に停止する。

## 再学習

新しい800件manifestから既存checkpointをwarm-startせず、Speaker Inversionを最初から作成する。

- token数16
- learning rate `0.01`
- seed `2`
- effective batch size16（batch4、gradient accumulation4）
- bf16、TF32
- gradient checkpointing無効
- condition dropoutはすべて0
- `rf_loss_mode: utterance_mean`
- 3000 optimizer steps
- 250 stepsごとにcheckpoint保存

seedを変更しないことで、既存Phase 1との差をmanifest変更へ限定する。訓練root、config、setup、
checkpoint、terminalは新しいversioned create-only領域へ保存する。

## 評価と分岐

step 1000、1500、2000、2500、3000を、既存と同じ5 checkpoint × 28 case、合計140 caseで評価する。
text、seed、style、reference、ECAPA、Whisper、全hard gateは変更しない。

### 合格

従来の自動policyを一切変更せず、次をすべて満たすcheckpointを合格とする。

- 140/140生成・評価成功
- hard 16件中14件以上がspeaker similarity 0.75以上
- 最大2外れ値が各0.72以上
- hard 16件平均が0.765以上
- その他hard gate失敗0

### echo追加条件

scratchで自動合格しなかった場合、全5 checkpoint中の最高平均を旧最高平均0.71898819と比較する。
新最高平均が0.73898819以上の場合だけ、同じ800件manifestとscratch最良exact checkpointから
learning rate `0.0001`、echo loss、最大1500 stepsの継続学習を1回行う。

改善が0.02未満ならechoを実行せず、データ再選定が有効でなかった証跡を残して終了する。echoでも
自動policyを満たさない場合は、追加seed、追加rate、追加subsetを探索せず区切る。

## 実行構成

新しいversioned tool bundleとrun rootを使用する。既存のPhase 1、full-clean、echo rootは読み取り専用の
入力として扱う。

処理順は次のとおりとする。

1. 手動overrideのpreflightとcreate-only保存
2. oop70 residual manifestのbuildと独立verify
3. GPU、競合process、v4 commit、base checkpoint、tokenizer、script hashのpreflight
4. scratch訓練
5. 140-case評価と自動decision
6. 改善条件を満たす場合だけecho訓練・評価
7. operator dispositionと最終verificationのcreate-only保存

detached supervisorは自分が起動したprocessだけを所有する。既存service、他の訓練process、stale pidを
停止・削除しない。

## 最終disposition

後続の`operator-disposition-v001.json`では12話者を次のいずれかに分類する。

- `AUTOMATED_ACCEPTED`
- `MANUAL_ACCEPTED`
- `REJECTED_AFTER_DURATION_CORRECTED_RETRAINING`

oop70が合格すれば12話者すべてがoperator上の実用合格となる。未達なら、11話者を実用合格、oop70を
再学習後不合格として閉じる。どちらの場合も自動判定の履歴とmanual overrideを別々に追跡できることを
最終verifierで確認する。

## テストと検証

runtime toolはTDDで作成する。

- OLS residualとduration bin境界の決定性
- 5 bin × 160件、重複0、元順序保持
- 0.72 floor、平均類似度、平均residualのfail-closed条件
- manifest、latent、reference、ECAPA model、scriptのhash binding
- manual overrideが元decisionを改変せず、選択embeddingと評価へ結び付くこと
- scratchと条件付きechoの状態遷移
- policy不変と140-case完全性
- v3、voice bank、service不変
- 終了時に学習・queue process 0、GPU idle

## 完了条件

- 3話者の手動実用合格overrideがimmutableな自動判定へhash bindingされている。
- oop70の800件manifestが決定的かつ独立検証可能である。
- oop70 scratchの3000-step訓練と140-case評価が完了している。
- 改善条件に従ってechoを実行またはskipした証跡がある。
- 12話者のoperator dispositionが閉じている。
- 配備、現行v3、voice bank、既存decisionに変更がない。

## 実行結果（2026-08-04）

### 手動override

`manual-acceptance-override-v001.json`をcreate-only作成し、SHA-256
`c1710ecd22a99ad402f2634805d5128a3d79e5203ef1f59c4283c5fb216fbde5`で固定した。3 embeddingは
すべて`F32[16,768]`かつfiniteで、各140-case評価と元の`NO_PRAGMATIC_CANDIDATE` decisionへ
hash bindingされている。

| 話者 | step | embedding SHA-256 | hard 16件平均 | operator判定 |
|---|---:|---|---:|---|
| `oop176_natsu_no_owari_sp_dcec9a11d3` | 750 | `750d95d3895f036d840e069dfb47018c4e98f6c61253f5cd185e384bc584aeb5` | 0.79867819 | `MANUAL_ACCEPTED` |
| `oop52_aibeya_2_sp_5d544fe890` | 750 | `201beaca8962ec04764a79f0949a40cd270a8f707cacd7d3532393daf57ef4ff` | 0.75985176 | `MANUAL_ACCEPTED` |
| `miu` | 1500 | `528fb261969cced5ce1508354330628b9e8dd01773d47d7684307c6dfa2401b5` | 0.76072490 | `MANUAL_ACCEPTED` |

### residual manifest

2209件から各duration bin 160件、計800件を選び、独立verifierは`PASS`だった。manifest SHA-256は
`eff0bebf5884b0c65d65998560c27352ed975edb838ccd14163d133279752562`、verification SHA-256は
`e039c92ae2053c1e05be5b5139e99e9f0bb58f9642317b2f9413938932b045d7`である。

- OLS係数: `[0.7479433943528998, 0.1445236441819881, -0.07789122255320478]`
- duration境界: `[2.25775, 4.098937500000001, 6.4464250000000005, 9.899912500000005]`
- eligible件数: `[244, 336, 380, 401, 440]`
- source平均類似度: `0.7829642909888729`
- selected平均類似度: `0.826399930468916`
- selected平均residual: `0.04233521836803627`
- floor違反 / latent欠落: `0 / 0`

| duration bin | 選択数 | 平均類似度 | 平均residual |
|---:|---:|---:|---:|
| 0 | 160 | 0.78106143 | 0.05254494 |
| 1 | 160 | 0.80178388 | 0.04275929 |
| 2 | 160 | 0.82789341 | 0.04488067 |
| 3 | 160 | 0.85023148 | 0.03752444 |
| 4 | 160 | 0.87102944 | 0.03396676 |

### oop70再学習・評価

`duration-residual-scratch-v003`は3000 stepを終了コード0で完走し、250 step刻み12 checkpointと
finalを保存した。最大GPU使用量は6410 MiB、terminal SHA-256は
`3208787dab5844b1500af08515642f29c67e3a994b202964a60b946926f998a7`である。評価は140/140生成成功、
全音声finiteだった。

| step | hard 16件平均 | 最小 | similarity未達 | その他失敗 | 判定 |
|---:|---:|---:|---:|---:|---|
| 1000 | 0.70438040 | 0.67063085 | 16 | 1 | rejected |
| 1500 | 0.71095193 | 0.68256550 | 16 | 1 | rejected |
| 2000 | 0.70733061 | 0.69386804 | 16 | 0 | rejected |
| 2500 | 0.71592508 | 0.68871471 | 15 | 0 | rejected |
| 3000 | 0.71197509 | 0.67454606 | 15 | 0 | rejected |

最高平均0.71592508は旧最高0.71898819より低く、echo開始閾値0.73898819にも届かなかったため、
branch decisionは`STOP_NO_IMPROVEMENT`、queue terminalは`SKIPPED_NO_IMPROVEMENT`とした。echoは
作成していない。単一話者acceptanceではraw evaluation rowsを正本としてpolicyを再計算し、集合用
candidate decisionはadvisoryなhash-bound sourceとして保持した。

### 最終disposition

`operator-disposition-v001.json`はfresh verifierと別process auditの双方で`PASS`。SHA-256は
`8d36021f1c45fe2a3a9a987cae2e321ac5a7dfdf2db351b2dded52aaa11935a1`で、内訳は
`AUTOMATED_ACCEPTED=8`、`MANUAL_ACCEPTED=3`、
`REJECTED_AFTER_DURATION_CORRECTED_RETRAINING=1`。runtimeはGPU free 11,078 MiB、training/queue process 0。
v3 Git commit `eaf74d6a19138f743acb5b71a445fd25a57db987`とtree
`b7ceab2449344e4c3a32c1e83ab5c71df429922a`はcleanで、voice bank snapshot SHA-256
`214809c34d6f530e5929b12d8bb5f9223ef52cf8dee5f4d53a617e41fca3d13d`も不変、配備は実施していない。

### 後続operator採用（2026-08-05）

再学習の自動結果は`NO_PRAGMATIC_CANDIDATE`のまま保持し、ユーザー指示により5候補中の最高平均だった
step 2500を手動採用した。採用embeddingは
`duration-residual-scratch-v003/training/checkpoint_0002500.speaker.safetensors`、SHA-256
`bfadf21004c16b0a71ed44e0884ab53342e36ed2f8efbbeedbc8073373ccbe9b`、平均
`0.7159250827723578`、最小`0.688714710818511`である。

この後続判断は`oop70-manual-acceptance-override-v001.json`へcreate-only保存し、元の自動acceptance、
candidate decision、training/evaluation terminal、140-case結果、embeddingへhash bindingした。最新の
`operator-disposition-v002.json`は`AUTOMATED_ACCEPTED=8`、`MANUAL_ACCEPTED=4`、`rejected=0`で、
SHA-256は`e014e64ad311ca4fc994eb245c272d13e3c65099af4db388cc394022d7299c34`。別process auditも`PASS`し、
配備、現行v3、voice bankに変更はない。
