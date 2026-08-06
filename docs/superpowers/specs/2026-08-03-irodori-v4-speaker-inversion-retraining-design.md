# Irodori-TTS v4 Speaker Inversion 再訓練設計

## 目的

`Aratako/Irodori-TTS-v4-Small` 用に作成した既存 Speaker Inversion 埋め込みの
話者同一性を改善する。v4-Small 本体、学習データ、評価ケース、品質閾値を固定し、
学習率と乱数 seed の効果だけを段階的に検証する。

再訓練の成功条件は、既存 evaluator が話者同一性を判定する 16 hard-gate case の
すべてで、正規化 ECAPA cosine similarity が `0.75` 以上になることである。既存の
明瞭性、音声健全性、style contrast の gate も同時に維持する。

## 現状

既存 v4 pilot は次の条件で完走した。

- base model: `Aratako/Irodori-TTS-v4-Small`
- upstream commit: `8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71`
- clean manifest: 2,223 rows、SHA-256
  `6fd6f8755a74130bec0ca985da45ef05841fcf047e788062888cc64d0a5f89dd`
- Speaker Inversion: 16 tokens、12,288 trainable parameters
- optimizer: AdamW
- learning rate: `0.01`
- seed: `2`
- effective batch size: `16`
- max steps: `3000`

step 3000 は平均 speaker similarity `0.76235022`、CER `0.0` だったが、16 hard-gate
case のうち 6 case が `0.75` 未満だった。5 checkpoint の結果は単調ではなく、
pass count は step 1000 が 23/28、step 1500 が 19/28、step 2000 が 22/28、
step 2500 が 21/28、step 3000 が 22/28 だった。このため、step 数の単純な延長は
採用しない。

## 検討した方式

### 現行 learning rate のまま seed だけを変える

乱数による変動は測定できるが、既存結果で見られた checkpoint 間の振れを抑える
方向には働かない。比較候補ではあるが、最初の再訓練には採用しない。

### learning rate を下げて段階探索する

モデル、manifest、token 数を固定し、learning rate と seed だけを変える。既存条件との
因果関係を保ちつつ、埋め込みの更新幅と初期値依存を評価できる。本設計ではこの方式を
採用する。

### token 数または学習データを変更する

表現容量やデータ分布を変えられる一方、最適化条件との効果分離ができなくなる。
第1段階の候補がすべて失敗した場合だけ、別設計として検討する。

## 再訓練候補

候補は GPU 上で直列実行する。各候補は新しい create-only root を使用し、既存の
checkpoint や評価成果物を上書きしない。

| 順序 | 候補 | 初期化 | learning rate | seed | max steps |
|---:|---|---|---:|---:|---:|
| 0 | 既存 baseline | scratch | 0.01 | 2 | 3000 |
| 1 | `lr0035_seed2` | scratch | 0.0035 | 2 | 3000 |
| 2 | `lr0035_seed7` | scratch | 0.0035 | 7 | 3000 |
| 3 | `lr001_seed2` | scratch | 0.001 | 2 | 3000 |

候補 1 を学習・評価し、合格すれば候補 2 と 3 は起動しない。候補 1 が不合格なら
候補 2、候補 2 も不合格なら候補 3 へ進む。これにより、合格後の不要な GPU 使用を
避けながら、learning rate と seed の両方を段階的に検証する。

次の条件は全候補で固定する。

- v4 model revision と model/tokenizer SHA-256
- upstream checkout と仮想環境
- clean manifest とその SHA-256
- Speaker Inversion 16 tokens
- batch size 4、gradient accumulation 4
- gradient checkpointing disabled
- bf16、TF32 enabled
- AdamW、weight decay 0
- caption、speaker、text condition dropout 0
- checkpoint interval 250 steps
- evaluation checkpoint: 1000、1500、2000、2500、3000

## 実行と証跡

各候補は preflight 後に detached supervisor から起動する。supervisor は所有する学習
process だけを監視し、終了時に exit code、checkpoint 一覧と SHA-256、最終 loss、
最新 step、GPU peak、GPU 解放状態を terminal evidence に保存する。

次のいずれかが発生した候補は失敗として終了し、同じ root で再開または上書きしない。

- preflight failure
- non-zero exit code
- OOM または traceback
- step 3000 未到達
- final checkpoint と step 3000 checkpoint の不一致
- checkpoint、設定、manifest、base model の hash 不一致
- 学習終了後も所有 process が残る

失敗時は新しい versioned root を作る。既存の失敗証跡は削除しない。

## 評価

各候補は既存比較と同じ 140 case を生成する。

- 5 checkpoints
- 7 text IDs
- 2 synthesis seeds
- 2 styles (`neutral`、`calm`)

生成、音声解析、ECAPA speaker similarity、Whisper CER、deterministic evaluator、
review packet 作成を既存と同じ順序で実行する。評価モデル、revision、参照 centroid、
閾値は変更しない。

候補の自動合格条件は次のすべてである。

1. 140/140 case の生成と metric 計算が成功する。
2. 16 hard-gate case の speaker similarity がすべて `0.75` 以上になる。
3. CER、duration、RMS、silence、clipping の既存 hard gate をすべて通る。
4. style contrast が既存の下限を通り、style similarity drop が上限内に収まる。
5. evaluator が少なくとも 1 checkpoint を `ELIGIBLE` と判定する。

平均 similarity だけの改善、既存 v4 より少ない不合格件数、loss の低下だけでは合格と
しない。

## 聴覚確認

自動合格した checkpoint がある場合も、配備前に人間が話者同一性と style を確認する。
選択 checkpoint の review packet と、既存 v4 baseline の対応 case を Mac の独立した
runtime-asset directory にコピーする。リポジトリへ WAV を追加しない。

聴覚確認で明確な話者差、音割れ、不自然な抑揚、caption による identity 崩れが確認
された場合は、evaluator の合格だけで配備しない。

## 安全境界

- 現在の v3 service と voice bank を停止、更新、置換しない。
- v4-Small の base model を再訓練しない。
- 既存 v3/v4 checkpoint と比較成果物を変更しない。
- 全候補を versioned create-only directory に保存する。
- 合格しても自動配備しない。
- 候補 3 まで不合格なら探索を停止し、token 数またはデータ構成の変更を別設計にする。

## 完了条件

本作業は、次のいずれかで完了する。

- 候補が自動合格し、聴覚確認用 packet と再現可能な証跡が作成される。
- 3候補すべてが不合格となり、各候補の immutable evidence と比較結果が残る。

いずれの場合も、配備状態と標準 v3 設定が開始前と同一であることを最終確認する。
