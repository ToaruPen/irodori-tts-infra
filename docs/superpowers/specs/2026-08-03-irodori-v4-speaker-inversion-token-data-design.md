# Irodori-TTS v4 Speaker Inversion token・データ再探索設計

## 目的

Irodori-TTS v4-Small の Speaker Inversion について、既存の16-token探索で残った
文章・style依存の話者同一性低下を解消する。固定済み2,223件のclean manifestを基準に、
token数、学習データ構成、初期化を一度に一要因ずつ変更し、既存の140-case評価契約で
再現可能な候補を探索する。

成功条件は、同一checkpointの16 hard-gate caseすべてでnormalized ECAPA speaker
similarityが`0.75`以上となり、CER、音声健全性、style contrastを含む既存の他のhard
gateも通過することである。合格候補が得られても自動配備は行わない。

診断Run IDは`v4-si-token-data-diagnostic-v001`とする。

## 固定する基準

- base model: `Aratako/Irodori-TTS-v4-Small`
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- upstream commit: `8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71`
- base model SHA-256:
  `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- tokenizerとそのhash
- 基準clean manifest: 2,223 rows、SHA-256
  `6fd6f8755a74130bec0ca985da45ef05841fcf047e788062888cc64d0a5f89dd`
- 25本の参照音声とECAPA centroid
- 7 text IDs、2 synthesis seeds、2 styles
- 既存evaluatorと全閾値
- effective batch size 16、AdamW、weight decay 0、bf16、TF32
- checkpoint interval 250、最終評価checkpoint 1000/1500/2000/2500/3000

## 観測事実

### 既存16-token探索

既存baselineはlearning rate `0.01`、seed `2`、scratch初期化である。最良のstep 1000は
hard-gate 16 case中5件がspeaker similarity `0.75`未満で、平均は
`0.7490631472749244`、最小は`0.6781481281387317`だった。

追加した3候補も140/140 caseの評価を完了したが、すべて不合格だった。

| 候補 | 最良step | 不合格数 | hard平均 | hard最小 |
|---|---:|---:|---:|---:|
| `lr0035_seed2` | 1500 | 7 | 0.7528069437 | 0.6973145136 |
| `lr0035_seed7` | 1000 | 6 | 0.7488060811 | 0.6716769832 |
| `lr001_seed2` | 2500 | 9 | 0.7406961598 | 0.6539117553 |

全候補でCERは`0.0`だった。learning rate低下とseed変更は失敗caseを移動させたが、
16/16通過には至らなかった。単純なstep延長も、baselineでstep 1000より後に失敗数が
増えるため採用しない。

### 失敗caseの偏り

baselineの5 checkpointを合わせたspeaker hard-gate失敗33件は、`control` 1件に対し、
`sentence_chinko` 11件、`sentence_manko` 7件、`sentence_unko` 14件だった。style別では
`neutral` 26件、`calm` 7件である。追加候補でも失敗は文章・style・seed間を移動した。

この分布は、特定の参照音声だけが判定を壊している場合よりも、生成条件に応じて
Speaker Inversion表現から取り出される声質が揺れている場合と整合する。

### token構造

v4実装ではSpeaker Inversionのtoken列をreference speaker encoderの代わりに直接使用し、
base modelは凍結される。token数は公式設定項目であり、学習対象parameterは
`token数 × 768`である。異なるtoken数のinit embeddingは公式loaderが受け付けない。

4つの最良checkpointを読み取り専用で解析したところ、16 tokenのcentered effective
rankは約`8.77`〜`12.49`、token間cosine平均は約`0.076`〜`0.092`だった。したがって
既存tokenは単純な重複ではなく、平均圧縮した派生embeddingをscratch学習の代替には
しない。

baseline step 1000ではtoken L2 normが`3.13`〜`24.90`、第1主成分が分散の約34.8%を
占めた。低learning-rate候補ではnormと第1主成分への集中が下がったが、hard-gate結果は
改善しなかった。このためnormの暴走だけを主因とはみなさない。

### 参照centroid

同じ25参照音声に対する既存auditでは、robust rule上の外れ値が1件あった。一方、
leave-one-out centroidのfull centroidに対するcosineは`0.9990`以上で、centroid driftは
最大約`0.0010`だった。生成16 caseのleave-one-out similarity差は最大約`0.0125`である。

閾値直近caseの判定は変わり得るが、`0.67`台の失敗を説明できない。既存比較との
互換性を守るため、参照集合と閾値は探索中に変更しない。

## 仮説順位

1. **token数と条件依存性**: 16-token列の容量がv4-Small上でidentityと演技を分離するのに
   適していない。8 tokenなら自由度を制限してidentityを安定化できる可能性があり、
   32 tokenならidentityとstyleを別tokenへ分離できる可能性がある。
2. **学習音声分布**: 2,223件は同一人物でも演技・音響分布が広く、固定tokenが評価参照の
   canonical voiceより広い混合分布を学習している。
3. **初期値依存**: scratch seedによる局所解の差が残る。token数またはデータ構成を決めた
   後で検証する。
4. **参照centroid不安定性**: 影響は小さく、固定評価契約を変える根拠にはならない。

## 検討した方式

### 既存16 tokenのseed探索を追加する

変更が小さい一方、seed 2と7で失敗caseが移動しただけで、改善方向が得られていない。
最初の追加実験には採用しない。

### 学習済み16 tokenを平均・複製して直接評価する

GPU時間は短いが、平均は学習済みtokenの分化を破壊し、複製はattention上ほぼ同値に
なり得る。scratchでtoken数を変えた学習の代理にならないため、昇格判断には使わない。

### token数をbracketし、その後にデータと初期化を検証する

8 tokenと32 tokenを同じmanifest、learning rate、seedでscratch学習すれば、既存16 tokenを
中央点として容量の方向を比較できる。不合格時だけECAPAで学習データ分布を監査し、
選択規則を固定した派生manifestを作る。最後に同一shapeだけでwarm-startを試す。

本設計はこの方式を採用する。

## 実験順序

### Stage 0: 読み取り専用診断

1. 既存checkpointのtoken norm、pairwise cosine、effective rankを記録する。
2. 基準clean manifestの2,223音声を、評価と同じ固定ECAPA modelでembedding化する。
3. 25参照音声の固定centroidに対する各学習音声のsimilarity分布、duration、text長との関係、
   下位tailのsource IDをcreate-only reportへ保存する。
4. 入力音声、manifest、参照音声、モデル、scriptのpathとSHA-256を証跡へ束縛する。

Stage 0は評価閾値や学習データを変更しない。

### Stage 1: token数bracket

候補をGPU上で直列実行する。

| 順序 | 候補 | token数 | 初期化 | LR | seed | manifest |
|---:|---|---:|---|---:|---:|---|
| 0 | 既存baseline | 16 | scratch | 0.01 | 2 | 基準2,223件 |
| 1 | `tokens8_scratch` | 8 | scratch | 0.01 | 2 | 基準2,223件 |
| 2 | `tokens32_scratch` | 32 | scratch | 0.01 | 2 | 基準2,223件 |

各候補を3,000 stepsまで学習し、5 checkpoint × 28 caseの140-case評価を行う。
`tokens8_scratch`が成功条件を満たせばStage 1を終了し、32 tokenは起動しない。不合格なら
32 tokenへ進む。平均値だけでなく、hard-gate失敗数、最小値、文章・style別分布で方向を
判断する。

### Stage 2: データ構成

Stage 1で合格候補がない場合だけ実施する。Stage 0 reportの分布を見てから閾値を選ぶが、
学習前に次を固定し、結果を見て変更しない。

- ECAPA similarityの採用quantile
- 最小・最大duration
- 必要ならtext長の範囲
- retained / excluded source ID一覧
- 派生manifestのrow countとSHA-256

最初の派生manifestは、基準manifestの順序を維持し、参照centroidに対するsimilarity下位25%を
除いた`central_q25`を第一候補とする。分布に明確な多峰性または破損clusterがある場合のみ、
その境界を根拠付きで優先する。token数はStage 1でhard-gate失敗数が最少のものを使い、
同数なら16 tokenを使う。その他の学習条件は固定する。

### Stage 3: 初期化

Stage 2までで合格候補がなく、最良候補が閾値近傍まで改善した場合だけ実施する。同じtoken数の
最良checkpointをinit embeddingとして使い、学習率を下げた短いrefinementを新規rootで行う。
異なるtoken数間の変形warm-startは行わない。

最初のrefinement条件はlearning rate `0.001`、seed `2`、最大1,500 steps、保存間隔250とする。
元checkpointとinit embeddingのSHA-256を束縛し、step番号は新規run内のstepとして扱う。

### Stage 4: 証拠に基づくrefinement継続

Stage 3の最終checkpointがbaselineに対してhard-gate失敗数、平均、最小のすべてを改善し、
他のhard gate失敗が0件だった場合に限り、同じtoken数・`central_q25`・learning rate
`0.001`・seed `2`で、さらに1回だけ1,500 stepsのwarm-start継続を行う。init embeddingは
Stage 3 step 1500そのものとし、SHA-256と`F32[16,768]`を束縛する。

新runのstep 250/500/750/1000/1500を140-caseで評価する。合格しない場合、Stage 3より
失敗数または最小similarityが改善しなければ、同条件の反復は停止する。

### Stage 5: 未評価checkpoint gap診断

Stage 4までで合格せず、保存間隔250に対して140-case対象外だったstep 1250が残る場合、
追加学習の前に保存済みexact checkpointだけで140-case診断を行う。5つの診断slotへbaseline、
Stage 3、Stage 4のcheckpointを割り当て、slot番号と元run/step/path/SHA-256の対応を
create-only evidenceへ保存する。embeddingの平均化、補間、外挿、reshapeは行わない。

この診断で合格checkpointがなければ、同じ学習条件の反復には戻らず、次のデータ構成仮説へ
進む。

### Stage 6: `central_q50` identity refinement

Stage 5で保存済みcheckpointに合格候補がなく、`central_q25` warm-startがbaselineの
hard-gate失敗数を5件から3件へ改善した場合、参照identityをさらに絞る次のデータ構成仮説を
1回だけ検証する。Stage 0で固定した同一ECAPA結果のmedian
`0.8050478070701037`をinclusive cutoffとし、2,223件中1,112件を元manifest順で保持する
`central_q50` manifestを新規作成する。評価結果を見てcutoff、retained ID、row countを変更しない。

学習はStage 4 step 1500のexact `F32[16,768]` embeddingをwarm-startとして使う。token数16、
learning rate `0.001`、seed `2`、最大1,500 steps、保存間隔250は維持し、変更点を
manifest、output root、init embeddingの束縛に限定する。新runのstep
250/500/750/1000/1500を既存140-case契約で評価する。

合格checkpointがなければ同じ`central_q50` refinementを反復しない。hard-gate失敗数または
最小similarityがStage 4を改善した場合だけ、未評価checkpointの追加診断を検討する。それ以外は
duration依存を除いた参照identity選別など、別の事前固定データ仮説へ進む。

### Stage 7: `central_q50` exact checkpoint gap診断

Stage 6の評価対象checkpointがStage 4よりhard-gate失敗数または最小similarityを改善し、保存済み
step 1250が未評価の場合、追加学習の前にexact checkpointだけで1回の140-case診断を行う。
5つの診断slotはStage 4 step 1500、Stage 6 step 250、500、1250、1500へ固定し、各slotと
元run/step/path/SHA-256/config/terminalの対応をcreate-only evidenceに保存する。平均化、補間、
外挿、reshapeは行わない。

合格checkpointがなければ`central_q50`には戻らず、duration依存を除いた参照identity選別を次の
データ構成仮説とする。

### Stage 8: duration-residual identity refinement

Stage 7までで合格せず、absolute ECAPA cutoffによる`central_q50`が参照durationを大きく偏らせた
場合、同じStage 0 ECAPA結果からduration効果を除いた残差で1回だけ参照を選別する。
全2,223件について

`speaker_similarity = 0.6624267827077653 + 0.0704180071156121 * log1p(duration_seconds) + residual`

を固定し、residualの25th percentile `-0.021322203746580615`以上の1,667件を元manifest順で
保持する。係数、cutoff、retained ID、row countは学習前に固定し、結果を見て変更しない。
この構成では保持群と除外群の平均durationはそれぞれ約6.494秒と6.368秒であり、
`central_q50`の8.038秒対4.886秒の偏りを避ける。

初期化はStage 6の最良だったexact step 500 `F32[16,768]` checkpointとする。token数16、
learning rate `0.001`、seed `2`、最大1,500 steps、保存間隔250を維持し、step
250/500/750/1000/1500を既存140-case契約で評価する。合格しなければ同じ残差選別を反復しない。

### Stage 9: duration-residual q50 identity refinement

Stage 8が不合格で、その最良checkpointもStage 6の最良checkpointに対してhard-gate失敗数を
減らさず、平均と最小similarityを改善しなかった場合、Stage 8と同じ固定OLS残差をさらに
identity中心へ絞る仮説を1回だけ検証する。全2,223件のresidualのmedian
`0.006422475306645192`以上の1,112件を元manifest順で保持する。係数、cutoff、retained ID、
row countは学習前に固定し、結果を見て変更しない。保持群と除外群の平均durationはそれぞれ
約6.235秒と6.690秒であり、absolute `central_q50`の8.038秒対4.886秒よりduration偏りが小さい。

初期化はStage 6の最良だったexact step 500 `F32[16,768]` checkpointへ戻し、Stage 8の
不合格checkpointを連鎖させない。token数16、learning rate `0.001`、seed `2`、最大1,500
steps、保存間隔250を維持し、step 250/500/750/1000/1500を既存140-case契約で評価する。
合格しなければ同じresidual q50選別を反復せず、参照構成軸を停止して最適化率または
初期化近傍を別の事前固定仮説として検討する。

### Stage 10: `central_q50` low-rate local refinement

Stage 9までの追加データ構成がStage 6最良のhard-gate失敗2件を減らさなかった場合、
参照構成軸を停止し、Stage 6最良点の近傍を1回だけ低learning rateで探索する。
Stage 6 exact step 500 `F32[16,768]`を初期値とし、そのcheckpointを生んだabsolute
`central_q50` 1,112件manifestへ戻す。token数16、seed `2`、最大1,500 steps、保存間隔250、
その他のoptimizer条件は維持し、learning rateだけを`0.00035`へ下げる。

`0.00035`は既存v3 quality-searchで使った局所refinement率であり、`0.001` runのstep 500付近を
より細かく探索するために学習前に固定する。step 250/500/750/1000/1500を既存140-case契約で
評価し、合格しなければ同じlow-rate continuationを反復しない。改善した未評価step 1250が
残る場合だけexact checkpoint gap診断を検討する。

### Stage 11: low-rate exact checkpoint gap diagnostic

Stage 10のstep 250がStage 6最良と同じhard-gate失敗2件を残しながら、平均similarityを
`0.7665460368446404`から`0.7687227570720304`へ、最小similarityを
`0.7083651962338371`から`0.7218517799056324`へ改善し、未評価のexact step 1250が残った場合、
新たな学習の前に1回だけcheckpoint gapを診断する。

既存の5つの評価slotへ、Stage 6 step 500、Stage 10 step 250、step 500、step 1250、
step 1500をこの順で割り当てる。各slotを元run、元step、絶対path、SHA-256、config、setup、
terminal、decisionへ束縛し、既存140-case契約で評価する。embeddingの平均化、補間、外挿、
reshapeその他の派生処理は行わない。合格checkpointがなければ同じ低learning-rate仮説を反復しない。

### Stage 12: Pareto initialization refinement

Stage 11が不合格で、同一の2ケース`sentence_manko/seed1234/calm`と
`sentence_unko/seed1234/neutral`がStage 6からStage 10まで残る場合、既評価checkpoint全体から
この2ケースの低い方を最大化するexact checkpointを初期化候補として1回だけ検証する。
固定済み評価証跡ではStage 3 q25 refinement step 500がそれぞれ
`0.736087317842197`、`0.7399610544304428`であり、2ケースのminimum `0.736087317842197`は
既評価checkpoint中で最大である。一方で同checkpointは全体で6件未達なので、そのまま候補にはしない。

Stage 3 exact step 500 `F32[16,768]`を初期値とし、Stage 6と同じabsolute `central_q50`
1,112件manifestを使って、learning rate `0.00035`、token数16、seed `2`、最大1,500 steps、
保存間隔250で局所refinementする。これはStage 10の不合格checkpointを連鎖する試行ではなく、
persistent 2ケースに対する別のexact initialization basinを検証する試行である。
step 250/500/750/1000/1500を既存140-case契約で評価し、不合格なら同じ初期化仮説を反復しない。

### Stage 13: absolute central q75 identity-core refinement

Stage 12が不合格で、既存reference-centroid auditが25参照間のpairwise similarity中央値
`0.6708094945427595`とrobust outlier 1件を示し、学習音声のabsolute centroid similarityを
q25からq50へ厳しくした既存試行が最良failure countを3件から2件へ改善している場合、
未検証のabsolute q75 identity coreを1回だけ評価する。

固定済みStage 0 ECAPA auditの75th percentile `0.8338747704317409`以上の556件を元manifest順で
保持し、latent pathをcreate-only rootへrebaseする。初期化は全試行中の最良であるStage 6 exact
step 500 `F32[16,768]`とし、learning rate `0.00035`、token数16、seed `2`、最大1,500 steps、
保存間隔250を使う。step 250/500/750/1000/1500を既存140-case契約で評価する。
これはduration補正ではなく、評価reference centroidに最も近い学習identity coreを検証する仮説であり、
不合格なら同じabsolute cutoff系列をさらに細分化または反復しない。

### Stage 14: `central_q50` seed-order sensitivity refinement

Stage 13が不合格で、absolute q50のStage 10が依然として全候補中の最良値
である場合、データ構成と初期化を固定したまま学習順序・乱数seedだけを1回変更する。
事前証拠は、同じ旧full-data、learning rate `0.0035`、3,000-step契約の比較で、
seed `2`の最良checkpointが類似度未達7件、seed `7`が6件と異なる局所解へ進んだこととする。

Stage 6 exact step 500 `F32[16,768]`を初期値とし、absolute `central_q50` 1,112件、
learning rate `0.00035`、token数16、最大1,500 steps、保存間隔250を維持し、seedだけを
`2`から`7`へ変える。step 250/500/750/1000/1500を既存140-case契約で評価する。
不合格なら同じq50・初期値・learning rateで別seedを連続探索せず、seed軸は停止する。

### Stage 15: seed7 exact checkpoint gap diagnostic

Stage 14が最良でも未達2件のまま不合格だが、step 500の平均および最小similarityと
step 750の最小similarityがStage 10の現最良値を一部上回り、保存済みstep 1250が
未評価の場合、追加学習前にそのexact checkpointを1回だけ140-case診断する。

5つの診断slotにStage 14のstep 250、500、750、1250、1500をそのまま割り当て、
元step/path/SHA-256/config/setup/terminal/decisionをcreate-only evidenceへ束縛する。
step 250、500、750、1500は決定性再現のcontrolとし、新規情報はstep 1250のみとする。
embeddingの平均化、補間、外挿、reshape、再学習は行わない。不合格ならseed軸には戻らない。

### Stage 16: 24-reference robust-centroid data audit

Stage 15までで合格せず、固定済みreference-centroid auditが25参照中1件
`oop53_aibeya_sp_f7269f5ffc:00000594`を固定ルール`median - 3*MAD`でoutlierと
判定している場合、学習データ選別用のcentroidだけを24 inlier参照から再計算する。
評価側の25参照、類似度定義、閾値、caseは変更しない。

固定2,223件を同じECAPA model/revisionで再embedding化し、24-reference centroidに対する
similarity、quantile、source orderをcreate-only auditに保存する。そのmedian以上の1,112件を
`robust_central_q50`と事前定義し、既存absolute `central_q50`とのJaccard similarityと
symmetric differenceを計算する。Jaccardが`0.98`以上なら選別差が小さすぎると判定し、
この仮説では学習しない。

### Stage 17: robust-centroid q50 local refinement

Stage 16のJaccard similarityが`0.98`未満で、入力hash、24参照、除外outlier、保持IDが
完全に束縛された場合だけ、`robust_central_q50` 1,112件で1回再学習する。
初期値は全試行の最良failure count 2かつ平均・最小の総合値が最良なStage 10
exact step 250 `F32[16,768]`とする。token数16、learning rate `0.00035`、seed `2`、
最大1,500 steps、保存間隔250で学習し、step 250/500/750/1000/1500を既存140-case
契約で評価する。不合格なら同じrobust-centroid選別を反復しない。

最初の実行がモデル・データ・optimizer以外の運用障害で中断した場合は、そのFAIL terminal、
config、setup、logのSHA-256と障害種別を固定し、学習条件を変えずに新しいcreate-only rootで
1回だけ再実行できる。2026-08-04の初回実行はstep 1427で`progress-evidence.json`の原子的置換が
一時的なWindows file lockに遭遇し、再試行のないsupervisorが所有学習processを停止した。
復旧runでは原子的置換に有限回のPermissionError retryだけを追加し、初回runのcheckpointを
初期値や評価候補には使わない。この復旧はrobust-centroid仮説の追加探索として数えない。

### Stage 18: full-clean generalization recovery

Stage 17の最良step 250が未達2件、平均`0.7706521148577734`、最小
`0.7215834480433574`で、Stage 10から平均だけ改善した一方、同じ
`sentence_manko/seed1234/calm`と`sentence_unko/seed1234/neutral`を残した場合、
robust-centroid系列を停止する。78件の入れ替えでpersistent failureが実質的に動かなかったため、
参照centroid cutoffの細分化は行わない。

次の独立したデータ仮説として、identity中央50%へ限定した局所解から固定2,223件clean manifestへ
戻し、発話多様性を再導入する。初期値は不合格なStage 17 checkpointを連鎖させず、最良failure
count 2と最良minimumを持つStage 10 exact step 250 `F32[16,768]`へ戻す。token数16、learning
rate `0.00035`、seed `2`、最大1,500 steps、保存間隔250を固定し、step
250/500/750/1000/1500を既存140-case契約で評価する。これはfull-clean low-rate recoveryを
1回だけ検証するもので、不合格なら同じmanifest・初期値・learning-rate条件を反復しない。

### Stage 19: official microbatch geometry

Stage 18が不合格で、同じ2つのpersistent failureが残る場合、データ選別系列を停止する。
固定upstream commitの公式Speaker Inversion設定
`configs/train_v4_small_speaker_inversion.yaml`（SHA-256
`d2ad1e3d345b34f1b2af4f137f6492f68d541d9366f8079db943a950d9e81964`）は
`batch_size=16`、`gradient_accumulation_steps=1`、`gradient_checkpointing=true`を使用する。
一方、これまでの試行は実効batch sizeこそ16だが、`batch_size=4`、
`gradient_accumulation_steps=4`、`gradient_checkpointing=false`だった。upstreamの`train.py`は
microbatchごとに`sample_stratified_logit_normal_t(batch_size=bsz, ...)`を呼ぶため、両者は同じ
最適化軌道ではない。

この未検証軸だけを分離し、固定`central_q50` 1,112件、Stage 10 exact step 250
`F32[16,768]`、token数16、learning rate `0.00035`、seed `2`、最大1,500 steps、保存間隔250を
維持して、公式の`16/1/gradient-checkpointing=true`で1回だけ再学習する。step
250/500/750/1000/1500を既存140-case契約で評価する。VRAM不足、学習失敗、または評価不合格なら
batch sizeを段階探索せず、このgeometry軸を停止する。seed `0`への変更を同時に行わず、原因を
microbatch geometryに限定する。

2026-08-04の初回v001起動は学習processを生成する前に、共通supervisorの旧geometry固定契約
（batch 4 / accumulation 4 / checkpointing false）でfail-closed停止した。config、setup、
preflight、launch、terminalのhashとtraining未開始を固定し、v002では共通契約検査を迂回せず、
同じ項目集合を公式16/1/trueの固定値で厳密比較する関数へ差し替える。入力、初期値、seed、
optimizer、予定stepsは変更せず、この復旧を追加のgeometry探索として数えない。

Stage 19 v002は全checkpointで不合格となり、最良failure countは3、最良minimumは
`0.6877897040607848`でStage 10の2件・`0.7218517799056324`を下回った。other hard gate failureは
0だったため、公式microbatch geometryは声質一致を改善しないと判定し、この軸を停止する。

### Stage 20: ultra-low-rate local continuation

Stage 10の`0.00035` continuationは最初の評価点step 250が全trajectoryの最良failure count 2かつ
最良minimumを持ち、それ以降は改善しなかった。最良領域を保存間隔より前に通過した可能性を
独立に検証するため、exact Stage 10 step 250へ戻し、学習率だけを`0.0001`へ下げる。

固定`central_q50` 1,112件、token数16、batch size 4、gradient accumulation 4、gradient
checkpointing false、seed 2、最大1,500 steps、保存間隔250を維持し、step
250/500/750/1000/1500を既存140-case契約で評価する。不合格なら同じ初期値・manifestでさらに
学習率を列挙せず、このultra-low-rate軸を停止する。

### Stage 21: ultra-low-rate exact checkpoint gap

Stage 20が全評価checkpointで同じ2件を残して不合格でも、step 500がStage 10 global bestに対して
平均similarityとminimum similarityを両方改善し、保存済みstep 1250が未評価なら、追加学習前に
そのexact checkpointを1回だけ140-case診断する。

診断slot 250/500/750/1000/1500へStage 20 exact step 250/500/750/1250/1500を割り当てる。
slot 1000だけが新規情報で、他4 slotは既評価checkpointの決定性controlとする。source
step/path/SHA-256/config/setup/terminal/decisionをcreate-only evidenceへ束縛し、embeddingの平均化、
補間、外挿、reshape、再学習は行わない。不合格ならultra-low-rate系列へ戻らない。

### Stage 22: aligned same-token convex-direction diagnostic

Stage 21が不合格で、Stage 10 step 250は全体failure count 2のglobal bestを保つ一方、Stage 12
step 1500がpersistent 2件をそれぞれ約`0.7443`、`0.7446`まで改善している場合、両者の間に
全体性能とpersistent case改善を両立する方向があるかを1回だけ診断する。

両embeddingは同じ`F32[16,768]`で、token-wise cosine監査により16/16で同じindexがrow最大、
対角cosine平均`0.9742916822433472`、global cosine`0.9873813390731812`であることを事前条件とする。
この対応が崩れた場合は補間しない。

Stage 10をalpha 0、Stage 12をalpha 1として、`0, 0.25, 0.5, 0.75, 1`の5点を事前固定し、
CPU NumPy float32で`(1-alpha)*A + alpha*B`を要素単位に計算する。両端はexact parent control、
内側3点だけをderived diagnosticとする。既存140-case契約で一度だけ評価し、derived embeddingを
最終checkpointや配備候補にはしない。内側点がglobal bestのfailure countまたはpersistent 2件を
改善する場合だけ、その1点を後続再訓練の初期値候補とする。改善がなければこの方向を停止し、
alphaの細分化や外挿は行わない。

### Stage 23: Pareto-basin full-clean ultra-low-rate recovery

Stage 22の内側3点がいずれもglobal bestのfailure count 2を改善しない場合、convex方向を停止する。
一方、Stage 12 exact step 1500はpersistent 2件を約`0.7443`、`0.7446`まで上げながら、別の
3件を増やしている。Stage 18でfull-clean generalization、Stage 20でultra-low-rate local
continuationを個別に検証済みであるため、このPareto basinを保ちながら全体性能を戻せるかを
1回だけ組合せ検証する。

初期値はStage 12 exact step 1500 `F32[16,768]`、dataはimmutable 2,223-row clean manifest、
token数16、learning rate `0.0001`、batch size 4、gradient accumulation 4、gradient
checkpointing false、seed 2、最大1,500 steps、保存間隔250とする。step
250/500/750/1000/1500を既存140-case契約で評価する。不合格なら同じbasin・full-clean・low-rate
条件を反復せず、このrecovery系列を停止する。

### Stage 24: token-weighted RF-loss sentence-identity refinement

Stage 23が不合格の場合、既存評価rootに保存された全exact checkpointを横断監査する。監査で
`sentence_unko / seed1234 / neutral`が115 evaluated checkpoint slotの全てで0.75未満、最高でも
`0.744594789105378`であり、failure count 2に到達した24 slotが全て同一の2 sentence caseだけを
残していることを確認できた場合だけ、
発話長に依存しない`utterance_mean` lossを停止して、upstream実装済みの`echo` lossを1回試す。
`echo`は有効latent token単位でRF誤差を正規化するため、長い学習発話の話者情報を相対的に
強く学習するという独立した最適化仮説であり、評価文のoversamplingや推論parameter変更ではない。

初期値はStage 20 exact step 500、manifestは固定1,112件`central_q50`、token数16、
LR0.0001、batch4/accumulation4/checkpointing-off、seed2、max1,500、save250を維持する。
source configから変更してよいのは`train.rf_loss_mode`、`train.output_dir`、
`train.speaker_inversion_init_embedding`だけとする。steps250/500/750/1000/1500を変更なしの
140-case契約で評価する。不合格ならloss-normalization軸を停止し、`echo`のrate・seed・data variantを
列挙しない。

### Stage 25: text-condition dropout identity disentanglement

Stage 24が同じpersistent 2件を残して不合格になった場合、loss-normalization軸を停止する。既存の
全候補は`text_condition_dropout: 0.0`であり、同じspeaker embeddingが文・style・seedの組合せで
通過と失敗に分かれている。upstreamはduration-predictor pathでもsample単位のtext-condition
dropoutを実装しているため、speaker identityを本文条件から分離する独立仮説として0.1を1回試す。

初期値はStage 20 exact step 500、manifestは固定1,112件`central_q50`、token数16、
LR0.0001、`rf_loss_mode: utterance_mean`、batch4/accumulation4/checkpointing-off、seed2、
max1,500、save250を維持する。変更してよいのは`train.text_condition_dropout`、
`train.output_dir`、`train.speaker_inversion_init_embedding`だけで、caption/speaker/duration dropoutは
0.0のままとする。既存140-case契約でsteps250/500/750/1000/1500を評価し、不合格なら確率変更や
caption dropoutとの組合せを列挙せず、この軸を停止する。

### Stage 26: cosine-decay drift stabilization

Stage 25が不合格ならtext-dropout軸を停止する。Stage 10、20、24、25はいずれも早期checkpointが
最良で、定率更新を続けてもfailure countが減らず、後半でminimumまたはmeanが悪化している。
upstreamの`cosine` schedulerはwarmup 0、max1,500 steps全体でbase LRから暗黙defaultの
`min_lr_scale: 0.1`へ単調減衰するため、初期の局所更新を残しつつ後半のembedding driftを抑える
独立仮説として1回だけ検証する。

初期値はStage 20 exact step 500、manifestは固定1,112件`central_q50`、token数16、base
LR0.0001、`rf_loss_mode: utterance_mean`、全condition dropout 0.0、batch4/accumulation4/
checkpointing-off、seed2、max1,500、save250を維持する。source configから変更してよいのは
`train.lr_scheduler`、`train.output_dir`、`train.speaker_inversion_init_embedding`だけとする。
steps250/500/750/1000/1500を変更なしの140-case契約で評価し、不合格ならscheduler種類・
warmup・minimum倍率を列挙せず、この軸を停止する。

## 実行契約

- 全runはversioned create-only rootに保存し、既存成果物を上書きしない。
- preflightでupstream commit、model/tokenizer/manifest hash、config、token数、初期化、GPU空き、
  競合process不在を検証する。
- detached supervisorは所有processだけを監視し、終了code、最終step、checkpoint hash、loss、
  GPU peak、GPU解放をterminal evidenceへ保存する。
- OOM、traceback、hash不一致、予定step未達、所有process残留をFAILとする。
- FAIL rootは再利用せず、新しいversioned rootを作る。
- 診断・学習・評価script自体のSHA-256を各runへ束縛する。

## 評価と停止規則

各3,000-step候補は既存の140-case評価を完全実行する。候補は次のすべてを満たす場合だけ
`ELIGIBLE`とする。

1. 140/140 caseの生成・解析・metric計算が成功する。
2. 同一checkpointの16 speaker hard-gate caseすべてが`0.75`以上である。
3. 既存のCER、duration、RMS、silence、clipping hard gateをすべて通る。
4. style contrastとstyle similarity dropが既存閾値内である。
5. evaluatorがcheckpointを`ELIGIBLE`と判定する。

成功した時点で後続の探索は停止する。成功しない候補も削除せず、比較証跡として保持する。

## 安全境界

- 現行v3 serviceを停止・置換しない。
- 配備済みvoice bankを変更しない。
- v4-Small base modelを更新・再訓練しない。
- immutable baselineと既存評価成果物を変更しない。
- 評価参照、閾値、text、style、seedを候補ごとに変更しない。
- 合格候補を自動配備しない。
- generated audio、model weight、checkpoint、datasetをrepositoryへ追加しない。

## 完了条件

次をすべて満たしたときにゴールを完了とする。

- 16 speaker hard-gate caseすべてが`0.75`以上のcheckpointがある。
- そのcheckpointが既存の他のhard gateも通過している。
- 140-case評価、学習、入力、model、scriptの再現可能なcreate-only証跡がある。
- 現行v3 service、voice bank、immutable baselineが開始前と同一である。
- 聴覚確認用packetがMac側のruntime-asset directoryにコピーされている。
