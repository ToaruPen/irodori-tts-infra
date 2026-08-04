# Irodori-TTS v4 pragmatic Speaker Inversion acceptance design

## 目的

既存の厳格評価契約とその`REJECTED`証跡は変更せず、運用上十分に近いSpeaker Inversion
checkpointを、外れ値に頑健な追加契約で一意に選定する。選定結果は配備を伴わない
create-only証跡として保存する。

## 背景

固定済みの140-case評価を26回、合計130 checkpoint枠で実行したが、16 speaker hard-gate
caseすべてがnormalized ECAPA similarity `0.75`以上となる候補は得られなかった。一方、
36枠はsimilarity失敗2件かつ他hard-gate失敗0件まで到達している。失敗は主に固定の
`seed=1234`を含む2ケースへ集中し、ユーザーは単一ケース最小値の厳密な0.75到達よりも
実用的な近似を優先する方針へ変更した。

## 検討した方式

1. 全caseのsimilarity閾値を`0.72`へ下げる。
   実装は単純だが、通常caseに対する既存の`0.75`契約まで弱めるため採用しない。
2. `0.75`を主閾値として維持し、最大2件だけ下限`0.72`までの外れ値を許容する。
   平均値と既存hard gateも併用でき、今回採用する。
3. 最小値を使わず平均similarityだけで判定する。
   低い単一caseを高いcaseが隠しうるため採用しない。

## 追加受入契約

checkpointは次のすべてを満たす場合に`PRAGMATICALLY_ELIGIBLE`とする。

1. 固定manifestの140/140 caseで生成、解析、metric計算が成功している。
2. 16 speaker hard-gate caseのうち14件以上が既存主閾値`0.75`以上である。
3. `0.75`未満のcaseは最大2件で、各similarityが`0.72`以上である。
4. 16件のmean speaker similarityが`0.765`以上である。
5. CER、duration、RMS、silence、clipping、tone、style contrast、style similarity dropの
   既存hard gateに失敗していない。
6. incomplete metric、生成失敗、非有限音声がない。

境界値はinclusiveとする。既存のstrict evaluatorによる`ELIGIBLE`候補が存在する場合は、
追加契約の候補より常に優先する。

## 選定規則

追加契約を満たす全checkpointを次の順で決定的に順位付けし、先頭の1件だけを選ぶ。

1. similarity failure countが少ない。
2. minimum speaker similarityが高い。
3. mean speaker similarityが高い。
4. candidate名の辞書順、checkpoint stepの昇順を決定的tie-breakerとする。

style contrastは既存閾値の通過条件として扱い、話者同一性の順位付けには使用しない。

## 証跡と安全境界

- 既存のevaluation results、candidate decision、checkpoint、strict thresholdを変更しない。
- versioned toolとversioned acceptance rootを新規作成し、再利用を拒否する。
- 入力となる全evaluation results、decision、training terminal、選定embeddingをSHA-256で束縛する。
- 出力はpolicy、全候補の判定理由、順位、選定checkpointのpath/hashを含む。
- status名はstrict結果と区別できる`PRAGMATICALLY_ELIGIBLE`を使う。
- 現行v3 service、配備済みvoice bank、v4 base modelを変更しない。
- 自動配備しない。配備は別の明示的なユーザー指示を必要とする。

## エラー処理

入力hashの変化、case数不足、重複case、metric欠落、選定checkpointのhash不一致、既存出力root、
複数の同順位候補を決定不能な状態はfail-closedとする。候補が0件の場合は
`NO_PRAGMATIC_CANDIDATE`をcreate-onlyで記録し、成功扱いにしない。

## 検証

実装前に次の失敗テストを作成する。

- 14件が`0.75`以上、残り2件が`0.72`以上、mean `0.765`以上なら通る。
- `0.75`未満が3件なら落ちる。
- 1件でも`0.72`未満なら落ちる。
- meanが`0.765`未満なら落ちる。
- similarity以外のhard gate失敗があれば落ちる。
- 複数候補からfailure count、minimum、meanの順で一意に選ぶ。
- 既存output rootの再利用を拒否する。

ローカルでRed→Green→Refactor、format、lint、pytest、py_compileを通し、リモートでは同一hashと
py_compileを確認する。その後、固定済み26評価・130枠に対して1回だけ再判定する。

## 実行結果

2026-08-04に固定済み26評価・130 checkpoint枠を再判定し、17枠が追加受入契約を満たした。
strict契約を満たす枠は0件だった。決定規則により、次を一意に選定した。

- candidate: `tokens16_central_q50_echo_lr0001_from_stage20step500_v001`
- checkpoint step: `750`
- strict pass count: `14/16`
- outlier count: `2/16`
- outlier similarity:
  - `sentence_manko / seed 1234 / calm`: `0.7265047083898645`
  - `sentence_unko / seed 1234 / neutral`: `0.7255887333554903`
- minimum speaker similarity: `0.7255887333554903`
- mean speaker similarity: `0.7694414503655422`
- other hard-gate failure count: `0`
- embedding SHA-256: `e858bedb0ad1b94e78673b1f07100d6f0490f5787f106bcdd8d886371ac7cdfb`

strict evaluatorのcandidate decisionは`REJECTED`のまま変更していない。そのSHA-256は
`d580e0cc1b815231da47fdb183a8be2d0b24b7fb051a25f723f3e905174ff924`である。追加判定は
`PRAGMATICALLY_ELIGIBLE`で、create-only decisionのSHA-256は
`a7cc8f76fed323e91d5fa25d652461f883c2c472eebf3bc07161ebb12aa155c1`である。

初版tool rootはcase行とsummaryに重複した`tone_candidate`を二重計上してfail-closedになり、
acceptance outputを作成しなかった。回帰テストを追加して修正した`v002`を新規rootへ配置した。
最終selector SHA-256は`9b68357088672472ca5e6c71afae434b1a45f9f8d36109bf98d5eaf3d5c4a736`、
test SHA-256は`1dfd312061cb779abfefb93f6ce3ce1b7922458db52a4ba255ae252ec8dff88c`で、
ローカル12テスト、format、lint、py_compileとリモートpy_compileを通過した。

選定embeddingの実体hashはdecisionの束縛値と一致し、選定元training terminalは`PASS`だった。
既知の訓練・評価PID 4件はすべて不存在で、`deployment_performed: false`、
`active_voice_bank_unchanged: true`を独立に確認した。現行serviceとvoice bankは変更していない。
