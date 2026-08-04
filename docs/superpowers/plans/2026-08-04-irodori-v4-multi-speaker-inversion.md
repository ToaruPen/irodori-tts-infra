# Irodori-TTS v4 複数話者 Speaker Inversion 実行計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** oop53で成立した実用判定を用い、残り11話者のv4-Small Speaker Inversionを隔離領域で作成・評価・選定する。

**Architecture:** 汎用のcreate-only preparer、訓練supervisor、評価supervisor、直列queueを一時runtime toolとして作り、固定された既存manifestと評価資産へ適用する。全話者を共通scratchで作成し、不合格話者だけ追加refinementする。配備は行わない。

**Tech Stack:** Python 3.11、Irodori-TTS v4-Small、PyTorch CUDA/bf16、SpeechBrain ECAPA-TDNN、Whisper large-v3-turbo、PowerShell 5.1、Windows OpenSSH

---

## Task 1: 12話者資産とGPUを棚卸しする

- [x] **Step 1:** 12話者catalogとclean manifestの存在、row count、SHA-256を確認する。
- [x] **Step 2:** 既存v3選定状況を確認し、一律warm-start不可を確認する。
- [x] **Step 3:** GPU空きと競合訓練process不在を確認する。

## Task 2: 汎用runtime toolをTDDで作る

**Files:**
- Create locally: `/private/tmp/irodori-v4-multispeaker-v001/`
- Create remotely: `v4_compatibility/_tools_multi_speaker_v001/`

- [x] **Step 1:** 11話者inventory、hash、順序を検証するcontract testを書く。
- [x] **Step 2:** 話者別scratch config/setupをexclusive作成するpreparerを実装する。
- [x] **Step 3:** 動的manifest/hashを検証する訓練supervisorを実装する。
- [x] **Step 4:** 動的model ID/reference assetsを扱う評価preparer/supervisorを実装する。
- [x] **Step 5:** 訓練→評価→実用判定を直列再開するqueueを実装する。
- [x] **Step 6:** pytest、Ruff、py_compile、fail-closed contract testを通す。

## Task 3: remote create-only rootを準備する

**Files:**
- Create remotely: `v4_compatibility/v4_speaker_inversion_multi_v001/`

- [x] **Step 1:** toolをimmutable versioned directoryへ転送し、local/remote SHA-256を照合する。
- [x] **Step 2:** upstream/model/tokenizer/11 manifest/11評価assetをpreflightする。
- [x] **Step 3:** queue setup evidenceと11話者job manifestをexclusive作成する。
- [x] **Step 4:** mismatch、既存root、競合processで学習前に停止することを確認する。

## Task 4: Phase 1共通scratchを訓練する

- [x] **Step 1:** detached serial queueを起動する。
- [x] **Step 2:** 各話者の進捗とGPU使用を監視し、運用失敗だけversioned retryへ分離する。
- [x] **Step 3:** 11話者すべてについて3,000-step terminal証跡とcheckpoint setを検証する。
- [x] **Step 4:** 全embeddingが`F32[16,768]`かつfiniteであることを独立検証する。

## Task 5: Phase 2共通評価と実用選定を行う

- [x] **Step 1:** 各話者の5 checkpoint × 28 caseを生成する。
- [x] **Step 2:** 音声健全性、ECAPA、CER、style gateを完走する。
- [x] **Step 3:** 話者ごとにstrict decisionと別のpragmatic decisionをcreate-only保存する。
- [x] **Step 4:** 実用合格話者を決定的に選定し、未達話者を抽出する。

## Task 6: 不合格話者だけrefinementする

- [x] **Step 1:** 各未達話者の最良exact checkpointを固定する。
- [x] **Step 2:** full-clean、LR0.0001、1,500-step局所継続を1回評価する。
- [x] **Step 3:** 必要な話者だけcentral_q50を固定作成し、echo lossを1回評価する。
- [x] **Step 4:** 合格または事前定義した探索終了まで話者別decisionを閉じる。

## Task 7: 独立検証と文書化を完了する

- [x] **Step 1:** 全manifest、config、script、checkpoint、評価、decisionのhashを再計算する。
- [x] **Step 2:** 12話者（oop53を含む）の最終一覧と未達caseを文書化する。
- [x] **Step 3:** service、voice bank、標準v3設定が不変であることを確認する。
- [x] **Step 4:** repositoryの関連テストと既知baseline failureを再確認する。

## 実行結果（2026-08-04）

最終verifierは `PASS`。12話者中8話者を実用合格、4話者を事前定義したbounded refinementの
終了により `REJECTED_EXHAUSTED` とした。合格モデルの配備は行っておらず、現行v3 service、
13話者voice bank、標準v3設定は開始前から不変である。

最終証跡:

- root: `v4_compatibility/v4_speaker_inversion_multi_v001`
- final verification: `final-verification-v001.json`
- verifier: `_tools_multi_speaker_final_verification_v005/verify_multispeaker_final.py`
- verifier SHA-256: `486d7007b6ca40039a6467d10c6868002457c0a725b4f9574d41133e7b99b1ef`
- v4 upstream: `8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71`、tracked clean
- final runtime: training/queue/service process 0、GPU free 11,070 MiB、utilization 0%

### 最終選定一覧

| 話者 | 最終段階 | step | 判定 | hard 16件の平均 / 最小 | similarity未達 / その他失敗 |
|---|---|---:|---|---:|---:|
| `oop53_aibeya_sp_f7269f5ffc` | pilot central-q50 + echo | 750 | pragmatic合格 | 0.769441 / 0.725589 | 2 / 0 |
| `oop176_natsu_no_owari_sp_dcec9a11d3` | central-q50 + echo | 750 | 探索終了・不合格 | 0.798678 / 0.766276 | 0 / 2 |
| `oop52_aibeya_2_sp_5d544fe890` | central-q50 + echo | 750 | 探索終了・不合格 | 0.759852 / 0.730189 | 5 / 0 |
| `oop54_aikagi_2_sp_85dded42a7` | Phase 1 scratch | 2000 | strict合格 | 0.797113 / 0.761845 | 0 / 0 |
| `miu` | central-q50 + echo | 1500 | 探索終了・不合格 | 0.760725 / 0.737230 | 2 / 0 |
| `oop68_maid_san_no_iru_kurashi_s_sp_e4da3225a4` | Phase 1 scratch | 2500 | strict合格 | 0.808669 / 0.787547 | 0 / 0 |
| `oop69_maid_san_no_iru_kurashi_sp_07497b0fbd` | Phase 1 scratch | 2000 | pragmatic合格 | 0.766413 / 0.741996 | 1 / 0 |
| `oop70_osananajimi_no_iru_kurashi_sp_7195504dbb` | central-q50 + echo | 750 | 探索終了・不合格 | 0.717470 / 0.693072 | 15 / 0 |
| `oop73_toshishita_kanojo_sp_6b50dbf844` | Phase 1 scratch | 1000 | strict合格 | 0.786743 / 0.759362 | 0 / 0 |
| `oop77_anabel_maidgarden_sp_451488a7c1` | Phase 1 scratch | 2000 | strict合格 | 0.825513 / 0.805358 | 0 / 0 |
| `narrator_sayoko` | Phase 1 scratch | 1500 | strict合格 | 0.808816 / 0.775159 | 0 / 0 |
| `kasumi` | Phase 1 scratch | 1000 | pragmatic合格 | 0.796382 / 0.729432 | 1 / 0 |

### bounded refinement後の未達case

- `oop176`: similarityは16/16で0.75以上だが、step 750の
  `sentence_manko/seed-1234/calm` と `sentence_manko/seed-1234/neutral` が
  `tone_candidate` となり、その他hard gate失敗2件で不合格。
- `oop52`: step 750でsimilarity未達5件。`sentence_chinko/1234/calm`、
  `sentence_manko/5678/neutral`、`sentence_unko/1234/calm`、
  `sentence_unko/1234/neutral`、`sentence_unko/5678/neutral`。範囲は
  0.730189〜0.745229、平均も0.759852で基準未達。
- `miu`: step 1500で `sentence_manko/5678/calm` が0.745769、
  `sentence_unko/5678/neutral` が0.737230。未達2件は許容上限内だが、平均0.760725が
  0.765未満のため不合格。
- `oop70`: step 750の16件中15件がsimilarity未達。内訳はcontrol 4/4、
  sentence_chinko 4/4、sentence_manko 3/4、sentence_unko 4/4で、範囲は
  0.693072〜0.737811。外れ値数・floor・平均のすべてで基準未達。

### 運用上の復旧履歴

- Phase 1本体は全訓練・評価を完了したが、kasumiのacceptance wrapper欠落によりqueue terminalは
  `PARTIAL_FAILURE`。`acceptance-recovery-v001.json`で既存10件と復旧1件をhash bindingし、
  11件の判定を閉じた。
- full-clean初回queueはrefinement用checkpoint stepを旧generatorが拒否したため停止した。
  訓練成果物を変更せず、専用generatorを明示した
  `phase3_full_clean_lr0001_eval_recovery_v001`で4話者各140/140評価を完了した。
- echoのv001〜v003は順にECAPA directory binding、bundle内依存、full-clean evaluation rootの
  暗黙前提をfail-closedで露出した。v004ではdirectory tree hash、依存ファイル集合、evaluation rootを
  明示契約にし、運用失敗0で `COMPLETE` した。失敗rootは上書きせず証跡として保持した。

### 検証記録

- final remote verifier: `PASS`、12 models、8 accepted、4 rejected after bounded refinement
- runtime tool tests: 61 passed
- 全runtime Python fileの`py_compile`: PASS
- isolated Ruff critical rules (`E9,F63,F7,F82`): PASS
- repository用の全Ruff profileを一時runtimeと取り込み済みupstream sourceへ適用した結果は、
  意図的なfile-level `noqa`、implicit namespace、repo固有style規約により186件。実行時toolの
  構文・contract test・remote hash検証とは分離した既知のscope差として記録し、成果物は変更していない。

## Operator disposition追補（2026-08-04）

元の自動判定はimmutableのまま保持し、ユーザーの聴感確認に基づく3話者の手動実用合格と、
oop70のduration補正再学習結果を別のoperator証跡へ集約した。最終内訳は自動合格8、手動合格3、
再学習後不合格1である。配備は行っていない。

| 話者 | operator判定 | step | embedding SHA-256 | 自動評価平均 |
|---|---|---:|---|---:|
| `oop176_natsu_no_owari_sp_dcec9a11d3` | `MANUAL_ACCEPTED` | 750 | `750d95d3895f036d840e069dfb47018c4e98f6c61253f5cd185e384bc584aeb5` | 0.79867819 |
| `oop52_aibeya_2_sp_5d544fe890` | `MANUAL_ACCEPTED` | 750 | `201beaca8962ec04764a79f0949a40cd270a8f707cacd7d3532393daf57ef4ff` | 0.75985176 |
| `miu` | `MANUAL_ACCEPTED` | 1500 | `528fb261969cced5ce1508354330628b9e8dd01773d47d7684307c6dfa2401b5` | 0.76072490 |

oop70はduration/text-length residualで選んだ800件を用いてscratchから3000 step再学習し、
5 checkpoint × 28 caseを140/140成功で評価した。最高平均はstep 2500の0.71592508で、
旧最高0.71898819を上回らず、echo開始条件0.73898819にも届かなかった。このため
`SKIPPED_NO_IMPROVEMENT`として探索を終了し、`REJECTED_AFTER_DURATION_CORRECTED_RETRAINING`とした。

最終証跡は
`v4_compatibility/v4_speaker_inversion_operator_v001/operator-disposition-v001.json`、SHA-256
`8d36021f1c45fe2a3a9a987cae2e321ac5a7dfdf2db351b2dded52aaa11935a1`。fresh verifierと別processの
auditはいずれも`PASS`で、GPU free 11,078 MiB、training/queue process 0、v3 Git treeと13話者
voice bankは開始前から不変だった。

### oop70最高checkpointの手動採用（2026-08-05）

ユーザー指示により、oop70は自動不合格履歴を変更せず、完了済みscratch 5候補のうち平均類似度が
最高だったstep 2500を手動採用した。平均は`0.7159250827723578`、最小は`0.688714710818511`、
embedding SHA-256は
`bfadf21004c16b0a71ed44e0884ab53342e36ed2f8efbbeedbc8073373ccbe9b`。tensorは
`F32[16,768]`かつ全値finiteである。

最新operator dispositionは
`v4_compatibility/v4_speaker_inversion_operator_v001/operator-disposition-v002.json`、SHA-256
`e014e64ad311ca4fc994eb245c272d13e3c65099af4db388cc394022d7299c34`。内訳は自動合格8、手動合格4、
不合格0となり、operator上は12話者すべて採用済みである。手動採用証跡
`oop70-manual-acceptance-override-v001.json`のSHA-256は
`5e91c1960a3f74373aa50412e374b5247a2de5f172efda0753ab0136db34acb1`。create時auditと別process auditは
ともに`PASS`、配備は行わず、v3・voice bankは不変である。
