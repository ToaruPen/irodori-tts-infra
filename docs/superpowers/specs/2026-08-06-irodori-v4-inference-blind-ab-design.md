# Irodori-TTS v4 推論設定 Blind AB 設計

**Status:** 実装・live browser/service smoke・score完了
**Date:** 2026-08-06

## 目的

Irodori-TTS v4-Smallの現行設定`24 steps / linear / neutral`と、高速化候補
`12 steps / sway / neutral`を、条件名を見ずに一人で簡単に聴き比べられるようにする。

既存の自動評価では、12/swayは全runtime voiceの匿名sweepでclient/server p95を約35%短縮し、
CER、speaker similarity、duration、RMS、silence、clippingのgateを通過した。production昇格に
残るgateは、人間が読み、声、ノイズ、自然さ・韻律、感情の明確な悪化を検出しないことである。

## Scope

- `/capabilities`が返す動的default voiceを1つだけ評価する。
- 固定評価文6件とseed `101`、`202`の直積、計12 pairを作る。
- 各pairで同じtext、voice、seedを24/linearと12/swayへ送る。
- A/B割当、画面上のpair順、serverへの条件送信順をランダム化する。
- ローカルHTMLで再生、回答、進捗保持、結果JSON保存を行う。
- private answer keyを使う別commandで条件を開示し、集計する。
- 出力はruntime assetとしてcreate-only directoryへ保存する。

## Non-goals

- moco operator UIへの統合
- 複数reviewerの認証、同期、合議、inter-rater統計
- runtime catalogの全voiceを使う聴覚評価
- 音量正規化、無音trim、resample、その他の音声加工
- browserからanswer keyまたはcondition名を読むこと
- score結果によるmoco設定の自動変更
- voice名、件数、順序をtest fixtureへ固定すること
- native streaming、checkpoint、voice bank、caption、CFG設定の比較

## 選択した方式

### ローカルHTML評価packet

terminalだけの対話式評価は、再生し直し、前の回答の修正、進捗確認が分かりにくい。moco UIへ
統合すると評価用runtime assetとproduction stateの境界が曖昧になる。したがって、Irodori infraが
自己完結したローカルHTML packetを生成する。

HTML packetはHTTP serverを必要としない。`index.html`、固定`review.js`、生成された
`manifest.js`、相対pathのWAVを`file://`で開く。条件対応表はpacket外の`private/`に置き、HTMLは
参照しない。

### Integrity保証境界

packet内のhashとprivate answer keyは、偶発的な破損や一部artifactだけの改変を検出するための
整合性情報であり、packet全体の暗号学的な真正性を保証する署名ではない。同じユーザー権限でpacket、
answer key、resultsをすべて整合的に改変できる攻撃者は非目標とする。これに耐える必要がある運用では、
packet外の別権限または別媒体に署名・MAC等の外部sealを保存しなければならない。

ただし実行可能な`index.html`と`review.js`については、scoreとbrowser open直前に、answer keyのdigestに
加えてrepository/package側のtrusted canonical bytesとの完全一致をbounded readで要求する。これにより
packet側UIとanswer keyだけを同時に改変しても受理しない。trusted repository/package asset自体を変更
できるsame-UID攻撃者と、検証完了後からbrowser実行までのsame-UID TOCTOUは上記非目標に含む。

`index.html`はstrict CSPでnetwork接続、object、base、form、frame、workerを禁止し、同一local packetの
script/mediaと固定inline styleだけを許可する。`manifest.js`は固定prefix/suffixとstrict JSON parserで
open前に検証する。動的manifestを非実行dataとして固定HTMLへ埋める方式はcanonical HTML照合を失い、
外部JSON読込は`file://`互換性と`connect-src 'none'`に反するため、今回は移行しない。

## ユーザー操作

### 1. Packet作成

```bash
just v4-inference-blind-ab prepare \
  --base-url http://127.0.0.1:18924 \
  --output-dir /tmp/irodori-blind-ab \
  --open
```

- `--base-url`は数値loopback HTTP URLだけを受け付ける。remote GPUは既存と同様、SSH tunnelを
  loopbackへ終端して利用する。
- `--output-dir`は存在してはならない。
- `--open`は生成完了後だけdefault browserで`packet/index.html`を開く。
- 成功時はpacket path、private key path、pair数だけを表示する。voice metadata、condition割当、
  generationは表示しない。

### 2. 回答

各画面は次だけを表示する。

- 評価文
- Aのnative audio control
- Bのnative audio control
- `Aが良い`
- `Bが良い`
- `同等`
- `判断できない`
- 任意の理由: `読み`、`声`、`ノイズ`、`自然さ・韻律`、`感情`
- `前へ`、`次へ`
- 現在位置と未回答数

autoplayは行わない。A/Bは任意の順で何度でも再生できる。回答はpacket IDをkeyにbrowser
localStorageへ保存し、reload後に復元する。全12件へ回答するまで最終結果のdownloadは無効にする。

### 3. Score

browserは`irodori-blind-ab-results.json`をdownloadする。答え合わせは次で行う。

```bash
just v4-inference-blind-ab score \
  --packet-root /tmp/irodori-blind-ab \
  --results ~/Downloads/irodori-blind-ab-results.json
```

scoreは人間向けsummaryをstderrへ、機械可読JSONをstdoutへ返す。

- 12/sway勝利数
- 24/linear勝利数
- 同等数
- 判断不能数
- 理由別の勝敗内訳
- baselineが優位かを調べるone-sided exact binomial p-value
- `no_detected_degradation`、`degraded`、`inconclusive`のいずれか

## Generation契約

### 固定条件

| 項目 | Baseline | Candidate |
| --- | --- | --- |
| num_steps | 24 | 12 |
| t_schedule_mode | linear | sway |
| style | neutral | neutral |
| num_candidates | 1 | 1 |

text、voice ID、runtime generation、seed、duration scale、CFG scale、sway coefficientはpair内で同一に
する。評価文は既存inference benchmarkの6文を再利用し、別の固定話者表やvoice listを持たない。

### Voice解決

1. `/health`と`/capabilities`を取得する。
2. `ready=true`かつ`readiness=ready`を要求する。
3. `default=true`のvoiceがちょうど1件であることを要求する。
4. 全synthesis requestへその`voice_id`と取得時の`if_generation`を送る。
5. generation mismatch、voice消失、model unloadはfallbackせずpacket作成を失敗させる。

話者ID、label、alias、catalog件数、順序はpublic/privateどちらの成果物にも保存しない。private keyは
voice IDとgenerationのSHA-256だけを再現証跡として持ち、raw値は持たない。

### Randomization

OSのCSPRNGからpacket seedを作る。12 pairのA/B割当はbaseline Aを6件、candidate Aを6件にした
配列をshuffleし、左右の偏りを固定する。画面上のpair順を独立にshuffleする。serverへの2条件の
送信順もpairごとに独立にrandomizeし、warm cacheや温度の一方向biasを減らす。

random seedと全割当はprivate keyだけへ保存する。public packetからseedを復元できないようにする。
scoreはprivate random seedから同じ決定的アルゴリズムでpair表示順、baseline side、条件送信順を
再構築し、public/private metadataとの順序込み完全一致を要求する。形式と6対6 balanceだけが正しい
改変もintegrity errorとして拒否する。

## Artifact構造

```text
<output-dir>/
├── packet/
│   ├── index.html
│   ├── review.js
│   ├── manifest.js
│   └── audio/
│       ├── <opaque-id>.wav
│       └── ...
└── private/
    └── answer-key.json
```

packet ID、pair ID、audio IDにはCSPRNGで生成した128-bit値のlowercase hex表現を使う。
`packet/manifest.js`は、次のmanifest payloadと`manifest_sha256`を持つ固定のJavaScript代入式とする。
SHA-256は、payloadをUTF-8、key sort、空白なしでcanonical JSON化したbytesに対して計算する。
これによりbrowserはWeb CryptoやHTTP fetchを使わずdigestをresultsへコピーでき、scoreはpayloadから
digestを再計算できる。

manifest payloadは次だけを含む。

- schema version
- random packet ID
- pair ID
- 表示するtext
- opaqueなA/B audio relative path
- reason enum

`manifest_sha256`以外にcondition名、num_steps、schedule、seed、voice metadata、generation、生成順、
answer keyは含めない。
`review.js`と`index.html`もcondition文字列を含めない。audio filenameはrandom opaque IDとし、
condition、pair順、seedをencodeしない。

`private/answer-key.json`は次を含む。

- schema version
- packet ID
- canonical public manifest payloadのSHA-256
- 各audioのSHA-256
- 固定`index.html`と`review.js`のSHA-256
- pair IDからbaseline/candidateのA/B割当への対応
- sample indexとseed
- 条件送信順
- randomization seed
- runtime generationとvoice IDのSHA-256

## Results契約

browserが作るresults JSONは次を含む。

```json
{
  "schema_version": "irodori-v4-inference-blind-ab-results/v1",
  "packet_id": "opaque",
  "manifest_sha256": "64 lowercase hex chars",
  "answers": [
    {
      "pair_id": "opaque",
      "choice": "a|b|same|unsure",
      "reasons": ["reading|voice|noise|prosody|emotion"]
    }
  ]
}
```

自由記述は保存しない。reviewer名、時刻、host情報、voice情報を保存しない。scoreは次を拒否する。

- 未知schema version
- packet IDまたはmanifest hash不一致
- pairの欠落、重複、未知ID
- 未知choiceまたはreason
- 同じreasonの重複
- answer key、manifest、audio hashの不一致
- `index.html`または`review.js`のhash不一致
- repository/package側canonical `index.html`または`review.js`との不一致
- private random seedから再構築した表示順、A/B割当、送信順との不一致

## Score判定

private keyでA/Bをbaseline/candidateへ戻し、次を数える。

- `candidate_wins`
- `baseline_wins`
- `same`
- `unsure`
- `decisive = candidate_wins + baseline_wins`

baselineがcandidateより好まれる確率を帰無仮説0.5としたone-sided exact binomial p-valueを、tiesと
unsureを除くdecisive pairで計算する。

1. `unsure >= 4`なら`inconclusive`。
2. それ以外で`baseline_wins > candidate_wins`かつp-value `<= 0.05`なら`degraded`。
3. それ以外は`no_detected_degradation`。

`same`は「差を検出しなかった」という有効回答として扱う。理由は補助集計であり、自動判定を
変更しない。結果が`no_detected_degradation`でもproduction設定は変更しない。採用にはscore、
既存自動metric、ユーザーの明示承認がすべて必要である。

## Resource・安全境界

- packetは24 WAV、各4 MiB、各60秒、合計96 MiBを上限とする。
- 全runは15分を上限とする。個別synthesisはmocoと同様に切断timeoutを設けず、run全体で止める。
- HTTP buffered responseは展開後8 MiBを上限とする。
- WAVはRIFF declared boundaryをchunk header、payload size、偶数padding単位で走査し、PCM16、sample
  rate、channel、frame count、duration、sizeを検証する。`fmt`と終端`data`は各1件だけ許可し、
  torchaudio出力互換の`fmt`後`LIST` metadataだけを各1 MiBまで無視する。`JUNK`と未知chunkは拒否する。
- output parentを先に検証し、所有する一時directoryへ生成後、完成時だけdestinationへrenameする。
- destinationが存在する、symlinkである、parent外へ解決される場合は拒否する。
- 失敗時に削除するのは、そのrunが`tempfile.mkdtemp`で作った一時directoryだけとする。
- standard service、voice bank、runtime設定、moco設定を変更しない。
- packet、results、scoreはgit管理対象外のruntime assetとし、commitしない。
- `--open`は完成packetをstrict再検証し、canonical UIまたは他artifactが変化していればbrowserを起動しない。

## Error処理

CLIはcontent-freeなstable codeと非zero exitを返す。

- `runtime_not_ready`
- `default_voice_unavailable`
- `runtime_generation_mismatch`
- `response_too_large`
- `invalid_wav`
- `audio_too_large`
- `audio_too_long`
- `blind_ab_timeout`
- `browser_open_failed`
- `output_exists`
- `unsafe_output_path`
- `packet_integrity_error`
- `invalid_results`
- `client_error`

remote message、voice metadata、text、filesystemのprivate absolute pathをerror JSONへ含めない。人間向け
stderrも、入力output/results path以外のprivate runtime情報を表示しない。
`--open`が失敗した場合は、完成済みpacketを削除せず`browser_open_failed`で終了し、packet pathを
stderrへ示して手動で開けるようにする。

## 実装境界

- `scripts/v4_inference_blind_ab.py`: prepare/score CLI、generation、randomization、integrity、score
- `scripts/assets/v4_inference_blind_ab/index.html`: 固定UI shell
- `scripts/assets/v4_inference_blind_ab/review.js`: playback、回答state、localStorage、results download
- `tests/scripts/test_v4_inference_blind_ab.py`: Python契約、artifact、score、resource、安全test
- `tests/js/v4_inference_blind_ab.test.js`相当は新しいJS toolchainをinfraへ追加しない。browser logicは小さく
  保ち、Node等の既存runnerがない限り、generated manifestとDOM hookのPython test、およびlive packetの
  手動smokeで確認する。
- `justfile`: `v4-inference-blind-ab` recipe

既存`benchmark_v4_inference.py`は通常runで音声を保持しない責務を維持する。blind packet作成を既存
benchmarkのoptionへ混在させない。評価文だけを単一ownerへ抽出する場合も、condition計測とpacket生成の
CLI責務は分ける。

## TDD・検証計画

### Unit

- dynamic default voiceをruntime catalogから解決する。
- voice名、件数、順序に依存しない。
- 24/linearと12/sway以外をpacket conditionへ入れない。
- 12 pairのA/Bが6対6で、pair順・送信順がinjectされたseedで再現する。
- public artifactへcondition、seed、voice、generationが漏れない。
- private mappingが全pair/audioを一意にcoverする。
- create-only、symlink、path traversal、atomic completionを守る。
- generation change、WAV/resource failureでpartial packetを残さない。
- resultsのmissing、duplicate、unknown、tamperを拒否する。
- exact binomialと3状態判定の境界を固定する。
- scoreがcondition別勝敗とreasonを正しく戻す。

### Repository gate

- `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py`
- `uv run ruff check ...`
- `uv run ruff format --check ...`
- `uv run mypy ...`
- `just check`
- `git diff --check`

### Live smoke

2026-08-06に次を確認した。

- tunnel先のstandard v4 serviceが`ready=true`であることを確認する。
- 動的default voiceで24 WAVを生成する。
- packetをbrowserで開き、再生、戻る、reload復元、全回答、downloadを確認する。
- scoreでpacket integrityと12回答を確認した。
- 8923の別serviceとvoice bankを変更せず、隔離用tunnelと一時sourceを削除し、標準serviceを
  `model_loaded=true`へ復旧した。

live smokeは`live`、`gpu`、必要なら`ssh` marker相当として通常testから除外し、自動deploymentや
production昇格を行わない。

## 2026-08-06 実施結果

動的default voiceを使って24 WAV、12 pairのpacketを生成した。ユーザーのblind回答は全12組が
`same`で、scoreは`candidate_wins=0`、`baseline_wins=0`、`same=12`、`unsure=0`、
`outcome=no_detected_degradation`だった。既存の客観metricとこの結果を根拠に、ユーザーはmocoの
現在の運用設定として`12 steps / sway`を明示承認した。

この選択はHTTP通信契約の固定値ではない。評価toolは比較条件を再現するが、mocoのproduction testは
選択中のsteps/scheduleを契約fixtureとして固定せず、有効な設定値をrequestへ転送できることだけを
検証する。

## Rollback・cleanup

この機能はproduction pathを変更しないため、rollbackは生成したpacket runtime assetの削除だけである。
削除対象はユーザーが指定した単一packet rootへ限定し、自動cleanup commandは設けない。score結果が
`degraded`または`inconclusive`なら24/linearを維持する。今回の`no_detected_degradation`と明示承認を
受け、mocoのactive configだけを12/swayへ変更した。必要なら同じ設定境界から24/linearへ戻す。
