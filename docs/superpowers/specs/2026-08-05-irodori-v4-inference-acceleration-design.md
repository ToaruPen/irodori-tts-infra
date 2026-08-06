# Irodori-TTS v4 推論高速化設計

**Status:** request sampling候補の客観評価完了、blind ABは未完、Windows/Linux compile・量子化候補は検証完了・不採用
**Date:** 2026-08-05

## 目的

moco が利用する Irodori-TTS v4-Small の生成方式、Speaker Inversion voice bank、
neutral caption mode を維持したまま、合成開始から完成 WAV 受信までの遅延を短縮する。
品質を落として速度だけを得る候補は採用せず、隔離測定と品質 gate を通過した設定だけを
通常経路の候補とする。

## 現状

- 評価開始時点のmoco baselineは `24 steps / linear / neutral / 1 candidate` である。
- Irodori server 自体の request default は40 stepsだが、mocoは24を明示送信する。
- runtime は CUDA上のBF16、codecはCUDA上のFP32、context KV cache有効で動作する。
- `torch.compile` は実装済みだがinfra設定では無効である。
- standard serviceはRTX 4070上で稼働中であり、GPUメモリの大半を保持している。
- 現行 `/synthesize_stream` は、一文節を完全生成・decodeした後のWAV byte列を分割する。
  推論途中のlatentまたはPCMを返すnative streamingではない。
- v4-Smallは全時間長のlatentへ非因果attentionを使い、各RF stepで全系列を更新してから
  DACVAE decodeする。この設計を変更するnative streamingは本作業に含めない。

## 検討する方式

### A. request単位のsampling最適化（最初に実施）

稼働中standard serviceを停止せず、`num_steps` と `t_schedule_mode` だけを変えて比較する。
product baselineの24/linearと、24/sway、20/sway、16/sway、16/linear、12/swayを同じ
text、voice、seedで交互実行する。モデル、埋め込み、caption、CFG scaleは変えない。

利点は、rollbackがrequest値だけで済み、compileやcheckpoint交換が不要なこと。欠点は、
step削減が読み、話者同一性、ノイズ除去、韻律へ影響し得ること。最初の推奨方式とする。

### B. runtime実行最適化（Aの後に実施）

`torch.compile`、可変発話長向けのdynamic shape、十分なwarm-upを隔離processで比較する。
sampling設定はbaselineに固定し、compile初回時間と定常時間を分けて記録する。

品質リスクは低いが、初回compile、shapeごとの再compile、VRAM増加があり得る。standard
serviceを停止・置換せずに別processを起動できるGPU余力がない場合は実行せず、明示的な
service停止許可を待つ。

### C. 量子化checkpoint（最後に実施）

公式のv4-Small Quantizedから、ハードウェアに適合するINT8 weight-onlyを最初に比較する。
FP8はRTX 4070の対応可否をpreflightで確認してから扱い、INT4は速度よりVRAM削減候補と
して後順位にする。Speaker Inversion embeddingは同じbase由来でも、量子化後の話者同一性を
当然とは扱わない。

native streaming、モデル再学習、codec交換はこの設計では採用しない。

## 所有境界

- Irodori infraはbenchmark、runtime設定、checkpoint互換、品質判定を所有する。
- mocoは合格済みsampling profileだけを送信し、model pathやquantization名を知らない。
- voiceは `/capabilities` のruntime catalogから選び、名前、件数、順序をfixtureへ固定しない。
- standard service、voice bank、runtime generationは自動で変更しない。

## Benchmark契約

benchmarkは `scripts/benchmark_v4_inference.py` に置き、次の性質を持つ。

- 数値loopback IPのIrodori HTTP API、または数値loopbackへ終端したSSH tunnelだけを使用する。
  direct HTTP transportを明示し、環境変数のproxyやDNS名を経由しない。
- `/capabilities` からdefault voiceを一意に解決する。defaultがなければ明示selectorを要求する。
- voice IDとruntime generationをrequestへ相関させ、途中変更をfail closedにする。
- 固定された非機密の日本語sampleとseedを使用する。
- condition列の先頭は現行baselineの`24 / linear`だけを許可し、全candidateのreferenceとする。
- condition順序をsampleごとに回転し、warm cacheや温度の一方向biasを減らす。
- WAVはmemory内で検査し、通常runでは保存しない。
- JSON summaryへ本文、voice ID/label/alias、generation、音声、captionを含めない。
- server契約は1 requestの`num_steps`を64以下に制限する。benchmarkは最大8 condition、
  8 seed、128 dynamic voice、2,048 synthesis trial、32,768 step-unitに制限し、catalogの
  実際のvoice名、件数、順序には依存しない。
- benchmark clientは非stream応答を8 MiB、decode対象WAVを4 MiBかつ60秒、run全体を15分に
  制限する。汎用async clientもbuffered応答とstreaming error本文に既定64 MiBの上限を持ち、
  callerがより小さい値を指定できる。圧縮応答は展開後の累積byte数で判定する。
- timeout、HTTP error、invalid response、invalid WAV、generation mismatchをallowlist済みの
  stable failure codeで返す。resource上限超過も`benchmark_workload_too_large`、
  `response_too_large`、`wav_too_large`、`audio_too_long`、`benchmark_timeout`へ正規化し、
  未知のremote error codeやvalidation詳細は出力しない。
- synthesisまたは検査に1件でも失敗したrunは、その時点でcontent-free failure JSONを返して
  非zero終了する。成功summaryは全trial成功を意味するため、固定値のfailure countを持たない。

各測定は次を記録する。

- client elapsed milliseconds
- server-reported synthesis elapsed milliseconds
- WAV duration、sample rate、channel count
- RMS、silence ratio、clipping ratio
- realtime factor (`server elapsed / audio duration`)

summaryはconditionごとのcount、p50、p95、mean、RMS、technical audio gateに加え、全clipの
RTF最大値・違反数・baseline悪化数、匿名trial順で対応付けたduration drift最大値・違反数を
保持する。server/client p95の15%短縮を含む`request_technical_gate_pass`も返すが、これはCER、
speaker similarity、blind ABを含むproduction採用gateではない。本文やvoiceを識別できるkeyは
対応付けに使わない。

## 実験段階

### Phase 1: default voice pilot

動的default voice、6本以上の短文・長文・読点・疑問・引用・難読寄りsample、3 seedで
全sampling conditionを比較する。明らかに遅い候補、technical audio failure、duration driftを
除外する。

### Phase 2: dynamic catalog sweep

Phase 1の上位2候補だけを、capabilitiesが返す全voiceに対して代表3文・2 seedで比較する。
voice数と順序は実行時に決まり、summaryにはvoiceを識別できる情報を残さない。

### Phase 3: quality metrics

baselineと最良候補について、既存のWhisper日本語transcriptionとspeaker metricを用い、
normalized CER、speaker similarity、technical audio metricを比較する。生成物が必要な場合は
create-only一時directoryへ置き、判定後に削除する。

### Phase 4: compile / quantization

standard serviceと競合しない時だけ、同じPhase 1入力でcompile eager/compiled、次にbase/
quantizedを比較する。compileは初回時間、warm steady-state、recompile数を別々に扱う。

## 採用gate

sampling候補は次をすべて満たす必要がある。

- server synthesis p95をbaseline比15%以上短縮する。
- client elapsed p95をbaseline比15%以上短縮する。
- synthesis、HTTP、WAV validation failureが0件。
- RTFが全sampleで1未満、かつbaselineより悪化しない。
- 各clipのduration差がbaseline比10%以内。
- silence anomaly、clipping anomalyが0件。
- normalized CERの悪化が絶対2 percentage points以内。
- speaker similarityの低下が0.03以内で、既存の話者hard gateを下回らない。
- blind ABで読み、話者、ノイズ、韻律、感情に明確な悪化がない。

compile候補は同じ品質gateに加え、warm steady-state p95を10%以上短縮し、compile後のpeak
VRAMが12GB境界を超えず、可変長sampleで反復recompileしないことを求める。

quantized候補は同じ品質gateを一切緩和しない。VRAMだけ減り速度が改善しない候補は、今回の
推論高速化として採用しない。

## Rolloutとrollback

1. request tuningをstandard serviceへ非配備で測定する。
2. 上位候補だけcatalog sweepと品質metricへ進める。
3. compile/quantizationは隔離processで測定する。
4. gate通過後も結果と設定を提示し、通常moco設定またはIrodori runtimeを明示的に変更する。
5. sampling rollbackはmocoの24/linearへ戻す。
6. runtime rollbackはcompile無効、現行base checkpointへ戻す。

## 2026-08-05 実測結果

標準serviceを停止または変更せず、mocoの`24 steps / linear / neutral`をbaselineとして
request-level比較を実施した。default voiceの6文・3 seed（各条件18件）では次の結果だった。

| condition | server p95 | client p95 | baseline比 server短縮 | technical audio |
| --- | ---: | ---: | ---: | --- |
| 24 / linear | 1,156 ms | 1,175 ms | baseline | pass |
| 16 / linear | 809 ms | 870 ms | 30.0% | pass |
| 16 / sway | 819 ms | 851 ms | 29.2% | pass |
| 12 / sway | 650 ms | 741 ms | 43.8% | pass |

capabilitiesがその時点で返した全voiceを、名前、件数、順序をfixtureへ固定せずに匿名sweepした。
最初の3文・1 seedの各42件では、`12 / sway`がserver p95を968 msから621 msへ35.8%、client
p95を1,014 msから676 msへ33.3%短縮した。このpilot後、benchmarkをfail-fast、loopback限定、
allowlist error、port検証、非finite値拒否、clip単位gateへ強化し、短文・長文・疑問・引用・
難読表記を含む6文・2 seedで再測定した。

最終の全catalog sweepは各条件168件となり、`12 / sway`はserver p95を1,939 msから1,268 msへ
34.6%、client p95を2,031 msから1,326 msへ34.7%短縮した。両条件とも失敗0件、technical audio
passで、candidateの`request_technical_gate_pass`はtrueだった。candidateは全clipのRTFが1未満、
baseline比RTF悪化0件、paired duration差最大0.0%で、RTF最大値は0.720だった。baselineは実行時の
負荷揺らぎによりRTF最大1.245、違反2件だったが、candidate側に違反はなかった。

既存のpinned ECAPAとWhisperをCPUで実行し、GPU serviceと競合させず品質を確認した。代表
voiceの6文では全候補の平均normalized CERは0.008772で同一だった。`12 / sway`の平均speaker
similarityは0.818995、最小は0.779723で、baselineの平均0.820167からの低下は0.001172だった。
さらにdynamic catalog全体の同一文paired確認ではbaseline/candidateともCERは全件0、CER悪化
件数0、baselineと`12 / sway`間のECAPA similarityは平均0.983961、最小0.936903だった。
本文、voice metadata、generation、WAV、transcriptは集計へ残さず、一時生成物は削除した。

以上から客観metric上のrequest-level最良候補を`12 steps / sway / neutral`とする。ただしblind
ABが未完であるためproduction winnerには昇格せず、mocoの既定値とactive configは`24 / linear`
を維持する。mocoはmodel artifactを知らず、明示的な試験時だけこの2個のsampling値をtyped
requestとして送れる。rollback値も`24 / linear`である。

2026-08-05のbaseline測定時はstandard serviceが11,482 MiB / 12,282 MiBを使用していたため、
別processによるcompileまたはquantization比較を実行しなかった。その後、明示承認された
maintenance windowでstandard serviceを停止し、以下のPhase 4 compile検証を続行した。

### 2026-08-06 compile検証

明示的に承認されたmaintenance windowでstandard serviceを停止し、同じGPUの固定
upstream runtimeにて小さなCUDA tensorを使う`torch.compile` preflightを実行した。停止後の
空きVRAMは11,108 MiB / 12,282 MiBだった。

- 高速化候補の標準Inductor backendは当初`missing_triton`で失敗した。
- `aot_eager` backendはCUDA上でfinite outputを返し、`torch.compile`のgraph capture自体は動作した。
- 実行環境はPyTorch 2.10.0+cu128で、Pythonから利用可能なTriton packageは存在しなかった。

追加承認後、PyTorch 2.10に対応する`triton-windows==3.6.0.post26`だけを`--no-deps`で
検証runtimeへ導入した。小規模CUDA tensorのInductor compileは成功し、初回3.39秒、
同一shapeの定常実行は約0.06ms、eagerとの差は最大`1.2e-7`だった。標準runtimeでは
GPU graphに含まれるCPU code生成にMSVCも必要だったが、Visual Studio Communityに既存の
toolchainをprocess-localな`VsDevCmd`環境で利用できたため、システム環境変数は変更していない。

同じcheckpoint、動的default voiceのembedding、neutral caption、seed 101、24 steps / linearで、短文3回、
長文1回、短文1回を別processのeager/static compileで比較した。

| 条件 | 短文初回 | 同一shape定常 | 長文へ変更 | 短文へ復帰 | unique graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 1.185秒 | 0.854–0.882秒 | 0.963秒 | 0.810秒 | 0 |
| static compile初回 | 86.450秒 | 0.378–0.482秒 | 68.425秒 | 34.979秒 | 11 |
| static compile disk cache再利用 | 17.707秒 | 0.365–0.403秒 | 8.920秒 | 5.620秒 | 11 |

static compileは同一shape定常2回の平均をeager比で約50%短縮し、disk cache再利用processでは
約56%短縮した。一方、発話長変更ごとに複数graphを
生成し、disk cache利用後も5.6–8.9秒の再compile待ちが生じた。`compile_dynamic=true`は
約4分のcompile後、Inductorのsymbolic shape処理でAssertionErrorとなり、最初の音声を返せなかった。
したがって可変長対話でcompile latencyを隠す経路は成立しない。

全compiled出力はfinite、clipping ratio 0で、duration、RMS、silence ratioもeagerと同程度だった。
ただし同じseedでもcompile/recompile境界ではeager波形と一致せず、最大絶対差は0.88–1.10、
平均絶対差は0.016–0.031だった。同一process・同一compiled graphの反復は一致したが、長文を
挟んで短文へ戻すと再び波形が変わった。この決定性差は技術的quality gateだけでは音質同等性を
証明できないことを意味する。

peak reserved VRAMはeager約4,894 MiB、static compile約4,906 MiBで、OOMは発生しなかった。
不採用理由はVRAMではなく、cold/recompile latency、dynamic compile failure、決定性差である。
Windows production runtimeでは`compile_model=false`を維持し、失敗したdynamic compileを
infraの公開設定へ追加しない。
検証後は両venvから`triton-windows`を削除し、一時scriptとeager reference tensorも削除した。
次回Phase 4を再開する場合はLinux CUDAで同じ可変長・決定性gateを先に通す。

検証終了後は同じdeployed sourceと設定でstandard serviceを再起動し、`/health`の
`model_loaded=true`と`/capabilities`の`ready=true`を確認した。deploy sync、`.env`の書き換え、
voice bankの変更は行っていない。

### 2026-08-06 eager operator検証

Windows standard runtimeのeager推論をPyTorch profilerで測定した。短文の24 steps / linearでは
`sample_rf`が定常合成時間の大半を占め、GPU operatorは`aten::mm`、BF16 GEMM、cuDNN
convolution、memory-efficient SDPAの順に支配的だった。SDPAは既に
`aten::_efficient_attention_forward`を使用していた。Windows版PyTorch 2.10.0+cu128はFlash
Attentionを含まず、Flash backendの強制実行は最初の音声を生成せずに拒否された。

FP32 matmulのTF32許可を`highest`と`high`で交互に各6回比較した。既定ではcuDNN TF32は有効、
matmul TF32は無効だった。小標本では`high`が速く見えたが、拡張測定では再現せず、`high`は
短文の平均を1.288秒から1.306秒、p95を1.405秒から1.481秒へ悪化させた。中程度文も平均を
1.429秒から1.481秒、p95を1.481秒から1.626秒へ悪化させた。同seed波形差は平均
`2.6e-7`から`3.4e-7`と小さかったが、速度gateを満たさないためTF32 matmulを有効化しない。

既存の`alternating` CFGもLinux eagerで比較したが、2つのbatch 1 forwardを逐次実行するため、
batch 3を1回実行する`independent`より遅かった。定常短文は0.780–0.812秒から
1.549–1.649秒へ、中程度文は1.069秒から1.589秒へ悪化した。CFG modeは変更しない。

### 2026-08-06 Linux CUDA再検証

既存のUbuntu 24.04 WSL2と同じRTX 4070に、固定source、Python 3.10、PyTorch
2.10.0+cu128、Triton 3.6.0の隔離環境を作成した。checkpoint、tokenizer、動的default voice、
neutral caption、seed 101、24 steps / linearはWindows検証と同一にした。小規模Inductor probeは
cold 8.36秒、同shape定常0.46ms、反復差0で成功した。

実モデルの可変長結果は次の通りだった。

| 条件 | 短文初回 | 同一shape定常 | 中程度文へ変更 | 短文へ復帰 | unique graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linux eager | 1.802秒 | 0.808–0.812秒 | 1.069秒 | 0.780秒 | 0 |
| static compile cold | 119.033秒 | 0.497–0.501秒 | 70.136秒 | 40.806秒 | 11 |
| static compile disk cache再利用 | 25.588秒 | 0.413–0.441秒 | 11.403秒 | 7.168秒 | 11 |

Linux static compileは同一shape定常を約38–47%短縮したが、disk cache後もshape変更の待ちを
対話用途の上限まで下げられなかった。同seedのeagerとの差は短文で平均0.00395、最大0.0783、
中程度文で最大0.245から0.348だった。全出力はfiniteでclipping ratio 0だったが、速度と
決定性の両gateを満たさない。peak reserved VRAMはeager/staticとも約4,650 MiBで、OOMは
不採用理由ではない。

`compile_dynamic=true`は約6分のcode generation後、Windowsと同じSymPy由来のInductor
AssertionErrorで最初の音声を生成できなかった。OS固有toolchainを除いてもdynamic shape問題は
再現したため、Linux移行や事前cacheだけでは可変長対話のcompile経路は成立しない。

### 2026-08-06 公式量子化checkpoint検証

同じbase modelの公式`Irodori-TTS-v4-Small-Quantized`から、一般用途推奨の
`int8-weight-only`と、RTX 4070が対応条件を満たす`float8-dynamic`を隔離比較した。Speaker
Inversion embedding、caption、seed、sampling値はbase eagerと同一にした。

- `int8-weight-only`はpeak reserved VRAMを4,650 MiBから3,146 MiBへ減らしたが、定常短文は
  0.969–1.192秒、中程度文は1.181秒でbase eagerより遅かった。
- `float8-dynamic`はpeak reserved VRAM 3,176 MiBだったが、定常短文を含む全発話が
  14.866–17.937秒となり、dynamic activation quantizationのoverheadが支配的だった。

両候補ともfinite、clipping ratio 0だったが、速度gateを満たさないためCER、speaker
similarity、blind ABへ進めない。今回の速度目的では量子化checkpointを採用せず、base
checkpoint、24 / linear、independent CFG、compile無効を維持する。

Linux検証用venv、Inductor cache、eager参照tensor、一時scriptは検証後に削除した。共有
Hugging Face cacheの量子化snapshotは別用途での再利用可能性があるため削除していない。
standard serviceは元のdeployed sourceと設定で再起動し、`model_loaded=true`、`ready=true`、
動的default voiceによるWAV合成を確認した。8923のserviceは停止・再起動せず、検証後も同じ
PIDでlistenしている。deploy sync、設定変更、voice bank交換は行っていない。

汎用async clientの展開後64 MiB上限は現在のinfra未コミット差分に含まれ、mocoが固定する
`2f169719`には上限指定APIがまだない。したがってinfra契約をcommitするまではmocoのpinを
動かさない。次のpin更新は同一migration単位で、mocoのhealth clientへ256 KiB、synthesis
clientへ既存の`max_wav_bytes`から算出したJSON上限を明示し、cross-repo testを通してから行う。
reflectionによる旧client互換や、設定可能なWAV上限の縮小は行わない。

training output、quantized checkpoint、benchmark winnerを自動promoteしない。

## 非目標

- native latent/PCM streaming
- arbitrary caption生成
- style preset追加
- voice bank、Speaker Inversion embeddingの再学習または交換
- GPU capacityの増加または並列synthesis
- watermarkの無効化
- standard serviceの無断停止・再起動
- 話者名、件数、順序を固定するtest

## リスク

- step削減は平均音質より先に、一部seed・難読文・特定voiceで破綻する可能性がある。
- swayの最適値はv4-Smallで公式品質保証されていない。
- compileは定常推論を速めても、初回と可変shapeの再compileで体感を悪化させ得る。
- quantizationはGPU kernel次第で速度が改善せず、話者同一性だけを落とし得る。
- standard serviceのGPU占有中はcompile/quantization比較を安全に実行できない。
- 自動metricは自然さを完全に表さないため、最終採用にはblind ABを残す。
