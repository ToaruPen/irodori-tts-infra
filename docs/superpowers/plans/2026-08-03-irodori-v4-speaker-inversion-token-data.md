# Irodori-TTS v4 Speaker Inversion Token/Data Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v4-Small Speaker Inversionのtoken数、学習データ構成、初期化を段階探索し、既存140-case契約で16 speaker hard-gateを全通過する候補をcreate-only証跡付きで得る。

**Architecture:** 既存v4 baselineのconfigと評価runtimeをimmutable sourceとして再利用し、新しい一時運用tool群で候補config、fail-closed supervisor、ECAPAデータ監査を作る。GPU runはWindows側のversioned rootへ直列に保存し、各段階をhashで束縛する。候補評価は既存のv4 140-case generator/analyzer/metrics/evaluatorをそのまま使う。

**Tech Stack:** Python 3.10/3.11、PyTorch、safetensors、SpeechBrain ECAPA、PyYAML、pytest、Windows PowerShell/SSH、Irodori-TTS v4-Small

---

## File structure

- Create: `/tmp/irodori-v4-si-token-data-tools/token_search_contract.py` — candidateと学習configの純粋な契約検証。
- Create: `/tmp/irodori-v4-si-token-data-tools/prepare_token_search.py` — baselineからcreate-only search/candidate rootを準備。
- Create: `/tmp/irodori-v4-si-token-data-tools/run_token_training_supervisor.py` — hash、process、GPU、完走条件をfail-closedで監視。
- Create: `/tmp/irodori-v4-si-token-data-tools/launch_token_training_detached.py` — supervisorをWindows detached processとして起動しhandoff証跡を作る。
- Create: `/tmp/irodori-v4-si-token-data-tools/audit_training_ecapa_distribution.py` — 2,223学習音声と25参照音声の固定ECAPA分布を監査。
- Create: `/tmp/irodori-v4-si-token-data-tools/build_central_manifest.py` — Stage 2だけで、監査結果から順序保存の派生manifestを作る。
- Create: `/tmp/irodori-v4-si-token-data-tools/test_*.py` — 純粋ロジック、改変拒否、create-only、launcher argvを検証。
- Reuse unchanged: `/tmp/irodori-v4-si-retraining-tools/prepare_irodori_v4_evaluation.py` — 5 checkpointの評価manifestとruntime bundle作成。
- Reuse unchanged: `/tmp/irodori-v4-si-retraining-tools/run_irodori_v4_candidate_evaluation_supervisor.py` — 140-case評価supervisor。
- Reuse unchanged: `/tmp/irodori-v4-si-retraining-tools/launch_irodori_v4_candidate_evaluation_detached.py` — 評価detached launcher。
- Create remotely: `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_token_data_v001` — 全create-only runtime証跡。
- Create locally: `/Users/sankenbisha/Downloads/irodori-v4-si-token-data-review` — 最終聴覚確認packet。

リポジトリのcommitはユーザーから明示依頼されていないため行わない。

### Task 1: Candidate contract

**Files:**
- Create: `/tmp/irodori-v4-si-token-data-tools/token_search_contract.py`
- Test: `/tmp/irodori-v4-si-token-data-tools/test_token_search_contract.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_scratch_contract_accepts_only_planned_token_candidate():
    candidate = Candidate("tokens8_scratch", 8, 0.01, 2, None)
    assert expected_training_contract(candidate)["speaker_inversion_tokens"] == 8

def test_contract_rejects_cross_shape_initialization():
    with pytest.raises(ValueError, match="shape"):
        validate_init_shape(token_count=8, shape=(16, 768))

def test_config_contract_detects_token_mismatch():
    expected = expected_training_contract(Candidate("tokens32_scratch", 32, 0.01, 2, None))
    actual = {**expected, "speaker_inversion_tokens": 16}
    with pytest.raises(RuntimeError, match="contract mismatch"):
        validate_contract(expected, actual)
```

- [ ] **Step 2: Run tests and verify Red**

Run:

```bash
uv run pytest -q /tmp/irodori-v4-si-token-data-tools/test_token_search_contract.py
```

Expected: collection failure because `token_search_contract` does not exist.

- [ ] **Step 3: Implement the minimal typed contract**

```python
@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    token_count: int
    learning_rate: float
    seed: int
    init_embedding: Path | None

def expected_training_contract(candidate: Candidate) -> dict[str, object]:
    return {
        "fresh_speaker_inversion": candidate.init_embedding is None,
        "speaker_inversion_tokens": candidate.token_count,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": False,
        "learning_rate": candidate.learning_rate,
        "max_steps": 3000,
        "save_every": 250,
        "log_every": 20,
        "seed": candidate.seed,
    }
```

`token_count`は`1..128`、learning rateはfinite positive、scratch候補のinitは`None`、warm-startは
`F32[token_count,768]`だけを許可する。configから同じfieldを抽出し、辞書の完全一致で検証する。

- [ ] **Step 4: Run tests and verify Green**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Create-only search preparation

**Files:**
- Create: `/tmp/irodori-v4-si-token-data-tools/prepare_token_search.py`
- Test: `/tmp/irodori-v4-si-token-data-tools/test_prepare_token_search.py`

- [ ] **Step 1: Write failing preparation tests**

Tests must assert:

```python
assert changed_leaf_paths(baseline, tokens8) == [
    "train.output_dir",
    "train.speaker_inversion_tokens",
]
assert setup["candidate"] == {
    "name": "tokens8_scratch",
    "token_count": 8,
    "learning_rate": 0.01,
    "seed": 2,
    "initialization": "scratch",
}
with pytest.raises(FileExistsError, match="refusing to reuse"):
    prepare_candidate(existing_root, ...)
```

Also mutate the baseline model SHA and manifest SHA independently and require rejection.

- [ ] **Step 2: Verify Red**

```bash
uv run pytest -q /tmp/irodori-v4-si-token-data-tools/test_prepare_token_search.py
```

Expected: missing module/function failure.

- [ ] **Step 3: Implement preparation**

The candidate table is exact and ordered:

```python
PLANNED = {
    "tokens8_scratch": Candidate("tokens8_scratch", 8, 0.01, 2, None),
    "tokens32_scratch": Candidate("tokens32_scratch", 32, 0.01, 2, None),
}
```

Load the immutable baseline `config.yaml`, `setup-evidence.json`, and `terminal-evidence.json`; verify their
recorded hashes and PASS/step-3000 status. Deep-copy config, change only
`train.speaker_inversion_tokens` and `train.output_dir`, then write `config.yaml` and
`setup-evidence.json` using exclusive creation. Bind base model, tokenizer, clean manifest, upstream,
baseline, script, candidate contract, and `deployment_performed: false`.

- [ ] **Step 4: Verify Green**

Run the command from Step 2. Expected: all tests pass.

### Task 3: Fail-closed training supervisor and launcher

**Files:**
- Create: `/tmp/irodori-v4-si-token-data-tools/run_token_training_supervisor.py`
- Create: `/tmp/irodori-v4-si-token-data-tools/launch_token_training_detached.py`
- Test: `/tmp/irodori-v4-si-token-data-tools/test_token_training_supervisor.py`
- Test: `/tmp/irodori-v4-si-token-data-tools/test_launch_token_training.py`

- [ ] **Step 1: Write failing supervisor tests**

Cover config/setup mismatch, wrong token count, dirty upstream, active train/service process, insufficient GPU,
nonzero child exit, step below 3000, missing periodic checkpoint, final/step3000 hash mismatch, and PASS.

```python
assert terminal["status"] == "PASS"
assert terminal["latest_logged_step"] == 3000
assert terminal["training_contract"]["speaker_inversion_tokens"] == 8
assert terminal["deployment_performed"] is False
```

Launcher test must assert fixed argv with `shell=False`, fresh launch-state enforcement, and that handoff is
written only after `supervisor-start-evidence.json` appears.

- [ ] **Step 2: Verify Red**

```bash
uv run pytest -q \
  /tmp/irodori-v4-si-token-data-tools/test_token_training_supervisor.py \
  /tmp/irodori-v4-si-token-data-tools/test_launch_token_training.py
```

- [ ] **Step 3: Implement supervisor and launcher**

Launch exactly:

```python
command = [
    str(v4_python), "-u", str(upstream / "train.py"),
    "--config", str(config),
    "--manifest", str(manifest),
    "--init-checkpoint", str(base_model),
    "--output-dir", str(output_dir),
    "--device", "cuda",
]
```

Before launch, bind and validate upstream commit, clean manifest SHA, model/tokenizer SHAs, candidate contract,
no competing train/service process, and at least10,500 MiB free GPU. Poll every5 seconds, atomically update
progress evidence, and retain peak GPU. At exit, require checkpoints 250..3000 plus final, finite final loss,
step3000, and identical final/step3000 hash. The launcher uses Windows flags
`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB` and never kills unrelated processes.

- [ ] **Step 4: Verify Green**

Run the command from Step 2. Expected: all tests pass.

### Task 4: Training-data ECAPA audit

**Files:**
- Create: `/tmp/irodori-v4-si-token-data-tools/audit_training_ecapa_distribution.py`
- Test: `/tmp/irodori-v4-si-token-data-tools/test_audit_training_ecapa_distribution.py`

- [ ] **Step 1: Write failing pure-function tests**

```python
rows = score_rows(source_rows, embeddings, centroid)
assert [row["source_id"] for row in rows] == ["speaker:0", "speaker:1"]
assert summarize(rows)["similarity"]["median"] == pytest.approx(0.75)
assert central_source_ids(rows, lower_quantile=0.25) == ("speaker:1",)
```

Also require duplicate source IDs, source/clean manifest identity mismatch, non-finite embedding, missing audio,
hash mismatch, and existing output root to fail before output mutation.

- [ ] **Step 2: Verify Red**

```bash
uv run pytest -q /tmp/irodori-v4-si-token-data-tools/test_audit_training_ecapa_distribution.py
```

- [ ] **Step 3: Implement audit**

Load `prepare-input.jsonl` and `clean-manifest.jsonl`, require identical ordered source IDs plus matching
`audio_sha256`, `pcm_sha256`, and text. Load the same 25 reference WAVs and the pinned
`speechbrain/spkrec-ecapa-voxceleb` revision used by evaluation. Use the production metric helper's
`SpeechBrainECAPA`, `aggregate_reference_centroid`, and `normalized_cosine_similarity`; decode OGG with
SoundFile and resample through the production helper. Write one scored JSONL row per source and a summary with
count, min/q01/q05/q10/q25/median/q75/q90/q95/q99/max, duration/text-length correlations, and bottom100 IDs.
Write output to a new root only, with hashes for all inputs and scripts.

- [ ] **Step 4: Verify Green**

Run the command from Step 2. Expected: all tests pass.

### Task 5: Verify and deploy operational bundle to the GPU host

**Files:** all files from Tasks1–4 plus unchanged evaluation tools.

- [ ] **Step 1: Run the complete temporary test suite**

```bash
uv run pytest -q /tmp/irodori-v4-si-token-data-tools
```

Expected: all tests pass.

- [ ] **Step 2: Compile scripts in both local and pinned remote runtimes**

```bash
uv run python -m py_compile /tmp/irodori-v4-si-token-data-tools/*.py
```

Copy scripts to the remote `_tools_token_data_v001` directory and run the v4 Python with `-m py_compile`.
Expected: exit0.

- [ ] **Step 3: Check runtime safety**

Use read-only process inventory and `nvidia-smi`; require no training/service process and at least10,500 MiB free.
Verify upstream commit/status and all pinned hashes. Expected: preflight PASS.

### Task 6: Run Stage 0 audit

**Files:**
- Create remotely: `...\v4_speaker_inversion_oop53_token_data_v001\diagnostics\training_ecapa_v001`

- [ ] **Step 1: Run audit detached or supervised**

Pass exact clean manifest, prepare-input, reference-wavs, production metrics script, ECAPA source/savedir, and
new output root. Expected:2,223 COMPLETE rows and terminal PASS.

- [ ] **Step 2: Verify evidence and choose the predeclared data path**

Recompute every output hash, verify row/source coverage, and record the distribution. Do not change the Stage1
manifest. Record whether Stage2 would use `central_q25` or a clearly evidenced cluster boundary.

### Task 7: Train and evaluate `tokens8_scratch`

**Files:**
- Create remotely: `...\v4_speaker_inversion_oop53_token_data_v001\candidates\tokens8_scratch`
- Create remotely: candidate `evaluation` subtree.

- [ ] **Step 1: Prepare candidate and run preflight**

Require the config diff to contain only output directory and token count. Expected contract:8 tokens, LR0.01,
seed2, scratch, 3,000 steps.

- [ ] **Step 2: Launch detached training and monitor to terminal evidence**

Do not interrupt the owned process while progress evidence advances. Expected: PASS, step3000,13 checkpoint
files including final, no owned process left, GPU released.

- [ ] **Step 3: Prepare and launch unchanged140-case evaluation**

Reuse the pinned evaluation preparer/supervisor/launcher. Expected:140/140 generation and evaluation COMPLETE.

- [ ] **Step 4: Decide eligibility**

Use the existing decision logic. If any checkpoint is ELIGIBLE and all16 speaker cases are at least0.75, skip
Task8 and proceed to Task10. Otherwise retain all evidence and continue.

### Task 8: Conditionally train and evaluate `tokens32_scratch`

**Files:**
- Create remotely: `...\v4_speaker_inversion_oop53_token_data_v001\candidates\tokens32_scratch`

- [ ] **Step 1: Repeat Task7 with the exact planned32-token contract**

The only config difference from baseline is output directory and `speaker_inversion_tokens: 32`. Run to3,000
and evaluate all140 cases.

- [ ] **Step 2: Compare 8/16/32 without changing gates**

Rank by: eligible checkpoint first, then minimum hard similarity, then hard failure count, then hard mean. Preserve
the full text/style/seed failure matrix. If eligible, proceed to Task10; otherwise continue to Task9.

### Task 9: Conditional data and initialization stages

**Files:**
- Create: `/tmp/irodori-v4-si-token-data-tools/build_central_manifest.py`
- Test: `/tmp/irodori-v4-si-token-data-tools/test_build_central_manifest.py`
- Create remotely: versioned `central_q25` candidate and, only if needed, same-shape refinement candidate.

- [ ] **Step 1: Red/Green-test deterministic manifest construction**

Require ordered intersection with clean manifest, exact retained count, no duplicate/missing IDs, copied rows
byte-equivalent after JSON decode, exclusive output, and provenance hashes. Run pytest and require PASS.

- [ ] **Step 2: Build `central_q25` and train the best token count**

Exclude only sources below the audit's precommitted25th-percentile cutoff. Train scratch with all other Stage1
conditions and run the full140-case evaluation.

- [ ] **Step 3: Conditionally run same-shape refinement**

Only if the best result is near threshold but not eligible, initialize from that exact-shape best checkpoint,
bind its hash, use LR0.001/seed2/max1,500/save250, and evaluate its checkpoint matrix in a separate root. Never
reshape an embedding between token counts.

- [ ] **Step 4: Conditionally continue an improving refinement once**

If the first refinement improves failure count, hard mean, and hard minimum over the baseline while all other
hard gates remain clear, initialize one new create-only run from its exact step-1500 embedding. Keep16 tokens,
`central_q25`, LR0.001, seed2, max1,500, and save interval250 unchanged. Evaluate new-run steps
250/500/750/1000/1500 as140 cases. Do not repeat the same continuation if it fails to improve either failure
count or minimum similarity over the first refinement.

- [ ] **Step 5: Evaluate exact checkpoint gaps before changing data again**

If unevaluated step-1250 files remain, build one create-only140-case diagnostic manifest from five exact saved
16-token checkpoints across the baseline and refinement chain. Bind each diagnostic slot to its original
run, step, path, and SHA-256. Do not average, interpolate, extrapolate, or reshape embeddings. If none is
eligible, proceed to a new data-composition hypothesis rather than repeating the same refinement.

- [ ] **Step 6: Run one precommitted `central_q50` identity refinement**

Build a create-only manifest by retaining the 1,112 rows whose pinned Stage0 ECAPA similarity is at least
the pre-existing median `0.8050478070701037`. Preserve source order and rebase latent paths exactly. Warm-start
from the exact Stage4 step-1500 `F32[16,768]` checkpoint, keeping16 tokens, LR0.001, seed2, max1,500, and
save250. Evaluate steps250/500/750/1000/1500 with the unchanged140-case contract. Do not repeat this data
hypothesis if it is rejected.

- [ ] **Step 7: Evaluate the `central_q50` exact checkpoint gap once**

Only if Stage6 improves failure count or minimum similarity, map Stage4 step1500 and Stage6 steps
250/500/1250/1500 to the five diagnostic slots. Bind every exact source checkpoint and run evidence, and run
the unchanged140-case evaluator. Do not derive embeddings or return to the same `central_q50` training
hypothesis if all slots are rejected.

- [ ] **Step 8: Run one duration-residual identity refinement**

Fit the precommitted one-variable OLS model on all2,223 pinned audit rows using `log1p(duration_seconds)`,
retain the1,667 rows at or above residual q25, preserve source order, and bind the exact coefficients/cutoff.
Warm-start from exact Stage6 step500, keep16 tokens/LR0.001/seed2/max1,500/save250, and run the unchanged
140-case evaluation. Do not repeat this residual-selection hypothesis if rejected.

- [ ] **Step 9: Run one duration-residual q50 identity refinement**

If Step8 is rejected without improving Stage6's failure count, mean, or minimum, keep the same pinned OLS
model but retain only the1,112 rows at or above residual median `0.006422475306645192`. Preserve source order,
bind the exact retained IDs and derived-manifest hashes, and warm-start again from exact Stage6 step500 rather
than chaining the rejected Step8 embedding. Keep16 tokens/LR0.001/seed2/max1,500/save250 and run the unchanged
140-case evaluation. Do not repeat this residual-q50 hypothesis if rejected.

- [ ] **Step 10: Run one low-rate local refinement from the q50 best checkpoint**

If Steps8–9 do not reduce Stage6's two similarity failures, stop changing data composition. Return to the exact
Stage6 step500 `F32[16,768]` embedding and its pinned absolute `central_q50` manifest. Change only the local
optimization rate to0.00035 while keeping16 tokens/seed2/max1,500/save250 and all other optimizer settings.
Evaluate steps250/500/750/1000/1500 with the unchanged140-case contract. Do not repeat this low-rate
continuation if rejected; consider an exact saved-checkpoint diagnostic only when the evaluated trajectory
improves Stage6 and leaves step1250 untested.

- [ ] **Step 11: Evaluate the low-rate exact checkpoint gap once**

Only if Step10 keeps Stage6's two similarity failures while improving both mean and minimum similarity, map
Stage6 step500 and Stage10 steps250/500/1250/1500 to the five diagnostic slots. Bind every exact source
checkpoint, run evidence file, decision, path, and SHA-256, then run the unchanged140-case evaluator once.
Do not average, interpolate, extrapolate, reshape, or otherwise derive embeddings. If all slots are rejected,
do not repeat the same low-rate continuation hypothesis.

- [ ] **Step 12: Run one Pareto-initialized q50 refinement**

If Step11 is rejected and the same two persistent cases remain, select the already evaluated exact checkpoint
that maximizes the lower similarity of those two cases. Bind the Stage3 q25-refinement step500 checkpoint and
its evaluation evidence, then refine it on the pinned 1,112-row absolute `central_q50` manifest with16 tokens,
LR0.00035, seed2, max1,500, and save250. Evaluate steps250/500/750/1000/1500 with the unchanged140-case
contract. This is a distinct initialization-basin test; do not chain the rejected Step10 checkpoint and do not
repeat the same initialization hypothesis if rejected.

- [ ] **Step 13: Run one absolute central-q75 identity-core refinement**

If Step12 is rejected and the pinned reference-centroid audit confirms heterogeneous references while the
existing absolute q25-to-q50 sequence improves the best failure count from three to two, retain exactly the
556 Stage0 rows at or above similarity q75 `0.8338747704317409`. Preserve source order and latent identity,
warm-start from exact Stage6 step500, and train with16 tokens, LR0.00035, seed2, max1,500, and save250.
Evaluate steps250/500/750/1000/1500 with the unchanged140-case contract. Do not further subdivide or repeat
the absolute-cutoff family if rejected.

- [ ] **Step 14: Run one q50 seed-order sensitivity refinement**

If Step13 is rejected and Stage10 remains the global best, bind the existing full-data LR0.0035 decisions
showing seed2 best failure count7 and seed7 best failure count6. Return to exact Stage6 step500 and the pinned
1,112-row absolute `central_q50` manifest. Keep16 tokens, LR0.00035, max1,500, save250, initialization, and all
other optimizer settings fixed; change only the training seed from2 to7. Evaluate
steps250/500/750/1000/1500 with the unchanged140-case contract. If rejected, stop this seed axis rather than
enumerating more seeds under the same q50/local-rate hypothesis.

- [ ] **Step 15: Evaluate the seed7 exact step-1250 gap once**

If Step14 remains rejected with two failures but its evaluated trajectory improves at least one global-best
mean/minimum statistic, map exact seed7 steps250/500/750/1250/1500 to the five diagnostic slots. Bind the
source config, setup, terminal, decision, checkpoint paths, and SHA-256 values. Re-run the unchanged140-case
contract; the four already evaluated slots act as deterministic controls and only step1250 adds information.
Do not derive or retrain an embedding, and do not return to the seed axis if the diagnostic is rejected.

- [ ] **Step 16: Audit a 24-reference robust centroid and precommit its q50 selection**

Bind the immutable reference-centroid audit and exclude exactly its sole robust outlier
`oop53_aibeya_sp_f7269f5ffc:00000594` from the data-selection centroid only. Re-embed the fixed2,223 training
audios with the same ECAPA model/revision, write complete create-only evidence, and select exactly the1,112
rows at or above the resulting median in clean-manifest order. Compare against absolute central-q50. Continue
to training only when Jaccard similarity is below0.98; never change the evaluator's25 references or threshold.

- [ ] **Step 17: Conditionally train one robust-centroid q50 refinement**

If Step16 passes its material-difference rule, build a create-only rebased manifest for the fixed1,112 IDs.
Warm-start from exact Stage10 step250 and keep16 tokens, LR0.00035, seed2, max1,500, and save250. Evaluate
steps250/500/750/1000/1500 under the unchanged140-case contract. Do not repeat this robust-centroid family if
rejected.

If the first run is interrupted solely by a bound progress-evidence atomic-replace PermissionError, retain that
FAIL root unchanged and launch one operational recovery in a new create-only root. The recovery must bind the
failed config/setup/terminal/log hashes, keep every training input and hyperparameter unchanged, and change only
the supervisor's finite retry handling for the transient Windows file lock. Do not use partial checkpoints from
the failed run as initialization or final evidence.

- [ ] **Step 18: Train one full-clean low-rate generalization recovery**

After the robust-centroid run is rejected with the same persistent two cases, stop that family. Return to the
immutable2,223-row clean manifest and exact Stage10 step250 rather than chaining a Stage17 checkpoint. Keep16
tokens, LR0.00035, seed2, max1,500, and save250. Evaluate steps250/500/750/1000/1500 under the unchanged
140-case contract. Treat this as one data-generalization trial and do not repeat it if rejected.

- [ ] **Step 19: Train once with the official v4-Small microbatch geometry**

After Step18 is rejected with the persistent cases, stop the data-selection family. Bind the pinned upstream
Speaker Inversion config SHA-256 and the per-microbatch stratified-timestep sampling implementation. Return to
the immutable Stage10 exact step250 and fixed1,112-row `central_q50` manifest. Keep16 tokens, LR0.00035,
seed2, max1,500, save250, data, initialization, and all optimizer settings fixed; change only from
batch4/accumulation4/checkpointing-off to the official batch16/accumulation1/checkpointing-on geometry. Evaluate
steps250/500/750/1000/1500 under the unchanged140-case contract. If it OOMs, fails, or is rejected, do not
enumerate intermediate batch sizes under this hypothesis.

If the first launch fails before spawning the training process because the shared supervisor's contract helper
hard-codes the legacy4/4/off geometry, preserve the complete FAIL root and bind its hashes. Retry once in a new
create-only root with a wrapper that replaces that helper by a strict16/1/on contract comparison while changing
no training input or hyperparameter. Treat this as operational recovery, not another geometry trial.

- [ ] **Step 20: Run one ultra-low-rate local continuation**

After Step19 is rejected and stopped, use the Stage10 trajectory evidence that its earliest evaluated checkpoint
is the global best while later checkpoints regress. Return to exact Stage10 step250 and fixed1,112-row
`central_q50`. Keep16 tokens, batch4/accumulation4/checkpointing-off, seed2, max1,500, save250, initialization,
data, and all other optimizer settings fixed; change only LR from0.00035 to0.0001. Evaluate
steps250/500/750/1000/1500 under the unchanged140-case contract. Do not enumerate still smaller rates if
rejected.

- [ ] **Step 21: Evaluate the ultra-low-rate exact step-1250 gap once**

If Step20 is rejected with the same two failures but an evaluated checkpoint improves both mean and minimum
over the Stage10 global best, map exact Stage20 steps250/500/750/1250/1500 to diagnostic slots
250/500/750/1000/1500. Bind source config, setup, terminal, decision, checkpoint paths, and SHA-256 values.
Only source step1250 adds information; the other four slots are determinism controls. Do not derive an embedding,
retrain, or return to the ultra-low-rate family if rejected.

- [ ] **Step 22: Run one aligned same-token convex-direction diagnostic**

After Step21 is rejected, bind exact Stage10 step250 and Stage12 step1500 plus their training/evaluation evidence.
Require all16 token rows to prefer the same-index parent by cosine. Precommit alpha0/0.25/0.5/0.75/1 and derive
only the three interior F32 embeddings with CPU elementwise arithmetic. Evaluate all five slots under the
unchanged140-case contract. Treat endpoints as controls and never promote a derived embedding. Continue to
retraining only if an interior point improves global-best failure count or the persistent pair; otherwise stop
this direction without alpha refinement or extrapolation.

- [ ] **Step 23: Train one Pareto-basin full-clean ultra-low-rate recovery**

After Step22's interior points fail to improve the global-best failure count, stop the convex direction. Use
exact Stage12 step1500, whose persistent pair is closest to threshold but which adds three other failures. Switch
to the immutable2,223-row full-clean manifest and LR0.0001 while keeping16 tokens, batch4/accumulation4/
checkpointing-off, seed2, max1,500, and save250. This combines the separately tested generalization and
ultra-low-rate mechanisms in the Pareto basin. Evaluate steps250/500/750/1000/1500 under the unchanged140-case
contract and do not repeat the combination if rejected.

- [ ] **Step 24: Train one token-weighted RF-loss sentence-identity refinement**

After Step23 is rejected, scan all immutable exact-checkpoint evaluation rows. Continue only if the persistent
`sentence_unko / seed1234 / neutral` hard case has never reached0.75 across all115 evaluated slots and all24
failure-count-2 frontier slots retain exactly the same two sentence failures. Warm-start from exact Stage20
step500, retain the fixed1,112-row `central_q50` manifest,
16 tokens, LR0.0001, batch4/accumulation4/checkpointing-off, seed2, max1,500, and save250. Change only
`rf_loss_mode` from `utterance_mean` to the upstream-supported token-weighted `echo` mode, plus output and
warm-start paths. Evaluate steps250/500/750/1000/1500 with the unchanged140-case contract. This tests whether
length-weighted training repairs sentence identity without target-text oversampling or inference changes. If
rejected, stop the loss-normalization axis rather than enumerating `echo` rate, seed, or data variants.

- [ ] **Step 25: Train one text-condition-dropout identity refinement**

After Step24 is rejected with the persistent pair, stop the loss-normalization axis. Return to exact Stage20
step500 and retain `central_q50`,16 tokens, LR0.0001, `utterance_mean`, batch4/accumulation4/
checkpointing-off, seed2, max1,500, and save250. Change only `text_condition_dropout` from0.0 to0.1 plus
output and warm-start paths. Keep caption, speaker, and duration condition dropouts at0.0. This tests the
upstream-supported per-sample conditioning path as a single identity/text disentanglement mechanism without
changing evaluation inference. Evaluate steps250/500/750/1000/1500 under the unchanged140-case contract. If
rejected, stop this axis without testing other probabilities or combining it with caption dropout.

- [ ] **Step 26: Train one cosine-decay drift-stabilization refinement**

After Step25 is rejected, stop the text-dropout axis. Bind the repeated early-checkpoint/late-regression
trajectories from Stages10,20,24,25 and the pinned upstream scheduler implementation. Warm-start from exact
Stage20 step500 and retain `central_q50`,16 tokens, base LR0.0001, `utterance_mean`, every condition dropout
at0.0, batch4/accumulation4/checkpointing-off, seed2, max1,500, and save250. Change only `lr_scheduler` from
`none` to upstream `cosine` plus output and warm-start paths; use its existing implicit warmup0 and minimum
LR scale0.1 defaults. Evaluate steps250/500/750/1000/1500 under the unchanged140-case contract. If rejected,
stop the scheduler axis without enumerating scheduler types, warmup lengths, or minimum scales.

### Task 10: Final verification and review packet

**Files:**
- Create remotely: final comparison/verification JSON and SHA256SUMS.
- Create locally: `/Users/sankenbisha/Downloads/irodori-v4-si-token-data-review`.

- [ ] **Step 1: Verify the winning140-case evidence from immutable inputs**

Recompute config, manifest, model, tokenizer, checkpoint, generation, metrics, evaluator, and script hashes.
Require16/16 speaker cases at least0.75 and all other hard gates PASS.

- [ ] **Step 2: Copy and verify the review packet**

Copy winning hard-gate WAVs and corresponding baseline WAVs plus a manifest to the Mac runtime-asset directory.
Recompute SHA-256 locally and require exact equality with remote evidence.

- [ ] **Step 3: Verify safety boundaries**

Confirm no train/supervisor process remains, GPU is released, v4 upstream tracked worktree is clean, and the
current v3 service/voice bank/deployment paths were not modified. Do not deploy.

- [ ] **Step 4: Mark the goal complete only after evidence passes**

If no candidate satisfies the objective, retain the goal active and continue with the next evidence-backed
candidate. If satisfied, mark the goal complete and report the final goal token usage returned by the goal tool.
