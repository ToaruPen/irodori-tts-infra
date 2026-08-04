# Irodori-TTS v4 Speaker Inversion Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v4-Small 用 Speaker Inversion を learning rate と seed の段階探索で再訓練し、既存の 16 hard-gate case をすべて speaker similarity 0.75 以上にする。

**Architecture:** 現行 v4 pilot を immutable baseline として残し、新しい create-only root に候補を直列作成する。各候補は学習 supervisor、140-case 評価 supervisor、deterministic evaluator の証跡で閉じ、合格候補が出た時点で後続 GPU 作業を停止する。v3 service、voice bank、v4 base model は変更しない。

**Tech Stack:** Python 3.11、Irodori-TTS v4-Small、PyTorch CUDA/bf16、Speaker Inversion、SpeechBrain ECAPA-TDNN、Whisper large-v3-turbo、PowerShell 5.1、Windows OpenSSH

---

## File and artifact map

- Design: `docs/superpowers/specs/2026-08-03-irodori-v4-speaker-inversion-retraining-design.md`
- Plan: `docs/superpowers/plans/2026-08-03-irodori-v4-speaker-inversion-retraining.md`
- Existing remote baseline:
  `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_pilot_v002`
- New remote search root:
  `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_retraining_v001`
- Remote training supervisor:
  `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools\run_irodori_v4_si_retraining_supervisor.py`
- Existing remote evaluation preparer:
  `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools\prepare_irodori_v4_evaluation.py`
- Remote v4-only evaluation supervisor:
  `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools\run_irodori_v4_candidate_evaluation_supervisor.py`
- Local human-review assets:
  `/Users/sankenbisha/Downloads/irodori-v4-si-retraining-review`

### Task 1: Freeze baseline and preflight the remote host

**Files:**
- Read: remote baseline `setup-evidence.json`, `terminal-evidence.json`, `config.yaml`
- Create: remote search root `search-setup-evidence.json`

- [ ] **Step 1: Verify the pinned inputs**

Run a read-only PowerShell check over SSH that verifies:

```text
upstream commit = 8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71
model SHA-256 = 5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
tokenizer.json SHA-256 = 6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d
tokenizer_config.json SHA-256 = d229a271c64de1a7939d20d3665498e873fa91d5ee2edf135d73ec752cb9c9d3
manifest SHA-256 = 6fd6f8755a74130bec0ca985da45ef05841fcf047e788062888cc64d0a5f89dd
baseline final SHA-256 = 3dd6f4eef078e4a3d41e46f8f05603c42538fd555ba4c1da06dda32bd30cbb41
```

Expected: all six bindings match and the upstream tracked worktree is clean.

- [ ] **Step 2: Verify runtime availability without stopping anything**

Run `nvidia-smi` and query `Win32_Process`. Classify command lines containing
`train.py` plus `Irodori-TTS-v4-8ca3acb` as training and command lines containing
`remote_server.py`, `uvicorn`, `irodori_tts_infra.server`, or `irodori-tts-server`
as conflicting services. Expected:

```text
free GPU memory >= 10500 MiB
active training process count = 0
conflicting service process count = 0
```

If a service or training process is active, do not stop it; leave the search unstarted and report the process evidence.

- [ ] **Step 3: Reserve the create-only search root**

Create the root only if it does not exist. Write `search-setup-evidence.json` with the baseline bindings, candidate order, success threshold `0.75`, hard-gate count `16`, and `deployment_performed: false` using exclusive creation.

Expected: later commands refuse to reuse an existing search root.

### Task 2: Parameterize the fail-closed training supervisor

**Files:**
- Read: remote `_tools/run_irodori_v4_si_pilot_supervisor.py`
- Create: remote `_tools/run_irodori_v4_si_retraining_supervisor.py`
- Create: disposable remote contract-test roots under `v4_speaker_inversion_oop53_retraining_v001/contract-tests/`

- [ ] **Step 1: Add learning-rate and seed bindings**

Start from the immutable pilot supervisor and change the candidate contract to require these fields in both `setup-evidence.json` and `config.yaml`:

```python
expected_contract = {
    "fresh_speaker_inversion": True,
    "speaker_inversion_tokens": 16,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": False,
    "learning_rate": expected_learning_rate,
    "max_steps": 3000,
    "save_every": 250,
    "log_every": 20,
    "seed": expected_seed,
}
```

The script arguments must be:

```text
run_root python upstream manifest checkpoint expected_learning_rate expected_seed
```

Use schema names `irodori-v4-si-retraining-*/v1`. Preserve exclusive writes, asset hashes, tracked-worktree validation, process refusal, minimum free VRAM, `PYTHONDONTWRITEBYTECODE=1`, checkpoint-set validation, final/step3000 hash equality, and GPU release evidence.

- [ ] **Step 2: Run syntax validation**

Run:

```powershell
& 'C:\Users\takut\Dev\Irodori-TTS-v4-8ca3acb\.venv\Scripts\python.exe' `
  -m py_compile `
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools\run_irodori_v4_si_retraining_supervisor.py'
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Verify fail-closed mismatch behavior**

Create a disposable setup whose evidence declares learning rate `0.0035` while its config contains `0.01`. Run the supervisor in the foreground.

Expected: non-zero exit before `training/` or `training.log` is created, with terminal evidence containing `config contract mismatch`.

- [ ] **Step 4: Verify allowed candidate contracts**

Load the three planned candidate documents without launching training and assert the parsed contracts are exactly:

```python
[
    {"learning_rate": 0.0035, "seed": 2},
    {"learning_rate": 0.0035, "seed": 7},
    {"learning_rate": 0.001, "seed": 2},
]
```

Expected: all other train/model fields equal the baseline config except `output_dir`.

### Task 3: Train candidate `lr0035_seed2`

**Files:**
- Create: remote `candidates/lr0035_seed2/config.yaml`
- Create: remote `candidates/lr0035_seed2/setup-evidence.json`
- Create: remote `candidates/lr0035_seed2/{training/,training.log,*evidence.json}`

- [ ] **Step 1: Prepare the candidate config**

Copy the baseline YAML in memory, then set only:

```yaml
train:
  learning_rate: 0.0035
  seed: 2
  output_dir: C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_retraining_v001\candidates\lr0035_seed2\training
```

Write the config and setup evidence exclusively. Bind the config, manifest, model, tokenizer, upstream commit, expected learning rate, and seed by SHA-256 or exact value.

- [ ] **Step 2: Verify the config diff**

Parse the baseline and candidate YAML and recursively list unequal leaf fields.

Expected exactly:

```text
train.learning_rate
train.output_dir
```

`train.seed` remains `2` and therefore must not appear in the diff.

- [ ] **Step 3: Launch detached training**

Start the retraining supervisor through `Start-Process` with redirected stdout/stderr and write reservation/parent-handoff evidence before returning.

Expected: the supervisor owns exactly one `train.py` child, creates `supervisor-start-evidence.json`, and reports step progress every 30 seconds.

- [ ] **Step 4: Monitor to a terminal state**

Poll `progress-evidence.json` without creating an automation or scheduler. Do not interrupt the process while progress advances.

Expected terminal PASS:

```text
latest_logged_step = 3000
checkpoint count = 13
checkpoint_0003000 SHA-256 = checkpoint_final SHA-256
exit_code = 0
no OOM or traceback
GPU utilization returns to idle
```

If training fails, retain its evidence and create a versioned retry root only for an operational failure. A completed-but-low-quality candidate proceeds to evaluation and is not retried in place.

### Task 4: Evaluate one v4 candidate with the fixed 140-case matrix

**Files:**
- Read: remote `_tools/prepare_irodori_v4_evaluation.py`
- Create: remote `_tools/run_irodori_v4_candidate_evaluation_supervisor.py`
- Create: remote `evaluations/lr0035_seed2/v4/{manifest,runtime,generation,analysis,metrics,selection}`
- Conditionally create: remote `evaluations/lr0035_seed7/v4/{manifest,runtime,generation,analysis,metrics,selection}`
- Conditionally create: remote `evaluations/lr001_seed2/v4/{manifest,runtime,generation,analysis,metrics,selection}`

- [ ] **Step 1: Build a v4-only evaluation supervisor**

Reuse the pinned commands from the completed comparison supervisor, but accept `evaluation_root` and `training_root` as arguments and execute only:

```python
stages = [
    "v4_preflight",
    "v4_generation",
    "v4_analysis",
    "v4_metrics",
    "v4_evaluate",
]
```

The evaluator exit codes `{0, 1}` are both operationally valid; exit `1` means no checkpoint was eligible, not a pipeline crash. The terminal status is `COMPLETE` only when generation verification passes, 140 successful cases exist, five checkpoint summaries exist, all stages have allowed exit codes, and the GPU is released.

- [ ] **Step 2: Syntax-check and fail-closed test the evaluator**

Run `python -m py_compile`. Then invoke it against a disposable evaluation root whose training terminal status is `FAIL`.

Expected: no generation starts and terminal evidence records `training terminal evidence is not a PASS`.

- [ ] **Step 3: Prepare and launch candidate evaluation**

Run `prepare_irodori_v4_evaluation.py` with these exact arguments:

```text
C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_retraining_v001\candidates\lr0035_seed2
C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_retraining_v001\evaluations\lr0035_seed2
C:\Users\takut\Dev\Irodori-TTS-v4-8ca3acb
D:\hf_cache\hub\models--Aratako--Irodori-TTS-v4-Small\snapshots\e4aaac4df355ff560dcd35e0dae272c3a759317b\model.safetensors
C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v3_v4_oop53_comparison_v001\v3_quality\manifest\evaluation-manifest.json
C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v3_v4_oop53_comparison_v001\v4\runtime\generate_v4_checkpoint_audio_remote.py
```

Expected setup evidence:

```text
checkpoints = [1000, 1500, 2000, 2500, 3000]
text IDs = 7
seeds = [1234, 5678]
styles = [neutral, calm]
case_count = 140
deployment_performed = false
```

Launch the v4-only evaluation supervisor detached and monitor its stage evidence to `COMPLETE`.

- [ ] **Step 4: Make the deterministic candidate decision**

Read `selection/selected-models.json`, `checkpoint-summary.jsonl`, and `evaluation-results.jsonl`. A candidate passes only if at least one selection exists and its 16 metric-gate rows all satisfy:

```python
speaker_similarity >= 0.75
```

Also require all other evaluator hard gates to pass. Write a create-only `candidate-decision.json` containing the selected checkpoint step and SHA-256, minimum/mean similarity, CER, style contrast, case counts, review packet binding, and `deployment_performed: false`.

### Task 5: Continue the conditional search only when required

**Files:**
- Conditionally create: remote `candidates/lr0035_seed7` and its evaluation
- Conditionally create: remote `candidates/lr001_seed2` and its evaluation
- Create: remote root `search-decision.json`

- [ ] **Step 1: Stop after an eligible candidate**

If `lr0035_seed2` passes, do not create later candidate roots. Record `stop_reason: eligible_candidate_found` and the selected binding in `search-decision.json`.

- [ ] **Step 2: Otherwise train and evaluate `lr0035_seed7`**

Create `candidates/lr0035_seed7/config.yaml` from the baseline with exactly these
three changed leaves:

```yaml
train:
  learning_rate: 0.0035
  seed: 7
  output_dir: C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_retraining_v001\candidates\lr0035_seed7\training
```

Write hash-bound setup evidence, launch the retraining supervisor with expected
learning rate `0.0035` and seed `7`, and require the same 13-checkpoint terminal
PASS contract. Prepare `evaluations/lr0035_seed7` using the pinned v3 manifest and
v4 generator paths listed in Task 4, launch the five-stage v4-only evaluation, and
write `candidate-decision.json`. Stop if `selected-models.json` contains a selection
whose 16 metric-gate rows all have speaker similarity at least `0.75` and whose
other hard gates pass.

- [ ] **Step 3: Otherwise train and evaluate `lr001_seed2`**

Create `candidates/lr001_seed2/config.yaml` from the baseline with exactly these
two changed leaves:

```yaml
train:
  learning_rate: 0.001
  seed: 2
  output_dir: C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_retraining_v001\candidates\lr001_seed2\training
```

Write hash-bound setup evidence, launch the retraining supervisor with expected
learning rate `0.001` and seed `2`, and require the 13-checkpoint terminal PASS
contract. Prepare `evaluations/lr001_seed2`, run the five-stage evaluation, and
write its deterministic candidate decision. After this decision, stop regardless
of outcome. If all candidates fail, record
`stop_reason: stage_one_candidates_exhausted`; do not change token count or dataset
under this plan.

### Task 6: Materialize review audio and final safety evidence

**Files:**
- Conditionally create: `/Users/sankenbisha/Downloads/irodori-v4-si-retraining-review/lr0035_seed2/`
- Conditionally create: `/Users/sankenbisha/Downloads/irodori-v4-si-retraining-review/lr0035_seed7/`
- Conditionally create: `/Users/sankenbisha/Downloads/irodori-v4-si-retraining-review/lr001_seed2/`
- Create: remote root `final-verification.json`

- [ ] **Step 1: Verify the selected artifacts**

Rehash the selected `.speaker.safetensors`, evaluation manifest, metrics provenance, checkpoint summary, selected models document, and review packet manifest. Compare every hash with its evidence binding.

Expected: all bindings match and no selected input changed after evaluation.

- [ ] **Step 2: Copy a matched listening packet to the Mac**

Copy the selected candidate's review WAVs plus the six corresponding baseline v4 WAVs into separate `candidate/` and `baseline/` directories. Preserve descriptive case names and verify source/destination SHA-256 equality.

Expected: local WAV files are 48 kHz, mono, 16-bit PCM and remain outside the repository.

- [ ] **Step 3: Verify deployment state is unchanged**

Capture the active service process, deployed voice-bank binding, standard v3 configuration, and GPU state. Write `final-verification.json` with:

```json
{
  "deployment_performed": false,
  "active_voice_bank_unchanged": true,
  "standard_v3_configuration_unchanged": true
}
```

Expected: no service restart, voice-bank replacement, or v4 deployment occurred.

- [ ] **Step 4: Run local document checks**

Run:

```bash
git diff --check -- \
  docs/superpowers/specs/2026-08-03-irodori-v4-speaker-inversion-retraining-design.md \
  docs/superpowers/plans/2026-08-03-irodori-v4-speaker-inversion-retraining.md
```

Expected: exit code 0. Do not stage or commit files unless the user explicitly requests it.
