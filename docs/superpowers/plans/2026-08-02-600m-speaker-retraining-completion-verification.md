# 600M Speaker Retraining Completion Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only, fail-closed verifier that proves all 12 600M Speaker Inversion training and evaluation artifacts are complete, reviewed, quiescent, and not deployed.

**Architecture:** A single standalone script loads immutable component artifacts, validates each boundary independently, and emits one atomic versioned completion report. Pure validation functions accept paths and normalized runtime snapshots so unit tests cover the contract without SSH, GPU, model weights, or service changes.

**Tech Stack:** Python 3.11+, standard library, NumPy, pytest, existing JSON/JSONL and safetensors wire formats.

---

## File Map

- Create `scripts/verify_600m_speaker_retraining_completion.py`: CLI, training/evaluation/review/staging validators, live read-only runtime probe, and atomic report writer.
- Create `tests/scripts/test_verify_600m_speaker_retraining_completion.py`: deterministic fixtures and contract-focused unit tests.
- Do not modify existing scripts, remote artifacts, statuses, voice-bank files, services, `justfile`, or package modules.

### Task 1: Training inventory, embedding, and log gate

**Files:**
- Create: `tests/scripts/test_verify_600m_speaker_retraining_completion.py`
- Create: `scripts/verify_600m_speaker_retraining_completion.py`

- [ ] **Step 1: Write the failing happy-path training test**

Build 12 temporary jobs with configs, manifests, logs, periodic checkpoints at
250 through 3000, and a byte-identical final checkpoint. Give Anabel 12 status
candidates and the other models 13. Assert `verify_training` returns 12
models, 13 files per model, 150 finite loss events, and exact SHA bindings.

```python
result = module.verify_training(
    training_jobs=fixture.jobs,
    training_status=fixture.status,
    training_launch_evidence=fixture.launch,
    runtime_snapshot=module.RuntimeSnapshot.idle(used_mib=973.0),
    gpu_memory_tolerance_mib=256.0,
)
assert len(result.models) == 12
assert all(row.checkpoint_count == 13 for row in result.models)
assert all(row.loss_event_count == 150 for row in result.models)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_verify_600m_speaker_retraining_completion.py::test_training_gate_accepts_exact_complete_run -q
```

Expected: FAIL because the verifier module does not exist.

- [ ] **Step 3: Implement the minimal training gate**

Add immutable result dataclasses and focused helpers named `verify_training`,
`validate_speaker_embedding`, and `parse_final_training_run`. Lock the training
constants before implementing the readers:

```python
EXPECTED_MODEL_COUNT = 12
PERIODIC_STEPS = tuple(range(250, 3001, 250))
LOGGED_STEPS = tuple(range(20, 3001, 20))
EXPECTED_EMBEDDING_SHAPE = (16, 768)
EXPECTED_CHECKPOINT_NAMES = frozenset(
    {f"checkpoint_{step:07d}.speaker.safetensors" for step in PERIODIC_STEPS}
    | {"checkpoint_final.speaker.safetensors"}
)
```

Validate exact job/current-status provenance, the corrected 12-or-13 status
candidate rule, exact 13 on-disk files, final/3000 hash equality, config fields,
finite loss suffix, launcher evidence, and runtime snapshot.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Add one failing test per training boundary**

Parameterize mutations for missing/extra checkpoint, wrong shape/dtype/nonfinite
payload, final/3000 mismatch, missing periodic status candidate, unrelated status
candidate, stale provenance SHA, later running status, missing/nonfinite loss,
incomplete step sequence, launcher failure, residual process, compute app, probe
error, and unreleased GPU memory. Add dedicated process-inventory cases proving
that the diagnostic command and its ancestors are excluded, a reused launcher
PID is not treated as ownership by itself, creation time disambiguates a
launcher record when present, and a semantically matching worker is still
rejected even with a different PID.

```python
@pytest.mark.parametrize("mutation,match", TRAINING_FAILURES)
def test_training_gate_fails_closed(fixture, mutation, match):
    mutation(fixture)
    with pytest.raises(ValueError, match=match):
        _verify_training(module, fixture)
```

- [ ] **Step 6: Run the new failures and confirm RED**

Run the full new test file. Expected: the new boundary cases fail for missing
validation, not fixture errors.

- [ ] **Step 7: Implement minimal boundary validation and confirm GREEN**

Complete strict readers, path containment, SHA validation, ANSI stripping,
final-run suffix selection, candidate subset validation, launcher binding, and
runtime thresholds. Match process command semantics after excluding the current
process, its ancestors, and diagnostic invocations; never classify by PID
alone. Re-run the full new test file; expected PASS.

### Task 2: Evaluation queue and per-model artifact gate

**Files:**
- Modify: `tests/scripts/test_verify_600m_speaker_retraining_completion.py`
- Modify: `scripts/verify_600m_speaker_retraining_completion.py`

- [ ] **Step 1: Write a failing 12-model evaluation happy-path test**

Create one manifests stage and four successful stages per model, with current
output snapshots. Create exact five-checkpoint, 140-case manifests/evaluation
results and v2 verification documents. Assert 49 stages, 12 models, and one
selection per model are returned.

```python
result = module.verify_evaluations(
    evaluation_config=fixture.evaluation_config,
    evaluation_status=fixture.evaluation_status,
    training=training_result,
)
assert result.stage_count == 49
assert len(result.models) == 12
assert all(row.case_count == 140 for row in result.models)
```

- [ ] **Step 2: Run the test and confirm RED**

Expected: FAIL because `verify_evaluations` is absent.

- [ ] **Step 3: Implement the minimal evaluation validator**

Add `verify_evaluations`, `snapshot_path`, and `validate_case_matrix`. Lock the
evaluation constants before implementing the strict readers:

```python
EXPECTED_EVALUATION_STAGES_PER_MODEL = (
    "generation",
    "analysis",
    "metrics",
    "evaluate",
)
EXPECTED_EVALUATION_STAGE_COUNT = 1 + EXPECTED_MODEL_COUNT * len(
    EXPECTED_EVALUATION_STAGES_PER_MODEL
)
EXPECTED_EVALUATION_STEPS = tuple(range(250, 3001, 250))
EXPECTED_TEXT_IDS = (
    "word_unko", "word_chinko", "word_manko",
    "sentence_unko", "sentence_chinko", "sentence_manko", "control",
)
EXPECTED_SEEDS = (1234, 5678)
EXPECTED_STYLES = ("neutral", "calm")
```

Require exact model/stage sets, current config/component/output bindings, absent
lock, v2 PASS artifacts, exact 140-case matrix, selected-manifest-training
identity, and current artifact hashes.

- [ ] **Step 4: Run the happy path and confirm GREEN**

Run the focused test. Expected: PASS.

- [ ] **Step 5: Add and verify fail-closed evaluation tests**

Cover missing/duplicate/stale stage, changed output snapshot, live lock, missing
control/nko case, duplicate case, wrong checkpoint set, non-PASS verification,
two selections, changed selected embedding, stale artifact hash, and mismatched
training provenance. Observe RED, implement the smallest checks, then observe
GREEN for the full file.

### Task 3: Human review decision gate

**Files:**
- Modify: `tests/scripts/test_verify_600m_speaker_retraining_completion.py`
- Modify: `scripts/verify_600m_speaker_retraining_completion.py`

- [ ] **Step 1: Write failing review-state tests**

Test the three required outcomes:

```python
assert module.verify_reviews(
    evaluations=evaluation_result,
    decisions_path=fixture.empty_decisions,
).status == "AWAITING_REVIEW"
assert module.verify_reviews(
    evaluations=evaluation_result,
    decisions_path=fixture.all_voice_decisions,
).status == "PASS"
with pytest.raises(ValueError, match="selected checkpoint"):
    module.verify_reviews(
        evaluations=evaluation_result,
        decisions_path=fixture.selected_tone_decision,
    )
```

Also assert non-selected `TONE` decisions remain grouped in report summaries.

- [ ] **Step 2: Run the review tests and confirm RED**

Expected: FAIL because the review contract is absent.

- [ ] **Step 3: Implement strict one-to-one decision binding**

Add `verify_reviews` and lock the decision contract before implementing the
one-to-one join:

```python
REVIEW_DECISION_SCHEMA = "speaker-checkpoint-review-decision/v1"
REVIEW_DECISIONS = frozenset({"VOICE", "TONE", "UNSURE"})
REVIEW_IDENTITY_FIELDS = (
    "case_id",
    "model_id",
    "checkpoint_step",
    "wav_sha256",
)
```

Validate schema, case/model/step/WAV identity, timezone-aware timestamp,
reviewer, enum, no duplicates/extras, packet manifest and copied asset hashes.
Return `AWAITING_REVIEW` only for otherwise-valid missing decisions. Reject
selected `TONE`/`UNSURE`; allow `VOICE`; preserve non-selected counts.

- [ ] **Step 4: Add invalid-decision tests and reach GREEN**

Cover stale WAV hash, duplicate/extra decision, invalid enum, naive timestamp,
missing packet asset, and changed copied asset. Run the full new test file until
all cases pass.

### Task 4: Staging, report, and CLI closure

**Files:**
- Modify: `tests/scripts/test_verify_600m_speaker_retraining_completion.py`
- Modify: `scripts/verify_600m_speaker_retraining_completion.py`

- [ ] **Step 1: Write failing staging and report tests**

Assert the final phase accepts a v1 staging report with exact selections,
unchanged baseline/current voice-bank hashes, false deployment flags, and absent
staging root. Assert it rejects every flag inversion, changed voice bank,
selection mismatch, and existing staging root.

- [ ] **Step 2: Run the tests and confirm RED**

Expected: FAIL because staging and top-level orchestration are absent.

- [ ] **Step 3: Implement staging validation and top-level report**

Add `verify_staging`, `build_completion_report`, and
`write_report_create_only`. Lock the report and staging contracts before
implementing orchestration:

```python
COMPLETION_SCHEMA = "600m-speaker-retraining-completion-verification/v1"
STAGING_SCHEMA = "speaker-model-staging-report/v1"
COMPLETION_STATUSES = frozenset({"PASS", "AWAITING_REVIEW", "FAIL"})
REQUIRED_NON_DEPLOYMENT_VALUES = {
    "deployment_performed": False,
    "active_voice_bank_unchanged": True,
    "proposed_staging_root_created": False,
}
```

Emit schema `600m-speaker-retraining-completion-verification/v1` with `PASS`,
`AWAITING_REVIEW`, or `FAIL`, all input/artifact hashes, runtime snapshot,
model summaries, decisions, non-deployment evidence, and verifier SHA.
Publish through an exclusively created temporary file and an atomic
no-overwrite operation so a concurrent writer cannot be replaced.

- [ ] **Step 4: Implement and test the CLI**

Inject the live probe into `main` for unit tests. Confirm training phase requires
only training inputs, final phase requires every final input, exit zero only on
PASS, atomic create-only behavior, no partial temporary file, and no input
mtime/hash changes.

- [ ] **Step 5: Run focused verification**

```bash
uv run pytest --no-cov tests/scripts/test_verify_600m_speaker_retraining_completion.py -q
uv run ruff check scripts/verify_600m_speaker_retraining_completion.py tests/scripts/test_verify_600m_speaker_retraining_completion.py
uv run ruff format --check scripts/verify_600m_speaker_retraining_completion.py tests/scripts/test_verify_600m_speaker_retraining_completion.py
```

Expected: all commands PASS.

### Task 5: Regression and completion verification

**Files:**
- Verify only; do not commit or modify unrelated files.

- [ ] **Step 1: Run adjacent script tests**

```bash
uv run pytest --no-cov \
  tests/scripts/test_run_600m_speaker_training_queue.py \
  tests/scripts/test_build_600m_checkpoint_evaluation_manifests.py \
  tests/scripts/test_run_600m_speaker_evaluation_queue.py \
  tests/scripts/test_evaluate_600m_speaker_checkpoints.py \
  tests/scripts/test_build_600m_speaker_staging_report.py \
  tests/scripts/test_verify_600m_speaker_retraining_completion.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the repository gate when practical**

```bash
just check
```

Expected: PASS, or report an exact unrelated dirty-tree/external-runtime blocker.

- [ ] **Step 3: Review the diff and verify ownership**

```bash
git diff --check
git status --short
git diff -- \
  docs/superpowers/specs/2026-08-02-600m-speaker-retraining-completion-verification-design.md \
  docs/superpowers/plans/2026-08-02-600m-speaker-retraining-completion-verification.md \
  scripts/verify_600m_speaker_retraining_completion.py \
  tests/scripts/test_verify_600m_speaker_retraining_completion.py
```

Expected: only the four owned files contain this task's changes. Do not commit.

## Self-Review

- Spec coverage: training provenance, corrected 12-or-13 status candidate rule,
  exact 13 files, final/3000 equality, finite losses, queue/runtime evidence,
  exact evaluation stages/matrix, human review resolution, and non-deployment
  evidence each map to an implementation task.
- Placeholder scan: no deferred implementation step or unspecified error-handling
  instruction remains.
- Type consistency: the result names and public validation entry points used by
  later tasks are defined in the task where they are introduced.
- Scope: no existing component contract or remote artifact is modified.
