# 600M VoiceDesign Speaker Inversion Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build maximally retained, beep-free training datasets from the original recordings and retrain 12 Speaker Inversion embeddings against the 600M VoiceDesign checkpoint.

**Architecture:** Keep source recordings and deployed embeddings immutable. A deterministic, label-calibrated audio audit classifies every unique recording, repairs only unambiguous censored captions, and emits reproducible clean datasets. Generate a 600M-compatible Speaker Inversion config from checkpoint metadata, pilot it on one voice, then train and evaluate 12 isolated outputs before any deployment replacement.

**Tech Stack:** Python 3.11, NumPy, soundfile, safetensors, Irodori-TTS upstream training runtime, pytest, ruff, mypy, Windows PowerShell over SSH, NVIDIA RTX 4070.

**Repository policy:** Do not commit, stage, or overwrite deployed model assets unless the user explicitly requests it. Generated audio, datasets, latents, and model weights remain outside Git.

---

### Task 1: Add expressive-safe audio classification

**Files:**
- Create: `scripts/speaker_dataset_quality.py`
- Create: `tests/scripts/test_speaker_dataset_quality.py`
- Reuse: `scripts/analyze_nko_beep_matrix.py`

- [x] **Step 1: Write failing synthetic-signal tests**

Add tests that exercise real NumPy samples:

```python
def test_classify_audio_excludes_matching_confirmed_tone() -> None:
    samples = insert_tone(frequency_hz=1_000.0, duration_seconds=0.25)
    result = module.classify_audio(
        samples,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(confirmed_1khz_signature("dataset-a"),),
    )
    assert result.decision == "EXCLUDE_CONFIRMED_TONE"


def test_classify_audio_keeps_broadband_breath() -> None:
    result = module.classify_audio(seed_breath_noise(), SAMPLE_RATE)
    assert result.decision == "KEEP"


def test_classify_audio_does_not_exclude_harmonic_vocalization() -> None:
    result = module.classify_audio(harmonic_vocalization(220.0), SAMPLE_RATE)
    assert result.decision in {"KEEP", "REVIEW"}


def test_classify_audio_reviews_low_level_nonzero_audio() -> None:
    result = module.classify_audio(low_level_breath(), SAMPLE_RATE)
    assert result.decision == "REVIEW"


def test_classify_audio_rejects_all_zero_audio() -> None:
    result = module.classify_audio(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert result.decision == "EXCLUDE_INVALID_AUDIO"
```

Cover 703.125Hz, 1kHz, and 12kHz beeps, clipped vocalization, short moans, and a fading pure tone.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_speaker_dataset_quality.py -q
```

Expected: collection or assertion failure because `speaker_dataset_quality.py` and its API do not exist.

- [x] **Step 3: Implement the minimal classifier**

Define immutable result types and derive the decision from the broad-band intervals plus voice-protection features:

```python
Decision = Literal[
    "KEEP",
    "KEEP_RECAPTIONED",
    "REVIEW",
    "EXCLUDE_CONFIRMED_TONE",
    "EXCLUDE_INVALID_AUDIO",
    "EXCLUDE_TRANSCRIPT_MISMATCH",
    "EXCLUDE_DUPLICATE",
]


@dataclass(frozen=True, slots=True)
class AudioDecision:
    decision: Decision
    reasons: tuple[str, ...]
    intervals: tuple[ToneInterval, ...]
    harmonic_ratio: float
    clipped_fraction: float
    rms_dbfs: float
```

`EXCLUDE_CONFIRMED_TONE` must require a stable pure-tone interval, weak harmonic evidence, and a match to a human-confirmed signature scoped to the same source dataset. An unregistered pure tone at any frequency remains `REVIEW`. Broadband breath and a harmonic series must never be auto-excluded. Nonzero low-level audio and clipping go to `REVIEW`; only empty/all-zero/non-finite audio is automatically invalid.

- [x] **Step 4: Run GREEN and style checks**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_speaker_dataset_quality.py -q
uv run ruff check scripts/speaker_dataset_quality.py tests/scripts/test_speaker_dataset_quality.py
uv run ruff format --check scripts/speaker_dataset_quality.py tests/scripts/test_speaker_dataset_quality.py
```

Expected: all pass.

### Task 2: Add caption repair and label overrides

**Files:**
- Modify: `scripts/speaker_dataset_quality.py`
- Modify: `tests/scripts/test_speaker_dataset_quality.py`

- [x] **Step 1: Write failing caption and label tests**

```python
def test_apply_label_override_restores_voice_candidate() -> None:
    result = module.apply_label_override(
        automatic="REVIEW",
        label=module.ReviewLabel(label="VOICE", reviewer="user", note="breath"),
    )
    assert result == "KEEP"


def test_apply_label_override_excludes_confirmed_beep() -> None:
    result = module.apply_label_override(
        automatic="REVIEW",
        label=module.ReviewLabel(label="TONE", reviewer="user", note="fixed tone"),
    )
    assert result == "EXCLUDE_CONFIRMED_TONE"


def test_apply_label_override_excludes_automatic_keep_labeled_as_tone() -> None:
    result = module.apply_label_override(
        automatic="KEEP",
        label=module.ReviewLabel(label="TONE", reviewer="user", note="listening review"),
    )
    assert result == "EXCLUDE_CONFIRMED_TONE"


def test_repair_caption_uses_only_explicit_rule() -> None:
    repaired = module.repair_caption(
        "おち◯ちんです",
        rules=(module.CaptionRule(source="おち◯ちん", replacement="おちんちん"),),
    )
    assert repaired.text == "おちんちんです"
    assert repaired.decision == "REPAIRED"


def test_repair_caption_reviews_unknown_marker() -> None:
    repaired = module.repair_caption("未知◯語", rules=())
    assert repaired.decision == "REVIEW"
```

- [x] **Step 2: Verify RED**

Run the narrow test file and confirm the missing APIs fail.

- [x] **Step 3: Implement explicit, auditable repair rules**

Add `ReviewLabel`, `CaptionRule`, and `CaptionRepair`. Never guess an unknown marker replacement. Persist the original text, repaired text, rule id, and reviewer label in serialized output.

- [x] **Step 4: Verify GREEN**

Run the narrow tests and ruff checks again.

### Task 3: Build reproducible source inventories and clean datasets

**Files:**
- Create: `scripts/build_clean_speaker_datasets.py`
- Create: `tests/scripts/test_build_clean_speaker_datasets.py`
- Modify: `scripts/audit_training_tones.py`

- [x] **Step 1: Write failing inventory tests**

Use temporary `index.json` files and tiny WAV fixtures to assert:

```python
def test_inventory_hashes_unique_audio_once(tmp_path: Path) -> None:
    rows = module.build_inventory(catalog_with_duplicate_paths(tmp_path))
    assert [row.decision for row in rows].count("EXCLUDE_DUPLICATE") == 1


def test_clean_dataset_keeps_expressive_voice_and_repairs_caption(tmp_path: Path) -> None:
    result = module.build_clean_dataset(
        catalog=expressive_catalog(tmp_path),
        labels=voice_labels(),
        caption_rules=known_rules(),
    )
    assert result.summary["kept"] == 2
    assert result.rows[0]["text"] == "はぁ、んっ"


def test_clean_dataset_never_emits_review_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unresolved REVIEW"):
        module.write_clean_dataset(unresolved_result(tmp_path))
```

Also assert that every decision has a rule version and reasons, and that source paths remain unchanged.

- [x] **Step 2: Verify RED**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_build_clean_speaker_datasets.py -q
```

- [x] **Step 3: Implement inventory and output writers**

The CLI accepts:

```text
--catalog-json PATH
--labels-jsonl PATH
--caption-rules-json PATH
--output-root PATH
--progress-every N
```

For each dataset write `source-inventory.jsonl`, `decisions.jsonl`, `review-candidates.jsonl`, `clean-dataset.jsonl`, and `summary.json`. Candidate WAV conversion happens only for `REVIEW` rows. `clean-dataset.jsonl` must not be emitted while an unresolved `REVIEW` exists.

- [x] **Step 4: Verify GREEN and run all script tests**

Run:

```bash
uv run pytest --no-cov tests/scripts -q
uv run ruff check scripts tests/scripts
uv run ruff format --check scripts tests/scripts
```

### Task 4: Inventory all 12 source datasets on Windows

**Files:**
- Generate outside Git: `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\source\catalog.json`
- Generate outside Git: per-dataset inventories under the same root

- [x] **Step 1: Resolve the source mapping**

Map the ten OOPPEENN source collections, Kasumi's original 772-row source, and Sayoko's original 2,469-row source. Map the OOP55 collection to output id `miu`; do not create a separate Ai output.

- [x] **Step 2: Record immutable source hashes and counts**

For each `index.json`, record SHA-256, source speaker name, row count, audio root, previous manifest, and previous embedding. Confirm all referenced audio exists before scanning.

- [x] **Step 3: Copy only versioned scripts to an isolated remote work directory**

Use `scp` to copy the tested scripts. Do not modify the running service checkout or current voice bank.

- [x] **Step 4: Run the first-pass broad audit**

Use the runtime Python with `soundfile` and save progress after each record. On Windows,
pin every subsequent audit, manifest, and training command through
`uv run --project C:\Users\takut\Dev\Irodori-TTS --no-sync --python C:\Users\takut\Dev\Irodori-TTS\.venv`;
do not rely on the PowerShell `python` resolution or synchronize dependencies during a run.
Expected result: every unique source audio receives exactly one decision row or an explicit error.

- [x] **Step 5: Register the pinned remote runtime in `just`**

Create `justfile` recipes `remote-python script *args` and `remote-audit round`.
Load `IRODORI_REMOTE_HOST` from `.env`, route Python through the pinned `uv --project`,
`--no-sync`, and `--python` arguments, and keep remote work paths in shared just variables.
No dedicated `justfile` tests are required. Verify `just --list`,
`just --dry-run remote-python -c 'print(1)'`, `just --dry-run remote-audit round-example`,
and a real interpreter probe that reports Python 3.10.20 and
`soundfile` 0.13.1. Do not embed credentials or synchronize the remote environment.

### Task 5: Calibrate the detector with user review and maximize retention

**Files:**
- Generate outside Git: `review/round-N/*.wav`
- Generate outside Git: `review/round-N/review-sheet.csv`
- Generate outside Git: `labels.jsonl`
- Generate outside Git: `caption-rules.json`

- [x] **Step 1: Cluster `REVIEW` candidates**

Group by dataset, dominant frequency bucket, harmonic ratio, duration, RMS, marker presence, and neighboring source-file sequence. Select boundary examples and at least one representative from every cluster.

- [x] **Step 2: Present a compact review packet**

Copy only the selected WAVs and a table of exact suspect intervals to the local visualization artifact directory. The user labels each as `TONE`, `VOICE`, or `UNSURE`.

- [x] **Step 3: Recalibrate deterministic rules**

Turn repeated labeled signatures into explicit rule-version changes. Do not train an opaque classifier unless deterministic features cannot separate labeled cases.

- [x] **Step 4: Rescan until the boundary stabilizes**

Only newly created boundary cases appear in subsequent rounds. Restore all `VOICE` rows to `KEEP`; retain `UNSURE` rows outside the training set until resolved.

- [x] **Step 5: Repair safe censored captions**

Add explicit caption rules for non-beep marker rows. Unknown or ambiguous repairs remain in review. Produce clean datasets only after unresolved review count reaches zero.

Round 2 confirmed that all 744 censored rows are labeled `TONE`; none remain in
`REVIEW`, so no non-beep censored caption requires repair. The caption-rule set remains empty.

### Task 6: Build clean manifests and the 600M training config

**Files:**
- Create: `scripts/prepare_600m_speaker_training.py`
- Create: `tests/scripts/test_prepare_600m_speaker_training.py`
- Generate outside Git: per-dataset `clean-manifest.jsonl`, latents, and training configs

- [x] **Step 1: Write failing metadata/config tests**

```python
def test_build_training_config_preserves_voicedesign_model_shape() -> None:
    config = module.build_training_config(VOICE_DESIGN_METADATA, manifest=MANIFEST, output=OUT)
    assert config["model"]["use_caption_condition"] is True
    assert config["model"]["use_speaker_condition"] is True
    assert config["model"]["duration_architecture"] == "token_sum_dual_adarn_zero_no_aux"
    assert config["train"]["speaker_inversion_enabled"] is True
    assert config["train"]["speaker_inversion_init_embedding"] is None


def test_training_manifest_has_empty_style_caption() -> None:
    row = module.to_training_manifest_row(CLEAN_ROW)
    assert row["caption"] == ""
    assert row["text"] == CLEAN_ROW["text"]
```

Test latent reuse only when source hash matches; otherwise require re-encoding.

- [x] **Step 2: Verify RED**

Run the new test file and confirm the API is missing.

- [x] **Step 3: Implement metadata extraction and config generation**

Read `config_json` from the 600M safetensors metadata. Copy the complete model config except the inference-only `max_text_len`, `max_caption_len`, and `fixed_target_latent_steps` keys. Move the text and caption limits into the train section, but set `fixed_target_latent_steps` to null while retaining `max_latent_steps: 750`, matching the upstream Speaker Inversion recipe and avoiding fixed 750-frame padding. Set Speaker Inversion to 16 newly initialized tokens, batch size 16, bf16, gradient checkpointing, checkpoints every 250 steps, and isolated output paths. Never initialize from a 500M embedding.

- [x] **Step 4: Reuse or encode latents**

Reuse an existing latent only when stable source id and source audio hash match its provenance and the tensor passes `(T, 32)`, nonempty, and finite checks. Transcript equality is not required because caption repair does not change the audio-derived latent. Re-encode restored rows and any row without verifiable provenance using upstream `prepare_manifest.py` on CUDA.

- [x] **Step 5: Verify manifest closure**

Assert that every `KEEP` row has one readable latent and that no excluded/review row appears in `clean-manifest.jsonl`.

### Task 7: Run a 600M pilot before batch training

**Files:**
- Generate outside Git: `pilot/<speaker>/outputs_600m_speaker_inversion/`
- Generate outside Git: pilot logs and WAVs

- [x] **Step 1: Stop if the active service prevents safe VRAM use**

Read `/health`, process ids, and VRAM. Do not stop the active service without explicit authorization. If simultaneous training cannot fit, report the concrete VRAM blocker before altering the service.

- [x] **Step 2: Run a 250-step smoke pilot**

Train Anabel or the smallest clean dataset from the 600M checkpoint. Confirm finite loss, only Speaker Inversion parameters are trainable, and a `.speaker.safetensors` checkpoint is written.

- [x] **Step 3: Generate pilot controls**

Generate ordinary text and the three `んこ` words with neutral and calm styles. Confirm no errors, no high-confidence narrowband tone, and a stable duration.

- [x] **Step 4: Continue the pilot to 3,000 steps**

Retain checkpoints every 250 steps and compare at least 1,000, 1,500, 2,000, 2,500, and 3,000 steps. Select by the common quality gates rather than final-step position.

### Task 8: Train all 12 Speaker Inversion models

**Files:**
- Generate outside Git: `training/<model-id>/outputs_600m_speaker_inversion/`
- Generate outside Git: `training-status.jsonl` and per-model logs

- [x] **Step 1: Create a resumable serial queue**

Each row records model id, clean manifest hash, 600M checkpoint hash, config hash, start/end time, exit code, last checkpoint, and log path. A successful existing row is skipped only when all hashes match.

- [ ] **Step 2: Finish Kasumi without changing its active configuration**

Keep the already-running Kasumi job at batch size 1, gradient accumulation 16, and
gradient checkpointing enabled until step 3,000. Stop the old queue from advancing to
the next speaker, but do not suspend or restart the Kasumi trainer.

- [ ] **Step 3: Benchmark single-process training throughput**

After Kasumi finishes and before the next full training job starts, benchmark A
`1/16/checkpointing`, B `2/8/checkpointing`, C `4/4/checkpointing`, and D
`2/8/no-checkpointing` in isolated versioned output directories. Keep the manifest,
seed, bf16, TF32, learning rate 0.01, max latent length, and effective batch size 16
fixed. Measure 50 optimizer steps after a 10-step performance warmup and record
optimizer steps/s, samples/s, full-run peak VRAM, steady GPU utilization/power, finite
loss, and OOM status. Preserve every failed candidate and require all four candidates
before selecting a winner.

- [ ] **Step 4: Apply the fastest safe setting non-destructively**

Require finite loss, no OOM, reproducible provenance, and full-run peak VRAM at most
10.5 GiB. Write versioned configs and a versioned training-jobs manifest for only the
ten pending speakers. Do not rewrite the original configs, original jobs manifest,
Anabel or Kasumi artifacts, deployed voice bank, or service state.

- [ ] **Step 5: Train the remaining models one at a time**

Use the pilot-proven settings. Continue after a model-specific failure, but never mark that model successful. Keep outputs isolated.

- [ ] **Step 6: Verify all output payloads**

Open every `.speaker.safetensors`, confirm tensor shape `(16, 768)`, float32, and finite values, and hash every candidate checkpoint. The embedding format has empty metadata, so record the 600M checkpoint SHA/revision, upstream commit, training config SHA, and clean manifest SHA in the external `training-status.jsonl` run manifest.

### Task 9: Evaluate and select one checkpoint per model

**Files:**
- Modify: `scripts/generate_nko_beep_matrix.py`
- Modify: `tests/scripts/test_generate_nko_beep_matrix.py`
- Generate outside Git: evaluation WAVs, tone analysis, similarity reports, selected-model manifest

- [x] **Step 1: Write a failing checkpoint-matrix test**

Add an input manifest that maps a stable model id to multiple checkpoint paths. Assert case ids include model id and checkpoint step without filename collisions.

- [x] **Step 2: Verify RED, implement, and verify GREEN**

Keep the existing deployed-speaker behavior unchanged while allowing an explicit checkpoint manifest for retraining evaluation.

- [ ] **Step 3: Generate the evaluation matrix**

For every candidate checkpoint, generate ordinary controls, all `んこ` cases, multiple seeds, and neutral/calm styles. Analyze every WAV and compute speaker similarity against representative clean source recordings.

- [ ] **Step 4: Select checkpoints deterministically**

Reject any checkpoint with an error or high-confidence beep. Among the remainder rank stability, intelligibility, similarity, and style identity retention. Prefer the earlier checkpoint on a metric tie.

- [ ] **Step 5: Produce the user review packet**

Copy only ambiguous generated candidates and their controls to the local artifact directory. Keep selected embeddings staged; do not overwrite the active voice bank.

### Task 10: Verify the implementation and staged result

**Files:**
- Update: `docs/superpowers/plans/2026-07-31-600m-speaker-inversion-retraining.md` checkboxes as evidence is produced
- Generate outside Git: final dataset and model reports

- [ ] **Step 1: Run focused local verification**

```bash
uv run pytest --no-cov tests/scripts -q
uv run ruff check scripts tests/scripts
uv run ruff format --check scripts tests/scripts
```

- [x] **Step 2: Run full local verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run vulture src/
uv run pytest
```

- [ ] **Step 3: Verify remote completion**

Require 12 successful training rows, 12 selected embedding hashes, zero missing manifests/latents, zero high-confidence source or generated beep, and an unchanged healthy active service.

- [ ] **Step 4: Report exact retention and unresolved limits**

For every dataset report original unique files, kept, restored from prior censorship, confirmed beep exclusions, invalid exclusions, duplicates, and final retention percentage. Link suspicious audio and selected staged assets. Do not claim deployment replacement until the user approves it.

## Plan self-review

- Spec coverage: immutable sources, 12-model scope, Ai/Miu merge, expressive-audio preservation, label calibration, caption repair, maximum unique-file retention, 600M-only initialization, pilot, batch training, checkpoint selection, and non-destructive staging are covered.
- Placeholder scan: no deferred implementation placeholders are present.
- Type consistency: decision names and label names match `2026-07-31-600m-speaker-retraining-clean-data-design.md`.
