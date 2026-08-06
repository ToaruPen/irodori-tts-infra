# 600M VoiceDesign and Nko Beep Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the 600M VoiceDesign infra migration and produce a reproducible, reviewable beep-screening result for every deployed Speaker Inversion embedding.

**Architecture:** Keep the public synthesis contract restricted to fixed style presets while forwarding caption and speaker conditioning to upstream Irodori-TTS. Add a bootstrap compatibility probe for the exact upstream fields required by the 600M checkpoint. Use standalone diagnostic scripts for the finite GPU generation matrix and NumPy-based narrowband analysis so generated audio stays outside the import package and outside Git.

**Tech Stack:** Python 3.11, Pydantic, FastAPI, NumPy, pytest, uv, PowerShell over SSH, Irodori-TTS, ffmpeg.

**Repository policy:** Do not commit. Existing uncommitted changes belong to the user and must be preserved.

---

### Task 1: Make bootstrap reject an incompatible upstream runtime

**Files:**
- Modify: `tests/deploy/test_remote.py`
- Modify: `src/irodori_tts_infra/deploy/remote/bootstrap.py`

- [ ] **Step 1: Add a failing bootstrap-script assertion**

Extend `test_bootstrap_creates_remote_dir_then_runtime_venv` with assertions that the generated PowerShell runs a Python compatibility probe after `uv pip check`. The probe must inspect:

```python
required_sampling_fields = {
    "caption",
    "ref_embed",
    "cfg_scale_caption",
    "cfg_scale_speaker",
    "cfg_guidance_mode",
}
required_model_fields = {
    "use_caption_condition",
    "use_speaker_condition",
}
```

Assert that `SamplingRequest`, `ModelConfig`, both required-field variable names, and the error label `Irodori-TTS runtime is incompatible with 600M VoiceDesign` appear in the decoded remote script.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest --no-cov tests/deploy/test_remote.py::test_bootstrap_creates_remote_dir_then_runtime_venv -q
```

Expected: failure because the bootstrap script does not contain a compatibility probe.

- [ ] **Step 3: Add the minimal compatibility probe**

Add `_runtime_compatibility_check_script(runtime_python: str) -> str` to `bootstrap.py`. It must return PowerShell that runs this Python body:

```python
import inspect

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import SamplingRequest

required_sampling_fields = {
    "caption",
    "ref_embed",
    "cfg_scale_caption",
    "cfg_scale_speaker",
    "cfg_guidance_mode",
}
required_model_fields = {
    "use_caption_condition",
    "use_speaker_condition",
}
sampling_fields = set(inspect.signature(SamplingRequest).parameters)
model_fields = set(inspect.signature(ModelConfig).parameters)
missing = sorted(
    required_sampling_fields.difference(sampling_fields)
    | required_model_fields.difference(model_fields)
)
if missing:
    raise RuntimeError(
        "Irodori-TTS runtime is incompatible with 600M VoiceDesign; "
        f"missing fields: {', '.join(missing)}"
    )
```

Append the probe after `uv pip check` in `_bootstrap_script`. If the Python process exits nonzero, PowerShell must stop bootstrap with the same compatibility label.

- [ ] **Step 4: Run the focused deploy tests and verify GREEN**

Run:

```bash
uv run pytest --no-cov tests/deploy/test_remote.py -q
```

Expected: all deploy tests pass.

### Task 2: Close remaining 600M VoiceDesign documentation and GPU-smoke gaps

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/deploy/rvc-training.md`
- Modify: `docs/deploy/windows.md`
- Modify: `tests/gpu/test_phase2_e2e_smoke.py`

- [ ] **Step 1: Update the GPU smoke test to exercise caption plus speaker**

Keep the existing narrator/dialogue test. Add a second GPU test that constructs:

```python
job = SynthesisJob(
    segment_index=0,
    text="落ち着いて読み上げます。",
    speaker=smoke_character_name,
    require_speaker=True,
    style="calm",
)
result = pipeline.synthesize_job(job)
```

Decode the returned WAV and assert positive frame count, sample rate, and elapsed time. This test is the real-runtime proof that the 600M checkpoint accepts both a fixed VoiceDesign caption and a Speaker Inversion embedding.

- [ ] **Step 2: Update current operator-facing text**

Change the package description from `v3 base` to `v3 VoiceDesign`. In the superseded RVC training document, describe the current standard runtime as `v3 VoiceDesign with Speaker Inversion`. In the Windows deployment guide, document:

```env
IRODORI_TTS_RUNTIME_CHECKPOINT=Aratako/Irodori-TTS-600M-v3-VoiceDesign
IRODORI_TTS_RUNTIME_CFG_SCALE_CAPTION=3.0
IRODORI_TTS_RUNTIME_WARMUP_STYLE=calm
```

State that bootstrap now rejects an upstream checkout missing the v3 VoiceDesign speaker/caption model fields.

- [ ] **Step 3: Run non-GPU checks for the touched files**

Run:

```bash
uv run ruff check src/irodori_tts_infra/deploy/remote/bootstrap.py tests/deploy/test_remote.py tests/gpu/test_phase2_e2e_smoke.py
uv run ruff format --check src/irodori_tts_infra/deploy/remote/bootstrap.py tests/deploy/test_remote.py tests/gpu/test_phase2_e2e_smoke.py
```

Expected: both commands exit zero.

### Task 3: Add a deterministic generation matrix script

**Files:**
- Create: `scripts/generate_nko_beep_matrix.py`
- Create: `tests/scripts/test_generate_nko_beep_matrix.py`

- [ ] **Step 1: Write failing pure-logic tests**

Load the script module with `importlib.util.spec_from_file_location`. Test that:

```python
cases = build_cases(
    speaker_paths=(Path("a.speaker.safetensors"), Path("b.speaker.safetensors")),
    text_cases=TEXT_CASES,
    seeds=SEEDS,
)
assert len(cases) == 2 * 7 * 2
assert len({case.case_id for case in cases}) == len(cases)
assert {case.style for case in cases} == {"neutral"}
```

Also test repeated `--speaker`, `--text-id`, and `--seed` filters through a pure `filter_cases` function.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_generate_nko_beep_matrix.py -q
```

Expected: failure because the script module does not exist.

- [ ] **Step 3: Implement the matrix generator**

Define immutable `TextCase` and `GenerationCase` dataclasses, these seven text cases:

```python
TEXT_CASES = (
    TextCase("word_unko", "うんこ。", False),
    TextCase("word_chinko", "ちんこ。", False),
    TextCase("word_manko", "まんこ。", False),
    TextCase("sentence_unko", "「うんこ」という言葉を読み上げます。", False),
    TextCase("sentence_chinko", "「ちんこ」という言葉を読み上げます。", False),
    TextCase("sentence_manko", "「まんこ」という言葉を読み上げます。", False),
    TextCase("control", "こんにちは。今日はいい天気ですね。", True),
)
SEEDS = (1234, 5678)
```

The CLI must accept:

```text
--speakers-dir PATH
--output-dir PATH
--checkpoint REPO_ID
--checkpoint-revision COMMIT_SHA
--checkpoint-sha256 FILE_SHA256
--style neutral|calm|cheerful|clear
--speaker STEM  (repeatable)
--text-id ID    (repeatable)
--seed INTEGER  (repeatable)
```

The official 600M VoiceDesign checkpoint supplies the pinned revision and file
SHA-256 defaults. A custom `--checkpoint` requires both identity arguments; do not
combine checkpoint overrides with `--checkpoint-manifest` evaluation mode.

Create one backend with `create_irodori_backend(IrodoriRuntimeSettings(...))`, synthesize each selected case using an absolute `ref_embed`, and write:

```text
<output-dir>/wav/<case_id>.wav
<output-dir>/generation-results.jsonl
<output-dir>/generation-config.json
```

Each JSONL row must contain the case ID, speaker filename, text ID, text, control
flag, seed, style, checkpoint identity, elapsed seconds, WAV path and SHA-256,
status, exception type, and exception message. Checkpoint-manifest rows also bind
the exact embedding, manifest, base checkpoint, and training provenance. Continue
after per-case `BackendUnavailableError`, `OSError`, `RuntimeError`, or `ValueError`;
always close the backend.

- [ ] **Step 4: Run generator unit tests and verify GREEN**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_generate_nko_beep_matrix.py -q
```

Expected: all tests pass without loading Irodori-TTS.

### Task 4: Add deterministic narrowband detection and candidate artifacts

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `scripts/analyze_nko_beep_matrix.py`
- Create: `tests/scripts/test_analyze_nko_beep_matrix.py`

- [ ] **Step 1: Write the detector tests before implementation**

Use `uv run --with 'numpy>=2,<3'` for RED. Generate three 48 kHz arrays:

```python
silence = np.zeros(48_000, dtype=np.float64)
t = np.arange(48_000, dtype=np.float64) / 48_000
beep = np.sin(2 * np.pi * 1_000 * t) * 0.5
voiced = sum(
    (0.15 / harmonic) * np.sin(2 * np.pi * 180 * harmonic * t)
    for harmonic in range(1, 7)
)
```

Assert that silence, a voice-range pure tone, the six-harmonic signal, and a
voiced signal with a dominant high harmonic produce no interval. A 300 ms beep
embedded in silence and a fading stepped beep must produce intervals. Add WAV
decoding tests for mono and stereo PCM16.

- [ ] **Step 2: Run the detector tests and verify RED**

Run:

```bash
uv run --with 'numpy>=2,<3' pytest --no-cov tests/scripts/test_analyze_nko_beep_matrix.py -q
```

Expected: failure because the analyzer does not exist.

- [ ] **Step 3: Implement the analyzer**

Use these fixed defaults:

```python
ToneConfig(
    window_size=2048,
    hop_size=480,
    analysis_min_frequency_hz=80.0,
    min_tone_frequency_hz=500.0,
    max_frequency_hz=5000.0,
    peak_half_width_hz=40.0,
    min_peak_energy_ratio=0.95,
    max_normalized_entropy=0.20,
    max_frequency_std_hz=80.0,
    min_rms_dbfs=-45.0,
    min_qualifying_frames=8,
    max_gap_frames=3,
    min_interval_span_frames=10,
)
```

Implement `read_pcm16_wav`, `detect_narrowband_intervals`, and JSONL analysis.
Use Hann-windowed `numpy.fft.rfft`. Compute tonal purity against the full
80–5,000 Hz analysis band so a voice fundamental below 500 Hz still contributes
to the denominator. Join qualifying frames across gaps of at most three frames
and reject a group when the standard deviation of its peak frequencies exceeds
80 Hz.

The CLI must accept an input generation directory and output analysis directory. It must write:

```text
analysis-results.jsonl
summary.csv
summary.md
spectrograms/<case_id>.png
```

Set `CLEAR`, `CANDIDATE`, or `ERROR` per row. Generate spectrogram PNGs only for candidates with:

```bash
ffmpeg -y -i INPUT.wav -lavfi showspectrumpic=s=1600x900:legend=1 OUTPUT.png
```

Include the complete detector configuration in every result and in `summary.md`.

- [ ] **Step 4: Add NumPy to the dev dependency group and lock**

Add:

```toml
"numpy>=2,<3",
```

to `[dependency-groups].dev`, then run:

```bash
uv lock
```

- [ ] **Step 5: Run analyzer tests and verify GREEN**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_analyze_nko_beep_matrix.py -q
```

Expected: all tests pass.

### Task 5: Run local verification

**Files:**
- All modified source, tests, scripts, and docs.

- [ ] **Step 1: Run focused tests**

```bash
uv run pytest --no-cov \
  tests/config/test_settings.py \
  tests/contracts/test_synthesis_contracts.py \
  tests/deploy/test_remote.py \
  tests/engine/backends/test_irodori.py \
  tests/engine/test_pipeline.py \
  tests/scripts/test_generate_nko_beep_matrix.py \
  tests/scripts/test_analyze_nko_beep_matrix.py \
  tests/server/routers/test_synthesis.py -q
```

- [ ] **Step 2: Run the full repository verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run vulture src/
uv run pytest
bash scripts/check-dotfiles.sh
npx secretlint .
```

`scripts/check-dotfiles.sh` is required by the shared AGENTS instructions but may be absent from this repository. If absent, record that exact blocker instead of creating an unrelated script. All other commands must pass.

### Task 6: Verify an isolated Windows deployment

**Files:**
- Remote temporary deployment only; do not replace the active `127.0.0.1:8924` service.

- [ ] **Step 1: Record the active service and GPU state**

Over SSH, record the process bound to port 8924 and `nvidia-smi` memory usage. Do not stop it.

- [ ] **Step 2: Create an isolated deploy directory**

Use:

```text
C:\Users\takut\Dev\irodori-tts-infra-codex-verify-20260731
```

Sync the local source there, copy the active deployment `.env`, and bootstrap the new `.runtime-venv`. The compatibility probe from Task 1 must pass.

- [ ] **Step 3: Start and verify on loopback port 8931**

Start the isolated service with host `127.0.0.1` and port `8931`. Poll `/health` until it reports `status=ok` and `model_loaded=true`. Send one `style=calm` request for a manifest speaker and validate the returned WAV.

- [ ] **Step 4: Run the GPU smoke test**

Run in the isolated runtime:

```powershell
uv run --no-sync pytest -m gpu tests/gpu/test_phase2_e2e_smoke.py -s
```

If the sync layout does not include tests, run the equivalent test from a separately copied test file without changing the active deployment.

- [ ] **Step 5: Stop only the isolated service**

Stop port 8931 before the generation matrix so the active 8924 service remains the only resident server model.

### Task 7: Generate and analyze all 182 primary samples

**Files:**
- Remote generated WAVs.
- Local artifact directory outside Git.

- [ ] **Step 1: Run the neutral matrix**

Copy `scripts/generate_nko_beep_matrix.py` to the isolated deployment and execute it with that deployment's runtime Python against:

```text
C:\Users\takut\Dev\Irodori-TTS\speakers
```

Expected: 182 JSONL result rows, one for every 13 × 7 × 2 combination.

- [ ] **Step 2: Copy results to the local artifact workspace**

Copy the matrix output into a dated directory outside the Git worktree. Preserve WAV filenames and JSONL metadata.

- [ ] **Step 3: Run narrowband analysis**

Run `scripts/analyze_nko_beep_matrix.py` locally. Confirm that `summary.csv`, `summary.md`, and every candidate spectrogram exist.

- [ ] **Step 4: Review candidates visually**

Inspect every candidate spectrogram. Promote a row to `REPRODUCED` only when the target phrase has a horizontal single-frequency band absent from the same speaker/seed control. Leave uncertain rows as `CANDIDATE`.

- [ ] **Step 5: Re-run candidates with calm caption**

For each `CANDIDATE` or `REPRODUCED` row, run the generator with `--style calm` and exact speaker, text ID, and seed filters. Analyze and copy those outputs beside the neutral artifacts.

### Task 8: Final handoff

**Files:**
- Local analysis summary and candidate artifacts.

- [ ] **Step 1: Verify artifact completeness**

Confirm:

```text
182 primary result rows
13 unique speaker filenames
7 text IDs
2 seeds
no missing WAV for successful rows
all candidates have PNG spectrograms
all candidates have calm comparison rows
```

- [ ] **Step 2: Report results**

Provide:

- infra verification commands and outcomes
- isolated GPU deployment outcome
- clear/error/candidate/reproduced counts
- one row per suspect with speaker filename, text, seed, frequency, interval, neutral WAV, calm WAV, and spectrogram links
- an explicit note that retraining was not performed

Ask the user to make only the final perceptual decision on the linked suspect WAVs.
