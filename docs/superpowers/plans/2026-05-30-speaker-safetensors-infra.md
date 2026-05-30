# Speaker Safetensors Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make trained `.speaker.safetensors` assets usable through the standard Irodori-TTS v3 voice-bank path without committing model weights.

**Architecture:** The runtime source of truth remains `VOICE_BANK_DIR/voice_bank_speakers.toml`. Clients send `speaker`; the server resolves `ref_embed` server-side and verifies referenced speaker files at startup or via an explicit deploy check.

**Tech Stack:** Python 3.11, Typer, FastAPI, PowerShell-over-SSH, uv, pytest.

---

### Task 1: Strict Voice-Bank Asset Validation

**Files:**
- Modify: `src/irodori_tts_infra/voice_bank/repository.py`
- Modify: `src/irodori_tts_infra/server/main.py`
- Modify: `tests/voice_bank/test_repository.py`
- Modify: `tests/server/test_app.py`
- Modify: `tests/gpu/test_phase2_e2e_smoke.py`

- [ ] **Step 1: Write failing repository tests**

Add tests proving that `ref_embed` must end in `.speaker.safetensors`, and that missing files are rejected only when `require_embedding_files=True`.

- [ ] **Step 2: Implement minimal repository validation**

Add `require_embedding_files: bool = False` to `load_voice_profile()`. Keep local client validation compatible by leaving the default false.

- [ ] **Step 3: Make server and GPU smoke strict**

Call `load_voice_profile(..., require_embedding_files=True)` in server startup and GPU smoke setup so runtime failures happen before synthesis.

- [ ] **Step 4: Verify**

Run:

```bash
uv run pytest --no-cov tests/voice_bank/test_repository.py tests/server/test_app.py -q
```

### Task 2: Remote Voice-Bank Verification Command

**Files:**
- Create: `src/irodori_tts_infra/deploy/remote/voice_bank.py`
- Modify: `src/irodori_tts_infra/deploy/remote/__init__.py`
- Modify: `src/irodori_tts_infra/deploy/cli.py`
- Modify: `tests/deploy/test_remote.py`

- [ ] **Step 1: Write failing deploy tests**

Add tests for `verify_voice_bank()` to ensure it runs the remote runtime Python, loads the deployed `.env`, and calls `load_voice_profile(..., require_embedding_files=True)`.

- [ ] **Step 2: Implement command**

Add `irodori-tts-deploy deploy-verify-voice-bank`. It must not sync weights; it only validates the remote `.env`, manifest, and referenced `.speaker.safetensors` files.

- [ ] **Step 3: Verify**

Run:

```bash
uv run pytest --no-cov tests/deploy/test_remote.py -q
```

### Task 3: Operator Documentation and Examples

**Files:**
- Create: `docs/deploy/voice_bank_speakers.ooppeenn.example.toml`
- Modify: `docs/deploy/windows.md`
- Modify: `.env.example`

- [ ] **Step 1: Document runtime layout**

Document that Windows keeps the actual files under `C:/Users/takut/Dev/Irodori-TTS/speakers`, while Git only tracks examples and validation code.

- [ ] **Step 2: Add OOPPEENN/Kasumi example manifest**

Create a TOML example using relative `speakers/*.speaker.safetensors` paths for `kasumi` plus the top trained OOPPEENN speakers.

- [ ] **Step 3: Verify docs references**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
```

### Task 4: Final Verification

**Files:**
- All modified source, tests, and docs.

- [ ] **Step 1: Run focused tests**

```bash
uv run pytest --no-cov tests/voice_bank/test_repository.py tests/server/test_app.py tests/deploy/test_remote.py -q
```

- [ ] **Step 2: Run project checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

- [ ] **Step 3: Report residual risk**

If Windows SSH/GPU verification is not run in this turn, explicitly report that `deploy-verify-voice-bank` and GPU synthesis still need a live remote run.
