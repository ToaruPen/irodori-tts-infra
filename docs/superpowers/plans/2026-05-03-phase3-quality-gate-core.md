# Phase 3 Quality Gate Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the deterministic core models and verdict logic for Phase 3 quality gates without pulling heavyweight audio-model dependencies into the default test loop.

**Architecture:** Keep the first slice inside `irodori_tts_infra.metrics`: immutable dataclasses for scores, thresholds, verdicts, and pure functions for identity, relative-margin, MOS, F0, and style threshold checks. External WavLM, ECAPA, F0, MOS, and style scorers remain adapters for later tasks because their dependencies and runtime model downloads need separate decisions.

**Tech Stack:** Python dataclasses, stdlib math, pytest, ruff, mypy.

---

### Task 1: Quality Gate Core Models and Verdicts

**Files:**
- Modify: `src/irodori_tts_infra/metrics/models.py`
- Modify: `src/irodori_tts_infra/metrics/quality.py`
- Modify: `src/irodori_tts_infra/metrics/__init__.py`
- Test: `tests/metrics/test_quality.py`

- [ ] **Step 1: Write failing tests for identity pass, hard fail, margin fail, and moan warning**

Add tests that construct `QualityGateInput`, call `evaluate_quality_gate`, and assert `QualityGateStatus.PASS`, `FAIL`, or `WARN`.

- [ ] **Step 2: Run the narrow test and verify RED**

Run: `uv run pytest tests/metrics/test_quality.py -q`

Expected: FAIL because the metrics models and `evaluate_quality_gate` are not implemented.

- [ ] **Step 3: Implement minimal immutable models and pure evaluation logic**

Define score models, thresholds, issue records, and verdict status in `metrics/models.py`. Implement `cosine_similarity`, `relative_margin`, configured-threshold missing-score failures, non-speech warning downgrades for identity-derived gates, and `evaluate_quality_gate` in `metrics/quality.py`.

- [ ] **Step 4: Run the narrow test and verify GREEN**

Run: `uv run pytest tests/metrics/test_quality.py -q`

Expected: PASS.

- [ ] **Step 5: Run repository checks**

Run:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

Expected: all pass.

### Explicit Follow-Up Scope

- Real WavLM/ECAPA/SpeechMOS/parselmouth/librosa adapters.
- Optional dependency grouping and `uv.lock` updates for heavy metric libraries.
- `voice_bank_rvc.toml` threshold schema or a separate threshold manifest.
- Engine/server/client integration for returning or enforcing quality scores.
- Windows GPU validation through `tailscale ssh ts-win-main`.
