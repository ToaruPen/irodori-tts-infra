# AGENTS.md

## Project

`irodori-tts-infra` is a Python 3.11+ infrastructure project for Irodori-TTS
v3 VoiceDesign synthesis with Speaker Inversion voice identity.

The import package lives under `src/irodori_tts_infra/`. Tests live under `tests/`.
Generated audio, model weights, datasets, checkpoints, and local secrets are not source files.

## Architecture

```text
Text + fixed style preset → Irodori-TTS v3 VoiceDesign
     + per-character Speaker Inversion embedding (.speaker.safetensors)
     → Multi-metric quality gate (automated pass/fail)
     → Playback / cache
```

Voice identity is supplied by the narrator or character `ref_embed` from
`voice_bank_speakers.toml`. Expressive direction comes from a strict public
`style` enum mapped server-side to fixed VoiceDesign captions. Arbitrary public
captions and RVC are not part of the standard path.

## Commands

Use `uv` for dependency management.

- Install/sync: `uv sync --all-extras`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`
- Format write: `uv run ruff format .`
- Type check: `uv run mypy`
- Default tests: `uv run pytest`
- Integration tests: `uv run pytest -m "integration"`
- GPU tests: `uv run pytest -m "gpu"`
- Dead code detection: `uv run vulture src/`
- Full verification: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run vulture src/ && uv run pytest`

## Quality Bar

Lint, type checking, and tests are the source of truth.

### Principles

- **TDD**: Write the failing test first. Then write the minimal code to make it pass. Then refactor.
- **YAGNI**: Do not build for hypothetical future requirements. Three similar lines beat a premature abstraction.
- **DRY**: Extract shared logic only when duplication is proven and stable. Premature DRY creates coupling.

### Rules

- Keep code direct. No wrapper classes, factories, protocols, hooks, or configuration layers unless the current code needs them.
- No placeholder comments, commented-out code, speculative TODOs, or docstrings that restate the function name.
- No swallowed exceptions. Catch specific types and either handle or re-raise with context.
- No `print` in library code. Use structured logging via `structlog`.
- Prefer small typed functions with explicit inputs and outputs.
- Public behavior changes need tests. Bug fixes need a failing-then-passing regression test.

### Coverage Tiers

Two coverage thresholds are enforced in CI:
- **Pure-logic modules** (text/, voice_bank/captions.py, voice_bank/models.py, engine/models.py, engine/protocols.py, engine/errors.py, contracts/): 100% line AND branch coverage. `__init__.py` re-exports are omitted.
- **Overall project**: ≥ 80% (line + branch combined; branch coverage is enabled globally).

Pure modules have no side effects — new branches there require corresponding tests before CI will pass. Use `# pragma: no cover` only for code paths that genuinely cannot run without external services (GPU, real Irodori runtime, network). Inline pragma comments must explain why.

## Testing

Default `pytest` excludes `integration`, `gpu`, and `slow` tests.

- `unit`: fast deterministic, no external services
- `integration`: network, SSH, subprocess, or service-dependent
- `gpu`: CUDA/GPU or real Irodori-TTS runtime
- `slow`: too slow for the default loop
- `ssh`: requires remote host access

Tests must not depend on model weights, generated audio, or remote machines unless marked.

## Hardware

- GPU server: Windows, RTX 4070 12GB, Tailscale VPN
- Client: macOS (Apple Silicon)
- Host/path configuration: see `.env` (not committed; copy from `.env.example`)
- Connection model and troubleshooting: see `docs/connection.md`

## Repository Hygiene

- Keep `uv.lock` committed
- Do not commit `.env`, credentials, model weights, checkpoints, generated audio, or datasets
- Small intentional fixtures under `tests/fixtures/`
- Commit only when explicitly asked

## Agent Workflow

Before editing, inspect the relevant files and existing patterns. After editing Python code, run the narrowest useful check first, then full verification when practical. If a check cannot run due to missing services, GPU, or SSH, report that explicitly.

## Scoped Instructions

Nested `AGENTS.md` files exist only when a subtree has different commands, constraints, or ownership.
