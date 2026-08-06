# AGENTS.md

## WHY — Project and Runtime Contract

`irodori-tts-infra` is a Python 3.11+ infrastructure project for Japanese TTS
using Irodori-TTS v4 Small VoiceDesign with Speaker Inversion voice identity.

The standard synthesis path is:

1. Text and a fixed public style preset enter the Irodori-TTS v4 VoiceDesign model.
2. The server resolves the narrator or character to a Speaker Inversion embedding.
3. A multi-metric quality gate produces an automated pass/fail result.
4. Passing audio is available for playback or caching.

The public `style` enum maps server-side to fixed VoiceDesign captions. Public
arbitrary captions and RVC are not part of the standard path.

## WHAT — Repository and Sources of Truth

- The import package is `src/irodori_tts_infra/`; tests are under `tests/`.
- Generated audio, model weights, datasets, checkpoints, and local secrets are
  runtime assets, not source files.
- `pyproject.toml` owns deterministic Python tool, test, and coverage configuration.
- The root `justfile` owns executable command expansion.
- `docs/` owns detailed architecture, deployment, and connection guidance;
  start with `docs/connection.md` for the remote topology and troubleshooting.
- The nearest nested `AGENTS.md` owns subtree-specific boundaries and rules.

Project instructions take priority over project docs, which take priority over
implementation. If these sources contradict one another, stop and report the exact
files and conflicting passages instead of guessing.

## HOW — Commands

The root `justfile` is the executable command catalog. Run `just --list` before
choosing a recipe. `uv` remains the package manager and command runner beneath it.

- `just sync`: install local development dependencies.
- `just test [ARGS]`: run default tests and forward pytest arguments.
- `just check`: run the full local lint, format, type, dead-code, and test gate.
- `just client [ARGS]` and `just deploy [ARGS]`: run the project CLIs.
- `just v4-inference-benchmark [ARGS]`: compare v4 sampling profiles against the
  active capability catalog without retaining text, voice metadata, or audio.
- `just v4-inference-blind-ab {prepare,score} [ARGS]`: prepare or score a
  create-only local fixed-profile listening packet; it never changes runtime or
  production configuration.
- `just remote-python SCRIPT [ARGS]` and `just remote-audit ROUND`: use the pinned
  upstream Windows runtime for remote scripts and dataset audits.
- `just speaker-{train-queue,speed-benchmark,apply-speed-selection,evaluation-queue,build-manifests,generate,analyze,metrics,evaluate,review-packet,staging-report}`:
  run the local training/evaluation stages; the matching `remote-speaker-*`
  recipes run them in the pinned Windows runtime. Remote generation uses the
  standalone 140-case, five-checkpoint generator.
- `just speaker-train-queue-detached {preflight,launch,status}` runs the pinned,
  fail-closed training supervisor; `just speaker-quality-run {prepare,finalize}`
  manages create-only quality-search/retraining evidence. Matching remote recipes
  use the versioned Windows launcher and quality-run bundle.
- `just speaker-checkpoint-search-{build,generate,evaluate}` runs the isolated,
  create-only 28-case single-checkpoint diagnostic path. Matching remote recipes use
  the pinned Windows runtime and never produce final checkpoint selections.
- `just speaker-midpoint-diagnostic-{build,generate,evaluate}` runs the isolated,
  create-only fixed-alpha derived-embedding diagnostic. Its synthetic step zero is
  case identity only; the path never creates or promotes a training checkpoint.
- `just speaker-reference-centroid-audit` runs the read-only reference identity and
  ECAPA centroid-stability diagnostic. The matching remote recipe uses the pinned
  Windows runtime and writes one create-only evidence report.
- `just speaker-evaluation-speed-v4 {prepare,preflight,launch}` retains the isolated
  speed-v4 evaluation launcher. The matching `speaker-evaluation-speed-v5` recipe is
  the retained legacy all-model launcher. `speaker-evaluation-speed-v6` runs its
  versioned fresh 140-case successor against the quality-successor training evidence
  without changing the speed-v5 path or artifacts.
  `just speaker-evaluation-speed-v6-detached {preflight,launch,status}` runs the
  fail-closed detached supervisor; the matching remote recipe uses its create-only
  versioned bundle and launch evidence under `evaluation_speed_v6`.
  `just speaker-verify-retraining [ARGS]` verifies immutable training evidence or the
  complete non-deployment workflow. Matching `remote-speaker-*` recipes run against
  the pinned Windows workspace.

Do not duplicate raw tool commands in instructions when a `just` recipe owns them.

## HOW — Development Workflow

- Inspect the relevant implementation, tests, and nearest scoped instructions before
  editing. Keep changes within the requested issue or ticket and preserve unrelated
  dirty worktree changes.
- For Python behavior changes, use Red → Green → Refactor. Bug fixes require a
  failing-then-passing regression test.
- Run the narrowest useful check first, then `just check` when practical. Report any
  check blocked by GPU, network, SSH, model weights, or another external dependency.
- Do not add architecture layers or perform broad refactors without discussion.
  Backward compatibility is not required unless the task explicitly requires it.
- Library code uses `structlog`; do not swallow exceptions. Handle specific failures
  or re-raise them with context.

## HOW — Tests and Coverage

Default pytest runs exclude `integration`, `gpu`, and `slow` tests. Available markers:

- `unit`: fast and deterministic; no network, GPU, or external services.
- `integration`: external services, network, SSH, or real subprocesses.
- `gpu`: CUDA/GPU hardware, model weights, or the real Irodori-TTS runtime.
- `slow`: too slow for the default local loop.
- `ssh`: requires access to a remote host.

Mark model-, network-, GPU-, and SSH-dependent tests appropriately; default tests
must not depend on generated audio, remote machines, or unavailable runtime assets.

CI enforces two coverage tiers with branch coverage enabled:

- 100% line and branch coverage for `text/*.py`, `voice_bank/captions.py`,
  `voice_bank/models.py`, `engine/models.py`, `engine/protocols.py`,
  `engine/errors.py`, and `contracts/*.py`, excluding `__init__.py` re-exports.
- At least 80% coverage for the overall project.

Use `# pragma: no cover` only for paths that genuinely cannot run without an external
service, GPU, network, or real Irodori runtime; explain the reason inline.

## HOW — Remote Runtime and Operational Safety

- The local infrastructure targets Python 3.11+.
- The GPU host is Windows with an RTX 4070 12GB and is reached over Tailscale/SSH.
- General connection and deployment settings belong in uncommitted `.env`; copy
  `.env.example` and follow `docs/connection.md`. The current 600M retraining workspace,
  upstream project, and virtual environment are intentionally pinned in `justfile`.
- Remote recipes invoke the upstream Irodori-TTS project and virtual environment via
  `uv run --project ... --no-sync --python ...`. They neither resolve an arbitrary
  PowerShell `python` nor synchronize the pinned upstream training environment.
- Remote checkpoint generation uses its standalone script because the infrastructure
  package is intentionally not installed into the pinned upstream environment.
- Do not stop or replace an active service, and do not replace the deployed voice
  bank, without explicit user authorization.

## HOW — Repository Hygiene

- Keep `uv.lock` committed.
- Do not commit `.env`, credentials, model weights, checkpoints, generated audio, or
  datasets.
- Keep only small, intentional fixtures under `tests/fixtures/`.
- Commit only when explicitly requested.

## Scoped Instructions

Nested `AGENTS.md` files under `src/irodori_tts_infra/` and `tests/` define local
dependency boundaries, ownership, and test placement. Apply the closest file in
addition to this root file.
