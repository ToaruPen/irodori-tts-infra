# 0001: Irodori Runtime Access Pattern

Date: 2026-04-16

Status: Accepted

## Context

Phase 1 kept the production engine, server, client, and deploy surfaces testable
with fake backends, but left the real Irodori-TTS runtime access pattern
unsettled. That decision now blocks Phase 2 because Irodori-TTS VoiceDesign and
RVC must run together on the Windows GPU host with an RTX 4070 12GB.

The prototype history disagrees on how the server should reach Irodori-TTS:

- Prototype `CLAUDE.md:32-36` described direct Python API access through
  `sys.path` injection and `IRODORI_TTS_DIR`.
- The active prototype server imported `irodori_tts` as an installed package.
- The current backend adapter already imports `irodori_tts.inference_runtime`
  lazily and converts import/runtime failures into `BackendUnavailableError`.

The decision must prefer VRAM sharing with RVC over all other concerns, then
error propagation and server survivability, then upgrade friction, then startup
cost.

## Decision

Install Irodori-TTS as a package in the same uv-managed Python environment as
the FastAPI server and import it lazily from the Irodori backend adapter.

## Rationale

VRAM sharing with RVC is the deciding criterion. Irodori and RVC need to coexist
on one RTX 4070 12GB, and the least risky way to coordinate that budget is to
load both runtimes in the same server process. A package install keeps normal
Python imports while allowing the engine/server layer to serialize work, prewarm
the Irodori runtime, inspect CUDA memory pressure, and later add RVC without
cross-process GPU coordination.

Error propagation remains acceptable because the backend imports Irodori lazily
inside the adapter boundary instead of at package import time. Import failures,
missing optional dependencies, checkpoint download failures, and runtime
construction errors should continue to become typed backend availability errors.
The server can then expose degraded health or startup failure with structured
logs instead of crashing from a top-level `irodori_tts` import.

Upgrade friction is lower than `sys.path` injection because the deployed
environment records what was installed. The Windows host should install
Irodori-TTS from a pinned Git commit or local checkout using `uv pip install`
inside the server environment. Updating Irodori becomes an explicit dependency
operation rather than changing an ambient directory on `sys.path`.

Startup cost is acceptable. The server already treats runtime warmup as part of
the GPU lifecycle, and package import does not add an extra process or IPC
boundary. Prewarming can still happen once in FastAPI lifespan before accepting
real synthesis work.

## Consequences

Positive:

- Irodori and Phase 2 RVC can share one server process, one scheduling policy,
  and one CUDA memory budget.
- Import behavior follows normal Python packaging semantics instead of ambient
  `sys.path` mutation.
- The existing lazy backend import shape remains valid and keeps CI free of real
  Irodori imports.
- Deployment can pin the exact Irodori revision used on the GPU host.

Negative:

- A severe native crash inside Irodori or PyTorch can still take down the server
  process.
- Server startup can fail or report degraded readiness if Irodori is not
  installed correctly.
- Dependency conflicts between Irodori, RVC, PyTorch, and this package must be
  solved in one environment instead of hidden behind process boundaries.
- The current optional dependency group does not yet install Irodori-TTS by
  itself.

Follow-up work:

- Integrate the two-step Windows install examples from the implementation notes
  into the deploy scripts and operator docs: first **Install Infra** with
  `uv pip install "irodori-tts-infra[server,irodori] @ ./path/to/source"` or
  `uv pip install "irodori-tts-infra[server,irodori] @
  git+https://github.com/ToaruPen/irodori-tts-infra.git@<commit-hash>"`; then
  **Install Irodori-TTS** with `uv pip install "irodori-tts @
  ./path/to/irodori-tts"` or `uv pip install "irodori-tts @
  git+https://github.com/<owner>/Irodori-TTS.git@<commit-hash>"`.
- Remove stale `IRODORI_TTS_DIR` guidance from prototype-era docs and examples
  after deploy scripts are updated.
- Make the production server entrypoint construct the real Irodori backend in
  lifespan and surface `BackendUnavailableError` cleanly.
- Add Phase 2 RVC loading so both runtimes use the same capacity-one GPU
  scheduling policy and documented VRAM budget.
- Add the GPU smoke-test implementation described below.

## GPU Smoke Test Acceptance

This follows the repository testing policy that GPU tests require both CUDA/GPU
and the real Irodori runtime and are excluded from default `pytest`.

Test discovery and execution:

- Place the first smoke test in `tests/smoke/test_irodori_gpu_smoke.py`.
- Register and mark it with `@pytest.mark.gpu_smoke`. Also mark it with the
  existing `@pytest.mark.gpu` marker so the default local test selection still
  excludes it.
- Run it on the Windows GPU host with `uv run pytest -m gpu_smoke tests/smoke/`;
  the raw pytest equivalent is `pytest -m gpu_smoke tests/smoke/`.
- On hosts without CUDA or without the configured real Irodori runtime, skip at
  collection or fixture setup with
  `@pytest.mark.skipif(not has_cuda(), reason="CUDA GPU and real Irodori runtime required")`
  or an equivalent fixture-level `pytest.skip`. Do not xfail those hosts because
  missing GPU/runtime prerequisites are environment mismatches, not expected
  product failures.

The smoke test must fail unless all of these assertions pass:

- Raw import check: `import irodori_tts.inference_runtime` succeeds from the
  installed package. If the raw module check fails, the smoke test observes the
  failure as `ImportError` or `ModuleNotFoundError` and fails immediately.
- Adapter/factory contract check: when the backend adapter or factory initializes
  with a missing runtime, missing optional dependency, or raw `ImportError`, the
  surfaced exception is `BackendUnavailableError`; raw import exceptions must not
  leak through the adapter/factory boundary.
- Precondition: required Irodori model checkpoints are already present in the
  host cache before running the smoke test. First-time checkpoint download is
  setup work and is not part of the timed warmup phase.
- With pre-cached checkpoints, backend construction and warmup complete within
  60 seconds. A warmup timeout beyond 60 seconds or runtime construction failure
  is a failed smoke test and must map to `BackendUnavailableError`.
- If checkpoint cache validation or checkpoint restore fails before the timed
  warmup, the smoke test fails as an environment setup failure; when that failure
  is observed through the adapter/factory boundary, it must surface as
  `BackendUnavailableError`.
- Backend exceptions raised through the adapter/factory during warmup or a
  single synthetic request are translated to `BackendUnavailableError`. Leaking
  raw Irodori, PyTorch, Hugging Face, or CUDA exceptions through the server
  boundary is a failed smoke test.
- During Irodori and RVC coexistence, the server process uses no more than
  10240 MiB of GPU memory on the RTX 4070 12GB host.

VRAM measurement procedure:

- Measure the server PID with `nvidia-smi
  --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` after
  Irodori warmup and again after RVC warmup in the same process.
- Record the real host output in the smoke-test log. Example passing evidence
  shape, not a checked-in baseline:
  - after Irodori warmup: `4242, 5900`
  - after Irodori plus RVC warmup: `4242, 9700`
- The threshold is `10240 MiB` because the Phase 2 budget reserves up to
  `6144 MiB` for Irodori and up to `4096 MiB` for RVC in the server process.
  On a 12GB RTX 4070, treated as `12288 MiB`, that leaves `2048 MiB` outside the
  process for Windows display use, driver/CUDA allocator overhead, transient
  buffers, and observability tools.
- A looser `11000 MiB` threshold would leave only `1288 MiB` of headroom. Reject
  that for the first Phase 2 smoke test; revisit it only after repeated real
  host measurements show stable memory pressure below the current limit.

## Alternatives Considered

### 1. Package Install

Chosen. This keeps Irodori in the same process as the server and future RVC
backend, preserves normal import semantics, supports pinned upgrades, and avoids
an IPC boundary while still allowing lazy import/error mapping.

### 2. `sys.path` Injection with `IRODORI_TTS_DIR`

Rejected. It has similar in-process VRAM characteristics to package install,
but it relies on ambient path mutation, makes the deployed revision harder to
audit, and can silently import the wrong checkout. It also keeps prototype-era
configuration in the hot path without providing better startup or error
behavior.

### 3. Subprocess Isolation

Rejected. It improves crash containment for Irodori import/runtime failures, but
it makes the top-weighted criterion worse: Irodori and RVC would live in
separate CUDA contexts, making 12GB VRAM sharing harder to coordinate and adding
IPC, serialization, and lifecycle complexity.

### 4. Remote Venv Wrapper

Rejected. A separate uv-managed venv under the Windows deploy tree would make
Irodori dependency conflicts more isolated, but it is still out-of-process and
therefore has the same VRAM-sharing problem as subprocess isolation. It also
adds a second environment to bootstrap, pin, monitor, and warm.

## Implementation Notes

- Keep `irodori_tts` and `huggingface_hub` imports inside
  `src/irodori_tts_infra/engine/backends/irodori.py` or narrower factory paths.
- Do not use `IRODORI_TTS_DIR` for runtime imports. Runtime configuration should
  continue to use `IRODORI_TTS_RUNTIME_*` for model checkpoint, devices,
  precision, warmup, decode mode, and compile settings.
- Windows deployment should install two packages into the same uv environment:
  **Install Infra** with PEP 508 extras syntax, then **Install Irodori-TTS** from
  a pinned source until the optional dependency can encode that source directly.
  For local checkouts, use `uv pip install "irodori-tts-infra[server,irodori] @
  ./path/to/source"` and `uv pip install "irodori-tts @
  ./path/to/irodori-tts"`. For Git sources, use `uv pip install
  "irodori-tts-infra[server,irodori] @
  git+https://github.com/ToaruPen/irodori-tts-infra.git@<commit-hash>"` and
  `uv pip install "irodori-tts @
  git+https://github.com/<owner>/Irodori-TTS.git@<commit-hash>"`, with both
  commit hashes pinned in the URLs.
- `CLAUDE.md` in this repository is currently a symlink to `AGENTS.md`; the
  referenced `CLAUDE.md:32-36` guidance belongs to the prototype history. Any
  surviving prototype instructions that mention `sys.path` or `IRODORI_TTS_DIR`
  should be replaced with this ADR when those docs are migrated.
- Phase 2 RVC work should assume a same-process runtime and implement explicit
  load order, warmup order, and memory/backpressure handling instead of a
  process boundary.
