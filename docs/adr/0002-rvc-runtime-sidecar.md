# 0002: RVC Runtime Sidecar

Date: 2026-04-17

Status: Accepted

## Context

[ADR 0001](0001-irodori-runtime-access.md) established same-process package
installation as the Irodori runtime access pattern because Irodori-TTS
VoiceDesign and RVC need to coexist on the Windows GPU host with an RTX 4070
12GB. That policy remains correct for Irodori, whose backend adapter lazily
imports `irodori_tts.inference_runtime` and maps import/runtime failures to
`BackendUnavailableError`.

RVC is different. The two-stage pipeline in
[the Irodori + RVC architecture](../irodori-rvc-architecture.md) depends on RVC
for per-character voice identity, but current Python-library packaging options
do not cleanly install on this project's Python 3.11 target. We ran three PyPI
smoke tests on macOS aarch64, Python 3.11.14, with `uv`:

1. `uv add rvc-python` resolved and installed, but
   `python -c "import rvc_python"` died inside `fairseq.dataclass.configs` at
   Python 3.11 dataclass strictness with
   `TypeError: non-default argument follows default argument`. The package pins
   upstream `fairseq==0.12.2`; patching requires forcing the
   `One-sixth/fairseq` fork, which is equivalent to vendoring.
2. `uv add rvc`, the `RVC-Project/Retrieval-based-Voice-Conversion` library
   package, failed while building pinned `av==11.0.0` against ffmpeg 8.x headers
   because `AV_OPT_TYPE_CHANNEL_LAYOUT` was renamed to
   `AV_OPT_TYPE_CHLAYOUT`. Fixing this requires an upstream dependency override.
3. `uv add "applio @ git+https://github.com/IAHispano/Applio"` failed because
   the repository `does not appear to be a Python project`; Applio ships no
   `setup.py` or `pyproject.toml`.

Conclusion: clean pip install is empirically dead for all current RVC Python
library candidates on this project's Python 3.11 target. The runtime decision
therefore needs an explicit exception to ADR 0001's same-process policy.
The macOS smoke results are packaging triage, not the Windows production
sign-off. Before first production use, the same installability and sidecar
contract checks must be repeated on the Windows RTX 4070 host and recorded with
the pinned RVC commit selected for that deployment.

This ADR also provides the runtime decision that later updates should cross-link
from [Windows GPU deployment](../deploy/windows.md) and
[the RVC training SOP](../deploy/rvc-training.md).

## Decision

Run RVC as a pinned Gradio HTTP sidecar in a separate Windows uv venv; reach it via `gradio_client`.

## Rationale

1. VRAM sharing.

   ADR 0001 made VRAM sharing the top criterion. This sidecar keeps the same
   capacity-one scheduling policy: the FastAPI server serializes GPU synthesis
   work so Irodori generation and RVC conversion do not run concurrently. The
   expected steady-state budget is still plausible on the RTX 4070 12GB:
   Irodori at 4-6GB, RVC at 2-4GB, plus CUDA contexts and allocator overhead at
   roughly 1GB, for a total below 12GB.

   That estimate is not a production sign-off. `torch.cuda.empty_cache()` does
   not free live tensors, and separate processes add separate CUDA context
   overhead. Production rollout requires empirical VRAM measurement on the
   Windows RTX 4070 12GB host with the long-running Irodori server and the
   long-running RVC sidecar both warmed.

2. Error propagation.

   The RVC adapter boundary should match the style of
   `src/irodori_tts_infra/engine/backends/irodori.py`: dependency, connection,
   timeout, and runtime-availability failures are translated into
   `BackendUnavailableError`. HTTP timeout and connection errors from the
   Gradio sidecar must not leak raw transport exceptions through the engine
   boundary.

3. Upgrade friction.

   The sidecar has its own pinned uv environment. RVC dependency churn in
   `numba`, `fairseq`, `numpy`, `gradio`, ffmpeg/`av`, and related inference
   packages is contained in that environment instead of being solved inside the
   main FastAPI/Irodori uv environment.

4. Startup cost.

   The RVC WebUI starts once per Windows host boot as a long-running local
   Gradio server. The main process keeps using the existing FastAPI server and
   Irodori backend. The RVC adapter calls the sidecar with
   `gradio_client.Client(...).predict(..., api_name="/infer_convert")`; the
   client can reuse the TCP connection instead of paying process startup cost
   per synthesis request.

## Consequences

Positive:

- RVC dependency conflicts are isolated from the main FastAPI/Irodori runtime.
- The main uv environment gains only the small `gradio_client` dependency for
  sidecar access.
- The official
  `RVC-Project/Retrieval-based-Voice-Conversion-WebUI` runtime can be pinned and
  operated without pretending it is a clean Python library package.
- The engine boundary still exposes typed backend availability failures instead
  of raw optional-dependency or transport failures.

Negative:

- Operators now manage two processes: the FastAPI server and the RVC Gradio
  sidecar.
- The separate sidecar process adds another CUDA context and allocator overhead
  on the 12GB GPU host.
- Cross-process HTTP latency is added, though it is negligible for batch or
  streaming synthesis compared with TTS generation and RVC conversion time.
- The main server no longer receives direct Python exception stacks from the RVC
  runtime; RVC details must come from the sidecar logs and mapped adapter errors.

Follow-up work:

- Add `deploy-rvc-start` and `deploy-rvc-stop` commands for the Windows host.
- Measure warmed Irodori plus warmed RVC VRAM on the RTX 4070 12GB host before
  production sign-off.
- Add `tests/engine/backends/test_rvc_sidecar.py`, marked with both `gpu` and
  `integration`, to verify the Gradio sidecar contract:
  - adapter initialization sends a short readiness sample to `/infer_convert`
    before the adapter is marked ready;
  - requests made while startup health checks are still running immediately raise
    `BackendUnavailableError` until the readiness sample succeeds;
  - startup health-check connection and timeout failures follow the fixed retry
    contract: exactly 3 attempts, 1500ms request timeout, 500ms then 1000ms
    backoff, and no jitter, then `BackendUnavailableError`;
  - startup health-check protocol errors or invalid response shapes immediately
    raise `BackendUnavailableError` without retry;
  - successful `predict(..., api_name="/infer_convert")` returns the expected
    converted-audio response shape and decodable WAV or PCM audio at the
    configured sample rate, configured channel count, and at least 100ms
    duration;
  - `gradio_client` connection failure maps to `BackendUnavailableError`;
  - HTTP timeout maps to `BackendUnavailableError`;
  - calls made after the sidecar stops map to `BackendUnavailableError`;
  - protocol errors or partial responses from `/infer_convert` map to
    `BackendUnavailableError` without retry;
  - raw HTTP, `gradio_client`, `httpx`, or `requests` exceptions do not escape
    the adapter boundary;
  - retry instrumentation proves exactly 3 attempts for connection/timeout
    failures, a 1500ms request timeout, 500ms then 1000ms backoff delays, and no
    jitter.
- On the Windows RTX 4070 host, repeat the RVC package installability checks and
  run the sidecar contract smoke with the pinned deployment commit before
  production sign-off.
- Update `docs/deploy/windows.md` and `docs/deploy/rvc-training.md` to link to
  this ADR when the deploy commands and first pinned RVC commit are available.

## Alternatives Considered

### A. Vendor Applio Inference Subset

Rejected. First smoke is estimated at 24-40 hours because the inference path
requires `transformers`, `scipy`, `librosa`, `soxr`, `faiss-cpu`,
`torchcrepe`, `torchfcpe`, `noisereduce`, `pedalboard`, and related model glue.
The long-horizon maintenance cost is high because the project would own a
private inference fork instead of a deployable runtime boundary.

### B. `inferrvc` PyPI Package

Rejected. It still needs the `One-sixth/fairseq` fork, creates an import-time
CUDA context on `cuda:0` that breaks non-CUDA development smoke tests, and has
only one release. It does not provide a lower-risk path than the official WebUI
sidecar.

### C. Custom stdin/stdout Subprocess Protocol

Rejected. Gradio already provides an HTTP API addressable by `api_name`, so a
custom wire format would add lifecycle, parsing, error-mapping, and streaming
surface area without solving a problem the official runtime already exposes.

### D. Defer Phase 2 RVC Entirely

Rejected. Deferring RVC also defers the core identity-consistency fix described
in the architecture document. The sidecar path gives Phase 2 a bounded runtime
integration target without blocking on Python packaging cleanup across the RVC
ecosystem.

## Relationship to ADR 0001

ADR 0001 established same-process as the Irodori runtime access pattern. This
ADR does NOT reverse that for Irodori. This ADR records RVC as a sidecar because
pip installability across modern Python 3.11 + ffmpeg 8.x is empirically broken
for all surveyed RVC libraries. If a clean Python 3.11+ RVC inference package
appears, this ADR should be revisited.

This exception applies only to RVC inference. It does not authorize moving
Irodori behind a sidecar, and it does not change the lazy import/error-mapping
style required by `src/irodori_tts_infra/engine/backends/irodori.py`.

## Implementation Notes

- `src/irodori_tts_infra/engine/backends/rvc.py` currently exists only as an
  empty placeholder, and `src/irodori_tts_infra/config/settings.py` does not yet
  define `IRODORI_RVC_SIDECAR_URL`. Those are follow-up implementation tasks.
- `gradio_client` will be added to the `rvc` optional-dependency group.
- The official `RVC-Project/Retrieval-based-Voice-Conversion-WebUI` will run in
  its own uv-managed virtual environment on the Windows GPU host, separate from
  the main FastAPI/Irodori uv environment.
- The sidecar will live in a directory such as `C:\Users\user\rvc-sidecar\` with
  its own `.venv` and a pinned git clone of
  `RVC-Project/Retrieval-based-Voice-Conversion-WebUI` at a specific commit.
  Record the commit SHA here once chosen during the first deploy task tracked by
  issue `#21`:
  `<pin RVC commit on first deploy>`.
- Start the sidecar with
  `python infer-web.py --port 7865 --noautoopen`.
- Port `7865` is the default sidecar port. The main server must configure the
  full URL with `IRODORI_RVC_SIDECAR_URL`.
- The RVC adapter at `src/irodori_tts_infra/engine/backends/rvc.py` must call
  `gradio_client.Client(...).predict(..., api_name="/infer_convert")` and must
  map connection, timeout, and unavailable-sidecar failures to
  `BackendUnavailableError`.
- Sidecar start/stop automation is out of scope for this ADR and will be handled
  by future deploy commands.

### Operational requirements

- `IRODORI_RVC_SIDECAR_URL` is the configured sidecar base URL. The adapter must
  use that URL when constructing `gradio_client.Client`.
- During adapter startup, the RVC adapter must verify sidecar readiness by
  sending a short health sample to `/infer_convert` at
  `IRODORI_RVC_SIDECAR_URL`. The sample must be deterministic valid audio: WAV
  or PCM, mono, 16-bit, 16kHz, 100ms to 500ms duration, and 1KiB to 128KiB.
  Connection and timeout failures use the fixed retry contract below. A
  protocol error, invalid response shape, partial response, or undecodable audio
  response is non-retryable and immediately raises `BackendUnavailableError`.
- The adapter starts in a not-ready state and remains not ready until the
  `/infer_convert` health sample succeeds. Any runtime conversion request
  received while not ready, including while startup health-check retries are in
  progress, must immediately raise `BackendUnavailableError`. After the health
  sample succeeds, the adapter transitions to ready and may serve runtime
  conversion requests.
- Runtime calls must distinguish transient network blips from persistent
  unavailability. Startup health checks and runtime calls use the same fixed
  retry contract for connection and timeout failures: 3 attempts total,
  1500ms per-attempt timeout, exponential backoff delays of 500ms then 1000ms
  between attempts, and no jitter. After the third failed attempt, the adapter
  maps the failure to `BackendUnavailableError`.
- Persistent unavailability includes an unreachable `IRODORI_RVC_SIDECAR_URL`, a
  stopped sidecar, repeated timeout, protocol error, and invalid or partial
  `/infer_convert` response. Protocol errors and invalid or partial responses
  are non-retryable; connection and timeout failures follow the fixed retry
  contract. None of those failures may leak raw transport or Gradio client
  exceptions through the engine boundary.
- Automatic sidecar crash recovery is out of scope for this ADR. Operators must
  restart the sidecar process manually until future deploy automation provides a
  supervisor or Windows service wrapper.
