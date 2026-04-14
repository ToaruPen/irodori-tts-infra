# Phase 1 Porting Plan

**Goal:** Port the experimental novel read-aloud prototype into `irodori-tts-infra` as a tested, production-grade Irodori-TTS VoiceDesign service and client.

**Phase 1 boundary:** Irodori-TTS only. The architecture remains aligned with the final two-stage plan, but RVC, quality gates, persistent cache, and observability are deferred to Phase 2+.

**Inputs inspected:**

- Prototype: `<prototype_root>/tts/parser.py`, `characters.py`, `tts_engine.py`, `player.py`, `read_aloud.py`, `remote_server.py`, `remote_synth.py`
- Prototype workflow docs: `<prototype_root>/tts/AGENTS.md`, `<prototype_root>/tts/CLAUDE.md`, `<prototype_root>/.agent/rules/writing_guidelines.md`, `<prototype_root>/.agent/skills/read_aloud/SKILL.md`

`<prototype_root>` is the experimental repo containing the original prototype (outside this repo). Subsequent references use the same placeholder.
- Architecture docs: `docs/irodori-rvc-architecture.md`, `docs/irodori-tts-optimization.md`
- Current skeleton: `src/irodori_tts_infra/` and matching empty tests under `tests/`

## Current State

- `src/irodori_tts_infra/` already has package boundaries for `text`, `voice_bank`, `config`, `contracts`, `engine`, `server`, `client`, `cache`, and `metrics`.
- Only `src/irodori_tts_infra/__init__.py` and `tests/test_smoke.py` contain code today.
- `audio/` is mentioned by scoped instructions but does not exist yet. Phase 1 should create it only if lightweight WAV/file/stream primitives are needed.
- `pyproject.toml` exposes `irodori-tts = "irodori_tts_infra.cli:app"` even though the skeleton has `client/cli.py`, not top-level `cli.py`.

## Dependency Order

Port in this order so each layer can be tested without importing heavier layers.

| Order | Layer | Modules | Depends on | Why first/next |
|---:|---|---|---|---|
| 1 | Foundation | `config`, `contracts` | `pydantic`, `pydantic-settings` | Shared settings and API models must stabilize before engine/server/client. |
| 2 | Text primitives | `text.models`, `text.markdown`, `text.speaker_tags` | stdlib | Pure parsing can be unit-tested from prototype turn files without TTS. `text.normalization` and `text.emotion` are Phase 2+ (left as empty stubs). |
| 3 | Voice primitives | `voice_bank.models`, `voice_bank.captions`, `voice_bank.repository` | `text`, `config` | Phase 1 only needs `characters.md` loading and caption resolution. |
| 4 | Engine core | `engine.models`, `engine.protocols`, `engine.errors`, `engine.pipeline` | `config`, `contracts`, `text`, `voice_bank`, `audio.*` if factored out | Orchestrates parsed segments into backend calls using fake backends in CI. |
| 5 | Irodori backend | `engine/backends/irodori.py` | engine core, `config`, `irodori_tts`, `huggingface_hub` (optional imports) | Heavy optional imports stay isolated and GPU tests are marked. |
| 6 | Transport server | `server.*`, `server/routers/*` | `contracts`, `engine`, `config`, `fastapi`, `uvicorn`, `audio.*` if factored out | FastAPI wraps engine behavior and serves HTTP byte streams without SCP/temp-path downloads. |
| 7 | HTTP clients | `client.sync`, `client.async_`, `client.errors` | `contracts`, `config`, `httpx`, `audio.*` if factored out | Clients can target the server API with `httpx.MockTransport` in CI. |
| 8 | CLI/read-aloud | `client.cli`, read-aloud orchestration | HTTP clients, `text`, `voice_bank`, `config`, `typer`, `rich`, playback helpers if needed | CLI logic depends on text parsing and caption resolution, not just transport. |
| 9 | Remote deploy | `deploy/remote/*` | server/engine packaging decisions | Windows deployment should mirror the production server, not the old standalone script. |

## Per-Module Breakdown

### `config/`

| Item | Plan |
|---|---|
| Prototype source | Hardcoded host/port in `<prototype_root>/tts/tts_engine.py:14-23`; CLI overrides in `read_aloud.py:50-57` and `read_aloud.py:86-93`; Windows temp/model constants in `remote_server.py:18-20`; runtime guidance in `CLAUDE.md:32-36`. |
| Extract | `ClientSettings`, `ServerSettings`, `IrodoriRuntimeSettings`, `PathSettings`, `load_settings()` or direct `BaseSettings` models. Defaults should include port `8923`, checkpoint `Aratako/Irodori-TTS-500M-v2-VoiceDesign`, `num_steps=30`, `cfg_scale_text=3.0`, `cfg_scale_caption=3.5`, `model_device`, `model_precision=bf16`, `codec_device`, `codec_precision=fp32`, temp WAV directory, warmup step count/text, `decode_mode`, KV-cache settings, and compile settings. Runtime defaults should cross-check `<prototype_root>/tts/remote_server.py`, `<prototype_root>/tts/remote_synth.py`, and `docs/irodori-tts-optimization.md`. |
| Split/merge | Move prototype constants out of `TTSEngine` and `remote_server.py`. Use `pydantic-settings` env loading as the default config layer, with CLI flags overriding loaded settings rather than defining defaults independently. PR 1 must resolve the console script mismatch by either changing `pyproject.toml` to `irodori_tts_infra.client.cli:app` or moving the CLI to top-level `cli.py`. |
| Tests | Unit tests for defaults, env overrides, invalid port/path values, import safety, and installed-script smoke coverage for `irodori-tts --help`. No network, filesystem writes, or Irodori imports. |
| Complexity | **M**. The code is small, but settings must stay explicit and avoid import side effects. |

### `contracts/`

| Item | Plan |
|---|---|
| Prototype source | Request/response dicts in `tts_engine.py:40-52`, `tts_engine.py:54-85`, `tts_engine.py:87-128`; server JSON handling in `remote_server.py:76-113`; stream header format in `remote_server.py:23-25`; health response in `remote_server.py:139-145`. |
| Extract | `SynthesisRequest`, `SynthesisSegment`, `BatchSynthesisRequest`, `SynthesisResult`, `BatchSynthesisResult`, `StreamHandshakeHeader`, `StreamChunkHeader`, `HealthResponse`, `VoiceProfileResponse`, shared error payloads. HTTP byte streaming is the Phase 1 audio transport contract: `/synthesize_stream` emits ordered framed WAV byte chunks; JSON responses must not expose server temp paths or require SCP download. |
| Split/merge | Replace ad hoc JSON dicts with pydantic models. `HealthResponse` replaces the prototype's plain `ok` text and MUST include `max_chunk_size: int` so clients can discover the server-advertised cap before streaming. **Stream framing spec (locked for Phase 1):** JSON-line header terminated by a single `\n` byte, followed by exactly `byte_length` bytes of WAV payload (for payload headers only). All integers are JSON numbers (portable, no endianness concerns). Two distinct header models sharing the same line format but disjoint fields disambiguate handshake from payload — parsers MUST route by `kind`: **`StreamHandshakeHeader`** `{"kind":"handshake","header_version":1,"max_chunk_size":N}` carries NO payload, MUST appear at most once, MUST be the first frame of a connection if present, MUST NOT use `segment_index` / `byte_length` / `final`; **`StreamChunkHeader`** `{"kind":"chunk","header_version":1,"segment_index":I,"byte_length":L,"final":B}` carries exactly `byte_length` payload bytes immediately after its line terminator. `segment_index` is monotonic per request, range `[0, 2^32)`, and starts at 0 for the first payload chunk; handshake presence does NOT shift or consume a `segment_index` slot. `byte_length` range `[0, max_chunk_size]`. `max_chunk_size` default: 4 MiB (servers MAY lower; clients MUST accept any value up to the advertised cap). Clients MUST discover the cap via `/health.max_chunk_size`; when a `StreamHandshakeHeader` is present, clients MUST prefer its `max_chunk_size` over the `/health` value for that connection. Header version bumps require explicit contract migration. |
| Tests | Unit tests for model validation, JSON round-trip, default synthesis parameters, ordered batch results, and stream header serialization. Byte-exact reconstruction test: synthesize a mock WAV, frame it into multiple chunks, parse the framed stream, assert reassembled bytes match source byte-for-byte. Interop test: server-side framer output fed into client-side parser via in-memory buffer, including a path with `StreamHandshakeHeader` + subsequent `StreamChunkHeader` frames. Boundary tests: `byte_length == 0`, `byte_length == max_chunk_size`. Handshake-specific tests MUST cover: handshake accepted as the first frame, subsequent payload chunks still required to have monotonic unique `segment_index` starting at 0, duplicate handshake rejected, handshake after payload rejected. Negative tests MUST cover: malformed header JSON, missing `\n` separator, payload shorter than `byte_length` (truncation), payload longer than `byte_length` (surplus bytes), oversized `byte_length` exceeding the connection's effective `max_chunk_size`, non-monotonic / gap / duplicate `segment_index`, unknown `kind`, and unknown `header_version`. All negative cases must fail fast with a parser error rather than silently accepting bytes. |
| Complexity | **S-M**. Straightforward schema work once the HTTP byte-streaming contract is fixed. |

### `text/`

| Item | Plan |
|---|---|
| Prototype source | `Segment` and `parse_turn()` in `parser.py:14-54`; metadata stripping in `parser.py:57-65`; narration flushing in `parser.py:68-71`; speaker tag format docs in `<prototype_root>/.agent/rules/writing_guidelines.md:38-70` and `<prototype_root>/.agent/skills/read_aloud/SKILL.md:64-104`. |
| Extract | `Segment`, `SegmentKind`, `SpeakerTag`, `parse_turn_markdown()`, `strip_turn_metadata()`, `parse_speaker_tag()`, `is_skippable_markdown_line()`. |
| Split/merge | Put markdown traversal in `text/markdown.py`; put `【Name:direction】「dialogue」` parsing in `text/speaker_tags.py`; keep models in `text/models.py`. `normalization.py` can hold Irodori-compatible punctuation normalization later, but Phase 1 should not duplicate Irodori preprocessing unless needed. |
| Tests | Unit tests for tagged dialogue, optional direction, bare dialogue, headings, `---` separators, metadata after `「「🏷️情報」:`, inner Japanese quotes such as `「『相性表』を見ましたか？」`, empty files, multiline narration joining, and malformed tags becoming narration. |
| Complexity | **S**. The prototype parser is compact and pure. |

### `voice_bank/`

| Item | Plan |
|---|---|
| Prototype source | Caption constants in `characters.py:6-8`; character block parser in `characters.py:51-91`; heuristic caption builder in `characters.py:94-147`; character file discovery and caption resolution in `read_aloud.py:14-48`; voice assignment docs in `<prototype_root>/.agent/skills/read_aloud/SKILL.md:106-113`. |
| Extract | `CharacterVoice`, `VoiceProfile`, `load_characters_markdown()`, `build_voicedesign_caption()`, `resolve_segment_caption()`, `find_characters_markdown()`. |
| Split/merge | Put caption heuristics in `captions.py`; repository/path discovery in `repository.py`; shared dataclasses/pydantic models in `models.py`. Phase 1 preserves the current caption heuristics with tests and later explicit overrides can layer on top. Defer durable manifests, aliases, narrator storage schema, and RVC model references to Phase 2+. Avoid carrying the prototype's broad heading skip rule from `characters.py:73-75` without tests. |
| Tests | Unit tests for `characters.md` parsing, headings with parenthetical readings, attr keys like `年齢`, `年齢/外見`, `性格`, gender/age/personality detection, narrator fallback, known speaker captions, unknown speaker captions, directed-dialogue caption injection, and missing character files. Integration tests can use small sanitized fixture `characters.md` files. |
| Complexity | **M**. The parser is small, but caption behavior needs regression coverage. |

### `audio/` (Create If Needed)

| Item | Plan |
|---|---|
| Prototype source | Temp WAV writes in `tts_engine.py:130-147`; legacy SCP download temp files in `tts_engine.py:149-165`; streaming byte writes/deletion in `remote_server.py:131-136`; macOS playback in `player.py:7-14`. |
| Extract | Minimal `WavBytes`, `WavFile`, `write_temp_wav()`, `safe_unlink()`, and maybe `AudioPlaybackCommand` only when a later PR needs shared helpers. |
| Split/merge | Do not create a standalone audio PR. Fold these helpers into the first engine, server, client, or CLI PR that needs them. Keep playback process calls in `client/sync.py` or `client/cli.py`, not core audio. Keep DSP, resampling, and quality analysis out of Phase 1. |
| Tests | Unit tests for temp file creation/deletion and byte preservation. Subprocess playback tests should mock `subprocess.run`; real `afplay` is not a default test. |
| Complexity | **S**. Useful only as a small boundary to prevent WAV temp-file logic from spreading. |

### `engine/`

| Item | Plan |
|---|---|
| Prototype source | Client-side batch/stream orchestration in `tts_engine.py:54-128`; server synthesis primitive in `remote_server.py:54-74`; runtime single-call wrapper in `remote_synth.py:36-51`. |
| Extract | `Synthesizer` protocol, `SynthesisJob`, `SynthesizedAudio`, `SynthesisPipeline`, `synthesize_segments()`, `EngineError`, `BackendUnavailableError`. |
| Split/merge | `engine/pipeline.py` should map parsed segments plus voice profiles into backend jobs. `engine/backends/*` should only implement concrete synthesis. Default GPU/runtime policy is one in-flight backend synthesis per runtime, enforced with a lock or capacity-one limiter plus bounded acquire timeout/backpressure. Do not put FastAPI, HTTP client code, SSH, SCP, or playback here. |
| Tests | Unit tests with fake synthesizer for ordering, per-segment caption assignment, backend error propagation, concurrent request serialization/backpressure, and no import of heavy runtime at package import. |
| Complexity | **M**. Most behavior is orchestration and can be tested with fakes. |

### `engine/backends/irodori.py`

| Item | Plan |
|---|---|
| Prototype source | Runtime loading in `remote_server.py:28-37`; `_synthesize_one()` in `remote_server.py:54-74`; warmup in `remote_server.py:157-163`; alternate singleton runtime in `remote_synth.py:16-33`; optimization defaults in `docs/irodori-tts-optimization.md:3-20`. |
| Extract | `IrodoriVoiceDesignBackend`, lazy `load_runtime()`, `build_sampling_request()`, `warm_up()`, `close()`. |
| Split/merge | Keep `huggingface_hub` and `irodori_tts` imports inside this module or method bodies. The backend should return bytes/path objects, not write directly into HTTP responses. Replace global class attrs (`remote_server.py:40-42`) and prototype `TTSHandler.gpu_lock` with instance state managed by server lifespan and the engine serialization policy. |
| Tests | Unit tests can monkeypatch a fake runtime and fake `save_wav`. Real model tests must be marked `gpu` and skipped by default. Add protocol compliance and fake-runtime concurrent request tests without importing Irodori. |
| Complexity | **L**. Runtime install/access pattern is unresolved, and this is the first heavy optional dependency boundary. |

### `engine/backends/rvc.py`

| Item | Plan |
|---|---|
| Prototype source | None in prototype. Architecture target described in `docs/irodori-rvc-architecture.md:80-108`. |
| Extract | Phase 1 should only preserve the empty module if the package skeleton keeps it. |
| Split/merge | Do not implement RVC conversion, model loading, training, or routing in Phase 1. |
| Tests | Import-only smoke if the module remains empty. No behavior tests yet. |
| Complexity | **S** for Phase 1 because it is intentionally deferred. |

### `server/`

| Item | Plan |
|---|---|
| Prototype source | HTTP dispatch in `remote_server.py:40-52`; single/batch handlers in `remote_server.py:76-113`; stream handler in `remote_server.py:115-138`; health in `remote_server.py:139-146`; startup/shutdown in `remote_server.py:152-176`. |
| Extract | `create_app()`, FastAPI lifespan runtime loading/unloading, dependency providers for pipeline/backend, `health`, `synthesis`, and `voices` routers, HTTP exception mapping. |
| Split/merge | Replace `http.server` with FastAPI. Keep route handlers thin: validate contracts, call pipeline/backend, return contracts or HTTP byte streaming responses. Avoid direct model loading in router files. Do not expose server-side temp paths or SCP cleanup in Phase 1 transport. Map engine backpressure/timeouts to explicit HTTP errors. |
| Tests | FastAPI `TestClient` tests with fake pipeline for `/health`, `/synthesize`, `/synthesize_batch`, `/synthesize_stream`, validation errors, stream byte preservation/order, and backend failure/backpressure mapping. Real Irodori server tests are `integration`/`gpu`. |
| Complexity | **M-L**. FastAPI setup is standard, but stream format, backpressure, and lifespan-heavy runtime handling need careful tests. |

### `client/`

| Item | Plan |
|---|---|
| Prototype source | HTTP client and streaming reader in `tts_engine.py:18-52` and `tts_engine.py:87-128`; playback in `player.py:7-14`; CLI flow in `read_aloud.py:50-123`; read-aloud UX in `<prototype_root>/.agent/skills/read_aloud/SKILL.md:23-63`. |
| Extract | `SyncIrodoriClient`, `AsyncIrodoriClient`, `ReadAloudOptions`, Typer CLI `app`, ordered stream playback/save logic, client errors. |
| Split/merge | `tts_engine.py` currently mixes HTTP, SSH/SCP cleanup, temp files, and progress printing. Split sync/async HTTP clients from CLI read-aloud orchestration. Use `httpx`, not `urllib.request`; do not retain SCP/temp-path download in Phase 1 normal flow. |
| Tests | Unit tests with `httpx.MockTransport` for health, single, batch, stream framing, byte-exact stream reconstruction, and timeout/error mapping. CLI tests with `typer.testing.CliRunner`. Playback tests mock `subprocess.run` and verify segment-order playback even if mocked stream chunks arrive out of order. |
| Complexity | **M**. Most logic is testable, but preserving ordered playback while receiving out-of-order stream chunks needs focused tests. |

### `deploy/remote/`

| Item | Plan |
|---|---|
| Prototype source | One-shot remote script in `remote_synth.py:1-67`; persistent server startup instructions in `remote_server.py:1-4`; old deployment notes in `tts/AGENTS.md:29-33`. |
| Extract | Windows launch script or docs that run the production FastAPI app with the Irodori backend. Optional helper for copying config to the GPU host. |
| Split/merge | Do not keep a second implementation of synthesis in deploy scripts. Deployment should invoke package entry points. |
| Tests | No network tests in default CI. Keep syntax/import checks and document manual smoke commands. SSH tests should be marked `ssh` and `integration`. |
| Complexity | **M**. Operationally important, but should stay thin once server/backend are done. |

### `cache/`

| Item | Plan |
|---|---|
| Prototype source | None. Prototype always synthesizes and uses temp WAV files. |
| Extract | Nothing for Phase 1 beyond keeping imports stable. |
| Split/merge | Do not add content-addressed cache policy in Phase 1. |
| Tests | Existing empty tests can remain until cache work starts. |
| Complexity | **S** for Phase 1 because it is out of scope. |

### `metrics/`

| Item | Plan |
|---|---|
| Prototype source | None. Final quality-gate design is in `docs/irodori-rvc-architecture.md:109-140`. |
| Extract | Nothing for Phase 1 beyond keeping imports stable. |
| Split/merge | Do not add WavLM, ECAPA-TDNN, F0, UTMOS/DNSMOS, or Prometheus/OpenTelemetry behavior in Phase 1. |
| Tests | Existing empty tests can remain until Phase 2+. |
| Complexity | **S** for Phase 1 because it is out of scope. |

## Incremental PR Strategy

Keep each PR independently reviewable and under roughly 500 net source lines where possible.

| PR | Scope | Tests | Notes |
|---:|---|---|---|
| 1 | Contracts and settings foundation: `contracts/*`, `config/*`, resolve `irodori-tts` package entry point. | `tests/contracts/*`, `tests/config/*`, smoke import, installed-script `irodori-tts --help`. | No FastAPI, no engine, no file parsing. The CLI target must be either `irodori_tts_infra.client.cli:app` or a real top-level `cli.py`. |
| 2 | Text parsing port: `text/models.py`, `text/markdown.py`, `text/speaker_tags.py`. | `tests/text/test_markdown.py`, `tests/text/test_speaker_tags.py`. | Use sanitized fixtures that cover speaker tags and metadata stripping. |
| 3 | Voice bank caption port: `voice_bank/models.py`, `voice_bank/captions.py`, `voice_bank/repository.py`. | `tests/voice_bank/test_captions.py`, repository path tests with temp dirs. | Only `characters.md` parsing and narrator/known/unknown/directed caption resolution. |
| 4 | Engine protocols and pipeline with fake backend. | `tests/engine/test_pipeline.py`, `tests/engine/backends/test_protocol_compliance.py`. | No Irodori import required. Include fake-runtime concurrent request serialization/backpressure tests. Add audio helpers here only if the engine actually needs them. |
| 5 | Irodori backend adapter. | Unit tests with monkeypatched runtime; optional `gpu` smoke test skipped by default. | Resolve runtime access pattern before this PR starts. |
| 6 | FastAPI server shell and routers with fake engine. | `tests/server/test_app.py`, `tests/server/routers/*`. | Server should pass CI without GPU or Irodori installed and must expose HTTP byte streaming. Add audio helpers here only if the server actually needs them. |
| 7 | Sync/async HTTP clients. | `tests/client/test_sync_client.py`, `tests/client/test_async_client.py`. | Use `httpx.MockTransport`; cover stream framing and byte-exact reconstruction. Add audio helpers here only if clients actually need them. |
| 8 | CLI read-aloud orchestration and playback. | CLI tests with `typer.testing.CliRunner`; mocked playback/subprocess tests. | Depends on HTTP clients, text parsing, and caption resolution. Verify ordered playback even when chunks arrive out of order. |
| 9 | Remote deployment wrapper and manual smoke docs. | Lint/import checks; optional `integration`/`ssh` test skeleton. | Should not duplicate server synthesis logic. |

## Risks and Decisions Needed

1. **Irodori-TTS runtime access pattern**
   - Prototype docs say direct Python API via `sys.path`/`IRODORI_TTS_DIR` (`CLAUDE.md:32-36`).
   - Active prototype server imports `irodori_tts` as an installed package (`remote_server.py:15-17`).
   - Decision needed: package install, `sys.path` injection, subprocess isolation, or a remote venv wrapper.
   - Blocks only PR 5, the Irodori backend adapter. Earlier PRs should use fake backends/runtimes.

## Phase 1 Does Not Include

- RVC model loading, voice conversion, training data extraction, or per-character RVC manifests.
- Durable voice-bank manifests, aliases, or RVC model references.
- SCP/temp-path audio download compatibility unless a later explicit compatibility requirement retains it.
- Multi-metric quality gates: WavLM, ECAPA-TDNN, F0, UTMOS, DNSMOS, speaking-rate, pause, or energy analysis.
- Persistent content-addressed cache.
- OpenTelemetry tracing, Prometheus metrics, dashboards, or production alerting.
- Automated candidate reranking, threshold calibration, or human review workflows.
- Dataset downloads, model checkpoint management beyond Irodori VoiceDesign runtime loading.
- Real GPU/SSH checks in default CI.
- UI work beyond a CLI client.

## Phase 1 Acceptance Criteria

- A turn markdown file can be parsed into ordered narration/dialogue segments.
- A `characters.md` file can resolve narrator, known speaker, unknown speaker, and directed-dialogue captions.
- The engine pipeline can synthesize ordered segments through a fake backend in CI.
- The Irodori backend can run behind a `gpu` marker or documented manual smoke command without affecting default CI.
- FastAPI exposes health and synthesis endpoints using shared contracts, including `HealthResponse` instead of the prototype plain `ok`. `HealthResponse` advertises the server's current `max_chunk_size` so clients can size their buffers before streaming.
- `/synthesize_batch` returns ordered results.
- `/synthesize_stream` framing preserves byte-exact WAV bytes and chunk order, verified by a server-framer ↔ client-parser interop test exercising header versioning, `byte_length` boundaries, and `max_chunk_size` enforcement. The parser must fail fast on malformed headers, missing separators, truncated / surplus payloads, oversized chunks, non-monotonic or duplicate `segment_index`, and unknown `header_version` (covered by negative tests).
- Engine enforces capacity-one GPU/runtime concurrency: at most one in-flight synthesis per runtime, acquire attempts beyond that either block up to a bounded timeout or surface a typed backpressure error, verified with fake-runtime tests driving concurrent requests.
- Sync/async clients can call the API and save or play returned WAVs with subprocess calls mocked in tests.
- CLI read-aloud plays chunks in segment order even when stream chunks arrive out of order.
- The installed console script `irodori-tts --help` runs successfully.
- Per-PR checks run the narrowest useful validation first. Full verification before Phase 1 completion is `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run vulture src/ && uv run pytest`.
