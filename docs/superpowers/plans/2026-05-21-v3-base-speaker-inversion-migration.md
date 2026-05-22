# Irodori-TTS v3 Base + Speaker Inversion Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standard VoiceDesign + RVC pipeline with Irodori-TTS v3 base synthesis driven by per-character Speaker Inversion embeddings.

**Architecture:** HTTP clients send portable speaker identity (`speaker=None` for narrator, character name for dialogue). The server-side pipeline resolves that identity against its own `voice_bank_speakers.toml`, builds a backend `SynthesisRequest` with the resolved `.speaker.safetensors` `ref_embed`, and sends it directly to the Irodori backend. RVC and VoiceDesign captions are removed from the standard path. Heavy runtime dependencies remain optional and are only imported inside backend/runtime code.

**Tech Stack:** Python 3.11+, uv, pydantic, pydantic-settings, pytest, Typer, upstream `irodori_tts.inference_runtime`.

---

## Task 1: Contract And Settings

**Files:**
- Modify: `src/irodori_tts_infra/contracts/synthesis.py`
- Modify: `src/irodori_tts_infra/config/settings.py`
- Test: `tests/contracts/test_synthesis_contracts.py`
- Test: `tests/config/test_settings.py`

- [ ] **Step 1: Write failing contract tests**

Change `test_synthesis_request_defaults_and_validation()` so public
`SynthesisRequest` requires `text`, accepts optional `speaker` for HTTP, and
keeps optional `ref_embed` for the backend boundary:

```python
request = SynthesisRequest(text="こんにちは", speaker="ミカ")

assert request.speaker == "ミカ"
assert request.ref_embed is None
assert request.num_steps == 40
assert request.cfg_scale_text == pytest.approx(3.0)
assert request.cfg_scale_speaker == pytest.approx(5.0)
assert request.seed is None
assert request.duration_scale == pytest.approx(1.0)
assert request.num_candidates == 1
assert request.t_schedule_mode == "linear"
assert request.sway_coeff == pytest.approx(-1.0)

with pytest.raises(ValidationError, match="text"):
    SynthesisRequest(text="")

with pytest.raises(ValidationError, match="speaker"):
    SynthesisRequest(text="こんにちは", speaker="   ")

with pytest.raises(ValidationError, match="num_candidates"):
    SynthesisRequest(text="こんにちは", speaker="ミカ", num_candidates=0)
```

- [ ] **Step 2: Run contract test and verify RED**

Run:

```bash
uv run pytest tests/contracts/test_synthesis_contracts.py::test_synthesis_request_defaults_and_validation -q
```

Expected: fail because `speaker`, optional `ref_embed`, and `cfg_scale_speaker`
do not exist yet.

- [ ] **Step 3: Implement v3 request fields**

Replace `SynthesisRequest` fields with:

```python
text: str = Field(min_length=1)
speaker: str | None = Field(default=None, min_length=1)
ref_embed: str | None = Field(default=None, min_length=1)
num_steps: int = Field(default=40, gt=0)
cfg_scale_text: float = Field(default=3.0, gt=0.0)
cfg_scale_speaker: float = Field(default=5.0, gt=0.0)
seed: int | None = None
duration_scale: float = Field(default=1.0, gt=0.0)
num_candidates: int = Field(default=1, gt=0)
t_schedule_mode: Literal["linear", "sway"] = "linear"
sway_coeff: float = -1.0
```

Update validators to reject blank `text`, `speaker`, and `ref_embed`.

- [ ] **Step 4: Write failing settings tests**

Update settings expectations:

```python
assert runtime.checkpoint == "Aratako/Irodori-TTS-500M-v3"
assert runtime.num_steps == 40
assert runtime.cfg_scale_text == pytest.approx(3.0)
assert runtime.cfg_scale_speaker == pytest.approx(5.0)
assert runtime.duration_scale == pytest.approx(1.0)
assert runtime.num_candidates == 1
assert runtime.t_schedule_mode == "linear"
assert runtime.sway_coeff == pytest.approx(-1.0)
assert runtime.decode_mode == "sequential"
```

Add env override checks for `IRODORI_TTS_RUNTIME_CFG_SCALE_SPEAKER`, `DURATION_SCALE`, `NUM_CANDIDATES`, `T_SCHEDULE_MODE`, and `SWAY_COEFF`.

- [ ] **Step 5: Implement v3 runtime settings**

Change `IrodoriRuntimeSettings` defaults:

```python
checkpoint = "Aratako/Irodori-TTS-500M-v3"
num_steps = 40
cfg_scale_speaker = 5.0
duration_scale = 1.0
num_candidates = 1
t_schedule_mode = "linear"
sway_coeff = -1.0
decode_mode = "sequential"
```

Remove `cfg_scale_caption`, `warmup_caption`, and VoiceDesign-specific defaults.

- [ ] **Step 6: Verify Task 1**

Run:

```bash
uv run pytest tests/contracts/test_synthesis_contracts.py tests/config/test_settings.py -q
```

Expected: pass.

## Task 2: Speaker Embedding Voice Bank

**Files:**
- Modify: `src/irodori_tts_infra/voice_bank/models.py`
- Modify: `src/irodori_tts_infra/voice_bank/repository.py`
- Modify: `src/irodori_tts_infra/voice_bank/captions.py`
- Modify: `src/irodori_tts_infra/voice_bank/__init__.py`
- Test: `tests/voice_bank/test_models.py`
- Test: `tests/voice_bank/test_repository.py`
- Test: `tests/voice_bank/test_captions.py`

- [ ] **Step 1: Write failing model tests**

Add tests for `SpeakerEmbeddingProfile`:

```python
profile = SpeakerEmbeddingProfile(ref_embed=Path("speakers/mika.speaker.safetensors"))
assert profile.ref_embed == Path("speakers/mika.speaker.safetensors")
```

and update `CharacterVoice` / `VoiceProfile` construction to require `speaker` and `narrator`.

- [ ] **Step 2: Run model tests and verify RED**

Run:

```bash
uv run pytest tests/voice_bank/test_models.py -q
```

Expected: fail because `SpeakerEmbeddingProfile` does not exist.

- [ ] **Step 3: Implement voice bank dataclasses**

Replace `RVCProfile` with:

```python
@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingProfile:
    ref_embed: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref_embed", Path(self.ref_embed))

@dataclass(frozen=True, slots=True)
class CharacterVoice:
    name: str
    speaker: SpeakerEmbeddingProfile

@dataclass(frozen=True, slots=True)
class VoiceProfile:
    characters: Mapping[str, CharacterVoice]
    narrator: SpeakerEmbeddingProfile
```

- [ ] **Step 4: Write failing manifest parser tests**

Update repository tests to use `voice_bank_speakers.toml`:

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."ミカ"]
ref_embed = "speakers/mika.speaker.safetensors"
```

Assert paths resolve relative to the manifest file and absolute paths are rejected.

- [ ] **Step 5: Implement speaker manifest parsing**

Add `SPEAKER_MANIFEST_FILENAME = "voice_bank_speakers.toml"` and replace RVC parsing with speaker embedding parsing. Keep `characters.md` validation: manifest characters not present in `characters.md` are rejected when `characters.md` exists.

- [ ] **Step 6: Update captions module**

Remove VoiceDesign caption generation from the standard public surface. Keep only character-name parsing if useful:

```python
def load_characters_markdown(content: str) -> dict[str, str]:
    ...
```

Return known character names without constructing captions.

- [ ] **Step 7: Verify Task 2**

Run:

```bash
uv run pytest tests/voice_bank/test_models.py tests/voice_bank/test_repository.py tests/voice_bank/test_captions.py -q
```

Expected: pass.

## Task 3: Pipeline Without RVC

**Files:**
- Modify: `src/irodori_tts_infra/engine/models.py`
- Modify: `src/irodori_tts_infra/engine/pipeline.py`
- Modify: `src/irodori_tts_infra/engine/protocols.py`
- Test: `tests/engine/test_pipeline.py`
- Test: `tests/engine/backends/test_protocol_compliance.py`

- [ ] **Step 1: Write failing pipeline planning tests**

Update tests so narration plans with `voice_profile.narrator.ref_embed`, known dialogue plans with that character's `ref_embed`, and unknown or missing dialogue speaker raises `BackendUnavailableError` or a dedicated configuration error.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run:

```bash
uv run pytest tests/engine/test_pipeline.py -q
```

Expected: fail because pipeline still resolves captions/RVC.

- [ ] **Step 3: Update `SynthesisJob`**

Replace caption/RVC fields with speaker identity plus the backend embedding:

```python
speaker: str | None = None
ref_embed: str | None = None
require_speaker: bool = False
cfg_scale_speaker: float = 5.0
seed: int | None = None
duration_scale: float = 1.0
num_candidates: int = 1
t_schedule_mode: str = "linear"
sway_coeff: float = -1.0
```

Update `to_request()`.

- [ ] **Step 4: Remove `VoiceConverter` from standard pipeline**

Remove constructor `voice_converter`, `voice_converter` property, `_resolve_rvc()`, and conversion call from `SynthesisPipeline`.

- [ ] **Step 5: Implement speaker embedding planning**

`plan_segment()` must:

- send narrator requests with `speaker=None`;
- mark Markdown dialogue jobs as requiring a speaker;
- keep HTTP requests portable by sending the character name, not a local
  absolute `ref_embed`;
- resolve `ref_embed` on the server-side pipeline from `VoiceProfile` before the
  backend call.

- [ ] **Step 6: Verify Task 3**

Run:

```bash
uv run pytest tests/engine/test_pipeline.py tests/engine/backends/test_protocol_compliance.py -q
```

Expected: pass.

## Task 4: Irodori v3 Backend

**Files:**
- Modify: `src/irodori_tts_infra/engine/backends/irodori.py`
- Modify: `src/irodori_tts_infra/engine/backends/__init__.py`
- Test: `tests/engine/backends/test_irodori.py`
- Test: `tests/engine/backends/test_protocol_compliance.py`

- [ ] **Step 1: Write failing backend forwarding tests**

Update `FakeSamplingRequest` and `test_synthesize_forwards_sampling_request_fields()` to assert `ref_embed`, `cfg_scale_speaker`, `seed`, `duration_scale`, `num_candidates`, `t_schedule_mode`, and `sway_coeff` are forwarded, and that `caption` / `no_ref` are not required.

- [ ] **Step 2: Run backend tests and verify RED**

Run:

```bash
uv run pytest tests/engine/backends/test_irodori.py -q
```

Expected: fail because backend still sends VoiceDesign fields.

- [ ] **Step 3: Rename backend class**

Rename `IrodoriVoiceDesignBackend` to `IrodoriBaseBackend`. Keep a temporary import alias only if required by tests during this task, then remove old naming from public exports.

- [ ] **Step 4: Forward v3 SamplingRequest fields**

Build request with:

```python
text=request.text,
ref_embed=request.ref_embed,
num_steps=request.num_steps,
cfg_scale_text=request.cfg_scale_text,
cfg_scale_speaker=request.cfg_scale_speaker,
seed=request.seed,
duration_scale=request.duration_scale,
num_candidates=request.num_candidates,
t_schedule_mode=request.t_schedule_mode,
sway_coeff=request.sway_coeff,
decode_mode=self._settings.decode_mode,
context_kv_cache=self._settings.context_kv_cache,
```

- [ ] **Step 5: Update warmup behavior**

Warmup should accept a `ref_embed` value from settings or be deferred until pipeline setup can supply narrator embedding. If settings cannot supply it, remove backend warmup's default no-ref path and make missing warmup embedding a clear `BackendUnavailableError`.

- [ ] **Step 6: Verify Task 4**

Run:

```bash
uv run pytest tests/engine/backends/test_irodori.py tests/engine/backends/test_protocol_compliance.py -q
```

Expected: pass.

## Task 5: Server, Client, Deploy, And GPU Smoke

**Files:**
- Modify: `src/irodori_tts_infra/server/routers/synthesis.py`
- Modify: `src/irodori_tts_infra/deploy/remote/bootstrap.py`
- Modify: `src/irodori_tts_infra/deploy/cli.py`
- Modify: `tests/server/routers/test_synthesis.py`
- Modify: `tests/deploy/test_remote.py`
- Modify: `tests/gpu/test_phase2_e2e_smoke.py`
- Modify: `.env.example`

- [ ] **Step 1: Update server request tests**

Update synthesis route tests so payloads use `ref_embed` instead of `caption`.

- [ ] **Step 2: Update route implementation**

Forward `ref_embed` and v3 request fields into `SynthesisJob` / `SynthesisRequest`; remove `cfg_scale_caption`.

- [ ] **Step 3: Add deploy bootstrap tests**

Add test proving bootstrap creates `.runtime-venv` with Python 3.11 and installs `Irodori-TTS[cu128]` plus infra.

- [ ] **Step 4: Implement deploy bootstrap**

Add CLI options:

```text
--irodori-tts-dir
--python-version
--torch-backend-extra
```

and generate the runtime venv bootstrap script.

- [ ] **Step 5: Update GPU smoke**

Change GPU smoke to require speaker manifest and v3 base. Remove RVC sidecar preconditions and assertions.

- [ ] **Step 6: Verify Task 5**

Run:

```bash
uv run pytest tests/server/routers/test_synthesis.py tests/deploy/test_remote.py -q
uv run pytest tests/gpu/test_phase2_e2e_smoke.py::test_irodori_runtime_imports_on_gpu_host -q
```

Expected: server/deploy tests pass; GPU import smoke skips cleanly when CUDA/runtime prerequisites are absent.

## Task 6: Documentation And Final Verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/irodori-rvc-architecture.md`
- Modify: `docs/irodori-tts-optimization.md`
- Modify: `docs/deploy/windows.md`
- Modify: `README.md`

- [ ] **Step 1: Update architecture docs**

Replace VoiceDesign + RVC standard architecture with v3 base + Speaker Inversion. Mark old RVC and VoiceDesign material as superseded or remove it when it only describes the previous standard path.

- [ ] **Step 2: Update deployment docs**

Document the dedicated runtime venv and required `voice_bank_speakers.toml`.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest tests/contracts/test_synthesis_contracts.py tests/config/test_settings.py tests/voice_bank tests/engine tests/server tests/deploy -q
```

- [ ] **Step 4: Run full verification**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run vulture src/
uv run pytest
```

Expected: all pass, or any blocker is documented with exact failing command and output summary.

## Self-Review

- Spec coverage: default checkpoint, Speaker Inversion voice bank, no RVC fallback, v3 request contract, deploy runtime venv, and GPU smoke are covered.
- Placeholder scan: no placeholders or "fill in later" steps remain.
- Type consistency: `ref_embed`, `cfg_scale_speaker`, `duration_scale`, `num_candidates`, `t_schedule_mode`, and `sway_coeff` match upstream `SamplingRequest`.
