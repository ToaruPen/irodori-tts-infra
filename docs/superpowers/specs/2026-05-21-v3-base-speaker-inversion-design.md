# Irodori-TTS v3 Base + Speaker Inversion Design

## Decision

Switch the standard synthesis architecture from:

```text
Text -> Irodori-TTS v2 VoiceDesign -> RVC
```

to:

```text
Text + character speaker embedding -> Irodori-TTS v3 base -> audio
```

There is no backward compatibility requirement for the old VoiceDesign/RVC data model. Existing names and APIs should be changed when keeping them would make the new architecture unclear.

## Goals

- Make `Aratako/Irodori-TTS-500M-v3` the default Irodori checkpoint.
- Make per-character Speaker Inversion embeddings the required voice identity source.
- Remove RVC from the standard pipeline.
- Remove caption as a required synthesis input.
- Expose the v3 inference controls that are needed immediately: `ref_embed`, `seed`, `duration_scale`, `num_candidates`, `t_schedule_mode`, `sway_coeff`, `cfg_scale_text`, `cfg_scale_speaker`, `decode_mode`, and `context_kv_cache`.
- Keep normal unit tests independent of real Irodori, PyTorch, model weights, GPU, network, and Hugging Face access.
- Keep real runtime checks under `gpu` / `integration` marked tests.

## Non-Goals

- Do not support the old VoiceDesign-first contract in the same API shape.
- Do not keep RVC fallback behavior in `SynthesisPipeline`.
- Do not add a generalized multi-backend abstraction layer.
- Do not train Speaker Inversion embeddings in this change. This work only consumes `.speaker.safetensors` files.
- Do not make quality claims until the Windows GPU smoke and listening checks run.

## Architecture

The pipeline becomes a single synthesis step. `SynthesisPipeline` resolves the segment's speaker embedding from the voice bank, builds a `SynthesisJob`, and sends one `SynthesisRequest` to the Irodori backend. The backend creates upstream `SamplingRequest` with `ref_embed` and v3 sampling parameters, then writes the returned audio to WAV bytes.

`RVCProfile`, `VoiceConverter`, and `_resolve_rvc()` are removed from the standard pipeline path. If any RVC code remains temporarily in the repository during migration, it must be isolated from the default flow and not referenced by new tests or docs.

## Data Model

Introduce a Speaker Inversion voice model:

```python
@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingProfile:
    ref_embed: Path

@dataclass(frozen=True, slots=True)
class CharacterVoice:
    name: str
    speaker: SpeakerEmbeddingProfile

@dataclass(frozen=True, slots=True)
class VoiceProfile:
    characters: Mapping[str, CharacterVoice]
    narrator: SpeakerEmbeddingProfile
```

Every dialogue speaker must resolve to a known `CharacterVoice`. Narration uses `VoiceProfile.narrator`. Missing embeddings are configuration errors, not fallbacks.

Manifest format should replace `voice_bank_rvc.toml` with a v3-specific manifest:

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."ミカ"]
ref_embed = "speakers/mika.speaker.safetensors"
```

Path values remain relative to the manifest file. Absolute paths are rejected for repository hygiene and portability.

`characters.md` may still provide the known character names, but it no longer generates VoiceDesign captions. If both files are present, the speaker manifest must not define characters absent from `characters.md`.

## Request Contract

The public HTTP `SynthesisRequest` should be v3-oriented while remaining
portable across client and server machines. Clients send a narrator request with
`speaker = None`, or a dialogue request with a character name from the server's
voice bank. The server resolves that speaker against its own
`voice_bank_speakers.toml` before calling the backend. Direct backend requests
receive the resolved `ref_embed`.

```python
class SynthesisRequest(_ContractModel):
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

Remove from the standard contract:

- `caption`
- `cfg_scale_caption`
- `no_ref`

`caption` can be reintroduced later only if a v3 VoiceDesign release or a separate style-control design needs it.

## Runtime Settings

`IrodoriRuntimeSettings.checkpoint` defaults to `Aratako/Irodori-TTS-500M-v3`.

Runtime defaults:

```python
num_steps = 40
cfg_scale_text = 3.0
cfg_scale_speaker = 5.0
duration_scale = 1.0
num_candidates = 1
t_schedule_mode = "linear"
sway_coeff = -1.0
decode_mode = "sequential"
context_kv_cache = True
```

`decode_mode` should default to `sequential` for lower VRAM pressure with candidates. GPU deployments can set `batch` explicitly after validation.

Warmup uses narrator embedding, not `no_ref=True`. Therefore runtime or server setup must be able to locate the voice profile before warmup. If no narrator embedding exists, startup should fail with a clear configuration error.

## Pipeline Behavior

Planning rules:

- Narration: use `VoiceProfile.narrator.ref_embed`.
- Dialogue with known speaker: use `VoiceProfile.characters[segment.speaker].speaker.ref_embed`.
- Dialogue without a speaker: fail.
- Dialogue with an unknown speaker: fail.

Segment direction remains text-side metadata only. It may be folded into the input text if a later design proves that is useful, but this change does not invent a prompt format for directions.

## Backend Behavior

`IrodoriVoiceDesignBackend` should be renamed or replaced with a v3-neutral name such as `IrodoriBaseBackend`.

The backend must forward:

```python
text=request.text
ref_embed=request.ref_embed
num_steps=request.num_steps
cfg_scale_text=request.cfg_scale_text
cfg_scale_speaker=request.cfg_scale_speaker
seed=request.seed
duration_scale=request.duration_scale
num_candidates=request.num_candidates
t_schedule_mode=request.t_schedule_mode
sway_coeff=request.sway_coeff
decode_mode=settings.decode_mode
context_kv_cache=settings.context_kv_cache
```

The backend must not send `caption` or `no_ref` in the default path.

## Deployment

Keep a Python 3.11 runtime environment for infra compatibility. Upstream Irodori-TTS currently has `.python-version` set to `3.10`, so deploy bootstrap must create a dedicated runtime venv instead of relying on `uv run` inside the Irodori-TTS project.

The Windows GPU host bootstrap should install both projects into the same runtime venv:

```powershell
uv venv .runtime-venv --python 3.11 --clear
uv pip install --python .runtime-venv/Scripts/python.exe -e "C:/Irodori-TTS[cu128]" -e "C:/irodori-tts-infra[server,irodori]"
uv pip check --python .runtime-venv/Scripts/python.exe
```

macOS can keep the local validation venv at:

```text
/Users/sankenbisha/Dev/.venvs/irodori-tts-infra-runtime
```

## Testing

Unit tests:

- Contract defaults and validation for v3 fields.
- Voice bank manifest parsing with relative path enforcement.
- Pipeline planning for narration, known dialogue speaker, missing speaker, and unknown speaker.
- Backend forwarding into fake `SamplingRequest`.
- Import-light behavior remains: importing config/server/pipeline must not import real `irodori_tts`, `torch`, or Hugging Face runtime modules.

GPU / integration tests:

- Real runtime import smoke under the dedicated runtime venv.
- v3 base synthesis smoke with a configured narrator `.speaker.safetensors`.
- Multi-segment pipeline smoke after at least one character embedding exists.

## Documentation Updates

Update project docs to say:

- The architecture is v3 base + Speaker Inversion, not VoiceDesign + RVC.
- VoiceDesign v2 and RVC are superseded for this repository's standard path.
- Voice bank setup requires `.speaker.safetensors` files.
- GPU smoke requires real embeddings and model access.

The existing plan at `docs/superpowers/plans/2026-05-21-irodori-tts-v3-runtime-followup.md` should be replaced or revised because it assumes VoiceDesign remains the default.

## Risks

- Speaker Inversion embeddings may not exist yet, so the first GPU smoke may be blocked on creating one.
- v3 base may not match VoiceDesign's caption-driven expressiveness. This is acceptable for the architecture switch but must be evaluated with listening tests later.
- Removing RVC simplifies the pipeline but removes the previous identity correction stage.
- Upstream Irodori-TTS dependency resolution can change; deployment must validate with `uv pip check`.

## Acceptance Criteria

- `IrodoriRuntimeSettings().checkpoint == "Aratako/Irodori-TTS-500M-v3"`.
- Public `SynthesisRequest` requires `text` and accepts `speaker`; server-side
  pipeline resolution supplies `ref_embed` before the backend call.
- `SynthesisPipeline` has no standard `VoiceConverter` or RVC conversion path.
- Voice bank manifests provide narrator and character Speaker Inversion embedding paths.
- Unit tests pass without real Irodori-TTS installed.
- Dedicated runtime venv can import both `irodori_tts.inference_runtime` and `irodori_tts_infra`.
- GPU smoke either synthesizes audio with v3 base or skips with a precise missing-precondition message.

## Self-Review

- Placeholder scan: no placeholder sections or ambiguous future-only requirements remain.
- Internal consistency: the standard path is consistently v3 base + Speaker Inversion with no RVC fallback.
- Scope check: this is a large but cohesive architecture migration; implementation should be split into model/contract, pipeline, backend, deploy, docs, and GPU smoke tasks.
- Ambiguity check: missing speakers and missing embeddings fail instead of falling back.
