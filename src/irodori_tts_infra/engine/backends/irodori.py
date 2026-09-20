from __future__ import annotations

import hashlib
import importlib
import io
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from irodori_tts_infra.contracts.synthesis import style_caption
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import SynthesizedAudio

if TYPE_CHECKING:
    from irodori_tts_infra.config.settings import IrodoriRuntimeSettings
    from irodori_tts_infra.engine.models import ResolvedSynthesisRequest

INSTALL_HINT = (
    "Irodori backend requires optional dependencies. Install: "
    "pip install irodori-tts huggingface-hub torch soundfile"
)

_MONO_AUDIO_RANK = 2

EncodeWavFn = Callable[[object, int], bytes]
RequestFactory = Callable[..., object]
RuntimeKeyFactory = Callable[..., object]
RuntimeFactory = Callable[[object], "RuntimeLike"]
HfSnapshotDownloadFn = Callable[..., str]


class RuntimeResultLike(Protocol):
    @property
    def audio(self) -> object: ...

    @property
    def sample_rate(self) -> int: ...


class RuntimeLike(Protocol):
    def synthesize(self, request: object) -> RuntimeResultLike: ...


class _InferenceRuntimeType(Protocol):
    def from_key(self, _: object) -> RuntimeLike: ...


class _InferenceRuntimeModule(Protocol):
    RuntimeKey: RuntimeKeyFactory
    SamplingRequest: RequestFactory
    InferenceRuntime: _InferenceRuntimeType


class _TorchModule(Protocol):
    float32: object


class _SoundFileModule(Protocol):
    def write(
        self,
        _file: object,
        data: object,
        samplerate: int,  # noqa: V107 - protocol parameter mirrors soundfile.write
        *,
        format: str,  # noqa: A002,V107 - protocol keyword mirrors soundfile.write
        subtype: str,  # noqa: V107 - protocol keyword mirrors soundfile.write
    ) -> object: ...


class _TensorLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def detach(self) -> _TensorLike: ...

    def to(self, **_kwargs: object) -> _TensorLike: ...

    def squeeze(self, _dim: int) -> _TensorLike: ...

    def numpy(self) -> object: ...


def _import_wav_encoder_modules() -> tuple[_TorchModule, _SoundFileModule]:
    try:
        torch = importlib.import_module("torch")
        soundfile = importlib.import_module("soundfile")
    except (ImportError, OSError) as exc:
        # soundfile and torch raise OSError when their native libraries cannot be loaded.
        raise BackendUnavailableError(INSTALL_HINT) from exc
    return cast(_TorchModule, torch), cast(_SoundFileModule, soundfile)  # noqa: TC006


def _encode_wav_bytes(audio: object, sample_rate: int) -> bytes:
    torch, soundfile = _import_wav_encoder_modules()
    tensor = (
        cast(_TensorLike, audio)  # noqa: TC006
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
    )
    # Upstream irodori_tts.inference_runtime returns SamplingResult.audio = trimmed_audios[0]: the
    # first candidate only, for any num_candidates, decoded by a mono codec as (1, samples). Fail
    # closed on anything else so a batched or samples-first tensor never becomes a multi-channel
    # WAV.
    if len(tensor.shape) != _MONO_AUDIO_RANK or tensor.shape[0] != 1:
        msg = f"Irodori runtime audio must be shaped (1, samples), got {tensor.shape}"
        raise BackendUnavailableError(msg)
    samples = tensor.squeeze(0).numpy()

    buffer = io.BytesIO()
    # The censor-beep guard only accepts 16-bit PCM, so never rely on soundfile's default.
    soundfile.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@runtime_checkable
class _UnloadableRuntime(Protocol):
    def unload(self) -> None: ...


class IrodoriBaseBackend:
    def __init__(
        self,
        runtime: RuntimeLike,
        settings: IrodoriRuntimeSettings,
        *,
        encode_wav_fn: EncodeWavFn = _encode_wav_bytes,
        sampling_request_cls: RequestFactory | None = None,
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        if sampling_request_cls is None:
            sampling_request_cls = _import_inference_runtime().SamplingRequest
        self._encode_wav_fn = encode_wav_fn
        self._sampling_request_cls = sampling_request_cls
        self._closed = False

    def synthesize(self, request: ResolvedSynthesisRequest) -> SynthesizedAudio:
        self._ensure_open()

        sampling_request = self._sampling_request_cls(
            text=request.text,
            caption=(
                request.delivery_caption
                if request.delivery_caption is not None
                else style_caption(request.style)
            ),
            ref_embed=request.ref_embed,
            num_steps=request.num_steps,
            cfg_scale_text=request.cfg_scale_text,
            cfg_scale_caption=request.cfg_scale_caption,
            cfg_scale_speaker=request.cfg_scale_speaker,
            cfg_guidance_mode="independent",
            seed=request.seed,
            duration_scale=request.duration_scale,
            num_candidates=request.num_candidates,
            t_schedule_mode=request.t_schedule_mode,
            sway_coeff=request.sway_coeff,
            decode_mode=self._settings.decode_mode,
            context_kv_cache=self._settings.context_kv_cache,
        )
        result = self._runtime.synthesize(sampling_request)
        sample_rate = int(result.sample_rate)
        wav_bytes = self._encode_wav_fn(result.audio, sample_rate)
        return SynthesizedAudio(wav_bytes=wav_bytes, sample_rate=sample_rate)

    def warm_up(self, *, ref_embed: str | None = None) -> None:
        self._ensure_open()
        if ref_embed is None or not ref_embed.strip():
            msg = "warmup ref_embed is required"
            raise BackendUnavailableError(msg)
        normalized_ref_embed = ref_embed.strip()
        request = self._sampling_request_cls(
            text=self._settings.warmup_text,
            caption=style_caption(self._settings.warmup_style),
            ref_embed=normalized_ref_embed,
            num_steps=self._settings.warmup_num_steps,
            cfg_scale_text=self._settings.cfg_scale_text,
            cfg_scale_caption=self._settings.cfg_scale_caption,
            cfg_scale_speaker=self._settings.cfg_scale_speaker,
            cfg_guidance_mode="independent",
            seed=self._settings.seed,
            duration_scale=self._settings.duration_scale,
            num_candidates=self._settings.num_candidates,
            t_schedule_mode=self._settings.t_schedule_mode,
            sway_coeff=self._settings.sway_coeff,
            decode_mode=self._settings.decode_mode,
            context_kv_cache=self._settings.context_kv_cache,
        )
        self._runtime.synthesize(request)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if isinstance(self._runtime, _UnloadableRuntime):
                self._runtime.unload()
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            msg = "backend is closed"
            raise BackendUnavailableError(msg)


def create_irodori_backend(
    settings: IrodoriRuntimeSettings,
    *,
    checkpoint_filename: str = "model.safetensors",
    snapshot_download_fn: HfSnapshotDownloadFn | None = None,
    runtime_factory: RuntimeFactory | None = None,
    runtime_key_cls: RuntimeKeyFactory | None = None,
    encode_wav_fn: EncodeWavFn = _encode_wav_bytes,
    sampling_request_cls: RequestFactory | None = None,
) -> IrodoriBaseBackend:
    if encode_wav_fn is _encode_wav_bytes:
        # Fail at startup rather than on the first synthesis request.
        _import_wav_encoder_modules()
    snapshot_fn = snapshot_download_fn or _import_snapshot_download()
    inference_runtime = _import_inference_runtime_if_needed(
        runtime_factory=runtime_factory,
        runtime_key_cls=runtime_key_cls,
        sampling_request_cls=sampling_request_cls,
    )
    resolved_runtime_key_cls = _runtime_key_cls(runtime_key_cls, inference_runtime)
    resolved_runtime_factory = _runtime_factory(runtime_factory, inference_runtime)
    resolved_sampling_request_cls = _sampling_request_cls(
        sampling_request_cls,
        inference_runtime,
    )

    try:
        snapshot_root = Path(
            snapshot_fn(
                repo_id=settings.checkpoint,
                revision=settings.checkpoint_revision,
                allow_patterns=[checkpoint_filename, "tokenizer/*"],
            )
        )
        checkpoint = snapshot_root / checkpoint_filename
        _verify_file_sha256(
            checkpoint,
            expected=settings.checkpoint_sha256,
            label="checkpoint",
        )
        _verify_bundled_tokenizer(snapshot_root, settings)
        runtime_key = resolved_runtime_key_cls(
            checkpoint=str(checkpoint),
            model_device=settings.model_device,
            model_precision=settings.model_precision,
            codec_device=settings.codec_device,
            codec_precision=settings.codec_precision,
            compile_model=settings.compile_model,
        )
        runtime = resolved_runtime_factory(runtime_key)
    except BackendUnavailableError:
        raise
    except (OSError, RuntimeError) as exc:
        msg = "Failed to create Irodori backend"
        raise BackendUnavailableError(msg) from exc

    return IrodoriBaseBackend(
        runtime=runtime,
        settings=settings,
        encode_wav_fn=encode_wav_fn,
        sampling_request_cls=resolved_sampling_request_cls,
    )


def _verify_file_sha256(path: Path, *, expected: str, label: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        msg = f"{label} is missing or unreadable: {path}"
        raise BackendUnavailableError(msg) from exc
    actual = digest.hexdigest()
    if actual != expected:
        msg = f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        raise BackendUnavailableError(msg)


def _verify_bundled_tokenizer(
    snapshot_root: Path,
    settings: IrodoriRuntimeSettings,
) -> None:
    json_pin = settings.checkpoint_tokenizer_json_sha256
    config_pin = settings.checkpoint_tokenizer_config_sha256
    if json_pin is None or config_pin is None:
        return
    tokenizer_root = snapshot_root / "tokenizer"
    _verify_file_sha256(
        tokenizer_root / "tokenizer.json",
        expected=json_pin,
        label="bundled tokenizer.json",
    )
    _verify_file_sha256(
        tokenizer_root / "tokenizer_config.json",
        expected=config_pin,
        label="bundled tokenizer_config.json",
    )


def _import_inference_runtime_if_needed(
    *,
    runtime_factory: RuntimeFactory | None,
    runtime_key_cls: RuntimeKeyFactory | None,
    sampling_request_cls: RequestFactory | None,
) -> _InferenceRuntimeModule | None:
    if (
        runtime_factory is not None
        and runtime_key_cls is not None
        and sampling_request_cls is not None
    ):
        return None
    return _import_inference_runtime()


def _runtime_key_cls(
    injected: RuntimeKeyFactory | None,
    inference_runtime: _InferenceRuntimeModule | None,
) -> RuntimeKeyFactory:
    if injected is not None:
        return injected
    return _require_inference_runtime(inference_runtime).RuntimeKey


def _runtime_factory(
    injected: RuntimeFactory | None,
    inference_runtime: _InferenceRuntimeModule | None,
) -> RuntimeFactory:
    if injected is not None:
        return injected
    return _require_inference_runtime(inference_runtime).InferenceRuntime.from_key


def _sampling_request_cls(
    injected: RequestFactory | None,
    inference_runtime: _InferenceRuntimeModule | None,
) -> RequestFactory:
    if injected is not None:
        return injected
    return _require_inference_runtime(inference_runtime).SamplingRequest


def _require_inference_runtime(
    inference_runtime: _InferenceRuntimeModule | None,
) -> _InferenceRuntimeModule:
    if inference_runtime is not None:
        return inference_runtime
    return _import_inference_runtime()


def _import_snapshot_download() -> HfSnapshotDownloadFn:
    try:
        module = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise BackendUnavailableError(INSTALL_HINT) from exc
    return cast(  # pragma: no cover - requires real huggingface_hub
        "HfSnapshotDownloadFn",
        module.snapshot_download,
    )


def _import_inference_runtime() -> _InferenceRuntimeModule:
    try:
        module = importlib.import_module("irodori_tts.inference_runtime")
    except ImportError as exc:
        raise BackendUnavailableError(INSTALL_HINT) from exc
    return cast(  # pragma: no cover - requires real Irodori runtime
        "_InferenceRuntimeModule",
        module,
    )
