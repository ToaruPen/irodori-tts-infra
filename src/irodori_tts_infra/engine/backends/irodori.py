from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from irodori_tts_infra.config.settings import PathSettings
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import SynthesizedAudio

if TYPE_CHECKING:
    from irodori_tts_infra.config.settings import IrodoriRuntimeSettings
    from irodori_tts_infra.contracts.synthesis import SynthesisRequest

INSTALL_HINT = (
    "Install optional Irodori dependencies with: pip install 'irodori-tts-infra[irodori]'"
)

SaveWavFn = Callable[[str, object, int], object]
RequestFactory = Callable[..., object]
RuntimeKeyFactory = Callable[..., object]
RuntimeFactory = Callable[[object], "RuntimeLike"]
HfHubDownloadFn = Callable[..., str]


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
    save_wav: SaveWavFn


@runtime_checkable
class _UnloadableRuntime(Protocol):
    def unload(self) -> None: ...


class IrodoriVoiceDesignBackend:
    def __init__(
        self,
        runtime: RuntimeLike,
        settings: IrodoriRuntimeSettings,
        *,
        save_wav_fn: SaveWavFn | None = None,
        sampling_request_cls: RequestFactory | None = None,
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        inference_runtime: _InferenceRuntimeModule | None = None
        if save_wav_fn is None:
            inference_runtime = _import_inference_runtime()
            save_wav_fn = inference_runtime.save_wav
        if sampling_request_cls is None:
            if inference_runtime is None:
                inference_runtime = _import_inference_runtime()
            sampling_request_cls = inference_runtime.SamplingRequest
        self._save_wav_fn = save_wav_fn
        self._sampling_request_cls = sampling_request_cls
        self._closed = False

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        if self._closed:
            msg = "backend is closed"
            raise BackendUnavailableError(msg)

        sampling_request = self._sampling_request_cls(
            text=request.text,
            caption=request.caption,
            no_ref=request.no_ref,
            num_steps=request.num_steps,
            cfg_scale_text=request.cfg_scale_text,
            cfg_scale_caption=request.cfg_scale_caption,
            # Verified against Irodori-TTS inference_runtime.py:
            # decode_mode and context_kv_cache are SamplingRequest fields.
            decode_mode=self._settings.decode_mode,
            context_kv_cache=self._settings.context_kv_cache,
        )
        result = self._runtime.synthesize(sampling_request)
        sample_rate = int(result.sample_rate)
        wav_bytes = self._save_result_to_wav_bytes(result.audio, sample_rate)
        return SynthesizedAudio(wav_bytes=wav_bytes, sample_rate=sample_rate)

    def warm_up(self) -> None:
        request = self._sampling_request_cls(
            text=self._settings.warmup_text,
            caption=self._settings.warmup_caption,
            no_ref=True,
            num_steps=self._settings.warmup_num_steps,
            cfg_scale_text=self._settings.cfg_scale_text,
            cfg_scale_caption=self._settings.cfg_scale_caption,
            decode_mode=self._settings.decode_mode,
            context_kv_cache=self._settings.context_kv_cache,
        )
        self._runtime.synthesize(request)

    def close(self) -> None:
        if self._closed:
            return
        if isinstance(self._runtime, _UnloadableRuntime):
            self._runtime.unload()
        self._closed = True

    def _save_result_to_wav_bytes(self, audio: object, sample_rate: int) -> bytes:
        temp_dir = PathSettings().temp_wav_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=temp_dir,
                suffix=".wav",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
            self._save_wav_fn(temp_path, audio, sample_rate)
            return Path(temp_path).read_bytes()
        finally:
            if temp_path is not None:
                _unlink_temp_file(temp_path)


def create_irodori_backend(
    settings: IrodoriRuntimeSettings,
    *,
    checkpoint_filename: str = "model.safetensors",
    hf_hub_download_fn: HfHubDownloadFn | None = None,
    runtime_factory: RuntimeFactory | None = None,
    runtime_key_cls: RuntimeKeyFactory | None = None,
    save_wav_fn: SaveWavFn | None = None,
    sampling_request_cls: RequestFactory | None = None,
) -> IrodoriVoiceDesignBackend:
    try:
        download_fn = hf_hub_download_fn or _import_hf_hub_download()
        inference_runtime = _import_inference_runtime_if_needed(
            runtime_factory=runtime_factory,
            runtime_key_cls=runtime_key_cls,
            save_wav_fn=save_wav_fn,
            sampling_request_cls=sampling_request_cls,
        )
        resolved_runtime_key_cls = _runtime_key_cls(runtime_key_cls, inference_runtime)
        resolved_runtime_factory = _runtime_factory(runtime_factory, inference_runtime)
        resolved_save_wav_fn = _save_wav_fn(save_wav_fn, inference_runtime)
        resolved_sampling_request_cls = _sampling_request_cls(
            sampling_request_cls,
            inference_runtime,
        )

        checkpoint = download_fn(
            repo_id=settings.checkpoint,
            filename=checkpoint_filename,
        )
        runtime_key = resolved_runtime_key_cls(
            checkpoint=checkpoint,
            model_device=settings.model_device,
            model_precision=settings.model_precision,
            codec_device=settings.codec_device,
            codec_precision=settings.codec_precision,
            compile_model=settings.compile_model,
        )
        runtime = resolved_runtime_factory(runtime_key)
    except BackendUnavailableError:
        raise
    except ImportError as exc:
        raise BackendUnavailableError(INSTALL_HINT) from exc
    except Exception as exc:
        msg = "Failed to create Irodori backend"
        raise BackendUnavailableError(msg) from exc

    return IrodoriVoiceDesignBackend(
        runtime=runtime,
        settings=settings,
        save_wav_fn=resolved_save_wav_fn,
        sampling_request_cls=resolved_sampling_request_cls,
    )


def _import_inference_runtime_if_needed(
    *,
    runtime_factory: RuntimeFactory | None,
    runtime_key_cls: RuntimeKeyFactory | None,
    save_wav_fn: SaveWavFn | None,
    sampling_request_cls: RequestFactory | None,
) -> _InferenceRuntimeModule | None:
    if (
        runtime_factory is not None
        and runtime_key_cls is not None
        and save_wav_fn is not None
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


def _save_wav_fn(
    injected: SaveWavFn | None,
    inference_runtime: _InferenceRuntimeModule | None,
) -> SaveWavFn:
    if injected is not None:
        return injected
    return _require_inference_runtime(inference_runtime).save_wav


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


def _import_hf_hub_download() -> HfHubDownloadFn:
    try:
        module = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise BackendUnavailableError(INSTALL_HINT) from exc
    return cast("HfHubDownloadFn", module.hf_hub_download)


def _import_inference_runtime() -> _InferenceRuntimeModule:
    try:
        module = importlib.import_module("irodori_tts.inference_runtime")
    except ImportError as exc:
        raise BackendUnavailableError(INSTALL_HINT) from exc
    return cast("_InferenceRuntimeModule", module)


def _unlink_temp_file(path: str) -> None:
    with suppress(FileNotFoundError):
        os.unlink(path)  # noqa: PTH108
