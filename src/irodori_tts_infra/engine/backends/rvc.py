"""RVC WebUI backend pinned to the official Gradio contract.

Pinned upstream contract (re-verify before any sidecar upgrade):
- `/infer_convert` binding:
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/7ef19867780cf703841ebafb565a4e47d1ea86ff/infer-web.py#L948-L967
- `/infer_change_voice` binding:
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/7ef19867780cf703841ebafb565a4e47d1ea86ff/infer-web.py#L1107-L1111
- `vc_single` signature and return shape:
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/7ef19867780cf703841ebafb565a4e47d1ea86ff/infer/modules/vc/modules.py#L146-L225

Official current main does not accept `model_path` directly on `/infer_convert`.
The backend must first load `weight_root/sid` through `/infer_change_voice`
using the dropdown value, then call `/infer_convert` with the pinned positional
signature.

This adapter does not resample output audio. The sidecar is expected to return
audio at `profile.sample_rate`; any mismatch is treated as a backend contract
failure.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import wave
from array import array
from collections.abc import Callable, Iterable
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import structlog

from irodori_tts_infra.config.settings import PathSettings, RVCSidecarSettings
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import SynthesizedAudio

if TYPE_CHECKING:
    from irodori_tts_infra.voice_bank.models import RVCProfile

INSTALL_HINT = "RVC backend requires optional dependencies. Install: pip install gradio_client"
_CHANGE_VOICE_API_NAME = "/infer_change_voice"
_PINNED_SPEAKER_ID = 0
_PINNED_F0_UP_KEY = 0
_PINNED_F0_FILE = None
_PINNED_F0_METHOD = "harvest"
_PINNED_FILE_INDEX = ""
_PINNED_FILE_INDEX2 = ""
_PINNED_INDEX_RATE = 0.88
_PINNED_FILTER_RADIUS = 3
_PINNED_RMS_MIX_RATE = 1
_PINNED_PROTECT = 0.33
_PAIR_LEN = 2

logger = structlog.get_logger(__name__)

ClientFactory = Callable[..., "GradioClientLike"]


class GradioClientLike(Protocol):
    def view_api(self) -> object: ...

    def predict(self, *_args: object, api_name: str) -> object: ...


@runtime_checkable
class _ClosableClient(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class _SessionCloseable(Protocol):
    @property
    def session(self) -> object: ...


@runtime_checkable
class _ListConvertible(Protocol):
    def tolist(self) -> object: ...


class RVCConverter:
    def __init__(
        self,
        client: GradioClientLike,
        settings: RVCSidecarSettings,
        *,
        temp_wav_dir: Path,
    ) -> None:
        self._client = client
        self._settings = settings
        self._temp_wav_dir = temp_wav_dir
        self._closed = False

    def warm_up(self) -> None:
        self._ensure_open()
        logger.debug("rvc_sidecar_warm_up", url=str(self._settings.url))
        try:
            self._client.view_api()
        except _client_errors() as exc:
            msg = "RVC sidecar warm-up failed"
            raise BackendUnavailableError(msg) from exc

    def convert(self, audio: SynthesizedAudio, *, profile: RVCProfile) -> SynthesizedAudio:
        self._ensure_open()

        temp_path = _write_temp_input_wav(audio, self._temp_wav_dir)
        response: object
        try:
            self._load_profile(profile)
            response = self._predict_convert(temp_path, profile)
        except BackendUnavailableError:
            raise
        except _client_errors() as exc:
            msg = "RVC conversion failed"
            raise BackendUnavailableError(msg) from exc
        finally:
            _unlink_temp_file(temp_path)

        return _response_to_audio(response, profile)

    def close(self) -> None:
        if self._closed:
            return
        try:
            closer = _resolve_closer(self._client)
            if closer is not None:
                closer()
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            msg = "backend is closed"
            raise BackendUnavailableError(msg)

    def _load_profile(self, profile: RVCProfile) -> None:
        logger.debug("rvc_sidecar_load_profile", sid=profile.model_path.name)
        self._client.predict(
            profile.model_path.name,
            _PINNED_PROTECT,
            _PINNED_PROTECT,
            api_name=_CHANGE_VOICE_API_NAME,
        )

    def _predict_convert(self, input_audio_path: str, profile: RVCProfile) -> object:
        logger.debug(
            "rvc_sidecar_convert",
            api_name=self._settings.api_name,
            sample_rate=profile.sample_rate,
        )
        return self._client.predict(
            _PINNED_SPEAKER_ID,
            input_audio_path,
            _PINNED_F0_UP_KEY,
            _PINNED_F0_FILE,
            _PINNED_F0_METHOD,
            _PINNED_FILE_INDEX,
            _PINNED_FILE_INDEX2,
            _PINNED_INDEX_RATE,
            _PINNED_FILTER_RADIUS,
            profile.sample_rate,
            _PINNED_RMS_MIX_RATE,
            _PINNED_PROTECT,
            api_name=self._settings.api_name,
        )


def create_rvc_backend(
    settings: RVCSidecarSettings,
    *,
    client_factory: ClientFactory | None = None,
) -> RVCConverter:
    resolved_client_factory = client_factory or _import_gradio_client()
    try:
        client = resolved_client_factory(
            src=str(settings.url),
            httpx_kwargs={"timeout": _timeout_for(settings)},
        )
    except BackendUnavailableError:
        raise
    except _client_errors() as exc:
        msg = "Failed to create RVC backend"
        raise BackendUnavailableError(msg) from exc

    return RVCConverter(
        client=client,
        settings=settings,
        temp_wav_dir=PathSettings().temp_wav_dir,
    )


def _import_gradio_client() -> ClientFactory:
    try:
        import gradio_client  # type: ignore[import-not-found,import-untyped,unused-ignore] # noqa: PLC0415
    except ImportError as exc:
        raise BackendUnavailableError(INSTALL_HINT) from exc
    return cast("ClientFactory", gradio_client.Client)


def _timeout_for(settings: RVCSidecarSettings) -> object:
    httpx_module = cast("Any", _import_httpx())
    return httpx_module.Timeout(
        timeout=settings.convert_timeout_seconds,
        connect=settings.connect_timeout_seconds,
    )


def _write_temp_input_wav(audio: SynthesizedAudio, temp_wav_dir: Path) -> str:
    temp_wav_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=temp_wav_dir,
        suffix=".wav",
        delete=False,
    ) as temp_file:
        temp_file.write(audio.wav_bytes)
        return temp_file.name


def _response_to_audio(response: object, profile: RVCProfile) -> SynthesizedAudio:
    if isinstance(response, (Path, str)):
        return _audio_from_path(Path(response), profile)
    if isinstance(response, tuple) and len(response) == _PAIR_LEN:
        if response[1] is None:
            msg = "RVC sidecar returned no audio"
            raise BackendUnavailableError(msg)
        if isinstance(response[1], (Path, str)):
            return _audio_from_path(Path(response[1]), profile)
        if not isinstance(response[1], tuple) or len(response[1]) != _PAIR_LEN:
            msg = "Unexpected RVC response shape"
            raise BackendUnavailableError(msg)
        sample_rate, audio_array = response[1]
        if not isinstance(sample_rate, int):
            msg = "Unexpected RVC response shape"
            raise BackendUnavailableError(msg)
        _validate_sample_rate(sample_rate, profile.sample_rate)
        return SynthesizedAudio(
            wav_bytes=_encode_wav_bytes(audio_array, sample_rate=sample_rate),
            sample_rate=sample_rate,
        )
    msg = "Unexpected RVC response shape"
    raise BackendUnavailableError(msg)


def _audio_from_path(path: Path, profile: RVCProfile) -> SynthesizedAudio:
    if not path.is_file():
        msg = "Unexpected RVC response shape"
        raise BackendUnavailableError(msg)
    wav_bytes = path.read_bytes()
    sample_rate = _read_wav_sample_rate(wav_bytes)
    _validate_sample_rate(sample_rate, profile.sample_rate)
    return SynthesizedAudio(wav_bytes=wav_bytes, sample_rate=sample_rate)


def _read_wav_sample_rate(wav_bytes: bytes) -> int:
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
            return wav_file.getframerate()
    except (EOFError, wave.Error) as exc:
        msg = "RVC output was not a WAV file"
        raise BackendUnavailableError(msg) from exc


def _validate_sample_rate(actual: int, expected: int) -> None:
    if actual != expected:
        msg = f"RVC returned sample rate {actual}, expected {expected}"
        raise BackendUnavailableError(msg)


def _encode_wav_bytes(audio_array: object, *, sample_rate: int) -> bytes:
    pcm_frames = array("h", (_to_pcm16(sample) for sample in _flatten_audio(audio_array)))
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_frames.tobytes())
        return buffer.getvalue()


def _flatten_audio(audio_array: object) -> list[float]:
    if isinstance(audio_array, _ListConvertible):
        audio_array = audio_array.tolist()
    if isinstance(audio_array, Iterable) and not isinstance(
        audio_array,
        (str, bytes, bytearray),
    ):
        samples: list[float] = []
        for item in audio_array:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes, bytearray)):
                samples.extend(_flatten_audio(item))
            else:
                try:
                    samples.append(float(item))
                except (TypeError, ValueError) as exc:
                    msg = "Unexpected RVC audio payload"
                    raise BackendUnavailableError(msg) from exc
        return samples
    msg = "Unexpected RVC audio payload"
    raise BackendUnavailableError(msg)


def _to_pcm16(sample: float) -> int:
    if not -1.0 <= sample <= 1.0:
        logger.warning("rvc_sidecar_pcm_sample_out_of_unit_range", sample=sample)
    value = round(sample * 32_767) if -1.0 <= sample <= 1.0 else round(sample)
    return max(-32_768, min(32_767, value))


def _resolve_closer(client: object) -> Callable[[], None] | None:
    if isinstance(client, _ClosableClient):
        return client.close
    if isinstance(client, _SessionCloseable):
        session_close = getattr(client.session, "close", None)
        if callable(session_close):
            return cast("Callable[[], None]", session_close)
    return None


def _unlink_temp_file(path: str) -> None:
    with suppress(FileNotFoundError):
        os.unlink(path)  # noqa: PTH108


def _client_errors() -> tuple[type[BaseException], ...]:
    httpx_module = cast("Any", _import_httpx())
    return (
        ConnectionError,
        TimeoutError,
        OSError,
        cast("type[BaseException]", httpx_module.HTTPError),
    )


def _import_httpx() -> object:
    return importlib.import_module("httpx")
