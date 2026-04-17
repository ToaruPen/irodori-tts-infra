from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from irodori_tts_infra.config.settings import RVCSidecarSettings
from irodori_tts_infra.engine.backends import rvc as rvc_module
from irodori_tts_infra.engine.backends.rvc import RVCConverter, create_rvc_backend
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import SynthesizedAudio
from irodori_tts_infra.voice_bank import RVCProfile

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.unit

INPUT_SAMPLE_RATE = 40_000
EXPECTED_PREDICT_CALLS = 2


class FakeSession:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class FakeRVCClient:
    def __init__(
        self,
        *,
        convert_result: object = ("Success", (INPUT_SAMPLE_RATE, [0.0, 0.25, -0.25])),
    ) -> None:
        self.convert_result = convert_result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.view_api_count = 0
        self.close_count = 0
        self.session = FakeSession()
        self.predict_exception: Exception | None = None
        self.predict_exception_api_name: str | None = None
        self.view_api_exception: Exception | None = None

    def view_api(self) -> dict[str, object]:
        if self.view_api_exception is not None:
            raise self.view_api_exception
        self.view_api_count += 1
        return {"named_endpoints": {"/infer_convert": {"parameters": 12}}}

    def predict(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        if (
            self.predict_exception is not None
            and self.predict_exception_api_name == kwargs["api_name"]
        ):
            raise self.predict_exception
        if kwargs["api_name"] == "/infer_change_voice":
            return {"loaded": args[0]}
        if kwargs["api_name"] == "/infer_convert":
            input_path = Path(str(args[1]))
            assert input_path.exists()
            return self.convert_result
        msg = f"unexpected api_name: {kwargs['api_name']}"
        raise AssertionError(msg)

    def close(self) -> None:
        self.close_count += 1


def sidecar_settings(**overrides: object) -> RVCSidecarSettings:
    data: dict[str, object] = {
        "url": "http://localhost:7865",
        "api_name": "/infer_convert",
        "connect_timeout_seconds": 10.0,
        "convert_timeout_seconds": 120.0,
    }
    data.update(overrides)
    return RVCSidecarSettings.model_validate(data)


def input_audio() -> SynthesizedAudio:
    return SynthesizedAudio(
        wav_bytes=_wav_bytes_for([0, 512, -512], sample_rate=INPUT_SAMPLE_RATE),
        sample_rate=INPUT_SAMPLE_RATE,
    )


def profile(**overrides: object) -> RVCProfile:
    model_path = cast("Path", overrides.pop("model_path", Path("models/chizuru.pth")))
    sample_rate = cast("int", overrides.pop("sample_rate", INPUT_SAMPLE_RATE))
    neutral_prototype = cast("Path | None", overrides.pop("neutral_prototype", None))
    state_prototypes = cast("Mapping[str, Path]", overrides.pop("state_prototypes", {}))
    assert not overrides
    return RVCProfile(
        model_path=model_path,
        sample_rate=sample_rate,
        neutral_prototype=neutral_prototype,
        state_prototypes=state_prototypes,
    )


def make_backend(
    client: FakeRVCClient | None = None,
    *,
    settings: RVCSidecarSettings | None = None,
) -> RVCConverter:
    return RVCConverter(
        client=client or FakeRVCClient(),
        settings=settings or sidecar_settings(),
    )


def _wav_bytes_for(samples: list[int], *, sample_rate: int) -> bytes:
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(
                b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples),
            )
        return buffer.getvalue()


def _wav_sample_rate(wav_bytes: bytes) -> int:
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.getframerate()


def test_convert_packs_official_webui_arguments_and_cleans_temp_input() -> None:
    client = FakeRVCClient()
    backend = make_backend(client)

    result = backend.convert(input_audio(), profile=profile())

    assert len(client.calls) == EXPECTED_PREDICT_CALLS
    assert client.calls[0] == (
        (profile().model_path.name, 0.33, 0.33),
        {"api_name": "/infer_change_voice"},
    )
    args, kwargs = client.calls[1]
    assert kwargs == {"api_name": "/infer_convert"}
    assert args[0] == 0
    assert Path(str(args[1])).suffix == ".wav"
    assert args[2:] == (0, None, "harvest", "", "", 0.88, 3, INPUT_SAMPLE_RATE, 1, 0.33)
    assert result.sample_rate == INPUT_SAMPLE_RATE
    assert _wav_sample_rate(result.wav_bytes) == INPUT_SAMPLE_RATE
    input_path = Path(str(args[1]))
    assert not input_path.exists()


def test_convert_accepts_path_response(tmp_path: Path) -> None:
    output_path = tmp_path / "converted.wav"
    output_path.write_bytes(_wav_bytes_for([0, 256, -256], sample_rate=INPUT_SAMPLE_RATE))
    backend = make_backend(FakeRVCClient(convert_result=("Success", str(output_path))))

    result = backend.convert(input_audio(), profile=profile())

    assert result == SynthesizedAudio(
        wav_bytes=output_path.read_bytes(),
        sample_rate=INPUT_SAMPLE_RATE,
    )


def test_convert_rejects_non_wav_path_response(tmp_path: Path) -> None:
    output_path = tmp_path / "converted.wav"
    output_path.write_bytes(b"not-a-wav")
    backend = make_backend(FakeRVCClient(convert_result=("Success", str(output_path))))

    with pytest.raises(BackendUnavailableError, match="not a WAV file"):
        backend.convert(input_audio(), profile=profile())


def test_convert_rejects_path_sample_rate_mismatch(tmp_path: Path) -> None:
    output_path = tmp_path / "converted.wav"
    output_path.write_bytes(_wav_bytes_for([0, 256, -256], sample_rate=48_000))
    backend = make_backend(FakeRVCClient(convert_result=("Success", str(output_path))))

    with pytest.raises(BackendUnavailableError, match="sample rate"):
        backend.convert(input_audio(), profile=profile())


def test_factory_raises_backend_unavailable_on_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ImportError("missing gradio_client")
    monkeypatch.setattr(rvc_module, "_gradio_client", None)
    monkeypatch.setattr(rvc_module, "_gradio_client_import_error", error)

    with pytest.raises(BackendUnavailableError, match="gradio_client") as exc_info:
        rvc_module.create_rvc_backend(sidecar_settings())

    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("error_cls", [ConnectionError, TimeoutError])
def test_factory_maps_transport_errors(error_cls: type[Exception]) -> None:
    error = error_cls("sidecar offline")

    def client_factory(**_kwargs: object) -> FakeRVCClient:
        raise error

    with pytest.raises(BackendUnavailableError, match="Failed to create RVC") as exc_info:
        create_rvc_backend(sidecar_settings(), client_factory=client_factory)

    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("error_cls", [ConnectionError, TimeoutError])
def test_convert_maps_transport_errors_and_cleans_temp_input(
    error_cls: type[Exception],
) -> None:
    client = FakeRVCClient()
    client.predict_exception = error_cls("network down")
    client.predict_exception_api_name = "/infer_convert"
    backend = make_backend(client)

    with pytest.raises(BackendUnavailableError, match="RVC conversion failed") as exc_info:
        backend.convert(input_audio(), profile=profile())

    assert isinstance(exc_info.value.__cause__, error_cls)
    args, _kwargs = client.calls[-1]
    assert not Path(str(args[1])).exists()


def test_convert_maps_profile_load_error_and_cleans_temp_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IRODORI_TTS_PATH_TEMP_WAV_DIR", str(tmp_path))
    client = FakeRVCClient()
    client.predict_exception = ConnectionError("profile load failed")
    client.predict_exception_api_name = "/infer_change_voice"
    backend = make_backend(client)

    with pytest.raises(BackendUnavailableError, match="RVC conversion failed") as exc_info:
        backend.convert(input_audio(), profile=profile())

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert not any(tmp_path.glob("*.wav"))


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (None, "Unexpected RVC response shape"),
        ((), "Unexpected RVC response shape"),
        (("Success", None), "RVC sidecar returned no audio"),
        (("Success", "not-audio"), "Unexpected RVC response shape"),
        (("Success", (None, None)), "Unexpected RVC response shape"),
    ],
)
def test_convert_rejects_unexpected_response_shape(response: object, match: str) -> None:
    backend = make_backend(FakeRVCClient(convert_result=response))

    with pytest.raises(BackendUnavailableError, match=match):
        backend.convert(input_audio(), profile=profile())


def test_warm_up_maps_unreachable_sidecar() -> None:
    client = FakeRVCClient()
    client.view_api_exception = TimeoutError("timed out")
    backend = make_backend(client)

    with pytest.raises(BackendUnavailableError, match="RVC sidecar warm-up failed") as exc_info:
        backend.warm_up()

    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_close_is_idempotent_with_session_close_fallback() -> None:
    class SessionOnlyClient:
        def __init__(self) -> None:
            self.session = FakeSession()

        @staticmethod
        def view_api() -> dict[str, object]:
            return {"named_endpoints": {}}

        @staticmethod
        def predict(*_args: object, **_kwargs: object) -> None:
            return None

    client = SessionOnlyClient()
    backend = RVCConverter(client=client, settings=sidecar_settings())

    backend.close()
    backend.close()

    assert client.session.close_count == 1


def test_sample_rate_mismatch_raises_backend_unavailable() -> None:
    backend = make_backend(FakeRVCClient(convert_result=("Success", (48_000, [0.0, 0.1]))))

    with pytest.raises(BackendUnavailableError, match="sample rate"):
        backend.convert(input_audio(), profile=profile())
