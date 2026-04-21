from __future__ import annotations

import builtins
import wave
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from irodori_tts_infra.config.settings import PathSettings, RVCSidecarSettings
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
    temp_wav_dir: Path | None = None,
) -> RVCConverter:
    return RVCConverter(
        client=client or FakeRVCClient(),
        settings=settings or sidecar_settings(),
        temp_wav_dir=temp_wav_dir or PathSettings().temp_wav_dir,
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


def _wav_samples(wav_bytes: bytes) -> list[int]:
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return [
        int.from_bytes(frames[index : index + 2], "little", signed=True)
        for index in range(0, len(frames), 2)
    ]


def test_convert_packs_official_webui_arguments_and_cleans_temp_input(tmp_path: Path) -> None:
    client = FakeRVCClient()
    backend = make_backend(client, temp_wav_dir=tmp_path)

    result = backend.convert(input_audio(), profile=profile())

    assert len(client.calls) == EXPECTED_PREDICT_CALLS
    assert client.calls[0] == (
        (profile().model_path.name, 0.33, 0.33),
        {"api_name": "/infer_change_voice"},
    )
    args, kwargs = client.calls[1]
    assert kwargs == {"api_name": "/infer_convert"}
    assert args[0] == 0
    input_path = Path(str(args[1]))
    assert input_path.parent == tmp_path
    assert input_path.suffix == ".wav"
    assert args[2:] == (0, None, "harvest", "", "", 0.88, 3, INPUT_SAMPLE_RATE, 1, 0.33)
    assert result.sample_rate == INPUT_SAMPLE_RATE
    assert _wav_sample_rate(result.wav_bytes) == INPUT_SAMPLE_RATE
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


def test_convert_accepts_direct_path_response(tmp_path: Path) -> None:
    output_path = tmp_path / "converted.wav"
    output_path.write_bytes(_wav_bytes_for([0, 256, -256], sample_rate=INPUT_SAMPLE_RATE))
    backend = make_backend(FakeRVCClient(convert_result=str(output_path)))

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
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "gradio_client":
            raise error
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

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


def test_factory_returns_converter_that_warms_up() -> None:
    clients: list[FakeRVCClient] = []

    def client_factory(**_kwargs: object) -> FakeRVCClient:
        client = FakeRVCClient()
        clients.append(client)
        return client

    backend = create_rvc_backend(sidecar_settings(), client_factory=client_factory)

    assert isinstance(backend, RVCConverter)
    backend.warm_up()
    assert clients[0].view_api_count == 1


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
    tmp_path: Path,
) -> None:
    client = FakeRVCClient()
    client.predict_exception = ConnectionError("profile load failed")
    client.predict_exception_api_name = "/infer_change_voice"
    backend = make_backend(client, temp_wav_dir=tmp_path)

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
        (("Success", 123), "Unexpected RVC response shape"),
        (("Success", (None, None)), "Unexpected RVC response shape"),
    ],
)
def test_convert_rejects_unexpected_response_shape(
    response: object,
    match: str,
    tmp_path: Path,
) -> None:
    backend = make_backend(FakeRVCClient(convert_result=response), temp_wav_dir=tmp_path)

    with pytest.raises(BackendUnavailableError, match=match):
        backend.convert(input_audio(), profile=profile())

    assert not any(tmp_path.glob("*.wav"))


def test_convert_preserves_nested_audio_array_order() -> None:
    backend = make_backend(
        FakeRVCClient(convert_result=("Success", (INPUT_SAMPLE_RATE, [[0.0], [0.5, -0.5]])))
    )

    result = backend.convert(input_audio(), profile=profile())

    assert _wav_samples(result.wav_bytes) == [0, 16_384, -16_384]


def test_convert_rejects_nested_non_numeric_audio_payload() -> None:
    backend = make_backend(
        FakeRVCClient(convert_result=("Success", (INPUT_SAMPLE_RATE, [[object()]])))
    )

    with pytest.raises(BackendUnavailableError, match="Unexpected RVC audio payload"):
        backend.convert(input_audio(), profile=profile())


def test_convert_rejects_deeply_nested_audio_payload() -> None:
    audio_payload: object = 0.0
    for _ in range(101):
        audio_payload = [audio_payload]
    backend = make_backend(
        FakeRVCClient(convert_result=("Success", (INPUT_SAMPLE_RATE, audio_payload)))
    )

    with pytest.raises(BackendUnavailableError, match="Unexpected RVC audio payload"):
        backend.convert(input_audio(), profile=profile())


def test_convert_warns_when_audio_samples_look_pcm_scaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class FakeLogger:
        @staticmethod
        def debug(_event: str, **_fields: object) -> None:
            return None

        @staticmethod
        def warning(event: str, **fields: object) -> None:
            events.append((event, fields))

    monkeypatch.setattr(rvc_module, "logger", FakeLogger())
    backend = make_backend(FakeRVCClient(convert_result=("Success", (INPUT_SAMPLE_RATE, [2.0]))))

    result = backend.convert(input_audio(), profile=profile())

    assert _wav_samples(result.wav_bytes) == [2]
    assert events == [("rvc_sidecar_pcm_sample_out_of_unit_range", {"sample": 2.0})]


@pytest.mark.parametrize("api_name", ["/infer_convert", "/infer_change_voice"])
def test_convert_maps_gradio_protocol_value_error(
    api_name: str,
    tmp_path: Path,
) -> None:
    client = FakeRVCClient()
    client.predict_exception = ValueError("None")
    client.predict_exception_api_name = api_name
    backend = make_backend(client, temp_wav_dir=tmp_path)

    with pytest.raises(BackendUnavailableError, match="RVC conversion failed") as exc_info:
        backend.convert(input_audio(), profile=profile())

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "None"
    assert not any(tmp_path.glob("*.wav"))


@pytest.mark.parametrize(
    "unrelated",
    [
        pytest.param(ValueError("programming bug"), id="non-None-string"),
        pytest.param(ValueError(None), id="None-object-not-string"),
    ],
)
def test_convert_does_not_wrap_unrelated_value_error(
    unrelated: ValueError,
    tmp_path: Path,
) -> None:
    client = FakeRVCClient()
    client.predict_exception = unrelated
    client.predict_exception_api_name = "/infer_convert"
    backend = make_backend(client, temp_wav_dir=tmp_path)

    with pytest.raises(ValueError) as exc_info:
        backend.convert(input_audio(), profile=profile())

    assert exc_info.value is unrelated
    assert not any(tmp_path.glob("*.wav"))


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        (-1.0, -32_767),
        (0.0, 0),
        (1.0, 32_767),
        (-1.0001, -1),
        (1.0001, 1),
        (-32_768.0, -32_768),
        (32_768.0, 32_767),
    ],
)
def test_convert_applies_pcm16_boundary_values(sample: float, expected: int) -> None:
    backend = make_backend(FakeRVCClient(convert_result=("Success", (INPUT_SAMPLE_RATE, [sample]))))

    result = backend.convert(input_audio(), profile=profile())

    assert _wav_samples(result.wav_bytes) == [expected]


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        (-1.0, -32_767),
        (1.0, 32_767),
        (-1.0001, -1),
        (1.0001, 1),
        (0.0, 0),
        (-32_768.0, -32_768),
        (32_767.0, 32_767),
        (-40_000.0, -32_768),
        (40_000.0, 32_767),
    ],
)
def test_to_pcm16_handles_float_normalized_and_pcm_scaled_samples(
    sample: float,
    expected: int,
) -> None:
    assert rvc_module._to_pcm16(sample) == expected  # noqa: SLF001 - direct private unit test


def test_warm_up_calls_view_api_once() -> None:
    client = FakeRVCClient()
    backend = make_backend(client)

    backend.warm_up()

    assert client.view_api_count == 1


def test_warm_up_maps_unreachable_sidecar() -> None:
    client = FakeRVCClient()
    client.view_api_exception = TimeoutError("timed out")
    backend = make_backend(client)

    with pytest.raises(BackendUnavailableError, match="RVC sidecar warm-up failed") as exc_info:
        backend.warm_up()

    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_close_is_idempotent_with_session_close_fallback(tmp_path: Path) -> None:
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
    backend = RVCConverter(client=client, settings=sidecar_settings(), temp_wav_dir=tmp_path)

    backend.close()
    backend.close()

    assert client.session.close_count == 1


def test_close_prefers_client_close_over_session_close() -> None:
    client = FakeRVCClient()
    backend = make_backend(client)

    backend.close()
    backend.close()

    assert client.close_count == 1
    assert client.session.close_count == 0


def test_convert_after_close_raises_backend_unavailable() -> None:
    backend = make_backend()

    backend.close()

    with pytest.raises(BackendUnavailableError, match="backend is closed"):
        backend.convert(input_audio(), profile=profile())


def test_sample_rate_mismatch_raises_backend_unavailable() -> None:
    backend = make_backend(FakeRVCClient(convert_result=("Success", (48_000, [0.0, 0.1]))))

    with pytest.raises(BackendUnavailableError, match="sample rate"):
        backend.convert(input_audio(), profile=profile())
