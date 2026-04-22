from __future__ import annotations

import subprocess  # noqa: S404
import sys
from typing import TYPE_CHECKING, Protocol

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import SynthesizedAudio
from irodori_tts_infra.server.app import create_app

if TYPE_CHECKING:
    from collections.abc import Callable

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline
    from irodori_tts_infra.voice_bank import RVCProfile

pytestmark = pytest.mark.unit


class _WarmableSynthesizer(Protocol):
    warm_up_calls: int
    close_calls: int


class _WarmableConverter(Protocol):
    warm_up_calls: int
    close_calls: int


class WarmupUnavailableSynthesizer(FakeSynthesizer):
    @staticmethod
    def warm_up() -> None:
        msg = "warmup backend unavailable"
        raise BackendUnavailableError(msg)


class WarmupUnavailableConverter:
    def __init__(self) -> None:
        self.close_calls = 0

    @staticmethod
    def warm_up() -> None:
        msg = "converter warmup failed"
        raise BackendUnavailableError(msg)

    def close(self) -> None:
        self.close_calls += 1

    @staticmethod
    def convert(audio: SynthesizedAudio, *, profile: RVCProfile) -> SynthesizedAudio:
        return SynthesizedAudio(wav_bytes=audio.wav_bytes, sample_rate=profile.sample_rate)


class CloseFailingConverter:
    def __init__(self) -> None:
        self.close_calls = 0

    @staticmethod
    def warm_up() -> None:
        return

    def close(self) -> None:
        self.close_calls += 1
        msg = "gradio client died"
        raise RuntimeError(msg)

    @staticmethod
    def convert(audio: SynthesizedAudio, *, profile: RVCProfile) -> SynthesizedAudio:
        return SynthesizedAudio(wav_bytes=audio.wav_bytes, sample_rate=profile.sample_rate)


def test_create_app_warms_up_and_closes_backend(
    pipeline_factory: Callable[..., SynthesisPipeline],
    warmable_synthesizer: _WarmableSynthesizer,
) -> None:
    app = create_app(pipeline_factory(warmable_synthesizer))

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert warmable_synthesizer.warm_up_calls == 1
        assert warmable_synthesizer.close_calls == 0
        assert response.json()["model_loaded"] is True
        assert response.json()["max_chunk_size"] == MAX_CHUNK_SIZE_BYTES

    assert warmable_synthesizer.close_calls == 1


def test_create_app_handles_backend_unavailable_on_warmup(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory(WarmupUnavailableSynthesizer()))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"
    assert "warmup backend unavailable" in body["detail"]


def test_create_app_warms_up_and_closes_voice_converter(
    pipeline_factory: Callable[..., SynthesisPipeline],
    warmable_synthesizer: _WarmableSynthesizer,
    warmable_converter: _WarmableConverter,
) -> None:
    app = create_app(
        pipeline_factory(warmable_synthesizer, voice_converter=warmable_converter),
    )

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert warmable_converter.warm_up_calls == 1
        assert warmable_converter.close_calls == 0
        assert response.json()["model_loaded"] is True

    assert warmable_converter.close_calls == 1
    assert warmable_synthesizer.close_calls == 1


def test_create_app_handles_converter_unavailable_on_warmup(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    converter = WarmupUnavailableConverter()
    app = create_app(pipeline_factory(FakeSynthesizer(), voice_converter=converter))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"
    assert "converter warmup failed" in body["detail"]
    assert converter.close_calls == 1


def test_create_app_close_error_in_converter_does_not_skip_backend_close(
    pipeline_factory: Callable[..., SynthesisPipeline],
    warmable_synthesizer: _WarmableSynthesizer,
) -> None:
    converter = CloseFailingConverter()
    app = create_app(
        pipeline_factory(warmable_synthesizer, voice_converter=converter),
    )

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    assert converter.close_calls == 1
    assert warmable_synthesizer.close_calls == 1


def test_server_import_is_lightweight() -> None:
    code = (
        "import sys\n"
        "import irodori_tts_infra.server.app\n"
        'blocked = {"irodori_tts", "huggingface_hub", "torch"}\n'
        "loaded = blocked & set(sys.modules)\n"
        'assert not loaded, f"heavy modules loaded: {loaded}"\n'
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
