from __future__ import annotations

import importlib
import subprocess  # noqa: S404
import sys
import threading
import time
from typing import TYPE_CHECKING, Protocol

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import (
    MAX_CHUNK_SIZE_BYTES,
    CapabilitiesResponse,
    ErrorPayload,
)
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.errors import BackendUnavailableError, VoiceBankInvalidError
from irodori_tts_infra.engine.models import PipelineConfig
from irodori_tts_infra.server.app import create_app, create_app_from_factory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline
    from irodori_tts_infra.voice_bank import VoiceProfile

pytestmark = pytest.mark.unit
EXPECTED_LIFESPAN_COUNT = 2
STATE_TRANSITION_TIMEOUT = 2.0


class _WarmableSynthesizer(Protocol):
    warm_up_calls: int
    warm_up_ref_embeds: list[str]
    close_calls: int


class WarmupUnavailableSynthesizer(FakeSynthesizer):
    @staticmethod
    def warm_up(**_kwargs: object) -> None:
        msg = "warmup backend unavailable"
        raise BackendUnavailableError(msg)


class TrackingWarmableSynthesizer(FakeSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.warm_up_calls = 0
        self.warm_up_ref_embeds: list[str] = []
        self.close_calls = 0

    def warm_up(self, *, ref_embed: str) -> None:
        self.warm_up_calls += 1
        self.warm_up_ref_embeds.append(ref_embed)

    def close(self) -> None:
        self.close_calls += 1


def _wait_for_readiness(
    client: TestClient,
    expected: str,
) -> CapabilitiesResponse:
    deadline = time.monotonic() + STATE_TRANSITION_TIMEOUT
    while time.monotonic() < deadline:
        response = CapabilitiesResponse.model_validate_json(client.get("/capabilities").text)
        if response.readiness == expected:
            return response
        time.sleep(0.001)
    pytest.fail(f"readiness did not transition to {expected}")


def test_create_app_warms_up_and_closes_backend(
    pipeline_factory: Callable[..., SynthesisPipeline],
    warmable_synthesizer: _WarmableSynthesizer,
) -> None:
    app = create_app(pipeline_factory(warmable_synthesizer))

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert warmable_synthesizer.warm_up_calls == 1
        assert warmable_synthesizer.warm_up_ref_embeds == [
            "speakers/narrator.speaker.safetensors",
        ]
        assert warmable_synthesizer.close_calls == 0
        assert response.json()["model_loaded"] is True
        assert response.json()["max_chunk_size"] == MAX_CHUNK_SIZE_BYTES

    assert warmable_synthesizer.close_calls == 1


def test_factory_app_recreates_pipeline_after_lifespan_shutdown(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    synthesizers: list[_WarmableSynthesizer] = []

    def build_pipeline() -> SynthesisPipeline:
        synthesizer = TrackingWarmableSynthesizer()
        synthesizers.append(synthesizer)
        return pipeline_factory(
            synthesizer,
            config=PipelineConfig(generation="fixture-generation"),
        )

    app = create_app_from_factory(build_pipeline, generation="fixture-generation")

    with TestClient(app) as client:
        assert client.get("/health").status_code == status.HTTP_200_OK
    with TestClient(app) as client:
        assert client.get("/health").status_code == status.HTTP_200_OK

    assert len(synthesizers) == EXPECTED_LIFESPAN_COUNT
    assert [synthesizer.warm_up_calls for synthesizer in synthesizers] == [1, 1]
    assert [synthesizer.close_calls for synthesizer in synthesizers] == [1, 1]


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
    assert body["detail"] == "Synthesis model is not loaded"


def test_factory_app_exposes_loading_then_atomically_publishes_runtime_catalog(
    pipeline_factory: Callable[..., SynthesisPipeline],
    catalog_profile_factory: Callable[[int], VoiceProfile],
) -> None:
    factory_entered = threading.Event()
    factory_release = threading.Event()
    synthesizer = TrackingWarmableSynthesizer()
    profile = catalog_profile_factory(3)

    def build_pipeline() -> SynthesisPipeline:
        factory_entered.set()
        assert factory_release.wait(timeout=STATE_TRANSITION_TIMEOUT)
        return pipeline_factory(
            synthesizer,
            config=PipelineConfig(generation="fixture-generation"),
            voice_profile=profile,
        )

    app = create_app_from_factory(build_pipeline, generation="fixture-generation")

    with TestClient(app) as client:
        try:
            assert factory_entered.wait(timeout=STATE_TRANSITION_TIMEOUT)
            loading = CapabilitiesResponse.model_validate_json(client.get("/capabilities").text)
            blocked = client.post("/synthesize", json={"text": "本文"})

            assert loading.readiness == "model_loading"
            assert loading.ready is False
            assert loading.voices == ()
            assert blocked.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert ErrorPayload.model_validate_json(blocked.text).code == "model_not_loaded"

            factory_release.set()
            ready = _wait_for_readiness(client, "ready")
            assert tuple(voice.id for voice in ready.voices) == tuple(
                voice.id for voice in profile.catalog
            )
            assert synthesizer.warm_up_calls == 1
        finally:
            factory_release.set()

    assert synthesizer.close_calls == 1


@pytest.mark.parametrize(
    ("error", "expected_readiness", "expected_code"),
    [
        (
            VoiceBankInvalidError("/private/secret/voice-bank.toml"),
            "voice_bank_invalid",
            "voice_bank_invalid",
        ),
        (
            RuntimeError("backend failed at /private/secret/checkpoint"),
            "model_not_loaded",
            "model_not_loaded",
        ),
    ],
)
def test_factory_app_maps_load_failures_to_safe_readiness_and_errors(
    error: Exception,
    expected_readiness: str,
    expected_code: str,
) -> None:
    def fail_factory() -> SynthesisPipeline:
        raise error

    app = create_app_from_factory(fail_factory, generation="fixture-generation")

    with TestClient(app, raise_server_exceptions=False) as client:
        capabilities = _wait_for_readiness(client, expected_readiness)
        response = client.post("/synthesize", json={"text": "本文"})

    assert capabilities.voices == ()
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    payload = ErrorPayload.model_validate_json(response.text)
    assert payload.code == expected_code
    assert "/private/secret" not in response.text


def test_factory_app_rejects_unconfigured_generation_without_calling_factory() -> None:
    calls = 0

    def build_pipeline() -> SynthesisPipeline:
        nonlocal calls
        calls += 1
        msg = "factory must not be called"
        raise AssertionError(msg)

    app = create_app_from_factory(build_pipeline)

    with TestClient(app) as client:
        capabilities = _wait_for_readiness(client, "voice_bank_invalid")

    assert capabilities.generation == "unconfigured"
    assert calls == 0


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


def test_server_main_exports_asgi_app_without_eager_runtime() -> None:
    code = (
        "import sys\n"
        "from irodori_tts_infra.server.main import app\n"
        "assert app.title\n"
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


def test_server_main_reports_voice_bank_invalid_without_voice_bank_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOICE_BANK_SPEAKER_MANIFEST", raising=False)
    monkeypatch.delenv("VOICE_BANK_DIR", raising=False)
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_PUBLIC_GENERATION", "fixture-generation")
    server_main = importlib.reload(importlib.import_module("irodori_tts_infra.server.main"))

    with TestClient(server_main.app) as client:
        capabilities = _wait_for_readiness(client, "voice_bank_invalid")

    assert capabilities.ready is False


def test_server_main_reports_voice_bank_invalid_when_embedding_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOICE_BANK_SPEAKER_MANIFEST", str(manifest))
    monkeypatch.delenv("VOICE_BANK_DIR", raising=False)
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_PUBLIC_GENERATION", "fixture-generation")
    server_main = importlib.reload(importlib.import_module("irodori_tts_infra.server.main"))

    with TestClient(server_main.app) as client:
        capabilities = _wait_for_readiness(client, "voice_bank_invalid")

    assert capabilities.ready is False
