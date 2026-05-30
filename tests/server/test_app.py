from __future__ import annotations

import importlib
import subprocess  # noqa: S404
import sys
from typing import TYPE_CHECKING, Protocol

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.server.app import create_app, create_app_from_factory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline

pytestmark = pytest.mark.unit
EXPECTED_LIFESPAN_COUNT = 2


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
        return pipeline_factory(synthesizer)

    app = create_app_from_factory(build_pipeline)

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
    assert "warmup backend unavailable" in body["detail"]


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


def test_server_main_startup_fails_without_voice_bank_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOICE_BANK_SPEAKER_MANIFEST", raising=False)
    monkeypatch.delenv("VOICE_BANK_DIR", raising=False)
    server_main = importlib.import_module("irodori_tts_infra.server.main")

    with (
        pytest.raises(ValueError, match="VOICE_BANK_SPEAKER_MANIFEST or VOICE_BANK_DIR"),
        TestClient(server_main.app),
    ):
        pass


def test_server_main_startup_fails_when_voice_bank_embedding_is_missing(
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
    server_main = importlib.import_module("irodori_tts_infra.server.main")

    with (
        pytest.raises(
            ValueError,
            match=r"speaker embedding file does not exist: .*narrator\.speaker\.safetensors",
        ),
        TestClient(server_main.app),
    ):
        pass
