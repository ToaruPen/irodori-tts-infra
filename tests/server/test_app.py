from __future__ import annotations

import subprocess  # noqa: S404
import sys
from typing import TYPE_CHECKING, Protocol

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES
from irodori_tts_infra.server.app import create_app

if TYPE_CHECKING:
    from collections.abc import Callable

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline

pytestmark = pytest.mark.unit


class _WarmableSynthesizer(Protocol):
    warm_up_calls: int
    close_calls: int


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
