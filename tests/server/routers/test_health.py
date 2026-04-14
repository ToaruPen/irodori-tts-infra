from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.server.app import create_app

if TYPE_CHECKING:
    from collections.abc import Callable

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline

pytestmark = pytest.mark.unit


def test_health_returns_health_response_with_configured_chunk_size(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())
    app.state.max_chunk_size = 64

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "detail": None,
        "max_chunk_size": 64,
    }


def test_health_returns_degraded_response_when_model_unloaded(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.model_loaded = False
        app.state.health_detail = "warmup failed"
        response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False
    assert response.json()["detail"] == "warmup failed"
