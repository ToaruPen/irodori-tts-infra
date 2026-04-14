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
