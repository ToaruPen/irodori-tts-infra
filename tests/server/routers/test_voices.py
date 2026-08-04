from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import CapabilitiesResponse
from irodori_tts_infra.engine.models import PipelineConfig
from irodori_tts_infra.server.app import create_app

if TYPE_CHECKING:
    from collections.abc import Callable

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline
    from irodori_tts_infra.voice_bank import VoiceProfile

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count", [0, 1, 4])
def test_capabilities_returns_runtime_catalog_without_fixed_names_or_order(
    count: int,
    pipeline_factory: Callable[..., SynthesisPipeline],
    catalog_profile_factory: Callable[[int], VoiceProfile],
) -> None:
    profile = catalog_profile_factory(count)
    app = create_app(
        pipeline_factory(
            config=PipelineConfig(generation="fixture-generation"),
            voice_profile=profile,
        )
    )

    with TestClient(app) as client:
        response = client.get("/capabilities")

    assert response.status_code == status.HTTP_200_OK
    capabilities = CapabilitiesResponse.model_validate_json(response.text)
    assert capabilities.generation == "fixture-generation"
    assert capabilities.ready is True
    assert capabilities.readiness == "ready"
    assert tuple(voice.id for voice in capabilities.voices) == tuple(
        voice.id for voice in profile.catalog
    )
    assert tuple(voice.label for voice in capabilities.voices) == tuple(
        voice.label for voice in profile.catalog
    )
    assert tuple(voice.aliases for voice in capabilities.voices) == tuple(
        voice.aliases for voice in profile.catalog
    )
    assert "ref_embed" not in response.text
    assert ".speaker.safetensors" not in response.text
    assert capabilities.conditioning.delivery_caption.supported is False
    assert capabilities.conditioning.emoji.supported is True
