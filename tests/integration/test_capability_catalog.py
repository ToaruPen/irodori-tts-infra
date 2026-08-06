from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import CapabilitiesResponse, SynthesisResult
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.models import PipelineConfig
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.server.app import create_app
from irodori_tts_infra.voice_bank import PortableVoice, SpeakerEmbeddingProfile, VoiceProfile

pytestmark = pytest.mark.unit


def _fixture_profile(count: int) -> VoiceProfile:
    narrator = SpeakerEmbeddingProfile(Path("speakers/fixture-narrator.speaker.safetensors"))
    catalog = tuple(
        PortableVoice(
            id=f"fixture-voice-{index}",
            label=f"Fixture voice {index}",
            aliases=(f"fixture-alias-{index}",),
            default=index == 0,
            speaker=SpeakerEmbeddingProfile(Path(f"speakers/fixture-{index}.speaker.safetensors")),
        )
        for index in range(count)
    )
    return VoiceProfile(characters={}, narrator=narrator, catalog=catalog)


@pytest.mark.parametrize("fixture_count", [1, 4])
def test_every_runtime_catalog_entry_synthesizes_with_advertised_generation(
    fixture_count: int,
) -> None:
    synthesizer = FakeSynthesizer()
    profile = _fixture_profile(fixture_count)
    app = create_app(
        SynthesisPipeline(
            synthesizer,
            profile,
            config=PipelineConfig(generation="fixture-generation"),
        )
    )

    with TestClient(app) as client:
        capabilities = CapabilitiesResponse.model_validate_json(client.get("/capabilities").text)
        assert capabilities.ready is True
        assert capabilities.voices

        for voice in capabilities.voices:
            response = client.post(
                "/synthesize",
                json={
                    "text": "fixture text",
                    "voice_id": voice.id,
                    "if_generation": capabilities.generation,
                },
            )
            assert response.status_code == status.HTTP_200_OK
            result = SynthesisResult.model_validate_json(response.text)
            assert result.wav_bytes.startswith(b"RIFF")
            assert b"WAVE" in result.wav_bytes
            assert math.isfinite(result.elapsed_seconds)
            assert result.elapsed_seconds >= 0.0
            current = CapabilitiesResponse.model_validate_json(client.get("/capabilities").text)
            assert current.readiness == "ready"
            assert current.generation == capabilities.generation

    assert len(synthesizer.calls) == len(capabilities.voices)
