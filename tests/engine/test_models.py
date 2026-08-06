from __future__ import annotations

import pytest
from pydantic import ValidationError

from irodori_tts_infra.engine.models import (
    PipelineConfig,
    ResolvedSynthesisRequest,
    SynthesisJob,
)

pytestmark = pytest.mark.unit


def test_synthesis_job_maps_versioned_voice_selection_to_contract_request() -> None:
    job = SynthesisJob(
        segment_index=0,
        text="fixture text",
        voice_id="fixture-voice",
        if_generation="fixture-generation",
    )

    request = job.to_request(ref_embed="speakers/fixture.speaker.safetensors")

    assert request.voice_id == job.voice_id
    assert request.if_generation == job.if_generation
    assert request.ref_embed == "speakers/fixture.speaker.safetensors"


def test_synthesis_job_requires_a_resolved_internal_embedding() -> None:
    job = SynthesisJob(segment_index=0, text="fixture text")

    with pytest.raises(ValueError, match="resolved ref_embed"):
        job.to_request()


def test_resolved_synthesis_request_rejects_blank_embedding() -> None:
    with pytest.raises(ValidationError, match="ref_embed"):
        ResolvedSynthesisRequest(text="fixture text", ref_embed="   ")


def test_pipeline_config_normalizes_and_requires_non_blank_generation() -> None:
    config = PipelineConfig(generation="  fixture-generation  ")

    assert config.generation == "fixture-generation"

    with pytest.raises(ValueError, match="generation"):
        PipelineConfig(generation="   ")
