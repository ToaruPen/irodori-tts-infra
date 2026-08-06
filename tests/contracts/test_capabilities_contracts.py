from __future__ import annotations

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import CapabilitiesResponse, Readiness, VoiceCapability

pytestmark = pytest.mark.unit


def _capability_voices(count: int) -> tuple[VoiceCapability, ...]:
    return tuple(
        VoiceCapability(
            id=f"fixture-voice-{index}",
            label=f"Fixture voice {index}",
            aliases=(f"fixture-alias-{index}",),
            default=index == 0,
        )
        for index in range(count)
    )


@pytest.mark.parametrize("count", [0, 1, 4])
def test_capabilities_preserve_runtime_catalog_without_assuming_names_or_order(
    count: int,
) -> None:
    voices = _capability_voices(count)

    response = CapabilitiesResponse(
        generation="fixture-generation",
        ready=True,
        readiness="ready",
        voices=voices,
    )

    assert response.contract_version == 1
    assert tuple(item.id for item in response.voices) == tuple(item.id for item in voices)
    assert response.conditioning.delivery_caption.supported is False
    assert response.conditioning.delivery_caption.max_chars is None
    assert response.conditioning.emoji.supported is True


@pytest.mark.parametrize(
    ("ready", "readiness"),
    [
        (True, "model_loading"),
        (False, "ready"),
    ],
)
def test_capabilities_reject_inconsistent_readiness(
    ready: bool,  # noqa: FBT001
    readiness: Readiness,
) -> None:
    with pytest.raises(ValidationError, match="ready"):
        CapabilitiesResponse(
            generation="fixture-generation",
            ready=ready,
            readiness=readiness,
            voices=(),
        )


def test_capabilities_reject_blank_generation_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="generation"):
        CapabilitiesResponse(
            generation="   ",
            ready=False,
            readiness="model_loading",
            voices=(),
        )

    with pytest.raises(ValidationError, match="unexpected"):
        CapabilitiesResponse.model_validate(
            {
                "generation": "fixture-generation",
                "ready": False,
                "readiness": "model_loading",
                "voices": [],
                "unexpected": True,
            }
        )


def test_capabilities_reject_public_caption_limits_while_caption_is_unsupported() -> None:
    with pytest.raises(ValidationError, match="max_chars"):
        CapabilitiesResponse.model_validate(
            {
                "generation": "fixture-generation",
                "ready": True,
                "readiness": "ready",
                "voices": [],
                "conditioning": {
                    "delivery_caption": {"supported": False, "max_chars": 120},
                    "emoji": {"supported": True},
                },
            }
        )


@pytest.mark.parametrize(
    "voices",
    [
        (
            VoiceCapability(id="voice-a", label="A"),
            VoiceCapability(id="voice-a", label="B"),
        ),
        (
            VoiceCapability(id="voice-a", label="A", aliases=("shared",)),
            VoiceCapability(id="voice-b", label="B", aliases=("shared",)),
        ),
        (
            VoiceCapability(id="voice-a", label="A", aliases=("voice-b",)),
            VoiceCapability(id="voice-b", label="B"),
        ),
        (
            VoiceCapability(id="voice-a", label="A", default=True),
            VoiceCapability(id="voice-b", label="B", default=True),
        ),
    ],
)
def test_capabilities_reject_ambiguous_runtime_catalog(
    voices: tuple[VoiceCapability, ...],
) -> None:
    with pytest.raises(ValidationError, match="catalog"):
        CapabilitiesResponse(
            generation="fixture-generation",
            ready=True,
            readiness="ready",
            voices=voices,
        )
