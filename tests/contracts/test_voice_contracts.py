from __future__ import annotations

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import VoiceCapability

pytestmark = pytest.mark.unit


def test_voice_capability_normalizes_public_text_fields() -> None:
    voice = VoiceCapability(
        id="  fixture-id  ",
        label="  Fixture label  ",
        aliases=("  first alias  ", "second alias"),
        default=True,
    )

    assert voice.id == "fixture-id"
    assert voice.label == "Fixture label"
    assert voice.aliases == ("first alias", "second alias")
    assert voice.default is True


@pytest.mark.parametrize(
    "payload",
    [
        {"id": " ", "label": "Fixture"},
        {"id": "fixture", "label": " "},
        {"id": "fixture", "label": "Fixture", "aliases": (" ",)},
        {"id": "fixture", "label": "Fixture", "aliases": ("same", " same ")},
        {"id": "fixture", "label": "Fixture", "aliases": "scalar"},
        {"id": "fixture", "label": "Fixture", "unexpected": True},
    ],
)
def test_voice_capability_rejects_invalid_or_ambiguous_metadata(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        VoiceCapability.model_validate(payload)
