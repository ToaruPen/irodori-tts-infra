from __future__ import annotations

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import VoiceProfileResponse

pytestmark = pytest.mark.unit


def test_voice_profile_rejects_blank_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        VoiceProfileResponse(name=" ")

    assert any(err.get("loc") == ("name",) for err in exc_info.value.errors())


def test_voice_profile_rejects_scalar_aliases() -> None:
    with pytest.raises(ValidationError) as exc_info:
        VoiceProfileResponse.model_validate(
            {
                "name": "x",
                "aliases": "narrator",
            },
        )

    assert any(err.get("loc") == ("aliases",) for err in exc_info.value.errors())


def test_voice_profile_response_does_not_serialize_ref_embed() -> None:
    profile = VoiceProfileResponse(name="Narrator", aliases=("語り手",))

    assert profile.model_dump() == {"name": "Narrator", "aliases": ("語り手",)}
