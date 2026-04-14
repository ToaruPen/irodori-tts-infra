from __future__ import annotations

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import VoiceProfileResponse

pytestmark = pytest.mark.unit


def test_voice_profile_rejects_blank_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        VoiceProfileResponse(name=" ", caption="y")

    assert any(err.get("loc") == ("name",) for err in exc_info.value.errors())


def test_voice_profile_rejects_scalar_aliases() -> None:
    with pytest.raises(ValidationError) as exc_info:
        VoiceProfileResponse.model_validate(
            {"name": "x", "caption": "y", "aliases": "narrator"},
        )

    assert any(err.get("loc") == ("aliases",) for err in exc_info.value.errors())
