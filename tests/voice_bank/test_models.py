from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from irodori_tts_infra.voice_bank import CharacterVoice, RVCProfile

if TYPE_CHECKING:
    from collections.abc import MutableMapping

pytestmark = pytest.mark.unit


def test_rvc_profile_compares_by_value() -> None:
    left = RVCProfile(
        model_path=Path("models/chizuru.pth"),
        sample_rate=40000,
        neutral_prototype=Path("prototypes/chizuru-neutral.npy"),
        state_prototypes={"happy": Path("prototypes/chizuru-happy.npy")},
    )
    right = RVCProfile(
        model_path=Path("models/chizuru.pth"),
        sample_rate=40000,
        neutral_prototype=Path("prototypes/chizuru-neutral.npy"),
        state_prototypes={"happy": Path("prototypes/chizuru-happy.npy")},
    )

    assert left == right


def test_rvc_profile_is_frozen() -> None:
    profile = RVCProfile(model_path=Path("models/chizuru.pth"), sample_rate=40000)

    with pytest.raises(FrozenInstanceError):
        profile.sample_rate = 48000  # type: ignore[misc]


def test_rvc_profile_state_prototypes_defaults_to_empty_mapping() -> None:
    profile = RVCProfile(model_path=Path("models/chizuru.pth"), sample_rate=40000)

    assert profile.state_prototypes == {}


def test_rvc_profile_state_prototypes_is_immutable_snapshot() -> None:
    state_prototypes = {"happy": Path("prototypes/chizuru-happy.npy")}

    profile = RVCProfile(
        model_path=Path("models/chizuru.pth"),
        sample_rate=40000,
        state_prototypes=state_prototypes,
    )
    state_prototypes["sad"] = Path("prototypes/chizuru-sad.npy")

    assert profile.state_prototypes == {"happy": Path("prototypes/chizuru-happy.npy")}
    with pytest.raises(TypeError):
        cast("MutableMapping[str, Path]", profile.state_prototypes)["angry"] = Path(
            "prototypes/chizuru-angry.npy",
        )


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_rvc_profile_rejects_non_positive_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError, match="sample_rate must be greater than 0"):
        RVCProfile(model_path=Path("models/chizuru.pth"), sample_rate=sample_rate)


def test_character_voice_defaults_to_no_rvc_profile() -> None:
    character = CharacterVoice(name="チヅル", caption="チヅルの声。")

    assert character.rvc is None
