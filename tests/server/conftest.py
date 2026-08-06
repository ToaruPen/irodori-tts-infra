from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.voice_bank import (
    CharacterVoice,
    PortableVoice,
    SpeakerEmbeddingProfile,
    VoiceProfile,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from irodori_tts_infra.engine.models import PipelineConfig
    from irodori_tts_infra.engine.protocols import Synthesizer


def server_profile() -> VoiceProfile:
    return VoiceProfile(
        characters={
            "ミカ": CharacterVoice(
                name="ミカ",
                speaker=SpeakerEmbeddingProfile(
                    "speakers/mika.speaker.safetensors",  # type: ignore[arg-type]
                ),
            ),
        },
        narrator=SpeakerEmbeddingProfile(
            "speakers/narrator.speaker.safetensors",  # type: ignore[arg-type]
        ),
    )


def catalog_profile(count: int) -> VoiceProfile:
    narrator = SpeakerEmbeddingProfile(
        "speakers/fixture-narrator.speaker.safetensors",  # type: ignore[arg-type]
    )
    catalog = tuple(
        PortableVoice(
            id=f"fixture-voice-{index}",
            label=f"Fixture voice {index}",
            aliases=(f"fixture-alias-{index}",),
            default=index == 0,
            speaker=SpeakerEmbeddingProfile(
                f"speakers/fixture-{index}.speaker.safetensors",  # type: ignore[arg-type]
            ),
        )
        for index in range(count)
    )
    return VoiceProfile(characters={}, narrator=narrator, catalog=catalog)


def make_pipeline(
    synthesizer: Synthesizer | None = None,
    *,
    config: PipelineConfig | None = None,
    voice_profile: VoiceProfile | None = None,
) -> SynthesisPipeline:
    return SynthesisPipeline(
        synthesizer or FakeSynthesizer(),
        voice_profile or server_profile(),
        config=config,
    )


class WarmableFakeSynthesizer(FakeSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.warm_up_calls = 0
        self.close_calls = 0
        self.warm_up_ref_embeds: list[str] = []

    def warm_up(self, *, ref_embed: str) -> None:
        self.warm_up_calls += 1
        self.warm_up_ref_embeds.append(ref_embed)

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(name="pipeline_factory")
def fixture_pipeline_factory() -> Callable[..., SynthesisPipeline]:
    def build(
        synthesizer: Synthesizer | None = None,
        config: PipelineConfig | None = None,
        voice_profile: VoiceProfile | None = None,
    ) -> SynthesisPipeline:
        return make_pipeline(synthesizer, config=config, voice_profile=voice_profile)

    return build


@pytest.fixture(name="catalog_profile_factory")
def fixture_catalog_profile_factory() -> Callable[[int], VoiceProfile]:
    return catalog_profile


@pytest.fixture(name="warmable_synthesizer")
def fixture_warmable_synthesizer() -> WarmableFakeSynthesizer:
    return WarmableFakeSynthesizer()
