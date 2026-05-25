from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.voice_bank import CharacterVoice, SpeakerEmbeddingProfile, VoiceProfile

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


def make_pipeline(
    synthesizer: Synthesizer | None = None,
    *,
    config: PipelineConfig | None = None,
) -> SynthesisPipeline:
    return SynthesisPipeline(
        synthesizer or FakeSynthesizer(),
        server_profile(),
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
    ) -> SynthesisPipeline:
        return make_pipeline(synthesizer, config=config)

    return build


@pytest.fixture(name="warmable_synthesizer")
def fixture_warmable_synthesizer() -> WarmableFakeSynthesizer:
    return WarmableFakeSynthesizer()
