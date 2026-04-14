from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.voice_bank import CharacterVoice, VoiceProfile

if TYPE_CHECKING:
    from collections.abc import Callable

    from irodori_tts_infra.engine.models import PipelineConfig
    from irodori_tts_infra.engine.protocols import Synthesizer


def server_profile() -> VoiceProfile:
    return VoiceProfile(
        characters={
            "ミカ": CharacterVoice(
                name="ミカ",
                caption="若い女性が、明るく楽しそうに話している。若々しい声。",
            ),
        },
        narrator_caption="落ち着いた大人の女性が読み上げている。",
        generic_dialogue_caption="若い人が自然な口調で話している。",
    )


def make_pipeline(
    synthesizer: Synthesizer | None = None,
    *,
    config: PipelineConfig | None = None,
) -> SynthesisPipeline:
    return SynthesisPipeline(synthesizer or FakeSynthesizer(), server_profile(), config=config)


class WarmableFakeSynthesizer(FakeSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.warm_up_calls = 0
        self.close_calls = 0

    def warm_up(self) -> None:
        self.warm_up_calls += 1

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
