from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.engine.backends.fake import FakeSynthesizer

if TYPE_CHECKING:
    from irodori_tts_infra.engine.protocols import Synthesizer

pytestmark = pytest.mark.unit


def test_fake_synthesizer_implements_synthesizer_protocol() -> None:
    synth: Synthesizer = FakeSynthesizer()

    assert callable(synth.synthesize)
