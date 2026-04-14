from __future__ import annotations

import pytest

from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.protocols import Synthesizer

pytestmark = pytest.mark.unit


def test_fake_synthesizer_implements_synthesizer_protocol() -> None:
    assert isinstance(FakeSynthesizer(), Synthesizer)
