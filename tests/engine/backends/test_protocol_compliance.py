from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.config.settings import RVCSidecarSettings
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.backends.rvc import RVCConverter
from irodori_tts_infra.engine.models import SynthesizedAudio
from irodori_tts_infra.engine.protocols import VoiceConverter
from irodori_tts_infra.voice_bank import RVCProfile

if TYPE_CHECKING:
    from irodori_tts_infra.engine.protocols import Synthesizer

pytestmark = pytest.mark.unit
PROTOCOL_SAMPLE_RATE = 40_000


def _protocol_wav_bytes() -> bytes:
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(PROTOCOL_SAMPLE_RATE)
            wav_file.writeframes(b"\x00\x00")
        return buffer.getvalue()


def test_fake_synthesizer_implements_synthesizer_protocol() -> None:
    synth: Synthesizer = FakeSynthesizer()

    assert callable(synth.synthesize)


class _ProtocolFakeRVCClient:
    @staticmethod
    def view_api() -> dict[str, object]:
        return {"named_endpoints": {"/infer_convert": {}}}

    @staticmethod
    def predict(*_args: object, **_kwargs: object) -> tuple[str, tuple[int, list[float]]]:
        return ("Success", (PROTOCOL_SAMPLE_RATE, [0.0]))

    @staticmethod
    def close() -> None:
        return None


def test_rvc_backend_implements_voice_converter_protocol() -> None:
    converter: VoiceConverter = RVCConverter(
        client=_ProtocolFakeRVCClient(),
        settings=RVCSidecarSettings(),
    )

    result = converter.convert(
        SynthesizedAudio(
            wav_bytes=_protocol_wav_bytes(),
            sample_rate=PROTOCOL_SAMPLE_RATE,
        ),
        profile=RVCProfile(
            model_path=Path("voice.pth"),
            sample_rate=PROTOCOL_SAMPLE_RATE,
        ),
    )

    assert isinstance(converter, VoiceConverter)
    assert callable(converter.convert)
    assert result.sample_rate == PROTOCOL_SAMPLE_RATE
