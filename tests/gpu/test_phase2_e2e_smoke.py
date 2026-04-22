"""Phase 2 end-to-end GPU smoke test.

This gate verifies the real Phase 2 chain in one pytest process:
Irodori VoiceDesign synthesis feeds RVC conversion for character dialogue.

Out of scope: HTTP routing, deployment orchestration, quality metrics,
multi-character coverage, and long-form synthesis.

Preconditions:
- Run on the Windows GPU host.
- The RVC sidecar is already running.
- VOICE_BANK_DIR points to a voice bank with voice_bank_rvc.toml and characters.md.
- At least one character in the voice bank has a populated RVCProfile.
- IRODORI_TTS_RUNTIME_* and IRODORI_RVC_SIDECAR_* environment variables are set
  for the host runtime.

Run:
    uv run pytest -m gpu tests/gpu/test_phase2_e2e_smoke.py -s
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.config.settings import IrodoriRuntimeSettings, RVCSidecarSettings
from irodori_tts_infra.engine.backends.irodori import create_irodori_backend
from irodori_tts_infra.engine.backends.rvc import RVCConverter, create_rvc_backend
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import PipelineConfig, SynthesizedAudio
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.text.models import Segment, SegmentKind
from irodori_tts_infra.voice_bank.models import RVCProfile, VoiceProfile
from irodori_tts_infra.voice_bank.repository import load_voice_profile

if TYPE_CHECKING:
    from collections.abc import Iterator

    from irodori_tts_infra.contracts.synthesis import SynthesisResult

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

EXPECTED_RESULT_COUNT = 2
MAX_SMOKE_SECONDS = 300
MIN_WAV_HEADER_BYTES = 44

SmokeSetup = tuple[SynthesisPipeline, "_SpyVoiceConverter", VoiceProfile, str]


class _SpyVoiceConverter:
    def __init__(self, inner: RVCConverter) -> None:
        self._inner = inner
        self.convert_calls: list[RVCProfile] = []

    def warm_up(self) -> None:
        self._inner.warm_up()

    def convert(self, audio: SynthesizedAudio, *, profile: RVCProfile) -> SynthesizedAudio:
        self.convert_calls.append(profile)
        return self._inner.convert(audio, profile=profile)

    def close(self) -> None:
        self._inner.close()


@pytest.fixture(scope="module")
def phase2_smoke_setup() -> Iterator[SmokeSetup]:
    voice_profile = _load_smoke_voice_profile()
    smoke_character_name = _smoke_character_name(voice_profile)

    backend = None
    spy = None
    try:
        try:
            irodori_settings = IrodoriRuntimeSettings()
            rvc_settings = RVCSidecarSettings()
            backend = create_irodori_backend(irodori_settings)
            rvc_converter = create_rvc_backend(rvc_settings)
            backend.warm_up()
            rvc_converter.warm_up()
        except BackendUnavailableError as exc:
            pytest.skip(f"GPU smoke backend unavailable during setup: {exc}")

        spy = _SpyVoiceConverter(rvc_converter)
        yield (
            SynthesisPipeline(
                backend,
                voice_profile,
                voice_converter=spy,
                config=PipelineConfig(capacity=1),
            ),
            spy,
            voice_profile,
            smoke_character_name,
        )
    finally:
        if spy is not None:
            with suppress(BackendUnavailableError, OSError):
                spy.close()
        if backend is not None:
            with suppress(BackendUnavailableError, OSError):
                backend.close()


def test_phase2_chain_dialogue_uses_rvc_and_narration_bypasses(
    phase2_smoke_setup: SmokeSetup,
) -> None:
    pipeline, spy, voice_profile, smoke_character_name = phase2_smoke_setup
    dialogue = Segment(
        kind=SegmentKind.DIALOGUE,
        speaker=smoke_character_name,
        text="こんにちは。",
    )
    narration = Segment(kind=SegmentKind.NARRATION, text="空は青かった。")

    result = pipeline.synthesize_batch([dialogue, narration])

    assert len(result.results) == EXPECTED_RESULT_COUNT
    for item in result.results:
        _assert_wav_result(item)
    assert len(spy.convert_calls) == 1
    assert spy.convert_calls[0] is voice_profile.characters[smoke_character_name].rvc
    assert result.total_elapsed_seconds > 0
    assert result.total_elapsed_seconds < MAX_SMOKE_SECONDS


def _load_smoke_voice_profile() -> VoiceProfile:
    voice_bank_dir_raw = os.environ.get("VOICE_BANK_DIR")
    if voice_bank_dir_raw is None:
        pytest.skip("VOICE_BANK_DIR is unset; configure the trained voice bank path")

    voice_bank_dir = Path(voice_bank_dir_raw)
    if not voice_bank_dir.is_dir():
        pytest.skip(f"VOICE_BANK_DIR does not resolve to a directory: {voice_bank_dir}")

    rvc_manifest = voice_bank_dir / "voice_bank_rvc.toml"
    characters_md = voice_bank_dir / "characters.md"
    if not rvc_manifest.is_file():
        pytest.skip(f"VOICE_BANK_DIR is missing voice_bank_rvc.toml: {rvc_manifest}")
    if not characters_md.is_file():
        pytest.skip(f"VOICE_BANK_DIR is missing characters.md: {characters_md}")

    return load_voice_profile(characters_md=characters_md, rvc_manifest=rvc_manifest)


def _smoke_character_name(voice_profile: VoiceProfile) -> str:
    for name, character in voice_profile.characters.items():
        if character.rvc is not None:
            return name
    pytest.skip("no trained RVC weights in VOICE_BANK_DIR; run RVC training SOP first")


def _assert_wav_result(result: SynthesisResult) -> None:
    assert result.wav_bytes.startswith(b"RIFF")
    assert len(result.wav_bytes) >= MIN_WAV_HEADER_BYTES
    assert result.elapsed_seconds > 0
