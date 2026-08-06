"""Phase 2 end-to-end GPU smoke test.

This gate verifies the real Phase 2 chain in one pytest process:
the configured Irodori-TTS VoiceDesign runtime uses Speaker Inversion embeddings
for narrator and character dialogue, including caption plus speaker conditioning.

Out of scope: HTTP routing, deployment orchestration, quality metrics,
multi-character coverage, and long-form synthesis.

Preconditions:
- Run on the Windows GPU host.
- VOICE_BANK_DIR points to a voice bank with voice_bank_speakers.toml.
- characters.md is optional, matching server startup behavior.
- At least one character in the speaker manifest has a .speaker.safetensors embedding.
- SMOKE_SPEAKER selects the character deterministically when the voice bank has
  more than one character embedding. With exactly one, the sole character is
  picked automatically.
- IRODORI_TTS_RUNTIME_* environment variables are set for the host runtime.

Run:
    uv run pytest -m gpu tests/gpu/test_phase2_e2e_smoke.py -s
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.config.settings import IrodoriRuntimeSettings
from irodori_tts_infra.engine.backends.irodori import create_irodori_backend
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import PipelineConfig, SynthesisJob
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.text.models import Segment, SegmentKind
from irodori_tts_infra.voice_bank.models import VoiceProfile
from irodori_tts_infra.voice_bank.repository import load_voice_profile

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from irodori_tts_infra.contracts.synthesis import SynthesisResult

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated\..*:DeprecationWarning",
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.nn\.utils\.weight_norm` is deprecated.*:FutureWarning",
    ),
    pytest.mark.filterwarnings(
        r"ignore:'audioop' is deprecated and slated for removal in Python 3\.13:"
        "DeprecationWarning",
    ),
    pytest.mark.filterwarnings(
        r"ignore:Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work:"
        "RuntimeWarning",
    ),
    pytest.mark.filterwarnings(
        r"ignore:unclosed file .*hparams\.yaml.*:ResourceWarning",
    ),
]

EXPECTED_RESULT_COUNT = 2
MAX_SMOKE_SECONDS = 300

SmokeSetup = tuple[SynthesisPipeline, VoiceProfile, str]


@pytest.fixture(scope="module")
def phase2_smoke_setup() -> Iterator[SmokeSetup]:
    voice_profile = _load_smoke_voice_profile()
    smoke_character_name = _smoke_character_name(voice_profile)

    backend = None
    setup_completed = False
    try:
        try:
            irodori_settings = IrodoriRuntimeSettings()
            backend = create_irodori_backend(irodori_settings)
            backend.warm_up(ref_embed=str(voice_profile.narrator.ref_embed))
            setup_completed = True
        except BackendUnavailableError as exc:
            pytest.skip(f"GPU smoke backend unavailable during setup: {exc}")

        yield (
            SynthesisPipeline(
                backend,
                voice_profile,
                config=PipelineConfig(capacity=1),
            ),
            voice_profile,
            smoke_character_name,
        )
    finally:
        if backend is not None:
            _close_component("Irodori backend", backend.close, setup_completed=setup_completed)


def _close_component(
    label: str,
    close_fn: Callable[[], None],
    *,
    setup_completed: bool,
) -> None:
    try:
        close_fn()
    except BackendUnavailableError as exc:
        if setup_completed:
            msg = f"failed to close {label} during gpu smoke teardown"
            raise RuntimeError(msg) from exc


def test_phase2_chain_uses_speaker_embeddings_for_dialogue_and_narration(
    phase2_smoke_setup: SmokeSetup,
) -> None:
    pipeline, _voice_profile, smoke_character_name = phase2_smoke_setup
    dialogue = Segment(
        kind=SegmentKind.DIALOGUE,
        speaker=smoke_character_name,
        text="こんにちは。",
    )
    narration = Segment(kind=SegmentKind.NARRATION, text="空は青かった。")

    result = pipeline.synthesize_batch([dialogue, narration])

    assert len(result.results) == EXPECTED_RESULT_COUNT
    dialogue_result, narration_result = result.results
    assert dialogue_result.segment_index == 0
    assert narration_result.segment_index == 1

    dialogue_nframes, dialogue_sample_rate = _decode_wav(dialogue_result)
    narration_nframes, narration_sample_rate = _decode_wav(narration_result)
    assert dialogue_nframes > 0
    assert narration_nframes > 0
    assert dialogue_result.elapsed_seconds > 0
    assert narration_result.elapsed_seconds > 0

    assert dialogue_sample_rate > 0
    assert narration_sample_rate > 0

    assert result.total_elapsed_seconds > 0
    assert result.total_elapsed_seconds < MAX_SMOKE_SECONDS


def test_voicedesign_combines_caption_and_speaker_embedding(
    phase2_smoke_setup: SmokeSetup,
) -> None:
    pipeline, _voice_profile, smoke_character_name = phase2_smoke_setup
    result = pipeline.synthesize_job(
        SynthesisJob(
            segment_index=0,
            text="落ち着いて読み上げます。",
            speaker=smoke_character_name,
            require_speaker=True,
            style="calm",
        ),
    )

    nframes, sample_rate = _decode_wav(result)
    assert nframes > 0
    assert sample_rate > 0
    assert result.elapsed_seconds > 0


def _load_smoke_voice_profile() -> VoiceProfile:
    voice_bank_dir_raw = os.environ.get("VOICE_BANK_DIR")
    if voice_bank_dir_raw is None:
        pytest.skip("VOICE_BANK_DIR is unset; configure the trained voice bank path")

    voice_bank_dir = Path(voice_bank_dir_raw)
    if not voice_bank_dir.is_dir():
        pytest.skip(f"VOICE_BANK_DIR does not resolve to a directory: {voice_bank_dir}")

    speaker_manifest = voice_bank_dir / "voice_bank_speakers.toml"
    if not speaker_manifest.is_file():
        pytest.skip(f"VOICE_BANK_DIR is missing voice_bank_speakers.toml: {speaker_manifest}")
    characters_md_candidate = voice_bank_dir / "characters.md"
    characters_md = characters_md_candidate if characters_md_candidate.is_file() else None

    return load_voice_profile(
        characters_md=characters_md,
        speaker_manifest=speaker_manifest,
        require_embedding_files=True,
    )


def _smoke_character_name(voice_profile: VoiceProfile) -> str:
    explicit = os.environ.get("SMOKE_SPEAKER")
    if explicit is not None:
        character = voice_profile.characters.get(explicit)
        if character is None:
            pytest.skip(f"SMOKE_SPEAKER={explicit!r} is not present in VOICE_BANK_DIR")
        return explicit

    speaker_characters = sorted(voice_profile.characters)
    if not speaker_characters:
        pytest.skip("no speaker embeddings in VOICE_BANK_DIR")
    if len(speaker_characters) > 1:
        pytest.skip(
            "multiple speaker-embedding characters in VOICE_BANK_DIR "
            f"({', '.join(speaker_characters)}); set SMOKE_SPEAKER to choose one"
        )
    return speaker_characters[0]


def _decode_wav(result: SynthesisResult) -> tuple[int, int]:
    assert result.wav_bytes.startswith(b"RIFF")
    assert result.wav_bytes[8:12] == b"WAVE"
    with wave.open(io.BytesIO(result.wav_bytes)) as reader:
        return reader.getnframes(), reader.getframerate()
