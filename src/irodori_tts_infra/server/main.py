from __future__ import annotations

import os
from pathlib import Path

from irodori_tts_infra.config.settings import IrodoriRuntimeSettings
from irodori_tts_infra.engine.backends.irodori import create_irodori_backend
from irodori_tts_infra.engine.errors import VoiceBankInvalidError
from irodori_tts_infra.engine.models import PipelineConfig
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.server.app import create_app_from_factory
from irodori_tts_infra.voice_bank.repository import (
    SPEAKER_MANIFEST_FILENAME,
    load_voice_profile,
)


def _build_pipeline(settings: IrodoriRuntimeSettings) -> SynthesisPipeline:
    try:
        speaker_manifest = _resolve_speaker_manifest()
        characters_md = _resolve_characters_markdown(speaker_manifest)
        voice_profile = load_voice_profile(
            characters_md,
            speaker_manifest=speaker_manifest,
            require_embedding_files=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        msg = "voice bank configuration is invalid"
        raise VoiceBankInvalidError(msg) from exc
    backend = create_irodori_backend(settings)
    return SynthesisPipeline(
        backend,
        voice_profile,
        config=PipelineConfig(generation=settings.public_generation),
    )


def _resolve_speaker_manifest() -> Path:
    explicit = os.environ.get("VOICE_BANK_SPEAKER_MANIFEST")
    if explicit:
        return Path(explicit).expanduser()

    voice_bank_dir = os.environ.get("VOICE_BANK_DIR")
    if voice_bank_dir:
        return Path(voice_bank_dir).expanduser() / SPEAKER_MANIFEST_FILENAME

    msg = "VOICE_BANK_SPEAKER_MANIFEST or VOICE_BANK_DIR is required"
    raise ValueError(msg)


def _resolve_characters_markdown(speaker_manifest: Path) -> Path | None:
    voice_bank_dir = os.environ.get("VOICE_BANK_DIR")
    if voice_bank_dir:
        candidate = Path(voice_bank_dir).expanduser() / "characters.md"
    else:
        candidate = speaker_manifest.parent / "characters.md"
    return candidate if candidate.is_file() else None


settings = IrodoriRuntimeSettings()
app = create_app_from_factory(
    lambda: _build_pipeline(settings),
    generation=settings.public_generation,
    emoji_conditioning_supported=settings.emoji_conditioning_supported,
)
