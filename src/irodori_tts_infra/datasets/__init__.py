from __future__ import annotations

from irodori_tts_infra.datasets.models import ExtractedClip, ExtractionIndex
from irodori_tts_infra.datasets.moe_speech import (
    DEFAULT_DATASET_REPO,
    DEFAULT_MAX_BYTES,
    DEFAULT_OUTPUT_SAMPLE_RATE,
    DatasetExtractionError,
    MoeSpeechRecord,
    NsfwSubsetUnavailableError,
    UnsupportedAudioFormatError,
    extract_character_dataset,
    stream_character_records,
)

__all__ = [
    "DEFAULT_DATASET_REPO",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_OUTPUT_SAMPLE_RATE",
    "DatasetExtractionError",
    "ExtractedClip",
    "ExtractionIndex",
    "MoeSpeechRecord",
    "NsfwSubsetUnavailableError",
    "UnsupportedAudioFormatError",
    "extract_character_dataset",
    "stream_character_records",
]
