from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/audit_training_tones.py")
SAMPLE_RATE = 48_000


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_training_tones", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_source_records_partitions_censored_and_uncensored_captions(
    tmp_path: Path,
) -> None:
    module = _load_script()
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "Speaker": "綾希",
                    "Text": "おち◯ちんです",
                    "FilePath": "綾希\\censored.ogg",
                },
                {
                    "Speaker": "綾希",
                    "Text": "普通の台詞です",
                    "FilePath": "綾希\\plain.ogg",
                },
                {
                    "Speaker": "別人",
                    "Text": "対象外です",
                    "FilePath": "別人\\other.ogg",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = module.load_source_records(index_path, speaker="綾希")

    assert [record.caption_has_censor for record in records] == [True, False]
    assert records[0].audio_path == tmp_path / "綾希" / "censored.ogg"
    assert records[1].text == "普通の台詞です"


@pytest.mark.parametrize("frequency_hz", [234.375, 12_000.0])
def test_analyze_training_record_finds_tones_outside_synthesis_screening_band(
    frequency_hz: float,
) -> None:
    module = _load_script()
    samples = np.zeros(SAMPLE_RATE, dtype=np.float64)
    start = round(0.2 * SAMPLE_RATE)
    end = round(0.5 * SAMPLE_RATE)
    t = np.arange(end - start, dtype=np.float64) / SAMPLE_RATE
    samples[start:end] = np.sin(2 * np.pi * frequency_hz * t) * 0.5
    record = module.SourceRecord(
        audio_path=Path("plain.ogg"),
        text="伏字のない台詞",
        speaker="綾希",
        caption_has_censor=False,
    )

    result = module.analyze_training_record(
        record,
        samples=samples,
        sample_rate=SAMPLE_RATE,
    )

    assert result["analysis_status"] == "CANDIDATE"
    assert result["caption_has_censor"] is False
    intervals = result["intervals"]
    assert isinstance(intervals, list)
    assert intervals[0]["frequency_hz"] == pytest.approx(frequency_hz, abs=25.0)
