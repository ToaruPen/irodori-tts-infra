from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/speaker_dataset_quality.py")
SAMPLE_RATE = 48_000


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("speaker_dataset_quality", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("frequency_hz", [703.125, 1_000.0])
def test_classify_audio_excludes_matching_confirmed_tone(frequency_hz: float) -> None:
    module = _load_script()
    samples = np.zeros(SAMPLE_RATE, dtype=np.float64)
    start = round(0.2 * SAMPLE_RATE)
    end = round(0.5 * SAMPLE_RATE)
    t = np.arange(end - start, dtype=np.float64) / SAMPLE_RATE
    samples[start:end] = 0.5 * np.sin(2 * np.pi * frequency_hz * t)

    signature = module.ToneSignature(
        signature_id="confirmed",
        dataset_id="dataset-a",
        center_frequency_hz=frequency_hz,
    )
    result = module.classify_audio(
        samples,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(signature,),
    )

    assert result.decision == "EXCLUDE_CONFIRMED_TONE"
    assert result.intervals
    assert "stable_pure_tone" in result.reasons


@pytest.mark.parametrize("frequency_hz", [350.0, 12_000.0])
def test_classify_audio_reviews_unregistered_pure_tone(frequency_hz: float) -> None:
    module = _load_script()
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    samples = 0.5 * np.sin(2 * np.pi * frequency_hz * t)

    result = module.classify_audio(
        samples,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(),
    )

    assert result.decision == "REVIEW"
    assert "unmatched_pure_tone" in result.reasons


def test_classify_audio_does_not_propagate_signature_across_datasets() -> None:
    module = _load_script()
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    samples = 0.5 * np.sin(2 * np.pi * 1_000.0 * t)
    signature = module.ToneSignature(
        signature_id="dataset-a-1khz",
        dataset_id="dataset-a",
        center_frequency_hz=1_000.0,
    )

    result = module.classify_audio(
        samples,
        SAMPLE_RATE,
        dataset_id="dataset-b",
        confirmed_signatures=(signature,),
    )

    assert result.decision == "REVIEW"


def test_classify_audio_keeps_broadband_breath() -> None:
    module = _load_script()
    rng = np.random.default_rng(42)
    envelope = np.sin(np.linspace(0.0, np.pi, SAMPLE_RATE)) ** 2
    breath = rng.normal(0.0, 0.04, SAMPLE_RATE) * envelope

    result = module.classify_audio(
        breath,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(),
    )

    assert result.decision == "KEEP"
    assert result.intervals == ()


def test_classify_audio_does_not_exclude_harmonic_vocalization() -> None:
    module = _load_script()
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    vocalization = sum(
        (0.2 / harmonic) * np.sin(2 * np.pi * 220.0 * harmonic * t) for harmonic in range(1, 8)
    )

    result = module.classify_audio(
        vocalization,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(),
    )

    assert result.decision in {"KEEP", "REVIEW"}
    assert result.decision != "EXCLUDE_CONFIRMED_TONE"


def test_classify_audio_keeps_short_harmonic_moan() -> None:
    module = _load_script()
    samples = np.zeros(SAMPLE_RATE, dtype=np.float64)
    start = round(0.2 * SAMPLE_RATE)
    end = round(0.34 * SAMPLE_RATE)
    t = np.arange(end - start, dtype=np.float64) / SAMPLE_RATE
    envelope = np.sin(np.linspace(0.0, np.pi, end - start)) ** 2
    samples[start:end] = envelope * (
        0.2 * np.sin(2 * np.pi * 190.0 * t)
        + 0.1 * np.sin(2 * np.pi * 380.0 * t)
        + 0.05 * np.sin(2 * np.pi * 570.0 * t)
    )

    result = module.classify_audio(
        samples,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(),
    )

    assert result.decision == "KEEP"


def test_classify_audio_reviews_low_level_nonzero_breath() -> None:
    module = _load_script()
    rng = np.random.default_rng(7)
    breath = rng.normal(0.0, 0.0001, SAMPLE_RATE)

    result = module.classify_audio(
        breath,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(),
    )

    assert result.decision == "REVIEW"
    assert "low_level_audio" in result.reasons


def test_classify_audio_reviews_extreme_clipping() -> None:
    module = _load_script()
    clipped = np.ones(SAMPLE_RATE, dtype=np.float64)
    clipped[::2] = -1.0

    result = module.classify_audio(
        clipped,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(),
    )

    assert result.decision == "REVIEW"
    assert result.clipped_fraction == pytest.approx(1.0)
    assert "extreme_clipping" in result.reasons


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([], dtype=np.float64),
        np.zeros(SAMPLE_RATE, dtype=np.float64),
        np.full(SAMPLE_RATE, np.nan, dtype=np.float64),
    ],
)
def test_classify_audio_rejects_invalid_audio(invalid: np.ndarray) -> None:
    module = _load_script()

    result = module.classify_audio(
        invalid,
        SAMPLE_RATE,
        dataset_id="dataset-a",
        confirmed_signatures=(),
    )

    assert result.decision == "EXCLUDE_INVALID_AUDIO"
    assert result.reasons


def test_classify_audio_rejects_invalid_sample_rate() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        module.classify_audio(
            np.ones(100, dtype=np.float64),
            0,
            dataset_id="dataset-a",
            confirmed_signatures=(),
        )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("VOICE", "KEEP"),
        ("TONE", "EXCLUDE_CONFIRMED_TONE"),
        ("UNSURE", "REVIEW"),
    ],
)
def test_apply_label_override_resolves_review_decision(label: str, expected: str) -> None:
    module = _load_script()
    review_label = module.ReviewLabel(
        label=label,
        reviewer="user",
        note="listening review",
    )

    result = module.apply_label_override("REVIEW", review_label)

    assert result == expected


def test_apply_label_override_excludes_automatic_keep_labeled_as_tone() -> None:
    module = _load_script()
    review_label = module.ReviewLabel(label="TONE", reviewer="user", note="")

    result = module.apply_label_override("KEEP", review_label)

    assert result == "EXCLUDE_CONFIRMED_TONE"


@pytest.mark.parametrize(
    ("automatic", "label"),
    [
        ("KEEP", "VOICE"),
        ("KEEP", "UNSURE"),
        ("EXCLUDE_INVALID_AUDIO", "TONE"),
        ("EXCLUDE_TRANSCRIPT_MISMATCH", "TONE"),
        ("EXCLUDE_DUPLICATE", "VOICE"),
    ],
)
def test_apply_label_override_preserves_non_review_decision(
    automatic: str,
    label: str,
) -> None:
    module = _load_script()
    review_label = module.ReviewLabel(label=label, reviewer="user", note="")

    result = module.apply_label_override(automatic, review_label)

    assert result == automatic


def test_repair_caption_uses_one_explicit_rule() -> None:
    module = _load_script()
    rule = module.CaptionRule(
        rule_id="ochinchin-nasal",
        source="おち◯ちん",
        replacement="おちんちん",
    )

    repaired = module.repair_caption("おち◯ちんです", rules=(rule,))

    assert repaired.original_text == "おち◯ちんです"
    assert repaired.text == "おちんちんです"
    assert repaired.decision == "REPAIRED"
    assert repaired.rule_id == "ochinchin-nasal"


def test_repair_caption_keeps_expressive_text_without_marker() -> None:
    module = _load_script()

    repaired = module.repair_caption("はぁ、んっ", rules=())

    assert repaired.text == "はぁ、んっ"
    assert repaired.decision == "UNCHANGED"
    assert repaired.rule_id is None


def test_repair_caption_reviews_unknown_marker() -> None:
    module = _load_script()

    repaired = module.repair_caption("未知◯語", rules=())

    assert repaired.text == "未知◯語"
    assert repaired.decision == "REVIEW"
    assert repaired.rule_id is None


def test_repair_caption_reviews_conflicting_rules() -> None:
    module = _load_script()
    rules = (
        module.CaptionRule(rule_id="first", source="◯", replacement="ん"),
        module.CaptionRule(rule_id="second", source="◯", replacement="ま"),
    )

    repaired = module.repair_caption("おち◯ちん", rules=rules)

    assert repaired.text == "おち◯ちん"
    assert repaired.decision == "REVIEW"
    assert repaired.rule_id is None
