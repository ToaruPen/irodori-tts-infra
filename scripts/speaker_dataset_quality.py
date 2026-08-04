from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_ANALYZER_PATH = Path(__file__).with_name("analyze_nko_beep_matrix.py")
_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "_speaker_dataset_quality_analyzer",
    _ANALYZER_PATH,
)
if _ANALYZER_SPEC is None or _ANALYZER_SPEC.loader is None:
    message = f"cannot load tone analyzer: {_ANALYZER_PATH}"
    raise RuntimeError(message)
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
sys.modules[_ANALYZER_SPEC.name] = _ANALYZER
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)

Decision = Literal[
    "KEEP",
    "KEEP_RECAPTIONED",
    "REVIEW",
    "EXCLUDE_CONFIRMED_TONE",
    "EXCLUDE_INVALID_AUDIO",
    "EXCLUDE_TRANSCRIPT_MISMATCH",
    "EXCLUDE_DUPLICATE",
]

BROAD_TONE_CONFIG = _ANALYZER.ToneConfig(
    analysis_min_frequency_hz=40.0,
    min_tone_frequency_hz=80.0,
    max_frequency_hz=20_000.0,
)
AUTO_EXCLUDE_MAX_FREQUENCY_STD_HZ = 2.0
AUTO_EXCLUDE_MIN_PEAK_ENERGY_RATIO = 0.95
AUTO_EXCLUDE_MAX_NORMALIZED_ENTROPY = 0.20
AUTO_EXCLUDE_MIN_DURATION_SECONDS = 0.12
AUTO_EXCLUDE_MAX_HARMONIC_RATIO = 0.03
LOW_LEVEL_RMS_DBFS = -60.0
EXTREME_CLIPPED_FRACTION = 0.01
CLIPPED_AMPLITUDE = 0.999
POWER_FLOOR = np.finfo(np.float64).tiny
MIN_FFT_SAMPLES = 2
CENSOR_MARKERS = ("◯", "○", "〇")  # noqa: RUF001 - source captions use these glyphs

ReviewLabelValue = Literal["TONE", "VOICE", "UNSURE"]
CaptionRepairDecision = Literal["UNCHANGED", "REPAIRED", "REVIEW"]


@dataclass(frozen=True, slots=True)
class AudioDecision:
    decision: Decision
    reasons: tuple[str, ...]
    intervals: tuple[object, ...]
    harmonic_ratio: float
    clipped_fraction: float
    rms_dbfs: float


@dataclass(frozen=True, slots=True)
class ToneSignature:
    signature_id: str
    dataset_id: str
    center_frequency_hz: float
    tolerance_fft_bins: float = 1.5


@dataclass(frozen=True, slots=True)
class ReviewLabel:
    label: ReviewLabelValue
    reviewer: str
    note: str


@dataclass(frozen=True, slots=True)
class CaptionRule:
    rule_id: str
    source: str
    replacement: str


@dataclass(frozen=True, slots=True)
class CaptionRepair:
    original_text: str
    text: str
    decision: CaptionRepairDecision
    rule_id: str | None


def apply_label_override(automatic: Decision, label: ReviewLabel) -> Decision:
    decisions: dict[ReviewLabelValue, Decision] = {
        "TONE": "EXCLUDE_CONFIRMED_TONE",
        "VOICE": "KEEP",
        "UNSURE": "REVIEW",
    }
    try:
        labeled_decision = decisions[label.label]
    except KeyError as exc:
        message = f"unsupported review label: {label.label}"
        raise ValueError(message) from exc
    if automatic == "REVIEW" or (automatic == "KEEP" and label.label == "TONE"):
        return labeled_decision
    return automatic


def repair_caption(text: str, *, rules: Sequence[CaptionRule]) -> CaptionRepair:
    if not _has_censor_marker(text):
        return CaptionRepair(
            original_text=text,
            text=text,
            decision="UNCHANGED",
            rule_id=None,
        )
    applicable = tuple(rule for rule in rules if rule.source and rule.source in text)
    if len(applicable) != 1:
        return CaptionRepair(
            original_text=text,
            text=text,
            decision="REVIEW",
            rule_id=None,
        )
    rule = applicable[0]
    repaired = text.replace(rule.source, rule.replacement)
    if _has_censor_marker(repaired):
        return CaptionRepair(
            original_text=text,
            text=text,
            decision="REVIEW",
            rule_id=None,
        )
    return CaptionRepair(
        original_text=text,
        text=repaired,
        decision="REPAIRED",
        rule_id=rule.rule_id,
    )


def _has_censor_marker(text: str) -> bool:
    return any(marker in text for marker in CENSOR_MARKERS)


def classify_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    dataset_id: str,
    confirmed_signatures: Sequence[ToneSignature],
) -> AudioDecision:
    if sample_rate <= 0:
        message = "sample_rate must be positive"
        raise ValueError(message)
    normalized = np.asarray(samples, dtype=np.float64)
    if normalized.ndim != 1:
        message = "samples must be one-dimensional"
        raise ValueError(message)
    invalid_reason = _invalid_reason(normalized)
    if invalid_reason is not None:
        return AudioDecision(
            decision="EXCLUDE_INVALID_AUDIO",
            reasons=(invalid_reason,),
            intervals=(),
            harmonic_ratio=0.0,
            clipped_fraction=0.0,
            rms_dbfs=float("-inf"),
        )

    rms = float(np.sqrt(np.mean(normalized**2)))
    rms_dbfs = float(20.0 * np.log10(max(rms, POWER_FLOOR)))
    clipped_fraction = float(np.mean(np.abs(normalized) >= CLIPPED_AMPLITUDE))
    intervals = _ANALYZER.detect_narrowband_intervals(
        normalized,
        sample_rate,
        BROAD_TONE_CONFIG,
    )
    harmonic_ratios = _harmonic_ratios(
        normalized,
        sample_rate=sample_rate,
        intervals=intervals,
    )
    harmonic_ratio = max(harmonic_ratios, default=0.0)
    matched_signature = next(
        (
            signature
            for interval, interval_harmonic_ratio in zip(
                intervals,
                harmonic_ratios,
                strict=True,
            )
            if _is_confirmed_tone(interval, harmonic_ratio=interval_harmonic_ratio)
            for signature in confirmed_signatures
            if _matches_signature(
                interval,
                signature=signature,
                dataset_id=dataset_id,
                sample_rate=sample_rate,
            )
        ),
        None,
    )
    if matched_signature is not None:
        return AudioDecision(
            decision="EXCLUDE_CONFIRMED_TONE",
            reasons=(
                "stable_pure_tone",
                f"confirmed_signature:{matched_signature.signature_id}",
            ),
            intervals=intervals,
            harmonic_ratio=harmonic_ratio,
            clipped_fraction=clipped_fraction,
            rms_dbfs=rms_dbfs,
        )
    review_reasons: list[str] = []
    if intervals:
        has_unmatched_pure_tone = any(
            _is_confirmed_tone(interval, harmonic_ratio=interval_harmonic_ratio)
            for interval, interval_harmonic_ratio in zip(
                intervals,
                harmonic_ratios,
                strict=True,
            )
        )
        review_reasons.append(
            "unmatched_pure_tone" if has_unmatched_pure_tone else "narrowband_candidate"
        )
    if clipped_fraction >= EXTREME_CLIPPED_FRACTION:
        review_reasons.append("extreme_clipping")
    if rms_dbfs <= LOW_LEVEL_RMS_DBFS:
        review_reasons.append("low_level_audio")
    return AudioDecision(
        decision="REVIEW" if review_reasons else "KEEP",
        reasons=tuple(review_reasons) if review_reasons else ("valid_voice_audio",),
        intervals=intervals,
        harmonic_ratio=harmonic_ratio,
        clipped_fraction=clipped_fraction,
        rms_dbfs=rms_dbfs,
    )


def _invalid_reason(samples: np.ndarray) -> str | None:
    if samples.size == 0:
        return "empty_audio"
    if not np.all(np.isfinite(samples)):
        return "non_finite_audio"
    if not np.any(samples):
        return "all_zero_audio"
    return None


def _is_confirmed_tone(interval: object, *, harmonic_ratio: float) -> bool:
    duration = float(interval.end_seconds - interval.start_seconds)
    return (
        interval.frequency_std_hz <= AUTO_EXCLUDE_MAX_FREQUENCY_STD_HZ
        and interval.peak_energy_ratio >= AUTO_EXCLUDE_MIN_PEAK_ENERGY_RATIO
        and interval.normalized_entropy <= AUTO_EXCLUDE_MAX_NORMALIZED_ENTROPY
        and duration >= AUTO_EXCLUDE_MIN_DURATION_SECONDS
        and harmonic_ratio <= AUTO_EXCLUDE_MAX_HARMONIC_RATIO
    )


def _matches_signature(
    interval: object,
    *,
    signature: ToneSignature,
    dataset_id: str,
    sample_rate: int,
) -> bool:
    if signature.dataset_id != dataset_id:
        return False
    fft_bin_width_hz = sample_rate / BROAD_TONE_CONFIG.window_size
    tolerance_hz = signature.tolerance_fft_bins * fft_bin_width_hz
    return abs(interval.frequency_hz - signature.center_frequency_hz) <= tolerance_hz


def _harmonic_ratios(
    samples: np.ndarray,
    *,
    sample_rate: int,
    intervals: tuple[object, ...],
) -> tuple[float, ...]:
    return tuple(
        _harmonic_ratio(
            samples,
            sample_rate=sample_rate,
            start_seconds=interval.start_seconds,
            end_seconds=interval.end_seconds,
            fundamental_hz=interval.frequency_hz,
        )
        for interval in intervals
    )


def _harmonic_ratio(
    samples: np.ndarray,
    *,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    fundamental_hz: float,
) -> float:
    start = max(0, round(start_seconds * sample_rate))
    end = min(samples.size, round(end_seconds * sample_rate))
    segment = samples[start:end]
    if segment.size < MIN_FFT_SAMPLES:
        return 0.0
    power = np.abs(np.fft.rfft(segment * np.hanning(segment.size))) ** 2
    frequencies = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    audible = (frequencies >= BROAD_TONE_CONFIG.analysis_min_frequency_hz) & (
        frequencies <= BROAD_TONE_CONFIG.max_frequency_hz
    )
    total_power = float(power[audible].sum())
    if total_power <= POWER_FLOOR:
        return 0.0
    harmonic_power = 0.0
    harmonic_hz = 2.0 * fundamental_hz
    while harmonic_hz <= BROAD_TONE_CONFIG.max_frequency_hz:
        mask = np.abs(frequencies - harmonic_hz) <= BROAD_TONE_CONFIG.peak_half_width_hz
        harmonic_power += float(power[mask].sum())
        harmonic_hz += fundamental_hz
    return harmonic_power / total_power
