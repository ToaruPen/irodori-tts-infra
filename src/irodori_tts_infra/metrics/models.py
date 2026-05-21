from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

MIN_MOS_SCORE = 1.0
MAX_MOS_SCORE = 5.0


class QualityGateStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - quality verdict, not a secret.
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class RelativeMargin:
    value: float
    nearest_character: str
    nearest_identity: float

    def __post_init__(self) -> None:
        _validate_finite(self.value, "value")
        _validate_non_blank(self.nearest_character, "nearest_character")
        _validate_unit_score(self.nearest_identity, "nearest_identity")


@dataclass(frozen=True, slots=True)
class QualityScores:
    primary_identity: float | None = None
    secondary_identity: float | None = None
    other_identities: Mapping[str, float] = field(default_factory=dict)
    utmos: float | None = None
    dnsmos: float | None = None
    f0_median_hz: float | None = None
    f0_range_hz: float | None = None
    f0_std_hz: float | None = None
    voiced_ratio: float | None = None
    speaking_rate: float | None = None
    pause_ratio: float | None = None
    rms_energy: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_unit_score(self.primary_identity, "primary_identity")
        _validate_optional_unit_score(self.secondary_identity, "secondary_identity")
        _validate_optional_mos_score(self.utmos, "utmos")
        _validate_optional_mos_score(self.dnsmos, "dnsmos")
        _validate_optional_non_negative(self.f0_median_hz, "f0_median_hz")
        _validate_optional_non_negative(self.f0_range_hz, "f0_range_hz")
        _validate_optional_non_negative(self.f0_std_hz, "f0_std_hz")
        _validate_optional_unit_score(self.voiced_ratio, "voiced_ratio")
        _validate_optional_non_negative(self.speaking_rate, "speaking_rate")
        _validate_optional_unit_score(self.pause_ratio, "pause_ratio")
        _validate_optional_non_negative(self.rms_energy, "rms_energy")
        object.__setattr__(
            self,
            "other_identities",
            MappingProxyType(
                {
                    _validated_character_name(name): _validated_unit_score(
                        score,
                        f"other_identities.{name}",
                    )
                    for name, score in self.other_identities.items()
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    primary_identity_min: float | None = None
    secondary_identity_min: float | None = None
    relative_margin_min: float | None = None
    utmos_min: float | None = None
    dnsmos_min: float | None = None
    f0_median_min_hz: float | None = None
    f0_median_max_hz: float | None = None
    f0_range_min_hz: float | None = None
    f0_range_max_hz: float | None = None
    f0_std_max_hz: float | None = None
    voiced_ratio_min: float | None = None
    speaking_rate_min: float | None = None
    speaking_rate_max: float | None = None
    pause_ratio_max: float | None = None
    rms_energy_min: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_unit_score(self.primary_identity_min, "primary_identity_min")
        _validate_optional_unit_score(self.secondary_identity_min, "secondary_identity_min")
        _validate_optional_unit_score(self.relative_margin_min, "relative_margin_min")
        _validate_optional_mos_score(self.utmos_min, "utmos_min")
        _validate_optional_mos_score(self.dnsmos_min, "dnsmos_min")
        _validate_optional_non_negative(self.f0_median_min_hz, "f0_median_min_hz")
        _validate_optional_non_negative(self.f0_median_max_hz, "f0_median_max_hz")
        _validate_optional_non_negative(self.f0_range_min_hz, "f0_range_min_hz")
        _validate_optional_non_negative(self.f0_range_max_hz, "f0_range_max_hz")
        _validate_optional_non_negative(self.f0_std_max_hz, "f0_std_max_hz")
        _validate_optional_unit_score(self.voiced_ratio_min, "voiced_ratio_min")
        _validate_optional_non_negative(self.speaking_rate_min, "speaking_rate_min")
        _validate_optional_non_negative(self.speaking_rate_max, "speaking_rate_max")
        _validate_optional_unit_score(self.pause_ratio_max, "pause_ratio_max")
        _validate_optional_non_negative(self.rms_energy_min, "rms_energy_min")
        _validate_optional_range(
            self.f0_median_min_hz,
            self.f0_median_max_hz,
            "f0_median_min_hz",
            "f0_median_max_hz",
        )
        _validate_optional_range(
            self.f0_range_min_hz,
            self.f0_range_max_hz,
            "f0_range_min_hz",
            "f0_range_max_hz",
        )
        if (
            self.speaking_rate_min is not None
            and self.speaking_rate_max is not None
            and self.speaking_rate_min > self.speaking_rate_max
        ):
            msg = "speaking_rate_min must be less than or equal to speaking_rate_max"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class QualityGateInput:
    scores: QualityScores
    thresholds: QualityThresholds
    non_speech: bool = False


@dataclass(frozen=True, slots=True)
class QualityGateIssue:
    code: str
    metric: str
    observed: float | None
    threshold: float
    warning: bool = False
    nearest_character: str | None = None

    def __post_init__(self) -> None:
        _validate_non_blank(self.code, "code")
        _validate_non_blank(self.metric, "metric")
        if self.observed is not None:
            _validate_finite(self.observed, "observed")
        _validate_finite(self.threshold, "threshold")
        if self.nearest_character is not None:
            _validate_non_blank(self.nearest_character, "nearest_character")


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    status: QualityGateStatus
    issues: tuple[QualityGateIssue, ...] = ()


def _validated_character_name(value: str) -> str:
    _validate_non_blank(value, "character name")
    return value


def _validated_unit_score(value: float, field_name: str) -> float:
    _validate_unit_score(value, field_name)
    return value


def _validate_optional_unit_score(value: float | None, field_name: str) -> None:
    if value is not None:
        _validate_unit_score(value, field_name)


def _validate_unit_score(value: float, field_name: str) -> None:
    _validate_finite(value, field_name)
    if value < 0.0 or value > 1.0:
        msg = f"{field_name} must be between 0.0 and 1.0"
        raise ValueError(msg)


def _validate_optional_mos_score(value: float | None, field_name: str) -> None:
    if value is None:
        return
    _validate_finite(value, field_name)
    if value < MIN_MOS_SCORE or value > MAX_MOS_SCORE:
        msg = f"{field_name} must be between {MIN_MOS_SCORE} and {MAX_MOS_SCORE}"
        raise ValueError(msg)


def _validate_optional_non_negative(value: float | None, field_name: str) -> None:
    if value is None:
        return
    _validate_finite(value, field_name)
    if value < 0.0:
        msg = f"{field_name} must be greater than or equal to 0.0"
        raise ValueError(msg)


def _validate_optional_range(
    lower: float | None,
    upper: float | None,
    lower_name: str,
    upper_name: str,
) -> None:
    if lower is not None and upper is not None and lower > upper:
        msg = f"{lower_name} must be less than or equal to {upper_name}"
        raise ValueError(msg)


def _validate_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        msg = f"{field_name} must be finite"
        raise ValueError(msg)


def _validate_non_blank(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must not be blank"
        raise ValueError(msg)
