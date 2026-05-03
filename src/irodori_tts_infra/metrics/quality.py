from __future__ import annotations

from math import sqrt
from operator import itemgetter

from irodori_tts_infra.metrics.models import (
    QualityGateInput,
    QualityGateIssue,
    QualityGateResult,
    QualityGateStatus,
    RelativeMargin,
)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        msg = "embedding lengths must match"
        raise ValueError(msg)
    if not left:
        msg = "embeddings must not be empty"
        raise ValueError(msg)

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        msg = "embeddings must not be zero vectors"
        raise ValueError(msg)
    return dot / (left_norm * right_norm)


def relative_margin(
    *,
    target_identity: float,
    other_identities: dict[str, float],
) -> RelativeMargin:
    if not other_identities:
        msg = "other_identities must not be empty"
        raise ValueError(msg)

    nearest_character, nearest_identity = max(
        other_identities.items(),
        key=itemgetter(1),
    )
    return RelativeMargin(
        value=target_identity - nearest_identity,
        nearest_character=nearest_character,
        nearest_identity=nearest_identity,
    )


def evaluate_quality_gate(request: QualityGateInput) -> QualityGateResult:
    issues: list[QualityGateIssue] = []
    issues.extend(_identity_issues(request))
    issues.extend(_quality_floor_issues(request))
    issues.extend(_f0_issues(request))
    issues.extend(_style_issues(request))

    if any(not issue.warning for issue in issues):
        status = QualityGateStatus.FAIL
    elif issues:
        status = QualityGateStatus.WARN
    else:
        status = QualityGateStatus.PASS
    return QualityGateResult(status=status, issues=tuple(issues))


def _identity_issues(request: QualityGateInput) -> tuple[QualityGateIssue, ...]:
    issues: list[QualityGateIssue] = []
    _append_min_issue(
        issues,
        metric="primary_identity",
        code="primary_identity_below_threshold",
        observed=request.scores.primary_identity,
        threshold=request.thresholds.primary_identity_min,
        warning=request.non_speech,
    )
    _append_min_issue(
        issues,
        metric="secondary_identity",
        code="secondary_identity_below_threshold",
        observed=request.scores.secondary_identity,
        threshold=request.thresholds.secondary_identity_min,
        warning=request.non_speech,
    )
    if (
        request.scores.primary_identity is not None
        and request.thresholds.relative_margin_min is not None
        and request.scores.other_identities
    ):
        margin = relative_margin(
            target_identity=request.scores.primary_identity,
            other_identities=dict(request.scores.other_identities),
        )
        if margin.value < request.thresholds.relative_margin_min:
            issues.append(
                QualityGateIssue(
                    code="relative_margin_below_threshold",
                    metric="relative_margin",
                    observed=margin.value,
                    threshold=request.thresholds.relative_margin_min,
                    warning=request.non_speech,
                    nearest_character=margin.nearest_character,
                ),
            )
    return tuple(issues)


def _quality_floor_issues(request: QualityGateInput) -> tuple[QualityGateIssue, ...]:
    issues: list[QualityGateIssue] = []
    _append_min_issue(
        issues,
        metric="utmos",
        code="utmos_below_threshold",
        observed=request.scores.utmos,
        threshold=request.thresholds.utmos_min,
    )
    _append_min_issue(
        issues,
        metric="dnsmos",
        code="dnsmos_below_threshold",
        observed=request.scores.dnsmos,
        threshold=request.thresholds.dnsmos_min,
    )
    return tuple(issues)


def _f0_issues(request: QualityGateInput) -> tuple[QualityGateIssue, ...]:
    issues: list[QualityGateIssue] = []
    _append_min_issue(
        issues,
        metric="f0_median_hz",
        code="f0_median_hz_below_threshold",
        observed=request.scores.f0_median_hz,
        threshold=request.thresholds.f0_median_min_hz,
    )
    _append_max_issue(
        issues,
        metric="f0_median_hz",
        code="f0_median_hz_above_threshold",
        observed=request.scores.f0_median_hz,
        threshold=request.thresholds.f0_median_max_hz,
    )
    _append_min_issue(
        issues,
        metric="f0_range_hz",
        code="f0_range_hz_below_threshold",
        observed=request.scores.f0_range_hz,
        threshold=request.thresholds.f0_range_min_hz,
    )
    _append_max_issue(
        issues,
        metric="f0_range_hz",
        code="f0_range_hz_above_threshold",
        observed=request.scores.f0_range_hz,
        threshold=request.thresholds.f0_range_max_hz,
    )
    _append_max_issue(
        issues,
        metric="f0_std_hz",
        code="f0_std_hz_above_threshold",
        observed=request.scores.f0_std_hz,
        threshold=request.thresholds.f0_std_max_hz,
    )
    _append_min_issue(
        issues,
        metric="voiced_ratio",
        code="voiced_ratio_below_threshold",
        observed=request.scores.voiced_ratio,
        threshold=request.thresholds.voiced_ratio_min,
    )
    return tuple(issues)


def _style_issues(request: QualityGateInput) -> tuple[QualityGateIssue, ...]:
    issues: list[QualityGateIssue] = []
    _append_min_issue(
        issues,
        metric="speaking_rate",
        code="speaking_rate_below_threshold",
        observed=request.scores.speaking_rate,
        threshold=request.thresholds.speaking_rate_min,
    )
    _append_max_issue(
        issues,
        metric="speaking_rate",
        code="speaking_rate_above_threshold",
        observed=request.scores.speaking_rate,
        threshold=request.thresholds.speaking_rate_max,
    )
    _append_max_issue(
        issues,
        metric="pause_ratio",
        code="pause_ratio_above_threshold",
        observed=request.scores.pause_ratio,
        threshold=request.thresholds.pause_ratio_max,
    )
    _append_min_issue(
        issues,
        metric="rms_energy",
        code="rms_energy_below_threshold",
        observed=request.scores.rms_energy,
        threshold=request.thresholds.rms_energy_min,
    )
    return tuple(issues)


def _append_min_issue(
    issues: list[QualityGateIssue],
    *,
    metric: str,
    code: str,
    observed: float | None,
    threshold: float | None,
    warning: bool = False,
) -> None:
    if threshold is None:
        return
    if observed is None:
        _append_missing_issue(issues, metric=metric, threshold=threshold, warning=warning)
        return
    if observed >= threshold:
        return
    issues.append(
        QualityGateIssue(
            code=code,
            metric=metric,
            observed=observed,
            threshold=threshold,
            warning=warning,
        ),
    )


def _append_max_issue(
    issues: list[QualityGateIssue],
    *,
    metric: str,
    code: str,
    observed: float | None,
    threshold: float | None,
) -> None:
    if threshold is None:
        return
    if observed is None:
        _append_missing_issue(issues, metric=metric, threshold=threshold)
        return
    if observed <= threshold:
        return
    issues.append(
        QualityGateIssue(
            code=code,
            metric=metric,
            observed=observed,
            threshold=threshold,
        ),
    )


def _append_missing_issue(
    issues: list[QualityGateIssue],
    *,
    metric: str,
    threshold: float,
    warning: bool = False,
) -> None:
    issues.append(
        QualityGateIssue(
            code=f"{metric}_missing",
            metric=metric,
            observed=None,
            threshold=threshold,
            warning=warning,
        ),
    )
