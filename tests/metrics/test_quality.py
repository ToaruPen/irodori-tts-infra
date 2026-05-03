from __future__ import annotations

import pytest

from irodori_tts_infra.metrics import (
    QualityGateInput,
    QualityGateStatus,
    QualityScores,
    QualityThresholds,
    cosine_similarity,
    evaluate_quality_gate,
    relative_margin,
)

pytestmark = pytest.mark.unit

PRIMARY_IDENTITY_OBSERVED = 0.79
PRIMARY_IDENTITY_THRESHOLD = 0.80
NEAREST_IDENTITY = 0.72


def test_quality_gate_passes_when_identity_and_margin_meet_thresholds() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(
                primary_identity=0.86,
                secondary_identity=0.83,
                other_identities={"ミカ": 0.42, "ユイ": 0.50},
            ),
            thresholds=QualityThresholds(
                primary_identity_min=0.80,
                secondary_identity_min=0.75,
                relative_margin_min=0.20,
            ),
        ),
    )

    assert result.status is QualityGateStatus.PASS
    assert result.issues == ()


def test_quality_gate_hard_fails_when_primary_identity_is_below_threshold() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(primary_identity=PRIMARY_IDENTITY_OBSERVED),
            thresholds=QualityThresholds(primary_identity_min=PRIMARY_IDENTITY_THRESHOLD),
        ),
    )

    assert result.status is QualityGateStatus.FAIL
    assert [issue.code for issue in result.issues] == ["primary_identity_below_threshold"]
    assert result.issues[0].observed == pytest.approx(PRIMARY_IDENTITY_OBSERVED)
    assert result.issues[0].threshold == pytest.approx(PRIMARY_IDENTITY_THRESHOLD)


def test_quality_gate_fails_when_target_is_too_close_to_other_character() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(
                primary_identity=0.84,
                other_identities={"ミカ": 0.76, "ユイ": 0.60},
            ),
            thresholds=QualityThresholds(
                primary_identity_min=0.80,
                relative_margin_min=0.10,
            ),
        ),
    )

    assert result.status is QualityGateStatus.FAIL
    assert [issue.code for issue in result.issues] == ["relative_margin_below_threshold"]
    assert result.issues[0].observed == pytest.approx(0.08)
    assert result.issues[0].nearest_character == "ミカ"


def test_quality_gate_fails_when_configured_score_is_missing() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(),
            thresholds=QualityThresholds(primary_identity_min=0.80),
        ),
    )

    assert result.status is QualityGateStatus.FAIL
    assert [issue.code for issue in result.issues] == ["primary_identity_missing"]
    assert result.issues[0].observed is None
    assert result.issues[0].threshold == pytest.approx(0.80)


def test_quality_gate_fails_when_relative_margin_inputs_are_missing() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(primary_identity=0.84),
            thresholds=QualityThresholds(relative_margin_min=0.10),
        ),
    )

    assert result.status is QualityGateStatus.FAIL
    assert [issue.code for issue in result.issues] == ["relative_margin_missing"]
    assert result.issues[0].observed is None
    assert result.issues[0].threshold == pytest.approx(0.10)


def test_quality_gate_warns_when_non_speech_relative_margin_inputs_are_missing() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(other_identities={"ミカ": 0.60}),
            thresholds=QualityThresholds(relative_margin_min=0.10),
            non_speech=True,
        ),
    )

    assert result.status is QualityGateStatus.WARN
    assert [issue.code for issue in result.issues] == ["relative_margin_missing"]
    assert result.issues[0].observed is None
    assert result.issues[0].threshold == pytest.approx(0.10)
    assert result.issues[0].warning is True


def test_quality_gate_warns_instead_of_hard_failing_identity_for_non_speech() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(primary_identity=0.55, secondary_identity=0.58),
            thresholds=QualityThresholds(
                primary_identity_min=0.60,
                secondary_identity_min=0.60,
            ),
            non_speech=True,
        ),
    )

    assert result.status is QualityGateStatus.WARN
    assert [issue.code for issue in result.issues] == [
        "primary_identity_below_threshold",
        "secondary_identity_below_threshold",
    ]
    assert {issue.warning for issue in result.issues} == {True}


def test_quality_gate_warns_for_non_speech_relative_margin() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(
                primary_identity=0.63,
                other_identities={"ミカ": 0.60},
            ),
            thresholds=QualityThresholds(
                primary_identity_min=0.60,
                relative_margin_min=0.10,
            ),
            non_speech=True,
        ),
    )

    assert result.status is QualityGateStatus.WARN
    assert [issue.code for issue in result.issues] == ["relative_margin_below_threshold"]
    assert result.issues[0].warning is True


def test_quality_gate_uses_mos_scale_for_quality_floor() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(utmos=3.4, dnsmos=3.8),
            thresholds=QualityThresholds(utmos_min=3.5, dnsmos_min=3.0),
        ),
    )

    assert result.status is QualityGateStatus.FAIL
    assert [issue.code for issue in result.issues] == ["utmos_below_threshold"]


def test_quality_gate_applies_f0_statistics_thresholds() -> None:
    result = evaluate_quality_gate(
        QualityGateInput(
            scores=QualityScores(
                f0_median_hz=210.0,
                f0_range_hz=90.0,
                f0_std_hz=42.0,
                voiced_ratio=0.62,
            ),
            thresholds=QualityThresholds(
                f0_median_min_hz=180.0,
                f0_median_max_hz=220.0,
                f0_range_min_hz=80.0,
                f0_range_max_hz=120.0,
                f0_std_max_hz=45.0,
                voiced_ratio_min=0.60,
            ),
        ),
    )

    assert result.status is QualityGateStatus.PASS
    assert result.issues == ()


def test_relative_margin_uses_nearest_other_character() -> None:
    margin = relative_margin(
        target_identity=0.91,
        other_identities={"ミカ": 0.60, "ユイ": NEAREST_IDENTITY},
    )

    assert margin.value == pytest.approx(0.19)
    assert margin.nearest_character == "ユイ"
    assert margin.nearest_identity == pytest.approx(NEAREST_IDENTITY)


def test_relative_margin_uses_character_name_as_tie_breaker() -> None:
    margin = relative_margin(
        target_identity=0.91,
        other_identities={"ユイ": NEAREST_IDENTITY, "ミカ": NEAREST_IDENTITY},
    )

    assert margin.nearest_character == "ミカ"
    assert margin.nearest_identity == pytest.approx(NEAREST_IDENTITY)


def test_relative_margin_rejects_empty_other_identity_set() -> None:
    with pytest.raises(ValueError, match="other_identities must not be empty"):
        relative_margin(target_identity=0.91, other_identities={})


def test_cosine_similarity_happy_path() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="embedding lengths must match"):
        cosine_similarity((1.0, 0.0), (1.0,))


def test_cosine_similarity_rejects_empty_embedding() -> None:
    with pytest.raises(ValueError, match="embeddings must not be empty"):
        cosine_similarity((), ())


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="embeddings must not be zero vectors"):
        cosine_similarity((0.0, 0.0), (1.0, 0.0))
