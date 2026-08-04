from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from types import ModuleType


class _ReviewCandidate(Protocol):
    audio_sha256: str
    caption_has_censor: bool
    dataset_id: str


class _CandidateCluster(Protocol):
    cluster_id: str
    candidates: Sequence[_ReviewCandidate]


pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/apply_speaker_review_labels.py")
PACKET_SCRIPT_PATH = Path("scripts/build_speaker_review_packet.py")


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_homogeneous_tone_cluster_propagates_and_creates_dataset_signature(
    tmp_path: Path,
) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_tone")
    candidate_path = tmp_path / "review-candidates.jsonl"
    candidates = [
        _candidate("first", source_index=10, frequency_hz=703.0),
        _candidate("second", source_index=11, frequency_hz=704.0),
        _candidate("third", source_index=12, frequency_hz=705.0),
    ]
    _write_jsonl(candidate_path, candidates)
    cluster_id = _single_cluster_id(candidate_path)
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(
        sheet_path,
        [
            _sheet_row(cluster_id, candidates[0], "TONE"),
            _sheet_row(cluster_id, candidates[1], "TONE"),
        ],
    )
    labels_path = tmp_path / "labels.jsonl"
    signatures_path = tmp_path / "tone-signatures.json"

    result = module.apply_review_labels(
        review_sheet=sheet_path,
        review_candidate_paths=[candidate_path],
        labels_path=labels_path,
        tone_signatures_path=signatures_path,
        reviewer="owner",
        cluster_version="cluster-v1",
        rule_version="rule-v2",
    )

    labels = _read_jsonl(labels_path)
    assert result.explicit_count == len(candidates) - 1
    assert result.propagated_count == 1
    assert {row["audio_sha256"] for row in labels} == {
        "sha-first",
        "sha-second",
        "sha-third",
    }
    assert {row["label"] for row in labels} == {"TONE"}
    assert {row["provenance"] for row in labels if row["audio_sha256"] == "sha-third"} == {
        "cluster_propagated"
    }
    assert all(row["cluster_version"] == "cluster-v1" for row in labels)
    assert all(row["rule_version"] == "rule-v2" for row in labels)
    assert all(row["reviewer"] == "owner" for row in labels)

    signatures = json.loads(signatures_path.read_text(encoding="utf-8"))
    assert len(signatures) == 1
    assert signatures[0]["dataset_id"] == "oop55"
    assert signatures[0]["center_frequency_hz"] == pytest.approx(703.5)
    assert signatures[0]["cluster_id"] == cluster_id
    assert signatures[0]["cluster_version"] == "cluster-v1"
    assert signatures[0]["rule_version"] == "rule-v2"


def test_homogeneous_tone_cluster_without_frequencies_propagates_without_signature(
    tmp_path: Path,
) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_tone_no_frequency")
    candidate_path = tmp_path / "review-candidates.jsonl"
    candidates = [
        _candidate("first", source_index=10, frequency_hz=None),
        _candidate("second", source_index=11, frequency_hz=None),
        _candidate("third", source_index=12, frequency_hz=None),
    ]
    _write_jsonl(candidate_path, candidates)
    cluster_id = _single_cluster_id(candidate_path)
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(
        sheet_path,
        [
            _sheet_row(cluster_id, candidates[0], "TONE"),
            _sheet_row(cluster_id, candidates[1], "TONE"),
        ],
    )
    labels_path = tmp_path / "labels.jsonl"
    signatures_path = tmp_path / "tone-signatures.json"

    result = module.apply_review_labels(
        review_sheet=sheet_path,
        review_candidate_paths=[candidate_path],
        labels_path=labels_path,
        tone_signatures_path=signatures_path,
        reviewer="owner",
        cluster_version="cluster-v1",
        rule_version="rule-v1",
    )

    labels = _read_jsonl(labels_path)
    assert result.explicit_count == len(candidates) - 1
    assert result.propagated_count == 1
    assert result.label_count == len(candidates)
    assert result.signature_count == 0
    assert {row["audio_sha256"] for row in labels} == {
        "sha-first",
        "sha-second",
        "sha-third",
    }
    assert {row["label"] for row in labels} == {"TONE"}
    assert {row["provenance"] for row in labels} == {"explicit", "cluster_propagated"}
    assert json.loads(signatures_path.read_text(encoding="utf-8")) == []


def test_censored_homogeneous_tone_cluster_with_frequencies_does_not_create_signature(
    tmp_path: Path,
) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_censored_tone")
    candidate_path = tmp_path / "review-candidates.jsonl"
    censored_candidates = [
        _candidate("first", source_index=10, frequency_hz=703.0, caption_has_censor=True),
        _candidate("second", source_index=11, frequency_hz=704.0, caption_has_censor=True),
        _candidate("third", source_index=12, frequency_hz=705.0, caption_has_censor=True),
    ]
    candidates = [
        *censored_candidates,
        _candidate("uncensored", source_index=13, frequency_hz=703.0),
    ]
    _write_jsonl(candidate_path, candidates)
    clusters = _clusters_by_censor(candidate_path)
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(
        sheet_path,
        [
            _sheet_row(clusters[True], censored_candidates[0], "TONE"),
            _sheet_row(clusters[True], censored_candidates[1], "TONE"),
            _sheet_row(clusters[False], candidates[-1], ""),
        ],
    )
    labels_path = tmp_path / "labels.jsonl"
    signatures_path = tmp_path / "tone-signatures.json"

    result = module.apply_review_labels(
        review_sheet=sheet_path,
        review_candidate_paths=[candidate_path],
        labels_path=labels_path,
        tone_signatures_path=signatures_path,
        reviewer="owner",
        cluster_version="cluster-v1",
        rule_version="rule-v1",
    )

    assert result.label_count == len(censored_candidates)
    assert result.signature_count == 0
    assert {row["audio_sha256"] for row in _read_jsonl(labels_path)} == {
        "sha-first",
        "sha-second",
        "sha-third",
    }
    assert json.loads(signatures_path.read_text(encoding="utf-8")) == []


def test_homogeneous_voice_propagation_stays_inside_its_cluster_and_dataset(
    tmp_path: Path,
) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_voice")
    candidate_path = tmp_path / "review-candidates.jsonl"
    candidates = [
        _candidate("a-first", dataset_id="dataset-a", source_index=10),
        _candidate("a-second", dataset_id="dataset-a", source_index=11),
        _candidate("b-same-frequency", dataset_id="dataset-b", source_index=10),
    ]
    _write_jsonl(candidate_path, candidates)
    clusters = _clusters_by_dataset(candidate_path)
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(
        sheet_path,
        [
            _sheet_row(clusters["dataset-a"], candidates[0], "VOICE"),
            _sheet_row(clusters["dataset-b"], candidates[2], ""),
        ],
    )
    labels_path = tmp_path / "labels.jsonl"
    signatures_path = tmp_path / "tone-signatures.json"

    module.apply_review_labels(
        review_sheet=sheet_path,
        review_candidate_paths=[candidate_path],
        labels_path=labels_path,
        tone_signatures_path=signatures_path,
        reviewer="owner",
        explicit_note="packet approved after random sampling",
        cluster_version="cluster-v1",
        rule_version="rule-v1",
    )

    labels = _read_jsonl(labels_path)
    assert {row["audio_sha256"] for row in labels} == {"sha-a-first", "sha-a-second"}
    assert {row["label"] for row in labels} == {"VOICE"}
    assert next(row for row in labels if row["audio_sha256"] == "sha-a-first")["note"] == (
        "packet approved after random sampling"
    )
    assert json.loads(signatures_path.read_text(encoding="utf-8")) == []


def test_mixed_unsure_or_empty_cluster_keeps_only_explicit_labels(
    tmp_path: Path,
) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_mixed")
    candidate_path = tmp_path / "review-candidates.jsonl"
    candidates = [
        _candidate("tone", source_index=20),
        _candidate("voice", source_index=21),
        _candidate("unsure", source_index=22),
        _candidate("empty", source_index=23),
        _candidate("unselected", source_index=24),
    ]
    _write_jsonl(candidate_path, candidates)
    cluster_id = _single_cluster_id(candidate_path)
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(
        sheet_path,
        [
            _sheet_row(cluster_id, candidates[0], "TONE"),
            _sheet_row(cluster_id, candidates[1], "VOICE"),
            _sheet_row(cluster_id, candidates[2], "UNSURE"),
            _sheet_row(cluster_id, candidates[3], ""),
        ],
    )
    labels_path = tmp_path / "labels.jsonl"
    signatures_path = tmp_path / "tone-signatures.json"

    result = module.apply_review_labels(
        review_sheet=sheet_path,
        review_candidate_paths=[candidate_path],
        labels_path=labels_path,
        tone_signatures_path=signatures_path,
        reviewer="owner",
        cluster_version="cluster-v1",
        rule_version="rule-v1",
    )

    labels = _read_jsonl(labels_path)
    assert result.explicit_count == len(labels)
    assert result.propagated_count == 0
    assert [(row["audio_sha256"], row["label"]) for row in labels] == [
        ("sha-tone", "TONE"),
        ("sha-unsure", "UNSURE"),
        ("sha-voice", "VOICE"),
    ]
    assert {row["provenance"] for row in labels} == {"explicit"}
    assert json.loads(signatures_path.read_text(encoding="utf-8")) == []


def test_existing_conflicting_label_fails_without_changing_outputs(tmp_path: Path) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_conflict")
    candidate_path = tmp_path / "review-candidates.jsonl"
    candidate = _candidate("same")
    _write_jsonl(candidate_path, [candidate])
    cluster_id = _single_cluster_id(candidate_path)
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(sheet_path, [_sheet_row(cluster_id, candidate, "TONE")])
    labels_path = tmp_path / "labels.jsonl"
    existing: dict[str, object] = {
        "audio_sha256": "sha-same",
        "label": "VOICE",
        "reviewer": "previous",
        "note": "listened",
        "cluster_id": cluster_id,
        "cluster_version": "cluster-v0",
        "rule_version": "rule-v0",
        "provenance": "explicit",
    }
    _write_jsonl(labels_path, [existing])
    signatures_path = tmp_path / "tone-signatures.json"
    signatures_path.write_text("[]\n", encoding="utf-8")
    labels_before = labels_path.read_bytes()
    signatures_before = signatures_path.read_bytes()

    with pytest.raises(ValueError, match=r"conflicting label.*sha-same"):
        module.apply_review_labels(
            review_sheet=sheet_path,
            review_candidate_paths=[candidate_path],
            labels_path=labels_path,
            tone_signatures_path=signatures_path,
            reviewer="owner",
            cluster_version="cluster-v1",
            rule_version="rule-v1",
        )

    assert labels_path.read_bytes() == labels_before
    assert signatures_path.read_bytes() == signatures_before


def test_reapplying_same_round_is_deterministic_and_deduplicates_signatures(
    tmp_path: Path,
) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_repeat")
    candidate_path = tmp_path / "review-candidates.jsonl"
    candidates = [_candidate("first", source_index=1), _candidate("second", source_index=2)]
    _write_jsonl(candidate_path, candidates)
    cluster_id = _single_cluster_id(candidate_path)
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(
        sheet_path,
        [_sheet_row(cluster_id, candidate, "TONE") for candidate in candidates],
    )
    labels_path = tmp_path / "labels.jsonl"
    signatures_path = tmp_path / "tone-signatures.json"
    arguments = {
        "review_sheet": sheet_path,
        "review_candidate_paths": [candidate_path],
        "labels_path": labels_path,
        "tone_signatures_path": signatures_path,
        "reviewer": "owner",
        "cluster_version": "cluster-v1",
        "rule_version": "rule-v1",
    }

    module.apply_review_labels(**arguments)
    first_labels = labels_path.read_bytes()
    first_signatures = signatures_path.read_bytes()
    module.apply_review_labels(**arguments)

    assert labels_path.read_bytes() == first_labels
    assert signatures_path.read_bytes() == first_signatures
    assert len(json.loads(signatures_path.read_text(encoding="utf-8"))) == 1


def test_main_writes_requested_outputs(tmp_path: Path) -> None:
    module = _load_script(SCRIPT_PATH, "apply_speaker_review_labels_main")
    candidate_path = tmp_path / "review-candidates.jsonl"
    candidate = _candidate("voice")
    _write_jsonl(candidate_path, [candidate])
    sheet_path = tmp_path / "review-sheet.csv"
    _write_review_sheet(
        sheet_path,
        [_sheet_row(_single_cluster_id(candidate_path), candidate, "VOICE")],
    )
    labels_path = tmp_path / "labels.jsonl"
    signatures_path = tmp_path / "tone-signatures.json"

    exit_code = module.main(
        [
            "--review-sheet",
            str(sheet_path),
            "--review-candidates",
            str(candidate_path),
            "--labels-jsonl",
            str(labels_path),
            "--tone-signatures-json",
            str(signatures_path),
            "--reviewer",
            "owner",
            "--cluster-version",
            "cluster-v1",
            "--rule-version",
            "rule-v1",
        ]
    )

    assert exit_code == 0
    assert _read_jsonl(labels_path)[0]["label"] == "VOICE"
    assert json.loads(signatures_path.read_text(encoding="utf-8")) == []


def _candidate(
    record_id: str,
    *,
    dataset_id: str = "oop55",
    source_index: int = 10,
    frequency_hz: float | None = 703.125,
    caption_has_censor: bool = False,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "dataset_id": dataset_id,
        "source_index": source_index,
        "audio_sha256": f"sha-{record_id}",
        "original_text": "本文◯" if caption_has_censor else "本文",
        "reasons": ["unmatched_pure_tone"],
        "candidate_wav_path": "candidate.wav",
        "harmonic_ratio": 0.01,
        "rms_dbfs": -40.0,
        "intervals": (
            []
            if frequency_hz is None
            else [
                {
                    "start_seconds": 0.1,
                    "end_seconds": 0.4,
                    "frequency_hz": frequency_hz,
                    "frequency_std_hz": 0.5,
                    "peak_energy_ratio": 0.98,
                    "normalized_entropy": 0.1,
                }
            ]
        ),
    }


def _single_cluster_id(candidate_path: Path) -> str:
    clusters = _clusters(candidate_path)
    assert len(clusters) == 1
    return clusters[0].cluster_id


def _clusters_by_dataset(candidate_path: Path) -> dict[str, str]:
    return {
        cluster.candidates[0].dataset_id: cluster.cluster_id
        for cluster in _clusters(candidate_path)
    }


def _clusters_by_censor(candidate_path: Path) -> dict[bool, str]:
    return {
        cluster.candidates[0].caption_has_censor: cluster.cluster_id
        for cluster in _clusters(candidate_path)
    }


def _clusters(candidate_path: Path) -> Sequence[_CandidateCluster]:
    module = _load_script(PACKET_SCRIPT_PATH, "build_speaker_review_packet_test_helper")
    return cast(
        "Sequence[_CandidateCluster]",
        module.cluster_candidates(module.load_candidates([candidate_path])),
    )


def _sheet_row(cluster_id: str, candidate: dict[str, object], label: str) -> dict[str, str]:
    interval = candidate["intervals"]
    assert isinstance(interval, list)
    dominant_frequency = ""
    if interval:
        dominant = interval[0]
        assert isinstance(dominant, dict)
        dominant_frequency = str(dominant["frequency_hz"])
    return {
        "cluster_id": cluster_id,
        "audio_sha256": str(candidate["audio_sha256"]),
        "dataset_id": str(candidate["dataset_id"]),
        "dominant_frequency_hz": dominant_frequency,
        "label": label,
    }


def _write_review_sheet(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "cluster_id",
                "audio_sha256",
                "dataset_id",
                "dominant_frequency_hz",
                "label",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
