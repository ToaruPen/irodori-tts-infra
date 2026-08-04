from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/build_speaker_review_packet.py")
EXPECTED_PACKET_ROWS = 2
MAX_LARGE_CLUSTER_SELECTIONS = 4
MAX_COARSE_CLUSTERS = 4
MAX_COARSE_SELECTIONS = 16


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_speaker_review_packet", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_candidates_uses_every_requested_feature_and_is_deterministic(
    tmp_path: Path,
) -> None:
    module = _load_script()
    input_path = tmp_path / "review-candidates.jsonl"
    rows = [
        _candidate("base", dataset_id="a", source_index=10),
        _candidate("dataset", dataset_id="b", source_index=10),
        _candidate("frequency", dataset_id="a", source_index=11, frequency_hz=1_500.0),
        _candidate("harmonic", dataset_id="a", source_index=12, harmonic_ratio=0.75),
        _candidate("duration", dataset_id="a", source_index=13, duration_seconds=1.40),
        _candidate("rms", dataset_id="a", source_index=14, rms_dbfs=-91.0),
        _candidate("marker", dataset_id="a", source_index=15, text="おち◯ちん"),
        _candidate("sequence", dataset_id="a", source_index=1_000),
    ]
    _write_jsonl(input_path, reversed(rows))

    first = module.cluster_candidates(module.load_candidates([input_path]))
    second = module.cluster_candidates(module.load_candidates([input_path]))

    assert first == second
    assert len(first) == len(rows)
    assert {cluster.candidates[0].record_id for cluster in first} == {
        row["record_id"] for row in rows
    }


def test_select_candidates_keeps_every_member_of_small_clusters(tmp_path: Path) -> None:
    module = _load_script()
    input_path = tmp_path / "review-candidates.jsonl"
    rows = [
        _candidate(f"candidate-{index}", source_index=20 + index, frequency_hz=710 + index)
        for index in range(4)
    ]
    _write_jsonl(input_path, rows)
    (cluster,) = module.cluster_candidates(module.load_candidates([input_path]))

    selected = module.select_candidates(cluster)

    assert {selection.candidate.record_id for selection in selected} == {
        row["record_id"] for row in rows
    }
    assert all(selection.roles for selection in selected)


def test_select_candidates_chooses_roles_without_duplicate_rows(tmp_path: Path) -> None:
    module = _load_script()
    input_path = tmp_path / "review-candidates.jsonl"
    rows = [
        _candidate(
            f"candidate-{index}",
            source_index=30 + index,
            frequency_hz=701.0 + index * 3,
            harmonic_ratio=0.011 + index * 0.004,
            duration_seconds=0.31 + index * 0.008,
            rms_dbfs=-49.0 + index,
        )
        for index in range(7)
    ]
    _write_jsonl(input_path, rows)
    (cluster,) = module.cluster_candidates(module.load_candidates([input_path]))

    selected = module.select_candidates(cluster)
    selected_ids = [selection.candidate.record_id for selection in selected]
    all_roles = {role for selection in selected for role in selection.roles}

    assert len(selected_ids) == len(set(selected_ids))
    assert len(selected) <= MAX_LARGE_CLUSTER_SELECTIONS
    assert all_roles == {"representative", "boundary_low", "boundary_high", "outlier"}


def test_initial_packet_coarsely_clusters_realistic_feature_variation(tmp_path: Path) -> None:
    module = _load_script()
    input_path = tmp_path / "review-candidates.jsonl"
    rows = [
        _candidate(
            f"candidate-{index}",
            source_index=100 + index,
            frequency_hz=500.0 + (index % 10) * 90.0,
            harmonic_ratio=(index % 8) * 0.05,
            duration_seconds=0.1 + (index % 9) * 0.09,
            rms_dbfs=-59.0 + (index % 8) * 5.0,
        )
        for index in range(100)
    ]
    _write_jsonl(input_path, rows)

    clusters = module.cluster_candidates(module.load_candidates([input_path]))
    selected = sum(len(module.select_candidates(cluster)) for cluster in clusters)

    assert len(clusters) <= MAX_COARSE_CLUSTERS
    assert selected <= MAX_COARSE_SELECTIONS


def test_build_review_packet_copies_selected_wavs_and_writes_label_sheet(
    tmp_path: Path,
) -> None:
    module = _load_script()
    first_wav = tmp_path / "first.wav"
    second_wav = tmp_path / "second.wav"
    first_wav.write_bytes(b"first audio")
    second_wav.write_bytes(b"second audio")
    input_path = tmp_path / "review-candidates.jsonl"
    first = _candidate("first", source_index=1, candidate_wav_path=first_wav.name)
    second = _candidate(
        "second",
        source_index=2,
        frequency_hz=712.0,
        candidate_wav_path=second_wav.name,
    )
    _write_jsonl(input_path, [second, first])
    output_dir = tmp_path / "review" / "round-1"

    result = module.build_review_packet([input_path], output_dir=output_dir)

    assert result.candidate_count == EXPECTED_PACKET_ROWS
    assert result.selected_count == EXPECTED_PACKET_ROWS
    with (output_dir / "review-sheet.csv").open(encoding="utf-8-sig", newline="") as sheet:
        rows = list(csv.DictReader(sheet))
    assert [row["record_id"] for row in rows] == ["first", "second"]
    assert all(not row["label"] for row in rows)
    assert all(row["label_options"] == "TONE|VOICE|UNSURE" for row in rows)
    assert json.loads(rows[0]["intervals_json"]) == first["intervals"]
    assert {row["caption_has_censor"] for row in rows} == {"False"}
    copied = [output_dir / row["review_wav"] for row in rows]
    assert [path.read_bytes() for path in copied] == [b"first audio", b"second audio"]
    assert first_wav.read_bytes() == b"first audio"
    assert second_wav.read_bytes() == b"second audio"


def test_main_accepts_multiple_inputs_and_deduplicates_repeated_records(
    tmp_path: Path,
) -> None:
    module = _load_script()
    wav_path = tmp_path / "candidate.wav"
    wav_path.write_bytes(b"candidate")
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    row = _candidate("same", candidate_wav_path=wav_path.name)
    _write_jsonl(first_input, [row])
    _write_jsonl(second_input, [row])
    output_dir = tmp_path / "packet"

    exit_code = module.main(
        [
            "--review-candidates",
            str(second_input),
            "--review-candidates",
            str(first_input),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    with (output_dir / "review-sheet.csv").open(encoding="utf-8-sig", newline="") as sheet:
        assert len(list(csv.DictReader(sheet))) == 1


def _candidate(  # noqa: PLR0913 - explicit feature overrides make cluster tests readable
    record_id: str,
    *,
    dataset_id: str = "oop55",
    source_index: int = 10,
    frequency_hz: float = 703.125,
    harmonic_ratio: float = 0.02,
    duration_seconds: float = 0.30,
    rms_dbfs: float = -45.0,
    text: str = "本文",
    candidate_wav_path: str = "candidate.wav",
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "dataset_id": dataset_id,
        "source_index": source_index,
        "audio_sha256": f"sha-{record_id}",
        "original_text": text,
        "reasons": ["narrowband_candidate"],
        "candidate_wav_path": candidate_wav_path,
        "harmonic_ratio": harmonic_ratio,
        "rms_dbfs": rms_dbfs,
        "intervals": [
            {
                "start_seconds": 0.1,
                "end_seconds": 0.1 + duration_seconds,
                "frequency_hz": frequency_hz,
                "frequency_std_hz": 1.0,
                "peak_energy_ratio": 0.96,
                "normalized_entropy": 0.18,
            }
        ],
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
