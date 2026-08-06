from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

FREQUENCY_BUCKET_WIDTH_HZ = 1_000.0
HARMONIC_RATIO_BUCKET_WIDTH = 0.50
DURATION_BUCKET_WIDTH_SECONDS = 1.0
RMS_BUCKET_WIDTH_DB = 30.0
NEIGHBOR_SEQUENCE_MAX_GAP = 500
SMALL_CLUSTER_LIMIT = 4
CENSOR_MARKERS = ("◯", "○", "〇")  # noqa: RUF001 - source captions use these glyphs

ClusterBaseKey: TypeAlias = tuple[str, int | None, int, int, int, bool]


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    record_id: str
    dataset_id: str
    source_index: int
    audio_sha256: str
    original_text: str
    reasons: tuple[str, ...]
    candidate_wav_path: Path
    intervals: tuple[dict[str, object], ...]
    dominant_frequency_hz: float | None
    duration_seconds: float
    harmonic_ratio: float
    rms_dbfs: float
    caption_has_censor: bool


@dataclass(frozen=True, slots=True)
class CandidateCluster:
    cluster_id: str
    frequency_bucket_hz: float | None
    harmonic_ratio_bucket: float
    duration_bucket_seconds: float
    rms_bucket_dbfs: float
    neighbor_sequence: str
    candidates: tuple[ReviewCandidate, ...]


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    candidate: ReviewCandidate
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PacketResult:
    candidate_count: int
    cluster_count: int
    selected_count: int
    review_sheet: Path


def load_candidates(paths: Sequence[Path]) -> tuple[ReviewCandidate, ...]:
    rows_by_id: dict[tuple[str, str], tuple[str, dict[str, object], Path]] = {}
    for path in sorted(paths, key=lambda item: str(item.resolve())):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                message = f"candidate line {line_number} must be an object: {path}"
                raise TypeError(message)
            row = {str(key): value for key, value in raw.items()}
            key = (str(row["dataset_id"]), str(row["record_id"]))
            fingerprint = json.dumps(row, ensure_ascii=False, sort_keys=True)
            previous = rows_by_id.get(key)
            if previous is not None:
                if previous[0] != fingerprint:
                    message = f"conflicting review candidate: {key[0]}/{key[1]}"
                    raise ValueError(message)
                continue
            rows_by_id[key] = (fingerprint, row, path)
    return tuple(
        sorted(
            (
                _parse_candidate(row, input_path=input_path)
                for _fingerprint, row, input_path in rows_by_id.values()
            ),
            key=_candidate_sort_key,
        )
    )


def cluster_candidates(
    candidates: Sequence[ReviewCandidate],
) -> tuple[CandidateCluster, ...]:
    grouped: dict[ClusterBaseKey, list[ReviewCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(_base_cluster_key(candidate), []).append(candidate)

    clusters: list[CandidateCluster] = []
    for base_key, base_candidates in sorted(grouped.items(), key=_sortable_base_key):
        runs = _neighbor_runs(sorted(base_candidates, key=_candidate_sort_key))
        for run in runs:
            first_index = run[0].source_index
            last_index = run[-1].source_index
            neighbor_sequence = f"{first_index:08d}-{last_index:08d}"
            cluster_key = (*base_key, neighbor_sequence)
            clusters.append(
                CandidateCluster(
                    cluster_id=_cluster_id(cluster_key),
                    frequency_bucket_hz=(
                        None if base_key[1] is None else base_key[1] * FREQUENCY_BUCKET_WIDTH_HZ
                    ),
                    harmonic_ratio_bucket=(base_key[2] * HARMONIC_RATIO_BUCKET_WIDTH),
                    duration_bucket_seconds=(base_key[3] * DURATION_BUCKET_WIDTH_SECONDS),
                    rms_bucket_dbfs=base_key[4] * RMS_BUCKET_WIDTH_DB,
                    neighbor_sequence=neighbor_sequence,
                    candidates=tuple(run),
                )
            )
    return tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id))


def select_candidates(cluster: CandidateCluster) -> tuple[CandidateSelection, ...]:
    candidates = cluster.candidates
    vectors = {candidate.record_id: _feature_vector(candidate) for candidate in candidates}
    center = tuple(
        statistics.median(vector[index] for vector in vectors.values())
        for index in range(len(next(iter(vectors.values()))))
    )

    def distance(candidate: ReviewCandidate) -> float:
        return sum(
            (value - median) ** 2
            for value, median in zip(vectors[candidate.record_id], center, strict=True)
        )

    def projection(candidate: ReviewCandidate) -> float:
        return sum(
            value - median
            for value, median in zip(vectors[candidate.record_id], center, strict=True)
        )

    by_projection = sorted(
        candidates,
        key=lambda item: (projection(item), _candidate_sort_key(item)),
    )
    roles_by_id: dict[str, set[str]] = {}

    def add_role(candidate: ReviewCandidate, role: str) -> None:
        roles_by_id.setdefault(candidate.record_id, set()).add(role)

    add_role(
        min(candidates, key=lambda item: (distance(item), _candidate_sort_key(item))),
        "representative",
    )
    add_role(by_projection[0], "boundary_low")
    add_role(by_projection[-1], "boundary_high")
    add_role(
        min(candidates, key=lambda item: (-distance(item), _candidate_sort_key(item))),
        "outlier",
    )
    if len(candidates) <= SMALL_CLUSTER_LIMIT:
        for candidate in candidates:
            if candidate.record_id not in roles_by_id:
                add_role(candidate, "small_cluster")

    return tuple(
        CandidateSelection(
            candidate=candidate,
            roles=tuple(sorted(roles_by_id[candidate.record_id], key=_role_order)),
        )
        for candidate in candidates
        if candidate.record_id in roles_by_id
    )


def build_review_packet(
    review_candidate_paths: Sequence[Path],
    *,
    output_dir: Path,
) -> PacketResult:
    candidates = load_candidates(review_candidate_paths)
    clusters = cluster_candidates(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []
    for cluster in clusters:
        for selection in select_candidates(cluster):
            candidate = selection.candidate
            filename = _review_filename(cluster, candidate)
            destination = audio_dir / filename
            shutil.copy2(candidate.candidate_wav_path, destination)
            rows.append(_sheet_row(cluster, selection, review_wav=Path("audio") / filename))
    rows.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            _as_int(row["source_index"], field="source_index"),
            str(row["record_id"]),
        )
    )
    review_sheet = output_dir / "review-sheet.csv"
    _write_review_sheet(review_sheet, rows)
    return PacketResult(
        candidate_count=len(candidates),
        cluster_count=len(clusters),
        selected_count=len(rows),
        review_sheet=review_sheet,
    )


def _parse_candidate(row: dict[str, object], *, input_path: Path) -> ReviewCandidate:
    raw_intervals = row.get("intervals", [])
    if not isinstance(raw_intervals, list) or not all(
        isinstance(interval, dict) for interval in raw_intervals
    ):
        message = f"intervals must be an object list: {row.get('record_id')}"
        raise TypeError(message)
    intervals = tuple(
        {str(key): value for key, value in interval.items()} for interval in raw_intervals
    )
    dominant = _dominant_interval(intervals)
    raw_wav_path = Path(str(row["candidate_wav_path"]))
    wav_path = raw_wav_path if raw_wav_path.is_absolute() else input_path.parent / raw_wav_path
    original_text = str(row.get("original_text", row.get("text", "")))
    reasons = row.get("reasons", [])
    if not isinstance(reasons, list):
        message = f"reasons must be a list: {row.get('record_id')}"
        raise TypeError(message)
    return ReviewCandidate(
        record_id=str(row["record_id"]),
        dataset_id=str(row["dataset_id"]),
        source_index=_as_int(row["source_index"], field="source_index"),
        audio_sha256=str(row["audio_sha256"]),
        original_text=original_text,
        reasons=tuple(str(reason) for reason in reasons),
        candidate_wav_path=wav_path,
        intervals=intervals,
        dominant_frequency_hz=(
            None if dominant is None else _as_float(dominant["frequency_hz"], field="frequency_hz")
        ),
        duration_seconds=0.0 if dominant is None else _interval_duration(dominant),
        harmonic_ratio=_as_float(row.get("harmonic_ratio", 0.0), field="harmonic_ratio"),
        rms_dbfs=_as_float(row.get("rms_dbfs", float("-inf")), field="rms_dbfs"),
        caption_has_censor=any(marker in original_text for marker in CENSOR_MARKERS),
    )


def _dominant_interval(
    intervals: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    if not intervals:
        return None
    return min(
        intervals,
        key=lambda interval: (
            -_interval_duration(interval),
            _as_float(interval["start_seconds"], field="start_seconds"),
            _as_float(interval["frequency_hz"], field="frequency_hz"),
        ),
    )


def _interval_duration(interval: dict[str, object]) -> float:
    return max(
        0.0,
        _as_float(interval["end_seconds"], field="end_seconds")
        - _as_float(interval["start_seconds"], field="start_seconds"),
    )


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"{field} must be an integer"
        raise TypeError(message)
    return value


def _as_float(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        message = f"{field} must be numeric"
        raise TypeError(message)
    return float(value)


def _base_cluster_key(candidate: ReviewCandidate) -> ClusterBaseKey:
    frequency_bucket = (
        None
        if candidate.dominant_frequency_hz is None
        else _bucket(candidate.dominant_frequency_hz, FREQUENCY_BUCKET_WIDTH_HZ)
    )
    return (
        candidate.dataset_id,
        frequency_bucket,
        _bucket(candidate.harmonic_ratio, HARMONIC_RATIO_BUCKET_WIDTH),
        _bucket(candidate.duration_seconds, DURATION_BUCKET_WIDTH_SECONDS),
        _bucket(candidate.rms_dbfs, RMS_BUCKET_WIDTH_DB),
        candidate.caption_has_censor,
    )


def _bucket(value: float, width: float) -> int:
    if not math.isfinite(value):
        return -(10**9) if value < 0 else 10**9
    return math.floor((value + 1e-12) / width)


def _neighbor_runs(
    candidates: Sequence[ReviewCandidate],
) -> tuple[tuple[ReviewCandidate, ...], ...]:
    runs: list[list[ReviewCandidate]] = []
    for candidate in candidates:
        if (
            not runs
            or candidate.source_index - runs[-1][-1].source_index > NEIGHBOR_SEQUENCE_MAX_GAP
        ):
            runs.append([candidate])
        else:
            runs[-1].append(candidate)
    return tuple(tuple(run) for run in runs)


def _feature_vector(candidate: ReviewCandidate) -> tuple[float, ...]:
    frequency = candidate.dominant_frequency_hz or 0.0
    return (
        frequency / FREQUENCY_BUCKET_WIDTH_HZ,
        candidate.harmonic_ratio / HARMONIC_RATIO_BUCKET_WIDTH,
        candidate.duration_seconds / DURATION_BUCKET_WIDTH_SECONDS,
        candidate.rms_dbfs / RMS_BUCKET_WIDTH_DB,
        candidate.source_index / NEIGHBOR_SEQUENCE_MAX_GAP,
    )


def _candidate_sort_key(candidate: ReviewCandidate) -> tuple[str, int, str, str]:
    return (
        candidate.dataset_id,
        candidate.source_index,
        candidate.record_id,
        candidate.audio_sha256,
    )


def _sortable_base_key(
    item: tuple[ClusterBaseKey, list[ReviewCandidate]],
) -> tuple[str, int, int, int, int, bool]:
    key = item[0]
    frequency = -(10**9) if key[1] is None else key[1]
    return (key[0], frequency, key[2], key[3], key[4], key[5])


def _cluster_id(key: tuple[object, ...]) -> str:
    serialized = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    dataset_id = str(key[0]).replace("/", "-").replace("\\", "-")
    return f"{dataset_id}-{digest}"


def _role_order(role: str) -> int:
    order = {
        "representative": 0,
        "boundary_low": 1,
        "boundary_high": 2,
        "outlier": 3,
        "small_cluster": 4,
    }
    return order[role]


def _review_filename(cluster: CandidateCluster, candidate: ReviewCandidate) -> str:
    safe_record_id = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in candidate.record_id
    ).strip("-")
    return f"{cluster.cluster_id}__{safe_record_id}__{candidate.audio_sha256[:12]}.wav"


def _sheet_row(
    cluster: CandidateCluster,
    selection: CandidateSelection,
    *,
    review_wav: Path,
) -> dict[str, object]:
    candidate = selection.candidate
    return {
        "cluster_id": cluster.cluster_id,
        "selection_roles": "|".join(selection.roles),
        "record_id": candidate.record_id,
        "dataset_id": candidate.dataset_id,
        "source_index": candidate.source_index,
        "audio_sha256": candidate.audio_sha256,
        "review_wav": review_wav.as_posix(),
        "dominant_frequency_hz": candidate.dominant_frequency_hz,
        "frequency_bucket_hz": cluster.frequency_bucket_hz,
        "harmonic_ratio": candidate.harmonic_ratio,
        "harmonic_ratio_bucket": cluster.harmonic_ratio_bucket,
        "duration_seconds": candidate.duration_seconds,
        "duration_bucket_seconds": cluster.duration_bucket_seconds,
        "rms_dbfs": candidate.rms_dbfs,
        "rms_bucket_dbfs": cluster.rms_bucket_dbfs,
        "caption_has_censor": candidate.caption_has_censor,
        "neighbor_sequence": cluster.neighbor_sequence,
        "intervals_json": json.dumps(
            candidate.intervals,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "reasons": "|".join(candidate.reasons),
        "original_text": candidate.original_text,
        "label": "",
        "label_options": "TONE|VOICE|UNSURE",
    }


def _write_review_sheet(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = list(_sheet_row_fieldnames())
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sheet_row_fieldnames() -> tuple[str, ...]:
    return (
        "cluster_id",
        "selection_roles",
        "record_id",
        "dataset_id",
        "source_index",
        "audio_sha256",
        "review_wav",
        "dominant_frequency_hz",
        "frequency_bucket_hz",
        "harmonic_ratio",
        "harmonic_ratio_bucket",
        "duration_seconds",
        "duration_bucket_seconds",
        "rms_dbfs",
        "rms_bucket_dbfs",
        "caption_has_censor",
        "neighbor_sequence",
        "intervals_json",
        "reasons",
        "original_text",
        "label",
        "label_options",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-candidates",
        action="append",
        dest="review_candidates",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_review_packet(args.review_candidates, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
