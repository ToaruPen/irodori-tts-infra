from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_PACKET_PATH = Path(__file__).with_name("build_speaker_review_packet.py")
_PACKET_SPEC = importlib.util.spec_from_file_location(
    "_apply_speaker_review_labels_packet",
    _PACKET_PATH,
)
if _PACKET_SPEC is None or _PACKET_SPEC.loader is None:
    message = f"cannot load review packet module: {_PACKET_PATH}"
    raise RuntimeError(message)
_PACKET = importlib.util.module_from_spec(_PACKET_SPEC)
sys.modules[_PACKET_SPEC.name] = _PACKET
_PACKET_SPEC.loader.exec_module(_PACKET)

ReviewLabelValue = Literal["TONE", "VOICE", "UNSURE"]
Provenance = Literal["explicit", "cluster_propagated"]
ALLOWED_LABELS = frozenset({"TONE", "VOICE", "UNSURE"})
DEFAULT_TOLERANCE_FFT_BINS = 1.5
FREQUENCY_DIGITS = 9


@dataclass(frozen=True, slots=True)
class ReviewSheetRow:
    cluster_id: str
    audio_sha256: str
    dataset_id: str
    dominant_frequency_hz: float | None
    label: ReviewLabelValue | None


@dataclass(frozen=True, slots=True)
class LabelRecord:
    audio_sha256: str
    label: ReviewLabelValue
    reviewer: str
    note: str
    cluster_id: str
    cluster_version: str
    rule_version: str
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ApplyResult:
    explicit_count: int
    propagated_count: int
    label_count: int
    signature_count: int


@dataclass(frozen=True, slots=True)
class _IncomingRecords:
    labels: Mapping[str, LabelRecord]
    signatures: tuple[dict[str, object], ...]
    explicit_count: int
    propagated_count: int


def apply_review_labels(
    *,
    review_sheet: Path,
    review_candidate_paths: Sequence[Path],
    labels_path: Path,
    tone_signatures_path: Path,
    reviewer: str,
    explicit_note: str = "manual review",
    cluster_version: str,
    rule_version: str,
) -> ApplyResult:
    _require_nonempty(reviewer, field="reviewer")
    _require_nonempty(explicit_note, field="explicit_note")
    _require_nonempty(cluster_version, field="cluster_version")
    _require_nonempty(rule_version, field="rule_version")
    sheet_rows = _load_review_sheet(review_sheet)
    clusters = _PACKET.cluster_candidates(_PACKET.load_candidates(review_candidate_paths))
    clusters_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    selected_by_cluster = _validate_and_group_sheet(sheet_rows, clusters_by_id=clusters_by_id)
    incoming = _build_incoming_records(
        selected_by_cluster,
        clusters_by_id=clusters_by_id,
        reviewer=reviewer,
        explicit_note=explicit_note,
        cluster_version=cluster_version,
        rule_version=rule_version,
    )
    merged_labels = _merge_labels(_load_existing_labels(labels_path), incoming.labels)
    merged_signatures = _merge_signatures(
        _load_existing_signatures(tone_signatures_path),
        incoming.signatures,
    )
    _write_labels(labels_path, merged_labels)
    _write_signatures(tone_signatures_path, merged_signatures)
    return ApplyResult(
        explicit_count=incoming.explicit_count,
        propagated_count=incoming.propagated_count,
        label_count=len(merged_labels),
        signature_count=len(merged_signatures),
    )


def _build_incoming_records(
    selected_by_cluster: Mapping[str, Sequence[ReviewSheetRow]],
    *,
    clusters_by_id: Mapping[str, object],
    reviewer: str,
    explicit_note: str,
    cluster_version: str,
    rule_version: str,
) -> _IncomingRecords:
    incoming_labels: dict[str, LabelRecord] = {}
    generated_signatures: list[dict[str, object]] = []
    explicit_count = 0
    propagated_count = 0
    for cluster_id in sorted(selected_by_cluster):
        rows = selected_by_cluster[cluster_id]
        cluster = clusters_by_id[cluster_id]
        explicit_rows = tuple(row for row in rows if row.label is not None)
        homogeneous_label = _homogeneous_cluster_label(rows)
        candidates = sorted(cluster.candidates, key=_candidate_sort_key)
        selected_hashes = {row.audio_sha256 for row in explicit_rows}

        if homogeneous_label in {"TONE", "VOICE"}:
            for candidate in candidates:
                provenance: Provenance = (
                    "explicit"
                    if candidate.audio_sha256 in selected_hashes
                    else "cluster_propagated"
                )
                record = _label_record(
                    audio_sha256=candidate.audio_sha256,
                    label=homogeneous_label,
                    reviewer=reviewer,
                    explicit_note=explicit_note,
                    cluster_id=cluster_id,
                    cluster_version=cluster_version,
                    rule_version=rule_version,
                    provenance=provenance,
                )
                _add_label(incoming_labels, record)
                if provenance == "explicit":
                    explicit_count += 1
                else:
                    propagated_count += 1
            if (
                homogeneous_label == "TONE"
                and not cluster.candidates[0].caption_has_censor
                and any(row.dominant_frequency_hz is not None for row in rows)
            ):
                generated_signatures.append(
                    _tone_signature(
                        rows,
                        dataset_id=cluster.candidates[0].dataset_id,
                        cluster_id=cluster_id,
                        cluster_version=cluster_version,
                        rule_version=rule_version,
                    )
                )
            continue

        for row in explicit_rows:
            record = _label_record(
                audio_sha256=row.audio_sha256,
                label=row.label,
                reviewer=reviewer,
                explicit_note=explicit_note,
                cluster_id=cluster_id,
                cluster_version=cluster_version,
                rule_version=rule_version,
                provenance="explicit",
            )
            _add_label(incoming_labels, record)
            explicit_count += 1
    return _IncomingRecords(
        labels=incoming_labels,
        signatures=tuple(generated_signatures),
        explicit_count=explicit_count,
        propagated_count=propagated_count,
    )


def _load_review_sheet(path: Path) -> tuple[ReviewSheetRow, ...]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        raw_rows = list(csv.DictReader(source))
    rows: list[ReviewSheetRow] = []
    for line_number, raw in enumerate(raw_rows, start=2):
        label_text = str(raw.get("label", "")).strip().upper()
        if label_text and label_text not in ALLOWED_LABELS:
            message = f"unsupported label at line {line_number}: {label_text}"
            raise ValueError(message)
        frequency_text = str(raw.get("dominant_frequency_hz", "")).strip()
        frequency = float(frequency_text) if frequency_text else None
        rows.append(
            ReviewSheetRow(
                cluster_id=_required_csv_value(raw, "cluster_id", line_number=line_number),
                audio_sha256=_required_csv_value(
                    raw,
                    "audio_sha256",
                    line_number=line_number,
                ),
                dataset_id=_required_csv_value(raw, "dataset_id", line_number=line_number),
                dominant_frequency_hz=frequency,
                label=label_text or None,
            )
        )
    return tuple(rows)


def _validate_and_group_sheet(
    rows: Sequence[ReviewSheetRow],
    *,
    clusters_by_id: Mapping[str, object],
) -> dict[str, tuple[ReviewSheetRow, ...]]:
    grouped: dict[str, list[ReviewSheetRow]] = {}
    seen_hashes: dict[str, ReviewSheetRow] = {}
    for row in rows:
        cluster = clusters_by_id.get(row.cluster_id)
        if cluster is None:
            message = f"unknown cluster in review sheet: {row.cluster_id}"
            raise ValueError(message)
        candidate = next(
            (item for item in cluster.candidates if item.audio_sha256 == row.audio_sha256),
            None,
        )
        if candidate is None:
            message = f"audio SHA-256 {row.audio_sha256} is not in cluster {row.cluster_id}"
            raise ValueError(message)
        if candidate.dataset_id != row.dataset_id:
            message = f"dataset mismatch for audio SHA-256: {row.audio_sha256}"
            raise ValueError(message)
        if not _same_frequency(candidate.dominant_frequency_hz, row.dominant_frequency_hz):
            message = f"frequency mismatch for audio SHA-256: {row.audio_sha256}"
            raise ValueError(message)
        previous = seen_hashes.get(row.audio_sha256)
        if previous is not None:
            message = f"duplicate review row for audio SHA-256: {row.audio_sha256}"
            raise ValueError(message)
        seen_hashes[row.audio_sha256] = row
        grouped.setdefault(row.cluster_id, []).append(row)
    return {
        cluster_id: tuple(sorted(cluster_rows, key=lambda item: item.audio_sha256))
        for cluster_id, cluster_rows in grouped.items()
    }


def _homogeneous_cluster_label(
    rows: Sequence[ReviewSheetRow],
) -> Literal["TONE", "VOICE"] | None:
    labels = {row.label for row in rows}
    if labels == {"TONE"}:
        return "TONE"
    if labels == {"VOICE"}:
        return "VOICE"
    return None


def _label_record(
    *,
    audio_sha256: str,
    label: ReviewLabelValue,
    reviewer: str,
    explicit_note: str,
    cluster_id: str,
    cluster_version: str,
    rule_version: str,
    provenance: Provenance,
) -> LabelRecord:
    note = (
        explicit_note
        if provenance == "explicit"
        else f"propagated from homogeneous {label} labels in {cluster_id}"
    )
    return LabelRecord(
        audio_sha256=audio_sha256,
        label=label,
        reviewer=reviewer,
        note=note,
        cluster_id=cluster_id,
        cluster_version=cluster_version,
        rule_version=rule_version,
        provenance=provenance,
    )


def _add_label(labels: dict[str, LabelRecord], record: LabelRecord) -> None:
    previous = labels.get(record.audio_sha256)
    if previous is not None and previous.label != record.label:
        message = f"conflicting label for audio SHA-256: {record.audio_sha256}"
        raise ValueError(message)
    if previous is None or (
        previous.provenance == "cluster_propagated" and record.provenance == "explicit"
    ):
        labels[record.audio_sha256] = record


def _tone_signature(
    rows: Sequence[ReviewSheetRow],
    *,
    dataset_id: str,
    cluster_id: str,
    cluster_version: str,
    rule_version: str,
) -> dict[str, object]:
    frequencies = [
        row.dominant_frequency_hz
        for row in rows
        if row.label == "TONE" and row.dominant_frequency_hz is not None
    ]
    if len(frequencies) != len(rows):
        message = f"TONE cluster requires a frequency for every selected row: {cluster_id}"
        raise ValueError(message)
    center = round(float(statistics.median(frequencies)), FREQUENCY_DIGITS)
    signature_key = f"{dataset_id}\0{center:.{FREQUENCY_DIGITS}f}"
    digest = hashlib.sha256(signature_key.encode("utf-8")).hexdigest()[:16]
    return {
        "signature_id": f"tone-{digest}",
        "dataset_id": dataset_id,
        "center_frequency_hz": center,
        "tolerance_fft_bins": DEFAULT_TOLERANCE_FFT_BINS,
        "cluster_id": cluster_id,
        "cluster_version": cluster_version,
        "rule_version": rule_version,
    }


def _load_existing_labels(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    labels: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            message = f"label line {line_number} must be an object"
            raise TypeError(message)
        normalized = {str(key): value for key, value in row.items()}
        audio_sha256 = str(normalized["audio_sha256"])
        if audio_sha256 in labels:
            message = f"duplicate existing label for audio SHA-256: {audio_sha256}"
            raise ValueError(message)
        label = str(normalized["label"])
        if label not in ALLOWED_LABELS:
            message = f"unsupported existing label: {label}"
            raise ValueError(message)
        labels[audio_sha256] = normalized
    return labels


def _merge_labels(
    existing: Mapping[str, dict[str, object]],
    incoming: Mapping[str, LabelRecord],
) -> dict[str, dict[str, object]]:
    merged = {audio_sha256: dict(row) for audio_sha256, row in existing.items()}
    for audio_sha256, record in incoming.items():
        previous = merged.get(audio_sha256)
        if previous is not None:
            if str(previous["label"]) != record.label:
                message = f"conflicting label for audio SHA-256: {audio_sha256}"
                raise ValueError(message)
            continue
        merged[audio_sha256] = {
            "audio_sha256": record.audio_sha256,
            "label": record.label,
            "reviewer": record.reviewer,
            "note": record.note,
            "cluster_id": record.cluster_id,
            "cluster_version": record.cluster_version,
            "rule_version": record.rule_version,
            "provenance": record.provenance,
        }
    return merged


def _load_existing_signatures(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        message = f"tone signatures must be an object list: {path}"
        raise TypeError(message)
    return tuple({str(key): value for key, value in row.items()} for row in payload)


def _merge_signatures(
    existing: Sequence[dict[str, object]],
    incoming: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    by_key: dict[tuple[str, float], dict[str, object]] = {}
    keys_by_id: dict[str, tuple[str, float]] = {}
    for signature in (*existing, *incoming):
        key = _signature_key(signature)
        signature_id = str(signature["signature_id"])
        prior_key = keys_by_id.get(signature_id)
        if prior_key is not None and prior_key != key:
            message = f"conflicting tone signature id: {signature_id}"
            raise ValueError(message)
        keys_by_id[signature_id] = key
        by_key.setdefault(key, dict(signature))
    return tuple(by_key[key] for key in sorted(by_key))


def _signature_key(signature: Mapping[str, object]) -> tuple[str, float]:
    return (
        str(signature["dataset_id"]),
        round(float(signature["center_frequency_hz"]), FREQUENCY_DIGITS),
    )


def _write_labels(path: Path, labels: Mapping[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for audio_sha256 in sorted(labels):
            output.write(json.dumps(labels[audio_sha256], ensure_ascii=False, sort_keys=True))
            output.write("\n")


def _write_signatures(path: Path, signatures: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(signatures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_sort_key(candidate: object) -> tuple[str, int, str]:
    return (candidate.dataset_id, candidate.source_index, candidate.audio_sha256)


def _same_frequency(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    return math.isclose(first, second, rel_tol=0.0, abs_tol=10**-FREQUENCY_DIGITS)


def _required_csv_value(
    row: Mapping[str, str | None],
    field: str,
    *,
    line_number: int,
) -> str:
    value = str(row.get(field, "") or "").strip()
    if not value:
        message = f"missing {field} at line {line_number}"
        raise ValueError(message)
    return value


def _require_nonempty(value: str, *, field: str) -> None:
    if not value.strip():
        message = f"{field} must not be empty"
        raise ValueError(message)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument(
        "--review-candidates",
        action="append",
        dest="review_candidates",
        type=Path,
        required=True,
    )
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--tone-signatures-json", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--explicit-note", default="manual review")
    parser.add_argument("--cluster-version", required=True)
    parser.add_argument("--rule-version", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    apply_review_labels(
        review_sheet=args.review_sheet,
        review_candidate_paths=args.review_candidates,
        labels_path=args.labels_jsonl,
        tone_signatures_path=args.tone_signatures_json,
        reviewer=args.reviewer,
        explicit_note=args.explicit_note,
        cluster_version=args.cluster_version,
        rule_version=args.rule_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
