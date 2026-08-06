# ruff: noqa: INP001
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_RULE_VERSION = "2026-07-31.v1"
DEFAULT_KASUMI_ROWS = 772
LATENT_FRAMES_PER_SECOND = 25.0
CENSOR_MARKERS = ("◯", "○", "〇")  # noqa: RUF001


@dataclass(frozen=True, slots=True)
class DirectDatasetSpec:
    dataset_id: str
    output_model_id: str
    speaker: str
    index_parts: tuple[str, ...]
    audio_root_parts: tuple[str, ...]


DIRECT_DATASETS = (
    DirectDatasetSpec(
        dataset_id="oop176_natsu_no_owari_sp_dcec9a11d3",
        output_model_id="oop176_natsu_no_owari_sp_dcec9a11d3",
        speaker="ミオ",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_cube_natsu_no_owari_7z",
            "CUBE_Natsu no Owari",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_cube_natsu_no_owari_7z",
            "CUBE_Natsu no Owari",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop52_aibeya_2_sp_5d544fe890",
        output_model_id="oop52_aibeya_2_sp_5d544fe890",
        speaker="朔",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_aibeya_2_7z",
            "Azarashi Soft_Aibeya 2",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_aibeya_2_7z",
            "Azarashi Soft_Aibeya 2",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop53_aibeya_sp_f7269f5ffc",
        output_model_id="oop53_aibeya_sp_f7269f5ffc",
        speaker="綾希",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_aibeya_7z",
            "Azarashi Soft_Aibeya",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_aibeya_7z",
            "Azarashi Soft_Aibeya",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop54_aikagi_2_sp_85dded42a7",
        output_model_id="oop54_aikagi_2_sp_85dded42a7",
        speaker="綾乃",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_aikagi_2_7z",
            "Azarashi Soft_Aikagi 2",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_aikagi_2_7z",
            "Azarashi Soft_Aikagi 2",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop55_aikagi_3_sp_683c9895cc",
        output_model_id="miu",
        speaker="アイ",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_aikagi_3_7z",
            "Azarashi Soft_Aikagi 3",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_aikagi_3_7z",
            "Azarashi Soft_Aikagi 3",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop68_maid_san_no_iru_kurashi_s_sp_e4da3225a4",
        output_model_id="oop68_maid_san_no_iru_kurashi_s_sp_e4da3225a4",
        speaker="エレナ",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_maid_san_no_iru_kurashi_s_7z",
            "Azarashi Soft_Maid-san no Iru Kurashi S",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_maid_san_no_iru_kurashi_s_7z",
            "Azarashi Soft_Maid-san no Iru Kurashi S",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop69_maid_san_no_iru_kurashi_sp_07497b0fbd",
        output_model_id="oop69_maid_san_no_iru_kurashi_sp_07497b0fbd",
        speaker="イヴ",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_maid_san_no_iru_kurashi_7z",
            "Azarashi Soft_Maid-san no Iru Kurashi",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_maid_san_no_iru_kurashi_7z",
            "Azarashi Soft_Maid-san no Iru Kurashi",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop70_osananajimi_no_iru_kurashi_sp_7195504dbb",
        output_model_id="oop70_osananajimi_no_iru_kurashi_sp_7195504dbb",
        speaker="舞雪",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_osananajimi_no_iru_kurashi_7z",
            "Azarashi Soft_Osananajimi no Iru Kurashi",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_osananajimi_no_iru_kurashi_7z",
            "Azarashi Soft_Osananajimi no Iru Kurashi",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop73_toshishita_kanojo_sp_6b50dbf844",
        output_model_id="oop73_toshishita_kanojo_sp_6b50dbf844",
        speaker="絢音",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_toshishita_kanojo_7z",
            "Azarashi Soft_Toshishita Kanojo",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_azarashi_soft_toshishita_kanojo_7z",
            "Azarashi Soft_Toshishita Kanojo",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="oop77_anabel_maidgarden_sp_451488a7c1",
        output_model_id="oop77_anabel_maidgarden_sp_451488a7c1",
        speaker="アナベル",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_barista_lab_anabel_maidgarden_7z",
            "Barista Lab_Anabel Maidgarden",
            "index.json",
        ),
        audio_root_parts=(
            "top_speakers_20260529",
            "extracted",
            "galgame_barista_lab_anabel_maidgarden_7z",
            "Barista Lab_Anabel Maidgarden",
        ),
    ),
    DirectDatasetSpec(
        dataset_id="narrator_toshiue_ama",
        output_model_id="narrator_sayoko",
        speaker="ama",
        index_parts=(
            "top_speakers_20260529",
            "indexes",
            "galgame_azarashi_soft_toshiue_kanojo_7z",
            "Azarashi Soft_Toshiue Kanojo",
            "index.json",
        ),
        audio_root_parts=(
            "narrator_toshiue_ama_20260530",
            "extracted",
            "Azarashi Soft_Toshiue Kanojo",
        ),
    ),
)


def create_catalog(
    *,
    training_root: Path,
    output_root: Path,
    rule_version: str,
    expected_kasumi_rows: int = DEFAULT_KASUMI_ROWS,
) -> dict[str, object]:
    direct_entries = [
        _build_direct_entry(training_root.resolve(), spec) for spec in DIRECT_DATASETS
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    kasumi_entry, kasumi_audit = _build_kasumi_entry(
        training_root=training_root.resolve(),
        output_root=output_root.resolve(),
        expected_rows=expected_kasumi_rows,
    )
    datasets = [*direct_entries, kasumi_entry]
    payload: dict[str, object] = {
        "rule_version": rule_version,
        "datasets": datasets,
    }
    _write_json(output_root / "catalog.json", payload)
    _write_json(
        output_root / "source-audit.json",
        {
            "rule_version": rule_version,
            "datasets": [
                {
                    "dataset_id": row["dataset_id"],
                    "source_index": row["source_index"],
                    "source_index_sha256": row["source_index_sha256"],
                    "source_speaker_rows": row["source_speaker_rows"],
                    "missing_audio_count": row["missing_audio_count"],
                }
                for row in datasets
            ],
            "kasumi_mapping_review_count": sum(row["decision"] == "REVIEW" for row in kasumi_audit),
        },
    )
    return payload


def _build_direct_entry(training_root: Path, spec: DirectDatasetSpec) -> dict[str, object]:
    index_path = training_root.joinpath(*spec.index_parts)
    audio_root = training_root.joinpath(*spec.audio_root_parts)
    rows = _load_json_list(index_path)
    speaker_rows = [row for row in rows if str(row.get("Speaker")) == spec.speaker]
    for row in speaker_rows:
        raw_audio_path = _source_audio_path(row, speaker=spec.speaker)
        audio_path = raw_audio_path if raw_audio_path.is_absolute() else audio_root / raw_audio_path
        if not audio_path.is_file():
            message = f"{spec.dataset_id}: missing audio: {audio_path}"
            raise FileNotFoundError(message)
    return {
        "dataset_id": spec.dataset_id,
        "output_model_id": spec.output_model_id,
        "index_json": str(index_path.resolve()),
        "speaker": spec.speaker,
        "audio_root": str(audio_root.resolve()),
        "source_index": str(index_path.resolve()),
        "source_index_sha256": _sha256(index_path),
        "source_index_rows": len(rows),
        "source_speaker_rows": len(speaker_rows),
        "censor_marker_rows": sum(
            any(marker in str(row.get("Text", "")) for marker in CENSOR_MARKERS)
            for row in speaker_rows
        ),
        "missing_audio_count": 0,
    }


def _source_audio_path(row: dict[str, object], *, speaker: str) -> Path:
    file_path = row.get("FilePath")
    if isinstance(file_path, str) and file_path:
        return Path(file_path)
    voice = row.get("Voice")
    if isinstance(voice, str) and voice:
        filename = voice if Path(voice).suffix else f"{voice}.ogg"
        return Path(speaker) / filename
    message = "source row requires FilePath or Voice"
    raise ValueError(message)


def _build_kasumi_entry(
    *,
    training_root: Path,
    output_root: Path,
    expected_rows: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = training_root / "0006_aiko" / "manifest.jsonl"
    metadata_path = training_root / "0006_aiko_check" / "metadata.csv"
    manifest_rows = _load_jsonl(manifest_path)
    if len(manifest_rows) != expected_rows:
        message = (
            f"kasumi source manifest must contain {expected_rows} rows, found {len(manifest_rows)}"
        )
        raise ValueError(message)
    metadata_rows = _load_metadata(metadata_path)
    normalized_rows, audit_rows = _map_kasumi_rows(
        manifest_rows=manifest_rows,
        metadata_rows=metadata_rows,
        metadata_path=metadata_path,
    )
    kasumi_root = output_root / "kasumi"
    kasumi_root.mkdir(parents=True, exist_ok=True)
    normalized_index = kasumi_root / "normalized-index.json"
    mapping_audit = kasumi_root / "mapping-audit.jsonl"
    _write_json(normalized_index, normalized_rows)
    _write_jsonl(mapping_audit, audit_rows)
    review_count = sum(row["decision"] == "REVIEW" for row in audit_rows)
    return (
        {
            "dataset_id": "kasumi",
            "output_model_id": "kasumi",
            "index_json": str(normalized_index.resolve()),
            "speaker": "藍子",
            "audio_root": str(metadata_path.parent.joinpath("speaker_audio").resolve()),
            "source_index": str(manifest_path.resolve()),
            "source_index_sha256": _sha256(manifest_path),
            "source_index_rows": len(manifest_rows),
            "source_speaker_rows": len(manifest_rows),
            "source_metadata_csv": str(metadata_path.resolve()),
            "source_metadata_sha256": _sha256(metadata_path),
            "source_metadata_rows": len(metadata_rows),
            "censor_marker_rows": sum(
                any(marker in str(row.get("text", "")) for marker in CENSOR_MARKERS)
                for row in manifest_rows
            ),
            "missing_audio_count": 0,
            "mapping_review_count": review_count,
            "mapping_audit_jsonl": str(mapping_audit.resolve()),
        },
        audit_rows,
    )


def _map_kasumi_rows(
    *,
    manifest_rows: Sequence[dict[str, object]],
    metadata_rows: Sequence[dict[str, str]],
    metadata_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    cursor = 0
    for manifest_index, manifest_row in enumerate(manifest_rows):
        text = str(manifest_row["text"])
        candidate_indices = [
            index
            for index in range(cursor, len(metadata_rows))
            if metadata_rows[index]["text"] == text
        ]
        if not candidate_indices:
            message = (
                f"kasumi manifest row {manifest_index} has no ordered metadata match: {text!r}"
            )
            raise ValueError(message)
        frame_candidate_indices = _frame_matching_candidate_indices(
            manifest_row,
            metadata_rows=metadata_rows,
            candidate_indices=candidate_indices,
        )
        if frame_candidate_indices:
            candidate_indices = frame_candidate_indices
        selected_index = candidate_indices[0]
        candidate_audio_paths = [
            _metadata_audio_path(metadata_path, metadata_rows[index]["audio"])
            for index in candidate_indices
        ]
        for audio_path in candidate_audio_paths:
            if not audio_path.is_file():
                message = f"kasumi: missing audio: {audio_path}"
                raise FileNotFoundError(message)
        decision = "REVIEW" if len(candidate_indices) > 1 else "MAPPED"
        selected_audio = candidate_audio_paths[0]
        normalized.append(
            {
                "Speaker": "藍子",
                "Text": text,
                "FilePath": str(selected_audio.resolve()),
                "SourceManifestIndex": manifest_index,
                "SourceMetadataIndex": selected_index,
                "MappingDecision": decision,
                "MappingCandidateMetadataIndices": candidate_indices,
            }
        )
        audit.append(
            {
                "dataset_id": "kasumi",
                "manifest_index": manifest_index,
                "text": text,
                "decision": decision,
                "selected_metadata_index": selected_index,
                "selected_audio_path": str(selected_audio.resolve()),
                "candidate_metadata_indices": candidate_indices,
                "candidate_audio_paths": [str(path.resolve()) for path in candidate_audio_paths],
            }
        )
        cursor = selected_index + 1
    return normalized, audit


def _frame_matching_candidate_indices(
    manifest_row: dict[str, object],
    *,
    metadata_rows: Sequence[dict[str, str]],
    candidate_indices: Sequence[int],
) -> list[int]:
    num_frames = manifest_row.get("num_frames")
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        return []
    matches: list[int] = []
    for index in candidate_indices:
        raw_duration = metadata_rows[index].get("duration", "").strip()
        try:
            duration = float(raw_duration)
        except ValueError:
            continue
        if (
            duration > 0.0
            and math.isfinite(duration)
            and math.ceil(duration * LATENT_FRAMES_PER_SECOND) == num_frames
        ):
            matches.append(index)
    return matches


def _load_json_list(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        message = f"JSON file must contain an object list: {path}"
        raise TypeError(message)
    return payload


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            message = f"JSONL line {line_number} must be an object: {path}"
            raise TypeError(message)
        rows.append(row)
    return rows


def _load_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = [dict(row) for row in csv.DictReader(source)]
    if not all(row.get("audio") and row.get("text") for row in rows):
        message = f"metadata CSV requires audio and text columns: {path}"
        raise ValueError(message)
    return rows


def _metadata_audio_path(metadata_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else metadata_path.parent / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    create_catalog(
        training_root=args.training_root,
        output_root=args.output_root,
        rule_version=args.rule_version,
        expected_kasumi_rows=args.expected_kasumi_rows,
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rule-version", default=DEFAULT_RULE_VERSION)
    parser.add_argument("--expected-kasumi-rows", type=int, default=DEFAULT_KASUMI_ROWS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
