from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import TracebackType
    from typing import TextIO

_QUALITY_PATH = Path(__file__).with_name("speaker_dataset_quality.py")
_QUALITY_SPEC = importlib.util.spec_from_file_location(
    "_build_clean_speaker_datasets_quality",
    _QUALITY_PATH,
)
if _QUALITY_SPEC is None or _QUALITY_SPEC.loader is None:
    message = f"cannot load dataset quality module: {_QUALITY_PATH}"
    raise RuntimeError(message)
_QUALITY = importlib.util.module_from_spec(_QUALITY_SPEC)
sys.modules[_QUALITY_SPEC.name] = _QUALITY
_QUALITY_SPEC.loader.exec_module(_QUALITY)

CaptionRule = _QUALITY.CaptionRule
ReviewLabel = _QUALITY.ReviewLabel
ToneSignature = _QUALITY.ToneSignature

Decoder = Callable[[Path], tuple[np.ndarray, int]]
CandidateWriter = Callable[[Path, np.ndarray, int], None]
PCM16_SCALE = 32_767.0


@dataclass(frozen=True, slots=True)
class DatasetCatalogEntry:
    dataset_id: str
    output_model_id: str
    index_json: Path
    speaker: str
    audio_root: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetRules:
    labels_by_audio_sha256: Mapping[str, ReviewLabel]
    caption_rules: Sequence[CaptionRule]
    confirmed_signatures: Sequence[ToneSignature]
    rule_version: str


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    entry: DatasetCatalogEntry
    record_id: str
    source_index: int
    speaker: str
    audio_path: Path
    audio_sha256: str
    original_text: str
    rule_version: str
    mapping_decision: str | None
    mapping_candidate_indices: tuple[int, ...]


@dataclass(slots=True)
class _InventoryState:
    first_record_by_hash: dict[str, tuple[str, str | None]]
    first_record_by_pcm_hash: dict[str, str]
    records: list[InventoryRecord]
    on_record: Callable[[InventoryRecord], None] | None

    def append(self, record: InventoryRecord) -> None:
        self.records.append(record)
        if self.on_record is not None:
            self.on_record(record)


@dataclass(frozen=True, slots=True)
class InventoryRecord:
    record_id: str
    dataset_id: str
    output_model_id: str
    source_index: int
    speaker: str
    audio_path: Path
    audio_sha256: str
    pcm_sha256: str | None
    original_text: str
    text: str
    decision: str
    reasons: tuple[str, ...]
    rule_version: str
    duplicate_of: str | None = None
    caption_rule_id: str | None = None
    review_label: str | None = None
    reviewer: str | None = None
    sample_rate: int | None = None
    intervals: tuple[dict[str, object], ...] = ()
    harmonic_ratio: float = 0.0
    clipped_fraction: float = 0.0
    rms_dbfs: float | None = None
    mapping_decision: str | None = None
    mapping_candidate_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CleanDataset:
    rows: tuple[dict[str, object], ...]
    summary: dict[str, int]


class _ProgressWriter:
    def __init__(self, path: Path, *, progress_every: int, dataset_id: str) -> None:
        self._path = path
        self._progress_every = progress_every
        self._dataset_id = dataset_id
        self._output: TextIO | None = None
        self._count = 0

    def __enter__(self) -> _ProgressWriter:  # noqa: PYI034 - remote scanner uses Python 3.10
        self._output = self._path.open("w", encoding="utf-8", newline="\n")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        if self._output is not None:
            self._output.close()

    def write(self, record: InventoryRecord) -> None:
        if self._output is None:
            message = "progress writer is not open"
            raise RuntimeError(message)
        self._output.write(
            json.dumps(serialize_inventory_record(record), ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        self._output.flush()
        self._count += 1
        if self._count % self._progress_every == 0:
            print(f"[{self._dataset_id}] scanned {self._count}")


def build_inventory(
    entry: DatasetCatalogEntry,
    *,
    labels_by_audio_sha256: Mapping[str, ReviewLabel],
    caption_rules: Sequence[CaptionRule],
    rule_version: str,
    decoder: Decoder,
    confirmed_signatures: Sequence[ToneSignature] = (),
    on_record: Callable[[InventoryRecord], None] | None = None,
) -> tuple[InventoryRecord, ...]:
    source_rows = json.loads(entry.index_json.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        message = f"index JSON must contain a list: {entry.index_json}"
        raise TypeError(message)

    rules = DatasetRules(
        labels_by_audio_sha256=labels_by_audio_sha256,
        caption_rules=caption_rules,
        confirmed_signatures=confirmed_signatures,
        rule_version=rule_version,
    )
    state = _InventoryState(
        first_record_by_hash={},
        first_record_by_pcm_hash={},
        records=[],
        on_record=on_record,
    )
    for source_index, source_row in enumerate(source_rows):
        source = _parse_source_record(
            entry,
            source_index=source_index,
            source_row=source_row,
            rule_version=rule_version,
        )
        if source is None:
            continue
        state.append(_record_for_source(source, state=state, rules=rules, decoder=decoder))
    return tuple(state.records)


def _parse_source_record(
    entry: DatasetCatalogEntry,
    *,
    source_index: int,
    source_row: object,
    rule_version: str,
) -> _SourceRecord | None:
    if not isinstance(source_row, dict):
        message = f"index row {source_index} must be an object"
        raise TypeError(message)
    speaker = str(source_row["Speaker"])
    if speaker != entry.speaker:
        return None
    audio_path = _resolve_audio_path(entry, _source_audio_path(source_row, speaker=speaker))
    return _SourceRecord(
        entry=entry,
        record_id=f"{entry.dataset_id}:{source_index:08d}",
        source_index=source_index,
        speaker=speaker,
        audio_path=audio_path,
        audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        original_text=str(source_row["Text"]),
        rule_version=rule_version,
        mapping_decision=(
            str(source_row["MappingDecision"])
            if source_row.get("MappingDecision") is not None
            else None
        ),
        mapping_candidate_indices=tuple(
            int(index) for index in source_row.get("MappingCandidateMetadataIndices", ())
        ),
    )


def _record_for_source(
    source: _SourceRecord,
    *,
    state: _InventoryState,
    rules: DatasetRules,
    decoder: Decoder,
) -> InventoryRecord:
    prior_encoded = state.first_record_by_hash.get(source.audio_sha256)
    if prior_encoded is not None:
        return _duplicate_record(
            source,
            pcm_sha256=prior_encoded[1],
            duplicate_of=prior_encoded[0],
            reason="duplicate_audio_sha256",
        )
    try:
        samples, sample_rate = decoder(source.audio_path)
    except (OSError, RuntimeError, ValueError) as exc:
        record = _invalid_audio_record(source, exc)
    else:
        pcm_sha256 = _pcm_sha256(samples, sample_rate=sample_rate)
        duplicate_of = state.first_record_by_pcm_hash.get(pcm_sha256)
        if duplicate_of is None:
            record = _build_unique_record(
                source=source,
                samples=samples,
                sample_rate=sample_rate,
                pcm_sha256=pcm_sha256,
                labels_by_audio_sha256=rules.labels_by_audio_sha256,
                caption_rules=rules.caption_rules,
                confirmed_signatures=rules.confirmed_signatures,
            )
            state.first_record_by_pcm_hash[pcm_sha256] = source.record_id
        else:
            record = _duplicate_record(
                source,
                pcm_sha256=pcm_sha256,
                duplicate_of=duplicate_of,
                reason="duplicate_pcm_sha256",
            )
    state.first_record_by_hash[source.audio_sha256] = (
        record.duplicate_of or record.record_id,
        record.pcm_sha256,
    )
    return record


def _build_unique_record(
    *,
    source: _SourceRecord,
    samples: np.ndarray,
    sample_rate: int,
    pcm_sha256: str,
    labels_by_audio_sha256: Mapping[str, ReviewLabel],
    caption_rules: Sequence[CaptionRule],
    confirmed_signatures: Sequence[ToneSignature],
) -> InventoryRecord:
    automatic = _QUALITY.classify_audio(
        samples,
        sample_rate,
        dataset_id=source.entry.dataset_id,
        confirmed_signatures=confirmed_signatures,
    )
    decision = automatic.decision
    reasons = list(automatic.reasons)
    label = labels_by_audio_sha256.get(source.audio_sha256)
    if label is not None:
        labeled_decision = _QUALITY.apply_label_override(decision, label)
        if labeled_decision != decision or decision == "REVIEW":
            if decision == "KEEP" and labeled_decision == "EXCLUDE_CONFIRMED_TONE":
                reasons = [reason for reason in reasons if reason != "valid_voice_audio"]
            decision = labeled_decision
            reasons.append(f"user_label:{label.label}")

    repair = _QUALITY.repair_caption(source.original_text, rules=caption_rules)
    if decision in {"KEEP", "REVIEW"}:
        if repair.decision == "REVIEW":
            decision = "REVIEW"
            reasons.append("caption_repair_required")
        elif repair.decision == "REPAIRED" and decision == "KEEP":
            decision = "KEEP_RECAPTIONED"
            reasons.append(f"caption_rule:{repair.rule_id}")
    if source.mapping_decision == "REVIEW" and decision in {
        "KEEP",
        "KEEP_RECAPTIONED",
        "REVIEW",
    }:
        decision = "REVIEW"
        reasons.append("source_mapping_ambiguous")

    return InventoryRecord(
        record_id=source.record_id,
        dataset_id=source.entry.dataset_id,
        output_model_id=source.entry.output_model_id,
        source_index=source.source_index,
        speaker=source.speaker,
        audio_path=source.audio_path,
        audio_sha256=source.audio_sha256,
        pcm_sha256=pcm_sha256,
        original_text=source.original_text,
        text=repair.text,
        decision=decision,
        reasons=tuple(reasons),
        rule_version=source.rule_version,
        caption_rule_id=repair.rule_id,
        review_label=label.label if label is not None else None,
        reviewer=label.reviewer if label is not None else None,
        sample_rate=sample_rate,
        intervals=tuple(asdict(interval) for interval in automatic.intervals),
        harmonic_ratio=automatic.harmonic_ratio,
        clipped_fraction=automatic.clipped_fraction,
        rms_dbfs=automatic.rms_dbfs,
        mapping_decision=source.mapping_decision,
        mapping_candidate_indices=source.mapping_candidate_indices,
    )


def _duplicate_record(
    source: _SourceRecord,
    *,
    pcm_sha256: str | None,
    duplicate_of: str,
    reason: str,
) -> InventoryRecord:
    return InventoryRecord(
        record_id=source.record_id,
        dataset_id=source.entry.dataset_id,
        output_model_id=source.entry.output_model_id,
        source_index=source.source_index,
        speaker=source.speaker,
        audio_path=source.audio_path,
        audio_sha256=source.audio_sha256,
        pcm_sha256=pcm_sha256,
        original_text=source.original_text,
        text=source.original_text,
        decision="EXCLUDE_DUPLICATE",
        reasons=(reason,),
        rule_version=source.rule_version,
        duplicate_of=duplicate_of,
        mapping_decision=source.mapping_decision,
        mapping_candidate_indices=source.mapping_candidate_indices,
    )


def _invalid_audio_record(source: _SourceRecord, error: Exception) -> InventoryRecord:
    return InventoryRecord(
        record_id=source.record_id,
        dataset_id=source.entry.dataset_id,
        output_model_id=source.entry.output_model_id,
        source_index=source.source_index,
        speaker=source.speaker,
        audio_path=source.audio_path,
        audio_sha256=source.audio_sha256,
        pcm_sha256=None,
        original_text=source.original_text,
        text=source.original_text,
        decision="EXCLUDE_INVALID_AUDIO",
        reasons=(f"decode_error:{type(error).__name__}", str(error)),
        rule_version=source.rule_version,
        mapping_decision=source.mapping_decision,
        mapping_candidate_indices=source.mapping_candidate_indices,
    )


def _pcm_sha256(samples: np.ndarray, *, sample_rate: int) -> str:
    normalized = np.ascontiguousarray(samples, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(sample_rate).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def process_dataset(
    entry: DatasetCatalogEntry,
    *,
    output_dir: Path,
    rules: DatasetRules,
    decoder: Decoder,
    candidate_writer: CandidateWriter,
    progress_every: int = 100,
) -> dict[str, int]:
    if progress_every <= 0:
        message = "progress_every must be positive"
        raise ValueError(message)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _ProgressWriter(
        output_dir / "scan-progress.jsonl",
        progress_every=progress_every,
        dataset_id=entry.dataset_id,
    ) as progress:
        records = build_inventory(
            entry,
            labels_by_audio_sha256=rules.labels_by_audio_sha256,
            caption_rules=rules.caption_rules,
            confirmed_signatures=rules.confirmed_signatures,
            rule_version=rules.rule_version,
            decoder=decoder,
            on_record=progress.write,
        )
    inventory_rows = tuple(_source_inventory_row(record) for record in records)
    decision_rows = tuple(serialize_inventory_record(record) for record in records)
    review_rows = [
        serialize_inventory_record(record) for record in records if record.decision == "REVIEW"
    ]
    if review_rows:
        candidate_dir = output_dir / "candidate-audio"
        candidate_dir.mkdir(exist_ok=True)
        for row, record in zip(
            review_rows,
            (record for record in records if record.decision == "REVIEW"),
            strict=True,
        ):
            candidate_path = candidate_dir / f"{entry.dataset_id}_{record.source_index:08d}.wav"
            samples, sample_rate = decoder(record.audio_path)
            candidate_writer(candidate_path, samples, sample_rate)
            row["candidate_wav_path"] = str(candidate_path)

    _write_jsonl(output_dir / "source-inventory.jsonl", inventory_rows)
    _write_jsonl(output_dir / "decisions.jsonl", decision_rows)
    _write_jsonl(output_dir / "review-candidates.jsonl", review_rows)
    result = build_clean_dataset(records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    clean_path = output_dir / "clean-dataset.jsonl"
    if review_rows:
        clean_path.unlink(missing_ok=True)
    else:
        write_clean_dataset(clean_path, records)
    return result.summary


def build_clean_dataset(records: Sequence[InventoryRecord]) -> CleanDataset:
    kept_records = tuple(
        record for record in records if record.decision in {"KEEP", "KEEP_RECAPTIONED"}
    )
    rows = tuple(
        {
            "audio": str(record.audio_path.resolve()),
            "text": record.text,
            "source_id": record.record_id,
            "audio_sha256": record.audio_sha256,
            "pcm_sha256": record.pcm_sha256,
        }
        for record in kept_records
    )
    decision_counts = {
        decision.lower(): sum(record.decision == decision for record in records)
        for decision in (
            "KEEP",
            "KEEP_RECAPTIONED",
            "REVIEW",
            "EXCLUDE_CONFIRMED_TONE",
            "EXCLUDE_INVALID_AUDIO",
            "EXCLUDE_TRANSCRIPT_MISMATCH",
            "EXCLUDE_DUPLICATE",
        )
    }
    return CleanDataset(
        rows=rows,
        summary={
            "total": len(records),
            "unique_audio": len(records) - decision_counts["exclude_duplicate"],
            "kept": len(kept_records),
            "excluded": len(records) - len(kept_records) - decision_counts["review"],
        }
        | decision_counts,
    )


def write_clean_dataset(path: Path, records: Sequence[InventoryRecord]) -> CleanDataset:
    unresolved = [record.record_id for record in records if record.decision == "REVIEW"]
    if unresolved:
        message = f"unresolved REVIEW rows: {len(unresolved)}"
        raise ValueError(message)
    result = build_clean_dataset(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in result.rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    return result


def serialize_inventory_record(record: InventoryRecord) -> dict[str, object]:
    serialized = asdict(record)
    serialized["audio_path"] = str(record.audio_path)
    return serialized


def _source_inventory_row(record: InventoryRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "output_model_id": record.output_model_id,
        "source_index": record.source_index,
        "speaker": record.speaker,
        "audio_path": str(record.audio_path),
        "audio_sha256": record.audio_sha256,
        "pcm_sha256": record.pcm_sha256,
        "original_text": record.original_text,
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")


def _resolve_audio_path(entry: DatasetCatalogEntry, raw_path: str) -> Path:
    path = Path(raw_path.replace("\\", "/"))
    if path.is_absolute():
        return path
    base = entry.audio_root if entry.audio_root is not None else entry.index_json.parent
    return base / path


def _source_audio_path(source_row: Mapping[str, object], *, speaker: str) -> str:
    file_path = source_row.get("FilePath")
    if isinstance(file_path, str) and file_path:
        return file_path
    voice = source_row.get("Voice")
    if isinstance(voice, str) and voice:
        filename = voice if Path(voice).suffix else f"{voice}.ogg"
        return str(Path(speaker) / filename)
    message = "source row requires FilePath or Voice"
    raise ValueError(message)


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    soundfile = importlib.import_module("soundfile")
    samples, sample_rate = soundfile.read(
        str(path),
        dtype="float64",
        always_2d=True,
    )
    return np.asarray(samples, dtype=np.float64).mean(axis=1), int(sample_rate)


def write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * PCM16_SCALE).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    catalog, rule_version = _load_catalog(args.catalog_json)
    rules = DatasetRules(
        labels_by_audio_sha256=_load_labels(args.labels_jsonl),
        caption_rules=_load_caption_rules(args.caption_rules_json),
        confirmed_signatures=_load_tone_signatures(args.tone_signatures_json),
        rule_version=rule_version,
    )
    summaries: dict[str, dict[str, int]] = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for entry in catalog:
        summaries[entry.dataset_id] = process_dataset(
            entry,
            output_dir=args.output_root / entry.dataset_id,
            rules=rules,
            decoder=decode_audio,
            candidate_writer=write_pcm16_wav,
            progress_every=args.progress_every,
        )
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _load_catalog(path: Path) -> tuple[tuple[DatasetCatalogEntry, ...], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        message = f"catalog must contain a datasets list: {path}"
        raise TypeError(message)
    entries = tuple(
        DatasetCatalogEntry(
            dataset_id=str(row["dataset_id"]),
            output_model_id=str(row["output_model_id"]),
            index_json=_resolve_catalog_path(path, str(row["index_json"])),
            speaker=str(row["speaker"]),
            audio_root=(
                _resolve_catalog_path(path, str(row["audio_root"]))
                if row.get("audio_root") is not None
                else None
            ),
        )
        for row in payload["datasets"]
        if isinstance(row, dict)
    )
    if len(entries) != len(payload["datasets"]):
        message = "every catalog dataset must be an object"
        raise TypeError(message)
    return entries, str(payload["rule_version"])


def _load_labels(path: Path) -> dict[str, ReviewLabel]:
    labels: dict[str, ReviewLabel] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            message = f"label line {line_number} must be an object"
            raise TypeError(message)
        audio_sha256 = str(row["audio_sha256"])
        if audio_sha256 in labels:
            message = f"duplicate label for audio SHA-256: {audio_sha256}"
            raise ValueError(message)
        labels[audio_sha256] = ReviewLabel(
            label=str(row["label"]),
            reviewer=str(row["reviewer"]),
            note=str(row.get("note", "")),
        )
    return labels


def _load_caption_rules(path: Path) -> tuple[CaptionRule, ...]:
    return tuple(
        CaptionRule(
            rule_id=str(row["rule_id"]),
            source=str(row["source"]),
            replacement=str(row["replacement"]),
        )
        for row in _load_json_objects(path)
    )


def _load_tone_signatures(path: Path | None) -> tuple[ToneSignature, ...]:
    if path is None:
        return ()
    return tuple(
        ToneSignature(
            signature_id=str(row["signature_id"]),
            dataset_id=str(row["dataset_id"]),
            center_frequency_hz=float(row["center_frequency_hz"]),
            tolerance_fft_bins=float(row.get("tolerance_fft_bins", 1.5)),
        )
        for row in _load_json_objects(path)
    )


def _load_json_objects(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        message = f"JSON file must contain an object list: {path}"
        raise TypeError(message)
    return tuple(payload)


def _resolve_catalog_path(catalog_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else catalog_path.parent / path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-json", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--caption-rules-json", type=Path, required=True)
    parser.add_argument("--tone-signatures-json", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
