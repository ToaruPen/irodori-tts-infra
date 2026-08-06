from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MANIFEST_SCHEMA_VERSION = "speaker-checkpoint-evaluation-manifest/v1"
INDEX_SCHEMA_VERSION = "speaker-checkpoint-evaluation-manifest-index/v1"
REFERENCE_SCHEMA_VERSION = "speaker-similarity-references/v1"
EXPECTED_MODEL_COUNT = 12
EXPECTED_REFERENCE_COUNT = 25
EXPECTED_EMBEDDING_SHAPE = (16, 768)
CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
SEEDS = (1234, 5678)
STYLES = ("neutral", "calm")
SAFETENSORS_HEADER_LENGTH_BYTES = 8
SAFETENSORS_OFFSET_COUNT = 2
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
SHA256_HEX_LENGTH = 64
CHECKPOINT_STEP_PATTERN = re.compile(r"checkpoint[-_](\d+)")


@dataclass(frozen=True, slots=True)
class Job:
    model_id: str
    clean_manifest: Path
    config: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Candidate:
    checkpoint_step: int
    embedding_path: Path
    embedding_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_step": self.checkpoint_step,
            "embedding_path": str(self.embedding_path),
            "embedding_sha256": self.embedding_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReferenceManifest:
    model_id: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class MetricModel:
    model_id: str
    revision: str
    source_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class BuildConfig:
    base_checkpoint: str
    base_checkpoint_sha256: str
    base_revision: str
    speaker_embedding: MetricModel
    transcription: MetricModel


@dataclass(frozen=True, slots=True)
class JobsContract:
    base_checkpoint_path: Path
    base_checkpoint_sha256: str
    checkpoint_revision: str
    upstream_commit: str

    def to_dict(self) -> dict[str, str]:
        return {
            "base_checkpoint_path": str(self.base_checkpoint_path),
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "checkpoint_revision": self.checkpoint_revision,
            "upstream_commit": self.upstream_commit,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(row: Mapping[str, object]) -> str:
    serialized = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def build_manifests(
    *,
    training_status: Path,
    training_jobs: Path,
    reference_wavs: Sequence[Path],
    output_dir: Path,
    config: BuildConfig,
) -> dict[str, object]:
    if output_dir.exists():
        message = f"output directory already exists: {output_dir}"
        raise FileExistsError(message)
    jobs, jobs_contract = _load_jobs(training_jobs, config=config)
    status_rows = _load_status_rows(
        training_status,
        jobs=jobs,
        jobs_contract=jobs_contract,
    )
    references = _load_reference_manifests(reference_wavs, jobs=jobs)
    manifests, index_entries = _build_payloads(
        jobs=jobs,
        status_rows=status_rows,
        references=references,
        upstream_commit=jobs_contract.upstream_commit,
        config=config,
    )
    index: dict[str, object] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "base_checkpoint": {
            "model_id": config.base_checkpoint,
            "sha256": config.base_checkpoint_sha256,
            "revision": config.base_revision,
        },
        "metric_models": {
            "speaker_embedding": config.speaker_embedding.to_dict(),
            "transcription": config.transcription.to_dict(),
        },
        "manifests": index_entries,
        "provenance": {
            "builder_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "jobs_contract": jobs_contract.to_dict(),
            "inputs": {
                "training_jobs": {
                    "path": str(training_jobs.resolve()),
                    "sha256": sha256_file(training_jobs),
                },
                "training_status": {
                    "path": str(training_status.resolve()),
                    "sha256": sha256_file(training_status),
                },
                "reference_wavs": [
                    {
                        "model_id": reference.model_id,
                        "path": str(reference.path),
                        "sha256": reference.sha256,
                    }
                    for reference in references.values()
                ],
            },
        },
    }
    _publish(output_dir, manifests=manifests, index=index)
    return index


def _load_jobs(
    path: Path,
    *,
    config: BuildConfig,
) -> tuple[tuple[Job, ...], JobsContract]:
    payload = _read_json(path)
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_MODEL_COUNT:
        message = f"training jobs must contain exactly {EXPECTED_MODEL_COUNT} jobs"
        raise ValueError(message)
    base_checkpoint_path = _resolve_input(
        payload,
        "base_checkpoint_path",
        base=path.parent,
        source="training jobs",
    )
    base_checkpoint_sha256 = _required_sha256(
        payload,
        "base_checkpoint_sha256",
        source="training jobs",
    )
    checkpoint_revision = _required_string(
        payload,
        "checkpoint_revision",
        source="training jobs",
    )
    upstream_commit = _required_string(payload, "upstream_commit", source="training jobs")
    if base_checkpoint_sha256 != config.base_checkpoint_sha256:
        message = "training jobs base_checkpoint_sha256 does not match CLI configuration"
        raise ValueError(message)
    if checkpoint_revision != config.base_revision:
        message = "training jobs checkpoint_revision does not match CLI configuration"
        raise ValueError(message)
    if not base_checkpoint_path.is_file():
        message = f"training jobs base checkpoint does not exist: {base_checkpoint_path}"
        raise ValueError(message)
    actual_base_checkpoint_sha256 = sha256_file(base_checkpoint_path)
    if actual_base_checkpoint_sha256 != base_checkpoint_sha256:
        message = "training jobs base checkpoint file SHA-256 does not match declared value"
        raise ValueError(message)
    contract = JobsContract(
        base_checkpoint_path=base_checkpoint_path,
        base_checkpoint_sha256=base_checkpoint_sha256,
        checkpoint_revision=checkpoint_revision,
        upstream_commit=upstream_commit,
    )
    jobs = tuple(_parse_job(raw, base=path.parent) for raw in raw_jobs)
    model_ids = [job.model_id for job in jobs]
    if len(set(model_ids)) != EXPECTED_MODEL_COUNT:
        message = "training jobs contain duplicate model_id"
        raise ValueError(message)
    if not any("anabel" in model_id.casefold() for model_id in model_ids):
        message = "training jobs must include the Anabel seed model"
        raise ValueError(message)
    return jobs, contract


def _parse_job(raw: object, *, base: Path) -> Job:
    if not isinstance(raw, dict):
        message = "training job entries must be objects"
        raise TypeError(message)
    model_id = _required_string(raw, "model_id", source="training job")
    if "/" in model_id or "\\" in model_id or model_id in {".", ".."}:
        message = f"training job model_id is unsafe: {model_id!r}"
        raise ValueError(message)
    clean_manifest = _resolve_input(raw, "clean_manifest", base=base, source=model_id)
    config = _resolve_input(raw, "config", base=base, source=model_id)
    output_dir = _resolve_path(raw, "output_dir", base=base, source=model_id)
    if not clean_manifest.is_file():
        message = f"clean manifest does not exist for {model_id}: {clean_manifest}"
        raise ValueError(message)
    if not config.is_file():
        message = f"training config does not exist for {model_id}: {config}"
        raise ValueError(message)
    return Job(
        model_id=model_id,
        clean_manifest=clean_manifest,
        config=config,
        output_dir=output_dir,
    )


def _load_status_rows(
    path: Path,
    *,
    jobs: Sequence[Job],
    jobs_contract: JobsContract,
) -> dict[str, dict[str, object]]:
    jobs_by_id = {job.model_id: job for job in jobs}
    indexed: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"invalid training status JSON on line {line_number}"
            raise ValueError(message) from exc
        if not isinstance(row, dict):
            message = f"training status line {line_number} must be an object"
            raise TypeError(message)
        if row.get("event") != "finished":
            continue
        model_id = _required_string(row, "model_id", source="finished status")
        job = jobs_by_id.get(model_id)
        if job is None:
            message = f"finished status has unknown model_id: {model_id}"
            raise ValueError(message)
        if (
            row.get("status") == "success"
            and row.get("exit_code") == 0
            and _status_matches_current_provenance(
                row,
                job=job,
                jobs_contract=jobs_contract,
            )
        ):
            indexed[model_id] = row
    missing_ids = jobs_by_id.keys() - indexed.keys()
    if missing_ids:
        message = (
            "training status has no reusable successful finished status for: "
            f"{', '.join(sorted(missing_ids))}"
        )
        raise ValueError(message)
    return indexed


def _status_matches_current_provenance(
    row: Mapping[str, object],
    *,
    job: Job,
    jobs_contract: JobsContract,
) -> bool:
    expected = {
        "clean_manifest_sha256": sha256_file(job.clean_manifest),
        "config_sha256": sha256_file(job.config),
        "checkpoint_sha256": jobs_contract.base_checkpoint_sha256,
        "checkpoint_revision": jobs_contract.checkpoint_revision,
        "upstream_commit": jobs_contract.upstream_commit,
    }
    return all(row.get(field) == value for field, value in expected.items())


def _load_reference_manifests(
    paths: Sequence[Path],
    *,
    jobs: Sequence[Job],
) -> dict[str, ReferenceManifest]:
    if len(paths) != EXPECTED_MODEL_COUNT:
        message = f"exactly {EXPECTED_MODEL_COUNT} --reference-wavs inputs are required"
        raise ValueError(message)
    references: dict[str, ReferenceManifest] = {}
    for path in paths:
        payload = _read_json(path)
        if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
            message = f"reference manifest has unsupported schema: {path}"
            raise ValueError(message)
        model_id = _required_string(payload, "model_id", source=f"reference manifest {path}")
        if model_id in references:
            message = f"duplicate reference manifest model_id: {model_id}"
            raise ValueError(message)
        if payload.get("all_reference_wavs_finite") is not True:
            message = f"reference manifest finite flag is not true for {model_id}"
            raise ValueError(message)
        if payload.get("all_selected_source_hashes_verified") is not True:
            message = f"reference manifest source hash flag is not true for {model_id}"
            raise ValueError(message)
        raw_references = payload.get("references")
        if not isinstance(raw_references, list) or len(raw_references) != EXPECTED_REFERENCE_COUNT:
            message = (
                f"reference manifest for {model_id} must contain exactly "
                f"{EXPECTED_REFERENCE_COUNT} references"
            )
            raise ValueError(message)
        seen_paths: set[Path] = set()
        for raw_reference in raw_references:
            _validate_reference(raw_reference, manifest_path=path, seen_paths=seen_paths)
        references[model_id] = ReferenceManifest(
            model_id=model_id,
            path=path.resolve(),
            sha256=sha256_file(path),
        )
    expected_ids = {job.model_id for job in jobs}
    if references.keys() != expected_ids:
        message = "reference manifest model_ids do not exactly match training jobs"
        raise ValueError(message)
    return references


def _validate_reference(
    raw: object,
    *,
    manifest_path: Path,
    seen_paths: set[Path],
) -> None:
    if not isinstance(raw, dict):
        message = f"reference entries must be objects: {manifest_path}"
        raise TypeError(message)
    raw_path = _required_string(raw, "reference_wav_path", source=str(manifest_path))
    reference_path = Path(raw_path)
    if not reference_path.is_absolute():
        reference_path = manifest_path.parent / reference_path
    reference_path = reference_path.resolve()
    if reference_path in seen_paths:
        message = f"duplicate reference WAV path: {reference_path}"
        raise ValueError(message)
    seen_paths.add(reference_path)
    expected_sha256 = _required_sha256(raw, "reference_wav_sha256", source=str(manifest_path))
    if not reference_path.is_file():
        message = f"reference WAV does not exist: {reference_path}"
        raise ValueError(message)
    if sha256_file(reference_path) != expected_sha256:
        message = f"reference WAV SHA-256 mismatch: {reference_path}"
        raise ValueError(message)
    _required_string(raw, "source_id", source=str(manifest_path))


def _build_payloads(
    *,
    jobs: Sequence[Job],
    status_rows: Mapping[str, dict[str, object]],
    references: Mapping[str, ReferenceManifest],
    upstream_commit: str,
    config: BuildConfig,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    manifests: dict[str, dict[str, object]] = {}
    index_entries: list[dict[str, object]] = []
    for job in jobs:
        row = status_rows[job.model_id]
        _validate_status_provenance(
            row,
            job=job,
            upstream_commit=upstream_commit,
            config=config,
        )
        candidates = _select_candidates(row, job=job)
        run_id = _resolve_run_id(row, job=job)
        reference = references[job.model_id]
        manifest = _manifest_payload(
            job=job,
            candidates=candidates,
            run_id=run_id,
            reference=reference,
            config=config,
        )
        relative_path = Path(job.model_id) / "evaluation-manifest.json"
        manifests[relative_path.as_posix()] = manifest
        selected = [candidate.to_dict() for candidate in candidates]
        index_entries.append(
            {
                "model_id": job.model_id,
                "manifest_path": relative_path.as_posix(),
                "manifest_sha256": "",
                "selected_candidates": selected,
                "provenance": {
                    "clean_manifest_sha256": row["clean_manifest_sha256"],
                    "config_sha256": row["config_sha256"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "checkpoint_revision": row["checkpoint_revision"],
                    "upstream_commit": row["upstream_commit"],
                    "run_id": run_id,
                },
            },
        )
    return manifests, index_entries


def _resolve_run_id(row: Mapping[str, object], *, job: Job) -> str:
    if "seeded_existing_run" not in row:
        return canonical_sha256(row)
    seeded_existing_run = row["seeded_existing_run"]
    if not isinstance(seeded_existing_run, dict):
        message = f"seeded existing run must be an object for {job.model_id}"
        raise TypeError(message)
    source = f"seeded existing run for {job.model_id}"
    raw_path = _required_string(
        seeded_existing_run,
        "run_provenance_path",
        source=source,
    )
    run_provenance_path = Path(raw_path)
    if not run_provenance_path.is_absolute():
        message = f"run provenance path must be absolute for {job.model_id}"
        raise ValueError(message)
    resolved_run_provenance_path = run_provenance_path.resolve()
    if (
        str(resolved_run_provenance_path) != raw_path
        or run_provenance_path.is_symlink()
        or not resolved_run_provenance_path.is_file()
    ):
        message = f"run provenance path is unsafe or missing for {job.model_id}"
        raise ValueError(message)
    declared_sha256 = _required_sha256(
        seeded_existing_run,
        "run_provenance_sha256",
        source=source,
    )
    run_provenance_bytes = resolved_run_provenance_path.read_bytes()
    if hashlib.sha256(run_provenance_bytes).hexdigest() != declared_sha256:
        message = f"run provenance SHA-256 mismatch for {job.model_id}"
        raise ValueError(message)
    try:
        run_provenance = json.loads(run_provenance_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"invalid JSON: {resolved_run_provenance_path}"
        raise ValueError(message) from exc
    if not isinstance(run_provenance, dict):
        message = f"JSON document must be an object: {resolved_run_provenance_path}"
        raise TypeError(message)
    provenance_model_id = _required_string(
        run_provenance,
        "model_id",
        source=f"run provenance {resolved_run_provenance_path}",
    )
    if provenance_model_id != job.model_id:
        message = f"run provenance model_id mismatch for {job.model_id}"
        raise ValueError(message)
    return declared_sha256


def _validate_status_provenance(
    row: Mapping[str, object],
    *,
    job: Job,
    upstream_commit: str,
    config: BuildConfig,
) -> None:
    expected = {
        "clean_manifest_sha256": sha256_file(job.clean_manifest),
        "config_sha256": sha256_file(job.config),
        "checkpoint_sha256": config.base_checkpoint_sha256,
        "checkpoint_revision": config.base_revision,
        "upstream_commit": upstream_commit,
    }
    mismatches = [field for field, value in expected.items() if row.get(field) != value]
    if mismatches:
        message = f"status provenance mismatch for {job.model_id}: {', '.join(mismatches)}"
        raise ValueError(message)


def _select_candidates(row: Mapping[str, object], *, job: Job) -> tuple[Candidate, ...]:
    raw_candidates = row.get("candidate_checkpoints")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        message = f"candidate checkpoints are missing for {job.model_id}"
        raise ValueError(message)
    selected: dict[int, Candidate] = {}
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            message = f"candidate checkpoint entries must be objects for {job.model_id}"
            raise TypeError(message)
        raw_path = _required_string(raw, "path", source=f"candidate {job.model_id}")
        path = Path(raw_path).resolve()
        matches = tuple(CHECKPOINT_STEP_PATTERN.finditer(str(path)))
        if not matches:
            continue
        step = int(matches[-1].group(1))
        if step not in CHECKPOINT_STEPS:
            continue
        if step in selected:
            message = f"duplicate candidate checkpoint step {step} for {job.model_id}"
            raise ValueError(message)
        if not path.is_relative_to(job.output_dir.resolve()):
            message = f"candidate checkpoint is outside output_dir for {job.model_id}: {path}"
            raise ValueError(message)
        if not path.is_file():
            message = f"candidate checkpoint does not exist: {path}"
            raise ValueError(message)
        expected_sha256 = _required_sha256(raw, "sha256", source=f"candidate {job.model_id}")
        actual_sha256 = validate_speaker_embedding(path)
        if actual_sha256 != expected_sha256:
            message = f"candidate checkpoint SHA-256 mismatch: {path}"
            raise ValueError(message)
        selected[step] = Candidate(
            checkpoint_step=step,
            embedding_path=path,
            embedding_sha256=actual_sha256,
        )
    if tuple(sorted(selected)) != CHECKPOINT_STEPS:
        message = f"candidate checkpoints for {job.model_id} must include exact required steps"
        raise ValueError(message)
    final = selected[CHECKPOINT_STEPS[-1]]
    if (
        row.get("last_checkpoint") != str(final.embedding_path)
        or row.get("last_checkpoint_sha256") != final.embedding_sha256
    ):
        message = f"last checkpoint binding mismatch for {job.model_id}"
        raise ValueError(message)
    return tuple(selected[step] for step in CHECKPOINT_STEPS)


def validate_speaker_embedding(path: Path) -> str:
    header, data_start = _read_safetensors_header(path)
    raw_tensor = header.get("speaker_embedding")
    if not isinstance(raw_tensor, dict):
        message = f"speaker_embedding tensor is missing: {path}"
        raise TypeError(message)
    if raw_tensor.get("dtype") != "F32":
        message = f"speaker_embedding must be F32: {path}"
        raise ValueError(message)
    if raw_tensor.get("shape") != list(EXPECTED_EMBEDDING_SHAPE):
        message = f"speaker_embedding must have shape {EXPECTED_EMBEDDING_SHAPE}: {path}"
        raise ValueError(message)
    offsets = raw_tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != SAFETENSORS_OFFSET_COUNT
        or not all(isinstance(offset, int) for offset in offsets)
    ):
        message = f"speaker_embedding has invalid data offsets: {path}"
        raise ValueError(message)
    start, end = offsets
    expected_bytes = int(np.prod(EXPECTED_EMBEDDING_SHAPE)) * np.dtype("<f4").itemsize
    if start < 0 or end - start != expected_bytes:
        message = f"speaker_embedding has invalid payload size: {path}"
        raise ValueError(message)
    with path.open("rb") as source:
        source.seek(data_start + start)
        payload = source.read(end - start)
    if len(payload) != expected_bytes:
        message = f"speaker_embedding payload is truncated: {path}"
        raise ValueError(message)
    tensor = np.frombuffer(payload, dtype="<f4").reshape(EXPECTED_EMBEDDING_SHAPE)
    if not np.isfinite(tensor).all():
        message = f"speaker_embedding must contain only finite values: {path}"
        raise ValueError(message)
    return sha256_file(path)


def _read_safetensors_header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as source:
        raw_length = source.read(SAFETENSORS_HEADER_LENGTH_BYTES)
        if len(raw_length) != SAFETENSORS_HEADER_LENGTH_BYTES:
            message = f"invalid safetensors header length: {path}"
            raise ValueError(message)
        header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
        if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
            message = f"invalid safetensors header size: {path}"
            raise ValueError(message)
        raw_header = source.read(header_length)
    if len(raw_header) != header_length:
        message = f"truncated safetensors header: {path}"
        raise ValueError(message)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"invalid safetensors JSON header: {path}"
        raise ValueError(message) from exc
    if not isinstance(header, dict):
        message = f"safetensors header must be an object: {path}"
        raise TypeError(message)
    return header, SAFETENSORS_HEADER_LENGTH_BYTES + header_length


def _manifest_payload(
    *,
    job: Job,
    candidates: Sequence[Candidate],
    run_id: str,
    reference: ReferenceManifest,
    config: BuildConfig,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "models": [
            {
                "model_id": job.model_id,
                "checkpoints": [
                    candidate.to_dict()
                    | {
                        "training_config_sha256": sha256_file(job.config),
                        "base_checkpoint": config.base_checkpoint,
                        "base_checkpoint_sha256": config.base_checkpoint_sha256,
                        "base_revision": config.base_revision,
                        "run_id": run_id,
                    }
                    for candidate in candidates
                ],
            },
        ],
        "text_ids": list(TEXT_IDS),
        "seeds": list(SEEDS),
        "styles": list(STYLES),
        "metrics_provenance": {
            "reference_wavs_sha256": reference.sha256,
            "speaker_embedding": config.speaker_embedding.to_dict(),
            "transcription": config.transcription.to_dict(),
        },
    }


def _publish(
    output_dir: Path,
    *,
    manifests: Mapping[str, Mapping[str, object]],
    index: dict[str, object],
) -> None:
    index_entries = index["manifests"]
    if not isinstance(index_entries, list):
        message = "index manifests must be a list"
        raise TypeError(message)
    indexed_entries = {
        str(entry["manifest_path"]): entry for entry in index_entries if isinstance(entry, dict)
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        ),
    )
    try:
        for relative, payload in manifests.items():
            destination = temporary / relative
            _write_json(destination, payload)
            indexed_entries[relative]["manifest_sha256"] = sha256_file(destination)
        _write_json(temporary / "manifest-index.json", index)
        temporary.rename(output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"invalid JSON: {path}"
        raise ValueError(message) from exc
    if not isinstance(payload, dict):
        message = f"JSON document must be an object: {path}"
        raise TypeError(message)
    return payload


def _resolve_input(
    row: Mapping[str, object],
    field: str,
    *,
    base: Path,
    source: str,
) -> Path:
    return _resolve_path(row, field, base=base, source=source).resolve()


def _resolve_path(
    row: Mapping[str, object],
    field: str,
    *,
    base: Path,
    source: str,
) -> Path:
    raw_path = _required_string(row, field, source=source)
    path = Path(raw_path)
    return path if path.is_absolute() else base / path


def _required_string(row: Mapping[str, Any], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        message = f"{source} requires nonempty string {field}"
        raise ValueError(message)
    return value


def _required_sha256(row: Mapping[str, Any], field: str, *, source: str) -> str:
    value = _required_string(row, field, source=source)
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        message = f"{source} requires lowercase SHA-256 {field}"
        raise ValueError(message)
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--training-jobs", type=Path, required=True)
    parser.add_argument("--reference-wavs", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--speaker-embedding-model-id", required=True)
    parser.add_argument("--speaker-embedding-revision", required=True)
    parser.add_argument("--speaker-embedding-source-sha256", required=True)
    parser.add_argument("--transcription-model-id", required=True)
    parser.add_argument("--transcription-revision", required=True)
    parser.add_argument("--transcription-source-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = BuildConfig(
        base_checkpoint=_required_string(
            {"value": args.base_checkpoint},
            "value",
            source="CLI base checkpoint",
        ),
        base_checkpoint_sha256=_required_sha256(
            {"value": args.base_checkpoint_sha256},
            "value",
            source="CLI base checkpoint",
        ),
        base_revision=_required_string(
            {"value": args.base_revision},
            "value",
            source="CLI base revision",
        ),
        speaker_embedding=MetricModel(
            model_id=_required_string(
                {"value": args.speaker_embedding_model_id},
                "value",
                source="CLI speaker embedding model",
            ),
            revision=_required_string(
                {"value": args.speaker_embedding_revision},
                "value",
                source="CLI speaker embedding revision",
            ),
            source_sha256=_required_sha256(
                {"value": args.speaker_embedding_source_sha256},
                "value",
                source="CLI speaker embedding model",
            ),
        ),
        transcription=MetricModel(
            model_id=_required_string(
                {"value": args.transcription_model_id},
                "value",
                source="CLI transcription model",
            ),
            revision=_required_string(
                {"value": args.transcription_revision},
                "value",
                source="CLI transcription revision",
            ),
            source_sha256=_required_sha256(
                {"value": args.transcription_source_sha256},
                "value",
                source="CLI transcription model",
            ),
        ),
    )
    build_manifests(
        training_status=args.training_status,
        training_jobs=args.training_jobs,
        reference_wavs=args.reference_wavs,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"evaluation manifests written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
