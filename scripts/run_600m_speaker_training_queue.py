from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess  # noqa: S404 - this script intentionally runs an operator-owned command queue.
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

Runner = Callable[[tuple[str, ...], Path], int]
Clock = Callable[[], str]

_EXPECTED_JOB_COUNT = 12
_EXPECTED_EMBEDDING_SHAPE = (16, 768)
_SAFETENSORS_HEADER_LENGTH_BYTES = 8
_SAFETENSORS_OFFSET_COUNT = 2
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_CHECKPOINT_STEP_PATTERN = re.compile(r"checkpoint[-_](\d+)")


@dataclass(frozen=True, slots=True)
class TrainingJob:
    model_id: str
    clean_manifest: Path
    config: Path
    output_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueueProvenance:
    checkpoint: Path
    checkpoint_revision: str
    upstream_commit: str


@dataclass(frozen=True, slots=True)
class ValidatedEmbedding:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class QueueResult:
    planned: tuple[str, ...]
    succeeded: tuple[str, ...]
    failed: tuple[str, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _JobProvenance:
    clean_manifest_sha256: str
    checkpoint_sha256: str
    checkpoint_revision: str
    config_sha256: str
    upstream_commit: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_jobs(
    path: Path,
    *,
    expected_count: int = _EXPECTED_JOB_COUNT,
) -> tuple[TrainingJob, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw_jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(raw_jobs, list) or len(raw_jobs) != expected_count:
        message = f"training job manifest must contain exactly {expected_count} jobs"
        raise ValueError(message)
    jobs = tuple(_parse_job(row, base_dir=path.parent) for row in raw_jobs)
    model_ids = [job.model_id for job in jobs]
    if len(set(model_ids)) != len(model_ids):
        message = "training job manifest contains duplicate model ids"
        raise ValueError(message)
    return jobs


def validate_speaker_embedding(path: Path) -> ValidatedEmbedding:
    header, data_start = _read_safetensors_header(path)
    raw_tensor = header.get("speaker_embedding")
    if not isinstance(raw_tensor, dict):
        message = f"speaker_embedding tensor is missing: {path}"
        raise TypeError(message)
    dtype = raw_tensor.get("dtype")
    if dtype != "F32":
        message = f"speaker_embedding must be float32, got {dtype!r}: {path}"
        raise ValueError(message)
    shape = raw_tensor.get("shape")
    if shape != list(_EXPECTED_EMBEDDING_SHAPE):
        message = (
            f"speaker_embedding shape must be {_EXPECTED_EMBEDDING_SHAPE}, got {shape!r}: {path}"
        )
        raise ValueError(message)
    offsets = raw_tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != _SAFETENSORS_OFFSET_COUNT
        or not all(isinstance(offset, int) for offset in offsets)
    ):
        message = f"speaker_embedding has invalid data offsets: {path}"
        raise ValueError(message)
    start, end = offsets
    expected_bytes = int(np.prod(_EXPECTED_EMBEDDING_SHAPE)) * np.dtype("<f4").itemsize
    if start < 0 or end - start != expected_bytes:
        message = f"speaker_embedding has invalid payload size: {path}"
        raise ValueError(message)
    with path.open("rb") as file:
        file.seek(data_start + start)
        payload = file.read(end - start)
    if len(payload) != expected_bytes:
        message = f"speaker_embedding payload is truncated: {path}"
        raise ValueError(message)
    tensor = np.frombuffer(payload, dtype="<f4").reshape(_EXPECTED_EMBEDDING_SHAPE)
    if not np.isfinite(tensor).all():
        message = f"speaker_embedding must contain only finite values: {path}"
        raise ValueError(message)
    return ValidatedEmbedding(path=path, sha256=sha256_file(path))


def run_training_queue(
    jobs: Sequence[TrainingJob],
    *,
    provenance: QueueProvenance,
    status_path: Path,
    runner: Runner | None = None,
    now: Clock | None = None,
    dry_run: bool = False,
) -> QueueResult:
    execute = runner or _run_subprocess
    clock = now or _utc_now
    checkpoint_sha256 = sha256_file(provenance.checkpoint)
    successful_rows = _successful_rows(status_path)
    planned: list[str] = []
    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for job in jobs:
        job_provenance = _build_job_provenance(job, provenance, checkpoint_sha256)
        if _has_reusable_success(successful_rows.get(job.model_id, ()), job_provenance):
            skipped.append(job.model_id)
            continue
        planned.append(job.model_id)
        if dry_run:
            continue
        started_at = clock()
        log_path = status_path.parent / "logs" / f"{job.model_id}.log"
        base_row = _base_status_row(job, job_provenance, started_at, log_path)
        _append_status(status_path, base_row | {"event": "started", "status": "running"})
        exit_code, execution_error = _execute_job(execute, job, log_path)
        completed_row = _complete_status_row(
            job,
            base_row,
            ended_at=clock(),
            exit_code=exit_code,
            execution_error=execution_error,
        )
        _append_status(status_path, completed_row)
        if completed_row["status"] == "success":
            succeeded.append(job.model_id)
        else:
            failed.append(job.model_id)

    return QueueResult(
        planned=tuple(planned),
        succeeded=tuple(succeeded),
        failed=tuple(failed),
        skipped=tuple(skipped),
    )


def _parse_job(raw_job: object, *, base_dir: Path) -> TrainingJob:
    if not isinstance(raw_job, dict):
        message = "each training job must be a JSON object"
        raise TypeError(message)
    model_id = _required_string(raw_job, "model_id")
    raw_command = raw_job.get("command")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or not all(isinstance(part, str) and part for part in raw_command)
    ):
        message = f"training job {model_id!r} command must be a nonempty string list"
        raise ValueError(message)
    return TrainingJob(
        model_id=model_id,
        clean_manifest=base_dir / _required_string(raw_job, "clean_manifest"),
        config=base_dir / _required_string(raw_job, "config"),
        output_dir=base_dir / _required_string(raw_job, "output_dir"),
        command=tuple(raw_command),
    )


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        message = f"training job field {key!r} must be a nonempty string"
        raise ValueError(message)
    return value


def _read_safetensors_header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as file:
        raw_length = file.read(_SAFETENSORS_HEADER_LENGTH_BYTES)
        if len(raw_length) != _SAFETENSORS_HEADER_LENGTH_BYTES:
            message = f"invalid safetensors header length: {path}"
            raise ValueError(message)
        header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
        if header_length <= 0 or header_length > _MAX_SAFETENSORS_HEADER_BYTES:
            message = f"invalid safetensors header size {header_length}: {path}"
            raise ValueError(message)
        raw_header = file.read(header_length)
    if len(raw_header) != header_length:
        message = f"truncated safetensors header: {path}"
        raise ValueError(message)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"invalid safetensors JSON header: {path}"
        raise ValueError(message) from exc
    if not isinstance(header, dict):
        message = f"safetensors header must be a JSON object: {path}"
        raise TypeError(message)
    return header, _SAFETENSORS_HEADER_LENGTH_BYTES + header_length


def _build_job_provenance(
    job: TrainingJob,
    provenance: QueueProvenance,
    checkpoint_sha256: str,
) -> _JobProvenance:
    return _JobProvenance(
        clean_manifest_sha256=sha256_file(job.clean_manifest),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_revision=provenance.checkpoint_revision,
        config_sha256=sha256_file(job.config),
        upstream_commit=provenance.upstream_commit,
    )


def _successful_rows(status_path: Path) -> dict[str, tuple[Mapping[str, object], ...]]:
    if not status_path.exists():
        return {}
    rows: dict[str, list[Mapping[str, object]]] = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or row.get("event") != "finished"
            or row.get("status") != "success"
        ):
            continue
        model_id = row.get("model_id")
        if isinstance(model_id, str):
            rows.setdefault(model_id, []).append(row)
    return {model_id: tuple(model_rows) for model_id, model_rows in rows.items()}


def _has_reusable_success(
    rows: Sequence[Mapping[str, object]],
    provenance: _JobProvenance,
) -> bool:
    expected = {
        "clean_manifest_sha256": provenance.clean_manifest_sha256,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "checkpoint_revision": provenance.checkpoint_revision,
        "config_sha256": provenance.config_sha256,
        "upstream_commit": provenance.upstream_commit,
    }
    for row in reversed(rows):
        if not all(row.get(key) == value for key, value in expected.items()):
            continue
        if _all_recorded_checkpoints_valid(row):
            return True
    return False


def _all_recorded_checkpoints_valid(row: Mapping[str, object]) -> bool:
    raw_candidates = row.get("candidate_checkpoints")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return False
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            return False
        path = raw_candidate.get("path")
        expected_sha256 = raw_candidate.get("sha256")
        if not isinstance(path, str) or not isinstance(expected_sha256, str):
            return False
        try:
            validated = validate_speaker_embedding(Path(path))
        except (OSError, TypeError, ValueError):
            return False
        if validated.sha256 != expected_sha256:
            return False
    last_candidate = raw_candidates[-1]
    return last_candidate.get("path") == row.get("last_checkpoint") and last_candidate.get(
        "sha256"
    ) == row.get("last_checkpoint_sha256")


def _base_status_row(
    job: TrainingJob,
    provenance: _JobProvenance,
    started_at: str,
    log_path: Path,
) -> dict[str, object]:
    return {
        "model_id": job.model_id,
        "clean_manifest_sha256": provenance.clean_manifest_sha256,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "checkpoint_revision": provenance.checkpoint_revision,
        "config_sha256": provenance.config_sha256,
        "upstream_commit": provenance.upstream_commit,
        "started_at": started_at,
        "ended_at": None,
        "exit_code": None,
        "log_path": str(log_path),
        "last_checkpoint": None,
        "last_checkpoint_sha256": None,
        "candidate_checkpoints": [],
        "error": None,
    }


def _execute_job(
    execute: Runner,
    job: TrainingJob,
    log_path: Path,
) -> tuple[int | None, str | None]:
    try:
        return execute(job.command, log_path), None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _complete_status_row(
    job: TrainingJob,
    base_row: Mapping[str, object],
    *,
    ended_at: str,
    exit_code: int | None,
    execution_error: str | None,
) -> dict[str, object]:
    if exit_code != 0:
        error = execution_error or f"training process exited with code {exit_code}"
        return dict(base_row) | {
            "event": "finished",
            "status": "failed",
            "ended_at": ended_at,
            "exit_code": exit_code,
            "error": error,
        }
    try:
        validated = _validate_candidate_checkpoints(job.output_dir)
    except (OSError, TypeError, ValueError) as exc:
        return dict(base_row) | {
            "event": "finished",
            "status": "failed",
            "ended_at": ended_at,
            "exit_code": exit_code,
            "error": f"{type(exc).__name__}: {exc}",
        }
    last_checkpoint = validated[-1]
    return dict(base_row) | {
        "event": "finished",
        "status": "success",
        "ended_at": ended_at,
        "exit_code": exit_code,
        "last_checkpoint": str(last_checkpoint.path),
        "last_checkpoint_sha256": last_checkpoint.sha256,
        "candidate_checkpoints": [
            {"path": str(candidate.path), "sha256": candidate.sha256} for candidate in validated
        ],
    }


def _validate_candidate_checkpoints(output_dir: Path) -> tuple[ValidatedEmbedding, ...]:
    validated = tuple(
        validate_speaker_embedding(path) for path in _candidate_checkpoints(output_dir)
    )
    if not validated:
        message = f"no .speaker.safetensors checkpoints found in {output_dir}"
        raise ValueError(message)
    return validated


def _candidate_checkpoints(output_dir: Path) -> tuple[Path, ...]:
    paths = tuple(output_dir.rglob("*.speaker.safetensors"))
    return tuple(sorted(paths, key=_checkpoint_sort_key))


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    matches = tuple(_CHECKPOINT_STEP_PATTERN.finditer(str(path)))
    step = int(matches[-1].group(1)) if matches else -1
    return step, path.stat().st_mtime_ns, str(path)


def _append_status(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _run_subprocess(command: tuple[str, ...], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        completed = subprocess.run(  # noqa: S603 - command is explicit operator-owned manifest input.
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - remote trainer uses Python 3.10


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_training_queue(
        load_training_jobs(args.jobs_json),
        provenance=QueueProvenance(
            checkpoint=args.checkpoint,
            checkpoint_revision=args.checkpoint_revision,
            upstream_commit=args.upstream_commit,
        ),
        status_path=args.status_path,
        dry_run=args.dry_run,
    )
    print(json.dumps({field: getattr(result, field) for field in result.__dataclass_fields__}))
    return int(bool(result.failed))


if __name__ == "__main__":
    raise SystemExit(main())
