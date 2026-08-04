# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0914, PLR0915, PLR0916, TRY003
# Validation errors deliberately retain path/model context; orchestration mirrors fixed contracts.
"""Read-only completion verifier for the 600M Speaker Inversion workflow."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import subprocess  # noqa: S404 - probes are fixed, read-only inventory commands.
import sys
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_UTC = timezone.utc  # noqa: UP017 - pinned Windows runtime uses Python 3.10.

EXPECTED_MODEL_COUNT = 12
EXPECTED_NEW_STATUS_ROW_COUNT = 20
EXPECTED_QUALITY_STATUS_ROW_COUNT = 2
EXPECTED_CHECKPOINT_COUNT = 13
EXPECTED_EVALUATION_CASE_COUNT = 140
MIN_COMPONENT_COMMAND_PARTS = 2
SAFETENSORS_HEADER_BYTES = 8
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
PERIODIC_STEPS = tuple(range(250, 3001, 250))
LOGGED_STEPS = tuple(range(20, 3001, 20))
EXPECTED_EMBEDDING_SHAPE = (16, 768)
EXPECTED_CHECKPOINT_NAMES = frozenset(
    {f"checkpoint_{step:07d}.speaker.safetensors" for step in PERIODIC_STEPS}
    | {"checkpoint_final.speaker.safetensors"}
)
EXPECTED_EVALUATION_STAGES_PER_MODEL = (
    "generation",
    "analysis",
    "metrics",
    "evaluate",
)
EXPECTED_EVALUATION_STAGE_COUNT = 49
EXPECTED_EVALUATION_STEPS = (1000, 1500, 2000, 2500, 3000)
EXPECTED_TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
EXPECTED_SEEDS = (1234, 5678)
EXPECTED_STYLES = ("neutral", "calm")
EXPECTED_HARD_GATE_TEXT_IDS = frozenset(
    {"sentence_unko", "sentence_chinko", "sentence_manko", "control"}
)
EXPECTED_HARD_GATE_METRIC_CASE_COUNT_PER_CHECKPOINT = 16
EXPECTED_DIAGNOSTIC_WORD_CASE_COUNT_PER_CHECKPOINT = 12
MAX_GPU_UTILIZATION_PERCENT = 100.0
ZIP_CREATE_SYSTEM_UNIX = 3
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_UNIX_MODE = 0o100644
ZIP_COMPRESSLEVEL = 9

COMPLETION_SCHEMA = "600m-speaker-retraining-completion-verification/v1"
QUALITY_RUN_EVIDENCE_SCHEMA = "speaker-quality-retrain-run-evidence/v1"
QUALITY_SETUP_EVIDENCE_SCHEMA = "speaker-quality-retrain-setup/v1"
EVALUATION_CONFIG_SCHEMA = "speaker-evaluation-queue/v1"
EVALUATION_STATUS_SCHEMA = "speaker-evaluation-queue-status/v1"
EVALUATION_VERIFICATION_SCHEMA = "speaker-checkpoint-evaluation-verification/v2"
RUNTIME_SNAPSHOT_SCHEMA = "speaker-evaluation-runtime-inputs/v1"
RUNTIME_CONFIG_NAME = "evaluation-queue-runtime.json"
RUNTIME_JOBS_NAME = "training-jobs-speed-v1.json"
RUNTIME_STATUS_NAME = "training-status.jsonl"
RUNTIME_MANIFEST_NAME = "snapshot-manifest.json"
UPSTREAM_RUNTIME_PROVENANCE_NAME = "upstream-runtime-provenance.json"
UPSTREAM_RUNTIME_PACKAGE_NAME = "upstream-runtime-package.zip"
UPSTREAM_RUNTIME_PROVENANCE_SCHEMA = "irodori-upstream-runtime-provenance/v1"
PINNED_UPSTREAM_COMMIT = "eaf74d6a19138f743acb5b71a445fd25a57db987"
GIT_HEAD_TREE = "HEAD^{tree}"
RUNTIME_COMPONENT_NAMES = (
    "run_600m_speaker_evaluation_queue.py",
    "build_600m_checkpoint_evaluation_manifests.py",
    "generate_600m_checkpoint_audio_remote.py",
    "analyze_nko_beep_matrix.py",
    "compute_600m_speaker_metrics.py",
    "evaluate_600m_speaker_checkpoints.py",
)
EXPECTED_RUNTIME_SNAPSHOT_FILES = frozenset(
    {
        RUNTIME_CONFIG_NAME,
        RUNTIME_JOBS_NAME,
        RUNTIME_STATUS_NAME,
        UPSTREAM_RUNTIME_PROVENANCE_NAME,
        UPSTREAM_RUNTIME_PACKAGE_NAME,
        *(f"scripts/{name}" for name in RUNTIME_COMPONENT_NAMES),
    }
)
STAGE_COMPONENT_NAMES = {
    "manifests": "build_600m_checkpoint_evaluation_manifests.py",
    "generation": "generate_600m_checkpoint_audio_remote.py",
    "analysis": "analyze_nko_beep_matrix.py",
    "metrics": "compute_600m_speaker_metrics.py",
    "evaluate": "evaluate_600m_speaker_checkpoints.py",
}
REVIEW_DECISION_SCHEMA = "speaker-checkpoint-review-decision/v1"
REVIEW_PACKET_SCHEMA = "speaker-checkpoint-review-packet/v1"
STAGING_SCHEMA = "speaker-model-staging-report/v1"
VOICE_BANK_SNAPSHOT_SCHEMA = "voice-bank-snapshot/v1"
REVIEW_DECISIONS = frozenset({"VOICE", "TONE", "UNSURE"})
COMPLETION_STATUSES = frozenset({"PASS", "AWAITING_REVIEW", "FAIL"})
REQUIRED_NON_DEPLOYMENT_VALUES = {
    "deployment_performed": False,
    "active_voice_bank_unchanged": True,
    "proposed_staging_root_created": False,
}

_ANSI_RE = re.compile(r"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\))")
_STEP_RE = re.compile(r"\bstep\s*(?:=|:|\s)\s*(\d+)\b", re.IGNORECASE)
_LOSS_RE = re.compile(
    r"\bloss\s*(?:=|:)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:inf|nan))\b",
    re.IGNORECASE,
)
_FINISHED_RE = re.compile(r"Training finished at step\s*=\s*3000\.", re.IGNORECASE)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONFLICTING_COMMAND_MARKERS = (
    "run_600m_speaker_training_queue.py",
    "launch_600m_training_queue_runtime.py",
    "launch_600m_training_queue_speed_v1.py",
    "run_600m_speaker_evaluation_queue.py",
    "launch_600m_speaker_evaluation_queue",
    "generate_600m_checkpoint_audio_remote.py",
    "compute_600m_speaker_metrics.py",
    "evaluate_600m_speaker_checkpoints.py",
    "build_600m_checkpoint_evaluation_manifests.py",
    "analyze_nko_beep_matrix.py",
    "train.py",
    "--multiprocessing-fork",
    "spawn_main(",
)
_RELATED_COMPUTE_EXECUTABLES = frozenset({"python.exe", "pythonw.exe"})


@dataclass(frozen=True, slots=True)
class ValidatedEmbedding:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class FileBinding:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class TrainingLogSummary:
    loss_event_count: int
    first_step: int
    last_step: int
    minimum_loss: float
    maximum_loss: float
    last_loss: float


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    conflicting_processes: tuple[dict[str, object], ...]
    related_compute_applications: tuple[dict[str, object], ...]
    gpu_memory_used_mib: float | None
    errors: tuple[str, ...]
    observed_at: str

    @classmethod
    def idle(
        cls,
        *,
        used_mib: float,
        observed_at: str = "1970-01-01T00:00:00+00:00",
    ) -> RuntimeSnapshot:
        return cls((), (), float(used_mib), (), observed_at)


@dataclass(frozen=True, slots=True)
class QualityRunLineage:
    model_id: str
    evidence: FileBinding
    setup_evidence: FileBinding
    training_jobs: FileBinding
    training_status: FileBinding
    queue_script: FileBinding
    source_diagnostic: FileBinding
    initialization_checkpoint: FileBinding


@dataclass(frozen=True, slots=True)
class TrainingModelSummary:
    model_id: str
    checkpoint_count: int
    loss_event_count: int
    config_sha256: str
    clean_manifest_sha256: str
    log_sha256: str
    output_dir: Path
    checkpoints: tuple[ValidatedEmbedding, ...]
    latest_status: Mapping[str, object]
    run_id: str
    config_path: Path
    clean_manifest_path: Path
    log_path: Path
    run_evidence_lineage: tuple[QualityRunLineage, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingVerification:
    models: tuple[TrainingModelSummary, ...]
    training_jobs: Path
    training_jobs_sha256: str
    training_status: Path
    training_status_sha256: str
    training_launch_evidence: Path
    training_launch_evidence_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    checkpoint_revision: str
    upstream_commit: str
    runtime_snapshot: RuntimeSnapshot
    base_training_jobs: FileBinding | None = None
    base_training_status: FileBinding | None = None
    training_run_evidence: tuple[QualityRunLineage, ...] = ()

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.models)


@dataclass(frozen=True, slots=True)
class EvaluationModelSummary:
    model_id: str
    evaluation_dir: Path
    manifest_path: Path
    case_count: int
    selected: Mapping[str, object]
    review_candidates: tuple[Mapping[str, object], ...]
    manifest_sha256: str | None = None
    evaluation_verification: FileBinding | None = None
    evaluation_results: FileBinding | None = None
    review_candidates_file: FileBinding | None = None
    review_packet_manifest: FileBinding | None = None
    review_packet_assets: tuple[FileBinding, ...] = ()
    selected_file: FileBinding | None = None


@dataclass(frozen=True, slots=True)
class EvaluationVerification:
    stage_count: int
    models: tuple[EvaluationModelSummary, ...]
    evaluation_config: Path
    evaluation_config_sha256: str
    evaluation_status: Path
    evaluation_status_sha256: str
    runtime_snapshot_manifest: FileBinding | None = None
    runtime_snapshot_files: tuple[FileBinding, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationStageContract:
    stage: str
    model_id: str | None
    component_path: Path | None
    command: tuple[str, ...]
    collision_paths: tuple[Path, ...]
    input_files: tuple[Path, ...]
    output_roots: tuple[Path, ...]
    required_outputs: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ReviewVerification:
    status: str
    candidate_count: int
    decision_count: int
    unresolved_ids: tuple[str, ...]
    grouped_decisions: Mapping[str, Mapping[str, Mapping[str, int]]]
    decisions_path: Path
    decisions_sha256: str


@dataclass(frozen=True, slots=True)
class StagingVerification:
    model_count: int
    selections: tuple[Mapping[str, object], ...]
    staging_report: Path
    staging_report_sha256: str
    proposed_staging_root: Path
    active_voice_bank_baseline: Path
    active_voice_bank_baseline_sha256: str
    active_voice_bank_current: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StagingOutputPreflight:
    staging_report_lexical: Path
    staging_report: FileBinding
    proposed_staging_root: Path


def sha256_file(path: Path) -> str:
    _require_regular_file(path, source="SHA-256 input")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_speaker_embedding(path: Path) -> ValidatedEmbedding:
    lexical = _require_regular_file(path, source="speaker checkpoint")
    resolved = lexical.resolve(strict=True)
    with resolved.open("rb") as source:
        contents = source.read()
    raw_length = contents[:SAFETENSORS_HEADER_BYTES]
    if len(raw_length) != SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"invalid safetensors header length: {resolved}")
    header_length = struct.unpack("<Q", raw_length)[0]
    if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"invalid safetensors header size: {resolved}")
    header_end = SAFETENSORS_HEADER_BYTES + header_length
    raw_header = contents[SAFETENSORS_HEADER_BYTES:header_end]
    if len(raw_header) != header_length:
        raise ValueError(f"truncated safetensors header: {resolved}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {resolved}") from exc
    if not isinstance(header, dict):
        raise TypeError(f"safetensors header must be an object: {resolved}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if set(tensors) != {"speaker_embedding"}:
        raise ValueError(f"safetensors must contain only speaker_embedding: {resolved}")
    tensor = tensors["speaker_embedding"]
    if not isinstance(tensor, dict):
        raise TypeError(f"speaker_embedding metadata must be an object: {resolved}")
    expected_size = int(np.prod(EXPECTED_EMBEDDING_SHAPE)) * 4
    if tensor.get("dtype") != "F32" or tensor.get("shape") != [16, 768]:
        raise ValueError(f"speaker_embedding dtype or shape mismatch: {resolved}")
    offsets = tensor.get("data_offsets")
    if offsets != [0, expected_size]:
        raise ValueError(f"speaker_embedding offsets or payload size mismatch: {resolved}")
    payload = contents[header_end:]
    if len(payload) != expected_size:
        raise ValueError(f"speaker_embedding file size mismatch: {resolved}")
    if not np.isfinite(np.frombuffer(payload, dtype="<f4")).all():
        raise ValueError(f"speaker_embedding contains nonfinite values: {resolved}")
    return ValidatedEmbedding(path=resolved, sha256=hashlib.sha256(contents).hexdigest())


def parse_final_training_run(log_text: str) -> TrainingLogSummary:
    clean = _ANSI_RE.sub("", log_text)
    lines = clean.splitlines()
    events: list[tuple[int, int, float]] = []
    terminals: list[int] = []
    for index, line in enumerate(lines):
        if _FINISHED_RE.search(line):
            terminals.append(index)
        step_match = _STEP_RE.search(line)
        loss_match = _LOSS_RE.search(line)
        if step_match is not None and loss_match is not None:
            loss = float(loss_match.group(1))
            events.append((index, int(step_match.group(1)), loss))
    starts = [index for index, step, _loss in events if step == LOGGED_STEPS[0]]
    if not starts:
        raise ValueError("final training run has no step 20 loss event")
    start = starts[-1]
    terminal = next((index for index in terminals if index > start), None)
    if terminal is None:
        raise ValueError("final training run is missing its terminal marker")
    if any(index > terminal for index, _step, _loss in events):
        raise ValueError("an incomplete run follows the last completed training run")
    suffix = [(step, loss) for index, step, loss in events if start <= index < terminal]
    if tuple(step for step, _loss in suffix) != LOGGED_STEPS:
        raise ValueError("final training run has an incomplete optimizer step sequence")
    losses = tuple(loss for _step, loss in suffix)
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("final training run contains nonfinite loss")
    return TrainingLogSummary(
        loss_event_count=len(losses),
        first_step=LOGGED_STEPS[0],
        last_step=LOGGED_STEPS[-1],
        minimum_loss=min(losses),
        maximum_loss=max(losses),
        last_loss=losses[-1],
    )


def normalize_runtime_snapshot(
    *,
    processes: Iterable[Mapping[str, object]],
    compute_applications: Iterable[Mapping[str, object]],
    gpu_memory_used_mib: float | None,
    errors: Iterable[str],
    excluded_pids: set[int] | frozenset[int] = frozenset(),
    observed_at: str | None = None,
) -> RuntimeSnapshot:
    process_rows = [dict(row) for row in processes]
    excluded = set(excluded_pids)
    conflicts = tuple(
        row
        for row in process_rows
        if _process_int(row, "pid", "ProcessId") not in excluded
        and _is_conflicting_command(_command_line(row))
    )
    conflict_pids = {
        pid for row in conflicts if (pid := _process_int(row, "pid", "ProcessId")) is not None
    }
    related_compute = tuple(
        dict(row)
        for row in compute_applications
        if _process_int(row, "pid", "ProcessId") not in excluded
        and _compute_application_related(row, conflict_pids=conflict_pids)
    )
    raw_errors = tuple(errors)
    if not all(isinstance(error, str) and error for error in raw_errors):
        raise TypeError("runtime probe errors must be nonempty strings")
    if gpu_memory_used_mib is not None and (
        isinstance(gpu_memory_used_mib, bool)
        or not isinstance(gpu_memory_used_mib, int | float)
        or not math.isfinite(float(gpu_memory_used_mib))
        or float(gpu_memory_used_mib) < 0
    ):
        raise ValueError("runtime GPU memory must be a finite nonnegative number")
    return RuntimeSnapshot(
        conflicting_processes=conflicts,
        related_compute_applications=related_compute,
        gpu_memory_used_mib=(
            float(gpu_memory_used_mib) if gpu_memory_used_mib is not None else None
        ),
        errors=raw_errors,
        observed_at=observed_at or datetime.now(_UTC).isoformat(),
    )


def _process_int(row: Mapping[str, object], *fields: str) -> int | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _command_line(row: Mapping[str, object]) -> str:
    for field in ("command_line", "CommandLine", "command", "process_name", "name", "Name"):
        value = row.get(field)
        if isinstance(value, str):
            return value
    return ""


def _is_conflicting_command(command: str) -> bool:
    normalized = command.casefold()
    return any(marker in normalized for marker in _CONFLICTING_COMMAND_MARKERS)


def _compute_application_related(
    row: Mapping[str, object],
    *,
    conflict_pids: set[int],
) -> bool:
    pid = _process_int(row, "pid", "ProcessId")
    if pid in conflict_pids:
        return True
    process_name = row.get("process_name", row.get("ProcessName"))
    if not isinstance(process_name, str):
        return False
    basename = re.split(r"[\\/]", process_name.strip().strip('"'))[-1].casefold()
    return basename in _RELATED_COMPUTE_EXECUTABLES


def _resolve_training_run_id(row: Mapping[str, object], *, model_id: str) -> str:
    if "seeded_existing_run" not in row:
        return _canonical_sha256(row)
    seeded_existing_run = row["seeded_existing_run"]
    if not isinstance(seeded_existing_run, dict):
        raise TypeError(f"seeded existing run must be an object for {model_id}")
    source = f"seeded existing run for {model_id}"
    raw_path = _required_string(
        seeded_existing_run,
        "run_provenance_path",
        source=source,
    )
    run_provenance_path = Path(raw_path)
    if not run_provenance_path.is_absolute():
        raise ValueError(f"run provenance path must be absolute for {model_id}")
    try:
        resolved_run_provenance_path = _require_regular_file(
            run_provenance_path,
            source=f"run provenance for {model_id}",
        ).resolve(strict=True)
    except ValueError as exc:
        raise ValueError(f"run provenance path is unsafe or missing for {model_id}") from exc
    if (
        str(resolved_run_provenance_path) != raw_path
        or run_provenance_path.is_symlink()
        or not resolved_run_provenance_path.is_file()
    ):
        raise ValueError(f"run provenance path is unsafe or missing for {model_id}")
    declared_sha256 = _required_sha256(
        seeded_existing_run,
        "run_provenance_sha256",
        source=source,
    )
    run_provenance_bytes = resolved_run_provenance_path.read_bytes()
    if hashlib.sha256(run_provenance_bytes).hexdigest() != declared_sha256:
        raise ValueError(f"run provenance SHA-256 mismatch for {model_id}")
    try:
        run_provenance = json.loads(run_provenance_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {resolved_run_provenance_path}") from exc
    if not isinstance(run_provenance, dict):
        raise TypeError(f"JSON document must be an object: {resolved_run_provenance_path}")
    provenance_model_id = _required_string(
        run_provenance,
        "model_id",
        source=f"run provenance {resolved_run_provenance_path}",
    )
    if provenance_model_id != model_id:
        raise ValueError(f"run provenance model_id mismatch for {model_id}")
    return declared_sha256


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, source: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{source} field set mismatch")


def _require_regular_file(path: Path, *, source: str) -> Path:
    lexical = _require_no_alias_components(path, source=source)
    if not os.path.lexists(lexical):
        raise ValueError(f"{source} file is missing: {lexical}")
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{source} file metadata is unavailable: {lexical}") from exc
    if lexical.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{source} must be a regular non-symlink file: {lexical}")
    return lexical


def _require_directory(path: Path, *, source: str) -> Path:
    lexical = _require_no_alias_components(path, source=source)
    if not os.path.lexists(lexical):
        raise ValueError(f"{source} directory is missing: {lexical}")
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{source} directory metadata is unavailable: {lexical}") from exc
    if lexical.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{source} must be a non-symlink directory: {lexical}")
    return lexical


def _require_no_alias_components(path: Path, *, source: str) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink() or _is_reparse_alias(current):
            raise ValueError(
                f"{source} must not contain a symlink, junction, reparse, or alias component: "
                f"{current}"
            )
    return lexical


def _is_reparse_alias(path: Path) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not reparse_flag:
        return False
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & reparse_flag)


def _canonical_artifact_path(
    raw: object,
    *,
    source: str,
    directory: bool = False,
) -> Path:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise ValueError(f"{source} path must be absolute")
    path = Path(raw)
    lexical = (
        _require_directory(path, source=source)
        if directory
        else _require_regular_file(path, source=source)
    )
    resolved = lexical.resolve(strict=True)
    if str(resolved) != raw:
        raise ValueError(f"{source} path is noncanonical, symlinked, or missing")
    return resolved


def _validated_file_binding(raw: object, *, source: str) -> FileBinding:
    if not isinstance(raw, dict):
        raise TypeError(f"{source} binding must be an object")
    _require_exact_keys(raw, {"path", "sha256"}, source=f"{source} binding")
    path = _canonical_artifact_path(raw.get("path"), source=source)
    sha256 = _required_sha256(raw, "sha256", source=source)
    if sha256_file(path) != sha256:
        raise ValueError(f"{source} SHA-256 mismatch")
    return FileBinding(path, sha256)


def _require_finite_number(value: object, *, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{source} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source} must be finite")
    return result


def _require_iso_datetime(value: object, *, source: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} must be a nonempty ISO-8601 timestamp")
    return _parse_datetime(value, source=source)


def _normalized_job_command(
    job: Mapping[str, object],
    *,
    config: Path,
    output: Path,
    manifest: Path,
    base_checkpoint: Path,
    source: str,
) -> tuple[str, ...]:
    command = job.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) and part for part in command):
        raise ValueError(f"{source} command must be a nonempty string list")
    normalized = list(command)
    expected_paths = {
        "--config": (config, "<CONFIG>"),
        "--output-dir": (output, "<OUTPUT_DIR>"),
        "--manifest": (manifest, "<MANIFEST>"),
        "--init-checkpoint": (base_checkpoint, "<BASE_CHECKPOINT>"),
    }
    for flag, (expected, placeholder) in expected_paths.items():
        if normalized.count(flag) != 1:
            raise ValueError(f"{source} command must contain exactly one {flag} argument")
        index = normalized.index(flag) + 1
        if index >= len(normalized):
            raise ValueError(f"{source} command is missing the {flag} value")
        configured = _resolve_path(
            normalized[index],
            base=expected.parent,
            source=f"{source} command {flag}",
        )
        if configured != expected:
            raise ValueError(f"{source} command {flag} path mismatch")
        normalized[index] = placeholder
    return tuple(normalized)


def _launch_status_tail(launch: Mapping[str, object]) -> tuple[int, str]:
    after_count = launch.get("status_row_count")
    if isinstance(after_count, bool) or not isinstance(after_count, int) or after_count < 0:
        raise ValueError("training launch evidence status_row_count is invalid")
    after_sha = _required_sha256(
        launch,
        "status_after_sha256",
        source="training launch evidence",
    )
    return after_count, after_sha


def _validate_quality_setup(  # noqa: PLR0913 - mirrors the fixed setup evidence contract.
    setup: Mapping[str, object],
    *,
    model_id: str,
    setup_path: Path,
    jobs_binding: FileBinding,
    queue_binding: FileBinding,
    status_path: Path,
    before_sha: str,
    current_job: Mapping[str, object],
    current_jobs_base: Path,
    predecessor_job: Mapping[str, object],
    predecessor_jobs_base: Path,
) -> tuple[FileBinding, FileBinding, Path, Path, Path]:
    source = f"quality setup evidence {setup_path}"
    _require_exact_keys(
        setup,
        {"schema_version", "created_at", "model_id", "reason", "changes", "paths", "sha256"},
        source=source,
    )
    if setup.get("schema_version") != QUALITY_SETUP_EVIDENCE_SCHEMA:
        raise ValueError(f"{source} schema mismatch")
    _require_iso_datetime(setup.get("created_at"), source=f"{source} created_at")
    if setup.get("model_id") != model_id:
        raise ValueError(f"{source} model_id mismatch")

    reason = _required_mapping(setup, "reason", source=source)
    _require_exact_keys(
        reason,
        {"source_diagnostic", "source_diagnostic_sha256", "source_best", "strategy"},
        source=f"{source} reason",
    )
    diagnostic_path = _canonical_artifact_path(
        reason.get("source_diagnostic"), source=f"{source} source diagnostic"
    )
    diagnostic_sha = _required_sha256(
        reason,
        "source_diagnostic_sha256",
        source=f"{source} reason",
    )
    if sha256_file(diagnostic_path) != diagnostic_sha:
        raise ValueError(f"{source} source diagnostic SHA-256 mismatch")
    strategy = reason.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError(f"{source} strategy must be nonempty")
    source_best = _required_mapping(reason, "source_best", source=f"{source} reason")
    _require_exact_keys(
        source_best,
        {
            "checkpoint_step",
            "hard_gate_pass_count",
            "hard_gate_case_count",
            "failing_case",
            "speaker_similarity",
            "required_minimum",
        },
        source=f"{source} source_best",
    )
    checkpoint_step = _required_int(
        source_best,
        "checkpoint_step",
        source=f"{source} source_best",
    )
    if checkpoint_step not in PERIODIC_STEPS:
        raise ValueError(f"{source} initialization checkpoint step is invalid")
    pass_count = _required_int(
        source_best,
        "hard_gate_pass_count",
        source=f"{source} source_best",
    )
    case_count = _required_int(
        source_best,
        "hard_gate_case_count",
        source=f"{source} source_best",
    )
    failing_case = source_best.get("failing_case")
    similarity = _require_finite_number(
        source_best.get("speaker_similarity"), source=f"{source} speaker_similarity"
    )
    minimum = _require_finite_number(
        source_best.get("required_minimum"), source=f"{source} required_minimum"
    )
    if (
        pass_count < 0
        or case_count <= 0
        or pass_count >= case_count
        or not isinstance(failing_case, str)
        or not failing_case
        or similarity >= minimum
    ):
        raise ValueError(f"{source} source diagnostic reason mismatch")

    paths = _required_mapping(setup, "paths", source=source)
    _require_exact_keys(
        paths,
        {"config", "jobs", "status", "queue_script", "output_dir"},
        source=f"{source} paths",
    )
    config_path = _canonical_artifact_path(paths.get("config"), source=f"{source} config")
    jobs_path = _canonical_artifact_path(paths.get("jobs"), source=f"{source} jobs")
    bound_status = _canonical_artifact_path(paths.get("status"), source=f"{source} status")
    bound_queue = _canonical_artifact_path(
        paths.get("queue_script"), source=f"{source} queue script"
    )
    output_dir = _canonical_artifact_path(
        paths.get("output_dir"), source=f"{source} output_dir", directory=True
    )
    if (
        jobs_path != jobs_binding.path
        or bound_status != status_path
        or bound_queue != queue_binding.path
    ):
        raise ValueError(f"{source} jobs, status, or queue script path mismatch")
    if (
        _job_path(current_job, "config", base=current_jobs_base) != config_path
        or _job_path(current_job, "output_dir", base=current_jobs_base) != output_dir
    ):
        raise ValueError(f"{source} effective job path mismatch")

    hashes = _required_mapping(setup, "sha256", source=source)
    _require_exact_keys(
        hashes,
        {"source_config", "config", "jobs", "status_seed", "queue_script"},
        source=f"{source} sha256",
    )
    predecessor_config = _job_path(predecessor_job, "config", base=predecessor_jobs_base)
    if (
        _required_sha256(hashes, "source_config", source=f"{source} sha256")
        != sha256_file(predecessor_config)
        or _required_sha256(hashes, "config", source=f"{source} sha256") != sha256_file(config_path)
        or _required_sha256(hashes, "jobs", source=f"{source} sha256") != jobs_binding.sha256
        or _required_sha256(hashes, "status_seed", source=f"{source} sha256") != before_sha
        or _required_sha256(hashes, "queue_script", source=f"{source} sha256")
        != queue_binding.sha256
    ):
        raise ValueError(f"{source} SHA-256 lineage mismatch")

    changes = _required_mapping(setup, "changes", source=source)
    _require_exact_keys(
        changes,
        {
            "learning_rate",
            "seed",
            "max_steps",
            "save_every",
            "batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "speaker_inversion_init_embedding",
            "speaker_inversion_init_embedding_sha256",
        },
        source=f"{source} changes",
    )
    learning_rate = _required_mapping(changes, "learning_rate", source=f"{source} changes")
    seed = _required_mapping(changes, "seed", source=f"{source} changes")
    _require_exact_keys(learning_rate, {"from", "to"}, source=f"{source} learning_rate")
    _require_exact_keys(seed, {"from", "to"}, source=f"{source} seed")
    new_learning_rate = _require_finite_number(
        learning_rate.get("to"), source=f"{source} learning_rate.to"
    )
    new_seed = _required_int(seed, "to", source=f"{source} seed")
    previous_seed = _required_int(seed, "from", source=f"{source} seed")
    if new_learning_rate <= 0:
        raise ValueError(f"{source} learning rate must be positive")
    init_path = _canonical_artifact_path(
        changes.get("speaker_inversion_init_embedding"),
        source=f"{source} initialization checkpoint",
    )
    init_sha = _required_sha256(
        changes,
        "speaker_inversion_init_embedding_sha256",
        source=f"{source} changes",
    )
    expected_init = (
        _job_path(predecessor_job, "output_dir", base=predecessor_jobs_base)
        / f"checkpoint_{checkpoint_step:07d}.speaker.safetensors"
    ).resolve()
    if init_path != expected_init or sha256_file(init_path) != init_sha:
        raise ValueError(f"{source} initialization checkpoint lineage mismatch")
    config = _read_json(config_path)
    train = config.get("train", config)
    if not isinstance(train, dict):
        raise TypeError(f"{source} config train must be an object")
    predecessor_config_document = _read_json(predecessor_config)
    predecessor_train = predecessor_config_document.get("train", predecessor_config_document)
    if not isinstance(predecessor_train, dict):
        raise TypeError(f"{source} predecessor config train must be an object")
    previous_learning_rate = _require_finite_number(
        learning_rate.get("from"), source=f"{source} learning_rate.from"
    )
    if (
        predecessor_train.get("learning_rate") != previous_learning_rate
        or predecessor_train.get("seed") != previous_seed
    ):
        raise ValueError(f"{source} predecessor config change binding mismatch")
    max_steps = _required_int(changes, "max_steps", source=f"{source} changes")
    save_every = _required_int(changes, "save_every", source=f"{source} changes")
    batch_size = _required_int(changes, "batch_size", source=f"{source} changes")
    accumulation = _required_int(
        changes,
        "gradient_accumulation_steps",
        source=f"{source} changes",
    )
    gradient_checkpointing = changes.get("gradient_checkpointing")
    if (
        max_steps != LOGGED_STEPS[-1]
        or save_every != PERIODIC_STEPS[0]
        or batch_size <= 0
        or accumulation <= 0
        or type(gradient_checkpointing) is not bool
    ):
        raise ValueError(f"{source} training change contract mismatch")
    expected_config_values = {
        "learning_rate": new_learning_rate,
        "seed": new_seed,
        "max_steps": max_steps,
        "save_every": save_every,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if any(train.get(key) != value for key, value in expected_config_values.items()):
        raise ValueError(f"{source} config change binding mismatch")
    configured_init = _canonical_artifact_path(
        train.get("speaker_inversion_init_embedding"),
        source=f"{source} config initialization checkpoint",
    )
    if configured_init != init_path:
        raise ValueError(f"{source} config initialization checkpoint mismatch")
    expected_config = copy.deepcopy(predecessor_config_document)
    expected_train = expected_config.get("train", expected_config)
    if not isinstance(expected_train, dict):
        raise TypeError(f"{source} expected config train must be an object")
    current_manifest = _job_path(current_job, "clean_manifest", base=current_jobs_base)
    expected_train.update(
        expected_config_values
        | {
            "manifest_path": str(current_manifest),
            "output_dir": str(output_dir),
            "speaker_inversion_init_embedding": str(init_path),
        }
    )
    if _canonical_sha256(config) != _canonical_sha256(expected_config):
        raise ValueError(f"{source} successor config contains undeclared drift")
    return (
        FileBinding(diagnostic_path, diagnostic_sha),
        FileBinding(init_path, init_sha),
        config_path,
        output_dir,
        predecessor_config,
    )


def _validate_quality_run_payload(  # noqa: PLR0913 - mirrors the fixed run evidence contract.
    run: Mapping[str, object],
    runtime_after: Mapping[str, object],
    *,
    model_id: str,
    started: Mapping[str, object],
    finished: Mapping[str, object],
    config: Path,
    manifest: Path,
    output_dir: Path,
    base_sha: str,
    launch_gpu_baseline: float,
    gpu_memory_tolerance_mib: float,
) -> tuple[tuple[ValidatedEmbedding, ...], Path, TrainingLogSummary]:
    source = f"quality run evidence for {model_id}"
    _require_exact_keys(
        run,
        {
            "started_at",
            "ended_at",
            "config_sha256",
            "clean_manifest_sha256",
            "base_checkpoint_sha256",
            "candidate_checkpoint_count",
            "checkpoints",
            "final_equals_step3000",
            "log",
        },
        source=f"{source} run",
    )
    started_at = _require_iso_datetime(run.get("started_at"), source=f"{source} started_at")
    ended_at = _require_iso_datetime(run.get("ended_at"), source=f"{source} ended_at")
    if (
        ended_at < started_at
        or run.get("started_at") != started.get("started_at")
        or run.get("ended_at") != finished.get("ended_at")
    ):
        raise ValueError(f"{source} timestamp mismatch")
    config_sha = sha256_file(config)
    manifest_sha = sha256_file(manifest)
    if (
        run.get("config_sha256") != config_sha
        or run.get("clean_manifest_sha256") != manifest_sha
        or run.get("base_checkpoint_sha256") != base_sha
        or run.get("candidate_checkpoint_count") != EXPECTED_CHECKPOINT_COUNT
        or run.get("final_equals_step3000") is not True
    ):
        raise ValueError(f"{source} provenance or completion mismatch")
    checkpoints = _validate_checkpoint_inventory(output_dir, model_id=model_id)
    raw_checkpoints = run.get("checkpoints")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != EXPECTED_CHECKPOINT_COUNT:
        raise ValueError(f"{source} checkpoint evidence count mismatch")
    actual_by_name = {checkpoint.path.name: checkpoint for checkpoint in checkpoints}
    seen: set[str] = set()
    for raw in raw_checkpoints:
        if not isinstance(raw, dict):
            raise TypeError(f"{source} checkpoint evidence must be an object")
        _require_exact_keys(raw, {"name", "path", "sha256"}, source=f"{source} checkpoint")
        name = _required_string(raw, "name", source=f"{source} checkpoint")
        binding = _validated_file_binding(
            {"path": raw.get("path"), "sha256": raw.get("sha256")},
            source=f"{source} checkpoint {name}",
        )
        expected = actual_by_name.get(name)
        if (
            expected is None
            or binding.path != expected.path
            or binding.sha256 != expected.sha256
            or name in seen
        ):
            raise ValueError(f"{source} checkpoint binding mismatch")
        seen.add(name)
    if seen != EXPECTED_CHECKPOINT_NAMES:
        raise ValueError(f"{source} checkpoint inventory mismatch")
    if (
        actual_by_name["checkpoint_final.speaker.safetensors"].sha256
        != actual_by_name["checkpoint_0003000.speaker.safetensors"].sha256
    ):
        raise ValueError(f"final checkpoint does not match step 3000 for {model_id}")

    raw_log = _required_mapping(run, "log", source=f"{source} run")
    _require_exact_keys(
        raw_log,
        {
            "path",
            "sha256",
            "loss_event_count",
            "loss_steps_exact",
            "loss_all_finite",
            "last_loss",
            "oom",
            "traceback",
        },
        source=f"{source} log",
    )
    log_binding = _validated_file_binding(
        {"path": raw_log.get("path"), "sha256": raw_log.get("sha256")},
        source=f"{source} log",
    )
    status_log = _canonical_artifact_path(finished.get("log_path"), source=f"{source} status log")
    log_summary = parse_final_training_run(
        log_binding.path.read_text(encoding="utf-8", errors="replace")
    )
    declared_last_loss = _require_finite_number(
        raw_log.get("last_loss"), source=f"{source} last_loss"
    )
    if (
        log_binding.path != status_log
        or raw_log.get("loss_event_count") != log_summary.loss_event_count
        or raw_log.get("loss_steps_exact") is not True
        or raw_log.get("loss_all_finite") is not True
        or raw_log.get("oom") is not False
        or raw_log.get("traceback") is not False
        or not math.isclose(declared_last_loss, log_summary.last_loss, rel_tol=1e-5, abs_tol=1e-8)
    ):
        raise ValueError(f"{source} log evidence mismatch")

    _require_exact_keys(
        runtime_after,
        {
            "gpu_memory_used_mib",
            "gpu_memory_total_mib",
            "gpu_utilization_percent",
            "gpu_power_watts",
            "active_training_processes",
        },
        source=f"{source} runtime_after",
    )
    used = _require_finite_number(
        runtime_after.get("gpu_memory_used_mib"), source=f"{source} GPU memory used"
    )
    total = _require_finite_number(
        runtime_after.get("gpu_memory_total_mib"), source=f"{source} GPU memory total"
    )
    utilization = _require_finite_number(
        runtime_after.get("gpu_utilization_percent"), source=f"{source} GPU utilization"
    )
    power = _require_finite_number(
        runtime_after.get("gpu_power_watts"), source=f"{source} GPU power"
    )
    if (
        used < 0
        or total <= 0
        or used > total
        or utilization < 0
        or utilization > MAX_GPU_UTILIZATION_PERCENT
        or power < 0
        or runtime_after.get("active_training_processes") != []
        or used > launch_gpu_baseline + gpu_memory_tolerance_mib
    ):
        raise ValueError(f"{source} runtime closure mismatch")
    return checkpoints, log_binding.path, log_summary


def _launch_gpu_baseline(launch: Mapping[str, object]) -> float:
    gpu_before = launch.get("gpu_before")
    if not isinstance(gpu_before, dict):
        raise TypeError("training launch evidence gpu_before is missing")
    baseline = gpu_before.get("used_mib")
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, int | float)
        or not math.isfinite(float(baseline))
        or float(baseline) < 0
    ):
        raise ValueError("training launch evidence gpu_before.used_mib is invalid")
    return float(baseline)


def _validate_quality_run_chain(  # noqa: PLR0913 - mirrors the immutable lineage contract.
    evidence_paths: Sequence[Path],
    *,
    launch: Mapping[str, object],
    base_jobs_document: Mapping[str, object],
    base_jobs: Sequence[Mapping[str, object]],
    base_jobs_path: Path,
    base_status: FileBinding,
    base_status_rows: Sequence[Mapping[str, object]],
    final_status_path: Path,
    base_checkpoint: Path,
    base_sha: str,
    revision: str,
    upstream: str,
    gpu_memory_tolerance_mib: float,
) -> tuple[
    tuple[Mapping[str, object], ...],
    FileBinding,
    FileBinding,
    tuple[Mapping[str, object], ...],
    tuple[QualityRunLineage, ...],
]:
    expected_count, expected_sha = _launch_status_tail(launch)
    effective_jobs = tuple(base_jobs)
    effective_jobs_document = base_jobs_document
    effective_jobs_base = base_jobs_path.parent
    effective_jobs_binding = FileBinding(base_jobs_path, sha256_file(base_jobs_path))
    original_ids = tuple(
        _required_string(job, "model_id", source="training job") for job in base_jobs
    )
    lineages: list[QualityRunLineage] = []
    seen_evidence: set[Path] = set()
    seen_status = {base_status.path}
    seen_configs = {_job_path(job, "config", base=base_jobs_path.parent) for job in base_jobs}
    seen_outputs = {_job_path(job, "output_dir", base=base_jobs_path.parent) for job in base_jobs}
    launch_gpu_baseline = _launch_gpu_baseline(launch)
    effective_status = base_status
    effective_status_rows = tuple(base_status_rows)

    for evidence_argument in evidence_paths:
        evidence_path = _require_regular_file(
            evidence_argument,
            source="training run evidence",
        ).resolve(strict=True)
        if evidence_path in seen_evidence or not evidence_path.is_file():
            raise ValueError("training run evidence is duplicate, symlinked, or missing")
        seen_evidence.add(evidence_path)
        evidence_sha = sha256_file(evidence_path)
        evidence = _read_json(evidence_path)
        source = f"training run evidence {evidence_path}"
        _require_exact_keys(
            evidence,
            {
                "schema_version",
                "created_at",
                "state",
                "model_id",
                "queue_exit_code",
                "setup_evidence",
                "training_jobs",
                "training_status",
                "queue_script",
                "invocation",
                "run",
                "runtime_after",
            },
            source=source,
        )
        model_id = _required_string(evidence, "model_id", source=source)
        if (
            evidence.get("schema_version") != QUALITY_RUN_EVIDENCE_SCHEMA
            or evidence.get("state") != "finished"
            or type(evidence.get("queue_exit_code")) is not int
            or evidence.get("queue_exit_code") != 0
            or original_ids.count(model_id) != 1
        ):
            raise ValueError(f"{source} completion or model contract mismatch")
        _require_iso_datetime(evidence.get("created_at"), source=f"{source} created_at")

        status_evidence = _required_mapping(evidence, "training_status", source=source)
        _require_exact_keys(
            status_evidence,
            {
                "path",
                "before_row_count",
                "before_sha256",
                "after_row_count",
                "after_sha256",
                "new_status_row_count",
                "new_started_model_ids",
                "new_finished_success_model_ids",
            },
            source=f"{source} training_status",
        )
        bound_status = _canonical_artifact_path(
            status_evidence.get("path"), source=f"{source} training status"
        )
        if not bound_status.is_relative_to(evidence_path.parent) or bound_status in seen_status:
            raise ValueError(f"{source} training status is aliased or outside its run root")
        current_status_pairs = _read_jsonl_raw_lines(bound_status)
        current_status_lines = tuple(raw for raw, _row in current_status_pairs)
        current_status_rows = tuple(row for _raw, row in current_status_pairs)
        before_count = _required_int(
            status_evidence,
            "before_row_count",
            source=f"{source} training_status",
        )
        before_sha = _required_sha256(
            status_evidence,
            "before_sha256",
            source=f"{source} training_status",
        )
        after_count = _required_int(
            status_evidence,
            "after_row_count",
            source=f"{source} training_status",
        )
        after_sha = _required_sha256(
            status_evidence,
            "after_sha256",
            source=f"{source} training_status",
        )
        if (
            before_count != expected_count
            or before_sha != expected_sha
            or hashlib.sha256(b"".join(current_status_lines[:before_count])).hexdigest()
            != before_sha
            or after_count != before_count + EXPECTED_QUALITY_STATUS_ROW_COUNT
            or after_count != len(current_status_lines)
            or hashlib.sha256(b"".join(current_status_lines)).hexdigest() != after_sha
            or status_evidence.get("new_status_row_count") != EXPECTED_QUALITY_STATUS_ROW_COUNT
            or status_evidence.get("new_started_model_ids") != [model_id]
            or status_evidence.get("new_finished_success_model_ids") != [model_id]
        ):
            raise ValueError(f"{source} status append-only chain mismatch")
        started, finished = current_status_rows[before_count:after_count]
        if not (
            started.get("model_id") == model_id
            and started.get("event") == "started"
            and started.get("status") == "running"
            and started.get("exit_code") is None
            and started.get("ended_at") is None
            and started.get("last_checkpoint") is None
            and started.get("last_checkpoint_sha256") is None
            and started.get("candidate_checkpoints") == []
            and started.get("error") is None
            and finished.get("model_id") == model_id
            and finished.get("event") == "finished"
            and finished.get("status") == "success"
            and type(finished.get("exit_code")) is int
            and finished.get("exit_code") == 0
            and finished.get("error") is None
            and finished.get("started_at") == started.get("started_at")
        ):
            raise ValueError(f"{source} status row order or state mismatch")

        setup_binding = _validated_file_binding(
            evidence.get("setup_evidence"), source=f"{source} setup evidence"
        )
        jobs_binding = _validated_file_binding(
            evidence.get("training_jobs"), source=f"{source} training jobs"
        )
        queue_binding = _validated_file_binding(
            evidence.get("queue_script"), source=f"{source} queue script"
        )
        if queue_binding.path.name != "run_600m_speaker_training_queue.py":
            raise ValueError(f"{source} queue script identity mismatch")
        invocation = _required_mapping(evidence, "invocation", source=source)
        _require_exact_keys(
            invocation,
            {"recipe", "checkpoint_revision", "upstream_commit"},
            source=f"{source} invocation",
        )
        recipe = invocation.get("recipe")
        if (
            not isinstance(recipe, str)
            or not recipe
            or invocation.get("checkpoint_revision") != revision
            or invocation.get("upstream_commit") != upstream
        ):
            raise ValueError(f"{source} invocation provenance mismatch")

        jobs_document = _read_json(jobs_binding.path)
        _require_exact_keys(
            jobs_document,
            {
                "schema_version",
                "created_at_utc",
                "base_checkpoint_path",
                "base_checkpoint_sha256",
                "checkpoint_revision",
                "upstream_commit",
                "queue_policy",
                "anabel_strategy",
                "jobs",
            },
            source=f"{source} training jobs",
        )
        _require_iso_datetime(
            jobs_document.get("created_at_utc"), source=f"{source} jobs created_at_utc"
        )
        if (
            jobs_document.get("queue_policy") != "serial_one_at_a_time"
            or jobs_document.get("anabel_strategy") != "reuse_existing_fresh_3000_run"
        ):
            raise ValueError(f"{source} training jobs policy metadata mismatch")
        allowed_top_level_changes = {
            "created_at_utc",
            "queue_policy",
            "anabel_strategy",
        }
        expected_top_level = {
            key: value for key, value in effective_jobs_document.items() if key != "jobs"
        }
        for key in allowed_top_level_changes:
            expected_top_level[key] = jobs_document.get(key)
        actual_top_level = {key: value for key, value in jobs_document.items() if key != "jobs"}
        if _canonical_sha256(actual_top_level) != _canonical_sha256(expected_top_level):
            raise ValueError(f"{source} changes undeclared training jobs metadata")
        successor_base = _canonical_artifact_path(
            jobs_document.get("base_checkpoint_path"), source=f"{source} base checkpoint"
        )
        if (
            jobs_document.get("schema_version") != 1
            or successor_base != base_checkpoint
            or jobs_document.get("base_checkpoint_sha256") != base_sha
            or jobs_document.get("checkpoint_revision") != revision
            or jobs_document.get("upstream_commit") != upstream
        ):
            raise ValueError(f"{source} base model provenance mismatch")
        successor_jobs = _training_jobs(jobs_document, base=jobs_binding.path.parent)
        successor_ids = tuple(
            _required_string(job, "model_id", source="training job") for job in successor_jobs
        )
        if successor_ids != original_ids:
            raise ValueError(f"{source} training job model order mismatch")
        target_index = original_ids.index(model_id)
        predecessor_job = effective_jobs[target_index]
        successor_job = successor_jobs[target_index]
        for index, (previous, current) in enumerate(
            zip(effective_jobs, successor_jobs, strict=True)
        ):
            allowed_job_changes = {"config", "output_dir", "command"}
            if index == target_index:
                expected_job = copy.deepcopy(dict(previous))
                for key in allowed_job_changes:
                    expected_job[key] = current.get(key)
                if _canonical_sha256(current) != _canonical_sha256(expected_job):
                    raise ValueError(f"{source} changes undeclared target job fields")
            elif _canonical_sha256(current) != _canonical_sha256(previous):
                raise ValueError(f"{source} changes a non-target job")
            previous_config = _job_path(previous, "config", base=effective_jobs_base)
            current_config = _canonical_artifact_path(
                current.get("config"), source=f"{source} job config"
            )
            previous_manifest = _job_path(previous, "clean_manifest", base=effective_jobs_base)
            current_manifest = _canonical_artifact_path(
                current.get("clean_manifest"), source=f"{source} job manifest"
            )
            previous_output = _job_path(previous, "output_dir", base=effective_jobs_base)
            current_output = _canonical_artifact_path(
                current.get("output_dir"), source=f"{source} job output", directory=True
            )
            previous_command = _normalized_job_command(
                previous,
                config=previous_config,
                output=previous_output,
                manifest=previous_manifest,
                base_checkpoint=base_checkpoint,
                source=f"predecessor job {original_ids[index]}",
            )
            current_command = _normalized_job_command(
                current,
                config=current_config,
                output=current_output,
                manifest=current_manifest,
                base_checkpoint=base_checkpoint,
                source=f"{source} job {original_ids[index]}",
            )
            if current_manifest != previous_manifest or current_command != previous_command:
                raise ValueError(f"{source} changes an undeclared job binding")
            if index == target_index:
                if (
                    current_config == previous_config
                    or current_output == previous_output
                    or current_config in seen_configs
                    or current_output in seen_outputs
                ):
                    raise ValueError(f"{source} successor job is not versioned create-only")
            elif current_config != previous_config or current_output != previous_output:
                raise ValueError(f"{source} changes a non-target job path")
        current_config = _job_path(successor_job, "config", base=jobs_binding.path.parent)
        current_manifest = _job_path(successor_job, "clean_manifest", base=jobs_binding.path.parent)
        current_output = _job_path(successor_job, "output_dir", base=jobs_binding.path.parent)
        setup = _read_json(setup_binding.path)
        diagnostic, initialization, setup_config, setup_output, _source_config = (
            _validate_quality_setup(
                setup,
                model_id=model_id,
                setup_path=setup_binding.path,
                jobs_binding=jobs_binding,
                queue_binding=queue_binding,
                status_path=bound_status,
                before_sha=before_sha,
                current_job=successor_job,
                current_jobs_base=jobs_binding.path.parent,
                predecessor_job=predecessor_job,
                predecessor_jobs_base=effective_jobs_base,
            )
        )
        if setup_config != current_config or setup_output != current_output:
            raise ValueError(f"{source} setup effective job mismatch")
        config_sha = sha256_file(current_config)
        manifest_sha = sha256_file(current_manifest)
        for row in (started, finished):
            if (
                row.get("config_sha256") != config_sha
                or row.get("clean_manifest_sha256") != manifest_sha
                or row.get("checkpoint_sha256") != base_sha
                or row.get("checkpoint_revision") != revision
                or row.get("upstream_commit") != upstream
            ):
                raise ValueError(f"{source} status provenance mismatch")
            _validate_optional_status_paths(
                row,
                manifest=current_manifest,
                config=current_config,
                output_dir=current_output,
                model_id=model_id,
            )
        raw_run = _required_mapping(evidence, "run", source=source)
        runtime_after = _required_mapping(evidence, "runtime_after", source=source)
        checkpoints, log_path, _log_summary = _validate_quality_run_payload(
            raw_run,
            runtime_after,
            model_id=model_id,
            started=started,
            finished=finished,
            config=current_config,
            manifest=current_manifest,
            output_dir=current_output,
            base_sha=base_sha,
            launch_gpu_baseline=launch_gpu_baseline,
            gpu_memory_tolerance_mib=gpu_memory_tolerance_mib,
        )
        _validate_status_candidates(finished, checkpoints=checkpoints, model_id=model_id)
        if (
            _canonical_artifact_path(started.get("log_path"), source=f"{source} started log")
            != log_path
        ):
            raise ValueError(f"{source} started log path mismatch")
        _validate_training_config(
            current_config,
            output_dir=current_output,
            manifest=current_manifest,
        )
        lineage = QualityRunLineage(
            model_id=model_id,
            evidence=FileBinding(evidence_path, evidence_sha),
            setup_evidence=setup_binding,
            training_jobs=jobs_binding,
            training_status=FileBinding(bound_status, after_sha),
            queue_script=queue_binding,
            source_diagnostic=diagnostic,
            initialization_checkpoint=initialization,
        )
        lineages.append(lineage)
        seen_status.add(bound_status)
        seen_configs.add(current_config)
        seen_outputs.add(current_output)
        effective_jobs = successor_jobs
        effective_jobs_document = jobs_document
        effective_jobs_base = jobs_binding.path.parent
        effective_jobs_binding = jobs_binding
        effective_status = FileBinding(bound_status, after_sha)
        effective_status_rows = current_status_rows
        expected_count = after_count
        expected_sha = after_sha

    if final_status_path != effective_status.path or sha256_file(final_status_path) != expected_sha:
        raise ValueError("final training status does not match the declared evidence chain")
    return (
        effective_jobs,
        effective_jobs_binding,
        effective_status,
        effective_status_rows,
        tuple(lineages),
    )


def verify_training(
    training_jobs: Path,
    training_status: Path,
    training_launch_evidence: Path,
    runtime_snapshot: RuntimeSnapshot,
    gpu_memory_tolerance_mib: float,
    *,
    training_run_evidence: Sequence[Path] = (),
) -> TrainingVerification:
    jobs_path = _require_regular_file(training_jobs, source="training jobs").resolve(strict=True)
    status_path = _require_regular_file(training_status, source="final training status").resolve(
        strict=True
    )
    launch_path = _require_regular_file(
        training_launch_evidence, source="training launch evidence"
    ).resolve(strict=True)
    jobs_document = _read_json(jobs_path)
    jobs = _training_jobs(jobs_document, base=jobs_path.parent)
    base_checkpoint = _resolve_path(
        jobs_document.get("base_checkpoint_path"),
        base=jobs_path.parent,
        source="training jobs base_checkpoint_path",
    )
    base_sha = _required_sha256(
        jobs_document,
        "base_checkpoint_sha256",
        source="training jobs",
    )
    if sha256_file(base_checkpoint) != base_sha:
        raise ValueError("training jobs base checkpoint SHA-256 mismatch")
    revision = _required_string(jobs_document, "checkpoint_revision", source="training jobs")
    upstream = _required_string(jobs_document, "upstream_commit", source="training jobs")
    if not _is_lower_hex(revision, length=40) or not _is_lower_hex(upstream, length=40):
        raise ValueError("training jobs revisions must be lowercase 40-character hex")
    job_ids = tuple(_required_string(job, "model_id", source="training job") for job in jobs)
    launch = _read_json(launch_path)
    base_status_path = _canonical_artifact_path(
        launch.get("status_path"),
        source="training launch evidence status_path",
    )
    base_status_pairs = _read_jsonl_raw_lines(base_status_path)
    base_status_rows = tuple(row for _raw, row in base_status_pairs)
    base_status_lines = tuple(raw for raw, _row in base_status_pairs)
    base_jobs_sha = sha256_file(jobs_path)
    _validate_launch_evidence(
        launch,
        jobs_path=jobs_path,
        jobs_sha=base_jobs_sha,
        status_path=base_status_path,
        base_checkpoint=base_checkpoint,
        base_sha=base_sha,
        revision=revision,
        upstream=upstream,
        pending_ids=job_ids[2:],
        jobs=jobs,
        status_rows=base_status_rows,
        status_lines=base_status_lines,
    )
    if training_run_evidence:
        _validate_training_predecessor(
            jobs,
            jobs_base=jobs_path.parent,
            status_rows=base_status_rows,
            status_base=base_status_path.parent,
            base_sha=base_sha,
            revision=revision,
            upstream=upstream,
        )
    base_status_binding = FileBinding(base_status_path, sha256_file(base_status_path))
    (
        effective_jobs,
        effective_jobs_binding,
        effective_status_binding,
        effective_status_rows,
        run_lineages,
    ) = _validate_quality_run_chain(
        training_run_evidence,
        launch=launch,
        base_jobs_document=jobs_document,
        base_jobs=jobs,
        base_jobs_path=jobs_path,
        base_status=base_status_binding,
        base_status_rows=base_status_rows,
        final_status_path=status_path,
        base_checkpoint=base_checkpoint,
        base_sha=base_sha,
        revision=revision,
        upstream=upstream,
        gpu_memory_tolerance_mib=gpu_memory_tolerance_mib,
    )
    status_by_model: dict[str, list[Mapping[str, object]]] = {model_id: [] for model_id in job_ids}
    for row in effective_status_rows:
        model_id = row.get("model_id")
        if isinstance(model_id, str) and model_id in status_by_model:
            status_by_model[model_id].append(row)
    lineages_by_model = {
        model_id: tuple(lineage for lineage in run_lineages if lineage.model_id == model_id)
        for model_id in job_ids
    }
    summaries: list[TrainingModelSummary] = []
    for job in effective_jobs:
        model_id = _required_string(job, "model_id", source="training job")
        model_rows = status_by_model[model_id]
        if not model_rows:
            raise ValueError(f"training status is missing model {model_id}")
        latest = model_rows[-1]
        if not (
            latest.get("event") == "finished"
            and latest.get("status") == "success"
            and type(latest.get("exit_code")) is int
            and latest.get("exit_code") == 0
        ):
            raise ValueError(f"latest training event is not successful for {model_id}")
        manifest = _job_path(job, "clean_manifest", base=effective_jobs_binding.path.parent)
        config = _job_path(job, "config", base=effective_jobs_binding.path.parent)
        output_dir = _job_path(job, "output_dir", base=effective_jobs_binding.path.parent)
        expected_provenance = {
            "clean_manifest_sha256": sha256_file(manifest),
            "config_sha256": sha256_file(config),
            "checkpoint_sha256": base_sha,
            "checkpoint_revision": revision,
            "upstream_commit": upstream,
        }
        mismatches = [key for key, value in expected_provenance.items() if latest.get(key) != value]
        if mismatches:
            raise ValueError(
                f"training status provenance mismatch for {model_id}: {', '.join(mismatches)}"
            )
        _validate_optional_status_paths(
            latest,
            manifest=manifest,
            config=config,
            output_dir=output_dir,
            model_id=model_id,
        )
        checkpoints = _validate_checkpoint_inventory(output_dir, model_id=model_id)
        by_name = {checkpoint.path.name: checkpoint for checkpoint in checkpoints}
        if (
            by_name["checkpoint_final.speaker.safetensors"].sha256
            != by_name["checkpoint_0003000.speaker.safetensors"].sha256
        ):
            raise ValueError(f"final checkpoint does not match step 3000 for {model_id}")
        _validate_status_candidates(latest, checkpoints=checkpoints, model_id=model_id)
        _validate_training_config(config, output_dir=output_dir, manifest=manifest)
        log_path = _resolve_existing_file_path(
            latest.get("log_path"), base=status_path.parent, source="log_path"
        )
        log_sha = sha256_file(log_path)
        log = parse_final_training_run(log_path.read_text(encoding="utf-8", errors="replace"))
        summaries.append(
            TrainingModelSummary(
                model_id=model_id,
                checkpoint_count=len(checkpoints),
                loss_event_count=log.loss_event_count,
                config_sha256=expected_provenance["config_sha256"],
                clean_manifest_sha256=expected_provenance["clean_manifest_sha256"],
                log_sha256=log_sha,
                output_dir=output_dir,
                checkpoints=checkpoints,
                latest_status=dict(latest),
                run_id=_resolve_training_run_id(latest, model_id=model_id),
                config_path=config,
                clean_manifest_path=manifest,
                log_path=log_path,
                run_evidence_lineage=lineages_by_model[model_id],
            )
        )
    _validate_runtime(
        runtime_snapshot,
        launch=launch,
        gpu_memory_tolerance_mib=gpu_memory_tolerance_mib,
    )
    return TrainingVerification(
        models=tuple(summaries),
        training_jobs=effective_jobs_binding.path,
        training_jobs_sha256=effective_jobs_binding.sha256,
        training_status=effective_status_binding.path,
        training_status_sha256=effective_status_binding.sha256,
        training_launch_evidence=launch_path,
        training_launch_evidence_sha256=sha256_file(launch_path),
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_sha,
        checkpoint_revision=revision,
        upstream_commit=upstream,
        runtime_snapshot=runtime_snapshot,
        base_training_jobs=(
            FileBinding(jobs_path, base_jobs_sha) if training_run_evidence else None
        ),
        base_training_status=(base_status_binding if training_run_evidence else None),
        training_run_evidence=run_lineages,
    )


def _training_jobs(
    document: Mapping[str, object],
    *,
    base: Path,
) -> tuple[Mapping[str, object], ...]:
    schema = document.get("schema_version")
    if type(schema) is not int or schema != 1:
        raise ValueError("training jobs schema_version must be numeric 1")
    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_MODEL_COUNT:
        raise ValueError("training jobs must contain exactly 12 jobs")
    jobs: list[Mapping[str, object]] = []
    model_ids: list[str] = []
    for raw in raw_jobs:
        if not isinstance(raw, dict):
            raise TypeError("training job entries must be objects")
        model_id = _required_string(raw, "model_id", source="training job")
        if model_id in model_ids or "/" in model_id or "\\" in model_id:
            raise ValueError(f"duplicate or unsafe training model id: {model_id}")
        model_ids.append(model_id)
        for field in ("clean_manifest", "config", "output_dir"):
            _job_path(raw, field, base=base)
        jobs.append(raw)
    return tuple(jobs)


def _job_path(job: Mapping[str, object], field: str, *, base: Path) -> Path:
    source = f"training job {field}"
    if field == "output_dir":
        return _resolve_existing_directory_path(job.get(field), base=base, source=source)
    return _resolve_existing_file_path(job.get(field), base=base, source=source)


def _validate_optional_status_paths(
    status: Mapping[str, object],
    *,
    manifest: Path,
    config: Path,
    output_dir: Path,
    model_id: str,
) -> None:
    expected = {
        "clean_manifest": manifest,
        "clean_manifest_path": manifest,
        "config": config,
        "config_path": config,
        "output_dir": output_dir,
    }
    for field, path in expected.items():
        value = status.get(field)
        if value is not None:
            resolved = (
                _resolve_existing_directory_path(value, base=path.parent, source=field)
                if field == "output_dir"
                else _resolve_existing_file_path(value, base=path.parent, source=field)
            )
            if resolved != path:
                raise ValueError(f"training status {field} path mismatch for {model_id}")


def _validate_training_predecessor(
    jobs: Sequence[Mapping[str, object]],
    *,
    jobs_base: Path,
    status_rows: Sequence[Mapping[str, object]],
    status_base: Path,
    base_sha: str,
    revision: str,
    upstream: str,
) -> None:
    rows_by_model: dict[str, list[Mapping[str, object]]] = {}
    for row in status_rows:
        model_id = row.get("model_id")
        if isinstance(model_id, str):
            rows_by_model.setdefault(model_id, []).append(row)
    for job in jobs:
        model_id = _required_string(job, "model_id", source="predecessor training job")
        model_rows = rows_by_model.get(model_id, [])
        if not model_rows:
            raise ValueError(f"predecessor training status is missing model {model_id}")
        latest = model_rows[-1]
        if not (
            latest.get("event") == "finished"
            and latest.get("status") == "success"
            and type(latest.get("exit_code")) is int
            and latest.get("exit_code") == 0
        ):
            raise ValueError(f"predecessor training status is not successful for {model_id}")
        manifest = _job_path(job, "clean_manifest", base=jobs_base)
        config = _job_path(job, "config", base=jobs_base)
        output_dir = _job_path(job, "output_dir", base=jobs_base)
        expected_provenance = {
            "clean_manifest_sha256": sha256_file(manifest),
            "config_sha256": sha256_file(config),
            "checkpoint_sha256": base_sha,
            "checkpoint_revision": revision,
            "upstream_commit": upstream,
        }
        mismatches = [key for key, value in expected_provenance.items() if latest.get(key) != value]
        if mismatches:
            raise ValueError(
                f"predecessor training provenance mismatch for {model_id}: {', '.join(mismatches)}"
            )
        _validate_optional_status_paths(
            latest,
            manifest=manifest,
            config=config,
            output_dir=output_dir,
            model_id=model_id,
        )
        checkpoints = _validate_checkpoint_inventory(output_dir, model_id=model_id)
        by_name = {checkpoint.path.name: checkpoint for checkpoint in checkpoints}
        if (
            by_name["checkpoint_final.speaker.safetensors"].sha256
            != by_name["checkpoint_0003000.speaker.safetensors"].sha256
        ):
            raise ValueError(f"predecessor final checkpoint mismatch for {model_id}")
        _validate_status_candidates(latest, checkpoints=checkpoints, model_id=model_id)
        _validate_training_config(config, output_dir=output_dir, manifest=manifest)
        log_path = _resolve_existing_file_path(
            latest.get("log_path"), base=status_base, source="predecessor log_path"
        )
        sha256_file(log_path)
        parse_final_training_run(log_path.read_text(encoding="utf-8", errors="replace"))


def _validate_checkpoint_inventory(
    output_dir: Path,
    *,
    model_id: str,
) -> tuple[ValidatedEmbedding, ...]:
    lexical_output = _require_directory(
        output_dir, source=f"training output directory for {model_id}"
    )
    resolved_output = lexical_output.resolve(strict=True)
    paths = tuple(sorted(lexical_output.glob("*.speaker.safetensors")))
    if {path.name for path in paths} != EXPECTED_CHECKPOINT_NAMES or len(
        paths
    ) != EXPECTED_CHECKPOINT_COUNT:
        raise ValueError(f"checkpoint inventory mismatch for {model_id}")
    for path in paths:
        lexical_checkpoint = _require_regular_file(path, source=f"checkpoint for {model_id}")
        if lexical_checkpoint.parent.resolve(strict=True) != resolved_output:
            raise ValueError(f"checkpoint is outside the training output for {model_id}")
    nested = tuple(
        path
        for path in lexical_output.rglob("*.speaker.safetensors")
        if path.parent != lexical_output
    )
    if nested:
        raise ValueError(f"unrelated nested checkpoint exists for {model_id}")
    return tuple(validate_speaker_embedding(path) for path in paths)


def _validate_status_candidates(
    status: Mapping[str, object],
    *,
    checkpoints: Sequence[ValidatedEmbedding],
    model_id: str,
) -> None:
    raw = status.get("candidate_checkpoints")
    if not isinstance(raw, list) or len(raw) not in {12, 13}:
        raise ValueError(f"status checkpoint candidates must contain 12 or 13 rows for {model_id}")
    by_path = {checkpoint.path: checkpoint.sha256 for checkpoint in checkpoints}
    recorded: list[Path] = []
    for candidate in raw:
        if not isinstance(candidate, dict):
            raise TypeError(f"status candidate must be an object for {model_id}")
        path = _resolve_existing_file_path(
            candidate.get("path"), base=checkpoints[0].path.parent, source="candidate"
        )
        sha = _required_sha256(candidate, "sha256", source="status candidate")
        if path not in by_path or by_path[path] != sha or path in recorded:
            raise ValueError(f"status checkpoint candidate mismatch for {model_id}")
        recorded.append(path)
    periodic = {
        checkpoint.path
        for checkpoint in checkpoints
        if checkpoint.path.name != "checkpoint_final.speaker.safetensors"
    }
    if not periodic.issubset(recorded):
        raise ValueError(f"status checkpoint candidates omit a periodic checkpoint for {model_id}")
    final = next(
        checkpoint.path
        for checkpoint in checkpoints
        if checkpoint.path.name == "checkpoint_final.speaker.safetensors"
    )
    if set(recorded) - periodic not in (set(), {final}):
        raise ValueError(
            f"status checkpoint candidates contain an unrelated checkpoint for {model_id}"
        )
    last = next(
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.path.name == "checkpoint_0003000.speaker.safetensors"
    )
    if (
        _resolve_existing_file_path(
            status.get("last_checkpoint"), base=last.path.parent, source="last_checkpoint"
        )
        != last.path
        or status.get("last_checkpoint_sha256") != last.sha256
    ):
        raise ValueError(f"last_checkpoint must bind periodic step 3000 for {model_id}")


def _validate_training_config(config_path: Path, *, output_dir: Path, manifest: Path) -> None:
    document = _read_json(config_path)
    train = document.get("train", document)
    if not isinstance(train, dict):
        raise TypeError(f"training config train must be an object: {config_path}")
    expected: dict[str, object] = {
        "speaker_inversion_enabled": True,
        "speaker_inversion_tokens": 16,
        "max_steps": 3000,
        "save_every": 250,
        "log_every": 20,
        "valid_ratio": 0.0,
        "checkpoint_best_n": 0,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if train.get(field) != value or type(train.get(field)) is not type(value)
    ]
    if mismatches:
        raise ValueError(f"training config contract mismatch: {', '.join(mismatches)}")
    for field, expected_path in (("output_dir", output_dir), ("manifest_path", manifest)):
        value = train.get(field)
        if (
            value is not None
            and _resolve_path(
                str(value),
                base=config_path.parent,
                source=f"training config {field}",
            )
            != expected_path
        ):
            raise ValueError(f"training config {field} path mismatch")


def _validate_launch_evidence(  # noqa: PLR0913 - mirrors the immutable evidence contract.
    launch: Mapping[str, object],
    *,
    jobs_path: Path,
    jobs_sha: str,
    status_path: Path,
    base_checkpoint: Path,
    base_sha: str,
    revision: str,
    upstream: str,
    pending_ids: Sequence[str],
    jobs: Sequence[Mapping[str, object]],
    status_rows: Sequence[Mapping[str, object]],
    status_lines: Sequence[bytes],
) -> None:
    if type(launch.get("schema_version")) is not int or launch.get("schema_version") != 1:
        raise ValueError("training launch evidence schema_version must be numeric 1")
    exit_code = launch.get("queue_exit_code", launch.get("exit_code"))
    errors = launch.get("completion_errors")
    active = launch.get("active_owned_processes_after")
    if not (
        launch.get("state") == "finished"
        and type(exit_code) is int
        and exit_code == 0
        and errors == []
        and launch.get("new_status_contract_valid") is True
        and launch.get("new_status_row_count") == EXPECTED_NEW_STATUS_ROW_COUNT
        and launch.get("new_started_model_ids") == list(pending_ids)
        and launch.get("new_finished_success_model_ids") == list(pending_ids)
        and launch.get("finished_success_model_ids")
        == [_required_string(job, "model_id", source="training job") for job in jobs]
        and launch.get("finished_failed_model_ids") == []
        and active == []
        and launch.get("gpu_memory_released") is True
    ):
        raise ValueError("training launch evidence completion contract mismatch")
    _require_evidence_binding(launch, ("training_jobs", "jobs"), jobs_path, jobs_sha)
    _require_evidence_path(launch, "status_path", status_path)
    _require_evidence_binding(
        launch,
        ("checkpoint",),
        base_checkpoint,
        base_sha,
    )
    _validate_status_evidence(
        launch,
        status_path=status_path,
        status_rows=status_rows,
        status_lines=status_lines,
        pending_jobs=jobs[2:],
    )
    if launch.get("checkpoint_revision", launch.get("base_revision")) != revision:
        raise ValueError("training launch evidence checkpoint revision mismatch")
    if launch.get("upstream_commit") != upstream:
        raise ValueError("training launch evidence upstream commit mismatch")
    script_path = _resolve_path(
        launch.get("launcher_script_path"),
        base=jobs_path.parent,
        source="launcher script",
    )
    if script_path.name != "launch_600m_training_queue_speed_v1.py":
        raise ValueError("training launch evidence script identity mismatch")
    if _required_sha256(
        launch,
        "launcher_script_sha256",
        source="training launch evidence",
    ) != sha256_file(script_path):
        raise ValueError("training launch evidence script SHA-256 mismatch")
    queue_script_path = _resolve_path(
        launch.get("queue_script_path"),
        base=jobs_path.parent,
        source="queue script",
    )
    if queue_script_path.name != "run_600m_speaker_training_queue.py":
        raise ValueError("training launch evidence queue script identity mismatch")
    if _required_sha256(
        launch,
        "queue_script_sha256",
        source="training launch evidence",
    ) != sha256_file(queue_script_path):
        raise ValueError("training launch evidence queue script SHA-256 mismatch")


def _validate_status_evidence(
    launch: Mapping[str, object],
    *,
    status_path: Path,
    status_rows: Sequence[Mapping[str, object]],
    status_lines: Sequence[bytes],
    pending_jobs: Sequence[Mapping[str, object]],
) -> None:
    before_count = launch.get("status_row_count_before")
    if isinstance(before_count, bool) or not isinstance(before_count, int) or before_count < 0:
        raise ValueError("training launch evidence status_row_count_before is invalid")
    after_count = launch.get("status_row_count")
    if isinstance(after_count, bool) or not isinstance(after_count, int):
        raise TypeError("training launch evidence status_row_count mismatch")
    if after_count != before_count + EXPECTED_NEW_STATUS_ROW_COUNT:
        if after_count == len(status_lines):
            raise ValueError("training status row count does not match launcher evidence")
        raise ValueError("training launch evidence status_row_count mismatch")
    if len(status_lines) != after_count or len(status_rows) != len(status_lines):
        raise ValueError("training status row count does not match launcher evidence")
    before_sha = hashlib.sha256(b"".join(status_lines[:before_count])).hexdigest()
    if launch.get("status_before_sha256") != before_sha:
        raise ValueError("training status prefix SHA-256 mismatch")
    after_sha = hashlib.sha256(b"".join(status_lines[:after_count])).hexdigest()
    if launch.get("status_after_sha256") != after_sha:
        raise ValueError("training status after SHA-256 mismatch")
    new_rows = status_rows[before_count:after_count]
    for index, job in enumerate(pending_jobs):
        model_id = _required_string(job, "model_id", source="training job")
        started = new_rows[index * 2]
        finished = new_rows[index * 2 + 1]
        if not (
            started.get("model_id") == model_id
            and started.get("event") == "started"
            and started.get("status") == "running"
            and started.get("exit_code") is None
            and finished.get("model_id") == model_id
            and finished.get("event") == "finished"
            and finished.get("status") == "success"
            and type(finished.get("exit_code")) is int
            and finished.get("exit_code") == 0
        ):
            raise ValueError(f"training status new row sequence mismatch for {model_id}")
        expected_log = _resolve_existing_file_path(
            str(status_path.parent / "logs" / f"{model_id}.log"),
            base=status_path.parent,
            source=f"expected training log for {model_id}",
        )
        resolved_logs = [
            _resolve_path(
                row.get("log_path"),
                base=status_path.parent,
                source=f"training status log for {model_id}",
            )
            for row in (started, finished)
        ]
        if resolved_logs != [expected_log, expected_log] or not expected_log.is_file():
            raise ValueError(f"training status log path mismatch for {model_id}")


def _require_evidence_binding(
    evidence: Mapping[str, object],
    names: Sequence[str],
    expected_path: Path,
    expected_sha: str,
) -> None:
    for name in names:
        binding = evidence.get(name)
        if isinstance(binding, dict):
            path = _resolve_path(binding.get("path"), base=expected_path.parent, source=name)
            sha = binding.get("sha256")
            if path != expected_path or sha != expected_sha:
                raise ValueError(f"training launch evidence {name} binding mismatch")
            return
        path_value = evidence.get(f"{name}_path", binding)
        sha_value = evidence.get(f"{name}_sha256")
        if isinstance(path_value, str) and isinstance(sha_value, str):
            path = _resolve_path(path_value, base=expected_path.parent, source=name)
            if path != expected_path or sha_value != expected_sha:
                raise ValueError(f"training launch evidence {name} binding mismatch")
            return
    raise ValueError(f"training launch evidence is missing {names[0]} binding")


def _require_evidence_path(
    evidence: Mapping[str, object],
    field: str,
    expected_path: Path,
) -> None:
    path = _resolve_path(evidence.get(field), base=expected_path.parent, source=field)
    if path != expected_path:
        raise ValueError(f"training launch evidence {field} binding mismatch")


def _validate_runtime(
    snapshot: RuntimeSnapshot,
    *,
    launch: Mapping[str, object],
    gpu_memory_tolerance_mib: float,
) -> None:
    if not math.isfinite(gpu_memory_tolerance_mib) or gpu_memory_tolerance_mib < 0:
        raise ValueError("GPU memory tolerance must be finite and nonnegative")
    if snapshot.errors:
        raise ValueError("runtime probe reported an error")
    if snapshot.conflicting_processes:
        raise ValueError("runtime contains a residual workflow process")
    if snapshot.related_compute_applications:
        raise ValueError("runtime contains a related NVIDIA compute application")
    if snapshot.gpu_memory_used_mib is None:
        raise ValueError("runtime GPU memory is unavailable")
    gpu_before = launch.get("gpu_before")
    if not isinstance(gpu_before, dict):
        raise TypeError("training launch evidence gpu_before is missing")
    baseline = gpu_before.get("used_mib")
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, int | float)
        or not math.isfinite(float(baseline))
    ):
        raise ValueError("training launch evidence gpu_before.used_mib is invalid")
    if snapshot.gpu_memory_used_mib > float(baseline) + gpu_memory_tolerance_mib:
        raise ValueError("runtime GPU memory has not returned to baseline")


def snapshot_path(path: Path) -> dict[str, object]:
    lexical = _require_no_alias_components(path, source="snapshot path")
    if lexical.is_file():
        resolved = _require_regular_file(lexical, source="snapshot path").resolve(strict=True)
        return {"path": str(resolved), "kind": "file", "files": {".": sha256_file(resolved)}}
    if not lexical.is_dir():
        raise ValueError(f"snapshot path does not exist: {lexical}")
    resolved = _require_directory(lexical, source="snapshot path").resolve(strict=True)
    files = {
        child.relative_to(resolved).as_posix(): sha256_file(child)
        for child in sorted(resolved.rglob("*"))
        if child.is_file()
    }
    if not files:
        raise ValueError(f"snapshot directory contains no files: {resolved}")
    return {"path": str(resolved), "kind": "directory", "files": files}


def validate_case_matrix(rows: Sequence[Mapping[str, object]], *, model_id: str) -> None:
    expected = {
        (model_id, step, text_id, seed, style)
        for step in EXPECTED_EVALUATION_STEPS
        for text_id in EXPECTED_TEXT_IDS
        for seed in EXPECTED_SEEDS
        for style in EXPECTED_STYLES
    }
    actual: list[tuple[object, object, object, object, object]] = [
        (
            row.get("model_id"),
            row.get("checkpoint_step"),
            row.get("text_id"),
            row.get("seed"),
            row.get("style"),
        )
        for row in rows
    ]
    case_ids = [row.get("case_id") for row in rows]
    if (
        len(rows) != EXPECTED_EVALUATION_CASE_COUNT
        or len(actual) != len(set(actual))
        or set(actual) != expected
        or not all(isinstance(case_id, str) and case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
    ):
        raise ValueError(f"evaluation case matrix mismatch for {model_id}")
    metric_gate_counts = {step: {True: 0, False: 0} for step in EXPECTED_EVALUATION_STEPS}
    for row in rows:
        text_id = row.get("text_id")
        metric_gate_applied = row.get("metric_gate_applied")
        expected_metric_gate = text_id in EXPECTED_HARD_GATE_TEXT_IDS
        if metric_gate_applied is not expected_metric_gate:
            raise ValueError(f"evaluation metric gate distribution mismatch for {model_id}")
        step = row.get("checkpoint_step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError(f"evaluation metric gate distribution mismatch for {model_id}")
        metric_gate_counts[step][expected_metric_gate] += 1
    if any(
        counts[True] != EXPECTED_HARD_GATE_METRIC_CASE_COUNT_PER_CHECKPOINT
        or counts[False] != EXPECTED_DIAGNOSTIC_WORD_CASE_COUNT_PER_CHECKPOINT
        for counts in metric_gate_counts.values()
    ):
        raise ValueError(f"evaluation metric gate distribution mismatch for {model_id}")


def verify_evaluations(
    evaluation_config: Path,
    evaluation_status: Path,
    training: TrainingVerification,
) -> EvaluationVerification:
    config_path = _require_regular_file(
        evaluation_config,
        source="evaluation config",
    ).resolve(strict=True)
    status_path = _require_regular_file(
        evaluation_status,
        source="evaluation status",
    ).resolve(strict=True)
    config = _read_json(config_path)
    if config.get("schema_version") != EVALUATION_CONFIG_SCHEMA:
        raise ValueError(f"evaluation config requires {EVALUATION_CONFIG_SCHEMA}")
    base = config_path.parent
    config_sha256 = sha256_file(config_path)
    lock_path = status_path.with_suffix(status_path.suffix + ".lock")
    if lock_path.exists():
        raise ValueError(f"evaluation queue lock is present: {lock_path}")
    _validate_evaluation_status_config_binding(
        status_path,
        config_path=config_path,
        config_sha256=config_sha256,
    )
    runtime_manifest, runtime_files = _validate_runtime_evaluation_inputs(
        config_path,
        config=config,
        training=training,
    )
    raw_base = config.get("base_checkpoint")
    if not isinstance(raw_base, dict):
        raise TypeError("evaluation config base_checkpoint must be an object")
    configured_base_path = _resolve_path(
        raw_base.get("path"), base=base, source="evaluation base checkpoint"
    )
    if (
        configured_base_path != training.base_checkpoint
        or raw_base.get("sha256") != training.base_checkpoint_sha256
        or raw_base.get("revision") != training.checkpoint_revision
        or sha256_file(configured_base_path) != training.base_checkpoint_sha256
    ):
        raise ValueError("evaluation base checkpoint identity mismatch")
    raw_models = config.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != EXPECTED_MODEL_COUNT:
        raise ValueError("evaluation config must contain exactly 12 models")
    configured_ids = [
        _required_string(raw, "model_id", source="evaluation model")
        if isinstance(raw, dict)
        else ""
        for raw in raw_models
    ]
    if (
        tuple(configured_ids) != training.model_ids
        or len(set(configured_ids)) != EXPECTED_MODEL_COUNT
    ):
        raise ValueError("evaluation model order must match training exactly")
    _validate_evaluation_stages(
        status_path,
        config_path=config_path,
        config_sha256=config_sha256,
        model_ids=training.model_ids,
    )
    manifest_root = _resolve_path(
        config.get("manifest_output_dir"), base=base, source="manifest_output_dir"
    )
    training_by_id = {model.model_id: model for model in training.models}
    summaries = tuple(
        _verify_model_evaluation(
            raw_model,
            base=base,
            manifest_root=manifest_root,
            training_model=training_by_id[model_id],
            training=training,
        )
        for raw_model, model_id in zip(raw_models, configured_ids, strict=True)
        if isinstance(raw_model, dict)
    )
    return EvaluationVerification(
        stage_count=EXPECTED_EVALUATION_STAGE_COUNT,
        models=summaries,
        evaluation_config=config_path,
        evaluation_config_sha256=config_sha256,
        evaluation_status=status_path,
        evaluation_status_sha256=sha256_file(status_path),
        runtime_snapshot_manifest=runtime_manifest,
        runtime_snapshot_files=runtime_files,
    )


def _validate_evaluation_status_config_binding(
    status_path: Path,
    *,
    config_path: Path,
    config_sha256: str,
) -> None:
    rows = _read_jsonl(status_path)
    if not rows:
        raise ValueError("evaluation status is empty")
    for row in rows:
        row_path = _resolve_path(
            row.get("config_path"),
            base=config_path.parent,
            source="evaluation status config_path",
        )
        if row_path != config_path or row.get("config_sha256") != config_sha256:
            raise ValueError("evaluation status config path or SHA-256 mismatch")


def _validate_runtime_evaluation_inputs(
    config_path: Path,
    *,
    config: Mapping[str, object],
    training: TrainingVerification,
) -> tuple[FileBinding, tuple[FileBinding, ...]]:
    root = config_path.parent.resolve()
    if config_path.name != RUNTIME_CONFIG_NAME:
        raise ValueError(f"evaluation config must be the runtime config {RUNTIME_CONFIG_NAME}")
    manifest_path = root / RUNTIME_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    raw_files = manifest.get("files")
    raw_sources = manifest.get("source_inputs")
    if (
        manifest.get("schema_version") != RUNTIME_SNAPSHOT_SCHEMA
        or not isinstance(raw_files, dict)
        or not isinstance(raw_sources, dict)
    ):
        raise ValueError("runtime snapshot manifest contract mismatch")
    if set(raw_files) != EXPECTED_RUNTIME_SNAPSHOT_FILES:
        raise ValueError("runtime snapshot producer file inventory mismatch")
    bindings: list[FileBinding] = []
    expected_entries = {Path(RUNTIME_MANIFEST_NAME)}
    for raw_relative, raw_binding in raw_files.items():
        if not isinstance(raw_relative, str) or not isinstance(raw_binding, dict):
            raise TypeError("runtime snapshot file bindings must be objects")
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_relative
            or raw_relative == RUNTIME_MANIFEST_NAME
        ):
            raise ValueError("runtime snapshot manifest contains an unsafe file path")
        path = _resolve_contained_path(
            raw_relative,
            base=root,
            source="runtime snapshot file",
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"runtime snapshot file is missing or symlinked: {path}")
        if set(raw_binding) != {"sha256", "size"}:
            raise ValueError(f"runtime snapshot file binding field mismatch: {path}")
        expected_sha = _required_sha256(
            raw_binding,
            "sha256",
            source="runtime snapshot file",
        )
        expected_size = raw_binding.get("size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"runtime snapshot file content mismatch: {path}")
        bindings.append(FileBinding(path, expected_sha))
        expected_entries.add(relative)
        expected_entries.update(parent for parent in relative.parents if parent != Path())
    actual_entries = {entry.relative_to(root) for entry in root.rglob("*")}
    if any(entry.is_symlink() for entry in root.rglob("*")):
        raise ValueError("runtime snapshot contains a symlink")
    if actual_entries != expected_entries:
        raise ValueError("runtime snapshot exact inventory mismatch")
    source_contents: dict[Path, bytes] = {}
    for raw_path, raw_sha in raw_sources.items():
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or not _is_sha256(raw_sha)
        ):
            raise ValueError("runtime snapshot source input binding is invalid")
        source_path = _canonical_artifact_path(
            raw_path,
            source="runtime snapshot source input",
        )
        contents = source_path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != raw_sha:
            raise ValueError(f"runtime snapshot source input changed: {source_path}")
        source_contents[source_path] = contents
    original_inputs = {
        training.training_jobs: training.training_jobs_sha256,
        training.training_status: training.training_status_sha256,
    }
    for original, expected_sha in original_inputs.items():
        if sha256_file(original) != expected_sha or raw_sources.get(str(original)) != expected_sha:
            raise ValueError(f"original training input drift or manifest mismatch: {original}")
    runtime_jobs = _resolve_path(
        config.get("training_jobs"), base=root, source="runtime training_jobs"
    )
    runtime_status = _resolve_path(
        config.get("training_status"), base=root, source="runtime training_status"
    )
    if runtime_jobs != root / RUNTIME_JOBS_NAME or runtime_status != root / RUNTIME_STATUS_NAME:
        raise ValueError("runtime config training snapshot path mismatch")
    if (
        runtime_jobs.read_bytes() != training.training_jobs.read_bytes()
        or runtime_status.read_bytes() != training.training_status.read_bytes()
    ):
        raise ValueError("runtime training snapshot differs from original training input")
    runtime_bytes = config_path.read_bytes()
    source_configs: list[Path] = []
    for source_path, contents in source_contents.items():
        try:
            document = json.loads(contents)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != EVALUATION_CONFIG_SCHEMA
        ):
            continue
        candidate = dict(document)
        candidate["training_jobs"] = str(runtime_jobs)
        candidate["training_status"] = str(runtime_status)
        encoded = (
            json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        if encoded == runtime_bytes:
            source_configs.append(source_path)
    if len(source_configs) != 1:
        raise ValueError("runtime config source input binding mismatch")
    expected_sources = {
        training.training_jobs,
        training.training_status,
        source_configs[0],
    }
    for binding in bindings:
        try:
            relative = binding.path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] != "scripts":
            continue
        matches = [
            source_path
            for source_path, contents in source_contents.items()
            if source_path.name == binding.path.name
            and hashlib.sha256(contents).hexdigest() == binding.sha256
        ]
        if len(matches) != 1:
            raise ValueError(f"runtime snapshot script source binding mismatch: {binding.path}")
        expected_sources.add(matches[0])
    if set(source_contents) != expected_sources:
        raise ValueError("runtime snapshot source input inventory mismatch")
    provenance = _validate_upstream_runtime_provenance(
        root / UPSTREAM_RUNTIME_PROVENANCE_NAME,
        config=config,
        training=training,
    )
    _validate_upstream_runtime_package(
        root / UPSTREAM_RUNTIME_PACKAGE_NAME,
        provenance=provenance,
    )
    return (
        FileBinding(manifest_path, sha256_file(manifest_path)),
        tuple(sorted(bindings, key=lambda binding: str(binding.path))),
    )


def _git_output(root: Path, *arguments: str) -> bytes:
    process = subprocess.run(  # noqa: S603 - fixed Git executable, no shell.
        ("git", "-C", str(root), *arguments),  # noqa: S607
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"upstream Git provenance command failed ({' '.join(arguments)}): {stderr}"
        )
    return process.stdout


def _validate_upstream_runtime_provenance(
    path: Path,
    *,
    config: Mapping[str, object],
    training: TrainingVerification,
) -> dict[str, object]:
    document = _read_json(path)
    _require_exact_keys(
        document,
        {"schema_version", "upstream_root", "commit", "tree", "package", "python_files"},
        source="upstream runtime provenance",
    )
    if document.get("schema_version") != UPSTREAM_RUNTIME_PROVENANCE_SCHEMA:
        raise ValueError("upstream runtime provenance schema mismatch")
    if training.upstream_commit != PINNED_UPSTREAM_COMMIT:
        raise ValueError("training upstream commit does not match pinned Irodori commit")
    root = _resolve_path(
        document.get("upstream_root"),
        base=path.parent,
        source="upstream runtime root",
    )
    configured_root = _resolve_path(
        config.get("upstream_root"),
        base=path.parent,
        source="evaluation upstream root",
    )
    if root != configured_root or str(root) != document.get("upstream_root"):
        raise ValueError("upstream runtime root binding mismatch")
    commit = _required_string(document, "commit", source="upstream runtime provenance")
    tree = _required_string(document, "tree", source="upstream runtime provenance")
    if commit != PINNED_UPSTREAM_COMMIT or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ValueError("upstream runtime commit or tree binding mismatch")
    if document.get("package") != "irodori_tts":
        raise ValueError("upstream runtime package binding mismatch")
    top_level = _require_directory(
        Path(_git_output(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()),
        source="upstream Git top-level",
    ).resolve(strict=True)
    head = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    current_tree = _git_output(root, "rev-parse", GIT_HEAD_TREE).decode("ascii").strip()
    if top_level != root or head != commit or current_tree != tree:
        raise ValueError("current upstream Git identity does not match runtime provenance")
    package_status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "irodori_tts",
    ).decode("utf-8", errors="replace")
    if package_status:
        raise ValueError("current upstream irodori_tts package is dirty or untracked")
    raw_files = document.get("python_files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("upstream runtime provenance requires Python file bindings")
    tracked = _git_output(root, "ls-files", "-z", "--", "irodori_tts")
    expected_paths = sorted(
        item for item in tracked.decode("utf-8").split("\0") if item and item.endswith(".py")
    )
    actual_paths: list[str] = []
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise TypeError("upstream runtime Python file binding must be an object")
        _require_exact_keys(raw, {"path", "sha256"}, source="upstream runtime Python file")
        relative = _required_string(raw, "path", source="upstream runtime Python file")
        if (
            Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or not relative.startswith("irodori_tts/")
            or not relative.endswith(".py")
        ):
            raise ValueError("upstream runtime Python file path is invalid")
        candidate = _resolve_contained_path(
            relative,
            base=root,
            source="upstream runtime Python file",
        )
        expected_sha = _required_sha256(raw, "sha256", source="upstream runtime Python file")
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or sha256_file(candidate) != expected_sha
        ):
            raise ValueError(f"upstream runtime Python file hash mismatch: {candidate}")
        actual_paths.append(relative)
    if actual_paths != expected_paths or len(set(actual_paths)) != len(actual_paths):
        raise ValueError("upstream runtime Python file inventory mismatch")
    return document


def _validate_upstream_runtime_package(
    path: Path,
    *,
    provenance: Mapping[str, object],
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"upstream runtime package archive is missing or symlinked: {path}")
    raw_files = provenance.get("python_files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("upstream runtime package provenance requires Python files")
    expected: dict[str, str] = {}
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise TypeError("upstream runtime package binding must be an object")
        relative = _required_string(raw, "path", source="upstream runtime package binding")
        expected_sha = _required_sha256(
            raw,
            "sha256",
            source="upstream runtime package binding",
        )
        expected[relative] = expected_sha
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != sorted(expected) or len(set(names)) != len(names):
                raise ValueError("upstream runtime package archive inventory mismatch")
            if archive.comment:
                raise ValueError("upstream runtime package archive comment must be empty")
            contents: list[tuple[str, bytes]] = []
            for info in infos:
                if (
                    info.is_dir()
                    or info.filename not in expected
                    or info.date_time != ZIP_TIMESTAMP
                    or info.create_system != ZIP_CREATE_SYSTEM_UNIX
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or info.flag_bits & 0x1
                    or info.external_attr >> 16 != ZIP_UNIX_MODE
                    or info.extra
                    or info.comment
                ):
                    raise ValueError(
                        f"upstream runtime package archive entry is invalid: {info.filename}"
                    )
                content = archive.read(info)
                if hashlib.sha256(content).hexdigest() != expected[info.filename]:
                    raise ValueError(
                        f"upstream runtime package archive hash mismatch: {info.filename}"
                    )
                contents.append((info.filename, content))
        canonical = io.BytesIO()
        with zipfile.ZipFile(
            canonical,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=ZIP_COMPRESSLEVEL,
            strict_timestamps=True,
        ) as rebuilt:
            for relative, content in contents:
                info = zipfile.ZipInfo(relative, date_time=ZIP_TIMESTAMP)
                info.create_system = ZIP_CREATE_SYSTEM_UNIX
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = ZIP_UNIX_MODE << 16
                info.internal_attr = 0
                rebuilt.writestr(
                    info,
                    content,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=ZIP_COMPRESSLEVEL,
                )
        if canonical.getvalue() != path.read_bytes():
            raise ValueError("upstream runtime package archive is not canonical")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"upstream runtime package archive is invalid: {path}") from exc


def _validate_evaluation_stages(
    status_path: Path,
    *,
    config_path: Path,
    config_sha256: str,
    model_ids: Sequence[str],
) -> None:
    config = _read_json(config_path)
    expected = {"manifests"} | {
        f"{model_id}:{stage}"
        for model_id in model_ids
        for stage in EXPECTED_EVALUATION_STAGES_PER_MODEL
    }
    latest: dict[str, Mapping[str, object]] = {}
    for row in _read_jsonl(status_path):
        stage = row.get("stage")
        if isinstance(stage, str):
            latest[stage] = row
    if set(latest) != expected:
        raise ValueError("evaluation status stage set mismatch")
    for stage in sorted(expected):
        stage_row = latest[stage]
        expected_model_id = None if stage == "manifests" else stage.split(":", 1)[0]
        if not (
            stage_row.get("schema_version") == EVALUATION_STATUS_SCHEMA
            and stage_row.get("event") == "finished"
            and stage_row.get("status") == "success"
            and type(stage_row.get("exit_code")) is int
            and stage_row.get("exit_code") == 0
            and stage_row.get("config_sha256") == config_sha256
            and _is_sha256(stage_row.get("stage_fingerprint"))
            and stage_row.get("model_id") == expected_model_id
        ):
            raise ValueError(f"evaluation stage is not current and successful: {stage}")
        row_config = _resolve_path(
            stage_row.get("config_path"), base=config_path.parent, source="stage config_path"
        )
        if row_config != config_path:
            raise ValueError(f"evaluation stage config path mismatch: {stage}")
        contract = _expected_evaluation_stage_contract(
            config,
            stage=stage,
            base=config_path.parent,
        )
        component = stage_row.get("component_script")
        if contract.component_path is None:
            if component is not None:
                raise ValueError(f"evaluation producer component mismatch: {stage}")
        elif not isinstance(component, dict) or set(component) != {"path", "sha256"}:
            raise ValueError(f"evaluation producer component mismatch: {stage}")
        else:
            script = _resolve_path(
                component.get("path"), base=config_path.parent, source="component script"
            )
            if script != contract.component_path or component.get("sha256") != sha256_file(
                contract.component_path
            ):
                raise ValueError(f"evaluation producer component mismatch: {stage}")
        command = stage_row.get("command")
        if not isinstance(command, list) or tuple(command) != contract.command:
            raise ValueError(f"evaluation producer command mismatch: {stage}")
        expected_outputs = [snapshot_path(path) for path in contract.output_roots]
        if stage_row.get("outputs") != expected_outputs:
            raise ValueError(f"evaluation producer output mismatch: {stage}")
        if stage_row.get("stage_fingerprint") != _current_stage_fingerprint(
            config,
            stage_row,
            base=config_path.parent,
        ):
            raise ValueError(f"evaluation stage fingerprint is stale: {stage}")


def _current_stage_fingerprint(
    config: Mapping[str, object],
    row: Mapping[str, object],
    *,
    base: Path,
) -> str:
    stage = _required_string(row, "stage", source="evaluation stage")
    contract = _expected_evaluation_stage_contract(config, stage=stage, base=base)
    return _stage_contract_fingerprint(contract)


def _stage_contract_fingerprint(contract: EvaluationStageContract) -> str:
    payload = {
        "stage": contract.stage,
        "model_id": contract.model_id,
        "command": contract.command,
        "collision_paths": [
            str(_require_no_alias_components(path, source="stage collision path").resolve())
            for path in contract.collision_paths
        ],
        "input_files": [
            {
                "path": str(_require_regular_file(path, source="stage input").resolve(strict=True)),
                "sha256": sha256_file(path),
            }
            for path in contract.input_files
        ],
        "output_roots": [
            str(_require_no_alias_components(path, source="stage output root").resolve())
            for path in contract.output_roots
        ],
        "required_outputs": [
            str(_require_no_alias_components(path, source="stage required output").resolve())
            for path in contract.required_outputs
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _expected_evaluation_stage_contract(
    config: Mapping[str, object],
    *,
    stage: str,
    base: Path,
) -> EvaluationStageContract:
    raw_models = config.get("models")
    if not isinstance(raw_models, list):
        raise TypeError("evaluation config models must be a list")
    manifest_root = _resolve_path(
        config.get("manifest_output_dir"), base=base, source="manifest output"
    )
    scripts_dir = base / "scripts"
    if stage == "manifests":
        manifest_component_path = scripts_dir / STAGE_COMPONENT_NAMES[stage]
        references = tuple(
            _resolve_path(model.get("reference_wavs"), base=base, source="reference_wavs")
            for model in raw_models
            if isinstance(model, dict)
        )
        base_checkpoint = _required_mapping(config, "base_checkpoint", source="evaluation config")
        metric_models = _required_mapping(config, "metric_models", source="evaluation config")
        speaker = _required_mapping(
            metric_models, "speaker_embedding", source="evaluation metric_models"
        )
        transcription = _required_mapping(
            metric_models, "transcription", source="evaluation metric_models"
        )
        training_status = _resolve_path(
            config.get("training_status"), base=base, source="training status"
        )
        training_jobs = _resolve_path(
            config.get("training_jobs"), base=base, source="training jobs"
        )
        manifest_command = [
            sys.executable,
            str(manifest_component_path),
            "--training-status",
            str(training_status),
            "--training-jobs",
            str(training_jobs),
            "--output-dir",
            str(manifest_root),
            "--base-checkpoint",
            _required_string(base_checkpoint, "model_id", source="base checkpoint"),
            "--base-checkpoint-sha256",
            _required_sha256(base_checkpoint, "sha256", source="base checkpoint"),
            "--base-revision",
            _required_string(base_checkpoint, "revision", source="base checkpoint"),
            "--speaker-embedding-model-id",
            _required_string(speaker, "model_id", source="speaker embedding model"),
            "--speaker-embedding-revision",
            _required_string(speaker, "revision", source="speaker embedding model"),
            "--speaker-embedding-source-sha256",
            _required_sha256(speaker, "source_sha256", source="speaker embedding model"),
            "--transcription-model-id",
            _required_string(transcription, "model_id", source="transcription model"),
            "--transcription-revision",
            _required_string(transcription, "revision", source="transcription model"),
            "--transcription-source-sha256",
            _required_sha256(transcription, "source_sha256", source="transcription model"),
        ]
        for reference in references:
            manifest_command.extend(("--reference-wavs", str(reference)))
        return EvaluationStageContract(
            stage=stage,
            model_id=None,
            component_path=manifest_component_path,
            command=tuple(manifest_command),
            collision_paths=(manifest_root,),
            input_files=(
                manifest_component_path,
                training_status,
                training_jobs,
                *references,
            ),
            output_roots=(manifest_root,),
            required_outputs=(manifest_root / "manifest-index.json",),
        )
    model_id, separator, operation = stage.partition(":")
    if not separator or operation not in EXPECTED_EVALUATION_STAGES_PER_MODEL:
        raise ValueError(f"invalid evaluation stage name: {stage}")
    raw_model = next(
        (
            model
            for model in raw_models
            if isinstance(model, dict) and model.get("model_id") == model_id
        ),
        None,
    )
    if raw_model is None:
        raise ValueError(f"evaluation stage has unknown model: {stage}")
    reuse = raw_model.get("reuse")
    component_path = (
        None if isinstance(reuse, dict) else scripts_dir / STAGE_COMPONENT_NAMES[operation]
    )
    collision_paths, input_files, output_roots, required_outputs = _model_stage_paths(
        config,
        raw_model,
        operation=operation,
        component_path=component_path,
        manifest_root=manifest_root,
        base=base,
    )
    command = _expected_model_stage_command(
        config,
        raw_model,
        operation=operation,
        component_path=component_path,
        manifest_root=manifest_root,
        base=base,
    )
    return EvaluationStageContract(
        stage=stage,
        model_id=model_id,
        component_path=component_path,
        command=command,
        collision_paths=collision_paths,
        input_files=input_files,
        output_roots=output_roots,
        required_outputs=required_outputs,
    )


def _expected_model_stage_command(
    config: Mapping[str, object],
    model: Mapping[str, object],
    *,
    operation: str,
    component_path: Path | None,
    manifest_root: Path,
    base: Path,
) -> tuple[str, ...]:
    if isinstance(model.get("reuse"), dict):
        return ()
    if component_path is None:
        raise ValueError("queue-owned evaluation stage requires a component")
    model_id = _required_string(model, "model_id", source="evaluation model")
    generation = _resolve_path(model.get("generation_dir"), base=base, source="generation_dir")
    analysis = _resolve_path(model.get("analysis_dir"), base=base, source="analysis_dir")
    metrics = _resolve_path(model.get("metrics_dir"), base=base, source="metrics_dir")
    evaluation = _resolve_path(model.get("evaluation_dir"), base=base, source="evaluation_dir")
    reference = _resolve_path(model.get("reference_wavs"), base=base, source="reference_wavs")
    manifest = manifest_root / model_id / "evaluation-manifest.json"
    if operation == "generation":
        base_checkpoint = _required_mapping(config, "base_checkpoint", source="evaluation config")
        return (
            sys.executable,
            str(component_path),
            "generate",
            "--checkpoint-manifest",
            str(manifest),
            "--base-checkpoint-path",
            str(_resolve_path(base_checkpoint.get("path"), base=base, source="base checkpoint")),
            "--upstream-root",
            str(_resolve_path(config.get("upstream_root"), base=base, source="upstream root")),
            "--upstream-runtime-provenance",
            str(base / UPSTREAM_RUNTIME_PROVENANCE_NAME),
            "--upstream-package-archive",
            str(base / UPSTREAM_RUNTIME_PACKAGE_NAME),
            "--output-dir",
            str(generation),
            "--upstream-runtime-provenance-sha256",
            sha256_file(base / UPSTREAM_RUNTIME_PROVENANCE_NAME),
            "--upstream-package-archive-sha256",
            sha256_file(base / UPSTREAM_RUNTIME_PACKAGE_NAME),
        )
    if operation == "analysis":
        return (
            sys.executable,
            str(component_path),
            "--generation-dir",
            str(generation),
            "--output-dir",
            str(analysis),
        )
    if operation == "metrics":
        metric_models = _required_mapping(config, "metric_models", source="evaluation config")
        speaker = _required_mapping(
            metric_models, "speaker_embedding", source="evaluation metric_models"
        )
        transcription = _required_mapping(
            metric_models, "transcription", source="evaluation metric_models"
        )
        return (
            sys.executable,
            str(component_path),
            "--generation-results",
            str(generation / "generation-results.jsonl"),
            "--reference-wavs",
            str(reference),
            "--output",
            str(metrics / "metrics-results.jsonl"),
            "--provenance-output",
            str(metrics / "metrics-results.provenance.json"),
            "--ecapa-source",
            str(_resolve_path(speaker.get("source"), base=base, source="ECAPA source")),
            "--ecapa-savedir",
            str(_resolve_path(speaker.get("savedir"), base=base, source="ECAPA savedir")),
            "--ecapa-model-id",
            _required_string(speaker, "model_id", source="speaker embedding model"),
            "--ecapa-revision",
            _required_string(speaker, "revision", source="speaker embedding model"),
            "--whisper-model",
            _required_string(transcription, "model_id", source="transcription model"),
            "--whisper-source",
            str(
                _resolve_path(transcription.get("source"), base=base, source="transcription source")
            ),
            "--whisper-revision",
            _required_string(transcription, "revision", source="transcription model"),
            "--whisper-device",
            _required_string(transcription, "device", source="transcription model"),
        )
    return (
        sys.executable,
        str(component_path),
        "--generation-results",
        str(generation / "generation-results.jsonl"),
        "--analysis-results",
        str(analysis / "analysis-results.jsonl"),
        "--metrics-results",
        str(metrics / "metrics-results.jsonl"),
        "--metrics-provenance",
        str(metrics / "metrics-results.provenance.json"),
        "--evaluation-manifest",
        str(manifest),
        "--output-dir",
        str(evaluation),
    )


def _model_stage_paths(
    config: Mapping[str, object],
    model: Mapping[str, object],
    *,
    operation: str,
    component_path: Path | None,
    manifest_root: Path,
    base: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    model_id = _required_string(model, "model_id", source="evaluation model")
    reuse = model.get("reuse")
    if isinstance(reuse, dict):
        return _reused_stage_paths(reuse, operation=operation, base=base)
    if component_path is None:
        raise ValueError(f"queue-owned stage requires a component script: {model_id}:{operation}")
    generation = _resolve_path(model.get("generation_dir"), base=base, source="generation_dir")
    analysis = _resolve_path(model.get("analysis_dir"), base=base, source="analysis_dir")
    metrics = _resolve_path(model.get("metrics_dir"), base=base, source="metrics_dir")
    evaluation = _resolve_path(model.get("evaluation_dir"), base=base, source="evaluation_dir")
    reference = _resolve_path(model.get("reference_wavs"), base=base, source="reference_wavs")
    manifest = manifest_root / model_id / "evaluation-manifest.json"
    if operation == "generation":
        base_checkpoint = config.get("base_checkpoint")
        if not isinstance(base_checkpoint, dict):
            raise TypeError("evaluation base_checkpoint must be an object")
        inputs = (
            component_path,
            manifest,
            _resolve_path(base_checkpoint.get("path"), base=base, source="base checkpoint"),
            base / UPSTREAM_RUNTIME_PROVENANCE_NAME,
            base / UPSTREAM_RUNTIME_PACKAGE_NAME,
        )
        return (
            (generation,),
            inputs,
            (generation,),
            (
                generation / "generation-results.jsonl",
                generation / "generation-verification.json",
            ),
        )
    if operation == "analysis":
        return (
            (analysis,),
            (
                component_path,
                generation / "generation-results.jsonl",
                generation / "generation-verification.json",
            ),
            (analysis,),
            (analysis / "analysis-results.jsonl",),
        )
    if operation == "metrics":
        return (
            (metrics,),
            (component_path, generation / "generation-results.jsonl", reference),
            (metrics,),
            (
                metrics / "metrics-results.jsonl",
                metrics / "metrics-results.provenance.json",
            ),
        )
    return (
        (evaluation,),
        (
            component_path,
            generation / "generation-results.jsonl",
            analysis / "analysis-results.jsonl",
            metrics / "metrics-results.jsonl",
            metrics / "metrics-results.provenance.json",
            manifest,
        ),
        (evaluation,),
        (
            evaluation / "selected-models.json",
            evaluation / "evaluation-verification.json",
        ),
    )


def _reused_stage_paths(
    reuse: Mapping[str, object],
    *,
    operation: str,
    base: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    outputs: tuple[Path, ...]
    if operation == "generation":
        generation = _resolve_path(
            reuse.get("generation_dir"), base=base, source="reused generation_dir"
        )
        proof_candidates = (
            generation / "generation-verification.json",
            generation / "canonicalization-report.json",
        )
        existing_proofs = tuple(path for path in proof_candidates if path.is_file())
        if len(existing_proofs) != 1:
            raise ValueError("exactly one generation proof required for reused evaluation")
        proof = existing_proofs[0]
        outputs = (generation / "generation-results.jsonl", proof)
    elif operation == "analysis":
        analysis = _resolve_path(reuse.get("analysis_dir"), base=base, source="reused analysis_dir")
        outputs = (analysis / "analysis-results.jsonl",)
    elif operation == "metrics":
        outputs = (
            _resolve_path(reuse.get("metrics_results"), base=base, source="reused metrics results"),
            _resolve_path(
                reuse.get("metrics_provenance"),
                base=base,
                source="reused metrics provenance",
            ),
        )
    else:
        evaluation = _resolve_path(
            reuse.get("evaluation_dir"), base=base, source="reused evaluation_dir"
        )
        return (
            (),
            (),
            (evaluation,),
            (
                evaluation / "selected-models.json",
                evaluation / "evaluation-verification.json",
            ),
        )
    return (), (), outputs, outputs


def _verify_model_evaluation(
    model: Mapping[str, object],
    *,
    base: Path,
    manifest_root: Path,
    training_model: TrainingModelSummary,
    training: TrainingVerification,
) -> EvaluationModelSummary:
    model_id = training_model.model_id
    reuse = model.get("reuse")
    reuse_mapping = reuse if isinstance(reuse, dict) else None
    evaluation_dir = _resolve_path(
        (reuse_mapping or model).get("evaluation_dir"),
        base=base,
        source=f"evaluation_dir for {model_id}",
    )
    manifest_value = (reuse_mapping or {}).get("evaluation_manifest")
    manifest_path = (
        _resolve_path(manifest_value, base=base, source="evaluation_manifest")
        if manifest_value is not None
        else _require_no_alias_components(
            manifest_root / model_id / "evaluation-manifest.json",
            source="evaluation_manifest",
        ).resolve()
    )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != "speaker-checkpoint-evaluation-manifest/v1"
        or manifest.get("text_ids") != list(EXPECTED_TEXT_IDS)
        or manifest.get("seeds") != list(EXPECTED_SEEDS)
        or manifest.get("styles") != list(EXPECTED_STYLES)
    ):
        raise ValueError(f"evaluation manifest dimensions mismatch for {model_id}")
    raw_manifest_models = manifest.get("models")
    if not isinstance(raw_manifest_models, list) or len(raw_manifest_models) != 1:
        raise ValueError(f"evaluation manifest model identity mismatch for {model_id}")
    manifest_model = raw_manifest_models[0]
    if not isinstance(manifest_model, dict) or manifest_model.get("model_id") != model_id:
        raise ValueError(f"evaluation manifest model identity mismatch for {model_id}")
    raw_checkpoints = manifest_model.get("checkpoints")
    if not isinstance(raw_checkpoints, list):
        raise TypeError(f"evaluation manifest checkpoints must be a list for {model_id}")
    steps = tuple(
        checkpoint.get("checkpoint_step")
        for checkpoint in raw_checkpoints
        if isinstance(checkpoint, dict)
    )
    if steps != EXPECTED_EVALUATION_STEPS or len(raw_checkpoints) != len(EXPECTED_EVALUATION_STEPS):
        raise ValueError(f"evaluation checkpoint set mismatch for {model_id}")
    training_checkpoints = {
        checkpoint.path.name: checkpoint for checkpoint in training_model.checkpoints
    }
    contracts: dict[int, Mapping[str, object]] = {}
    for raw in raw_checkpoints:
        if not isinstance(raw, dict):
            raise TypeError(f"evaluation checkpoint entry must be an object for {model_id}")
        step = raw.get("checkpoint_step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError(f"evaluation checkpoint step must be an integer for {model_id}")
        embedding = _resolve_path(
            raw.get("embedding_path"), base=manifest_path.parent, source="embedding"
        )
        expected_training = training_checkpoints.get(f"checkpoint_{step:07d}.speaker.safetensors")
        if expected_training is None or embedding != expected_training.path:
            raise ValueError(f"evaluation embedding path does not match training for {model_id}")
        expected = {
            "embedding_sha256": expected_training.sha256,
            "training_config_sha256": training_model.config_sha256,
            "base_checkpoint_sha256": training.base_checkpoint_sha256,
            "base_revision": training.checkpoint_revision,
            "run_id": training_model.run_id,
        }
        if any(raw.get(field) != value for field, value in expected.items()):
            raise ValueError(f"evaluation checkpoint provenance mismatch for {model_id}")
        if sha256_file(embedding) != expected_training.sha256:
            raise ValueError(f"evaluation checkpoint file changed for {model_id}")
        contracts[step] = raw
    results_path = evaluation_dir / "evaluation-results.jsonl"
    results = _read_jsonl(results_path)
    validate_case_matrix(results, model_id=model_id)
    selected_path = evaluation_dir / "selected-models.json"
    selected_document = _read_json(selected_path)
    selections = selected_document.get("selections")
    if (
        selected_document.get("schema_version") != "speaker-checkpoint-evaluation/v1"
        or not isinstance(selections, list)
        or len(selections) != 1
        or not isinstance(selections[0], dict)
    ):
        raise ValueError(f"evaluation must contain exactly one selection for {model_id}")
    selected = selections[0]
    selected_step = selected.get("checkpoint_step")
    contract = contracts.get(selected_step) if isinstance(selected_step, int) else None
    if contract is None or any(selected.get(field) != value for field, value in contract.items()):
        raise ValueError(f"selected checkpoint identity mismatch for {model_id}")
    verification_path = evaluation_dir / "evaluation-verification.json"
    verification = _read_json(verification_path)
    if (
        verification.get("schema_version") != EVALUATION_VERIFICATION_SCHEMA
        or verification.get("status") != "PASS"
        or verification.get("selected") != selected
        or verification.get("checkpoint_count") != len(EXPECTED_EVALUATION_STEPS)
        or verification.get("evaluation_case_count") != EXPECTED_EVALUATION_CASE_COUNT
        or verification.get("hard_gate_metric_case_count_per_checkpoint")
        != EXPECTED_HARD_GATE_METRIC_CASE_COUNT_PER_CHECKPOINT
        or verification.get("diagnostic_word_case_count_per_checkpoint")
        != EXPECTED_DIAGNOSTIC_WORD_CASE_COUNT_PER_CHECKPOINT
    ):
        raise ValueError(f"evaluation verification did not pass for {model_id}")
    _validate_artifact_hashes(
        verification.get("artifact_sha256"),
        base=evaluation_dir,
        required_paths=(
            results_path,
            evaluation_dir / "checkpoint-summary.jsonl",
            evaluation_dir / "review-candidates.jsonl",
            evaluation_dir / "evaluation-config.json",
            selected_path,
            *(path for path in (evaluation_dir / "review_packet").rglob("*") if path.is_file()),
        ),
    )
    candidates_path = evaluation_dir / "review-candidates.jsonl"
    candidates = tuple(_read_jsonl(candidates_path))
    packet_path = evaluation_dir / "review_packet" / "manifest.json"
    _validate_review_packet(candidates, packet_path=packet_path)
    packet_manifest_path = _require_regular_file(
        packet_path,
        source="review packet manifest",
    ).resolve(strict=True)
    packet_assets_list: list[FileBinding] = []
    for path in sorted(packet_path.parent.rglob("*")):
        if not path.is_file():
            continue
        resolved_asset = _require_regular_file(
            path,
            source="review packet asset",
        ).resolve(strict=True)
        if resolved_asset != packet_manifest_path:
            packet_assets_list.append(FileBinding(resolved_asset, sha256_file(path)))
    packet_assets = tuple(packet_assets_list)
    resolved_verification = _require_regular_file(
        verification_path,
        source="evaluation verification",
    ).resolve(strict=True)
    resolved_results = _require_regular_file(
        results_path,
        source="evaluation results",
    ).resolve(strict=True)
    resolved_candidates = _require_regular_file(
        candidates_path,
        source="review candidates",
    ).resolve(strict=True)
    resolved_selected = _require_regular_file(
        selected_path,
        source="selected models",
    ).resolve(strict=True)
    return EvaluationModelSummary(
        model_id=model_id,
        evaluation_dir=evaluation_dir,
        manifest_path=manifest_path,
        case_count=len(results),
        selected=dict(selected),
        review_candidates=candidates,
        manifest_sha256=sha256_file(manifest_path),
        evaluation_verification=FileBinding(resolved_verification, sha256_file(verification_path)),
        evaluation_results=FileBinding(resolved_results, sha256_file(results_path)),
        review_candidates_file=FileBinding(resolved_candidates, sha256_file(candidates_path)),
        review_packet_manifest=FileBinding(packet_manifest_path, sha256_file(packet_path)),
        review_packet_assets=packet_assets,
        selected_file=FileBinding(resolved_selected, sha256_file(selected_path)),
    )


def _validate_artifact_hashes(
    raw: object,
    *,
    base: Path,
    required_paths: Sequence[Path],
) -> None:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("evaluation verification artifact hashes are missing")
    bound: set[Path] = set()
    for raw_path, raw_sha in raw.items():
        if not isinstance(raw_path, str) or not _is_sha256(raw_sha):
            raise ValueError("evaluation verification artifact hash is invalid")
        path = _resolve_path(raw_path, base=base, source="evaluation artifact")
        if sha256_file(path) != raw_sha:
            raise ValueError(f"evaluation artifact hash mismatch: {path}")
        bound.add(path)
    missing = {
        _require_regular_file(path, source="required evaluation artifact").resolve(strict=True)
        for path in required_paths
    } - bound
    if missing:
        raise ValueError(f"evaluation verification omits artifact hash: {min(missing)}")


def _validate_review_packet(
    candidates: Sequence[Mapping[str, object]],
    *,
    packet_path: Path,
) -> None:
    packet = _read_json(packet_path)
    raw_packet_candidates = packet.get("review_candidates")
    if packet.get("schema_version") != REVIEW_PACKET_SCHEMA or not isinstance(
        raw_packet_candidates, list
    ):
        raise ValueError(f"review packet is invalid: {packet_path}")
    candidate_ids = [
        _required_string(row, "case_id", source="review candidate") for row in candidates
    ]
    packet_ids = [
        _required_string(row, "case_id", source="review packet candidate")
        if isinstance(row, dict)
        else ""
        for row in raw_packet_candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)) or packet_ids != candidate_ids:
        raise ValueError("review packet candidate set mismatch")
    root = packet_path.parent
    for candidate, packet_candidate in zip(candidates, raw_packet_candidates, strict=True):
        if not isinstance(packet_candidate, dict):
            raise TypeError("review packet candidate must be an object")
        wav = packet_candidate.get("wav")
        if not isinstance(wav, dict):
            raise TypeError(f"review packet candidate WAV is missing: {candidate['case_id']}")
        _validate_packet_asset(wav, root=root, expected_sha=candidate.get("wav_sha256"))
        source_path = wav.get("source_path")
        if source_path is not None and source_path != candidate.get("wav_path"):
            raise ValueError(f"review packet WAV source mismatch: {candidate['case_id']}")
        spectrogram = packet_candidate.get("spectrogram")
        if spectrogram is not None:
            if not isinstance(spectrogram, dict):
                raise TypeError("review packet spectrogram must be an object")
            _validate_packet_asset(spectrogram, root=root, expected_sha=None)
        controls = packet_candidate.get("paired_controls")
        if not isinstance(controls, list):
            raise TypeError("review packet paired_controls must be a list")
        raw_candidate_controls = candidate.get("paired_controls", [])
        if not isinstance(raw_candidate_controls, list):
            raise TypeError("review candidate paired_controls must be a list")
        candidate_controls = {
            control.get("case_id"): control
            for control in raw_candidate_controls
            if isinstance(control, dict)
        }
        packet_control_ids = [
            control.get("case_id") if isinstance(control, dict) else None for control in controls
        ]
        if packet_control_ids != list(candidate_controls):
            raise ValueError(f"review packet paired control set mismatch: {candidate['case_id']}")
        for control in controls:
            if not isinstance(control, dict) or not isinstance(control.get("wav"), dict):
                raise TypeError("review packet paired control must contain a WAV")
            candidate_control = candidate_controls[control["case_id"]]
            _validate_packet_asset(
                control["wav"],
                root=root,
                expected_sha=candidate_control.get("wav_sha256"),
            )


def _validate_packet_asset(
    asset: Mapping[str, object],
    *,
    root: Path,
    expected_sha: object,
) -> None:
    path = _resolve_contained_path(asset.get("path"), base=root, source="review packet asset")
    declared = _required_sha256(asset, "sha256", source="review packet asset")
    if expected_sha is not None and declared != expected_sha:
        raise ValueError(f"review packet asset candidate SHA-256 mismatch: {path}")
    if sha256_file(path) != declared:
        raise ValueError(f"review packet copied asset changed: {path}")


def verify_reviews(
    evaluations: EvaluationVerification,
    decisions_path: Path,
) -> ReviewVerification:
    path = _require_regular_file(
        decisions_path,
        source="review decisions",
    ).resolve(strict=True)
    candidates: dict[str, tuple[EvaluationModelSummary, Mapping[str, object]]] = {}
    for model in evaluations.models:
        _validate_review_packet(
            model.review_candidates,
            packet_path=model.evaluation_dir / "review_packet" / "manifest.json",
        )
        for candidate in model.review_candidates:
            case_id = _required_string(candidate, "case_id", source="review candidate")
            if case_id in candidates:
                raise ValueError(f"duplicate review candidate case_id: {case_id}")
            if candidate.get("model_id") != model.model_id:
                raise ValueError(f"review candidate model identity mismatch: {case_id}")
            _required_int(candidate, "checkpoint_step", source="review candidate")
            _required_sha256(candidate, "wav_sha256", source="review candidate")
            candidates[case_id] = (model, candidate)
    decisions = _read_jsonl(path)
    by_case: dict[str, Mapping[str, object]] = {}
    expected_fields = {
        "schema_version",
        "case_id",
        "model_id",
        "checkpoint_step",
        "wav_sha256",
        "reviewer",
        "reviewed_at",
        "decision",
    }
    for decision in decisions:
        if set(decision) != expected_fields:
            raise ValueError("review decision field set mismatch")
        if decision.get("schema_version") != REVIEW_DECISION_SCHEMA:
            raise ValueError("review decision schema_version mismatch")
        case_id = _required_string(decision, "case_id", source="review decision")
        if case_id in by_case:
            raise ValueError(f"duplicate review decision: {case_id}")
        candidate_entry = candidates.get(case_id)
        if candidate_entry is None:
            raise ValueError(f"extra review decision: {case_id}")
        _model, candidate = candidate_entry
        for field in ("model_id", "checkpoint_step", "wav_sha256"):
            if decision.get(field) != candidate.get(field):
                raise ValueError(f"stale review decision identity for {case_id}: {field}")
        reviewer = decision.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"review decision reviewer is missing: {case_id}")
        reviewed_at = _required_string(decision, "reviewed_at", source="review decision")
        parsed = _parse_datetime(reviewed_at, source="review decision reviewed_at")
        if parsed.utcoffset() is None:
            raise ValueError(f"review decision timestamp must be timezone-aware: {case_id}")
        if decision.get("decision") not in REVIEW_DECISIONS:
            raise ValueError(f"review decision enum is invalid: {case_id}")
        by_case[case_id] = decision
    unresolved = tuple(sorted(set(candidates) - set(by_case)))
    grouped: dict[str, dict[str, dict[str, int]]] = {}
    for case_id, reviewed_decision in by_case.items():
        model, candidate = candidates[case_id]
        checkpoint_step = _required_int(candidate, "checkpoint_step", source="review candidate")
        outcome = str(reviewed_decision["decision"])
        selected_step = model.selected.get("checkpoint_step")
        if checkpoint_step == selected_step and outcome in {"TONE", "UNSURE"}:
            raise ValueError(
                f"selected checkpoint has disallowed {outcome} review decision: {case_id}"
            )
        model_counts = grouped.setdefault(model.model_id, {})
        checkpoint_counts = model_counts.setdefault(str(checkpoint_step), {})
        checkpoint_counts[outcome] = checkpoint_counts.get(outcome, 0) + 1
    status = "AWAITING_REVIEW" if unresolved else "PASS"
    return ReviewVerification(
        status=status,
        candidate_count=len(candidates),
        decision_count=len(by_case),
        unresolved_ids=unresolved,
        grouped_decisions=grouped,
        decisions_path=path,
        decisions_sha256=sha256_file(path),
    )


def verify_staging(
    evaluations: EvaluationVerification,
    staging_report: Path,
) -> StagingVerification:
    path = _require_regular_file(staging_report, source="staging report").resolve(strict=True)
    report = _read_json(path)
    if (
        report.get("schema_version") != STAGING_SCHEMA
        or report.get("status") != "PASS"
        or report.get("model_count") != EXPECTED_MODEL_COUNT
    ):
        raise ValueError("staging report completion contract mismatch")
    for field, expected in REQUIRED_NON_DEPLOYMENT_VALUES.items():
        if report.get(field) is not expected:
            raise ValueError(f"staging report non-deployment flag mismatch: {field}")
    raw_selections = report.get("selections")
    if not isinstance(raw_selections, list) or len(raw_selections) != EXPECTED_MODEL_COUNT:
        raise ValueError("staging report must contain exactly 12 selections")
    expected_models = {model.model_id: model for model in evaluations.models}
    seen: set[str] = set()
    for selection in raw_selections:
        if not isinstance(selection, dict):
            raise TypeError("staging selection must be an object")
        model_id = _required_string(selection, "model_id", source="staging selection")
        if model_id in seen or model_id not in expected_models:
            raise ValueError(f"staging selection model set mismatch: {model_id}")
        seen.add(model_id)
        selected_expected = expected_models[model_id].selected
        if any(selection.get(field) != value for field, value in selected_expected.items()):
            raise ValueError(f"staging selection identity mismatch: {model_id}")
        embedding = _resolve_existing_file_path(
            selection.get("embedding_path"), base=path.parent, source="staging embedding"
        )
        declared = _required_sha256(selection, "embedding_sha256", source="staging selection")
        if sha256_file(embedding) != declared:
            raise ValueError(f"staging selected embedding changed: {model_id}")
    if seen != set(expected_models):
        raise ValueError("staging selection model set mismatch")
    proposed_root_lexical = _require_absent_proposed_staging_root(
        report,
        report_path=path,
        source="proposed staging root",
    )
    baseline = _resolve_existing_file_path(
        report.get("active_voice_bank_snapshot"),
        base=path.parent,
        source="active voice bank snapshot",
    )
    baseline_sha = _required_sha256(
        report, "active_voice_bank_snapshot_sha256", source="staging report"
    )
    if sha256_file(baseline) != baseline_sha:
        raise ValueError("active voice bank baseline SHA-256 mismatch")
    current = _verify_voice_bank_baseline(baseline)
    if report.get("active_voice_bank_current") != current:
        raise ValueError("active voice bank current snapshot mismatch")
    return StagingVerification(
        model_count=len(raw_selections),
        selections=tuple(dict(selection) for selection in raw_selections),
        staging_report=path,
        staging_report_sha256=sha256_file(path),
        proposed_staging_root=proposed_root_lexical.resolve(strict=False),
        active_voice_bank_baseline=baseline,
        active_voice_bank_baseline_sha256=baseline_sha,
        active_voice_bank_current=current,
    )


def _verify_voice_bank_baseline(path: Path) -> dict[str, object]:
    baseline = _read_json(path)
    if baseline.get("schema_version") != VOICE_BANK_SNAPSHOT_SCHEMA:
        raise ValueError("voice bank baseline schema mismatch")
    root = _resolve_path(
        baseline.get("voice_bank_root"), base=path.parent, source="voice bank root"
    )
    raw_manifest = baseline.get("manifest")
    raw_speakers = baseline.get("speakers")
    if not isinstance(raw_manifest, dict) or not isinstance(raw_speakers, list):
        raise TypeError("voice bank baseline manifest and speakers are required")
    manifest = root / "voice_bank_speakers.toml"
    _validate_baseline_file(raw_manifest, manifest)
    expected_names: set[str] = set()
    for raw in raw_speakers:
        if not isinstance(raw, dict):
            raise TypeError("voice bank baseline speaker must be an object")
        name = _required_string(raw, "name", source="voice bank speaker")
        if name in expected_names:
            raise ValueError(f"duplicate voice bank baseline speaker: {name}")
        expected_names.add(name)
        _validate_baseline_file(raw, root / "speakers" / name)
    actual = tuple(sorted((root / "speakers").glob("*.speaker.safetensors")))
    if {speaker.name for speaker in actual} != expected_names:
        raise ValueError("active voice bank speaker inventory changed")
    if baseline.get("speaker_count") != len(actual):
        raise ValueError("voice bank baseline speaker_count mismatch")
    return {
        "root": str(root),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "speaker_count": len(actual),
        "speakers": [
            {
                "name": speaker.name,
                "path": str(speaker),
                "sha256": sha256_file(speaker),
                "size": speaker.stat().st_size,
            }
            for speaker in actual
        ],
    }


def _validate_baseline_file(binding: Mapping[str, object], actual: Path) -> None:
    declared_path = binding.get("path")
    resolved_actual = _require_regular_file(
        actual,
        source="active voice bank file",
    ).resolve(strict=True)
    if (
        declared_path is not None
        and _resolve_existing_file_path(
            str(declared_path),
            base=actual.parent,
            source="voice bank baseline path",
        )
        != resolved_actual
    ):
        raise ValueError(f"voice bank baseline path mismatch: {actual}")
    if binding.get("sha256") != sha256_file(actual) or binding.get("size") != actual.stat().st_size:
        raise ValueError(f"active voice bank file changed: {actual}")


def build_completion_report(
    *,
    phase: str,
    status: str,
    training: TrainingVerification | None = None,
    evaluations: EvaluationVerification | None = None,
    reviews: ReviewVerification | None = None,
    staging: StagingVerification | None = None,
    failure_reasons: Sequence[str] = (),
    verified_at: str | None = None,
) -> dict[str, object]:
    if phase not in {"training", "final"}:
        raise ValueError("completion phase must be training or final")
    if status not in COMPLETION_STATUSES:
        raise ValueError("completion status is invalid")
    script_path = Path(__file__).resolve()
    report: dict[str, object] = {
        "schema_version": COMPLETION_SCHEMA,
        "status": status,
        "phase": phase,
        "verified_at": verified_at or datetime.now(_UTC).isoformat(),
        "checks": _completion_checks(
            phase=phase,
            training=training,
            evaluations=evaluations,
            reviews=reviews,
            staging=staging,
            failure_reasons=failure_reasons,
        ),
        "verifier": {"path": str(script_path), "sha256": sha256_file(script_path)},
    }
    if training is not None:
        report["training"] = _training_report(training)
    if evaluations is not None:
        report["evaluations"] = _evaluation_report(evaluations)
    if reviews is not None:
        report["reviews"] = {
            "status": reviews.status,
            "candidate_count": reviews.candidate_count,
            "decision_count": reviews.decision_count,
            "unresolved_ids": list(reviews.unresolved_ids),
            "grouped_decisions": reviews.grouped_decisions,
            "decisions": _binding(reviews.decisions_path, reviews.decisions_sha256),
        }
    if staging is not None:
        report["staging"] = {
            "model_count": staging.model_count,
            "report": _binding(staging.staging_report, staging.staging_report_sha256),
            "active_voice_bank_baseline": _binding(
                staging.active_voice_bank_baseline,
                staging.active_voice_bank_baseline_sha256,
            ),
            "active_voice_bank_current": staging.active_voice_bank_current,
            "selections": list(staging.selections),
            "proposed_staging_root": str(staging.proposed_staging_root),
            **REQUIRED_NON_DEPLOYMENT_VALUES,
        }
    return report


def _completion_checks(
    *,
    phase: str,
    training: TrainingVerification | None,
    evaluations: EvaluationVerification | None,
    reviews: ReviewVerification | None,
    staging: StagingVerification | None,
    failure_reasons: Sequence[str],
) -> dict[str, dict[str, object]]:
    names = (
        ("training",)
        if phase == "training"
        else (
            "training",
            "evaluations",
            "reviews",
            "staging",
        )
    )
    values: dict[str, object | None] = {
        "training": training,
        "evaluations": evaluations,
        "reviews": reviews,
        "staging": staging,
    }
    checks: dict[str, dict[str, object]] = {}
    failure_assigned = False
    for name in names:
        value = values[name]
        passed = value is not None
        reasons: list[str] = []
        if name == "reviews" and reviews is not None and reviews.status != "PASS":
            passed = False
            reasons.append("review decisions are unresolved")
        if not passed and failure_reasons and not failure_assigned:
            reasons.extend(failure_reasons)
            failure_assigned = True
        checks[name] = {"passed": passed, "reasons": reasons}
    return checks


def _training_report(training: TrainingVerification) -> dict[str, object]:
    models: list[dict[str, object]] = []
    for model in training.models:
        model_report: dict[str, object] = {
            "model_id": model.model_id,
            "checkpoint_count": model.checkpoint_count,
            "loss_event_count": model.loss_event_count,
            "config": _binding(model.config_path, model.config_sha256),
            "clean_manifest": _binding(model.clean_manifest_path, model.clean_manifest_sha256),
            "log": _binding(model.log_path, model.log_sha256),
            "output_dir": str(model.output_dir),
            "latest_status": dict(model.latest_status),
            "run_id": model.run_id,
            "checkpoints": [
                _binding(checkpoint.path, checkpoint.sha256) for checkpoint in model.checkpoints
            ],
        }
        if model.run_evidence_lineage:
            model_report["run_evidence_lineage"] = [
                _quality_run_lineage_report(lineage) for lineage in model.run_evidence_lineage
            ]
        models.append(model_report)
    report: dict[str, object] = {
        "jobs": _binding(training.training_jobs, training.training_jobs_sha256),
        "status": _binding(training.training_status, training.training_status_sha256),
        "launch_evidence": _binding(
            training.training_launch_evidence, training.training_launch_evidence_sha256
        ),
        "base_checkpoint": _binding(training.base_checkpoint, training.base_checkpoint_sha256),
        "checkpoint_revision": training.checkpoint_revision,
        "upstream_commit": training.upstream_commit,
        "runtime": _jsonable(asdict(training.runtime_snapshot)),
        "models": models,
    }
    if training.base_training_jobs is not None:
        report["base_jobs"] = _file_binding(training.base_training_jobs)
    if training.base_training_status is not None:
        report["base_status"] = _file_binding(training.base_training_status)
    if training.training_run_evidence:
        report["run_evidence"] = [
            _quality_run_lineage_report(lineage) for lineage in training.training_run_evidence
        ]
    return report


def _quality_run_lineage_report(lineage: QualityRunLineage) -> dict[str, object]:
    return {
        "model_id": lineage.model_id,
        "evidence": _file_binding(lineage.evidence),
        "setup_evidence": _file_binding(lineage.setup_evidence),
        "training_jobs": _file_binding(lineage.training_jobs),
        "training_status": _file_binding(lineage.training_status),
        "queue_script": _file_binding(lineage.queue_script),
        "source_diagnostic": _file_binding(lineage.source_diagnostic),
        "initialization_checkpoint": _file_binding(lineage.initialization_checkpoint),
    }


def _evaluation_report(evaluations: EvaluationVerification) -> dict[str, object]:
    if evaluations.runtime_snapshot_manifest is None:
        raise ValueError("evaluation runtime snapshot audit binding is missing")
    models: list[dict[str, object]] = []
    for model in evaluations.models:
        if (
            model.manifest_sha256 is None
            or model.evaluation_verification is None
            or model.evaluation_results is None
            or model.review_candidates_file is None
            or model.review_packet_manifest is None
            or model.selected_file is None
        ):
            raise ValueError(f"evaluation audit bindings are incomplete for {model.model_id}")
        models.append(
            {
                "model_id": model.model_id,
                "evaluation_dir": str(model.evaluation_dir),
                "case_count": model.case_count,
                "manifest": _binding(model.manifest_path, model.manifest_sha256),
                "evaluation_verification": _file_binding(model.evaluation_verification),
                "evaluation_results": _file_binding(model.evaluation_results),
                "review_candidates": _file_binding(model.review_candidates_file),
                "review_candidate_count": len(model.review_candidates),
                "review_packet_manifest": _file_binding(model.review_packet_manifest),
                "review_packet_assets": [
                    _file_binding(asset) for asset in model.review_packet_assets
                ],
                "selected": {
                    "selection": dict(model.selected),
                    "artifact": _file_binding(model.selected_file),
                },
            }
        )
    return {
        "stage_count": evaluations.stage_count,
        "config": _binding(evaluations.evaluation_config, evaluations.evaluation_config_sha256),
        "status": _binding(evaluations.evaluation_status, evaluations.evaluation_status_sha256),
        "runtime_snapshot_manifest": _file_binding(evaluations.runtime_snapshot_manifest),
        "runtime_snapshot_files": [
            _file_binding(binding) for binding in evaluations.runtime_snapshot_files
        ],
        "models": models,
    }


def ensure_output_safe(output: Path, protected_paths: Iterable[Path]) -> None:
    resolved_output = _require_no_alias_components(
        output,
        source="completion report output",
    ).resolve(strict=False)
    candidates = (
        resolved_output,
        resolved_output.with_suffix(resolved_output.suffix + ".tmp"),
    )
    for protected in protected_paths:
        resolved_protected = _require_no_alias_components(
            protected,
            source="completion protected path",
        ).resolve(strict=False)
        for candidate in candidates:
            if (
                candidate == resolved_protected
                or candidate in resolved_protected.parents
                or resolved_protected in candidate.parents
            ):
                raise ValueError(
                    f"completion report output overlaps protected path: "
                    f"{candidate} and {resolved_protected}"
                )


def _require_absent_proposed_staging_root(
    report: Mapping[str, object],
    *,
    report_path: Path,
    source: str,
) -> Path:
    raw_proposed = report.get("proposed_staging_root")
    if not isinstance(raw_proposed, str) or not raw_proposed:
        raise ValueError("staging report proposed staging root must be a nonempty path")
    proposed = Path(raw_proposed).expanduser()
    if not proposed.is_absolute():
        proposed = report_path.parent / proposed
    if os.path.lexists(proposed):
        try:
            proposed.lstat()
        except OSError as exc:
            raise ValueError(f"{source} metadata is unavailable: {proposed}") from exc
        raise ValueError(f"{source} exists or is symlinked: {proposed}")
    return _require_no_alias_components(proposed, source=source)


def _preflight_staging_output(
    staging_report: Path,
    output: Path,
) -> StagingOutputPreflight:
    raw_report = staging_report.expanduser().absolute()
    lexical = _require_regular_file(raw_report, source="staging report preflight")
    resolved_report = lexical.resolve(strict=True)
    report = _read_json(lexical)
    proposed = _require_absent_proposed_staging_root(
        report,
        report_path=resolved_report,
        source="proposed staging root",
    )
    ensure_output_safe(output, (proposed,))
    return StagingOutputPreflight(
        staging_report_lexical=raw_report,
        staging_report=FileBinding(
            path=resolved_report,
            sha256=sha256_file(lexical),
        ),
        proposed_staging_root=proposed,
    )


def _revalidate_staging_output(
    preflight: StagingOutputPreflight,
    output: Path,
) -> None:
    lexical = _require_regular_file(
        preflight.staging_report_lexical,
        source="staging report publication recheck",
    )
    resolved_report = lexical.resolve(strict=True)
    if resolved_report != preflight.staging_report.path:
        raise ValueError("staging report changed after preflight: resolved path")
    if sha256_file(lexical) != preflight.staging_report.sha256:
        raise ValueError("staging report changed after preflight: SHA-256")
    report = _read_json(lexical)
    proposed = _require_absent_proposed_staging_root(
        report,
        report_path=resolved_report,
        source="proposed staging root publication recheck",
    )
    if proposed != preflight.proposed_staging_root:
        raise ValueError("staging report changed after preflight: proposed staging root")
    ensure_output_safe(output, (proposed,))


def _completion_protected_paths(
    args: argparse.Namespace,
    *,
    training: TrainingVerification | None,
    evaluations: EvaluationVerification | None,
    reviews: ReviewVerification | None,
    staging: StagingVerification | None,
) -> tuple[Path, ...]:
    protected: set[Path] = {
        path
        for path in (
            args.training_jobs,
            args.training_status,
            args.training_launch_evidence,
            args.evaluation_config,
            args.evaluation_status,
            args.review_decisions,
            args.staging_report,
        )
        if isinstance(path, Path)
    }
    protected.update(
        path for path in getattr(args, "training_run_evidence", ()) if isinstance(path, Path)
    )
    if training is not None:
        protected.update(
            {
                training.training_jobs,
                training.training_status,
                training.training_launch_evidence,
                training.base_checkpoint,
            }
        )
        if training.base_training_jobs is not None:
            protected.add(training.base_training_jobs.path)
        if training.base_training_status is not None:
            protected.add(training.base_training_status.path)
        for lineage in training.training_run_evidence:
            protected.update(
                {
                    lineage.evidence.path,
                    lineage.setup_evidence.path,
                    lineage.training_jobs.path,
                    lineage.training_status.path,
                    lineage.queue_script.path,
                    lineage.source_diagnostic.path,
                    lineage.initialization_checkpoint.path,
                }
            )
        launch = _read_json(training.training_launch_evidence)
        launcher = launch.get("launcher_script_path")
        if isinstance(launcher, str):
            protected.add(Path(launcher))
        for training_model in training.models:
            protected.update(
                {
                    training_model.config_path,
                    training_model.clean_manifest_path,
                    training_model.log_path,
                    training_model.output_dir,
                    *(checkpoint.path for checkpoint in training_model.checkpoints),
                }
            )
    if evaluations is not None:
        protected.update({evaluations.evaluation_config, evaluations.evaluation_status})
        if evaluations.runtime_snapshot_manifest is not None:
            protected.add(evaluations.runtime_snapshot_manifest.path)
        protected.update(binding.path for binding in evaluations.runtime_snapshot_files)
        for evaluation_model in evaluations.models:
            protected.update({evaluation_model.evaluation_dir, evaluation_model.manifest_path})
            for binding in (
                evaluation_model.evaluation_verification,
                evaluation_model.evaluation_results,
                evaluation_model.review_candidates_file,
                evaluation_model.review_packet_manifest,
                evaluation_model.selected_file,
            ):
                if binding is not None:
                    protected.add(binding.path)
            protected.update(asset.path for asset in evaluation_model.review_packet_assets)
        protected.update(_evaluation_snapshot_paths(evaluations.evaluation_status))
    if reviews is not None:
        protected.add(reviews.decisions_path)
    if staging is not None:
        protected.update(
            {
                staging.staging_report,
                staging.proposed_staging_root,
                staging.active_voice_bank_baseline,
            }
        )
        current_root = staging.active_voice_bank_current.get("root")
        if isinstance(current_root, str):
            protected.add(Path(current_root))
        current_manifest = staging.active_voice_bank_current.get("manifest_path")
        if isinstance(current_manifest, str):
            protected.add(Path(current_manifest))
        current_speakers = staging.active_voice_bank_current.get("speakers")
        if isinstance(current_speakers, list):
            for speaker in current_speakers:
                if isinstance(speaker, dict) and isinstance(speaker.get("path"), str):
                    protected.add(Path(str(speaker["path"])))
        for selection in staging.selections:
            embedding = selection.get("embedding_path")
            if isinstance(embedding, str):
                protected.add(Path(embedding))
    return tuple(sorted(protected, key=str))


def _evaluation_snapshot_paths(status_path: Path) -> set[Path]:
    paths: set[Path] = set()
    for row in _read_jsonl(status_path):
        outputs = row.get("outputs")
        if not isinstance(outputs, list):
            continue
        for snapshot in outputs:
            if isinstance(snapshot, dict) and isinstance(snapshot.get("path"), str):
                paths.add(Path(str(snapshot["path"])))
    return paths


def write_report_create_only(path: Path, payload: Mapping[str, object]) -> None:
    output = _require_no_alias_components(path, source="completion report output")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_jsonable(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor: int | None = None
    temporary_created = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        temporary_created = True
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, output)
        temporary.unlink()
        # Windows does not consistently allow opening directories for fsync.
        with contextlib.suppress(PermissionError):
            _fsync_directory(output.parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def probe_runtime(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    platform_name: str | None = None,
    current_pid: int | None = None,
) -> RuntimeSnapshot:
    execute = runner or subprocess.run
    platform = platform_name or os.name
    own_pid = current_pid if current_pid is not None else os.getpid()
    errors: list[str] = []
    processes: list[dict[str, object]] = []
    compute: list[dict[str, object]] = []
    gpu_memory: float | None = None
    try:
        process_command = (
            (
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "Get-CimInstance Win32_Process | Select-Object "
                    "ProcessId,ParentProcessId,CreationDate,CommandLine,Name | "
                    "ConvertTo-Json -Compress"
                ),
            )
            if platform == "nt"
            else ("ps", "-axo", "pid=,ppid=,lstart=,command=")
        )
        completed = execute(  # Fixed read-only process inventory.
            process_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode:
            errors.append(completed.stderr.strip() or "process inventory failed")
        elif platform == "nt":
            processes.extend(
                _parse_windows_process_inventory(completed.stdout, current_pid=own_pid)
            )
        else:
            for line in completed.stdout.splitlines():
                match = re.match(r"\s*(\d+)\s+(\d+)\s+(.{24})\s+(.*)", line)
                if match:
                    processes.append(
                        {
                            "pid": int(match.group(1)),
                            "parent_pid": int(match.group(2)),
                            "creation_time": match.group(3).strip(),
                            "command_line": match.group(4),
                        }
                    )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    excluded = _inventory_ancestor_pids(processes, current_pid=own_pid)
    try:
        completed = execute(  # Fixed read-only NVIDIA inventory.
            (
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode:
            errors.append(completed.stderr.strip() or "GPU memory query failed")
        else:
            gpu_memory = float(completed.stdout.strip().splitlines()[0])
        applications = execute(  # Fixed read-only NVIDIA inventory.
            (
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if applications.returncode:
            errors.append(applications.stderr.strip() or "compute application query failed")
        else:
            for line in applications.stdout.splitlines():
                if not line.strip():
                    continue
                pid, separator, name = line.partition(",")
                raw_pid = pid.strip()
                compute.append(
                    {
                        "pid": int(raw_pid) if separator and raw_pid.isdigit() else None,
                        "process_name": name.strip() if separator else line.strip(),
                    }
                )
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return normalize_runtime_snapshot(
        processes=processes,
        compute_applications=compute,
        gpu_memory_used_mib=gpu_memory,
        errors=errors,
        excluded_pids=excluded,
    )


def _parse_windows_process_inventory(
    stdout: str,
    *,
    current_pid: int,
) -> list[dict[str, object]]:
    if not stdout.strip():
        raise ValueError("process inventory is empty")
    payload: object = json.loads(stdout)
    raw_rows = payload if isinstance(payload, list) else [payload]
    if not raw_rows or not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError("process inventory must be a nonempty object list")
    rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
    if not all(_process_int(row, "ProcessId") is not None for row in rows):
        raise ValueError("process inventory rows require ProcessId")
    if isinstance(current_pid, bool) or current_pid <= 0:
        raise ValueError("current process PID must be a positive integer")
    current_rows = [row for row in rows if _process_int(row, "ProcessId") == current_pid]
    if len(current_rows) != 1:
        raise ValueError("process inventory must contain exactly one current process row")
    current = current_rows[0]
    parent_pid = _process_int(current, "ParentProcessId")
    if parent_pid is None or parent_pid < 0:
        raise ValueError("current process row requires a nonnegative ParentProcessId")
    command_line = current.get("CommandLine")
    if not isinstance(command_line, str) or not command_line.strip():
        raise ValueError("current process row requires a nonempty CommandLine")
    return rows


def _inventory_ancestor_pids(
    processes: Sequence[Mapping[str, object]],
    *,
    current_pid: int,
) -> set[int]:
    parents = {
        pid: parent
        for row in processes
        if (pid := _process_int(row, "pid", "ProcessId")) is not None
        and (parent := _process_int(row, "parent_pid", "ParentProcessId")) is not None
    }
    ancestors = {current_pid}
    cursor = current_pid
    while (parent := parents.get(cursor)) is not None and parent > 0 and parent not in ancestors:
        ancestors.add(parent)
        cursor = parent
    return ancestors


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("training", "final"), required=True)
    parser.add_argument("--training-jobs", type=Path, required=True)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--training-launch-evidence", type=Path, required=True)
    parser.add_argument("--training-run-evidence", type=Path, action="append", default=[])
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--evaluation-status", type=Path)
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--staging-report", type=Path)
    parser.add_argument("--gpu-memory-tolerance-mib", type=float, default=256.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_probe: Callable[[], RuntimeSnapshot] | None = None,
    now: Callable[[], str] | None = None,
) -> int:
    args = _parse_args(argv)
    input_paths = (
        ("training jobs", args.training_jobs),
        ("training status", args.training_status),
        ("training launch evidence", args.training_launch_evidence),
        *(("training run evidence", path) for path in args.training_run_evidence),
        ("evaluation config", args.evaluation_config),
        ("evaluation status", args.evaluation_status),
        ("review decisions", args.review_decisions),
        ("staging report", args.staging_report),
    )
    for source, path in input_paths:
        if isinstance(path, Path):
            _require_no_alias_components(path, source=source)
    raw_output = args.output.expanduser().absolute()
    raw_output_temp = raw_output.with_suffix(raw_output.suffix + ".tmp")
    if os.path.lexists(raw_output) or os.path.lexists(raw_output_temp):
        raise FileExistsError(f"refusing to overwrite completion report: {raw_output}")
    raw_output = _require_no_alias_components(raw_output, source="completion report output")
    output = raw_output.resolve(strict=False)
    launch_evidence_parent = (
        _require_no_alias_components(
            args.training_launch_evidence,
            source="training launch evidence",
        )
        .resolve()
        .parent
    )
    if output.parent != launch_evidence_parent:
        raise ValueError(
            "completion report output parent must match training launch evidence directory"
        )
    if os.path.lexists(output) or os.path.lexists(output.with_suffix(output.suffix + ".tmp")):
        raise FileExistsError(f"refusing to overwrite completion report: {output}")
    final_paths = (
        args.evaluation_config,
        args.evaluation_status,
        args.review_decisions,
        args.staging_report,
    )
    if args.phase == "final" and any(path is None for path in final_paths):
        raise ValueError(
            "final phase requires evaluation config/status, review decisions, and staging report"
        )
    staging_preflight: StagingOutputPreflight | None = None
    if args.phase == "final":
        staging_report = _not_none(args.staging_report, source="staging report")
        staging_preflight = _preflight_staging_output(staging_report, output)
    verified_at = (now or (lambda: datetime.now(_UTC).isoformat()))()
    training: TrainingVerification | None = None
    evaluations: EvaluationVerification | None = None
    reviews: ReviewVerification | None = None
    staging: StagingVerification | None = None
    try:
        runtime = (runtime_probe or probe_runtime)()
        training = verify_training(
            args.training_jobs,
            args.training_status,
            args.training_launch_evidence,
            runtime,
            args.gpu_memory_tolerance_mib,
            training_run_evidence=tuple(args.training_run_evidence),
        )
        if args.phase == "final":
            evaluation_config, evaluation_status, decisions, staging_report = final_paths
            evaluation_config = _not_none(evaluation_config, source="evaluation config")
            evaluation_status = _not_none(evaluation_status, source="evaluation status")
            decisions = _not_none(decisions, source="review decisions")
            staging_report = _not_none(staging_report, source="staging report")
            evaluations = verify_evaluations(evaluation_config, evaluation_status, training)
            reviews = verify_reviews(evaluations, decisions)
            staging = verify_staging(evaluations, staging_report)
        status = reviews.status if reviews is not None else "PASS"
        report = build_completion_report(
            phase=args.phase,
            status=status,
            training=training,
            evaluations=evaluations,
            reviews=reviews,
            staging=staging,
            verified_at=verified_at,
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        report = build_completion_report(
            phase=args.phase,
            status="FAIL",
            training=training,
            evaluations=evaluations,
            reviews=reviews,
            staging=staging,
            failure_reasons=(f"{type(exc).__name__}: {exc}",),
            verified_at=verified_at,
        )
        status = "FAIL"
    ensure_output_safe(
        output,
        _completion_protected_paths(
            args,
            training=training,
            evaluations=evaluations,
            reviews=reviews,
            staging=staging,
        ),
    )
    if staging_preflight is not None:
        _revalidate_staging_output(staging_preflight, output)
    write_report_create_only(output, report)
    return 0 if status == "PASS" else 1


def _read_json(path: Path) -> dict[str, object]:
    lexical = _require_regular_file(path, source="JSON input")
    try:
        payload: Any = json.loads(lexical.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    lexical = _require_regular_file(path, source="JSONL input")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lexical.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _read_jsonl_raw_lines(path: Path) -> list[tuple[bytes, dict[str, object]]]:
    lexical = _require_regular_file(path, source="JSONL input")
    raw_lines = lexical.read_bytes().splitlines(keepends=True)
    if any(not line.endswith(b"\n") for line in raw_lines):
        raise ValueError(f"JSONL must end every row with newline: {path}")
    rows = _read_jsonl(path)
    if len(rows) != len(raw_lines):
        raise ValueError(f"JSONL may not contain blank rows: {path}")
    return list(zip(raw_lines, rows, strict=True))


def _required_string(row: Mapping[str, object], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} requires nonempty string {field}")
    return value


def _required_mapping(
    row: Mapping[str, object], field: str, *, source: str
) -> Mapping[str, object]:
    value = row.get(field)
    if not isinstance(value, dict):
        raise TypeError(f"{source} requires object {field}")
    return value


def _required_int(row: Mapping[str, object], field: str, *, source: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{source} requires integer {field}")
    return value


def _required_sha256(row: Mapping[str, object], field: str, *, source: str) -> str:
    value = row.get(field)
    if not _is_sha256(value):
        raise ValueError(f"{source} requires lowercase SHA-256 {field}")
    return str(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _is_lower_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_path(raw: object, *, base: Path, source: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{source} must be a nonempty path")
    path = Path(raw)
    lexical = path if path.is_absolute() else base / path
    return _require_no_alias_components(lexical, source=source).resolve()


def _resolve_existing_file_path(raw: object, *, base: Path, source: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{source} must be a nonempty path")
    path = Path(raw)
    lexical = path if path.is_absolute() else base / path
    return _require_regular_file(lexical, source=source).resolve(strict=True)


def _resolve_existing_directory_path(raw: object, *, base: Path, source: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{source} must be a nonempty path")
    path = Path(raw)
    lexical = path if path.is_absolute() else base / path
    return _require_directory(lexical, source=source).resolve(strict=True)


def _resolve_contained_path(raw: object, *, base: Path, source: str) -> Path:
    root = _require_no_alias_components(base, source=f"{source} root").resolve()
    resolved = _resolve_path(raw, base=root, source=source)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{source} escapes its root")
    return resolved


def _canonical_sha256(row: Mapping[str, object]) -> str:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_datetime(value: str, *, source: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{source} must be ISO-8601") from exc


def _binding(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256}


def _file_binding(binding: FileBinding) -> dict[str, str]:
    return _binding(binding.path, binding.sha256)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _not_none(path: Path | None, *, source: str) -> Path:
    if path is None:
        raise TypeError(f"{source} is required")
    return path


if __name__ == "__main__":  # pragma: no cover - exercised through main unit tests.
    raise SystemExit(main())
