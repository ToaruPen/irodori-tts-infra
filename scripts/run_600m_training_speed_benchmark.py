# ruff: noqa: EM101, EM102, TRY003 - operational errors retain candidate and path context.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess  # noqa: S404 - benchmark intentionally runs fixed trainer and GPU sampler commands.
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "speaker-training-speed-benchmark/v1"
RESULT_SCHEMA_VERSION = "speaker-training-speed-candidate/v1"
STATUS_SCHEMA_VERSION = "speaker-training-speed-status/v1"
TRAINING_JOBS_SCHEMA_VERSION = 1
EXPECTED_TRAINING_JOB_COUNT = 12
ANABEL_MODEL_ID = "oop77_anabel_maidgarden_sp_451488a7c1"
KASUMI_MODEL_ID = "kasumi"
NEXT_MODEL_ID = "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
EXPECTED_TRAINING_PREFIX = (ANABEL_MODEL_ID, KASUMI_MODEL_ID, NEXT_MODEL_ID)
GLOBAL_BATCH_SIZE = 16
PERF_WARMUP_STEPS = 10
MEASUREMENT_STEPS = 50
MAX_STEPS = PERF_WARMUP_STEPS + MEASUREMENT_STEPS
PEAK_VRAM_LIMIT_MIB = 10_500.0
DEFAULT_SAMPLER_INTERVAL_SECONDS = 0.5
MIN_STEADY_GPU_SAMPLES = 10
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 120.0
DEFAULT_CLEANUP_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_CLEANUP_MEMORY_TOLERANCE_MIB = 256.0
EXPECTED_COMMIT_LENGTH = 40
EXPECTED_SEED = 0
EXPECTED_MAX_LATENT_STEPS = 750
STEP_PATTERN = re.compile(r"\bstep=(?P<step>\d+)\b")
LOSS_PATTERN = re.compile(r"\bloss=(?P<loss>[^\s]+)")
LR_PATTERN = re.compile(r"\blr=(?P<lr>[^\s]+)")
OOM_PATTERN = re.compile(r"(?:cuda\s+)?out of memory", re.IGNORECASE)
GPU_HEADER = (
    "timestamp",
    "index",
    "utilization_gpu_percent",
    "memory_used_mib",
    "power_draw_w",
    "temperature_c",
    "error",
    "raw_csv",
)
GPU_QUERY_FIELD_COUNT = 5
JSON_NUMBER_PATTERN = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")

LineSink = Callable[[str, str | None], None]
Runner = Callable[["CandidatePlan", tuple[str, ...], LineSink], int]
Sampler = Callable[["CandidatePlan", Path, float, threading.Event], None]
GitInspector = Callable[[Path], Mapping[str, object]]
EnvironmentInspector = Callable[[], Mapping[str, object]]
RuntimeProbe = Callable[["CandidatePlan", str], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    candidate_id: str
    spec: CandidateSpec
    root: Path
    output_dir: Path
    config_path: Path
    upstream_root: Path


@dataclass(frozen=True, slots=True)
class TrainingJobBinding:
    model_id: str
    clean_manifest: Path
    config: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class TrainingQueueContract:
    jobs: tuple[TrainingJobBinding, ...]
    base_checkpoint: Path
    base_checkpoint_sha256: str
    checkpoint_revision: str
    upstream_commit: str


CANDIDATE_SPECS = (
    CandidateSpec("A", 1, 16, gradient_checkpointing=True),
    CandidateSpec("B", 2, 8, gradient_checkpointing=True),
    CandidateSpec("C", 4, 4, gradient_checkpointing=True),
    CandidateSpec("D", 2, 8, gradient_checkpointing=False),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _candidate_overrides(spec: CandidateSpec) -> dict[str, object]:
    return {
        "batch_size": spec.batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "gradient_checkpointing": spec.gradient_checkpointing,
    }


def _load_training_queue_contract(
    path: Path,
    *,
    base_config: Path,
    manifest: Path,
    base_checkpoint: Path,
    upstream_commit: str,
) -> TrainingQueueContract:
    document = _read_json(path)
    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version != TRAINING_JOBS_SCHEMA_VERSION:
        raise ValueError("training jobs schema_version must be numeric 1")
    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_TRAINING_JOB_COUNT:
        raise ValueError(f"training jobs must contain exactly {EXPECTED_TRAINING_JOB_COUNT} jobs")
    jobs = tuple(_parse_training_job(row, base=path.parent) for row in raw_jobs)
    model_ids = tuple(job.model_id for job in jobs)
    if len(set(model_ids)) != EXPECTED_TRAINING_JOB_COUNT:
        raise ValueError("training jobs must contain unique model_id values")
    if model_ids[: len(EXPECTED_TRAINING_PREFIX)] != EXPECTED_TRAINING_PREFIX:
        raise ValueError(
            "training job order must start with Anabel, Kasumi, and the OOP69 benchmark job"
        )
    third = jobs[2]
    if third.config != base_config.resolve() or third.clean_manifest != manifest.resolve():
        raise ValueError("benchmark config and manifest must match the third training job")
    if third.output_dir.exists():
        raise FileExistsError(f"third training job output must be absent: {third.output_dir}")
    declared_checkpoint = _resolve_queue_path(
        document,
        "base_checkpoint_path",
        base=path.parent,
        source="training jobs",
    )
    declared_checkpoint_sha256 = _queue_hex(
        document,
        "base_checkpoint_sha256",
        length=64,
        source="training jobs",
    )
    if declared_checkpoint != base_checkpoint.resolve():
        raise ValueError("CLI base checkpoint does not match training jobs")
    if sha256_file(declared_checkpoint) != declared_checkpoint_sha256:
        raise ValueError("training jobs base checkpoint SHA-256 does not match the file")
    checkpoint_revision = _queue_hex(
        document,
        "checkpoint_revision",
        length=EXPECTED_COMMIT_LENGTH,
        source="training jobs",
    )
    declared_upstream = _queue_hex(
        document,
        "upstream_commit",
        length=EXPECTED_COMMIT_LENGTH,
        source="training jobs",
    )
    if declared_upstream != upstream_commit:
        raise ValueError("CLI upstream_commit does not match training jobs")
    return TrainingQueueContract(
        jobs=jobs,
        base_checkpoint=declared_checkpoint,
        base_checkpoint_sha256=declared_checkpoint_sha256,
        checkpoint_revision=checkpoint_revision,
        upstream_commit=declared_upstream,
    )


def _parse_training_job(raw: object, *, base: Path) -> TrainingJobBinding:
    if not isinstance(raw, dict):
        raise TypeError("training job entries must be objects")
    model_id = _queue_string(raw, "model_id", source="training job")
    clean_manifest = _resolve_queue_path(
        raw, "clean_manifest", base=base, source=f"training job {model_id}"
    )
    config = _resolve_queue_path(raw, "config", base=base, source=f"training job {model_id}")
    output_dir = _resolve_queue_path(
        raw, "output_dir", base=base, source=f"training job {model_id}"
    )
    if not clean_manifest.is_file():
        raise FileNotFoundError(f"training job manifest does not exist: {clean_manifest}")
    if not config.is_file():
        raise FileNotFoundError(f"training job config does not exist: {config}")
    return TrainingJobBinding(
        model_id=model_id,
        clean_manifest=clean_manifest,
        config=config,
        output_dir=output_dir,
    )


def _validate_training_status(path: Path, *, contract: TrainingQueueContract) -> None:
    rows = _read_jsonl(path)
    jobs_by_id = {job.model_id: job for job in contract.jobs}
    history: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        model_id = _queue_string(row, "model_id", source="training status")
        if model_id not in jobs_by_id:
            raise ValueError(f"training status contains unknown model_id: {model_id}")
        if row.get("event") not in {"started", "finished"}:
            raise ValueError(f"training status event is invalid for {model_id}")
        history.setdefault(model_id, []).append(row)
    for model_id, label in (
        (ANABEL_MODEL_ID, "Anabel"),
        (KASUMI_MODEL_ID, "Kasumi"),
    ):
        model_history = history.get(model_id)
        if not model_history:
            raise ValueError(f"{label} current status must be finished success")
        current = model_history[-1]
        if not (
            current.get("event") == "finished"
            and current.get("status") == "success"
            and current.get("exit_code") == 0
        ):
            raise ValueError(f"{label} current status must be finished success")
        _validate_success_status(
            current,
            job=jobs_by_id[model_id],
            contract=contract,
        )
    if history.get(NEXT_MODEL_ID):
        raise ValueError("third training job must be unstarted before the benchmark")
    pending_ids = {job.model_id for job in contract.jobs[2:]}
    if any(
        model_history[-1].get("status") == "success"
        for model_id, model_history in history.items()
        if model_id in pending_ids
    ):
        raise ValueError("remaining training jobs must be unfinished")


def _validate_success_status(
    row: Mapping[str, object],
    *,
    job: TrainingJobBinding,
    contract: TrainingQueueContract,
) -> None:
    expected = {
        "clean_manifest_sha256": sha256_file(job.clean_manifest),
        "config_sha256": sha256_file(job.config),
        "checkpoint_sha256": contract.base_checkpoint_sha256,
        "checkpoint_revision": contract.checkpoint_revision,
        "upstream_commit": contract.upstream_commit,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(
                f"successful training status {field} does not match for {job.model_id}"
            )
    if not job.output_dir.is_dir():
        raise FileNotFoundError(f"successful training output does not exist: {job.output_dir}")
    raw_candidates = row.get("candidate_checkpoints")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"successful training status has no checkpoints for {job.model_id}")
    candidates: list[tuple[Path, str]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise TypeError("training status checkpoint entries must be objects")
        candidate_path = Path(
            _queue_string(raw_candidate, "path", source="training status checkpoint")
        ).resolve()
        candidate_sha256 = _queue_hex(
            raw_candidate,
            "sha256",
            length=64,
            source="training status checkpoint",
        )
        if not candidate_path.is_relative_to(job.output_dir):
            raise ValueError(f"training checkpoint is outside output directory: {candidate_path}")
        if not candidate_path.is_file():
            raise FileNotFoundError(f"training checkpoint does not exist: {candidate_path}")
        if sha256_file(candidate_path) != candidate_sha256:
            raise ValueError(f"training checkpoint SHA-256 does not match: {candidate_path}")
        candidates.append((candidate_path, candidate_sha256))
    last_path, last_sha256 = candidates[-1]
    if (
        row.get("last_checkpoint") != str(last_path)
        or row.get("last_checkpoint_sha256") != last_sha256
    ):
        raise ValueError(f"last training checkpoint binding does not match for {job.model_id}")


def _queue_string(row: Mapping[str, object], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} requires nonempty string {field}")
    return value


def _queue_hex(
    row: Mapping[str, object],
    field: str,
    *,
    length: int,
    source: str,
) -> str:
    value = _queue_string(row, field, source=source)
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{source} {field} must be {length}-character lowercase hex")
    return value


def _resolve_queue_path(
    row: Mapping[str, object],
    field: str,
    *,
    base: Path,
    source: str,
) -> Path:
    path = Path(_queue_string(row, field, source=source))
    return (path if path.is_absolute() else base / path).resolve()


def _inspect_git_state(upstream_root: Path) -> dict[str, object]:
    head = _git_stdout(upstream_root, "rev-parse", "HEAD").strip()
    tracked_worktree = subprocess.run(  # noqa: S603 - fixed read-only Git command.
        ("git", "-C", str(upstream_root), "diff", "--quiet", "--"),  # noqa: S607
        check=False,
    )
    index = subprocess.run(  # noqa: S603 - fixed read-only Git command.
        ("git", "-C", str(upstream_root), "diff", "--cached", "--quiet", "--"),  # noqa: S607
        check=False,
    )
    if tracked_worktree.returncode not in {0, 1} or index.returncode not in {0, 1}:
        raise RuntimeError("failed to inspect upstream tracked Git state")
    untracked = sorted(
        line
        for line in _git_stdout(
            upstream_root, "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        if line
    )
    return {
        "head": head,
        "tracked_worktree_clean": tracked_worktree.returncode == 0,
        "index_clean": index.returncode == 0,
        "untracked_files": untracked,
    }


def _git_stdout(upstream_root: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed read-only Git command.
        ("git", "-C", str(upstream_root), *args),  # noqa: S607
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git inspection failed: {detail}")
    return completed.stdout


def _normalize_git_state(raw: Mapping[str, object]) -> dict[str, object]:
    head = raw.get("head")
    tracked_clean = raw.get("tracked_worktree_clean")
    index_clean = raw.get("index_clean")
    untracked = raw.get("untracked_files")
    if not isinstance(head, str):
        raise TypeError("Git inspection head must be a string")
    if not isinstance(tracked_clean, bool) or not isinstance(index_clean, bool):
        raise TypeError("Git inspection cleanliness fields must be booleans")
    if not isinstance(untracked, list) or not all(isinstance(path, str) for path in untracked):
        raise TypeError("Git inspection untracked_files must be a string list")
    sorted_untracked = sorted(set(untracked))
    encoded = json.dumps(sorted_untracked, ensure_ascii=False, separators=(",", ":"))
    return {
        "head": head,
        "tracked_worktree_clean": tracked_clean,
        "index_clean": index_clean,
        "untracked_files": sorted_untracked,
        "untracked_count": len(sorted_untracked),
        "untracked_files_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _validate_git_state(state: Mapping[str, object], *, expected_commit: str) -> None:
    if state["head"] != expected_commit:
        raise ValueError(
            f"upstream HEAD does not match expected commit: {state['head']} != {expected_commit}"
        )
    if state["tracked_worktree_clean"] is not True:
        raise ValueError("upstream tracked worktree is dirty")
    if state["index_clean"] is not True:
        raise ValueError("upstream index is dirty")


def _inspect_environment() -> dict[str, object]:
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415

        torch_state: dict[str, object] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:  # noqa: BLE001 - preflight records unavailable runtime metadata.
        torch_state = {
            "version": None,
            "cuda_version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    command = (
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed read-only GPU preflight command.
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        raw = completed.stdout.strip()
        rows = list(csv.reader(raw.splitlines())) if raw else []
        if completed.returncode or len(rows) != 1 or len(rows[0]) != GPU_QUERY_FIELD_COUNT:
            error = completed.stderr.strip() or "nvidia-smi returned invalid GPU metadata"
            gpu: dict[str, object] = {"error": error, "raw_csv": raw}
        else:
            fields = [field.strip() for field in rows[0]]
            gpu = {
                "name": fields[0],
                "uuid": fields[1],
                "driver_version": fields[2],
                "memory_total_mib": float(fields[3]),
                "power_limit_w": float(fields[4]),
                "error": None,
                "raw_csv": raw,
            }
    except (OSError, ValueError) as exc:
        gpu = {"error": f"{type(exc).__name__}: {exc}", "raw_csv": ""}
    return {
        "python": {"version": sys.version, "executable": sys.executable},
        "torch": torch_state,
        "gpu": gpu,
    }


def _normalize_environment(raw: Mapping[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(raw, ensure_ascii=False))
    if not isinstance(normalized, dict):
        raise TypeError("environment inspection must be a JSON object")
    return normalized


def _validate_environment(environment: Mapping[str, object]) -> None:
    gpu = environment.get("gpu")
    if not isinstance(gpu, dict):
        raise TypeError("environment inspection requires GPU metadata")
    if gpu.get("error"):
        raise ValueError(f"GPU preflight failed: {gpu['error']}")
    for field in ("name", "uuid", "driver_version", "memory_total_mib", "power_limit_w"):
        if gpu.get(field) in {None, ""}:
            raise ValueError(f"GPU preflight did not return {field}")


def run_benchmark(  # noqa: PLR0913, PLR0914, PLR0915 - explicit benchmark orchestration.
    *,
    base_config: Path,
    manifest: Path,
    base_checkpoint: Path,
    upstream_root: Path,
    upstream_commit: str,
    training_jobs: Path,
    training_status: Path,
    output_root: Path,
    runner: Runner | None = None,
    sampler: Sampler | None = None,
    sampler_interval: float = DEFAULT_SAMPLER_INTERVAL_SECONDS,
    git_inspector: GitInspector | None = None,
    environment_inspector: EnvironmentInspector | None = None,
    runtime_probe: RuntimeProbe | None = None,
    cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    cleanup_poll_interval: float = DEFAULT_CLEANUP_POLL_INTERVAL_SECONDS,
    cleanup_memory_tolerance_mib: float = DEFAULT_CLEANUP_MEMORY_TOLERANCE_MIB,
) -> dict[str, object]:
    inputs = _validate_inputs(
        base_config=base_config,
        manifest=manifest,
        base_checkpoint=base_checkpoint,
        upstream_root=upstream_root,
        upstream_commit=upstream_commit,
        training_jobs=training_jobs,
        training_status=training_status,
        sampler_interval=sampler_interval,
        cleanup_timeout=cleanup_timeout,
        cleanup_poll_interval=cleanup_poll_interval,
        cleanup_memory_tolerance_mib=cleanup_memory_tolerance_mib,
    )
    base_document = _read_json(base_config)
    _validate_base_config(base_document)
    queue_contract = _load_training_queue_contract(
        training_jobs,
        base_config=base_config,
        manifest=manifest,
        base_checkpoint=base_checkpoint,
        upstream_commit=upstream_commit,
    )
    _validate_training_status(training_status, contract=queue_contract)
    inspect_git = git_inspector or _inspect_git_state
    git_state = _normalize_git_state(inspect_git(upstream_root))
    _validate_git_state(git_state, expected_commit=upstream_commit)
    inspect_environment = environment_inspector or _inspect_environment
    environment = _normalize_environment(inspect_environment())
    _validate_environment(environment)
    _reserve_output_root(output_root)
    provenance = _build_provenance(
        base_config=base_config,
        manifest=manifest,
        base_checkpoint=base_checkpoint,
        upstream_root=upstream_root,
        upstream_commit=upstream_commit,
        trainer_python=inputs["trainer_python"],
        trainer_script=inputs["trainer_script"],
        training_jobs=training_jobs,
        training_status=training_status,
        git_state=git_state,
        environment=environment,
    )
    execute = runner or _run_trainer
    sample_gpu = sampler or _sample_nvidia_smi
    probe_runtime = runtime_probe or _probe_runtime
    status_path = output_root / "status.jsonl"
    results: list[dict[str, object]] = []
    for spec in CANDIDATE_SPECS:
        plan = _prepare_candidate(
            spec,
            base_document=base_document,
            manifest=manifest,
            output_root=output_root,
            upstream_root=upstream_root,
        )
        command = _training_command(
            plan,
            trainer_python=inputs["trainer_python"],
            trainer_script=inputs["trainer_script"],
            manifest=manifest,
            base_checkpoint=base_checkpoint,
        )
        runtime_guard_path = plan.root / "runtime-guard.jsonl"
        before_state = _normalize_runtime_state(probe_runtime(plan, "before"))
        _append_jsonl(
            runtime_guard_path,
            {"phase": "before", **before_state},
            durable=True,
        )
        baseline_memory = before_state["gpu_memory_used_mib"]
        if not _runtime_state_safe_before(before_state):
            result = _blocked_candidate_result(
                plan,
                provenance=provenance,
                reason="runtime_not_quiescent_before_start",
                runtime_guard_path=runtime_guard_path,
                before_state=before_state,
            )
            result_path = plan.root / "result.json"
            _write_json_atomic(result_path, result)
            _append_status(
                status_path,
                {
                    "schema_version": STATUS_SCHEMA_VERSION,
                    "event": "blocked",
                    "status": "failed",
                    "candidate_id": spec.candidate_id,
                    "timestamp": _utc_now(),
                    "eligible": False,
                    "result_path": str(result_path.resolve()),
                    "result_sha256": sha256_file(result_path),
                    "error": "runtime_not_quiescent_before_start",
                },
            )
            results.append(result)
            break
        if not isinstance(baseline_memory, int | float):
            raise TypeError("runtime guard baseline GPU memory must be numeric")
        _append_status(
            status_path,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "event": "started",
                "status": "running",
                "candidate_id": spec.candidate_id,
                "timestamp": _utc_now(),
                "command": list(command),
                "config_path": str(plan.config_path.resolve()),
                "config_sha256": sha256_file(plan.config_path),
            },
        )
        result = _run_candidate(
            plan,
            command=command,
            runner=execute,
            sampler=sample_gpu,
            sampler_interval=sampler_interval,
            provenance=provenance,
        )
        after_state, runtime_is_quiescent = _wait_for_runtime_cleanup(
            plan,
            probe=probe_runtime,
            runtime_guard_path=runtime_guard_path,
            baseline_gpu_memory_mib=float(baseline_memory),
            timeout=cleanup_timeout,
            poll_interval=cleanup_poll_interval,
            memory_tolerance_mib=cleanup_memory_tolerance_mib,
        )
        result["runtime_guard"] = {
            "before": before_state,
            "after": after_state,
            "gpu_memory_release_threshold_mib": float(baseline_memory)
            + cleanup_memory_tolerance_mib,
        }
        artifacts = result["artifacts"]
        if not isinstance(artifacts, dict):
            raise TypeError("candidate artifacts must be an object")
        artifacts["runtime_guard"] = _file_binding(runtime_guard_path)
        metrics = result["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("candidate metrics must be an object")
        if not runtime_is_quiescent:
            reasons = metrics["ineligible_reasons"]
            if not isinstance(reasons, list):
                raise TypeError("candidate ineligible reasons must be a list")
            reasons.append("runtime_not_quiescent")
            metrics["eligible"] = False
        result_path = plan.root / "result.json"
        _write_json_atomic(result_path, result)
        _append_status(
            status_path,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "event": "finished",
                "status": "success" if metrics["eligible"] else "failed",
                "candidate_id": spec.candidate_id,
                "timestamp": _utc_now(),
                "exit_code": metrics["exit_code"],
                "eligible": metrics["eligible"],
                "result_path": str(result_path.resolve()),
                "result_sha256": sha256_file(result_path),
                "error": result["execution_error"],
            },
        )
        results.append(result)
        if not runtime_is_quiescent:
            break
    recommendation = _recommend(results)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if recommendation is not None else "FAILED",
        "constraints": {
            "candidate_parallelism": 1,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "measurement_optimizer_steps": MEASUREMENT_STEPS,
            "perf_warmup_optimizer_steps": PERF_WARMUP_STEPS,
            "torch_compile": False,
            "multi_model_parallelism": False,
        },
        "candidates": results,
        "recommended_candidate": recommendation,
        "measurement_boundary": {
            "definition": "(60-10)/(timestamp(step=60)-timestamp(step=10))",
            "start_step": PERF_WARMUP_STEPS,
            "end_step": MAX_STEPS,
            "measured_optimizer_steps": MEASUREMENT_STEPS,
        },
        "provenance": provenance,
    }
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def _validate_inputs(  # noqa: PLR0913 - mirrors the public provenance boundary.
    *,
    base_config: Path,
    manifest: Path,
    base_checkpoint: Path,
    upstream_root: Path,
    upstream_commit: str,
    training_jobs: Path,
    training_status: Path,
    sampler_interval: float,
    cleanup_timeout: float,
    cleanup_poll_interval: float,
    cleanup_memory_tolerance_mib: float,
) -> dict[str, Path]:
    required_files = (
        base_config,
        manifest,
        base_checkpoint,
        training_jobs,
        training_status,
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"benchmark input does not exist: {missing[0]}")
    trainer_python = upstream_root / ".venv" / "Scripts" / "python.exe"
    trainer_script = upstream_root / "train.py"
    for path in (trainer_python, trainer_script):
        if not path.is_file():
            raise FileNotFoundError(f"upstream training component does not exist: {path}")
    if len(upstream_commit) != EXPECTED_COMMIT_LENGTH or any(
        character not in "0123456789abcdef" for character in upstream_commit
    ):
        raise ValueError("upstream commit must be a lowercase 40-character Git SHA")
    if not math.isfinite(sampler_interval) or sampler_interval <= 0:
        raise ValueError("sampler interval must be a positive finite number")
    if not math.isfinite(cleanup_timeout) or cleanup_timeout < 0:
        raise ValueError("cleanup timeout must be a nonnegative finite number")
    if not math.isfinite(cleanup_poll_interval) or cleanup_poll_interval <= 0:
        raise ValueError("cleanup poll interval must be a positive finite number")
    if not math.isfinite(cleanup_memory_tolerance_mib) or cleanup_memory_tolerance_mib < 0:
        raise ValueError("cleanup memory tolerance must be a nonnegative finite number")
    return {"trainer_python": trainer_python, "trainer_script": trainer_script}


def _validate_base_config(document: Mapping[str, object]) -> None:
    model = document.get("model")
    train = document.get("train")
    if not isinstance(model, dict) or not isinstance(train, dict):
        raise TypeError("base config requires object model and train sections")
    for field in ("max_latent_steps", "seed"):
        if field not in train:
            raise ValueError(f"base config train section requires fixed field {field}")
    if train["seed"] != EXPECTED_SEED or isinstance(train["seed"], bool):
        raise ValueError(f"base config seed must equal {EXPECTED_SEED}")
    if train["max_latent_steps"] != EXPECTED_MAX_LATENT_STEPS or isinstance(
        train["max_latent_steps"], bool
    ):
        raise ValueError(f"base config max_latent_steps must equal {EXPECTED_MAX_LATENT_STEPS}")


def _reserve_output_root(output_root: Path) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.mkdir()
    except FileExistsError:
        raise FileExistsError(f"benchmark output root already exists: {output_root}") from None


def _prepare_candidate(
    spec: CandidateSpec,
    *,
    base_document: Mapping[str, object],
    manifest: Path,
    output_root: Path,
    upstream_root: Path,
) -> CandidatePlan:
    root = output_root / f"candidate-{spec.candidate_id}"
    root.mkdir()
    output_dir = root / "output"
    config_path = root / "training-config.json"
    document = json.loads(json.dumps(base_document))
    train = document.get("train")
    if not isinstance(train, dict):
        raise TypeError("base config train section must be an object")
    train.update(
        {
            "manifest_path": str(manifest.resolve()),
            "output_dir": str(output_dir.resolve()),
            "batch_size": spec.batch_size,
            "gradient_accumulation_steps": spec.gradient_accumulation_steps,
            "gradient_checkpointing": spec.gradient_checkpointing,
            "allow_tf32": True,
            "precision": "bf16",
            "compile_model": False,
            "learning_rate": 0.01,
            "lr_scheduler": "none",
            "max_steps": MAX_STEPS,
            "log_every": 1,
            "save_every": MAX_STEPS,
            "valid_ratio": 0.0,
            "valid_every": 0,
            "wandb_enabled": False,
            "wandb_project": None,
            "wandb_entity": None,
            "wandb_run_name": None,
        }
    )
    if spec.batch_size * spec.gradient_accumulation_steps != GLOBAL_BATCH_SIZE:
        raise ValueError(f"candidate {spec.candidate_id} does not preserve global batch 16")
    _write_training_config(config_path, document)
    return CandidatePlan(
        candidate_id=spec.candidate_id,
        spec=spec,
        root=root,
        output_dir=output_dir,
        config_path=config_path,
        upstream_root=upstream_root,
    )


def _training_command(
    plan: CandidatePlan,
    *,
    trainer_python: Path,
    trainer_script: Path,
    manifest: Path,
    base_checkpoint: Path,
) -> tuple[str, ...]:
    return (
        str(trainer_python),
        "-u",
        str(trainer_script),
        "--config",
        str(plan.config_path),
        "--manifest",
        str(manifest.resolve()),
        "--init-checkpoint",
        str(base_checkpoint.absolute()),
        "--output-dir",
        str(plan.output_dir.resolve()),
        "--device",
        "cuda",
    )


def _run_candidate(  # noqa: PLR0914 - artifact metrics stay together for one candidate.
    plan: CandidatePlan,
    *,
    command: tuple[str, ...],
    runner: Runner,
    sampler: Sampler,
    sampler_interval: float,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    raw_log_path = plan.root / "raw.log"
    events_path = plan.root / "step-events.jsonl"
    gpu_path = plan.root / "nvidia-smi.csv"
    raw_log_path.touch()
    events_path.touch()
    state: dict[str, object] = {"oom_detected": False}
    sink = _line_sink(raw_log_path, events_path=events_path, state=state)
    stop_event = threading.Event()
    sampler_errors: list[str] = []
    sampler_thread = threading.Thread(
        target=_sampler_worker,
        args=(sampler, plan, gpu_path, sampler_interval, stop_event, sampler_errors),
        name=f"gpu-sampler-{plan.candidate_id}",
        daemon=True,
    )
    sampler_thread.start()
    exit_code: int | None = None
    execution_error: str | None = None
    try:
        exit_code = runner(plan, command, sink)
    except Exception as exc:  # noqa: BLE001 - candidate failure must not abort later candidates.
        execution_error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_event.set()
        sampler_thread.join(timeout=max(5.0, sampler_interval * 4))
        if sampler_thread.is_alive():
            sampler_errors.append("GPU sampler did not stop before timeout")
    _ensure_gpu_csv(gpu_path, errors=sampler_errors)
    events = _read_jsonl(events_path)
    gpu_samples = _read_gpu_samples(gpu_path)
    measurements = _measure(events, gpu_samples=gpu_samples)
    reasons = _ineligible_reasons(
        exit_code=exit_code,
        execution_error=execution_error,
        oom_detected=bool(state["oom_detected"]),
        measurements=measurements,
    )
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in {
            "config": plan.config_path,
            "raw_log": raw_log_path,
            "step_events": events_path,
            "gpu_samples": gpu_path,
        }.items()
    }
    metrics = {
        "measured_optimizer_steps": measurements["measurement_step_count"],
        "steady_optimizer_steps_per_second": measurements["optimizer_steps_per_second"],
        "steady_samples_per_second": measurements["samples_per_second"],
        "peak_vram_mib": measurements["full_run_peak_vram_mib"],
        "steady_peak_vram_mib": measurements["steady_peak_vram_mib"],
        "gpu_utilization_percent": measurements["steady_gpu_utilization_percent"],
        "power_watts": measurements["steady_power_draw_w"],
        "loss_finite": measurements["losses_finite"],
        "learning_rate_fixed": measurements["learning_rates_fixed"],
        "observed_step_sequence_valid": measurements["observed_step_sequence_valid"],
        "oom": bool(state["oom_detected"]),
        "exit_code": exit_code,
        "eligible": not reasons,
        "ineligible_reasons": reasons,
        "reached_step": measurements["reached_step"],
        "full_run_peak_vram_mib": measurements["full_run_peak_vram_mib"],
        "full_run_gpu_utilization_percent": measurements["full_run_gpu_utilization_percent"],
        "full_run_power_watts": measurements["full_run_power_draw_w"],
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "id": plan.candidate_id,
        "batch_size": plan.spec.batch_size,
        "gradient_accumulation_steps": plan.spec.gradient_accumulation_steps,
        "gradient_checkpointing": plan.spec.gradient_checkpointing,
        "effective_global_batch_size": GLOBAL_BATCH_SIZE,
        "overrides": _candidate_overrides(plan.spec),
        "metrics": metrics,
        "execution_error": execution_error,
        "sampler_errors": sampler_errors,
        "artifacts": artifacts,
        "provenance": provenance,
    }


def _line_sink(
    raw_log_path: Path,
    *,
    events_path: Path,
    state: dict[str, object],
) -> LineSink:
    def emit(line: str, timestamp: str | None = None) -> None:
        normalized = line.rstrip("\r\n")
        with raw_log_path.open("a", encoding="utf-8", newline="\n") as raw_log:
            raw_log.write(normalized + "\n")
            raw_log.flush()
        if OOM_PATTERN.search(normalized):
            state["oom_detected"] = True
        step_match = STEP_PATTERN.search(normalized)
        loss_match = LOSS_PATTERN.search(normalized)
        lr_match = LR_PATTERN.search(normalized)
        if step_match is None or loss_match is None or lr_match is None:
            return
        loss_token = loss_match.group("loss")
        lr_token = lr_match.group("lr")
        loss_value = _finite_float_or_none(loss_token)
        lr_value = _finite_float_or_none(lr_token)
        event = {
            "timestamp": timestamp or _utc_now(),
            "step": int(step_match.group("step")),
            "loss": loss_token,
            "loss_value": loss_value,
            "loss_finite": loss_value is not None,
            "lr": lr_token,
            "lr_value": lr_value,
            "raw": normalized,
        }
        _append_jsonl(events_path, event)

    return emit


def _finite_float_or_none(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _measure(
    events: Sequence[Mapping[str, object]],
    *,
    gpu_samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reached_step = max((cast("int", event["step"]) for event in events), default=0)
    losses_finite = bool(events) and all(event.get("loss_finite") is True for event in events)
    observed_steps = [cast("int", event["step"]) for event in events]
    observed_step_sequence_valid = observed_steps == list(range(1, MAX_STEPS + 1))
    learning_rates_fixed = bool(events) and all(
        isinstance(event.get("lr_value"), int | float)
        and math.isclose(cast("float", event["lr_value"]), 0.01, rel_tol=1e-9, abs_tol=1e-12)
        for event in events
    )
    measurement_events = [
        event for event in events if PERF_WARMUP_STEPS < cast("int", event["step"]) <= MAX_STEPS
    ]
    measurement_step_count = len(measurement_events)
    by_step = {cast("int", event["step"]): event for event in events}
    step_rate: float | None = None
    steady_start: datetime | None = None
    steady_end: datetime | None = None
    if PERF_WARMUP_STEPS in by_step and MAX_STEPS in by_step:
        steady_start = _parse_timestamp(str(by_step[PERF_WARMUP_STEPS]["timestamp"]))
        steady_end = _parse_timestamp(str(by_step[MAX_STEPS]["timestamp"]))
        elapsed = (steady_end - steady_start).total_seconds()
        if elapsed > 0:
            step_rate = MEASUREMENT_STEPS / elapsed
    steady_samples = [
        row
        for row in gpu_samples
        if steady_start is not None
        and steady_end is not None
        and steady_start <= _parse_timestamp(str(row["timestamp"])) <= steady_end
    ]
    return {
        "reached_step": reached_step,
        "losses_finite": losses_finite,
        "observed_step_sequence_valid": observed_step_sequence_valid,
        "learning_rates_fixed": learning_rates_fixed,
        "measurement_step_count": measurement_step_count,
        "optimizer_steps_per_second": _rounded(step_rate),
        "samples_per_second": _rounded(
            step_rate * GLOBAL_BATCH_SIZE if step_rate is not None else None
        ),
        "steady_gpu_utilization_percent": _numeric_stats(steady_samples, "utilization_gpu_percent"),
        "steady_power_draw_w": _numeric_stats(steady_samples, "power_draw_w"),
        "steady_peak_vram_mib": _rounded(_max_numeric(steady_samples, "memory_used_mib")),
        "full_run_gpu_utilization_percent": _numeric_stats(gpu_samples, "utilization_gpu_percent"),
        "full_run_power_draw_w": _numeric_stats(gpu_samples, "power_draw_w"),
        "full_run_peak_vram_mib": _rounded(_max_numeric(gpu_samples, "memory_used_mib")),
    }


def _ineligible_reasons(
    *,
    exit_code: int | None,
    execution_error: str | None,
    oom_detected: bool,
    measurements: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    if exit_code != 0 or execution_error is not None:
        reasons.append("exit_nonzero")
    if oom_detected:
        reasons.append("oom_detected")
    if measurements["losses_finite"] is not True:
        reasons.append("nonfinite_loss")
    if measurements["observed_step_sequence_valid"] is not True:
        reasons.append("step_sequence_not_1_through_60")
    if measurements["learning_rates_fixed"] is not True:
        reasons.append("learning_rate_not_fixed_1e-2")
    if measurements["measurement_step_count"] != MEASUREMENT_STEPS:
        reasons.append("measurement_steps_not_50")
    if measurements["reached_step"] != MAX_STEPS:
        reasons.append("did_not_reach_step_60")
    if measurements["optimizer_steps_per_second"] is None:
        reasons.append("optimizer_rate_unavailable")
    utilization = measurements["steady_gpu_utilization_percent"]
    steady_sample_count = utilization.get("sample_count") if isinstance(utilization, dict) else 0
    if not isinstance(steady_sample_count, int) or steady_sample_count < MIN_STEADY_GPU_SAMPLES:
        reasons.append("steady_gpu_samples_below_10")
    peak_vram = measurements["full_run_peak_vram_mib"]
    if not isinstance(peak_vram, int | float):
        reasons.append("peak_vram_unavailable")
    elif float(peak_vram) > PEAK_VRAM_LIMIT_MIB:
        reasons.append("peak_vram_exceeds_10500_mib")
    return reasons


def _recommend(results: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    if [row.get("id") for row in results] != [spec.candidate_id for spec in CANDIDATE_SPECS]:
        return None
    eligible = [row for row in results if _candidate_metrics(row).get("eligible") is True]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda row: (
            -cast("float", _candidate_metrics(row)["steady_optimizer_steps_per_second"]),
            cast("float", _candidate_metrics(row)["peak_vram_mib"]),
            str(row["id"]),
        ),
    )
    return dict(selected)


def _candidate_metrics(candidate: Mapping[str, object]) -> Mapping[str, object]:
    metrics = candidate.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _is_conflicting_runtime_command(command_line: str, plan: CandidatePlan) -> bool:
    normalized = command_line.casefold()
    benchmark_needles = (
        str(plan.config_path.resolve()).casefold(),
        str(plan.output_dir.resolve()).casefold(),
        str((plan.upstream_root / "train.py").resolve()).casefold(),
    )
    queue_scripts = (
        "run_600m_speaker_training_queue.py",
        "launch_600m_training_queue_runtime.py",
    )
    is_dataloader = "--multiprocessing-fork" in normalized or "spawn_main(" in normalized
    return (
        any(needle in normalized for needle in benchmark_needles)
        or any(script in normalized for script in queue_scripts)
        or is_dataloader
    )


def _probe_runtime(plan: CandidatePlan, _phase: str) -> dict[str, object]:
    process_command = (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
            "ConvertTo-Json -Compress"
        ),
    )
    errors: list[str] = []
    matching_processes: list[dict[str, object]] = []
    try:
        completed = subprocess.run(  # noqa: S603 - fixed read-only process inventory command.
            process_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode:
            errors.append(completed.stderr.strip() or "process inventory failed")
        elif completed.stdout.strip():
            payload: object = json.loads(completed.stdout)
            rows = payload if isinstance(payload, list) else [payload]
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                command_line = str(raw.get("CommandLine") or "")
                if not _is_conflicting_runtime_command(command_line, plan):
                    continue
                matching_processes.append(
                    {
                        "pid": raw.get("ProcessId"),
                        "parent_pid": raw.get("ParentProcessId"),
                        "name": raw.get("Name"),
                        "command_line": command_line,
                    }
                )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    gpu_memory: float | None = None
    gpu_memory_command = (
        "nvidia-smi",
        "--id=0",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed read-only GPU memory query.
            gpu_memory_command,
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
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "timestamp": _utc_now(),
        "matching_processes": matching_processes,
        "gpu_memory_used_mib": gpu_memory,
        "errors": errors,
    }


def _normalize_runtime_state(raw: Mapping[str, object]) -> dict[str, object]:
    timestamp = raw.get("timestamp")
    processes = raw.get("matching_processes")
    memory = raw.get("gpu_memory_used_mib")
    errors = raw.get("errors", [])
    if not isinstance(timestamp, str):
        raise TypeError("runtime probe timestamp must be a string")
    _parse_timestamp(timestamp)
    if not isinstance(processes, list) or not all(isinstance(row, dict) for row in processes):
        raise TypeError("runtime probe matching_processes must be an object list")
    if memory is not None and not isinstance(memory, int | float):
        raise TypeError("runtime probe gpu_memory_used_mib must be numeric or null")
    if not isinstance(errors, list) or not all(isinstance(error, str) for error in errors):
        raise TypeError("runtime probe errors must be a string list")
    return {
        "timestamp": timestamp,
        "matching_processes": processes,
        "gpu_memory_used_mib": float(memory) if isinstance(memory, int | float) else None,
        "errors": errors,
    }


def _runtime_state_safe_before(state: Mapping[str, object]) -> bool:
    return (
        not state["matching_processes"]
        and isinstance(state["gpu_memory_used_mib"], int | float)
        and not state["errors"]
    )


def _runtime_state_safe_after(
    state: Mapping[str, object],
    *,
    baseline_gpu_memory_mib: float,
    memory_tolerance_mib: float,
) -> bool:
    memory = state["gpu_memory_used_mib"]
    return (
        not state["matching_processes"]
        and isinstance(memory, int | float)
        and float(memory) <= baseline_gpu_memory_mib + memory_tolerance_mib
        and not state["errors"]
    )


def _wait_for_runtime_cleanup(
    plan: CandidatePlan,
    *,
    probe: RuntimeProbe,
    runtime_guard_path: Path,
    baseline_gpu_memory_mib: float,
    timeout: float,
    poll_interval: float,
    memory_tolerance_mib: float,
) -> tuple[dict[str, object], bool]:
    deadline = time.monotonic() + timeout
    while True:
        state = _normalize_runtime_state(probe(plan, "after"))
        safe = _runtime_state_safe_after(
            state,
            baseline_gpu_memory_mib=baseline_gpu_memory_mib,
            memory_tolerance_mib=memory_tolerance_mib,
        )
        _append_jsonl(
            runtime_guard_path,
            {"phase": "after", "quiescent": safe, **state},
            durable=True,
        )
        if safe or time.monotonic() >= deadline:
            return state, safe
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def _blocked_candidate_result(
    plan: CandidatePlan,
    *,
    provenance: Mapping[str, object],
    reason: str,
    runtime_guard_path: Path,
    before_state: Mapping[str, object],
) -> dict[str, object]:
    empty_stats = {"sample_count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "id": plan.candidate_id,
        "batch_size": plan.spec.batch_size,
        "gradient_accumulation_steps": plan.spec.gradient_accumulation_steps,
        "gradient_checkpointing": plan.spec.gradient_checkpointing,
        "effective_global_batch_size": GLOBAL_BATCH_SIZE,
        "overrides": _candidate_overrides(plan.spec),
        "metrics": {
            "measured_optimizer_steps": 0,
            "steady_optimizer_steps_per_second": None,
            "steady_samples_per_second": None,
            "peak_vram_mib": None,
            "gpu_utilization_percent": empty_stats,
            "power_watts": empty_stats,
            "loss_finite": False,
            "learning_rate_fixed": False,
            "observed_step_sequence_valid": False,
            "oom": False,
            "exit_code": None,
            "eligible": False,
            "ineligible_reasons": [reason],
            "reached_step": 0,
            "full_run_peak_vram_mib": None,
            "full_run_gpu_utilization_percent": empty_stats,
            "full_run_power_watts": empty_stats,
        },
        "execution_error": reason,
        "sampler_errors": [],
        "runtime_guard": {"before": before_state, "after": None},
        "artifacts": {
            "config": _file_binding(plan.config_path),
            "runtime_guard": _file_binding(runtime_guard_path),
        },
        "provenance": provenance,
    }


def _run_trainer(plan: CandidatePlan, command: tuple[str, ...], emit: LineSink) -> int:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(  # noqa: S603 - fixed trainer command is built from validated paths.
        command,
        cwd=plan.upstream_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("trainer stdout pipe was not created")
    for line in process.stdout:
        emit(line, None)
    return process.wait()


def _sampler_worker(  # noqa: PLR0917 - thread target receives explicit immutable dependencies.
    sampler: Sampler,
    plan: CandidatePlan,
    path: Path,
    interval: float,
    stop_event: threading.Event,
    errors: list[str],
) -> None:
    try:
        sampler(plan, path, interval, stop_event)
    except Exception as exc:  # noqa: BLE001 - sampler failure is recorded with candidate artifacts.
        errors.append(f"{type(exc).__name__}: {exc}")


def _sample_nvidia_smi(
    _plan: CandidatePlan,
    path: Path,
    interval: float,
    stop_event: threading.Event,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(  # noqa: PLR1702 - keep sampling through malformed rows.
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(GPU_HEADER)
        while True:
            timestamp = _utc_now()
            command = (
                "nvidia-smi",
                "--id=0",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            )
            try:
                completed = subprocess.run(  # noqa: S603 - fixed read-only GPU telemetry command.
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                raw = completed.stdout.strip()
                error = completed.stderr.strip() if completed.returncode else ""
                rows = [line for line in raw.splitlines() if line.strip()]
                if completed.returncode or not rows:
                    writer.writerow((timestamp, "", "", "", "", "", error, raw))
                else:
                    for row in rows:
                        fields = [field.strip() for field in row.split(",")]
                        if len(fields) != GPU_QUERY_FIELD_COUNT:
                            writer.writerow((timestamp, "", "", "", "", "", "invalid CSV", row))
                            continue
                        writer.writerow((timestamp, *fields, "", row))
            except OSError as exc:
                writer.writerow((timestamp, "", "", "", "", "", f"{type(exc).__name__}: {exc}", ""))
            destination.flush()
            if stop_event.wait(interval):
                return


def _ensure_gpu_csv(path: Path, *, errors: Sequence[str]) -> None:
    if path.is_file():
        if errors:
            with path.open("a", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                for error in errors:
                    writer.writerow((_utc_now(), "", "", "", "", "", error, ""))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(GPU_HEADER)
        for error in errors:
            writer.writerow((_utc_now(), "", "", "", "", "", error, ""))


def _read_gpu_samples(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for raw in csv.DictReader(source):
            if raw.get("error"):
                continue
            timestamp = raw.get("timestamp")
            if not timestamp:
                continue
            try:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "utilization_gpu_percent": float(raw["utilization_gpu_percent"]),
                        "memory_used_mib": float(raw["memory_used_mib"]),
                        "power_draw_w": float(raw["power_draw_w"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _max_numeric(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = _numeric_values(rows, field)
    return max(values) if values else None


def _mean_numeric(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = _numeric_values(rows, field)
    return statistics.fmean(values) if values else None


def _numeric_stats(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    values = _numeric_values(rows, field)
    return {
        "sample_count": len(values),
        "minimum": _rounded(min(values) if values else None),
        "mean": _rounded(statistics.fmean(values) if values else None),
        "maximum": _rounded(max(values) if values else None),
    }


def _numeric_values(rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _parse_timestamp(raw: str) -> datetime:
    timestamp = datetime.fromisoformat(raw)
    if timestamp.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {raw}")
    return timestamp


def _build_provenance(  # noqa: PLR0913 - provenance records every benchmark authority.
    *,
    base_config: Path,
    manifest: Path,
    base_checkpoint: Path,
    training_jobs: Path,
    training_status: Path,
    upstream_root: Path,
    upstream_commit: str,
    trainer_python: Path,
    trainer_script: Path,
    git_state: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    manifest_binding: dict[str, object] = {**_file_binding(manifest)}
    manifest_binding["row_count"] = _manifest_row_count(manifest)
    return {
        "base_config": _file_binding(base_config),
        "manifest": manifest_binding,
        "base_checkpoint": _file_binding(base_checkpoint),
        "training_jobs": _file_binding(training_jobs),
        "training_status": _file_binding(training_status),
        "script": _file_binding(Path(__file__)),
        "environment": dict(environment),
        "upstream": {
            "root": str(upstream_root.resolve()),
            "commit": upstream_commit,
            **git_state,
            "trainer_python": _file_binding(trainer_python),
            "trainer_script": _file_binding(trainer_script),
        },
    }


def _file_binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _manifest_row_count(path: Path) -> int:
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload: object = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"manifest row must be an object: {path}:{line_number}")
        count += 1
    return count


def _append_status(path: Path, row: Mapping[str, object]) -> None:
    _append_jsonl(path, row, durable=True)


def _append_jsonl(
    path: Path,
    row: Mapping[str, object],
    *,
    durable: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        destination.flush()
        if durable:
            os.fsync(destination.fileno())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_training_config(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(_expand_scientific_json_numbers(encoded) + "\n", encoding="utf-8")


def _expand_scientific_json_numbers(encoded: str) -> str:
    chunks: list[str] = []
    cursor = 0
    while cursor < len(encoded):
        if encoded[cursor] == '"':
            string_start = cursor
            cursor += 1
            while encoded[cursor] != '"':
                cursor += 2 if encoded[cursor] == "\\" else 1
            cursor += 1
            chunks.append(encoded[string_start:cursor])
            continue
        number = JSON_NUMBER_PATTERN.match(encoded, cursor)
        if number is None:
            chunks.append(encoded[cursor])
            cursor += 1
            continue
        token = number.group()
        if "e" in token.casefold():
            token = format(Decimal(token), "f")
            if "." not in token:
                token += ".0"
        chunks.append(token)
        cursor = number.end()
    return "".join(chunks)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary JSON output: {temporary}")
    try:
        _write_json(temporary, payload)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload: object = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - remote runtime is Python 3.10.


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--training-jobs", type=Path, required=True)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sampler-interval",
        type=float,
        default=DEFAULT_SAMPLER_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--cleanup-timeout",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--cleanup-poll-interval",
        type=float,
        default=DEFAULT_CLEANUP_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--cleanup-memory-tolerance-mib",
        type=float,
        default=DEFAULT_CLEANUP_MEMORY_TOLERANCE_MIB,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner | None = None,
    sampler: Sampler | None = None,
    git_inspector: GitInspector | None = None,
    environment_inspector: EnvironmentInspector | None = None,
    runtime_probe: RuntimeProbe | None = None,
    cleanup_timeout: float | None = None,
    cleanup_poll_interval: float | None = None,
) -> int:
    args = _parse_args(argv)
    summary = run_benchmark(
        base_config=args.base_config,
        manifest=args.manifest,
        base_checkpoint=args.base_checkpoint,
        upstream_root=args.upstream_root,
        upstream_commit=args.upstream_commit,
        training_jobs=args.training_jobs,
        training_status=args.training_status,
        output_root=args.output_root,
        runner=runner,
        sampler=sampler,
        sampler_interval=args.sampler_interval,
        git_inspector=git_inspector,
        environment_inspector=environment_inspector,
        runtime_probe=runtime_probe,
        cleanup_timeout=args.cleanup_timeout if cleanup_timeout is None else cleanup_timeout,
        cleanup_poll_interval=(
            args.cleanup_poll_interval if cleanup_poll_interval is None else cleanup_poll_interval
        ),
        cleanup_memory_tolerance_mib=args.cleanup_memory_tolerance_mib,
    )
    print(json.dumps(summary["recommended_candidate"], ensure_ascii=False, sort_keys=True))
    return int(summary["recommended_candidate"] is None)


if __name__ == "__main__":
    raise SystemExit(main())
