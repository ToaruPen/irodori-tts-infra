# ruff: noqa: BLE001, C901, EM101, EM102, PLR0912, PLR0913, PLR0914, PLR0915, S404, TRY003, TRY004, TRY301
# Operational failures retain exact artifact context; subprocesses use fixed argv and no shell.

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import math
import os
import re
import socket
import stat
import subprocess
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence


_UTC = timezone.utc  # noqa: UP017 - pinned Windows runtime uses Python 3.10.
MIN_FREE_GPU_MIB = 10_500.0
EXPECTED_JOB_COUNT = 12
WINDOWS_DETACHED_FLAGS = 0x00000008 | 0x00000200 | 0x01000000
RESERVATION_SCHEMA = "speaker-training-detached-reservation/v1"
PARENT_HANDOFF_SCHEMA = "speaker-training-detached-parent-handoff/v1"
SUPERVISOR_START_SCHEMA = "speaker-training-detached-supervisor-start/v1"
TERMINAL_SCHEMA = "speaker-training-detached-terminal/v1"
SPAWN_FAILURE_SCHEMA = "speaker-training-detached-spawn-failure/v1"
RESERVATION_FAILURE_SCHEMA = "speaker-training-detached-reservation-failure/v1"
BOOTSTRAP_FAILURE_SCHEMA = "speaker-training-detached-bootstrap-failure/v1"
LOCK_SCHEMA = "speaker-training-detached-lock/v1"
QUEUE_LOCK_SCHEMA = "speaker-training-detached-queue-lock/v1"
OUTPUT_IDENTITY_SCHEMA = "speaker-training-detached-output-identity/v1"
LOCK_MUTATION_PROTOCOL = "speaker-training-detached-v002-cooperative-os-mutex/v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_EVIDENCE_ROOT_RE = re.compile(r"speaker-training-detached-v[0-9]{3,}\Z")
_LAUNCH_ID_RE = re.compile(r"launch-v([0-9]{3,})\Z")
_TRAINING_MARKERS = (
    "run_600m_speaker_training_queue.py",
    "train_speaker_inversion.py",
    "speaker_inversion",
)
_GENERIC_TRAIN_SCRIPT_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])train\.py(?![A-Za-z0-9_.-])", re.IGNORECASE
)
_SERVICE_MARKERS = (
    "remote_server.py",
    "uvicorn",
    "irodori_tts_infra.server",
    "irodori-tts-server",
)

_RESERVATION_NAME = "reservation-evidence.json"
_PARENT_HANDOFF_NAME = "parent-handoff-evidence.json"
_SUPERVISOR_START_NAME = "supervisor-start-evidence.json"
_TERMINAL_NAME = "terminal-final-evidence.json"
_SPAWN_FAILURE_NAME = "spawn-failure-evidence.json"
_RESERVATION_FAILURE_NAME = "reservation-failure-evidence.json"
_BOOTSTRAP_FAILURE_NAME = "supervisor-bootstrap-error.json"
_ACTIVE_LOCK_NAME = "supervisor.lock"
_FINAL_LOCK_NAME = "supervisor-lock-final.json"
_FINAL_QUEUE_LOCK_NAME = "queue-lock-final.json"
_FOREIGN_QUEUE_LOCK_SNAPSHOT_NAME = "foreign-queue-lock-snapshot.json"
_QUEUE_LOG_NAME = "queue.log"


class ContractArguments(TypedDict):
    queue_script: Path
    expected_queue_sha256: str
    jobs_path: Path
    expected_jobs_sha256: str
    status_path: Path
    checkpoint_path: Path
    expected_checkpoint_sha256: str
    checkpoint_revision: str
    upstream_root: Path
    expected_upstream_commit: str
    python_path: Path
    evidence_root: Path
    detached_script: Path
    target_model_id: str


class RuntimeSnapshot:
    __slots__ = (
        "errors",
        "gpu_free_mib",
        "gpu_total_mib",
        "gpu_used_mib",
        "observed_at",
        "processes",
        "service_processes",
        "training_processes",
    )

    def __init__(
        self,
        *,
        observed_at: str,
        processes: Sequence[Mapping[str, object]],
        training_processes: Sequence[Mapping[str, object]],
        service_processes: Sequence[Mapping[str, object]],
        gpu_total_mib: float | None,
        gpu_used_mib: float | None,
        gpu_free_mib: float | None,
        errors: Sequence[str],
    ) -> None:
        self.observed_at = observed_at
        self.processes = tuple(dict(row) for row in processes)
        self.training_processes = tuple(dict(row) for row in training_processes)
        self.service_processes = tuple(dict(row) for row in service_processes)
        self.gpu_total_mib = gpu_total_mib
        self.gpu_used_mib = gpu_used_mib
        self.gpu_free_mib = gpu_free_mib
        self.errors = tuple(errors)

    def as_dict(self) -> dict[str, object]:
        return {
            "observed_at": self.observed_at,
            "processes": self.processes,
            "training_processes": self.training_processes,
            "service_processes": self.service_processes,
            "gpu_total_mib": self.gpu_total_mib,
            "gpu_used_mib": self.gpu_used_mib,
            "gpu_free_mib": self.gpu_free_mib,
            "errors": self.errors,
        }


def _utc_now() -> str:
    return datetime.now(_UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _nominal_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))  # noqa: PTH100 - resolve() would follow aliases.


def _is_filesystem_alias(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _require_alias_free(path: Path, *, label: str) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    current = Path(lexical.anchor)
    parts = lexical.parts[1:] if lexical.anchor else lexical.parts
    for part in parts:
        current /= part
        if current != current.parent and _is_filesystem_alias(current):
            raise ValueError(f"{label} has a symlink, junction, or reparse alias: {current}")
    nominal = _nominal_absolute(path)
    for candidate in (nominal, *nominal.parents):
        if candidate != candidate.parent and _is_filesystem_alias(candidate):
            raise ValueError(f"{label} has a symlink, junction, or reparse alias: {candidate}")
    return nominal


def _file_binding(path: Path) -> dict[str, object] | None:
    nominal = _require_alias_free(path, label="bound file")
    if not nominal.is_file():
        return None
    return {
        "path": str(nominal.resolve()),
        "sha256": sha256_file(nominal),
        "size": nominal.stat().st_size,
    }


def _required_file_binding(path: Path) -> dict[str, object]:
    binding = _file_binding(path)
    if binding is None:
        raise RuntimeError(f"required immutable evidence is missing: {path}")
    return binding


def _verify_pinned_file(path: Path, expected_sha256: str, *, label: str) -> dict[str, object]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} SHA-256 pin is not finalized: {expected_sha256}")
    binding = _file_binding(path)
    if binding is None:
        raise ValueError(f"{label} is unsafe or missing: {path}")
    if binding["sha256"] != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha256}, "
            f"actual={binding['sha256']}, path={binding['path']}"
        )
    return binding


def _verify_unpinned_file(path: Path, *, label: str) -> dict[str, object]:
    binding = _file_binding(path)
    if binding is None:
        raise ValueError(f"{label} is unsafe or missing: {path}")
    return binding


def _queue_lock_path(status_path: Path) -> Path:
    return status_path.with_suffix(status_path.suffix + ".detached.lock")


def _validate_output_identity(
    *,
    evidence_root: Path,
    status_path: Path,
    expected: object | None = None,
    launch_dir: Path | None = None,
) -> dict[str, str]:
    root = _require_alias_free(evidence_root, label="training evidence root")
    if _EVIDENCE_ROOT_RE.fullmatch(root.name) is None:
        raise ValueError(f"expected a versioned evidence root: {root}")
    status = _require_alias_free(status_path, label="training status path")
    queue_lock = _require_alias_free(_queue_lock_path(status), label="training queue lock")
    launches = _require_alias_free(root / "launches", label="training launches root")
    if launches.parent != root:
        raise ValueError("training launches root identity mismatch")
    identity = {
        "schema_version": OUTPUT_IDENTITY_SCHEMA,
        "evidence_root": str(root),
        "launches_root": str(launches),
        "status_path": str(status),
        "queue_lock_path": str(queue_lock),
    }
    if expected is not None and identity != expected:
        raise RuntimeError(
            f"fixed output identity mismatch: expected={expected}, actual={identity}"
        )
    if launch_dir is not None:
        launch = _require_alias_free(launch_dir, label="training launch directory")
        if launch.parent != launches or _LAUNCH_ID_RE.fullmatch(launch.name) is None:
            raise ValueError(f"launch directory is outside fixed launches root: {launch}")
    return identity


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _resolve_manifest_path(raw: object, *, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"training jobs require nonempty {label}")
    value = Path(raw)
    return _nominal_absolute(value if value.is_absolute() else base / value)


def _verify_upstream(
    root: Path,
    expected_commit: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> dict[str, str]:
    if _COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError(f"upstream commit pin is not finalized: {expected_commit}")
    nominal = _require_alias_free(root, label="upstream root")
    if not nominal.is_dir():
        raise ValueError(f"upstream root is unsafe or missing: {nominal}")
    execute = runner or subprocess.run
    command = ("git", "-C", str(nominal), "rev-parse", "HEAD")
    completed = execute(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(f"upstream Git verification failed: {completed.stderr.strip()}")
    actual = completed.stdout.strip()
    if actual != expected_commit:
        raise ValueError(
            f"upstream commit mismatch: expected={expected_commit}, actual={actual}, path={nominal}"
        )
    status_command = (
        "git",
        "-C",
        str(nominal),
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    status = execute(
        status_command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    if status.returncode:
        raise RuntimeError(f"upstream worktree verification failed: {status.stderr.strip()}")
    if status.stdout.strip():
        raise ValueError(f"upstream tracked worktree is not clean: {status.stdout.strip()}")

    train_path = _require_alias_free(nominal / "train.py", label="upstream train.py")
    package_path = _require_alias_free(nominal / "irodori_tts", label="upstream irodori_tts")
    if not train_path.is_file():
        raise ValueError(f"upstream train.py is missing or not a regular file: {train_path}")
    if not package_path.is_dir():
        raise ValueError(f"upstream irodori_tts is missing or not a directory: {package_path}")

    tracked_command = (
        "git",
        "-C",
        str(nominal),
        "ls-files",
        "--error-unmatch",
        "--",
        "train.py",
        "irodori_tts",
    )
    tracked = execute(
        tracked_command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    tracked_paths = tuple(line.strip() for line in tracked.stdout.splitlines() if line.strip())
    train_is_tracked = "train.py" in tracked_paths
    package_is_tracked = any(path.startswith("irodori_tts/") for path in tracked_paths)
    if not train_is_tracked:
        raise ValueError("upstream train.py must be tracked at the pinned commit")
    if not package_is_tracked:
        raise ValueError("upstream irodori_tts must contain tracked source at the pinned commit")
    if tracked.returncode:
        raise RuntimeError(f"upstream critical path verification failed: {tracked.stderr.strip()}")
    for relative in tracked_paths:
        if relative != "train.py" and not relative.startswith("irodori_tts/"):
            continue
        tracked_path = _require_alias_free(
            nominal / relative,
            label="tracked upstream path",
        )
        if not tracked_path.is_file():
            raise ValueError(f"tracked upstream file is missing or not regular: {tracked_path}")

    critical_untracked: list[str] = []
    for extra_flags in (
        ("--others", "--exclude-standard"),
        ("--others", "--ignored", "--exclude-standard"),
    ):
        untracked_command = (
            "git",
            "-C",
            str(nominal),
            "ls-files",
            *extra_flags,
            "--",
            "train.py",
            "irodori_tts",
        )
        untracked = execute(
            untracked_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
        if untracked.returncode:
            raise RuntimeError(
                f"upstream untracked source verification failed: {untracked.stderr.strip()}"
            )
        critical_untracked.extend(
            line.strip() for line in untracked.stdout.splitlines() if line.strip()
        )
    importable_suffixes = {
        *importlib.machinery.SOURCE_SUFFIXES,
        *importlib.machinery.BYTECODE_SUFFIXES,
        *importlib.machinery.EXTENSION_SUFFIXES,
        ".pyd",
        ".pyi",
        ".pyx",
    }
    untracked_source = tuple(
        path
        for path in critical_untracked
        if path == "train.py"
        or any(path.casefold().endswith(suffix.casefold()) for suffix in importable_suffixes)
    )
    if untracked_source:
        raise ValueError(f"untracked source in critical upstream scope: {untracked_source[0]}")
    return {
        "path": str(nominal.resolve()),
        "commit": actual,
        "tracked_worktree": "clean",
        "critical_paths": "tracked-clean-no-untracked-source",
    }


def _required_job_string(job: Mapping[str, object], name: str, *, model_id: str) -> str:
    value = job.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"training job {model_id!r} requires nonempty {name}")
    return value


def _job_input_contracts(
    raw_jobs: Sequence[object],
    *,
    jobs_base: Path,
    target_model_id: str,
    base_checkpoint: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    model_ids: list[str] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise TypeError("training jobs must contain only job objects")
        job = cast("Mapping[str, object]", raw_job)
        model_id = _required_job_string(job, "model_id", model_id="unknown")
        if model_id in model_ids:
            raise ValueError(f"training jobs contain duplicate model_id: {model_id}")
        model_ids.append(model_id)
        manifest = _resolve_manifest_path(
            job.get("clean_manifest"), base=jobs_base, label=f"{model_id} clean_manifest"
        )
        config = _resolve_manifest_path(
            job.get("config"), base=jobs_base, label=f"{model_id} config"
        )
        output = _resolve_manifest_path(
            job.get("output_dir"), base=jobs_base, label=f"{model_id} output_dir"
        )
        manifest_binding = _verify_unpinned_file(manifest, label=f"{model_id} clean manifest")
        config_binding = _verify_unpinned_file(config, label=f"{model_id} config")
        output = _require_alias_free(output, label=f"{model_id} output directory")
        command = job.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ValueError(f"training job {model_id!r} command must be a nonempty string list")
        expected_command_paths = {
            "--config": config,
            "--manifest": manifest,
            "--init-checkpoint": base_checkpoint,
            "--output-dir": output,
        }
        for flag, expected in expected_command_paths.items():
            indexes = [index for index, part in enumerate(command) if part == flag]
            if len(indexes) != 1 or indexes[0] + 1 >= len(command):
                raise ValueError(f"training job {model_id!r} requires exactly one {flag} value")
            command_path = _resolve_manifest_path(
                command[indexes[0] + 1],
                base=jobs_base,
                label=f"{model_id} command {flag}",
            )
            if command_path != expected:
                raise ValueError(f"training job {model_id!r} command {flag} path mismatch")
        config_document = _read_json(config)
        train = config_document.get("train")
        if not isinstance(train, dict):
            raise TypeError(f"training config train must be an object for {model_id}")
        configured_manifest = _resolve_manifest_path(
            train.get("manifest_path"), base=config.parent, label=f"{model_id} manifest_path"
        )
        configured_output = _resolve_manifest_path(
            train.get("output_dir"), base=config.parent, label=f"{model_id} output_dir"
        )
        if configured_manifest != manifest or configured_output != output:
            raise ValueError(f"training config paths do not match job paths for {model_id}")
        row: dict[str, object] = {
            "model_id": model_id,
            "clean_manifest": manifest_binding,
            "config": config_binding,
            "output_dir": str(output),
            "command": list(command),
        }
        if model_id == target_model_id:
            init_path = _resolve_manifest_path(
                train.get("speaker_inversion_init_embedding"),
                base=config.parent,
                label=f"{model_id} speaker_inversion_init_embedding",
            )
            init_binding = _verify_unpinned_file(
                init_path, label=f"{model_id} speaker inversion init embedding"
            )
            configured_sha = train.get("speaker_inversion_init_embedding_sha256")
            if configured_sha is not None and configured_sha != init_binding["sha256"]:
                raise ValueError(
                    f"training config initialization checkpoint SHA-256 mismatch for {model_id}"
                )
            row["speaker_inversion_init_embedding"] = init_binding
        rows.append(row)
    if model_ids.count(target_model_id) != 1:
        raise ValueError("target model must occur exactly once in training jobs")
    return rows, model_ids


def verify_contract(
    *,
    queue_script: Path,
    expected_queue_sha256: str,
    jobs_path: Path,
    expected_jobs_sha256: str,
    status_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    checkpoint_revision: str,
    upstream_root: Path,
    expected_upstream_commit: str,
    python_path: Path,
    evidence_root: Path,
    detached_script: Path,
    target_model_id: str,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, object]:
    output_identity = _validate_output_identity(
        evidence_root=evidence_root,
        status_path=status_path,
    )
    queue = _verify_pinned_file(queue_script, expected_queue_sha256, label="training queue")
    jobs = _verify_pinned_file(jobs_path, expected_jobs_sha256, label="training jobs")
    checkpoint = _verify_pinned_file(
        checkpoint_path,
        expected_checkpoint_sha256,
        label="base checkpoint",
    )
    python = _verify_unpinned_file(python_path, label="Python executable")
    detached = _verify_unpinned_file(detached_script, label="detached launcher")
    upstream = _verify_upstream(upstream_root, expected_upstream_commit, runner=git_runner)
    jobs_document = _read_json(Path(cast("str", jobs["path"])))
    raw_jobs = jobs_document.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_JOB_COUNT:
        raise ValueError("training jobs must contain exactly 12 jobs")
    job_inputs, model_ids = _job_input_contracts(
        raw_jobs,
        jobs_base=Path(cast("str", jobs["path"])).parent,
        target_model_id=target_model_id,
        base_checkpoint=Path(cast("str", checkpoint["path"])),
    )
    declared_checkpoint = _resolve_manifest_path(
        jobs_document.get("base_checkpoint_path"),
        base=Path(cast("str", jobs["path"])).parent,
        label="base_checkpoint_path",
    )
    if declared_checkpoint != Path(cast("str", checkpoint["path"])):
        raise ValueError("training jobs base checkpoint path mismatch")
    if jobs_document.get("base_checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("training jobs base checkpoint SHA-256 mismatch")
    if jobs_document.get("checkpoint_revision") != checkpoint_revision:
        raise ValueError("training jobs checkpoint revision mismatch")
    if jobs_document.get("upstream_commit") != expected_upstream_commit:
        raise ValueError("training jobs upstream commit mismatch")
    return {
        "queue_script": queue,
        "jobs": jobs,
        "status_before": _file_binding(status_path),
        "checkpoint": checkpoint,
        "checkpoint_revision": checkpoint_revision,
        "upstream": upstream,
        "python": python,
        "detached_launcher": detached,
        "output_identity": output_identity,
        "target_model_id": target_model_id,
        "expected_skipped_model_ids": [
            model_id for model_id in model_ids if model_id != target_model_id
        ],
        "job_inputs": job_inputs,
    }


def _pid(row: Mapping[str, object]) -> int | None:
    for name in ("pid", "ProcessId"):
        value = row.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _parent_pid(row: Mapping[str, object]) -> int | None:
    for name in ("parent_pid", "ParentProcessId"):
        value = row.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _command_line(row: Mapping[str, object]) -> str:
    for name in ("command_line", "CommandLine", "name", "Name"):
        value = row.get(name)
        if isinstance(value, str):
            return value
    return ""


def _command_has_token(command_line: str, token: str) -> bool:
    normalized = command_line.casefold()
    expected = token.casefold()
    pattern = rf"(?:^|[\s=])[\"']?{re.escape(expected)}[\"']?(?=$|\s)"
    return re.search(pattern, normalized) is not None


def _is_training_command(
    command_line: str,
    contract: Mapping[str, object] | None = None,
) -> bool:
    folded = command_line.casefold()
    if any(marker in folded for marker in _TRAINING_MARKERS):
        return True
    if _GENERIC_TRAIN_SCRIPT_RE.search(command_line) is not None:
        return True
    if contract is None:
        return False
    raw_inputs = contract.get("job_inputs")
    if not isinstance(raw_inputs, list):
        raise RuntimeError("training contract job inputs are invalid")
    for raw in raw_inputs:
        if not isinstance(raw, dict):
            raise RuntimeError("training contract job input is invalid")
        config = raw.get("config")
        if (
            isinstance(config, dict)
            and isinstance(config.get("path"), str)
            and _command_has_token(command_line, cast("str", config["path"]))
        ):
            return True
    return False


def _ancestor_pids(rows: Sequence[Mapping[str, object]], *, current_pid: int) -> set[int]:
    parents = {
        pid: parent
        for row in rows
        if (pid := _pid(row)) is not None and (parent := _parent_pid(row)) is not None
    }
    excluded = {current_pid}
    cursor = current_pid
    while (parent := parents.get(cursor)) is not None and parent not in excluded and parent > 0:
        excluded.add(parent)
        cursor = parent
    return excluded


def _parse_processes(stdout: str) -> list[dict[str, object]]:
    payload: object = json.loads(stdout)
    raw = payload if isinstance(payload, list) else [payload]
    if not raw or not all(isinstance(row, dict) for row in raw):
        raise ValueError("process inventory must be a nonempty object list")
    rows = [dict(cast("Mapping[str, object]", row)) for row in raw]
    if not any(_pid(row) is not None for row in rows):
        raise ValueError("process inventory requires a positive ProcessId")
    return rows


def _parse_gpu_memory(stdout: str) -> tuple[float, float]:
    total_raw, used_raw = (part.strip() for part in stdout.strip().splitlines()[0].split(",", 1))
    total, used = float(total_raw), float(used_raw)
    if not all(math.isfinite(value) and value >= 0 for value in (total, used)) or used > total:
        raise ValueError("GPU memory values are invalid")
    return total, used


def probe_runtime(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    current_pid: int | None = None,
) -> RuntimeSnapshot:
    execute = runner or subprocess.run
    own_pid = current_pid or os.getpid()
    errors: list[str] = []
    processes: list[dict[str, object]] = []
    total: float | None = None
    used: float | None = None
    try:
        completed = execute(
            (
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "Get-CimInstance Win32_Process | Select-Object "
                    "ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
                ),
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
        if completed.returncode:
            errors.append(completed.stderr.strip() or "process inventory failed")
        else:
            processes = _parse_processes(completed.stdout)
            if sum(_pid(row) == own_pid for row in processes) != 1:
                raise ValueError("process inventory must contain exactly one current process")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    try:
        completed = execute(
            (
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
        if completed.returncode:
            errors.append(completed.stderr.strip() or "GPU memory query failed")
        else:
            total, used = _parse_gpu_memory(completed.stdout)
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    excluded = _ancestor_pids(processes, current_pid=own_pid)
    visible = tuple(row for row in processes if _pid(row) not in excluded)
    training = tuple(row for row in visible if _is_training_command(_command_line(row)))
    services = tuple(
        row
        for row in visible
        if any(marker in _command_line(row).casefold() for marker in _SERVICE_MARKERS)
    )
    return RuntimeSnapshot(
        observed_at=_utc_now(),
        processes=visible,
        training_processes=training,
        service_processes=services,
        gpu_total_mib=total,
        gpu_used_mib=used,
        gpu_free_mib=total - used if total is not None and used is not None else None,
        errors=errors,
    )


def _existing_supervisor_locks(evidence_root: Path) -> tuple[Path, ...]:
    launches = evidence_root / "launches"
    return tuple(sorted(launches.glob("launch-v*/supervisor.lock"))) if launches.is_dir() else ()


def _require_safe_runtime(
    snapshot: RuntimeSnapshot,
    *,
    evidence_root: Path,
    status_path: Path,
    allowed_supervisor_lock: Path | None = None,
    allowed_queue_lock: Path | None = None,
    contract: Mapping[str, object] | None = None,
) -> None:
    if snapshot.errors:
        raise RuntimeError(f"runtime probe failed closed: {snapshot.errors}")
    queue_lock = _queue_lock_path(status_path)
    if os.path.lexists(queue_lock) and _nominal_absolute(queue_lock) != (
        _nominal_absolute(allowed_queue_lock) if allowed_queue_lock is not None else None
    ):
        raise RuntimeError(f"training queue lock already exists: {queue_lock}")
    allowed_supervisor = (
        _nominal_absolute(allowed_supervisor_lock) if allowed_supervisor_lock is not None else None
    )
    locks = tuple(
        lock
        for lock in _existing_supervisor_locks(evidence_root)
        if _nominal_absolute(lock) != allowed_supervisor
    )
    if locks:
        raise RuntimeError(f"detached supervisor lock already exists: {locks[0]}")
    contract_training = tuple(
        row for row in snapshot.processes if _is_training_command(_command_line(row), contract)
    )
    training_processes = tuple(snapshot.training_processes) + tuple(
        row for row in contract_training if row not in snapshot.training_processes
    )
    if training_processes:
        raise RuntimeError(f"training-owned process is already running: {training_processes}")
    if snapshot.service_processes:
        raise RuntimeError(
            f"TTS service must be stopped by the operator: {snapshot.service_processes}"
        )
    if snapshot.gpu_free_mib is None or snapshot.gpu_free_mib < MIN_FREE_GPU_MIB:
        raise RuntimeError(
            f"insufficient free GPU memory: required_mib={MIN_FREE_GPU_MIB}, "
            f"actual_mib={snapshot.gpu_free_mib}"
        )


def _queue_command(contract: Mapping[str, object], *, dry_run: bool) -> tuple[str, ...]:
    queue = cast("Mapping[str, object]", contract["queue_script"])
    jobs = cast("Mapping[str, object]", contract["jobs"])
    checkpoint = cast("Mapping[str, object]", contract["checkpoint"])
    python = cast("Mapping[str, object]", contract["python"])
    upstream = cast("Mapping[str, object]", contract["upstream"])
    identity = cast("Mapping[str, object]", contract["output_identity"])
    command = (
        cast("str", python["path"]),
        cast("str", queue["path"]),
        "--jobs-json",
        cast("str", jobs["path"]),
        "--status-path",
        cast("str", identity["status_path"]),
        "--checkpoint",
        cast("str", checkpoint["path"]),
        "--checkpoint-revision",
        cast("str", contract["checkpoint_revision"]),
        "--upstream-commit",
        cast("str", upstream["commit"]),
    )
    return (*command, "--dry-run") if dry_run else command


def _last_json_object(text: str, *, source: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"{source} has no JSON object line")


def _validate_queue_report(
    report: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    completed: bool,
) -> None:
    target = contract.get("target_model_id")
    skipped = contract.get("expected_skipped_model_ids")
    expected = {
        "planned": [target],
        "succeeded": [target] if completed else [],
        "failed": [],
        "skipped": skipped,
    }
    label = "completion" if completed else "dry-run"
    mismatched = any(report.get(name) != value for name, value in expected.items())
    if set(report) != set(expected) or mismatched:
        raise RuntimeError(f"training queue {label} report is unsafe: {dict(report)}")


def _run_queue_dry_run(
    contract: Mapping[str, object],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> dict[str, Any]:
    execute = runner or subprocess.run
    command = _queue_command(contract, dry_run=True)
    upstream = cast("Mapping[str, object]", contract["upstream"])
    completed = execute(
        command,
        cwd=cast("str", upstream["path"]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"training queue dry-run failed: exit_code={completed.returncode}, "
            f"stderr={completed.stderr.strip()}"
        )
    report = _last_json_object(completed.stdout, source="training queue dry-run")
    _validate_queue_report(report, contract, completed=False)
    return report


def preflight_detached(
    *,
    queue_script: Path,
    expected_queue_sha256: str,
    jobs_path: Path,
    expected_jobs_sha256: str,
    status_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    checkpoint_revision: str,
    upstream_root: Path,
    expected_upstream_commit: str,
    python_path: Path,
    evidence_root: Path,
    detached_script: Path,
    target_model_id: str,
    probe: Callable[[], RuntimeSnapshot] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, object]:
    arguments: ContractArguments = {
        "queue_script": queue_script,
        "expected_queue_sha256": expected_queue_sha256,
        "jobs_path": jobs_path,
        "expected_jobs_sha256": expected_jobs_sha256,
        "status_path": status_path,
        "checkpoint_path": checkpoint_path,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_revision": checkpoint_revision,
        "upstream_root": upstream_root,
        "expected_upstream_commit": expected_upstream_commit,
        "python_path": python_path,
        "evidence_root": evidence_root,
        "detached_script": detached_script,
        "target_model_id": target_model_id,
    }
    contract = verify_contract(**arguments, git_runner=git_runner)
    snapshot = (probe or probe_runtime)()
    _require_safe_runtime(
        snapshot,
        evidence_root=evidence_root,
        status_path=status_path,
        contract=contract,
    )
    report = _run_queue_dry_run(contract, runner=runner)
    after = verify_contract(**arguments, git_runner=git_runner)
    if after != contract:
        raise RuntimeError("training inputs or mutable status changed during preflight")
    runtime_after = (probe or probe_runtime)()
    _require_safe_runtime(
        runtime_after,
        evidence_root=evidence_root,
        status_path=status_path,
        contract=contract,
    )
    return {
        "passed": True,
        "launch_performed": False,
        "checked_at": _utc_now(),
        "contract": contract,
        "queue_dry_run": report,
        "runtime": snapshot.as_dict(),
        "runtime_after_dry_run": runtime_after.as_dict(),
        "minimum_free_gpu_mib": MIN_FREE_GPU_MIB,
    }


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _write_bytes_create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes_create_only(path, _json_bytes(payload))


def reserve_launch_directory(launches_root: Path, launch_id: str | None = None) -> Path:
    launches = _require_alias_free(launches_root, label="training launches root")
    if launches.name != "launches" or _EVIDENCE_ROOT_RE.fullmatch(launches.parent.name) is None:
        raise ValueError(f"launches root is outside a versioned evidence root: {launches}")
    launches.mkdir(parents=True, exist_ok=True)
    if launch_id is not None:
        if _LAUNCH_ID_RE.fullmatch(launch_id) is None:
            raise ValueError(f"invalid launch id: {launch_id}")
        destination = launches / launch_id
        try:
            destination.mkdir()
        except FileExistsError:
            raise FileExistsError(f"refusing to reuse launch directory: {destination}") from None
        return _require_alias_free(destination, label="training launch directory")
    versions = [
        int(match.group(1))
        for child in launches.iterdir()
        if (match := _LAUNCH_ID_RE.fullmatch(child.name)) is not None
    ]
    version = max(versions, default=0) + 1
    while True:
        destination = launches / f"launch-v{version:03d}"
        try:
            destination.mkdir()
        except FileExistsError:
            version += 1
            continue
        return _require_alias_free(destination, label="training launch directory")


def _lock_payload(
    *,
    schema: str,
    token: str,
    supervisor_pid: int | None,
    status_path: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": schema,
        "launch_token": token,
        "hostname": socket.gethostname(),
        "launcher_pid": os.getpid(),
        "supervisor_pid": supervisor_pid,
        "updated_at": _utc_now(),
        "mutation_protocol": LOCK_MUTATION_PROTOCOL,
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
    }
    if status_path is not None:
        payload["status_path"] = status_path
    return payload


def _is_owned_v002_lock(
    payload: Mapping[str, object],
    *,
    token: str,
    schema: str | None = None,
) -> bool:
    actual_schema = payload.get("schema_version")
    return (
        payload.get("launch_token") == token
        and actual_schema in {LOCK_SCHEMA, QUEUE_LOCK_SCHEMA}
        and (schema is None or actual_schema == schema)
        and payload.get("mutation_protocol") == LOCK_MUTATION_PROTOCOL
        and payload.get("launcher_sha256") == sha256_file(Path(__file__).resolve())
    )


@contextmanager
def _lock_mutex(path: Path) -> Iterator[None]:
    # v002 invariant: every cooperating launcher serializes create/update/archive here.
    # A lock without this protocol and the exact launcher binding is never mutated.
    mutex = _require_alias_free(
        path.with_name(f".{path.name}.mutation-mutex"), label="lock mutation mutex"
    )
    mutex.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(mutex, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        try:
            if os.name == "nt":  # pragma: no cover - exercised on the pinned Windows host.
                import msvcrt  # noqa: PLC0415

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(  # type: ignore[attr-defined]
                    descriptor,
                    msvcrt.LK_NBLCK,  # type: ignore[attr-defined]
                    1,
                )
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError(f"lock mutation mutex is busy: {path}") from exc
        yield
    finally:
        if acquired:
            if os.name == "nt":  # pragma: no cover - exercised on the pinned Windows host.
                import msvcrt  # noqa: PLC0415

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(  # type: ignore[attr-defined]
                    descriptor,
                    msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                    1,
                )
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _create_lock(
    path: Path,
    payload: Mapping[str, object],
    *,
    conflict_snapshot: Path | None = None,
) -> None:
    with _lock_mutex(path):
        try:
            _write_json_create_only(path, payload)
        except FileExistsError:
            if conflict_snapshot is not None:
                _write_bytes_create_only(conflict_snapshot, path.read_bytes())
            raise


def _read_lock_bytes(path: Path) -> tuple[dict[str, Any], bytes, os.stat_result]:
    with path.open("rb") as source:
        content = source.read()
        metadata = os.fstat(source.fileno())
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise TypeError(f"lock JSON document must be an object: {path}")
    return payload, content, metadata


def _assert_lock_unchanged(path: Path, content: bytes, metadata: os.stat_result) -> None:
    current = path.stat()
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_ctime_ns, metadata.st_size)
    current_identity = (current.st_dev, current.st_ino, current.st_ctime_ns, current.st_size)
    if current_identity != identity or path.read_bytes() != content:
        raise RuntimeError(f"lock ownership changed during mutation: {path}")


def _archive_lock(source: Path, destination: Path, *, token: str) -> dict[str, object]:
    with _lock_mutex(source):
        payload, content, metadata = _read_lock_bytes(source)
        if not _is_owned_v002_lock(payload, token=token):
            raise RuntimeError(f"lock ownership mismatch: {source}")
        created = False
        try:
            _write_bytes_create_only(destination, content)
            created = True
            _assert_lock_unchanged(source, content, metadata)
            source.unlink()
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return _required_file_binding(destination)


def _archive_owned_or_snapshot_foreign(
    source: Path,
    destination: Path,
    *,
    token: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    archived = _file_binding(destination)
    if archived is not None:
        archived_payload = _read_json(destination)
        if not _is_owned_v002_lock(archived_payload, token=token):
            raise RuntimeError(f"archived lock ownership mismatch: {destination}")
    if not source.is_file():
        if archived is None:
            raise RuntimeError(f"lock disappeared without an archive: {source}")
        return archived, None
    current = _read_json(source)
    if not _is_owned_v002_lock(current, token=token):
        return archived, _required_file_binding(source)
    if archived is not None:
        raise RuntimeError(f"owned active lock conflicts with its archive: {source}")
    try:
        return _archive_lock(source, destination, token=token), None
    except RuntimeError as exc:
        if "lock ownership changed" not in str(exc):
            raise
        foreign = _required_file_binding(source)
        if _is_owned_v002_lock(_read_json(source), token=token):
            raise RuntimeError(
                f"owned lock mutation failed without ownership loss: {source}"
            ) from exc
        return None, foreign


def _wait_for_worker_registration(
    supervisor_lock: Path,
    queue_lock: Path,
    *,
    token: str,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        supervisor = _read_json(supervisor_lock)
        queue = _read_json(queue_lock)
        if supervisor.get("launch_token") != token or queue.get("launch_token") != token:
            raise RuntimeError("detached worker lock ownership mismatch")
        supervisor_pid = supervisor.get("supervisor_pid")
        if (
            isinstance(supervisor_pid, int)
            and not isinstance(supervisor_pid, bool)
            and supervisor_pid > 0
            and queue.get("supervisor_pid") == supervisor_pid
        ):
            return supervisor_pid
        if time.monotonic() >= deadline:
            raise RuntimeError("real supervisor worker did not register before timeout")
        time.sleep(0.05)


def launch_detached(
    *,
    queue_script: Path,
    expected_queue_sha256: str,
    jobs_path: Path,
    expected_jobs_sha256: str,
    status_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    checkpoint_revision: str,
    upstream_root: Path,
    expected_upstream_commit: str,
    python_path: Path,
    evidence_root: Path,
    detached_script: Path,
    target_model_id: str,
    probe: Callable[[], RuntimeSnapshot] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    popen: Callable[..., Any] | None = None,
    launch_id: str | None = None,
    worker_registration_timeout_seconds: float = 30.0,
) -> dict[str, object]:
    arguments: ContractArguments = {
        "queue_script": queue_script,
        "expected_queue_sha256": expected_queue_sha256,
        "jobs_path": jobs_path,
        "expected_jobs_sha256": expected_jobs_sha256,
        "status_path": status_path,
        "checkpoint_path": checkpoint_path,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_revision": checkpoint_revision,
        "upstream_root": upstream_root,
        "expected_upstream_commit": expected_upstream_commit,
        "python_path": python_path,
        "evidence_root": evidence_root,
        "detached_script": detached_script,
        "target_model_id": target_model_id,
    }
    preflight = preflight_detached(
        **arguments,
        probe=probe,
        runner=runner,
        git_runner=git_runner,
    )
    contract = cast("dict[str, object]", preflight["contract"])
    current = verify_contract(**arguments, git_runner=git_runner)
    if current != contract:
        raise RuntimeError("training contract changed after preflight")
    identity = cast("dict[str, str]", contract["output_identity"])
    launch_dir = reserve_launch_directory(Path(identity["launches_root"]), launch_id)
    _validate_output_identity(
        evidence_root=Path(identity["evidence_root"]),
        status_path=Path(identity["status_path"]),
        expected=identity,
        launch_dir=launch_dir,
    )
    token = uuid.uuid4().hex
    reservation_path = launch_dir / _RESERVATION_NAME
    handoff_path = launch_dir / _PARENT_HANDOFF_NAME
    supervisor_lock = launch_dir / _ACTIVE_LOCK_NAME
    queue_lock = Path(identity["queue_lock_path"])
    foreign_queue_snapshot = launch_dir / _FOREIGN_QUEUE_LOCK_SNAPSHOT_NAME
    reservation = {
        "schema_version": RESERVATION_SCHEMA,
        "launch_token": token,
        "status": "RESERVED",
        "created_at": _utc_now(),
        "launch_dir": str(launch_dir),
        "contract": contract,
        "detached_launcher": contract["detached_launcher"],
        "python": contract["python"],
        "output_identity": identity,
        "preflight": preflight,
    }
    _write_json_create_only(reservation_path, reservation)
    _create_lock(
        supervisor_lock,
        _lock_payload(schema=LOCK_SCHEMA, token=token, supervisor_pid=None),
    )
    try:
        _create_lock(
            queue_lock,
            _lock_payload(
                schema=QUEUE_LOCK_SCHEMA,
                token=token,
                supervisor_pid=None,
                status_path=identity["status_path"],
            ),
            conflict_snapshot=foreign_queue_snapshot,
        )
    except Exception as exc:
        archived_supervisor = _archive_lock(
            supervisor_lock,
            launch_dir / _FINAL_LOCK_NAME,
            token=token,
        )
        _write_json_create_only(
            launch_dir / _RESERVATION_FAILURE_NAME,
            {
                "schema_version": RESERVATION_FAILURE_SCHEMA,
                "launch_token": token,
                "status": "RESERVATION_FAILED",
                "failed_at": _utc_now(),
                "contract": contract,
                "detached_launcher": contract["detached_launcher"],
                "python": contract["python"],
                "output_identity": identity,
                "previous_evidence": _required_file_binding(reservation_path),
                "archived_supervisor_lock": archived_supervisor,
                "foreign_queue_lock": _file_binding(foreign_queue_snapshot),
                "foreign_queue_lock_path": str(queue_lock),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    detached = cast("Mapping[str, object]", contract["detached_launcher"])
    python = cast("Mapping[str, object]", contract["python"])
    upstream = cast("Mapping[str, object]", contract["upstream"])
    command = (
        cast("str", python["path"]),
        cast("str", detached["path"]),
        "supervise",
        "--launch-dir",
        str(launch_dir),
        "--token",
        token,
    )
    spawn = popen or subprocess.Popen
    spawn_pid: int | None = None
    supervisor_pid: int | None = None
    try:
        process = spawn(
            command,
            cwd=cast("str", upstream["path"]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=WINDOWS_DETACHED_FLAGS,
            shell=False,
        )
        spawn_pid = process.pid
        if not isinstance(spawn_pid, int) or isinstance(spawn_pid, bool) or spawn_pid <= 0:
            raise RuntimeError("detached supervisor returned an invalid PID")
        supervisor_pid = _wait_for_worker_registration(
            supervisor_lock,
            queue_lock,
            token=token,
            timeout_seconds=worker_registration_timeout_seconds,
        )
        _reverify_contract(contract, git_runner=git_runner, require_status_before=True)
        handoff = {
            "schema_version": PARENT_HANDOFF_SCHEMA,
            "launch_token": token,
            "status": "DETACHED",
            "created_at": _utc_now(),
            "spawn_pid": spawn_pid,
            "supervisor_pid": supervisor_pid,
            "supervisor_command": list(command),
            "contract": contract,
            "detached_launcher": contract["detached_launcher"],
            "python": contract["python"],
            "output_identity": identity,
            "previous_evidence": _required_file_binding(reservation_path),
        }
        _write_json_create_only(handoff_path, handoff)
    except Exception as exc:
        try:
            archived_supervisor = _archive_lock(
                supervisor_lock,
                launch_dir / _FINAL_LOCK_NAME,
                token=token,
            )
            archived_queue = _archive_lock(
                queue_lock,
                launch_dir / _FINAL_QUEUE_LOCK_NAME,
                token=token,
            )
            _write_json_create_only(
                launch_dir / _SPAWN_FAILURE_NAME,
                {
                    "schema_version": SPAWN_FAILURE_SCHEMA,
                    "launch_token": token,
                    "status": "SPAWN_FAILED",
                    "failed_at": _utc_now(),
                    "spawn_pid": spawn_pid,
                    "supervisor_pid": supervisor_pid,
                    "contract": contract,
                    "detached_launcher": contract["detached_launcher"],
                    "python": contract["python"],
                    "output_identity": identity,
                    "previous_evidence": _required_file_binding(reservation_path),
                    "archived_supervisor_lock": archived_supervisor,
                    "archived_queue_lock": archived_queue,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except Exception as lock_exc:
            _record_bootstrap_error(
                launch_dir=launch_dir,
                token=token,
                error=lock_exc,
            )
        raise
    return {
        "status": "DETACHED",
        "launch_dir": str(launch_dir),
        "reservation_path": str(reservation_path),
        "evidence_path": str(handoff_path),
        "spawn_pid": spawn_pid,
        "supervisor_pid": supervisor_pid,
    }


def _read_phase(
    path: Path,
    *,
    schema: str,
    status: str | None = None,
    token: str | None = None,
    previous: Path | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != schema:
        raise RuntimeError(f"immutable evidence chain has an invalid schema: {path}")
    if status is not None and payload.get("status") != status:
        raise RuntimeError(f"immutable evidence chain has an invalid status: {path}")
    phase_token = payload.get("launch_token")
    if not isinstance(phase_token, str) or not phase_token:
        raise RuntimeError(f"immutable evidence chain has an invalid token: {path}")
    if token is not None and phase_token != token:
        raise RuntimeError(f"immutable evidence chain token mismatch: {path}")
    if previous is not None and payload.get("previous_evidence") != _required_file_binding(
        previous
    ):
        raise RuntimeError(f"immutable evidence chain binding mismatch: {path}")
    return payload


def _require_phase_identity(
    payload: Mapping[str, object],
    reservation: Mapping[str, object],
    *,
    path: Path,
) -> None:
    for name in ("contract", "detached_launcher", "python", "output_identity"):
        if payload.get(name) != reservation.get(name):
            raise RuntimeError(f"immutable evidence chain {name} mismatch: {path}")


def _update_lock(
    path: Path,
    *,
    schema: str,
    token: str,
    supervisor_pid: int,
    status_path: str | None = None,
) -> None:
    with _lock_mutex(path):
        current, content, metadata = _read_lock_bytes(path)
        if not _is_owned_v002_lock(current, token=token, schema=schema):
            raise RuntimeError(f"lock ownership mismatch: {path}")
        temporary = path.with_name(f".{path.name}.{token}.tmp")
        try:
            with temporary.open("xb") as destination:
                destination.write(
                    _json_bytes(
                        _lock_payload(
                            schema=schema,
                            token=token,
                            supervisor_pid=supervisor_pid,
                            status_path=status_path,
                        )
                    )
                )
                destination.flush()
                os.fsync(destination.fileno())
            _assert_lock_unchanged(path, content, metadata)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _wait_for_parent_handoff(
    launch_dir: Path,
    *,
    token: str,
    supervisor_pid: int,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    handoff_path = launch_dir / _PARENT_HANDOFF_NAME
    while True:
        if handoff_path.is_file():
            handoff = _read_phase(
                handoff_path,
                schema=PARENT_HANDOFF_SCHEMA,
                status="DETACHED",
                token=token,
                previous=launch_dir / _RESERVATION_NAME,
            )
            if handoff.get("supervisor_pid") != supervisor_pid:
                raise RuntimeError("parent handoff supervisor PID mismatch")
            return handoff
        if time.monotonic() >= deadline:
            raise RuntimeError("detached supervisor parent handoff did not complete")
        time.sleep(0.05)


def _contract_arguments(contract: Mapping[str, object]) -> ContractArguments:
    queue = cast("Mapping[str, object]", contract["queue_script"])
    jobs = cast("Mapping[str, object]", contract["jobs"])
    checkpoint = cast("Mapping[str, object]", contract["checkpoint"])
    upstream = cast("Mapping[str, object]", contract["upstream"])
    python = cast("Mapping[str, object]", contract["python"])
    detached = cast("Mapping[str, object]", contract["detached_launcher"])
    identity = cast("Mapping[str, object]", contract["output_identity"])
    return {
        "queue_script": Path(cast("str", queue["path"])),
        "expected_queue_sha256": cast("str", queue["sha256"]),
        "jobs_path": Path(cast("str", jobs["path"])),
        "expected_jobs_sha256": cast("str", jobs["sha256"]),
        "status_path": Path(cast("str", identity["status_path"])),
        "checkpoint_path": Path(cast("str", checkpoint["path"])),
        "expected_checkpoint_sha256": cast("str", checkpoint["sha256"]),
        "checkpoint_revision": cast("str", contract["checkpoint_revision"]),
        "upstream_root": Path(cast("str", upstream["path"])),
        "expected_upstream_commit": cast("str", upstream["commit"]),
        "python_path": Path(cast("str", python["path"])),
        "evidence_root": Path(cast("str", identity["evidence_root"])),
        "detached_script": Path(cast("str", detached["path"])),
        "target_model_id": cast("str", contract["target_model_id"]),
    }


def _reverify_contract(
    contract: Mapping[str, object],
    *,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    require_status_before: bool,
) -> None:
    try:
        current = verify_contract(**_contract_arguments(contract), git_runner=git_runner)
    except Exception as exc:
        raise RuntimeError(f"training contract changed: {type(exc).__name__}: {exc}") from exc
    for name in (
        "queue_script",
        "jobs",
        "checkpoint",
        "checkpoint_revision",
        "upstream",
        "python",
        "detached_launcher",
        "output_identity",
        "target_model_id",
        "expected_skipped_model_ids",
        "job_inputs",
    ):
        if current.get(name) != contract.get(name):
            raise RuntimeError(f"training contract changed: {name}")
    if require_status_before and current.get("status_before") != contract.get("status_before"):
        raise RuntimeError("training mutable status changed before queue start")


def _validate_lock(
    path: Path,
    *,
    schema: str,
    token: str,
    supervisor_pid: object,
    status_path: str | None = None,
) -> None:
    payload = _read_json(path)
    identity_mismatch = (
        not _is_owned_v002_lock(payload, token=token, schema=schema)
        or payload.get("supervisor_pid") != supervisor_pid
    )
    status_mismatch = status_path is not None and payload.get("status_path") != status_path
    if identity_mismatch or status_mismatch:
        raise RuntimeError(f"immutable evidence chain lock mismatch: {path}")


def _require_archived_locks(
    payload: Mapping[str, object],
    *,
    launch_dir: Path,
    token: str,
    supervisor_pid: object,
) -> None:
    supervisor_path = launch_dir / _FINAL_LOCK_NAME
    queue_path = launch_dir / _FINAL_QUEUE_LOCK_NAME
    if payload.get("archived_supervisor_lock") != _required_file_binding(supervisor_path):
        raise RuntimeError("immutable evidence chain archived supervisor lock mismatch")
    if payload.get("archived_queue_lock") != _required_file_binding(queue_path):
        raise RuntimeError("immutable evidence chain archived queue lock mismatch")
    _validate_lock(
        supervisor_path,
        schema=LOCK_SCHEMA,
        token=token,
        supervisor_pid=supervisor_pid,
    )
    _validate_lock(
        queue_path,
        schema=QUEUE_LOCK_SCHEMA,
        token=token,
        supervisor_pid=supervisor_pid,
        status_path=cast(
            "str",
            cast("Mapping[str, object]", payload["output_identity"])["status_path"],
        ),
    )


def _require_foreign_lock_snapshot(
    foreign: object,
    *,
    expected_path: Path,
    own_token: str,
) -> None:
    if not isinstance(foreign, dict):
        raise RuntimeError("immutable failure evidence foreign lock snapshot is invalid")
    expected = str(_nominal_absolute(expected_path).resolve())
    foreign_sha = foreign.get("sha256")
    size = foreign.get("size")
    identity_invalid = foreign.get("path") != expected or not isinstance(foreign_sha, str)
    hash_invalid = isinstance(foreign_sha, str) and _SHA256_RE.fullmatch(foreign_sha) is None
    size_invalid = not isinstance(size, int) or isinstance(size, bool)
    size_negative = isinstance(size, int) and not isinstance(size, bool) and size < 0
    if identity_invalid or hash_invalid or size_invalid or size_negative:
        raise RuntimeError("immutable failure evidence foreign lock snapshot is invalid")
    if expected_path.is_file():
        if _file_binding(expected_path) != foreign:
            raise RuntimeError("foreign lock changed after failure evidence was recorded")
        if _is_owned_v002_lock(_read_json(expected_path), token=own_token):
            raise RuntimeError("failure evidence misclassified an owned lock as foreign")


def _require_frozen_foreign_lock_snapshot(
    foreign: object,
    *,
    snapshot_path: Path,
    live_path: Path,
    recorded_live_path: object,
    own_token: str,
) -> None:
    if foreign != _required_file_binding(snapshot_path):
        raise RuntimeError("immutable failure evidence foreign lock snapshot binding mismatch")
    if recorded_live_path != str(_nominal_absolute(live_path).resolve()):
        raise RuntimeError("immutable failure evidence foreign lock path mismatch")
    if _is_owned_v002_lock(_read_json(snapshot_path), token=own_token):
        raise RuntimeError("failure evidence misclassified an owned lock as foreign")


def _require_failure_locks(
    payload: Mapping[str, object],
    *,
    launch_dir: Path,
    token: str,
    supervisor_pid: object,
) -> None:
    identity = cast("Mapping[str, object]", payload["output_identity"])
    rows = (
        (
            "supervisor",
            launch_dir / _FINAL_LOCK_NAME,
            launch_dir / _ACTIVE_LOCK_NAME,
            LOCK_SCHEMA,
            None,
        ),
        (
            "queue",
            launch_dir / _FINAL_QUEUE_LOCK_NAME,
            Path(cast("str", identity["queue_lock_path"])),
            QUEUE_LOCK_SCHEMA,
            cast("str", identity["status_path"]),
        ),
    )
    for name, archived_path, active_path, schema, status_path in rows:
        archived = payload.get(f"archived_{name}_lock")
        foreign = payload.get(f"foreign_{name}_lock")
        if archived is None and foreign is None:
            raise RuntimeError(f"immutable failure evidence lost {name} lock disposition")
        if archived is not None:
            if archived != _required_file_binding(archived_path):
                raise RuntimeError(f"immutable failure evidence archived {name} lock mismatch")
            _validate_lock(
                archived_path,
                schema=schema,
                token=token,
                supervisor_pid=supervisor_pid,
                status_path=status_path,
            )
        if foreign is not None:
            _require_foreign_lock_snapshot(
                foreign,
                expected_path=active_path,
                own_token=token,
            )
        elif active_path.exists():
            raise RuntimeError(f"immutable failure evidence retained owned {name} lock")


def _validate_launch_chain(  # noqa: PLR0911 - each immutable terminal phase is explicit.
    launch_dir: Path,
) -> tuple[str, dict[str, Any], Path]:
    launch = _require_alias_free(launch_dir, label="training launch directory")
    reservation_path = launch / _RESERVATION_NAME
    reservation = _read_phase(
        reservation_path,
        schema=RESERVATION_SCHEMA,
        status="RESERVED",
    )
    token = cast("str", reservation["launch_token"])
    identity = reservation.get("output_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("reservation output identity is invalid")
    _validate_output_identity(
        evidence_root=Path(cast("str", identity["evidence_root"])),
        status_path=Path(cast("str", identity["status_path"])),
        expected=identity,
        launch_dir=launch,
    )
    active_supervisor = launch / _ACTIVE_LOCK_NAME
    active_queue = Path(cast("str", identity["queue_lock_path"]))
    handoff_path = launch / _PARENT_HANDOFF_NAME
    start_path = launch / _SUPERVISOR_START_NAME
    terminal_path = launch / _TERMINAL_NAME
    spawn_failure_path = launch / _SPAWN_FAILURE_NAME
    reservation_failure_path = launch / _RESERVATION_FAILURE_NAME
    bootstrap_path = launch / _BOOTSTRAP_FAILURE_NAME
    if reservation_failure_path.is_file():
        failure = _read_phase(
            reservation_failure_path,
            schema=RESERVATION_FAILURE_SCHEMA,
            status="RESERVATION_FAILED",
            token=token,
            previous=reservation_path,
        )
        _require_phase_identity(failure, reservation, path=reservation_failure_path)
        if any(path.exists() for path in (active_supervisor, handoff_path, start_path)):
            raise RuntimeError("immutable evidence chain has conflicting reservation artifacts")
        supervisor_final = launch / _FINAL_LOCK_NAME
        if failure.get("archived_supervisor_lock") != _required_file_binding(supervisor_final):
            raise RuntimeError("immutable evidence chain archived supervisor lock mismatch")
        _validate_lock(
            supervisor_final,
            schema=LOCK_SCHEMA,
            token=token,
            supervisor_pid=None,
        )
        foreign = failure.get("foreign_queue_lock")
        if foreign is not None:
            _require_frozen_foreign_lock_snapshot(
                foreign,
                snapshot_path=launch / _FOREIGN_QUEUE_LOCK_SNAPSHOT_NAME,
                live_path=active_queue,
                recorded_live_path=failure.get("foreign_queue_lock_path"),
                own_token=token,
            )
        elif (launch / _FOREIGN_QUEUE_LOCK_SNAPSHOT_NAME).exists():
            raise RuntimeError("reservation failure omitted its foreign lock snapshot")
        return "RESERVATION_FAILED", failure, reservation_failure_path
    if spawn_failure_path.is_file():
        failure = _read_phase(
            spawn_failure_path,
            schema=SPAWN_FAILURE_SCHEMA,
            status="SPAWN_FAILED",
            token=token,
            previous=reservation_path,
        )
        _require_phase_identity(failure, reservation, path=spawn_failure_path)
        if any(
            path.exists() for path in (active_supervisor, active_queue, handoff_path, start_path)
        ):
            raise RuntimeError("immutable evidence chain has conflicting spawn artifacts")
        _require_archived_locks(
            failure,
            launch_dir=launch,
            token=token,
            supervisor_pid=failure.get("supervisor_pid"),
        )
        return "SPAWN_FAILED", failure, spawn_failure_path
    if not handoff_path.is_file():
        if bootstrap_path.is_file():
            bootstrap = _read_phase(
                bootstrap_path,
                schema=BOOTSTRAP_FAILURE_SCHEMA,
                status="SUPERVISOR_ERROR",
                token=token,
                previous=reservation_path,
            )
            _require_phase_identity(bootstrap, reservation, path=bootstrap_path)
            _require_failure_locks(
                bootstrap,
                launch_dir=launch,
                token=token,
                supervisor_pid=bootstrap.get("supervisor_pid"),
            )
            return "SUPERVISOR_ERROR", bootstrap, bootstrap_path
        if not active_supervisor.is_file() or not active_queue.is_file():
            raise RuntimeError("immutable evidence chain is missing active locks")
        supervisor_payload = _read_json(active_supervisor)
        registered_pid = supervisor_payload.get("supervisor_pid")
        if registered_pid is not None and (
            not isinstance(registered_pid, int)
            or isinstance(registered_pid, bool)
            or registered_pid <= 0
        ):
            raise RuntimeError("immutable evidence chain lock mismatch: invalid supervisor PID")
        _validate_lock(
            active_supervisor,
            schema=LOCK_SCHEMA,
            token=token,
            supervisor_pid=registered_pid,
        )
        _validate_lock(
            active_queue,
            schema=QUEUE_LOCK_SCHEMA,
            token=token,
            supervisor_pid=registered_pid,
            status_path=cast("str", identity["status_path"]),
        )
        return "RESERVED", reservation, reservation_path
    handoff = _read_phase(
        handoff_path,
        schema=PARENT_HANDOFF_SCHEMA,
        status="DETACHED",
        token=token,
        previous=reservation_path,
    )
    _require_phase_identity(handoff, reservation, path=handoff_path)
    supervisor_pid = handoff.get("supervisor_pid")
    if (
        not isinstance(supervisor_pid, int)
        or isinstance(supervisor_pid, bool)
        or supervisor_pid <= 0
    ):
        raise RuntimeError("immutable evidence chain has an invalid supervisor PID")
    latest_status = "DETACHED"
    latest = handoff
    latest_path = handoff_path
    if start_path.is_file():
        start = _read_phase(
            start_path,
            schema=SUPERVISOR_START_SCHEMA,
            status="RUNNING",
            token=token,
            previous=handoff_path,
        )
        _require_phase_identity(start, reservation, path=start_path)
        if start.get("supervisor_pid") != supervisor_pid:
            raise RuntimeError("immutable evidence chain supervisor PID mismatch")
        latest_status, latest, latest_path = "RUNNING", start, start_path
    if terminal_path.is_file():
        if not start_path.is_file():
            raise RuntimeError("terminal evidence has no supervisor start evidence")
        terminal = _read_phase(
            terminal_path,
            schema=TERMINAL_SCHEMA,
            token=token,
            previous=start_path,
        )
        _require_phase_identity(terminal, reservation, path=terminal_path)
        if terminal.get("status") not in {"SUCCEEDED", "FAILED"}:
            raise RuntimeError("immutable evidence chain has an invalid terminal status")
        if any(path.exists() for path in (active_supervisor, active_queue)):
            raise RuntimeError("immutable evidence chain retained an active terminal lock")
        _require_archived_locks(
            terminal,
            launch_dir=launch,
            token=token,
            supervisor_pid=supervisor_pid,
        )
        return cast("str", terminal["status"]), terminal, terminal_path
    if bootstrap_path.is_file():
        bootstrap = _read_phase(
            bootstrap_path,
            schema=BOOTSTRAP_FAILURE_SCHEMA,
            status="SUPERVISOR_ERROR",
            token=token,
            previous=latest_path,
        )
        _require_phase_identity(bootstrap, reservation, path=bootstrap_path)
        _require_failure_locks(
            bootstrap,
            launch_dir=launch,
            token=token,
            supervisor_pid=bootstrap.get("supervisor_pid"),
        )
        return "SUPERVISOR_ERROR", bootstrap, bootstrap_path
    if not active_supervisor.is_file() or not active_queue.is_file():
        raise RuntimeError("immutable evidence chain is missing live locks")
    _validate_lock(
        active_supervisor,
        schema=LOCK_SCHEMA,
        token=token,
        supervisor_pid=supervisor_pid,
    )
    _validate_lock(
        active_queue,
        schema=QUEUE_LOCK_SCHEMA,
        token=token,
        supervisor_pid=supervisor_pid,
        status_path=cast("str", identity["status_path"]),
    )
    return latest_status, latest, latest_path


def _validate_target_status_delta(
    contract: Mapping[str, object],
) -> dict[str, object]:
    identity = cast("Mapping[str, object]", contract["output_identity"])
    status_path = Path(cast("str", identity["status_path"]))
    content = status_path.read_bytes()
    before = contract.get("status_before")
    offset = 0
    if before is not None:
        if not isinstance(before, dict):
            raise RuntimeError("training status-before binding is invalid")
        size = before.get("size")
        expected_sha = before.get("sha256")
        if not isinstance(size, int) or not isinstance(expected_sha, str) or len(content) < size:
            raise RuntimeError("training status was truncated during queue execution")
        if hashlib.sha256(content[:size]).hexdigest() != expected_sha:
            raise RuntimeError("training status prefix changed during queue execution")
        offset = size
    try:
        delta_lines = [line for line in content[offset:].decode("utf-8").splitlines() if line]
        rows = [json.loads(line) for line in delta_lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("training target status delta is invalid JSON") from exc
    if len(rows) != 2 or not all(isinstance(row, dict) for row in rows):  # noqa: PLR2004
        raise RuntimeError("training target status delta must contain exactly two rows")
    started, finished = cast("list[dict[str, object]]", rows)
    target = contract.get("target_model_id")
    started_expected = {"model_id": target, "event": "started", "status": "running"}
    finished_expected = {
        "model_id": target,
        "event": "finished",
        "status": "success",
        "exit_code": 0,
    }
    started_invalid = any(started.get(name) != value for name, value in started_expected.items())
    finished_invalid = any(finished.get(name) != value for name, value in finished_expected.items())
    ended_at = finished.get("ended_at")
    if started_invalid or finished_invalid or not isinstance(ended_at, str) or not ended_at:
        raise RuntimeError(
            "training target status delta does not prove started and finished success"
        )
    raw_inputs = cast("list[dict[str, object]]", contract["job_inputs"])
    target_inputs = next(row for row in raw_inputs if row.get("model_id") == target)
    provenance = {
        "clean_manifest_sha256": cast("Mapping[str, object]", target_inputs["clean_manifest"])[
            "sha256"
        ],
        "checkpoint_sha256": cast("Mapping[str, object]", contract["checkpoint"])["sha256"],
        "checkpoint_revision": contract["checkpoint_revision"],
        "config_sha256": cast("Mapping[str, object]", target_inputs["config"])["sha256"],
        "upstream_commit": cast("Mapping[str, object]", contract["upstream"])["commit"],
    }
    for row in (started, finished):
        if any(row.get(name) != value for name, value in provenance.items()):
            raise RuntimeError("training target status delta provenance mismatch")
    if started.get("started_at") != finished.get("started_at"):
        raise RuntimeError("training target status delta start timestamp mismatch")
    candidates = finished.get("candidate_checkpoints")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("training target status delta has no candidate checkpoints")
    output = Path(cast("str", target_inputs["output_dir"]))
    candidate_bindings: list[dict[str, object]] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict) or not isinstance(raw_candidate.get("path"), str):
            raise RuntimeError("training target status delta has an invalid checkpoint")
        candidate_path = _require_alias_free(
            Path(cast("str", raw_candidate["path"])), label="target candidate checkpoint"
        )
        try:
            candidate_path.relative_to(output)
        except ValueError as exc:
            raise RuntimeError(
                "training target checkpoint escapes target output directory"
            ) from exc
        binding = _required_file_binding(candidate_path)
        if raw_candidate.get("sha256") != binding["sha256"]:
            raise RuntimeError("training target checkpoint SHA-256 mismatch")
        candidate_bindings.append(binding)
    last = candidate_bindings[-1]
    if (
        finished.get("last_checkpoint") != last["path"]
        or finished.get("last_checkpoint_sha256") != last["sha256"]
    ):
        raise RuntimeError("training target final checkpoint binding mismatch")
    return {
        "model_id": target,
        "started": started,
        "finished": finished,
        "candidate_checkpoints": candidate_bindings,
        "status_delta_sha256": hashlib.sha256(content[offset:]).hexdigest(),
        "status_delta_size": len(content) - offset,
    }


def run_supervisor(
    *,
    launch_dir: Path,
    token: str,
    probe: Callable[[], RuntimeSnapshot] | None = None,
    popen: Callable[..., Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    current_pid: int | None = None,
) -> int:
    launch = _require_alias_free(launch_dir, label="training launch directory")
    reservation_path = launch / _RESERVATION_NAME
    handoff_path = launch / _PARENT_HANDOFF_NAME
    start_path = launch / _SUPERVISOR_START_NAME
    terminal_path = launch / _TERMINAL_NAME
    supervisor_lock = launch / _ACTIVE_LOCK_NAME
    log_path = launch / _QUEUE_LOG_NAME
    reservation = _read_phase(
        reservation_path,
        schema=RESERVATION_SCHEMA,
        status="RESERVED",
        token=token,
    )
    contract = reservation.get("contract")
    identity = reservation.get("output_identity")
    if not isinstance(contract, dict) or not isinstance(identity, dict):
        raise RuntimeError("reservation training contract is invalid")
    _validate_output_identity(
        evidence_root=Path(cast("str", identity["evidence_root"])),
        status_path=Path(cast("str", identity["status_path"])),
        expected=identity,
        launch_dir=launch,
    )
    queue_lock = Path(cast("str", identity["queue_lock_path"]))
    supervisor_pid = current_pid if current_pid is not None else os.getpid()
    if (
        not isinstance(supervisor_pid, int)
        or isinstance(supervisor_pid, bool)
        or supervisor_pid <= 0
    ):
        raise ValueError("detached supervisor requires a positive current PID")
    _update_lock(
        supervisor_lock,
        schema=LOCK_SCHEMA,
        token=token,
        supervisor_pid=supervisor_pid,
    )
    _update_lock(
        queue_lock,
        schema=QUEUE_LOCK_SCHEMA,
        token=token,
        supervisor_pid=supervisor_pid,
        status_path=cast("str", identity["status_path"]),
    )
    handoff = _wait_for_parent_handoff(
        launch,
        token=token,
        supervisor_pid=supervisor_pid,
    )
    _require_phase_identity(handoff, reservation, path=handoff_path)
    _reverify_contract(contract, git_runner=git_runner, require_status_before=True)
    runtime_before = (probe or probe_runtime)()
    _require_safe_runtime(
        runtime_before,
        evidence_root=Path(cast("str", identity["evidence_root"])),
        status_path=Path(cast("str", identity["status_path"])),
        allowed_supervisor_lock=supervisor_lock,
        allowed_queue_lock=queue_lock,
        contract=contract,
    )
    queue_preflight = _run_queue_dry_run(contract, runner=runner)
    _reverify_contract(contract, git_runner=git_runner, require_status_before=True)
    start = {
        "schema_version": SUPERVISOR_START_SCHEMA,
        "launch_token": token,
        "status": "RUNNING",
        "started_at": _utc_now(),
        "spawn_pid": handoff["spawn_pid"],
        "supervisor_pid": supervisor_pid,
        "contract": contract,
        "detached_launcher": reservation["detached_launcher"],
        "python": reservation["python"],
        "output_identity": identity,
        "runtime_before_queue": runtime_before.as_dict(),
        "queue_dry_run": queue_preflight,
        "previous_evidence": _required_file_binding(handoff_path),
    }
    _write_json_create_only(start_path, start)
    command = _queue_command(contract, dry_run=False)
    queue_exit_code: int | None = None
    error: str | None = None
    try:
        spawn = popen or subprocess.Popen
        with log_path.open("xb") as output:
            child_env = os.environ.copy()
            child_env["PYTHONDONTWRITEBYTECODE"] = "1"
            process = spawn(
                command,
                cwd=cast("str", cast("Mapping[str, object]", contract["upstream"])["path"]),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                close_fds=True,
                shell=False,
            )
            queue_exit_code = process.wait()
            output.flush()
            os.fsync(output.fileno())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        _reverify_contract(contract, git_runner=git_runner, require_status_before=False)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    runtime_after = (probe or probe_runtime)()
    try:
        _require_safe_runtime(
            runtime_after,
            evidence_root=Path(cast("str", identity["evidence_root"])),
            status_path=Path(cast("str", identity["status_path"])),
            allowed_supervisor_lock=supervisor_lock,
            allowed_queue_lock=queue_lock,
            contract=contract,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    log_binding = _file_binding(log_path)
    status_binding = _file_binding(Path(cast("str", identity["status_path"])))
    summary: dict[str, Any] | None = None
    target_status_delta: dict[str, object] | None = None
    if log_binding is not None:
        try:
            summary = _last_json_object(log_path.read_text(encoding="utf-8"), source="queue log")
        except (UnicodeDecodeError, ValueError):
            summary = None
    if summary is not None:
        try:
            _validate_queue_report(summary, contract, completed=True)
            target_status_delta = _validate_target_status_delta(contract)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    archived_supervisor = _archive_lock(
        supervisor_lock,
        launch / _FINAL_LOCK_NAME,
        token=token,
    )
    archived_queue = _archive_lock(
        queue_lock,
        launch / _FINAL_QUEUE_LOCK_NAME,
        token=token,
    )
    summary_failed = summary is None or target_status_delta is None
    failed = any(
        (
            error is not None,
            queue_exit_code != 0,
            summary_failed,
            status_binding is None,
            bool(runtime_after.errors),
            bool(runtime_after.training_processes),
            bool(runtime_after.service_processes),
        )
    )
    terminal_status = "FAILED" if failed else "SUCCEEDED"
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "launch_token": token,
        "status": terminal_status,
        "finished_at": _utc_now(),
        "spawn_pid": handoff["spawn_pid"],
        "supervisor_pid": supervisor_pid,
        "contract": contract,
        "detached_launcher": reservation["detached_launcher"],
        "python": reservation["python"],
        "output_identity": identity,
        "previous_evidence": _required_file_binding(start_path),
        "archived_supervisor_lock": archived_supervisor,
        "archived_queue_lock": archived_queue,
        "queue_command": list(command),
        "queue_exit_code": queue_exit_code,
        "queue_summary": summary,
        "queue_log": log_binding,
        "queue_status": status_binding,
        "target_status_delta": target_status_delta,
        "runtime_after": runtime_after.as_dict(),
        "error": error,
    }
    _write_json_create_only(terminal_path, terminal)
    return 0 if terminal_status == "SUCCEEDED" else 1


def read_status(evidence_root: Path) -> dict[str, object]:
    root = _require_alias_free(evidence_root, label="training evidence root")
    if _EVIDENCE_ROOT_RE.fullmatch(root.name) is None:
        raise ValueError(f"expected a versioned evidence root: {root}")
    launches = root / "launches"
    if not launches.is_dir():
        return {"status": "NOT_LAUNCHED", "launches_root": str(launches)}
    candidates = [
        (int(match.group(1)), child)
        for child in launches.iterdir()
        if child.is_dir() and (match := _LAUNCH_ID_RE.fullmatch(child.name)) is not None
    ]
    if not candidates:
        return {"status": "NOT_LAUNCHED", "launches_root": str(launches)}
    _, latest = max(candidates)
    status, evidence, evidence_path = _validate_launch_chain(latest)
    return {
        "status": status,
        "launch_dir": str(latest.resolve()),
        "evidence_path": str(evidence_path.resolve()),
        "evidence": evidence,
        "supervisor_lock_active": (latest / _ACTIVE_LOCK_NAME).is_file(),
    }


def _record_bootstrap_error(
    *,
    launch_dir: Path,
    token: str,
    error: BaseException,
) -> None:
    launch = _require_alias_free(launch_dir, label="training launch directory")
    reservation_path = launch / _RESERVATION_NAME
    reservation = _read_phase(
        reservation_path,
        schema=RESERVATION_SCHEMA,
        status="RESERVED",
        token=token,
    )
    spawn_failure_path = launch / _SPAWN_FAILURE_NAME
    if spawn_failure_path.is_file():
        _read_phase(
            spawn_failure_path,
            schema=SPAWN_FAILURE_SCHEMA,
            status="SPAWN_FAILED",
            token=token,
            previous=reservation_path,
        )
        return
    identity = reservation.get("output_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("reservation output identity is invalid")
    handoff_path = launch / _PARENT_HANDOFF_NAME
    start_path = launch / _SUPERVISOR_START_NAME
    previous_path = (
        start_path
        if start_path.is_file()
        else handoff_path
        if handoff_path.is_file()
        else reservation_path
    )
    previous = _read_json(previous_path)
    if previous.get("launch_token") != token:
        raise RuntimeError("bootstrap previous evidence token mismatch")
    supervisor_lock = launch / _ACTIVE_LOCK_NAME
    queue_lock = Path(cast("str", identity["queue_lock_path"]))
    final_supervisor_lock = launch / _FINAL_LOCK_NAME
    final_queue_lock = launch / _FINAL_QUEUE_LOCK_NAME
    archived_supervisor, foreign_supervisor = _archive_owned_or_snapshot_foreign(
        supervisor_lock,
        final_supervisor_lock,
        token=token,
    )
    archived_queue, foreign_queue = _archive_owned_or_snapshot_foreign(
        queue_lock,
        final_queue_lock,
        token=token,
    )
    supervisor_pid = previous.get("supervisor_pid")
    if supervisor_pid is None and archived_supervisor is not None:
        supervisor_pid = _read_json(final_supervisor_lock).get("supervisor_pid")
    payload = {
        "schema_version": BOOTSTRAP_FAILURE_SCHEMA,
        "launch_token": token,
        "status": "SUPERVISOR_ERROR",
        "failed_at": _utc_now(),
        "spawn_pid": previous.get("spawn_pid"),
        "supervisor_pid": supervisor_pid,
        "contract": reservation["contract"],
        "detached_launcher": reservation["detached_launcher"],
        "python": reservation["python"],
        "output_identity": identity,
        "previous_evidence": _required_file_binding(previous_path),
        "archived_supervisor_lock": archived_supervisor,
        "archived_queue_lock": archived_queue,
        "foreign_supervisor_lock": foreign_supervisor,
        "foreign_queue_lock": foreign_queue,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    _write_json_create_only(launch / _BOOTSTRAP_FAILURE_NAME, payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "launch", "status", "supervise"))
    parser.add_argument("--queue-script", type=Path)
    parser.add_argument("--queue-sha256", dest="expected_queue_sha256")
    parser.add_argument("--jobs-json", dest="jobs_path", type=Path)
    parser.add_argument("--jobs-sha256", dest="expected_jobs_sha256")
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path)
    parser.add_argument("--checkpoint-sha256", dest="expected_checkpoint_sha256")
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--upstream-commit", dest="expected_upstream_commit")
    parser.add_argument("--python", dest="python_path", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--target-model-id")
    parser.add_argument("--launch-id")
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--token")
    return parser.parse_args(argv)


def _operational_arguments(args: argparse.Namespace) -> ContractArguments:
    fields = (
        ("queue_script", "--queue-script"),
        ("expected_queue_sha256", "--queue-sha256"),
        ("jobs_path", "--jobs-json"),
        ("expected_jobs_sha256", "--jobs-sha256"),
        ("status_path", "--status-path"),
        ("checkpoint_path", "--checkpoint"),
        ("expected_checkpoint_sha256", "--checkpoint-sha256"),
        ("checkpoint_revision", "--checkpoint-revision"),
        ("upstream_root", "--upstream-root"),
        ("expected_upstream_commit", "--upstream-commit"),
        ("python_path", "--python"),
        ("evidence_root", "--evidence-root"),
        ("target_model_id", "--target-model-id"),
    )
    values: dict[str, object] = {}
    for name, option in fields:
        value = getattr(args, name)
        if value is None:
            raise ValueError(f"{args.mode} requires {option}")
        values[name] = value
    return ContractArguments(
        queue_script=cast("Path", values["queue_script"]),
        expected_queue_sha256=cast("str", values["expected_queue_sha256"]),
        jobs_path=cast("Path", values["jobs_path"]),
        expected_jobs_sha256=cast("str", values["expected_jobs_sha256"]),
        status_path=cast("Path", values["status_path"]),
        checkpoint_path=cast("Path", values["checkpoint_path"]),
        expected_checkpoint_sha256=cast("str", values["expected_checkpoint_sha256"]),
        checkpoint_revision=cast("str", values["checkpoint_revision"]),
        upstream_root=cast("Path", values["upstream_root"]),
        expected_upstream_commit=cast("str", values["expected_upstream_commit"]),
        python_path=cast("Path", values["python_path"]),
        evidence_root=cast("Path", values["evidence_root"]),
        detached_script=Path(__file__).resolve(),
        target_model_id=cast("str", values["target_model_id"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "status":
        if args.evidence_root is None:
            raise ValueError("status requires --evidence-root")
        result = read_status(args.evidence_root)
    elif args.mode == "supervise":
        if args.launch_dir is None or args.token is None:
            raise ValueError("supervise requires --launch-dir and --token")
        try:
            return run_supervisor(launch_dir=args.launch_dir, token=args.token)
        except Exception as exc:
            try:
                _record_bootstrap_error(
                    launch_dir=args.launch_dir,
                    token=args.token,
                    error=exc,
                )
            except Exception as evidence_exc:
                print(
                    json.dumps(
                        {
                            "status": "SUPERVISOR_ERROR",
                            "error": f"{type(exc).__name__}: {exc}",
                            "evidence_error": f"{type(evidence_exc).__name__}: {evidence_exc}",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            return 1
    else:
        operational = _operational_arguments(args)
        if args.mode == "preflight":
            result = preflight_detached(**operational)
        else:
            result = launch_detached(**operational, launch_id=args.launch_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
