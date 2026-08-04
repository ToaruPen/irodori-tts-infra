# ruff: noqa: BLE001, EM101, EM102, PLR0913, PLR0914, PLR0915, S404, TRY003, TRY301
# Operational failures retain exact artifact context; subprocesses use fixed argv and no shell.

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_UTC = timezone.utc  # noqa: UP017 - pinned Windows runtime uses Python 3.10.


REMOTE_ROOT = Path(r"C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731")
UPSTREAM_ROOT = Path(r"C:\Users\takut\Dev\Irodori-TTS")
PINNED_PYTHON = UPSTREAM_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_OUTPUT_ROOT = REMOTE_ROOT / "evaluation_speed_v7"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v7.jsonl"
DEFAULT_LAUNCHES_ROOT = DEFAULT_OUTPUT_ROOT / "launches"
PENDING_V5_LAUNCHER_BUNDLE_NAME = "PENDING_REMOTE_SPEED_V7_LAUNCHER_BUNDLE_NAME"
V5_LAUNCHER_BUNDLE_NAME = "evaluation_speed_v7_v1"
V5_LAUNCHER_PATH = (
    REMOTE_ROOT
    / "scripts"
    / V5_LAUNCHER_BUNDLE_NAME
    / "launch_600m_speaker_evaluation_queue_speed_v5.py"
)
PENDING_DETACHED_BUNDLE_NAME = "PENDING_REMOTE_SPEED_V7_DETACHED_BUNDLE_NAME"
DETACHED_BUNDLE_NAME = "evaluation_speed_v7_detached_v1"
DETACHED_SCRIPT_PATH = (
    REMOTE_ROOT
    / "scripts"
    / DETACHED_BUNDLE_NAME
    / "launch_600m_speaker_evaluation_queue_speed_v7_detached.py"
)

PENDING_V5_LAUNCHER_SHA256 = "PENDING_REMOTE_SPEED_V7_LAUNCHER_SHA256"
EXPECTED_V5_LAUNCHER_SHA256 = PENDING_V5_LAUNCHER_SHA256
RESERVATION_SCHEMA = "speaker-evaluation-speed-v7-detached-reservation/v1"
PARENT_HANDOFF_SCHEMA = "speaker-evaluation-speed-v7-detached-parent-handoff/v1"
SUPERVISOR_START_SCHEMA = "speaker-evaluation-speed-v7-detached-supervisor-start/v1"
TERMINAL_SCHEMA = "speaker-evaluation-speed-v7-detached-terminal/v1"
SPAWN_FAILURE_SCHEMA = "speaker-evaluation-speed-v7-detached-spawn-failure/v1"
BOOTSTRAP_FAILURE_SCHEMA = "speaker-evaluation-speed-v7-detached-bootstrap-failure/v1"
BOOTSTRAP_PRE_HANDOFF_FAILURE_SCHEMA = (
    "speaker-evaluation-speed-v7-detached-bootstrap-pre-handoff-failure/v1"
)
LOCK_SCHEMA = "speaker-evaluation-speed-v7-detached-lock/v1"
OUTPUT_IDENTITY_SCHEMA = "speaker-evaluation-speed-v7-detached-output-identity/v1"
MIN_FREE_GPU_MIB = 10_500.0
WINDOWS_DETACHED_FLAGS = 0x00000008 | 0x00000200 | 0x01000000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LAUNCH_ID_RE = re.compile(r"launch-v([0-9]{3,})\Z")
_V5_BUNDLE_RE = re.compile(r"evaluation_speed_v7_v[0-9]+\Z")
_DETACHED_BUNDLE_RE = re.compile(r"evaluation_speed_v7_detached_v[0-9]+\Z")
_EVALUATION_LAUNCHER_RE = re.compile(
    r"(?:^|[\\/\s\"'])"
    r"launch_600m_speaker_evaluation_queue_speed_v(?:2|3|4|5|6|7)"
    r"(?:_detached)?\.py(?:$|[\s\"'])",
    re.IGNORECASE,
)
_REMOTE_SERVER_RE = re.compile(
    r"(?:^|[\\/\s\"'])remote_server\.py(?:$|[\s\"'])",
    re.IGNORECASE,
)
_EVALUATION_MARKERS = (
    "run_600m_speaker_evaluation_queue.py",
    "build_600m_checkpoint_evaluation_manifests.py",
    "generate_600m_checkpoint_audio_remote.py",
    "analyze_nko_beep_matrix.py",
    "compute_600m_speaker_metrics.py",
    "evaluate_600m_speaker_checkpoints.py",
)
_SERVICE_MARKERS = (
    "uvicorn",
    "irodori_tts_infra.server",
    "irodori-tts-server",
)

_RESERVATION_NAME = "reservation-evidence.json"
_PARENT_HANDOFF_NAME = "parent-handoff-evidence.json"
_SUPERVISOR_START_NAME = "supervisor-start-evidence.json"
_TERMINAL_NAME = "terminal-final-evidence.json"
_SPAWN_FAILURE_NAME = "spawn-failure-evidence.json"
_BOOTSTRAP_FAILURE_NAME = "supervisor-bootstrap-error.json"
_ACTIVE_LOCK_NAME = "supervisor.lock"
_FINAL_LOCK_NAME = "supervisor-lock-final.json"


def _is_positive_pid(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class RuntimeSnapshot:
    __slots__ = (
        "errors",
        "evaluation_processes",
        "gpu_free_mib",
        "gpu_total_mib",
        "gpu_used_mib",
        "observed_at",
        "processes",
        "service_processes",
    )

    def __init__(
        self,
        *,
        observed_at: str,
        processes: Sequence[Mapping[str, object]],
        evaluation_processes: Sequence[Mapping[str, object]],
        service_processes: Sequence[Mapping[str, object]],
        gpu_total_mib: float | None,
        gpu_used_mib: float | None,
        gpu_free_mib: float | None,
        errors: Sequence[str],
    ) -> None:
        self.observed_at = observed_at
        self.processes = tuple(dict(row) for row in processes)
        self.evaluation_processes = tuple(dict(row) for row in evaluation_processes)
        self.service_processes = tuple(dict(row) for row in service_processes)
        self.gpu_total_mib = gpu_total_mib
        self.gpu_used_mib = gpu_used_mib
        self.gpu_free_mib = gpu_free_mib
        self.errors = tuple(errors)

    def as_dict(self) -> dict[str, object]:
        return {
            "observed_at": self.observed_at,
            "processes": self.processes,
            "evaluation_processes": self.evaluation_processes,
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


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _write_json_create_only(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes_create_only(path, _json_bytes(payload))


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _is_filesystem_alias(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)


def _nominal_absolute(path: Path) -> Path:
    # abspath normalizes dot segments without following filesystem aliases.
    return Path(os.path.abspath(path))  # noqa: PTH100 - resolve() would follow aliases.


def _require_alias_free_ancestors(path: Path, *, label: str) -> None:
    nominal = _nominal_absolute(path)
    for candidate in (nominal, *nominal.parents):
        if candidate == candidate.parent:
            continue
        if _is_filesystem_alias(candidate):
            raise ValueError(f"{label} has a symlink, junction, or reparse alias: {candidate}")


def _lexical_absolute(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _require_alias_free_lexical_components(path: Path, *, label: str) -> None:
    lexical = _lexical_absolute(path)
    current = Path(lexical.anchor)
    parts = lexical.parts[1:] if lexical.anchor else lexical.parts
    for part in parts:
        current /= part
        if current != current.parent and _is_filesystem_alias(current):
            raise ValueError(f"{label} has a symlink, junction, or reparse alias: {current}")


def _require_alias_free_output_path(path: Path, *, label: str) -> Path:
    _require_alias_free_lexical_components(path, label=label)
    nominal = _nominal_absolute(path)
    _require_alias_free_ancestors(nominal, label=label)
    return nominal


def _launch_artifact_paths(launch_dir: Path) -> tuple[tuple[Path, str], ...]:
    return (
        (launch_dir / _RESERVATION_NAME, "reservation evidence"),
        (launch_dir / _PARENT_HANDOFF_NAME, "parent handoff evidence"),
        (launch_dir / _SUPERVISOR_START_NAME, "supervisor start evidence"),
        (launch_dir / _TERMINAL_NAME, "terminal evidence"),
        (launch_dir / _SPAWN_FAILURE_NAME, "spawn failure evidence"),
        (launch_dir / _BOOTSTRAP_FAILURE_NAME, "bootstrap failure evidence"),
        (launch_dir / _ACTIVE_LOCK_NAME, "active supervisor lock"),
        (launch_dir / _FINAL_LOCK_NAME, "archived supervisor lock"),
        (launch_dir / "queue.log", "queue log"),
    )


def _require_alias_free_launch_artifacts(launch_dir: Path) -> Path:
    nominal_launch = _require_alias_free_output_path(
        launch_dir,
        label="evaluation launch directory",
    )
    for artifact_path, label in _launch_artifact_paths(launch_dir):
        nominal_artifact = _require_alias_free_output_path(artifact_path, label=label)
        if nominal_artifact.parent != nominal_launch:
            raise ValueError(f"{label} is outside fixed launch directory: {artifact_path}")
    return nominal_launch


def _validate_output_paths(
    *,
    output_root: Path,
    status_path: Path,
    expected_identity: object | None = None,
    launch_dir: Path | None = None,
) -> dict[str, str]:
    nominal_root = _require_alias_free_output_path(
        output_root,
        label="evaluation output root",
    )
    nominal_status = _require_alias_free_output_path(
        status_path,
        label="evaluation status path",
    )
    nominal_queue_lock = _require_alias_free_output_path(
        _queue_lock_path(status_path),
        label="evaluation queue lock",
    )
    nominal_launches = _require_alias_free_output_path(
        output_root / "launches",
        label="evaluation launches root",
    )
    if not nominal_status.is_relative_to(nominal_root):
        raise ValueError(f"evaluation status path is outside output root: {nominal_status}")
    if not nominal_queue_lock.is_relative_to(nominal_root):
        raise ValueError(f"evaluation queue lock is outside output root: {nominal_queue_lock}")
    if nominal_launches.parent != nominal_root:
        raise ValueError(f"evaluation launches root identity mismatch: {nominal_launches}")
    identity = {
        "schema_version": OUTPUT_IDENTITY_SCHEMA,
        "output_root": str(nominal_root),
        "status_path": str(nominal_status),
        "queue_lock_path": str(nominal_queue_lock),
        "launches_root": str(nominal_launches),
    }
    if expected_identity is not None and expected_identity != identity:
        raise RuntimeError(
            f"fixed output identity mismatch: expected={expected_identity}, actual={identity}"
        )
    if launch_dir is not None:
        nominal_launch = _require_alias_free_launch_artifacts(launch_dir)
        if (
            nominal_launch.parent != nominal_launches
            or _LAUNCH_ID_RE.fullmatch(nominal_launch.name) is None
        ):
            raise ValueError(f"launch directory is outside fixed launches root: {launch_dir}")
    return identity


def _validate_reservation_output_identity(
    reservation: Mapping[str, object],
    *,
    launch_dir: Path,
) -> dict[str, str]:
    expected_identity = reservation.get("output_identity")
    if not isinstance(expected_identity, dict):
        raise TypeError("reservation has an invalid fixed output identity")
    output_root = expected_identity.get("output_root")
    status_path = expected_identity.get("status_path")
    if not isinstance(output_root, str) or not isinstance(status_path, str):
        raise TypeError("reservation has invalid fixed output paths")
    identity = _validate_output_paths(
        output_root=Path(output_root),
        status_path=Path(status_path),
        expected_identity=expected_identity,
        launch_dir=launch_dir,
    )
    nominal_launch = _nominal_absolute(launch_dir)
    if (
        reservation.get("launch_dir") != str(nominal_launch)
        or reservation.get("output_root") != identity["output_root"]
        or reservation.get("status_path") != identity["status_path"]
    ):
        raise RuntimeError("reservation output path binding mismatch")
    return identity


def _verify_pinned_file(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, str]:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError(f"{label} SHA-256 pin is not finalized: {expected_sha256}")
    nominal = _nominal_absolute(path)
    _require_alias_free_ancestors(nominal, label=label)
    resolved = nominal.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is unsafe or missing: {path}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual}, path={resolved}"
        )
    return {"path": str(resolved), "sha256": actual}


def _verify_launcher(path: Path, expected_sha256: str) -> dict[str, str]:
    return _verify_pinned_file(path, expected_sha256, label="v5 launcher")


def _verify_detached_script(path: Path) -> dict[str, str]:
    nominal = _nominal_absolute(path)
    _require_alias_free_ancestors(nominal, label="speed-v7 detached launcher")
    resolved = nominal.resolve()
    if not resolved.is_file():
        raise ValueError(f"speed-v7 detached launcher is unsafe or missing: {path}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verify_python_executable(path: Path) -> dict[str, str]:
    nominal = _nominal_absolute(path)
    _require_alias_free_ancestors(nominal, label="Python executable")
    resolved = nominal.resolve()
    if not resolved.is_file():
        raise ValueError(f"Python executable is unsafe or missing: {path}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _assert_resolved_bundle_names(
    *,
    v5_launcher_bundle_name: str,
    detached_bundle_name: str,
) -> None:
    if v5_launcher_bundle_name == PENDING_V5_LAUNCHER_BUNDLE_NAME:
        raise ValueError(PENDING_V5_LAUNCHER_BUNDLE_NAME)
    if detached_bundle_name == PENDING_DETACHED_BUNDLE_NAME:
        raise ValueError(PENDING_DETACHED_BUNDLE_NAME)
    if _V5_BUNDLE_RE.fullmatch(v5_launcher_bundle_name) is None:
        raise ValueError(
            f"expected a versioned speed-v7 launcher bundle: {v5_launcher_bundle_name}"
        )
    if _DETACHED_BUNDLE_RE.fullmatch(detached_bundle_name) is None:
        raise ValueError(f"expected a versioned speed-v7 detached bundle: {detached_bundle_name}")


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


def _is_evaluation_command(command_line: str) -> bool:
    folded = command_line.casefold()
    return _EVALUATION_LAUNCHER_RE.search(command_line) is not None or any(
        marker in folded for marker in _EVALUATION_MARKERS
    )


def _is_service_command(command_line: str) -> bool:
    folded = command_line.casefold()
    return _REMOTE_SERVER_RE.search(command_line) is not None or any(
        marker in folded for marker in _SERVICE_MARKERS
    )


def _ancestor_pids(
    rows: Sequence[Mapping[str, object]],
    *,
    current_pid: int,
) -> set[int]:
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
    raw_rows = payload if isinstance(payload, list) else [payload]
    if not raw_rows or not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError("process inventory must be a nonempty object list")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row = dict(cast("Mapping[str, object]", raw))
        value = row.get("ProcessId", row.get("pid"))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("process inventory rows require nonnegative ProcessId")
        if value > 0:
            rows.append(row)
    if not rows:
        raise ValueError("process inventory requires at least one positive ProcessId")
    return rows


def _parse_gpu_memory(stdout: str) -> tuple[float, float]:
    first = stdout.strip().splitlines()[0]
    total_raw, used_raw = (part.strip() for part in first.split(",", maxsplit=1))
    total = float(total_raw)
    used = float(used_raw)
    if not all(math.isfinite(value) and value >= 0 for value in (total, used)):
        raise ValueError("GPU memory values must be finite and nonnegative")
    if used > total:
        raise ValueError("GPU used memory exceeds total memory")
    return total, used


def probe_runtime(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    current_pid: int | None = None,
) -> RuntimeSnapshot:
    execute = runner or subprocess.run
    own_pid = current_pid if current_pid is not None else os.getpid()
    errors: list[str] = []
    processes: list[dict[str, object]] = []
    gpu_total: float | None = None
    gpu_used: float | None = None
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
            gpu_total, gpu_used = _parse_gpu_memory(completed.stdout)
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    excluded = _ancestor_pids(processes, current_pid=own_pid)
    visible = tuple(row for row in processes if _pid(row) not in excluded)
    evaluation = tuple(row for row in visible if _is_evaluation_command(_command_line(row)))
    services = tuple(row for row in visible if _is_service_command(_command_line(row)))
    return RuntimeSnapshot(
        observed_at=_utc_now(),
        processes=visible,
        evaluation_processes=evaluation,
        service_processes=services,
        gpu_total_mib=gpu_total,
        gpu_used_mib=gpu_used,
        gpu_free_mib=(
            gpu_total - gpu_used if gpu_total is not None and gpu_used is not None else None
        ),
        errors=errors,
    )


def _queue_lock_path(status_path: Path) -> Path:
    return status_path.with_suffix(status_path.suffix + ".lock")


def _existing_supervisor_locks(output_root: Path) -> tuple[Path, ...]:
    launches = output_root / "launches"
    if not launches.exists():
        return ()
    return tuple(sorted(launches.glob("launch-v*/supervisor.lock")))


def _require_safe_runtime(
    snapshot: RuntimeSnapshot,
    *,
    output_root: Path,
    status_path: Path,
    allowed_supervisor_lock: Path | None = None,
) -> None:
    if snapshot.errors:
        raise RuntimeError(f"runtime probe failed closed: {snapshot.errors}")
    queue_lock = _queue_lock_path(status_path)
    if os.path.lexists(queue_lock):
        raise RuntimeError(f"evaluation queue lock already exists: {queue_lock}")
    allowed_lock = (
        _nominal_absolute(allowed_supervisor_lock) if allowed_supervisor_lock is not None else None
    )
    supervisor_locks = tuple(
        lock
        for lock in _existing_supervisor_locks(output_root)
        if _nominal_absolute(lock) != allowed_lock
    )
    if supervisor_locks:
        raise RuntimeError(f"detached supervisor lock already exists: {supervisor_locks[0]}")
    if snapshot.evaluation_processes:
        raise RuntimeError(
            f"evaluation-owned process is already running: {snapshot.evaluation_processes}"
        )
    if snapshot.service_processes:
        raise RuntimeError(f"TTS service must be stopped: {snapshot.service_processes}")
    if snapshot.gpu_free_mib is None or snapshot.gpu_free_mib < MIN_FREE_GPU_MIB:
        raise RuntimeError(
            "insufficient free GPU memory: "
            f"required_mib={MIN_FREE_GPU_MIB}, actual_mib={snapshot.gpu_free_mib}"
        )


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


def preflight_detached(
    *,
    launcher_path: Path,
    expected_launcher_sha256: str,
    detached_script: Path,
    python_path: Path,
    output_root: Path,
    status_path: Path,
    probe: Callable[[], RuntimeSnapshot] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, object]:
    output_identity = _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
    )
    launcher = _verify_launcher(launcher_path, expected_launcher_sha256)
    detached = _verify_detached_script(detached_script)
    python = _verify_python_executable(python_path)
    snapshot = (probe or probe_runtime)()
    _require_safe_runtime(snapshot, output_root=output_root, status_path=status_path)
    execute = runner or subprocess.run
    command = (python["path"], launcher["path"], "preflight")
    completed = execute(
        command,
        cwd=UPSTREAM_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "speed-v7 v5 preflight failed: "
            f"exit_code={completed.returncode}, stderr={completed.stderr.strip()}"
        )
    report = _last_json_object(completed.stdout, source="speed-v7 v5 preflight")
    if report.get("passed") is not True or report.get("launch_performed") is not False:
        raise RuntimeError(f"speed-v7 v5 preflight report is not launch-safe: {report}")
    _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=output_identity,
    )
    _verify_launcher(launcher_path, expected_launcher_sha256)
    _verify_detached_script(detached_script)
    if _verify_python_executable(python_path) != python:
        raise RuntimeError("Python executable binding mismatch after v5 preflight")
    runtime_after_v5_preflight = (probe or probe_runtime)()
    _require_safe_runtime(
        runtime_after_v5_preflight,
        output_root=output_root,
        status_path=status_path,
    )
    return {
        "passed": True,
        "checked_at": _utc_now(),
        "v5_launcher": launcher,
        "detached_launcher": detached,
        "python": python,
        "output_identity": output_identity,
        "v5_preflight": report,
        "runtime": snapshot.as_dict(),
        "runtime_after_v5_preflight": runtime_after_v5_preflight.as_dict(),
        "queue_lock_path": output_identity["queue_lock_path"],
        "minimum_free_gpu_mib": MIN_FREE_GPU_MIB,
    }


def reserve_launch_directory(launches_root: Path, launch_id: str | None = None) -> Path:
    nominal_launches_root = _require_alias_free_output_path(
        launches_root,
        label="evaluation launches root",
    )
    launches_root.mkdir(parents=True, exist_ok=True)
    _require_alias_free_output_path(launches_root, label="evaluation launches root")
    if launch_id is not None:
        if _LAUNCH_ID_RE.fullmatch(launch_id) is None:
            raise ValueError(f"invalid launch id: {launch_id}")
        destination = launches_root / launch_id
        try:
            destination.mkdir()
        except FileExistsError:
            raise FileExistsError(f"refusing to reuse launch directory: {destination}") from None
        return _require_alias_free_output_path(
            destination,
            label="evaluation launch directory",
        )
    version = 1
    existing_versions = [
        int(match.group(1))
        for child in launches_root.iterdir()
        if (match := _LAUNCH_ID_RE.fullmatch(child.name)) is not None
    ]
    if existing_versions:
        version = max(existing_versions) + 1
    while True:
        destination = launches_root / f"launch-v{version:03d}"
        try:
            destination.mkdir()
        except FileExistsError:
            version += 1
            continue
        nominal_destination = _require_alias_free_output_path(
            destination,
            label="evaluation launch directory",
        )
        if nominal_destination.parent != nominal_launches_root:
            raise ValueError(
                f"launch directory is outside fixed launches root: {nominal_destination}"
            )
        return nominal_destination


def _lock_payload(*, token: str, supervisor_pid: int | None) -> dict[str, object]:
    return {
        "schema_version": LOCK_SCHEMA,
        "launch_token": token,
        "hostname": socket.gethostname(),
        "launcher_pid": os.getpid(),
        "supervisor_pid": supervisor_pid,
        "updated_at": _utc_now(),
    }


def _update_lock(path: Path, *, token: str, supervisor_pid: int) -> None:
    current = _read_json(path)
    if current.get("launch_token") != token:
        raise RuntimeError(f"supervisor lock ownership mismatch: {path}")
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(
                _json_bytes(_lock_payload(token=token, supervisor_pid=supervisor_pid))
            )
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _wait_for_worker_registration(
    lock_path: Path,
    *,
    token: str,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        lock = _read_json(lock_path)
        if lock.get("launch_token") != token:
            raise RuntimeError(f"supervisor lock ownership mismatch: {lock_path}")
        supervisor_pid = lock.get("supervisor_pid")
        if _is_positive_pid(supervisor_pid):
            return cast("int", supervisor_pid)
        if time.monotonic() >= deadline:
            raise RuntimeError("real supervisor worker did not register before timeout")
        time.sleep(0.05)


def launch_detached(
    *,
    launcher_path: Path,
    expected_launcher_sha256: str,
    python_path: Path,
    detached_script: Path,
    upstream_root: Path,
    output_root: Path,
    status_path: Path,
    probe: Callable[[], RuntimeSnapshot] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    popen: Callable[..., Any] | None = None,
    launch_id: str | None = None,
    worker_registration_timeout_seconds: float = 30.0,
) -> dict[str, object]:
    preflight = preflight_detached(
        launcher_path=launcher_path,
        expected_launcher_sha256=expected_launcher_sha256,
        detached_script=detached_script,
        python_path=python_path,
        output_root=output_root,
        status_path=status_path,
        probe=probe,
        runner=runner,
    )
    detached = _verify_detached_script(detached_script)
    if detached != preflight["detached_launcher"]:
        raise RuntimeError("detached launcher changed after preflight")
    python = _verify_python_executable(python_path)
    if python != preflight["python"]:
        raise RuntimeError("Python executable binding mismatch after preflight")
    output_identity = _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=preflight["output_identity"],
    )
    script = Path(detached["path"])
    launch_dir = reserve_launch_directory(output_root / "launches", launch_id)
    token = uuid.uuid4().hex
    reservation_path = launch_dir / _RESERVATION_NAME
    handoff_path = launch_dir / _PARENT_HANDOFF_NAME
    lock_path = launch_dir / _ACTIVE_LOCK_NAME
    _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=output_identity,
        launch_dir=launch_dir,
    )
    reservation: dict[str, object] = {
        "schema_version": RESERVATION_SCHEMA,
        "launch_token": token,
        "status": "RESERVED",
        "created_at": _utc_now(),
        "launch_dir": str(launch_dir),
        "output_root": output_identity["output_root"],
        "status_path": output_identity["status_path"],
        "output_identity": output_identity,
        "python": python,
        "detached_launcher": detached,
        "v5_launcher": preflight["v5_launcher"],
        "preflight": preflight,
    }
    _write_json_create_only(reservation_path, reservation)
    _write_json_create_only(lock_path, _lock_payload(token=token, supervisor_pid=None))
    command = (
        python["path"],
        str(script),
        "supervise",
        "--launch-dir",
        str(launch_dir),
        "--token",
        token,
    )
    spawn = popen or subprocess.Popen
    spawn_pid: int | None = None
    registered_supervisor_pid: int | None = None
    try:
        process = spawn(
            command,
            cwd=upstream_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=WINDOWS_DETACHED_FLAGS,
            shell=False,
        )
        spawn_pid = process.pid
        if not _is_positive_pid(spawn_pid):
            raise RuntimeError("detached supervisor returned an invalid PID")
        registered_supervisor_pid = _wait_for_worker_registration(
            lock_path,
            token=token,
            timeout_seconds=worker_registration_timeout_seconds,
        )
        handoff = {
            "schema_version": PARENT_HANDOFF_SCHEMA,
            "launch_token": token,
            "status": "DETACHED",
            "created_at": _utc_now(),
            "spawn_pid": spawn_pid,
            "supervisor_pid": registered_supervisor_pid,
            "supervisor_command": list(command),
            "detached_launcher": detached,
            "v5_launcher": preflight["v5_launcher"],
            "python": python,
            "output_identity": output_identity,
            "previous_evidence": _required_file_binding(reservation_path),
        }
        _write_json_create_only(handoff_path, handoff)
    except Exception as exc:
        archived_lock = _archive_lock(lock_path, token=token)
        failure_path = launch_dir / _SPAWN_FAILURE_NAME
        _write_json_create_only(
            failure_path,
            {
                "schema_version": SPAWN_FAILURE_SCHEMA,
                "launch_token": token,
                "status": "SPAWN_FAILED",
                "failed_at": _utc_now(),
                "spawn_pid": spawn_pid,
                "supervisor_pid": registered_supervisor_pid,
                "detached_launcher": detached,
                "v5_launcher": preflight["v5_launcher"],
                "python": python,
                "output_identity": output_identity,
                "previous_evidence": _required_file_binding(reservation_path),
                "archived_lock": archived_lock,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    return {
        "status": "DETACHED",
        "launch_dir": str(launch_dir),
        "reservation_path": str(reservation_path),
        "evidence_path": str(handoff_path),
        "spawn_pid": spawn_pid,
        "supervisor_pid": registered_supervisor_pid,
    }


def _file_binding(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _required_file_binding(path: Path) -> dict[str, object]:
    binding = _file_binding(path)
    if binding is None:
        raise RuntimeError(f"required immutable evidence is missing: {path}")
    return binding


def _read_phase(
    path: Path,
    *,
    expected_schema: str,
    expected_status: str | None = None,
    token: str | None = None,
    previous_path: Path | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != expected_schema:
        raise RuntimeError(f"immutable evidence chain has an invalid schema: {path}")
    if expected_status is not None and payload.get("status") != expected_status:
        raise RuntimeError(f"immutable evidence chain has an invalid phase status: {path}")
    phase_token = payload.get("launch_token")
    if not isinstance(phase_token, str) or not phase_token:
        raise RuntimeError(f"immutable evidence chain has an invalid token: {path}")
    if token is not None and phase_token != token:
        raise RuntimeError(f"immutable evidence chain token mismatch: {path}")
    if previous_path is not None and payload.get("previous_evidence") != (
        _required_file_binding(previous_path)
    ):
        raise RuntimeError(f"immutable evidence chain binding mismatch: {path}")
    return payload


def _require_phase_identity(
    payload: Mapping[str, object],
    reservation: Mapping[str, object],
    *,
    path: Path,
) -> None:
    for name in ("detached_launcher", "v5_launcher", "python", "output_identity"):
        if payload.get(name) != reservation.get(name):
            raise RuntimeError(f"immutable evidence chain {name} mismatch: {path}")


def _require_archived_lock(
    payload: Mapping[str, object],
    *,
    launch_dir: Path,
    token: str,
    expected_supervisor_pid: object,
) -> None:
    final_lock_path = launch_dir / _FINAL_LOCK_NAME
    if payload.get("archived_lock") != _required_file_binding(final_lock_path):
        raise RuntimeError(f"immutable evidence chain archived lock mismatch: {final_lock_path}")
    _validate_lock_payload(
        final_lock_path,
        token=token,
        expected_supervisor_pid=expected_supervisor_pid,
        label="archived lock",
    )


def _validate_lock_payload(
    path: Path,
    *,
    token: str,
    expected_supervisor_pid: object,
    label: str,
    allow_registered_pid: bool = False,
) -> None:
    try:
        payload = _read_json(path)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"immutable evidence chain {label} is unreadable: {path}") from exc
    if payload.get("schema_version") != LOCK_SCHEMA:
        raise RuntimeError(f"immutable evidence chain {label} schema mismatch: {path}")
    if payload.get("launch_token") != token:
        raise RuntimeError(f"immutable evidence chain {label} token mismatch: {path}")
    supervisor_pid = payload.get("supervisor_pid")
    if allow_registered_pid:
        if supervisor_pid is not None and not _is_positive_pid(supervisor_pid):
            raise RuntimeError(f"immutable evidence chain {label} PID mismatch: {path}")
    elif supervisor_pid != expected_supervisor_pid:
        raise RuntimeError(f"immutable evidence chain {label} PID mismatch: {path}")


def _validate_launch_chain(  # noqa: C901, PLR0912 - explicit fail-closed phase machine.
    launch_dir: Path,
) -> tuple[str, dict[str, Any], Path]:
    launch_dir = _require_alias_free_launch_artifacts(launch_dir)
    reservation_path = launch_dir / _RESERVATION_NAME
    reservation = _read_phase(
        reservation_path,
        expected_schema=RESERVATION_SCHEMA,
        expected_status="RESERVED",
    )
    _validate_reservation_output_identity(reservation, launch_dir=launch_dir)
    token = cast("str", reservation["launch_token"])
    active_lock = launch_dir / _ACTIVE_LOCK_NAME
    final_lock = launch_dir / _FINAL_LOCK_NAME
    if active_lock.is_file() and final_lock.is_file():
        raise RuntimeError("immutable evidence chain has both active and archived locks")

    spawn_failure_path = launch_dir / _SPAWN_FAILURE_NAME
    handoff_path = launch_dir / _PARENT_HANDOFF_NAME
    start_path = launch_dir / _SUPERVISOR_START_NAME
    terminal_path = launch_dir / _TERMINAL_NAME
    bootstrap_path = launch_dir / _BOOTSTRAP_FAILURE_NAME

    if spawn_failure_path.is_file():
        if any(path.exists() for path in (handoff_path, start_path, terminal_path, bootstrap_path)):
            raise RuntimeError("immutable evidence chain has conflicting spawn artifacts")
        failure = _read_phase(
            spawn_failure_path,
            expected_schema=SPAWN_FAILURE_SCHEMA,
            expected_status="SPAWN_FAILED",
            token=token,
            previous_path=reservation_path,
        )
        _require_phase_identity(failure, reservation, path=spawn_failure_path)
        _require_archived_lock(
            failure,
            launch_dir=launch_dir,
            token=token,
            expected_supervisor_pid=failure.get("supervisor_pid"),
        )
        if active_lock.exists():
            raise RuntimeError("immutable evidence chain retained an active lock after failure")
        return "SPAWN_FAILED", failure, spawn_failure_path

    if not handoff_path.is_file():
        if any(path.exists() for path in (start_path, terminal_path)):
            raise RuntimeError("immutable evidence chain is missing parent handoff evidence")
        if bootstrap_path.is_file():
            bootstrap = _read_phase(
                bootstrap_path,
                expected_schema=BOOTSTRAP_PRE_HANDOFF_FAILURE_SCHEMA,
                expected_status="SUPERVISOR_ERROR",
                token=token,
                previous_path=reservation_path,
            )
            _require_phase_identity(bootstrap, reservation, path=bootstrap_path)
            if bootstrap.get("spawn_pid") is not None:
                raise RuntimeError("immutable evidence chain has a pre-handoff bootstrap spawn PID")
            _require_archived_lock(
                bootstrap,
                launch_dir=launch_dir,
                token=token,
                expected_supervisor_pid=bootstrap.get("supervisor_pid"),
            )
            if active_lock.exists():
                raise RuntimeError(
                    "immutable evidence chain retained an active pre-handoff bootstrap lock"
                )
            return "SUPERVISOR_ERROR", bootstrap, bootstrap_path
        if final_lock.exists():
            raise RuntimeError("immutable evidence chain has an unbound archived lock")
        if not active_lock.is_file():
            raise RuntimeError("immutable evidence chain is missing its active lock")
        _validate_lock_payload(
            active_lock,
            token=token,
            expected_supervisor_pid=None,
            label="active lock",
            allow_registered_pid=True,
        )
        return "RESERVED", reservation, reservation_path

    handoff = _read_phase(
        handoff_path,
        expected_schema=PARENT_HANDOFF_SCHEMA,
        expected_status="DETACHED",
        token=token,
        previous_path=reservation_path,
    )
    _require_phase_identity(handoff, reservation, path=handoff_path)
    if not _is_positive_pid(handoff.get("spawn_pid")):
        raise RuntimeError("immutable evidence chain has an invalid spawn PID")
    if not _is_positive_pid(handoff.get("supervisor_pid")):
        raise RuntimeError("immutable evidence chain has an invalid supervisor PID")

    if start_path.is_file():
        start = _read_phase(
            start_path,
            expected_schema=SUPERVISOR_START_SCHEMA,
            expected_status="RUNNING",
            token=token,
            previous_path=handoff_path,
        )
        _require_phase_identity(start, reservation, path=start_path)
        if start.get("spawn_pid") != handoff.get("spawn_pid") or start.get(
            "supervisor_pid"
        ) != handoff.get("supervisor_pid"):
            raise RuntimeError("immutable evidence chain has an invalid PID handoff")
        latest_status = "RUNNING"
        latest_payload = start
        latest_path = start_path
    else:
        latest_status = "DETACHED"
        latest_payload = handoff
        latest_path = handoff_path

    if terminal_path.is_file():
        if not start_path.is_file():
            raise RuntimeError("immutable evidence chain terminal evidence has no start evidence")
        terminal = _read_phase(
            terminal_path,
            expected_schema=TERMINAL_SCHEMA,
            token=token,
            previous_path=start_path,
        )
        _require_phase_identity(terminal, reservation, path=terminal_path)
        _require_archived_lock(
            terminal,
            launch_dir=launch_dir,
            token=token,
            expected_supervisor_pid=terminal.get("supervisor_pid"),
        )
        if terminal.get("spawn_pid") != handoff.get("spawn_pid") or terminal.get(
            "supervisor_pid"
        ) != latest_payload.get("supervisor_pid"):
            raise RuntimeError("immutable evidence chain terminal PID mismatch")
        if active_lock.exists():
            raise RuntimeError("immutable evidence chain retained an active terminal lock")
        status = terminal.get("status")
        if status not in {"SUCCEEDED", "FAILED"}:
            raise RuntimeError("immutable evidence chain has an invalid terminal status")
        latest_status = cast("str", status)
        latest_payload = terminal
        latest_path = terminal_path

    if bootstrap_path.is_file():
        if terminal_path.exists() or spawn_failure_path.exists():
            raise RuntimeError("immutable evidence chain has conflicting failure artifacts")
        bootstrap_previous = start_path if start_path.is_file() else handoff_path
        bootstrap = _read_phase(
            bootstrap_path,
            expected_schema=BOOTSTRAP_FAILURE_SCHEMA,
            expected_status="SUPERVISOR_ERROR",
            token=token,
            previous_path=bootstrap_previous,
        )
        _require_phase_identity(bootstrap, reservation, path=bootstrap_path)
        _require_archived_lock(
            bootstrap,
            launch_dir=launch_dir,
            token=token,
            expected_supervisor_pid=bootstrap.get("supervisor_pid"),
        )
        if active_lock.exists():
            raise RuntimeError("immutable evidence chain retained an active bootstrap lock")
        latest_status = "SUPERVISOR_ERROR"
        latest_payload = bootstrap
        latest_path = bootstrap_path
    elif latest_status in {"DETACHED", "RUNNING"}:
        if not active_lock.is_file() or final_lock.exists():
            raise RuntimeError("immutable evidence chain has an invalid live lock state")
        _validate_lock_payload(
            active_lock,
            token=token,
            expected_supervisor_pid=latest_payload.get("supervisor_pid"),
            label="active lock",
        )

    return latest_status, latest_payload, latest_path


def _status_binding(path: Path) -> dict[str, object] | None:
    binding = _file_binding(path)
    if binding is None:
        return None
    binding["row_count"] = sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return binding


def _archive_lock(path: Path, *, token: str) -> dict[str, object]:
    current = _read_json(path)
    if current.get("launch_token") != token:
        raise RuntimeError(f"supervisor lock ownership mismatch: {path}")
    destination = path.with_name(_FINAL_LOCK_NAME)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite final supervisor lock: {destination}")
    _write_bytes_create_only(destination, path.read_bytes())
    path.unlink()
    return _required_file_binding(destination)


def _wait_for_parent_handoff(
    launch_dir: Path,
    *,
    token: str,
    supervisor_pid: int,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    if not _is_positive_pid(supervisor_pid):
        raise ValueError("detached supervisor requires a positive current PID")
    deadline = time.monotonic() + timeout_seconds
    while True:
        status, evidence, _ = _validate_launch_chain(launch_dir)
        if evidence.get("launch_token") != token:
            raise RuntimeError(f"launch evidence ownership mismatch: {launch_dir}")
        if status == "DETACHED":
            if evidence.get("supervisor_pid") != supervisor_pid:
                raise RuntimeError("parent handoff supervisor PID does not match this worker")
            return evidence
        if status in {"SPAWN_FAILED", "SUPERVISOR_ERROR"} or time.monotonic() >= deadline:
            raise RuntimeError("detached supervisor parent handoff did not complete")
        time.sleep(0.05)


def run_supervisor(
    *,
    launch_dir: Path,
    token: str,
    launcher_path: Path,
    expected_launcher_sha256: str,
    python_path: Path,
    upstream_root: Path,
    output_root: Path,
    status_path: Path,
    probe: Callable[[], RuntimeSnapshot] | None = None,
    popen: Callable[..., Any] | None = None,
    current_pid: int | None = None,
) -> int:
    output_identity = _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        launch_dir=launch_dir,
    )
    launch_dir = _nominal_absolute(launch_dir)
    reservation_path = launch_dir / _RESERVATION_NAME
    handoff_path = launch_dir / _PARENT_HANDOFF_NAME
    start_path = launch_dir / _SUPERVISOR_START_NAME
    terminal_path = launch_dir / _TERMINAL_NAME
    lock_path = launch_dir / _ACTIVE_LOCK_NAME
    log_path = launch_dir / "queue.log"
    supervisor_pid = current_pid if current_pid is not None else os.getpid()
    if not _is_positive_pid(supervisor_pid):
        raise ValueError("detached supervisor requires a positive current PID")
    reservation = _read_phase(
        reservation_path,
        expected_schema=RESERVATION_SCHEMA,
        token=token,
    )
    reserved_output_identity = _validate_reservation_output_identity(
        reservation,
        launch_dir=launch_dir,
    )
    _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=reservation.get("output_identity"),
        launch_dir=launch_dir,
    )
    if output_identity != reserved_output_identity:
        raise RuntimeError("reservation fixed output identity mismatch")
    _update_lock(lock_path, token=token, supervisor_pid=supervisor_pid)
    handoff = _wait_for_parent_handoff(
        launch_dir,
        token=token,
        supervisor_pid=supervisor_pid,
    )
    self_path = Path(__file__).resolve()
    self_binding = _verify_detached_script(self_path)
    if handoff.get("detached_launcher") != self_binding:
        raise RuntimeError("detached launcher binding mismatch")
    launcher = _verify_launcher(launcher_path, expected_launcher_sha256)
    if handoff.get("v5_launcher") != launcher:
        raise RuntimeError("launch evidence v5 launcher binding mismatch")
    python = _verify_python_executable(python_path)
    if reservation.get("python") != python or handoff.get("python") != python:
        raise RuntimeError("Python executable binding mismatch after reservation")
    runtime_before_queue = (probe or probe_runtime)()
    _require_safe_runtime(
        runtime_before_queue,
        output_root=output_root,
        status_path=status_path,
        allowed_supervisor_lock=lock_path,
    )
    _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=output_identity,
        launch_dir=launch_dir,
    )
    _update_lock(lock_path, token=token, supervisor_pid=supervisor_pid)
    start = {
        "schema_version": SUPERVISOR_START_SCHEMA,
        "launch_token": token,
        "status": "RUNNING",
        "started_at": _utc_now(),
        "spawn_pid": handoff["spawn_pid"],
        "supervisor_pid": supervisor_pid,
        "detached_launcher": self_binding,
        "v5_launcher": launcher,
        "python": python,
        "output_identity": output_identity,
        "runtime_before_queue": runtime_before_queue.as_dict(),
        "previous_evidence": _required_file_binding(handoff_path),
    }
    _write_json_create_only(start_path, start)
    command = (python["path"], launcher["path"], "launch")
    _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=output_identity,
        launch_dir=launch_dir,
    )
    queue_exit_code: int | None = None
    error: str | None = None
    try:
        spawn = popen or subprocess.Popen
        with log_path.open("xb") as output:
            process = spawn(
                command,
                cwd=upstream_root,
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
        _verify_launcher(launcher_path, expected_launcher_sha256)
        if _verify_detached_script(self_path) != self_binding:
            raise RuntimeError("detached launcher changed while the queue was running")
        if _verify_python_executable(python_path) != python:
            raise RuntimeError("Python executable binding mismatch after queue completion")
    except (OSError, RuntimeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    queue_lock_path = _queue_lock_path(status_path)
    queue_lock_present_after_queue = os.path.lexists(queue_lock_path)
    runtime_after = (probe or probe_runtime)()
    log_binding = _file_binding(log_path)
    queue_status_binding = _status_binding(status_path)
    queue_summary: dict[str, Any] | None = None
    if log_binding is not None:
        with contextlib.suppress(ValueError, UnicodeDecodeError):
            queue_summary = _last_json_object(
                log_path.read_text(encoding="utf-8"), source="speed-v7 queue log"
            )
    summary_failed = (
        queue_summary is None
        or not isinstance(queue_summary.get("failed"), list)
        or bool(queue_summary["failed"])
    )
    runtime_payload = runtime_after.as_dict()
    runtime_payload["process_residue"] = runtime_after.evaluation_processes
    _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=output_identity,
        launch_dir=launch_dir,
    )
    archived_lock = _archive_lock(lock_path, token=token)
    queue_lock_present_before_terminal = os.path.lexists(queue_lock_path)
    failed_checks = (
        error is not None,
        queue_exit_code != 0,
        summary_failed,
        queue_status_binding is None,
        queue_lock_present_after_queue,
        queue_lock_present_before_terminal,
        bool(runtime_after.errors),
        bool(runtime_after.evaluation_processes),
        bool(runtime_after.service_processes),
    )
    terminal_status = "FAILED" if any(failed_checks) else "SUCCEEDED"
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "launch_token": token,
        "status": terminal_status,
        "finished_at": _utc_now(),
        "spawn_pid": handoff["spawn_pid"],
        "supervisor_pid": supervisor_pid,
        "detached_launcher": self_binding,
        "v5_launcher": launcher,
        "python": python,
        "output_identity": output_identity,
        "previous_evidence": _required_file_binding(start_path),
        "archived_lock": archived_lock,
        "queue_command": list(command),
        "queue_exit_code": queue_exit_code,
        "queue_summary": queue_summary,
        "queue_log": log_binding,
        "queue_status": queue_status_binding,
        "queue_status_lock": {
            "path": str(_nominal_absolute(queue_lock_path)),
            "present_after_queue": queue_lock_present_after_queue,
            "present_before_terminal": queue_lock_present_before_terminal,
        },
        "runtime_after": runtime_payload,
        "error": error,
    }
    _validate_output_paths(
        output_root=output_root,
        status_path=status_path,
        expected_identity=output_identity,
        launch_dir=launch_dir,
    )
    _write_json_create_only(terminal_path, terminal)
    return 0 if terminal_status == "SUCCEEDED" else 1


def _record_supervisor_bootstrap_error(
    *,
    launch_dir: Path,
    token: str,
    error: BaseException,
) -> None:
    launch_dir = _require_alias_free_launch_artifacts(launch_dir)
    reservation_path = launch_dir / _RESERVATION_NAME
    handoff_path = launch_dir / _PARENT_HANDOFF_NAME
    start_path = launch_dir / _SUPERVISOR_START_NAME
    lock_path = launch_dir / _ACTIVE_LOCK_NAME
    error_path = launch_dir / _BOOTSTRAP_FAILURE_NAME
    reservation = _read_phase(
        reservation_path,
        expected_schema=RESERVATION_SCHEMA,
        token=token,
    )
    _validate_reservation_output_identity(reservation, launch_dir=launch_dir)
    if start_path.is_file():
        previous_path = start_path
        schema = BOOTSTRAP_FAILURE_SCHEMA
    elif handoff_path.is_file():
        previous_path = handoff_path
        schema = BOOTSTRAP_FAILURE_SCHEMA
    else:
        previous_path = reservation_path
        schema = BOOTSTRAP_PRE_HANDOFF_FAILURE_SCHEMA
    previous = _read_json(previous_path)
    if previous.get("launch_token") != token:
        raise RuntimeError(f"launch evidence ownership mismatch: {previous_path}")
    final_lock_path = launch_dir / _FINAL_LOCK_NAME
    if lock_path.is_file():
        lock_payload = _read_json(lock_path)
        archived_lock = _archive_lock(lock_path, token=token)
    else:
        lock_payload = _read_json(final_lock_path)
        if lock_payload.get("launch_token") != token:
            raise RuntimeError(f"supervisor lock ownership mismatch: {final_lock_path}")
        archived_lock = _required_file_binding(final_lock_path)
    payload = {
        "schema_version": schema,
        "launch_token": token,
        "status": "SUPERVISOR_ERROR",
        "failed_at": _utc_now(),
        "spawn_pid": previous.get("spawn_pid"),
        "supervisor_pid": lock_payload.get("supervisor_pid"),
        "detached_launcher": reservation["detached_launcher"],
        "v5_launcher": reservation["v5_launcher"],
        "python": reservation["python"],
        "output_identity": reservation["output_identity"],
        "previous_evidence": _required_file_binding(previous_path),
        "archived_lock": archived_lock,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    _write_json_create_only(error_path, payload)


def read_status(output_root: Path) -> dict[str, object]:
    nominal_output_root = _require_alias_free_output_path(
        output_root,
        label="evaluation output root",
    )
    launches_root = _require_alias_free_output_path(
        output_root / "launches",
        label="evaluation launches root",
    )
    if not launches_root.is_dir():
        return {"status": "NOT_LAUNCHED", "launches_root": str(launches_root)}
    if launches_root.parent != nominal_output_root:
        raise RuntimeError("evaluation launches root fixed identity mismatch")
    launches = [
        (int(match.group(1)), child)
        for child in launches_root.iterdir()
        if (match := _LAUNCH_ID_RE.fullmatch(child.name)) is not None
    ]
    if not launches:
        return {"status": "NOT_LAUNCHED", "launches_root": str(launches_root)}
    _, latest = max(launches)
    status, evidence, evidence_path = _validate_launch_chain(latest)
    return {
        "status": status,
        "launch_dir": str(_nominal_absolute(latest)),
        "evidence_path": str(_nominal_absolute(evidence_path)),
        "evidence": evidence,
        "supervisor_lock_active": (latest / _ACTIVE_LOCK_NAME).is_file(),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "launch", "status", "supervise"))
    parser.add_argument("--launch-id")
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--token")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode != "status":
        _assert_resolved_bundle_names(
            v5_launcher_bundle_name=V5_LAUNCHER_BUNDLE_NAME,
            detached_bundle_name=DETACHED_BUNDLE_NAME,
        )
    if args.mode == "preflight":
        result = preflight_detached(
            launcher_path=V5_LAUNCHER_PATH,
            expected_launcher_sha256=EXPECTED_V5_LAUNCHER_SHA256,
            detached_script=DETACHED_SCRIPT_PATH,
            python_path=PINNED_PYTHON,
            output_root=DEFAULT_OUTPUT_ROOT,
            status_path=DEFAULT_STATUS_PATH,
        )
    elif args.mode == "launch":
        result = launch_detached(
            launcher_path=V5_LAUNCHER_PATH,
            expected_launcher_sha256=EXPECTED_V5_LAUNCHER_SHA256,
            python_path=PINNED_PYTHON,
            detached_script=DETACHED_SCRIPT_PATH,
            upstream_root=UPSTREAM_ROOT,
            output_root=DEFAULT_OUTPUT_ROOT,
            status_path=DEFAULT_STATUS_PATH,
            launch_id=args.launch_id,
        )
    elif args.mode == "status":
        result = read_status(DEFAULT_OUTPUT_ROOT)
    else:
        if args.launch_dir is None or args.token is None:
            raise ValueError("supervise requires --launch-dir and --token")
        try:
            return run_supervisor(
                launch_dir=args.launch_dir,
                token=args.token,
                launcher_path=V5_LAUNCHER_PATH,
                expected_launcher_sha256=EXPECTED_V5_LAUNCHER_SHA256,
                python_path=PINNED_PYTHON,
                upstream_root=UPSTREAM_ROOT,
                output_root=DEFAULT_OUTPUT_ROOT,
                status_path=DEFAULT_STATUS_PATH,
            )
        except Exception as exc:
            _record_supervisor_bootstrap_error(
                launch_dir=args.launch_dir,
                token=args.token,
                error=exc,
            )
            return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
