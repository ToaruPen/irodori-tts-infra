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
DEFAULT_OUTPUT_ROOT = REMOTE_ROOT / "evaluation_speed_v6"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v6.jsonl"
DEFAULT_LAUNCHES_ROOT = DEFAULT_OUTPUT_ROOT / "launches"
V4_LAUNCHER_PATH = (
    REMOTE_ROOT
    / "scripts"
    / "evaluation_speed_v6_v4"
    / "launch_600m_speaker_evaluation_queue_speed_v4.py"
)
DETACHED_SCRIPT_PATH = (
    REMOTE_ROOT
    / "scripts"
    / "evaluation_speed_v6_detached_v6"
    / "launch_600m_speaker_evaluation_queue_speed_v6_detached.py"
)

EXPECTED_V4_LAUNCHER_SHA256 = "91f38bb35f9b8f5dffd31a49f464ffd814404fe9cdbe10ab5ad35e2b4de7f9da"
EVIDENCE_SCHEMA = "speaker-evaluation-speed-v6-detached-launch/v1"
LOCK_SCHEMA = "speaker-evaluation-speed-v6-detached-lock/v1"
MIN_FREE_GPU_MIB = 10_500.0
WINDOWS_DETACHED_FLAGS = 0x00000008 | 0x00000200 | 0x01000000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LAUNCH_ID_RE = re.compile(r"launch-v([0-9]{3,})\Z")
_EVALUATION_MARKERS = (
    "run_600m_speaker_evaluation_queue.py",
    "launch_600m_speaker_evaluation_queue_speed_v4.py",
    "launch_600m_speaker_evaluation_queue_speed_v6_detached.py",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(_json_bytes(payload))
        destination.flush()
        os.fsync(destination.fileno())


def _atomic_update_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    token: str,
) -> None:
    current = _read_json(path)
    if current.get("launch_token") != token:
        raise RuntimeError(f"launch evidence ownership mismatch: {path}")
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(_json_bytes(payload))
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _verify_launcher(path: Path, expected_sha256: str) -> dict[str, str]:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("v4 launcher SHA-256 pin is not finalized")
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"v4 launcher is unsafe or missing: {path}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            "v4 launcher SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual}, path={resolved}"
        )
    return {"path": str(resolved), "sha256": actual}


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
    evaluation = tuple(
        row
        for row in visible
        if any(marker in _command_line(row).casefold() for marker in _EVALUATION_MARKERS)
    )
    services = tuple(
        row
        for row in visible
        if any(marker in _command_line(row).casefold() for marker in _SERVICE_MARKERS)
    )
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
) -> None:
    if snapshot.errors:
        raise RuntimeError(f"runtime probe failed closed: {snapshot.errors}")
    queue_lock = _queue_lock_path(status_path)
    if os.path.lexists(queue_lock):
        raise RuntimeError(f"evaluation queue lock already exists: {queue_lock}")
    supervisor_locks = _existing_supervisor_locks(output_root)
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
    python_path: Path,
    output_root: Path,
    status_path: Path,
    probe: Callable[[], RuntimeSnapshot] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, object]:
    launcher = _verify_launcher(launcher_path, expected_launcher_sha256)
    snapshot = (probe or probe_runtime)()
    _require_safe_runtime(snapshot, output_root=output_root, status_path=status_path)
    execute = runner or subprocess.run
    command = (str(python_path), launcher["path"], "preflight")
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
            "speed-v6 v4 preflight failed: "
            f"exit_code={completed.returncode}, stderr={completed.stderr.strip()}"
        )
    report = _last_json_object(completed.stdout, source="speed-v6 v4 preflight")
    if report.get("passed") is not True or report.get("launch_performed") is not False:
        raise RuntimeError(f"speed-v6 v4 preflight report is not launch-safe: {report}")
    _verify_launcher(launcher_path, expected_launcher_sha256)
    return {
        "passed": True,
        "checked_at": _utc_now(),
        "v4_launcher": launcher,
        "v4_preflight": report,
        "runtime": snapshot.as_dict(),
        "queue_lock_path": str(_queue_lock_path(status_path).resolve()),
        "minimum_free_gpu_mib": MIN_FREE_GPU_MIB,
    }


def reserve_launch_directory(launches_root: Path, launch_id: str | None = None) -> Path:
    launches_root.mkdir(parents=True, exist_ok=True)
    if launch_id is not None:
        if _LAUNCH_ID_RE.fullmatch(launch_id) is None:
            raise ValueError(f"invalid launch id: {launch_id}")
        destination = launches_root / launch_id
        try:
            destination.mkdir()
        except FileExistsError:
            raise FileExistsError(f"refusing to reuse launch directory: {destination}") from None
        return destination.resolve()
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
        return destination.resolve()


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
) -> dict[str, object]:
    preflight = preflight_detached(
        launcher_path=launcher_path,
        expected_launcher_sha256=expected_launcher_sha256,
        python_path=python_path,
        output_root=output_root,
        status_path=status_path,
        probe=probe,
        runner=runner,
    )
    script = detached_script.resolve()
    if detached_script.is_symlink() or not script.is_file():
        raise ValueError(f"detached launcher is unsafe or missing: {detached_script}")
    launch_dir = reserve_launch_directory(output_root / "launches", launch_id)
    token = uuid.uuid4().hex
    evidence_path = launch_dir / "launch-evidence.json"
    lock_path = launch_dir / "supervisor.lock"
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA,
        "launch_token": token,
        "status": "RESERVED",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "launch_dir": str(launch_dir),
        "output_root": str(output_root.resolve()),
        "status_path": str(status_path.resolve()),
        "python": {
            "path": str(python_path),
            "sha256": sha256_file(python_path) if python_path.is_file() else None,
        },
        "detached_launcher": {"path": str(script), "sha256": sha256_file(script)},
        "v4_launcher": preflight["v4_launcher"],
        "preflight": preflight,
        "spawn_pid": None,
        "supervisor_pid": None,
        "queue_exit_code": None,
        "queue_summary": None,
        "queue_log": None,
        "queue_status": None,
        "runtime_after": None,
    }
    _write_json_create_only(evidence_path, evidence)
    _write_json_create_only(lock_path, _lock_payload(token=token, supervisor_pid=None))
    command = (
        str(python_path),
        str(script),
        "supervise",
        "--launch-dir",
        str(launch_dir),
        "--token",
        token,
    )
    spawn = popen or subprocess.Popen
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
        supervisor_pid = process.pid
        if not _is_positive_pid(supervisor_pid):
            raise RuntimeError("detached supervisor returned an invalid PID")
        _update_lock(lock_path, token=token, supervisor_pid=supervisor_pid)
        evidence.update(
            {
                "status": "DETACHED",
                "updated_at": _utc_now(),
                "spawn_pid": supervisor_pid,
                "supervisor_pid": supervisor_pid,
                "supervisor_command": list(command),
            }
        )
        _atomic_update_json(evidence_path, evidence, token=token)
    except Exception as exc:
        evidence.update(
            {
                "status": "SPAWN_FAILED",
                "updated_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_update_json(evidence_path, evidence, token=token)
        _archive_lock(lock_path, token=token)
        raise
    return {
        "status": "DETACHED",
        "launch_dir": str(launch_dir),
        "evidence_path": str(evidence_path),
        "spawn_pid": evidence["spawn_pid"],
        "supervisor_pid": evidence["supervisor_pid"],
    }


def _file_binding(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _status_binding(path: Path) -> dict[str, object] | None:
    binding = _file_binding(path)
    if binding is None:
        return None
    binding["row_count"] = sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return binding


def _archive_lock(path: Path, *, token: str) -> None:
    current = _read_json(path)
    if current.get("launch_token") != token:
        raise RuntimeError(f"supervisor lock ownership mismatch: {path}")
    destination = path.with_name("supervisor-lock-final.json")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite final supervisor lock: {destination}")
    path.replace(destination)


def _wait_for_parent_handoff(
    evidence_path: Path,
    *,
    token: str,
    supervisor_pid: int,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    if not _is_positive_pid(supervisor_pid):
        raise ValueError("detached supervisor requires a positive current PID")
    deadline = time.monotonic() + timeout_seconds
    while True:
        evidence = _read_json(evidence_path)
        if evidence.get("launch_token") != token:
            raise RuntimeError(f"launch evidence ownership mismatch: {evidence_path}")
        spawn_pid = evidence.get("spawn_pid")
        if (
            evidence.get("status") == "DETACHED"
            and _is_positive_pid(spawn_pid)
            and evidence.get("supervisor_pid") == spawn_pid
        ):
            return evidence
        if evidence.get("status") in {"SPAWN_FAILED", "SUPERVISOR_ERROR"} or (
            time.monotonic() >= deadline
        ):
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
    resolved_launch_dir = launch_dir.resolve()
    expected_parent = (output_root.resolve() / "launches").resolve()
    if (
        resolved_launch_dir.parent != expected_parent
        or _LAUNCH_ID_RE.fullmatch(resolved_launch_dir.name) is None
    ):
        raise ValueError(f"launch directory is outside fixed launches root: {launch_dir}")
    evidence_path = launch_dir / "launch-evidence.json"
    lock_path = launch_dir / "supervisor.lock"
    log_path = launch_dir / "queue.log"
    supervisor_pid = current_pid if current_pid is not None else os.getpid()
    evidence = _wait_for_parent_handoff(
        evidence_path,
        token=token,
        supervisor_pid=supervisor_pid,
    )
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        raise RuntimeError(f"launch evidence ownership mismatch: {evidence_path}")
    self_path = Path(__file__).resolve()
    self_binding = {"path": str(self_path), "sha256": sha256_file(self_path)}
    if evidence.get("detached_launcher") != self_binding:
        raise RuntimeError("detached launcher binding mismatch")
    launcher = _verify_launcher(launcher_path, expected_launcher_sha256)
    if evidence.get("v4_launcher") != launcher:
        raise RuntimeError("launch evidence v4 launcher binding mismatch")
    _update_lock(lock_path, token=token, supervisor_pid=supervisor_pid)
    evidence.update(
        {
            "status": "RUNNING",
            "updated_at": _utc_now(),
            "started_at": _utc_now(),
            "supervisor_pid": supervisor_pid,
        }
    )
    _atomic_update_json(evidence_path, evidence, token=token)
    command = (str(python_path), launcher["path"], "launch")
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
    except (OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    runtime_after = (probe or probe_runtime)()
    log_binding = _file_binding(log_path)
    queue_summary: dict[str, Any] | None = None
    if log_binding is not None:
        with contextlib.suppress(ValueError, UnicodeDecodeError):
            queue_summary = _last_json_object(
                log_path.read_text(encoding="utf-8"), source="speed-v6 queue log"
            )
    summary_failed = (
        queue_summary is None
        or not isinstance(queue_summary.get("failed"), list)
        or bool(queue_summary["failed"])
    )
    runtime_payload = runtime_after.as_dict()
    runtime_payload["process_residue"] = runtime_after.evaluation_processes
    failed_checks = (
        error is not None,
        queue_exit_code != 0,
        summary_failed,
        bool(runtime_after.errors),
        bool(runtime_after.evaluation_processes),
        bool(runtime_after.service_processes),
    )
    terminal_status = "FAILED" if any(failed_checks) else "SUCCEEDED"
    evidence.update(
        {
            "status": terminal_status,
            "updated_at": _utc_now(),
            "finished_at": _utc_now(),
            "queue_command": list(command),
            "queue_exit_code": queue_exit_code,
            "queue_summary": queue_summary,
            "queue_log": log_binding,
            "queue_status": _status_binding(status_path),
            "runtime_after": runtime_payload,
            "error": error,
        }
    )
    _atomic_update_json(evidence_path, evidence, token=token)
    _archive_lock(lock_path, token=token)
    return 0 if terminal_status == "SUCCEEDED" else 1


def _record_supervisor_bootstrap_error(
    *,
    launch_dir: Path,
    token: str,
    error: BaseException,
) -> None:
    evidence_path = launch_dir / "launch-evidence.json"
    lock_path = launch_dir / "supervisor.lock"
    error_path = launch_dir / "supervisor-bootstrap-error.json"
    payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "launch_token": token,
        "status": "SUPERVISOR_ERROR",
        "failed_at": _utc_now(),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    _write_json_create_only(error_path, payload)
    if evidence_path.is_file():
        evidence = _read_json(evidence_path)
        if evidence.get("launch_token") == token:
            evidence.update(
                {
                    "status": "SUPERVISOR_ERROR",
                    "updated_at": _utc_now(),
                    "error": payload["error"],
                    "supervisor_bootstrap_error": _file_binding(error_path),
                }
            )
            _atomic_update_json(evidence_path, evidence, token=token)
    if lock_path.is_file():
        _archive_lock(lock_path, token=token)


def read_status(output_root: Path) -> dict[str, object]:
    launches_root = output_root / "launches"
    if not launches_root.is_dir():
        return {"status": "NOT_LAUNCHED", "launches_root": str(launches_root.resolve())}
    launches = sorted(
        child for child in launches_root.iterdir() if _LAUNCH_ID_RE.fullmatch(child.name)
    )
    if not launches:
        return {"status": "NOT_LAUNCHED", "launches_root": str(launches_root.resolve())}
    latest = launches[-1]
    evidence_path = latest / "launch-evidence.json"
    evidence = _read_json(evidence_path)
    return {
        "status": evidence.get("status"),
        "launch_dir": str(latest.resolve()),
        "evidence": evidence,
        "supervisor_lock_active": (latest / "supervisor.lock").is_file(),
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
    if args.mode == "preflight":
        result = preflight_detached(
            launcher_path=V4_LAUNCHER_PATH,
            expected_launcher_sha256=EXPECTED_V4_LAUNCHER_SHA256,
            python_path=PINNED_PYTHON,
            output_root=DEFAULT_OUTPUT_ROOT,
            status_path=DEFAULT_STATUS_PATH,
        )
    elif args.mode == "launch":
        result = launch_detached(
            launcher_path=V4_LAUNCHER_PATH,
            expected_launcher_sha256=EXPECTED_V4_LAUNCHER_SHA256,
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
                launcher_path=V4_LAUNCHER_PATH,
                expected_launcher_sha256=EXPECTED_V4_LAUNCHER_SHA256,
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
