# ruff: noqa: S404

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

SCRIPT_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v6_detached.py")
SUPERVISOR_PID = 4321
SPAWN_PID = 2468


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("speed_v6_detached", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_v4_launcher_uses_final_v6_bundle() -> None:
    module = _load_script()

    assert module.V4_LAUNCHER_PATH == (
        module.REMOTE_ROOT
        / "scripts"
        / "evaluation_speed_v6_v4"
        / "launch_600m_speaker_evaluation_queue_speed_v4.py"
    )
    assert module.EXPECTED_V4_LAUNCHER_SHA256 == (
        "91f38bb35f9b8f5dffd31a49f464ffd814404fe9cdbe10ab5ad35e2b4de7f9da"
    )
    assert module.DETACHED_SCRIPT_PATH == (
        module.REMOTE_ROOT
        / "scripts"
        / "evaluation_speed_v6_detached_v6"
        / "launch_600m_speaker_evaluation_queue_speed_v6_detached.py"
    )
    assert module.WINDOWS_DETACHED_FLAGS == 0x01000208  # noqa: PLR2004


def _write_launcher(path: Path, content: bytes = b"print('launcher')\n") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _safe_runtime(module: ModuleType) -> Any:  # noqa: ANN401
    return module.RuntimeSnapshot(
        observed_at="2026-08-03T00:00:00+00:00",
        processes=(),
        evaluation_processes=(),
        service_processes=(),
        gpu_total_mib=12_288.0,
        gpu_used_mib=512.0,
        gpu_free_mib=11_776.0,
        errors=(),
    )


def test_process_inventory_ignores_windows_idle_pid_zero() -> None:
    module = _load_script()

    rows = module._parse_processes(  # noqa: SLF001 - direct Windows parser contract.
        json.dumps(
            [
                {"ProcessId": 0, "ParentProcessId": 0, "Name": "System Idle Process"},
                {"ProcessId": 42, "ParentProcessId": 1, "Name": "python.exe"},
            ]
        )
    )

    assert rows == [{"ProcessId": 42, "ParentProcessId": 1, "Name": "python.exe"}]


def test_parent_handoff_accepts_a_distinct_venv_shim_spawn_pid(tmp_path: Path) -> None:
    module = _load_script()
    evidence_path = tmp_path / "launch-evidence.json"
    token = "shim-handoff-token"  # noqa: S105 - ownership token fixture.
    evidence_path.write_text(
        json.dumps(
            {
                "launch_token": token,
                "status": "DETACHED",
                "spawn_pid": SPAWN_PID,
                "supervisor_pid": SPAWN_PID,
            }
        ),
        encoding="utf-8",
    )

    evidence = module._wait_for_parent_handoff(  # noqa: SLF001 - handoff contract.
        evidence_path,
        token=token,
        supervisor_pid=SUPERVISOR_PID,
        timeout_seconds=0,
    )

    assert evidence["spawn_pid"] == SPAWN_PID


def test_supervisor_bootstrap_error_is_persisted_and_lock_archived(tmp_path: Path) -> None:
    module = _load_script()
    launch_dir = tmp_path / "launch-v001"
    launch_dir.mkdir()
    token = "owned-token"  # noqa: S105 - ownership token fixture, not a credential.
    module._write_json_create_only(  # noqa: SLF001 - bootstrap evidence contract.
        launch_dir / "launch-evidence.json",
        {"schema_version": module.EVIDENCE_SCHEMA, "launch_token": token, "status": "DETACHED"},
    )
    module._write_json_create_only(  # noqa: SLF001 - bootstrap lock contract.
        launch_dir / "supervisor.lock",
        module._lock_payload(token=token, supervisor_pid=42),  # noqa: SLF001
    )

    try:
        raise RuntimeError("bootstrap exploded")  # noqa: EM101, TRY003, TRY301
    except RuntimeError as error:
        module._record_supervisor_bootstrap_error(  # noqa: SLF001
            launch_dir=launch_dir,
            token=token,
            error=error,
        )

    evidence = json.loads((launch_dir / "launch-evidence.json").read_text(encoding="utf-8"))
    failure = json.loads(
        (launch_dir / "supervisor-bootstrap-error.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "SUPERVISOR_ERROR"
    assert evidence["error"] == "RuntimeError: bootstrap exploded"
    assert failure["traceback"].startswith("Traceback")
    assert not (launch_dir / "supervisor.lock").exists()
    assert (launch_dir / "supervisor-lock-final.json").is_file()


def test_preflight_fails_closed_before_invoking_unpinned_v4(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v4.py"
    _write_launcher(launcher)
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="v4 launcher SHA-256 mismatch"):
        module.preflight_detached(
            launcher_path=launcher,
            expected_launcher_sha256="0" * 64,
            python_path=tmp_path / "python.exe",
            output_root=tmp_path / "evaluation_speed_v6",
            status_path=tmp_path / "evaluation_speed_v6/status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=lambda command, **_kwargs: calls.append(command),
        )

    assert calls == []


def test_preflight_invokes_pinned_v4_without_shell_and_binds_report(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v4.py"
    launcher_sha256 = _write_launcher(launcher)
    output_root = tmp_path / "evaluation_speed_v6"
    output_root.mkdir()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"passed": True, "launch_performed": False}) + "\n",
            stderr="",
        )

    report = module.preflight_detached(
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        python_path=tmp_path / "python.exe",
        output_root=output_root,
        status_path=output_root / "status.jsonl",
        probe=lambda: _safe_runtime(module),
        runner=runner,
    )

    assert calls[0][0] == (str(tmp_path / "python.exe"), str(launcher), "preflight")
    assert calls[0][1]["shell"] is False
    assert report["v4_launcher"] == {
        "path": str(launcher.resolve()),
        "sha256": launcher_sha256,
    }
    assert report["v4_preflight"]["passed"] is True
    assert report["runtime"]["gpu_free_mib"] == pytest.approx(11_776.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("queue_lock", "evaluation queue lock already exists"),
        ("supervisor_lock", "detached supervisor lock already exists"),
        ("evaluation_process", "evaluation-owned process is already running"),
        ("service_process", "TTS service must be stopped"),
        ("low_gpu", "insufficient free GPU memory"),
        ("probe_error", "runtime probe failed closed"),
    ],
)
def test_preflight_rejects_unsafe_runtime(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v4.py"
    launcher_sha256 = _write_launcher(launcher)
    output_root = tmp_path / "evaluation_speed_v6"
    output_root.mkdir()
    status_path = output_root / "status.jsonl"
    snapshot = _safe_runtime(module)
    if mutation == "queue_lock":
        status_path.with_suffix(".jsonl.lock").write_text("locked", encoding="utf-8")
    elif mutation == "supervisor_lock":
        lock = output_root / "launches/launch-v001/supervisor.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("locked", encoding="utf-8")
    elif mutation == "evaluation_process":
        snapshot = module.RuntimeSnapshot(
            **{
                **snapshot.as_dict(),
                "evaluation_processes": (
                    {
                        "pid": 10,
                        "command_line": "run_600m_speaker_evaluation_queue.py",
                    },
                ),
            }
        )
    elif mutation == "service_process":
        snapshot = module.RuntimeSnapshot(
            **{
                **snapshot.as_dict(),
                "service_processes": (
                    {
                        "pid": 11,
                        "command_line": "uvicorn irodori_tts_infra.server.app:app",
                    },
                ),
            }
        )
    elif mutation == "low_gpu":
        snapshot = module.RuntimeSnapshot(
            **{**snapshot.as_dict(), "gpu_used_mib": 2_500.0, "gpu_free_mib": 9_788.0}
        )
    else:
        snapshot = module.RuntimeSnapshot(
            **{**snapshot.as_dict(), "errors": ("nvidia-smi failed",)}
        )
    calls: list[object] = []

    with pytest.raises(RuntimeError, match=message):
        module.preflight_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            output_root=output_root,
            status_path=status_path,
            probe=lambda: snapshot,
            runner=lambda *_args, **_kwargs: calls.append(True),
        )

    assert calls == []


def test_reserve_launch_directory_is_versioned_create_only(tmp_path: Path) -> None:
    module = _load_script()
    launches = tmp_path / "evaluation_speed_v6/launches"

    first = module.reserve_launch_directory(launches)
    second = module.reserve_launch_directory(launches)

    assert first.name == "launch-v001"
    assert second.name == "launch-v002"
    with pytest.raises(FileExistsError, match="refusing to reuse launch directory"):
        module.reserve_launch_directory(launches, launch_id="launch-v002")


def test_launch_spawns_detached_supervisor_with_create_only_evidence(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v4.py"
    launcher_sha256 = _write_launcher(launcher)
    detached_script = tmp_path / "detached.py"
    detached_script.write_text("# detached\n", encoding="utf-8")
    output_root = tmp_path / "evaluation_speed_v6"
    output_root.mkdir()
    run_calls: list[object] = []
    popen_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        run_calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"passed": True, "launch_performed": False}),
            stderr="",
        )

    def popen(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        popen_calls.append((command, kwargs))
        return SimpleNamespace(pid=SUPERVISOR_PID)

    result = module.launch_detached(
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        python_path=tmp_path / "python.exe",
        detached_script=detached_script,
        upstream_root=tmp_path,
        output_root=output_root,
        status_path=output_root / "status.jsonl",
        probe=lambda: _safe_runtime(module),
        runner=runner,
        popen=popen,
    )

    launch_dir = Path(result["launch_dir"])
    evidence = json.loads((launch_dir / "launch-evidence.json").read_text(encoding="utf-8"))
    lock = json.loads((launch_dir / "supervisor.lock").read_text(encoding="utf-8"))
    assert len(run_calls) == 1
    assert popen_calls[0][0] == (
        str(tmp_path / "python.exe"),
        str(detached_script.resolve()),
        "supervise",
        "--launch-dir",
        str(launch_dir.resolve()),
        "--token",
        evidence["launch_token"],
    )
    assert popen_calls[0][1]["shell"] is False
    assert popen_calls[0][1]["creationflags"] == module.WINDOWS_DETACHED_FLAGS
    assert evidence["status"] == "DETACHED"
    assert evidence["spawn_pid"] == SUPERVISOR_PID
    assert evidence["supervisor_pid"] == SUPERVISOR_PID
    assert lock["supervisor_pid"] == SUPERVISOR_PID
    assert not (launch_dir / "queue.log").exists()


def test_spawn_failure_preserves_evidence_and_archives_active_lock(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v4.py"
    launcher_sha256 = _write_launcher(launcher)
    detached_script = tmp_path / "detached.py"
    detached_script.write_text("# detached\n", encoding="utf-8")
    output_root = tmp_path / "evaluation_speed_v6"
    output_root.mkdir()

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"passed": True, "launch_performed": False}),
            stderr="",
        )

    def popen(_command: tuple[str, ...], **_kwargs: object) -> None:
        message = "spawn denied"
        raise OSError(message)

    with pytest.raises(OSError, match="spawn denied"):
        module.launch_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            detached_script=detached_script,
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=runner,
            popen=popen,
        )

    launch_dir = output_root / "launches/launch-v001"
    evidence = json.loads((launch_dir / "launch-evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "SPAWN_FAILED"
    assert "spawn denied" in evidence["error"]
    assert not (launch_dir / "supervisor.lock").exists()
    assert (launch_dir / "supervisor-lock-final.json").is_file()


def test_supervisor_captures_queue_and_finalizes_hashes_and_lock(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v4.py"
    launcher_sha256 = _write_launcher(launcher)
    output_root = tmp_path / "evaluation_speed_v6"
    status_path = output_root / "status.jsonl"
    output_root.mkdir()
    launch_dir = output_root / "launches/launch-v001"
    launch_dir.mkdir(parents=True)
    token = "a" * 32
    initial = {
        "schema_version": module.EVIDENCE_SCHEMA,
        "launch_token": token,
        "status": "DETACHED",
        "spawn_pid": SPAWN_PID,
        "supervisor_pid": SPAWN_PID,
        "detached_launcher": {
            "path": str(SCRIPT_PATH.resolve()),
            "sha256": module.sha256_file(SCRIPT_PATH),
        },
        "v4_launcher": {"path": str(launcher.resolve()), "sha256": launcher_sha256},
        "status_path": str(status_path.resolve()),
    }
    (launch_dir / "launch-evidence.json").write_text(json.dumps(initial), encoding="utf-8")
    (launch_dir / "supervisor.lock").write_text(
        json.dumps({"launch_token": token, "supervisor_pid": SPAWN_PID}),
        encoding="utf-8",
    )
    status_path.write_text('{"stage":"evaluate","status":"success"}\n', encoding="utf-8")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class Process:
        returncode = 0

        def wait(self) -> int:
            return self.returncode

    def popen(command: tuple[str, ...], **kwargs: object) -> Process:
        calls.append((command, kwargs))
        output = cast("Any", kwargs["stdout"])
        output.write(b'{"succeeded":["all"],"skipped":[],"reused":[],"failed":[]}\n')
        output.flush()
        return Process()

    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=token,
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        python_path=tmp_path / "python.exe",
        upstream_root=tmp_path,
        output_root=output_root,
        status_path=status_path,
        probe=lambda: _safe_runtime(module),
        popen=popen,
        current_pid=SUPERVISOR_PID,
    )

    evidence = json.loads((launch_dir / "launch-evidence.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert calls[0][0] == (str(tmp_path / "python.exe"), str(launcher), "launch")
    assert calls[0][1]["shell"] is False
    assert evidence["status"] == "SUCCEEDED"
    assert evidence["spawn_pid"] == SPAWN_PID
    assert evidence["supervisor_pid"] == SUPERVISOR_PID
    assert evidence["queue_exit_code"] == 0
    assert evidence["queue_summary"]["failed"] == []
    assert evidence["queue_log"]["sha256"] == module.sha256_file(launch_dir / "queue.log")
    assert evidence["queue_status"]["sha256"] == module.sha256_file(status_path)
    assert evidence["queue_status"]["row_count"] == 1
    assert evidence["runtime_after"]["process_residue"] == []
    assert not (launch_dir / "supervisor.lock").exists()
    assert (launch_dir / "supervisor-lock-final.json").is_file()


def test_supervisor_rejects_changed_detached_launcher_binding(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v4.py"
    launcher_sha256 = _write_launcher(launcher)
    output_root = tmp_path / "evaluation_speed_v6"
    launch_dir = output_root / "launches/launch-v001"
    launch_dir.mkdir(parents=True)
    token = "b" * 32
    evidence = {
        "schema_version": module.EVIDENCE_SCHEMA,
        "launch_token": token,
        "status": "DETACHED",
        "spawn_pid": SUPERVISOR_PID,
        "supervisor_pid": SUPERVISOR_PID,
        "detached_launcher": {
            "path": str(SCRIPT_PATH.resolve()),
            "sha256": "0" * 64,
        },
        "v4_launcher": {"path": str(launcher.resolve()), "sha256": launcher_sha256},
    }
    (launch_dir / "launch-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    (launch_dir / "supervisor.lock").write_text(
        json.dumps({"launch_token": token, "supervisor_pid": SUPERVISOR_PID}),
        encoding="utf-8",
    )
    calls: list[object] = []

    with pytest.raises(RuntimeError, match="detached launcher binding mismatch"):
        module.run_supervisor(
            launch_dir=launch_dir,
            token=token,
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: _safe_runtime(module),
            popen=lambda *_args, **_kwargs: calls.append(True),
            current_pid=SUPERVISOR_PID,
        )

    assert calls == []


def test_just_and_agents_catalog_register_detached_speed_v6() -> None:
    justfile = Path("justfile").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "speaker-evaluation-speed-v6-detached" in justfile
    assert "remote-speaker-evaluation-speed-v6-detached" in justfile
    assert "evaluation_speed_v6_detached_v6" in justfile
    assert "speaker-evaluation-speed-v6-detached" in agents
