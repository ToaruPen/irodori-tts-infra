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

SCRIPT_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v7_detached.py")
V5_LAUNCHER_SOURCE_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v5.py")
SUPERVISOR_PID = 4321
SPAWN_PID = 2468


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("speed_v7_detached", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v7_defaults_keep_launcher_sha_pending_until_successor_is_final() -> None:
    module = _load_script()

    assert module.DEFAULT_OUTPUT_ROOT == module.REMOTE_ROOT / "evaluation_speed_v7"
    assert module.DEFAULT_STATUS_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v7.jsonl"
    )
    assert module.V5_LAUNCHER_BUNDLE_NAME == "evaluation_speed_v7_v1"
    assert module.V5_LAUNCHER_PATH == (
        module.REMOTE_ROOT
        / "scripts"
        / "evaluation_speed_v7_v1"
        / "launch_600m_speaker_evaluation_queue_speed_v5.py"
    )
    assert module.EXPECTED_V5_LAUNCHER_SHA256 == module.PENDING_V5_LAUNCHER_SHA256
    assert hashlib.sha256(V5_LAUNCHER_SOURCE_PATH.read_bytes()).hexdigest() != (
        module.EXPECTED_V5_LAUNCHER_SHA256
    )
    assert module.DETACHED_BUNDLE_NAME == "evaluation_speed_v7_detached_v1"
    assert module.DETACHED_SCRIPT_PATH == (
        module.REMOTE_ROOT
        / "scripts"
        / "evaluation_speed_v7_detached_v1"
        / "launch_600m_speaker_evaluation_queue_speed_v7_detached.py"
    )
    assert module.RESERVATION_SCHEMA == ("speaker-evaluation-speed-v7-detached-reservation/v1")
    assert module.PARENT_HANDOFF_SCHEMA == (
        "speaker-evaluation-speed-v7-detached-parent-handoff/v1"
    )
    assert module.SUPERVISOR_START_SCHEMA == (
        "speaker-evaluation-speed-v7-detached-supervisor-start/v1"
    )
    assert module.TERMINAL_SCHEMA == "speaker-evaluation-speed-v7-detached-terminal/v1"
    assert module.LOCK_SCHEMA == "speaker-evaluation-speed-v7-detached-lock/v1"
    assert module.WINDOWS_DETACHED_FLAGS == 0x01000208  # noqa: PLR2004


def test_pending_launcher_pin_and_bundle_names_fail_closed(tmp_path: Path) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="PENDING_REMOTE_SPEED_V7_LAUNCHER_SHA256"):
        module._verify_launcher(  # noqa: SLF001 - fail-closed pin contract.
            tmp_path / "missing-launcher.py",
            module.PENDING_V5_LAUNCHER_SHA256,
        )
    with pytest.raises(ValueError, match="PENDING_REMOTE_SPEED_V7_LAUNCHER_BUNDLE_NAME"):
        module._assert_resolved_bundle_names(  # noqa: SLF001 - fail-closed path contract.
            v5_launcher_bundle_name=module.PENDING_V5_LAUNCHER_BUNDLE_NAME,
            detached_bundle_name="evaluation_speed_v7_detached_v001",
        )
    with pytest.raises(ValueError, match="PENDING_REMOTE_SPEED_V7_DETACHED_BUNDLE_NAME"):
        module._assert_resolved_bundle_names(  # noqa: SLF001 - fail-closed path contract.
            v5_launcher_bundle_name="evaluation_speed_v7_v001",
            detached_bundle_name=module.PENDING_DETACHED_BUNDLE_NAME,
        )
    with pytest.raises(ValueError, match="versioned speed-v7 launcher bundle"):
        module._assert_resolved_bundle_names(  # noqa: SLF001 - versioned path contract.
            v5_launcher_bundle_name="evaluation_speed_v7",
            detached_bundle_name="evaluation_speed_v7_detached_v001",
        )
    with pytest.raises(ValueError, match="versioned speed-v7 detached bundle"):
        module._assert_resolved_bundle_names(  # noqa: SLF001 - versioned path contract.
            v5_launcher_bundle_name="evaluation_speed_v7_v001",
            detached_bundle_name="evaluation_speed_v7_detached",
        )

    module._assert_resolved_bundle_names(  # noqa: SLF001 - finalized bundle contract.
        v5_launcher_bundle_name="evaluation_speed_v7_v001",
        detached_bundle_name="evaluation_speed_v7_detached_v001",
    )


def _write_launcher(path: Path, content: bytes = b"print('launcher')\n") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_detached(path: Path, content: bytes = b"# detached\n") -> str:
    return _write_launcher(path, content)


def _write_python(path: Path, content: bytes = b"python-runtime\n") -> str:
    return _write_launcher(path, content)


def test_create_only_json_is_complete_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    destination = tmp_path / "evidence.json"
    payload = {"schema_version": "test/v1", "value": "complete"}
    observed: list[dict[str, object]] = []
    original_link = module.os.link

    def observe_atomic_publish(source: Path, target: Path) -> None:
        assert Path(target) == destination
        assert not destination.exists()
        observed.append(json.loads(Path(source).read_text(encoding="utf-8")))
        original_link(source, target)

    monkeypatch.setattr(module.os, "link", observe_atomic_publish)

    module._write_json_create_only(destination, payload)  # noqa: SLF001

    assert observed == [payload]
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(".*.tmp")) == []


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


def _unsafe_service_runtime(module: ModuleType) -> Any:  # noqa: ANN401
    snapshot = _safe_runtime(module)
    return module.RuntimeSnapshot(
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


def _successful_preflight(
    command: tuple[str, ...],
    **_kwargs: object,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=json.dumps({"passed": True, "launch_performed": False}),
        stderr="",
    )


def _successful_queue(
    _command: tuple[str, ...],
    **kwargs: object,
) -> SimpleNamespace:
    output = cast("Any", kwargs["stdout"])
    output.write(b'{"succeeded":["all"],"skipped":[],"reused":[],"failed":[]}\n')
    output.flush()
    return SimpleNamespace(wait=lambda: 0)


def _launch_fixture(
    module: ModuleType,
    tmp_path: Path,
    *,
    detached_script: Path = SCRIPT_PATH,
) -> tuple[Path, Path, Path, str, dict[str, object]]:
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()
    status_path = output_root / "status.jsonl"

    def registered_worker(command: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        launch_dir = Path(command[command.index("--launch-dir") + 1])
        token = command[command.index("--token") + 1]
        module._update_lock(  # noqa: SLF001 - real-worker registration contract.
            launch_dir / "supervisor.lock",
            token=token,
            supervisor_pid=SUPERVISOR_PID,
        )
        return SimpleNamespace(pid=SPAWN_PID)

    result = module.launch_detached(
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        python_path=tmp_path / "python.exe",
        detached_script=detached_script,
        upstream_root=tmp_path,
        output_root=output_root,
        status_path=status_path,
        probe=lambda: _safe_runtime(module),
        runner=_successful_preflight,
        popen=registered_worker,
    )
    return launcher, output_root, status_path, launcher_sha256, result


def _finish_successful_launch(
    module: ModuleType,
    tmp_path: Path,
) -> tuple[Path, Path, Path, str, dict[str, object]]:
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    status_path.write_text('{"stage":"evaluate","status":"success"}\n', encoding="utf-8")
    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=handoff["launch_token"],
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        python_path=tmp_path / "python.exe",
        upstream_root=tmp_path,
        output_root=output_root,
        status_path=status_path,
        probe=lambda: _safe_runtime(module),
        popen=_successful_queue,
        current_pid=SUPERVISOR_PID,
    )
    assert exit_code == 0
    return launcher, output_root, status_path, launcher_sha256, result


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


@pytest.mark.parametrize(
    "command_line",
    [
        (
            'C:\\Python\\python.exe "C:\\work\\scripts\\'
            'launch_600m_speaker_evaluation_queue_speed_v2.py" launch'
        ),
        (
            'C:\\Python\\python.exe "C:\\work\\scripts\\'
            'launch_600m_speaker_evaluation_queue_speed_v3.py" preflight'
        ),
        "python launch_600m_speaker_evaluation_queue_speed_v4.py launch",
        "python launch_600m_speaker_evaluation_queue_speed_v5.py launch",
        "python launch_600m_speaker_evaluation_queue_speed_v6_detached.py supervise",
        "python launch_600m_speaker_evaluation_queue_speed_v7_detached.py supervise",
    ],
)
def test_legacy_evaluation_launchers_are_classified_as_unsafe(command_line: str) -> None:
    module = _load_script()

    assert module._is_evaluation_command(command_line) is True  # noqa: SLF001


def test_legacy_remote_server_entrypoint_is_classified_as_unsafe() -> None:
    module = _load_script()

    assert (
        module._is_service_command(  # noqa: SLF001 - service safety contract.
            r'C:\Python\python.exe "C:\work\remote_server.py" --port 8000'
        )
        is True
    )


def test_similarly_named_files_do_not_trigger_process_classification() -> None:
    module = _load_script()

    assert (
        module._is_evaluation_command(  # noqa: SLF001 - semantic boundary contract.
            "python relaunch_600m_speaker_evaluation_queue_speed_v2.py launch"
        )
        is False
    )
    assert module._is_service_command("python not_remote_server.py") is False  # noqa: SLF001


def test_parent_handoff_accepts_a_distinct_venv_shim_spawn_pid(tmp_path: Path) -> None:
    module = _load_script()
    _, _, _, _, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text(encoding="utf-8"))

    handoff = module._wait_for_parent_handoff(  # noqa: SLF001 - handoff contract.
        launch_dir,
        token=reservation["launch_token"],
        supervisor_pid=SUPERVISOR_PID,
        timeout_seconds=0,
    )

    assert handoff["spawn_pid"] == SPAWN_PID
    assert handoff["supervisor_pid"] == SUPERVISOR_PID
    assert handoff["previous_evidence"] == module._file_binding(  # noqa: SLF001
        launch_dir / "reservation-evidence.json"
    )


def test_supervisor_bootstrap_error_is_persisted_and_lock_archived(tmp_path: Path) -> None:
    module = _load_script()
    _, output_root, _, _, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    reservation_path = launch_dir / "reservation-evidence.json"
    handoff_path = launch_dir / "parent-handoff-evidence.json"
    reservation_before = reservation_path.read_bytes()
    handoff_before = handoff_path.read_bytes()
    handoff = json.loads(handoff_before)
    token = cast("str", handoff["launch_token"])

    try:
        raise RuntimeError("bootstrap exploded")  # noqa: EM101, TRY003, TRY301
    except RuntimeError as error:
        module._record_supervisor_bootstrap_error(  # noqa: SLF001
            launch_dir=launch_dir,
            token=token,
            error=error,
        )

    failure = json.loads(
        (launch_dir / "supervisor-bootstrap-error.json").read_text(encoding="utf-8")
    )
    assert reservation_path.read_bytes() == reservation_before
    assert handoff_path.read_bytes() == handoff_before
    assert failure["status"] == "SUPERVISOR_ERROR"
    assert failure["error"] == "RuntimeError: bootstrap exploded"
    assert failure["traceback"].startswith("Traceback")
    assert failure["previous_evidence"] == module._file_binding(handoff_path)  # noqa: SLF001
    assert not (launch_dir / "supervisor.lock").exists()
    final_lock = launch_dir / "supervisor-lock-final.json"
    assert failure["archived_lock"] == module._file_binding(final_lock)  # noqa: SLF001
    assert module.read_status(output_root)["status"] == "SUPERVISOR_ERROR"


def test_bootstrap_error_reuses_a_lock_archived_before_terminal_write(tmp_path: Path) -> None:
    module = _load_script()
    _, output_root, _, _, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    token = cast("str", handoff["launch_token"])
    archived_lock = module._archive_lock(  # noqa: SLF001 - terminal ordering contract.
        launch_dir / "supervisor.lock",
        token=token,
    )

    try:
        raise RuntimeError("terminal write exploded")  # noqa: EM101, TRY003, TRY301
    except RuntimeError as error:
        module._record_supervisor_bootstrap_error(  # noqa: SLF001
            launch_dir=launch_dir,
            token=token,
            error=error,
        )

    failure = json.loads(
        (launch_dir / "supervisor-bootstrap-error.json").read_text(encoding="utf-8")
    )
    assert failure["archived_lock"] == archived_lock
    assert module.read_status(output_root)["status"] == "SUPERVISOR_ERROR"


def test_worker_timeout_before_parent_handoff_records_terminal_bootstrap_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    reservation_path = launch_dir / "reservation-evidence.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    (launch_dir / "parent-handoff-evidence.json").unlink()
    monotonic_values = iter((0.0, 31.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))

    try:
        module.run_supervisor(
            launch_dir=launch_dir,
            token=reservation["launch_token"],
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=status_path,
            probe=lambda: _safe_runtime(module),
            popen=lambda *_args, **_kwargs: pytest.fail("queue must not start"),
            current_pid=SUPERVISOR_PID,
        )
    except RuntimeError as error:
        if "parent handoff did not complete" not in str(error):
            pytest.fail(f"unexpected worker timeout error: {error}")
        module._record_supervisor_bootstrap_error(  # noqa: SLF001
            launch_dir=launch_dir,
            token=reservation["launch_token"],
            error=error,
        )
    else:
        pytest.fail("worker timeout must fail closed")

    failure_path = launch_dir / "supervisor-bootstrap-error.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["schema_version"] == module.BOOTSTRAP_PRE_HANDOFF_FAILURE_SCHEMA
    assert failure["previous_evidence"] == module._file_binding(reservation_path)  # noqa: SLF001
    assert failure["supervisor_pid"] == SUPERVISOR_PID
    assert not (launch_dir / "supervisor.lock").exists()
    assert (launch_dir / "supervisor-lock-final.json").is_file()
    assert module.read_status(output_root)["status"] == "SUPERVISOR_ERROR"

    report = module.preflight_detached(
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        detached_script=SCRIPT_PATH,
        python_path=tmp_path / "python.exe",
        output_root=output_root,
        status_path=status_path,
        probe=lambda: _safe_runtime(module),
        runner=_successful_preflight,
    )
    assert report["passed"] is True
    assert module.reserve_launch_directory(output_root / "launches").name == "launch-v002"


def test_status_rejects_spawn_pid_in_pre_handoff_bootstrap_phase(tmp_path: Path) -> None:
    module = _load_script()
    _, output_root, _, _, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    reservation_path = launch_dir / "reservation-evidence.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    (launch_dir / "parent-handoff-evidence.json").unlink()
    archived_lock = module._archive_lock(  # noqa: SLF001
        launch_dir / "supervisor.lock",
        token=reservation["launch_token"],
    )
    module._write_json_create_only(  # noqa: SLF001
        launch_dir / "supervisor-bootstrap-error.json",
        {
            "schema_version": module.BOOTSTRAP_PRE_HANDOFF_FAILURE_SCHEMA,
            "launch_token": reservation["launch_token"],
            "status": "SUPERVISOR_ERROR",
            "failed_at": "2026-08-03T00:00:00+00:00",
            "spawn_pid": SPAWN_PID,
            "supervisor_pid": SUPERVISOR_PID,
            "detached_launcher": reservation["detached_launcher"],
            "v5_launcher": reservation["v5_launcher"],
            "python": reservation["python"],
            "output_identity": reservation["output_identity"],
            "previous_evidence": module._file_binding(reservation_path),  # noqa: SLF001
            "archived_lock": archived_lock,
            "error": "RuntimeError: parent disappeared",
            "traceback": "Traceback",
        },
    )

    with pytest.raises(RuntimeError, match="pre-handoff bootstrap spawn PID"):
        module.read_status(output_root)


def test_preflight_fails_closed_before_invoking_unpinned_v5(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    _write_detached(detached)
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="v5 launcher SHA-256 mismatch"):
        module.preflight_detached(
            launcher_path=launcher,
            expected_launcher_sha256="0" * 64,
            detached_script=detached,
            python_path=tmp_path / "python.exe",
            output_root=tmp_path / "evaluation_speed_v7",
            status_path=tmp_path / "evaluation_speed_v7/status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=lambda command, **_kwargs: calls.append(command),
        )

    assert calls == []


def test_preflight_invokes_pinned_v5_without_shell_and_binds_report(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    detached_sha256 = _write_detached(detached)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
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
        detached_script=detached,
        python_path=tmp_path / "python.exe",
        output_root=output_root,
        status_path=output_root / "status.jsonl",
        probe=lambda: _safe_runtime(module),
        runner=runner,
    )

    assert calls[0][0] == (str(tmp_path / "python.exe"), str(launcher), "preflight")
    assert calls[0][1]["shell"] is False
    assert report["v5_launcher"] == {
        "path": str(launcher.resolve()),
        "sha256": launcher_sha256,
    }
    assert report["detached_launcher"] == {
        "path": str(detached.resolve()),
        "sha256": detached_sha256,
    }
    assert report["python"] == {
        "path": str((tmp_path / "python.exe").resolve()),
        "sha256": module.sha256_file(tmp_path / "python.exe"),
    }
    assert report["output_identity"] == {
        "schema_version": module.OUTPUT_IDENTITY_SCHEMA,
        "output_root": str(output_root),
        "status_path": str(output_root / "status.jsonl"),
        "queue_lock_path": str(output_root / "status.jsonl.lock"),
        "launches_root": str(output_root / "launches"),
    }
    assert report["v5_preflight"]["passed"] is True
    assert report["runtime"]["gpu_free_mib"] == pytest.approx(11_776.0)


@pytest.mark.parametrize("aliased_path", ["output_root", "status_parent"])
def test_preflight_rejects_initial_output_path_aliases(
    tmp_path: Path,
    aliased_path: str,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    _write_detached(detached)
    python_path = tmp_path / "python.exe"
    _write_python(python_path)
    output_root = tmp_path / "evaluation_speed_v7"
    if aliased_path == "output_root":
        real_root = tmp_path / "real-output"
        real_root.mkdir()
        output_root.symlink_to(real_root, target_is_directory=True)
        status_path = output_root / "status.jsonl"
    else:
        output_root.mkdir()
        real_status_parent = output_root / "real-status"
        real_status_parent.mkdir()
        status_parent = output_root / "status-link"
        status_parent.symlink_to(real_status_parent, target_is_directory=True)
        status_path = status_parent / "status.jsonl"
    calls: list[object] = []

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return _successful_preflight(command)

    with pytest.raises(ValueError, match="symlink, junction, or reparse alias"):
        module.preflight_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            detached_script=detached,
            python_path=python_path,
            output_root=output_root,
            status_path=status_path,
            probe=lambda: _safe_runtime(module),
            runner=runner,
        )

    assert calls == []


def test_preflight_rejects_alias_hidden_by_parent_traversal(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    _write_detached(detached)
    python_path = tmp_path / "python.exe"
    _write_python(python_path)
    scope = tmp_path / "scope"
    scope.mkdir()
    traversal_target = tmp_path / "traversal-target"
    traversal_target.mkdir()
    (scope / "alias").symlink_to(traversal_target, target_is_directory=True)
    output_root = scope / "alias" / ".." / "evaluation_speed_v7"
    (tmp_path / "evaluation_speed_v7").mkdir()

    with pytest.raises(ValueError, match="symlink, junction, or reparse alias"):
        module.preflight_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            detached_script=detached,
            python_path=python_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=_successful_preflight,
        )


def test_launch_rejects_output_alias_swapped_in_by_v5_preflight(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    _write_detached(detached)
    python_path = tmp_path / "python.exe"
    _write_python(python_path)
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()
    replacement = tmp_path / "replacement-output"
    replacement.mkdir()
    popen_calls: list[object] = []

    def swap_during_preflight(
        command: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        output_root.rmdir()
        output_root.symlink_to(replacement, target_is_directory=True)
        return _successful_preflight(command)

    with pytest.raises(ValueError, match="symlink, junction, or reparse alias"):
        module.launch_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=python_path,
            detached_script=detached,
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=swap_during_preflight,
            popen=lambda *_args, **_kwargs: popen_calls.append(True),
        )

    assert popen_calls == []
    assert not (replacement / "launches").exists()


def test_preflight_rejects_windows_reparse_output_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    _write_detached(detached)
    python_path = tmp_path / "python.exe"
    _write_python(python_path)
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()
    original_lstat = Path.lstat

    class ReparseStat:
        st_file_attributes = 0x400

        def __init__(self, original: object) -> None:
            self.original = original

        def __getattr__(self, name: str) -> object:
            return getattr(self.original, name)

    def simulated_lstat(path: Path) -> object:
        metadata = original_lstat(path)
        if Path(path) == output_root:
            return ReparseStat(metadata)
        return metadata

    monkeypatch.setattr(Path, "lstat", simulated_lstat)

    with pytest.raises(ValueError, match="symlink, junction, or reparse alias"):
        module.preflight_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            detached_script=detached,
            python_path=python_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=_successful_preflight,
        )


@pytest.mark.parametrize(
    "blocked_binding",
    ["v5_launcher", "detached_launcher", "python"],
)
def test_preflight_rejects_windows_reparse_bundle_ancestors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_binding: str,
) -> None:
    module = _load_script()
    launcher = tmp_path / "bundles/v5/launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "bundles/detached/detached.py"
    _write_detached(detached)
    python_path = tmp_path / "runtime/python.exe"
    _write_python(python_path)
    blocked_ancestor = {
        "v5_launcher": launcher.parent,
        "detached_launcher": detached.parent,
        "python": python_path.parent,
    }[blocked_binding]
    original_lstat = Path.lstat

    class ReparseStat:
        st_file_attributes = 0x400

        def __init__(self, original: object) -> None:
            self.original = original

        def __getattr__(self, name: str) -> object:
            return getattr(self.original, name)

    def simulated_lstat(path: Path) -> object:
        metadata = original_lstat(path)
        if Path(path) == blocked_ancestor:
            return ReparseStat(metadata)
        return metadata

    monkeypatch.setattr(Path, "lstat", simulated_lstat)

    with pytest.raises(ValueError, match="symlink, junction, or reparse alias"):
        module.preflight_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            detached_script=detached,
            python_path=python_path,
            output_root=tmp_path / "evaluation_speed_v7",
            status_path=tmp_path / "evaluation_speed_v7/status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=_successful_preflight,
        )


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
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    _write_detached(detached)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
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
            detached_script=detached,
            python_path=tmp_path / "python.exe",
            output_root=output_root,
            status_path=status_path,
            probe=lambda: snapshot,
            runner=lambda *_args, **_kwargs: calls.append(True),
        )

    assert calls == []


def test_launch_reprobes_runtime_after_v5_preflight_before_reservation(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached = tmp_path / "detached.py"
    _write_detached(detached)
    python_path = tmp_path / "python.exe"
    _write_python(python_path)
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()
    snapshots = iter((_safe_runtime(module), _unsafe_service_runtime(module)))
    popen_calls: list[object] = []

    with pytest.raises(RuntimeError, match="TTS service must be stopped"):
        module.launch_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=python_path,
            detached_script=detached,
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: next(snapshots),
            runner=_successful_preflight,
            popen=lambda *_args, **_kwargs: popen_calls.append(True),
        )

    assert popen_calls == []
    assert not (output_root / "launches").exists()


def test_reserve_launch_directory_is_versioned_create_only(tmp_path: Path) -> None:
    module = _load_script()
    launches = tmp_path / "evaluation_speed_v7/launches"

    first = module.reserve_launch_directory(launches)
    second = module.reserve_launch_directory(launches)

    assert first.name == "launch-v001"
    assert second.name == "launch-v002"
    with pytest.raises(FileExistsError, match="refusing to reuse launch directory"):
        module.reserve_launch_directory(launches, launch_id="launch-v002")


def test_launch_spawns_detached_supervisor_with_immutable_phase_evidence(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached_script = tmp_path / "detached.py"
    _write_detached(detached_script)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
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
        launch_dir = Path(command[command.index("--launch-dir") + 1])
        token = command[command.index("--token") + 1]
        module._update_lock(  # noqa: SLF001 - simulate child worker registration.
            launch_dir / "supervisor.lock",
            token=token,
            supervisor_pid=SUPERVISOR_PID,
        )
        return SimpleNamespace(pid=SPAWN_PID)

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
    reservation_path = launch_dir / "reservation-evidence.json"
    handoff_path = launch_dir / "parent-handoff-evidence.json"
    reservation_bytes = reservation_path.read_bytes()
    reservation = json.loads(reservation_bytes)
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    lock = json.loads((launch_dir / "supervisor.lock").read_text(encoding="utf-8"))
    assert len(run_calls) == 1
    assert popen_calls[0][0] == (
        str(tmp_path / "python.exe"),
        str(detached_script.resolve()),
        "supervise",
        "--launch-dir",
        str(launch_dir.resolve()),
        "--token",
        reservation["launch_token"],
    )
    assert popen_calls[0][1]["shell"] is False
    assert popen_calls[0][1]["creationflags"] == module.WINDOWS_DETACHED_FLAGS
    assert reservation["status"] == "RESERVED"
    assert reservation["output_identity"] == reservation["preflight"]["output_identity"]
    assert reservation["python"] == {
        "path": str((tmp_path / "python.exe").resolve()),
        "sha256": module.sha256_file(tmp_path / "python.exe"),
    }
    assert handoff["status"] == "DETACHED"
    assert handoff["output_identity"] == reservation["output_identity"]
    assert handoff["python"] == reservation["python"]
    assert handoff["spawn_pid"] == SPAWN_PID
    assert handoff["supervisor_pid"] == SUPERVISOR_PID
    assert handoff["previous_evidence"] == module._file_binding(reservation_path)  # noqa: SLF001
    assert lock["supervisor_pid"] == SUPERVISOR_PID
    assert reservation_path.read_bytes() == reservation_bytes
    with pytest.raises(FileExistsError):
        module._write_json_create_only(reservation_path, reservation)  # noqa: SLF001
    assert not (launch_dir / "launch-evidence.json").exists()
    assert not (launch_dir / "queue.log").exists()


def test_spawn_failure_preserves_evidence_and_archives_active_lock(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached_script = tmp_path / "detached.py"
    _write_detached(detached_script)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()
    reservation_during_spawn: list[bytes] = []

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"passed": True, "launch_performed": False}),
            stderr="",
        )

    def popen(_command: tuple[str, ...], **_kwargs: object) -> None:
        reservation_during_spawn.append(
            (output_root / "launches/launch-v001/reservation-evidence.json").read_bytes()
        )
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
    reservation_path = launch_dir / "reservation-evidence.json"
    failure_path = launch_dir / "spawn-failure-evidence.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert reservation_path.read_bytes() == reservation_during_spawn[0]
    assert failure["status"] == "SPAWN_FAILED"
    assert "spawn denied" in failure["error"]
    assert failure["previous_evidence"] == module._file_binding(reservation_path)  # noqa: SLF001
    assert not (launch_dir / "supervisor.lock").exists()
    final_lock = launch_dir / "supervisor-lock-final.json"
    assert failure["archived_lock"] == module._file_binding(final_lock)  # noqa: SLF001
    assert module.read_status(output_root)["status"] == "SPAWN_FAILED"


def test_parent_handoff_failure_records_the_known_spawn_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached_script = tmp_path / "detached.py"
    _write_detached(detached_script)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()
    original_write = module._write_json_create_only  # noqa: SLF001

    def fail_handoff(path: Path, payload: dict[str, object]) -> None:
        if path.name == "parent-handoff-evidence.json":
            message = "handoff denied"
            raise OSError(message)
        original_write(path, payload)

    monkeypatch.setattr(module, "_write_json_create_only", fail_handoff)

    def registered_worker(command: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        launch_dir = Path(command[command.index("--launch-dir") + 1])
        token = command[command.index("--token") + 1]
        module._update_lock(  # noqa: SLF001 - simulate child worker registration.
            launch_dir / "supervisor.lock",
            token=token,
            supervisor_pid=SUPERVISOR_PID,
        )
        return SimpleNamespace(pid=SPAWN_PID)

    with pytest.raises(OSError, match="handoff denied"):
        module.launch_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            detached_script=detached_script,
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=_successful_preflight,
            popen=registered_worker,
        )

    failure = json.loads(
        (output_root / "launches/launch-v001/spawn-failure-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["spawn_pid"] == SPAWN_PID
    assert failure["supervisor_pid"] == SUPERVISOR_PID


def test_parent_fails_closed_when_the_real_worker_does_not_register(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    detached_script = tmp_path / "detached.py"
    _write_detached(detached_script)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()

    with pytest.raises(RuntimeError, match="real supervisor worker did not register"):
        module.launch_detached(
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            detached_script=detached_script,
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=output_root / "status.jsonl",
            probe=lambda: _safe_runtime(module),
            runner=_successful_preflight,
            popen=lambda *_args, **_kwargs: SimpleNamespace(pid=SPAWN_PID),
            worker_registration_timeout_seconds=0,
        )

    failure = json.loads(
        (output_root / "launches/launch-v001/spawn-failure-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["spawn_pid"] == SPAWN_PID
    assert failure["supervisor_pid"] is None


def test_supervisor_captures_queue_and_finalizes_hashes_and_lock(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    reservation_path = launch_dir / "reservation-evidence.json"
    handoff_path = launch_dir / "parent-handoff-evidence.json"
    reservation_before = reservation_path.read_bytes()
    reservation = json.loads(reservation_before)
    handoff_before = handoff_path.read_bytes()
    token = cast("str", json.loads(handoff_before)["launch_token"])
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

    start_path = launch_dir / "supervisor-start-evidence.json"
    terminal_path = launch_dir / "terminal-final-evidence.json"
    start = json.loads(start_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert calls[0][0] == (str(tmp_path / "python.exe"), str(launcher), "launch")
    assert calls[0][1]["shell"] is False
    assert reservation_path.read_bytes() == reservation_before
    assert handoff_path.read_bytes() == handoff_before
    assert start["status"] == "RUNNING"
    assert start["spawn_pid"] == SPAWN_PID
    assert start["supervisor_pid"] == SUPERVISOR_PID
    assert start["python"] == reservation["python"]
    assert start["output_identity"] == reservation["output_identity"]
    assert start["runtime_before_queue"]["gpu_free_mib"] == pytest.approx(11_776.0)
    assert start["previous_evidence"] == module._file_binding(handoff_path)  # noqa: SLF001
    assert terminal["status"] == "SUCCEEDED"
    assert terminal["spawn_pid"] == SPAWN_PID
    assert terminal["supervisor_pid"] == SUPERVISOR_PID
    assert terminal["python"] == reservation["python"]
    assert terminal["output_identity"] == reservation["output_identity"]
    assert terminal["v5_launcher"] == {
        "path": str(launcher.resolve()),
        "sha256": launcher_sha256,
    }
    assert terminal["previous_evidence"] == module._file_binding(start_path)  # noqa: SLF001
    assert terminal["queue_exit_code"] == 0
    assert terminal["queue_summary"]["failed"] == []
    assert terminal["queue_log"]["sha256"] == module.sha256_file(launch_dir / "queue.log")
    assert terminal["queue_status"]["sha256"] == module.sha256_file(status_path)
    assert terminal["queue_status"]["row_count"] == 1
    assert terminal["runtime_after"]["process_residue"] == []
    assert terminal["queue_status_lock"] == {
        "path": str(status_path.with_suffix(".jsonl.lock")),
        "present_after_queue": False,
        "present_before_terminal": False,
    }
    assert not (launch_dir / "supervisor.lock").exists()
    final_lock = launch_dir / "supervisor-lock-final.json"
    assert terminal["archived_lock"] == module._file_binding(final_lock)  # noqa: SLF001
    assert module.read_status(output_root)["status"] == "SUCCEEDED"


def test_supervisor_fails_closed_when_queue_status_binding_is_absent(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))

    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=handoff["launch_token"],
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        python_path=tmp_path / "python.exe",
        upstream_root=tmp_path,
        output_root=output_root,
        status_path=status_path,
        probe=lambda: _safe_runtime(module),
        popen=_successful_queue,
        current_pid=SUPERVISOR_PID,
    )

    terminal = json.loads((launch_dir / "terminal-final-evidence.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert terminal["status"] == "FAILED"
    assert terminal["queue_status"] is None


def test_supervisor_reprobes_runtime_immediately_before_queue_launch(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    popen_calls: list[object] = []

    with pytest.raises(RuntimeError, match="TTS service must be stopped"):
        module.run_supervisor(
            launch_dir=launch_dir,
            token=handoff["launch_token"],
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=status_path,
            probe=lambda: _unsafe_service_runtime(module),
            popen=lambda *_args, **_kwargs: popen_calls.append(True),
            current_pid=SUPERVISOR_PID,
        )

    assert popen_calls == []


def test_supervisor_rejects_output_alias_swapped_before_queue_launch(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text(encoding="utf-8"))
    moved_root = tmp_path / "moved-evaluation-speed-v7"
    output_root.rename(moved_root)
    output_root.symlink_to(moved_root, target_is_directory=True)
    popen_calls: list[object] = []

    with pytest.raises(ValueError, match="symlink, junction, or reparse alias"):
        module.run_supervisor(
            launch_dir=launch_dir,
            token=reservation["launch_token"],
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=status_path,
            probe=lambda: _safe_runtime(module),
            popen=lambda *_args, **_kwargs: popen_calls.append(True),
            current_pid=SUPERVISOR_PID,
        )

    assert popen_calls == []


def test_supervisor_rejects_output_alias_swapped_before_terminal_publish(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    status_path.write_text('{"stage":"evaluate","status":"success"}\n', encoding="utf-8")
    moved_root = tmp_path / "moved-evaluation-speed-v7"

    class Process:
        @staticmethod
        def wait() -> int:
            output_root.rename(moved_root)
            output_root.symlink_to(moved_root, target_is_directory=True)
            return 0

    def popen(_command: tuple[str, ...], **kwargs: object) -> Process:
        output = cast("Any", kwargs["stdout"])
        output.write(b'{"succeeded":["all"],"skipped":[],"reused":[],"failed":[]}\n')
        output.flush()
        return Process()

    with pytest.raises(ValueError, match="symlink, junction, or reparse alias"):
        module.run_supervisor(
            launch_dir=launch_dir,
            token=handoff["launch_token"],
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

    moved_launch = moved_root / "launches" / launch_dir.name
    assert not (moved_launch / "terminal-final-evidence.json").exists()


def test_supervisor_rejects_python_binding_changed_after_reservation(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    (tmp_path / "python.exe").write_bytes(b"changed-python-runtime\n")
    popen_calls: list[object] = []

    with pytest.raises(RuntimeError, match="Python executable binding mismatch"):
        module.run_supervisor(
            launch_dir=launch_dir,
            token=handoff["launch_token"],
            launcher_path=launcher,
            expected_launcher_sha256=launcher_sha256,
            python_path=tmp_path / "python.exe",
            upstream_root=tmp_path,
            output_root=output_root,
            status_path=status_path,
            probe=lambda: _safe_runtime(module),
            popen=lambda *_args, **_kwargs: popen_calls.append(True),
            current_pid=SUPERVISOR_PID,
        )

    assert popen_calls == []


def test_supervisor_fails_terminal_when_python_changes_during_queue(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    status_path.write_text('{"stage":"evaluate","status":"success"}\n', encoding="utf-8")

    class Process:
        @staticmethod
        def wait() -> int:
            (tmp_path / "python.exe").write_bytes(b"changed-during-queue\n")
            return 0

    def popen(_command: tuple[str, ...], **kwargs: object) -> Process:
        output = cast("Any", kwargs["stdout"])
        output.write(b'{"succeeded":["all"],"skipped":[],"reused":[],"failed":[]}\n')
        output.flush()
        return Process()

    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=handoff["launch_token"],
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

    terminal = json.loads((launch_dir / "terminal-final-evidence.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert terminal["status"] == "FAILED"
    assert "Python executable binding mismatch" in terminal["error"]


def test_supervisor_fails_terminal_when_queue_status_lock_remains(
    tmp_path: Path,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    status_path.write_text('{"stage":"evaluate","status":"success"}\n', encoding="utf-8")
    queue_lock = status_path.with_suffix(".jsonl.lock")

    class Process:
        @staticmethod
        def wait() -> int:
            queue_lock.write_text("residue", encoding="utf-8")
            return 0

    def popen(_command: tuple[str, ...], **kwargs: object) -> Process:
        output = cast("Any", kwargs["stdout"])
        output.write(b'{"succeeded":["all"],"skipped":[],"reused":[],"failed":[]}\n')
        output.flush()
        return Process()

    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=handoff["launch_token"],
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

    terminal = json.loads((launch_dir / "terminal-final-evidence.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert terminal["status"] == "FAILED"
    assert terminal["queue_status_lock"]["present_after_queue"] is True
    assert terminal["queue_status_lock"]["present_before_terminal"] is True


def test_supervisor_fails_terminal_when_queue_status_lock_reappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    status_path.write_text('{"stage":"evaluate","status":"success"}\n', encoding="utf-8")
    queue_lock = status_path.with_suffix(".jsonl.lock")
    original_lexists = module.os.path.lexists
    observations = iter((False, False, True))

    def controlled_lexists(path: Path) -> bool:
        if Path(path) == queue_lock:
            return next(observations)
        return bool(original_lexists(path))

    monkeypatch.setattr(module.os.path, "lexists", controlled_lexists)

    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=handoff["launch_token"],
        launcher_path=launcher,
        expected_launcher_sha256=launcher_sha256,
        python_path=tmp_path / "python.exe",
        upstream_root=tmp_path,
        output_root=output_root,
        status_path=status_path,
        probe=lambda: _safe_runtime(module),
        popen=_successful_queue,
        current_pid=SUPERVISOR_PID,
    )

    terminal = json.loads((launch_dir / "terminal-final-evidence.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert terminal["status"] == "FAILED"
    assert terminal["queue_status_lock"]["present_after_queue"] is False
    assert terminal["queue_status_lock"]["present_before_terminal"] is True


def test_supervisor_rejects_changed_detached_launcher_binding(tmp_path: Path) -> None:
    module = _load_script()
    detached_script = tmp_path / "detached.py"
    _write_detached(detached_script)
    launcher, output_root, status_path, launcher_sha256, result = _launch_fixture(
        module,
        tmp_path,
        detached_script=detached_script,
    )
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff = json.loads((launch_dir / "parent-handoff-evidence.json").read_text(encoding="utf-8"))
    token = cast("str", handoff["launch_token"])
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
            status_path=status_path,
            probe=lambda: _safe_runtime(module),
            popen=lambda *_args, **_kwargs: calls.append(True),
            current_pid=SUPERVISOR_PID,
        )

    assert calls == []


def test_status_rejects_a_broken_immutable_evidence_chain(tmp_path: Path) -> None:
    module = _load_script()
    _, output_root, _, _, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    reservation_path = launch_dir / "reservation-evidence.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation["created_at"] = "tampered"
    reservation_path.write_text(json.dumps(reservation), encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable evidence chain"):
        module.read_status(output_root)


def test_status_rejects_an_invalid_phase_status(tmp_path: Path) -> None:
    module = _load_script()
    _, output_root, _, _, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    handoff_path = launch_dir / "parent-handoff-evidence.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["status"] = "RUNNING"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid phase status"):
        module.read_status(output_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "wrong-lock-schema"),
        ("launch_token", "wrong-lock-token"),
        ("supervisor_pid", SUPERVISOR_PID + 1),
    ],
)
def test_status_rejects_tampered_active_lock_payload(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    module = _load_script()
    _, output_root, _, _, result = _launch_fixture(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    lock_path = launch_dir / "supervisor.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock[field] = value
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(RuntimeError, match="active lock"):
        module.read_status(output_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("launch_token", "wrong-final-token"),
        ("supervisor_pid", SUPERVISOR_PID + 1),
    ],
)
def test_status_rejects_tampered_archived_lock_payload_even_if_rebound(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    module = _load_script()
    _, output_root, _, _, result = _finish_successful_launch(module, tmp_path)
    launch_dir = Path(cast("str", result["launch_dir"]))
    lock_path = launch_dir / "supervisor-lock-final.json"
    terminal_path = launch_dir / "terminal-final-evidence.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock[field] = value
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["archived_lock"] = module._file_binding(lock_path)  # noqa: SLF001
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    with pytest.raises(RuntimeError, match="archived lock"):
        module.read_status(output_root)


def test_status_selects_latest_launch_by_numeric_version(tmp_path: Path) -> None:
    module = _load_script()
    launcher = tmp_path / "launch_v5.py"
    launcher_sha256 = _write_launcher(launcher)
    _write_python(tmp_path / "python.exe")
    output_root = tmp_path / "evaluation_speed_v7"
    output_root.mkdir()

    for launch_id, message in (
        ("launch-v999", "older failure"),
        ("launch-v1000", "newer failure"),
    ):

        def fail_spawn(
            _command: tuple[str, ...],
            failure_message: str = message,
            **_kwargs: object,
        ) -> None:
            raise OSError(failure_message)

        with pytest.raises(OSError, match=message):
            module.launch_detached(
                launcher_path=launcher,
                expected_launcher_sha256=launcher_sha256,
                python_path=tmp_path / "python.exe",
                detached_script=SCRIPT_PATH,
                upstream_root=tmp_path,
                output_root=output_root,
                status_path=output_root / "status.jsonl",
                probe=lambda: _safe_runtime(module),
                runner=_successful_preflight,
                popen=fail_spawn,
                launch_id=launch_id,
            )

    status = module.read_status(output_root)
    assert Path(cast("str", status["launch_dir"])).name == "launch-v1000"
    assert status["evidence"]["error"] == "OSError: newer failure"
