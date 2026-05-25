from __future__ import annotations

import subprocess  # noqa: S404
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from irodori_tts_infra.config import ServerSettings
from irodori_tts_infra.deploy import cli
from irodori_tts_infra.deploy.remote import _common as remote_common  # noqa: PLC2701
from irodori_tts_infra.deploy.remote import bootstrap, service, sync

if TYPE_CHECKING:
    from collections.abc import Sequence


pytestmark = pytest.mark.unit


def make_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "package.py").write_text("", encoding="utf-8")
    (root / "README.md").write_text("# test\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (root / "uv.lock").write_text("# lock\n", encoding="utf-8")
    (root / ".env.example").write_text("IRODORI_REMOTE_HOST=user@host\n", encoding="utf-8")


def record_commands(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> list[tuple[list[str], bool]]:
    commands: list[tuple[list[str], bool]] = []

    def fake_run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append((list(command), check))
        return subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(module, "_run", fake_run)
    return commands


def remote_command(command: list[str]) -> str:
    assert command[:2] == ["ssh", "gpu"]
    return command[2]


def ps_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def test_sync_uses_rsync_with_expected_sources_and_excludes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    make_project(tmp_path)
    monkeypatch.setattr(
        "irodori_tts_infra.deploy.remote.sync.shutil.which",
        lambda name: "/usr/bin/rsync" if name == "rsync" else None,
    )
    commands = record_commands(monkeypatch, sync)

    sync.sync_project(remote_host="gpu", remote_dir="C:/irodori", repo_root=tmp_path)

    assert commands == [
        (
            [
                "rsync",
                "-az",
                "--delete",
                "-e",
                "ssh",
                "--exclude",
                ".env",
                "--exclude",
                ".git/",
                "--exclude",
                ".mypy_cache/",
                "--exclude",
                ".pytest_cache/",
                "--exclude",
                ".ruff_cache/",
                "--exclude",
                ".venv/",
                "--exclude",
                "__pycache__/",
                "--exclude",
                "audio/",
                "--exclude",
                "checkpoints/",
                "--exclude",
                "data/",
                "--exclude",
                "models/",
                "--exclude",
                "outputs/",
                "--exclude",
                "runs/",
                str(tmp_path / "src"),
                str(tmp_path / "README.md"),
                str(tmp_path / "pyproject.toml"),
                str(tmp_path / "uv.lock"),
                str(tmp_path / ".env.example"),
                "gpu:C:/irodori/",
            ],
            True,
        ),
    ]


def test_sync_falls_back_to_ssh_mkdir_and_scp_when_rsync_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    make_project(tmp_path)
    monkeypatch.setattr("irodori_tts_infra.deploy.remote.sync.shutil.which", lambda _name: None)
    commands = record_commands(monkeypatch, sync)

    sync.sync_project(remote_host="gpu", remote_dir="C:/irodori", repo_root=tmp_path)

    assert commands[0][0][:2] == ["ssh", "gpu"]
    assert "New-Item" in remote_command(commands[0][0])
    assert "C:/irodori" in remote_command(commands[0][0])
    assert commands[1] == (
        [
            "scp",
            "-r",
            f"{tmp_path / 'src'}",
            str(tmp_path / "README.md"),
            str(tmp_path / "pyproject.toml"),
            str(tmp_path / "uv.lock"),
            str(tmp_path / ".env.example"),
            "gpu:C:/irodori/",
        ],
        True,
    )


def test_sync_resolves_remote_host_and_dir_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    make_project(tmp_path)
    monkeypatch.setenv("IRODORI_REMOTE_HOST", "env-user@env-host")
    monkeypatch.setenv("IRODORI_DEPLOY_DIR", "D:/apps/irodori")
    monkeypatch.setattr(
        "irodori_tts_infra.deploy.remote.sync.shutil.which",
        lambda name: "/usr/bin/rsync" if name == "rsync" else None,
    )
    commands = record_commands(monkeypatch, sync)

    sync.sync_project(repo_root=tmp_path)

    assert commands[0][0][-1] == "env-user@env-host:D:/apps/irodori/"


def test_sync_project_requires_remote_host_when_environment_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    make_project(tmp_path)
    monkeypatch.delenv("IRODORI_REMOTE_HOST", raising=False)

    with pytest.raises(ValueError, match="remote host is required"):
        sync.sync_project(repo_root=tmp_path)


def test_sync_project_rejects_missing_deploy_source_items(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing deploy source item"):
        sync.sync_project(remote_host="gpu", remote_dir="C:/irodori", repo_root=tmp_path)


def test_resolve_remote_dir_rejects_blank_string() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        sync.resolve_remote_dir("   ")


def test_sync_project_propagates_subprocess_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    make_project(tmp_path)
    monkeypatch.setattr(
        "irodori_tts_infra.deploy.remote.sync.shutil.which",
        lambda name: "/usr/bin/rsync" if name == "rsync" else None,
    )

    def fail_run(
        command: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        raise subprocess.CalledProcessError(1, list(command), "out", "err")

    monkeypatch.setattr(sync, "_run", fail_run)

    with pytest.raises(subprocess.CalledProcessError):
        sync.sync_project(remote_host="gpu", remote_dir="C:/irodori", repo_root=tmp_path)


def test_bootstrap_creates_remote_dir_then_runtime_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, bootstrap)

    bootstrap.bootstrap_remote(
        remote_host="gpu",
        remote_dir="C:/irodori",
        irodori_tts_dir="C:/Irodori-TTS",
        python_version="3.11",
        torch_backend_extra="cu128",
    )

    assert "New-Item" in remote_command(commands[0][0])
    assert "C:/irodori" in remote_command(commands[0][0])
    bootstrap_script = remote_command(commands[1][0])
    assert "Set-Location -LiteralPath 'C:/irodori'" in bootstrap_script
    assert "uv.lock is required; run deploy-sync first" in bootstrap_script
    assert "uv venv '.runtime-venv' --python '3.11' --clear" in bootstrap_script
    assert "uv sync --all-extras --locked --python" in bootstrap_script
    assert "Irodori-TTS[cu128] @ file:///C:/Irodori-TTS" in bootstrap_script
    assert "'.[server,irodori]'" not in bootstrap_script
    pip_check = (
        "uv pip check --python $(Join-Path (Get-Location) '.runtime-venv/Scripts/python.exe')"
    )
    assert pip_check in bootstrap_script


def test_bootstrap_file_url_percent_encodes_path_component() -> None:
    assert (
        bootstrap._path_to_file_url("C:/Irodori TTS/repo#1")  # noqa: SLF001
        == "file:///C:/Irodori%20TTS/repo%231"
    )


def test_bootstrap_file_url_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="irodori_tts_dir must be an absolute path"):
        bootstrap._path_to_file_url("vendor/Irodori-TTS")  # noqa: SLF001


def test_bootstrap_resolves_runtime_options_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRODORI_TTS_DIR", "D:/Irodori-TTS")
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_PYTHON", "3.12")
    monkeypatch.setenv("IRODORI_TTS_TORCH_BACKEND_EXTRA", "cu121")
    commands = record_commands(monkeypatch, bootstrap)

    bootstrap.bootstrap_remote(remote_host="gpu", remote_dir="C:/irodori")

    bootstrap_script = remote_command(commands[1][0])
    assert "uv venv '.runtime-venv' --python '3.12' --clear" in bootstrap_script
    assert "Irodori-TTS[cu121] @ file:///D:/Irodori-TTS" in bootstrap_script


def test_start_service_uses_uvicorn_and_pid_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, service)

    service.start_service(remote_host="gpu", remote_dir="C:/irodori", port=9001)

    script = remote_command(commands[0][0])
    assert "Join-Path (Get-Location) '.uvicorn.pid'" in script
    assert "Start-Process -FilePath $runtimePython" in script
    assert "'-m', 'uvicorn', 'irodori_tts_infra.server.main:app'" in script
    assert "$port = '9001'" in script
    assert "'--host', $serverHost, '--port', $port" in script
    assert "Set-Content -LiteralPath $pidFile -Value $process.Id" in script


def test_start_service_uses_remote_env_host_and_port_when_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRODORI_TTS_SERVER_HOST", "127.0.0.9")
    monkeypatch.setenv("IRODORI_TTS_SERVER_PORT", "9999")
    commands = record_commands(monkeypatch, service)

    service.start_service(remote_host="gpu", remote_dir="C:/irodori")

    script = remote_command(commands[0][0])
    assert "$serverHost = if ($env:IRODORI_TTS_SERVER_HOST)" in script
    assert "$port = if ($env:IRODORI_TTS_SERVER_PORT)" in script
    assert f"{{ {ps_string(ServerSettings.model_fields['host'].default)} }}" in script
    assert f"{{ '{ServerSettings.model_fields['port'].default}' }}" in script
    assert "127.0.0.9" not in script
    assert "9999" not in script
    assert "'--host', $serverHost, '--port', $port" in script


def test_start_service_loads_remote_env_before_start_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, service)

    service.start_service(remote_host="gpu", remote_dir="C:/irodori", port=9001)

    script = remote_command(commands[0][0])
    assert "Join-Path (Get-Location) '.env'" in script
    assert "[Environment]::SetEnvironmentVariable($name, $value, 'Process')" in script
    assert script.index("[Environment]::SetEnvironmentVariable") < script.index("Start-Process")


def test_env_example_uses_current_server_settings_prefix() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "IRODORI_TTS_SERVER_PORT=8923" in env_example
    assert "IRODORI_SERVER_PORT" not in env_example


def test_rvc_training_sop_is_marked_superseded() -> None:
    rvc_training = Path("docs/deploy/rvc-training.md").read_text(encoding="utf-8")

    assert "Superseded" in rvc_training
    assert "voice_bank_speakers.toml" in rvc_training
    assert "voice_bank_rvc.toml" not in rvc_training


def test_start_service_quotes_server_host_for_powershell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, service)
    server_host = "127.0.0.1'; Write-Output injected; '"

    service.start_service(
        remote_host="gpu",
        remote_dir="C:/irodori",
        server_host=server_host,
        port=9001,
    )

    script = remote_command(commands[0][0])
    assert "$serverHost = '127.0.0.1''; Write-Output injected; '''" in script
    assert "$port = '9001'" in script
    assert "'--host', $serverHost, '--port', $port" in script


def test_stop_service_reads_pid_stops_process_and_removes_pid_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, service)

    service.stop_service(remote_host="gpu", remote_dir="C:/irodori")

    script = remote_command(commands[0][0])
    assert "Get-Content -LiteralPath $pidFile" in script
    assert "Stop-Process -Id $pid -Force" in script
    assert "Remove-Item -LiteralPath $pidFile" in script


def test_status_service_checks_pid_file_without_raising_for_stopped_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, service)

    service.status_service(remote_host="gpu", remote_dir="C:/irodori")

    script = remote_command(commands[0][0])
    assert commands[0][1] is False
    assert "Test-Path -LiteralPath $pidFile" in script
    assert "Get-Process -Id $pid" in script


def test_shared_run_reraises_timeout_and_logs_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def log_event(event: str, **values: object) -> None:
        events.append((event, values))

    def timeout_run(
        command: Sequence[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, float)
        raise subprocess.TimeoutExpired(cmd=list(command), timeout=timeout)

    monkeypatch.setattr(
        remote_common,
        "_LOGGER",
        SimpleNamespace(info=log_event, warning=log_event),
    )
    monkeypatch.setattr("irodori_tts_infra.deploy.remote._common.subprocess.run", timeout_run)

    with pytest.raises(subprocess.TimeoutExpired):
        remote_common._run(["ssh", "gpu"], timeout=1.25)  # noqa: SLF001

    assert events == [
        ("deploy_remote_command", {"command": ["ssh", "gpu"]}),
        (
            "deploy_remote_command_timeout",
            {"command": ["ssh", "gpu"], "timeout": 1.25},
        ),
    ]


@pytest.mark.parametrize(
    ("command_name", "function_name"),
    [
        ("deploy-sync", "sync_project"),
        ("deploy-bootstrap", "bootstrap_remote"),
        ("deploy-start", "start_service"),
        ("deploy-stop", "stop_service"),
        ("deploy-status", "status_service"),
    ],
)
def test_deploy_cli_exposes_self_contained_commands(
    monkeypatch: pytest.MonkeyPatch,
    command_name: str,
    function_name: str,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_command(**kwargs: object) -> subprocess.CompletedProcess[str] | None:
        remote_host = kwargs.get("remote_host")
        remote_dir = kwargs.get("remote_dir")
        assert remote_host is None or isinstance(remote_host, str)
        assert remote_dir is None or isinstance(remote_dir, str)
        calls.append((function_name, remote_host or "", remote_dir or ""))
        if function_name == "status_service":
            return subprocess.CompletedProcess(["status"], 0, "running 123\n", "")
        return None

    monkeypatch.setattr(cli, function_name, fake_command)

    result = CliRunner().invoke(
        cli.app,
        [command_name, "--remote-host", "gpu", "--remote-dir", "C:/irodori"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(function_name, "gpu", "C:/irodori")]
