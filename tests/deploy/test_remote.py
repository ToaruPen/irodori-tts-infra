from __future__ import annotations

import subprocess  # noqa: S404
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from irodori_tts_infra.deploy import cli
from irodori_tts_infra.deploy.remote import bootstrap, service, sync

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


pytestmark = pytest.mark.unit


def make_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "package.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
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
                str(tmp_path / "pyproject.toml"),
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
            str(tmp_path / "pyproject.toml"),
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


def test_bootstrap_creates_remote_dir_then_runs_uv_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, bootstrap)

    bootstrap.bootstrap_remote(remote_host="gpu", remote_dir="C:/irodori")

    assert "New-Item" in remote_command(commands[0][0])
    assert "C:/irodori" in remote_command(commands[0][0])
    bootstrap_script = remote_command(commands[1][0])
    assert "Set-Location -LiteralPath 'C:/irodori'" in bootstrap_script
    assert "uv sync --extra server --extra irodori" in bootstrap_script


def test_start_service_uses_uvicorn_and_pid_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_commands(monkeypatch, service)

    service.start_service(remote_host="gpu", remote_dir="C:/irodori", port=9001)

    script = remote_command(commands[0][0])
    assert "Join-Path (Get-Location) '.uvicorn.pid'" in script
    assert "Start-Process -FilePath 'uv'" in script
    assert "'run', 'uvicorn', 'irodori_tts_infra.server.main:app'" in script
    assert "'--host', '0.0.0.0', '--port', '9001'" in script
    assert "Set-Content -LiteralPath $pidFile -Value $process.Id" in script


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

    def fake_command(
        *,
        remote_host: str | None = None,
        remote_dir: str | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
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
