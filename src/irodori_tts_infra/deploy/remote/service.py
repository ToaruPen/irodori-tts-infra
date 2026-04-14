from __future__ import annotations

import subprocess  # noqa: S404

import structlog

from irodori_tts_infra.config import ServerSettings
from irodori_tts_infra.deploy.remote.bootstrap import _powershell, _ps_quote
from irodori_tts_infra.deploy.remote.sync import (
    resolve_remote_dir,
    resolve_remote_host,
)

_LOGGER = structlog.get_logger(__name__)
_APP_TARGET = "irodori_tts_infra.server.main:app"


def start_service(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
    server_host: str | None = None,
    port: int | None = None,
) -> None:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    settings = ServerSettings()
    resolved_server_host = server_host or settings.host
    resolved_port = port or settings.port
    _run(
        [
            "ssh",
            host,
            _powershell(
                _start_script(directory, server_host=resolved_server_host, port=resolved_port),
            ),
        ],
    )


def stop_service(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
) -> None:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    _run(["ssh", host, _powershell(_stop_script(directory))])


def status_service(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
) -> subprocess.CompletedProcess[str]:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    return _run(["ssh", host, _powershell(_status_script(directory))], check=False)


def _start_script(remote_dir: str, *, server_host: str, port: int) -> str:
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        "$pidFile = Join-Path (Get-Location) '.uvicorn.pid'; "
        "if (Test-Path -LiteralPath $pidFile) { "
        "$pid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue; "
        "if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) { "
        'Write-Output "running $pid"; exit 0 } }; '
        "$process = Start-Process -FilePath 'uv' "
        "-ArgumentList @("
        f"'run', 'uvicorn', '{_APP_TARGET}', "
        f"'--host', '{server_host}', '--port', '{port}'"
        ") -PassThru -WindowStyle Hidden; "
        "Set-Content -LiteralPath $pidFile -Value $process.Id; "
        "Write-Output $process.Id"
    )


def _stop_script(remote_dir: str) -> str:
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        "$pidFile = Join-Path (Get-Location) '.uvicorn.pid'; "
        "if (!(Test-Path -LiteralPath $pidFile)) { "
        'Write-Output "stopped"; exit 0 }; '
        "$pid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue; "
        "if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) { "
        "Stop-Process -Id $pid -Force }; "
        "Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue; "
        'Write-Output "stopped"'
    )


def _status_script(remote_dir: str) -> str:
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        "$pidFile = Join-Path (Get-Location) '.uvicorn.pid'; "
        "if (!(Test-Path -LiteralPath $pidFile)) { "
        'Write-Output "stopped"; exit 1 }; '
        "$pid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue; "
        "if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) { "
        'Write-Output "running $pid"; exit 0 }; '
        "Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue; "
        'Write-Output "stopped"; exit 1'
    )


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    _LOGGER.info("deploy_remote_command", command=command)
    return subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=check,
        text=True,
    )
