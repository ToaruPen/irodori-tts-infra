from __future__ import annotations

from typing import TYPE_CHECKING

from irodori_tts_infra.config.settings import LOOPBACK_HOSTS, ServerSettings
from irodori_tts_infra.deploy.remote._common import (
    _load_env_script,
    _powershell,
    _ps_quote,
    _run,
)
from irodori_tts_infra.deploy.remote.sync import (
    resolve_remote_dir,
    resolve_remote_host,
)

if TYPE_CHECKING:
    import subprocess

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
    _run(
        [
            "ssh",
            host,
            _powershell(
                _start_script(directory, server_host=server_host, port=port),
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


def _start_script(remote_dir: str, *, server_host: str | None, port: int | None) -> str:
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        f"{_load_env_script()}"
        f"{_server_bind_script(server_host=server_host, port=port)}"
        "$pidFile = Join-Path (Get-Location) '.uvicorn.pid'; "
        "$runtimePython = Join-Path (Get-Location) '.runtime-venv/Scripts/python.exe'; "
        "if (Test-Path -LiteralPath $pidFile) { "
        "$pid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue; "
        "if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) { "
        'Write-Output "running $pid"; exit 0 } }; '
        "$process = Start-Process -FilePath $runtimePython "
        "-ArgumentList @("
        f"'-m', 'uvicorn', '{_APP_TARGET}', "
        "'--host', $serverHost, '--port', $port"
        ") -PassThru -WindowStyle Hidden; "
        "Set-Content -LiteralPath $pidFile -Value $process.Id; "
        "Write-Output $process.Id"
    )


def _server_bind_script(*, server_host: str | None, port: int | None) -> str:
    default_host = _ps_quote(str(ServerSettings.model_fields["host"].default))
    default_port = _ps_quote(str(ServerSettings.model_fields["port"].default))
    host_expr = (
        _ps_quote(ServerSettings(host=server_host).host)
        if server_host is not None
        else (
            "if ($env:IRODORI_TTS_SERVER_HOST) "
            f"{{ $env:IRODORI_TTS_SERVER_HOST }} else {{ {default_host} }}"
        )
    )
    port_expr = (
        _ps_quote(str(port))
        if port is not None
        else (
            "if ($env:IRODORI_TTS_SERVER_PORT) "
            f"{{ $env:IRODORI_TTS_SERVER_PORT }} else {{ {default_port} }}"
        )
    )
    allowed_hosts = ", ".join(_ps_quote(host) for host in sorted(LOOPBACK_HOSTS))
    return (
        f"$serverHost = {host_expr}; "
        f"if (@({allowed_hosts}) -notcontains $serverHost) "
        "{ throw 'server host must be loopback' }; "
        f"$port = {port_expr}; "
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
