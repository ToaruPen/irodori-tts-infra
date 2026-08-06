from __future__ import annotations

import base64
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
_DETACHED_LAUNCHER = (
    "import subprocess\n"
    "import sys\n"
    "host = sys.argv[2]\n"
    "port = sys.argv[3]\n"
    "with open(sys.argv[4], 'wb') as stdout_log, "
    "open(sys.argv[5], 'wb') as stderr_log:\n"
    "    flags = (subprocess.CREATE_BREAKAWAY_FROM_JOB "
    "| subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)\n"
    "    process = subprocess.Popen(\n"
    "        [sys.executable, '-m', 'uvicorn', "
    f"'{_APP_TARGET}', '--host', host, '--port', port],\n"
    "        stdin=subprocess.DEVNULL,\n"
    "        stdout=stdout_log,\n"
    "        stderr=stderr_log,\n"
    "        close_fds=True,\n"
    "        creationflags=flags,\n"
    "    )\n"
    "print(process.pid)\n"
)
_DETACHED_BOOTSTRAP = "import base64,sys;exec(base64.urlsafe_b64decode(sys.argv[1]))"
_DETACHED_PAYLOAD = base64.urlsafe_b64encode(_DETACHED_LAUNCHER.encode()).decode()


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
        f"{_read_service_process_script()}"
        f"if ({_managed_service_condition()}) {{ "
        'Write-Output "running $servicePid"; exit 0 } }; '
        "$stdoutLog = Join-Path (Get-Location) '.uvicorn.stdout.log'; "
        "$stderrLog = Join-Path (Get-Location) '.uvicorn.stderr.log'; "
        f"$detachedBootstrap = {_ps_quote(_DETACHED_BOOTSTRAP)}; "
        f"$detachedPayload = {_ps_quote(_DETACHED_PAYLOAD)}; "
        "$launcherPid = & $runtimePython -c $detachedBootstrap $detachedPayload "
        "$serverHost $port $stdoutLog $stderrLog; "
        "if ($LASTEXITCODE -ne 0 -or !$launcherPid) "
        "{ throw 'detached uvicorn launcher failed' }; "
        "$serviceProcess = $null; "
        "for ($attempt = 0; $attempt -lt 50 -and !$serviceProcess; $attempt++) { "
        "$serviceProcess = Get-CimInstance Win32_Process "
        '-Filter "ParentProcessId = $launcherPid" -ErrorAction SilentlyContinue '
        f"| Where-Object {{ $_.CommandLine -like '*{_APP_TARGET}*' }} "
        "| Select-Object -First 1; "
        "if (!$serviceProcess) { Start-Sleep -Milliseconds 100 } }; "
        "if (!$serviceProcess) { "
        "$launcherProcess = Get-CimInstance Win32_Process "
        '-Filter "ProcessId = $launcherPid" -ErrorAction SilentlyContinue; '
        f"if ($launcherProcess.CommandLine -like '*{_APP_TARGET}*') "
        "{ $serviceProcess = $launcherProcess } }; "
        "if (!$serviceProcess) { throw 'uvicorn process did not start' }; "
        "$servicePid = $serviceProcess.ProcessId; "
        "Set-Content -LiteralPath $pidFile -Value $servicePid; "
        "Write-Output $servicePid"
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
        f"{_read_service_process_script()}"
        f"if ({_managed_service_condition()}) {{ "
        "Stop-Process -Id $servicePid -Force }; "
        "Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue; "
        'Write-Output "stopped"'
    )


def _status_script(remote_dir: str) -> str:
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        "$pidFile = Join-Path (Get-Location) '.uvicorn.pid'; "
        "if (!(Test-Path -LiteralPath $pidFile)) { "
        'Write-Output "stopped"; exit 1 }; '
        f"{_read_service_process_script()}"
        f"if ({_managed_service_condition()}) {{ "
        'Write-Output "running $servicePid"; exit 0 }; '
        "Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue; "
        'Write-Output "stopped"; exit 1'
    )


def _read_service_process_script() -> str:
    return (
        "$servicePid = Get-Content -LiteralPath $pidFile "
        "-ErrorAction SilentlyContinue; "
        "$serviceProcess = if ($servicePid) { "
        'Get-CimInstance Win32_Process -Filter "ProcessId = $servicePid" '
        "-ErrorAction SilentlyContinue } else { $null }; "
    )


def _managed_service_condition() -> str:
    return f"$serviceProcess -and $serviceProcess.CommandLine -like '*{_APP_TARGET}*'"
