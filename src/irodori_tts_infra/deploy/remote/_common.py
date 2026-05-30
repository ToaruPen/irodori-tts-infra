from __future__ import annotations

import base64
import subprocess  # noqa: S404

import structlog

_LOGGER = structlog.get_logger(__name__)


def _run(
    command: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    _LOGGER.info("deploy_remote_command", command=command)
    try:
        return subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            check=check,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _LOGGER.warning("deploy_remote_command_timeout", command=command, timeout=timeout)
        raise


def _powershell(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _load_env_script() -> str:
    return (
        "$envFile = Join-Path (Get-Location) '.env'; "
        "if (Test-Path -LiteralPath $envFile) { "
        "Get-Content -LiteralPath $envFile | ForEach-Object { "
        "$line = $_.Trim([char]0xFEFF).Trim(); "
        "if ($line -and !$line.StartsWith('#')) { "
        "$separator = $line.IndexOf('='); "
        "if ($separator -gt 0) { "
        "$name = $line.Substring(0, $separator).Trim(); "
        "$value = $line.Substring($separator + 1).Trim().Trim('\"').Trim(\"'\"); "
        "if ($name) { [Environment]::SetEnvironmentVariable($name, $value, 'Process') } "
        "} } } }; "
    )
