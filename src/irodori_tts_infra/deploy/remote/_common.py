from __future__ import annotations

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
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _LOGGER.warning("deploy_remote_command_timeout", command=command, timeout=timeout)
        raise
