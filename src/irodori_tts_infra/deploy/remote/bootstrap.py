from __future__ import annotations

from irodori_tts_infra.deploy.remote._common import _run
from irodori_tts_infra.deploy.remote.sync import (
    resolve_remote_dir,
    resolve_remote_host,
)


def bootstrap_remote(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
) -> None:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    _run(["ssh", host, _powershell(_mkdir_script(directory))])
    _run(["ssh", host, _powershell(_bootstrap_script(directory))])


def _bootstrap_script(remote_dir: str) -> str:
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; uv sync --extra server --extra irodori"
    )


def _mkdir_script(remote_dir: str) -> str:
    return f"New-Item -ItemType Directory -Force -Path {_ps_quote(remote_dir)} | Out-Null"


def _powershell(script: str) -> str:
    return f"powershell -NoProfile -ExecutionPolicy Bypass -Command {script}"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
