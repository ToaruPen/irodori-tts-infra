from __future__ import annotations

import os

from irodori_tts_infra.deploy.remote._common import _run
from irodori_tts_infra.deploy.remote.sync import (
    resolve_remote_dir,
    resolve_remote_host,
)


def bootstrap_remote(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
    irodori_tts_dir: str | None = None,
    python_version: str | None = None,
    torch_backend_extra: str | None = None,
) -> None:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    irodori_dir = resolve_irodori_tts_dir(irodori_tts_dir)
    runtime_python = resolve_runtime_python(python_version)
    torch_extra = resolve_torch_backend_extra(torch_backend_extra)
    _run(["ssh", host, _powershell(_mkdir_script(directory))])
    _run(
        [
            "ssh",
            host,
            _powershell(
                _bootstrap_script(
                    directory,
                    irodori_tts_dir=irodori_dir,
                    python_version=runtime_python,
                    torch_backend_extra=torch_extra,
                ),
            ),
        ],
    )


def resolve_irodori_tts_dir(value: str | None) -> str:
    directory = value if value is not None else os.getenv("IRODORI_TTS_DIR")
    if directory is None:
        msg = "irodori tts dir is required"
        raise ValueError(msg)
    _reject_blank("irodori_tts_dir", directory)
    return directory


def resolve_runtime_python(value: str | None) -> str:
    python_version = value if value is not None else os.getenv("IRODORI_TTS_RUNTIME_PYTHON", "3.11")
    _reject_blank("python_version", python_version)
    return python_version


def resolve_torch_backend_extra(value: str | None) -> str:
    torch_backend_extra = (
        value if value is not None else os.getenv("IRODORI_TTS_TORCH_BACKEND_EXTRA", "cu128")
    )
    _reject_blank("torch_backend_extra", torch_backend_extra)
    return torch_backend_extra


def _bootstrap_script(
    remote_dir: str,
    *,
    irodori_tts_dir: str,
    python_version: str,
    torch_backend_extra: str,
) -> str:
    runtime_python = "$(Join-Path (Get-Location) '.runtime-venv/Scripts/python.exe')"
    irodori_requirement = (
        f"Irodori-TTS[{torch_backend_extra}] @ {_path_to_file_url(irodori_tts_dir)}"
    )
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        f"uv venv '.runtime-venv' --python {_ps_quote(python_version)} --clear; "
        f"uv pip install --python {runtime_python} {_ps_quote(irodori_requirement)}; "
        f"uv pip install --python {runtime_python} {_ps_quote('.[server,irodori]')}; "
        f"uv pip check --python {runtime_python}"
    )


def _mkdir_script(remote_dir: str) -> str:
    return f"New-Item -ItemType Directory -Force -Path {_ps_quote(remote_dir)} | Out-Null"


def _powershell(script: str) -> str:
    return f"powershell -NoProfile -ExecutionPolicy Bypass -Command {script}"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _path_to_file_url(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        return f"file://{normalized}"
    return f"file:///{normalized}"


def _reject_blank(name: str, value: str) -> None:
    if not value.strip():
        msg = f"{name} must not be blank"
        raise ValueError(msg)
