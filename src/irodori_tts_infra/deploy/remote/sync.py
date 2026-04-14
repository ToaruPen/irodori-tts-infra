from __future__ import annotations

import os
import shutil
from pathlib import Path

from irodori_tts_infra.deploy.remote._common import _run

DEFAULT_REMOTE_DIR = "C:/irodori-tts-infra"
RSYNC_EXCLUDES = (
    ".env",
    ".git/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    "__pycache__/",
    "audio/",
    "checkpoints/",
    "data/",
    "models/",
    "outputs/",
    "runs/",
)
_SYNC_ITEMS = ("src", "pyproject.toml", ".env.example")


def sync_project(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
    repo_root: Path | str | None = None,
) -> None:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    root = Path.cwd() if repo_root is None else Path(repo_root)
    _validate_sync_sources(root)

    if shutil.which("rsync") is not None:
        _run(_rsync_command(host, directory, root))
        return

    _run(["ssh", host, _powershell(_mkdir_script(directory))])
    _run(_scp_command(host, directory, root))


def resolve_remote_host(remote_host: str | None = None) -> str:
    host = remote_host if remote_host is not None else os.environ.get("IRODORI_REMOTE_HOST")
    if host is None or not host.strip():
        msg = "remote host is required; pass --remote-host or set IRODORI_REMOTE_HOST"
        raise ValueError(msg)
    return host.strip()


def resolve_remote_dir(remote_dir: str | None = None) -> str:
    directory = remote_dir if remote_dir is not None else os.environ.get("IRODORI_DEPLOY_DIR")
    if directory is None:
        return DEFAULT_REMOTE_DIR
    if not directory.strip():
        msg = "remote directory must not be blank"
        raise ValueError(msg)
    return directory.strip()


def _rsync_command(remote_host: str, remote_dir: str, repo_root: Path) -> list[str]:
    command = ["rsync", "-az", "--delete", "-e", "ssh"]
    for pattern in RSYNC_EXCLUDES:
        command.extend(["--exclude", pattern])
    remote_target = remote_host + ":" + _remote_dir_with_trailing_slash(remote_dir)
    command.extend(
        [
            str(repo_root / "src"),
            str(repo_root / "pyproject.toml"),
            str(repo_root / ".env.example"),
            remote_target,
        ],
    )
    return command


def _scp_command(remote_host: str, remote_dir: str, repo_root: Path) -> list[str]:
    remote_target = remote_host + ":" + _remote_dir_with_trailing_slash(remote_dir)
    return [
        "scp",
        "-r",
        str(repo_root / "src"),
        str(repo_root / "pyproject.toml"),
        str(repo_root / ".env.example"),
        remote_target,
    ]


def _validate_sync_sources(repo_root: Path) -> None:
    missing = [name for name in _SYNC_ITEMS if not (repo_root / name).exists()]
    if missing:
        msg = f"repo root is missing deploy source item(s): {', '.join(missing)}"
        raise ValueError(msg)


def _remote_dir_with_trailing_slash(remote_dir: str) -> str:
    if remote_dir.endswith(("/", "\\")):
        return remote_dir
    return f"{remote_dir}/"


def _mkdir_script(remote_dir: str) -> str:
    return f"New-Item -ItemType Directory -Force -Path {_ps_quote(remote_dir)} | Out-Null"


def _powershell(script: str) -> str:
    return f"powershell -NoProfile -ExecutionPolicy Bypass -Command {script}"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
