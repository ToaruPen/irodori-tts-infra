from __future__ import annotations

from typing import TYPE_CHECKING

from irodori_tts_infra.deploy.remote._common import _load_env_script, _powershell, _ps_quote, _run
from irodori_tts_infra.deploy.remote.sync import (
    resolve_remote_dir,
    resolve_remote_host,
)

if TYPE_CHECKING:
    import subprocess


def verify_voice_bank(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
) -> subprocess.CompletedProcess[str]:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    return _run(["ssh", host, _powershell(_verify_script(directory))])


def _verify_script(remote_dir: str) -> str:
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        f"{_load_env_script()}"
        "$runtimePython = Join-Path (Get-Location) '.runtime-venv/Scripts/python.exe'; "
        "if (!(Test-Path -LiteralPath $runtimePython)) { "
        "throw 'runtime Python is missing; run deploy-bootstrap first' }; "
        "$code = @'\n"
        "from irodori_tts_infra.server.main import "
        "_resolve_characters_markdown, _resolve_speaker_manifest\n"
        "from irodori_tts_infra.voice_bank.repository import load_voice_profile\n"
        "speaker_manifest = _resolve_speaker_manifest()\n"
        "characters_md = _resolve_characters_markdown(speaker_manifest)\n"
        "profile = load_voice_profile(characters_md, speaker_manifest=speaker_manifest, "
        "require_embedding_files=True)\n"
        "print(f'voice bank ok: {speaker_manifest} ({len(profile.characters)} "
        "character speaker(s))')\n"
        "'@; "
        "& $runtimePython -c $code"
    )
