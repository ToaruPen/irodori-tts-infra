from __future__ import annotations

import shutil
import subprocess  # noqa: S404

import pytest

from irodori_tts_infra.deploy.remote._common import (  # noqa: PLC2701
    _powershell_stdin,
    _stdin_payload,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("powershell") is None,
        reason="requires Windows PowerShell",
    ),
]


def run_stdin_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        _powershell_stdin().split(" "),
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        input=_stdin_payload(script),
        text=True,
    )


def test_stdin_script_throw_propagates_as_failure_exit_code() -> None:
    result = run_stdin_script("Write-Output 'before'; if ($true) { throw 'boom' }; 'after'")

    assert result.returncode == 1
    assert result.stdout.split() == ["before"]


def test_stdin_script_explicit_exit_zero_is_success() -> None:
    result = run_stdin_script("Write-Output 'running 42'; exit 0; Write-Output 'after'")

    assert result.returncode == 0
    assert result.stdout.strip() == "running 42"


def test_stdin_script_preserves_non_ascii_text() -> None:
    path = "C:/Users/\u30c6\u30b9\u30c8/irodori"

    result = run_stdin_script(
        f"$p = '{path}'; ($p.ToCharArray() | ForEach-Object {{ [int]$_ }}) -join ','"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == ",".join(str(ord(char)) for char in path)
