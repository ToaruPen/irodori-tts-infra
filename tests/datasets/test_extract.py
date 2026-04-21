from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import typer
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from typer.testing import CliRunner

from irodori_tts_infra.datasets import extract
from irodori_tts_infra.datasets.models import ExtractedClip, ExtractionIndex
from irodori_tts_infra.datasets.moe_speech import (
    NsfwSubsetUnavailableError,
    UnsupportedAudioFormatError,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_output(output: str) -> str:
    return ANSI_RE.sub("", output)


def _readable_output(output: str) -> str:
    plain = _plain_output(output)
    without_box = re.sub(r"[╭╮╰╯─│]", " ", plain)
    return " ".join(without_box.split())


def test_cli_runs_extraction_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_extract_character_dataset(**kwargs: object) -> ExtractionIndex:
        captured.update(kwargs)
        return ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=10,
            total_duration_s=1.0,
            characters={"alice": (ExtractedClip(path="alice_000.wav", duration_s=1.0),)},
        )

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        [
            "--character",
            "alice",
            "--out",
            str(tmp_path),
            "--max-bytes",
            "99",
            "--sample-rate",
            "24000",
            "--include-nsfw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "character": "alice",
        "out_dir": tmp_path.resolve(),
        "max_bytes": 99,
        "sample_rate": 24_000,
        "include_nsfw": True,
    }
    assert "Wrote 1 clip(s)" in result.output


def test_cli_reports_unavailable_non_nsfw_subset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_extract_character_dataset(**kwargs: object) -> ExtractionIndex:
        captured.update(kwargs)
        msg = "litagin/moe-speech does not publish a separate non-NSFW subset"
        raise NsfwSubsetUnavailableError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "non-NSFW subset" in result.output
    assert captured["include_nsfw"] is False


@pytest.mark.parametrize(
    ("message", "expected_hint"),
    [
        ("character must not be blank", "--character"),
        ("out_dir must be empty before extraction", "--out"),
        ("sample_rate must be between 16000 and 48000", "--sample-rate"),
        ("max_bytes must be positive", "--max-bytes"),
        ("unsupported moe-speech path: x.wav", None),
    ],
)
def test_cli_reports_validation_value_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    message: str,
    expected_hint: str | None,
) -> None:
    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        raise ValueError(message)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path), "--include-nsfw"],
        color=False,
    )

    assert result.exit_code != 0
    output = _readable_output(result.output)
    assert message in output
    if expected_hint is None:
        assert "Invalid value for" not in output
    else:
        assert f"Invalid value for {expected_hint}" in output
    assert "Traceback" not in result.output


def test_cli_reports_unsupported_audio_format_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        msg = "moe-speech clips must be mono WAV files"
        raise UnsupportedAudioFormatError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path), "--include-nsfw"],
        color=False,
    )

    assert result.exit_code != 0
    assert "moe-speech clips must be mono WAV files" in _plain_output(result.output)
    assert "Traceback" not in result.output


def test_cli_reports_gated_repo_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        msg = "gated repo access is required"
        raise GatedRepoError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path)],
        color=False,
    )

    output = _readable_output(result.output)
    assert result.exit_code != 0
    assert "gated repo access is required" in output
    assert "Invalid value for --include-nsfw/--no-include-nsfw" in output
    assert "--include-nsfw" in output
    assert "Traceback" not in result.output


def test_cli_reports_gated_repo_error_with_include_nsfw_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        msg = "gated repo access is required"
        raise GatedRepoError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path), "--include-nsfw"],
        color=False,
    )

    output = _readable_output(result.output)
    assert result.exit_code != 0
    assert "gated repo access is required" in output
    assert "Accept the dataset terms at huggingface.co before retrying." in output
    assert "Invalid value for --include-nsfw" not in output
    assert "Invalid value for" not in output
    assert "Traceback" not in result.output


def test_cli_reports_hf_hub_http_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        msg = "401 Client Error: Unauthorized for url"
        raise HfHubHTTPError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path), "--include-nsfw"],
        color=False,
    )

    assert result.exit_code != 0
    assert "401 Client Error: Unauthorized for url" in _plain_output(result.output)
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "case",
    [
        (
            ValueError("sample_rate must be between 16000 and 48000"),
            True,
            ValueError,
        ),
        (
            UnsupportedAudioFormatError("moe-speech clips must be mono WAV files"),
            True,
            UnsupportedAudioFormatError,
        ),
        (
            GatedRepoError("gated repo access is required"),
            False,
            GatedRepoError,
        ),
        (
            HfHubHTTPError("401 Client Error: Unauthorized for url"),
            True,
            HfHubHTTPError,
        ),
        (
            NsfwSubsetUnavailableError(
                "litagin/moe-speech does not publish a separate non-NSFW subset",
            ),
            False,
            NsfwSubsetUnavailableError,
        ),
    ],
)
def test_main_preserves_original_exception_as_cause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: tuple[Exception, bool, type[Exception]],
) -> None:
    raised, include_nsfw, expected_cause = case

    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        raise raised

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    with pytest.raises(typer.BadParameter) as exc_info:
        extract.main(character="alice", out=str(tmp_path), include_nsfw=include_nsfw)

    assert isinstance(exc_info.value.__cause__, expected_cause)


def test_cli_rejects_invalid_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        nonlocal called
        called = True
        msg = "should not be called for invalid sample-rate"
        raise AssertionError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path), "--sample-rate", "15999"],
    )

    assert result.exit_code != 0
    assert "16000<=x<=48000" in result.output
    assert called is False


def test_cli_rejects_blank_character(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        nonlocal called
        called = True
        msg = "should not be called for blank character"
        raise AssertionError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "   ", "--out", str(tmp_path)],
        color=False,
    )

    assert result.exit_code != 0
    assert "--character is required" in _plain_output(result.output)
    assert called is False


def test_cli_rejects_blank_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        nonlocal called
        called = True
        msg = "should not be called for blank output directory"
        raise AssertionError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", "   "],
        color=False,
    )

    assert result.exit_code != 0
    assert "--out is required" in _plain_output(result.output)
    assert called is False


def test_cli_rejects_existing_file_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False
    out_path = tmp_path / "dataset"
    out_path.write_text("not a directory", encoding="utf-8")

    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        nonlocal called
        called = True
        msg = "should not be called for invalid output path"
        raise AssertionError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(out_path), "--include-nsfw"],
        color=False,
    )

    assert result.exit_code != 0
    assert "out_dir must be a directory" in _plain_output(result.output)
    assert called is False


def test_cli_rejects_non_empty_out_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False
    out_path = tmp_path / "dataset"
    out_path.mkdir()
    (out_path / "existing.txt").write_text("existing", encoding="utf-8")

    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        nonlocal called
        called = True
        msg = "should not be called for non-empty output directory"
        raise AssertionError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(out_path), "--include-nsfw"],
        color=False,
    )

    assert result.exit_code != 0
    assert "out_dir must be empty before extraction" in _plain_output(result.output)
    assert called is False


def test_main_requires_character_and_out_options(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="--character is required"):
        extract.main(character=None, out=str(tmp_path))

    with pytest.raises(typer.BadParameter, match="--out is required"):
        extract.main(character="alice", out=None)

    with pytest.raises(typer.BadParameter, match="--character is required"):
        extract.main(character="   ", out=str(tmp_path))

    with pytest.raises(typer.BadParameter, match="--out is required"):
        extract.main(character="alice", out="   ")


def test_main_rejects_non_string_character_and_out(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="character"):
        extract.main(character=object(), out=str(tmp_path))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="out"):
        extract.main(character="alice", out=tmp_path)  # type: ignore[arg-type]


def test_cli_help_shows_usage() -> None:
    result = CliRunner().invoke(extract.app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
