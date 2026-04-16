from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from irodori_tts_infra.datasets import extract
from irodori_tts_infra.datasets.models import ExtractedClip, ExtractionIndex
from irodori_tts_infra.datasets.moe_speech import NsfwSubsetUnavailableError

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


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
    def fake_extract_character_dataset(**_kwargs: object) -> ExtractionIndex:
        msg = "litagin/moe-speech does not publish a separate non-NSFW subset"
        raise NsfwSubsetUnavailableError(msg)

    monkeypatch.setattr(extract, "extract_character_dataset", fake_extract_character_dataset)

    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path), "--no-include-nsfw"],
    )

    assert result.exit_code != 0
    assert "non-NSFW subset" in result.output


def test_cli_rejects_invalid_sample_rate(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        extract.app,
        ["--character", "alice", "--out", str(tmp_path), "--sample-rate", "15999"],
    )

    assert result.exit_code != 0
    assert "16000<=x<=48000" in result.output


def test_cli_help_shows_usage() -> None:
    result = CliRunner().invoke(extract.app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
