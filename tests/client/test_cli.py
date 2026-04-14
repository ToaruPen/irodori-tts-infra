from __future__ import annotations

import subprocess  # noqa: S404
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import pytest
from typer.testing import CliRunner

from irodori_tts_infra.client import cli
from irodori_tts_infra.voice_bank import (
    DEFAULT_GENERIC_DIALOGUE_CAPTION,
    DEFAULT_NARRATOR_CAPTION,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from irodori_tts_infra.contracts import SynthesisRequest

pytestmark = pytest.mark.unit


class FakeSyncIrodoriClient:
    instances: ClassVar[list[FakeSyncIrodoriClient]] = []
    wav_by_text: ClassVar[dict[str, bytes]] = {}

    def __init__(self, *, base_url: str | None = None) -> None:
        self.base_url = base_url
        self.requests: list[SynthesisRequest] = []
        self.closed = False
        self.instances.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object,
    ) -> None:
        self.closed = True

    def synthesize_stream(self, request: SynthesisRequest) -> Iterator[bytes]:
        self.requests.append(request)
        yield self.wav_by_text[request.text]


def test_order_audio_segments_reassembles_out_of_order_chunks() -> None:
    ordered = cli._order_audio_segments(  # noqa: SLF001
        [(2, b"third"), (0, b"first"), (1, b"second")]
    )

    assert ordered == [(0, b"first"), (1, b"second"), (2, b"third")]


def test_read_aloud_synthesizes_and_plays_segments_in_resolved_caption_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text(
        "# Turn\n地の文です。\n【チヅル:小声で】「こんにちは」\n【不明】「だれ?」\n",
        encoding="utf-8",
    )
    (tmp_path / "characters.md").write_text(
        "## チヅル\n- **性格**: クール\n- **年齢/外見**: 高校生の女子\n",
        encoding="utf-8",
    )
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.wav_by_text = {
        "地の文です。": b"narration-wav",
        "こんにちは": b"known-wav",
        "だれ?": b"unknown-wav",
    }
    played_audio: list[bytes] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert command[0] == "afplay"
        played_audio.append(Path(command[-1]).read_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code == 0, result.output
    assert played_audio == [b"narration-wav", b"known-wav", b"unknown-wav"]
    client = FakeSyncIrodoriClient.instances[0]
    assert client.closed is True
    assert [request.text for request in client.requests] == ["地の文です。", "こんにちは", "だれ?"]
    assert client.requests[0].caption == DEFAULT_NARRATOR_CAPTION
    assert client.requests[1].caption == (
        "若い女性が、落ち着いたクールな調子で小声で話している。クリアな音質。若々しい声。"
    )
    assert client.requests[2].caption == DEFAULT_GENERIC_DIALOGUE_CAPTION


def test_read_aloud_missing_turn_file_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["read-aloud", str(tmp_path / "missing.md")])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_read_aloud_uses_default_profile_without_characters_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("地の文です。\n「こんにちは」\n", encoding="utf-8")
    save_dir = tmp_path / "audio"
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.wav_by_text = {
        "地の文です。": b"narration-wav",
        "こんにちは": b"dialogue-wav",
    }
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--save-dir", str(save_dir)],
    )

    assert result.exit_code == 0, result.output
    requests = FakeSyncIrodoriClient.instances[0].requests
    assert [request.caption for request in requests] == [
        DEFAULT_NARRATOR_CAPTION,
        DEFAULT_GENERIC_DIALOGUE_CAPTION,
    ]
    assert (save_dir / "0000.wav").read_bytes() == b"narration-wav"
    assert (save_dir / "0001.wav").read_bytes() == b"dialogue-wav"


def test_read_aloud_remote_host_override_builds_client_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.wav_by_text = {"本文です。": b"wav"}
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    def fake_run(_command: list[str], *, check: bool) -> None:
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--remote-host", "100.112.161.83"],
    )

    assert result.exit_code == 0, result.output
    assert FakeSyncIrodoriClient.instances[0].base_url == "http://100.112.161.83:8923"


def test_read_aloud_save_dir_writes_ordered_audio_files_and_skips_playback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("一。\n二。", encoding="utf-8")
    save_dir = tmp_path / "wav"
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.wav_by_text = {"一。二。": b"combined-wav"}
    playback_calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        playback_calls.append(command)

    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--save-dir", str(save_dir)],
    )

    assert result.exit_code == 0, result.output
    assert playback_calls == []
    assert (save_dir / "0000.wav").read_bytes() == b"combined-wav"
