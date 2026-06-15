from __future__ import annotations

import subprocess  # noqa: S404
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import pytest
import typer
from typer.testing import CliRunner
from typing_extensions import override

from irodori_tts_infra.client import cli
from irodori_tts_infra.client.errors import ClientUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from irodori_tts_infra.contracts import SynthesisRequest

pytestmark = pytest.mark.unit


class FakeSyncIrodoriClient:
    instances: ClassVar[list[FakeSyncIrodoriClient]] = []
    events: ClassVar[list[tuple[str, str | bytes]]] = []
    wav_chunks_by_text: ClassVar[dict[str, list[bytes]]] = {}

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
        self.events.append(("synthesize", request.text))
        yield from self.wav_chunks_by_text[request.text]


class FailingSyncIrodoriClient(FakeSyncIrodoriClient):
    @override
    def synthesize_stream(self, request: SynthesisRequest) -> Iterator[bytes]:
        self.requests.append(request)
        message = "connection failed"
        raise ClientUnavailableError(message, endpoint="/synthesize_stream")


def write_speaker_manifest(root: Path) -> Path:
    manifest = root / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"

[characters."ミカ"]
ref_embed = "speakers/mika.speaker.safetensors"
""",
        encoding="utf-8",
    )
    return manifest


def write_narrator_only_speaker_manifest(root: Path) -> Path:
    manifest = root / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
""",
        encoding="utf-8",
    )
    return manifest


def test_read_aloud_synthesizes_and_plays_segments_in_speaker_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text(
        "# Turn\n地の文です。\n【チヅル:小声で】「こんにちは」\n",
        encoding="utf-8",
    )
    (tmp_path / "characters.md").write_text(
        """
## チヅル
- **性格**: クール
- **年齢/外見**: 高校生の女子

## ミカ
- **性格**: 明るい
""",
        encoding="utf-8",
    )
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {
        "地の文です。": [b"narration-", b"wav"],
        "こんにちは": [b"known-wav"],
    }
    played_audio: list[bytes] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert command[0] == "afplay"
        played_audio.append(Path(command[-1]).read_bytes())
        FakeSyncIrodoriClient.events.append(("play", played_audio[-1]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code == 0, result.output
    assert played_audio == [b"narration-wav", b"known-wav"]
    assert FakeSyncIrodoriClient.events == [
        ("synthesize", "地の文です。"),
        ("play", b"narration-wav"),
        ("synthesize", "こんにちは"),
        ("play", b"known-wav"),
    ]
    client = FakeSyncIrodoriClient.instances[0]
    assert client.closed is True
    assert [request.text for request in client.requests] == ["地の文です。", "こんにちは"]
    assert client.requests[0].speaker is None
    assert client.requests[0].ref_embed is None
    assert client.requests[1].speaker == "チヅル"
    assert client.requests[1].ref_embed is None


def test_read_aloud_splits_long_segments_before_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text(
        "朝の教室には柔らかい光が差し込み、窓際の机だけが少し暖かかった。",
        encoding="utf-8",
    )
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {
        "朝の教室には柔らかい光が差し込み、": [b"first"],
        "窓際の机だけが少し暖かかった。": [b"second"],
    }
    played_audio: list[bytes] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        played_audio.append(Path(command[-1]).read_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "DEFAULT_TTS_MAX_CHARS", 24)

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code == 0, result.output
    client = FakeSyncIrodoriClient.instances[0]
    assert [request.text for request in client.requests] == [
        "朝の教室には柔らかい光が差し込み、",
        "窓際の机だけが少し暖かかった。",
    ]
    assert played_audio == [b"first", b"second"]


def test_read_aloud_preserves_speaker_when_splitting_dialogue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text(
        "【チヅル】「近くに来て。もう少しだけ、静かに話して。」",
        encoding="utf-8",
    )
    (tmp_path / "characters.md").write_text(
        """
## チヅル
- **性格**: クール

## ミカ
- **性格**: 明るい
""",
        encoding="utf-8",
    )
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {
        "近くに来て。": [b"first"],
        "もう少しだけ、": [b"second"],
        "静かに話して。": [b"third"],
    }
    played_audio: list[bytes] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        played_audio.append(Path(command[-1]).read_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "DEFAULT_TTS_MAX_CHARS", 12)

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code == 0, result.output
    client = FakeSyncIrodoriClient.instances[0]
    assert [request.text for request in client.requests] == [
        "近くに来て。",
        "もう少しだけ、",
        "静かに話して。",
    ]
    assert [request.speaker for request in client.requests] == ["チヅル", "チヅル", "チヅル"]
    assert played_audio == [b"first", b"second", b"third"]


def test_read_aloud_reports_too_many_prepared_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("一。二。三。", encoding="utf-8")
    write_narrator_only_speaker_manifest(tmp_path)

    monkeypatch.setattr(cli, "DEFAULT_TTS_MAX_SEGMENTS", 2)

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code != 0
    assert "too many TTS" in result.output
    assert "segments" in result.output


def test_read_aloud_removes_temp_wav_after_playback_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {"本文です。": [b"wav"]}
    played_paths: list[Path] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        temp_path = Path(command[-1])
        assert temp_path.exists()
        played_paths.append(temp_path)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code != 0
    assert len(played_paths) == 1
    assert not played_paths[0].exists()


def test_read_aloud_missing_turn_file_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["read-aloud", str(tmp_path / "missing.md")])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_validate_optional_local_profile_reports_invalid_speaker_profile(
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    characters = tmp_path / "characters.md"
    characters.write_text("## ミカ\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."いない"]
ref_embed = "speakers/missing.speaker.safetensors"
""",
        encoding="utf-8",
    )

    with pytest.raises(typer.BadParameter) as exc_info:
        cli._validate_optional_local_profile(  # noqa: SLF001
            turn_file=turn_file,
            characters=characters,
            speaker_manifest=manifest,
        )

    assert "invalid --speaker-manifest/--characters" in str(exc_info.value)


def test_read_aloud_synthesis_failure_exits_with_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    FailingSyncIrodoriClient.instances = []
    monkeypatch.setattr(cli, "SyncIrodoriClient", FailingSyncIrodoriClient)

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code != 0
    assert "connection failed" in result.output


def test_read_aloud_uses_speaker_manifest_without_characters_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("地の文です。\n【ミカ】「こんにちは」\n", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    save_dir = tmp_path / "audio"
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {
        "地の文です。": [b"narration-wav"],
        "こんにちは": [b"dialogue-wav"],
    }
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--save-dir", str(save_dir)],
    )

    assert result.exit_code == 0, result.output
    requests = FakeSyncIrodoriClient.instances[0].requests
    assert [(request.speaker, request.ref_embed) for request in requests] == [
        (None, None),
        ("ミカ", None),
    ]
    assert (save_dir / "segment-0000.wav").read_bytes() == b"narration-wav"
    assert (save_dir / "segment-0001.wav").read_bytes() == b"dialogue-wav"


def test_read_aloud_sends_speaker_without_local_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("地の文です。\n【ミカ】「こんにちは」\n", encoding="utf-8")
    save_dir = tmp_path / "audio"
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {
        "地の文です。": [b"narration-wav"],
        "こんにちは": [b"dialogue-wav"],
    }
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    result = CliRunner().invoke(
        cli.app,
        [
            "read-aloud",
            str(turn_file),
            "--remote-host",
            "gpu.example.test",
            "--save-dir",
            str(save_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    requests = FakeSyncIrodoriClient.instances[0].requests
    assert [(request.speaker, request.ref_embed) for request in requests] == [
        (None, None),
        ("ミカ", None),
    ]


def test_read_aloud_does_not_reject_speaker_missing_from_local_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("【リナ】「こんにちは」\n", encoding="utf-8")
    write_narrator_only_speaker_manifest(tmp_path)
    save_dir = tmp_path / "audio"
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {"こんにちは": [b"dialogue-wav"]}
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--save-dir", str(save_dir)],
    )

    assert result.exit_code == 0, result.output
    request = FakeSyncIrodoriClient.instances[0].requests[0]
    assert request.speaker == "リナ"
    assert request.ref_embed is None


def test_read_aloud_remote_host_override_builds_client_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {"本文です。": [b"wav"]}
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
    assert FakeSyncIrodoriClient.instances[0].requests[0].ref_embed is None


def test_read_aloud_remote_host_with_explicit_port_skips_default_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {"本文です。": [b"wav"]}
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    def fake_run(_command: list[str], *, check: bool) -> None:
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--remote-host", "example.com:9000"],
    )

    assert result.exit_code == 0, result.output
    assert FakeSyncIrodoriClient.instances[0].base_url == "http://example.com:9000"


def test_read_aloud_remote_host_with_https_prefix_preserves_scheme(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {"本文です。": [b"wav"]}
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    def fake_run(_command: list[str], *, check: bool) -> None:
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--remote-host", "https://api.example.com/"],
    )

    assert result.exit_code == 0, result.output
    assert FakeSyncIrodoriClient.instances[0].base_url == "https://api.example.com"


def test_read_aloud_save_dir_clears_stale_wav_files_and_skips_playback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("一。\n二。", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    save_dir = tmp_path / "wav"
    save_dir.mkdir()
    (save_dir / "segment-0000.wav").write_bytes(b"old-first")
    (save_dir / "segment-0001.wav").write_bytes(b"stale-second")
    (save_dir / "segment-10000.wav").write_bytes(b"stale-large-run")
    (save_dir / "0000.wav").write_bytes(b"user-numeric-wav")
    (save_dir / "personal.wav").write_bytes(b"user-wav")
    (save_dir / "notes.txt").write_text("keep me", encoding="utf-8")
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {
        "一。": [b"first-wav"],
        "二。": [b"second-wav"],
    }
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
    assert (save_dir / "segment-0000.wav").read_bytes() == b"first-wav"
    assert (save_dir / "segment-0001.wav").read_bytes() == b"second-wav"
    assert not (save_dir / "segment-10000.wav").exists()
    assert (save_dir / "0000.wav").read_bytes() == b"user-numeric-wav"
    assert (save_dir / "personal.wav").read_bytes() == b"user-wav"
    assert (save_dir / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_read_aloud_blank_remote_host_exits_with_error(tmp_path: Path) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    write_speaker_manifest(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--remote-host", "   "],
    )

    assert result.exit_code != 0
    assert "remote host must not be blank" in result.output


def test_read_aloud_blank_player_command_exits_with_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("本文です。", encoding="utf-8")
    write_speaker_manifest(tmp_path)
    FakeSyncIrodoriClient.instances = []
    FakeSyncIrodoriClient.events = []
    FakeSyncIrodoriClient.wav_chunks_by_text = {"本文です。": [b"wav"]}
    monkeypatch.setattr(cli, "SyncIrodoriClient", FakeSyncIrodoriClient)

    result = CliRunner().invoke(
        cli.app,
        ["read-aloud", str(turn_file), "--player-command", "   "],
    )

    assert result.exit_code != 0
    assert "player command must not be blank" in result.output


def test_read_aloud_empty_turn_file_exits_with_error(tmp_path: Path) -> None:
    turn_file = tmp_path / "turn.md"
    turn_file.write_text("", encoding="utf-8")

    result = CliRunner().invoke(cli.app, ["read-aloud", str(turn_file)])

    assert result.exit_code != 0
    assert "turn file contains no readable segments" in result.output
