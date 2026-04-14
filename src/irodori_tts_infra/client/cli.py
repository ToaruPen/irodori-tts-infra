from __future__ import annotations

import shlex
import subprocess  # noqa: S404
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.progress import Progress

from irodori_tts_infra import __version__
from irodori_tts_infra.client.sync import SyncIrodoriClient
from irodori_tts_infra.config import ClientSettings
from irodori_tts_infra.contracts import SynthesisRequest
from irodori_tts_infra.text import Segment, parse_turn_markdown
from irodori_tts_infra.voice_bank import (
    DEFAULT_NARRATOR_CAPTION,
    find_characters_markdown,
    load_voice_profile,
    resolve_segment_caption,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from irodori_tts_infra.voice_bank.models import VoiceProfile

app = typer.Typer(no_args_is_help=True)
AudioSegment = tuple[int, bytes]


def _version_callback(value: bool) -> None:  # noqa: FBT001
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    *,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            help="Show the installed package version.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    _ = version


@app.command("read-aloud")
def read_aloud(
    turn_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Markdown turn file to synthesize.",
        ),
    ],
    *,
    characters: Annotated[
        Path | None,
        typer.Option(
            "--characters",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="characters.md path. Defaults to discovery from the turn file.",
        ),
    ] = None,
    remote_host: Annotated[
        str | None,
        typer.Option(
            "--remote-host",
            help="Override the Irodori server host or base URL.",
        ),
    ] = None,
    narrator_caption: Annotated[
        str | None,
        typer.Option(
            "--narrator-caption",
            help="VoiceDesign caption for narration segments.",
        ),
    ] = None,
    save_dir: Annotated[
        Path | None,
        typer.Option(
            "--save-dir",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
            help="Write ordered WAV files instead of playing them.",
        ),
    ] = None,
    player_command: Annotated[
        str,
        typer.Option(
            "--player-command",
            help="Playback command used when --save-dir is not set.",
        ),
    ] = "afplay",
) -> None:
    segments = parse_turn_markdown(turn_file.read_text(encoding="utf-8"))
    if not segments:
        message = "turn file contains no readable segments"
        raise typer.BadParameter(message)

    profile = _load_profile(
        turn_file=turn_file,
        characters=characters,
        narrator_caption=narrator_caption,
    )
    base_url = _base_url_from_remote_host(remote_host)
    with SyncIrodoriClient(base_url=base_url) as client:
        audio_segments = _synthesize_audio_segments(client, segments, profile)
        if save_dir is not None:
            saved = _save_audio_segments(audio_segments, save_dir)
            typer.echo(f"Saved {saved} WAV file(s) to {save_dir}")
            return

        _play_audio_segments(audio_segments, _player_command_parts(player_command))


def _load_profile(
    *,
    turn_file: Path,
    characters: Path | None,
    narrator_caption: str | None,
) -> VoiceProfile:
    caption = narrator_caption if narrator_caption is not None else DEFAULT_NARRATOR_CAPTION
    if not caption.strip():
        message = "narrator caption must not be blank"
        raise typer.BadParameter(message)

    characters_path = characters if characters is not None else find_characters_markdown(turn_file)
    return load_voice_profile(characters_path, narrator_caption=caption)


def _base_url_from_remote_host(remote_host: str | None) -> str | None:
    if remote_host is None:
        return None

    host = remote_host.strip()
    if not host:
        message = "remote host must not be blank"
        raise typer.BadParameter(message)
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")

    settings = ClientSettings()
    if ":" in host:
        return f"http://{host}"
    return f"http://{host}:{settings.port}"


def _synthesize_audio_segments(
    client: SyncIrodoriClient,
    segments: list[Segment],
    profile: VoiceProfile,
) -> Iterator[AudioSegment]:
    with Progress() as progress:
        task_id = progress.add_task("Synthesizing", total=len(segments))
        for segment_index, segment in enumerate(segments):
            request = SynthesisRequest(
                text=segment.text,
                caption=resolve_segment_caption(segment, profile),
            )
            wav_bytes = b"".join(client.synthesize_stream(request))
            yield segment_index, wav_bytes
            progress.advance(task_id)


def _save_audio_segments(audio_segments: Iterable[AudioSegment], save_dir: Path) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in save_dir.glob("segment-[0-9][0-9][0-9][0-9].wav"):
        if wav_path.is_file():
            wav_path.unlink()

    saved = 0
    for segment_index, wav_bytes in audio_segments:
        (save_dir / f"segment-{segment_index:04d}.wav").write_bytes(wav_bytes)
        saved += 1
    return saved


def _play_audio_segments(audio_segments: Iterable[AudioSegment], player_command: list[str]) -> None:
    for segment_index, wav_bytes in audio_segments:
        _play_wav_bytes(wav_bytes, player_command, segment_index=segment_index)


def _play_wav_bytes(
    wav_bytes: bytes,
    player_command: list[str],
    *,
    segment_index: int,
) -> None:
    temp_path = _write_temp_wav(wav_bytes, segment_index=segment_index)
    try:
        subprocess.run([*player_command, str(temp_path)], check=True)  # noqa: S603
    finally:
        temp_path.unlink(missing_ok=True)


def _write_temp_wav(wav_bytes: bytes, *, segment_index: int) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"irodori-tts-{segment_index:04d}-",
        suffix=".wav",
        delete=False,
    ) as temp_file:
        temp_file.write(wav_bytes)
        return Path(temp_file.name)


def _player_command_parts(player_command: str) -> list[str]:
    parts = shlex.split(player_command)
    if not parts:
        message = "player command must not be blank"
        raise typer.BadParameter(message)
    return parts
