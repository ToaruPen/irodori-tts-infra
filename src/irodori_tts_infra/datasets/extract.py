from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from irodori_tts_infra.datasets.models import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE
from irodori_tts_infra.datasets.moe_speech import (
    DEFAULT_MAX_BYTES,
    DEFAULT_OUTPUT_SAMPLE_RATE,
    NsfwSubsetUnavailableError,
    extract_character_dataset,
)

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(
    *,
    character: Annotated[
        str | None,
        typer.Option("--character", help="Speaker identifier inside litagin/moe-speech."),
    ] = None,
    out: Annotated[
        str | None,
        typer.Option(
            "--out",
            help="Directory to write extracted WAV files and index.json into.",
        ),
    ] = None,
    max_bytes: Annotated[
        int,
        typer.Option("--max-bytes", min=1, help="Maximum total output size in bytes."),
    ] = DEFAULT_MAX_BYTES,
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            min=MIN_SAMPLE_RATE,
            max=MAX_SAMPLE_RATE,
            help="Output WAV sample rate in Hz.",
        ),
    ] = DEFAULT_OUTPUT_SAMPLE_RATE,
    include_nsfw: Annotated[
        bool,
        typer.Option(
            "--include-nsfw/--no-include-nsfw",
            help="Allow extraction from the gated not-for-all-audiences moe-speech dataset.",
        ),
    ] = False,
) -> None:
    character = _strip_option(character, name="character")
    out = _strip_option(out, name="out")

    if not character:
        msg = "--character is required"
        raise typer.BadParameter(msg, param_hint="--character")
    if not out:
        msg = "--out is required"
        raise typer.BadParameter(msg, param_hint="--out")

    out_path = Path(out).expanduser().resolve()
    _validate_out_path(out_path)
    try:
        index = extract_character_dataset(
            character=character,
            out_dir=out_path,
            max_bytes=max_bytes,
            sample_rate=sample_rate,
            include_nsfw=include_nsfw,
        )
    except NsfwSubsetUnavailableError as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--include-nsfw/--no-include-nsfw",
        ) from exc

    clip_count = sum(len(clips) for clips in index.characters.values())
    typer.echo(f"Wrote {clip_count} clip(s) and index.json to {out_path}")


def _strip_option(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{name} must be a string"
        raise TypeError(msg)
    return value.strip()


def _validate_out_path(out_path: Path) -> None:
    if not out_path.exists():
        return
    if not out_path.is_dir():
        msg = "out_dir must be a directory"
        raise typer.BadParameter(msg, param_hint="--out")
    if any(out_path.iterdir()):
        msg = "out_dir must be empty before extraction"
        raise typer.BadParameter(msg, param_hint="--out")


if __name__ == "__main__":
    app()
