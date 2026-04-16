from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer

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
        Path | None,
        typer.Option(
            "--out",
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            writable=True,
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
            min=16_000,
            max=48_000,
            help="Output WAV sample rate in Hz.",
        ),
    ] = DEFAULT_OUTPUT_SAMPLE_RATE,
    include_nsfw: Annotated[
        bool,
        typer.Option(
            "--include-nsfw/--no-include-nsfw",
            help="Allow extraction from the gated not-for-all-audiences moe-speech dataset.",
        ),
    ] = True,
) -> None:
    if character is None:
        msg = "--character is required"
        raise typer.BadParameter(msg, param_hint="--character")
    if out is None:
        msg = "--out is required"
        raise typer.BadParameter(msg, param_hint="--out")

    try:
        index = extract_character_dataset(
            character=character,
            out_dir=out,
            max_bytes=max_bytes,
            sample_rate=sample_rate,
            include_nsfw=include_nsfw,
        )
    except NsfwSubsetUnavailableError as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--include-nsfw/--no-include-nsfw",
        ) from exc

    clip_count = len(index.characters.get(character.strip(), ()))
    typer.echo(f"Wrote {clip_count} clip(s) and index.json to {out}")


if __name__ == "__main__":
    app()
