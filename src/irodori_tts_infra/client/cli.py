from __future__ import annotations

from typing import Annotated

import typer

from irodori_tts_infra import __version__

app = typer.Typer(no_args_is_help=True)


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
