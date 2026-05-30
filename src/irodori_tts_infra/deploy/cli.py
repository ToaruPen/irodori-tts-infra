from __future__ import annotations

from subprocess import CalledProcessError  # noqa: S404
from typing import Annotated, NoReturn

import typer

from irodori_tts_infra.deploy.remote.bootstrap import bootstrap_remote
from irodori_tts_infra.deploy.remote.service import (
    start_service,
    status_service,
    stop_service,
)
from irodori_tts_infra.deploy.remote.sync import sync_project
from irodori_tts_infra.deploy.remote.voice_bank import verify_voice_bank

app = typer.Typer(no_args_is_help=True)

RemoteHostOption = Annotated[
    str | None,
    typer.Option(
        "--remote-host",
        help="SSH host, e.g. user@gpu-host. Defaults to IRODORI_REMOTE_HOST.",
    ),
]
RemoteDirOption = Annotated[
    str | None,
    typer.Option(
        "--remote-dir",
        help="Windows project directory. Defaults to IRODORI_DEPLOY_DIR.",
    ),
]


@app.command("deploy-sync")
def deploy_sync(
    *,
    remote_host: RemoteHostOption = None,
    remote_dir: RemoteDirOption = None,
    repo_root: Annotated[
        str | None,
        typer.Option(
            "--repo-root",
            help="Local repository root to copy.",
        ),
    ] = None,
) -> None:
    try:
        if repo_root is None:
            sync_project(remote_host=remote_host, remote_dir=remote_dir)
        else:
            sync_project(remote_host=remote_host, remote_dir=remote_dir, repo_root=repo_root)
    except CalledProcessError as exc:
        _raise_remote_process_error(exc)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("deploy sync complete")


@app.command("deploy-bootstrap")
def deploy_bootstrap(
    *,
    remote_host: RemoteHostOption = None,
    remote_dir: RemoteDirOption = None,
    irodori_tts_dir: Annotated[
        str | None,
        typer.Option(
            "--irodori-tts-dir",
            help="Windows Irodori-TTS checkout directory. Defaults to IRODORI_TTS_DIR.",
        ),
    ] = None,
    python_version: Annotated[
        str | None,
        typer.Option(
            "--python-version",
            help=(
                "Python version for the runtime venv. "
                "Defaults to IRODORI_TTS_RUNTIME_PYTHON or 3.11."
            ),
        ),
    ] = None,
    torch_backend_extra: Annotated[
        str | None,
        typer.Option(
            "--torch-backend-extra",
            help=(
                "Irodori-TTS optional extra for the torch backend. "
                "Defaults to IRODORI_TTS_TORCH_BACKEND_EXTRA or cu128."
            ),
        ),
    ] = None,
) -> None:
    try:
        bootstrap_remote(
            remote_host=remote_host,
            remote_dir=remote_dir,
            irodori_tts_dir=irodori_tts_dir,
            python_version=python_version,
            torch_backend_extra=torch_backend_extra,
        )
    except CalledProcessError as exc:
        _raise_remote_process_error(exc)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("deploy bootstrap complete")


@app.command("deploy-start")
def deploy_start(
    *,
    remote_host: RemoteHostOption = None,
    remote_dir: RemoteDirOption = None,
    server_host: Annotated[
        str | None,
        typer.Option("--server-host", help="Host passed to uvicorn on Windows."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65_535, help="Port passed to uvicorn on Windows."),
    ] = None,
) -> None:
    try:
        if server_host is None and port is None:
            start_service(remote_host=remote_host, remote_dir=remote_dir)
        else:
            start_service(
                remote_host=remote_host,
                remote_dir=remote_dir,
                server_host=server_host,
                port=port,
            )
    except CalledProcessError as exc:
        _raise_remote_process_error(exc)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("deploy start complete")


@app.command("deploy-stop")
def deploy_stop(
    *,
    remote_host: RemoteHostOption = None,
    remote_dir: RemoteDirOption = None,
) -> None:
    try:
        stop_service(remote_host=remote_host, remote_dir=remote_dir)
    except CalledProcessError as exc:
        _raise_remote_process_error(exc)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("deploy stop complete")


@app.command("deploy-status")
def deploy_status(
    *,
    remote_host: RemoteHostOption = None,
    remote_dir: RemoteDirOption = None,
) -> None:
    try:
        result = status_service(remote_host=remote_host, remote_dir=remote_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if result.stdout:
        typer.echo(result.stdout.strip())
    raise typer.Exit(result.returncode)


@app.command("deploy-verify-voice-bank")
def deploy_verify_voice_bank(
    *,
    remote_host: RemoteHostOption = None,
    remote_dir: RemoteDirOption = None,
) -> None:
    try:
        result = verify_voice_bank(remote_host=remote_host, remote_dir=remote_dir)
    except CalledProcessError as exc:
        _raise_remote_process_error(exc)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if result.stdout:
        typer.echo(result.stdout.strip())
    typer.echo("voice bank verified")


def _raise_remote_process_error(exc: CalledProcessError) -> NoReturn:
    if exc.stdout:
        typer.echo(str(exc.stdout).strip(), err=True)
    if exc.stderr:
        typer.echo(str(exc.stderr).strip(), err=True)
    raise typer.Exit(exc.returncode) from exc
