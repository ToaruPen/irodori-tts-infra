from __future__ import annotations

from typing import Annotated

import typer

from irodori_tts_infra.deploy.remote.bootstrap import bootstrap_remote
from irodori_tts_infra.deploy.remote.service import (
    start_service,
    status_service,
    stop_service,
)
from irodori_tts_infra.deploy.remote.sync import sync_project

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
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("deploy sync complete")


@app.command("deploy-bootstrap")
def deploy_bootstrap(
    *,
    remote_host: RemoteHostOption = None,
    remote_dir: RemoteDirOption = None,
) -> None:
    try:
        bootstrap_remote(remote_host=remote_host, remote_dir=remote_dir)
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
