# Windows GPU Deployment

Phase 1 deployment copies this repository to the Windows GPU host and starts the
FastAPI server with a PID file. It is not a Windows service yet.

## One-Time Windows Setup

Install and verify these manually on the Windows host:

- OpenSSH Server, reachable from the macOS client with `ssh user@hostname`
- `uv`, available on the SSH user's `PATH`
- Git-compatible Python build tools required by the Irodori runtime
- Irodori-TTS runtime dependencies for the local GPU environment
- HuggingFace authentication:

```powershell
huggingface-cli login
```

The first model download and any HuggingFace gated-model checks are still manual
for Phase 1.

## Environment

On macOS, keep deployment connection settings in the local `.env` or shell:

```env
IRODORI_REMOTE_HOST=user@hostname
IRODORI_DEPLOY_DIR=C:\Users\user\irodori-tts-infra
```

On Windows, place the runtime `.env` in the deployed repository root:

```text
C:\Users\user\irodori-tts-infra\.env
```

The Windows `.env` should contain server/runtime settings such as
`IRODORI_SERVER_PORT`, `IRODORI_RUNTIME_*`, and
`IRODORI_PATH_TEMP_WAV_DIR`. Do not commit this file.

## Expected Layout

After `deploy-sync`, the Windows directory should look like this:

```text
C:\Users\user\irodori-tts-infra\
  .env
  .env.example
  .uvicorn.pid
  pyproject.toml
  src\
    irodori_tts_infra\
```

`.uvicorn.pid` is created by `deploy-start` and removed by `deploy-stop`.

## Commands

Run these from the macOS worktree:

```bash
irodori-tts-deploy deploy-sync
irodori-tts-deploy deploy-bootstrap
irodori-tts-deploy deploy-start
irodori-tts-deploy deploy-status
irodori-tts-deploy deploy-stop
```

`deploy-sync` prefers `rsync` over SSH. If `rsync` is unavailable locally, it
creates the remote directory with `ssh` and copies `src/`, `pyproject.toml`, and
`.env.example` with `scp`.

`deploy-bootstrap` runs this on the Windows host:

```powershell
uv sync --extra server --extra irodori
```

`deploy-start` launches:

```powershell
uv run uvicorn irodori_tts_infra.server.main:app --host 0.0.0.0 --port 8923
```

The PID-file wrapper is intentionally minimal. If the server fails during import
or startup, inspect the Windows shell environment and run the `uv run uvicorn ...`
command manually for the full error output.

## RVC Training

See the [RVC training SOP](rvc-training.md) for per-character RVC model training.
