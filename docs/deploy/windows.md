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
`IRODORI_TTS_SERVER_PORT`, `IRODORI_TTS_RUNTIME_*`, and
`IRODORI_TTS_PATH_TEMP_WAV_DIR`. Do not commit this file.

## Expected Layout

After `deploy-sync`, the Windows directory should look like this:

```text
C:\Users\user\irodori-tts-infra\
  .env
  .env.example
  .uvicorn.pid
  README.md
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
creates the remote directory with `ssh` and copies `src/`, `README.md`,
`pyproject.toml`, and `.env.example` with `scp`.

`deploy-bootstrap` creates a dedicated runtime venv and installs upstream
Irodori-TTS plus this package:

```powershell
uv venv '.runtime-venv' --python '3.11' --clear
uv pip install --python .runtime-venv\Scripts\python.exe 'Irodori-TTS[cu128] @ file:///C:/path/to/Irodori-TTS'
uv pip install --python .runtime-venv\Scripts\python.exe '.[server,irodori]'
uv pip check --python .runtime-venv\Scripts\python.exe
```

Override the upstream checkout, Python version, or Torch backend extra with:

```bash
irodori-tts-deploy deploy-bootstrap \
  --irodori-tts-dir 'C:/path/to/Irodori-TTS' \
  --python-version 3.11 \
  --torch-backend-extra cu128
```

`deploy-start` loads `.env` from the deployed repository root into the process
environment, then launches:

```powershell
.runtime-venv\Scripts\python.exe -m uvicorn irodori_tts_infra.server.main:app --host 0.0.0.0 --port 8923
```

The PID-file wrapper is intentionally minimal. If the server fails during import
or startup, inspect the Windows shell environment and run the `.runtime-venv`
Python command manually for the full error output.

## Voice Bank

The deployed voice bank must include `voice_bank_speakers.toml` and the
referenced `.speaker.safetensors` files:

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
```

RVC training is superseded for the standard path.
