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
IRODORI_REMOTE_HOST=user@100.x.y.z
IRODORI_DEPLOY_DIR=C:\Users\user\irodori-tts-infra
```

Use the Tailscale SSH address for `IRODORI_REMOTE_HOST`. See
[`docs/connection.md`](../connection.md) for the connection model, health
checks, and the difference between the standard infra server and the legacy
`say.py` helper server.

On Windows, place the runtime `.env` in the deployed repository root:

```text
C:\Users\user\irodori-tts-infra\.env
```

The Windows `.env` should contain server/runtime settings such as
`IRODORI_TTS_SERVER_PORT`, `IRODORI_TTS_RUNTIME_*`, and
`IRODORI_TTS_PATH_TEMP_WAV_DIR`. Do not commit this file.

For the current trained speaker embeddings, the Windows runtime voice bank can
point at the Irodori-TTS checkout that owns the speaker files:

```env
VOICE_BANK_DIR=C:\Users\takut\Dev\Irodori-TTS
IRODORI_TTS_SERVER_PORT=8923
```

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
irodori-tts-deploy deploy-verify-voice-bank
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
.runtime-venv\Scripts\python.exe -m uvicorn irodori_tts_infra.server.main:app --host 0.0.0.0 --port $env:IRODORI_TTS_SERVER_PORT
```

The PID-file wrapper is intentionally minimal. If the server fails during import
or startup, inspect the Windows shell environment and run the `.runtime-venv`
Python command manually for the full error output.

`deploy-verify-voice-bank` runs the deployed runtime Python on the Windows host,
loads the deployed `.env`, resolves `VOICE_BANK_DIR` or
`VOICE_BANK_SPEAKER_MANIFEST`, and validates that every manifest entry points to
an existing `.speaker.safetensors` file. Run it after `deploy-bootstrap` and
before `deploy-start`.

## Voice Bank

The deployed voice bank must include `voice_bank_speakers.toml` and the
referenced `.speaker.safetensors` files:

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
```

The actual `.speaker.safetensors` files stay outside Git. For the trained
OOPPEENN/Kasumi set, copy
`docs/deploy/voice_bank_speakers.ooppeenn.example.toml` to
`C:\Users\takut\Dev\Irodori-TTS\voice_bank_speakers.toml` and keep the files
under `C:\Users\takut\Dev\Irodori-TTS\speakers\`.

The standard FastAPI path accepts `speaker` names and rejects public `ref_embed`
values. Older local helper scripts under `/Users/sankenbisha/Dev/Test/tts` may
still pass `ref_embed` directly to their own test server; do not treat that as
the infra server contract.

RVC training is superseded for the standard path.
