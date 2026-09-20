# Windows GPU Deployment

Phase 1 deployment copies this repository to the Windows GPU host and starts the
FastAPI server with a PID file. It is not a Windows service yet.

## Local PC Setup

Local use needs neither the old GPU PC nor SSH. Install Git for Windows, `uv`,
`just`, FFmpeg 8 shared libraries, and a current NVIDIA driver. The pinned
TorchCodec supports FFmpeg 4 through 8; do not install FFmpeg 9 for this runtime.
With WinGet, use `winget install --id Gyan.FFmpeg.Shared --version 8.1.2 --exact --source winget`.
Open a new terminal after installation so the FFmpeg directory is on `PATH`.
Verify the actual GPU with `nvidia-smi`.
Use Python 3.11 explicitly instead of a new PC's default Python version.

Clone this repository and upstream Irodori-TTS into sibling directories. The
v4 baseline recorded in this repository is upstream commit
`8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71`; use a dedicated checkout at that
revision. Do not update an existing training checkout in place.

From this repository in PowerShell, with `uv` and `just` on `PATH`:

```powershell
$ErrorActionPreference = 'Stop'
$infra = (Get-Location).Path
$upstream = Join-Path (Split-Path $infra) 'Irodori-TTS'
# just recipes use a POSIX shell; use Git Bash, not the WSL bash launcher.
$env:Path = (Join-Path $env:ProgramFiles 'Git\bin') + ';' + $env:Path
uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
$env:UV_PYTHON = '3.11'
just --list
just sync
if ($LASTEXITCODE -ne 0) { throw 'Development setup failed' }
```

Keep development and GPU dependencies in separate environments. Install the GPU
dependencies through the upstream lockfile so its CUDA package indexes and Git
dependencies are honored. Installing only an upstream package extra from a file
URL does not reproduce those source settings.

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $infra '.runtime-venv'
try {
    uv sync --project $upstream --locked --extra cu128 --python 3.11
    if ($LASTEXITCODE -ne 0) { throw 'GPU dependency installation failed' }
} finally {
    Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
}
$runtimePython = Join-Path $infra '.runtime-venv\Scripts\python.exe'
# The pinned upstream is not built by uv sync; register it after syncing its dependencies.
uv pip install --python $runtimePython --no-deps -e $upstream
if ($LASTEXITCODE -ne 0) { throw 'Upstream package installation failed' }
uv pip install --python $runtimePython -e '.[all]'
if ($LASTEXITCODE -ne 0) { throw 'Infra runtime installation failed' }
uv pip check --python $runtimePython
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency check failed' }
& $runtimePython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
```

Copy `.env.example` to `.env` only when `.env` does not already exist. Set
`IRODORI_DEPLOY_DIR`, `IRODORI_TTS_DIR`, and `VOICE_BANK_DIR` to paths on this PC.
Keep the checkpoint revision and hashes from the example. Restore the actual
`voice_bank_speakers.toml` and its referenced `.speaker.safetensors` files from
backup, preserving relative paths and using embeddings trained for this model.
An empty folder or the example manifest is not a restored voice bank. Set a new
opaque `IRODORI_TTS_RUNTIME_PUBLIC_GENERATION` once the runtime/voice pair is ready.

After restoring and validating the voice bank, start the local API from this
repository (the first start may download model and codec assets):

```powershell
& .runtime-venv\Scripts\python.exe -m uvicorn irodori_tts_infra.server.main:app --env-file .env --host 127.0.0.1 --port 8924
```

From a second terminal, inspect `/health` and `/capabilities` at
`http://127.0.0.1:8924`. Check `model_loaded=true` before synthesis. Keep the server
on loopback. An active process or a successful HTTP response alone does not mean
the model and voice bank are ready.

The full development gate includes POSIX FIFO, symlink, and permission tests;
run it in WSL/Linux, matching CI. Keep the WSL virtual environment separate from
the Windows environments. From this checkout in WSL, with a current `just`:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/irodori-tts-infra/venv"
export UV_PYTHON=3.11
just sync
just check
```

The `remote-speaker-*` recipes retain historical training workspace pins. They
are not new-PC setup commands and do not restore datasets or trained embeddings.

## One-Time Windows Setup

For deployment from another machine, additionally verify on the Windows host:

- OpenSSH Server, reachable from the macOS client with `ssh user@hostname`
- `uv`, available on the SSH user's `PATH`
- Git-compatible Python build tools required by the Irodori runtime
- Irodori-TTS runtime dependencies for the local GPU environment
- HuggingFace authentication when using gated assets:

```powershell
hf auth login
```

Public model assets download on first use. Gated assets additionally require
authentication and acceptance of the corresponding repository's access terms.

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
`IRODORI_TTS_SERVER_HOST`, `IRODORI_TTS_SERVER_PORT`, and `IRODORI_TTS_RUNTIME_*`.
Do not commit this file.

The standard runtime uses the pinned v4.1 Small VoiceDesign checkpoint (v4 Small with a
retrained duration predictor; speaker embeddings must be trained on the same checkpoint):

```env
IRODORI_TTS_RUNTIME_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small
IRODORI_TTS_RUNTIME_CHECKPOINT_REVISION=2b28324dc263ed5e6638b3cf3dd94c82ead07b4b
IRODORI_TTS_RUNTIME_CHECKPOINT_SHA256=c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6
IRODORI_TTS_RUNTIME_CHECKPOINT_TOKENIZER_JSON_SHA256=6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d
IRODORI_TTS_RUNTIME_CHECKPOINT_TOKENIZER_CONFIG_SHA256=d229a271c64de1a7939d20d3665498e873fa91d5ee2edf135d73ec752cb9c9d3
IRODORI_TTS_RUNTIME_CFG_SCALE_CAPTION=3.0
IRODORI_TTS_RUNTIME_WARMUP_STYLE=calm
```

For the current trained speaker embeddings, the Windows runtime voice bank can
point at the Irodori-TTS checkout that owns the speaker files:

```env
VOICE_BANK_DIR=C:\Users\takut\Dev\Irodori-TTS
IRODORI_TTS_SERVER_HOST=127.0.0.1
IRODORI_TTS_SERVER_PORT=8924
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
uv pip install --python .runtime-venv\Scripts\python.exe '.[all]'
uv pip check --python .runtime-venv\Scripts\python.exe
```

After installation, bootstrap inspects the upstream `SamplingRequest` and
`ModelConfig` signatures. It fails when the checkout lacks the speaker and
caption and speaker fields required by the v4 VoiceDesign checkpoint.

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
.runtime-venv\Scripts\python.exe -m uvicorn irodori_tts_infra.server.main:app --host 127.0.0.1 --port $env:IRODORI_TTS_SERVER_PORT
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
