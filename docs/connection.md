# Connection

On the Windows GPU host itself, clients connect directly to loopback. Remote
clients reach the Windows GPU host through an SSH tunnel over Tailscale.
The HTTP server binds only to Windows loopback; do not expose it through LAN
addresses or public port forwarding.

## Hosts

- Client: local Windows, or a remote machine with an SSH client
- GPU host: Windows with a CUDA-capable NVIDIA GPU; inspect it with `nvidia-smi`
- Remote transport: OpenSSH over the current host's Tailscale address
- Standard HTTP port: `8924`

For local setup, follow [Windows deployment](deploy/windows.md#local-pc-setup).
No SSH server or `IRODORI_REMOTE_HOST` is needed for local use. Check the API
with `Invoke-RestMethod http://127.0.0.1:8924/health` in PowerShell.

For remote use, keep the concrete host in local `.env` or shell state:

```env
IRODORI_REMOTE_HOST=user@100.x.y.z
IRODORI_TTS_SERVER_PORT=8924
```

`IRODORI_REMOTE_HOST` must be the SSH target, including the Windows user name.
Use a Tailscale IP address or a Tailscale MagicDNS name. Do not commit a
machine-specific `.env`.

## Quick Checks

From macOS:

```bash
ssh "$IRODORI_REMOTE_HOST" "hostname"
```

If `IRODORI_REMOTE_HOST` is not exported, pass the target explicitly:

```bash
ssh user@100.x.y.z "hostname"
```

Check the standard infra FastAPI server:

```bash
ssh -N \
  -L "${IRODORI_TTS_SERVER_PORT}:127.0.0.1:${IRODORI_TTS_SERVER_PORT}" \
  "$IRODORI_REMOTE_HOST"
```

Keep that command running, then check the forwarded endpoint from another
terminal:

```bash
curl "http://127.0.0.1:${IRODORI_TTS_SERVER_PORT}/health"
```

If the health check times out, first verify SSH access. Then check whether the
Windows process is listening on loopback at `IRODORI_TTS_SERVER_PORT` and that
the local forwarding command is still running. Ensure the variable is defined
before running the check.

```powershell
netstat -ano | findstr ":$env:IRODORI_TTS_SERVER_PORT"
```

## Standard Infra Server

The standard path is the deployed `irodori_tts_infra` FastAPI server on the
Windows host.

Start and stop it from the macOS worktree with the deploy CLI:

```bash
just deploy deploy-sync
just deploy deploy-bootstrap
just deploy deploy-verify-voice-bank
just deploy deploy-start
just deploy deploy-status
just deploy deploy-stop
```

The Windows runtime `.env` must point at the local voice bank that contains
`voice_bank_speakers.toml` and the referenced `.speaker.safetensors` files:

```env
VOICE_BANK_DIR=C:\Users\takut\Dev\Irodori-TTS
IRODORI_TTS_SERVER_HOST=127.0.0.1
IRODORI_TTS_SERVER_PORT=8924
```

The standard HTTP API publishes its active generation, safe readiness, and
portable voice catalog at `GET /capabilities`. The generation is an opaque token
for the runtime and voice-bank pair. Clients should cache it and send it with
the selected portable voice ID so a runtime change fails closed instead of
silently choosing another voice.

`voice_bank_speakers.toml` owns `voice_id`, display `label`, legacy `aliases`,
and the optional `default` marker. Aliases must resolve uniquely; an ambiguous
manifest is invalid. Public clients must not receive or send raw `ref_embed`
paths, checkpoint or tokenizer identifiers, hashes, or other model-artifact
metadata.

The public capability contract reports whether free-form delivery captions are
supported and their maximum length. Style names such as `calm`, `cheerful`,
`clear`, `alluring`, and `lewd` are server-owned convenience presets, not
Irodori-TTS enums or public voice-catalog fields. A request may use either one
non-neutral preset or one `delivery_caption`, never both. Delivery-caption
support describes the API contract of the pinned v4.1 VoiceDesign runtime, so it
is reported as supported regardless of model load state; use `readiness` to tell
whether synthesis is currently available. The capability value changed without a
`contract_version` bump, so clients built against the earlier unsupported shape
must be rebuilt.

The v4 base model occasionally renders explicit words as a machine censor beep
(a steady tone near 710 Hz or 1 kHz), even with speaker embeddings trained on
beep-free audio. The server inspects every synthesized segment and regenerates
a beeping one with the next seed, up to four attempts in total. If every attempt
beeps, the segment fails closed as `backend_unavailable`; the server log records
`censor_beep_detected` for each rejected attempt, without the request text.

Do not restart the service, replace the standard voice bank, or change the
standard generation as part of a repository-only migration. Each operation
requires separate approval and an explicit rollback target.

## Test Repository Clients

The helpers under `/Users/sankenbisha/Dev/Test/tts` use the same standard API.
`TTSEngine` opens an SSH local forward to Windows loopback port `8924`, waits
for `status=ok` and `model_loaded=true`, and closes only the tunnel it created.
Callers send a deployed `speaker` name and either one fixed public `style` or one
validated `delivery_caption`; model files and raw upstream caption fields remain
server-side.

From macOS:

```bash
cd /Users/sankenbisha/Dev/Test/tts
python3 say.py カスミ "こんにちは" --style calm
python3 say.py カスミ "こんばんは" --delivery-caption \
  "雨の夜、親しい相手の耳元で囁くように、吐息を少し交えて話す。"
python3 read_aloud.py ../chat/<setting>/<scenario>/turns/turn_XX.md
```

Set `IRODORI_TTS_BASE_URL` or pass `--base-url` only when a standard endpoint is
already reachable and tunneling should be bypassed. Do not point these clients
at an upstream v3 `remote_server.py` process.

## Troubleshooting

- SSH fails: confirm Tailscale is connected on both machines and use the
  Tailscale address in `IRODORI_REMOTE_HOST`.
- HTTP health times out: confirm the correct server process is listening on
  the configured loopback port (`8924` for the pinned Test deployment) and the
  local SSH forwarding command is still running.
- `say.py` cannot find a speaker: compare the requested character name with
  `GET /capabilities` and the active server-side voice-bank manifest.
- Synthesis of one sentence returns `backend_unavailable` while others succeed:
  check the server log for `censor_beep_detected`. Four consecutive beeping
  attempts reject the segment instead of playing the beep.
- Synthesis rejects `ref_embed` or raw `caption`: this is expected. Use a
  deployed `speaker` name with either a fixed `style` or `delivery_caption`.
- Synthesis rejects `delivery_caption`: verify that it is 1–300 characters,
  contains no control characters or Unicode line/paragraph separators, and is
  not combined with a non-neutral style.
