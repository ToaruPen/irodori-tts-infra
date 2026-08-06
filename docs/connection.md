# Connection

This project reaches the Windows GPU host through an SSH tunnel over Tailscale.
The HTTP server binds only to Windows loopback; do not expose it through LAN
addresses or public port forwarding.

## Hosts

- Client: macOS
- GPU host: Windows, RTX 4070, reachable through Tailscale
- SSH transport: OpenSSH over the Tailscale address
- Standard HTTP port: `8924`

Keep the concrete host in local `.env` or shell state:

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

The current public capability contract reports free-form delivery captions as
unsupported. Style names such as `calm`, `cheerful`, and `clear` are
server-owned convenience presets, not Irodori-TTS v4 recommendations or public
voice-catalog fields.

Do not restart the service, replace the standard voice bank, or change the
standard generation as part of a repository-only migration. Each operation
requires separate approval and an explicit rollback target.

## Test Repository Clients

The helpers under `/Users/sankenbisha/Dev/Test/tts` use the same standard API.
`TTSEngine` opens an SSH local forward to Windows loopback port `8924`, waits
for `status=ok` and `model_loaded=true`, and closes only the tunnel it created.
Callers send a deployed `speaker` name and one fixed public `style`; model files
and free-form captions remain server-side.

From macOS:

```bash
cd /Users/sankenbisha/Dev/Test/tts
python3 say.py カスミ "こんにちは" --style calm
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
- Synthesis rejects `ref_embed` or a caption: this is expected. Use a deployed
  `speaker` name and a fixed `style` with the standard infra server.
