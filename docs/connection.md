# Connection

This project reaches the Windows GPU host through Tailscale. Do not rely on LAN
addresses, public port forwarding, or non-Tailscale hostnames for normal agent
work.

## Hosts

- Client: macOS
- GPU host: Windows, RTX 4070, reachable through Tailscale
- SSH transport: OpenSSH over the Tailscale address
- Default HTTP port: `8923`

Keep the concrete host in local `.env` or shell state:

```env
IRODORI_REMOTE_HOST=user@100.x.y.z
IRODORI_TTS_SERVER_PORT=8923
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
curl "http://100.x.y.z:${IRODORI_TTS_SERVER_PORT}/health"
```

If the health check times out, first verify SSH access. Then check whether the
Windows process is listening on `IRODORI_TTS_SERVER_PORT`. Ensure the variable
is defined before running the check.

```powershell
netstat -ano | findstr ":$env:IRODORI_TTS_SERVER_PORT"
```

## Standard Infra Server

The standard path is the deployed `irodori_tts_infra` FastAPI server on the
Windows host.

Start and stop it from the macOS worktree with the deploy CLI:

```bash
irodori-tts-deploy deploy-sync
irodori-tts-deploy deploy-bootstrap
irodori-tts-deploy deploy-verify-voice-bank
irodori-tts-deploy deploy-start
irodori-tts-deploy deploy-status
irodori-tts-deploy deploy-stop
```

The Windows runtime `.env` must point at the local voice bank that contains
`voice_bank_speakers.toml` and the referenced `.speaker.safetensors` files:

```env
VOICE_BANK_DIR=C:\Users\takut\Dev\Irodori-TTS
IRODORI_TTS_SERVER_PORT=8923
```

The standard HTTP API accepts portable `speaker` names. It resolves those names
server-side through `voice_bank_speakers.toml`; public clients should not send
raw `ref_embed` paths to the infra server.

## Legacy `say.py` Server

Local helper scripts under `/Users/sankenbisha/Dev/Test/tts`, including
`say.py`, use a separate legacy `remote_server.py` contract. That server accepts
direct `ref_embed` values and is not the same process as the standard infra
FastAPI server.

Start the legacy server on Windows only when those helper scripts need it:

```powershell
cd C:\Users\takut\Dev\Irodori-TTS
$env:IRODORI_TTS_SPEAKER_DIR='C:\Users\takut\Dev\Irodori-TTS\speakers'
.\.venv\Scripts\python.exe remote_server.py --port 8923
```

Then from macOS:

```bash
cd /Users/sankenbisha/Dev/Test/tts
python3 say.py カスミ "こんにちは" --ref-embed kasumi.speaker.safetensors
```

For this legacy path, pass the speaker file name relative to
`IRODORI_TTS_SPEAKER_DIR`, such as `kasumi.speaker.safetensors`. Do not pass
`speakers/kasumi.speaker.safetensors` unless `IRODORI_TTS_SPEAKER_DIR` is
changed to the parent directory.

## Troubleshooting

- SSH fails: confirm Tailscale is connected on both machines and use the
  Tailscale address in `IRODORI_REMOTE_HOST`.
- HTTP health times out: confirm the correct server process is running and
  listening on `8923`.
- `say.py` cannot find a speaker: confirm the legacy server is running and the
  requested `.speaker.safetensors` exists under
  `C:\Users\takut\Dev\Irodori-TTS\speakers`.
- Infra synthesis rejects `ref_embed`: this is expected. Use `speaker` names
  with the standard infra server, or use the legacy `say.py` server only for
  direct speaker-file experiments.
