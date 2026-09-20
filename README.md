# irodori-tts-infra

Infrastructure for Japanese TTS using Irodori-TTS v4.1 VoiceDesign with Speaker
Inversion embeddings.

## Runtime Path

```text
Text + (fixed style | delivery_caption) -> Irodori-TTS v4.1 VoiceDesign
                                        + Speaker Inversion ref_embed -> WAV
```

Voice selection comes from `voice_bank_speakers.toml`. Clients may choose one of
`neutral`, `calm`, `cheerful`, `clear`, `alluring`, or `lewd`; the server maps it
to a fixed VoiceDesign caption. Alternatively, clients may send one Japanese
free-form `delivery_caption` with `style=neutral`. Preset captions and free-form
captions are separate modes and are never merged. The raw upstream `caption`
field and RVC are not part of the standard path.

The default checkpoint is
`Aratako/Irodori-TTS-v4.1-Small` at its repository-pinned revision and hashes.

## Runtime capabilities

`GET /capabilities` is the public source of truth for the active runtime
generation, readiness, and portable voice catalog. The generation is an opaque
token identifying one runtime and voice-bank pair; set
`IRODORI_TTS_RUNTIME_PUBLIC_GENERATION` to a new value whenever either member of
the pair changes.

Portable voice metadata belongs in `voice_bank_speakers.toml`. Entries may
define `voice_id`, `label`, `aliases`, and `default`. Aliases exist only to
resolve legacy client names and must be globally unambiguous. Ambiguous IDs or
aliases make startup fail closed.

Public responses never expose checkpoint, tokenizer, hash, or embedding paths.
The capability catalog reports whether request-scoped `delivery_caption` is
supported and its maximum length. The six public style names remain server-owned
convenience presets rather than Irodori-TTS enums.

Example preset request:

```json
{"text":"こんばんは。","speaker":"カスミ","style":"alluring"}
```

Example free-form request:

```json
{
  "text": "待っていたよ。",
  "speaker": "カスミ",
  "style": "neutral",
  "delivery_caption": "雨の夜、親しい相手の耳元で囁くように、吐息を少し交えて話す。"
}
```

`delivery_caption` is stripped, must contain 1–300 Unicode code points, and
must not contain control characters or Unicode line/paragraph separators.
Sending it with a non-neutral `style` returns a validation error.

Changing the standard generation, replacing its voice bank, or restarting the
service is an explicit operational step. Repository checks and training output
must never promote those changes automatically.

## Development

For a new Windows GPU PC, start with [local setup](docs/deploy/windows.md#local-pc-setup).
Local use does not require the historical GPU host or an SSH tunnel.

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy
```

## Voice Bank

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
```
