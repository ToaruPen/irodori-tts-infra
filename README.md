# irodori-tts-infra

Infrastructure for Japanese TTS using Irodori-TTS v4 VoiceDesign with Speaker
Inversion embeddings.

## Runtime Path

```text
Text + fixed style -> Irodori-TTS v4 VoiceDesign
                   + Speaker Inversion ref_embed -> WAV
```

Voice selection comes from `voice_bank_speakers.toml`. Clients choose one of
`neutral`, `calm`, `cheerful`, or `clear`; the server maps it to a fixed
VoiceDesign caption. Arbitrary captions and RVC are not part of the standard
path.

The default checkpoint is
`Aratako/Irodori-TTS-v4-Small` at its repository-pinned revision and hashes.

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
Free-form delivery captions are not supported by the current public contract;
`neutral`, `calm`, `cheerful`, and `clear` remain server-owned convenience
presets rather than Irodori-TTS v4 enums.

Changing the standard generation, replacing its voice bank, or restarting the
service is an explicit operational step. Repository checks and training output
must never promote those changes automatically.

## Development

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
