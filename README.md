# irodori-tts-infra

Infrastructure for Japanese TTS using Irodori-TTS v3 VoiceDesign with Speaker
Inversion embeddings.

## Runtime Path

```text
Text + fixed style -> Irodori-TTS v3 VoiceDesign
                   + Speaker Inversion ref_embed -> WAV
```

Voice selection comes from `voice_bank_speakers.toml`. Clients choose one of
`neutral`, `calm`, `cheerful`, or `clear`; the server maps it to a fixed
VoiceDesign caption. Arbitrary captions and RVC are not part of the standard
path.

The default checkpoint is
`Aratako/Irodori-TTS-600M-v3-VoiceDesign`.

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
