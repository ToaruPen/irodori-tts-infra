# irodori-tts-infra

Infrastructure for Japanese novel TTS using Irodori-TTS v3 base and Speaker
Inversion embeddings.

## Runtime Path

```text
Text -> Irodori-TTS v3 base -> Speaker Inversion ref_embed -> WAV
```

Voice selection comes from `voice_bank_speakers.toml`; RVC and VoiceDesign
captions are not part of the standard path.

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
