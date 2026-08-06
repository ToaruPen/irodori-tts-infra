# Superseded: Windows GPU RVC Training SOP

This SOP is retained only as a historical pointer. It is not an active deploy
or voice-bank procedure for this repository.

The standard runtime path now uses Irodori-TTS v4 VoiceDesign with Speaker Inversion
embeddings. Voice identity is selected with `voice_bank_speakers.toml` and
`.speaker.safetensors` references, then sent directly to the Irodori backend.
There is no standard RVC conversion stage.

Use the current deployment guide and voice-bank manifest contract instead:

- [Windows GPU deployment](windows.md)
- [Irodori v4 VoiceDesign architecture](../irodori-rvc-architecture.md)

Historical RVC decisions may remain in ADRs and planning notes, but they must
not be used as current operator instructions.
