# Irodori-TTS v3 Speaker Inversion Architecture

> Supersedes the previous VoiceDesign + RVC standard pipeline.

## Standard Pipeline

```text
Text + speaker tag
  -> Irodori-TTS v3 base
  -> Speaker Inversion embedding (.speaker.safetensors)
  -> Multi-metric quality gate
  -> Playback / cache
```

The default checkpoint is `Aratako/Irodori-TTS-500M-v3`. Each narrator or
character voice is selected by a Speaker Inversion embedding, not by a
VoiceDesign caption and not by an RVC conversion stage.

## Voice Bank Manifest

The standard voice bank manifest is `voice_bank_speakers.toml`:

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
```

Paths are relative to the manifest file. When `characters.md` is present,
manifest character names must also be present in that markdown file.

Missing narrator embeddings, missing dialogue speakers, and unknown character
speakers are configuration errors. The pipeline does not fall back to a generic
voice.

## Runtime Contract

Public HTTP requests carry portable speaker identity:

- `text`
- `speaker` for dialogue, omitted for narrator
- `num_steps`
- `cfg_scale_text`
- `cfg_scale_speaker`
- `seed`
- `duration_scale`
- `num_candidates`
- `t_schedule_mode`
- `sway_coeff`

The server resolves `speaker` against its own `voice_bank_speakers.toml` before
the backend call. The backend boundary then receives the resolved `ref_embed`
plus the same v3 sampling fields. Public HTTP clients must not send local
`ref_embed` paths because client and GPU server filesystems may differ.

The backend must not send `caption`, `cfg_scale_caption`, or `no_ref` in the
standard path.

## Superseded RVC Material

Earlier design notes used VoiceDesign for expressive source audio and RVC for
identity correction. That path is no longer the standard architecture for this
repository. Historical RVC docs may remain under ADR or old planning files as
decision history only; implementation, tests, and new operational docs should
target v3 base + Speaker Inversion.

## Quality Gate Direction

Identity metrics still matter, but the target is now the v3 Speaker Inversion
output directly. Future quality gates should compare generated audio against the
selected character embedding/reference corpus and fail fast on missing speaker
configuration before synthesis begins.
