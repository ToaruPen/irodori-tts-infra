# Irodori-TTS v4 VoiceDesign + Speaker Inversion Architecture

> The filename is retained for historical links. RVC and the earlier v3 standard
> paths are superseded; the current path retains server-side Speaker Inversion.

## Standard Pipeline

```text
Text + speaker tag + fixed style
  -> Irodori-TTS v4 VoiceDesign
  + Speaker Inversion embedding (.speaker.safetensors)
  -> Multi-metric quality gate
  -> Playback / cache
```

The default checkpoint is `Aratako/Irodori-TTS-v4-Small`. Each
narrator or character voice is selected by a Speaker Inversion embedding.
The fixed VoiceDesign caption controls delivery rather than identity, and there
is no RVC conversion stage.

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
- `style`
- `cfg_scale_caption`
- `cfg_scale_speaker`
- `seed`
- `duration_scale`
- `num_candidates`
- `t_schedule_mode`
- `sway_coeff`

The server resolves `speaker` against its own `voice_bank_speakers.toml` before
the backend call. The backend boundary then receives the resolved `ref_embed`
plus the same private sampling fields. Public HTTP clients must not send local
`ref_embed` paths because client and GPU server filesystems may differ.

Public clients must not send `caption` or `no_ref`. The server maps `style` to
a fixed caption and sends it with `cfg_scale_caption` to the backend.

## Superseded RVC Material

Earlier design notes used VoiceDesign for expressive source audio and RVC for
identity correction. RVC is no longer part of the standard architecture.
Historical RVC and v3 documents may remain as decision history;
implementation and current operational docs target VoiceDesign plus Speaker
Inversion.

## Quality Gate Direction

Identity metrics still matter, but the target is now the v4 Speaker Inversion
output directly. Future quality gates should compare generated audio against the
selected character embedding/reference corpus and fail fast on missing speaker
configuration before synthesis begins.
