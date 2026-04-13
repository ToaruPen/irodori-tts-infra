# Irodori-TTS + RVC Voice Pipeline Architecture

> Design spec for a novel TTS reader with character voice consistency.
> Decided: 2026-04-13 through research, experimentation, and Codex collaboration.

## Problem Statement

Irodori-TTS VoiceDesign produces the highest quality Japanese speech synthesis with rich expressiveness (emoji annotations, caption-based control). However, it **cannot maintain voice identity** across different text content. Experiments show cosine similarity as low as 0.07 between normal dialogue and moaning text from the "same" caption+seed configuration.

The root cause is structural: VoiceDesign's architecture replaces the speaker conditioning branch with a caption encoder (`use_speaker_condition=False`). Voice identity is entangled with text content and cannot be decoupled through parameter tuning alone.

## Architecture Decision

**Separate content generation from voice identity:**

- **Irodori-TTS VoiceDesign** = content, prosody, emotion, expressiveness
- **RVC (Retrieval-based Voice Conversion)** = character voice identity

```
Input text + speaker tag + emoji annotations
  → Irodori-TTS VoiceDesign (expressive source audio generation)
  → RVC per-character model (voice identity conversion)
  → Multi-metric quality gate (automated pass/fail)
  → Playback / cache
```

## Why This Architecture

### Alternatives Evaluated and Rejected

| Approach | Why rejected |
|----------|-------------|
| VoiceDesign seed+caption lock | Cross-text identity drift: cosine 0.07-0.66 (Experiment C) |
| Base v2 + ref_wav | No caption support; moaning text likely still drifts even with reference anchor |
| Base v2 + voice bank (per-state refs) | Fragments identity across states; complex curation for uncertain gain |
| SBV2 fine-tuning | 2-5 days training, lower quality than Irodori, extensive manual tuning ("craftsman work") |
| VoiceDesign + candidate reranking | 8-16x synthesis cost; doesn't solve fundamental entanglement |

### Why RVC Wins

1. Directly decouples expressiveness from identity — the only approach that addresses the root cause
2. Preserves 100% of Irodori's expressiveness (emoji, prosody, emotion)
3. Lightweight training: 10-30 min of audio, hours not days
4. Real-time inference on RTX 4070
5. Training data available: Japanese-Eroge-Voice-V2 (MIT license, both SFW+NSFW per character)

## Experimental Evidence

### Experiment Setup
- Model: `Aratako/Irodori-TTS-500M-v2-VoiceDesign`
- Speaker embedding: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- Caption: "若い女性が、少し低めで落ち着いた声で、穏やかに話している。息遣いは自然で、音質はクリア。"

### Results Summary

| Experiment | Condition | Mean Cosine | Min | Max | Assessment |
|------------|-----------|-------------|-----|-----|------------|
| A | Same seed + same text (5 runs) | 1.000 | 1.000 | 1.000 | Perfect reproduction |
| B | Different seeds, same text (10 seeds) | 0.592 | 0.228 | 0.782 | High drift |
| C | Same seed, different texts (10 texts) | 0.366 | 0.066 | 0.664 | **Highest drift** |
| D | Same seed, different emoji (7 variants) | 0.732 | 0.470 | 0.968 | Emoji-dependent |

### Key Findings

1. **Seed fixation provides perfect determinism** for identical inputs (Exp A: 1.000)
2. **Text content is the dominant drift factor** — moaning text "あっ…んっ…" yields 0.07 cosine against normal dialogue (Exp C)
3. **Most emoji are safe**: 😊happy/😠angry/😰flustered stay 0.93-0.97 (Exp D)
4. **Problematic emoji**: 🥵aroused (0.47), 👂whisper (0.68), 😢sad (0.66) cause significant drift (Exp D)
5. **VoiceDesign ignores ref_wav by design**: source code confirms `use_speaker_condition = not use_caption_condition`

## Irodori-TTS Model Architecture (Confirmed)

| Model | Voice Control | ref_wav | Caption | Best For |
|-------|--------------|---------|---------|----------|
| Base v2 (`Irodori-TTS-500M-v2`) | Reference audio | **Active** | Disabled | Voice cloning |
| VoiceDesign (`...v2-VoiceDesign`) | Caption text | **Ignored** | Active | Expressive generation |

Both models share: emoji annotation system, text preprocessing, DACVAE codec, seed parameter.

## RVC Integration

### Training Data Source

**Japanese-Eroge-Voice-V2** (HuggingFace, MIT license)
- 1.03M clips, 2657 hours total, 520 parquet shards (~440MB each, ~228GB total)
- Per-character filtering via `char_id` field
- Top characters: 8-13 hours of audio (far more than RVC needs)
- Contains both normal dialogue and explicit/moaning content per character
- Transcriptions included

**Data extraction strategy**: Stream shards one at a time, extract target char_id clips, delete shard. Keeps disk usage under 1GB per character.

### RVC Training Requirements

- Audio: 10-30 minutes of clean, single-speaker clips
- Format: WAV, mono, 16-48kHz
- Hardware: RTX 4070 12GB sufficient
- Training time: Hours, not days
- Should include diverse content: normal speech, emotional, breathy, moaning

### Per-Character RVC Model

Each major character gets:
1. Curated audio dataset (from eroge voice data or other source)
2. Trained RVC model
3. Speaker embedding prototype (centroid of training data embeddings)
4. State-specific prototypes (normal, emotional, intimate, moan)

## Quality Gate System

### Multi-Metric Battery

| Layer | Metric | Role | Library |
|-------|--------|------|---------|
| Primary identity | WavLM-large SV cosine | TTS standard (F5-TTS, Seed-TTS-eval) | `seed-tts-eval` |
| Secondary identity | ECAPA-TDNN cosine | Disagreement detection (two-model consensus) | `speechbrain` |
| Relative margin | Target - nearest other character | "Who does it sound like?" ranking | Custom |
| F0 statistics | Median/range/std/voiced ratio | Pitch consistency (missed by embeddings) | `parselmouth` |
| Quality floor | UTMOS / DNSMOS | Minimum audio quality | `SpeechMOS` |
| Style | Speaking rate, pause ratio, energy | Character cadence consistency | `librosa` |

### State-Aware Thresholds

| State | Cosine Gate | Notes |
|-------|------------|-------|
| Normal dialogue | ≥ 0.80 | Compare against neutral prototype |
| Happy / angry / flustered | ≥ 0.80 | Compare against neutral prototype |
| Sad / whisper | ≥ 0.65-0.70 | Softer gate due to acoustic difference |
| Aroused / intimate | ≥ 0.60-0.70 | Compare against intimate prototype |
| Short moans | ≥ 0.50-0.60 | Soft warning + spot review; compare against moan prototype |
| Long moans | N/A | Compare against moan prototype, not neutral |

### Moaning/Non-Speech Handling

Speaker embeddings trained on normal speech are unreliable for non-linguistic vocalizations. For moan segments:
- Downgrade identity gate to soft warning
- Compare against moan-specific prototype (not neutral)
- Check consistency with adjacent dialogue segments
- Require two embedding models to agree before hard-failing

## Speaker Assignment

| Speaker Type | Pipeline | Rationale |
|-------------|----------|-----------|
| Narrator | VoiceDesign direct | Identity variety acceptable; expressiveness priority |
| Major characters | VoiceDesign → RVC | Identity consistency required |
| Minor/one-off characters | VoiceDesign seed+caption lock | Cost-effective; minor drift acceptable |

## Automation

### Human-Required (Once)

1. **Character voice selection** — listen to samples, choose preferred voice (~30 min)
2. **Threshold calibration** — rate 20-50 samples as ○/×, agent derives thresholds (~30-60 min)
3. **RVC quality spot-check** — verify conversion artifacts are acceptable (~15 min)

### Fully Automated (Agent Loop)

- Parameter exploration (seed, cfg_scale, num_steps grid search)
- Audio generation (Irodori → RVC pipeline)
- Multi-metric scoring
- Threshold-based pass/fail
- RVC training data quality filtering
- RVC training execution and checkpoint comparison
- Failed sample retry with alternate parameters
- Cache management

## Hardware & Network

- **GPU server**: Windows PC, RTX 4070 12GB, Tailscale VPN
- **Client**: macOS (Apple Silicon)
- **VRAM budget**: Irodori (~4-6GB) + RVC (~2-4GB) = fits in 12GB simultaneously
- **Latency**: Irodori ~1.3s/segment + RVC real-time conversion

## Key References

- Irodori-TTS: https://github.com/Aratako/Irodori-TTS
- VoiceDesign model: https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign
- Base v2 model: https://huggingface.co/Aratako/Irodori-TTS-500M-v2
- RVC WebUI: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Japanese-Eroge-Voice-V2: https://huggingface.co/datasets/NandemoGHS/Japanese-Eroge-Voice-V2
- Seed-TTS-eval (WavLM metrics): https://github.com/BytedanceSpeech/seed-tts-eval
- SpeechBrain ECAPA: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- Emoji annotations: see `docs/irodori-tts-optimization.md`

## Comparison Models Tested

| Model | Quality | Identity | Speed | Verdict |
|-------|---------|----------|-------|---------|
| Irodori VoiceDesign | Excellent | Poor (0.07-1.0) | 1.3s/seg | Best generation, needs RVC for identity |
| zonoko-nsfw (SBV2) | Good | Fixed (single voice) | 0.25s/seg | Fast but less expressive than Irodori |
| Irodori Base v2 | Good | Better (untested numerically) | ~1.3s/seg | No caption control |

## Next Steps

1. Stop full dataset download; use shard-streaming extraction for target character
2. Extract 10-30 min of character audio from Japanese-Eroge-Voice-V2
3. Install and test RVC on Windows GPU machine
4. Train RVC model on extracted character audio
5. Test Irodori → RVC pipeline end-to-end
6. Measure identity consistency with multi-metric battery
7. Calibrate thresholds with human listening
8. Build production pipeline with automated quality gates
