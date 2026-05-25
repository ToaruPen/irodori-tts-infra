# Irodori-TTS Optimization Reference

## Inference Speed

| Parameter | Default | Recommended | Effect |
|---|---|---|---|
| `num_steps` | 40 | 20-30 (speed) / 40 (quality) | Euler steps. Fewer = faster, lower quality |
| `compile_model` | False | True (after warmup) | torch.compile() acceleration. May not work on MPS |
| `decode_mode` | sequential | batch | Parallel DACVAE codec decode |
| `context_kv_cache` | True | True | Precompute text/speaker KV cache |

## Audio Quality Tuning

| Parameter | Default | Notes |
|---|---|---|
| `cfg_scale_text` | 3.0 | Higher = more text-faithful, may sound unnatural |
| `cfg_scale_speaker` | 5.0 | Higher = more speaker identity consistency |
| `duration_scale` | 1.0 | Higher = slower/longer delivery |
| `t_schedule_mode` | linear | Use `sway` only when testing sway sampling behavior |
| `sway_coeff` | -1.0 | Sway schedule coefficient |
| `trim_tail` | True | Remove trailing silence |
| `num_candidates` | 1 | Generate N candidates, pick best |

## Emoji Annotations (insert in text)

| Emoji | Effect |
|---|---|
| 👂 | Whisper / close to ear |
| 😮‍💨 | Sigh / exhale |
| ⏸️ | Pause / silence |
| 🤭 | Giggle / chuckle |
| 🥵 | Panting / moaning |
| 📢 | Echo / reverb |
| 😏 | Teasing / flirty |
| 🥺 | Trembling voice / unconfident |
| 🌬️ | Heavy breathing |
| 😮 | Gasp |
| 👅 | Licking / wet sounds |
| 💋 | Lip noise |
| 🫶 | Gentle / soft |
| 😭 | Sobbing / crying |
| 😱 | Scream / shriek |
| 😪 | Sleepy / drowsy |
| ⏩ | Fast speech |
| 📞 | Phone / speaker filter |
| 🐢 | Slow speech |
| 🥤 | Swallowing |
| 🤧 | Cough / sneeze / sniffling |
| 😒 | Tongue click / tch |
| 😰 | Flustered / stammering |
| 😆 | Joyful / happy |
| 😠 | Angry / sulky |
| 😲 | Surprised / amazed |
| 🥱 | Yawn |
| 😖 | Pained / distressed |
| 😟 | Worried |
| 🫣 | Embarrassed / shy |
| 🙄 | Exasperated |
| 😊 | Happy / cheerful |
| 👌 | Acknowledgment / nodding |
| 🙏 | Pleading |
| 🥴 | Drunk |
| 🎵 | Humming |
| 🤐 | Muffled voice |
| 😌 | Relieved / content |
| 🤔 | Questioning / thinking |

Repeating an emoji strengthens its effect.

## Speaker Inversion Embeddings

Use `voice_bank_speakers.toml` to select `.speaker.safetensors` files:

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
```

The synthesis path requires a narrator embedding and an explicit dialogue
speaker that exists in the manifest. There is no generic fallback voice.

## Text Preprocessing (auto-applied by Irodori-TTS)

- Fullwidth `？！` → halfwidth `?!`
- `～〜` → `ー`
- `...` `..` → `…`, max 2 consecutive `…`
- Fullwidth spaces/tabs removed
- NFKC normalization

**Known limitation**: Kanji reading accuracy is relatively weak. Convert complex kanji to hiragana/katakana for better results.

## MPS (Apple Silicon) Notes

- Use `model_device="mps"`, `codec_device="cpu"` (safer)
- Precision: `fp32` recommended (bf16 may be unstable on MPS)
- `compile_model=True` untested on MPS — disable if issues occur
- KV cache likely works on MPS
