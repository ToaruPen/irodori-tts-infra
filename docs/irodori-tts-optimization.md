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
| `cfg_scale_caption` | 3.0 | Higher = stronger style/emotion. Try 3.5-4.0 |
| `cfg_scale_speaker` | 5.0 | Higher = more speaker identity consistency |
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

## VoiceDesign Caption Best Practices

Structure (2-3 sentences in Japanese):
1. Speaker attributes: gender, age, pitch (e.g. "低い声の女性", "やや高めの男性")
2. Emotion/state: current emotional tone
3. Audio quality/distance: mic distance, environment

Examples:
- `低い声の女性が、苛立ちを隠せない様子で焦って話している。クリアな音質。`
- `落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。`
- `やや高めの男性の声で、気遣いを見せて申し訳なさそうなトーンでやさしく話してほしい。`

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
