# moe-speech extraction

`litagin/moe-speech` is currently a gated raw-audio HuggingFace dataset, not a parquet-backed tabular dataset. The extractor in `irodori_tts_infra.datasets.moe_speech` therefore uses deterministic HuggingFace Hub file listing and per-file download instead of pretending the repo exposes a labeled streaming table.

## Usage

```bash
uv run python -m irodori_tts_infra.datasets.extract \
  --character 00013899 \
  --out ./data/moe-speech/00013899 \
  --max-bytes 1073741824 \
  --sample-rate 24000 \
  --include-nsfw
```

The command writes:

1. resampled mono 16-bit WAV files
2. `index.json`

## Index schema

`ExtractionIndex` serializes to JSON with:

- `dataset`
- `sample_rate`
- `include_nsfw`
- `total_bytes`
- `total_duration_s`
- `characters`

`characters` maps a character identifier to ordered clip entries:

```json
{
  "characters": {
    "00013899": [
      ["00013899_000.wav", 4.21]
    ]
  }
}
```

Clip paths are relative to the directory containing `index.json`, which keeps the index deterministic and portable.

## NSFW flag semantics

`litagin/moe-speech` is published as `not-for-all-audiences`, but it does not currently expose a separate public non-NSFW subset or per-clip NSFW labels.

- `--include-nsfw`: allow extraction from the dataset as published
- `--no-include-nsfw`: fail fast with a clear error instead of silently pretending a clean subset exists

That fail-fast behavior is intentional. It keeps the CLI honest until a real subset or alternate fallback dataset is wired in.

## Determinism

The extractor sorts HuggingFace repo paths before downloading any clip. Re-running the same extraction for the same character, sample rate, and disk cap produces the same file order and the same `index.json` ordering.

## Disk cap guidance

The default cap is 1 GiB per character. Extraction stops before writing the clip that would exceed the limit. This keeps the output bounded without deleting already-written files.

If you need a smaller training set:

1. lower `--max-bytes`
2. keep the default deterministic ordering so repeated runs stay reproducible
3. choose the final RVC training subset from the written clips and `index.json`

## Troubleshooting

| Problem | Meaning | Fix |
|---|---|---|
| HuggingFace auth failure | The dataset is manually gated | Accept the dataset terms on HuggingFace and authenticate before running the CLI |
| `--no-include-nsfw` fails | moe-speech has no published clean subset | Re-run with `--include-nsfw` or use a different dataset |
| Unsupported audio format error | A clip was not mono 16-bit WAV as expected | Inspect the offending source file; the extractor only writes mono 16-bit WAV output |

## Future JEV2 hook

The current implementation keeps the extraction path centered on a single record iterator contract. A future JEV2 fallback can plug into that seam once there is a concrete need for alternate dataset coverage or clip-level NSFW filtering.
