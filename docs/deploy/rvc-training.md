# Windows GPU RVC Training SOP

## Purpose & Scope

This SOP covers Phase 2 issue
[#21](https://github.com/ToaruPen/irodori-tts-infra/issues/21) sub-task 2b:
training per-character RVC voice models on the Windows GPU host after Phase 1
deployment is already working. It is for model training and checkpoint handoff,
not for runtime inference wiring or production request handling.

## Prerequisites

- Windows GPU host bootstrapped through Phase 1
  [`deploy-bootstrap`](windows.md#commands).
- The server uv environment is active and all RVC/Irodori packages are installed
  into that same environment, per
  [ADR 0001](../adr/0001-irodori-runtime-access.md).
- NVIDIA driver and CUDA stack are installed for the RVC library selected in
  sub-task 2c: `<TBD after 2c merges>`.
- HuggingFace is authenticated before pulling gated datasets:

```powershell
huggingface-cli login
```

- Character audio has been extracted with
  `irodori_tts_infra.datasets.extract` from issue #21 sub-task 2a. The dataset
  extractor and `docs/datasets/extraction.md` are currently tracked in
  [PR #26](https://github.com/ToaruPen/irodori-tts-infra/pull/26).

## RVC Library Install

> **TBD — resolved once sub-task 2c selects the RVC inference library (see issue
> #21).** Install into the same uv env per ADR 0001:

```powershell
# Template only. Replace the package, source, and commit after sub-task 2c lands.
uv pip install "<rvc-library>" --pin <commit>
```

Do not install RVC into a separate virtual environment. The Phase 2 runtime
assumes Irodori-TTS and RVC share one uv-managed Python environment and one CUDA
memory budget. See
[ADR 0001](../adr/0001-irodori-runtime-access.md) for the binding runtime and
install policy.

## Dataset Requirements

Use 10-30 minutes of clean, single-character audio for each RVC model. Audio
should be mono WAV, 16-48 kHz, with clips selected for stable pronunciation,
low background noise, and the emotional states needed by the character.

NSFW handling follows the dataset extraction policy from issue #21 sub-task 2a:
`litagin/moe-speech` is gated and currently has no public clean subset or
per-clip NSFW labels, so extraction must explicitly opt into the published
dataset with `--include-nsfw` or fail fast with `--no-include-nsfw`. See
`docs/datasets/extraction.md` after PR #26 lands.

Expected extracted layout:

```text
datasets/
  <character>/
    clips/
      0000.wav
      0001.wav
    index.json
```

The sub-task 2a extractor currently writes WAV files and `index.json` into the
selected output directory. If it writes clips flat instead of under `clips/`,
preserve the generated `index.json` and either point the training command at that
directory or move the WAV files under `clips/` with the index updated to match.

## Training Command Template

> **Manual verification required on Windows GPU host.**

The exact CLI depends on the RVC library selected in sub-task 2c. Keep the
command surface explicit and replace only the `<TBD after 2c merges>` values once
that library is chosen.

```powershell
# Template only. Do not run until sub-task 2c provides the real RVC CLI.
uv run <TBD after 2c merges> train `
  --dataset "C:\Users\user\irodori-tts-infra\datasets\<character>" `
  --output-dir "C:\Users\user\irodori-tts-infra\models\rvc\<character>" `
  --sample-rate 40000 `
  --epochs 200 `
  --batch-size <TBD after 2c merges; fit within 12GB VRAM> `
  --amp <TBD after 2c merges> `
  --resume-from-checkpoint <optional checkpoint path if supported>
```

Initial target configuration:

- sample rate: `40000`
- epochs: approximately `200`
- batch size: the largest stable value that fits the RTX 4070 12GB host
- AMP: enabled when the selected RVC library supports it safely
- resume: use the library's checkpoint flag only when a previous run has a
  compatible checkpoint

## Checkpoint Management

Use this Windows output convention for the final checkpoint:

```text
C:\Users\user\irodori-tts-infra\models\rvc\<character>\<character>.pth
```

The repository already excludes model artifacts. Existing `.gitignore` patterns
include:

```text
models/
checkpoints/
weights/
*.pth
```

Keep generated indexes, logs, intermediate checkpoints, and tensorboard runs out
of source control unless a small intentional fixture is explicitly added under
`tests/fixtures/`.

For backup or transfer to macOS, zip the per-character model directory on
Windows and copy it over SSH:

```powershell
Compress-Archive `
  -Path "C:\Users\user\irodori-tts-infra\models\rvc\<character>" `
  -DestinationPath "C:\Users\user\irodori-tts-infra\models\rvc\<character>.zip" `
  -Force

scp "C:\Users\user\irodori-tts-infra\models\rvc\<character>.zip" `
  user@mac-host:/path/to/private/model-backups/
```

## Voice Bank Manifest Integration

After training and spot-checking, create or update `voice_bank_rvc.toml` in the
same directory that contains the story's `characters.md`. The schema is defined
in `src/irodori_tts_infra/voice_bank/repository.py`: all paths must be relative
to the TOML file, and each manifest character must already exist in
`characters.md`.

```toml
[characters."チヅル"]
model_path = "models/rvc/chizuru/chizuru.pth"
sample_rate = 40000
neutral_prototype = "prototypes/chizuru-neutral.npy"

[characters."チヅル".state_prototypes]
happy = "prototypes/chizuru-happy.npy"
sad = "prototypes/chizuru-sad.npy"
whisper = "prototypes/chizuru-whisper.npy"
intimate = "prototypes/chizuru-intimate.npy"
moan = "prototypes/chizuru-moan.npy"
```

State-specific prototype extraction is part of the per-character RVC model plan
in [the RVC architecture doc](../irodori-rvc-architecture.md#per-character-rvc-model).
Use neutral prototypes for normal dialogue and state prototypes for acoustically
different states such as whisper, intimate, and moan.

## Quality Spot-Check Procedure

> **Manual verification required on Windows GPU host.**

The current Phase 1 CLI command is `irodori-tts read-aloud`, not
`irodori-tts synthesize`. Create a short turn file with 4-6 lines for the target
character:

```text
【チヅル】「今日はいい天気ですね。」
【チヅル:嬉しそうに】「やった、成功しました。」
【チヅル:悲しそうに】「少しだけ、寂しかったです。」
【チヅル:小声で囁くように】「しっ……誰かいるみたい。」
【チヅル:息を整えながら】「大丈夫、もう落ち着きました。」
```

Run synthesis through the Phase 1 server and save the ordered WAV files:

```powershell
uv run irodori-tts read-aloud ".\spotcheck\<character>.md" `
  --characters ".\spotcheck\characters.md" `
  --remote-host "http://127.0.0.1:8923" `
  --save-dir ".\spotcheck\irodori-source\<character>"
```

Then run those files through the selected RVC conversion command:

```powershell
# Template only. Replace after sub-task 2c defines the RVC conversion CLI.
uv run <TBD after 2c merges> <convert-command> `
  <model-flag TBD after 2c merges> "C:\Users\user\irodori-tts-infra\models\rvc\<character>\<character>.pth" `
  <input-dir-flag TBD after 2c merges> ".\spotcheck\irodori-source\<character>" `
  <output-dir-flag TBD after 2c merges> ".\spotcheck\rvc-output\<character>" `
  <sample-rate-flag TBD after 2c merges> 40000
```

Human listening pass:

- pronunciation is intelligible and unchanged from the source phrase
- character identity is recognizable and consistent across states
- no obvious metallic, buzzing, clipping, pitch-jump, or breath artifacts
- whisper/sad/intimate states preserve the intended expression without becoming
  another character

Numeric targets come from
[`docs/irodori-rvc-architecture.md` Quality Gate](../irodori-rvc-architecture.md#quality-gate-system):

- normal dialogue, happy, angry, flustered: cosine `>= 0.80`
- sad and whisper: cosine `>= 0.65-0.70`
- aroused/intimate: cosine `>= 0.60-0.70` against the intimate prototype
- short moans: cosine `>= 0.50-0.60` as a soft warning plus spot review
- long moans: compare against the moan prototype instead of neutral

If a model fails the listening pass or numeric gate, retrain with a cleaner or
larger dataset, reduce noisy clips, or split prototypes by state. If it passes,
commit only the manifest entry and documentation; keep the `.pth` checkpoint and
prototype arrays in private artifact storage.

## Troubleshooting

| Problem | Action |
|---|---|
| VRAM OOM | Reduce `--batch-size`, shorten parallel preprocessing, or prune large dataset shards. If AMP is unstable for the selected library, turn it off even though memory use may rise. |
| CUDA or driver mismatch | Re-check the driver, CUDA, PyTorch, and selected RVC library compatibility matrix from sub-task 2c. Reinstall into the same uv env rather than creating a second venv. |
| Less than 10 minutes usable audio | Extract more clips, relax the disk cap, or choose a different character/source. Do not train a production model from a tiny set unless it is explicitly labeled experimental. |
| Checkpoint loading errors after RVC upgrade | Keep the exact RVC library commit with each training run. If loading breaks after an upgrade, reinstall the recorded commit or retrain with the new library and update the manifest only after spot-checking. |

## Cross-Links

- [ADR 0001: Irodori Runtime Access Pattern](../adr/0001-irodori-runtime-access.md)
- [Issue #21](https://github.com/ToaruPen/irodori-tts-infra/issues/21)
- [RVC architecture and quality gates](../irodori-rvc-architecture.md)
- `docs/datasets/extraction.md` from issue #21 sub-task 2a / PR #26
- [Windows GPU deployment](windows.md)
