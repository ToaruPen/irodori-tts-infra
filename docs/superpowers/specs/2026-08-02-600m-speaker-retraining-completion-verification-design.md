# 600M Speaker Retraining Completion Verification Design

## Purpose

Provide one fail-closed, read-only verifier that proves the 12-model 600M
VoiceDesign Speaker Inversion retraining and evaluation run is complete without
changing the running queue, versioned remote evidence, generated evaluation
artifacts, active voice bank, proposed staging directory, or service state.

The verifier closes gaps between the existing component contracts. It does not
replace the training queue, evaluation queue, evaluator, review packet builder,
or staging report.

The threat model is a trusted operator on the dedicated GPU host. Hash and path
bindings detect accidental drift, stale evidence, and partial execution; they are
not a signature scheme against an attacker who can rewrite the verifier and every
bound artifact. Adding an external signed policy is outside this workflow.

## Scope and Ownership

The implementation is limited to:

- `scripts/verify_600m_speaker_retraining_completion.py`
- `tests/scripts/test_verify_600m_speaker_retraining_completion.py`
- this design and its implementation plan

Existing status JSONL, remote launcher evidence, v1-v4 benchmark output,
evaluation output, review packets, voice-bank files, and service state are
immutable inputs. The verifier may only create its explicitly requested report
path, using an atomic create-only write.

## Existing Contracts Reused

The verifier independently checks and cross-binds these existing contracts:

- training jobs and `training-status.jsonl`
- the speed-v1 launcher evidence for the ten pending models
- the 600M VoiceDesign base checkpoint identity and pinned upstream commit
- evaluation queue configuration and status
- per-model evaluation manifests and `evaluation-verification.json` v2
- per-model review candidates and self-contained review packets
- the staging report and active voice-bank baseline

The existing evaluator may report `PASS` while non-selected checkpoints still
have review candidates. Therefore evaluator `PASS` is necessary but not
sufficient for final completion.

## Command Contract

The verifier has two phases:

```text
--phase training
--phase final
```

Both phases require:

- `--training-jobs`
- `--training-status`
- `--training-launch-evidence`
- `--output`

The final phase also requires:

- `--evaluation-config`
- `--evaluation-status`
- `--review-decisions`
- `--staging-report`

`--gpu-memory-tolerance-mib` defaults to 256 MiB, matching the existing
speed-v1 launcher. Runtime process and GPU probes are read-only and injectable
for unit tests.

The process exits zero only for `PASS`. Missing user decisions produce
`AWAITING_REVIEW`; every other unmet contract produces `FAIL`. Both non-PASS
states exit nonzero.

## Training Gate

### Job and provenance closure

The jobs document must contain exactly 12 unique model IDs in queue order. For
each model, the latest status event must be `finished/success` with exit code
zero. A later `started/running` event invalidates an earlier success.

Each latest success row must match the current files and declared run identity:

- clean manifest path and SHA-256
- training config path and SHA-256
- base checkpoint path and SHA-256
- base revision
- upstream commit
- output directory
- log path

The report binds the SHA-256 of the jobs document, complete status JSONL, every
config, every clean manifest, every log, the base checkpoint, and every
checkpoint file.

### Exact checkpoint inventory

Each output directory must contain exactly these 13 Speaker Inversion files:

```text
checkpoint_0000250.speaker.safetensors
checkpoint_0000500.speaker.safetensors
...
checkpoint_0003000.speaker.safetensors
checkpoint_final.speaker.safetensors
```

No additional `.speaker.safetensors` file is allowed. Every file must contain
only one valid `speaker_embedding` payload with dtype `F32`, shape `(16, 768)`,
valid offsets and size, and finite values. The final and step-3000 files must be
byte-identical by SHA-256.

The status candidate set must contain every periodic checkpoint exactly once.
It may omit or include `checkpoint_final` because the seeded Anabel status has
12 candidates while queue-produced statuses may have 13. Every recorded
candidate must name one member of the exact on-disk set and carry its current
SHA-256. No duplicate or unrelated candidate is accepted. `last_checkpoint`
must bind the step-3000 periodic file and hash.

### Config and loss closure

Each config must declare:

- `speaker_inversion_enabled: true`
- `speaker_inversion_tokens: 16`
- `max_steps: 3000`
- `save_every: 250`
- `log_every: 20`
- `valid_ratio: 0.0`
- `checkpoint_best_n: 0`

The verifier removes terminal control sequences and parses the final complete
training-run segment in the stable log. That segment must contain the exact
logged optimizer-step sequence 20, 40, ..., 3000, a finite `loss` value on
every event, and a terminal `Training finished at step=3000.` line. Historical
prefixes from a prior failed or restarted attempt do not invalidate a later
complete segment; an incomplete final segment does.

### Queue launcher and runtime closure

The speed-v1 launch evidence must have `schema_version: 1`, `state: finished`,
queue exit code zero, no completion errors, a valid new-status contract, the
ten pending success IDs in the same order as jobs 3-12, no active owned process
after completion, and `gpu_memory_released: true`. Its jobs/status/base
checkpoint/revision/upstream/script bindings must match the current inputs.

The verifier also performs a live read-only check. It excludes only its own process
and its enumerated ancestor chain. Other commands are never excluded by diagnostic
keywords. A PID is never sufficient evidence because Windows may reuse a
completed launcher's PID for an unrelated data-loader process. Matching uses
command semantics. NVIDIA compute rows are rejected independently when their PID
matches a workflow conflict or their executable is Python; unrelated WDDM desktop
and UI rows are ignored. The live gate requires:

- no training queue, training launcher, evaluation queue, checkpoint
  generator, metrics worker, evaluator, upstream trainer, or multiprocessing
  data-loader process;
- no Python or workflow-related NVIDIA compute application outside the verifier's
  excluded process chain;
- no probe error; and
- current GPU memory at most launcher `gpu_before.used_mib` plus the configured
  tolerance.

The launch evidence SHA-256 and normalized live snapshot are stored in the
completion report.

## Final Evaluation Gate

The final phase first reruns the complete training gate.

### Evaluation queue closure

The canonical evaluation input is the versioned evaluation runtime configuration. Its
`runtime-inputs-v1` manifest must have the producer's exact eleven-file runtime
inventory: the runtime config, frozen jobs/status, upstream runtime provenance,
a deterministic archive of the exact tracked `irodori_tts` Python package, and
six fixed component scripts. Every file SHA-256 and size, the exact nine
original source bindings (source config, source jobs/status, and six component
scripts), and the source config-to-runtime-config transformation are revalidated.
Frozen jobs/status must
be byte-identical to the current completed training inputs even though their paths
differ. The runtime configuration must contain the same 12 models in the same
order and reference the same base checkpoint identity.

The package archive must contain exactly the Python paths and content hashes in
the upstream provenance document, with deterministic ZIP metadata and no extra,
missing, encrypted, directory, or symlink-like entries. Generation imports
`irodori_tts` from this archive rather than the mutable upstream worktree, and its
generation verification binds both the archive and provenance hashes.

The evaluation status must bind that runtime config path and SHA-256 and contain a
current successful row for exactly 49 stages. Each row's component, command,
fingerprint, and output roots are reconstructed from the producer contract instead
of trusting the row's declarations:

- one `manifests` stage;
- generation, analysis, metrics, and evaluate stages for each of 12 models.

Every accepted row must use the status v1 schema, bind the current evaluation
config SHA-256, have exit code zero, a valid lowercase stage fingerprint, the
current component-script SHA when a component exists, and an output snapshot
that still exactly matches the files on disk. The evaluation queue lock must
not exist.

### Per-model evaluation closure

For each model, the verifier requires:

- evaluation verification schema v2 with `status: PASS`;
- exactly 5 checkpoints at steps 1000, 1500, 2000, 2500, and 3000;
- exactly 140 evaluation cases;
- the exact 7 text IDs, 2 seeds, and 2 styles from the manifest;
- both normal `control` text and all word/sentence `unko`, `chinko`, and
  `manko` cases;
- one selected checkpoint whose embedding and provenance match the manifest,
  training status, and current file hash;
- artifact hashes and review-packet copied-asset hashes that still match.

For the reused Anabel generation stage, exactly one of
`generation-verification.json` and `canonicalization-report.json` must exist,
matching the evaluation queue producer contract.

The verifier does not alter deterministic selection. A human `VOICE` decision
does not automatically re-include a checkpoint conservatively rejected by the
current evaluator.

## Human Review Decision Contract

`review-decisions.jsonl` is a new input contract with one object per review
candidate:

```json
{
  "schema_version": "speaker-checkpoint-review-decision/v1",
  "case_id": "...",
  "model_id": "...",
  "checkpoint_step": 2000,
  "wav_sha256": "...",
  "reviewer": "user",
  "reviewed_at": "2026-08-02T00:00:00+00:00",
  "decision": "VOICE"
}
```

Allowed decisions are `VOICE`, `TONE`, and `UNSURE`. Identity fields and the
WAV hash must match the review candidate exactly. Timestamps must be timezone
aware. Duplicate, extra, stale, or conflicting decisions fail.

The candidate set and decision set must be one-to-one across all 12 evaluation
directories. Missing decisions yield `AWAITING_REVIEW`, after the verifier has
proved that the corresponding review packet and assets are complete. The
verifier never assigns `VOICE` itself.

Final policy:

- unresolved candidate count must be zero;
- a selected checkpoint may have `VOICE` decisions;
- any selected-checkpoint `TONE` or `UNSURE` decision fails;
- non-selected `TONE` and `UNSURE` decisions remain in the report grouped by
  model and checkpoint, preserving the rejection and selection rationale.

## Non-Deployment Gate

The staging report must be schema v1 and retain all of these values:

- `status: PASS`
- `model_count: 12`
- exactly the same 12 selected embeddings and hashes
- `deployment_performed: false`
- `active_voice_bank_unchanged: true`
- `proposed_staging_root_created: false`

The current active voice bank is re-hashed against the report's baseline, and
the proposed staging root must still not exist. The verifier never copies an
embedding or creates the staging root.

## Output Contract

The create-only report uses schema
`600m-speaker-retraining-completion-verification/v1` and contains:

- `status`: `PASS`, `AWAITING_REVIEW`, or `FAIL`
- phase and verification timestamp
- a deterministic check list with pass/fail facts and reasons
- input and artifact path/SHA-256 bindings
- per-model training/checkpoint/loss summaries
- normalized runtime evidence
- per-model evaluation and selected-checkpoint summaries
- review totals, unresolved IDs, and checkpoint-grouped decision counts
- staging and non-deployment evidence
- verifier script SHA-256

The verifier creates a temporary sibling with exclusive creation, durably
flushes it, then publishes with an atomic no-overwrite operation. It refuses to
overwrite either an existing report or temporary file, including a concurrent
creation race. The requested report must be a direct child of the training launch
evidence directory, so even an early validation failure cannot write into a
training, evaluation, staging, or active voice-bank artifact tree.

## Failure Model

Structural, identity, hash, finite-value, matrix, runtime, and non-deployment
violations produce a report with `FAIL` where safe, then exit nonzero. Missing
review decisions alone produce `AWAITING_REVIEW`. Invalid decision rows are
contract failures, not awaiting input. Unexpected exceptions remove the
temporary report and surface the exception without mutating inputs.
