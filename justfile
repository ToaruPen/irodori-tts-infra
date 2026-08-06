set default-list
set positional-arguments

remote_uv := 'C:\Users\takut\.local\bin\uv.exe'
remote_project := 'C:\Users\takut\Dev\Irodori-TTS'
remote_venv := 'C:\Users\takut\Dev\Irodori-TTS\.venv'
remote_work_root := 'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731'
remote_audit_script := remote_work_root + '\scripts\build_clean_speaker_datasets.py'
remote_analysis_script := remote_work_root + '\scripts\analyze_nko_beep_matrix.py'
remote_evaluation_manifest_script := remote_work_root + '\scripts\build_600m_checkpoint_evaluation_manifests.py'
remote_review_packet_script := remote_work_root + '\scripts\build_speaker_review_packet.py'
remote_staging_report_script := remote_work_root + '\scripts\build_600m_speaker_staging_report.py'
remote_metrics_script := remote_work_root + '\scripts\compute_600m_speaker_metrics.py'
remote_evaluate_script := remote_work_root + '\scripts\evaluate_600m_speaker_checkpoints.py'
remote_generation_script := remote_work_root + '\scripts\generate_600m_checkpoint_audio_remote.py'
remote_search_manifest_script := remote_work_root + '\scripts\checkpoint_search_v2\build_600m_speaker_checkpoint_search_manifest.py'
remote_search_generation_script := remote_work_root + '\scripts\checkpoint_search_v2\generate_600m_speaker_checkpoint_search_remote.py'
remote_search_evaluate_script := remote_work_root + '\scripts\checkpoint_search_v2\evaluate_600m_speaker_checkpoint_search.py'
remote_midpoint_derive_script := remote_work_root + '\scripts\checkpoint_midpoint_v3\derive_600m_speaker_midpoint_diagnostic.py'
remote_midpoint_generation_script := remote_work_root + '\scripts\checkpoint_midpoint_v3\generate_600m_speaker_midpoint_diagnostic_remote.py'
remote_midpoint_evaluate_script := remote_work_root + '\scripts\checkpoint_midpoint_v3\evaluate_600m_speaker_midpoint_diagnostic.py'
remote_reference_centroid_audit_script := remote_work_root + '\scripts\audit_600m_speaker_reference_centroid.py'
remote_evaluation_queue_script := remote_work_root + '\scripts\run_600m_speaker_evaluation_queue.py'
remote_training_queue_script := remote_work_root + '\scripts\run_600m_speaker_training_queue.py'
remote_training_queue_detached_launcher := remote_work_root + '\scripts\speaker_training_queue_detached_v004\launch_600m_speaker_training_queue_detached.py'
remote_quality_run_script := remote_work_root + '\scripts\quality_runs_v3\manage_600m_speaker_quality_runs.py'
remote_speed_benchmark_script := remote_work_root + '\scripts\run_600m_training_speed_benchmark.py'
remote_speed_selection_script := remote_work_root + '\scripts\apply_600m_training_speed_selection.py'
remote_evaluation_speed_v4_launcher := remote_work_root + '\scripts\launch_600m_speaker_evaluation_queue_speed_v2.py'
remote_evaluation_speed_v5_launcher := remote_work_root + '\scripts\launch_600m_speaker_evaluation_queue_speed_v3.py'
remote_evaluation_speed_v6_launcher := remote_work_root + '\scripts\evaluation_speed_v6_v4\launch_600m_speaker_evaluation_queue_speed_v4.py'
remote_evaluation_speed_v6_detached_launcher := remote_work_root + '\scripts\evaluation_speed_v6_detached_v6\launch_600m_speaker_evaluation_queue_speed_v6_detached.py'
remote_completion_verifier := remote_work_root + '\scripts\verify_600m_speaker_retraining_completion.py'
remote_catalog := remote_work_root + '\source\catalog.json'
remote_labels := remote_work_root + '\labels.jsonl'
remote_caption_rules := remote_work_root + '\caption-rules.json'
remote_tone_signatures := remote_work_root + '\tone-signatures.json'
remote_audit_root := remote_work_root + '\audit'
remote_output_reserve := 'from pathlib import Path; import sys; p = Path(sys.argv[1]); root = Path(sys.argv[2]); p.parent != root and sys.exit(f"output path must be a direct child of {root}: {p}"); p.mkdir()'

# Install all local development dependencies.
sync:
    uv sync --all-extras

# Run Ruff lint checks.
lint:
    uv run ruff check .

# Check Python formatting without changing files.
format-check:
    uv run ruff format --check .

# Format Python files with Ruff.
format:
    uv run ruff format .

# Run the mypy type checker.
typecheck:
    uv run mypy

# Run default tests, forwarding any extra pytest arguments.
test *args:
    uv run pytest "$@"

# Run integration tests.
test-integration:
    uv run pytest -m "integration"

# Run GPU tests.
test-gpu:
    uv run pytest -m "gpu"

# Detect dead code under src/.
dead-code:
    uv run vulture src/

# Run the full local verification suite, stopping on the first failure.
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run vulture src/
    uv run pytest

# Run the Irodori-TTS client CLI, forwarding all arguments.
client *args:
    uv run irodori-tts "$@"

# Run the deployment CLI, forwarding all arguments.
deploy *args:
    uv run irodori-tts-deploy "$@"

# Generate the fixed checkpoint evaluation matrix, forwarding all arguments.
speaker-generate *args:
    uv run python scripts/generate_nko_beep_matrix.py "$@"

# Analyze generated audio for narrowband tone candidates.
speaker-analyze *args:
    uv run python scripts/analyze_nko_beep_matrix.py "$@"

# Build strict per-model checkpoint evaluation manifests from completed training status.
speaker-build-manifests *args:
    uv run python scripts/build_600m_checkpoint_evaluation_manifests.py "$@"

# Compute checkpoint speaker-similarity and transcription metrics.
speaker-metrics *args:
    uv run python scripts/compute_600m_speaker_metrics.py "$@"

# Select Speaker Inversion checkpoints through the deterministic quality gate.
speaker-evaluate *args:
    uv run python scripts/evaluate_600m_speaker_checkpoints.py "$@"

# Build a create-only single-checkpoint diagnostic search manifest.
speaker-checkpoint-search-build *args:
    uv run python scripts/build_600m_speaker_checkpoint_search_manifest.py "$@"

# Generate the isolated 28-case single-checkpoint diagnostic matrix.
speaker-checkpoint-search-generate *args:
    uv run python scripts/generate_600m_speaker_checkpoint_search_remote.py "$@"

# Verify and score an isolated 28-case single-checkpoint diagnostic matrix.
speaker-checkpoint-search-evaluate *args:
    uv run python scripts/evaluate_600m_speaker_checkpoint_search.py "$@"

# Derive the fixed alpha=0.5 diagnostic embedding in a create-only versioned directory.
speaker-midpoint-diagnostic-build *args:
    uv run python scripts/derive_600m_speaker_midpoint_diagnostic.py "$@"

# Generate the isolated 28-case matrix for a derived midpoint diagnostic embedding.
speaker-midpoint-diagnostic-generate *args:
    uv run python scripts/generate_600m_speaker_midpoint_diagnostic_remote.py "$@"

# Verify and score a derived midpoint diagnostic without promoting it to production.
speaker-midpoint-diagnostic-evaluate *args:
    uv run python scripts/evaluate_600m_speaker_midpoint_diagnostic.py "$@"

# Audit reference identity and ECAPA centroid stability without changing quality gates.
speaker-reference-centroid-audit *args:
    uv run python scripts/audit_600m_speaker_reference_centroid.py "$@"

# Run the resumable 12-model Speaker Inversion training queue.
speaker-train-queue *args:
    uv run python scripts/run_600m_speaker_training_queue.py "$@"

# Preflight, launch, or inspect a pinned detached Speaker Inversion training queue.
speaker-train-queue-detached *args:
    uv run python scripts/launch_600m_speaker_training_queue_detached.py "$@"

# Prepare or finalize a create-only Speaker Inversion quality-search/retraining run.
speaker-quality-run *args:
    uv run python scripts/manage_600m_speaker_quality_runs.py "$@"

# Benchmark serial 600M trainer memory/speed candidates and select the fastest safe one.
speaker-speed-benchmark *args:
    uv run python scripts/run_600m_training_speed_benchmark.py "$@"

# Apply a verified speed-benchmark winner to the pending Speaker Inversion jobs.
speaker-apply-speed-selection *args:
    uv run python scripts/apply_600m_training_speed_selection.py "$@"

# Run the resumable serial evaluation queue after all 12 training jobs succeed.
speaker-evaluation-queue *args:
    uv run python scripts/run_600m_speaker_evaluation_queue.py "$@"

# Materialize a self-contained suspicious-audio review packet.
speaker-review-packet *args:
    uv run python scripts/build_speaker_review_packet.py "$@"

# Build a verified non-destructive model staging report without copying embeddings.
speaker-staging-report *args:
    uv run python scripts/build_600m_speaker_staging_report.py "$@"

# Prepare, preflight, or launch the isolated speed-v4 evaluation queue.
speaker-evaluation-speed-v4 *args:
    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v2.py "$@"

# Prepare, preflight, or launch the retained legacy speed-v5 evaluation queue.
speaker-evaluation-speed-v5 *args:
    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v3.py "$@"

# Prepare, preflight, or launch the fresh isolated speed-v6 evaluation queue.
speaker-evaluation-speed-v6 *args:
    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v4.py "$@"

# Preflight, launch, or inspect the detached speed-v6 evaluation supervisor.
speaker-evaluation-speed-v6-detached *args:
    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v6_detached.py "$@"

# Verify immutable training evidence or the complete non-deployment workflow.
speaker-verify-retraining *args:
    uv run python scripts/verify_600m_speaker_retraining_completion.py "$@"

# Run a Python script in the pinned upstream Windows environment without syncing.
remote-python script *args:
    #!/usr/bin/env bash
    set -euo pipefail

    remote_host=''
    if [[ -r .env ]]; then
        while IFS= read -r env_line || [[ -n "$env_line" ]]; do
            case "$env_line" in
                IRODORI_REMOTE_HOST=*)
                    remote_host=${env_line#*=}
                    remote_host=${remote_host%$'\r'}
                    break
                    ;;
            esac
        done < .env
    fi
    if [[ -z "$remote_host" ]]; then
        echo "IRODORI_REMOTE_HOST must be nonempty in .env" >&2
        exit 2
    fi

    args_payload=$(uv run --no-sync python -c 'import base64,json,sys; print(base64.urlsafe_b64encode(json.dumps(sys.argv[1:], ensure_ascii=False).encode()).decode())' "$@")
    remote_bootstrap='import base64,json,subprocess,sys;a=json.loads(base64.urlsafe_b64decode(sys.argv[1]));raise SystemExit(subprocess.run([sys.executable,*a]).returncode)'

    powershell_command='& '
    for argument in '{{ remote_uv }}' run --project '{{ remote_project }}' --no-sync --python '{{ remote_venv }}' python -c "$remote_bootstrap" "$args_payload"; do
        escaped_argument=${argument//\'/\'\'}
        powershell_command+="'$escaped_argument' "
    done

    encoded_command=$(printf '%s' "$powershell_command" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\r\n')
    ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$remote_host" powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand "$encoded_command"

# Atomically reserve audit/<round>, then audit the fixed remote dataset inputs.
remote-audit round: (remote-python '-c' (remote_output_reserve) (remote_audit_root + '\' + round) (remote_audit_root)) (remote-python (remote_audit_script) '--catalog-json' (remote_catalog) '--labels-jsonl' (remote_labels) '--caption-rules-json' (remote_caption_rules) '--tone-signatures-json' (remote_tone_signatures) '--output-root' (remote_audit_root + '\' + round))

# Compute checkpoint metrics in the pinned Windows runtime.
remote-speaker-metrics *args:
    just remote-python '{{ remote_metrics_script }}' "$@"

# Analyze generated audio in the pinned Windows runtime.
remote-speaker-analyze *args:
    just remote-python '{{ remote_analysis_script }}' "$@"

# Build strict per-model evaluation manifests in the pinned Windows runtime.
remote-speaker-build-manifests *args:
    just remote-python '{{ remote_evaluation_manifest_script }}' "$@"

# Generate one model's fixed 140-case, five-checkpoint matrix in the pinned Windows runtime.
remote-speaker-generate *args:
    just remote-python '{{ remote_generation_script }}' "$@"

# Evaluate and select checkpoints in the pinned Windows runtime.
remote-speaker-evaluate *args:
    just remote-python '{{ remote_evaluate_script }}' "$@"

# Build a create-only diagnostic search manifest in the pinned Windows runtime.
remote-speaker-checkpoint-search-build *args:
    just remote-python '{{ remote_search_manifest_script }}' "$@"

# Generate a diagnostic search matrix in the pinned Windows runtime.
remote-speaker-checkpoint-search-generate *args:
    just remote-python '{{ remote_search_generation_script }}' "$@"

# Verify and score a diagnostic search matrix in the pinned Windows runtime.
remote-speaker-checkpoint-search-evaluate *args:
    just remote-python '{{ remote_search_evaluate_script }}' "$@"

# Derive the fixed alpha=0.5 diagnostic embedding in the pinned Windows runtime.
remote-speaker-midpoint-diagnostic-build *args:
    just remote-python '{{ remote_midpoint_derive_script }}' "$@"

# Generate the derived midpoint diagnostic matrix in the pinned Windows runtime.
remote-speaker-midpoint-diagnostic-generate *args:
    just remote-python '{{ remote_midpoint_generation_script }}' "$@"

# Verify and score the derived midpoint diagnostic in the pinned Windows runtime.
remote-speaker-midpoint-diagnostic-evaluate *args:
    just remote-python '{{ remote_midpoint_evaluate_script }}' "$@"

# Audit reference identity and centroid stability in the pinned Windows runtime.
remote-speaker-reference-centroid-audit *args:
    just remote-python '{{ remote_reference_centroid_audit_script }}' "$@"

# Materialize a self-contained review packet in the pinned Windows runtime.
remote-speaker-review-packet *args:
    just remote-python '{{ remote_review_packet_script }}' "$@"

# Build a non-destructive model staging report in the pinned Windows runtime.
remote-speaker-staging-report *args:
    just remote-python '{{ remote_staging_report_script }}' "$@"

# Run the resumable training queue in the pinned Windows runtime.
remote-speaker-train-queue *args:
    just remote-python '{{ remote_training_queue_script }}' "$@"

# Preflight, launch, or inspect the versioned detached training supervisor remotely.
remote-speaker-train-queue-detached *args:
    just remote-python '{{ remote_training_queue_detached_launcher }}' "$@"

# Prepare or finalize a create-only quality run in the pinned Windows runtime.
remote-speaker-quality-run *args:
    just remote-python '{{ remote_quality_run_script }}' "$@"

# Benchmark serial 600M trainer candidates in the pinned Windows runtime.
remote-speaker-speed-benchmark *args:
    just remote-python '{{ remote_speed_benchmark_script }}' "$@"

# Apply a verified speed-benchmark winner in the pinned Windows runtime.
remote-speaker-apply-speed-selection *args:
    just remote-python '{{ remote_speed_selection_script }}' "$@"

# Run the resumable serial evaluation queue in the pinned Windows runtime.
remote-speaker-evaluation-queue *args:
    just remote-python '{{ remote_evaluation_queue_script }}' "$@"

# Prepare, preflight, or launch the isolated speed-v4 evaluation queue remotely.
remote-speaker-evaluation-speed-v4 *args:
    just remote-python '{{ remote_evaluation_speed_v4_launcher }}' "$@"

# Prepare, preflight, or launch the retained legacy speed-v5 evaluation queue remotely.
remote-speaker-evaluation-speed-v5 *args:
    just remote-python '{{ remote_evaluation_speed_v5_launcher }}' "$@"

# Prepare, preflight, or launch the fresh isolated speed-v6 evaluation queue remotely.
remote-speaker-evaluation-speed-v6 *args:
    just remote-python '{{ remote_evaluation_speed_v6_launcher }}' "$@"

# Preflight, launch, or inspect the detached speed-v6 evaluation supervisor remotely.
remote-speaker-evaluation-speed-v6-detached *args:
    just remote-python '{{ remote_evaluation_speed_v6_detached_launcher }}' "$@"

# Verify immutable training evidence or the complete non-deployment workflow remotely.
remote-speaker-verify-retraining *args:
    just remote-python '{{ remote_completion_verifier }}' "$@"
