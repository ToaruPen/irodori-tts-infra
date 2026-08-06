# ruff: noqa: C901, PLR0912, PLR0914, PLR0915, PLR2004, S404, S603, S607, SLF001
# Fixtures mirror the fixed wire contract and inject subprocess results without execution.
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType
    from typing import IO, Any, Protocol

    class CheckpointLike(Protocol):
        path: Path
        sha256: str

    class TrainingModelLike(Protocol):
        model_id: str
        checkpoints: Sequence[CheckpointLike]
        config_sha256: str
        run_id: str

    class TrainingLike(Protocol):
        models: Sequence[TrainingModelLike]
        base_checkpoint: Path
        base_checkpoint_sha256: str
        checkpoint_revision: str
        upstream_commit: str
        training_jobs: Path
        training_status: Path
        model_ids: Sequence[str]

    class EvaluationModelLike(Protocol):
        model_id: str
        selected: Mapping[str, object]
        review_candidates: Sequence[Mapping[str, object]]

    class EvaluationsLike(Protocol):
        models: Sequence[EvaluationModelLike]


class TrainingFixture(TypedDict):
    root: Path
    jobs: Path
    base_status: Path
    status: Path
    launch: Path
    status_rows: list[dict[str, object]]


class EvaluationFixture(TypedDict):
    root: Path
    config: Path
    source_config: Path
    runtime_manifest: Path
    snapshot_jobs: Path
    snapshot_status: Path
    status: Path
    evaluation_dirs: list[Path]
    decisions: Path


pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/verify_600m_speaker_retraining_completion.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_600m_speaker_retraining_completion",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_embedding(
    path: Path,
    *,
    finite: bool = True,
    extra: bool = False,
    value: float = 1.0,
) -> None:
    values = np.full((16, 768), value, dtype="<f4")
    if not finite:
        values.flat[0] = np.nan
    payload = values.tobytes()
    header: dict[str, object] = {
        "speaker_embedding": {
            "dtype": "F32",
            "shape": [16, 768],
            "data_offsets": [0, len(payload)],
        }
    }
    if extra:
        header["unexpected"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded) + len(padding)) + encoded + padding + payload)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fixture_stage_contract(  # noqa: PLR0911 - mirrors the five fixed producer stages.
    stage: str,
    config: dict[str, Any],
    runtime_components: Mapping[str, Path],
) -> tuple[Path | None, list[str], tuple[Path, ...]]:
    models = config["models"]
    manifest_root = Path(config["manifest_output_dir"])
    if stage == "manifests":
        component = runtime_components["build_600m_checkpoint_evaluation_manifests.py"]
        speaker = config["metric_models"]["speaker_embedding"]
        transcription = config["metric_models"]["transcription"]
        command = [
            sys.executable,
            str(component),
            "--training-status",
            config["training_status"],
            "--training-jobs",
            config["training_jobs"],
            "--output-dir",
            config["manifest_output_dir"],
            "--base-checkpoint",
            config["base_checkpoint"]["model_id"],
            "--base-checkpoint-sha256",
            config["base_checkpoint"]["sha256"],
            "--base-revision",
            config["base_checkpoint"]["revision"],
            "--speaker-embedding-model-id",
            speaker["model_id"],
            "--speaker-embedding-revision",
            speaker["revision"],
            "--speaker-embedding-source-sha256",
            speaker["source_sha256"],
            "--transcription-model-id",
            transcription["model_id"],
            "--transcription-revision",
            transcription["revision"],
            "--transcription-source-sha256",
            transcription["source_sha256"],
        ]
        for model in models:
            command.extend(("--reference-wavs", model["reference_wavs"]))
        return component, command, (manifest_root,)
    model_id, operation = stage.split(":", 1)
    model = next(model for model in models if model["model_id"] == model_id)
    reuse = model.get("reuse")
    if isinstance(reuse, dict):
        if operation == "generation":
            generation = Path(reuse["generation_dir"])
            return (
                None,
                [],
                (
                    generation / "generation-results.jsonl",
                    generation / "generation-verification.json",
                ),
            )
        if operation == "analysis":
            return None, [], (Path(reuse["analysis_dir"]) / "analysis-results.jsonl",)
        if operation == "metrics":
            return (
                None,
                [],
                (
                    Path(reuse["metrics_results"]),
                    Path(reuse["metrics_provenance"]),
                ),
            )
        return None, [], (Path(reuse["evaluation_dir"]),)
    generation = Path(model["generation_dir"])
    analysis = Path(model["analysis_dir"])
    metrics = Path(model["metrics_dir"])
    evaluation = Path(model["evaluation_dir"])
    manifest = manifest_root / model_id / "evaluation-manifest.json"
    component_names = {
        "generation": "generate_600m_checkpoint_audio_remote.py",
        "analysis": "analyze_nko_beep_matrix.py",
        "metrics": "compute_600m_speaker_metrics.py",
        "evaluate": "evaluate_600m_speaker_checkpoints.py",
    }
    component = runtime_components[component_names[operation]]
    if operation == "generation":
        command = [
            sys.executable,
            str(component),
            "generate",
            "--checkpoint-manifest",
            str(manifest),
            "--base-checkpoint-path",
            config["base_checkpoint"]["path"],
            "--upstream-root",
            config["upstream_root"],
            "--upstream-runtime-provenance",
            str(Path(config["training_jobs"]).parent / "upstream-runtime-provenance.json"),
            "--upstream-package-archive",
            str(Path(config["training_jobs"]).parent / "upstream-runtime-package.zip"),
            "--output-dir",
            str(generation),
            "--upstream-runtime-provenance-sha256",
            hashlib.sha256(
                (
                    Path(config["training_jobs"]).parent / "upstream-runtime-provenance.json"
                ).read_bytes()
            ).hexdigest(),
            "--upstream-package-archive-sha256",
            hashlib.sha256(
                (Path(config["training_jobs"]).parent / "upstream-runtime-package.zip").read_bytes()
            ).hexdigest(),
        ]
        return component, command, (generation,)
    if operation == "analysis":
        command = [
            sys.executable,
            str(component),
            "--generation-dir",
            str(generation),
            "--output-dir",
            str(analysis),
        ]
        return component, command, (analysis,)
    if operation == "metrics":
        speaker = config["metric_models"]["speaker_embedding"]
        transcription = config["metric_models"]["transcription"]
        command = [
            sys.executable,
            str(component),
            "--generation-results",
            str(generation / "generation-results.jsonl"),
            "--reference-wavs",
            model["reference_wavs"],
            "--output",
            str(metrics / "metrics-results.jsonl"),
            "--provenance-output",
            str(metrics / "metrics-results.provenance.json"),
            "--ecapa-source",
            speaker["source"],
            "--ecapa-savedir",
            speaker["savedir"],
            "--ecapa-model-id",
            speaker["model_id"],
            "--ecapa-revision",
            speaker["revision"],
            "--whisper-model",
            transcription["model_id"],
            "--whisper-source",
            transcription["source"],
            "--whisper-revision",
            transcription["revision"],
            "--whisper-device",
            transcription["device"],
        ]
        return component, command, (metrics,)
    command = [
        sys.executable,
        str(component),
        "--generation-results",
        str(generation / "generation-results.jsonl"),
        "--analysis-results",
        str(analysis / "analysis-results.jsonl"),
        "--metrics-results",
        str(metrics / "metrics-results.jsonl"),
        "--metrics-provenance",
        str(metrics / "metrics-results.provenance.json"),
        "--evaluation-manifest",
        str(manifest),
        "--output-dir",
        str(evaluation),
    ]
    return component, command, (evaluation,)


def _write_training_fixture(tmp_path: Path, module: ModuleType) -> TrainingFixture:
    root = tmp_path / "queue"
    upstream = root / "upstream"
    package = upstream / "irodori_tts"
    package.mkdir(parents=True)
    (package / "runtime.py").write_text("RUNTIME = True\n", encoding="utf-8")
    (upstream / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(upstream, "init")
    _git(upstream, "config", "user.email", "test@example.invalid")
    _git(upstream, "config", "user.name", "Test")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-m", "fixture")
    upstream_commit = _git(upstream, "rev-parse", "HEAD")
    setattr(module, "PINNED_UPSTREAM_COMMIT", upstream_commit)  # noqa: B010
    base_checkpoint = root / "base.safetensors"
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_bytes(b"600m base")
    jobs = []
    status_rows = []
    for index in range(12):
        model_id = ("Anabel", "Kasumi")[index] if index < 2 else f"model-{index:02d}"
        manifest = root / "datasets" / model_id / "clean.jsonl"
        config = root / "configs" / f"{model_id}.json"
        output_dir = root / "training" / model_id
        if index == 0:
            log = root / "pilot-logs" / "special-anabel-complete.log"
        elif index == 1:
            log = root / "legacy-logs" / "kasumi-seeded.log"
        else:
            log = root / "logs" / f"{model_id}.log"
        manifest.parent.mkdir(parents=True)
        config.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"source_id": model_id}) + "\n", encoding="utf-8")
        config.write_text(
            json.dumps(
                {
                    "train": {
                        "manifest_path": str(manifest),
                        "output_dir": str(output_dir),
                        "speaker_inversion_enabled": True,
                        "speaker_inversion_tokens": 16,
                        "speaker_inversion_init_embedding": None,
                        "learning_rate": 0.001,
                        "seed": 0,
                        "batch_size": 8,
                        "gradient_accumulation_steps": 2,
                        "gradient_checkpointing": True,
                        "max_steps": 3000,
                        "save_every": 250,
                        "log_every": 20,
                        "valid_ratio": 0.0,
                        "checkpoint_best_n": 0,
                    }
                }
            ),
            encoding="utf-8",
        )
        log.write_text(
            "\n".join(
                [f"step={step} loss={1 / step:.8f}" for step in range(20, 3001, 20)]
                + ["Training finished at step=3000."]
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoints = []
        for step in range(250, 3001, 250):
            checkpoint = output_dir / f"checkpoint_{step:07d}.speaker.safetensors"
            _write_embedding(checkpoint)
            checkpoints.append({"path": str(checkpoint), "sha256": module.sha256_file(checkpoint)})
        final = output_dir / "checkpoint_final.speaker.safetensors"
        final.write_bytes((output_dir / "checkpoint_0003000.speaker.safetensors").read_bytes())
        if index != 0:
            checkpoints.append({"path": str(final), "sha256": module.sha256_file(final)})
        jobs.append(
            {
                "model_id": model_id,
                "clean_manifest": str(manifest),
                "config": str(config),
                "output_dir": str(output_dir),
                "command": [
                    "python",
                    "-u",
                    "train.py",
                    "--config",
                    str(config),
                    "--manifest",
                    str(manifest),
                    "--init-checkpoint",
                    str(base_checkpoint),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda",
                ],
            }
        )
        status_rows.append(
            {
                "event": "finished",
                "status": "success",
                "model_id": model_id,
                "clean_manifest_sha256": module.sha256_file(manifest),
                "config_sha256": module.sha256_file(config),
                "checkpoint_sha256": module.sha256_file(base_checkpoint),
                "checkpoint_revision": "c" * 40,
                "upstream_commit": upstream_commit,
                "started_at": "2026-08-02T00:00:00+00:00",
                "ended_at": "2026-08-02T01:00:00+00:00",
                "exit_code": 0,
                "log_path": str(log),
                "last_checkpoint": str(output_dir / "checkpoint_0003000.speaker.safetensors"),
                "last_checkpoint_sha256": module.sha256_file(
                    output_dir / "checkpoint_0003000.speaker.safetensors"
                ),
                "candidate_checkpoints": checkpoints,
                "error": None,
            }
        )
    jobs_path = root / "training-jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": "2026-08-02T00:00:00+00:00",
                "queue_policy": "serial_one_at_a_time",
                "anabel_strategy": "reuse_existing_fresh_3000_run",
                "base_checkpoint_path": str(base_checkpoint),
                "base_checkpoint_sha256": module.sha256_file(base_checkpoint),
                "checkpoint_revision": "c" * 40,
                "upstream_commit": upstream_commit,
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    prefix_rows = status_rows[:2]
    new_rows: list[dict[str, object]] = []
    for finished in status_rows[2:]:
        started = dict(finished) | {
            "event": "started",
            "status": "running",
            "ended_at": None,
            "exit_code": None,
            "last_checkpoint": None,
            "last_checkpoint_sha256": None,
            "candidate_checkpoints": [],
        }
        new_rows.extend((started, finished))
    status_rows = [*prefix_rows, *new_rows]
    status_path = root / "training-status.jsonl"
    status_before_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in prefix_rows)
    status_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in status_rows),
        encoding="utf-8",
    )
    launcher_script = root / "launch_600m_training_queue_speed_v1.py"
    launcher_script.write_text("# immutable launcher\n", encoding="utf-8")
    queue_script = root / "run_600m_speaker_training_queue.py"
    queue_script.write_text("# immutable queue\n", encoding="utf-8")
    launch_path = root / "launch-evidence.json"
    launch_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "finished",
                "queue_exit_code": 0,
                "completion_errors": [],
                "new_status_contract_valid": True,
                "status_row_count_before": len(prefix_rows),
                "status_before_sha256": hashlib.sha256(status_before_text.encode()).hexdigest(),
                "status_after_sha256": module.sha256_file(status_path),
                "status_row_count": len(status_rows),
                "new_status_row_count": 20,
                "new_started_model_ids": [row["model_id"] for row in jobs[2:]],
                "new_finished_success_model_ids": [row["model_id"] for row in jobs[2:]],
                "finished_success_model_ids": [row["model_id"] for row in jobs],
                "finished_failed_model_ids": [],
                "active_owned_processes_after": [],
                "gpu_memory_released": True,
                "gpu_before": {"used_mib": 900.0},
                "training_jobs_path": str(jobs_path),
                "training_jobs_sha256": module.sha256_file(jobs_path),
                "status_path": str(status_path),
                "checkpoint_path": str(base_checkpoint),
                "checkpoint_sha256": module.sha256_file(base_checkpoint),
                "checkpoint_revision": "c" * 40,
                "upstream_commit": upstream_commit,
                "launcher_script_path": str(launcher_script),
                "launcher_script_sha256": module.sha256_file(launcher_script),
                "queue_script_path": str(queue_script),
                "queue_script_sha256": module.sha256_file(queue_script),
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "jobs": jobs_path,
        "base_status": status_path,
        "status": status_path,
        "launch": launch_path,
        "status_rows": status_rows,
    }


def _write_evaluation_fixture(
    tmp_path: Path,
    module: ModuleType,
    training: TrainingLike,
) -> EvaluationFixture:
    root = tmp_path / "evaluation-queue"
    root.mkdir()
    upstream_root = next(
        candidate / "upstream"
        for candidate in training.training_jobs.parents
        if (candidate / "upstream").is_dir()
    )
    manifest_root = root / "manifests"
    models = []
    evaluation_dirs = []
    for index, training_model in enumerate(training.models):
        model_id = training_model.model_id
        evaluation_dir = root / "evaluation" / model_id
        evaluation_dir.mkdir(parents=True)
        evaluation_dirs.append(evaluation_dir)
        reference = root / "references" / f"{model_id}.json"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text("{}\n", encoding="utf-8")
        generation_dir = root / "generation" / model_id
        analysis_dir = root / "analysis" / model_id
        metrics_dir = root / "metrics" / model_id
        model_row: dict[str, object] = {
            "model_id": model_id,
            "reference_wavs": str(reference),
        }
        if index == 0:
            model_row["reuse"] = {
                "generation_dir": str(generation_dir),
                "analysis_dir": str(analysis_dir),
                "metrics_results": str(metrics_dir / "metrics-results.jsonl"),
                "metrics_provenance": str(metrics_dir / "metrics-results.provenance.json"),
                "evaluation_manifest": str(manifest_root / model_id / "evaluation-manifest.json"),
                "evaluation_dir": str(evaluation_dir),
            }
        else:
            model_row.update(
                {
                    "generation_dir": str(generation_dir),
                    "analysis_dir": str(analysis_dir),
                    "metrics_dir": str(metrics_dir),
                    "evaluation_dir": str(evaluation_dir),
                }
            )
        models.append(model_row)
        for required in (
            generation_dir / "generation-results.jsonl",
            generation_dir / "generation-verification.json",
            analysis_dir / "analysis-results.jsonl",
            metrics_dir / "metrics-results.jsonl",
            metrics_dir / "metrics-results.provenance.json",
        ):
            required.parent.mkdir(parents=True, exist_ok=True)
            required.write_text("{}\n", encoding="utf-8")
        checkpoints_by_name = {
            checkpoint.path.name: checkpoint for checkpoint in training_model.checkpoints
        }
        checkpoint_rows = []
        for step in module.EXPECTED_EVALUATION_STEPS:
            checkpoint = checkpoints_by_name[f"checkpoint_{step:07d}.speaker.safetensors"]
            checkpoint_rows.append(
                {
                    "model_id": model_id,
                    "checkpoint_step": step,
                    "embedding_path": str(checkpoint.path),
                    "embedding_sha256": checkpoint.sha256,
                    "training_config_sha256": training_model.config_sha256,
                    "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                    "base_checkpoint_sha256": training.base_checkpoint_sha256,
                    "base_revision": training.checkpoint_revision,
                    "run_id": training_model.run_id,
                }
            )
        manifest = manifest_root / model_id / "evaluation-manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
                    "models": [{"model_id": model_id, "checkpoints": checkpoint_rows}],
                    "text_ids": list(module.EXPECTED_TEXT_IDS),
                    "seeds": list(module.EXPECTED_SEEDS),
                    "styles": list(module.EXPECTED_STYLES),
                }
            ),
            encoding="utf-8",
        )
        results = [
            {
                "case_id": f"{model_id}:{step}:{text_id}:{seed}:{style}",
                "model_id": model_id,
                "checkpoint_step": step,
                "text_id": text_id,
                "seed": seed,
                "style": style,
                "metric_gate_applied": text_id
                in {"sentence_unko", "sentence_chinko", "sentence_manko", "control"},
            }
            for step in module.EXPECTED_EVALUATION_STEPS
            for text_id in module.EXPECTED_TEXT_IDS
            for seed in module.EXPECTED_SEEDS
            for style in module.EXPECTED_STYLES
        ]
        (evaluation_dir / "evaluation-results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
        )
        (evaluation_dir / "checkpoint-summary.jsonl").write_text("{}\n", encoding="utf-8")
        (evaluation_dir / "evaluation-config.json").write_text("{}\n", encoding="utf-8")
        selected = checkpoint_rows[0] | {"rank": 1}
        selected_path = evaluation_dir / "selected-models.json"
        selected_path.write_text(
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-evaluation/v1",
                    "selections": [selected],
                }
            ),
            encoding="utf-8",
        )
        wav = root / "candidate-audio" / f"{model_id}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(f"wav:{model_id}".encode())
        candidate = {
            "case_id": results[0]["case_id"],
            "model_id": model_id,
            "checkpoint_step": 1000,
            "wav_path": str(wav),
            "wav_sha256": module.sha256_file(wav),
        }
        (evaluation_dir / "review-candidates.jsonl").write_text(
            json.dumps(candidate) + "\n", encoding="utf-8"
        )
        packet = evaluation_dir / "review_packet"
        (packet / "audio").mkdir(parents=True)
        copied = packet / "audio" / f"{model_id}.wav"
        copied.write_bytes(wav.read_bytes())
        (packet / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-review-packet/v1",
                    "review_candidates": [
                        {
                            "case_id": candidate["case_id"],
                            "wav": {
                                "path": f"audio/{model_id}.wav",
                                "sha256": module.sha256_file(copied),
                            },
                            "spectrogram": None,
                            "paired_controls": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        artifacts = [
            evaluation_dir / "evaluation-results.jsonl",
            evaluation_dir / "checkpoint-summary.jsonl",
            evaluation_dir / "review-candidates.jsonl",
            evaluation_dir / "evaluation-config.json",
            selected_path,
            packet / "manifest.json",
            copied,
        ]
        (evaluation_dir / "evaluation-verification.json").write_text(
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-evaluation-verification/v2",
                    "status": "PASS",
                    "selected": selected,
                    "checkpoint_count": 5,
                    "evaluation_case_count": 140,
                    "hard_gate_metric_case_count_per_checkpoint": 16,
                    "diagnostic_word_case_count_per_checkpoint": 12,
                    "artifact_sha256": {
                        str(path.resolve()): module.sha256_file(path) for path in artifacts
                    },
                }
            ),
            encoding="utf-8",
        )
    (manifest_root / "manifest-index.json").write_text("{}\n", encoding="utf-8")
    config = root / "evaluation-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "speaker-evaluation-queue/v1",
                "training_status": str(training.training_status),
                "training_jobs": str(training.training_jobs),
                "manifest_output_dir": str(manifest_root),
                "base_checkpoint": {
                    "model_id": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                    "path": str(training.base_checkpoint),
                    "sha256": training.base_checkpoint_sha256,
                    "revision": training.checkpoint_revision,
                },
                "upstream_root": str(upstream_root),
                "metric_models": {
                    "speaker_embedding": {
                        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                        "revision": "speaker-revision",
                        "source_sha256": "1" * 64,
                        "source": str(root / "models" / "ecapa"),
                        "savedir": str(root / "runtime-cache" / "ecapa"),
                    },
                    "transcription": {
                        "model_id": "openai/whisper-large-v3-turbo",
                        "revision": "whisper-revision",
                        "source_sha256": "2" * 64,
                        "source": str(root / "models" / "whisper"),
                        "device": "cuda:0",
                    },
                },
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    component_names = (
        "run_600m_speaker_evaluation_queue.py",
        "build_600m_checkpoint_evaluation_manifests.py",
        "generate_600m_checkpoint_audio_remote.py",
        "analyze_nko_beep_matrix.py",
        "compute_600m_speaker_metrics.py",
        "evaluate_600m_speaker_checkpoints.py",
    )
    source_scripts = {name: root / "source-scripts" / name for name in component_names}
    for name, path in source_scripts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
    stages = ["manifests"] + [
        f"{model_id}:{stage}"
        for model_id in training.model_ids
        for stage in module.EXPECTED_EVALUATION_STAGES_PER_MODEL
    ]
    status_rows = []
    for stage in stages:
        if stage == "manifests":
            output_root = manifest_root
        else:
            model_id, operation = stage.split(":", 1)
            output_root = (
                root / "evaluation" / model_id
                if operation == "evaluate"
                else root / operation / model_id
            )
        row = {
            "schema_version": "speaker-evaluation-queue-status/v1",
            "config_path": str(config),
            "config_sha256": module.sha256_file(config),
            "stage_fingerprint": "0" * 64,
            "component_script": None,
            "stage": stage,
            "model_id": None if stage == "manifests" else stage.split(":", 1)[0],
            "event": "finished",
            "status": "success",
            "exit_code": 0,
            "outputs": [module.snapshot_path(output_root)],
            "command": [],
        }
        status_rows.append(row)
    status = root / "evaluation-status.jsonl"
    runtime_root = root / "runtime-inputs-v1"
    runtime_root.mkdir()
    runtime_scripts = runtime_root / "scripts"
    runtime_scripts.mkdir()
    runtime_jobs = runtime_root / "training-jobs-speed-v1.json"
    runtime_training_status = runtime_root / "training-status.jsonl"
    runtime_config = runtime_root / "evaluation-queue-runtime.json"
    runtime_jobs.write_bytes(training.training_jobs.read_bytes())
    runtime_training_status.write_bytes(training.training_status.read_bytes())
    runtime_components = {name: runtime_scripts / name for name in component_names}
    for name, path in runtime_components.items():
        path.write_bytes(source_scripts[name].read_bytes())
    runtime_document = json.loads(config.read_text(encoding="utf-8"))
    runtime_document["training_jobs"] = str(runtime_jobs)
    runtime_document["training_status"] = str(runtime_training_status)
    runtime_config.write_text(
        json.dumps(runtime_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upstream_root = Path(runtime_document["upstream_root"])
    upstream_python = sorted((upstream_root / "irodori_tts").rglob("*.py"))
    upstream_provenance = runtime_root / "upstream-runtime-provenance.json"
    upstream_provenance.write_text(
        json.dumps(
            {
                "schema_version": "irodori-upstream-runtime-provenance/v1",
                "upstream_root": str(upstream_root.resolve()),
                "commit": training.upstream_commit,
                "tree": _git(upstream_root, "rev-parse", "HEAD^{tree}"),
                "package": "irodori_tts",
                "python_files": [
                    {
                        "path": path.relative_to(upstream_root).as_posix(),
                        "sha256": module.sha256_file(path),
                    }
                    for path in upstream_python
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    upstream_archive = runtime_root / "upstream-runtime-package.zip"
    with zipfile.ZipFile(
        upstream_archive,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in upstream_python:
            relative = path.relative_to(upstream_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    snapshot_files = (
        runtime_config,
        runtime_jobs,
        runtime_training_status,
        upstream_provenance,
        upstream_archive,
        *runtime_components.values(),
    )
    runtime_manifest = runtime_root / "snapshot-manifest.json"
    runtime_manifest.write_text(
        json.dumps(
            {
                "schema_version": "speaker-evaluation-runtime-inputs/v1",
                "source_inputs": {
                    str(config.resolve()): module.sha256_file(config),
                    str(training.training_jobs): module.sha256_file(training.training_jobs),
                    str(training.training_status): module.sha256_file(training.training_status),
                    **{
                        str(path.resolve()): module.sha256_file(path)
                        for path in source_scripts.values()
                    },
                },
                "files": {
                    path.relative_to(runtime_root).as_posix(): {
                        "sha256": module.sha256_file(path),
                        "size": path.stat().st_size,
                    }
                    for path in snapshot_files
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_config_sha = module.sha256_file(runtime_config)
    for row in status_rows:
        row["config_path"] = str(runtime_config)
        row["config_sha256"] = runtime_config_sha
        component, command, output_roots = _fixture_stage_contract(
            str(row["stage"]),
            runtime_document,
            runtime_components,
        )
        row["component_script"] = (
            {"path": str(component), "sha256": module.sha256_file(component)}
            if component is not None
            else None
        )
        row["command"] = command
        row["outputs"] = [module.snapshot_path(path) for path in output_roots]
        row["stage_fingerprint"] = module._current_stage_fingerprint(
            runtime_document,
            row,
            base=runtime_root,
        )
    status.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in status_rows),
        encoding="utf-8",
    )
    decisions = root / "review-decisions.jsonl"
    decisions.write_text("", encoding="utf-8")
    return {
        "root": root,
        "config": runtime_config,
        "source_config": config,
        "runtime_manifest": runtime_manifest,
        "snapshot_jobs": runtime_jobs,
        "snapshot_status": runtime_training_status,
        "status": status,
        "evaluation_dirs": evaluation_dirs,
        "decisions": decisions,
    }


def _refresh_evaluation_status_stage(
    fixture: EvaluationFixture,
    module: ModuleType,
    *,
    model_id: str,
) -> None:
    runtime_document = json.loads(fixture["config"].read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in fixture["status"].read_text(encoding="utf-8").splitlines()]
    row = next(item for item in rows if item["stage"] == f"{model_id}:evaluate")
    evaluation_dir = fixture["root"] / "evaluation" / model_id
    row["outputs"] = [module.snapshot_path(evaluation_dir)]
    row["stage_fingerprint"] = module._current_stage_fingerprint(
        runtime_document,
        row,
        base=fixture["config"].parent,
    )
    fixture["status"].write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )


def _refresh_launch_status_hashes(
    fixture: TrainingFixture,
    module: ModuleType,
) -> None:
    launch = json.loads(fixture["launch"].read_text(encoding="utf-8"))
    lines = fixture["status"].read_bytes().splitlines(keepends=True)
    before_count = launch["status_row_count_before"]
    launch["status_before_sha256"] = hashlib.sha256(b"".join(lines[:before_count])).hexdigest()
    current = module.sha256_file(fixture["status"])
    launch["status_after_sha256"] = current
    launch["status_row_count"] = len(lines)
    fixture["launch"].write_text(json.dumps(launch), encoding="utf-8")


def _seed_existing_training_run(
    fixture: TrainingFixture,
    module: ModuleType,
) -> tuple[Path, str]:
    seeded_status = fixture["status_rows"][0]
    model_id = seeded_status["model_id"]
    assert isinstance(model_id, str)
    run_provenance = Path(fixture["root"]) / "pilot-logs" / "anabel-run-provenance.json"
    run_provenance.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
    run_provenance_sha256 = module.sha256_file(run_provenance)
    seeded_status["seeded_existing_run"] = {
        "run_provenance_path": str(run_provenance.resolve()),
        "run_provenance_sha256": run_provenance_sha256,
    }
    fixture["status"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fixture["status_rows"]),
        encoding="utf-8",
    )
    _refresh_launch_status_hashes(fixture, module)
    return run_provenance, run_provenance_sha256


def _append_quality_run_evidence(
    fixture: TrainingFixture,
    module: ModuleType,
    *,
    model_id: str,
    version: str,
    predecessor_jobs: Path | None = None,
    init_step: int = 2500,
) -> tuple[Path, Path]:
    source_jobs_path = predecessor_jobs or fixture["jobs"]
    source_jobs = json.loads(source_jobs_path.read_text(encoding="utf-8"))
    source_job = next(job for job in source_jobs["jobs"] if job["model_id"] == model_id)
    source_root = source_jobs_path.parent
    source_config = (source_root / source_job["config"]).resolve()
    source_output = (source_root / source_job["output_dir"]).resolve()
    manifest = (source_root / source_job["clean_manifest"]).resolve()
    init_embedding = source_output / f"checkpoint_{init_step:07d}.speaker.safetensors"

    run_root = fixture["root"] / "quality-runs" / model_id / version
    output_dir = run_root / "training"
    output_dir.mkdir(parents=True)
    config = run_root / "config.json"
    config_payload = json.loads(source_config.read_text(encoding="utf-8"))
    source_train = dict(config_payload["train"])
    config_payload["train"].update(
        {
            "manifest_path": str(manifest),
            "output_dir": str(output_dir),
            "learning_rate": 0.0003,
            "seed": 1,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": True,
            "speaker_inversion_init_embedding": str(init_embedding),
        }
    )
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    log = run_root / "training.log"
    log.write_text(
        "\n".join(
            [f"step={step} loss={1 / step:.8f}" for step in range(20, 3001, 20)]
            + ["Training finished at step=3000."]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoints = []
    for step in range(250, 3001, 250):
        checkpoint = output_dir / f"checkpoint_{step:07d}.speaker.safetensors"
        _write_embedding(checkpoint, value=1.0 + init_step / 10000)
        checkpoints.append(
            {
                "name": checkpoint.name,
                "path": str(checkpoint),
                "sha256": module.sha256_file(checkpoint),
            }
        )
    final = output_dir / "checkpoint_final.speaker.safetensors"
    final.write_bytes((output_dir / "checkpoint_0003000.speaker.safetensors").read_bytes())
    checkpoints.append(
        {"name": final.name, "path": str(final), "sha256": module.sha256_file(final)}
    )

    jobs = dict(source_jobs)
    jobs["base_checkpoint_path"] = str(
        (source_root / source_jobs["base_checkpoint_path"]).resolve()
    )
    jobs["created_at_utc"] = "2026-08-02T02:00:00+00:00"
    jobs["queue_policy"] = "serial_one_at_a_time"
    jobs["anabel_strategy"] = "reuse_existing_fresh_3000_run"
    jobs["jobs"] = [dict(job) for job in source_jobs["jobs"]]
    job = next(job for job in jobs["jobs"] if job["model_id"] == model_id)
    successor_command = list(source_job["command"])
    successor_command[successor_command.index("--config") + 1] = str(config)
    successor_command[successor_command.index("--output-dir") + 1] = str(output_dir)
    job.update(
        {
            "config": str(config),
            "output_dir": str(output_dir),
            "command": successor_command,
        }
    )
    jobs_path = run_root / f"training-jobs-{version}.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")

    predecessor_status = fixture["status"]
    before_bytes = predecessor_status.read_bytes()
    before_count = len(before_bytes.splitlines())
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    config_sha = module.sha256_file(config)
    manifest_sha = module.sha256_file(manifest)
    base_sha = source_jobs["base_checkpoint_sha256"]
    common_status = {
        "model_id": model_id,
        "clean_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "checkpoint_sha256": base_sha,
        "checkpoint_revision": source_jobs["checkpoint_revision"],
        "upstream_commit": source_jobs["upstream_commit"],
        "started_at": "2026-08-02T03:00:00+00:00",
        "log_path": str(log),
    }
    started = common_status | {
        "event": "started",
        "status": "running",
        "ended_at": None,
        "exit_code": None,
        "last_checkpoint": None,
        "last_checkpoint_sha256": None,
        "candidate_checkpoints": [],
        "error": None,
    }
    finished = common_status | {
        "event": "finished",
        "status": "success",
        "ended_at": "2026-08-02T04:00:00+00:00",
        "exit_code": 0,
        "last_checkpoint": str(output_dir / "checkpoint_0003000.speaker.safetensors"),
        "last_checkpoint_sha256": module.sha256_file(
            output_dir / "checkpoint_0003000.speaker.safetensors"
        ),
        "candidate_checkpoints": [
            {"path": checkpoint["path"], "sha256": checkpoint["sha256"]}
            for checkpoint in checkpoints
        ],
        "error": None,
    }
    status_path = run_root / f"training-status-{version}.jsonl"
    status_path.write_bytes(before_bytes)
    with status_path.open("a", encoding="utf-8") as status_file:
        status_file.write(json.dumps(started, sort_keys=True) + "\n")
        status_file.write(json.dumps(finished, sort_keys=True) + "\n")
    fixture["status_rows"].extend((started, finished))
    after_count = before_count + 2
    after_sha = module.sha256_file(status_path)
    fixture["status"] = status_path

    diagnostic = run_root / "source-diagnostic.json"
    diagnostic.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
    queue_script = fixture["root"] / "run_600m_speaker_training_queue.py"
    setup = run_root / "setup-evidence.json"
    setup.write_text(
        json.dumps(
            {
                "schema_version": "speaker-quality-retrain-setup/v1",
                "created_at": "2026-08-02T02:00:00+00:00",
                "model_id": model_id,
                "reason": {
                    "source_diagnostic": str(diagnostic),
                    "source_diagnostic_sha256": module.sha256_file(diagnostic),
                    "source_best": {
                        "checkpoint_step": init_step,
                        "hard_gate_pass_count": 15,
                        "hard_gate_case_count": 16,
                        "failing_case": "sentence_manko",
                        "speaker_similarity": 0.7,
                        "required_minimum": 0.75,
                    },
                    "strategy": "initialize from the strongest predecessor checkpoint",
                },
                "changes": {
                    "learning_rate": {
                        "from": source_train["learning_rate"],
                        "to": 0.0003,
                    },
                    "seed": {"from": source_train["seed"], "to": 1},
                    "max_steps": 3000,
                    "save_every": 250,
                    "batch_size": 8,
                    "gradient_accumulation_steps": 2,
                    "gradient_checkpointing": True,
                    "speaker_inversion_init_embedding": str(init_embedding),
                    "speaker_inversion_init_embedding_sha256": module.sha256_file(init_embedding),
                },
                "paths": {
                    "config": str(config),
                    "jobs": str(jobs_path),
                    "status": str(status_path),
                    "queue_script": str(queue_script),
                    "output_dir": str(output_dir),
                },
                "sha256": {
                    "source_config": module.sha256_file(source_config),
                    "config": config_sha,
                    "jobs": module.sha256_file(jobs_path),
                    "status_seed": before_sha,
                    "queue_script": module.sha256_file(queue_script),
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = run_root / "run-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "speaker-quality-retrain-run-evidence/v1",
                "created_at": "2026-08-02T04:01:00+00:00",
                "state": "finished",
                "model_id": model_id,
                "queue_exit_code": 0,
                "setup_evidence": {
                    "path": str(setup),
                    "sha256": module.sha256_file(setup),
                },
                "training_jobs": {
                    "path": str(jobs_path),
                    "sha256": module.sha256_file(jobs_path),
                },
                "training_status": {
                    "path": str(status_path),
                    "before_row_count": before_count,
                    "before_sha256": before_sha,
                    "after_row_count": after_count,
                    "after_sha256": after_sha,
                    "new_status_row_count": 2,
                    "new_started_model_ids": [model_id],
                    "new_finished_success_model_ids": [model_id],
                },
                "queue_script": {
                    "path": str(queue_script),
                    "sha256": module.sha256_file(queue_script),
                },
                "invocation": {
                    "recipe": "speaker-quality-retrain",
                    "checkpoint_revision": source_jobs["checkpoint_revision"],
                    "upstream_commit": source_jobs["upstream_commit"],
                },
                "run": {
                    "started_at": started["started_at"],
                    "ended_at": finished["ended_at"],
                    "config_sha256": config_sha,
                    "clean_manifest_sha256": manifest_sha,
                    "base_checkpoint_sha256": base_sha,
                    "candidate_checkpoint_count": 13,
                    "checkpoints": checkpoints,
                    "final_equals_step3000": True,
                    "log": {
                        "path": str(log),
                        "sha256": module.sha256_file(log),
                        "loss_event_count": 150,
                        "loss_steps_exact": True,
                        "loss_all_finite": True,
                        "last_loss": 1 / 3000,
                        "oom": False,
                        "traceback": False,
                    },
                },
                "runtime_after": {
                    "gpu_memory_used_mib": 973.0,
                    "gpu_memory_total_mib": 12282.0,
                    "gpu_utilization_percent": 0.0,
                    "gpu_power_watts": 12.0,
                    "active_training_processes": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return evidence, jobs_path


def _refresh_quality_setup_binding(
    payload: dict[str, Any],
    module: ModuleType,
    setup: dict[str, Any],
) -> None:
    setup_path = Path(payload["setup_evidence"]["path"])
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    payload["setup_evidence"]["sha256"] = module.sha256_file(setup_path)


def _refresh_quality_config_provenance(
    evidence: Path,
    module: ModuleType,
    *,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    setup_path = Path(payload["setup_evidence"]["path"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    config_path = Path(setup["paths"]["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mutate(config)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = module.sha256_file(config_path)
    setup["sha256"]["config"] = config_sha
    _refresh_quality_setup_binding(payload, module, setup)
    payload["run"]["config_sha256"] = config_sha
    status_path = Path(payload["training_status"]["path"])
    rows = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
    rows[-2]["config_sha256"] = config_sha
    rows[-1]["config_sha256"] = config_sha
    status_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload["training_status"]["after_sha256"] = module.sha256_file(status_path)
    evidence.write_text(json.dumps(payload), encoding="utf-8")


def _refresh_quality_jobs_provenance(
    evidence: Path,
    module: ModuleType,
    *,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    jobs_path = Path(payload["training_jobs"]["path"])
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    mutate(jobs)
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    jobs_sha = module.sha256_file(jobs_path)
    payload["training_jobs"]["sha256"] = jobs_sha
    setup_path = Path(payload["setup_evidence"]["path"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["sha256"]["jobs"] = jobs_sha
    _refresh_quality_setup_binding(payload, module, setup)
    evidence.write_text(json.dumps(payload), encoding="utf-8")


def _write_all_voice_decisions(path: Path, evaluations: EvaluationsLike) -> None:
    rows = [
        {
            "schema_version": "speaker-checkpoint-review-decision/v1",
            "case_id": candidate["case_id"],
            "model_id": candidate["model_id"],
            "checkpoint_step": candidate["checkpoint_step"],
            "wav_sha256": candidate["wav_sha256"],
            "reviewer": "user",
            "reviewed_at": "2026-08-02T00:00:00+00:00",
            "decision": "VOICE",
        }
        for model in evaluations.models
        for candidate in model.review_candidates
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_staging_fixture(
    tmp_path: Path,
    module: ModuleType,
    evaluations: EvaluationsLike,
) -> Path:
    voice_bank_root = tmp_path / "active-voice-bank"
    speakers = voice_bank_root / "speakers"
    speakers.mkdir(parents=True)
    manifest = voice_bank_root / "voice_bank_speakers.toml"
    manifest.write_text('[narrator]\nref_embed = "speakers/current.speaker.safetensors"\n')
    speaker = speakers / "current.speaker.safetensors"
    speaker.write_bytes(b"active speaker")
    baseline = tmp_path / "voice-bank-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "voice-bank-snapshot/v1",
                "voice_bank_root": str(voice_bank_root),
                "manifest": {
                    "path": str(manifest),
                    "sha256": module.sha256_file(manifest),
                    "size": manifest.stat().st_size,
                },
                "speaker_count": 1,
                "speakers": [
                    {
                        "path": str(speaker),
                        "name": speaker.name,
                        "sha256": module.sha256_file(speaker),
                        "size": speaker.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    current = {
        "root": str(voice_bank_root.resolve()),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": module.sha256_file(manifest),
        "speaker_count": 1,
        "speakers": [
            {
                "name": speaker.name,
                "path": str(speaker.resolve()),
                "sha256": module.sha256_file(speaker),
                "size": speaker.stat().st_size,
            }
        ],
    }
    staging = tmp_path / "staging-report.json"
    staging.write_text(
        json.dumps(
            {
                "schema_version": "speaker-model-staging-report/v1",
                "status": "PASS",
                "deployment_performed": False,
                "active_voice_bank_unchanged": True,
                "active_voice_bank_snapshot": str(baseline),
                "active_voice_bank_snapshot_sha256": module.sha256_file(baseline),
                "active_voice_bank_current": current,
                "proposed_staging_root": str(tmp_path / "proposed-staging"),
                "proposed_staging_root_created": False,
                "model_count": 12,
                "selections": [dict(model.selected) for model in evaluations.models],
            }
        ),
        encoding="utf-8",
    )
    return staging


def _completion_argv(
    training: TrainingFixture,
    evaluation: EvaluationFixture,
    *,
    staging: Path,
    output: Path,
) -> list[str]:
    return [
        "--phase",
        "final",
        "--training-jobs",
        str(training["jobs"]),
        "--training-status",
        str(training["status"]),
        "--training-launch-evidence",
        str(training["launch"]),
        "--evaluation-config",
        str(evaluation["config"]),
        "--evaluation-status",
        str(evaluation["status"]),
        "--review-decisions",
        str(evaluation["decisions"]),
        "--staging-report",
        str(staging),
        "--output",
        str(output),
    ]


def test_parse_args_preserves_training_run_evidence_cli_order(tmp_path: Path) -> None:
    module = _load_script()
    first = tmp_path / "quality-run-v1.json"
    second = tmp_path / "quality-run-v2.json"

    args = module._parse_args(
        [
            "--phase",
            "training",
            "--training-jobs",
            str(tmp_path / "jobs.json"),
            "--training-status",
            str(tmp_path / "status.jsonl"),
            "--training-launch-evidence",
            str(tmp_path / "launch.json"),
            "--training-run-evidence",
            str(first),
            "--training-run-evidence",
            str(second),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.training_run_evidence == [first, second]


def test_embedding_and_complete_log_contract(tmp_path: Path) -> None:
    module = _load_script()
    embedding = tmp_path / "embedding.speaker.safetensors"
    _write_embedding(embedding)
    validated = module.validate_speaker_embedding(embedding)
    log = "\n".join(
        [f"step={step} loss={step / 1000:.3f}" for step in range(20, 3001, 20)]
        + ["\x1b[32mTraining finished at step=3000.\x1b[0m"]
    )

    parsed = module.parse_final_training_run(log)

    assert validated.path == embedding.resolve()
    assert len(validated.sha256) == 64
    assert parsed.loss_event_count == 150
    assert parsed.last_step == 3000


def test_embedding_validation_reads_and_hashes_one_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    embedding = tmp_path / "embedding.speaker.safetensors"
    _write_embedding(embedding)
    original_open = Path.open
    read_opens = 0

    def counting_open(
        path: Path,
        mode: str = "r",
        *,
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        nonlocal read_opens
        if path.resolve() == embedding.resolve() and mode == "rb":
            read_opens += 1
        return original_open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", counting_open)

    validated = module.validate_speaker_embedding(embedding)

    assert read_opens == 1
    assert validated.sha256 == hashlib.sha256(embedding.read_bytes()).hexdigest()


def test_training_gate_accepts_exact_complete_run(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)

    result = module.verify_training(
        fixture["jobs"],
        fixture["status"],
        fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )

    assert len(result.models) == 12
    assert all(model.checkpoint_count == 13 for model in result.models)
    assert all(model.loss_event_count == 150 for model in result.models)
    assert len(result.models[0].latest_status["candidate_checkpoints"]) == 12
    assert all(
        len(model.latest_status["candidate_checkpoints"]) == 13 for model in result.models[1:]
    )
    assert all(
        model.run_id == module._canonical_sha256(model.latest_status) for model in result.models
    )
    legacy_report = module._training_report(result)
    assert "base_jobs" not in legacy_report
    assert "base_status" not in legacy_report
    assert "run_evidence" not in legacy_report
    assert all("run_evidence_lineage" not in model for model in legacy_report["models"])


def test_training_gate_accepts_one_quality_successor_run(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )

    result = module.verify_training(
        fixture["jobs"],
        fixture["status"],
        fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
        training_run_evidence=(evidence,),
    )

    anabel = result.models[0]
    assert result.training_jobs == jobs.resolve()
    assert result.base_training_status is not None
    assert result.base_training_status.path == fixture["base_status"].resolve()
    assert result.training_status == fixture["status"].resolve()
    assert result.training_status != result.base_training_status.path
    assert result.base_checkpoint == (fixture["root"] / "base.safetensors").resolve()
    assert anabel.config_path.parent == evidence.parent
    assert anabel.output_dir == evidence.parent / "training"
    assert [binding.evidence.path for binding in anabel.run_evidence_lineage] == [
        evidence.resolve()
    ]
    args = module._parse_args(
        [
            "--phase",
            "training",
            "--training-jobs",
            str(fixture["jobs"]),
            "--training-status",
            str(fixture["status"]),
            "--training-launch-evidence",
            str(fixture["launch"]),
            "--training-run-evidence",
            str(evidence),
            "--output",
            str(fixture["root"] / "report.json"),
        ]
    )
    protected = set(
        module._completion_protected_paths(
            args,
            training=result,
            evaluations=None,
            reviews=None,
            staging=None,
        )
    )
    lineage = anabel.run_evidence_lineage[0]
    assert fixture["base_status"].resolve() in protected
    assert {
        lineage.evidence.path,
        lineage.setup_evidence.path,
        lineage.training_jobs.path,
        lineage.training_status.path,
        lineage.queue_script.path,
        lineage.source_diagnostic.path,
        lineage.initialization_checkpoint.path,
    }.issubset(protected)


def test_training_gate_rejects_quality_setup_without_queue_binding(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, _jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    setup_path = Path(payload["setup_evidence"]["path"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["paths"].pop("queue_script")
    setup["sha256"].pop("queue_script")
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    payload["setup_evidence"]["sha256"] = module.sha256_file(setup_path)
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="paths field set mismatch"):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(evidence,),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("config", "predecessor.*config|provenance"),
        ("manifest", "predecessor.*manifest|provenance"),
        ("init_checkpoint", "predecessor.*checkpoint|candidate"),
        ("periodic_checkpoint", "predecessor.*checkpoint|candidate"),
    ],
)
def test_training_gate_rejects_stale_base_predecessor_before_successor_overlay(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    if mutation == "config":
        config = fixture["root"] / "configs" / "Anabel.json"
        payload = json.loads(config.read_text(encoding="utf-8"))
        payload["train"]["stale_predecessor"] = True
        config.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "manifest":
        manifest = fixture["root"] / "datasets" / "Anabel" / "clean.jsonl"
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"source_id": "stale"}) + "\n")
    else:
        step = 2500 if mutation == "init_checkpoint" else 2250
        checkpoint = (
            fixture["root"] / "training" / "Anabel" / f"checkpoint_{step:07d}.speaker.safetensors"
        )
        _write_embedding(checkpoint, value=1.75)
    evidence, _jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )

    with pytest.raises(ValueError, match=match):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(evidence,),
        )


def test_training_phase_reports_quality_run_lineage_and_bindings(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    output = fixture["root"] / "training-completion.json"

    exit_code = module.main(
        [
            "--phase",
            "training",
            "--training-jobs",
            str(fixture["jobs"]),
            "--training-status",
            str(fixture["status"]),
            "--training-launch-evidence",
            str(fixture["launch"]),
            "--training-run-evidence",
            str(evidence),
            "--output",
            str(output),
        ],
        runtime_probe=lambda: module.RuntimeSnapshot.idle(used_mib=973.0),
        now=lambda: "2026-08-02T05:00:00+00:00",
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["training"]["jobs"]["path"] == str(jobs.resolve())
    assert report["training"]["base_jobs"]["path"] == str(fixture["jobs"].resolve())
    assert report["training"]["base_status"]["path"] == str(fixture["base_status"].resolve())
    assert report["training"]["run_evidence"][0]["evidence"]["path"] == str(evidence.resolve())
    assert report["training"]["run_evidence"][0]["training_status"]["path"] == str(
        fixture["status"].resolve()
    )
    assert report["training"]["models"][0]["run_evidence_lineage"][0]["model_id"] == ("Anabel")


def test_training_gate_uses_last_serial_successor_for_same_model(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    first, first_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    second, second_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v2",
        predecessor_jobs=first_jobs,
    )

    result = module.verify_training(
        fixture["jobs"],
        fixture["status"],
        fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
        training_run_evidence=(first, second),
    )

    assert result.training_jobs == second_jobs.resolve()
    assert result.models[0].config_path.parent == second.parent
    assert [lineage.evidence.path for lineage in result.models[0].run_evidence_lineage] == [
        first.resolve(),
        second.resolve(),
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("config", "SHA-256 lineage|config"),
        ("checkpoint", r"checkpoint.*SHA-256 mismatch|checkpoint binding mismatch"),
    ],
)
def test_training_gate_rejects_stale_serial_predecessor_before_next_overlay(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    first, first_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    jobs = json.loads(first_jobs.read_text(encoding="utf-8"))
    anabel = next(job for job in jobs["jobs"] if job["model_id"] == "Anabel")
    if mutation == "config":
        config = Path(anabel["config"])
        payload = json.loads(config.read_text(encoding="utf-8"))
        payload["train"]["stale_between_runs"] = True
        config.write_text(json.dumps(payload), encoding="utf-8")
    else:
        checkpoint = Path(anabel["output_dir"]) / "checkpoint_0002500.speaker.safetensors"
        _write_embedding(checkpoint, value=1.875)
    second, _second_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v2",
        predecessor_jobs=first_jobs,
    )

    with pytest.raises(ValueError, match=match):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(first, second),
        )


def test_training_gate_overlays_successors_for_multiple_models(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    first, first_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    second, _second_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Kasumi",
        version="v1",
        predecessor_jobs=first_jobs,
    )

    result = module.verify_training(
        fixture["jobs"],
        fixture["status"],
        fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
        training_run_evidence=(first, second),
    )

    assert result.models[0].config_path.parent == first.parent
    assert result.models[1].config_path.parent == second.parent
    assert len(result.models[0].run_evidence_lineage) == 1
    assert len(result.models[1].run_evidence_lineage) == 1


def test_evaluation_gate_uses_effective_successor_jobs_and_status(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, _jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    training = module.verify_training(
        fixture["jobs"],
        fixture["status"],
        fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
        training_run_evidence=(evidence,),
    )
    evaluation = _write_evaluation_fixture(tmp_path, module, training)

    verified = module.verify_evaluations(
        evaluation["config"],
        evaluation["status"],
        training,
    )

    assert verified.models[0].model_id == "Anabel"
    assert training.models[0].config_path.parent == evidence.parent


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("before_sha", "append-only chain"),
        ("before_count", "append-only chain"),
        ("after_sha", "append-only chain"),
        ("after_count", "append-only chain"),
        ("row_order", "row order"),
        ("model_id", "append-only chain|model_id"),
        ("jobs_sha", "training jobs SHA-256"),
        ("config_sha", "provenance or completion"),
        ("output", "effective job path"),
        ("init_checkpoint", "initialization checkpoint lineage"),
        ("setup_queue_path", "queue script path mismatch"),
        ("setup_queue_sha", "SHA-256 lineage mismatch"),
        ("queue_sha", "queue script SHA-256"),
        ("uncovered_suffix", "append-only chain"),
    ],
)
def test_training_gate_rejects_tampered_quality_run_evidence(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, _jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    status_evidence = payload["training_status"]
    if mutation == "before_sha":
        status_evidence["before_sha256"] = "0" * 64
    elif mutation == "before_count":
        status_evidence["before_row_count"] -= 1
    elif mutation == "after_sha":
        status_evidence["after_sha256"] = "0" * 64
    elif mutation == "after_count":
        status_evidence["after_row_count"] -= 1
    elif mutation == "row_order":
        lines = fixture["status"].read_text(encoding="utf-8").splitlines(keepends=True)
        lines[-2], lines[-1] = lines[-1], lines[-2]
        fixture["status"].write_text("".join(lines), encoding="utf-8")
        status_evidence["after_sha256"] = module.sha256_file(fixture["status"])
    elif mutation == "model_id":
        payload["model_id"] = "Kasumi"
    elif mutation == "jobs_sha":
        payload["training_jobs"]["sha256"] = "0" * 64
    elif mutation == "config_sha":
        payload["run"]["config_sha256"] = "0" * 64
    elif mutation in {
        "output",
        "init_checkpoint",
        "setup_queue_path",
        "setup_queue_sha",
    }:
        setup_path = Path(payload["setup_evidence"]["path"])
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if mutation == "output":
            setup["paths"]["output_dir"] = str(fixture["root"] / "training" / "Anabel")
        elif mutation == "init_checkpoint":
            setup["changes"]["speaker_inversion_init_embedding_sha256"] = "0" * 64
        elif mutation == "setup_queue_path":
            substitute = fixture["root"] / "substitute" / "run_600m_speaker_training_queue.py"
            substitute.parent.mkdir()
            substitute.write_bytes(Path(setup["paths"]["queue_script"]).read_bytes())
            setup["paths"]["queue_script"] = str(substitute)
        else:
            setup["sha256"]["queue_script"] = "0" * 64
        setup_path.write_text(json.dumps(setup), encoding="utf-8")
        payload["setup_evidence"]["sha256"] = module.sha256_file(setup_path)
    elif mutation == "queue_sha":
        payload["queue_script"]["sha256"] = "0" * 64
    else:
        with fixture["status"].open("a", encoding="utf-8") as status_file:
            status_file.write(json.dumps({"model_id": "undeclared"}) + "\n")
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=match):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(evidence,),
        )


def test_training_gate_rejects_duplicate_and_reordered_quality_evidence(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    first, first_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    second, _second_jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Kasumi",
        version="v1",
        predecessor_jobs=first_jobs,
    )

    with pytest.raises(ValueError, match="duplicate"):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(first, first),
        )
    with pytest.raises(ValueError, match="append-only chain"):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(second, first),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("base_bytes", "row count|after SHA"),
        ("run_bytes", "append-only chain"),
        ("outside_path", "outside its run root"),
        ("same_path_alias", "aliased|outside its run root"),
        ("missing_file", "missing"),
        ("final_cli_mismatch", "final training status"),
    ],
)
def test_training_gate_rejects_invalid_versioned_status_lineage(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, _jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    cli_status = fixture["status"]
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if mutation == "base_bytes":
        with fixture["base_status"].open("a", encoding="utf-8") as status_file:
            status_file.write(json.dumps({"model_id": "base-tamper"}) + "\n")
    elif mutation == "run_bytes":
        rows = [json.loads(line) for line in cli_status.read_text(encoding="utf-8").splitlines()]
        rows[-1]["status"] = "failed"
        cli_status.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    elif mutation in {"outside_path", "same_path_alias"}:
        if mutation == "outside_path":
            replacement = fixture["root"] / "wrong-versioned-status.jsonl"
            replacement.write_bytes(cli_status.read_bytes())
        else:
            replacement = fixture["base_status"]
        payload["training_status"]["path"] = str(replacement)
        setup_path = Path(payload["setup_evidence"]["path"])
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        setup["paths"]["status"] = str(replacement)
        _refresh_quality_setup_binding(payload, module, setup)
        evidence.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "missing_file":
        cli_status.unlink()
    else:
        cli_status = fixture["base_status"]

    with pytest.raises((FileNotFoundError, TypeError, ValueError), match=match):
        module.verify_training(
            fixture["jobs"],
            cli_status,
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(evidence,),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("model_config", "undeclared drift"),
        ("train_config", "undeclared drift"),
        ("target_job", "undeclared target job"),
        ("non_target_job", "non-target job"),
        ("top_level", "field set mismatch"),
    ],
)
def test_training_gate_rejects_undeclared_successor_delta(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, _jobs = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    if mutation in {"model_config", "train_config"}:

        def mutate_config(config: dict[str, Any]) -> None:
            if mutation == "model_config":
                config["model"] = {"undeclared": True}
            else:
                config["train"]["undeclared"] = True

        _refresh_quality_config_provenance(evidence, module, mutate=mutate_config)
    else:

        def mutate_jobs(jobs: dict[str, Any]) -> None:
            if mutation == "target_job":
                jobs["jobs"][0]["undeclared"] = True
            elif mutation == "non_target_job":
                jobs["jobs"][1]["command"].append("--undeclared")
            else:
                jobs["undeclared"] = True

        _refresh_quality_jobs_provenance(evidence, module, mutate=mutate_jobs)

    with pytest.raises((TypeError, ValueError), match=match):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(evidence,),
        )


@pytest.mark.parametrize(
    "flag",
    ["--config", "--output-dir", "--manifest", "--init-checkpoint"],
)
@pytest.mark.parametrize("attack", ["different_path", "missing", "duplicate"])
def test_training_gate_rejects_quality_successor_command_path_attacks(
    tmp_path: Path,
    flag: str,
    attack: str,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    evidence, jobs_path = _append_quality_run_evidence(
        fixture,
        module,
        model_id="Anabel",
        version="v1",
    )
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    target = next(job for job in jobs["jobs"] if job["model_id"] == "Anabel")
    command = target["command"]
    value_index = command.index(flag) + 1
    original_value = command[value_index]
    if attack == "different_path":
        command[value_index] = str(tmp_path / f"attacker-{flag.removeprefix('--')}")
    elif attack == "missing":
        del command[value_index - 1 : value_index + 1]
    else:
        command.extend([flag, original_value])
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    jobs_sha = module.sha256_file(jobs_path)

    setup_path = Path(evidence_payload["setup_evidence"]["path"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["sha256"]["jobs"] = jobs_sha
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    evidence_payload["training_jobs"]["sha256"] = jobs_sha
    evidence_payload["setup_evidence"]["sha256"] = module.sha256_file(setup_path)
    evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"command.*{flag}"):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
            training_run_evidence=(evidence,),
        )


def test_seeded_training_run_matches_declared_evaluation_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    run_provenance, run_provenance_sha256 = _seed_existing_training_run(fixture, module)
    original_open = Path.open
    provenance_read_count = 0

    def count_provenance_reads(
        path: Path,
        mode: str = "r",
        *,
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        nonlocal provenance_read_count
        if path == run_provenance and mode == "rb":
            provenance_read_count += 1
        return original_open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", count_provenance_reads)
    training = module.verify_training(
        fixture["jobs"],
        fixture["status"],
        fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    seeded_contract = replace(
        training,
        models=(
            replace(training.models[0], run_id=run_provenance_sha256),
            *training.models[1:],
        ),
    )
    evaluation_fixture = _write_evaluation_fixture(tmp_path, module, seeded_contract)

    result = module.verify_evaluations(
        evaluation_fixture["config"],
        evaluation_fixture["status"],
        training,
    )

    assert training.models[0].run_id == run_provenance_sha256
    assert training.models[0].run_id != module._canonical_sha256(training.models[0].latest_status)
    assert result.models[0].model_id == "Anabel"
    assert provenance_read_count == 1


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("nested_object", "seeded existing run must be an object"),
        ("missing_path", "requires nonempty string run_provenance_path"),
        ("relative_path", "run provenance path must be absolute"),
        ("noncanonical_path", "run provenance path is unsafe or missing"),
        ("symlink", "run provenance path is unsafe or missing"),
        ("missing_file", "run provenance path is unsafe or missing"),
        ("sha256_format", "requires lowercase SHA-256 run_provenance_sha256"),
        ("sha256_mismatch", "run provenance SHA-256 mismatch"),
        ("invalid_json", "invalid JSON"),
        ("provenance_json", "JSON document must be an object"),
        ("missing_model_id", "requires nonempty string model_id"),
        ("empty_model_id", "requires nonempty string model_id"),
        ("model_id_mismatch", "run provenance model_id mismatch"),
    ],
)
def test_seeded_training_run_rejects_invalid_run_provenance(
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    run_provenance, _run_provenance_sha256 = _seed_existing_training_run(fixture, module)
    seeded_status = fixture["status_rows"][0]
    seeded_existing_run = seeded_status["seeded_existing_run"]
    assert isinstance(seeded_existing_run, dict)
    if failure == "nested_object":
        seeded_status["seeded_existing_run"] = "invalid"
    elif failure == "missing_path":
        seeded_existing_run.pop("run_provenance_path")
    elif failure == "relative_path":
        seeded_existing_run["run_provenance_path"] = run_provenance.name
    elif failure == "noncanonical_path":
        intermediate = run_provenance.parent / "intermediate"
        intermediate.mkdir()
        seeded_existing_run["run_provenance_path"] = str(intermediate / ".." / run_provenance.name)
    elif failure == "symlink":
        symlink = run_provenance.with_name("anabel-run-provenance-link.json")
        symlink.symlink_to(run_provenance)
        seeded_existing_run["run_provenance_path"] = str(symlink)
    elif failure == "missing_file":
        seeded_existing_run["run_provenance_path"] = str(
            run_provenance.with_name("missing-run-provenance.json")
        )
    elif failure == "sha256_format":
        seeded_existing_run["run_provenance_sha256"] = "invalid"
    elif failure == "sha256_mismatch":
        seeded_existing_run["run_provenance_sha256"] = "0" * 64
    elif failure == "invalid_json":
        run_provenance.write_text("{", encoding="utf-8")
        seeded_existing_run["run_provenance_sha256"] = module.sha256_file(run_provenance)
    elif failure == "provenance_json":
        run_provenance.write_text("[]", encoding="utf-8")
        seeded_existing_run["run_provenance_sha256"] = module.sha256_file(run_provenance)
    elif failure == "missing_model_id":
        run_provenance.write_text("{}", encoding="utf-8")
        seeded_existing_run["run_provenance_sha256"] = module.sha256_file(run_provenance)
    elif failure == "empty_model_id":
        run_provenance.write_text(json.dumps({"model_id": ""}), encoding="utf-8")
        seeded_existing_run["run_provenance_sha256"] = module.sha256_file(run_provenance)
    else:
        run_provenance.write_text(
            json.dumps({"model_id": "different-model"}),
            encoding="utf-8",
        )
        seeded_existing_run["run_provenance_sha256"] = module.sha256_file(run_provenance)
    fixture["status"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fixture["status_rows"]),
        encoding="utf-8",
    )
    _refresh_launch_status_hashes(fixture, module)

    with pytest.raises((TypeError, ValueError), match=match):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_checkpoint", "inventory"),
        ("nonfinite_checkpoint", "nonfinite"),
        ("final_mismatch", "final checkpoint"),
        ("candidate_inventory", "candidates omit"),
        ("bad_config", "config contract"),
        ("later_running", "row count|latest training event"),
        ("launcher_failed", "launch evidence"),
        ("status_row_count", "status_row_count"),
        ("status_prefix", "prefix SHA"),
        ("new_row_order", "new row sequence"),
        ("queue_script", "queue script SHA-256"),
        ("residual_process", "residual workflow process"),
        ("compute_application", "NVIDIA compute"),
        ("probe_error", "probe reported"),
        ("gpu_not_released", "GPU memory"),
    ],
)
def test_training_gate_fails_closed(tmp_path: Path, mutation: str, match: str) -> None:
    module = _load_script()
    fixture = _write_training_fixture(tmp_path, module)
    runtime = module.RuntimeSnapshot.idle(used_mib=973.0)
    first_output = Path(fixture["root"]) / "training" / "Anabel"
    if mutation == "missing_checkpoint":
        (first_output / "checkpoint_0000250.speaker.safetensors").unlink()
    elif mutation == "nonfinite_checkpoint":
        _write_embedding(first_output / "checkpoint_0000250.speaker.safetensors", finite=False)
    elif mutation == "final_mismatch":
        final = first_output / "checkpoint_final.speaker.safetensors"
        _write_embedding(final, value=2.0)
    elif mutation == "candidate_inventory":
        rows = fixture["status_rows"]
        candidates = rows[0]["candidate_checkpoints"]
        assert isinstance(candidates, list)
        final = first_output / "checkpoint_final.speaker.safetensors"
        rows[0]["candidate_checkpoints"] = [
            *candidates[:-1],
            {"path": str(final), "sha256": module.sha256_file(final)},
        ]
        fixture["status"].write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        _refresh_launch_status_hashes(fixture, module)
    elif mutation == "bad_config":
        config = Path(fixture["root"]) / "configs" / "Anabel.json"
        payload = json.loads(config.read_text(encoding="utf-8"))
        payload["train"]["max_steps"] = 2999
        config.write_text(json.dumps(payload), encoding="utf-8")
        rows = fixture["status_rows"]
        assert isinstance(rows, list)
        rows[0]["config_sha256"] = module.sha256_file(config)
        Path(fixture["status"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        _refresh_launch_status_hashes(fixture, module)
    elif mutation == "later_running":
        rows = fixture["status_rows"]
        assert isinstance(rows, list)
        later = dict(rows[0]) | {"event": "started", "status": "running", "exit_code": None}
        with Path(fixture["status"]).open("a", encoding="utf-8") as status:
            status.write(json.dumps(later) + "\n")
        _refresh_launch_status_hashes(fixture, module)
    elif mutation == "launcher_failed":
        path = Path(fixture["launch"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["queue_exit_code"] = 9
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "status_row_count":
        path = fixture["launch"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status_row_count"] = 21
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "status_prefix":
        path = fixture["launch"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status_before_sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "new_row_order":
        rows = fixture["status_rows"]
        rows[2], rows[4] = rows[4], rows[2]
        fixture["status"].write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        path = fixture["launch"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_sha = module.sha256_file(fixture["status"])
        payload["status_after_sha256"] = current_sha
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "queue_script":
        (fixture["root"] / "run_600m_speaker_training_queue.py").write_text(
            "# changed queue\n", encoding="utf-8"
        )
    elif mutation == "residual_process":
        runtime = module.normalize_runtime_snapshot(
            processes=[{"pid": 99, "command_line": "python train.py --config cfg.json"}],
            compute_applications=[],
            gpu_memory_used_mib=973,
            errors=[],
        )
    elif mutation == "compute_application":
        runtime = module.normalize_runtime_snapshot(
            processes=[],
            compute_applications=[{"pid": 99, "process_name": "python.exe"}],
            gpu_memory_used_mib=973,
            errors=[],
        )
    elif mutation == "probe_error":
        runtime = module.RuntimeSnapshot(
            conflicting_processes=(),
            related_compute_applications=(),
            gpu_memory_used_mib=973.0,
            errors=("CIM failed",),
            observed_at="2026-08-02T00:00:00+00:00",
        )
    else:
        runtime = module.RuntimeSnapshot.idle(used_mib=1200.0)

    with pytest.raises(ValueError, match=match):
        module.verify_training(
            fixture["jobs"],
            fixture["status"],
            fixture["launch"],
            runtime,
            256.0,
        )


@pytest.mark.parametrize("mutation", ["nonfinite", "extra_tensor"])
def test_embedding_fails_closed(tmp_path: Path, mutation: str) -> None:
    module = _load_script()
    embedding = tmp_path / "embedding.speaker.safetensors"
    _write_embedding(
        embedding,
        finite=mutation != "nonfinite",
        extra=mutation == "extra_tensor",
    )

    with pytest.raises(ValueError):
        module.validate_speaker_embedding(embedding)


@pytest.mark.parametrize(
    "log",
    [
        "step=20 loss=1.0\nTraining finished at step=3000.\n",
        "\n".join(
            [f"step={step} loss=nan" for step in range(20, 3001, 20)]
            + ["Training finished at step=3000."]
        ),
        "\n".join(f"step={step} loss=1.0" for step in range(20, 3001, 20)),
    ],
)
def test_log_fails_closed(log: str) -> None:
    module = _load_script()

    with pytest.raises(ValueError):
        module.parse_final_training_run(log)


def test_runtime_snapshot_idle_and_semantic_process_filter() -> None:
    module = _load_script()
    idle = module.RuntimeSnapshot.idle(used_mib=973.0)
    assert idle.gpu_memory_used_mib == pytest.approx(973.0)
    assert idle.errors == ()

    raw = [
        {"pid": 10, "parent_pid": 1, "command_line": "python train.py --ancestor"},
        {
            "pid": 11,
            "parent_pid": 1,
            "command_line": ("python train.py --note verify_600m_speaker_retraining_completion.py"),
        },
        {
            "pid": 12,
            "parent_pid": 1,
            "command_line": "python train.py --note powershell-ConvertTo-Json",
        },
        {
            "pid": 13,
            "parent_pid": 1,
            "command_line": "python train.py --note 'remote-python -c diagnostic'",
        },
    ]
    snapshot = module.normalize_runtime_snapshot(
        processes=raw,
        compute_applications=[],
        gpu_memory_used_mib=1000,
        errors=[],
        excluded_pids={10},
    )
    assert [row["pid"] for row in snapshot.conflicting_processes] == [11, 12, 13]


def test_runtime_snapshot_filters_wddm_ui_but_keeps_python_and_conflict_pid() -> None:
    module = _load_script()
    processes = [{"pid": 30, "command_line": "custom-worker.exe train.py --config cfg.json"}]
    compute = [
        {"pid": 10, "process_name": "python.exe"},
        {"pid": 20, "process_name": "python.exe"},
        {"pid": 21, "process_name": r"C:\Python310\pythonw.exe"},
        {"pid": 30, "process_name": "custom-worker.exe"},
        {"pid": 40, "process_name": "dwm.exe"},
        {"pid": 41, "process_name": "explorer.exe"},
        {"pid": 42, "process_name": "ordinary-ui.exe"},
        {"process_name": "unknown compute application"},
    ]

    snapshot = module.normalize_runtime_snapshot(
        processes=processes,
        compute_applications=compute,
        gpu_memory_used_mib=973,
        errors=[],
        excluded_pids={10},
    )

    assert snapshot.related_compute_applications == (compute[1], compute[2], compute[3])


def test_windows_probe_uses_cim_and_inventory_ancestor_exclusion() -> None:
    module = _load_script()
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "powershell.exe":
            payload = [
                {
                    "ProcessId": 100,
                    "ParentProcessId": 50,
                    "CreationDate": "20260802010000.000000+000",
                    "CommandLine": "python train.py --config ancestor-child.json",
                    "Name": "python.exe",
                },
                {
                    "ProcessId": 50,
                    "ParentProcessId": 1,
                    "CreationDate": "20260802000000.000000+000",
                    "CommandLine": "powershell verifier ancestor",
                    "Name": "powershell.exe",
                },
                {
                    "ProcessId": 200,
                    "ParentProcessId": 1,
                    "CreationDate": "20260802020000.000000+000",
                    "CommandLine": "python launch_600m_training_queue_speed_v1.py",
                    "Name": "python.exe",
                },
                {
                    "ProcessId": 300,
                    "ParentProcessId": 1,
                    "CreationDate": "20260802030000.000000+000",
                    "CommandLine": "remote-python -c diagnostic train.py",
                    "Name": "python.exe",
                },
                {
                    "ProcessId": 4,
                    "ParentProcessId": 0,
                    "CreationDate": "20260801000000.000000+000",
                    "CommandLine": None,
                    "Name": "System",
                },
            ]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if "--query-gpu=memory.used" in command:
            return subprocess.CompletedProcess(command, 0, "973\n", "")
        return subprocess.CompletedProcess(command, 0, "200, python.exe\n", "")

    snapshot = module.probe_runtime(
        runner=runner,
        platform_name="nt",
        current_pid=100,
    )

    assert calls[0][0] == "powershell.exe"
    assert [row["ProcessId"] for row in snapshot.conflicting_processes] == [200, 300]
    assert [row["pid"] for row in snapshot.related_compute_applications] == [200]
    assert snapshot.errors == ()


@pytest.mark.parametrize("cim_stdout", ["", "null", "[]", '[1, "invalid"]'])
def test_windows_probe_rejects_empty_or_invalid_cim_inventory(cim_stdout: str) -> None:
    module = _load_script()

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "powershell.exe":
            return subprocess.CompletedProcess(command, 0, cim_stdout, "")
        if "--query-gpu=memory.used" in command:
            return subprocess.CompletedProcess(command, 0, "973\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    snapshot = module.probe_runtime(
        runner=runner,
        platform_name="nt",
        current_pid=100,
    )

    assert snapshot.errors
    assert any("process inventory" in error for error in snapshot.errors)


@pytest.mark.parametrize(
    "current_row",
    [
        None,
        {"ProcessId": 100, "CommandLine": "python verifier.py", "Name": "python.exe"},
        {
            "ProcessId": 100,
            "ParentProcessId": -1,
            "CommandLine": "python verifier.py",
            "Name": "python.exe",
        },
        {
            "ProcessId": 100,
            "ParentProcessId": "50",
            "CommandLine": "python verifier.py",
            "Name": "python.exe",
        },
        {"ProcessId": 100, "ParentProcessId": 50, "CommandLine": None, "Name": "python.exe"},
        {"ProcessId": 100, "ParentProcessId": 50, "CommandLine": "  ", "Name": "python.exe"},
    ],
)
def test_windows_probe_requires_complete_current_process_row(
    current_row: dict[str, object] | None,
) -> None:
    module = _load_script()
    inventory: list[dict[str, object]] = [
        {
            "ProcessId": 50,
            "ParentProcessId": 1,
            "CommandLine": None,
            "Name": "System",
        }
    ]
    if current_row is not None:
        inventory.append(current_row)

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "powershell.exe":
            return subprocess.CompletedProcess(command, 0, json.dumps(inventory), "")
        if "--query-gpu=memory.used" in command:
            return subprocess.CompletedProcess(command, 0, "973\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    snapshot = module.probe_runtime(
        runner=runner,
        platform_name="nt",
        current_pid=100,
    )

    assert snapshot.errors
    assert any("current process" in error for error in snapshot.errors)


def test_windows_probe_detects_python_compute_with_missing_pid_or_cim_command() -> None:
    module = _load_script()

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "powershell.exe":
            inventory = [
                {
                    "ProcessId": 100,
                    "ParentProcessId": 50,
                    "CommandLine": "python verifier.py",
                    "Name": "python.exe",
                },
                {
                    "ProcessId": 44040,
                    "ParentProcessId": 1,
                    "CommandLine": None,
                    "Name": "python.exe",
                },
            ]
            return subprocess.CompletedProcess(command, 0, json.dumps(inventory), "")
        if "--query-gpu=memory.used" in command:
            return subprocess.CompletedProcess(command, 0, "973\n", "")
        return subprocess.CompletedProcess(
            command,
            0,
            "N/A, python.exe\n44040, python.exe\n",
            "",
        )

    snapshot = module.probe_runtime(
        runner=runner,
        platform_name="nt",
        current_pid=100,
    )

    assert snapshot.related_compute_applications == (
        {"pid": None, "process_name": "python.exe"},
        {"pid": 44040, "process_name": "python.exe"},
    )


def test_case_matrix_and_snapshot_are_exact(tmp_path: Path) -> None:
    module = _load_script()
    hard_gate_text_ids = {
        "sentence_unko",
        "sentence_chinko",
        "sentence_manko",
        "control",
    }
    rows = [
        {
            "model_id": "model-00",
            "checkpoint_step": step,
            "text_id": text_id,
            "seed": seed,
            "style": style,
            "case_id": f"{step}:{text_id}:{seed}:{style}",
            "metric_gate_applied": text_id in hard_gate_text_ids,
        }
        for step in module.EXPECTED_EVALUATION_STEPS
        for text_id in module.EXPECTED_TEXT_IDS
        for seed in module.EXPECTED_SEEDS
        for style in module.EXPECTED_STYLES
    ]
    output = tmp_path / "output"
    output.mkdir()
    (output / "artifact.txt").write_text("immutable\n", encoding="utf-8")

    module.validate_case_matrix(rows, model_id="model-00")
    snapshot = module.snapshot_path(output)

    assert module.EXPECTED_EVALUATION_STEPS == (1000, 1500, 2000, 2500, 3000)
    assert module.EXPECTED_EVALUATION_CASE_COUNT == 140
    assert len(rows) == 140
    assert snapshot["kind"] == "directory"
    assert snapshot["files"] == {"artifact.txt": module.sha256_file(output / "artifact.txt")}

    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="matrix"):
        module.validate_case_matrix(rows, model_id="model-00")


def test_case_matrix_rejects_wrong_metric_gate_distribution() -> None:
    module = _load_script()
    hard_gate_text_ids = {
        "sentence_unko",
        "sentence_chinko",
        "sentence_manko",
        "control",
    }
    rows = [
        {
            "model_id": "model-00",
            "checkpoint_step": step,
            "text_id": text_id,
            "seed": seed,
            "style": style,
            "case_id": f"{step}:{text_id}:{seed}:{style}",
            "metric_gate_applied": text_id in hard_gate_text_ids,
        }
        for step in module.EXPECTED_EVALUATION_STEPS
        for text_id in module.EXPECTED_TEXT_IDS
        for seed in module.EXPECTED_SEEDS
        for style in module.EXPECTED_STYLES
    ]
    rows[0]["metric_gate_applied"] = True

    with pytest.raises(ValueError, match="metric gate distribution"):
        module.validate_case_matrix(rows, model_id="model-00")


def test_evaluation_gate_accepts_exact_stage_and_case_closure(tmp_path: Path) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)

    result = module.verify_evaluations(fixture["config"], fixture["status"], training)

    assert result.stage_count == 49
    assert len(result.models) == 12
    assert all(model.case_count == 140 for model in result.models)
    assert all(model.selected["checkpoint_step"] == 1000 for model in result.models)


def test_evaluation_gate_allows_untracked_upstream_files_outside_package(tmp_path: Path) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    (training_fixture["root"] / "upstream" / "scratch.txt").write_text(
        "allowed\n", encoding="utf-8"
    )

    result = module.verify_evaluations(fixture["config"], fixture["status"], training)

    assert result.stage_count == 49


@pytest.mark.parametrize("mutation", ["tracked", "untracked"])
def test_evaluation_gate_rejects_current_dirty_upstream_package(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    package = training_fixture["root"] / "upstream" / "irodori_tts"
    target = package / ("runtime.py" if mutation == "tracked" else "new.py")
    target.write_text("CHANGED = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="package is dirty or untracked"):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


def test_generation_stage_fingerprint_binds_runtime_provenance(tmp_path: Path) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    runtime_root = fixture["config"].parent
    provenance = runtime_root / "upstream-runtime-provenance.json"
    provenance.write_bytes(provenance.read_bytes() + b"\n")
    snapshot_manifest = json.loads(fixture["runtime_manifest"].read_text(encoding="utf-8"))
    snapshot_manifest["files"][provenance.name] = {
        "sha256": module.sha256_file(provenance),
        "size": provenance.stat().st_size,
    }
    fixture["runtime_manifest"].write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="producer command mismatch"):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


def test_runtime_snapshot_rejects_tampered_upstream_package_archive(tmp_path: Path) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    runtime_root = fixture["config"].parent
    archive_path = runtime_root / "upstream-runtime-package.zip"
    replacement = runtime_root / "upstream-runtime-package.changed.zip"
    with zipfile.ZipFile(archive_path, "r") as source:
        entries = [(info, source.read(info)) for info in source.infolist()]
    with zipfile.ZipFile(
        replacement,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as destination:
        for index, (info, contents) in enumerate(entries):
            destination.writestr(
                info,
                contents + (b"# changed\n" if index == 0 else b""),
                compresslevel=9,
            )
    replacement.replace(archive_path)
    snapshot_manifest = json.loads(fixture["runtime_manifest"].read_text(encoding="utf-8"))
    snapshot_manifest["files"][archive_path.name] = {
        "sha256": module.sha256_file(archive_path),
        "size": archive_path.stat().st_size,
    }
    fixture["runtime_manifest"].write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="package archive hash mismatch"):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("checkpoint_count", 5),
        ("evaluation_case_count", 140),
        ("hard_gate_metric_case_count_per_checkpoint", 16),
        ("diagnostic_word_case_count_per_checkpoint", 12),
    ],
)
@pytest.mark.parametrize("mutation", ["omitted", "incorrect"])
def test_evaluation_verification_requires_exact_matrix_metadata(
    tmp_path: Path,
    field: str,
    expected: int,
    mutation: str,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    first = fixture["evaluation_dirs"][0]
    verification_path = first / "evaluation-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if mutation == "omitted":
        verification.pop(field)
    else:
        verification[field] = expected + 1
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    _refresh_evaluation_status_stage(fixture, module, model_id=training.model_ids[0])

    with pytest.raises(ValueError, match="evaluation verification did not pass"):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("snapshot_tamper", "differs from original"),
        ("manifest_hash", "file content mismatch"),
        ("manifest_extra", "producer file inventory"),
        ("manifest_missing", "producer file inventory"),
        ("snapshot_extra", "exact inventory"),
        ("original_drift", "source input changed|original training input drift"),
        ("path_escape", "producer file inventory"),
        ("runtime_config_source", "source input binding"),
    ],
)
def test_runtime_evaluation_snapshot_fails_closed(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    manifest_path = fixture["runtime_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    if mutation == "snapshot_tamper":
        snapshot_jobs = fixture["snapshot_jobs"]
        snapshot_jobs.write_bytes(snapshot_jobs.read_bytes() + b"\n")
        files["training-jobs-speed-v1.json"] = {
            "sha256": module.sha256_file(snapshot_jobs),
            "size": snapshot_jobs.stat().st_size,
        }
    elif mutation == "manifest_hash":
        files["evaluation-queue-runtime.json"]["sha256"] = "0" * 64
    elif mutation == "manifest_extra":
        files["ghost.txt"] = {"sha256": "0" * 64, "size": 0}
    elif mutation == "manifest_missing":
        del files["scripts/run_600m_speaker_evaluation_queue.py"]
    elif mutation == "snapshot_extra":
        (manifest_path.parent / "unowned.txt").write_text("extra\n", encoding="utf-8")
    elif mutation == "original_drift":
        training.training_status.write_bytes(training.training_status.read_bytes() + b"\n")
    elif mutation == "path_escape":
        files["../escape.json"] = dict(files["evaluation-queue-runtime.json"])
    else:
        runtime_config = fixture["config"]
        runtime_document = json.loads(runtime_config.read_text(encoding="utf-8"))
        runtime_document["runtime_only_mutation"] = True
        runtime_config.write_text(
            json.dumps(runtime_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files["evaluation-queue-runtime.json"] = {
            "sha256": module.sha256_file(runtime_config),
            "size": runtime_config.stat().st_size,
        }
        rows = [
            json.loads(line) for line in fixture["status"].read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["config_sha256"] = module.sha256_file(runtime_config)
            row["stage_fingerprint"] = module._current_stage_fingerprint(
                runtime_document,
                row,
                base=runtime_config.parent,
            )
        fixture["status"].write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    if mutation not in {"snapshot_extra", "original_drift"}:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=match):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


@pytest.mark.parametrize("mutation", ["paired_ghost", "paired_script_source"])
def test_runtime_snapshot_rejects_paired_entries_outside_producer_inventory(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    manifest_path = fixture["runtime_manifest"]
    runtime_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    added_content = b"# paired snapshot entry\n"
    if mutation == "paired_ghost":
        added = runtime_root / "ghost.txt"
    else:
        added = runtime_root / "scripts" / "unexpected_component.py"
        source = fixture["root"] / "source-scripts" / added.name
        source.write_bytes(added_content)
        manifest["source_inputs"][str(source.resolve())] = module.sha256_file(source)
    added.write_bytes(added_content)
    manifest["files"][added.relative_to(runtime_root).as_posix()] = {
        "sha256": module.sha256_file(added),
        "size": added.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="producer file inventory"):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


def test_evaluation_rejects_source_config_when_status_binds_runtime_config(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)

    with pytest.raises(ValueError, match="status config path or SHA-256 mismatch"):
        module.verify_evaluations(fixture["source_config"], fixture["status"], training)


@pytest.mark.parametrize("mutation", ["external_component", "bogus_output"])
def test_evaluation_status_rejects_self_consistent_nonproducer_stage_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    runtime_config = json.loads(fixture["config"].read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in fixture["status"].read_text(encoding="utf-8").splitlines()]
    target_stage = f"{training.model_ids[1]}:generation"
    row = next(item for item in rows if item["stage"] == target_stage)
    if mutation == "external_component":
        external = fixture["root"] / "source-scripts" / "generate_600m_checkpoint_audio_remote.py"
        row["component_script"] = {
            "path": str(external),
            "sha256": module.sha256_file(external),
        }
        row["command"][1] = str(external)
        row["stage_fingerprint"] = module._current_stage_fingerprint(
            runtime_config,
            row,
            base=fixture["config"].parent,
        )
    else:
        bogus = fixture["root"] / "bogus-generation"
        bogus.mkdir()
        (bogus / "generation-results.jsonl").write_text("{}\n", encoding="utf-8")
        row["outputs"] = [module.snapshot_path(bogus)]
        row["stage_fingerprint"] = module._current_stage_fingerprint(
            runtime_config,
            row,
            base=fixture["config"].parent,
        )
    fixture["status"].write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"producer (component|output|command)"):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


def test_evaluation_rejects_reused_generation_with_two_proof_files(tmp_path: Path) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    runtime_config = json.loads(fixture["config"].read_text(encoding="utf-8"))
    reused_generation = Path(runtime_config["models"][0]["reuse"]["generation_dir"])
    (reused_generation / "canonicalization-report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one generation proof required"):
        module.verify_evaluations(fixture["config"], fixture["status"], training)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_stage", "stage set"),
        ("stale_fingerprint", "fingerprint"),
        ("changed_snapshot", "fingerprint|snapshot changed|producer output"),
        ("live_lock", "lock"),
        ("bad_matrix", "snapshot changed|matrix|producer output"),
        ("stale_embedding", "checkpoint file changed"),
        ("changed_packet", "snapshot changed|artifact hash|copied asset|producer output"),
    ],
)
def test_evaluation_gate_fails_closed(tmp_path: Path, mutation: str, match: str) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    fixture = _write_evaluation_fixture(tmp_path, module, training)
    status = Path(fixture["status"])
    first = fixture["evaluation_dirs"][0]
    if mutation == "missing_stage":
        lines = status.read_text(encoding="utf-8").splitlines()
        status.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    elif mutation == "stale_fingerprint":
        rows = [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines()]
        rows[0]["stage_fingerprint"] = "0" * 64
        status.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif mutation == "changed_snapshot":
        rows = [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines()]
        output = rows[0]["outputs"][0]
        output_root = Path(output["path"])
        relative = next(iter(output["files"]))
        (output_root / relative).write_text("changed\n", encoding="utf-8")
    elif mutation == "live_lock":
        status.with_suffix(status.suffix + ".lock").write_text("locked\n", encoding="utf-8")
    elif mutation == "bad_matrix":
        results = first / "evaluation-results.jsonl"
        lines = results.read_text(encoding="utf-8").splitlines()
        results.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        verification = first / "evaluation-verification.json"
        payload = json.loads(verification.read_text(encoding="utf-8"))
        payload["artifact_sha256"][str(results.resolve())] = module.sha256_file(results)
        verification.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "stale_embedding":
        training.models[0].checkpoints[3].path.write_bytes(b"changed")
    else:
        packet_asset = first / "review_packet" / "audio" / "Anabel.wav"
        packet_asset.write_bytes(b"changed")

    with pytest.raises(ValueError, match=match):
        module.verify_evaluations(fixture["config"], status, training)


def test_review_decisions_have_three_outcomes(tmp_path: Path) -> None:
    module = _load_script()
    wav = tmp_path / "candidate.wav"
    wav.write_bytes(b"candidate")
    packet_root = tmp_path / "evaluation" / "review_packet"
    (packet_root / "audio").mkdir(parents=True)
    copied = packet_root / "audio" / "candidate.wav"
    copied.write_bytes(wav.read_bytes())
    nonselected_wav = tmp_path / "candidate-1500.wav"
    nonselected_wav.write_bytes(b"candidate-1500")
    copied_nonselected = packet_root / "audio" / "candidate-1500.wav"
    copied_nonselected.write_bytes(nonselected_wav.read_bytes())
    candidate = {
        "case_id": "case-1",
        "model_id": "model-00",
        "checkpoint_step": 1000,
        "wav_path": str(wav),
        "wav_sha256": module.sha256_file(wav),
    }
    nonselected_candidate = {
        "case_id": "case-2",
        "model_id": "model-00",
        "checkpoint_step": 1500,
        "wav_path": str(nonselected_wav),
        "wav_sha256": module.sha256_file(nonselected_wav),
    }
    (packet_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-review-packet/v1",
                "review_candidates": [
                    {
                        "case_id": "case-1",
                        "wav": {
                            "path": "audio/candidate.wav",
                            "sha256": module.sha256_file(copied),
                        },
                        "spectrogram": None,
                        "paired_controls": [],
                    },
                    {
                        "case_id": "case-2",
                        "wav": {
                            "path": "audio/candidate-1500.wav",
                            "sha256": module.sha256_file(copied_nonselected),
                        },
                        "spectrogram": None,
                        "paired_controls": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    model = module.EvaluationModelSummary(
        model_id="model-00",
        evaluation_dir=packet_root.parent,
        manifest_path=tmp_path / "manifest.json",
        case_count=140,
        selected={"model_id": "model-00", "checkpoint_step": 1000},
        review_candidates=(candidate, nonselected_candidate),
    )
    evaluations = module.EvaluationVerification(
        stage_count=49,
        models=(model,),
        evaluation_config=tmp_path / "config.json",
        evaluation_config_sha256="a" * 64,
        evaluation_status=tmp_path / "status.jsonl",
        evaluation_status_sha256="b" * 64,
    )
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("", encoding="utf-8")
    assert module.verify_reviews(evaluations, decisions).status == "AWAITING_REVIEW"

    row = {
        "schema_version": "speaker-checkpoint-review-decision/v1",
        "case_id": "case-1",
        "model_id": "model-00",
        "checkpoint_step": 1000,
        "wav_sha256": module.sha256_file(wav),
        "reviewer": "user",
        "reviewed_at": "2026-08-02T00:00:00+00:00",
        "decision": "VOICE",
    }
    decisions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert module.verify_reviews(evaluations, decisions).status == "AWAITING_REVIEW"

    nonselected_row = {
        **row,
        "case_id": "case-2",
        "checkpoint_step": 1500,
        "wav_sha256": module.sha256_file(nonselected_wav),
        "decision": "TONE",
    }
    decisions.write_text(
        json.dumps(row) + "\n" + json.dumps(nonselected_row) + "\n", encoding="utf-8"
    )
    reviewed = module.verify_reviews(evaluations, decisions)
    assert reviewed.status == "PASS"
    assert reviewed.grouped_decisions["model-00"]["1500"] == {"TONE": 1}

    row["decision"] = "TONE"
    decisions.write_text(
        json.dumps(row) + "\n" + json.dumps(nonselected_row) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="selected checkpoint"):
        module.verify_reviews(evaluations, decisions)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "duplicate"),
        ("stale_hash", "stale"),
        ("naive_time", "timezone-aware"),
        ("invalid_enum", "enum"),
        ("changed_asset", "copied asset"),
    ],
)
def test_review_gate_rejects_invalid_decisions_and_assets(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    evaluation_fixture = _write_evaluation_fixture(tmp_path, module, training)
    evaluations = module.verify_evaluations(
        evaluation_fixture["config"], evaluation_fixture["status"], training
    )
    decisions = Path(evaluation_fixture["decisions"])
    _write_all_voice_decisions(decisions, evaluations)
    rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "stale_hash":
        rows[0]["wav_sha256"] = "0" * 64
    elif mutation == "naive_time":
        rows[0]["reviewed_at"] = "2026-08-02T00:00:00"
    elif mutation == "invalid_enum":
        rows[0]["decision"] = "MAYBE"
    else:
        asset = evaluations.models[0].evaluation_dir / "review_packet" / "audio" / "Anabel.wav"
        asset.write_bytes(b"changed")
    decisions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        module.verify_reviews(evaluations, decisions)


def test_staging_gate_and_full_cli_are_read_only(tmp_path: Path) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    evaluation_fixture = _write_evaluation_fixture(tmp_path, module, training)
    evaluations = module.verify_evaluations(
        evaluation_fixture["config"], evaluation_fixture["status"], training
    )
    decisions = Path(evaluation_fixture["decisions"])
    staging = _write_staging_fixture(tmp_path, module, evaluations)
    verified = module.verify_staging(evaluations, staging)
    assert verified.model_count == 12
    awaiting_output = training_fixture["root"] / "awaiting-completion.json"
    assert (
        module.main(
            _completion_argv(
                training_fixture,
                evaluation_fixture,
                staging=staging,
                output=awaiting_output,
            ),
            runtime_probe=lambda: module.RuntimeSnapshot.idle(used_mib=973.0),
            now=lambda: "2026-08-02T11:00:00+00:00",
        )
        == 1
    )
    awaiting = json.loads(awaiting_output.read_text(encoding="utf-8"))
    assert awaiting["status"] == "AWAITING_REVIEW"
    assert awaiting["reviews"]["unresolved_ids"]
    assert awaiting["checks"]["reviews"]["reasons"] == ["review decisions are unresolved"]

    _write_all_voice_decisions(decisions, evaluations)
    tracked = [
        Path(training_fixture["jobs"]),
        Path(training_fixture["status"]),
        Path(training_fixture["launch"]),
        Path(evaluation_fixture["config"]),
        Path(evaluation_fixture["status"]),
        decisions,
        staging,
    ]
    before = {path: (module.sha256_file(path), path.stat().st_mtime_ns) for path in tracked}
    output = training_fixture["root"] / "completion.json"
    result = module.main(
        _completion_argv(
            training_fixture,
            evaluation_fixture,
            staging=staging,
            output=output,
        ),
        runtime_probe=lambda: module.RuntimeSnapshot.idle(used_mib=973.0),
        now=lambda: "2026-08-02T12:00:00+00:00",
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "600m-speaker-retraining-completion-verification/v1"
    assert report["status"] == "PASS"
    assert set(report["checks"]) == {"training", "evaluations", "reviews", "staging"}
    assert all(check["passed"] for check in report["checks"].values())
    assert set(report["evaluations"]) >= {
        "runtime_snapshot_manifest",
        "runtime_snapshot_files",
    }
    training_audit = report["training"]["models"][0]
    assert set(training_audit) >= {"config", "clean_manifest", "log", "checkpoints"}
    assert len(training_audit["checkpoints"]) == 13
    evaluation_audit = report["evaluations"]["models"][0]
    assert set(evaluation_audit) >= {
        "manifest",
        "evaluation_verification",
        "evaluation_results",
        "review_candidates",
        "review_packet_manifest",
        "review_packet_assets",
        "selected",
    }
    assert report["reviews"]["decisions"]["sha256"] == module.sha256_file(decisions)
    assert set(report["staging"]) >= {
        "report",
        "active_voice_bank_baseline",
        "active_voice_bank_current",
        "selections",
    }
    assert report["staging"]["deployment_performed"] is False
    assert before == {path: (module.sha256_file(path), path.stat().st_mtime_ns) for path in tracked}

    proposed_root = Path(json.loads(staging.read_text(encoding="utf-8"))["proposed_staging_root"])
    unsafe_output = proposed_root / "completion.json"
    with pytest.raises(ValueError, match="output parent"):
        module.main(
            _completion_argv(
                training_fixture,
                evaluation_fixture,
                staging=staging,
                output=unsafe_output,
            ),
            runtime_probe=lambda: module.RuntimeSnapshot.idle(used_mib=973.0),
        )
    assert not proposed_root.exists()


def test_cli_rejects_output_outside_launch_evidence_directory_before_validation(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    output = training["root"] / "training" / "Anabel" / "completion.json"
    runtime_called = False

    def unexpected_probe() -> object:
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError

    with pytest.raises(ValueError, match="output parent"):
        module.main(
            [
                "--phase",
                "training",
                "--training-jobs",
                str(training["jobs"]),
                "--training-status",
                str(training["status"]),
                "--training-launch-evidence",
                str(training["launch"]),
                "--output",
                str(output),
            ],
            runtime_probe=unexpected_probe,
        )

    assert runtime_called is False
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["output_alias", "dangling_proposed_root"])
def test_cli_preflights_raw_proposed_staging_root_before_runtime_or_training(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    evaluation_fixture = _write_evaluation_fixture(tmp_path, module, training)
    evaluations = module.verify_evaluations(
        evaluation_fixture["config"], evaluation_fixture["status"], training
    )
    _write_all_voice_decisions(evaluation_fixture["decisions"], evaluations)
    staging = _write_staging_fixture(tmp_path, module, evaluations)
    output = training_fixture["root"] / "unsafe-completion.json"
    staging_payload = json.loads(staging.read_text(encoding="utf-8"))
    if mutation == "output_alias":
        staging_payload["proposed_staging_root"] = str(output)
    else:
        proposed = training_fixture["root"] / "dangling-proposed"
        proposed.symlink_to(training_fixture["root"] / "missing-proposed", target_is_directory=True)
        staging_payload["proposed_staging_root"] = str(proposed)
    staging.write_text(json.dumps(staging_payload), encoding="utf-8")
    launch = json.loads(training_fixture["launch"].read_text(encoding="utf-8"))
    launch["queue_exit_code"] = 9
    training_fixture["launch"].write_text(json.dumps(launch), encoding="utf-8")
    runtime_called = False

    def unexpected_runtime() -> object:
        nonlocal runtime_called
        runtime_called = True
        return module.RuntimeSnapshot.idle(used_mib=973.0)

    with pytest.raises(ValueError, match=r"proposed staging root|overlaps protected path"):
        module.main(
            _completion_argv(
                training_fixture,
                evaluation_fixture,
                staging=staging,
                output=output,
            ),
            runtime_probe=unexpected_runtime,
        )

    assert runtime_called is False
    assert not os.path.lexists(output)
    assert not os.path.lexists(output.with_suffix(output.suffix + ".tmp"))


def test_cli_rechecks_runtime_created_dangling_proposed_staging_root(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    evaluation_fixture = _write_evaluation_fixture(tmp_path, module, training)
    evaluations = module.verify_evaluations(
        evaluation_fixture["config"], evaluation_fixture["status"], training
    )
    _write_all_voice_decisions(evaluation_fixture["decisions"], evaluations)
    staging = _write_staging_fixture(tmp_path, module, evaluations)
    output = training_fixture["root"] / "runtime-dangling-completion.json"
    proposed = Path(json.loads(staging.read_text(encoding="utf-8"))["proposed_staging_root"])

    def create_dangling_proposed_root() -> object:
        proposed.symlink_to(tmp_path / "missing-proposed", target_is_directory=True)
        return module.RuntimeSnapshot.idle(used_mib=973.0)

    with pytest.raises(ValueError, match="proposed staging root"):
        module.main(
            _completion_argv(
                training_fixture,
                evaluation_fixture,
                staging=staging,
                output=output,
            ),
            runtime_probe=create_dangling_proposed_root,
        )

    assert proposed.is_symlink()
    assert not os.path.lexists(output)
    assert not os.path.lexists(output.with_suffix(output.suffix + ".tmp"))


def test_cli_rechecks_staging_report_before_publishing_early_training_failure(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    evaluation_fixture = _write_evaluation_fixture(tmp_path, module, training)
    evaluations = module.verify_evaluations(
        evaluation_fixture["config"], evaluation_fixture["status"], training
    )
    _write_all_voice_decisions(evaluation_fixture["decisions"], evaluations)
    staging = _write_staging_fixture(tmp_path, module, evaluations)
    output = training_fixture["root"] / "mutated-staging-completion.json"
    launch = json.loads(training_fixture["launch"].read_text(encoding="utf-8"))
    launch["queue_exit_code"] = 9
    training_fixture["launch"].write_text(json.dumps(launch), encoding="utf-8")

    def alias_proposed_root_to_output() -> object:
        staging_payload = json.loads(staging.read_text(encoding="utf-8"))
        staging_payload["proposed_staging_root"] = str(output)
        staging.write_text(json.dumps(staging_payload), encoding="utf-8")
        return module.RuntimeSnapshot.idle(used_mib=973.0)

    with pytest.raises(ValueError, match="staging report changed after preflight"):
        module.main(
            _completion_argv(
                training_fixture,
                evaluation_fixture,
                staging=staging,
                output=output,
            ),
            runtime_probe=alias_proposed_root_to_output,
        )

    assert not os.path.lexists(output)
    assert not os.path.lexists(output.with_suffix(output.suffix + ".tmp"))


def test_cli_rejects_dangling_output_symlink_before_runtime(tmp_path: Path) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    output = training["root"] / "completion.json"
    output.symlink_to(training["root"] / "missing-completion.json")
    runtime_called = False

    def unexpected_runtime() -> object:
        nonlocal runtime_called
        runtime_called = True
        return module.RuntimeSnapshot.idle(used_mib=973.0)

    with pytest.raises(FileExistsError, match="overwrite completion report"):
        module.main(
            [
                "--phase",
                "training",
                "--training-jobs",
                str(training["jobs"]),
                "--training-status",
                str(training["status"]),
                "--training-launch-evidence",
                str(training["launch"]),
                "--output",
                str(output),
            ],
            runtime_probe=unexpected_runtime,
        )

    assert runtime_called is False
    assert output.is_symlink()


def test_cli_rejects_training_jobs_beneath_symlinked_ancestor(tmp_path: Path) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    alias_root = tmp_path / "queue-alias"
    alias_root.symlink_to(training["root"], target_is_directory=True)
    output = training["root"] / "completion.json"
    runtime_called = False

    def unexpected_runtime() -> object:
        nonlocal runtime_called
        runtime_called = True
        return module.RuntimeSnapshot.idle(used_mib=973.0)

    with pytest.raises(ValueError, match=r"training jobs.*symlink|alias|reparse"):
        module.main(
            [
                "--phase",
                "training",
                "--training-jobs",
                str(alias_root / training["jobs"].name),
                "--training-status",
                str(training["status"]),
                "--training-launch-evidence",
                str(training["launch"]),
                "--output",
                str(output),
            ],
            runtime_probe=unexpected_runtime,
        )

    assert runtime_called is False
    assert not output.exists()


def test_cli_rejects_output_beneath_symlinked_ancestor_before_runtime(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    alias_root = tmp_path / "queue-alias"
    alias_root.symlink_to(training["root"], target_is_directory=True)
    aliased_output = alias_root / "completion.json"
    runtime_called = False

    def unexpected_runtime() -> object:
        nonlocal runtime_called
        runtime_called = True
        return module.RuntimeSnapshot.idle(used_mib=973.0)

    with pytest.raises(ValueError, match=r"symlink|alias|reparse"):
        module.main(
            [
                "--phase",
                "training",
                "--training-jobs",
                str(training["jobs"]),
                "--training-status",
                str(training["status"]),
                "--training-launch-evidence",
                str(training["launch"]),
                "--output",
                str(aliased_output),
            ],
            runtime_probe=unexpected_runtime,
        )

    assert runtime_called is False
    assert not (training["root"] / "completion.json").exists()


def test_training_rejects_base_checkpoint_beneath_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    alias_root = tmp_path / "queue-alias"
    alias_root.symlink_to(training["root"], target_is_directory=True)
    jobs = json.loads(training["jobs"].read_text(encoding="utf-8"))
    jobs["base_checkpoint_path"] = str(alias_root / "base.safetensors")
    training["jobs"].write_text(json.dumps(jobs), encoding="utf-8")
    launch = json.loads(training["launch"].read_text(encoding="utf-8"))
    launch["training_jobs_sha256"] = module.sha256_file(training["jobs"])
    launch["checkpoint_path"] = jobs["base_checkpoint_path"]
    training["launch"].write_text(json.dumps(launch), encoding="utf-8")

    with pytest.raises(ValueError, match=r"base_checkpoint_path.*symlink|alias|reparse"):
        module.verify_training(
            training["jobs"],
            training["status"],
            training["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
        )


def test_training_rejects_predecessor_config_beneath_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    alias_root = tmp_path / "queue-alias"
    alias_root.symlink_to(training["root"], target_is_directory=True)
    jobs = json.loads(training["jobs"].read_text(encoding="utf-8"))
    jobs["jobs"][0]["config"] = str(alias_root / "configs" / "Anabel.json")
    training["jobs"].write_text(json.dumps(jobs), encoding="utf-8")
    launch = json.loads(training["launch"].read_text(encoding="utf-8"))
    launch["training_jobs_sha256"] = module.sha256_file(training["jobs"])
    training["launch"].write_text(json.dumps(launch), encoding="utf-8")

    with pytest.raises(ValueError, match=r"training job config.*symlink|alias|reparse"):
        module.verify_training(
            training["jobs"],
            training["status"],
            training["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
        )


def test_path_validation_rejects_windows_reparse_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    ancestor = tmp_path / "junction"
    ancestor.mkdir()
    target = ancestor / "input.json"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_is_reparse_alias",
        lambda path: path == ancestor,
        raising=False,
    )

    with pytest.raises(ValueError, match="reparse"):
        module._require_regular_file(target, source="reparse fixture")


def test_windows_reparse_detector_reads_file_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    reparse_flag = 0x400

    class Metadata:
        st_file_attributes = reparse_flag

    monkeypatch.setattr(module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", reparse_flag, raising=False)
    monkeypatch.setattr(Path, "lstat", lambda _path: Metadata())

    assert module._is_reparse_alias(Path("junction")) is True


def test_training_rejects_checkpoint_symlink_to_outside_output(tmp_path: Path) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    checkpoint = (
        training["root"] / "training" / "Anabel" / ("checkpoint_0000250.speaker.safetensors")
    )
    outside = training["root"] / "outside-checkpoint.speaker.safetensors"
    outside.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(outside)

    with pytest.raises(ValueError, match=r"checkpoint.*symlink|regular file|outside"):
        module.verify_training(
            training["jobs"],
            training["status"],
            training["launch"],
            module.RuntimeSnapshot.idle(used_mib=973.0),
            256.0,
        )


def test_cli_converts_checkpoint_directory_to_fail_report(tmp_path: Path) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    checkpoint = (
        training["root"] / "training" / "Anabel" / ("checkpoint_0000250.speaker.safetensors")
    )
    checkpoint.unlink()
    checkpoint.mkdir()
    output = training["root"] / "completion.json"

    result = module.main(
        [
            "--phase",
            "training",
            "--training-jobs",
            str(training["jobs"]),
            "--training-status",
            str(training["status"]),
            "--training-launch-evidence",
            str(training["launch"]),
            "--output",
            str(output),
        ],
        runtime_probe=lambda: module.RuntimeSnapshot.idle(used_mib=973.0),
    )

    assert result == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert "checkpoint" in report["checks"]["training"]["reasons"][0]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is not supported on this platform")
def test_cli_converts_checkpoint_fifo_to_fail_report(tmp_path: Path) -> None:
    module = _load_script()
    training = _write_training_fixture(tmp_path, module)
    checkpoint = training["root"] / "training" / "Anabel" / "checkpoint_0000250.speaker.safetensors"
    checkpoint.unlink()
    os.mkfifo(checkpoint)
    output = training["root"] / "completion.json"

    result = module.main(
        [
            "--phase",
            "training",
            "--training-jobs",
            str(training["jobs"]),
            "--training-status",
            str(training["status"]),
            "--training-launch-evidence",
            str(training["launch"]),
            "--output",
            str(output),
        ],
        runtime_probe=lambda: module.RuntimeSnapshot.idle(used_mib=973.0),
    )

    assert result == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert "regular non-symlink file" in report["checks"]["training"]["reasons"][0]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("deployment", "deployment_performed"),
        ("selection", "selection identity"),
        ("voice_bank", "voice bank file changed"),
        ("staging_exists", "staging root exists"),
        ("dangling_staging", "staging root exists"),
    ],
)
def test_staging_gate_fails_closed(tmp_path: Path, mutation: str, match: str) -> None:
    module = _load_script()
    training_fixture = _write_training_fixture(tmp_path, module)
    training = module.verify_training(
        training_fixture["jobs"],
        training_fixture["status"],
        training_fixture["launch"],
        module.RuntimeSnapshot.idle(used_mib=973.0),
        256.0,
    )
    evaluation_fixture = _write_evaluation_fixture(tmp_path, module, training)
    evaluations = module.verify_evaluations(
        evaluation_fixture["config"], evaluation_fixture["status"], training
    )
    staging = _write_staging_fixture(tmp_path, module, evaluations)
    payload = json.loads(staging.read_text(encoding="utf-8"))
    if mutation == "deployment":
        payload["deployment_performed"] = True
    elif mutation == "selection":
        payload["selections"][0]["checkpoint_step"] = 1500
    elif mutation == "voice_bank":
        speaker = Path(payload["active_voice_bank_current"]["speakers"][0]["path"])
        speaker.write_bytes(b"changed")
    elif mutation == "staging_exists":
        Path(payload["proposed_staging_root"]).mkdir()
    else:
        Path(payload["proposed_staging_root"]).symlink_to(
            tmp_path / "missing-proposed",
            target_is_directory=True,
        )
    staging.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        module.verify_staging(evaluations, staging)


def test_report_writer_is_create_only_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    output = tmp_path / "report.json"
    module.write_report_create_only(output, {"status": "PASS"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "PASS"}
    with pytest.raises(FileExistsError):
        module.write_report_create_only(output, {"status": "FAIL"})
    assert not output.with_suffix(".json.tmp").exists()

    windows_output = tmp_path / "windows-report.json"
    monkeypatch.setattr(
        module,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(PermissionError("unsupported")),
    )
    module.write_report_create_only(windows_output, {"status": "PASS"})
    assert json.loads(windows_output.read_text(encoding="utf-8")) == {"status": "PASS"}


def test_report_writer_preserves_concurrent_temp_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    blocked = tmp_path / "blocked.json"
    blocked_temp = blocked.with_suffix(".json.tmp")
    blocked_temp.write_text("concurrent-temp", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.write_report_create_only(blocked, {"status": "PASS"})
    assert blocked_temp.read_text(encoding="utf-8") == "concurrent-temp"
    assert not blocked.exists()

    raced = tmp_path / "raced.json"

    def competing_link(_source: Path, destination: Path) -> None:
        destination.write_text("concurrent-output", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(module.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        module.write_report_create_only(raced, {"status": "PASS"})
    assert raced.read_text(encoding="utf-8") == "concurrent-output"
    assert not raced.with_suffix(".json.tmp").exists()
