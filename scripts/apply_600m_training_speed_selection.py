from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BENCHMARK_SCHEMA = "speaker-training-speed-benchmark/v1"
TRAINING_JOBS_SCHEMA = 1
STATUS_SCHEMA = "speaker-training-speed-selection/v1"
EXPECTED_JOB_COUNT = 12
EXPECTED_COMPLETED_COUNT = 2
EXPECTED_CHANGED_COUNT = EXPECTED_JOB_COUNT - EXPECTED_COMPLETED_COUNT
EXPECTED_EFFECTIVE_GLOBAL_BATCH_SIZE = 16
EXPECTED_MEASURED_OPTIMIZER_STEPS = 50
MIN_ELIGIBLE_TELEMETRY_SAMPLES = 10
EXPECTED_CANDIDATE_IDS = frozenset({"A", "B", "C", "D"})
EXPECTED_CANDIDATE_SPECS = {
    "A": (1, 16, True),
    "B": (2, 8, True),
    "C": (4, 4, True),
    "D": (2, 8, False),
}
EXPECTED_COMPLETED_MODEL_IDS = frozenset({"oop77_anabel_maidgarden_sp_451488a7c1", "kasumi"})
MAX_PEAK_VRAM_MIB = 10_500.0
SPEED_CONFIG_NAME = "training-config-speed-v1.json"
SHA256_LENGTH = 64
GIT_COMMIT_LENGTH = 40
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
SCIENTIFIC_JSON_NUMBER_PATTERN = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d+",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    steady_optimizer_steps_per_second: float | None
    peak_vram_mib: float | None
    eligible: bool
    raw: dict[str, object]


@dataclass(frozen=True, slots=True)
class Job:
    model_id: str
    clean_manifest: Path
    config: Path
    output_dir: Path
    command: tuple[str, ...]
    config_argument_index: int
    raw: dict[str, object]
    config_document: dict[str, object]


@dataclass(frozen=True, slots=True)
class Change:
    job: Job
    target_config: Path
    config_document: dict[str, object]
    output_job: dict[str, object]
    changed_fields: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class BenchmarkProvenance:
    input_paths: tuple[Path, ...]
    training_status: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_speed_selection(  # noqa: PLR0914 - transaction authorities stay explicit.
    *,
    benchmark_summary: Path,
    training_jobs: Path,
    completed_model_ids: Sequence[str],
    output_jobs: Path,
    status_output: Path,
) -> dict[str, object]:
    benchmark_path = benchmark_summary.resolve()
    jobs_path = training_jobs.resolve()
    output_jobs_path = output_jobs.resolve()
    status_path = status_output.resolve()
    jobs_document, jobs = _load_training_jobs(jobs_path)
    completed = _validate_completed_models(completed_model_ids, jobs=jobs)
    base_checkpoint = _jobs_base_checkpoint(jobs_document, base=jobs_path.parent)
    candidate, benchmark_provenance = _load_benchmark(
        benchmark_path,
        jobs=jobs,
        completed_model_ids=completed,
        base_checkpoint=base_checkpoint,
        expected_training_jobs=jobs_path,
        expected_upstream_commit=_required_hex(
            jobs_document,
            "upstream_commit",
            length=GIT_COMMIT_LENGTH,
            source="training jobs",
        ),
    )
    input_paths = {
        benchmark_path,
        jobs_path,
        base_checkpoint,
        Path(__file__).resolve(),
        *benchmark_provenance.input_paths,
        *(job.config for job in jobs),
        *(job.clean_manifest for job in jobs),
    }
    input_sha256 = {path: sha256_file(path) for path in input_paths}
    changes = _build_changes(
        jobs,
        completed_model_ids=completed,
        candidate=candidate,
        jobs_base=jobs_path.parent,
    )
    if len(changes) != EXPECTED_CHANGED_COUNT:
        message = f"speed selection must change exactly {EXPECTED_CHANGED_COUNT} jobs"
        raise ValueError(message)

    output_document = copy.deepcopy(jobs_document)
    output_document["jobs"] = [
        (
            change.output_job
            if (change := changes.get(job.model_id)) is not None
            else copy.deepcopy(job.raw)
        )
        for job in jobs
    ]
    targets = [change.target_config for change in changes.values()]
    targets.extend((output_jobs_path, status_path))
    _validate_publish_targets(targets)
    return _stage_and_publish(
        benchmark_path=benchmark_path,
        jobs_path=jobs_path,
        training_status_path=benchmark_provenance.training_status,
        base_checkpoint=base_checkpoint,
        output_jobs_path=output_jobs_path,
        status_path=status_path,
        jobs=jobs,
        completed_model_ids=completed,
        candidate=candidate,
        changes=changes,
        output_document=output_document,
        publish_targets=targets,
        input_sha256=input_sha256,
    )


def _load_benchmark(
    path: Path,
    *,
    jobs: Sequence[Job],
    completed_model_ids: Sequence[str],
    base_checkpoint: Path,
    expected_training_jobs: Path,
    expected_upstream_commit: str,
) -> tuple[Candidate, BenchmarkProvenance]:
    document = _read_json(path)
    if document.get("schema_version") != BENCHMARK_SCHEMA:
        message = f"benchmark requires schema_version {BENCHMARK_SCHEMA}"
        raise ValueError(message)
    if document.get("status") != "PASS":
        message = "benchmark status PASS is required"
        raise ValueError(message)
    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        message = "benchmark candidates must be a nonempty list"
        raise ValueError(message)
    candidates = [_validate_candidate(row, source="benchmark candidate") for row in raw_candidates]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        message = "benchmark candidates contain duplicate ids"
        raise ValueError(message)
    if set(candidate_ids) != EXPECTED_CANDIDATE_IDS:
        message = "benchmark candidates must contain exactly A, B, C, D"
        raise ValueError(message)
    candidate_specs = {
        candidate.candidate_id: (
            candidate.batch_size,
            candidate.gradient_accumulation_steps,
            candidate.gradient_checkpointing,
        )
        for candidate in candidates
    }
    if candidate_specs != EXPECTED_CANDIDATE_SPECS:
        message = "benchmark candidate IDs must each match their fixed spec"
        raise ValueError(message)
    recommended = _validate_candidate(
        document.get("recommended_candidate"),
        source="recommended candidate",
        require_eligible=True,
    )
    matched = [
        candidate for candidate in candidates if candidate.candidate_id == recommended.candidate_id
    ]
    if len(matched) != 1 or matched[0].raw != recommended.raw:
        message = "recommended candidate must exactly match one benchmark candidate"
        raise ValueError(message)
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        message = "benchmark must contain at least one eligible candidate"
        raise ValueError(message)
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -_present_candidate_metric(
                candidate.steady_optimizer_steps_per_second,
                field="steady_optimizer_steps_per_second",
            ),
            _present_candidate_metric(candidate.peak_vram_mib, field="peak_vram_mib"),
            candidate.candidate_id,
        ),
    )
    if ranked[0].raw != recommended.raw:
        message = "recommended candidate must be the fastest eligible candidate"
        raise ValueError(message)
    provenance_inputs = _validate_benchmark_provenance(
        document.get("provenance"),
        jobs=jobs,
        completed_model_ids=completed_model_ids,
        base_checkpoint=base_checkpoint,
        expected_training_jobs=expected_training_jobs,
        expected_upstream_commit=expected_upstream_commit,
    )
    return recommended, provenance_inputs


def _validate_candidate(
    value: object,
    *,
    source: str,
    require_eligible: bool = False,
) -> Candidate:
    if not isinstance(value, dict):
        message = f"{source} must be an object"
        raise TypeError(message)
    candidate_id = _required_string(value, "id", source=source)
    batch_size = _required_positive_int(value, "batch_size", source=source)
    accumulation = _required_positive_int(
        value,
        "gradient_accumulation_steps",
        source=source,
    )
    gradient_checkpointing = value.get("gradient_checkpointing")
    if not isinstance(gradient_checkpointing, bool):
        message = f"{source} gradient_checkpointing must be boolean"
        raise TypeError(message)
    effective_batch_size = _required_positive_int(
        value,
        "effective_global_batch_size",
        source=source,
    )
    if (
        effective_batch_size != EXPECTED_EFFECTIVE_GLOBAL_BATCH_SIZE
        or batch_size * accumulation != effective_batch_size
    ):
        message = (
            f"{source} effective_global_batch_size must equal "
            f"{EXPECTED_EFFECTIVE_GLOBAL_BATCH_SIZE} and batch_size * "
            "gradient_accumulation_steps"
        )
        raise ValueError(message)
    overrides = value.get("overrides")
    expected_overrides = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if overrides != expected_overrides:
        message = f"{source} overrides must exactly match the speed fields"
        raise ValueError(message)
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        message = f"{source} metrics must be an object"
        raise TypeError(message)
    optimizer_speed, peak_vram, eligible = _validate_candidate_metrics(
        metrics,
        source=source,
        require_eligible=require_eligible,
    )
    return Candidate(
        candidate_id=candidate_id,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=gradient_checkpointing,
        steady_optimizer_steps_per_second=optimizer_speed,
        peak_vram_mib=peak_vram,
        eligible=eligible,
        raw=copy.deepcopy(value),
    )


def _validate_candidate_metrics(
    metrics: Mapping[str, object],
    *,
    source: str,
    require_eligible: bool,
) -> tuple[float | None, float | None, bool]:
    measured_steps = _required_nonnegative_int(
        metrics,
        "measured_optimizer_steps",
        source=f"{source} metrics",
    )
    optimizer_speed = _optional_finite_number(
        metrics,
        "steady_optimizer_steps_per_second",
        source=f"{source} metrics",
    )
    sample_speed = _optional_finite_number(
        metrics,
        "steady_samples_per_second",
        source=f"{source} metrics",
    )
    peak_vram = _optional_finite_number(
        metrics,
        "peak_vram_mib",
        source=f"{source} metrics",
    )
    for field in ("loss_finite", "oom", "eligible"):
        if not isinstance(metrics.get(field), bool):
            message = f"{source} metrics {field} must be boolean"
            raise TypeError(message)
    eligible = bool(metrics["eligible"])
    exit_code = metrics.get("exit_code")
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        message = f"{source} metrics exit_code must be an integer or null"
        raise TypeError(message)
    raw_reasons = metrics.get("ineligible_reasons")
    if not isinstance(raw_reasons, list) or not all(
        isinstance(reason, str) and reason for reason in raw_reasons
    ):
        message = f"{source} metrics ineligible_reasons must be a string list"
        raise ValueError(message)
    if eligible == bool(raw_reasons):
        message = f"{source} metrics eligible and ineligible_reasons are inconsistent"
        raise ValueError(message)
    gpu_telemetry = metrics.get("gpu_utilization_percent")
    power_telemetry = metrics.get("power_watts")
    if gpu_telemetry is not None:
        _validate_metric_summary(
            gpu_telemetry,
            source=f"{source} GPU utilization",
            lower=0.0,
            upper=100.0,
            allow_empty=not eligible,
            minimum_sample_count=MIN_ELIGIBLE_TELEMETRY_SAMPLES if eligible else 0,
        )
    if power_telemetry is not None:
        _validate_metric_summary(
            power_telemetry,
            source=f"{source} power",
            lower=0.0,
            upper=None,
            allow_empty=not eligible,
            minimum_sample_count=MIN_ELIGIBLE_TELEMETRY_SAMPLES if eligible else 0,
        )
    if require_eligible and not eligible:
        message = f"{source} metrics eligible must be true"
        raise ValueError(message)
    if eligible:
        _validate_eligible_metrics(
            source=source,
            measured_steps=measured_steps,
            optimizer_speed=optimizer_speed,
            sample_speed=sample_speed,
            peak_vram=peak_vram,
            gpu_telemetry=gpu_telemetry,
            power_telemetry=power_telemetry,
            loss_finite=metrics["loss_finite"],
            oom=metrics["oom"],
            exit_code=exit_code,
        )
    return optimizer_speed, peak_vram, eligible


def _validate_eligible_metrics(  # noqa: PLR0913 - measured gates stay explicit.
    *,
    source: str,
    measured_steps: int,
    optimizer_speed: float | None,
    sample_speed: float | None,
    peak_vram: float | None,
    gpu_telemetry: object,
    power_telemetry: object,
    loss_finite: object,
    oom: object,
    exit_code: int | None,
) -> None:
    if measured_steps != EXPECTED_MEASURED_OPTIMIZER_STEPS:
        message = (
            f"{source} measured_optimizer_steps must equal {EXPECTED_MEASURED_OPTIMIZER_STEPS}"
        )
        raise ValueError(message)
    if optimizer_speed is None or optimizer_speed <= 0.0:
        message = f"{source} metrics steady_optimizer_steps_per_second must be positive"
        raise ValueError(message)
    if sample_speed is None or sample_speed <= 0.0:
        message = f"{source} metrics steady_samples_per_second must be positive"
        raise ValueError(message)
    expected_sample_speed = optimizer_speed * EXPECTED_EFFECTIVE_GLOBAL_BATCH_SIZE
    if not math.isclose(sample_speed, expected_sample_speed, rel_tol=1e-6, abs_tol=1e-8):
        message = f"{source} metrics steady_samples_per_second is inconsistent"
        raise ValueError(message)
    if peak_vram is None or peak_vram <= 0.0 or peak_vram > MAX_PEAK_VRAM_MIB:
        message = f"{source} metrics peak_vram_mib must be within (0, {MAX_PEAK_VRAM_MIB}]"
        raise ValueError(message)
    if gpu_telemetry is None or power_telemetry is None:
        message = f"{source} eligible metrics require GPU utilization and power summaries"
        raise ValueError(message)
    if not all((loss_finite is True, oom is False, exit_code == 0)):
        message = f"{source} requires loss_finite true, oom false, and exit_code 0"
        raise ValueError(message)


def _validate_metric_summary(
    value: object,
    *,
    source: str,
    lower: float,
    upper: float | None,
    allow_empty: bool,
    minimum_sample_count: int,
) -> None:
    if not isinstance(value, dict):
        message = f"{source} summary must be an object"
        raise TypeError(message)
    sample_count = _required_nonnegative_int(value, "sample_count", source=source)
    if sample_count == 0:
        if allow_empty and all(
            value.get(field) is None for field in ("minimum", "mean", "maximum")
        ):
            return
        message = f"{source} empty summary is invalid"
        raise ValueError(message)
    if sample_count < minimum_sample_count:
        message = f"{source} requires at least {minimum_sample_count} samples"
        raise ValueError(message)
    minimum = _required_finite_number(value, "minimum", source=source)
    mean = _required_finite_number(value, "mean", source=source)
    maximum = _required_finite_number(value, "maximum", source=source)
    if not lower <= minimum <= mean <= maximum or (upper is not None and maximum > upper):
        message = f"{source} summary bounds are invalid"
        raise ValueError(message)


def _validate_benchmark_provenance(
    value: object,
    *,
    jobs: Sequence[Job],
    completed_model_ids: Sequence[str],
    base_checkpoint: Path,
    expected_training_jobs: Path,
    expected_upstream_commit: str,
) -> BenchmarkProvenance:
    if not isinstance(value, dict):
        message = "benchmark provenance must be an object"
        raise TypeError(message)
    base_config = _validate_file_binding(
        value.get("base_config"),
        source="benchmark base_config",
    )
    manifest = _validate_file_binding(value.get("manifest"), source="benchmark manifest")
    bound_checkpoint = _validate_file_binding(
        value.get("base_checkpoint"),
        source="benchmark base_checkpoint",
    )
    benchmark_script = _validate_file_binding(value.get("script"), source="benchmark script")
    bound_training_jobs = _validate_file_binding(
        value.get("training_jobs"),
        source="benchmark training_jobs",
    )
    if bound_training_jobs != expected_training_jobs:
        message = "benchmark training_jobs does not match --training-jobs"
        raise ValueError(message)
    training_status = _validate_file_binding(
        value.get("training_status"),
        source="benchmark training_status",
    )
    benchmark_jobs = [
        job for job in jobs if job.config == base_config and job.clean_manifest == manifest
    ]
    if len(benchmark_jobs) != 1:
        message = "benchmark base_config and manifest must bind the same training job"
        raise ValueError(message)
    benchmark_job = benchmark_jobs[0]
    if benchmark_job.model_id in set(completed_model_ids):
        message = "benchmark config and manifest must belong to a pending model"
        raise ValueError(message)
    if benchmark_job.output_dir.exists():
        message = "benchmark pending model output directory must not exist"
        raise FileExistsError(message)
    if bound_checkpoint != base_checkpoint:
        message = "benchmark base_checkpoint does not match training jobs"
        raise ValueError(message)
    upstream = value.get("upstream")
    if not isinstance(upstream, dict):
        message = "benchmark upstream provenance must be an object"
        raise TypeError(message)
    root = Path(_required_string(upstream, "root", source="benchmark upstream")).resolve()
    if not root.is_dir():
        message = f"benchmark upstream root does not exist: {root}"
        raise ValueError(message)
    commit = _required_hex(
        upstream,
        "commit",
        length=GIT_COMMIT_LENGTH,
        source="benchmark upstream",
    )
    if commit != expected_upstream_commit:
        message = "benchmark upstream commit does not match training jobs"
        raise ValueError(message)
    trainer_python = _validate_file_binding(
        upstream.get("trainer_python"),
        source="benchmark upstream trainer_python",
    )
    trainer_script = _validate_file_binding(
        upstream.get("trainer_script"),
        source="benchmark upstream trainer_script",
    )
    if not trainer_python.is_relative_to(root) or not trainer_script.is_relative_to(root):
        message = "benchmark trainer bindings must be inside upstream root"
        raise ValueError(message)
    return BenchmarkProvenance(
        input_paths=(
            base_config,
            manifest,
            bound_checkpoint,
            benchmark_script,
            bound_training_jobs,
            training_status,
            trainer_python,
            trainer_script,
        ),
        training_status=training_status,
    )


def _validate_file_binding(value: object, *, source: str) -> Path:
    if not isinstance(value, dict):
        message = f"{source} provenance must be an object"
        raise TypeError(message)
    path = Path(_required_string(value, "path", source=source)).resolve()
    expected_sha256 = _required_hex(value, "sha256", length=SHA256_LENGTH, source=source)
    if not path.is_file():
        message = f"{source} path does not exist: {path}"
        raise ValueError(message)
    if sha256_file(path) != expected_sha256:
        message = f"{source} SHA-256 does not match: {path}"
        raise ValueError(message)
    return path


def _load_training_jobs(path: Path) -> tuple[dict[str, object], tuple[Job, ...]]:
    document = _read_json(path)
    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version != TRAINING_JOBS_SCHEMA:
        message = f"training jobs require schema_version {TRAINING_JOBS_SCHEMA}"
        raise ValueError(message)
    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_JOB_COUNT:
        message = f"training jobs must contain exactly {EXPECTED_JOB_COUNT} jobs"
        raise ValueError(message)
    jobs = tuple(_load_job(row, base=path.parent) for row in raw_jobs)
    model_ids = [job.model_id for job in jobs]
    if len(model_ids) != len(set(model_ids)):
        message = "training jobs contain duplicate model ids"
        raise ValueError(message)
    return document, jobs


def _jobs_base_checkpoint(document: Mapping[str, object], *, base: Path) -> Path:
    raw_path = _required_string(
        document,
        "base_checkpoint_path",
        source="training jobs",
    )
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else base / path).resolve()
    expected_sha256 = _required_hex(
        document,
        "base_checkpoint_sha256",
        length=SHA256_LENGTH,
        source="training jobs",
    )
    if not resolved.is_file():
        message = f"training jobs base checkpoint does not exist: {resolved}"
        raise ValueError(message)
    if sha256_file(resolved) != expected_sha256:
        message = "training jobs base checkpoint SHA-256 does not match"
        raise ValueError(message)
    return resolved


def _load_job(value: object, *, base: Path) -> Job:
    if not isinstance(value, dict):
        message = "training job entries must be objects"
        raise TypeError(message)
    model_id = _required_string(value, "model_id", source="training job")
    if MODEL_ID_PATTERN.fullmatch(model_id) is None:
        message = f"training job has unsafe model_id: {model_id!r}"
        raise ValueError(message)
    clean_manifest = _resolved_job_path(value, "clean_manifest", base=base, model_id=model_id)
    config = _resolved_job_path(value, "config", base=base, model_id=model_id)
    output_dir = _resolved_job_path(value, "output_dir", base=base, model_id=model_id)
    if not clean_manifest.is_file():
        message = f"training job clean_manifest does not exist for {model_id}: {clean_manifest}"
        raise ValueError(message)
    if not config.is_file():
        message = f"training job config does not exist for {model_id}: {config}"
        raise ValueError(message)
    raw_command = value.get("command")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or not all(isinstance(part, str) and part for part in raw_command)
    ):
        message = f"training job command must be a nonempty string list for {model_id}"
        raise ValueError(message)
    command = tuple(raw_command)
    config_indexes = [index for index, part in enumerate(command) if part == "--config"]
    if len(config_indexes) != 1 or config_indexes[0] + 1 >= len(command):
        message = f"training job command requires exactly one --config value for {model_id}"
        raise ValueError(message)
    config_argument_index = config_indexes[0] + 1
    command_config = Path(command[config_argument_index])
    if not command_config.is_absolute():
        command_config = base / command_config
    if command_config.resolve() != config:
        message = f"training job --config does not bind config path for {model_id}"
        raise ValueError(message)
    config_document = _read_json(config)
    _validate_training_config(
        config_document,
        model_id=model_id,
        clean_manifest=clean_manifest,
        output_dir=output_dir,
        config_base=config.parent,
    )
    return Job(
        model_id=model_id,
        clean_manifest=clean_manifest,
        config=config,
        output_dir=output_dir,
        command=command,
        config_argument_index=config_argument_index,
        raw=copy.deepcopy(value),
        config_document=config_document,
    )


def _validate_training_config(
    value: Mapping[str, object],
    *,
    model_id: str,
    clean_manifest: Path,
    output_dir: Path,
    config_base: Path,
) -> None:
    model = value.get("model")
    train = value.get("train")
    if not isinstance(model, dict) or not model:
        message = f"training config model must be a nonempty object for {model_id}"
        raise ValueError(message)
    if not isinstance(train, dict):
        message = f"training config train must be an object for {model_id}"
        raise TypeError(message)
    expected = {
        "precision": "bf16",
        "allow_tf32": True,
        "learning_rate": 0.01,
        "lr_scheduler": "none",
        "compile_model": False,
        "max_latent_steps": 750,
        "seed": 0,
    }
    for field, expected_value in expected.items():
        if train.get(field) != expected_value or isinstance(train.get(field), bool) != isinstance(
            expected_value,
            bool,
        ):
            message = f"training config {field} invariant failed for {model_id}"
            raise ValueError(message)
    batch_size = _required_positive_int(train, "batch_size", source=f"training config {model_id}")
    accumulation = _required_positive_int(
        train,
        "gradient_accumulation_steps",
        source=f"training config {model_id}",
    )
    if batch_size * accumulation != EXPECTED_EFFECTIVE_GLOBAL_BATCH_SIZE:
        message = f"training config effective global batch must be 16 for {model_id}"
        raise ValueError(message)
    if not isinstance(train.get("gradient_checkpointing"), bool):
        message = f"training config gradient_checkpointing must be boolean for {model_id}"
        raise TypeError(message)
    manifest_path = _resolved_config_path(
        train.get("manifest_path"),
        base=config_base,
        source=f"training config manifest_path for {model_id}",
    )
    configured_output = _resolved_config_path(
        train.get("output_dir"),
        base=config_base,
        source=f"training config output_dir for {model_id}",
    )
    if manifest_path != clean_manifest:
        message = f"training config manifest_path does not match job for {model_id}"
        raise ValueError(message)
    if configured_output != output_dir:
        message = f"training config output_dir does not match job for {model_id}"
        raise ValueError(message)


def _validate_completed_models(
    values: Sequence[str],
    *,
    jobs: Sequence[Job],
) -> tuple[str, ...]:
    completed = tuple(values)
    if len(completed) != EXPECTED_COMPLETED_COUNT or set(completed) != EXPECTED_COMPLETED_MODEL_IDS:
        expected = ", ".join(sorted(EXPECTED_COMPLETED_MODEL_IDS))
        message = f"required completed model ids are exactly: {expected}"
        raise ValueError(message)
    ordered = tuple(job.model_id for job in jobs if job.model_id in EXPECTED_COMPLETED_MODEL_IDS)
    if len(ordered) != EXPECTED_COMPLETED_COUNT:
        message = "required completed model ids must exist in training jobs"
        raise ValueError(message)
    return ordered


def _build_changes(
    jobs: Sequence[Job],
    *,
    completed_model_ids: Sequence[str],
    candidate: Candidate,
    jobs_base: Path,
) -> dict[str, Change]:
    completed = set(completed_model_ids)
    changes: dict[str, Change] = {}
    for job in jobs:
        if job.model_id in completed:
            if not job.output_dir.is_dir():
                message = f"completed model output directory does not exist: {job.output_dir}"
                raise ValueError(message)
            continue
        if job.output_dir.exists():
            message = f"pending model output directory already exists: {job.output_dir}"
            raise FileExistsError(message)
        target_config = job.clean_manifest.parent / SPEED_CONFIG_NAME
        if job.config.parent != target_config.parent:
            message = (
                "source and speed configs must share the same directory to preserve "
                f"relative path semantics for {job.model_id}"
            )
            raise ValueError(message)
        new_document = copy.deepcopy(job.config_document)
        train = new_document.get("train")
        if not isinstance(train, dict):  # guarded by _validate_training_config
            raise TypeError
        speed_values = {
            "batch_size": candidate.batch_size,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
            "gradient_checkpointing": candidate.gradient_checkpointing,
        }
        changed_fields = {
            field: {"before": train[field], "after": value} for field, value in speed_values.items()
        }
        train.update(speed_values)
        _validate_training_config(
            new_document,
            model_id=job.model_id,
            clean_manifest=job.clean_manifest,
            output_dir=job.output_dir,
            config_base=job.config.parent,
        )
        if new_document.get("model") != job.config_document.get("model"):
            message = f"speed selection changed model config for {job.model_id}"
            raise ValueError(message)
        output_job = copy.deepcopy(job.raw)
        output_job["config"] = _serialize_job_path(target_config, base=jobs_base)
        command = list(job.command)
        command[job.config_argument_index] = str(target_config.resolve())
        if any(
            before != after
            for index, (before, after) in enumerate(zip(job.command, command, strict=True))
            if index != job.config_argument_index
        ):
            message = f"speed selection changed non-config command arguments for {job.model_id}"
            raise ValueError(message)
        output_job["command"] = command
        changes[job.model_id] = Change(
            job=job,
            target_config=target_config,
            config_document=new_document,
            output_job=output_job,
            changed_fields=changed_fields,
        )
    return changes


def _stage_and_publish(  # noqa: PLR0913 - transaction inputs remain explicit.
    *,
    benchmark_path: Path,
    jobs_path: Path,
    training_status_path: Path,
    base_checkpoint: Path,
    output_jobs_path: Path,
    status_path: Path,
    jobs: Sequence[Job],
    completed_model_ids: Sequence[str],
    candidate: Candidate,
    changes: Mapping[str, Change],
    output_document: Mapping[str, object],
    publish_targets: Sequence[Path],
    input_sha256: Mapping[Path, str],
) -> dict[str, object]:
    output_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=".speed-selection-stage-", dir=output_jobs_path.parent),
    )
    published: list[Path] = []
    try:
        staged_configs: dict[str, Path] = {}
        for index, change in enumerate(changes.values()):
            staged = stage_root / "configs" / f"{index:02d}.json"
            _write_training_config(staged, change.config_document)
            staged_configs[change.job.model_id] = staged
        staged_jobs = stage_root / "training-jobs.json"
        _write_json(staged_jobs, output_document)
        status = _build_status(
            benchmark_path=benchmark_path,
            jobs_path=jobs_path,
            training_status_path=training_status_path,
            base_checkpoint=base_checkpoint,
            output_jobs_path=output_jobs_path,
            jobs=jobs,
            completed_model_ids=completed_model_ids,
            candidate=candidate,
            changes=changes,
            staged_configs=staged_configs,
            staged_jobs=staged_jobs,
        )
        staged_status = stage_root / "status.json"
        _write_json(staged_status, status)
        _verify_inputs_unchanged(input_sha256)
        _validate_publish_targets(publish_targets)
        staged_targets = [
            (staged_configs[change.job.model_id], change.target_config)
            for change in changes.values()
        ]
        staged_targets.extend(((staged_jobs, output_jobs_path), (staged_status, status_path)))
        for staged, target in staged_targets:
            os.link(staged, target)
            published.append(target)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    else:
        return status
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _build_status(  # noqa: PLR0913 - report binds every transaction input.
    *,
    benchmark_path: Path,
    jobs_path: Path,
    training_status_path: Path,
    base_checkpoint: Path,
    output_jobs_path: Path,
    jobs: Sequence[Job],
    completed_model_ids: Sequence[str],
    candidate: Candidate,
    changes: Mapping[str, Change],
    staged_configs: Mapping[str, Path],
    staged_jobs: Path,
) -> dict[str, object]:
    return {
        "schema_version": STATUS_SCHEMA,
        "status": "PASS",
        "completed_model_ids": list(completed_model_ids),
        "changed_model_count": len(changes),
        "benchmark_binding": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "schema_version": BENCHMARK_SCHEMA,
            "recommended_candidate_id": candidate.candidate_id,
        },
        "inputs": {
            "benchmark_summary": {
                "path": str(benchmark_path),
                "sha256": sha256_file(benchmark_path),
            },
            "training_jobs": {
                "path": str(jobs_path),
                "sha256": sha256_file(jobs_path),
            },
            "training_status": {
                "path": str(training_status_path),
                "sha256": sha256_file(training_status_path),
            },
            "base_checkpoint": {
                "path": str(base_checkpoint),
                "sha256": sha256_file(base_checkpoint),
            },
            "configs": {
                job.model_id: {"path": str(job.config), "sha256": sha256_file(job.config)}
                for job in jobs
            },
            "clean_manifests": {
                job.model_id: {
                    "path": str(job.clean_manifest),
                    "sha256": sha256_file(job.clean_manifest),
                }
                for job in jobs
            },
            "applier_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "outputs": {
            "training_jobs": {
                "path": str(output_jobs_path),
                "sha256": sha256_file(staged_jobs),
            },
            "configs": {
                model_id: {
                    "path": str(change.target_config.resolve()),
                    "sha256": sha256_file(staged_configs[model_id]),
                }
                for model_id, change in changes.items()
            },
        },
        "changes": [
            {
                "model_id": change.job.model_id,
                "source_config": {
                    "path": str(change.job.config),
                    "sha256": sha256_file(change.job.config),
                },
                "selected_config": {
                    "path": str(change.target_config.resolve()),
                    "sha256": sha256_file(staged_configs[change.job.model_id]),
                },
                "changed_fields": change.changed_fields,
                "command_config": {
                    "index": change.job.config_argument_index,
                    "before": change.job.command[change.job.config_argument_index],
                    "after": str(change.target_config.resolve()),
                },
                "output_dir": str(change.job.output_dir),
            }
            for change in changes.values()
        ],
    }


def _validate_publish_targets(paths: Sequence[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        message = "speed selection publish targets must be unique"
        raise ValueError(message)
    existing = [path for path in resolved if path.exists()]
    if existing:
        message = f"refusing to overwrite existing speed selection artifact: {existing[0]}"
        raise FileExistsError(message)


def _verify_inputs_unchanged(input_sha256: Mapping[Path, str]) -> None:
    changed = [
        path
        for path, expected_sha256 in input_sha256.items()
        if not path.is_file() or sha256_file(path) != expected_sha256
    ]
    if changed:
        message = f"input changed during speed selection: {min(changed)}"
        raise ValueError(message)


def _resolved_job_path(
    row: Mapping[str, object],
    field: str,
    *,
    base: Path,
    model_id: str,
) -> Path:
    raw_path = _required_string(row, field, source=f"training job {model_id}")
    path = Path(raw_path)
    return (path if path.is_absolute() else base / path).resolve()


def _resolved_config_path(value: object, *, base: Path, source: str) -> Path:
    if not isinstance(value, str) or not value:
        message = f"{source} must be a nonempty path string"
        raise ValueError(message)
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _serialize_job_path(path: Path, *, base: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(base.resolve()))
    except ValueError:
        return str(resolved)


def _required_string(row: Mapping[str, object], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        message = f"{source} requires nonempty string {field}"
        raise ValueError(message)
    return value


def _required_positive_int(row: Mapping[str, object], field: str, *, source: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        message = f"{source} {field} must be a positive integer"
        raise ValueError(message)
    return value


def _required_nonnegative_int(
    row: Mapping[str, object],
    field: str,
    *,
    source: str,
) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = f"{source} {field} must be a nonnegative integer"
        raise ValueError(message)
    return value


def _required_finite_number(row: Mapping[str, object], field: str, *, source: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{source} {field} must be numeric"
        raise TypeError(message)
    numeric = float(value)
    if not math.isfinite(numeric):
        message = f"{source} {field} must be finite"
        raise ValueError(message)
    return numeric


def _optional_finite_number(
    row: Mapping[str, object],
    field: str,
    *,
    source: str,
) -> float | None:
    if row.get(field) is None:
        return None
    return _required_finite_number(row, field, source=source)


def _present_candidate_metric(value: float | None, *, field: str) -> float:
    if value is None:  # eligible candidates are validated to contain this metric
        message = f"eligible candidate is missing {field}"
        raise ValueError(message)
    return value


def _required_hex(
    row: Mapping[str, object],
    field: str,
    *,
    length: int,
    source: str,
) -> str:
    value = _required_string(row, field, source=source)
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        message = f"{source} {field} must be {length}-character lowercase hex"
        raise ValueError(message)
    return value


def _read_json(path: Path) -> dict[str, object]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"invalid JSON: {path}"
        raise ValueError(message) from exc
    if not isinstance(value, dict):
        message = f"JSON document must be an object: {path}"
        raise TypeError(message)
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_training_config(path: Path, value: Mapping[str, object]) -> None:
    document = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_expand_scientific_json_numbers(document) + "\n", encoding="utf-8")


def _expand_scientific_json_numbers(document: str) -> str:
    chunks: list[str] = []
    outside_start = 0
    string_start: int | None = None
    escaped = False
    for index, character in enumerate(document):
        if string_start is None:
            if character == '"':
                chunks.append(_expand_scientific_number_tokens(document[outside_start:index]))
                string_start = index
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            chunks.append(document[string_start : index + 1])
            outside_start = index + 1
            string_start = None
    if string_start is not None:
        message = "serialized training config contains an unterminated JSON string"
        raise ValueError(message)
    chunks.append(_expand_scientific_number_tokens(document[outside_start:]))
    return "".join(chunks)


def _expand_scientific_number_tokens(document: str) -> str:
    def replace(match: re.Match[str]) -> str:
        expanded = format(Decimal(match.group(0)), "f")
        return expanded if "." in expanded else f"{expanded}.0"

    return SCIENTIFIC_JSON_NUMBER_PATTERN.sub(replace, document)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--training-jobs", type=Path, required=True)
    parser.add_argument("--completed-model-id", action="append", required=True)
    parser.add_argument("--output-jobs", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    status = apply_speed_selection(
        benchmark_summary=args.benchmark_summary,
        training_jobs=args.training_jobs,
        completed_model_ids=args.completed_model_id,
        output_jobs=args.output_jobs,
        status_output=args.status_output,
    )
    benchmark_binding = status.get("benchmark_binding")
    if not isinstance(benchmark_binding, dict):  # internal report contract
        message = "speed selection status is missing benchmark binding"
        raise TypeError(message)
    print(
        "speed selection applied: "
        f"{status['changed_model_count']} pending jobs, "
        f"candidate {benchmark_binding['recommended_candidate_id']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
