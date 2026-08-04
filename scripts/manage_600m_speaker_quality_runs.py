# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, TRY003

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

SEARCH_EVALUATION_SCHEMA = "speaker-checkpoint-search-evaluation/v1"
MIN_SPEAKER_SIMILARITY = 0.75
EXPECTED_JOB_COUNT = 12
SEARCH_MAX_STEPS = 250
RETRAIN_MAX_STEPS = 3000
SAVE_EVERY = 250
LOG_EVERY = 20
SHA256_LENGTH = 64
SAFETENSORS_HEADER_BYTES = 8
MAX_GPU_UTILIZATION_PERCENT = 100.0
EXPECTED_HARD_GATE_CASE_COUNT = 16
PERIODIC_STEPS = frozenset(range(250, 3001, 250))
EXPECTED_JOBS_KEYS = {
    "schema_version",
    "created_at_utc",
    "base_checkpoint_path",
    "base_checkpoint_sha256",
    "checkpoint_revision",
    "upstream_commit",
    "queue_policy",
    "anabel_strategy",
    "jobs",
}
QUEUE_SCRIPT_NAME = "run_600m_speaker_training_queue.py"
_LOSS_RE = re.compile(
    r"\bstep\s*(?:=|:|\s)\s*(\d+)\b.*?\bloss\s*(?:=|:)\s*([^\s]+)",
    re.IGNORECASE,
)
_OOM_RE = re.compile(r"\boom\b|cuda\s+out\s+of\s+memory", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class TrainingLogSummary:
    steps: tuple[int, ...]
    last_loss: float


@dataclass(frozen=True, slots=True)
class SourceMetricCase:
    similarity: float
    failing_case: str
    tie_break: tuple[str, str, str, str]


def extract_source_best(
    diagnostic_path: Path,
    *,
    model_id: str,
    checkpoint_step: int,
) -> dict[str, object]:
    snapshot = _snapshot_file(diagnostic_path, source="source diagnostic")
    payload = _diagnostic_payload(snapshot.data)
    return extract_source_best_payload(
        payload,
        model_id=model_id,
        checkpoint_step=checkpoint_step,
    )


def extract_source_best_payload(
    payload: object,
    *,
    model_id: str,
    checkpoint_step: int,
) -> dict[str, object]:
    if isinstance(payload, list):
        return _source_best_from_result_rows(
            payload,
            model_id=model_id,
            checkpoint_step=checkpoint_step,
        )
    if not isinstance(payload, dict) or payload.get("schema_version") != SEARCH_EVALUATION_SCHEMA:
        raise ValueError("unsupported source diagnostic schema")
    if payload.get("model_id") != model_id or payload.get("checkpoint_step") != checkpoint_step:
        raise ValueError("source diagnostic identity mismatch")
    case_count = _integer(payload.get("hard_gate_metric_case_count"), "hard-gate case count")
    pass_count = _integer(payload.get("speaker_similarity_pass_count"), "hard-gate pass count")
    rows = payload.get("per_case_metrics")
    if not isinstance(rows, list):
        raise TypeError("source diagnostic per_case_metrics must be a list")
    if len(rows) != case_count:
        raise ValueError("source diagnostic hard-gate case count mismatch")
    cases = _source_metric_cases(rows)
    computed_pass_count = sum(case.similarity >= MIN_SPEAKER_SIMILARITY for case in cases)
    if pass_count != computed_pass_count or not 0 <= pass_count < case_count:
        raise ValueError("source diagnostic pass count is inconsistent with its hard-gate cases")
    selected = _minimum_failing_case(cases)
    if payload.get("min_speaker_similarity") != selected.similarity:
        raise ValueError("source diagnostic minimum speaker similarity mismatch")
    return {
        "checkpoint_step": checkpoint_step,
        "hard_gate_pass_count": pass_count,
        "hard_gate_case_count": case_count,
        "failing_case": selected.failing_case,
        "speaker_similarity": selected.similarity,
        "required_minimum": MIN_SPEAKER_SIMILARITY,
    }


def _diagnostic_payload(data: bytes) -> object:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("source diagnostic must be UTF-8") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[object] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid source diagnostic JSONL row {line_number}") from exc
        return rows


def _source_best_from_result_rows(
    raw_rows: list[object],
    *,
    model_id: str,
    checkpoint_step: int,
) -> dict[str, object]:
    if not raw_rows or not all(isinstance(row, dict) for row in raw_rows):
        raise TypeError("source evaluator results must contain JSON objects")
    rows = cast("list[dict[str, Any]]", raw_rows)
    if any(
        row.get("evaluation_schema_version") != "speaker-checkpoint-evaluation/v1" for row in rows
    ):
        raise ValueError("unsupported source evaluator result schema")
    metric_rows = [
        row
        for row in rows
        if row.get("model_id") == model_id
        and row.get("checkpoint_step") == checkpoint_step
        and row.get("metric_gate_applied") is True
    ]
    if len(metric_rows) != EXPECTED_HARD_GATE_CASE_COUNT:
        raise ValueError("source evaluator result must contain exactly 16 target hard-gate cases")
    identities = [(row.get("text_id"), row.get("seed"), row.get("style")) for row in metric_rows]
    if len(set(identities)) != len(identities):
        raise ValueError("source evaluator result contains duplicate hard-gate cases")
    cases = _source_metric_cases(metric_rows)
    pass_count = sum(case.similarity >= MIN_SPEAKER_SIMILARITY for case in cases)
    if not 0 <= pass_count < len(cases):
        raise ValueError("source evaluator result must contain at least one failing hard-gate case")
    selected = _minimum_failing_case(cases)
    return {
        "checkpoint_step": checkpoint_step,
        "hard_gate_pass_count": pass_count,
        "hard_gate_case_count": len(metric_rows),
        "failing_case": selected.failing_case,
        "speaker_similarity": selected.similarity,
        "required_minimum": MIN_SPEAKER_SIMILARITY,
    }


def _source_metric_cases(rows: Sequence[object]) -> tuple[SourceMetricCase, ...]:
    cases: list[SourceMetricCase] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise TypeError("source diagnostic case must be an object")
        similarity = raw_row.get("speaker_similarity")
        if isinstance(similarity, bool) or not isinstance(similarity, int | float):
            raise TypeError("source diagnostic speaker_similarity must be numeric")
        value = float(similarity)
        if not math.isfinite(value):
            raise ValueError("source diagnostic speaker_similarity must be finite")
        text_id = raw_row.get("text_id")
        if not isinstance(text_id, str) or not text_id:
            raise ValueError("source diagnostic case must have a text_id")
        seed = raw_row.get("seed")
        if seed is not None and type(seed) is not int:
            raise TypeError("source diagnostic case seed must be an integer when present")
        style = raw_row.get("style")
        if style is not None and (not isinstance(style, str) or not style):
            raise TypeError("source diagnostic case style must be nonempty when present")
        case_id = raw_row.get("case_id")
        if case_id is not None and (not isinstance(case_id, str) or not case_id):
            raise TypeError("source diagnostic case_id must be nonempty when present")
        cases.append(
            SourceMetricCase(
                similarity=value,
                failing_case=case_id or text_id,
                tie_break=(
                    text_id,
                    str(seed) if seed is not None else "",
                    style or "",
                    case_id or "",
                ),
            )
        )
    return tuple(cases)


def _minimum_failing_case(cases: Sequence[SourceMetricCase]) -> SourceMetricCase:
    failing = [case for case in cases if case.similarity < MIN_SPEAKER_SIMILARITY]
    if not failing:
        raise ValueError("source diagnostic must contain at least one failing hard-gate case")
    return min(failing, key=lambda case: (case.similarity, *case.tie_break))


def prepare_quality_run(
    *,
    kind: str,
    predecessor_jobs: Path,
    predecessor_status: Path,
    source_diagnostic: Path,
    model_id: str,
    init_checkpoint_step: int,
    learning_rate: float,
    seed: int,
    run_root: Path,
    queue_script: Path,
    strategy: str | None = None,
) -> dict[str, object]:
    max_steps = _max_steps(kind)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be nonempty")
    if type(init_checkpoint_step) is not int or init_checkpoint_step not in PERIODIC_STEPS:
        raise ValueError("init checkpoint_step must be a periodic predecessor step")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
        raise TypeError("learning_rate must be numeric")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    direct_snapshots = tuple(
        _snapshot_file(path, source=source)
        for path, source in (
            (predecessor_jobs, "predecessor jobs"),
            (predecessor_status, "predecessor status"),
            (source_diagnostic, "source diagnostic"),
            (queue_script, "queue script"),
        )
    )
    jobs_snapshot, status_snapshot, diagnostic_snapshot, queue_snapshot = direct_snapshots
    if queue_snapshot.path.name != QUEUE_SCRIPT_NAME:
        raise ValueError(f"queue script must be named {QUEUE_SCRIPT_NAME}")
    jobs_document = _json_object(jobs_snapshot.data, source="predecessor jobs")
    jobs = jobs_document.get("jobs")
    if (
        not isinstance(jobs, list)
        or len(jobs) != EXPECTED_JOB_COUNT
        or not all(isinstance(job, dict) for job in jobs)
    ):
        raise ValueError("predecessor jobs must contain exactly 12 job objects")
    model_ids = [_required_string(job, "model_id", source="predecessor job") for job in jobs]
    if len(set(model_ids)) != EXPECTED_JOB_COUNT:
        raise ValueError("predecessor jobs contain duplicate model ids")
    if model_ids.count(model_id) != 1:
        raise ValueError("target model must occur exactly once in predecessor jobs")
    target_index = model_ids.index(model_id)
    target = jobs[target_index]
    if not isinstance(target, dict):
        raise TypeError("target predecessor job must be an object")

    jobs_base = jobs_snapshot.path.parent
    config_path = _job_path(target, "config", base=jobs_base)
    manifest_path = _job_path(target, "clean_manifest", base=jobs_base)
    predecessor_output = _job_path(target, "output_dir", base=jobs_base)
    base_checkpoint = _bound_path(
        jobs_document.get("base_checkpoint_path"),
        base=jobs_base,
        source="base checkpoint",
    )
    lexical_init_checkpoint = (
        predecessor_output / f"checkpoint_{init_checkpoint_step:07d}.speaker.safetensors"
    )
    _require_no_alias_components(lexical_init_checkpoint, source="initialization checkpoint")
    init_checkpoint = lexical_init_checkpoint.resolve()
    if init_checkpoint.parent != predecessor_output.resolve():
        raise ValueError("initialization checkpoint escapes predecessor output")
    derived_snapshots = tuple(
        _snapshot_file(path, source=source)
        for path, source in (
            (config_path, "predecessor config"),
            (manifest_path, "clean manifest"),
            (base_checkpoint, "base checkpoint"),
            (init_checkpoint, "initialization checkpoint"),
        )
    )
    config_snapshot, manifest_snapshot, base_snapshot, init_snapshot = derived_snapshots
    _validate_jobs_provenance(jobs_document, base_snapshot)
    latest = _latest_successful_target_status(
        status_snapshot.data,
        model_id=model_id,
        config_sha256=config_snapshot.sha256,
        manifest_sha256=manifest_snapshot.sha256,
        base_sha256=base_snapshot.sha256,
        checkpoint_revision=_required_string(
            jobs_document, "checkpoint_revision", source="predecessor jobs"
        ),
        upstream_commit=_required_string(
            jobs_document, "upstream_commit", source="predecessor jobs"
        ),
        init_checkpoint=init_checkpoint,
        init_sha256=init_snapshot.sha256,
    )
    created_at = _required_iso_datetime(latest.get("ended_at"), source="predecessor ended_at")
    source_best = extract_source_best_payload(
        _diagnostic_payload(diagnostic_snapshot.data),
        model_id=model_id,
        checkpoint_step=init_checkpoint_step,
    )

    final_root = _new_run_root(run_root)
    new_output = final_root / "training"
    new_config = final_root / "config.json"
    new_jobs = final_root / "training-jobs.json"
    new_status = final_root / "training-status.jsonl"
    new_setup = final_root / "setup-evidence.json"
    config_document = _json_object(config_snapshot.data, source="predecessor config")
    successor_config = _successor_config(
        config_document,
        manifest=manifest_path,
        output_dir=new_output,
        init_checkpoint=init_checkpoint,
        learning_rate=float(learning_rate),
        seed=seed,
        max_steps=max_steps,
        init_sha256=init_snapshot.sha256,
    )
    config_bytes = _config_json_bytes(successor_config)
    successor_jobs = copy.deepcopy(jobs_document)
    raw_successor_jobs = successor_jobs.get("jobs")
    if not isinstance(raw_successor_jobs, list):
        raise TypeError("successor jobs must be a list")
    successor_target = raw_successor_jobs[target_index]
    if not isinstance(successor_target, dict):
        raise TypeError("successor target job must be an object")
    successor_target["config"] = str(new_config)
    successor_target["output_dir"] = str(new_output)
    successor_target["command"] = _rewrite_target_command(
        target.get("command"),
        predecessor_config=config_path,
        predecessor_output=predecessor_output,
        successor_config=new_config,
        successor_output=new_output,
        manifest=manifest_path,
        base_checkpoint=base_snapshot.path,
    )
    jobs_bytes = _json_bytes(successor_jobs)
    strategy_text = strategy or "initialize from the strongest predecessor checkpoint"
    if not strategy_text:
        raise ValueError("strategy must be nonempty")
    setup = _setup_payload(
        kind=kind,
        created_at=created_at,
        model_id=model_id,
        diagnostic=diagnostic_snapshot,
        source_best=source_best,
        strategy=strategy_text,
        source_config=config_snapshot,
        config_path=new_config,
        config_bytes=config_bytes,
        jobs_path=new_jobs,
        jobs_bytes=jobs_bytes,
        status_path=new_status,
        status_snapshot=status_snapshot,
        queue_script=queue_snapshot,
        output_dir=new_output,
        init_checkpoint=init_snapshot,
        predecessor_config=config_document,
        successor_config=successor_config,
    )
    setup_bytes = _json_bytes(setup)

    _recheck_snapshots((*direct_snapshots, *derived_snapshots))
    final_root.mkdir()
    new_output.mkdir()
    _write_new(new_config, config_bytes)
    _write_new(new_jobs, jobs_bytes)
    _write_new(new_status, status_snapshot.data)
    _write_new(new_setup, setup_bytes)
    return {
        "kind": kind,
        "model_id": model_id,
        "run_root": str(final_root),
        "config": str(new_config),
        "jobs": str(new_jobs),
        "status": str(new_status),
        "setup_evidence": str(new_setup),
        "output_dir": str(new_output),
    }


def finalize_quality_run(
    *,
    setup_evidence: Path,
    training_jobs: Path,
    training_status: Path,
    queue_script: Path,
    queue_exit_code: int,
    runtime_after: Path,
    output: Path,
) -> dict[str, object]:
    if type(queue_exit_code) is not int or queue_exit_code != 0:
        raise ValueError("queue exit code must be integer zero")
    direct_snapshots = tuple(
        _snapshot_file(path, source=source)
        for path, source in (
            (setup_evidence, "setup evidence"),
            (training_jobs, "training jobs"),
            (training_status, "training status"),
            (queue_script, "queue script"),
            (runtime_after, "runtime-after snapshot"),
        )
    )
    setup_snapshot, jobs_snapshot, status_snapshot, queue_snapshot, runtime_snapshot = (
        direct_snapshots
    )
    if queue_snapshot.path.name != QUEUE_SCRIPT_NAME:
        raise ValueError(f"queue script must be named {QUEUE_SCRIPT_NAME}")
    setup = _json_object(setup_snapshot.data, source="setup evidence")
    schema = setup.get("schema_version")
    if schema == "speaker-quality-search-setup/v1":
        kind = "search"
    elif schema == "speaker-quality-retrain-setup/v1":
        kind = "retrain"
    else:
        raise ValueError("unsupported setup evidence schema")
    expected_setup_fields = {
        "schema_version",
        "created_at",
        "model_id",
        "reason",
        "changes",
        "paths",
        "sha256",
    }
    if kind == "search":
        expected_setup_fields.add("candidate")
    _require_exact_keys(setup, expected_setup_fields, source="setup evidence")
    run_root = setup_snapshot.path.parent
    output_path = output.expanduser().absolute()
    _require_no_alias_components(output_path, source="run evidence output")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite run evidence: {output_path}")
    if output_path.parent.resolve(strict=True) != run_root:
        raise ValueError("run evidence output must be directly under the prepared run root")
    paths = setup.get("paths")
    hashes = setup.get("sha256")
    reason = setup.get("reason")
    changes = setup.get("changes")
    if not all(isinstance(value, dict) for value in (paths, hashes, reason, changes)):
        raise TypeError("setup evidence paths/hashes/reason/changes must be objects")
    paths = cast("dict[str, Any]", paths)
    hashes = cast("dict[str, Any]", hashes)
    reason = cast("dict[str, Any]", reason)
    changes = cast("dict[str, Any]", changes)
    _require_exact_keys(
        paths,
        {"config", "jobs", "status", "queue_script", "output_dir"},
        source="setup evidence paths",
    )
    _require_exact_keys(
        hashes,
        {"source_config", "config", "jobs", "status_seed", "queue_script"},
        source="setup evidence sha256",
    )
    _require_exact_keys(
        reason,
        {"source_diagnostic", "source_diagnostic_sha256", "source_best", "strategy"},
        source="setup evidence reason",
    )
    source_best = reason.get("source_best")
    if not isinstance(source_best, dict):
        raise TypeError("setup evidence reason source_best must be an object")
    _require_exact_keys(
        source_best,
        {
            "checkpoint_step",
            "hard_gate_pass_count",
            "hard_gate_case_count",
            "failing_case",
            "speaker_similarity",
            "required_minimum",
        },
        source="setup evidence reason source_best",
    )
    _require_exact_keys(
        changes,
        {
            "learning_rate",
            "seed",
            "max_steps",
            "save_every",
            "batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "speaker_inversion_init_embedding",
            "speaker_inversion_init_embedding_sha256",
        },
        source="setup evidence changes",
    )
    for field in ("learning_rate", "seed"):
        change = changes.get(field)
        if not isinstance(change, dict):
            raise TypeError(f"setup evidence changes {field} must be an object")
        _require_exact_keys(
            change,
            {"from", "to"},
            source=f"setup evidence changes {field}",
        )
    if kind == "search":
        candidate = setup.get("candidate")
        if not isinstance(candidate, dict):
            raise TypeError("setup evidence candidate must be an object")
        _require_exact_keys(
            candidate,
            {
                "model_id",
                "init_label",
                "speaker_inversion_init_embedding",
                "speaker_inversion_init_embedding_sha256",
            },
            source="setup evidence candidate",
        )
    if _bound_path(paths.get("jobs"), base=run_root, source="setup jobs") != jobs_snapshot.path:
        raise ValueError("setup evidence jobs path mismatch")
    if (
        _bound_path(paths.get("status"), base=run_root, source="setup status")
        != status_snapshot.path
    ):
        raise ValueError("setup evidence status path mismatch")
    if hashes.get("jobs") != jobs_snapshot.sha256:
        raise ValueError("setup evidence jobs SHA-256 mismatch")
    setup_queue_script = _bound_path(
        paths.get("queue_script"),
        base=run_root,
        source="setup queue script",
    )
    if setup_queue_script != queue_snapshot.path:
        raise ValueError("setup queue script path mismatch")
    if hashes.get("queue_script") != queue_snapshot.sha256:
        raise ValueError("setup queue script SHA-256 mismatch")
    before_sha = hashes.get("status_seed")
    if not isinstance(before_sha, str) or len(before_sha) != SHA256_LENGTH:
        raise ValueError("setup evidence status seed SHA-256 is invalid")

    jobs = _json_object(jobs_snapshot.data, source="training jobs")
    raw_jobs = jobs.get("jobs")
    if (
        not isinstance(raw_jobs, list)
        or len(raw_jobs) != EXPECTED_JOB_COUNT
        or not all(isinstance(job, dict) for job in raw_jobs)
    ):
        raise ValueError("training jobs must contain exactly 12 job objects")
    model_ids = [_required_string(job, "model_id", source="training job") for job in raw_jobs]
    if len(set(model_ids)) != EXPECTED_JOB_COUNT:
        raise ValueError("training jobs contain duplicate model ids")
    model_id = _required_string(setup, "model_id", source="setup evidence")
    if model_ids.count(model_id) != 1:
        raise ValueError("prepared target model is missing from training jobs")
    target = raw_jobs[model_ids.index(model_id)]
    if not isinstance(target, dict):
        raise TypeError("target training job must be an object")
    config_path = _job_path(target, "config", base=jobs_snapshot.path.parent)
    manifest_path = _job_path(target, "clean_manifest", base=jobs_snapshot.path.parent)
    output_dir = _job_path(target, "output_dir", base=jobs_snapshot.path.parent)
    if config_path != _bound_path(paths.get("config"), base=run_root, source="setup config"):
        raise ValueError("setup config path mismatch")
    if output_dir != _bound_path(paths.get("output_dir"), base=run_root, source="setup output"):
        raise ValueError("setup output path mismatch")
    if not output_dir.is_relative_to(run_root):
        raise ValueError("training output escapes the prepared run root")
    base_checkpoint = _bound_path(
        jobs.get("base_checkpoint_path"), base=jobs_snapshot.path.parent, source="base checkpoint"
    )
    init_checkpoint = _bound_path(
        changes.get("speaker_inversion_init_embedding"),
        base=run_root,
        source="initialization checkpoint",
    )
    diagnostic = _bound_path(
        reason.get("source_diagnostic"), base=run_root, source="source diagnostic"
    )
    derived_snapshots = tuple(
        _snapshot_file(path, source=source)
        for path, source in (
            (config_path, "prepared config"),
            (manifest_path, "clean manifest"),
            (base_checkpoint, "base checkpoint"),
            (init_checkpoint, "initialization checkpoint"),
            (diagnostic, "source diagnostic"),
        )
    )
    config_snapshot, manifest_snapshot, base_snapshot, init_snapshot, diagnostic_snapshot = (
        derived_snapshots
    )
    if hashes.get("config") != config_snapshot.sha256:
        raise ValueError("setup config SHA-256 mismatch")
    if changes.get("speaker_inversion_init_embedding_sha256") != init_snapshot.sha256:
        raise ValueError("setup initialization checkpoint SHA-256 mismatch")
    if reason.get("source_diagnostic_sha256") != diagnostic_snapshot.sha256:
        raise ValueError("setup source diagnostic SHA-256 mismatch")
    _validate_jobs_provenance(jobs, base_snapshot)
    _validate_prepared_config(
        kind=kind,
        config=_json_object(config_snapshot.data, source="prepared config"),
        manifest=manifest_path,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
    )
    _validate_target_command(
        target.get("command"),
        config=config_path,
        output=output_dir,
        manifest=manifest_path,
        base_checkpoint=base_checkpoint,
    )

    status_lines = status_snapshot.data.splitlines(keepends=True)
    if any(not line.endswith(b"\n") for line in status_lines):
        raise ValueError("training status must end every JSONL row with a newline")
    prefix_matches = [
        count
        for count in range(len(status_lines) + 1)
        if hashlib.sha256(b"".join(status_lines[:count])).hexdigest() == before_sha
    ]
    if len(prefix_matches) != 1:
        raise ValueError("training status seed is missing or ambiguous")
    before_count = prefix_matches[0]
    if len(status_lines) != before_count + 2:
        raise ValueError("training status append drift: expected exactly two new rows")
    rows = _status_rows(status_snapshot.data)
    started, finished = rows[before_count:]
    _validate_appended_status(
        started,
        finished,
        model_id=model_id,
        config=config_snapshot,
        manifest=manifest_snapshot,
        base=base_snapshot,
        checkpoint_revision=_required_string(jobs, "checkpoint_revision", source="training jobs"),
        upstream_commit=_required_string(jobs, "upstream_commit", source="training jobs"),
    )
    if [row for row in rows if row.get("model_id") == model_id][-1] != finished:
        raise ValueError("latest target status is not the appended successful finish")

    checkpoint_snapshots = _checkpoint_inventory(kind=kind, output_dir=output_dir)
    expected_last_name = (
        "checkpoint_0000250.speaker.safetensors"
        if kind == "search"
        else "checkpoint_0003000.speaker.safetensors"
    )
    by_name = {snapshot.path.name: snapshot for snapshot in checkpoint_snapshots}
    last = by_name[expected_last_name]
    if (
        _bound_path(
            finished.get("last_checkpoint"),
            base=status_snapshot.path.parent,
            source="last checkpoint",
        )
        != last.path
        or finished.get("last_checkpoint_sha256") != last.sha256
    ):
        raise ValueError("finished status last checkpoint mismatch")
    candidates = finished.get("candidate_checkpoints")
    if not isinstance(candidates, list) or len(candidates) != len(checkpoint_snapshots):
        raise ValueError("finished status checkpoint candidate count mismatch")
    candidate_bindings = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("status checkpoint candidate must be an object")
        candidate_bindings.append(
            (
                _bound_path(
                    candidate.get("path"), base=status_snapshot.path.parent, source="candidate"
                ),
                candidate.get("sha256"),
            )
        )
    expected_bindings = {(snapshot.path, snapshot.sha256) for snapshot in checkpoint_snapshots}
    if (
        len(set(candidate_bindings)) != len(candidate_bindings)
        or set(candidate_bindings) != expected_bindings
    ):
        raise ValueError("finished status checkpoint bindings mismatch")
    final_snapshot = by_name["checkpoint_final.speaker.safetensors"]
    if final_snapshot.data != last.data or final_snapshot.sha256 != last.sha256:
        raise ValueError("final checkpoint does not equal the terminal periodic checkpoint")

    log_path = _bound_path(finished.get("log_path"), base=status_snapshot.path.parent, source="log")
    log_snapshot = _snapshot_file(log_path, source="training log")
    log_summary = _training_log_summary(kind=kind, data=log_snapshot.data)
    runtime = _json_object(runtime_snapshot.data, source="runtime-after snapshot")
    _validate_runtime_after(runtime)
    run_checkpoints = [
        {"name": snapshot.path.name, "path": str(snapshot.path), "sha256": snapshot.sha256}
        for snapshot in sorted(checkpoint_snapshots, key=lambda item: item.path.name)
    ]
    started_at = _required_iso_datetime(started.get("started_at"), source="started_at")
    ended_at = _required_iso_datetime(finished.get("ended_at"), source="ended_at")
    run: dict[str, object] = {
        "started_at": started_at,
        "ended_at": ended_at,
        "config_sha256": config_snapshot.sha256,
        "clean_manifest_sha256": manifest_snapshot.sha256,
        "base_checkpoint_sha256": base_snapshot.sha256,
        "candidate_checkpoint_count": len(checkpoint_snapshots),
        "checkpoints": run_checkpoints,
    }
    if _parse_iso_datetime(ended_at, source="ended_at") < _parse_iso_datetime(
        started_at, source="started_at"
    ):
        raise ValueError("training timestamps are reversed")
    if kind == "search":
        run["config_path"] = str(config_snapshot.path)
        run["final_equals_step250"] = True
        run["log"] = {
            "path": str(log_snapshot.path),
            "sha256": log_snapshot.sha256,
            "loss_event_count": len(log_summary.steps),
            "loss_steps": list(log_summary.steps),
            "loss_all_finite": True,
            "last_loss": log_summary.last_loss,
            "oom": False,
            "traceback": False,
        }
    else:
        run["final_equals_step3000"] = True
        run["log"] = {
            "path": str(log_snapshot.path),
            "sha256": log_snapshot.sha256,
            "loss_event_count": len(log_summary.steps),
            "loss_steps_exact": True,
            "loss_all_finite": True,
            "last_loss": log_summary.last_loss,
            "oom": False,
            "traceback": False,
        }
    evidence: dict[str, object] = {
        "schema_version": f"speaker-quality-{kind}-run-evidence/v1",
        "created_at": str(run["ended_at"]),
        "state": "finished",
        "model_id": model_id,
        "queue_exit_code": queue_exit_code,
        "setup_evidence": {"path": str(setup_snapshot.path), "sha256": setup_snapshot.sha256},
        "training_jobs": {"path": str(jobs_snapshot.path), "sha256": jobs_snapshot.sha256},
        "training_status": {
            "path": str(status_snapshot.path),
            "before_row_count": before_count,
            "before_sha256": before_sha,
            "after_row_count": len(rows),
            "after_sha256": status_snapshot.sha256,
            "new_status_row_count": 2,
            "new_started_model_ids": [model_id],
            "new_finished_success_model_ids": [model_id],
        },
        "queue_script": {"path": str(queue_snapshot.path), "sha256": queue_snapshot.sha256},
        "invocation": {
            "recipe": f"speaker-quality-{kind}",
            "checkpoint_revision": jobs["checkpoint_revision"],
            "upstream_commit": jobs["upstream_commit"],
        },
        "run": run,
        "runtime_after": runtime,
    }
    evidence_bytes = _json_bytes(evidence)
    _recheck_snapshots(
        (
            *direct_snapshots,
            *derived_snapshots,
            *checkpoint_snapshots,
            log_snapshot,
        )
    )
    current_checkpoints = _checkpoint_inventory(kind=kind, output_dir=output_dir)
    if [(item.path, item.sha256) for item in current_checkpoints] != [
        (item.path, item.sha256) for item in checkpoint_snapshots
    ]:
        raise ValueError("checkpoint inventory changed after snapshot")
    _write_atomic_create_only(output_path, evidence_bytes)
    return evidence


def _max_steps(kind: str) -> int:
    if kind == "search":
        return SEARCH_MAX_STEPS
    if kind == "retrain":
        return RETRAIN_MAX_STEPS
    raise ValueError("kind must be 'search' or 'retrain'")


def _snapshot_file(path: Path, *, source: str) -> FileSnapshot:
    lexical = path.expanduser().absolute()
    _require_no_alias_components(lexical, source=source)
    try:
        metadata = lexical.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"{source} is missing: {lexical}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{source} must be a regular file: {lexical}")
    data = lexical.read_bytes()
    return FileSnapshot(
        path=lexical.resolve(strict=True),
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mtime_ns=metadata.st_mtime_ns,
    )


def _require_no_alias_components(path: Path, *, source: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink() or _is_reparse_alias(current):
            raise ValueError(
                f"{source} must not contain a symlink, junction, reparse, or alias component: "
                f"{current}"
            )


def _is_reparse_alias(path: Path) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not reparse_flag:
        return False
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & reparse_flag)


def _recheck_snapshots(snapshots: tuple[FileSnapshot, ...]) -> None:
    for snapshot in snapshots:
        current = _snapshot_file(snapshot.path, source="snapshotted input")
        if (
            current.data != snapshot.data
            or current.device != snapshot.device
            or current.inode != snapshot.inode
            or current.mtime_ns != snapshot.mtime_ns
        ):
            raise ValueError(f"input changed after snapshot: {snapshot.path}")


def _json_object(data: bytes, *, source: str) -> dict[str, Any]:
    try:
        payload: Any = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} must contain valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{source} must be a JSON object")
    return payload


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _config_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (_render_config_json(payload, depth=0) + "\n").encode("utf-8")


def _render_config_json(value: object, *, depth: int) -> str:
    indent = "  " * depth
    child_indent = "  " * (depth + 1)
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        if not all(isinstance(key, str) for key in value):
            raise TypeError("config JSON object keys must be strings")
        items = [
            f"{child_indent}{json.dumps(key, ensure_ascii=False)}: "
            f"{_render_config_json(value[key], depth=depth + 1)}"
            for key in sorted(value)
        ]
        return "{\n" + ",\n".join(items) + f"\n{indent}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [f"{child_indent}{_render_config_json(item, depth=depth + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{indent}]"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("config JSON floats must be finite")
        return format(Decimal(str(value)), "f")
    if value is None or isinstance(value, bool | int | str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"unsupported config JSON value: {type(value).__name__}")


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, source: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{source} field set mismatch")


def _required_string(row: Mapping[str, object], key: str, *, source: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} {key} must be a nonempty string")
    return value


def _bound_path(value: object, *, base: Path, source: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} path must be nonempty")
    lexical = Path(value)
    bound = (lexical if lexical.is_absolute() else base / lexical).expanduser().absolute()
    _require_no_alias_components(bound, source=source)
    return bound.resolve()


def _job_path(job: Mapping[str, object], field: str, *, base: Path) -> Path:
    return _bound_path(job.get(field), base=base, source=f"job {field}")


def _validate_jobs_provenance(jobs: Mapping[str, object], base: FileSnapshot) -> None:
    if set(jobs) != EXPECTED_JOBS_KEYS:
        raise ValueError("predecessor jobs keys do not match the queue contract")
    declared = jobs.get("base_checkpoint_sha256")
    if declared != base.sha256:
        raise ValueError("predecessor base checkpoint SHA-256 mismatch")
    if (
        jobs.get("schema_version") != 1
        or jobs.get("queue_policy") != "serial_one_at_a_time"
        or jobs.get("anabel_strategy") != "reuse_existing_fresh_3000_run"
    ):
        raise ValueError("predecessor jobs queue policy mismatch")
    _required_iso_datetime(jobs.get("created_at_utc"), source="predecessor jobs created_at_utc")
    _required_string(jobs, "checkpoint_revision", source="predecessor jobs")
    _required_string(jobs, "upstream_commit", source="predecessor jobs")


def _status_rows(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        try:
            row: Any = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid predecessor status JSONL row {line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"predecessor status row {line_number} must be an object")
        rows.append(row)
    return rows


def _latest_successful_target_status(
    data: bytes,
    *,
    model_id: str,
    config_sha256: str,
    manifest_sha256: str,
    base_sha256: str,
    checkpoint_revision: str,
    upstream_commit: str,
    init_checkpoint: Path,
    init_sha256: str,
) -> dict[str, Any]:
    matching = [row for row in _status_rows(data) if row.get("model_id") == model_id]
    if not matching:
        raise ValueError("predecessor status has no target row")
    latest = matching[-1]
    if not (
        latest.get("event") == "finished"
        and latest.get("status") == "success"
        and type(latest.get("exit_code")) is int
        and latest.get("exit_code") == 0
        and latest.get("error") is None
    ):
        raise ValueError("latest predecessor target status is not a successful finish")
    expected = {
        "config_sha256": config_sha256,
        "clean_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": base_sha256,
        "checkpoint_revision": checkpoint_revision,
        "upstream_commit": upstream_commit,
    }
    if any(latest.get(key) != value for key, value in expected.items()):
        raise ValueError("latest predecessor target status provenance mismatch")
    candidates = latest.get("candidate_checkpoints")
    if not isinstance(candidates, list):
        raise TypeError("latest predecessor checkpoint candidates must be a list")
    expected_binding = (init_checkpoint.resolve(), init_sha256)
    bindings = {
        (
            _bound_path(candidate.get("path"), base=init_checkpoint.parent, source="candidate"),
            candidate.get("sha256"),
        )
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    if expected_binding not in bindings:
        raise ValueError("initialization checkpoint is not bound by latest predecessor success")
    return latest


def _required_iso_datetime(value: object, *, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} must be nonempty")
    _parse_iso_datetime(value, source=source)
    return value


def _parse_iso_datetime(value: str, *, source: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{source} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{source} must include a timezone")
    return parsed


def _new_run_root(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError(f"refusing to overwrite run root: {lexical}")
    parent = lexical.parent
    try:
        _require_no_alias_components(parent, source="run root parent")
    except ValueError as exc:
        raise ValueError(f"run root parent must be a real directory: {parent}") from exc
    if not parent.is_dir():
        raise ValueError(f"run root parent must be a real directory: {parent}")
    resolved_parent = parent.resolve(strict=True)
    resolved = resolved_parent / lexical.name
    if resolved.parent != resolved_parent or not lexical.name:
        raise ValueError("run root escapes its parent")
    return resolved


def _successor_config(
    predecessor: Mapping[str, object],
    *,
    manifest: Path,
    output_dir: Path,
    init_checkpoint: Path,
    learning_rate: float,
    seed: int,
    max_steps: int,
    init_sha256: str,
) -> dict[str, Any]:
    successor = copy.deepcopy(dict(predecessor))
    train = successor.get("train")
    if not isinstance(train, dict):
        raise TypeError("predecessor config train must be an object")
    configured_manifest = _bound_path(
        train.get("manifest_path"), base=manifest.parent, source="config manifest"
    )
    if configured_manifest != manifest.resolve():
        raise ValueError("predecessor config manifest does not match target job")
    for field in ("batch_size", "gradient_accumulation_steps"):
        value = train.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"predecessor config train.{field} must be a positive integer")
    if type(train.get("gradient_checkpointing")) is not bool:
        raise ValueError("predecessor config train.gradient_checkpointing must be boolean")
    fixed_contract = {
        "speaker_inversion_enabled": True,
        "speaker_inversion_tokens": 16,
        "valid_ratio": 0.0,
        "checkpoint_best_n": 0,
    }
    if any(
        train.get(field) != expected or type(train.get(field)) is not type(expected)
        for field, expected in fixed_contract.items()
    ):
        raise ValueError("predecessor config does not match the fixed speaker training contract")
    if train.get("log_every") != LOG_EVERY:
        raise ValueError(f"predecessor config train.log_every must be {LOG_EVERY}")
    existing_init_sha = train.get("speaker_inversion_init_embedding_sha256")
    if existing_init_sha is not None and existing_init_sha != init_sha256:
        raise ValueError("predecessor config has a stale initialization checkpoint SHA-256")
    train.update(
        {
            "manifest_path": str(manifest.resolve()),
            "output_dir": str(output_dir.resolve()),
            "learning_rate": learning_rate,
            "seed": seed,
            "max_steps": max_steps,
            "save_every": SAVE_EVERY,
            "speaker_inversion_init_embedding": str(init_checkpoint.resolve()),
        }
    )
    return successor


def _validate_target_command(
    raw: object,
    *,
    config: Path,
    output: Path,
    manifest: Path,
    base_checkpoint: Path,
) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(part, str) and part for part in raw):
        raise ValueError("target command must be a nonempty string list")
    result = list(raw)
    expected_paths = {
        "--config": config,
        "--manifest": manifest,
        "--init-checkpoint": base_checkpoint,
        "--output-dir": output,
    }
    for flag, expected in expected_paths.items():
        if result.count(flag) != 1:
            raise ValueError(f"target command must contain exactly one {flag} argument")
        index = result.index(flag) + 1
        if index >= len(result):
            raise ValueError(f"target command is missing the {flag} value")
        actual = _bound_path(result[index], base=expected.parent, source=f"target command {flag}")
        if actual != expected.resolve():
            raise ValueError(f"target command {flag} path mismatch")
    return result


def _rewrite_target_command(
    raw: object,
    *,
    predecessor_config: Path,
    predecessor_output: Path,
    successor_config: Path,
    successor_output: Path,
    manifest: Path,
    base_checkpoint: Path,
) -> list[str]:
    result = _validate_target_command(
        raw,
        config=predecessor_config,
        output=predecessor_output,
        manifest=manifest,
        base_checkpoint=base_checkpoint,
    )
    for flag, successor in (
        ("--config", successor_config),
        ("--output-dir", successor_output),
    ):
        result[result.index(flag) + 1] = str(successor.resolve())
    return _validate_target_command(
        result,
        config=successor_config,
        output=successor_output,
        manifest=manifest,
        base_checkpoint=base_checkpoint,
    )


def _setup_payload(
    *,
    kind: str,
    created_at: str,
    model_id: str,
    diagnostic: FileSnapshot,
    source_best: Mapping[str, object],
    strategy: str,
    source_config: FileSnapshot,
    config_path: Path,
    config_bytes: bytes,
    jobs_path: Path,
    jobs_bytes: bytes,
    status_path: Path,
    status_snapshot: FileSnapshot,
    queue_script: FileSnapshot,
    output_dir: Path,
    init_checkpoint: FileSnapshot,
    predecessor_config: Mapping[str, object],
    successor_config: Mapping[str, object],
) -> dict[str, object]:
    before_train = predecessor_config.get("train")
    after_train = successor_config.get("train")
    if not isinstance(before_train, dict) or not isinstance(after_train, dict):
        raise TypeError("training config train must be an object")
    reason = {
        "source_diagnostic": str(diagnostic.path),
        "source_diagnostic_sha256": diagnostic.sha256,
        "source_best": dict(source_best),
        "strategy": strategy,
    }
    changes = {
        "learning_rate": {
            "from": before_train.get("learning_rate"),
            "to": after_train.get("learning_rate"),
        },
        "seed": {"from": before_train.get("seed"), "to": after_train.get("seed")},
        "max_steps": after_train.get("max_steps"),
        "save_every": after_train.get("save_every"),
        "batch_size": after_train.get("batch_size"),
        "gradient_accumulation_steps": after_train.get("gradient_accumulation_steps"),
        "gradient_checkpointing": after_train.get("gradient_checkpointing"),
        "speaker_inversion_init_embedding": str(init_checkpoint.path),
        "speaker_inversion_init_embedding_sha256": init_checkpoint.sha256,
    }
    payload: dict[str, object] = {
        "schema_version": f"speaker-quality-{kind}-setup/v1",
        "created_at": created_at,
        "model_id": model_id,
        "reason": reason,
        "changes": changes,
        "paths": {
            "config": str(config_path),
            "jobs": str(jobs_path),
            "status": str(status_path),
            "queue_script": str(queue_script.path),
            "output_dir": str(output_dir),
        },
        "sha256": {
            "source_config": source_config.sha256,
            "config": hashlib.sha256(config_bytes).hexdigest(),
            "jobs": hashlib.sha256(jobs_bytes).hexdigest(),
            "status_seed": status_snapshot.sha256,
            "queue_script": queue_script.sha256,
        },
    }
    if kind == "search":
        payload["candidate"] = {
            "model_id": model_id,
            "init_label": f"original{source_best['checkpoint_step']}",
            "speaker_inversion_init_embedding": str(init_checkpoint.path),
            "speaker_inversion_init_embedding_sha256": init_checkpoint.sha256,
        }
    return payload


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        os.close(descriptor)


def _validate_prepared_config(
    *,
    kind: str,
    config: Mapping[str, object],
    manifest: Path,
    output_dir: Path,
    init_checkpoint: Path,
) -> None:
    train = config.get("train")
    if not isinstance(train, dict):
        raise TypeError("prepared config train must be an object")
    if (
        train.get("max_steps") != _max_steps(kind)
        or train.get("save_every") != SAVE_EVERY
        or train.get("log_every") != LOG_EVERY
    ):
        raise ValueError("prepared config fixed training steps mismatch")
    for field in ("learning_rate",):
        value = train.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"prepared config train.{field} must be finite")
    if type(train.get("seed")) is not int:
        raise ValueError("prepared config train.seed must be an integer")
    if (
        _bound_path(train.get("manifest_path"), base=manifest.parent, source="config manifest")
        != manifest
        or _bound_path(train.get("output_dir"), base=output_dir.parent, source="config output")
        != output_dir
        or _bound_path(
            train.get("speaker_inversion_init_embedding"),
            base=init_checkpoint.parent,
            source="config initialization checkpoint",
        )
        != init_checkpoint
    ):
        raise ValueError("prepared config path binding mismatch")


def _validate_appended_status(
    started: Mapping[str, object],
    finished: Mapping[str, object],
    *,
    model_id: str,
    config: FileSnapshot,
    manifest: FileSnapshot,
    base: FileSnapshot,
    checkpoint_revision: str,
    upstream_commit: str,
) -> None:
    if not (
        started.get("model_id") == model_id
        and started.get("event") == "started"
        and started.get("status") == "running"
        and started.get("ended_at") is None
        and started.get("exit_code") is None
        and started.get("last_checkpoint") is None
        and started.get("last_checkpoint_sha256") is None
        and started.get("candidate_checkpoints") == []
        and started.get("error") is None
        and finished.get("model_id") == model_id
        and finished.get("event") == "finished"
        and finished.get("status") == "success"
        and type(finished.get("exit_code")) is int
        and finished.get("exit_code") == 0
        and finished.get("error") is None
        and finished.get("started_at") == started.get("started_at")
        and finished.get("log_path") == started.get("log_path")
    ):
        raise ValueError("appended status rows must be started then finished success")
    expected = {
        "config_sha256": config.sha256,
        "clean_manifest_sha256": manifest.sha256,
        "checkpoint_sha256": base.sha256,
        "checkpoint_revision": checkpoint_revision,
        "upstream_commit": upstream_commit,
    }
    for row in (started, finished):
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("appended status provenance mismatch")


def _checkpoint_inventory(*, kind: str, output_dir: Path) -> tuple[FileSnapshot, ...]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("training output must be a real directory")
    periodic_steps = (250,) if kind == "search" else tuple(range(250, 3001, 250))
    expected_names = {
        *(f"checkpoint_{step:07d}.speaker.safetensors" for step in periodic_steps),
        "checkpoint_final.speaker.safetensors",
    }
    lexical = tuple(output_dir.glob("*.speaker.safetensors"))
    nested = tuple(
        path for path in output_dir.rglob("*.speaker.safetensors") if path.parent != output_dir
    )
    if (
        nested
        or len(lexical) != len(expected_names)
        or {path.name for path in lexical} != expected_names
    ):
        raise ValueError("checkpoint inventory contains a missing or extra checkpoint")
    snapshots = tuple(
        _snapshot_file(path, source="training checkpoint") for path in sorted(lexical)
    )
    for snapshot in snapshots:
        if snapshot.path.parent != output_dir.resolve():
            raise ValueError("training checkpoint escapes output directory")
        _validate_embedding_bytes(snapshot.data, source=str(snapshot.path))
    return snapshots


def _validate_embedding_bytes(data: bytes, *, source: str) -> None:
    if len(data) < SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"speaker embedding header is truncated: {source}")
    header_length = struct.unpack("<Q", data[:8])[0]
    if header_length <= 0 or 8 + header_length > len(data):
        raise ValueError(f"speaker embedding header size is invalid: {source}")
    try:
        header: Any = json.loads(data[8 : 8 + header_length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"speaker embedding header JSON is invalid: {source}") from exc
    tensor = header.get("speaker_embedding") if isinstance(header, dict) else None
    expected_bytes = 16 * 768 * 4
    if not isinstance(tensor, dict) or (
        tensor.get("dtype") != "F32"
        or tensor.get("shape") != [16, 768]
        or tensor.get("data_offsets") != [0, expected_bytes]
    ):
        raise ValueError(f"speaker embedding tensor contract mismatch: {source}")
    payload = data[8 + header_length :]
    if len(payload) != expected_bytes:
        raise ValueError(f"speaker embedding payload size mismatch: {source}")
    if not all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", payload)):
        raise ValueError(f"speaker embedding contains nonfinite values: {source}")


def _training_log_summary(*, kind: str, data: bytes) -> TrainingLogSummary:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("training log must be UTF-8") from exc
    if _OOM_RE.search(text):
        raise ValueError("training log contains OOM evidence")
    if re.search(r"traceback", text, re.IGNORECASE):
        raise ValueError("training log contains traceback evidence")
    matches = _LOSS_RE.findall(text)
    steps: list[int] = []
    losses: list[float] = []
    for raw_step, raw_loss in matches:
        try:
            steps.append(int(raw_step))
            losses.append(float(raw_loss))
        except ValueError as exc:
            raise ValueError("training log contains an invalid loss") from exc
    expected_steps = list(range(20, 250, 20)) if kind == "search" else list(range(20, 3001, 20))
    if steps != expected_steps:
        raise ValueError("training log loss steps are missing, duplicated, or extra")
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise ValueError("training log losses must all be finite")
    terminal = 250 if kind == "search" else 3000
    marker = re.compile(rf"Training finished at step\s*=\s*{terminal}\.", re.IGNORECASE)
    if not marker.search(text):
        raise ValueError("training log terminal marker mismatch")
    return TrainingLogSummary(steps=tuple(steps), last_loss=losses[-1])


def _validate_runtime_after(runtime: Mapping[str, object]) -> None:
    expected = {
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_utilization_percent",
        "gpu_power_watts",
        "active_training_processes",
    }
    if set(runtime) != expected:
        raise ValueError("runtime-after keys do not match the closure contract")
    values = []
    for field in (
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_utilization_percent",
        "gpu_power_watts",
    ):
        value = runtime.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"runtime-after {field} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"runtime-after {field} must be finite")
        values.append(number)
    used, total, utilization, power = values
    if (
        used < 0
        or total <= 0
        or used > total
        or not 0 <= utilization <= MAX_GPU_UTILIZATION_PERCENT
        or power < 0
    ):
        raise ValueError("runtime-after numeric closure is invalid")
    if runtime.get("active_training_processes") != []:
        raise ValueError("runtime-after active training processes must be empty")


def _write_atomic_create_only(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite output: {path}") from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except PermissionError:
            # Windows does not consistently allow opening directories for fsync.
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or finalize a create-only 600M speaker quality run."
    )
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--kind", choices=("search", "retrain"), required=True)
    prepare.add_argument("--predecessor-jobs", type=Path, required=True)
    prepare.add_argument("--predecessor-status", type=Path, required=True)
    prepare.add_argument("--source-diagnostic", type=Path, required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--init-checkpoint-step", type=int, required=True)
    prepare.add_argument("--learning-rate", type=float, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--queue-script", type=Path, required=True)
    prepare.add_argument("--strategy")

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--setup-evidence", type=Path, required=True)
    finalize.add_argument("--training-jobs", type=Path, required=True)
    finalize.add_argument("--training-status", type=Path, required=True)
    finalize.add_argument("--queue-script", type=Path, required=True)
    finalize.add_argument("--queue-exit-code", type=int, required=True)
    finalize.add_argument("--runtime-after", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "prepare":
        payload = prepare_quality_run(
            kind=args.kind,
            predecessor_jobs=args.predecessor_jobs,
            predecessor_status=args.predecessor_status,
            source_diagnostic=args.source_diagnostic,
            model_id=args.model_id,
            init_checkpoint_step=args.init_checkpoint_step,
            learning_rate=args.learning_rate,
            seed=args.seed,
            run_root=args.run_root,
            queue_script=args.queue_script,
            strategy=args.strategy,
        )
    else:
        payload = finalize_quality_run(
            setup_evidence=args.setup_evidence,
            training_jobs=args.training_jobs,
            training_status=args.training_status,
            queue_script=args.queue_script,
            queue_exit_code=args.queue_exit_code,
            runtime_after=args.runtime_after,
            output=args.output,
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _integer(value: object, source: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{source} must be an integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
