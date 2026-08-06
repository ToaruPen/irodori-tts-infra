# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR2004, TRY003, TRY004
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SEARCH_SCHEMA = "speaker-checkpoint-search-manifest/v1"
RUN_EVIDENCE_SCHEMA = "speaker-quality-search-run-evidence/v1"
SOURCE_SCHEMA = "speaker-checkpoint-evaluation-manifest/v1"
TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
SEEDS = (1234, 5678)
STYLES = ("neutral", "calm")
SOURCE_STEPS = tuple(range(250, 3001, 250))
SEARCH_STEP = 250
SEARCH_LOG_EVERY = 20
SEARCH_LOSS_STEPS = tuple(range(SEARCH_LOG_EVERY, SEARCH_STEP, SEARCH_LOG_EVERY))
CANONICAL_CHECKPOINT_NAME = "checkpoint_0000250.speaker.safetensors"
EMBEDDING_SHAPE = (16, 768)
SHA256_LENGTH = 64
LOSS_LINE_RE = re.compile(r"^step=(\d+) loss=([^\s]+)", re.MULTILINE)
OOM_RE = re.compile(r"\boom\b|cuda\s+out\s+of\s+memory", re.IGNORECASE)
FINISHED_MARKER = "Training finished at step=250."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_search_manifest(
    *,
    source_manifest: Path,
    source_manifest_sha256: str,
    embedding: Path,
    embedding_sha256: str,
    training_config: Path,
    training_config_sha256: str,
    training_run_evidence: Path,
    model_id: str,
    run_id: str,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite search manifest: {output}")
    if not run_id:
        raise ValueError("run_id must be nonempty")
    source_path = source_manifest.resolve()
    embedding_path = embedding.resolve()
    config_path = training_config.resolve()
    evidence_path = training_run_evidence.resolve()
    _validate_hash(source_path, source_manifest_sha256, source="source manifest")
    _validate_hash(embedding_path, embedding_sha256, source="embedding")
    _validate_hash(config_path, training_config_sha256, source="training config")
    source = _read_json(source_path)
    _validate_source_manifest(source, model_id=model_id, manifest_dir=source_path.parent)
    config = _read_json(config_path)
    _validate_search_training_config(config)
    if embedding_path.name != CANONICAL_CHECKPOINT_NAME:
        raise ValueError(f"search checkpoint must be named {CANONICAL_CHECKPOINT_NAME}")
    _validate_embedding(embedding_path)
    _validate_run_evidence(
        evidence_path,
        model_id=model_id,
        run_id=run_id,
        config_path=config_path,
        config_sha256=training_config_sha256,
        embedding_path=embedding_path,
        embedding_sha256=embedding_sha256,
        base_checkpoint_sha256=str(source["models"][0]["checkpoints"][0]["base_checkpoint_sha256"]),
    )
    source_model = source["models"][0]
    source_checkpoint = source_model["checkpoints"][0]
    metrics = source["metrics_provenance"]
    payload: dict[str, object] = {
        "schema_version": SEARCH_SCHEMA,
        "model_id": model_id,
        "run_id": run_id,
        "checkpoint": {
            "checkpoint_step": SEARCH_STEP,
            "embedding_path": str(embedding_path),
            "embedding_sha256": embedding_sha256,
            "training_config_path": str(config_path),
            "training_config_sha256": training_config_sha256,
            "base_checkpoint": source_checkpoint["base_checkpoint"],
            "base_checkpoint_sha256": source_checkpoint["base_checkpoint_sha256"],
            "base_revision": source_checkpoint["base_revision"],
            "run_id": run_id,
        },
        "text_ids": list(TEXT_IDS),
        "seeds": list(SEEDS),
        "styles": list(STYLES),
        "metrics_provenance": metrics,
        "source_evaluation_manifest": {
            "path": str(source_path),
            "sha256": source_manifest_sha256,
        },
        "training_run_evidence": {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
        },
        "provenance": {
            "builder_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            }
        },
    }
    _validate_search_source_binding(payload, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
    return payload


def _validate_search_training_config(config: Mapping[str, object]) -> None:
    train = config.get("train")
    if not isinstance(train, dict):
        raise TypeError("training config train must be an object")
    for field in ("max_steps", "save_every"):
        value = train.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != SEARCH_STEP:
            raise ValueError(f"training config train.{field} must be integer 250")
    log_every = train.get("log_every")
    if (
        isinstance(log_every, bool)
        or not isinstance(log_every, int)
        or log_every != SEARCH_LOG_EVERY
    ):
        raise ValueError("training config train.log_every must be integer 20")


def _validate_command_path_argument(
    command: list[str],
    *,
    flag: str,
    expected: Path,
    base: Path,
    required: bool,
) -> None:
    count = command.count(flag)
    invalid_count = count != 1 if required else count > 1
    if invalid_count:
        qualifier = "exactly one" if required else "at most one"
        raise ValueError(f"search training job command must contain {qualifier} {flag} argument")
    if count == 0:
        return
    value_index = command.index(flag) + 1
    if value_index >= len(command):
        raise ValueError(f"search training job command is missing the {flag} value")
    if _resolve_path(command[value_index], base=base) != expected:
        raise ValueError(f"search training job command {flag} path mismatch")


def _validate_target_job_command(
    job: Mapping[str, object],
    *,
    config_path: Path,
    clean_manifest: Path,
    output_dir: Path,
) -> None:
    command = job.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) and part for part in command):
        raise ValueError("search training job command must be a list of nonempty strings")
    for flag, expected, required in (
        ("--config", config_path, True),
        ("--manifest", clean_manifest, False),
        ("--output-dir", output_dir, False),
    ):
        _validate_command_path_argument(
            command,
            flag=flag,
            expected=expected,
            base=config_path.parent,
            required=required,
        )


def _validate_training_log(
    raw_log: Mapping[str, object],
    *,
    log_path: Path,
) -> None:
    _require_exact_keys(
        raw_log,
        {
            "path",
            "sha256",
            "loss_event_count",
            "loss_steps",
            "loss_all_finite",
            "last_loss",
            "oom",
            "traceback",
        },
        source="search run evidence log",
    )
    event_count = _nonbool_int(
        raw_log.get("loss_event_count"),
        source="search run evidence log loss_event_count",
    )
    if event_count <= 0 or event_count != len(SEARCH_LOSS_STEPS):
        raise ValueError("search run evidence log loss_event_count must be positive and equal 12")
    if raw_log.get("loss_steps") != list(SEARCH_LOSS_STEPS):
        raise ValueError(
            "search run evidence log loss_steps must exactly match steps 20 through 240"
        )
    declared_last = raw_log.get("last_loss")
    if isinstance(declared_last, bool) or not isinstance(declared_last, int | float):
        raise TypeError("search run evidence log last_loss must be numeric")
    declared_last_float = float(declared_last)
    if not math.isfinite(declared_last_float):
        raise ValueError("search run evidence log last_loss must be finite")
    if (
        raw_log.get("loss_all_finite") is not True
        or raw_log.get("oom") is not False
        or raw_log.get("traceback") is not False
    ):
        raise ValueError("search run evidence log metadata reports failed quality checks")

    log_text = log_path.read_text(encoding="utf-8")
    matches = LOSS_LINE_RE.findall(log_text)
    try:
        steps = tuple(int(step) for step, _loss in matches)
        losses = tuple(float(loss) for _step, loss in matches)
    except ValueError as exc:
        raise ValueError("search training log contains an invalid loss event") from exc
    if steps != SEARCH_LOSS_STEPS or len(matches) != event_count:
        raise ValueError("search training log loss steps/count mismatch")
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise ValueError("search training log contains nonfinite loss")
    if not math.isclose(declared_last_float, losses[-1], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("search training log last_loss metadata mismatch")
    if OOM_RE.search(log_text):
        raise ValueError("search training log contains OOM evidence")
    if "Traceback" in log_text:
        raise ValueError("search training log contains Traceback evidence")
    if FINISHED_MARKER not in log_text:
        raise ValueError("search training log is missing the step 250 finish marker")


def _validate_run_evidence(
    path: Path,
    *,
    model_id: str,
    run_id: str,
    config_path: Path,
    config_sha256: str,
    embedding_path: Path,
    embedding_sha256: str,
    base_checkpoint_sha256: str,
) -> None:
    evidence = _read_json(path)
    _require_exact_keys(
        evidence,
        {
            "schema_version",
            "created_at",
            "state",
            "model_id",
            "queue_exit_code",
            "setup_evidence",
            "training_jobs",
            "training_status",
            "queue_script",
            "invocation",
            "run",
            "runtime_after",
        },
        source="search run evidence",
    )
    if (
        evidence.get("schema_version") != RUN_EVIDENCE_SCHEMA
        or evidence.get("state") != "finished"
        or evidence.get("model_id") != model_id
        or not isinstance(evidence.get("created_at"), str)
        or not evidence.get("created_at")
        or type(evidence.get("queue_exit_code")) is not int
        or evidence.get("queue_exit_code") != 0
    ):
        raise ValueError("search run evidence identity or state mismatch")
    run_root = path.parent
    if run_root.name != run_id or not config_path.is_relative_to(run_root):
        raise ValueError("search run evidence path/config does not match run_id root")
    for field in ("setup_evidence", "queue_script"):
        _validate_binding(
            evidence.get(field),
            base=run_root,
            source=f"search run evidence {field}",
        )
    jobs_path = _validate_binding(
        evidence.get("training_jobs"),
        base=run_root,
        source="search run evidence training_jobs",
    )
    jobs = _read_json(jobs_path)
    raw_jobs = jobs.get("jobs")
    if (
        not isinstance(raw_jobs, list)
        or len(raw_jobs) != 12
        or not all(isinstance(job, dict) for job in raw_jobs)
    ):
        raise ValueError("search training jobs must contain the actual 12-job queue")
    model_ids = [job.get("model_id") for job in raw_jobs if isinstance(job, dict)]
    if (
        not all(isinstance(candidate, str) and candidate for candidate in model_ids)
        or len(set(model_ids)) != len(model_ids)
        or model_ids.count(model_id) != 1
    ):
        raise ValueError("search training jobs contain duplicate or missing model_id")
    job = next(job for job in raw_jobs if isinstance(job, dict) and job.get("model_id") == model_id)
    job_config = _resolve_path(job.get("config"), base=jobs_path.parent)
    clean_manifest = _resolve_path(job.get("clean_manifest"), base=jobs_path.parent)
    output_dir = _resolve_path(job.get("output_dir"), base=jobs_path.parent)
    if (
        job_config != config_path
        or not output_dir.is_relative_to(run_root)
        or not embedding_path.is_relative_to(output_dir)
    ):
        raise ValueError("search training job does not match model/config/checkpoint path")
    config = _read_json(config_path)
    _validate_search_training_config(config)
    train = config.get("train")
    if not isinstance(train, dict):
        raise TypeError("training config train must be an object")
    configured_manifest = _resolve_path(train.get("manifest_path"), base=config_path.parent)
    configured_output = _resolve_path(train.get("output_dir"), base=config_path.parent)
    if configured_manifest != clean_manifest:
        raise ValueError("training config manifest_path does not match target job clean_manifest")
    if configured_output != output_dir:
        raise ValueError("training config output_dir does not match target job output_dir")
    _validate_target_job_command(
        job,
        config_path=config_path,
        clean_manifest=clean_manifest,
        output_dir=output_dir,
    )

    raw_status = evidence.get("training_status")
    if not isinstance(raw_status, dict):
        raise TypeError("search run evidence training_status must be an object")
    _require_exact_keys(
        raw_status,
        {
            "path",
            "before_row_count",
            "before_sha256",
            "after_row_count",
            "after_sha256",
            "new_status_row_count",
            "new_started_model_ids",
            "new_finished_success_model_ids",
        },
        source="search run evidence training_status",
    )
    status_path = _resolve_path(raw_status.get("path"), base=run_root)
    status_pairs = _read_jsonl_raw_lines(status_path)
    status_lines = [raw for raw, _row in status_pairs]
    status_rows = [row for _raw, row in status_pairs]
    before_count = _nonbool_int(raw_status.get("before_row_count"), source="before_row_count")
    after_count = _nonbool_int(raw_status.get("after_row_count"), source="after_row_count")
    before_sha = _require_sha(raw_status.get("before_sha256"), source="before_sha256")
    after_sha = _require_sha(raw_status.get("after_sha256"), source="after_sha256")
    if (
        before_count < 0
        or after_count != before_count + 2
        or after_count != len(status_rows)
        or hashlib.sha256(b"".join(status_lines[:before_count])).hexdigest() != before_sha
        or sha256_file(status_path) != after_sha
        or raw_status.get("new_status_row_count") != 2
        or raw_status.get("new_started_model_ids") != [model_id]
        or raw_status.get("new_finished_success_model_ids") != [model_id]
    ):
        raise ValueError("search training status append-only chain mismatch")
    started, status = status_rows[before_count:after_count]
    if not (
        started.get("model_id") == model_id
        and started.get("event") == "started"
        and started.get("status") == "running"
        and started.get("exit_code") is None
        and started.get("candidate_checkpoints") == []
        and status.get("model_id") == model_id
        and status.get("event") == "finished"
        and status.get("status") == "success"
        and type(status.get("exit_code")) is int
        and status.get("exit_code") == 0
        and status.get("error") is None
        and status.get("started_at") == started.get("started_at")
    ):
        raise ValueError("search training status appended rows must be started then success")
    model_status_rows = [row for row in status_rows if row.get("model_id") == model_id]
    if not model_status_rows or model_status_rows[-1] != status:
        raise ValueError("search training status latest model row must be finished success")

    clean_manifest_sha256 = sha256_file(clean_manifest)
    for row in (started, status):
        if (
            row.get("config_sha256") != config_sha256
            or row.get("clean_manifest_sha256") != clean_manifest_sha256
            or row.get("checkpoint_sha256") != base_checkpoint_sha256
        ):
            raise ValueError("search training status provenance mismatch")
    if (
        _resolve_path(status.get("last_checkpoint"), base=status_path.parent) != embedding_path
        or status.get("last_checkpoint_sha256") != embedding_sha256
    ):
        raise ValueError("search training status last checkpoint mismatch")
    candidates = status.get("candidate_checkpoints")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 2
        or not all(isinstance(candidate, dict) for candidate in candidates)
    ):
        raise ValueError("search training status checkpoint binding mismatch")
    status_bindings = {
        (
            _resolve_path(candidate.get("path"), base=status_path.parent),
            _require_sha(candidate.get("sha256"), source="search status checkpoint"),
        )
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    for checkpoint_path, checkpoint_sha in status_bindings:
        _validate_hash(checkpoint_path, checkpoint_sha, source="search status checkpoint")
        if not checkpoint_path.is_relative_to(output_dir):
            raise ValueError("search status checkpoint escapes job output_dir")
        _validate_embedding(checkpoint_path)
    if len(status_bindings) != 2:
        raise ValueError("search training status checkpoint list contains duplicates")

    run = evidence.get("run")
    if not isinstance(run, dict):
        raise TypeError("search run evidence run must be an object")
    _require_exact_keys(
        run,
        {
            "started_at",
            "ended_at",
            "config_path",
            "config_sha256",
            "clean_manifest_sha256",
            "base_checkpoint_sha256",
            "candidate_checkpoint_count",
            "checkpoints",
            "final_equals_step250",
            "log",
        },
        source="search run evidence run",
    )
    checkpoints = run.get("checkpoints")
    if (
        _resolve_path(run.get("config_path"), base=run_root) != config_path
        or run.get("config_sha256") != config_sha256
        or run.get("clean_manifest_sha256") != clean_manifest_sha256
        or run.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or run.get("candidate_checkpoint_count") != 2
        or run.get("final_equals_step250") is not True
        or run.get("started_at") != started.get("started_at")
        or run.get("ended_at") != status.get("ended_at")
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 2
        or not all(isinstance(checkpoint, dict) for checkpoint in checkpoints)
    ):
        raise ValueError("search run evidence must prove final_equals_step250 at step 250")
    run_bindings: set[tuple[Path, str]] = set()
    names: set[str] = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        _require_exact_keys(
            checkpoint,
            {"name", "path", "sha256"},
            source="search run evidence checkpoint",
        )
        checkpoint_path = _resolve_path(checkpoint.get("path"), base=path.parent)
        checkpoint_sha = _require_sha(
            checkpoint.get("sha256"), source="search run evidence checkpoint"
        )
        if checkpoint.get("name") != checkpoint_path.name:
            raise ValueError("search run evidence checkpoint name/path mismatch")
        _validate_hash(checkpoint_path, checkpoint_sha, source="search run evidence checkpoint")
        if not checkpoint_path.is_relative_to(output_dir):
            raise ValueError("search run evidence checkpoint escapes job output_dir")
        _validate_embedding(checkpoint_path)
        names.add(str(checkpoint.get("name")))
        run_bindings.add((checkpoint_path, checkpoint_sha))
    expected_names = {CANONICAL_CHECKPOINT_NAME, "checkpoint_final.speaker.safetensors"}
    by_name = {
        str(checkpoint.get("name")): (
            _resolve_path(checkpoint.get("path"), base=run_root),
            _require_sha(checkpoint.get("sha256"), source="search run evidence checkpoint"),
        )
        for checkpoint in checkpoints
        if isinstance(checkpoint, dict)
    }
    if (
        names != expected_names
        or len(run_bindings) != 2
        or run_bindings != status_bindings
        or by_name[CANONICAL_CHECKPOINT_NAME] != (embedding_path, embedding_sha256)
        or by_name["checkpoint_final.speaker.safetensors"][1] != embedding_sha256
    ):
        raise ValueError("search run evidence checkpoint binding mismatch")
    raw_log = run.get("log")
    if not isinstance(raw_log, dict):
        raise TypeError("search run evidence log must be an object")
    log_path = _resolve_path(raw_log.get("path"), base=run_root)
    log_sha = _require_sha(raw_log.get("sha256"), source="search run evidence log")
    _validate_hash(log_path, log_sha, source="search run evidence log")
    if _resolve_path(status.get("log_path"), base=status_path.parent) != log_path:
        raise ValueError("search run evidence log binding mismatch")
    _validate_training_log(raw_log, log_path=log_path)
    if not isinstance(evidence.get("invocation"), dict) or not isinstance(
        evidence.get("runtime_after"), dict
    ):
        raise TypeError("search run evidence invocation/runtime_after must be objects")


def _validate_binding(
    raw: object,
    *,
    base: Path,
    source: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> Path:
    if not isinstance(raw, dict):
        raise TypeError(f"{source} must be an object")
    _require_exact_keys(raw, {"path", "sha256"}, source=source)
    path = _resolve_path(raw.get("path"), base=base)
    sha256 = _require_sha(raw.get("sha256"), source=source)
    _validate_hash(path, sha256, source=source)
    if expected_path is not None and path != expected_path:
        raise ValueError(f"{source} path mismatch")
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f"{source} SHA-256 mismatch")
    return path


def _require_exact_keys(
    row: Mapping[str, object],
    expected: set[str],
    *,
    source: str,
) -> None:
    if set(row) != expected:
        raise ValueError(f"{source} keys must exactly match {sorted(expected)}")


def _resolve_path(value: object, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("bound path must be nonempty")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _read_jsonl_raw_lines(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    rows = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append((raw_line, row))
    return rows


def _nonbool_int(value: object, *, source: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{source} must be an integer")
    return value


def _validate_source_manifest(
    payload: Mapping[str, Any],
    *,
    model_id: str,
    manifest_dir: Path,
) -> None:
    if payload.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"source manifest schema_version must be {SOURCE_SCHEMA}")
    for field, expected in (("text_ids", TEXT_IDS), ("seeds", SEEDS), ("styles", STYLES)):
        if tuple(payload.get(field, ())) != expected:
            raise ValueError(f"source manifest {field} must exactly match {expected}")
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError("source manifest must contain exactly one model")
    if models[0].get("model_id") != model_id:
        raise ValueError("source manifest model_id does not match requested model_id")
    checkpoints = models[0].get("checkpoints")
    if (
        not isinstance(checkpoints, list)
        or tuple(
            row.get("checkpoint_step") if isinstance(row, dict) else None for row in checkpoints
        )
        != SOURCE_STEPS
    ):
        raise ValueError(f"source manifest checkpoint steps must exactly match {SOURCE_STEPS}")
    base_contracts: set[tuple[object, object, object]] = set()
    for row in checkpoints:
        if not isinstance(row, dict):
            raise TypeError("source checkpoint must be an object")
        for field in (
            "embedding_sha256",
            "training_config_sha256",
            "base_checkpoint_sha256",
        ):
            _require_sha(row.get(field), source=f"source checkpoint {field}")
        for field in ("base_checkpoint", "base_revision", "run_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"source checkpoint {field} must be nonempty")
        raw_embedding_path = row.get("embedding_path")
        if not isinstance(raw_embedding_path, str) or not raw_embedding_path:
            raise ValueError("source checkpoint embedding_path must be nonempty")
        source_embedding = Path(raw_embedding_path)
        if not source_embedding.is_absolute():
            source_embedding = manifest_dir / source_embedding
        expected_embedding_sha = str(row["embedding_sha256"])
        if (
            not source_embedding.is_file()
            or sha256_file(source_embedding) != expected_embedding_sha
        ):
            raise ValueError(f"source checkpoint embedding SHA-256 mismatch: {source_embedding}")
        _validate_embedding(source_embedding)
        base_contracts.add(
            (
                row["base_checkpoint"],
                row["base_checkpoint_sha256"],
                row["base_revision"],
            )
        )
    if len(base_contracts) != 1:
        raise ValueError("source checkpoints must share one base checkpoint contract")
    metrics = payload.get("metrics_provenance")
    if not isinstance(metrics, dict):
        raise TypeError("source manifest metrics_provenance must be an object")
    _require_sha(metrics.get("reference_wavs_sha256"), source="reference_wavs_sha256")
    for name in ("speaker_embedding", "transcription"):
        model = metrics.get(name)
        if not isinstance(model, dict):
            raise TypeError(f"source metric model {name} must be an object")
        for field in ("model_id", "revision"):
            if not isinstance(model.get(field), str) or not model[field]:
                raise ValueError(f"source metric model {name} {field} must be nonempty")
        _require_sha(model.get("source_sha256"), source=f"source metric model {name}")


def _validate_search_source_binding(
    search: Mapping[str, object],
    source: Mapping[str, Any],
) -> None:
    models = source.get("models")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise ValueError("source evaluation manifest model contract is missing")
    checkpoints = models[0].get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints or not isinstance(checkpoints[0], dict):
        raise ValueError("source evaluation manifest checkpoint contract is missing")
    search_checkpoint = search.get("checkpoint")
    if not isinstance(search_checkpoint, dict):
        raise TypeError("search checkpoint must be an object")
    source_checkpoint = checkpoints[0]
    base_fields = ("base_checkpoint", "base_checkpoint_sha256", "base_revision")
    if any(search_checkpoint.get(field) != source_checkpoint.get(field) for field in base_fields):
        raise ValueError("search base checkpoint contract does not match source; source drift")
    if search.get("metrics_provenance") != source.get("metrics_provenance"):
        raise ValueError("search metrics provenance does not match source; source drift")


def _validate_hash(path: Path, expected: str, *, source: str) -> None:
    _require_sha(expected, source=source)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{source} SHA-256 mismatch: {path}")


def _validate_embedding(path: Path) -> None:
    try:
        with path.open("rb") as source:
            raw_length = source.read(8)
            if len(raw_length) != 8:
                raise ValueError("embedding safetensors header is truncated")
            header_length = struct.unpack("<Q", raw_length)[0]
            header = json.loads(source.read(header_length))
            tensor = header.get("speaker_embedding") if isinstance(header, dict) else None
            if not isinstance(tensor, dict):
                raise ValueError("speaker_embedding tensor is missing")
            if tensor.get("dtype") != "F32":
                raise ValueError("speaker_embedding must be F32")
            if tensor.get("shape") != list(EMBEDDING_SHAPE):
                raise ValueError(f"speaker_embedding shape must be {EMBEDDING_SHAPE}")
            offsets = tensor.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ValueError("speaker_embedding offsets are invalid")
            start, end = offsets
            source.seek(8 + header_length + start)
            values = np.frombuffer(source.read(end - start), dtype="<f4")
            if values.size != math.prod(EMBEDDING_SHAPE):
                raise ValueError("speaker_embedding payload size is invalid")
            if not np.isfinite(values).all():
                raise ValueError("speaker_embedding must contain only finite values")
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        raise ValueError(f"invalid embedding: {path}") from exc


def _require_sha(value: object, *, source: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{source} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--embedding-sha256", required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--training-config-sha256", required=True)
    parser.add_argument("--training-run-evidence", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_search_manifest(
        source_manifest=args.source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        embedding=args.embedding,
        embedding_sha256=args.embedding_sha256,
        training_config=args.training_config,
        training_config_sha256=args.training_config_sha256,
        training_run_evidence=args.training_run_evidence,
        model_id=args.model_id,
        run_id=args.run_id,
        output=args.output,
    )
    print(json.dumps({"manifest": str(args.output.resolve()), "model_id": payload["model_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
