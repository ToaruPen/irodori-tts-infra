# ruff: noqa: EM101, EM102, TRY003 - operational errors retain exact artifact context.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class QueueResult(Protocol):
    succeeded: tuple[str, ...]
    skipped: tuple[str, ...]
    reused: tuple[str, ...]
    failed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingJobBinding:
    model_id: str
    clean_manifest: Path
    config: Path


@dataclass(frozen=True, slots=True)
class TrainingContract:
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_revision: str
    upstream_commit: str
    jobs: tuple[TrainingJobBinding, ...]


@dataclass(frozen=True, slots=True)
class VerifiedSource:
    path: Path
    content: bytes
    sha256: str
    snapshot_relative: Path | None


@dataclass(frozen=True, slots=True)
class VerifiedContext:
    report: dict[str, object]
    config_document: dict[str, Any]
    queue_module: ModuleType
    queue_config: Any
    sources: tuple[VerifiedSource, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    root: Path
    scripts_dir: Path
    config_path: Path
    jobs_path: Path
    training_status: Path
    files: Mapping[Path, bytes]
    config_sha256: str


EXPECTED_MODEL_COUNT = 12
ANABEL_MODEL_ID = "oop77_anabel_maidgarden_sp_451488a7c1"
REMOTE_ROOT = Path(r"C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731")
DEFAULT_OUTPUT_ROOT = REMOTE_ROOT / "evaluation_speed_v4"
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-queue-speed-v4.json"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v4.jsonl"
DEFAULT_JOBS_PATH = REMOTE_ROOT / "training" / "training-jobs-speed-v1.json"
DEFAULT_SCRIPTS_DIR = REMOTE_ROOT / "scripts"
DEFAULT_SOURCE_CONFIG_PATH = REMOTE_ROOT / "status" / "evaluation-queue-official-v1.json"
RUNTIME_SNAPSHOT_NAME = "runtime-inputs-v1"
RUNTIME_CONFIG_NAME = "evaluation-queue-runtime.json"
RUNTIME_JOBS_NAME = "training-jobs-speed-v1.json"
RUNTIME_STATUS_NAME = "training-status.jsonl"
RUNTIME_MANIFEST_NAME = "snapshot-manifest.json"
RUNTIME_SNAPSHOT_SCHEMA = "speaker-evaluation-runtime-inputs/v1"

# These pins bind the launcher to the reviewed remote artifacts. Update only by
# preparing a new version after reviewing the corresponding content.
PENDING_CONFIG_SHA256 = "PENDING_REMOTE_PREPARE_CONFIG_SHA256"
EXPECTED_CONFIG_SHA256 = "472fb1ec315def4423a0cc60ad9169a6c4395a6b587a21ea128b1ef9e332b5ce"
EXPECTED_SOURCE_CONFIG_SHA256 = "33109e7ea9b62b014d59ce0673ea3d4e50c45f9e60cebf13e06463e2e5e4fd02"
EXPECTED_JOBS_SHA256 = "206f8fe9d1428a5aa9426c215ee5f092e4546e44fd4bc094e92478180ef163c6"
EXPECTED_COMPONENT_SHA256 = {
    "run_600m_speaker_evaluation_queue.py": (
        "60e2456a5d51e6a4b935a64fdbd0140647f187d1599350039f51f77b7dbfef70"
    ),
    "build_600m_checkpoint_evaluation_manifests.py": (
        "e3f62e07f07c949fe60d4db00a7eef11dbd1ae9111a7628dd88982a2702d0e93"
    ),
    "generate_600m_checkpoint_audio_remote.py": (
        "9688f956fb2be1f148f4583a77f41f5774e033a93f6d25dc0db835412b099312"
    ),
    "analyze_nko_beep_matrix.py": (
        "06c23b489975843bf080b7fa70ebb41abac19c269b6cfe5bcafec019a0693dc1"
    ),
    "compute_600m_speaker_metrics.py": (
        "fa83491f0ee2f1e1f21c8d833ba90557ed67c885a2512c9c05104faf3b14a407"
    ),
    "evaluate_600m_speaker_checkpoints.py": (
        "cb28b8541956fcfdb549e1ee6e87905176ab4acf8b96809fd6da60655a10c738"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bytes(path.read_bytes(), source=path)


def _read_json_bytes(content: bytes, *, source: Path) -> dict[str, Any]:
    document = json.loads(content.decode("utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"JSON document must be an object: {source}")
    return document


def _verified_source(
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
    snapshot_relative: Path | None,
) -> VerifiedSource:
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha256}, actual={actual}, path={path}"
        )
    return VerifiedSource(
        path=path.resolve(),
        content=content,
        sha256=actual,
        snapshot_relative=snapshot_relative,
    )


def _require_sha256(path: Path, *, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, path={path}"
        )
    return actual


def _load_evaluation_module(
    path: Path,
    *,
    source_bytes: bytes | None = None,
) -> ModuleType:
    verified_source = path.read_bytes() if source_bytes is None else source_bytes
    source_sha256 = hashlib.sha256(verified_source).hexdigest()
    module_name = f"_speaker_evaluation_queue_speed_v4_{source_sha256[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    code = compile(verified_source, str(path), "exec")
    exec(code, module.__dict__)  # noqa: S102 - executes reviewed, SHA-pinned operator source.
    return module


def _required_string(row: Mapping[str, object], key: str, *, source: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} requires nonempty string {key}")
    return value


def _require_under(path: Path, *, root: Path, label: str) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} is outside speed-v4 output root: {resolved}")


def _validate_operational_paths(
    *,
    config_path: Path,
    status_path: Path,
    output_root: Path,
) -> None:
    expected = (
        (output_root, DEFAULT_OUTPUT_ROOT, "output root"),
        (config_path, DEFAULT_CONFIG_PATH, "config"),
        (status_path, DEFAULT_STATUS_PATH, "status"),
    )
    for actual, fixed, label in expected:
        if actual.resolve() != fixed.resolve():
            raise ValueError(
                f"operational {label} must equal fixed speed-v4 {label}: "
                f"expected={fixed.resolve()}, actual={actual.resolve()}"
            )


def _validate_config_roots(
    document: Mapping[str, object],
    *,
    config_path: Path,
    status_path: Path,
    output_root: Path,
    expected_jobs_path: Path,
) -> None:
    if document.get("schema_version") != "speaker-evaluation-queue/v1":
        raise ValueError("speed-v4 config must retain speaker-evaluation-queue/v1 schema")
    configured_jobs = Path(
        _required_string(document, "training_jobs", source="queue config")
    ).resolve()
    if configured_jobs != expected_jobs_path.resolve():
        raise ValueError(f"queue config is not bound to speed-v1 jobs: {configured_jobs}")
    _require_under(config_path, root=output_root, label="queue config")
    _require_under(status_path, root=output_root, label="evaluation status")
    _require_under(
        Path(_required_string(document, "manifest_output_dir", source="queue config")),
        root=output_root,
        label="manifest output",
    )
    metric_models = document.get("metric_models")
    if not isinstance(metric_models, dict):
        raise TypeError("queue config metric_models must be an object")
    speaker_embedding = metric_models.get("speaker_embedding")
    if not isinstance(speaker_embedding, dict):
        raise TypeError("queue config speaker_embedding must be an object")
    _require_under(
        Path(
            _required_string(
                speaker_embedding,
                "savedir",
                source="speaker embedding config",
            )
        ),
        root=output_root,
        label="speaker embedding cache",
    )


def _training_job_ids(path: Path) -> list[str]:
    return _training_job_ids_document(_read_json(path))


def _training_job_ids_document(document: Mapping[str, object]) -> list[str]:
    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"speed-v1 jobs must contain exactly {EXPECTED_MODEL_COUNT} jobs")
    job_ids: list[str] = []
    for job in raw_jobs:
        if not isinstance(job, dict):
            raise TypeError("training jobs must be objects")
        job_ids.append(_required_string(job, "model_id", source="training job"))
    return job_ids


def _resolve_input(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_training_contract(
    path: Path,
    *,
    document: Mapping[str, object] | None = None,
) -> TrainingContract:
    jobs_document = _read_json(path) if document is None else document
    raw_jobs = jobs_document.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"speed-v1 jobs must contain exactly {EXPECTED_MODEL_COUNT} jobs")
    jobs: list[TrainingJobBinding] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise TypeError("training jobs must be objects")
        model_id = _required_string(raw_job, "model_id", source="training job")
        jobs.append(
            TrainingJobBinding(
                model_id=model_id,
                clean_manifest=_resolve_input(
                    path.parent,
                    _required_string(raw_job, "clean_manifest", source=model_id),
                ),
                config=_resolve_input(
                    path.parent,
                    _required_string(raw_job, "config", source=model_id),
                ),
            )
        )
    model_ids = [job.model_id for job in jobs]
    if len(set(model_ids)) != EXPECTED_MODEL_COUNT:
        raise ValueError("speed-v1 jobs contain duplicate model ids")
    return TrainingContract(
        checkpoint=_resolve_input(
            path.parent,
            _required_string(jobs_document, "base_checkpoint_path", source="training jobs"),
        ),
        checkpoint_sha256=_required_string(
            jobs_document,
            "base_checkpoint_sha256",
            source="training jobs",
        ),
        checkpoint_revision=_required_string(
            jobs_document,
            "checkpoint_revision",
            source="training jobs",
        ),
        upstream_commit=_required_string(
            jobs_document,
            "upstream_commit",
            source="training jobs",
        ),
        jobs=tuple(jobs),
    )


def _latest_training_status(
    path: Path,
    *,
    model_ids: set[str],
) -> dict[str, Mapping[str, object]]:
    return _latest_training_status_bytes(
        path.read_bytes(),
        source=path,
        model_ids=model_ids,
    )


def _latest_training_status_bytes(
    content: bytes,
    *,
    source: Path,
    model_ids: set[str],
) -> dict[str, Mapping[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid training status JSON on line {line_number}: {source}"
            ) from exc
        if not isinstance(row, dict):
            raise TypeError(f"training status line {line_number} must be an object")
        model_id = row.get("model_id")
        if isinstance(model_id, str) and model_id in model_ids:
            latest[model_id] = row
    return latest


def _is_successful_finished_status(row: Mapping[str, object]) -> bool:
    if row.get("event") != "finished" or row.get("status") != "success":
        return False
    exit_code = row.get("exit_code")
    return exit_code == 0 and not isinstance(exit_code, bool)


def _validate_latest_training_success(
    *,
    config_document: Mapping[str, object],
    config_path: Path,
    jobs_path: Path,
    jobs_document: Mapping[str, object] | None = None,
    status_content: bytes | None = None,
) -> Path:
    contract = _load_training_contract(jobs_path, document=jobs_document)
    raw_status_path = _required_string(
        config_document,
        "training_status",
        source="queue config",
    )
    status_path = _resolve_input(config_path.parent, raw_status_path)
    model_ids = {job.model_id for job in contract.jobs}
    latest = (
        _latest_training_status(status_path, model_ids=model_ids)
        if status_content is None
        else _latest_training_status_bytes(
            status_content,
            source=status_path,
            model_ids=model_ids,
        )
    )
    for job in contract.jobs:
        row = latest.get(job.model_id)
        if row is None or not _is_successful_finished_status(row):
            raise ValueError(
                f"latest training status is not finished/success/exit_code=0 for {job.model_id}"
            )
        expected = {
            "clean_manifest_sha256": sha256_file(job.clean_manifest),
            "config_sha256": sha256_file(job.config),
            "checkpoint_sha256": contract.checkpoint_sha256,
            "checkpoint_revision": contract.checkpoint_revision,
            "upstream_commit": contract.upstream_commit,
        }
        mismatches = [field for field, value in expected.items() if row.get(field) != value]
        if mismatches:
            raise ValueError(
                f"latest training status provenance mismatch for {job.model_id}: "
                + ", ".join(mismatches)
            )
    actual_checkpoint_sha256 = sha256_file(contract.checkpoint)
    if actual_checkpoint_sha256 != contract.checkpoint_sha256:
        raise ValueError(
            "training jobs base checkpoint SHA-256 mismatch: "
            f"expected={contract.checkpoint_sha256}, actual={actual_checkpoint_sha256}"
        )
    return status_path


def _validate_models(
    document: Mapping[str, object],
    *,
    output_root: Path,
) -> tuple[list[str], list[str]]:
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"speed-v4 config must contain exactly {EXPECTED_MODEL_COUNT} models")
    model_ids: list[str] = []
    reused_model_ids: list[str] = []
    queue_owned_outputs: set[Path] = set()
    output_fields = ("generation_dir", "analysis_dir", "metrics_dir", "evaluation_dir")
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise TypeError("evaluation models must be objects")
        model_id = _required_string(raw_model, "model_id", source="evaluation model")
        model_ids.append(model_id)
        reuse = raw_model.get("reuse")
        if reuse is not None:
            if model_id != ANABEL_MODEL_ID:
                raise ValueError(f"only {ANABEL_MODEL_ID} may reuse canonical evaluation")
            if not isinstance(reuse, dict):
                raise TypeError(f"reuse for {model_id} must be an object")
            if any(field in raw_model for field in output_fields):
                raise ValueError(f"reused model {model_id} cannot own speed-v4 outputs")
            reused_model_ids.append(model_id)
            continue
        for field in output_fields:
            output = Path(_required_string(raw_model, field, source=model_id)).resolve()
            _require_under(output, root=output_root, label=f"{model_id} {field}")
            if output in queue_owned_outputs:
                raise ValueError(f"duplicate speed-v4 output path: {output}")
            queue_owned_outputs.add(output)
    return model_ids, reused_model_ids


def _validate_isolated_config(
    document: Mapping[str, object],
    *,
    config_path: Path,
    status_path: Path,
    output_root: Path,
    expected_jobs_path: Path,
    verified_job_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    _validate_config_roots(
        document,
        config_path=config_path,
        status_path=status_path,
        output_root=output_root,
        expected_jobs_path=expected_jobs_path,
    )
    job_ids = (
        _training_job_ids(expected_jobs_path) if verified_job_ids is None else verified_job_ids
    )
    model_ids, reused_model_ids = _validate_models(document, output_root=output_root)
    if model_ids != job_ids:
        raise ValueError("speed-v1 job order and speed-v4 evaluation order must match exactly")
    if reused_model_ids != [ANABEL_MODEL_ID]:
        raise ValueError(f"speed-v4 must reuse only the canonical {ANABEL_MODEL_ID} result")
    return model_ids, reused_model_ids


def prepare_speed_v4_config(
    *,
    source_path: Path,
    destination: Path,
    status_path: Path,
    output_root: Path,
    expected_source_sha256: str,
    jobs_path: Path,
    expected_jobs_sha256: str,
) -> dict[str, object]:
    source_sha256 = _require_sha256(
        source_path,
        expected=expected_source_sha256,
        label="source config",
    )
    jobs_sha256 = _require_sha256(
        jobs_path,
        expected=expected_jobs_sha256,
        label="training jobs",
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite speed-v4 config: {destination}")
    document = copy.deepcopy(_read_json(source_path))
    document["training_jobs"] = str(jobs_path.resolve())
    document["manifest_output_dir"] = str((output_root / "manifests").resolve())
    metric_models = document.get("metric_models")
    if not isinstance(metric_models, dict):
        raise TypeError("source config metric_models must be an object")
    speaker_embedding = metric_models.get("speaker_embedding")
    if not isinstance(speaker_embedding, dict):
        raise TypeError("source config speaker_embedding must be an object")
    speaker_embedding["savedir"] = str((output_root / "runtime-cache" / "ecapa").resolve())
    raw_models = document.get("models")
    if not isinstance(raw_models, list):
        raise TypeError("source config models must be a list")
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise TypeError("source evaluation models must be objects")
        if "reuse" in raw_model:
            continue
        model_id = _required_string(raw_model, "model_id", source="evaluation model")
        model_root = (output_root / "models" / model_id).resolve()
        raw_model["generation_dir"] = str(model_root / "generation")
        raw_model["analysis_dir"] = str(model_root / "analysis")
        raw_model["metrics_dir"] = str(model_root / "metrics")
        raw_model["evaluation_dir"] = str(model_root / "selection")

    model_ids, reused_model_ids = _validate_isolated_config(
        document,
        config_path=destination,
        status_path=status_path,
        output_root=output_root,
        expected_jobs_path=jobs_path,
    )
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary config: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "prepared": True,
        "launch_performed": False,
        "model_count": len(model_ids),
        "model_ids": model_ids,
        "reused_model_ids": reused_model_ids,
        "source_config": str(source_path.resolve()),
        "source_config_sha256": source_sha256,
        "training_jobs": str(jobs_path.resolve()),
        "training_jobs_sha256": jobs_sha256,
        "config_path": str(destination.resolve()),
        "config_sha256": sha256_file(destination),
    }


def _load_verified_queue_config(
    module: ModuleType,
    *,
    config_path: Path,
    config_document: Mapping[str, object],
    config_sha256: str,
) -> object:
    original_read_json = getattr(module, "_read_json")  # noqa: B009
    original_sha256_file = getattr(module, "sha256_file")  # noqa: B009
    resolved_config = config_path.resolve()

    def read_json(path: Path) -> dict[str, Any]:
        if path.resolve() == resolved_config:
            return copy.deepcopy(dict(config_document))
        return cast("dict[str, Any]", original_read_json(path))

    def bound_sha256_file(path: Path) -> str:
        if path.resolve() == resolved_config:
            return config_sha256
        return cast("str", original_sha256_file(path))

    setattr(module, "_read_json", read_json)  # noqa: B010
    setattr(module, "sha256_file", bound_sha256_file)  # noqa: B010
    try:
        return cast("object", module.load_queue_config(config_path))
    finally:
        setattr(module, "_read_json", original_read_json)  # noqa: B010
        setattr(module, "sha256_file", original_sha256_file)  # noqa: B010


def _assert_verified_sources_unchanged(sources: Sequence[VerifiedSource]) -> None:
    for source in sources:
        if source.path.read_bytes() != source.content:
            raise ValueError(f"verified source changed before snapshot: {source.path}")


def _verify_context(  # noqa: PLR0914 - binds all launch inputs in one read-only pass.
    *,
    config_path: Path,
    status_path: Path,
    scripts_dir: Path,
    output_root: Path,
    expected_config_sha256: str,
    expected_jobs_path: Path,
    expected_jobs_sha256: str,
    expected_component_sha256: Mapping[str, str],
) -> VerifiedContext:
    if expected_config_sha256 == PENDING_CONFIG_SHA256:
        raise ValueError(
            f"EXPECTED_CONFIG_SHA256 is {PENDING_CONFIG_SHA256}; "
            "run remote prepare, review its config_sha256, and pin that hash before "
            "preflight or launch"
        )
    config_source = _verified_source(
        config_path,
        expected_sha256=expected_config_sha256,
        label="config",
        snapshot_relative=None,
    )
    jobs_source = _verified_source(
        expected_jobs_path,
        expected_sha256=expected_jobs_sha256,
        label="training jobs",
        snapshot_relative=Path(RUNTIME_JOBS_NAME),
    )
    config_document = _read_json_bytes(config_source.content, source=config_path)
    jobs_document = _read_json_bytes(jobs_source.content, source=expected_jobs_path)
    verified_job_ids = _training_job_ids_document(jobs_document)
    component_sha256: dict[str, str] = {}
    component_sources: list[VerifiedSource] = []
    for name, expected in expected_component_sha256.items():
        source = _verified_source(
            scripts_dir / name,
            expected_sha256=expected,
            label="component",
            snapshot_relative=Path("scripts") / name,
        )
        component_sources.append(source)
        component_sha256[name] = source.sha256
    model_ids, reused_model_ids = _validate_isolated_config(
        config_document,
        config_path=config_path,
        status_path=status_path,
        output_root=output_root,
        expected_jobs_path=expected_jobs_path,
        verified_job_ids=verified_job_ids,
    )
    raw_status_path = _required_string(
        config_document,
        "training_status",
        source="queue config",
    )
    training_status_path = _resolve_input(config_path.parent, raw_status_path)
    status_source = _verified_source(
        training_status_path,
        expected_sha256=None,
        label="training status",
        snapshot_relative=Path(RUNTIME_STATUS_NAME),
    )
    _validate_latest_training_success(
        config_document=config_document,
        config_path=config_path,
        jobs_path=expected_jobs_path,
        jobs_document=jobs_document,
        status_content=status_source.content,
    )
    evaluation_source = next(
        (
            source
            for source in component_sources
            if source.path.name == "run_600m_speaker_evaluation_queue.py"
        ),
        None,
    )
    if evaluation_source is None:
        raise ValueError("component pins must include run_600m_speaker_evaluation_queue.py")
    queue_module = _load_evaluation_module(
        evaluation_source.path,
        source_bytes=evaluation_source.content,
    )
    queue_config = _load_verified_queue_config(
        queue_module,
        config_path=config_path,
        config_document=config_document,
        config_sha256=config_source.sha256,
    )
    queue_module._validate_ready_training(  # noqa: SLF001 - pinned operational contract.
        queue_config
    )
    sources = (
        config_source,
        jobs_source,
        status_source,
        *component_sources,
    )
    _assert_verified_sources_unchanged(sources)
    report = {
        "passed": True,
        "launch_performed": False,
        "model_count": len(model_ids),
        "model_ids": model_ids,
        "reused_model_ids": reused_model_ids,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_source.sha256,
        "training_jobs": str(expected_jobs_path.resolve()),
        "training_jobs_sha256": jobs_source.sha256,
        "training_status_sha256": status_source.sha256,
        "status_path": str(status_path.resolve()),
        "component_sha256": component_sha256,
    }
    return VerifiedContext(
        report=report,
        config_document=config_document,
        queue_module=queue_module,
        queue_config=queue_config,
        sources=sources,
    )


def preflight(
    *,
    config_path: Path,
    status_path: Path,
    scripts_dir: Path,
    output_root: Path,
    expected_config_sha256: str,
    expected_jobs_path: Path,
    expected_jobs_sha256: str,
    expected_component_sha256: Mapping[str, str],
) -> dict[str, object]:
    return _verify_context(
        config_path=config_path,
        status_path=status_path,
        scripts_dir=scripts_dir,
        output_root=output_root,
        expected_config_sha256=expected_config_sha256,
        expected_jobs_path=expected_jobs_path,
        expected_jobs_sha256=expected_jobs_sha256,
        expected_component_sha256=expected_component_sha256,
    ).report


def _runtime_snapshot(context: VerifiedContext, *, output_root: Path) -> RuntimeSnapshot:
    root = (output_root / RUNTIME_SNAPSHOT_NAME).resolve()
    scripts_dir = root / "scripts"
    config_path = root / RUNTIME_CONFIG_NAME
    jobs_path = root / RUNTIME_JOBS_NAME
    training_status = root / RUNTIME_STATUS_NAME
    runtime_document = copy.deepcopy(context.config_document)
    runtime_document["training_jobs"] = str(jobs_path)
    runtime_document["training_status"] = str(training_status)
    runtime_config = (
        json.dumps(
            runtime_document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    files = {
        source.snapshot_relative: source.content
        for source in context.sources
        if source.snapshot_relative is not None
    }
    files[Path(RUNTIME_CONFIG_NAME)] = runtime_config
    manifest = {
        "schema_version": RUNTIME_SNAPSHOT_SCHEMA,
        "source_inputs": {str(source.path): source.sha256 for source in context.sources},
        "files": {
            relative.as_posix(): {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
            for relative, content in sorted(files.items(), key=lambda item: item[0].as_posix())
        },
    }
    files[Path(RUNTIME_MANIFEST_NAME)] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return RuntimeSnapshot(
        root=root,
        scripts_dir=scripts_dir,
        config_path=config_path,
        jobs_path=jobs_path,
        training_status=training_status,
        files=files,
        config_sha256=hashlib.sha256(runtime_config).hexdigest(),
    )


def _expected_snapshot_entries(files: Mapping[Path, bytes]) -> set[Path]:
    entries = set(files)
    for relative in files:
        entries.update(parent for parent in relative.parents if parent != Path())
    return entries


def _validate_runtime_snapshot(snapshot: RuntimeSnapshot) -> None:
    actual_entries = {path.relative_to(snapshot.root) for path in snapshot.root.rglob("*")}
    expected_entries = _expected_snapshot_entries(snapshot.files)
    if actual_entries != expected_entries:
        raise ValueError(
            "runtime snapshot file set mismatch: "
            f"expected={sorted(map(str, expected_entries))}, "
            f"actual={sorted(map(str, actual_entries))}"
        )
    for relative, expected in snapshot.files.items():
        path = snapshot.root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"runtime snapshot content mismatch: {path}")


def _write_new_snapshot(snapshot: RuntimeSnapshot) -> None:
    temporary = snapshot.root.with_name(f".{snapshot.root.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary runtime snapshot: {temporary}")
    temporary.mkdir()
    try:
        for relative, content in snapshot.files.items():
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        temporary.rename(snapshot.root)
    except FileExistsError:
        shutil.rmtree(temporary, ignore_errors=True)
        if not snapshot.root.exists():
            raise
        _validate_runtime_snapshot(snapshot)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _materialize_runtime_snapshot(snapshot: RuntimeSnapshot) -> None:
    if snapshot.root.exists():
        _validate_runtime_snapshot(snapshot)
        return
    _write_new_snapshot(snapshot)
    _validate_runtime_snapshot(snapshot)


def launch(
    *,
    config_path: Path,
    status_path: Path,
    scripts_dir: Path,
    output_root: Path,
    expected_config_sha256: str,
    expected_jobs_path: Path,
    expected_jobs_sha256: str,
    expected_component_sha256: Mapping[str, str],
) -> QueueResult:
    _validate_operational_paths(
        config_path=config_path,
        status_path=status_path,
        output_root=output_root,
    )
    context = _verify_context(
        config_path=config_path,
        status_path=status_path,
        scripts_dir=scripts_dir,
        output_root=output_root,
        expected_config_sha256=expected_config_sha256,
        expected_jobs_path=expected_jobs_path,
        expected_jobs_sha256=expected_jobs_sha256,
        expected_component_sha256=expected_component_sha256,
    )
    snapshot = _runtime_snapshot(context, output_root=output_root)
    runtime_config = replace(
        cast("Any", context.queue_config),
        source_path=snapshot.config_path,
        source_sha256=snapshot.config_sha256,
        training_status=snapshot.training_status,
        training_jobs=snapshot.jobs_path,
    )
    with context.queue_module.evaluation_queue_lock(
        config=runtime_config,
        status_path=status_path,
    ):
        _assert_verified_sources_unchanged(context.sources)
        _materialize_runtime_snapshot(snapshot)
        return cast(
            "QueueResult",
            getattr(  # noqa: B009
                context.queue_module,
                "_run_evaluation_queue_locked",
            )(
                runtime_config,
                status_path=status_path,
                scripts_dir=snapshot.scripts_dir,
                runner=None,
                now=None,
            ),
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "preflight", "launch"))
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--scripts-dir", type=Path, default=DEFAULT_SCRIPTS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_operational_paths(
        config_path=args.config,
        status_path=args.status_path,
        output_root=args.output_root,
    )
    if args.mode == "prepare":
        prepare_result = prepare_speed_v4_config(
            source_path=args.source_config,
            destination=args.config,
            status_path=args.status_path,
            output_root=args.output_root,
            expected_source_sha256=EXPECTED_SOURCE_CONFIG_SHA256,
            jobs_path=DEFAULT_JOBS_PATH,
            expected_jobs_sha256=EXPECTED_JOBS_SHA256,
        )
        print(json.dumps(prepare_result, ensure_ascii=False, sort_keys=True))
        return 0
    common = {
        "config_path": args.config,
        "status_path": args.status_path,
        "scripts_dir": args.scripts_dir,
        "output_root": args.output_root,
        "expected_config_sha256": EXPECTED_CONFIG_SHA256,
        "expected_jobs_path": DEFAULT_JOBS_PATH,
        "expected_jobs_sha256": EXPECTED_JOBS_SHA256,
        "expected_component_sha256": EXPECTED_COMPONENT_SHA256,
    }
    if args.mode == "preflight":
        print(json.dumps(preflight(**common), ensure_ascii=False, sort_keys=True))
        return 0
    queue_result = launch(**common)
    print(
        json.dumps(
            {
                "succeeded": queue_result.succeeded,
                "skipped": queue_result.skipped,
                "reused": queue_result.reused,
                "failed": queue_result.failed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return int(bool(queue_result.failed))


if __name__ == "__main__":
    raise SystemExit(main())
