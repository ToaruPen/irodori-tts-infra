# ruff: noqa: EM101, EM102, TRY003 - operational errors retain contextual paths and stages.

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import socket
import subprocess  # noqa: S404 - this script runs a fixed, operator-owned evaluation pipeline.
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, TypeGuard

SCHEMA_VERSION = "speaker-evaluation-queue/v1"
STATUS_SCHEMA_VERSION = "speaker-evaluation-queue-status/v1"
LOCK_SCHEMA_VERSION = "speaker-evaluation-queue-lock/v1"
EXPECTED_MODEL_COUNT = 12
EXPECTED_GENERATION_COUNT = 140
EXPECTED_CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
UPSTREAM_RUNTIME_PROVENANCE_NAME = "upstream-runtime-provenance.json"
UPSTREAM_RUNTIME_PACKAGE_NAME = "upstream-runtime-package.zip"
RUNTIME_SNAPSHOT_NAME = "runtime-inputs-v1"
RUNTIME_MANIFEST_NAME = "snapshot-manifest.json"
RUNTIME_SNAPSHOT_SCHEMA = "speaker-evaluation-runtime-inputs/v1"
SHA256_LENGTH = 64
MIN_COMPONENT_COMMAND_PARTS = 2
EXPECTED_RUNTIME_SOURCE_COUNT = 9
RUNTIME_COMPONENT_NAMES = (
    "run_600m_speaker_evaluation_queue.py",
    "build_600m_checkpoint_evaluation_manifests.py",
    "generate_600m_checkpoint_audio_remote.py",
    "analyze_nko_beep_matrix.py",
    "compute_600m_speaker_metrics.py",
    "evaluate_600m_speaker_checkpoints.py",
)

Runner = Callable[[tuple[str, ...], Path], int]
Clock = Callable[[], str]


@dataclass(frozen=True, slots=True)
class BaseCheckpoint:
    model_id: str
    path: Path
    sha256: str
    revision: str


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingModel:
    model_id: str
    revision: str
    source_sha256: str
    source: Path
    savedir: Path


@dataclass(frozen=True, slots=True)
class TranscriptionModel:
    model_id: str
    revision: str
    source_sha256: str
    source: Path
    device: str


@dataclass(frozen=True, slots=True)
class MetricModels:
    speaker_embedding: SpeakerEmbeddingModel
    transcription: TranscriptionModel


@dataclass(frozen=True, slots=True)
class ReusedEvaluation:
    generation_dir: Path
    analysis_dir: Path
    metrics_results: Path
    metrics_provenance: Path
    evaluation_manifest: Path
    evaluation_dir: Path


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    model_id: str
    reference_wavs: Path
    generation_dir: Path | None
    analysis_dir: Path | None
    metrics_dir: Path | None
    evaluation_dir: Path | None
    reuse: ReusedEvaluation | None


@dataclass(frozen=True, slots=True)
class QueueConfig:
    source_path: Path
    source_sha256: str
    training_status: Path
    training_jobs: Path
    manifest_output_dir: Path
    base_checkpoint: BaseCheckpoint
    upstream_root: Path
    metric_models: MetricModels
    models: tuple[ModelEvaluation, ...]


@dataclass(frozen=True, slots=True)
class QueueResult:
    succeeded: tuple[str, ...]
    skipped: tuple[str, ...]
    reused: tuple[str, ...]
    failed: tuple[str, ...]


class QueueLockedError(RuntimeError):
    """Raised when another live process owns the queue status lock."""


@dataclass(frozen=True, slots=True)
class _Stage:
    key: str
    model_id: str | None
    command: tuple[str, ...]
    collision_paths: tuple[Path, ...]
    input_files: tuple[Path, ...]
    output_roots: tuple[Path, ...]
    required_outputs: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeFileBinding:
    relative: str
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class _RuntimeSnapshotGuard:
    root: Path
    manifest_path: Path
    manifest_bytes: bytes
    files: tuple[_RuntimeFileBinding, ...]

    def verify(self) -> None:
        _verify_runtime_snapshot_guard(self)

    def expected_sha256(self, path: Path) -> str | None:
        nominal = _nominal_absolute(path)
        for binding in self.files:
            if binding.path == nominal:
                return binding.sha256
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime_snapshot_guard(
    config: QueueConfig,
    *,
    scripts_dir: Path,
) -> _RuntimeSnapshotGuard | None:
    root = _nominal_absolute(config.source_path.parent)
    manifest_path = root / RUNTIME_MANIFEST_NAME
    provenance_path = root / UPSTREAM_RUNTIME_PROVENANCE_NAME
    package_path = root / UPSTREAM_RUNTIME_PACKAGE_NAME
    has_runtime_assets = provenance_path.exists() and package_path.exists()
    if not manifest_path.exists() and root.name.casefold() != RUNTIME_SNAPSHOT_NAME.casefold():
        return None
    if not manifest_path.is_file():
        raise ValueError(f"frozen runtime snapshot manifest is missing: {manifest_path}")
    _require_alias_free_path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    document = _runtime_manifest_document(manifest_bytes)
    raw_files = document.get("files")
    if not isinstance(raw_files, dict):
        raise TypeError("frozen runtime snapshot files must be an object")
    declares_runtime_package = UPSTREAM_RUNTIME_PACKAGE_NAME in raw_files
    if not has_runtime_assets and not declares_runtime_package:
        return None
    _validate_runtime_source_inputs(document.get("source_inputs"))
    relative_names = _validated_runtime_relative_names(raw_files)
    expected_names = _expected_runtime_inventory(config, scripts_dir=scripts_dir, root=root)
    if relative_names != expected_names:
        raise ValueError(
            "frozen runtime snapshot manifest inventory mismatch: "
            f"expected={expected_names}, actual={relative_names}"
        )
    bindings = _runtime_file_bindings(raw_files, root=root, relative_names=relative_names)
    guard = _RuntimeSnapshotGuard(
        root=root,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
        files=bindings,
    )
    guard.verify()
    return guard


def _runtime_manifest_document(manifest_bytes: bytes) -> dict[str, Any]:
    try:
        document = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen runtime snapshot manifest is invalid JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "source_inputs",
        "files",
    }:
        raise ValueError("frozen runtime snapshot manifest contract mismatch")
    if document.get("schema_version") != RUNTIME_SNAPSHOT_SCHEMA:
        raise ValueError("frozen runtime snapshot manifest schema mismatch")
    return document


def _validated_runtime_relative_names(raw_files: Mapping[object, object]) -> list[str]:
    raw_names = list(raw_files)
    if not all(isinstance(relative, str) for relative in raw_names):
        raise ValueError("frozen runtime snapshot paths must be strings")
    relative_names = [relative for relative in raw_names if isinstance(relative, str)]
    if relative_names != sorted(relative_names):
        raise ValueError("frozen runtime snapshot file inventory must be sorted")
    for relative in relative_names:
        if not _valid_runtime_relative(relative):
            raise ValueError(f"unsafe frozen runtime snapshot path: {relative!r}")
    casefolded = [relative.casefold() for relative in relative_names]
    if len(set(casefolded)) != len(casefolded):
        raise ValueError("frozen runtime snapshot has a case-insensitive collision")
    return relative_names


def _runtime_file_bindings(
    raw_files: Mapping[object, object],
    *,
    root: Path,
    relative_names: Sequence[str],
) -> tuple[_RuntimeFileBinding, ...]:
    bindings: list[_RuntimeFileBinding] = []
    for relative in relative_names:
        raw_binding = raw_files[relative]
        if not isinstance(raw_binding, dict) or set(raw_binding) != {"sha256", "size"}:
            raise ValueError(f"frozen runtime snapshot binding contract mismatch: {relative}")
        digest = _runtime_binding_sha256(raw_binding.get("sha256"), relative=relative)
        size = raw_binding.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"frozen runtime snapshot size is invalid: {relative}")
        bindings.append(
            _RuntimeFileBinding(
                relative=relative,
                path=root.joinpath(*PurePosixPath(relative).parts),
                sha256=digest,
                size=size,
            )
        )
    return tuple(bindings)


def _runtime_binding_sha256(value: object, *, relative: str) -> str:
    if not _valid_lower_sha256(value):
        raise ValueError(f"frozen runtime snapshot SHA-256 is invalid: {relative}")
    return value


def _validate_runtime_source_inputs(value: object) -> None:
    if not isinstance(value, dict) or len(value) != EXPECTED_RUNTIME_SOURCE_COUNT:
        raise ValueError(
            "frozen runtime snapshot must contain exactly "
            f"{EXPECTED_RUNTIME_SOURCE_COUNT} source bindings"
        )
    paths = list(value)
    if len({path.casefold() for path in paths if isinstance(path, str)}) != len(paths):
        raise ValueError("frozen runtime source bindings have a case-insensitive collision")
    for path, digest in value.items():
        if not isinstance(path, str) or not path or not Path(path).is_absolute():
            raise ValueError("frozen runtime source binding path is invalid")
        if not _valid_lower_sha256(digest):
            raise ValueError(f"frozen runtime source binding SHA-256 is invalid: {path}")


def _expected_runtime_inventory(
    config: QueueConfig,
    *,
    scripts_dir: Path,
    root: Path,
) -> list[str]:
    nominal_scripts = _nominal_absolute(scripts_dir)
    if nominal_scripts != root / "scripts":
        raise ValueError("frozen runtime scripts directory must be the bundle scripts directory")
    paths = (
        config.source_path,
        config.training_jobs,
        config.training_status,
        root / UPSTREAM_RUNTIME_PROVENANCE_NAME,
        root / UPSTREAM_RUNTIME_PACKAGE_NAME,
        *(nominal_scripts / name for name in RUNTIME_COMPONENT_NAMES),
    )
    relative_names: list[str] = []
    for path in paths:
        nominal = _nominal_absolute(path)
        try:
            relative = nominal.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"frozen runtime input is outside its bundle: {nominal}") from exc
        if not _valid_runtime_relative(relative):
            raise ValueError(f"unsafe frozen runtime input path: {relative!r}")
        relative_names.append(relative)
    if len(set(relative_names)) != len(relative_names):
        raise ValueError("frozen runtime expected inventory contains duplicates")
    if len({relative.casefold() for relative in relative_names}) != len(relative_names):
        raise ValueError("frozen runtime expected inventory has a case-insensitive collision")
    return sorted(relative_names)


def _valid_runtime_relative(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    pure = PurePosixPath(value)
    return (
        not pure.is_absolute()
        and pure.as_posix() == value
        and all(part not in {"", ".", ".."} for part in pure.parts)
    )


def _valid_lower_sha256(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _nominal_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))  # noqa: PTH100 - resolve() would follow aliases.


def _is_filesystem_alias(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(file_attributes & 0x400)


def _require_alias_free_path(path: Path) -> None:
    nominal = _nominal_absolute(path)
    for component in (nominal, *nominal.parents):
        if component == component.parent:
            continue
        if _is_filesystem_alias(component):
            raise ValueError(f"frozen runtime snapshot contains a filesystem alias: {component}")


def _verify_runtime_snapshot_guard(guard: _RuntimeSnapshotGuard) -> None:
    _require_alias_free_path(guard.manifest_path)
    try:
        current_manifest = guard.manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("frozen runtime snapshot manifest changed after guard load") from exc
    if current_manifest != guard.manifest_bytes:
        raise ValueError("frozen runtime snapshot manifest changed after guard load")
    actual_paths = _runtime_snapshot_actual_files(guard.root)
    expected_paths = sorted([RUNTIME_MANIFEST_NAME, *(binding.relative for binding in guard.files)])
    if actual_paths != expected_paths:
        raise ValueError(
            "frozen runtime snapshot file set mismatch: "
            f"expected={expected_paths}, actual={actual_paths}"
        )
    for binding in guard.files:
        _require_alias_free_path(binding.path)
        try:
            size = binding.path.stat().st_size
        except OSError as exc:
            raise ValueError(f"frozen runtime snapshot content changed: {binding.path}") from exc
        if size != binding.size or sha256_file(binding.path) != binding.sha256:
            raise ValueError(f"frozen runtime snapshot content changed: {binding.path}")


def _runtime_snapshot_actual_files(root: Path) -> list[str]:
    files: list[str] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            raise ValueError(f"frozen runtime snapshot file set changed: {directory}") from exc
        for child in children:
            relative = child.relative_to(root).as_posix()
            if _is_filesystem_alias(child):
                files.append(relative)
            elif child.is_dir():
                pending.append(child)
            elif child.is_file():
                files.append(relative)
            else:
                files.append(relative)
    return sorted(files)


def _verify_runtime_guard(guard: _RuntimeSnapshotGuard | None) -> None:
    if guard is not None:
        guard.verify()


def _bound_sha256(path: Path, *, runtime_guard: _RuntimeSnapshotGuard | None) -> str:
    if runtime_guard is not None:
        expected = runtime_guard.expected_sha256(path)
        if expected is not None:
            return expected
    return sha256_file(path)


def load_queue_config(path: Path) -> QueueConfig:
    document = _read_json(path)
    if document.get("schema_version") != SCHEMA_VERSION:
        message = f"queue config schema_version must be {SCHEMA_VERSION!r}"
        raise ValueError(message)
    base = path.parent
    base_raw = _required_mapping(document, "base_checkpoint")
    metrics_raw = _required_mapping(document, "metric_models")
    speaker_raw = _required_mapping(metrics_raw, "speaker_embedding")
    transcription_raw = _required_mapping(metrics_raw, "transcription")
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != EXPECTED_MODEL_COUNT:
        message = f"queue config must contain exactly {EXPECTED_MODEL_COUNT} models"
        raise ValueError(message)
    models = tuple(_parse_model(raw, base=base) for raw in raw_models)
    model_ids = [model.model_id for model in models]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("queue config contains duplicate model ids")
    device = _required_string(transcription_raw, "device")
    if device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("transcription device must be cpu, cuda, or cuda:0")
    return QueueConfig(
        source_path=path,
        source_sha256=sha256_file(path),
        training_status=_resolve(base, _required_string(document, "training_status")),
        training_jobs=_resolve(base, _required_string(document, "training_jobs")),
        manifest_output_dir=_resolve(
            base,
            _required_string(document, "manifest_output_dir"),
        ),
        base_checkpoint=BaseCheckpoint(
            model_id=_required_string(base_raw, "model_id"),
            path=_resolve(base, _required_string(base_raw, "path")),
            sha256=_required_sha256(base_raw, "sha256"),
            revision=_required_string(base_raw, "revision"),
        ),
        upstream_root=_resolve(base, _required_string(document, "upstream_root")),
        metric_models=MetricModels(
            speaker_embedding=SpeakerEmbeddingModel(
                model_id=_required_string(speaker_raw, "model_id"),
                revision=_required_string(speaker_raw, "revision"),
                source_sha256=_required_sha256(speaker_raw, "source_sha256"),
                source=_resolve(base, _required_string(speaker_raw, "source")),
                savedir=_resolve(base, _required_string(speaker_raw, "savedir")),
            ),
            transcription=TranscriptionModel(
                model_id=_required_string(transcription_raw, "model_id"),
                revision=_required_string(transcription_raw, "revision"),
                source_sha256=_required_sha256(transcription_raw, "source_sha256"),
                source=_resolve(base, _required_string(transcription_raw, "source")),
                device=device,
            ),
        ),
        models=models,
    )


def run_evaluation_queue(
    config: QueueConfig,
    *,
    status_path: Path,
    scripts_dir: Path,
    runner: Runner | None = None,
    now: Clock | None = None,
) -> QueueResult:
    with evaluation_queue_lock(config=config, status_path=status_path):
        return _run_evaluation_queue_locked(
            config,
            status_path=status_path,
            scripts_dir=scripts_dir,
            runner=runner,
            now=now,
        )


def _run_evaluation_queue_locked(
    config: QueueConfig,
    *,
    status_path: Path,
    scripts_dir: Path,
    runner: Runner | None,
    now: Clock | None,
) -> QueueResult:
    runtime_guard = _load_runtime_snapshot_guard(config, scripts_dir=scripts_dir)
    _validate_ready_training(config)
    execute = runner or _run_subprocess
    clock = now or _utc_now
    successful_rows = _successful_rows(status_path)
    succeeded: list[str] = []
    skipped: list[str] = []
    reused: list[str] = []
    failed: list[str] = []

    manifest_stage = _manifest_stage(config, scripts_dir=scripts_dir)
    stage_status = _run_stage(
        manifest_stage,
        config=config,
        status_path=status_path,
        successful_rows=successful_rows,
        execute=execute,
        clock=clock,
        runtime_guard=runtime_guard,
    )
    _record_result(
        manifest_stage.key,
        stage_status,
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
    )
    if stage_status == "failed":
        return QueueResult(tuple(succeeded), tuple(skipped), tuple(reused), tuple(failed))

    for model in config.models:
        if model.reuse is not None:
            _verify_runtime_guard(runtime_guard)
            _validate_canonical_reuse(config, model=model)
            for stage in _reused_stages(model):
                stage_status = _reuse_stage(
                    stage,
                    config=config,
                    status_path=status_path,
                    successful_rows=successful_rows,
                    clock=clock,
                    runtime_guard=runtime_guard,
                )
                if stage_status == "skipped":
                    skipped.append(stage.key)
                else:
                    reused.append(stage.key)
            continue
        for stage in _model_stages(
            config,
            model=model,
            scripts_dir=scripts_dir,
            runtime_guard=runtime_guard,
        ):
            stage_status = _run_stage(
                stage,
                config=config,
                status_path=status_path,
                successful_rows=successful_rows,
                execute=execute,
                clock=clock,
                runtime_guard=runtime_guard,
            )
            _record_result(
                stage.key,
                stage_status,
                succeeded=succeeded,
                skipped=skipped,
                failed=failed,
            )
            if stage_status == "failed":
                break
    return QueueResult(tuple(succeeded), tuple(skipped), tuple(reused), tuple(failed))


def queue_lock_path(status_path: Path) -> Path:
    return status_path.with_suffix(status_path.suffix + ".lock")


@contextlib.contextmanager
def evaluation_queue_lock(
    *,
    config: QueueConfig,
    status_path: Path,
) -> Iterator[Path]:
    path = queue_lock_path(status_path)
    token = uuid.uuid4().hex
    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "token": token,
        "config_path": str(config.source_path.resolve()),
        "config_sha256": config.source_sha256,
        "status_path": str(status_path.resolve()),
        "created_at": _utc_now(),
    }
    _create_lock(path, payload=payload)
    try:
        yield path
    finally:
        _release_owned_lock(path, token=token)


def _create_lock(path: Path, *, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if attempt or not _recover_same_host_stale_lock(path):
                raise QueueLockedError(f"evaluation queue is already locked: {path}") from None
            continue
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as destination:
                destination.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return
    raise QueueLockedError(f"evaluation queue is already locked: {path}")


def _recover_same_host_stale_lock(path: Path) -> bool:
    try:
        payload = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    owner = _stale_same_host_owner(payload)
    if owner is None:
        return False
    _, token = owner
    try:
        current = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if current.get("token") != token:
        return False
    path.unlink()
    return True


def _stale_same_host_owner(payload: Mapping[str, object]) -> tuple[int, str] | None:
    pid = payload.get("pid")
    hostname = payload.get("hostname")
    token = payload.get("token")
    if payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        return None
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if hostname != socket.gethostname():
        return None
    if not isinstance(token, str) or not token:
        return None
    if _pid_is_alive(pid):
        return None
    return pid, token


def _pid_is_alive(pid: int) -> bool:
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    inherit_handle = 0
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(process_query_limited_information, inherit_handle, pid)
    if not handle:
        return int(kernel32.GetLastError()) == error_access_denied
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _release_owned_lock(path: Path, *, token: str) -> None:
    try:
        payload = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    if payload.get("token") == token:
        path.unlink(missing_ok=True)


def _validate_canonical_reuse(config: QueueConfig, *, model: ModelEvaluation) -> None:
    reuse = model.reuse
    if reuse is None:
        raise TypeError(f"reuse is missing for {model.model_id}")
    generation_results = reuse.generation_dir / "generation-results.jsonl"
    generation_sha256 = sha256_file(generation_results)
    _validate_generation_proof(
        reuse,
        model_id=model.model_id,
        generation_sha256=generation_sha256,
    )
    selected_path = reuse.evaluation_dir / "selected-models.json"
    selected_models = _read_json(selected_path)
    input_sha256 = selected_models.get("input_sha256")
    if not isinstance(input_sha256, dict):
        raise TypeError(
            f"canonical reuse binding failed for {model.model_id}: selected input hashes"
        )
    current_input_sha256 = {
        "generation_results": generation_sha256,
        "analysis_results": sha256_file(reuse.analysis_dir / "analysis-results.jsonl"),
        "metrics_results": sha256_file(reuse.metrics_results),
        "metrics_provenance": sha256_file(reuse.metrics_provenance),
        "evaluation_manifest": sha256_file(reuse.evaluation_manifest),
    }
    if set(input_sha256) != set(current_input_sha256) or any(
        input_sha256.get(key) != value for key, value in current_input_sha256.items()
    ):
        raise ValueError(
            f"canonical reuse binding failed for {model.model_id}: selected inputs changed"
        )
    selections = selected_models.get("selections")
    if (
        not isinstance(selections, list)
        or len(selections) != 1
        or not isinstance(selections[0], dict)
        or selections[0].get("model_id") != model.model_id
    ):
        raise ValueError(f"canonical reuse binding failed for {model.model_id}: selection mismatch")
    selected = selections[0]
    _validate_evaluation_verification(
        reuse.evaluation_dir / "evaluation-verification.json",
        selected_path=selected_path,
        selected=selected,
        model_id=model.model_id,
    )
    _validate_semantic_manifest_binding(
        canonical_path=reuse.evaluation_manifest,
        current_path=config.manifest_output_dir / model.model_id / "evaluation-manifest.json",
        model_id=model.model_id,
        selected=selected,
    )


def _generation_proof_path(reuse: ReusedEvaluation, *, model_id: str) -> Path:
    candidates = (
        reuse.generation_dir / "generation-verification.json",
        reuse.generation_dir / "canonicalization-report.json",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            f"canonical reuse binding failed for {model_id}: exactly one generation proof required"
        )
    return existing[0]


def _validate_generation_proof(
    reuse: ReusedEvaluation,
    *,
    model_id: str,
    generation_sha256: str,
) -> None:
    proof_path = _generation_proof_path(reuse, model_id=model_id)
    proof = _read_json(proof_path)
    if proof_path.name == "generation-verification.json":
        valid = (
            proof.get("schema_version") == "speaker-checkpoint-audio-generation-verification/v1"
            and proof.get("passed") is True
            and proof.get("model_id") == model_id
            and proof.get("row_count") == EXPECTED_GENERATION_COUNT
            and proof.get("generation_results_sha256") == generation_sha256
        )
    else:
        canonical_sha256 = proof.get("canonical_sha256")
        counts = proof.get("counts")
        valid = (
            proof.get("schema_version") == "speaker-canonicalization/v1"
            and proof.get("model_id") == model_id
            and isinstance(canonical_sha256, dict)
            and canonical_sha256.get("generation_results") == generation_sha256
            and isinstance(counts, dict)
            and counts.get("generation") == EXPECTED_GENERATION_COUNT
        )
    if not valid:
        raise ValueError(f"canonical reuse binding failed for {model_id}: invalid generation proof")


def _validate_evaluation_verification(
    path: Path,
    *,
    selected_path: Path,
    selected: Mapping[str, object],
    model_id: str,
) -> None:
    verification = _read_json(path)
    artifact_sha256 = verification.get("artifact_sha256")
    valid = (
        verification.get("schema_version") == "speaker-checkpoint-evaluation-verification/v2"
        and verification.get("status") == "PASS"
        and verification.get("selected") == selected
        and isinstance(artifact_sha256, dict)
        and artifact_sha256.get(str(selected_path)) == sha256_file(selected_path)
    )
    if not valid:
        raise ValueError(f"canonical reuse binding failed for {model_id}: evaluation verification")


def _validate_semantic_manifest_binding(
    *,
    canonical_path: Path,
    current_path: Path,
    model_id: str,
    selected: Mapping[str, object],
) -> None:
    canonical = _manifest_semantics(canonical_path, model_id=model_id)
    current = _manifest_semantics(current_path, model_id=model_id)
    if canonical["matrix"] != current["matrix"]:
        raise ValueError(f"semantic manifest binding failed for {model_id}: matrix")
    if canonical["metrics_provenance"] != current["metrics_provenance"]:
        raise ValueError(f"semantic manifest binding failed for {model_id}: metrics provenance")
    canonical_checkpoints = canonical["checkpoints"]
    current_checkpoints = current["checkpoints"]
    if not isinstance(canonical_checkpoints, dict) or not isinstance(current_checkpoints, dict):
        raise TypeError(f"semantic manifest binding failed for {model_id}: checkpoints")
    if canonical_checkpoints.keys() != current_checkpoints.keys():
        raise ValueError(f"semantic manifest binding failed for {model_id}: checkpoint steps")
    for step in canonical_checkpoints:
        canonical_candidate = canonical_checkpoints[step]
        current_candidate = current_checkpoints[step]
        if not isinstance(canonical_candidate, dict) or not isinstance(current_candidate, dict):
            raise TypeError(f"semantic manifest binding failed for {model_id}: candidate")
        canonical_comparable = dict(canonical_candidate)
        current_comparable = dict(current_candidate)
        canonical_comparable.pop("base_checkpoint", None)
        current_comparable.pop("base_checkpoint", None)
        if canonical_comparable != current_comparable:
            raise ValueError(f"semantic manifest binding failed for {model_id}: checkpoint {step}")
    selected_step = selected.get("checkpoint_step")
    selected_candidate = canonical_checkpoints.get(selected_step)
    if not isinstance(selected_candidate, dict) or any(
        selected.get(field) != value for field, value in selected_candidate.items()
    ):
        raise ValueError(f"semantic manifest binding failed for {model_id}: selection")


def _manifest_semantics(path: Path, *, model_id: str) -> dict[str, object]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != "speaker-checkpoint-evaluation-manifest/v1":
        raise ValueError(f"semantic manifest binding failed for {model_id}: schema")
    raw_models = manifest.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != 1:
        raise ValueError(f"semantic manifest binding failed for {model_id}: identity")
    raw_model = raw_models[0]
    if not isinstance(raw_model, dict) or raw_model.get("model_id") != model_id:
        raise ValueError(f"semantic manifest binding failed for {model_id}: identity")
    raw_checkpoints = raw_model.get("checkpoints")
    if not isinstance(raw_checkpoints, list):
        raise TypeError(f"semantic manifest binding failed for {model_id}: checkpoints")
    checkpoints: dict[int, dict[str, object]] = {}
    for raw_candidate in raw_checkpoints:
        if not isinstance(raw_candidate, dict):
            raise TypeError(f"semantic manifest binding failed for {model_id}: candidate")
        step = raw_candidate.get("checkpoint_step")
        if not isinstance(step, int) or isinstance(step, bool) or step in checkpoints:
            raise ValueError(f"semantic manifest binding failed for {model_id}: checkpoint step")
        checkpoints[step] = raw_candidate
    if tuple(checkpoints) != EXPECTED_CHECKPOINT_STEPS:
        raise ValueError(
            f"semantic manifest binding failed for {model_id}: checkpoint order or set"
        )
    return {
        "checkpoints": checkpoints,
        "matrix": {field: manifest.get(field) for field in ("text_ids", "seeds", "styles")},
        "metrics_provenance": manifest.get("metrics_provenance"),
    }


def _validate_ready_training(config: QueueConfig) -> None:
    job_model_ids = _training_job_model_ids(config.training_jobs)
    configured_model_ids = [model.model_id for model in config.models]
    if job_model_ids != configured_model_ids:
        raise ValueError("training job order and evaluation model order must match exactly")
    successful_model_ids = _successful_training_model_ids(config.training_status, job_model_ids)
    missing = [model_id for model_id in job_model_ids if model_id not in successful_model_ids]
    if missing:
        raise ValueError(
            "all 12 training models must have successful finished status before evaluation: "
            + ", ".join(missing)
        )
    if sha256_file(config.base_checkpoint.path) != config.base_checkpoint.sha256:
        raise ValueError("base checkpoint SHA-256 does not match queue config")
    required_paths = [
        config.upstream_root,
        config.metric_models.speaker_embedding.source,
        config.metric_models.transcription.source,
        *(model.reference_wavs for model in config.models),
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"required evaluation input is missing: {missing_paths[0]}")


def _training_job_model_ids(path: Path) -> list[str]:
    jobs_document = _read_json(path)
    raw_jobs = jobs_document.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"training jobs must contain exactly {EXPECTED_MODEL_COUNT} jobs")
    job_model_ids = [
        _required_string(_object(row, source="training job"), "model_id") for row in raw_jobs
    ]
    if len(set(job_model_ids)) != EXPECTED_MODEL_COUNT:
        raise ValueError("training jobs contain duplicate model ids")
    return job_model_ids


def _successful_training_model_ids(path: Path, job_model_ids: Sequence[str]) -> set[str]:
    successful_model_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid training status JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"training status line {line_number} must be an object")
        if row.get("event") == "finished" and row.get("status") == "success":
            model_id = row.get("model_id")
            if isinstance(model_id, str) and model_id in job_model_ids:
                successful_model_ids.add(model_id)
    return successful_model_ids


def _manifest_stage(config: QueueConfig, *, scripts_dir: Path) -> _Stage:
    component_script = scripts_dir / "build_600m_checkpoint_evaluation_manifests.py"
    command = [
        sys.executable,
        str(component_script),
        "--training-status",
        str(config.training_status),
        "--training-jobs",
        str(config.training_jobs),
        "--output-dir",
        str(config.manifest_output_dir),
        "--base-checkpoint",
        config.base_checkpoint.model_id,
        "--base-checkpoint-sha256",
        config.base_checkpoint.sha256,
        "--base-revision",
        config.base_checkpoint.revision,
        "--speaker-embedding-model-id",
        config.metric_models.speaker_embedding.model_id,
        "--speaker-embedding-revision",
        config.metric_models.speaker_embedding.revision,
        "--speaker-embedding-source-sha256",
        config.metric_models.speaker_embedding.source_sha256,
        "--transcription-model-id",
        config.metric_models.transcription.model_id,
        "--transcription-revision",
        config.metric_models.transcription.revision,
        "--transcription-source-sha256",
        config.metric_models.transcription.source_sha256,
    ]
    for model in config.models:
        command.extend(("--reference-wavs", str(model.reference_wavs)))
    return _Stage(
        key="manifests",
        model_id=None,
        command=tuple(command),
        collision_paths=(config.manifest_output_dir,),
        input_files=(
            component_script,
            config.training_status,
            config.training_jobs,
            *(model.reference_wavs for model in config.models),
        ),
        output_roots=(config.manifest_output_dir,),
        required_outputs=(config.manifest_output_dir / "manifest-index.json",),
    )


def _model_stages(
    config: QueueConfig,
    *,
    model: ModelEvaluation,
    scripts_dir: Path,
    runtime_guard: _RuntimeSnapshotGuard | None = None,
) -> tuple[_Stage, ...]:
    if any(
        path is None
        for path in (
            model.generation_dir,
            model.analysis_dir,
            model.metrics_dir,
            model.evaluation_dir,
        )
    ):
        raise TypeError(f"queue-owned outputs are missing for {model.model_id}")
    generation_dir = _not_none(model.generation_dir)
    analysis_dir = _not_none(model.analysis_dir)
    metrics_dir = _not_none(model.metrics_dir)
    evaluation_dir = _not_none(model.evaluation_dir)
    manifest = config.manifest_output_dir / model.model_id / "evaluation-manifest.json"
    metrics_results = metrics_dir / "metrics-results.jsonl"
    metrics_provenance = metrics_dir / "metrics-results.provenance.json"
    generation_script = scripts_dir / "generate_600m_checkpoint_audio_remote.py"
    upstream_runtime_provenance = config.source_path.parent / UPSTREAM_RUNTIME_PROVENANCE_NAME
    upstream_runtime_package = config.source_path.parent / UPSTREAM_RUNTIME_PACKAGE_NAME
    analysis_script = scripts_dir / "analyze_nko_beep_matrix.py"
    metrics_script = scripts_dir / "compute_600m_speaker_metrics.py"
    evaluation_script = scripts_dir / "evaluate_600m_speaker_checkpoints.py"
    return (
        _Stage(
            key=f"{model.model_id}:generation",
            model_id=model.model_id,
            command=(
                sys.executable,
                str(generation_script),
                "generate",
                "--checkpoint-manifest",
                str(manifest),
                "--base-checkpoint-path",
                str(config.base_checkpoint.path),
                "--upstream-root",
                str(config.upstream_root),
                "--upstream-runtime-provenance",
                str(upstream_runtime_provenance),
                "--upstream-package-archive",
                str(upstream_runtime_package),
                "--output-dir",
                str(generation_dir),
                "--upstream-runtime-provenance-sha256",
                _bound_sha256(upstream_runtime_provenance, runtime_guard=runtime_guard),
                "--upstream-package-archive-sha256",
                _bound_sha256(upstream_runtime_package, runtime_guard=runtime_guard),
            ),
            collision_paths=(generation_dir,),
            input_files=(
                generation_script,
                manifest,
                config.base_checkpoint.path,
                upstream_runtime_provenance,
                upstream_runtime_package,
            ),
            output_roots=(generation_dir,),
            required_outputs=(
                generation_dir / "generation-results.jsonl",
                generation_dir / "generation-verification.json",
            ),
        ),
        _Stage(
            key=f"{model.model_id}:analysis",
            model_id=model.model_id,
            command=(
                sys.executable,
                str(analysis_script),
                "--generation-dir",
                str(generation_dir),
                "--output-dir",
                str(analysis_dir),
            ),
            collision_paths=(analysis_dir,),
            input_files=(
                analysis_script,
                generation_dir / "generation-results.jsonl",
                generation_dir / "generation-verification.json",
            ),
            output_roots=(analysis_dir,),
            required_outputs=(analysis_dir / "analysis-results.jsonl",),
        ),
        _Stage(
            key=f"{model.model_id}:metrics",
            model_id=model.model_id,
            command=(
                sys.executable,
                str(metrics_script),
                "--generation-results",
                str(generation_dir / "generation-results.jsonl"),
                "--reference-wavs",
                str(model.reference_wavs),
                "--output",
                str(metrics_results),
                "--provenance-output",
                str(metrics_provenance),
                "--ecapa-source",
                str(config.metric_models.speaker_embedding.source),
                "--ecapa-savedir",
                str(config.metric_models.speaker_embedding.savedir),
                "--ecapa-model-id",
                config.metric_models.speaker_embedding.model_id,
                "--ecapa-revision",
                config.metric_models.speaker_embedding.revision,
                "--whisper-model",
                config.metric_models.transcription.model_id,
                "--whisper-source",
                str(config.metric_models.transcription.source),
                "--whisper-revision",
                config.metric_models.transcription.revision,
                "--whisper-device",
                config.metric_models.transcription.device,
            ),
            collision_paths=(metrics_dir,),
            input_files=(
                metrics_script,
                generation_dir / "generation-results.jsonl",
                model.reference_wavs,
            ),
            output_roots=(metrics_dir,),
            required_outputs=(metrics_results, metrics_provenance),
        ),
        _Stage(
            key=f"{model.model_id}:evaluate",
            model_id=model.model_id,
            command=(
                sys.executable,
                str(evaluation_script),
                "--generation-results",
                str(generation_dir / "generation-results.jsonl"),
                "--analysis-results",
                str(analysis_dir / "analysis-results.jsonl"),
                "--metrics-results",
                str(metrics_results),
                "--metrics-provenance",
                str(metrics_provenance),
                "--evaluation-manifest",
                str(manifest),
                "--output-dir",
                str(evaluation_dir),
            ),
            collision_paths=(evaluation_dir,),
            input_files=(
                evaluation_script,
                generation_dir / "generation-results.jsonl",
                analysis_dir / "analysis-results.jsonl",
                metrics_results,
                metrics_provenance,
                manifest,
            ),
            output_roots=(evaluation_dir,),
            required_outputs=(
                evaluation_dir / "selected-models.json",
                evaluation_dir / "evaluation-verification.json",
            ),
        ),
    )


def _reused_stages(model: ModelEvaluation) -> tuple[_Stage, ...]:
    reuse = model.reuse
    if reuse is None:
        raise TypeError(f"reuse is missing for {model.model_id}")
    return (
        _Stage(
            key=f"{model.model_id}:generation",
            model_id=model.model_id,
            command=(),
            collision_paths=(),
            input_files=(),
            output_roots=(
                reuse.generation_dir / "generation-results.jsonl",
                _generation_proof_path(reuse, model_id=model.model_id),
            ),
            required_outputs=(
                reuse.generation_dir / "generation-results.jsonl",
                _generation_proof_path(reuse, model_id=model.model_id),
            ),
        ),
        _Stage(
            key=f"{model.model_id}:analysis",
            model_id=model.model_id,
            command=(),
            collision_paths=(),
            input_files=(),
            output_roots=(reuse.analysis_dir / "analysis-results.jsonl",),
            required_outputs=(reuse.analysis_dir / "analysis-results.jsonl",),
        ),
        _Stage(
            key=f"{model.model_id}:metrics",
            model_id=model.model_id,
            command=(),
            collision_paths=(),
            input_files=(),
            output_roots=(reuse.metrics_results, reuse.metrics_provenance),
            required_outputs=(reuse.metrics_results, reuse.metrics_provenance),
        ),
        _Stage(
            key=f"{model.model_id}:evaluate",
            model_id=model.model_id,
            command=(),
            collision_paths=(),
            input_files=(),
            output_roots=(reuse.evaluation_dir,),
            required_outputs=(
                reuse.evaluation_dir / "selected-models.json",
                reuse.evaluation_dir / "evaluation-verification.json",
            ),
        ),
    )


def _run_stage(
    stage: _Stage,
    *,
    config: QueueConfig,
    status_path: Path,
    successful_rows: Mapping[str, Sequence[Mapping[str, object]]],
    execute: Runner,
    clock: Clock,
    runtime_guard: _RuntimeSnapshotGuard | None,
) -> str:
    _verify_runtime_guard(runtime_guard)
    if _has_reusable_stage_success(
        successful_rows.get(stage.key, ()),
        stage=stage,
        runtime_guard=runtime_guard,
    ):
        _verify_runtime_guard(runtime_guard)
        return "skipped"
    _verify_runtime_guard(runtime_guard)
    started_at = clock()
    log_path = status_path.parent / "logs" / f"{stage.key.replace(':', '__')}.log"
    base_row = _base_status_row(
        stage,
        config=config,
        started_at=started_at,
        log_path=log_path,
        runtime_guard=runtime_guard,
    )
    _append_status(status_path, base_row | {"event": "started", "status": "running"})
    collisions = [path for path in stage.collision_paths if path.exists()]
    if collisions:
        error = f"output path already exists: {collisions[0]}"
        _append_status(
            status_path,
            base_row
            | {
                "event": "finished",
                "status": "failed",
                "ended_at": clock(),
                "error": error,
            },
        )
        raise FileExistsError(error)
    _verify_started_stage_runtime_guard(
        runtime_guard,
        status_path=status_path,
        base_row=base_row,
        clock=clock,
    )
    try:
        exit_code = execute(stage.command, log_path)
    except OSError as exc:
        _append_status(
            status_path,
            base_row
            | {
                "event": "finished",
                "status": "failed",
                "ended_at": clock(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return "failed"
    if exit_code != 0:
        _verify_started_stage_runtime_guard(
            runtime_guard,
            status_path=status_path,
            base_row=base_row,
            clock=clock,
        )
        _append_status(
            status_path,
            base_row
            | {
                "event": "finished",
                "status": "failed",
                "ended_at": clock(),
                "exit_code": exit_code,
                "error": f"subprocess exited with code {exit_code}",
            },
        )
        return "failed"
    try:
        outputs = _validated_output_snapshot(stage)
    except (OSError, ValueError) as exc:
        _append_status(
            status_path,
            base_row
            | {
                "event": "finished",
                "status": "failed",
                "ended_at": clock(),
                "exit_code": exit_code,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return "failed"
    _verify_started_stage_runtime_guard(
        runtime_guard,
        status_path=status_path,
        base_row=base_row,
        clock=clock,
        exit_code=exit_code,
    )
    success_row = base_row | {
        "event": "finished",
        "status": "success",
        "ended_at": clock(),
        "exit_code": exit_code,
        "outputs": outputs,
    }
    _append_status(status_path, success_row)
    return "success"


def _reuse_stage(
    stage: _Stage,
    *,
    config: QueueConfig,
    status_path: Path,
    successful_rows: Mapping[str, Sequence[Mapping[str, object]]],
    clock: Clock,
    runtime_guard: _RuntimeSnapshotGuard | None,
) -> str:
    _verify_runtime_guard(runtime_guard)
    if _has_reusable_stage_success(
        successful_rows.get(stage.key, ()),
        stage=stage,
        runtime_guard=runtime_guard,
    ):
        _verify_runtime_guard(runtime_guard)
        return "skipped"
    _verify_runtime_guard(runtime_guard)
    started_at = clock()
    base_row = _base_status_row(
        stage,
        config=config,
        started_at=started_at,
        log_path=None,
        runtime_guard=runtime_guard,
    )
    _append_status(
        status_path,
        base_row | {"event": "started", "status": "running", "reused": True},
    )
    try:
        outputs = _validated_output_snapshot(stage)
    except (OSError, ValueError) as exc:
        _append_status(
            status_path,
            base_row
            | {
                "event": "finished",
                "status": "failed",
                "ended_at": clock(),
                "error": f"{type(exc).__name__}: {exc}",
                "reused": True,
            },
        )
        raise
    _verify_started_stage_runtime_guard(
        runtime_guard,
        status_path=status_path,
        base_row=base_row,
        clock=clock,
        exit_code=0,
        reused=True,
    )
    success_row = base_row | {
        "event": "finished",
        "status": "success",
        "ended_at": clock(),
        "exit_code": 0,
        "outputs": outputs,
        "reused": True,
    }
    _append_status(status_path, success_row)
    return "reused"


def _verify_started_stage_runtime_guard(
    runtime_guard: _RuntimeSnapshotGuard | None,
    *,
    status_path: Path,
    base_row: Mapping[str, object],
    clock: Clock,
    exit_code: int | None = None,
    reused: bool = False,
) -> None:
    try:
        _verify_runtime_guard(runtime_guard)
    except ValueError as exc:
        failure = dict(base_row)
        failure.update(
            {
                "event": "finished",
                "status": "failed",
                "ended_at": clock(),
                "exit_code": exit_code,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if reused:
            failure["reused"] = True
        _append_status(status_path, failure)
        raise


def _base_status_row(
    stage: _Stage,
    *,
    config: QueueConfig,
    started_at: str,
    log_path: Path | None,
    runtime_guard: _RuntimeSnapshotGuard | None,
) -> dict[str, object]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "config_path": str(config.source_path.resolve()),
        "config_sha256": config.source_sha256,
        "stage_fingerprint": _stage_fingerprint(stage, runtime_guard=runtime_guard),
        "component_script": _component_script_binding(stage, runtime_guard=runtime_guard),
        "stage": stage.key,
        "model_id": stage.model_id,
        "command": list(stage.command),
        "started_at": started_at,
        "ended_at": None,
        "exit_code": None,
        "log_path": str(log_path) if log_path is not None else None,
        "outputs": [],
        "error": None,
    }


def _validated_output_snapshot(stage: _Stage) -> list[dict[str, object]]:
    missing = [path for path in stage.required_outputs if not path.is_file()]
    if missing:
        raise ValueError(f"stage {stage.key} did not produce required output: {missing[0]}")
    return [_snapshot_path(path) for path in stage.output_roots]


def _snapshot_path(path: Path) -> dict[str, object]:
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "kind": "file",
            "files": {".": sha256_file(path)},
        }
    if not path.is_dir():
        raise FileNotFoundError(f"output root does not exist: {path}")
    files = {
        child.relative_to(path).as_posix(): sha256_file(child)
        for child in sorted(path.rglob("*"))
        if child.is_file()
    }
    if not files:
        raise ValueError(f"output directory contains no files: {path}")
    return {"path": str(path.resolve()), "kind": "directory", "files": files}


def _has_reusable_stage_success(
    rows: Sequence[Mapping[str, object]],
    *,
    stage: _Stage,
    runtime_guard: _RuntimeSnapshotGuard | None = None,
) -> bool:
    stage_fingerprint = _stage_fingerprint(stage, runtime_guard=runtime_guard)
    for row in reversed(rows):
        if row.get("stage_fingerprint") != stage_fingerprint:
            continue
        recorded_outputs = row.get("outputs")
        if not isinstance(recorded_outputs, list):
            continue
        current = _validated_output_snapshot(stage)
        if current != recorded_outputs:
            raise ValueError(f"recorded successful outputs changed for stage {stage.key}")
        return True
    return False


def _successful_rows(
    status_path: Path,
) -> dict[str, tuple[Mapping[str, object], ...]]:
    if not status_path.exists():
        return {}
    rows: dict[str, list[Mapping[str, object]]] = {}
    for line_number, line in enumerate(status_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid queue status JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"queue status line {line_number} must be an object")
        stage = row.get("stage")
        if (
            row.get("event") == "finished"
            and row.get("status") == "success"
            and isinstance(stage, str)
        ):
            rows.setdefault(stage, []).append(row)
    return {stage: tuple(stage_rows) for stage, stage_rows in rows.items()}


def _stage_fingerprint(
    stage: _Stage,
    *,
    runtime_guard: _RuntimeSnapshotGuard | None = None,
) -> str:
    inputs = [
        {
            "path": str(path.resolve()),
            "sha256": _bound_sha256(path, runtime_guard=runtime_guard),
        }
        for path in stage.input_files
    ]
    payload = {
        "stage": stage.key,
        "model_id": stage.model_id,
        "command": stage.command,
        "collision_paths": [str(path.resolve()) for path in stage.collision_paths],
        "input_files": inputs,
        "output_roots": [str(path.resolve()) for path in stage.output_roots],
        "required_outputs": [str(path.resolve()) for path in stage.required_outputs],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _component_script_binding(
    stage: _Stage,
    *,
    runtime_guard: _RuntimeSnapshotGuard | None = None,
) -> dict[str, str] | None:
    if len(stage.command) < MIN_COMPONENT_COMMAND_PARTS:
        return None
    path = Path(stage.command[1])
    return {
        "path": str(path.resolve()),
        "sha256": _bound_sha256(path, runtime_guard=runtime_guard),
    }


def _append_status(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _run_subprocess(command: tuple[str, ...], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        completed = subprocess.run(  # noqa: S603 - command is a fixed pipeline derived from operator config.
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def _record_result(
    key: str,
    status: str,
    *,
    succeeded: list[str],
    skipped: list[str],
    failed: list[str],
) -> None:
    if status == "success":
        succeeded.append(key)
    elif status == "skipped":
        skipped.append(key)
    else:
        failed.append(key)


def _object(value: object, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{source} must be an object")
    return value


def _not_none(value: Path | None) -> Path:
    if value is None:
        raise TypeError("required output path is missing")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - remote runtime is Python 3.10.


def _parse_model(raw: object, *, base: Path) -> ModelEvaluation:
    if not isinstance(raw, dict):
        raise TypeError("each model evaluation must be an object")
    model_id = _required_string(raw, "model_id")
    reuse_raw = raw.get("reuse")
    if reuse_raw is not None:
        if not isinstance(reuse_raw, dict):
            raise TypeError(f"reuse for {model_id} must be an object")
        forbidden = {
            "generation_dir",
            "analysis_dir",
            "metrics_dir",
            "evaluation_dir",
        }.intersection(raw)
        if forbidden:
            message = f"reused model {model_id} cannot define queue output directories"
            raise ValueError(message)
        reuse = ReusedEvaluation(
            generation_dir=_resolve(base, _required_string(reuse_raw, "generation_dir")),
            analysis_dir=_resolve(base, _required_string(reuse_raw, "analysis_dir")),
            metrics_results=_resolve(base, _required_string(reuse_raw, "metrics_results")),
            metrics_provenance=_resolve(
                base,
                _required_string(reuse_raw, "metrics_provenance"),
            ),
            evaluation_manifest=_resolve(
                base,
                _required_string(reuse_raw, "evaluation_manifest"),
            ),
            evaluation_dir=_resolve(base, _required_string(reuse_raw, "evaluation_dir")),
        )
        generation_dir = analysis_dir = metrics_dir = evaluation_dir = None
    else:
        reuse = None
        generation_dir = _resolve(base, _required_string(raw, "generation_dir"))
        analysis_dir = _resolve(base, _required_string(raw, "analysis_dir"))
        metrics_dir = _resolve(base, _required_string(raw, "metrics_dir"))
        evaluation_dir = _resolve(base, _required_string(raw, "evaluation_dir"))
    return ModelEvaluation(
        model_id=model_id,
        reference_wavs=_resolve(base, _required_string(raw, "reference_wavs")),
        generation_dir=generation_dir,
        analysis_dir=analysis_dir,
        metrics_dir=metrics_dir,
        evaluation_dir=evaluation_dir,
        reuse=reuse,
    )


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return document


def _required_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"field {key!r} must be an object")
    return value


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a nonempty string")
    return value


def _required_sha256(row: Mapping[str, Any], key: str) -> str:
    value = _required_string(row, key)
    if len(value) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"field {key!r} must be a lowercase SHA-256")
    return value


def _resolve(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base / path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).parent)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner | None = None,
) -> int:
    args = _parse_args(argv)
    result = run_evaluation_queue(
        load_queue_config(args.config),
        status_path=args.status_path,
        scripts_dir=args.scripts_dir,
        runner=runner,
    )
    print(
        json.dumps(
            {
                "succeeded": result.succeeded,
                "skipped": result.skipped,
                "reused": result.reused,
                "failed": result.failed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return int(bool(result.failed))


if __name__ == "__main__":
    raise SystemExit(main())
