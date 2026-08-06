# ruff: noqa: EM101, EM102, PLR0913, PLR0914, PLR2004, TRY003
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

DERIVATION_SCHEMA = "speaker-derived-diagnostic-derivation/v1"
MANIFEST_SCHEMA = "speaker-derived-diagnostic-manifest/v1"
DIAGNOSTIC_KIND = "derived diagnostic embedding"
FORMULA = "M = 0.5 * P + 0.5 * F"
ALPHA = 0.5
CASE_IDENTITY_STEP = 0
EMBEDDING_SHAPE = (16, 768)
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
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SEARCH_VALIDATOR = Path(__file__).with_name("generate_600m_speaker_checkpoint_search_remote.py")
SEARCH_BUILDER = Path(__file__).with_name("build_600m_speaker_checkpoint_search_manifest.py")
PRODUCTION_GENERATOR = Path(__file__).with_name("generate_600m_checkpoint_audio_remote.py")


def _source_validator() -> ModuleType:
    name = "_speaker_midpoint_source_validator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, SEARCH_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source validator: {SEARCH_VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_embedding(path: Path) -> np.ndarray:
    return _decode_embedding(_snapshot_regular_file(path, source="speaker embedding")[0])


def derive_midpoint(
    *,
    source_manifest: Path,
    source_manifest_sha256: str,
    original_embedding: Path,
    original_embedding_sha256: str,
    candidate_f_embedding: Path,
    candidate_f_embedding_sha256: str,
    model_id: str,
    output_root: Path,
    version: str,
) -> dict[str, object]:
    if not model_id:
        raise ValueError("model_id must be nonempty")
    if not VERSION_RE.fullmatch(version) or version in {".", ".."}:
        raise ValueError("version must be a safe nonempty direct-child name")

    builder_path = Path(__file__).resolve(strict=True)
    builder_snapshot = _snapshot_regular_file(builder_path, source="midpoint builder")
    dependency_snapshots = _snapshot_dependencies()
    source_path, source_snapshot = _snapshot_bound_file(
        source_manifest,
        source_manifest_sha256,
        source="source evaluation manifest",
    )
    original_path, original_snapshot = _snapshot_bound_file(
        original_embedding,
        original_embedding_sha256,
        source="original step1000 embedding",
    )
    candidate_path, candidate_snapshot = _snapshot_bound_file(
        candidate_f_embedding,
        candidate_f_embedding_sha256,
        source="candidate F embedding",
    )

    source_payload = _read_json_bytes(source_snapshot[0], source=source_path)
    lineage = _validate_search_parent_lineage(
        source_path=source_path,
        source_payload=source_payload,
        model_id=model_id,
        original_path=original_path,
        original_sha256=original_snapshot[1],
        candidate_path=candidate_path,
        candidate_sha256=candidate_snapshot[1],
    )
    original_values = _decode_embedding(original_snapshot[0])
    candidate_values = _decode_embedding(candidate_snapshot[0])
    midpoint = (original_values * np.float32(ALPHA) + candidate_values * np.float32(ALPHA)).astype(
        "<f4", copy=False
    )
    if midpoint.shape != EMBEDDING_SHAPE or not np.isfinite(midpoint).all():
        raise ValueError("derived midpoint must be finite F32[16,768]")
    embedding_bytes = _encode_embedding(midpoint)
    embedding_sha = sha256_bytes(embedding_bytes)

    root = _require_existing_nominal_directory(output_root, source="output root")
    output = Path(os.path.abspath(root / version))  # noqa: PTH100
    if output.parent != root:
        raise ValueError("midpoint output must be a direct child of output root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite midpoint diagnostic: {output}")

    source_checkpoint = _required_mapping(source_payload, "checkpoint")
    metrics = source_payload["metrics_provenance"]
    embedding_path = output / "derived-midpoint.speaker.safetensors"
    derivation_path = output / "derivation.json"
    manifest_path = output / "diagnostic-manifest.json"
    derivation: dict[str, object] = {
        "schema_version": DERIVATION_SCHEMA,
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "model_id": model_id,
        "alpha": ALPHA,
        "formula": FORMULA,
        "normalization": "none",
        "arithmetic": "CPU NumPy elementwise float32",
        "dtype": "F32",
        "shape": list(EMBEDDING_SHAPE),
        "finite": True,
        "parents": {
            "original_step1000": {
                "path": str(original_path),
                "sha256": original_snapshot[1],
            },
            "candidate_f": {
                "path": str(candidate_path),
                "sha256": candidate_snapshot[1],
            },
        },
        "output_embedding": {
            "path": str(embedding_path),
            "sha256": embedding_sha,
        },
        "source_evaluation_manifest": {
            "path": str(source_path),
            "sha256": source_snapshot[1],
        },
        "validated_parent_lineage": lineage,
        "base_model": {
            "model_id": source_checkpoint["base_checkpoint"],
            "revision": source_checkpoint["base_revision"],
            "checkpoint_sha256": source_checkpoint["base_checkpoint_sha256"],
        },
        "metrics_provenance": metrics,
        "builder_script": {
            "path": str(builder_path),
            "sha256": builder_snapshot[1],
        },
        "validator_dependencies": [
            {"path": str(path), "sha256": snapshot[1]} for path, snapshot in dependency_snapshots
        ],
    }
    derivation_bytes = _json_bytes(derivation)
    derivation_sha = sha256_bytes(derivation_bytes)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "model_id": model_id,
        "checkpoint_step": CASE_IDENTITY_STEP,
        "checkpoint_step_semantics": "synthetic case identity only; not a training step",
        "derived_embedding": {
            "path": str(embedding_path),
            "sha256": embedding_sha,
            "dtype": "F32",
            "shape": list(EMBEDDING_SHAPE),
            "finite": True,
        },
        "derivation": {"path": str(derivation_path), "sha256": derivation_sha},
        "parents": derivation["parents"],
        "source_evaluation_manifest": derivation["source_evaluation_manifest"],
        "validated_parent_lineage": derivation["validated_parent_lineage"],
        "base_model": derivation["base_model"],
        "metrics_provenance": metrics,
        "text_ids": list(TEXT_IDS),
        "seeds": list(SEEDS),
        "styles": list(STYLES),
        "builder_script": derivation["builder_script"],
        "validator_dependencies": derivation["validator_dependencies"],
        "manifest_path": str(manifest_path),
    }
    manifest_bytes = _json_bytes(manifest)

    output.mkdir(exist_ok=False)
    _require_nominal_directory(output, source="midpoint diagnostic output")
    _write_exclusive(embedding_path, embedding_bytes)
    _write_exclusive(derivation_path, derivation_bytes)
    _write_exclusive(manifest_path, manifest_bytes)

    for path, snapshot, source in (
        (source_path, source_snapshot, "source evaluation manifest"),
        (original_path, original_snapshot, "original step1000 embedding"),
        (candidate_path, candidate_snapshot, "candidate F embedding"),
        (builder_path, builder_snapshot, "midpoint builder"),
        *(
            (path, snapshot, f"validator dependency {path.name}")
            for path, snapshot in dependency_snapshots
        ),
    ):
        _validate_snapshot_unchanged(path, snapshot, source=source)
    for path, expected, source in (
        (embedding_path, embedding_sha, "derived embedding"),
        (derivation_path, derivation_sha, "derivation"),
        (manifest_path, sha256_bytes(manifest_bytes), "diagnostic manifest"),
    ):
        _validate_published(path, expected, source=source)
    return manifest


def validate_derived_manifest(
    path: Path,
    *,
    snapshot: tuple[bytes, str] | None = None,
) -> dict[str, Any]:
    manifest_path = _require_regular_direct_file(path, source="derived diagnostic manifest")
    manifest_snapshot = snapshot or _snapshot_regular_file(
        manifest_path, source="derived diagnostic manifest"
    )
    payload = _read_json_bytes(manifest_snapshot[0], source=manifest_path)
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"derived manifest requires schema_version {MANIFEST_SCHEMA}")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "diagnostic_kind",
            "model_id",
            "checkpoint_step",
            "checkpoint_step_semantics",
            "derived_embedding",
            "derivation",
            "parents",
            "source_evaluation_manifest",
            "validated_parent_lineage",
            "base_model",
            "metrics_provenance",
            "text_ids",
            "seeds",
            "styles",
            "builder_script",
            "validator_dependencies",
            "manifest_path",
        },
        source="derived diagnostic manifest",
    )
    if payload.get("diagnostic_kind") != DIAGNOSTIC_KIND:
        raise ValueError("derived manifest diagnostic_kind mismatch")
    if payload.get("checkpoint_step") != CASE_IDENTITY_STEP:
        raise ValueError("derived manifest checkpoint_step must be synthetic step 0")
    if payload.get("checkpoint_step_semantics") != (
        "synthetic case identity only; not a training step"
    ):
        raise ValueError("derived manifest step semantics mismatch")
    for field, expected in (("text_ids", TEXT_IDS), ("seeds", SEEDS), ("styles", STYLES)):
        if tuple(payload.get(field, ())) != expected:
            raise ValueError(f"derived manifest {field} mismatch")
    if payload.get("manifest_path") != str(manifest_path):
        raise ValueError("derived manifest path binding mismatch")

    derivation_row = _required_mapping(payload, "derivation")
    derivation_path, derivation_snapshot = _snapshot_bound_file(
        _resolved_path(derivation_row, "path", manifest_path.parent),
        _required_sha(derivation_row, "sha256"),
        source="derivation",
    )
    derivation = _read_json_bytes(derivation_snapshot[0], source=derivation_path)
    _validate_derivation(derivation, manifest=payload, manifest_path=manifest_path)
    embedding_row = _required_mapping(payload, "derived_embedding")
    embedding_path, embedding_snapshot = _snapshot_bound_file(
        _resolved_path(embedding_row, "path", manifest_path.parent),
        _required_sha(embedding_row, "sha256"),
        source="derived embedding",
    )
    _decode_embedding(embedding_snapshot[0])
    parents = _required_mapping(payload, "parents")
    parent_values = []
    parent_snapshots: list[tuple[Path, tuple[bytes, str], str]] = []
    for name in ("original_step1000", "candidate_f"):
        row = _required_mapping(parents, name)
        parent_path, parent_snapshot = _snapshot_bound_file(
            _resolved_path(row, "path", manifest_path.parent),
            _required_sha(row, "sha256"),
            source=f"{name} parent",
        )
        parent_values.append(_decode_embedding(parent_snapshot[0]))
        parent_snapshots.append((parent_path, parent_snapshot, f"{name} parent"))
    expected_midpoint = (
        parent_values[0] * np.float32(ALPHA) + parent_values[1] * np.float32(ALPHA)
    ).astype("<f4", copy=False)
    actual = _decode_embedding(embedding_snapshot[0])
    if not np.array_equal(actual, expected_midpoint):
        raise ValueError("derived embedding does not exactly equal the declared midpoint")

    source_row = _required_mapping(payload, "source_evaluation_manifest")
    source_path, source_snapshot = _snapshot_bound_file(
        _resolved_path(source_row, "path", manifest_path.parent),
        _required_sha(source_row, "sha256"),
        source="source evaluation manifest",
    )
    source_payload = _read_json_bytes(source_snapshot[0], source=source_path)
    model_id = _required_string(payload, "model_id")
    lineage = _validate_search_parent_lineage(
        source_path=source_path,
        source_payload=source_payload,
        model_id=model_id,
        original_path=parent_snapshots[0][0],
        original_sha256=parent_snapshots[0][1][1],
        candidate_path=parent_snapshots[1][0],
        candidate_sha256=parent_snapshots[1][1][1],
    )
    if payload.get("validated_parent_lineage") != lineage:
        raise ValueError("validated parent lineage does not match source evidence")
    _validate_source_binding(payload, source_payload)
    for check_path, check_snapshot, source in (
        (manifest_path, manifest_snapshot, "derived diagnostic manifest"),
        (derivation_path, derivation_snapshot, "derivation"),
        (embedding_path, embedding_snapshot, "derived embedding"),
        (source_path, source_snapshot, "source evaluation manifest"),
        *parent_snapshots,
    ):
        _validate_snapshot_unchanged(check_path, check_snapshot, source=source)
    return payload


def _validate_derivation(
    derivation: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_path: Path,
) -> None:
    expected = {
        "schema_version": DERIVATION_SCHEMA,
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "model_id": manifest.get("model_id"),
        "alpha": ALPHA,
        "formula": FORMULA,
        "normalization": "none",
        "arithmetic": "CPU NumPy elementwise float32",
        "dtype": "F32",
        "shape": list(EMBEDDING_SHAPE),
        "finite": True,
        "parents": manifest.get("parents"),
        "output_embedding": {
            "path": _required_mapping(manifest, "derived_embedding").get("path"),
            "sha256": _required_mapping(manifest, "derived_embedding").get("sha256"),
        },
        "source_evaluation_manifest": manifest.get("source_evaluation_manifest"),
        "validated_parent_lineage": manifest.get("validated_parent_lineage"),
        "base_model": manifest.get("base_model"),
        "metrics_provenance": manifest.get("metrics_provenance"),
        "builder_script": manifest.get("builder_script"),
        "validator_dependencies": manifest.get("validator_dependencies"),
    }
    if derivation != expected:
        raise ValueError("derivation contract mismatch")
    builder = _required_mapping(derivation, "builder_script")
    builder_path, builder_snapshot = _snapshot_bound_file(
        _resolved_path(builder, "path", manifest_path.parent),
        _required_sha(builder, "sha256"),
        source="midpoint builder",
    )
    _validate_snapshot_unchanged(builder_path, builder_snapshot, source="midpoint builder")
    dependencies = derivation.get("validator_dependencies")
    if not isinstance(dependencies, list) or len(dependencies) != 3:
        raise ValueError("validator_dependencies must contain the exact three dependencies")
    expected_paths = (SEARCH_VALIDATOR, SEARCH_BUILDER, PRODUCTION_GENERATOR)
    for row, expected_path in zip(dependencies, expected_paths, strict=True):
        if not isinstance(row, dict):
            raise TypeError("validator dependency must be an object")
        path, snapshot = _snapshot_bound_file(
            _resolved_path(row, "path", manifest_path.parent),
            _required_sha(row, "sha256"),
            source="validator dependency",
        )
        if path != expected_path.resolve():
            raise ValueError("validator dependency path mismatch")
        _validate_snapshot_unchanged(path, snapshot, source="validator dependency")


def _validate_source_dimensions(source: Mapping[str, object]) -> None:
    for field, expected in (("text_ids", TEXT_IDS), ("seeds", SEEDS), ("styles", STYLES)):
        actual = source.get(field)
        if not isinstance(actual, list) or tuple(actual) != expected:
            raise ValueError(f"source evaluation manifest {field} mismatch")


def _validate_source_binding(derived: Mapping[str, object], source: Mapping[str, Any]) -> None:
    _validate_source_dimensions(source)
    checkpoint = _required_mapping(source, "checkpoint")
    expected_base = {
        "model_id": checkpoint["base_checkpoint"],
        "revision": checkpoint["base_revision"],
        "checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
    }
    if derived.get("base_model") != expected_base:
        raise ValueError("derived base model lineage does not match source")
    if derived.get("metrics_provenance") != source.get("metrics_provenance"):
        raise ValueError("derived metrics provenance does not match source")


def _validate_search_parent_lineage(
    *,
    source_path: Path,
    source_payload: Mapping[str, Any],
    model_id: str,
    original_path: Path,
    original_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
) -> dict[str, object]:
    plan = _source_validator().load_search_plan(source_path)
    if plan.model_id != model_id or len(plan.checkpoints) != 1:
        raise ValueError("source search manifest model/candidate mismatch")
    candidate = plan.checkpoints[0]
    if candidate.embedding_path != candidate_path or candidate.embedding_sha256 != candidate_sha256:
        raise ValueError("candidate F does not match the verified search manifest checkpoint")
    _validate_source_dimensions(source_payload)

    raw_search_source = _required_mapping(source_payload, "source_evaluation_manifest")
    search_source_path, search_source_snapshot = _snapshot_bound_file(
        _resolved_path(raw_search_source, "path", source_path.parent),
        _required_sha(raw_search_source, "sha256"),
        source="search source evaluation manifest",
    )
    _read_json_bytes(search_source_snapshot[0], source=search_source_path)

    evidence_row = _required_mapping(source_payload, "training_run_evidence")
    evidence_path, evidence_snapshot = _snapshot_bound_file(
        _resolved_path(evidence_row, "path", source_path.parent),
        _required_sha(evidence_row, "sha256"),
        source="candidate F training run evidence",
    )
    evidence = _read_json_bytes(evidence_snapshot[0], source=evidence_path)
    setup_row = _required_mapping(evidence, "setup_evidence")
    setup_path, setup_snapshot = _snapshot_bound_file(
        _resolved_path(setup_row, "path", evidence_path.parent),
        _required_sha(setup_row, "sha256"),
        source="candidate F setup evidence",
    )
    setup = _read_json_bytes(setup_snapshot[0], source=setup_path)
    if setup.get("schema_version") != "speaker-quality-search-setup/v1":
        raise ValueError("candidate F setup evidence schema mismatch")
    setup_candidate = _required_mapping(setup, "candidate")
    if setup_candidate.get("init_label") != "original1000":
        raise ValueError("candidate F setup init_label must be original1000")
    if original_path.name != "checkpoint_0001000.speaker.safetensors":
        raise ValueError(
            "original1000 embedding must be named checkpoint_0001000.speaker.safetensors"
        )
    setup_init_path = _resolved_path(
        setup_candidate, "speaker_inversion_init_embedding", setup_path.parent
    ).resolve()
    setup_init_sha = _required_sha(setup_candidate, "speaker_inversion_init_embedding_sha256")
    if setup_init_path != original_path or setup_init_sha != original_sha256:
        raise ValueError("candidate F setup evidence does not bind original step1000")

    checkpoint_row = _required_mapping(source_payload, "checkpoint")
    config_path, config_snapshot = _snapshot_bound_file(
        _resolved_path(checkpoint_row, "training_config_path", source_path.parent),
        _required_sha(checkpoint_row, "training_config_sha256"),
        source="candidate F training config",
    )
    config = _read_json_bytes(config_snapshot[0], source=config_path)
    train = _required_mapping(config, "train")
    configured_init = _resolved_path(
        train, "speaker_inversion_init_embedding", config_path.parent
    ).resolve()
    if configured_init != original_path:
        raise ValueError("candidate F training config does not bind original step1000")
    configured_sha = train.get("speaker_inversion_init_embedding_sha256")
    if configured_sha is not None and configured_sha != original_sha256:
        raise ValueError("candidate F training config init embedding SHA mismatch")

    for path, snapshot, source in (
        (
            source_path,
            _snapshot_regular_file(source_path, source="source search manifest"),
            "source search manifest",
        ),
        (search_source_path, search_source_snapshot, "search source evaluation manifest"),
        (evidence_path, evidence_snapshot, "candidate F training run evidence"),
        (setup_path, setup_snapshot, "candidate F setup evidence"),
        (config_path, config_snapshot, "candidate F training config"),
    ):
        _validate_snapshot_unchanged(path, snapshot, source=source)
    return {
        "search_source_evaluation_manifest": {
            "path": str(search_source_path),
            "sha256": search_source_snapshot[1],
        },
        "candidate_f_training_run_evidence": {
            "path": str(evidence_path),
            "sha256": evidence_snapshot[1],
        },
        "candidate_f_setup_evidence": {
            "path": str(setup_path),
            "sha256": setup_snapshot[1],
        },
        "candidate_f_training_config": {
            "path": str(config_path),
            "sha256": config_snapshot[1],
        },
    }


def _snapshot_dependencies() -> tuple[tuple[Path, tuple[bytes, str]], ...]:
    return tuple(
        (
            path.resolve(strict=True),
            _snapshot_regular_file(path, source=f"validator dependency {path.name}"),
        )
        for path in (SEARCH_VALIDATOR, SEARCH_BUILDER, PRODUCTION_GENERATOR)
    )


def _decode_embedding(payload: bytes) -> np.ndarray:
    if len(payload) < 8:
        raise ValueError("embedding safetensors header is truncated")
    header_length = struct.unpack("<Q", payload[:8])[0]
    if header_length <= 0 or 8 + header_length > len(payload):
        raise ValueError("embedding safetensors header size is invalid")
    try:
        header = json.loads(payload[8 : 8 + header_length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("embedding safetensors header is invalid") from exc
    _validate_safetensors_header_keys(header)
    return _decode_embedding_tensor(header, payload, header_length)


def _validate_safetensors_header_keys(header: object) -> None:
    if not isinstance(header, dict):
        raise TypeError("safetensors header must be an object")
    if (
        not set(header) <= {"__metadata__", "speaker_embedding"}
        or "speaker_embedding" not in header
    ):
        raise ValueError("safetensors must contain exactly one speaker_embedding tensor")
    metadata = header.get("__metadata__")
    if "__metadata__" in header and (
        not isinstance(metadata, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
        )
    ):
        raise ValueError("safetensors __metadata__ must be a mapping of strings to strings")


def _decode_embedding_tensor(
    header: Mapping[str, object], payload: bytes, header_length: int
) -> np.ndarray:
    tensor = header["speaker_embedding"]
    if not isinstance(tensor, dict):
        raise TypeError("speaker_embedding tensor must be an object")
    if tensor.get("dtype") != "F32":
        raise ValueError("speaker_embedding must be F32")
    if tensor.get("shape") != list(EMBEDDING_SHAPE):
        raise ValueError(f"speaker_embedding shape must be {EMBEDDING_SHAPE}")
    expected_bytes = math.prod(EMBEDDING_SHAPE) * 4
    if tensor.get("data_offsets") != [0, expected_bytes]:
        raise ValueError("speaker_embedding offsets are invalid")
    raw = payload[8 + header_length :]
    if len(raw) != expected_bytes:
        raise ValueError("speaker_embedding payload size is invalid")
    values = np.frombuffer(raw, dtype="<f4").reshape(EMBEDDING_SHAPE).copy()
    if not np.isfinite(values).all():
        raise ValueError("speaker_embedding must contain only finite values")
    return values


def _encode_embedding(values: np.ndarray) -> bytes:
    raw = np.asarray(values, dtype="<f4").tobytes(order="C")
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": "F32",
                "shape": list(EMBEDDING_SHAPE),
                "data_offsets": [0, len(raw)],
            }
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + raw


def _snapshot_bound_file(
    path: Path, expected: str, *, source: str
) -> tuple[Path, tuple[bytes, str]]:
    _required_sha({"sha": expected}, "sha")
    resolved = _require_regular_direct_file(path, source=source)
    snapshot = _snapshot_regular_file(resolved, source=source)
    if snapshot[1] != expected:
        raise ValueError(f"{source} SHA-256 mismatch: {resolved}")
    return resolved, snapshot


def _require_regular_direct_file(path: Path, *, source: str) -> Path:
    nominal = Path(os.path.abspath(path))  # noqa: PTH100
    for candidate in (nominal, *nominal.parents):
        if candidate == candidate.parent:
            break
        _require_no_alias(candidate, source=source)
    try:
        metadata = nominal.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{source} must exist: {nominal}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{source} must be a regular non-alias file: {nominal}")
    return nominal.resolve(strict=True)


def _require_no_alias(path: Path, *, source: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{source} must not use a symlink, junction, or reparse alias: {path}")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse and getattr(metadata, "st_file_attributes", 0) & reparse:
        raise ValueError(f"{source} must not use a symlink, junction, or reparse alias: {path}")


def _require_existing_nominal_directory(path: Path, *, source: str) -> Path:
    nominal = Path(os.path.abspath(path))  # noqa: PTH100
    for candidate in (nominal, *nominal.parents):
        if candidate == candidate.parent:
            break
        _require_no_alias(candidate, source=source)
    if not nominal.is_dir():
        raise ValueError(f"{source} must be an existing directory: {nominal}")
    return nominal.resolve(strict=True)


def _require_nominal_directory(path: Path, *, source: str) -> None:
    _require_no_alias(path, source=source)
    if not path.is_dir():
        raise ValueError(f"{source} must be a directory: {path}")


def _snapshot_regular_file(path: Path, *, source: str) -> tuple[bytes, str]:
    resolved = _require_regular_direct_file(path, source=source)
    payload = resolved.read_bytes()
    return payload, sha256_bytes(payload)


def _validate_snapshot_unchanged(path: Path, snapshot: tuple[bytes, str], *, source: str) -> None:
    current = _snapshot_regular_file(path, source=source)
    if current != snapshot:
        raise ValueError(f"{source} changed after input snapshot: {path}")


def _validate_published(path: Path, expected: str, *, source: str) -> None:
    current = _snapshot_regular_file(path, source=source)
    if current[1] != expected:
        raise ValueError(f"{source} changed during publication: {path}")


def _resolved_path(row: Mapping[str, object], field: str, base: Path) -> Path:
    value = _required_string(row, field)
    path = Path(value)
    return path if path.is_absolute() else base / path


def _required_mapping(row: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = row.get(field)
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _require_exact_keys(row: Mapping[str, object], expected: set[str], *, source: str) -> None:
    if set(row) != expected:
        raise ValueError(f"{source} keys must exactly match {sorted(expected)}")


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _required_sha(row: Mapping[str, object], field: str) -> str:
    value = _required_string(row, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_json_bytes(payload: bytes, *, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must be an object: {source}")
    return value


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as destination:
        destination.write(payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--original-embedding", type=Path, required=True)
    parser.add_argument("--original-embedding-sha256", required=True)
    parser.add_argument("--candidate-f-embedding", type=Path, required=True)
    parser.add_argument("--candidate-f-embedding-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = derive_midpoint(
        source_manifest=args.source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        original_embedding=args.original_embedding,
        original_embedding_sha256=args.original_embedding_sha256,
        candidate_f_embedding=args.candidate_f_embedding,
        candidate_f_embedding_sha256=args.candidate_f_embedding_sha256,
        model_id=args.model_id,
        output_root=args.output_root,
        version=args.version,
    )
    print(json.dumps({"manifest": result["manifest_path"], "model_id": result["model_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
