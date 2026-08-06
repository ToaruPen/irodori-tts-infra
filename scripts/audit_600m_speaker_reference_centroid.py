# ruff: noqa: C901, EM101, PLR0912, PLR0913, PLR0914, PLR0915, SLF001, TRY003
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

SCHEMA_VERSION = "speaker-reference-centroid-audit/v1"
REFERENCE_SCHEMA_VERSION = "speaker-similarity-references/v1"
METRICS_PROVENANCE_SCHEMA_VERSION = "speaker-metrics-extraction/v1"
SEARCH_GENERATION_CASE_SCHEMA = "speaker-checkpoint-search-generation-case/v1"
FULL_MATRIX_CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
FULL_MATRIX_TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
EXPECTED_REFERENCE_COUNT = 25
EXPECTED_QUANTILES = (1, 2, 3, 4, 5)
EXPECTED_REFERENCES_PER_QUANTILE = 5
METRIC_GATE_TEXT_IDS = (
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
EXPECTED_SEEDS = (1234, 5678)
EXPECTED_STYLES = ("neutral", "calm")
EXPECTED_METRIC_GATE_CASE_COUNT = 16
SIMILARITY_TOLERANCE = 1e-8
SHA256_HEX_LENGTH = 64
OUTLIER_MAD_MULTIPLIER = 3.0
SOURCE_HASH_FIELDS = (
    "round4_clean_dataset_sha256",
    "round4_decisions_sha256",
    "round4_summary_sha256",
    "training_clean_manifest_sha256",
    "training_latent_provenance_sha256",
    "training_provenance_sha256",
)
REFERENCE_IDENTITY_FIELDS = ("audio_sha256", "pcm_sha256", "text")
GENERATION_IDENTITY_FIELDS = (
    "case_id",
    "model_id",
    "checkpoint_step",
    "checkpoint",
    "speaker_filename",
    "embedding_path",
    "embedding_sha256",
    "evaluation_manifest_sha256",
    "base_checkpoint_sha256",
    "text_id",
    "seed",
    "style",
    "wav_path",
    "wav_sha256",
    "provenance",
)
NONVERBAL_TEXT_MARKERS = (
    "吐息",
    "喘ぎ",
    "あえぎ",
    "ため息",
    "息遣い",
    "笑い声",
    "泣き声",
    "うめき声",
    "呻き声",
    "咳",
    "くしゃみ",
    "呼吸音",
)
METRICS_SCRIPT = Path(__file__).with_name("compute_600m_speaker_metrics.py")


class SpeakerEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def source_sha256(self) -> str: ...

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


def _load_metrics_module() -> ModuleType:
    module_name = "_speaker_metrics_for_reference_centroid_audit"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, METRICS_SCRIPT)
    if spec is None or spec.loader is None:
        message = f"cannot load speaker metrics helpers: {METRICS_SCRIPT}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


metrics = _load_metrics_module()


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    name: str
    path: Path
    data: bytes
    sha256: str


def sha256_file(path: Path) -> str:
    return cast("str", metrics.sha256_file(path))


def run_audit(
    *,
    reference_wavs_path: Path,
    clean_manifest_path: Path,
    generation_results_path: Path,
    metrics_results_path: Path,
    metrics_provenance_path: Path,
    output_path: Path,
    ecapa_source: Path,
    embedder: SpeakerEmbedder,
    checkpoint_step: int | None = None,
) -> dict[str, object]:
    inputs = {
        "reference_wavs": _snapshot_file(
            "reference_wavs",
            reference_wavs_path,
            source="reference manifest",
        ),
        "clean_manifest": _snapshot_file(
            "clean_manifest",
            clean_manifest_path,
            source="clean manifest",
        ),
        "generation_results": _snapshot_file(
            "generation_results",
            generation_results_path,
            source="generation results",
        ),
        "metrics_results": _snapshot_file(
            "metrics_results",
            metrics_results_path,
            source="metrics results",
        ),
        "metrics_provenance": _snapshot_file(
            "metrics_provenance",
            metrics_provenance_path,
            source="metrics provenance",
        ),
    }
    audit_script = _snapshot_file("script", Path(__file__), source="audit script")
    metrics_script = _snapshot_file(
        "metrics_helper_script",
        METRICS_SCRIPT,
        source="metrics helper script",
    )
    ecapa_source = ecapa_source.resolve(strict=True)
    actual_ecapa_sha256 = cast("str", metrics.sha256_tree(ecapa_source))
    if embedder.source_sha256 != actual_ecapa_sha256:
        message = "ECAPA source SHA-256 does not match loaded embedder"
        raise ValueError(message)

    clean_rows = _load_clean_manifest(inputs["clean_manifest"])
    reference_payload = _read_json_snapshot(inputs["reference_wavs"])
    model_id, model_prefix, reference_rows = _validate_reference_manifest(
        reference_payload,
        manifest_path=inputs["reference_wavs"].path,
        clean_manifest_path=inputs["clean_manifest"].path,
        clean_manifest_sha256=inputs["clean_manifest"].sha256,
        clean_rows=clean_rows,
    )
    generation_rows = _load_generation_rows(inputs["generation_results"])
    metrics._validate_generation_audio(
        generation_rows,
        base=inputs["generation_results"].path.parent,
    )
    gate_rows, checkpoint_run_identity, generation_selection = _validate_generation_matrix(
        generation_rows,
        model_id=model_id,
        checkpoint_step=checkpoint_step,
    )
    metric_rows = _load_jsonl_snapshot(inputs["metrics_results"], source="metrics results")
    metric_rows_by_id = _index_rows(metric_rows, source="metrics results")
    metrics_provenance = _read_json_snapshot(inputs["metrics_provenance"])
    _validate_metrics_provenance(
        metrics_provenance,
        generation_results_path=inputs["generation_results"].path,
        generation_results_sha256=inputs["generation_results"].sha256,
        reference_wavs_path=inputs["reference_wavs"].path,
        reference_wavs_sha256=inputs["reference_wavs"].sha256,
        reference_rows=reference_rows,
        generation_rows=generation_rows,
        reference_base=inputs["reference_wavs"].path.parent,
        embedder=embedder,
        ecapa_source_sha256=actual_ecapa_sha256,
    )
    bound_audio = _capture_bound_audio_snapshots(
        reference_rows=reference_rows,
        generation_rows=generation_rows,
        generation_base=inputs["generation_results"].path.parent,
    )

    references = _embed_references(
        reference_rows,
        manifest_base=inputs["reference_wavs"].path.parent,
        embedder=embedder,
    )
    reference_embeddings = [embedding for _, _, embedding in references]
    full_centroid = metrics.aggregate_reference_centroid(reference_embeddings)

    leave_one_out_centroids = [
        metrics.aggregate_reference_centroid(
            reference_embeddings[:index] + reference_embeddings[index + 1 :],
        )
        for index in range(len(reference_embeddings))
    ]
    reference_analysis = _reference_analysis(
        references,
        full_centroid=full_centroid,
        leave_one_out_centroids=leave_one_out_centroids,
    )
    generated_analysis = _generated_analysis(
        gate_rows,
        metrics_rows=metric_rows_by_id,
        generation_base=inputs["generation_results"].path.parent,
        generation_results_sha256=inputs["generation_results"].sha256,
        embedder=embedder,
        full_centroid=full_centroid,
        leave_one_out_centroids=leave_one_out_centroids,
        reference_rows=reference_rows,
        checkpoint_run_identity=checkpoint_run_identity,
    )
    outlier_count = cast("int", reference_analysis["outlier_count"])
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_identity": {
            "model_id": model_id,
            "model_prefix": model_prefix,
            "clean_manifest_row_count": len(clean_rows),
            "reference_count": len(reference_rows),
            "source_identity_consistent": True,
        },
        "generation_selection": generation_selection,
        "identity_verification": {
            "reference_source_id_clean_manifest_occurrence": "exactly_one",
            "clean_manifest_join_fields": [
                "source_id",
                *REFERENCE_IDENTITY_FIELDS,
            ],
            "clean_manifest_audio_path_field": (
                "present_entries_verified"
                if any("audio_path" in row or "audio" in row for row in clean_rows)
                else "absent_reference_audio_path_verified_by_current_audio_sha256"
            ),
            "reference_source_audio_current_hash_verified": True,
            "reference_wav_current_hash_verified": True,
            "reference_wav_path_and_sha256_unique": True,
            "source_audio_and_pcm_identity_unique": True,
            "duration_quantile_contract_verified": True,
            "clean_manifest_common_model_prefix_verified": True,
            "optional_clean_model_id_fields_verified_when_present": True,
        },
        "summary": {
            "identity_consistent": True,
            "centroid_stable": outlier_count == 0,
            "centroid_stability_definition": (
                "true exactly when the documented untuned robust full-centroid "
                "reference rule flags no outliers"
            ),
            "reference_outlier_count": outlier_count,
            "metrics_similarity_verified": True,
            "checkpoint_run_identity": checkpoint_run_identity,
        },
        "reference_analysis": reference_analysis,
        "generated_analysis": generated_analysis,
        "provenance": {
            "script": {
                "path": str(audit_script.path),
                "sha256": audit_script.sha256,
            },
            "reused_metrics_helpers": {
                "path": str(metrics_script.path),
                "sha256": metrics_script.sha256,
            },
            "inputs": {
                name: {"path": str(snapshot.path), "sha256": snapshot.sha256}
                for name, snapshot in inputs.items()
            },
            "ecapa": {
                "source": str(ecapa_source),
                "savedir": str(getattr(embedder, "savedir", "not_exposed_by_embedder")),
                "model_id": embedder.model_id,
                "revision": embedder.revision,
                "source_sha256": actual_ecapa_sha256,
            },
            "reference_source_hashes": reference_payload["source_hashes"],
            "reference_source_hash_verification": {
                "training_clean_manifest_sha256": "verified_against_current_file",
                "other_source_hashes": "validated_sha256_declarations_only_no_paths_in_manifest",
            },
            "publication_reverification": {
                "status": "passed_before_atomic_publication",
                "explicit_input_count": len(inputs),
                "script_count": 2,
                "bound_audio_count": len(bound_audio),
                "ecapa_tree_verified": True,
            },
        },
    }
    _write_json_create_only(
        output_path,
        report,
        before_publish=lambda: _verify_audit_inputs_unchanged(
            explicit_inputs=tuple(inputs.values()),
            scripts=(audit_script, metrics_script),
            bound_audio=bound_audio,
            ecapa_source=ecapa_source,
            ecapa_source_sha256=actual_ecapa_sha256,
        ),
    )
    return report


def _snapshot_file(name: str, path: Path, *, source: str) -> FileSnapshot:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        message = f"{source} must be a regular file: {path}"
        raise ValueError(message)
    data = resolved.read_bytes()
    return FileSnapshot(
        name=name,
        path=resolved,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _read_json_snapshot(snapshot: FileSnapshot) -> dict[str, object]:
    payload: Any = json.loads(snapshot.data.decode("utf-8"))
    if not isinstance(payload, dict):
        message = f"JSON document must be an object: {snapshot.path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _load_jsonl_snapshot(
    snapshot: FileSnapshot,
    *,
    source: str,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(snapshot.data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            message = f"{source} contains blank row at line {line_number}"
            raise ValueError(message)
        value: Any = json.loads(line)
        if not isinstance(value, dict):
            message = f"{source} row {line_number} must be an object"
            raise TypeError(message)
        rows.append(cast("dict[str, object]", value))
    if not rows:
        message = f"{source} must be nonempty"
        raise ValueError(message)
    return tuple(rows)


def _load_clean_manifest(snapshot: FileSnapshot) -> tuple[dict[str, object], ...]:
    rows = _load_jsonl_snapshot(snapshot, source="clean manifest")
    seen_source_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        source_id = _required_string(row, "source_id", source=f"clean row {line_number}")
        if source_id in seen_source_ids:
            message = f"duplicate clean manifest source_id: {source_id}"
            raise ValueError(message)
        seen_source_ids.add(source_id)
        _required_sha(row, "audio_sha256", source=f"clean row {line_number}")
        _required_sha(row, "pcm_sha256", source=f"clean row {line_number}")
        _required_string(row, "text", source=f"clean row {line_number}", allow_empty=True)
    return rows


def _load_generation_rows(snapshot: FileSnapshot) -> tuple[dict[str, object], ...]:
    raw_rows = _load_jsonl_snapshot(snapshot, source="generation results")
    rows: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for line_number, raw_row in enumerate(raw_rows, start=1):
        row = cast(
            "dict[str, object]",
            metrics.validate_generation_row(raw_row, line_number=line_number),
        )
        case_id = cast("str", row["case_id"])
        if case_id in seen_case_ids:
            message = f"duplicate case_id at line {line_number}: {case_id}"
            raise ValueError(message)
        seen_case_ids.add(case_id)
        rows.append(row)
    return tuple(rows)


def _validate_reference_manifest(
    payload: Mapping[str, object],
    *,
    manifest_path: Path,
    clean_manifest_path: Path,
    clean_manifest_sha256: str,
    clean_rows: Sequence[Mapping[str, object]],
) -> tuple[str, str, tuple[dict[str, object], ...]]:
    if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        message = f"reference manifest requires schema_version {REFERENCE_SCHEMA_VERSION}"
        raise ValueError(message)
    model_id = _required_string(payload, "model_id", source="reference manifest")
    if payload.get("all_reference_wavs_finite") is not True:
        raise ValueError("reference manifest all_reference_wavs_finite must be true")
    if payload.get("all_selected_source_hashes_verified") is not True:
        raise ValueError("reference manifest all_selected_source_hashes_verified must be true")
    if payload.get("selected_count") != EXPECTED_REFERENCE_COUNT:
        message = f"reference manifest selected_count must be {EXPECTED_REFERENCE_COUNT}"
        raise ValueError(message)
    healthy_population_count = payload.get("healthy_population_count")
    if (
        not isinstance(healthy_population_count, int)
        or isinstance(healthy_population_count, bool)
        or not EXPECTED_REFERENCE_COUNT <= healthy_population_count <= len(clean_rows)
    ):
        raise ValueError(
            "reference manifest healthy_population_count must cover the selected "
            "references and not exceed the clean manifest"
        )
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise TypeError("reference manifest source_hashes must be an object")
    for field in SOURCE_HASH_FIELDS:
        _required_sha(source_hashes, field, source="reference manifest source_hashes")
    if source_hashes["training_clean_manifest_sha256"] != clean_manifest_sha256:
        raise ValueError("reference manifest training_clean_manifest_sha256 mismatch")
    quantile_ranges = _validate_selection_contract(
        payload,
        healthy_population_count=healthy_population_count,
    )

    raw_references = payload.get("references")
    if not isinstance(raw_references, list) or len(raw_references) != EXPECTED_REFERENCE_COUNT:
        message = f"reference manifest must contain exactly {EXPECTED_REFERENCE_COUNT} references"
        raise ValueError(message)
    references: list[dict[str, object]] = []
    seen_reference_ids: set[str] = set()
    identity_owners: dict[str, dict[object, str]] = {
        "resolved reference WAV": {},
        "reference_wav_sha256": {},
        "resolved source audio": {},
        "audio_sha256": {},
        "pcm_sha256": {},
    }
    selection_orders = {quantile: set() for quantile in EXPECTED_QUANTILES}
    clean_by_id = {cast("str", row["source_id"]): row for row in clean_rows}
    for index, raw in enumerate(raw_references, start=1):
        if not isinstance(raw, dict):
            message = f"reference entry {index} must be an object"
            raise TypeError(message)
        row = cast("dict[str, object]", raw)
        source_id = _required_string(row, "source_id", source=f"reference entry {index}")
        if source_id in seen_reference_ids:
            message = f"duplicate reference source_id: {source_id}"
            raise ValueError(message)
        seen_reference_ids.add(source_id)
        clean_row = clean_by_id.get(source_id)
        if clean_row is None:
            message = f"reference source_id is missing from clean manifest: {source_id}"
            raise ValueError(message)
        for field in REFERENCE_IDENTITY_FIELDS:
            if row.get(field) != clean_row.get(field):
                message = f"{source_id}: source identity mismatch for {field}"
                raise ValueError(message)
        source_path, reference_path = _validate_reference_row(
            row,
            manifest_path=manifest_path,
            source_id=source_id,
        )
        _validate_reference_quantile(
            row,
            source_id=source_id,
            quantile_ranges=quantile_ranges,
            selection_orders=selection_orders,
        )
        for label, value in (
            ("resolved reference WAV", reference_path),
            ("reference_wav_sha256", row["reference_wav_sha256"]),
            ("resolved source audio", source_path),
            ("audio_sha256", row["audio_sha256"]),
            ("pcm_sha256", row["pcm_sha256"]),
        ):
            _claim_unique_reference_identity(
                owners=identity_owners[label],
                label=label,
                value=value,
                source_id=source_id,
            )
        _validate_optional_clean_audio_path(
            clean_row,
            clean_manifest_path=clean_manifest_path,
            expected=source_path,
            source_id=source_id,
        )
        references.append(
            {
                **row,
                "audio_path": str(source_path),
                "reference_wav_path": str(reference_path),
            },
        )

    expected_orders = set(range(1, EXPECTED_REFERENCES_PER_QUANTILE + 1))
    if any(orders != expected_orders for orders in selection_orders.values()):
        raise ValueError("reference selection_order_within_quantile must be unique 1 through 5")

    model_prefix = _validate_model_identity(
        model_id=model_id,
        clean_rows=clean_rows,
        reference_rows=references,
    )
    return model_id, model_prefix, tuple(references)


def _validate_selection_contract(
    payload: Mapping[str, object],
    *,
    healthy_population_count: int,
) -> dict[int, tuple[float, float]]:
    strategy = payload.get("selection_strategy")
    if not isinstance(strategy, dict):
        raise TypeError("reference manifest selection_strategy must be an object")
    expected_strategy_fields = {
        "duration_quantiles",
        "duration_stratification",
        "health_filter",
        "references_per_quantile",
        "selection_within_each_quantile",
        "source",
    }
    if set(strategy) != expected_strategy_fields:
        raise ValueError("reference manifest selection_strategy fields are invalid")
    if strategy.get("references_per_quantile") != EXPECTED_REFERENCES_PER_QUANTILE:
        raise ValueError("reference manifest references_per_quantile must be 5")
    quantiles = payload.get("quantiles")
    if not isinstance(quantiles, list) or len(quantiles) != len(EXPECTED_QUANTILES):
        raise ValueError("reference manifest must declare exactly five duration quantiles")
    declared: list[int] = []
    total_population = 0
    ranges: dict[int, tuple[float, float]] = {}
    previous_maximum: float | None = None
    for raw in quantiles:
        if not isinstance(raw, dict):
            raise TypeError("reference manifest quantile entries must be objects")
        quantile = raw.get("quantile")
        if not isinstance(quantile, int) or isinstance(quantile, bool):
            raise TypeError("reference manifest quantile must be an integer")
        declared.append(quantile)
        if raw.get("selected_count") != EXPECTED_REFERENCES_PER_QUANTILE:
            raise ValueError("reference manifest quantile selected_count must be 5")
        population_count = raw.get("population_count")
        if (
            not isinstance(population_count, int)
            or isinstance(population_count, bool)
            or population_count < EXPECTED_REFERENCES_PER_QUANTILE
        ):
            raise ValueError("reference manifest quantile population_count is invalid")
        total_population += population_count
        minimum = _required_finite_number(
            raw,
            "population_min_seconds",
            source=f"duration quantile {quantile}",
        )
        maximum = _required_finite_number(
            raw,
            "population_max_seconds",
            source=f"duration quantile {quantile}",
        )
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("reference manifest quantile duration range is invalid")
        if previous_maximum is not None and minimum < previous_maximum:
            raise ValueError("reference manifest quantile duration ranges overlap or are unordered")
        ranges[quantile] = (minimum, maximum)
        previous_maximum = maximum
    if tuple(declared) != EXPECTED_QUANTILES:
        raise ValueError("reference manifest quantiles must be ordered 1 through 5")
    if total_population != healthy_population_count:
        raise ValueError(
            "reference manifest quantile population counts do not sum to healthy_population_count"
        )
    return ranges


def _validate_reference_quantile(
    row: Mapping[str, object],
    *,
    source_id: str,
    quantile_ranges: Mapping[int, tuple[float, float]],
    selection_orders: dict[int, set[int]],
) -> None:
    quantile = cast("int", row["duration_quantile"])
    duration = float(cast("float", row["duration_seconds"]))
    minimum, maximum = quantile_ranges[quantile]
    if not minimum <= duration <= maximum:
        message = (
            f"{source_id}: duration is outside declared duration range for quantile {quantile}"
        )
        raise ValueError(message)
    order = row.get("selection_order_within_quantile")
    if (
        not isinstance(order, int)
        or isinstance(order, bool)
        or not 1 <= order <= EXPECTED_REFERENCES_PER_QUANTILE
    ):
        message = f"{source_id}: selection_order_within_quantile must be 1 through 5"
        raise ValueError(message)
    if order in selection_orders[quantile]:
        message = f"{source_id}: duplicate selection_order_within_quantile in quantile {quantile}"
        raise ValueError(message)
    selection_orders[quantile].add(order)


def _claim_unique_reference_identity(
    *,
    owners: dict[object, str],
    label: str,
    value: object,
    source_id: str,
) -> None:
    owner = owners.get(value)
    if owner is not None and owner != source_id:
        message = f"duplicate {label} across source_ids: {owner}, {source_id}"
        raise ValueError(message)
    owners[value] = source_id


def _validate_reference_row(
    row: Mapping[str, object],
    *,
    manifest_path: Path,
    source_id: str,
) -> tuple[Path, Path]:
    source_path = _resolved_input_path(
        row.get("audio_path"),
        base=manifest_path.parent,
        source=f"{source_id} source audio",
    )
    reference_path = _resolved_input_path(
        row.get("reference_wav_path"),
        base=manifest_path.parent,
        source=f"{source_id} reference WAV",
    )
    for path, field in (
        (source_path, "audio_sha256"),
        (reference_path, "reference_wav_sha256"),
    ):
        expected = _required_sha(row, field, source=source_id)
        if sha256_file(path) != expected:
            message = f"{source_id}: current {field} does not match {path}"
            raise ValueError(message)
    if row.get("reference_wav_finite") is not True:
        message = f"{source_id}: reference_wav_finite must be true"
        raise ValueError(message)
    quantile = row.get("duration_quantile")
    if quantile not in EXPECTED_QUANTILES:
        message = f"{source_id}: duration_quantile must be 1 through 5"
        raise ValueError(message)
    duration = row.get("duration_seconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        message = f"{source_id}: duration_seconds must be numeric"
        raise TypeError(message)
    if not math.isfinite(float(duration)) or float(duration) <= 0.0:
        message = f"{source_id}: duration_seconds must be positive and finite"
        raise ValueError(message)
    return source_path, reference_path


def _resolved_input_path(value: object, *, base: Path, source: str) -> Path:
    if not isinstance(value, str) or not value:
        message = f"{source} path must be a nonempty string"
        raise ValueError(message)
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    if not path.is_file():
        message = f"{source} must be a regular file: {path}"
        raise ValueError(message)
    return path


def _validate_optional_clean_audio_path(
    clean_row: Mapping[str, object],
    *,
    clean_manifest_path: Path,
    expected: Path,
    source_id: str,
) -> None:
    for field in ("audio_path", "audio"):
        if field not in clean_row:
            continue
        actual = _resolved_input_path(
            clean_row[field],
            base=clean_manifest_path.parent,
            source=f"{source_id} clean {field}",
        )
        if actual != expected:
            message = f"{source_id}: clean {field} does not match reference source audio_path"
            raise ValueError(message)


def _validate_model_identity(
    *,
    model_id: str,
    clean_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
) -> str:
    prefixes = {
        cast("str", row["source_id"]).split(":", maxsplit=1)[0]
        for row in (*clean_rows, *reference_rows)
        if ":" in cast("str", row["source_id"])
    }
    if len(prefixes) != 1:
        raise ValueError("clean/reference source_ids must have one common model prefix")
    model_prefix = next(iter(prefixes))
    if any(":" not in cast("str", row["source_id"]) for row in clean_rows):
        raise ValueError("all clean source_ids must contain a model prefix separator")
    # Public model ids may be explicit aliases (for example, ``miu``) for one
    # source corpus.  Identity is established by the one-prefix join and the
    # reference/clean hashes, not by a naming convention between those ids.
    declared_model_values = {
        cast("str", row[field])
        for row in clean_rows
        for field in ("model_id", "output_model_id")
        if field in row and isinstance(row[field], str)
    }
    if declared_model_values and declared_model_values != {model_id}:
        raise ValueError("clean manifest declared model identity is inconsistent")
    return model_prefix


def _embed_references(
    rows: Sequence[dict[str, object]],
    *,
    manifest_base: Path,
    embedder: SpeakerEmbedder,
) -> list[tuple[dict[str, object], Path, np.ndarray]]:
    embedded: list[tuple[dict[str, object], Path, np.ndarray]] = []
    for row in rows:
        path = Path(cast("str", row["reference_wav_path"]))
        if not path.is_absolute():
            path = manifest_base / path
        path = path.resolve(strict=True)
        samples, sample_rate = metrics.read_wav(path)
        normalized = metrics.resample_audio(samples, sample_rate, metrics.TARGET_SAMPLE_RATE)
        embedding = embedder.embed(normalized, metrics.TARGET_SAMPLE_RATE)
        embedded.append((row, path, embedding))
    return embedded


def _validate_generation_matrix(
    rows: Sequence[Mapping[str, object]],
    *,
    model_id: str,
    checkpoint_step: int | None,
) -> tuple[
    tuple[Mapping[str, object], ...],
    dict[str, object],
    dict[str, object],
]:
    if any(row.get("model_id") != model_id for row in rows):
        raise ValueError("generation model_id does not match reference model_id")
    if all(
        "schema_version" in row and row.get("schema_version") == SEARCH_GENERATION_CASE_SCHEMA
        for row in rows
    ):
        selected_rows = tuple(rows)
        selection_mode = "search_matrix_single_checkpoint"
    elif all("schema_version" not in row for row in rows):
        if checkpoint_step is None:
            raise ValueError("full generation matrix requires --checkpoint-step")
        expected_full_matrix = {
            (step, text_id, seed, style)
            for step in FULL_MATRIX_CHECKPOINT_STEPS
            for text_id in FULL_MATRIX_TEXT_IDS
            for seed in EXPECTED_SEEDS
            for style in EXPECTED_STYLES
        }
        actual_full_matrix = {
            (
                row.get("checkpoint_step"),
                row.get("text_id"),
                row.get("seed"),
                row.get("style"),
            )
            for row in rows
        }
        if len(rows) != len(expected_full_matrix) or actual_full_matrix != expected_full_matrix:
            raise ValueError("generation results do not contain the exact 140-case full matrix")
        if checkpoint_step not in FULL_MATRIX_CHECKPOINT_STEPS:
            raise ValueError("checkpoint_step is not present in the full generation matrix")
        if any(row.get("status") != "SUCCESS" for row in rows):
            raise ValueError("all full-matrix generation rows must be SUCCESS")
        selected_rows = tuple(row for row in rows if row.get("checkpoint_step") == checkpoint_step)
        selection_mode = "full_matrix_explicit_checkpoint"
    else:
        raise ValueError("generation results contain mixed or unsupported case schemas")

    if any(row.get("status") != "SUCCESS" for row in selected_rows):
        raise ValueError("all selected generation rows must be SUCCESS")
    selected_identities = tuple(_checkpoint_run_identity(row) for row in selected_rows)
    checkpoint_run_identity = selected_identities[0]
    if any(identity != checkpoint_run_identity for identity in selected_identities[1:]):
        raise ValueError("selected generation checkpoint/run identity mismatch")

    gate_rows = tuple(row for row in selected_rows if row.get("text_id") in METRIC_GATE_TEXT_IDS)
    expected = {
        (text_id, seed, style)
        for text_id in METRIC_GATE_TEXT_IDS
        for seed in EXPECTED_SEEDS
        for style in EXPECTED_STYLES
    }
    actual = {(row.get("text_id"), row.get("seed"), row.get("style")) for row in gate_rows}
    if len(gate_rows) != EXPECTED_METRIC_GATE_CASE_COUNT or actual != expected:
        raise ValueError("generation results do not contain the exact 16-case metric gate matrix")
    if any(row.get("status") != "SUCCESS" for row in gate_rows):
        raise ValueError("all metric-gate generation rows must be SUCCESS")
    selected_checkpoint_step = cast("int", checkpoint_run_identity["checkpoint_step"])
    if checkpoint_step is not None and checkpoint_step != selected_checkpoint_step:
        raise ValueError("requested checkpoint_step does not match generation checkpoint")
    generation_selection = {
        "mode": selection_mode,
        "requested_checkpoint_step": checkpoint_step,
        "selected_checkpoint_step": selected_checkpoint_step,
        "input_case_count": len(rows),
        "selected_case_count": len(selected_rows),
        "metric_gate_case_count": len(gate_rows),
    }
    return gate_rows, checkpoint_run_identity, generation_selection


def _checkpoint_run_identity(row: Mapping[str, object]) -> dict[str, object]:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("metric-gate provenance must be an object")
    checkpoint = _required_string(row, "checkpoint", source="metric-gate row")
    provenance_checkpoint = _required_string(
        provenance,
        "base_checkpoint",
        source="metric-gate provenance",
    )
    if provenance_checkpoint != checkpoint:
        raise ValueError("metric-gate base_checkpoint provenance mismatch")
    checkpoint_step = row.get("checkpoint_step")
    if not isinstance(checkpoint_step, int) or isinstance(checkpoint_step, bool):
        raise TypeError("metric-gate checkpoint_step must be an integer")
    return {
        "model_id": _required_string(row, "model_id", source="metric-gate row"),
        "checkpoint_step": checkpoint_step,
        "speaker_filename": _required_string(
            row,
            "speaker_filename",
            source="metric-gate row",
        ),
        "embedding_path": _required_string(row, "embedding_path", source="metric-gate row"),
        "embedding_sha256": _required_sha(row, "embedding_sha256", source="metric-gate row"),
        "training_config_sha256": _required_sha(
            provenance,
            "training_config_sha256",
            source="metric-gate provenance",
        ),
        "base_checkpoint": checkpoint,
        "base_checkpoint_sha256": _required_sha(
            row,
            "base_checkpoint_sha256",
            source="metric-gate row",
        ),
        "base_revision": _required_string(
            provenance,
            "base_revision",
            source="metric-gate provenance",
        ),
        "run_id": _required_string(provenance, "run_id", source="metric-gate provenance"),
        "evaluation_manifest_sha256": _required_sha(
            row,
            "evaluation_manifest_sha256",
            source="metric-gate row",
        ),
        "provenance": dict(provenance),
    }


def _index_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source: str,
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for line_number, row in enumerate(rows, start=1):
        case_id = _required_string(row, "case_id", source=f"{source} row {line_number}")
        if case_id in indexed:
            message = f"duplicate {source} case_id: {case_id}"
            raise ValueError(message)
        indexed[case_id] = row
    return indexed


def _validate_metrics_provenance(
    payload: Mapping[str, object],
    *,
    generation_results_path: Path,
    generation_results_sha256: str,
    reference_wavs_path: Path,
    reference_wavs_sha256: str,
    reference_rows: Sequence[Mapping[str, object]],
    generation_rows: Sequence[Mapping[str, object]],
    reference_base: Path,
    embedder: SpeakerEmbedder,
    ecapa_source_sha256: str,
) -> None:
    if payload.get("schema_version") != METRICS_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("metrics provenance schema_version mismatch")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise TypeError("metrics provenance models must be an object")
    speaker = models.get("speaker_embedding")
    if not isinstance(speaker, dict):
        raise TypeError("metrics provenance speaker_embedding must be an object")
    if speaker.get("model_id") != embedder.model_id:
        raise ValueError("metrics provenance ECAPA model_id mismatch")
    if speaker.get("revision") != embedder.revision:
        raise ValueError("metrics provenance ECAPA revision mismatch")
    if speaker.get("source_sha256") != ecapa_source_sha256:
        raise ValueError("metrics provenance ECAPA source SHA-256 mismatch")
    transcription = models.get("transcription")
    if not isinstance(transcription, dict):
        raise TypeError("metrics provenance transcription must be an object")
    _required_string(transcription, "model_id", source="metrics transcription")
    _required_string(transcription, "revision", source="metrics transcription")
    _required_sha(transcription, "source_sha256", source="metrics transcription")

    input_sha = payload.get("input_sha256")
    if not isinstance(input_sha, dict):
        raise TypeError("metrics provenance input_sha256 must be an object")
    if input_sha.get("generation_results") != generation_results_sha256:
        raise ValueError("metrics provenance generation_results SHA-256 mismatch")
    if input_sha.get("reference_wavs") != reference_wavs_sha256:
        raise ValueError("metrics provenance reference_wavs SHA-256 mismatch")
    expected_references = {}
    for row in reference_rows:
        reference_path = Path(cast("str", row["reference_wav_path"]))
        if not reference_path.is_absolute():
            reference_path = reference_base / reference_path
        expected_references[reference_path.resolve(strict=True)] = cast(
            "str",
            row["reference_wav_sha256"],
        )
    expected_generated: dict[Path, str] = {}
    for row in generation_rows:
        if row.get("status") != "SUCCESS":
            continue
        generated_path = Path(cast("str", row["wav_path"]))
        if not generated_path.is_absolute():
            generated_path = generation_results_path.parent / generated_path
        expected_generated[generated_path.resolve(strict=True)] = cast(
            "str",
            row["wav_sha256"],
        )
    _validate_provenance_audio_map(
        input_sha.get("reference_audio"),
        expected=expected_references,
        base=reference_wavs_path.parent,
        source="reference_audio",
    )
    _validate_provenance_audio_map(
        input_sha.get("generated_audio"),
        expected=expected_generated,
        base=generation_results_path.parent,
        source="generated_audio",
    )


def _validate_provenance_audio_map(
    value: object,
    *,
    expected: Mapping[Path, str],
    base: Path,
    source: str,
) -> None:
    if not isinstance(value, dict):
        message = f"metrics provenance {source} must be an object"
        raise TypeError(message)
    actual: dict[Path, str] = {}
    for raw_path, raw_sha in value.items():
        if not isinstance(raw_path, str) or not _is_sha(raw_sha):
            message = f"metrics provenance {source} entry is invalid"
            raise ValueError(message)
        path = Path(raw_path)
        if not path.is_absolute():
            path = base / path
        path = path.resolve(strict=True)
        if not path.is_file() or sha256_file(path) != raw_sha:
            message = f"metrics provenance {source} current hash mismatch: {path}"
            raise ValueError(message)
        actual[path] = raw_sha
    if actual != expected:
        message = f"metrics provenance {source} artifact set mismatch"
        raise ValueError(message)


def _capture_bound_audio_snapshots(
    *,
    reference_rows: Sequence[Mapping[str, object]],
    generation_rows: Sequence[Mapping[str, object]],
    generation_base: Path,
) -> tuple[FileSnapshot, ...]:
    snapshots: list[FileSnapshot] = []
    for row in reference_rows:
        source_id = cast("str", row["source_id"])
        for name, path_field, sha_field in (
            (f"reference_source_audio:{source_id}", "audio_path", "audio_sha256"),
            (f"reference_wav:{source_id}", "reference_wav_path", "reference_wav_sha256"),
        ):
            snapshot = _snapshot_file(
                name,
                Path(cast("str", row[path_field])),
                source=name,
            )
            if snapshot.sha256 != row[sha_field]:
                message = f"{name} SHA-256 does not match reference manifest"
                raise ValueError(message)
            snapshots.append(snapshot)
    for row in generation_rows:
        if row.get("status") != "SUCCESS":
            continue
        case_id = cast("str", row["case_id"])
        wav_path = Path(cast("str", row["wav_path"]))
        if not wav_path.is_absolute():
            wav_path = generation_base / wav_path
        name = f"generated_wav:{case_id}"
        snapshot = _snapshot_file(name, wav_path, source=name)
        if snapshot.sha256 != row["wav_sha256"]:
            message = f"{name} SHA-256 does not match generation results"
            raise ValueError(message)
        snapshots.append(snapshot)
    return tuple(snapshots)


def _verify_audit_inputs_unchanged(
    *,
    explicit_inputs: Sequence[FileSnapshot],
    scripts: Sequence[FileSnapshot],
    bound_audio: Sequence[FileSnapshot],
    ecapa_source: Path,
    ecapa_source_sha256: str,
) -> None:
    for snapshot in (*explicit_inputs, *scripts, *bound_audio):
        _verify_snapshot_unchanged(snapshot)
    try:
        current_ecapa_sha256 = cast("str", metrics.sha256_tree(ecapa_source))
    except (OSError, TypeError, ValueError) as exc:
        message = f"ecapa_source changed during audit: {ecapa_source}"
        raise ValueError(message) from exc
    if current_ecapa_sha256 != ecapa_source_sha256:
        message = f"ecapa_source changed during audit: {ecapa_source}"
        raise ValueError(message)


def _verify_snapshot_unchanged(snapshot: FileSnapshot) -> None:
    try:
        current_sha256 = sha256_file(snapshot.path)
    except (OSError, TypeError, ValueError) as exc:
        message = f"{snapshot.name} changed during audit: {snapshot.path}"
        raise ValueError(message) from exc
    if current_sha256 != snapshot.sha256:
        message = f"{snapshot.name} changed during audit: {snapshot.path}"
        raise ValueError(message)


def _reference_analysis(
    references: Sequence[tuple[dict[str, object], Path, np.ndarray]],
    *,
    full_centroid: np.ndarray,
    leave_one_out_centroids: Sequence[np.ndarray],
) -> dict[str, object]:
    similarities = [
        float(metrics.normalized_cosine_similarity(embedding, full_centroid))
        for _, _, embedding in references
    ]
    median = float(np.median(np.asarray(similarities, dtype=np.float64)))
    mad = float(np.median(np.abs(np.asarray(similarities) - median)))
    threshold, outlier_flags = _robust_outliers(similarities)
    outlier_ids = [
        cast("str", row["source_id"])
        for (row, _, _), is_outlier in zip(references, outlier_flags, strict=True)
        if is_outlier
    ]
    pairwise = [
        float(metrics.normalized_cosine_similarity(references[left][2], references[right][2]))
        for left in range(len(references))
        for right in range(left + 1, len(references))
    ]
    per_reference = [
        {
            "source_id": row["source_id"],
            "text": row["text"],
            "text_class": _text_class(cast("str", row["text"])),
            "duration_quantile": row["duration_quantile"],
            "duration_seconds": row["duration_seconds"],
            "reference_wav_path": str(path.resolve()),
            "reference_wav_sha256": row["reference_wav_sha256"],
            "source_audio_path": row["audio_path"],
            "source_audio_sha256": row["audio_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "clean_manifest_identity_fields_verified": list(REFERENCE_IDENTITY_FIELDS),
            "similarity_to_full_centroid": similarity,
            "outlier": row["source_id"] in outlier_ids,
        }
        for (row, path, _), similarity in zip(references, similarities, strict=True)
    ]
    leave_one_out = [
        {
            "excluded_source_id": row["source_id"],
            "centroid_similarity_to_full": similarity,
            "centroid_drift": 1.0 - similarity,
        }
        for (row, _, _), similarity in zip(
            references,
            (
                float(metrics.normalized_cosine_similarity(centroid, full_centroid))
                for centroid in leave_one_out_centroids
            ),
            strict=True,
        )
    ]
    quantile_centroids: dict[int, np.ndarray] = {}
    duration_quantile_rows: list[dict[str, object]] = []
    for quantile in EXPECTED_QUANTILES:
        embeddings = [
            embedding for row, _, embedding in references if row["duration_quantile"] == quantile
        ]
        if len(embeddings) != EXPECTED_REFERENCES_PER_QUANTILE:
            message = f"duration quantile {quantile} must contain exactly five references"
            raise ValueError(message)
        centroid = metrics.aggregate_reference_centroid(embeddings)
        quantile_centroids[quantile] = centroid
        duration_quantile_rows.append(
            {
                "duration_quantile": quantile,
                "reference_count": len(embeddings),
                "similarity_to_full_centroid": float(
                    metrics.normalized_cosine_similarity(centroid, full_centroid),
                ),
            },
        )
    mutual = [
        {
            "left_duration_quantile": left,
            "right_duration_quantile": right,
            "centroid_similarity": float(
                metrics.normalized_cosine_similarity(
                    quantile_centroids[left],
                    quantile_centroids[right],
                ),
            ),
        }
        for left in EXPECTED_QUANTILES
        for right in EXPECTED_QUANTILES
        if left < right
    ]
    text_split = _text_split(references, full_centroid=full_centroid)
    return {
        "per_reference": per_reference,
        "pairwise_similarity": _stats(pairwise),
        "leave_one_out": leave_one_out,
        "duration_quantile_centroids": duration_quantile_rows,
        "duration_quantile_mutual_similarities": mutual,
        "text_split": text_split,
        "outlier_rule": {
            "method": "median_minus_3_mad",
            "description": (
                "Flag similarity_to_full_centroid values strictly below median - 3*MAD; "
                "this fixed robust diagnostic rule is not tuned to the quality gate threshold."
            ),
            "mad_zero_behavior": (
                "The threshold equals the median; values strictly below median are flagged, "
                "while values equal to the median are not."
            ),
            "median": median,
            "mad": mad,
            "multiplier": OUTLIER_MAD_MULTIPLIER,
            "threshold": threshold,
        },
        "outlier_count": len(outlier_ids),
        "outlier_source_ids": outlier_ids,
    }


def _robust_outliers(values: Sequence[float]) -> tuple[float, tuple[bool, ...]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("robust outlier rule requires nonempty finite values")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    threshold = median - OUTLIER_MAD_MULTIPLIER * mad
    return threshold, tuple(bool(value < threshold) for value in array)


def _text_split(
    references: Sequence[tuple[dict[str, object], Path, np.ndarray]],
    *,
    full_centroid: np.ndarray,
) -> dict[str, object]:
    groups: dict[str, list[np.ndarray]] = {"expressive_nonverbal": [], "lexical": []}
    for row, _, embedding in references:
        groups[_text_class(cast("str", row["text"]))].append(embedding)

    def group_summary(name: str) -> dict[str, object]:
        embeddings = groups[name]
        similarity = None
        if embeddings:
            centroid = metrics.aggregate_reference_centroid(embeddings)
            similarity = float(metrics.normalized_cosine_similarity(centroid, full_centroid))
        return {"count": len(embeddings), "centroid_similarity_to_full": similarity}

    expressive = group_summary("expressive_nonverbal")
    lexical = group_summary("lexical")
    return {
        "criterion": {
            "normalization": "Unicode NFKC and whitespace removal",
            "expressive_nonverbal_when_text_contains_any_marker": list(NONVERBAL_TEXT_MARKERS),
            "otherwise": "lexical",
        },
        "gate_effect": "diagnostic_only",
        "expressive_nonverbal_count": expressive["count"],
        "lexical_count": lexical["count"],
        "expressive_nonverbal": expressive,
        "lexical": lexical,
    }


def _text_class(text: str) -> str:
    normalized = "".join(
        character for character in unicodedata.normalize("NFKC", text) if not character.isspace()
    )
    if any(marker in normalized for marker in NONVERBAL_TEXT_MARKERS):
        return "expressive_nonverbal"
    return "lexical"


def _generated_analysis(
    gate_rows: Sequence[Mapping[str, object]],
    *,
    metrics_rows: Mapping[str, Mapping[str, object]],
    generation_base: Path,
    generation_results_sha256: str,
    embedder: SpeakerEmbedder,
    full_centroid: np.ndarray,
    leave_one_out_centroids: Sequence[np.ndarray],
    reference_rows: Sequence[Mapping[str, object]],
    checkpoint_run_identity: Mapping[str, object],
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for row in gate_rows:
        case_id = cast("str", row["case_id"])
        metric_row = metrics_rows.get(case_id)
        if metric_row is None:
            message = f"metric-gate case is missing metrics row: {case_id}"
            raise ValueError(message)
        if any(metric_row.get(field) != row.get(field) for field in GENERATION_IDENTITY_FIELDS):
            message = f"{case_id}: metrics identity does not match generation"
            raise ValueError(message)
        if metric_row.get("metrics_status") != "COMPLETE":
            message = f"{case_id}: metrics_status must be COMPLETE"
            raise ValueError(message)
        if metric_row.get("generation_results_sha256") != generation_results_sha256:
            message = f"{case_id}: metrics generation_results_sha256 mismatch"
            raise ValueError(message)
        wav_path = Path(cast("str", row["wav_path"]))
        if not wav_path.is_absolute():
            wav_path = generation_base / wav_path
        wav_path = wav_path.resolve(strict=True)
        samples, sample_rate = metrics.read_wav(wav_path)
        normalized = metrics.resample_audio(samples, sample_rate, metrics.TARGET_SAMPLE_RATE)
        embedding = embedder.embed(normalized, metrics.TARGET_SAMPLE_RATE)
        full_similarity = float(metrics.normalized_cosine_similarity(embedding, full_centroid))
        supplied = metric_row.get("speaker_similarity")
        if not isinstance(supplied, (int, float)) or isinstance(supplied, bool):
            message = f"{case_id}: metrics speaker_similarity must be numeric"
            raise TypeError(message)
        if not math.isclose(
            full_similarity,
            float(supplied),
            rel_tol=0.0,
            abs_tol=SIMILARITY_TOLERANCE,
        ):
            message = f"{case_id}: speaker_similarity does not match recomputed full centroid"
            raise ValueError(message)
        leave_out = []
        for reference, centroid in zip(reference_rows, leave_one_out_centroids, strict=True):
            similarity = float(metrics.normalized_cosine_similarity(embedding, centroid))
            leave_out.append(
                {
                    "excluded_source_id": reference["source_id"],
                    "speaker_similarity": similarity,
                    "delta_from_full_centroid": similarity - full_similarity,
                },
            )
        cases.append(
            {
                "case_id": case_id,
                "text_id": row["text_id"],
                "seed": row["seed"],
                "style": row["style"],
                "wav_path": str(wav_path),
                "wav_sha256": row["wav_sha256"],
                "checkpoint_run_identity": dict(checkpoint_run_identity),
                "full_centroid_similarity": full_similarity,
                "supplied_metrics_similarity": float(supplied),
                "leave_one_out": leave_out,
            },
        )
    return {
        "metric_gate_case_count": len(cases),
        "metrics_similarity_verified": True,
        "similarity_absolute_tolerance": SIMILARITY_TOLERANCE,
        "checkpoint_run_identity": dict(checkpoint_run_identity),
        "cases": cases,
    }


def _stats(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("statistics require nonempty finite values")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _required_string(
    row: Mapping[str, object],
    field: str,
    *,
    source: str,
    allow_empty: bool = False,
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or (not allow_empty and not value):
        message = f"{source} requires string {field}"
        raise ValueError(message)
    return value


def _required_finite_number(
    row: Mapping[str, object],
    field: str,
    *,
    source: str,
) -> float:
    value = row.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        message = f"{source} requires finite number {field}"
        raise ValueError(message)
    return float(value)


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_sha(row: Mapping[str, object], field: str, *, source: str) -> str:
    value = row.get(field)
    if not _is_sha(value):
        message = f"{source} requires lowercase SHA-256 {field}"
        raise ValueError(message)
    return cast("str", value)


def _write_json_create_only(
    path: Path,
    payload: Mapping[str, object],
    *,
    before_publish: Callable[[], None],
) -> None:
    output = _safe_output_path(path)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        _safe_output_path(output)
        before_publish()
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            message = f"refusing to overwrite reference centroid audit: {output}"
            raise FileExistsError(message) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output_path(path: Path) -> Path:
    output = Path(os.path.abspath(path))  # noqa: PTH100 - resolving would follow output aliases.
    try:
        metadata = output.lstat()
    except FileNotFoundError:
        pass
    else:
        if _is_link_or_reparse(metadata):
            message = (
                "reference centroid audit output must not be a symbolic link or "
                f"reparse point: {output}"
            )
            raise ValueError(message)
        message = f"refusing to overwrite reference centroid audit: {output}"
        raise FileExistsError(message)
    _ensure_safe_parent(output.parent)
    return output


def _ensure_safe_parent(parent: Path) -> None:
    lexical_chain = tuple(reversed((parent, *parent.parents)))
    for candidate in lexical_chain:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            candidate.mkdir()
            metadata = candidate.lstat()
        if _is_link_or_reparse(metadata):
            message = (
                "reference centroid audit parent must not be a symbolic link or "
                f"reparse point: {candidate}"
            )
            raise ValueError(message)
        if not stat.S_ISDIR(metadata.st_mode):
            message = f"reference centroid audit parent must be a directory: {candidate}"
            raise ValueError(message)


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(file_attributes & reparse_attribute)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-wavs", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--generation-results", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--metrics-results", type=Path, required=True)
    parser.add_argument("--metrics-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ecapa-source", type=Path, required=True)
    parser.add_argument("--ecapa-savedir", type=Path, required=True)
    parser.add_argument("--ecapa-model-id", required=True)
    parser.add_argument("--ecapa-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    embedder = metrics.SpeechBrainECAPA.load(
        source=args.ecapa_source,
        savedir=args.ecapa_savedir,
        model_id=args.ecapa_model_id,
        revision=args.ecapa_revision,
    )
    run_audit(
        reference_wavs_path=args.reference_wavs,
        clean_manifest_path=args.clean_manifest,
        generation_results_path=args.generation_results,
        metrics_results_path=args.metrics_results,
        metrics_provenance_path=args.metrics_provenance,
        output_path=args.output,
        ecapa_source=args.ecapa_source,
        embedder=embedder,
        checkpoint_step=args.checkpoint_step,
    )
    print(f"reference centroid audit written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
