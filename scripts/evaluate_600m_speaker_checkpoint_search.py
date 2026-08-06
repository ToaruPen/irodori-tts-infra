# ruff: noqa: ANN401, C901, EM101, EM102, PLR0912, PLR0914, PLR2004, SLF001, TRY003
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from types import ModuleType

SEARCH_SCHEMA = "speaker-checkpoint-search-manifest/v1"
GENERATION_VERIFICATION_SCHEMA = "speaker-checkpoint-search-generation-verification/v1"
GENERATION_CONFIG_SCHEMA = "speaker-checkpoint-search-generation/v1"
GENERATION_CASE_SCHEMA = "speaker-checkpoint-search-generation-case/v1"
EVALUATION_CASE_SCHEMA = "speaker-checkpoint-search-evaluation-case/v1"
EVALUATION_SCHEMA = "speaker-checkpoint-search-evaluation/v1"
VERIFICATION_SCHEMA = "speaker-checkpoint-search-evaluation-verification/v1"
PRODUCTION_EVALUATOR = Path(__file__).with_name("evaluate_600m_speaker_checkpoints.py")
SEARCH_BUILDER = Path(__file__).with_name("build_600m_speaker_checkpoint_search_manifest.py")
SEARCH_GENERATOR = Path(__file__).with_name("generate_600m_speaker_checkpoint_search_remote.py")
PRODUCTION_GENERATOR = Path(__file__).with_name("generate_600m_checkpoint_audio_remote.py")
SEARCH_STEP = 250
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
METRIC_GATE_TEXT_IDS = frozenset({"sentence_unko", "sentence_chinko", "sentence_manko", "control"})
EXPECTED_CASE_COUNT = 28
EXPECTED_METRIC_CASE_COUNT = 16
MIN_SPEAKER_SIMILARITY = 0.75
MAX_STYLE_SIMILARITY_DROP = 0.08
MIN_STYLE_CONTRAST = 0.01
IDENTITY_FIELDS = ("wav_path", "wav_sha256", "provenance")


def _production() -> ModuleType:
    name = "_speaker_checkpoint_production_evaluator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, PRODUCTION_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production evaluator: {PRODUCTION_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _builder() -> ModuleType:
    name = "_speaker_checkpoint_search_builder_for_evaluator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, SEARCH_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load search builder: {SEARCH_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_search_manifest(
    path: Path,
    *,
    snapshot: tuple[bytes, str] | None = None,
) -> Any:
    manifest_path = path.resolve()
    payload = (
        _read_json_snapshot(snapshot, path=manifest_path)
        if snapshot is not None
        else _read_json(manifest_path)
    )
    if payload.get("schema_version") != SEARCH_SCHEMA:
        raise ValueError(f"search manifest requires schema_version {SEARCH_SCHEMA}")
    for field, expected in (("text_ids", TEXT_IDS), ("seeds", SEEDS), ("styles", STYLES)):
        if tuple(payload.get(field, ())) != expected:
            raise ValueError(f"search manifest {field} must exactly match {expected}")
    model_id = _required_string(payload, "model_id", source="search manifest")
    run_id = _required_string(payload, "run_id", source="search manifest")
    source = payload.get("source_evaluation_manifest")
    if not isinstance(source, dict):
        raise TypeError("source_evaluation_manifest must be an object")
    source_path = _resolved_path(source.get("path"), base=manifest_path.parent)
    source_sha = _required_sha(source, "sha256")
    _validate_file_hash(source_path, source_sha, source="source evaluation manifest")
    source_payload = _read_json(source_path)
    _builder()._validate_source_manifest(
        source_payload,
        model_id=_required_string(payload, "model_id", source="search manifest"),
        manifest_dir=source_path.parent,
    )
    _builder()._validate_search_source_binding(payload, source_payload)
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_step") != SEARCH_STEP:
        raise ValueError(f"search manifest requires checkpoint step {SEARCH_STEP}")
    if checkpoint.get("run_id") != run_id:
        raise ValueError("search checkpoint run_id does not match manifest")
    embedding_path = _resolved_path(checkpoint.get("embedding_path"), base=manifest_path.parent)
    config_path = _resolved_path(checkpoint.get("training_config_path"), base=manifest_path.parent)
    config_sha = _required_sha(checkpoint, "training_config_sha256")
    _validate_file_hash(config_path, config_sha, source="training config")
    config = _read_json(config_path)
    _builder()._validate_search_training_config(config)
    production = _production()
    contract = production.CheckpointContract(
        model_id=model_id,
        checkpoint_step=SEARCH_STEP,
        embedding_path=embedding_path,
        embedding_sha256=_required_sha(checkpoint, "embedding_sha256"),
        training_config_sha256=config_sha,
        base_checkpoint=_required_string(checkpoint, "base_checkpoint", source="checkpoint"),
        base_checkpoint_sha256=_required_sha(checkpoint, "base_checkpoint_sha256"),
        base_revision=_required_string(checkpoint, "base_revision", source="checkpoint"),
        run_id=run_id,
        evaluation_manifest_sha256=snapshot[1] if snapshot is not None else _sha(manifest_path),
    )
    production._validate_embedding(contract)
    evidence = payload.get("training_run_evidence")
    if not isinstance(evidence, dict):
        raise TypeError("training_run_evidence must be an object")
    evidence_path = _resolved_path(evidence.get("path"), base=manifest_path.parent)
    evidence_sha = _required_sha(evidence, "sha256")
    _validate_file_hash(evidence_path, evidence_sha, source="training run evidence")
    _builder()._validate_run_evidence(
        evidence_path,
        model_id=model_id,
        run_id=run_id,
        config_path=config_path,
        config_sha256=config_sha,
        embedding_path=embedding_path,
        embedding_sha256=contract.embedding_sha256,
        base_checkpoint_sha256=contract.base_checkpoint_sha256,
    )
    metrics = payload.get("metrics_provenance")
    if not isinstance(metrics, dict):
        raise TypeError("search manifest metrics_provenance must be an object")
    speaker = metrics.get("speaker_embedding")
    transcription = metrics.get("transcription")
    if not isinstance(speaker, dict) or not isinstance(transcription, dict):
        raise TypeError("search manifest metric models must be objects")
    return production.EvaluationManifest(
        checkpoints={(model_id, SEARCH_STEP): contract},
        text_ids=TEXT_IDS,
        seeds=SEEDS,
        styles=STYLES,
        reference_wavs_sha256=_required_sha(metrics, "reference_wavs_sha256"),
        speaker_embedding_model_id=_required_string(speaker, "model_id", source="speaker model"),
        speaker_embedding_revision=_required_string(speaker, "revision", source="speaker model"),
        speaker_embedding_source_sha256=_required_sha(speaker, "source_sha256"),
        transcription_model_id=_required_string(
            transcription, "model_id", source="transcription model"
        ),
        transcription_revision=_required_string(
            transcription, "revision", source="transcription model"
        ),
        transcription_source_sha256=_required_sha(transcription, "source_sha256"),
    )


def validate_case_matrix(
    identities: Sequence[tuple[str, int, str, int, str]],
    *,
    model_id: str,
) -> None:
    expected = {
        (model_id, SEARCH_STEP, text_id, seed, style)
        for text_id in TEXT_IDS
        for seed in SEEDS
        for style in STYLES
    }
    actual = set(identities)
    if len(actual) != len(identities):
        raise ValueError("duplicate search matrix case")
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"search matrix missing expected case: {min(missing)}")
    if extra:
        raise ValueError(f"search matrix has unexpected case: {min(extra)}")


def validate_artifact_identity(
    generation: Mapping[str, Mapping[str, object]],
    other: Mapping[str, Mapping[str, object]],
    *,
    source: str,
) -> None:
    if generation.keys() != other.keys():
        raise ValueError(f"{source} artifact identity case set mismatch")
    for case_id, generation_row in generation.items():
        other_row = other[case_id]
        if any(generation_row.get(field) != other_row.get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"{case_id}: {source} artifact identity mismatch")


def validate_generation_case_schemas(
    rows: Mapping[str, Mapping[str, object]],
    *,
    source: str,
) -> None:
    invalid = [
        case_id
        for case_id, row in rows.items()
        if row.get("schema_version") != GENERATION_CASE_SCHEMA
    ]
    if invalid:
        raise ValueError(f"{source} row schema mismatch: {invalid[0]}")


def bind_search_evaluation_case_schema(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [{**row, "evaluation_schema_version": EVALUATION_CASE_SCHEMA} for row in rows]


def reserve_output(path: Path) -> Path:
    output = _prepare_output_parent(path, source="search evaluation")
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite search evaluation: {output}") from exc
    _require_nominal_directory(output, source="search evaluation output")
    return output.resolve(strict=True)


def _prepare_output_parent(path: Path, *, source: str) -> Path:
    output = Path(os.path.abspath(path))  # noqa: PTH100 - resolve() would follow aliases.
    missing: list[Path] = []
    candidate = output.parent
    while candidate != candidate.parent:
        try:
            candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate)
            candidate = candidate.parent
            continue
        _require_nominal_directory(candidate, source=f"{source} parent")
        if not candidate.is_dir():
            raise ValueError(f"{source} parent must be a directory: {candidate}")
        break
    for ancestor in (candidate, *candidate.parents):
        if ancestor != ancestor.parent:
            _require_nominal_directory(ancestor, source=f"{source} parent")
    for directory in reversed(missing):
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{source} parent changed during reservation: {directory}"
            ) from exc
        _require_nominal_directory(directory, source=f"{source} parent")
    return output


def _require_nominal_directory(path: Path, *, source: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{source} must not be a symlink, junction, or reparse alias: {path}")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if reparse_flag and file_attributes & reparse_flag:
        raise ValueError(f"{source} must not be a symlink, junction, or reparse alias: {path}")


def summarize_search(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError(f"search evaluation requires exactly {EXPECTED_CASE_COUNT} cases")
    identities = [
        (
            str(row.get("model_id")),
            _required_int(row, "checkpoint_step"),
            str(row.get("text_id")),
            _required_int(row, "seed"),
            str(row.get("style")),
        )
        for row in rows
    ]
    model_ids = {identity[0] for identity in identities}
    if len(model_ids) != 1:
        raise ValueError("search evaluation must contain exactly one model_id")
    validate_case_matrix(identities, model_id=next(iter(model_ids)))
    metric_rows = [row for row in rows if row.get("metric_gate_applied") is True]
    if len(metric_rows) != EXPECTED_METRIC_CASE_COUNT:
        raise ValueError(
            f"search evaluation requires exactly {EXPECTED_METRIC_CASE_COUNT} metric-gate cases"
        )
    similarities = [
        value
        for row in metric_rows
        if (value := _optional_finite_float(row, "speaker_similarity")) is not None
    ]
    sorted_similarities = sorted(similarities)
    rejections = _collect_reasons(rows, "rejection_reasons")
    incomplete = _collect_reasons(rows, "incomplete_reasons")
    reviews = _collect_reasons(rows, "review_reasons")
    if any(value < MIN_SPEAKER_SIMILARITY for value in similarities):
        rejections.add("low_speaker_similarity")
    style_pairs: dict[tuple[str, int], dict[str, float]] = {}
    for row in metric_rows:
        key = (str(row["text_id"]), _required_int(row, "seed"))
        similarity = _optional_finite_float(row, "speaker_similarity")
        if similarity is not None:
            style_pairs.setdefault(key, {})[str(row["style"])] = similarity
    if len(style_pairs) != 8 or any(set(pair) != set(STYLES) for pair in style_pairs.values()):
        incomplete.add("missing_style_pair")
    elif any(
        pair["calm"] < pair["neutral"] - MAX_STYLE_SIMILARITY_DROP for pair in style_pairs.values()
    ):
        rejections.add("style_similarity_not_preserved")
    style_contrast = _style_contrast(rows)
    if style_contrast < MIN_STYLE_CONTRAST:
        reviews.add("insufficient_style_contrast")
    if incomplete:
        status = "INCOMPLETE"
    elif rejections:
        status = "REJECTED"
    elif reviews:
        status = "REVIEW"
    else:
        status = "ELIGIBLE"
    per_case = [
        {
            "text_id": row["text_id"],
            "seed": row["seed"],
            "style": row["style"],
            "speaker_similarity": row["speaker_similarity"],
            "normalized_cer": row["normalized_cer"],
            "evaluation_status": row["evaluation_status"],
            "rejection_reasons": row.get("rejection_reasons", []),
        }
        for row in metric_rows
    ]
    return {
        "schema_version": EVALUATION_SCHEMA,
        "model_id": next(iter(model_ids)),
        "checkpoint_step": SEARCH_STEP,
        "status": status,
        "case_count": len(rows),
        "hard_gate_metric_case_count": len(metric_rows),
        "speaker_similarity_pass_count": sum(
            value >= MIN_SPEAKER_SIMILARITY for value in similarities
        ),
        "min_speaker_similarity": sorted_similarities[0] if sorted_similarities else None,
        "second_min_speaker_similarity": (
            sorted_similarities[1] if len(sorted_similarities) >= 2 else None
        ),
        "mean_speaker_similarity": (
            sum(similarities) / len(similarities) if similarities else None
        ),
        "style_contrast": style_contrast,
        "per_case_metrics": per_case,
        "rejection_reasons": sorted(rejections),
        "incomplete_reasons": sorted(incomplete),
        "review_reasons": sorted(reviews),
    }


def _style_contrast(rows: Sequence[Mapping[str, object]]) -> float:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row.get("text_id")), _required_int(row, "seed"))
        grouped.setdefault(key, {})[str(row.get("style"))] = row
    values = []
    for pair in grouped.values():
        neutral_audio = pair.get("neutral", {}).get("audio")
        calm_audio = pair.get("calm", {}).get("audio")
        if not isinstance(neutral_audio, dict) or not isinstance(calm_audio, dict):
            continue
        neutral_duration = _optional_positive_float(neutral_audio.get("duration_seconds"))
        calm_duration = _optional_positive_float(calm_audio.get("duration_seconds"))
        neutral_rms = _optional_positive_float(neutral_audio.get("rms"))
        calm_rms = _optional_positive_float(calm_audio.get("rms"))
        if (
            neutral_duration is None
            or calm_duration is None
            or neutral_rms is None
            or calm_rms is None
        ):
            continue
        values.append(
            abs(math.log(calm_duration / neutral_duration))
            + abs(20.0 * math.log10(calm_rms / neutral_rms)) / 20.0
        )
    return round(sum(values) / len(values), 8) if values else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_paths = {
        "search_manifest": args.search_manifest,
        "generation_results": args.generation_results,
        "generation_verification": args.generation_verification,
        "analysis_results": args.analysis_results,
        "metrics_results": args.metrics_results,
        "metrics_provenance": args.metrics_provenance,
    }
    input_snapshots = {
        name: _snapshot_file(path, source=name) for name, path in input_paths.items()
    }
    input_hashes = {name: snapshot[1] for name, snapshot in input_snapshots.items()}
    manifest = load_search_manifest(
        args.search_manifest,
        snapshot=input_snapshots["search_manifest"],
    )
    _validate_generation_evidence(
        args.generation_verification,
        generation_results=args.generation_results,
        search_manifest=args.search_manifest,
        manifest=manifest,
        verification_snapshot=input_snapshots["generation_verification"],
        generation_results_sha256=input_hashes["generation_results"],
        search_manifest_sha256=input_hashes["search_manifest"],
    )
    production = _production()
    generation = production._index_rows(
        _read_jsonl_snapshot(
            input_snapshots["generation_results"],
            path=args.generation_results,
        ),
        source="generation",
    )
    analysis = production._index_rows(
        _read_jsonl_snapshot(
            input_snapshots["analysis_results"],
            path=args.analysis_results,
        ),
        source="analysis",
    )
    metrics = production._index_rows(
        _read_jsonl_snapshot(
            input_snapshots["metrics_results"],
            path=args.metrics_results,
        ),
        source="metrics",
    )
    validate_generation_case_schemas(generation, source="generation")
    validate_generation_case_schemas(analysis, source="analysis")
    validate_artifact_identity(generation, analysis, source="analysis")
    validate_artifact_identity(generation, metrics, source="metrics")
    production._validate_inputs(generation, analysis, metrics)
    model_id = next(iter(manifest.checkpoints))[0]
    validate_case_matrix(
        [
            (
                str(row["model_id"]),
                int(row["checkpoint_step"]),
                str(row["text_id"]),
                int(row["seed"]),
                str(row["style"]),
            )
            for row in generation.values()
        ],
        model_id=model_id,
    )
    production._validate_expected_matrix(generation, manifest=manifest)
    production._validate_checkpoint_contract(generation, manifest=manifest)
    provenance = _read_json_snapshot(
        input_snapshots["metrics_provenance"],
        path=args.metrics_provenance,
    )
    production._validate_metrics_provenance(
        provenance,
        generation_results_sha256=input_hashes["generation_results"],
        manifest=manifest,
    )
    production._validate_metric_rows(
        metrics, generation_results_sha256=input_hashes["generation_results"]
    )
    production._validate_audio_bindings(
        generation,
        analysis,
        metrics_provenance=provenance,
        generation_base=args.generation_results.parent,
        analysis_base=args.analysis_results.parent,
        provenance_base=args.metrics_provenance.parent,
    )
    bound_wavs = _collect_bound_wavs(
        generation,
        metrics_provenance=provenance,
        generation_base=args.generation_results.parent,
        provenance_base=args.metrics_provenance.parent,
    )
    evaluations = bind_search_evaluation_case_schema(
        [
            production._evaluate_case(
                generation[case_id],
                analysis[case_id],
                metrics.get(case_id),
                generation_base=args.generation_results.parent,
                config=production.DEFAULT_CONFIG,
            )
            for case_id in sorted(
                generation, key=lambda value: production._case_sort_key(generation[value])
            )
        ]
    )
    summary = summarize_search(evaluations)
    for name, snapshot in input_snapshots.items():
        _validate_snapshot_unchanged(
            snapshot,
            path=input_paths[name],
            source=name,
        )
    _validate_bound_wavs_unchanged(bound_wavs)
    output_dir = reserve_output(args.output_dir)
    _write_jsonl(output_dir / "search-evaluation-results.jsonl", evaluations)
    _write_json(output_dir / "search-summary.json", summary)
    _write_json(
        output_dir / "search-verification.json",
        {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "PASS",
            "quality_status": summary["status"],
            "input_sha256": input_hashes,
            "case_count": len(evaluations),
            "hard_gate_metric_case_count": summary["hard_gate_metric_case_count"],
            "search_summary_sha256": _sha(output_dir / "search-summary.json"),
            "search_results_sha256": _sha(output_dir / "search-evaluation-results.jsonl"),
            "evaluator_script_sha256": _sha(Path(__file__).resolve()),
            "production_evaluator_script_sha256": _sha(PRODUCTION_EVALUATOR),
        },
    )
    print(json.dumps({"verified": True, "quality_status": summary["status"]}))
    return 0


def _validate_generation_verification(
    payload: Mapping[str, object],
    *,
    generation_results: Path,
    generation_results_sha256: str,
) -> None:
    errors = []
    if payload.get("schema_version") != GENERATION_VERIFICATION_SCHEMA:
        errors.append("generation verification schema mismatch")
    if payload.get("passed") is not True:
        errors.append("generation verification did not pass")
    if payload.get("row_count") != EXPECTED_CASE_COUNT:
        errors.append("generation verification row_count mismatch")
    if payload.get("generation_results_sha256") != generation_results_sha256:
        errors.append("generation verification results SHA mismatch")
    path = _resolved_path(payload.get("generation_results_path"), base=generation_results.parent)
    if path != generation_results.resolve():
        errors.append("generation verification results path mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_generation_evidence(
    verification_path: Path,
    *,
    generation_results: Path,
    search_manifest: Path,
    manifest: Any,
    verification_snapshot: tuple[bytes, str] | None = None,
    generation_results_sha256: str | None = None,
    search_manifest_sha256: str | None = None,
) -> None:
    verification = (
        _read_json_snapshot(verification_snapshot, path=verification_path)
        if verification_snapshot is not None
        else _read_json(verification_path)
    )
    _require_exact_keys(
        verification,
        {
            "schema_version",
            "passed",
            "model_id",
            "case_count",
            "row_count",
            "status_counts",
            "case_ids_unique",
            "all_audio_finite",
            "search_manifest_path",
            "search_manifest_sha256",
            "generation_config_path",
            "generation_config_sha256",
            "generation_results_path",
            "generation_results_sha256",
            "base_checkpoint_path",
            "base_checkpoint_model_id",
            "base_checkpoint_sha256",
            "base_revision",
            "search_generator_script",
            "search_generator_script_sha256",
            "production_generator_script",
            "production_generator_script_sha256",
        },
        source="generation verification",
    )
    generation_sha = generation_results_sha256 or _sha(generation_results)
    _validate_generation_verification(
        verification,
        generation_results=generation_results,
        generation_results_sha256=generation_sha,
    )
    config_path = _resolved_path(
        verification.get("generation_config_path"), base=verification_path.parent
    )
    config_sha = _required_sha(verification, "generation_config_sha256")
    config_snapshot = _snapshot_file(config_path, source="generation config")
    if config_snapshot[1] != config_sha:
        raise ValueError(f"generation config SHA-256 mismatch: {config_path}")
    config = _read_json_snapshot(config_snapshot, path=config_path)
    _require_exact_keys(
        config,
        {
            "schema_version",
            "model_id",
            "case_count",
            "search_manifest_path",
            "search_manifest_sha256",
            "search_generator_script",
            "search_generator_script_sha256",
            "production_generator_script",
            "production_generator_script_sha256",
            "base_checkpoint_path",
            "base_checkpoint_model_id",
            "base_checkpoint_sha256",
            "base_revision",
            "text_ids",
            "seeds",
            "styles",
        },
        source="generation config",
    )
    if config.get("schema_version") != GENERATION_CONFIG_SCHEMA:
        raise ValueError("generation config schema mismatch")
    manifest_path = search_manifest.resolve()
    manifest_sha = search_manifest_sha256 or _sha(manifest_path)
    model_id, checkpoint_step = next(iter(manifest.checkpoints))
    contract = manifest.checkpoints[model_id, checkpoint_step]
    base_path = _resolved_path(config.get("base_checkpoint_path"), base=config_path.parent)
    expected = {
        "model_id": model_id,
        "case_count": EXPECTED_CASE_COUNT,
        "search_manifest_path": str(manifest_path),
        "search_manifest_sha256": manifest_sha,
        "base_checkpoint_path": str(base_path),
        "base_checkpoint_model_id": contract.base_checkpoint,
        "base_checkpoint_sha256": contract.base_checkpoint_sha256,
        "base_revision": contract.base_revision,
        "search_generator_script": str(SEARCH_GENERATOR.resolve()),
        "search_generator_script_sha256": _sha(SEARCH_GENERATOR),
        "production_generator_script": str(PRODUCTION_GENERATOR.resolve()),
        "production_generator_script_sha256": _sha(PRODUCTION_GENERATOR),
    }
    config_mismatches = [field for field, value in expected.items() if config.get(field) != value]
    if tuple(config.get("text_ids", ())) != TEXT_IDS:
        config_mismatches.append("text_ids")
    if tuple(config.get("seeds", ())) != SEEDS:
        config_mismatches.append("seeds")
    if tuple(config.get("styles", ())) != STYLES:
        config_mismatches.append("styles")
    if config_mismatches:
        raise ValueError(f"generation config contract mismatch: {sorted(set(config_mismatches))}")
    _validate_file_hash(base_path, contract.base_checkpoint_sha256, source="base checkpoint")
    verification_expected = expected | {
        "case_count": EXPECTED_CASE_COUNT,
        "row_count": EXPECTED_CASE_COUNT,
        "generation_config_path": str(config_path.resolve()),
        "generation_config_sha256": config_sha,
        "generation_results_path": str(generation_results.resolve()),
        "generation_results_sha256": generation_sha,
        "status_counts": {"SUCCESS": EXPECTED_CASE_COUNT},
        "case_ids_unique": True,
        "all_audio_finite": True,
    }
    verification_mismatches = [
        field for field, value in verification_expected.items() if verification.get(field) != value
    ]
    if verification_mismatches:
        raise ValueError(
            f"generation verification contract mismatch: {sorted(verification_mismatches)}"
        )
    _validate_snapshot_unchanged(
        config_snapshot,
        path=config_path,
        source="generation config",
    )


def _require_exact_keys(
    row: Mapping[str, object],
    expected: set[str],
    *,
    source: str,
) -> None:
    if set(row) != expected:
        raise ValueError(f"{source} keys must exactly match {sorted(expected)}")


def _collect_reasons(rows: Sequence[Mapping[str, object]], field: str) -> set[str]:
    reasons: set[str] = set()
    for row in rows:
        raw = row.get(field, [])
        if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
            raise TypeError(f"{field} must be a list of strings")
        reasons.update(raw)
    return reasons


def _required_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _required_finite_float(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _optional_finite_float(row: Mapping[str, object], field: str) -> float | None:
    if row.get(field) is None:
        return None
    return _required_finite_float(row, field)


def _optional_positive_float(value: object) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        return None
    return float(value)


def _required_string(row: Mapping[str, object], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} {field} must be nonempty")
    return value


def _required_sha(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_file_hash(path: Path, expected: str, *, source: str) -> None:
    if not path.is_file() or _sha(path) != expected:
        raise ValueError(f"{source} SHA-256 mismatch: {path}")


def _resolved_path(value: object, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be nonempty")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _snapshot_file(path: Path, *, source: str) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot snapshot {source}: {path}") from exc
    return payload, hashlib.sha256(payload).hexdigest()


def _read_json_snapshot(snapshot: tuple[bytes, str], *, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _read_jsonl_snapshot(snapshot: tuple[bytes, str], *, path: Path) -> list[dict[str, Any]]:
    try:
        lines = snapshot[0].decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid JSONL: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_snapshot_unchanged(
    snapshot: tuple[bytes, str],
    *,
    path: Path,
    source: str,
) -> None:
    if not path.is_file() or _sha(path) != snapshot[1]:
        raise ValueError(f"{source} changed after input snapshot: {path}")


def _collect_bound_wavs(
    generation: Mapping[str, Mapping[str, object]],
    *,
    metrics_provenance: Mapping[str, object],
    generation_base: Path,
    provenance_base: Path,
) -> dict[Path, str]:
    bindings: dict[Path, str] = {}

    def bind(raw_path: object, raw_sha: object, *, base: Path, source: str) -> None:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{source} path must be nonempty")
        if (
            not isinstance(raw_sha, str)
            or len(raw_sha) != 64
            or any(character not in "0123456789abcdef" for character in raw_sha)
        ):
            raise ValueError(f"{source} SHA-256 must be lowercase")
        raw = Path(raw_path)
        path = raw if raw.is_absolute() else base / raw
        previous = bindings.get(path)
        if previous is not None and previous != raw_sha:
            raise ValueError(f"{source} has conflicting WAV bindings: {path}")
        bindings[path] = raw_sha

    for case_id, row in generation.items():
        if row.get("status") == "SUCCESS":
            bind(
                row.get("wav_path"),
                row.get("wav_sha256"),
                base=generation_base,
                source=f"{case_id} generated WAV",
            )
    input_sha256 = metrics_provenance.get("input_sha256")
    if not isinstance(input_sha256, dict):
        raise TypeError("metrics provenance input_sha256 must be an object")
    for group in ("generated_audio", "reference_audio"):
        artifacts = input_sha256.get(group)
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f"metrics provenance {group} must be nonempty")
        for raw_path, raw_sha in artifacts.items():
            bind(
                raw_path,
                raw_sha,
                base=provenance_base,
                source=f"metrics provenance {group}",
            )
    return bindings


def _validate_bound_wavs_unchanged(bindings: Mapping[Path, str]) -> None:
    for path, expected in bindings.items():
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"bound WAV changed after validation: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-manifest", type=Path, required=True)
    parser.add_argument("--generation-results", type=Path, required=True)
    parser.add_argument("--generation-verification", type=Path, required=True)
    parser.add_argument("--analysis-results", type=Path, required=True)
    parser.add_argument("--metrics-results", type=Path, required=True)
    parser.add_argument("--metrics-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
