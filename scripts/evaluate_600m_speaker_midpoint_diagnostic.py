# ruff: noqa: ANN401, EM101, EM102, PLR0914, PLR2004, SLF001, TRY003
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

DIAGNOSTIC_KIND = "derived diagnostic embedding"
CASE_IDENTITY_STEP = 0
EXPECTED_CASE_COUNT = 28
EXPECTED_METRIC_CASE_COUNT = 16
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
CASE_SCHEMA = "speaker-derived-diagnostic-generation-case/v1"
EVALUATION_CASE_SCHEMA = "speaker-derived-diagnostic-evaluation-case/v1"
SUMMARY_SCHEMA = "speaker-derived-diagnostic-evaluation/v1"
VERIFICATION_SCHEMA = "speaker-derived-diagnostic-evaluation-verification/v1"
GENERATION_SCHEMA = "speaker-derived-diagnostic-generation/v1"
GENERATION_VERIFICATION_SCHEMA = "speaker-derived-diagnostic-generation-verification/v1"
PRODUCTION_EVALUATOR = Path(__file__).with_name("evaluate_600m_speaker_checkpoints.py")
SEARCH_EVALUATOR = Path(__file__).with_name("evaluate_600m_speaker_checkpoint_search.py")
GENERATOR = Path(__file__).with_name("generate_600m_speaker_midpoint_diagnostic_remote.py")
PRODUCTION_GENERATOR = Path(__file__).with_name("generate_600m_checkpoint_audio_remote.py")
IDENTITY_FIELDS = ("wav_path", "wav_sha256", "provenance")


def _production() -> ModuleType:
    return _load_module("_speaker_midpoint_production_evaluator", PRODUCTION_EVALUATOR)


def _search() -> ModuleType:
    return _load_module("_speaker_midpoint_search_evaluator", SEARCH_EVALUATOR)


def _generator() -> ModuleType:
    return _load_module("_speaker_midpoint_generator_for_evaluator", GENERATOR)


def _load_module(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def midpoint_decision(*, status: str, min_similarity: float | None) -> str:
    if status != "ELIGIBLE" or min_similarity is None or not math.isfinite(min_similarity):
        return "FAILED_DIAGNOSTIC"
    if min_similarity >= 0.753:
        return "STRONG_DIAGNOSTIC"
    if min_similarity >= 0.75:
        return "BOUNDARY_DIAGNOSTIC"
    return "FAILED_DIAGNOSTIC"


def validate_case_matrix(
    identities: Sequence[tuple[str, int, str, int, str]], *, model_id: str
) -> None:
    expected = {
        (model_id, CASE_IDENTITY_STEP, text_id, seed, style)
        for text_id in TEXT_IDS
        for seed in SEEDS
        for style in STYLES
    }
    actual = set(identities)
    if len(actual) != len(identities):
        raise ValueError("duplicate derived diagnostic matrix case")
    if missing := expected - actual:
        raise ValueError(f"derived diagnostic matrix missing expected case: {min(missing)}")
    if extra := actual - expected:
        raise ValueError(f"derived diagnostic matrix has unexpected case: {min(extra)}")


def load_evaluation_manifest(
    path: Path, *, snapshot: tuple[bytes, str] | None = None
) -> tuple[Any, Any]:
    manifest_path = _require_regular_direct_file(path, source="derived manifest")
    manifest_snapshot = snapshot or _snapshot(manifest_path, source="derived manifest")
    if _sha(manifest_path) != manifest_snapshot[1]:
        raise ValueError("derived manifest snapshot mismatch")
    plan = _generator().load_derived_plan(manifest_path)
    if plan.evaluation_manifest_sha256 != manifest_snapshot[1]:
        raise ValueError("derived manifest changed while loading")
    payload = json.loads(manifest_snapshot[0].decode("utf-8"))
    metrics = _required_mapping(payload, "metrics_provenance")
    speaker = _required_mapping(metrics, "speaker_embedding")
    transcription = _required_mapping(metrics, "transcription")
    production = _production()
    checkpoint = plan.checkpoints[0]
    contract = production.CheckpointContract(
        model_id=plan.model_id,
        checkpoint_step=CASE_IDENTITY_STEP,
        embedding_path=checkpoint.embedding_path,
        embedding_sha256=checkpoint.embedding_sha256,
        training_config_sha256=plan.derivation_sha256,
        base_checkpoint=plan.base_checkpoint,
        base_checkpoint_sha256=plan.base_checkpoint_sha256,
        base_revision=plan.base_revision,
        run_id=checkpoint.run_id,
        evaluation_manifest_sha256=plan.evaluation_manifest_sha256,
    )
    production._validate_embedding(contract)
    manifest = production.EvaluationManifest(
        checkpoints={(plan.model_id, CASE_IDENTITY_STEP): contract},
        text_ids=TEXT_IDS,
        seeds=SEEDS,
        styles=STYLES,
        reference_wavs_sha256=_required_sha(metrics, "reference_wavs_sha256"),
        speaker_embedding_model_id=_required_string(speaker, "model_id"),
        speaker_embedding_revision=_required_string(speaker, "revision"),
        speaker_embedding_source_sha256=_required_sha(speaker, "source_sha256"),
        transcription_model_id=_required_string(transcription, "model_id"),
        transcription_revision=_required_string(transcription, "revision"),
        transcription_source_sha256=_required_sha(transcription, "source_sha256"),
    )
    return manifest, plan


def summarize_derived(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    # Reuse the frozen search quality gate by changing only its synthetic case identity.
    search_rows = [{**row, "checkpoint_step": _search().SEARCH_STEP} for row in rows]
    summary = _search().summarize_search(search_rows)
    minimum = summary["min_speaker_similarity"]
    decision = midpoint_decision(
        status=str(summary["status"]),
        min_similarity=float(minimum) if isinstance(minimum, int | float) else None,
    )
    return {
        **summary,
        "schema_version": SUMMARY_SCHEMA,
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "checkpoint_step": CASE_IDENTITY_STEP,
        "checkpoint_step_semantics": "synthetic case identity only; not a training step",
        "midpoint_decision": decision,
        "production_promotion_allowed": False,
        "decision_policy": {
            "strong": "quality status ELIGIBLE and min speaker similarity >= 0.753",
            "boundary": "quality status ELIGIBLE and 0.750 <= min speaker similarity < 0.753",
            "fail": "all other outcomes",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dependency_snapshots = _snapshot_dependencies()
    dependency_hashes = {path: snapshot[1] for path, snapshot in dependency_snapshots}
    input_paths = {
        "derived_manifest": args.derived_manifest,
        "generation_results": args.generation_results,
        "generation_verification": args.generation_verification,
        "analysis_results": args.analysis_results,
        "metrics_results": args.metrics_results,
        "metrics_provenance": args.metrics_provenance,
    }
    snapshots = {name: _snapshot(path, source=name) for name, path in input_paths.items()}
    hashes = {name: snapshot[1] for name, snapshot in snapshots.items()}
    manifest, plan = load_evaluation_manifest(
        args.derived_manifest, snapshot=snapshots["derived_manifest"]
    )
    _validate_generation_evidence(
        args.generation_verification,
        generation_results=args.generation_results,
        derived_manifest=args.derived_manifest,
        plan=plan,
        verification_snapshot=snapshots["generation_verification"],
        generation_results_sha256=hashes["generation_results"],
        dependency_hashes=dependency_hashes,
    )
    production = _production()
    generation = production._index_rows(
        _read_jsonl_snapshot(snapshots["generation_results"], path=args.generation_results),
        source="generation",
    )
    analysis = production._index_rows(
        _read_jsonl_snapshot(snapshots["analysis_results"], path=args.analysis_results),
        source="analysis",
    )
    metrics = production._index_rows(
        _read_jsonl_snapshot(snapshots["metrics_results"], path=args.metrics_results),
        source="metrics",
    )
    _validate_case_schemas(generation, source="generation")
    _validate_case_schemas(analysis, source="analysis")
    _validate_artifact_identity(generation, analysis, source="analysis")
    _validate_artifact_identity(generation, metrics, source="metrics")
    production._validate_inputs(generation, analysis, metrics)
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
        model_id=plan.model_id,
    )
    production._validate_expected_matrix(generation, manifest=manifest)
    _validate_derived_checkpoint_contract(generation, plan=plan)
    provenance = _read_json_snapshot(snapshots["metrics_provenance"], path=args.metrics_provenance)
    production._validate_metrics_provenance(
        provenance,
        generation_results_sha256=hashes["generation_results"],
        manifest=manifest,
    )
    production._validate_metric_rows(
        metrics, generation_results_sha256=hashes["generation_results"]
    )
    production._validate_audio_bindings(
        generation,
        analysis,
        metrics_provenance=provenance,
        generation_base=args.generation_results.parent,
        analysis_base=args.analysis_results.parent,
        provenance_base=args.metrics_provenance.parent,
    )
    bound_wavs = _search()._collect_bound_wavs(
        generation,
        metrics_provenance=provenance,
        generation_base=args.generation_results.parent,
        provenance_base=args.metrics_provenance.parent,
    )
    evaluated = [
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
    evaluations = [
        {**row, "evaluation_schema_version": EVALUATION_CASE_SCHEMA} for row in evaluated
    ]
    summary = summarize_derived(evaluations)
    for name, snapshot in snapshots.items():
        _validate_snapshot_unchanged(snapshot, path=input_paths[name], source=name)
    _search()._validate_bound_wavs_unchanged(bound_wavs)
    _generator()._validate_plan_unchanged(plan)
    _validate_dependencies_unchanged(dependency_snapshots)
    output = reserve_output(args.output_dir)
    results_path = output / "midpoint-evaluation-results.jsonl"
    summary_path = output / "midpoint-summary.json"
    _write_jsonl(results_path, evaluations)
    _write_json(summary_path, summary)
    _write_json(
        output / "midpoint-verification.json",
        {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "PASS",
            "diagnostic_kind": DIAGNOSTIC_KIND,
            "quality_status": summary["status"],
            "midpoint_decision": summary["midpoint_decision"],
            "production_promotion_allowed": False,
            "input_sha256": hashes,
            "case_count": len(evaluations),
            "hard_gate_metric_case_count": summary["hard_gate_metric_case_count"],
            "summary_sha256": _sha(summary_path),
            "results_sha256": _sha(results_path),
            "evaluator_script_sha256": dependency_hashes[Path(__file__).resolve()],
            "dependency_scripts": [
                {"path": str(path), "sha256": snapshot[1]}
                for path, snapshot in dependency_snapshots
            ],
        },
    )
    _validate_dependencies_unchanged(dependency_snapshots)
    print(
        json.dumps(
            {
                "verified": True,
                "quality_status": summary["status"],
                "midpoint_decision": summary["midpoint_decision"],
            }
        )
    )
    return 0


def _validate_derived_checkpoint_contract(
    rows: Mapping[str, Mapping[str, object]], *, plan: Any
) -> None:
    checkpoint = plan.checkpoints[0]
    expected_provenance = {
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "derivation_sha256": plan.derivation_sha256,
        "original_embedding_sha256": plan.original_embedding_sha256,
        "candidate_f_embedding_sha256": plan.candidate_f_embedding_sha256,
        "base_checkpoint": plan.base_checkpoint,
        "base_revision": plan.base_revision,
    }
    for case_id, row in rows.items():
        expected = {
            "model_id": plan.model_id,
            "checkpoint_step": CASE_IDENTITY_STEP,
            "checkpoint": plan.base_checkpoint,
            "speaker_filename": checkpoint.embedding_path.name,
            "embedding_path": str(checkpoint.embedding_path),
            "embedding_sha256": checkpoint.embedding_sha256,
            "evaluation_manifest_sha256": plan.evaluation_manifest_sha256,
            "base_checkpoint_sha256": plan.base_checkpoint_sha256,
            "provenance": expected_provenance,
        }
        mismatches = [field for field, value in expected.items() if row.get(field) != value]
        if mismatches:
            raise ValueError(f"{case_id}: derived checkpoint contract mismatch: {mismatches}")


def _validate_generation_evidence(
    verification_path: Path,
    *,
    generation_results: Path,
    derived_manifest: Path,
    plan: Any,
    verification_snapshot: tuple[bytes, str],
    generation_results_sha256: str,
    dependency_hashes: Mapping[Path, str],
) -> None:
    verification = _read_json_snapshot(verification_snapshot, path=verification_path)
    if verification.get("schema_version") != GENERATION_VERIFICATION_SCHEMA:
        raise ValueError("derived generation verification schema mismatch")
    _require_exact_keys(
        verification,
        {
            "schema_version",
            "diagnostic_kind",
            "passed",
            "model_id",
            "case_count",
            "row_count",
            "status_counts",
            "case_ids_unique",
            "all_audio_finite",
            "derived_manifest_path",
            "derived_manifest_sha256",
            "derivation_path",
            "derivation_sha256",
            "original_embedding_sha256",
            "candidate_f_embedding_sha256",
            "generation_config_path",
            "generation_config_sha256",
            "generation_results_path",
            "generation_results_sha256",
            "base_checkpoint_path",
            "base_checkpoint_model_id",
            "base_checkpoint_sha256",
            "base_revision",
            "generator_script",
            "generator_script_sha256",
            "production_generator_script",
            "production_generator_script_sha256",
            "dependency_scripts",
        },
        source="derived generation verification",
    )
    config_path = _resolved_path(
        verification.get("generation_config_path"), base=verification_path.parent
    )
    config_snapshot = _snapshot(config_path, source="generation config")
    config = _read_json_snapshot(config_snapshot, path=config_path)
    if config.get("schema_version") != GENERATION_SCHEMA:
        raise ValueError("derived generation config schema mismatch")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "diagnostic_kind",
            "model_id",
            "case_count",
            "synthetic_checkpoint_step",
            "synthetic_checkpoint_step_semantics",
            "derived_manifest_path",
            "derived_manifest_sha256",
            "derivation_path",
            "derivation_sha256",
            "original_embedding_sha256",
            "candidate_f_embedding_sha256",
            "derived_embedding_path",
            "derived_embedding_sha256",
            "generator_script",
            "generator_script_sha256",
            "production_generator_script",
            "production_generator_script_sha256",
            "dependency_scripts",
            "base_checkpoint_path",
            "base_checkpoint_model_id",
            "base_checkpoint_sha256",
            "base_revision",
            "text_ids",
            "seeds",
            "styles",
        },
        source="derived generation config",
    )
    base_path = _resolved_path(config.get("base_checkpoint_path"), base=config_path.parent)
    expected_common = {
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "model_id": plan.model_id,
        "derived_manifest_path": str(derived_manifest.resolve()),
        "derived_manifest_sha256": plan.evaluation_manifest_sha256,
        "derivation_path": str(plan.derivation_path),
        "derivation_sha256": plan.derivation_sha256,
        "original_embedding_sha256": plan.original_embedding_sha256,
        "candidate_f_embedding_sha256": plan.candidate_f_embedding_sha256,
        "base_checkpoint_path": str(base_path),
        "base_checkpoint_model_id": plan.base_checkpoint,
        "base_checkpoint_sha256": plan.base_checkpoint_sha256,
        "base_revision": plan.base_revision,
        "generator_script": str(GENERATOR.resolve()),
        "generator_script_sha256": dependency_hashes[GENERATOR.resolve()],
        "production_generator_script": str(PRODUCTION_GENERATOR.resolve()),
        "production_generator_script_sha256": dependency_hashes[PRODUCTION_GENERATOR.resolve()],
        "dependency_scripts": config.get("dependency_scripts"),
    }
    expected_generator_dependencies = [
        {"path": str(path), "sha256": snapshot[1]} for path, snapshot in plan.dependency_snapshots
    ]
    if config.get("dependency_scripts") != expected_generator_dependencies:
        raise ValueError("derived generator dependency snapshot mismatch")
    expected_common["dependency_scripts"] = expected_generator_dependencies
    config_expected = expected_common | {
        "case_count": EXPECTED_CASE_COUNT,
        "synthetic_checkpoint_step": CASE_IDENTITY_STEP,
        "synthetic_checkpoint_step_semantics": "case identity only; not a training step",
        "derived_embedding_path": str(plan.checkpoints[0].embedding_path),
        "derived_embedding_sha256": plan.checkpoints[0].embedding_sha256,
    }
    mismatches = [field for field, value in config_expected.items() if config.get(field) != value]
    for field, expected in (("text_ids", TEXT_IDS), ("seeds", SEEDS), ("styles", STYLES)):
        if tuple(config.get(field, ())) != expected:
            mismatches.append(field)
    if mismatches:
        raise ValueError(f"derived generation config contract mismatch: {sorted(set(mismatches))}")
    if _sha(base_path) != plan.base_checkpoint_sha256:
        raise ValueError("base checkpoint SHA-256 mismatch")
    verification_expected = expected_common | {
        "passed": True,
        "case_count": EXPECTED_CASE_COUNT,
        "row_count": EXPECTED_CASE_COUNT,
        "status_counts": {"SUCCESS": EXPECTED_CASE_COUNT},
        "case_ids_unique": True,
        "all_audio_finite": True,
        "generation_config_path": str(config_path),
        "generation_config_sha256": config_snapshot[1],
        "generation_results_path": str(generation_results.resolve()),
        "generation_results_sha256": generation_results_sha256,
    }
    mismatches = [
        field for field, value in verification_expected.items() if verification.get(field) != value
    ]
    if mismatches:
        raise ValueError(f"derived generation verification contract mismatch: {sorted(mismatches)}")
    _validate_snapshot_unchanged(config_snapshot, path=config_path, source="generation config")


def _validate_case_schemas(rows: Mapping[str, Mapping[str, object]], *, source: str) -> None:
    for case_id, row in rows.items():
        if row.get("schema_version") != CASE_SCHEMA:
            raise ValueError(f"{case_id}: {source} derived row schema mismatch")


def _validate_artifact_identity(
    generation: Mapping[str, Mapping[str, object]],
    other: Mapping[str, Mapping[str, object]],
    *,
    source: str,
) -> None:
    if generation.keys() != other.keys():
        raise ValueError(f"{source} artifact identity case set mismatch")
    for case_id, row in generation.items():
        if any(row.get(field) != other[case_id].get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"{case_id}: {source} artifact identity mismatch")


def reserve_output(path: Path) -> Path:
    output = _prepare_output_parent(path, source="derived diagnostic evaluation")
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite derived evaluation: {output}") from exc
    _require_nominal_directory(output, source="derived evaluation output")
    return output.resolve(strict=True)


def _prepare_output_parent(path: Path, *, source: str) -> Path:
    output = Path(os.path.abspath(path))  # noqa: PTH100
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
        break
    for ancestor in (candidate, *candidate.parents):
        if ancestor != ancestor.parent:
            _require_no_alias(ancestor, source=f"{source} parent")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=False)
        _require_nominal_directory(directory, source=f"{source} parent")
    return output


def _require_regular_direct_file(path: Path, *, source: str) -> Path:
    nominal = Path(os.path.abspath(path))  # noqa: PTH100
    for candidate in (nominal, *nominal.parents):
        if candidate == candidate.parent:
            break
        _require_no_alias(candidate, source=source)
    metadata = nominal.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{source} must be a regular non-alias file: {nominal}")
    return nominal.resolve(strict=True)


def _require_no_alias(path: Path, *, source: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{source} must not use symlink, junction, or reparse aliases: {path}")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse and getattr(metadata, "st_file_attributes", 0) & reparse:
        raise ValueError(f"{source} must not use symlink, junction, or reparse aliases: {path}")


def _require_nominal_directory(path: Path, *, source: str) -> None:
    _require_no_alias(path, source=source)
    if not path.is_dir():
        raise ValueError(f"{source} must be a directory: {path}")


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
        raise ValueError(f"{field} must be nonempty")
    return value


def _required_sha(row: Mapping[str, object], field: str) -> str:
    value = _required_string(row, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _resolved_path(value: object, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be nonempty")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _snapshot(path: Path, *, source: str) -> tuple[bytes, str]:
    resolved = _require_regular_direct_file(path, source=source)
    payload = resolved.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _snapshot_dependencies() -> tuple[tuple[Path, tuple[bytes, str]], ...]:
    return tuple(
        (path.resolve(strict=True), _snapshot(path, source=f"dependency {path.name}"))
        for path in (
            Path(__file__),
            PRODUCTION_EVALUATOR,
            SEARCH_EVALUATOR,
            GENERATOR,
            PRODUCTION_GENERATOR,
        )
    )


def _validate_dependencies_unchanged(
    snapshots: Sequence[tuple[Path, tuple[bytes, str]]],
) -> None:
    for path, snapshot in snapshots:
        if _snapshot(path, source=f"dependency {path.name}") != snapshot:
            raise ValueError(f"evaluator dependency changed after snapshot: {path}")


def _read_json_snapshot(snapshot: tuple[bytes, str], *, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON must be an object: {path}")
    return payload


def _read_jsonl_snapshot(snapshot: tuple[bytes, str], *, path: Path) -> list[dict[str, Any]]:
    try:
        lines = snapshot[0].decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid JSONL: {path}") from exc
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row must be an object at {path}:{number}")
        rows.append(row)
    return rows


def _validate_snapshot_unchanged(snapshot: tuple[bytes, str], *, path: Path, source: str) -> None:
    if _snapshot(path, source=source) != snapshot:
        raise ValueError(f"{source} changed after input snapshot: {path}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-manifest", type=Path, required=True)
    parser.add_argument("--generation-results", type=Path, required=True)
    parser.add_argument("--generation-verification", type=Path, required=True)
    parser.add_argument("--analysis-results", type=Path, required=True)
    parser.add_argument("--metrics-results", type=Path, required=True)
    parser.add_argument("--metrics-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
