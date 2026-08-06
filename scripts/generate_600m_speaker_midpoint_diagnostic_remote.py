# ruff: noqa: EM101, EM102, SLF001, TRY003
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

DERIVED_MANIFEST_SCHEMA = "speaker-derived-diagnostic-manifest/v1"
GENERATION_SCHEMA = "speaker-derived-diagnostic-generation/v1"
VERIFICATION_SCHEMA = "speaker-derived-diagnostic-generation-verification/v1"
CASE_SCHEMA = "speaker-derived-diagnostic-generation-case/v1"
DIAGNOSTIC_KIND = "derived diagnostic embedding"
CASE_IDENTITY_STEP = 0
EXPECTED_CASE_COUNT = 28
PRODUCTION_GENERATOR = Path(__file__).with_name("generate_600m_checkpoint_audio_remote.py")
DERIVER = Path(__file__).with_name("derive_600m_speaker_midpoint_diagnostic.py")


@dataclass(frozen=True, slots=True)
class DerivedPlan:
    model_id: str
    checkpoints: tuple[Any, ...]
    evaluation_manifest_path: Path
    evaluation_manifest_sha256: str
    base_checkpoint: str
    base_checkpoint_sha256: str
    base_revision: str
    derivation_path: Path
    derivation_sha256: str
    original_embedding_sha256: str
    candidate_f_embedding_sha256: str
    dependency_snapshots: tuple[tuple[Path, tuple[bytes, str]], ...]


def _production() -> ModuleType:
    return _load_module("_speaker_midpoint_production_generator", PRODUCTION_GENERATOR)


def _deriver() -> ModuleType:
    return _load_module("_speaker_midpoint_deriver", DERIVER)


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


def load_derived_plan(path: Path) -> DerivedPlan:
    dependency_snapshots = _snapshot_dependencies()
    manifest_path = _require_regular_direct_file(path, source="derived diagnostic manifest")
    manifest_snapshot = _snapshot(manifest_path)
    payload = _deriver().validate_derived_manifest(manifest_path, snapshot=manifest_snapshot)
    if payload.get("schema_version") != DERIVED_MANIFEST_SCHEMA:
        raise ValueError("derived diagnostic manifest schema mismatch")
    embedding = _required_mapping(payload, "derived_embedding")
    base = _required_mapping(payload, "base_model")
    parents = _required_mapping(payload, "parents")
    original = _required_mapping(parents, "original_step1000")
    candidate_f = _required_mapping(parents, "candidate_f")
    derivation = _required_mapping(payload, "derivation")
    production = _production()
    candidate = production.CheckpointCandidate(
        checkpoint_step=CASE_IDENTITY_STEP,
        embedding_path=_resolved_path(embedding, "path", manifest_path.parent),
        embedding_sha256=production._required_sha256(embedding, "sha256"),
        # Internal adapter value only. It is never emitted as training provenance.
        training_config_sha256=production._required_sha256(derivation, "sha256"),
        base_checkpoint=_required_string(base, "model_id"),
        base_checkpoint_sha256=production._required_sha256(base, "checkpoint_sha256"),
        base_revision=_required_string(base, "revision"),
        run_id="derived-diagnostic-midpoint-alpha-0.5",
    )
    _deriver().read_embedding(candidate.embedding_path)
    plan = DerivedPlan(
        model_id=_required_string(payload, "model_id"),
        checkpoints=(candidate,),
        evaluation_manifest_path=manifest_path,
        evaluation_manifest_sha256=manifest_snapshot[1],
        base_checkpoint=candidate.base_checkpoint,
        base_checkpoint_sha256=candidate.base_checkpoint_sha256,
        base_revision=candidate.base_revision,
        derivation_path=_resolved_path(derivation, "path", manifest_path.parent),
        derivation_sha256=candidate.training_config_sha256,
        original_embedding_sha256=production._required_sha256(original, "sha256"),
        candidate_f_embedding_sha256=production._required_sha256(candidate_f, "sha256"),
        dependency_snapshots=dependency_snapshots,
    )
    _validate_plan_unchanged(plan)
    return plan


def build_derived_cases(plan: DerivedPlan) -> tuple[Any, ...]:
    cases = tuple(_production().build_cases(plan))
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"midpoint generation requires exactly {EXPECTED_CASE_COUNT} cases")
    if any(case.checkpoint.checkpoint_step != CASE_IDENTITY_STEP for case in cases):
        raise ValueError("midpoint cases must use synthetic identity step 0")
    return cases


def bind_derived_case_provenance(
    row: Mapping[str, object], *, plan: DerivedPlan
) -> dict[str, object]:
    if row.get("checkpoint_step") != CASE_IDENTITY_STEP:
        raise ValueError("derived generation row must use synthetic identity step 0")
    return {
        **row,
        "schema_version": CASE_SCHEMA,
        "provenance": {
            "diagnostic_kind": DIAGNOSTIC_KIND,
            "derivation_sha256": plan.derivation_sha256,
            "original_embedding_sha256": plan.original_embedding_sha256,
            "candidate_f_embedding_sha256": plan.candidate_f_embedding_sha256,
            "base_checkpoint": plan.base_checkpoint,
            "base_revision": plan.base_revision,
        },
    }


def build_generation_config(*, plan: DerivedPlan, base_checkpoint_path: Path) -> dict[str, object]:
    production = _production()
    return {
        "schema_version": GENERATION_SCHEMA,
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "model_id": plan.model_id,
        "case_count": EXPECTED_CASE_COUNT,
        "synthetic_checkpoint_step": CASE_IDENTITY_STEP,
        "synthetic_checkpoint_step_semantics": "case identity only; not a training step",
        "derived_manifest_path": str(plan.evaluation_manifest_path),
        "derived_manifest_sha256": plan.evaluation_manifest_sha256,
        "derivation_path": str(plan.derivation_path),
        "derivation_sha256": plan.derivation_sha256,
        "original_embedding_sha256": plan.original_embedding_sha256,
        "candidate_f_embedding_sha256": plan.candidate_f_embedding_sha256,
        "derived_embedding_path": str(plan.checkpoints[0].embedding_path),
        "derived_embedding_sha256": plan.checkpoints[0].embedding_sha256,
        "generator_script": str(Path(__file__).resolve()),
        "generator_script_sha256": _dependency_sha(plan, Path(__file__)),
        "production_generator_script": str(PRODUCTION_GENERATOR.resolve()),
        "production_generator_script_sha256": _dependency_sha(plan, PRODUCTION_GENERATOR),
        "dependency_scripts": [
            {"path": str(path), "sha256": snapshot[1]}
            for path, snapshot in plan.dependency_snapshots
        ],
        "base_checkpoint_path": str(base_checkpoint_path.resolve()),
        "base_checkpoint_model_id": plan.base_checkpoint,
        "base_checkpoint_sha256": plan.base_checkpoint_sha256,
        "base_revision": plan.base_revision,
        "text_ids": list(production.EXPECTED_TEXT_IDS),
        "seeds": list(production.EXPECTED_SEEDS),
        "styles": list(production.EXPECTED_STYLES),
    }


def generate_derived(
    *,
    plan: DerivedPlan,
    base_checkpoint_path: Path,
    upstream_root: Path,
    output_path: Path,
) -> int:
    production = _production()
    production.validate_base_checkpoint(base_checkpoint_path, plan=plan)
    cases = build_derived_cases(plan)
    output, wav_dir = reserve_output(output_path)
    config_path = output / "generation-config.json"
    _write_json(
        config_path,
        build_generation_config(plan=plan, base_checkpoint_path=base_checkpoint_path),
    )
    results_path = output / "generation-results.jsonl"
    runtime_api = production._load_runtime_api(upstream_root)
    runtime = production._create_runtime(runtime_api, checkpoint_path=base_checkpoint_path)
    rows: list[dict[str, object]] = []
    try:
        with results_path.open("x", encoding="utf-8", newline="\n") as destination:
            for index, case in enumerate(cases, start=1):
                generated = production._generate_case_result(
                    case,
                    plan=plan,
                    wav_dir=wav_dir,
                    runtime=runtime,
                    runtime_api=runtime_api,
                )
                row = bind_derived_case_provenance(generated, plan=plan)
                rows.append(row)
                destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                destination.flush()
                print(f"[{index}/{len(cases)}] {case.case_id}: {row['status']}", flush=True)
    finally:
        production._unload_runtime(runtime)
    counts = Counter(str(row["status"]) for row in rows)
    passed = (
        len(rows) == EXPECTED_CASE_COUNT
        and counts == {"SUCCESS": EXPECTED_CASE_COUNT}
        and len({str(row["case_id"]) for row in rows}) == EXPECTED_CASE_COUNT
        and all(row.get("audio_finite") is True for row in rows)
    )
    verification_path = output / "generation-verification.json"
    verification = {
        "schema_version": VERIFICATION_SCHEMA,
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "passed": passed,
        "model_id": plan.model_id,
        "case_count": EXPECTED_CASE_COUNT,
        "row_count": len(rows),
        "status_counts": dict(counts),
        "case_ids_unique": len({str(row["case_id"]) for row in rows}) == len(rows),
        "all_audio_finite": all(row.get("audio_finite") is True for row in rows),
        "derived_manifest_path": str(plan.evaluation_manifest_path),
        "derived_manifest_sha256": plan.evaluation_manifest_sha256,
        "derivation_path": str(plan.derivation_path),
        "derivation_sha256": plan.derivation_sha256,
        "original_embedding_sha256": plan.original_embedding_sha256,
        "candidate_f_embedding_sha256": plan.candidate_f_embedding_sha256,
        "generation_config_path": str(config_path),
        "generation_config_sha256": production.sha256_file(config_path),
        "generation_results_path": str(results_path),
        "generation_results_sha256": production.sha256_file(results_path),
        "base_checkpoint_path": str(base_checkpoint_path.resolve()),
        "base_checkpoint_model_id": plan.base_checkpoint,
        "base_checkpoint_sha256": plan.base_checkpoint_sha256,
        "base_revision": plan.base_revision,
        "generator_script": str(Path(__file__).resolve()),
        "generator_script_sha256": _dependency_sha(plan, Path(__file__)),
        "production_generator_script": str(PRODUCTION_GENERATOR.resolve()),
        "production_generator_script_sha256": _dependency_sha(plan, PRODUCTION_GENERATOR),
        "dependency_scripts": [
            {"path": str(path), "sha256": snapshot[1]}
            for path, snapshot in plan.dependency_snapshots
        ],
    }
    _write_json(verification_path, verification)
    _validate_plan_unchanged(plan)
    if production.sha256_file(base_checkpoint_path.resolve()) != plan.base_checkpoint_sha256:
        raise ValueError("base checkpoint changed during generation")
    print(json.dumps({"passed": passed, "verification": str(verification_path)}), flush=True)
    return 0 if passed else 1


def _validate_plan_unchanged(plan: DerivedPlan) -> None:
    _validate_dependencies_unchanged(plan)
    if _snapshot(plan.evaluation_manifest_path)[1] != plan.evaluation_manifest_sha256:
        raise ValueError("derived diagnostic manifest changed after snapshot")
    payload = _deriver().validate_derived_manifest(plan.evaluation_manifest_path)
    if _required_mapping(payload, "derivation").get("sha256") != plan.derivation_sha256:
        raise ValueError("derived diagnostic derivation binding changed")


def _snapshot_dependencies() -> tuple[tuple[Path, tuple[bytes, str]], ...]:
    return tuple(
        (path.resolve(strict=True), _snapshot(path))
        for path in (Path(__file__), DERIVER, PRODUCTION_GENERATOR)
    )


def _dependency_sha(plan: DerivedPlan, path: Path) -> str:
    resolved = path.resolve(strict=True)
    for candidate, snapshot in plan.dependency_snapshots:
        if candidate == resolved:
            return snapshot[1]
    raise ValueError(f"unbound generator dependency: {resolved}")


def _validate_dependencies_unchanged(plan: DerivedPlan) -> None:
    for path, snapshot in plan.dependency_snapshots:
        if _snapshot(path) != snapshot:
            raise ValueError(f"generator dependency changed after snapshot: {path}")


def reserve_output(path: Path) -> tuple[Path, Path]:
    output = _prepare_output_parent(path, source="derived diagnostic generation")
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite derived generation: {output}") from exc
    _require_nominal_directory(output, source="derived generation output")
    resolved = output.resolve(strict=True)
    wav_dir = resolved / "wav"
    wav_dir.mkdir(exist_ok=False)
    return resolved, wav_dir


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
        raise ValueError(f"{source} must not be a symlink, junction, or reparse alias: {path}")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse and getattr(metadata, "st_file_attributes", 0) & reparse:
        raise ValueError(f"{source} must not be a symlink, junction, or reparse alias: {path}")


def _require_nominal_directory(path: Path, *, source: str) -> None:
    _require_no_alias(path, source=source)
    if not path.is_dir():
        raise ValueError(f"{source} must be a directory: {path}")


def _snapshot(path: Path) -> tuple[bytes, str]:
    payload = _require_regular_direct_file(path, source="snapshot input").read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _required_mapping(row: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = row.get(field)
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be nonempty")
    return value


def _resolved_path(row: Mapping[str, object], field: str, base: Path) -> Path:
    value = _required_string(row, field)
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "generate"))
    parser.add_argument("--derived-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint-path", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "generate" and args.output_dir is None:
        parser.error("generate mode requires --output-dir")
    if args.mode == "preflight" and args.output_dir is not None:
        parser.error("preflight mode does not accept --output-dir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_derived_plan(args.derived_manifest)
    production = _production()
    base_sha = production.validate_base_checkpoint(args.base_checkpoint_path, plan=plan)
    production.validate_upstream_root(args.upstream_root)
    if args.mode == "preflight":
        _validate_plan_unchanged(plan)
        print(
            json.dumps(
                {
                    "passed": True,
                    "diagnostic_kind": DIAGNOSTIC_KIND,
                    "model_id": plan.model_id,
                    "case_count": len(build_derived_cases(plan)),
                    "derived_manifest_sha256": plan.evaluation_manifest_sha256,
                    "derivation_sha256": plan.derivation_sha256,
                    "base_checkpoint_sha256": base_sha,
                    "derived_embedding_sha256": plan.checkpoints[0].embedding_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    return generate_derived(
        plan=plan,
        base_checkpoint_path=args.base_checkpoint_path,
        upstream_root=args.upstream_root,
        output_path=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
