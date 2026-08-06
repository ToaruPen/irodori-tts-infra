# ruff: noqa: ANN401, EM101, EM102, PLR0914, SLF001, TRY003
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

SEARCH_SCHEMA = "speaker-checkpoint-search-manifest/v1"
SEARCH_STEP = 250
EXPECTED_CASE_COUNT = 28
PRODUCTION_GENERATOR = Path(__file__).with_name("generate_600m_checkpoint_audio_remote.py")
SEARCH_BUILDER = Path(__file__).with_name("build_600m_speaker_checkpoint_search_manifest.py")
SEARCH_GENERATION_SCHEMA = "speaker-checkpoint-search-generation/v1"
SEARCH_VERIFICATION_SCHEMA = "speaker-checkpoint-search-generation-verification/v1"
SEARCH_CASE_SCHEMA = "speaker-checkpoint-search-generation-case/v1"


def _production() -> ModuleType:
    name = "_speaker_checkpoint_production_generator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, PRODUCTION_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production generator: {PRODUCTION_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _builder() -> ModuleType:
    name = "_speaker_checkpoint_search_builder"
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


def load_search_plan(path: Path) -> Any:
    manifest_path = path.resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema_version") != SEARCH_SCHEMA:
        raise TypeError(f"checkpoint search manifest requires schema_version {SEARCH_SCHEMA}")
    production = _production()
    for field, expected in (
        ("text_ids", tuple(production.EXPECTED_TEXT_IDS)),
        ("seeds", tuple(production.EXPECTED_SEEDS)),
        ("styles", tuple(production.EXPECTED_STYLES)),
    ):
        if tuple(payload.get(field, ())) != expected:
            raise ValueError(f"checkpoint search manifest {field} must exactly match {expected}")
    model_id = _required_string(payload, "model_id")
    run_id = _required_string(payload, "run_id")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_step") != SEARCH_STEP:
        raise ValueError(f"checkpoint search manifest requires checkpoint step {SEARCH_STEP}")
    if checkpoint.get("run_id") != run_id:
        raise ValueError("checkpoint search run_id does not match checkpoint provenance")
    training_config_path = _resolved_path(checkpoint, "training_config_path", manifest_path.parent)
    training_config_sha = production._required_sha256(checkpoint, "training_config_sha256")
    _validate_file_hash(training_config_path, training_config_sha, source="training config")
    training_config = _read_json(training_config_path)
    _builder()._validate_search_training_config(training_config)
    candidate = production._parse_candidate(checkpoint, manifest_dir=manifest_path.parent)
    _builder()._validate_embedding(candidate.embedding_path)
    source = payload.get("source_evaluation_manifest")
    if not isinstance(source, dict):
        raise TypeError("source_evaluation_manifest must be an object")
    source_path = _resolved_path(source, "path", manifest_path.parent)
    source_sha = production._required_sha256(source, "sha256")
    _validate_file_hash(source_path, source_sha, source="source evaluation manifest")
    source_payload = _read_json(source_path)
    _builder()._validate_source_manifest(
        source_payload,
        model_id=model_id,
        manifest_dir=source_path.parent,
    )
    _builder()._validate_search_source_binding(payload, source_payload)
    evidence = payload.get("training_run_evidence")
    if not isinstance(evidence, dict):
        raise TypeError("training_run_evidence must be an object")
    evidence_path = _resolved_path(evidence, "path", manifest_path.parent)
    evidence_sha = production._required_sha256(evidence, "sha256")
    _validate_file_hash(evidence_path, evidence_sha, source="training run evidence")
    _builder()._validate_run_evidence(
        evidence_path,
        model_id=model_id,
        run_id=run_id,
        config_path=training_config_path,
        config_sha256=training_config_sha,
        embedding_path=candidate.embedding_path,
        embedding_sha256=candidate.embedding_sha256,
        base_checkpoint_sha256=candidate.base_checkpoint_sha256,
    )
    if candidate.run_id != run_id:
        raise ValueError("checkpoint candidate run_id does not match search manifest")
    return production.GenerationPlan(
        model_id=model_id,
        checkpoints=(candidate,),
        evaluation_manifest_path=manifest_path,
        evaluation_manifest_sha256=production.sha256_file(manifest_path),
        base_checkpoint=candidate.base_checkpoint,
        base_checkpoint_sha256=candidate.base_checkpoint_sha256,
        base_revision=candidate.base_revision,
    )


def build_search_cases(plan: Any) -> tuple[Any, ...]:
    cases = tuple(_production().build_cases(plan))
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"search generation requires exactly {EXPECTED_CASE_COUNT} cases")
    return cases


def bind_search_case_schema(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [{**row, "schema_version": SEARCH_CASE_SCHEMA} for row in rows]


def build_generation_config(*, plan: Any, checkpoint_path: Path) -> dict[str, object]:
    production = _production()
    return {
        "schema_version": SEARCH_GENERATION_SCHEMA,
        "model_id": plan.model_id,
        "case_count": EXPECTED_CASE_COUNT,
        "search_manifest_path": str(plan.evaluation_manifest_path),
        "search_manifest_sha256": plan.evaluation_manifest_sha256,
        "search_generator_script": str(Path(__file__).resolve()),
        "search_generator_script_sha256": production.sha256_file(Path(__file__).resolve()),
        "production_generator_script": str(PRODUCTION_GENERATOR.resolve()),
        "production_generator_script_sha256": production.sha256_file(PRODUCTION_GENERATOR),
        "base_checkpoint_path": str(checkpoint_path.resolve()),
        "base_checkpoint_model_id": plan.base_checkpoint,
        "base_checkpoint_sha256": plan.base_checkpoint_sha256,
        "base_revision": plan.base_revision,
        "text_ids": list(production.EXPECTED_TEXT_IDS),
        "seeds": list(production.EXPECTED_SEEDS),
        "styles": list(production.EXPECTED_STYLES),
    }


def build_generation_verification(
    *,
    plan: Any,
    checkpoint_path: Path,
    config_path: Path,
    results_path: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    production = _production()
    counts = Counter(str(row["status"]) for row in rows)
    unique_count = len({str(row["case_id"]) for row in rows})
    passed = (
        len(rows) == EXPECTED_CASE_COUNT
        and counts["SUCCESS"] == EXPECTED_CASE_COUNT
        and counts["ERROR"] == 0
        and unique_count == EXPECTED_CASE_COUNT
        and all(row.get("schema_version") == SEARCH_CASE_SCHEMA for row in rows)
    )
    return {
        "schema_version": SEARCH_VERIFICATION_SCHEMA,
        "passed": passed,
        "model_id": plan.model_id,
        "case_count": EXPECTED_CASE_COUNT,
        "row_count": len(rows),
        "status_counts": dict(counts),
        "case_ids_unique": unique_count == len(rows),
        "all_audio_finite": all(row.get("audio_finite") is True for row in rows),
        "search_manifest_path": str(plan.evaluation_manifest_path),
        "search_manifest_sha256": plan.evaluation_manifest_sha256,
        "generation_config_path": str(config_path),
        "generation_config_sha256": production.sha256_file(config_path),
        "generation_results_path": str(results_path),
        "generation_results_sha256": production.sha256_file(results_path),
        "base_checkpoint_path": str(checkpoint_path.resolve()),
        "base_checkpoint_model_id": plan.base_checkpoint,
        "base_checkpoint_sha256": plan.base_checkpoint_sha256,
        "base_revision": plan.base_revision,
        "search_generator_script": str(Path(__file__).resolve()),
        "search_generator_script_sha256": production.sha256_file(Path(__file__).resolve()),
        "production_generator_script": str(PRODUCTION_GENERATOR.resolve()),
        "production_generator_script_sha256": production.sha256_file(PRODUCTION_GENERATOR),
    }


def generate_search(
    *,
    plan: Any,
    checkpoint_path: Path,
    upstream_root: Path,
    output_path: Path,
) -> int:
    production = _production()
    production.validate_base_checkpoint(checkpoint_path, plan=plan)
    cases = build_search_cases(plan)
    output_dir, wav_dir = reserve_output(output_path)
    config_path = output_dir / "generation-config.json"
    config = build_generation_config(plan=plan, checkpoint_path=checkpoint_path)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_path = output_dir / "generation-results.jsonl"
    runtime_api = production._load_runtime_api(upstream_root)
    runtime = production._create_runtime(runtime_api, checkpoint_path=checkpoint_path)
    try:
        rows = _write_search_generation_results(
            cases,
            plan=plan,
            wav_dir=wav_dir,
            results_path=results_path,
            runtime=runtime,
            runtime_api=runtime_api,
        )
    finally:
        production._unload_runtime(runtime)
    verification_path = output_dir / "generation-verification.json"
    verification = build_generation_verification(
        plan=plan,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        results_path=results_path,
        rows=rows,
    )
    verification_path.write_text(
        json.dumps(
            verification,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    passed = verification["passed"] is True
    print(json.dumps({"passed": passed, "verification": str(verification_path)}), flush=True)
    return 0 if passed else 1


def _write_search_generation_results(
    cases: Sequence[Any],
    *,
    plan: Any,
    wav_dir: Path,
    results_path: Path,
    runtime: Any,
    runtime_api: Mapping[str, Any],
) -> list[dict[str, object]]:
    production = _production()
    rows: list[dict[str, object]] = []
    with results_path.open("x", encoding="utf-8", newline="\n") as results_file:
        for index, case in enumerate(cases, start=1):
            generated = production._generate_case_result(
                case,
                plan=plan,
                wav_dir=wav_dir,
                runtime=runtime,
                runtime_api=runtime_api,
            )
            row = bind_search_case_schema([generated])[0]
            rows.append(row)
            results_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            results_file.flush()
            print(f"[{index}/{len(cases)}] {case.case_id}: {row['status']}", flush=True)
    return rows


def reserve_output(path: Path) -> tuple[Path, Path]:
    output = _prepare_output_parent(path, source="search generation")
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite search generation: {output}") from exc
    _require_nominal_directory(output, source="search generation output")
    resolved = output.resolve(strict=True)
    wav_dir = resolved / "wav"
    wav_dir.mkdir(exist_ok=False)
    return resolved, wav_dir


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


def _resolved_path(row: Mapping[str, object], field: str, base: Path) -> Path:
    value = _required_string(row, field)
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _validate_file_hash(path: Path, expected: str, *, source: str) -> None:
    if not path.is_file() or _production().sha256_file(path) != expected:
        raise ValueError(f"{source} SHA-256 mismatch: {path}")


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"checkpoint search manifest requires nonempty string {field}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return payload


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "generate"))
    parser.add_argument("--search-manifest", type=Path, required=True)
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
    plan = load_search_plan(args.search_manifest)
    production = _production()
    base_sha = production.validate_base_checkpoint(args.base_checkpoint_path, plan=plan)
    production.validate_upstream_root(args.upstream_root)
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "passed": True,
                    "model_id": plan.model_id,
                    "case_count": len(build_search_cases(plan)),
                    "search_manifest_sha256": plan.evaluation_manifest_sha256,
                    "base_checkpoint_sha256": base_sha,
                    "embedding_sha256": plan.checkpoints[0].embedding_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    return generate_search(
        plan=plan,
        checkpoint_path=args.base_checkpoint_path,
        upstream_root=args.upstream_root,
        output_path=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
