from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

EXPECTED_MODEL_COUNT = 12
VOICE_BANK_SNAPSHOT_SCHEMA = "voice-bank-snapshot/v1"
EVALUATION_SCHEMA = "speaker-checkpoint-evaluation/v1"
EVALUATION_VERIFICATION_SCHEMA = "speaker-checkpoint-evaluation-verification/v2"
REPORT_SCHEMA = "speaker-model-staging-report/v1"
SHA256_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_staging_report(
    *,
    training_jobs: Path,
    evaluation_dirs: Sequence[Path],
    voice_bank_baseline: Path,
    voice_bank_root: Path,
    staging_root: Path,
) -> dict[str, object]:
    jobs_path = training_jobs.resolve()
    baseline_path = voice_bank_baseline.resolve()
    bank_root = voice_bank_root.resolve()
    proposed_root = staging_root.resolve()
    _validate_staging_root(proposed_root, voice_bank_root=bank_root)
    job_ids = _load_job_ids(jobs_path)
    bank_state = _verify_voice_bank_unchanged(
        baseline_path,
        voice_bank_root=bank_root,
    )
    selections = _load_evaluations(
        evaluation_dirs,
        expected_model_ids=job_ids,
        staging_root=proposed_root,
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS",
        "deployment_performed": False,
        "active_voice_bank_unchanged": True,
        "active_voice_bank_snapshot": str(baseline_path),
        "active_voice_bank_snapshot_sha256": sha256_file(baseline_path),
        "active_voice_bank_current": bank_state,
        "training_jobs": str(jobs_path),
        "training_jobs_sha256": sha256_file(jobs_path),
        "proposed_staging_root": str(proposed_root),
        "proposed_staging_root_created": False,
        "model_count": len(selections),
        "selections": selections,
    }


def _validate_staging_root(staging_root: Path, *, voice_bank_root: Path) -> None:
    if staging_root.exists():
        message = f"proposed staging root already exists: {staging_root}"
        raise FileExistsError(message)
    if _paths_overlap(staging_root, voice_bank_root):
        message = "proposed staging root must be separate from the active voice bank"
        raise ValueError(message)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _load_job_ids(path: Path) -> tuple[str, ...]:
    payload = _read_json(path)
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_MODEL_COUNT:
        message = f"training jobs must contain exactly {EXPECTED_MODEL_COUNT} jobs"
        raise ValueError(message)
    model_ids: list[str] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            message = "training job entries must be objects"
            raise TypeError(message)
        model_id = _required_string(raw_job, "model_id", source="training job")
        if "/" in model_id or "\\" in model_id or model_id in {".", ".."}:
            message = f"unsafe training model_id: {model_id!r}"
            raise ValueError(message)
        model_ids.append(model_id)
    if len(set(model_ids)) != EXPECTED_MODEL_COUNT:
        message = "training jobs contain duplicate model_id"
        raise ValueError(message)
    return tuple(model_ids)


def _verify_voice_bank_unchanged(
    baseline_path: Path,
    *,
    voice_bank_root: Path,
) -> dict[str, object]:
    baseline = _read_json(baseline_path)
    if baseline.get("schema_version") != VOICE_BANK_SNAPSHOT_SCHEMA:
        message = f"voice bank baseline requires schema_version {VOICE_BANK_SNAPSHOT_SCHEMA}"
        raise ValueError(message)
    baseline_root = Path(
        _required_string(baseline, "voice_bank_root", source="voice bank baseline"),
    ).resolve()
    if baseline_root != voice_bank_root:
        message = "voice bank baseline root does not match the active voice bank"
        raise ValueError(message)
    manifest = voice_bank_root / "voice_bank_speakers.toml"
    raw_manifest = baseline.get("manifest")
    if not isinstance(raw_manifest, dict):
        message = "voice bank baseline manifest must be an object"
        raise TypeError(message)
    _verify_baseline_file(
        raw_manifest,
        actual_path=manifest,
        expected_name=None,
        source="voice bank manifest",
    )
    raw_speakers = baseline.get("speakers")
    if not isinstance(raw_speakers, list):
        message = "voice bank baseline speakers must be a list"
        raise TypeError(message)
    if baseline.get("speaker_count") != len(raw_speakers):
        message = "voice bank baseline speaker_count does not match speakers"
        raise ValueError(message)
    actual_speakers = tuple(
        sorted((voice_bank_root / "speakers").glob("*.speaker.safetensors")),
    )
    baseline_names: set[str] = set()
    for raw_speaker in raw_speakers:
        if not isinstance(raw_speaker, dict):
            message = "voice bank baseline speaker entries must be objects"
            raise TypeError(message)
        name = _required_string(raw_speaker, "name", source="voice bank speaker")
        if name in baseline_names:
            message = f"duplicate voice bank baseline speaker: {name}"
            raise ValueError(message)
        baseline_names.add(name)
        _verify_baseline_file(
            raw_speaker,
            actual_path=voice_bank_root / "speakers" / name,
            expected_name=name,
            source=f"voice bank speaker {name}",
        )
    actual_names = {path.name for path in actual_speakers}
    if actual_names != baseline_names:
        message = "active voice bank speaker set has changed from baseline"
        raise ValueError(message)
    return {
        "root": str(voice_bank_root),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "speaker_count": len(actual_speakers),
        "speakers": [
            {
                "name": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in actual_speakers
        ],
    }


def _verify_baseline_file(
    row: Mapping[str, object],
    *,
    actual_path: Path,
    expected_name: str | None,
    source: str,
) -> None:
    declared_path = Path(_required_string(row, "path", source=source)).resolve()
    if declared_path != actual_path.resolve():
        message = f"{source} path does not match active voice bank"
        raise ValueError(message)
    if expected_name is not None and row.get("name") != expected_name:
        message = f"{source} name does not match path"
        raise ValueError(message)
    if not actual_path.is_file():
        message = f"{source} does not exist: {actual_path}"
        raise ValueError(message)
    expected_sha256 = _required_sha256(row, "sha256", source=source)
    if sha256_file(actual_path) != expected_sha256:
        message = f"{source} SHA-256 changed from baseline"
        raise ValueError(message)
    if row.get("size") != actual_path.stat().st_size:
        message = f"{source} size changed from baseline"
        raise ValueError(message)


def _load_evaluations(
    evaluation_dirs: Sequence[Path],
    *,
    expected_model_ids: Sequence[str],
    staging_root: Path,
) -> list[dict[str, object]]:
    if len(evaluation_dirs) != EXPECTED_MODEL_COUNT:
        message = f"exactly {EXPECTED_MODEL_COUNT} --evaluation-dir inputs are required"
        raise ValueError(message)
    indexed: dict[str, dict[str, object]] = {}
    for raw_dir in evaluation_dirs:
        evaluation_dir = raw_dir.resolve()
        selected_path = evaluation_dir / "selected-models.json"
        verification_path = evaluation_dir / "evaluation-verification.json"
        selected_document = _read_json(selected_path)
        verification = _read_json(verification_path)
        selection = _validate_evaluation(
            selected_document,
            verification=verification,
            selected_path=selected_path,
            verification_path=verification_path,
            evaluation_dir=evaluation_dir,
            staging_root=staging_root,
        )
        model_id = str(selection["model_id"])
        if model_id in indexed:
            message = f"duplicate evaluation model_id: {model_id}"
            raise ValueError(message)
        indexed[model_id] = selection
    if indexed.keys() != set(expected_model_ids):
        message = "evaluation model_ids do not exactly match training jobs"
        raise ValueError(message)
    return [indexed[model_id] for model_id in expected_model_ids]


def _validate_evaluation(
    selected_document: Mapping[str, object],
    *,
    verification: Mapping[str, object],
    selected_path: Path,
    verification_path: Path,
    evaluation_dir: Path,
    staging_root: Path,
) -> dict[str, object]:
    if selected_document.get("schema_version") != EVALUATION_SCHEMA:
        message = f"selected models requires schema_version {EVALUATION_SCHEMA}"
        raise ValueError(message)
    raw_selections = selected_document.get("selections")
    if not isinstance(raw_selections, list) or len(raw_selections) != 1:
        message = "selected models must contain exactly one selection"
        raise ValueError(message)
    selected = raw_selections[0]
    if not isinstance(selected, dict):
        message = "selected model entry must be an object"
        raise TypeError(message)
    if verification.get("schema_version") != EVALUATION_VERIFICATION_SCHEMA:
        message = f"evaluation verification requires {EVALUATION_VERIFICATION_SCHEMA}"
        raise ValueError(message)
    if verification.get("status") != "PASS":
        message = f"evaluation did not pass: {verification_path}"
        raise ValueError(message)
    if verification.get("selected") != selected:
        message = f"evaluation selected identity mismatch: {verification_path}"
        raise ValueError(message)
    artifact_sha256 = verification.get("artifact_sha256")
    if not isinstance(artifact_sha256, dict) or artifact_sha256.get(
        str(selected_path)
    ) != sha256_file(selected_path):
        message = f"evaluation verification does not bind selected-models.json: {verification_path}"
        raise ValueError(message)
    model_id = _required_string(selected, "model_id", source="selected model")
    checkpoint_step = selected.get("checkpoint_step")
    if not isinstance(checkpoint_step, int) or isinstance(checkpoint_step, bool):
        message = f"selected model {model_id} requires integer checkpoint_step"
        raise TypeError(message)
    embedding_path = Path(
        _required_string(selected, "embedding_path", source=f"selected model {model_id}"),
    ).resolve()
    if not embedding_path.is_file():
        message = f"selected embedding does not exist: {embedding_path}"
        raise ValueError(message)
    embedding_sha256 = _required_sha256(
        selected,
        "embedding_sha256",
        source=f"selected model {model_id}",
    )
    if sha256_file(embedding_path) != embedding_sha256:
        message = f"selected embedding SHA-256 mismatch: {embedding_path}"
        raise ValueError(message)
    for field in ("training_config_sha256", "base_checkpoint_sha256", "run_id"):
        _required_sha256(selected, field, source=f"selected model {model_id}")
    _required_string(selected, "base_checkpoint", source=f"selected model {model_id}")
    _required_string(selected, "base_revision", source=f"selected model {model_id}")
    return dict(selected) | {
        "embedding_path": str(embedding_path),
        "embedding_verified": True,
        "evaluation_dir": str(evaluation_dir),
        "selected_models_path": str(selected_path),
        "selected_models_sha256": sha256_file(selected_path),
        "evaluation_verification_path": str(verification_path),
        "evaluation_verification_sha256": sha256_file(verification_path),
        "evaluation_verified": True,
        "proposed_staging_path": str(staging_root / f"{model_id}.speaker.safetensors"),
        "staged": False,
    }


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"invalid JSON: {path}"
        raise ValueError(message) from exc
    if not isinstance(payload, dict):
        message = f"JSON document must be an object: {path}"
        raise TypeError(message)
    return payload


def _required_string(row: Mapping[str, Any], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        message = f"{source} requires nonempty string {field}"
        raise ValueError(message)
    return value


def _required_sha256(row: Mapping[str, Any], field: str, *, source: str) -> str:
    value = _required_string(row, field, source=source)
    if len(value) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        message = f"{source} requires lowercase SHA-256 {field}"
        raise ValueError(message)
    return value


def write_report(path: Path, payload: Mapping[str, object]) -> None:
    output = path.resolve()
    if output.exists():
        message = f"refusing to overwrite existing report: {output}"
        raise FileExistsError(message)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        message = f"refusing to overwrite staging report: {temporary}"
        raise FileExistsError(message)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-jobs", type=Path, required=True)
    parser.add_argument("--evaluation-dir", action="append", type=Path, required=True)
    parser.add_argument("--voice-bank-baseline", type=Path, required=True)
    parser.add_argument("--voice-bank-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve()
    voice_bank_root = args.voice_bank_root.resolve()
    if _paths_overlap(output, voice_bank_root):
        message = "staging report output must be separate from the active voice bank"
        raise ValueError(message)
    if output.exists():
        message = f"refusing to overwrite existing report: {output}"
        raise FileExistsError(message)
    report = build_staging_report(
        training_jobs=args.training_jobs,
        evaluation_dirs=args.evaluation_dir,
        voice_bank_baseline=args.voice_bank_baseline,
        voice_bank_root=args.voice_bank_root,
        staging_root=args.staging_root,
    )
    write_report(output, report)
    print(f"non-destructive staging report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
