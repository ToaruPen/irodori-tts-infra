from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess  # noqa: S404 - fixed read-only Git provenance probes.
import sys
import time
import wave
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    class UpstreamGuardLike(Protocol):
        def verify(self, point: str) -> None: ...
        def binding(self) -> dict[str, object]: ...


MANIFEST_SCHEMA_VERSION = "speaker-checkpoint-evaluation-manifest/v1"
EXPECTED_CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
EXPECTED_TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
EXPECTED_SEEDS = (1234, 5678)
EXPECTED_STYLES = ("neutral", "calm")
EXPECTED_CASE_COUNT = (
    len(EXPECTED_CHECKPOINT_STEPS)
    * len(EXPECTED_TEXT_IDS)
    * len(EXPECTED_SEEDS)
    * len(EXPECTED_STYLES)
)
SHA256_LENGTH = 64
UPSTREAM_PROVENANCE_SCHEMA = "irodori-upstream-runtime-provenance/v1"
UPSTREAM_RUNTIME_PROVENANCE_NAME = "upstream-runtime-provenance.json"
UPSTREAM_RUNTIME_PACKAGE_NAME = "upstream-runtime-package.zip"
PINNED_UPSTREAM_COMMIT = "eaf74d6a19138f743acb5b71a445fd25a57db987"

TEXTS = {
    "word_unko": ("うんこ。", False),
    "word_chinko": ("ちんこ。", False),
    "word_manko": ("まんこ。", False),
    "sentence_unko": ("「うんこ」という言葉を読み上げます。", False),
    "sentence_chinko": ("「ちんこ」という言葉を読み上げます。", False),
    "sentence_manko": ("「まんこ」という言葉を読み上げます。", False),
    "control": ("こんにちは。今日はいい天気ですね。", True),
}
STYLE_CAPTIONS = {
    "neutral": None,
    "calm": "穏やかで優しい女性の声で、自然に話す。",
}


class RuntimeResultLike(Protocol):
    audio: object
    sample_rate: int
    used_seed: int


class RuntimeLike(Protocol):
    def synthesize(
        self,
        request: object,
        *,
        log_fn: object | None = None,
    ) -> RuntimeResultLike: ...


@dataclass(frozen=True, slots=True)
class CheckpointCandidate:
    checkpoint_step: int
    embedding_path: Path
    embedding_sha256: str
    training_config_sha256: str
    base_checkpoint: str
    base_checkpoint_sha256: str
    base_revision: str
    run_id: str


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    model_id: str
    checkpoints: tuple[CheckpointCandidate, ...]
    evaluation_manifest_path: Path
    evaluation_manifest_sha256: str
    base_checkpoint: str
    base_checkpoint_sha256: str
    base_revision: str


@dataclass(frozen=True, slots=True)
class GenerationCase:
    model_id: str
    checkpoint: CheckpointCandidate
    text_id: str
    text: str
    control: bool
    seed: int
    style: str
    caption: str | None

    @property
    def case_id(self) -> str:
        return (
            f"{self.model_id}__checkpoint-{self.checkpoint.checkpoint_step}__"
            f"{self.text_id}__seed-{self.seed}__{self.style}"
        )


@dataclass(slots=True)
class UpstreamRuntimeProvenance:
    path: Path
    sha256: str
    upstream_root: Path
    commit: str
    tree: str
    python_files: tuple[tuple[str, str], ...]
    package_archive: Path
    package_archive_sha256: str
    validation_points: list[str]

    def verify(self, point: str) -> None:
        _verify_upstream_runtime(self)
        self.validation_points.append(point)

    def binding(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "upstream_root": str(self.upstream_root),
            "commit": self.commit,
            "tree": self.tree,
            "package_archive": str(self.package_archive),
            "package_archive_sha256": self.package_archive_sha256,
            "python_file_count": len(self.python_files),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> bytes:
    process = subprocess.run(  # noqa: S603 - fixed Git executable and read-only arguments.
        ("git", "-C", str(root), *arguments),  # noqa: S607
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        message = f"upstream Git provenance command failed ({' '.join(arguments)}): {stderr}"
        raise ValueError(message)
    return process.stdout


def load_upstream_runtime_provenance(
    path: Path,
    *,
    expected_sha256: str,
    upstream_root: Path,
    package_archive: Path,
    expected_package_archive_sha256: str,
) -> UpstreamRuntimeProvenance:
    _require_alias_free_runtime_asset(path)
    _require_alias_free_runtime_asset(package_archive)
    provenance_path = path.resolve()
    if provenance_path.is_symlink() or not provenance_path.is_file():
        message = f"upstream runtime provenance is missing or symlinked: {provenance_path}"
        raise ValueError(message)
    actual_sha256 = sha256_file(provenance_path)
    if actual_sha256 != expected_sha256:
        message = (
            "upstream runtime provenance SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
        raise ValueError(message)
    document = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = "upstream runtime provenance must be an object"
        raise TypeError(message)
    expected_fields = {
        "schema_version",
        "upstream_root",
        "commit",
        "tree",
        "package",
        "python_files",
    }
    if (
        set(document) != expected_fields
        or document.get("schema_version") != UPSTREAM_PROVENANCE_SCHEMA
    ):
        message = "upstream runtime provenance contract mismatch"
        raise ValueError(message)
    root = upstream_root.resolve()
    if document.get("upstream_root") != str(root) or document.get("package") != "irodori_tts":
        message = "upstream runtime provenance root or package mismatch"
        raise ValueError(message)
    commit = _required_provenance_hex(document, "commit", length=40)
    if commit != PINNED_UPSTREAM_COMMIT:
        message = f"upstream commit mismatch: expected={PINNED_UPSTREAM_COMMIT}, actual={commit}"
        raise ValueError(message)
    tree = _required_provenance_hex(document, "tree", length=40)
    raw_files = document.get("python_files")
    if not isinstance(raw_files, list) or not raw_files:
        message = "upstream runtime provenance requires Python file bindings"
        raise ValueError(message)
    python_files: list[tuple[str, str]] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            message = "upstream runtime Python file binding contract mismatch"
            raise ValueError(message)
        relative = raw.get("path")
        digest = raw.get("sha256")
        if not _valid_python_relative(relative) or not _valid_lower_hex(
            digest,
            length=SHA256_LENGTH,
        ):
            message = "upstream runtime Python file binding is invalid"
            raise ValueError(message)
        python_files.append((cast("str", relative), cast("str", digest)))
    if python_files != sorted(python_files) or len({path for path, _sha in python_files}) != len(
        python_files
    ):
        message = "upstream runtime Python file inventory must be sorted and unique"
        raise ValueError(message)
    archive = package_archive.resolve()
    guard = UpstreamRuntimeProvenance(
        path=provenance_path,
        sha256=actual_sha256,
        upstream_root=root,
        commit=commit,
        tree=tree,
        python_files=tuple(python_files),
        package_archive=archive,
        package_archive_sha256=expected_package_archive_sha256,
        validation_points=[],
    )
    _verify_upstream_runtime(guard)
    return guard


def _require_alias_free_runtime_asset(path: Path) -> None:
    nominal = Path(os.path.abspath(path))  # noqa: PTH100 - preserve lexical aliases.
    for component in (nominal, *nominal.parents):
        if component == component.parent:
            continue
        try:
            metadata = component.lstat()
        except OSError:
            continue
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        if component.is_symlink() or bool(file_attributes & 0x400):
            message = f"upstream runtime asset contains a filesystem alias: {component}"
            raise ValueError(message)


def _required_provenance_hex(
    document: Mapping[str, object],
    field: str,
    *,
    length: int,
) -> str:
    value = document.get(field)
    if not _valid_lower_hex(value, length=length):
        message = f"upstream runtime provenance {field} is invalid"
        raise ValueError(message)
    return cast("str", value)


def _valid_lower_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_python_relative(value: object) -> bool:
    return (
        isinstance(value, str)
        and not Path(value).is_absolute()
        and Path(value).as_posix() == value
        and value.startswith("irodori_tts/")
        and value.endswith(".py")
    )


def _verify_upstream_runtime(guard: UpstreamRuntimeProvenance) -> None:
    if sha256_file(guard.path) != guard.sha256:
        message = "upstream runtime provenance SHA-256 changed"
        raise ValueError(message)
    root = guard.upstream_root
    top_level = Path(
        _git_output(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    head = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git_output(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    package_status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "irodori_tts",
    ).decode("utf-8", errors="replace")
    if top_level != root or head != guard.commit or tree != guard.tree:
        message = "upstream Git identity changed or does not match provenance"
        raise ValueError(message)
    if package_status:
        message = "upstream irodori_tts package is dirty or contains untracked files"
        raise ValueError(message)
    tracked = _git_output(root, "ls-files", "-z", "--", "irodori_tts").decode("utf-8")
    tracked_python = sorted(path for path in tracked.split("\0") if path.endswith(".py"))
    if tracked_python != [path for path, _digest in guard.python_files]:
        message = "upstream tracked Python file inventory mismatch"
        raise ValueError(message)
    for relative, expected in guard.python_files:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or sha256_file(candidate) != expected:
            message = f"upstream Python file hash mismatch: {candidate}"
            raise ValueError(message)
    _verify_package_archive(guard)


def _verify_package_archive(guard: UpstreamRuntimeProvenance) -> None:
    archive = guard.package_archive
    if archive.is_symlink() or not archive.is_file():
        message = f"upstream package archive is missing or symlinked: {archive}"
        raise ValueError(message)
    actual_archive_sha256 = sha256_file(archive)
    if actual_archive_sha256 != guard.package_archive_sha256:
        message = (
            "upstream package archive SHA-256 mismatch: "
            f"expected={guard.package_archive_sha256}, actual={actual_archive_sha256}"
        )
        raise ValueError(message)
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        expected_names = [path for path, _digest in guard.python_files]
        if names != expected_names or len(set(names)) != len(names):
            message = "upstream package archive entry inventory mismatch"
            raise ValueError(message)
        expected_sha = dict(guard.python_files)
        for info in package.infolist():
            archive_sha = hashlib.sha256(package.read(info)).hexdigest()
            if info.is_dir() or archive_sha != expected_sha[info.filename]:
                message = f"upstream package archive entry hash mismatch: {info.filename}"
                raise ValueError(message)


def load_generation_plan(path: Path) -> GenerationPlan:
    manifest_path = path.resolve()
    payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        message = f"checkpoint manifest requires schema_version {MANIFEST_SCHEMA_VERSION}"
        raise TypeError(message)
    _validate_dimensions(payload)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != 1:
        message = "remote generation requires exactly one model per checkpoint manifest"
        raise ValueError(message)
    raw_model = raw_models[0]
    if not isinstance(raw_model, dict):
        message = "checkpoint manifest model entry must be an object"
        raise TypeError(message)
    model_id = _required_string(raw_model, "model_id")
    raw_checkpoints = raw_model.get("checkpoints")
    if not isinstance(raw_checkpoints, list):
        message = f"checkpoint manifest checkpoints must be a list for {model_id}"
        raise TypeError(message)
    steps = tuple(
        row.get("checkpoint_step") if isinstance(row, dict) else None for row in raw_checkpoints
    )
    if steps != EXPECTED_CHECKPOINT_STEPS:
        message = f"checkpoint steps must exactly match {EXPECTED_CHECKPOINT_STEPS}"
        raise ValueError(message)
    checkpoints = tuple(
        _parse_candidate(row, manifest_dir=manifest_path.parent) for row in raw_checkpoints
    )
    base_contracts = {
        (
            candidate.base_checkpoint,
            candidate.base_checkpoint_sha256,
            candidate.base_revision,
        )
        for candidate in checkpoints
    }
    if len(base_contracts) != 1:
        message = "checkpoint candidates do not share one base checkpoint contract"
        raise ValueError(message)
    base_checkpoint, base_sha256, base_revision = next(iter(base_contracts))
    return GenerationPlan(
        model_id=model_id,
        checkpoints=checkpoints,
        evaluation_manifest_path=manifest_path,
        evaluation_manifest_sha256=sha256_file(manifest_path),
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_sha256,
        base_revision=base_revision,
    )


def _validate_dimensions(payload: Mapping[str, object]) -> None:
    expected = {
        "text_ids": EXPECTED_TEXT_IDS,
        "seeds": EXPECTED_SEEDS,
        "styles": EXPECTED_STYLES,
    }
    for field, values in expected.items():
        actual = payload.get(field)
        if not isinstance(actual, list) or tuple(actual) != values:
            message = f"checkpoint manifest {field} must exactly match {values}"
            raise ValueError(message)


def _parse_candidate(raw: object, *, manifest_dir: Path) -> CheckpointCandidate:
    if not isinstance(raw, dict):
        message = "checkpoint candidate must be an object"
        raise TypeError(message)
    step = raw.get("checkpoint_step")
    if not isinstance(step, int) or isinstance(step, bool):
        message = "checkpoint_step must be an integer"
        raise TypeError(message)
    raw_path = _required_string(raw, "embedding_path")
    embedding_path = Path(raw_path)
    if not embedding_path.is_absolute():
        embedding_path = manifest_dir / embedding_path
    embedding_path = embedding_path.resolve()
    if not embedding_path.is_file():
        message = f"checkpoint embedding does not exist: {embedding_path}"
        raise ValueError(message)
    embedding_sha256 = _required_sha256(raw, "embedding_sha256")
    if sha256_file(embedding_path) != embedding_sha256:
        message = f"embedding SHA-256 mismatch: {embedding_path}"
        raise ValueError(message)
    return CheckpointCandidate(
        checkpoint_step=step,
        embedding_path=embedding_path,
        embedding_sha256=embedding_sha256,
        training_config_sha256=_required_sha256(raw, "training_config_sha256"),
        base_checkpoint=_required_string(raw, "base_checkpoint"),
        base_checkpoint_sha256=_required_sha256(raw, "base_checkpoint_sha256"),
        base_revision=_required_string(raw, "base_revision"),
        run_id=_required_string(raw, "run_id"),
    )


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        message = f"checkpoint manifest requires nonempty string {field}"
        raise ValueError(message)
    return value


def _required_sha256(row: Mapping[str, object], field: str) -> str:
    value = _required_string(row, field)
    if len(value) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        message = f"checkpoint manifest {field} must be a lowercase SHA-256 digest"
        raise ValueError(message)
    return value


def build_cases(plan: GenerationPlan) -> tuple[GenerationCase, ...]:
    return tuple(
        GenerationCase(
            model_id=plan.model_id,
            checkpoint=checkpoint,
            text_id=text_id,
            text=TEXTS[text_id][0],
            control=TEXTS[text_id][1],
            seed=seed,
            style=style,
            caption=STYLE_CAPTIONS[style],
        )
        for checkpoint in plan.checkpoints
        for text_id in EXPECTED_TEXT_IDS
        for seed in EXPECTED_SEEDS
        for style in EXPECTED_STYLES
    )


def validate_base_checkpoint(path: Path, *, plan: GenerationPlan) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        message = f"base checkpoint does not exist: {resolved}"
        raise ValueError(message)
    actual = sha256_file(resolved)
    if actual != plan.base_checkpoint_sha256:
        message = (
            "base checkpoint SHA-256 mismatch: "
            f"expected {plan.base_checkpoint_sha256}, got {actual}"
        )
        raise ValueError(message)
    return actual


def reserve_output(path: Path) -> tuple[Path, Path]:
    output_dir = path.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        message = f"refusing to overwrite existing output directory: {output_dir}"
        raise FileExistsError(message) from exc
    wav_dir = output_dir / "wav"
    wav_dir.mkdir()
    return output_dir, wav_dir


def build_success_row(
    case: GenerationCase,
    *,
    plan: GenerationPlan,
    wav_path: Path,
    elapsed_seconds: float,
    audio_metadata: Mapping[str, object],
) -> dict[str, object]:
    return (
        _base_result_row(
            case,
            plan=plan,
            status="SUCCESS",
            elapsed_seconds=elapsed_seconds,
        )
        | dict(audio_metadata)
        | {
            "wav_path": str(wav_path.resolve()),
            "wav_sha256": sha256_file(wav_path.resolve()),
            "wav_size": wav_path.stat().st_size,
            "exception_type": None,
            "exception_message": None,
        }
    )


def _build_error_row(
    case: GenerationCase,
    *,
    plan: GenerationPlan,
    elapsed_seconds: float,
    error: Exception,
) -> dict[str, object]:
    return _base_result_row(
        case,
        plan=plan,
        status="ERROR",
        elapsed_seconds=elapsed_seconds,
    ) | {
        "audio_finite": False,
        "wav_path": None,
        "wav_sha256": None,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
    }


def _base_result_row(
    case: GenerationCase,
    *,
    plan: GenerationPlan,
    status: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    candidate = case.checkpoint
    return {
        "case_id": case.case_id,
        "model_id": case.model_id,
        "checkpoint_step": candidate.checkpoint_step,
        "checkpoint": candidate.base_checkpoint,
        "speaker_filename": candidate.embedding_path.name,
        "embedding_path": str(candidate.embedding_path),
        "embedding_sha256": candidate.embedding_sha256,
        "evaluation_manifest_sha256": plan.evaluation_manifest_sha256,
        "base_checkpoint_sha256": candidate.base_checkpoint_sha256,
        "text_id": case.text_id,
        "text": case.text,
        "control": case.control,
        "seed": case.seed,
        "style": case.style,
        "caption": case.caption,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "num_steps": 40,
        "cfg_scale_text": 3.0,
        "cfg_scale_caption": 3.0,
        "cfg_scale_speaker": 5.0,
        "cfg_guidance_mode": "independent",
        "duration_scale": 1.0,
        "num_candidates": 1,
        "t_schedule_mode": "linear",
        "sway_coeff": -1.0,
        "decode_mode": "sequential",
        "context_kv_cache": True,
        "provenance": {
            "training_config_sha256": candidate.training_config_sha256,
            "base_checkpoint": candidate.base_checkpoint,
            "base_revision": candidate.base_revision,
            "run_id": candidate.run_id,
        },
    }


def _audio_metadata(audio: object, wav_path: Path) -> dict[str, object]:
    detach = getattr(audio, "detach", None)
    if not callable(detach):
        message = "generated audio does not provide detach()"
        raise TypeError(message)
    tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if not callable(cpu):
        message = "generated audio does not provide cpu()"
        raise TypeError(message)
    cpu_tensor = cpu()
    to_numpy = getattr(cpu_tensor, "numpy", None)
    if not callable(to_numpy):
        message = "generated audio does not provide numpy()"
        raise TypeError(message)
    samples = np.asarray(to_numpy())
    if samples.size == 0 or not np.isfinite(samples).all():
        message = "generated audio contains non-finite samples"
        raise ValueError(message)
    with wave.open(str(wav_path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        num_frames = reader.getnframes()
    return {
        "audio_dtype": str(samples.dtype),
        "audio_shape": list(samples.shape),
        "audio_finite": True,
        "audio_min": float(samples.min()),
        "audio_max": float(samples.max()),
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "channels": channels,
        "num_frames": num_frames,
        "duration_seconds": num_frames / sample_rate,
    }


def _generation_config(
    *,
    plan: GenerationPlan,
    checkpoint_path: Path,
    cases: Sequence[GenerationCase],
    upstream_guard: UpstreamRuntimeProvenance | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": "speaker-checkpoint-audio-generation/v1",
        "model_id": plan.model_id,
        "case_count": len(cases),
        "checkpoint_manifest": str(plan.evaluation_manifest_path),
        "checkpoint_manifest_sha256": plan.evaluation_manifest_sha256,
        "generator_script": str(Path(__file__).resolve()),
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
        "base_checkpoint": str(checkpoint_path.absolute()),
        "base_checkpoint_sha256": plan.base_checkpoint_sha256,
        "text_ids": list(EXPECTED_TEXT_IDS),
        "seeds": list(EXPECTED_SEEDS),
        "styles": list(EXPECTED_STYLES),
    }
    if upstream_guard is not None:
        config["upstream_runtime"] = upstream_guard.binding()
    return config


def generate(
    *,
    plan: GenerationPlan,
    checkpoint_path: Path,
    upstream_root: Path,
    upstream_guard: UpstreamRuntimeProvenance,
    output_path: Path,
) -> int:
    validate_base_checkpoint(checkpoint_path, plan=plan)
    cases = build_cases(plan)
    output_dir, wav_dir = reserve_output(output_path)
    config_path = output_dir / "generation-config.json"
    config = _generation_config(
        plan=plan,
        checkpoint_path=checkpoint_path,
        cases=cases,
        upstream_guard=upstream_guard,
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_path = output_dir / "generation-results.jsonl"
    upstream_guard.verify("before_import")
    runtime_api = _load_runtime_api(upstream_root, package_archive=upstream_guard.package_archive)
    _verify_imported_upstream_modules(upstream_guard.package_archive)
    upstream_guard.verify("after_import")
    upstream_guard.verify("base_model_before_load")
    runtime = _create_runtime(
        runtime_api,
        checkpoint_path=checkpoint_path,
    )
    upstream_guard.verify("base_model_after_load")
    try:
        rows = _write_generation_results(
            cases,
            plan=plan,
            wav_dir=wav_dir,
            results_path=results_path,
            runtime=runtime,
            runtime_api=runtime_api,
            upstream_guard=upstream_guard,
        )
    finally:
        _unload_runtime(runtime)
        upstream_guard.verify("after_generation")
        _verify_imported_upstream_modules(upstream_guard.package_archive)

    counts = Counter(str(row["status"]) for row in rows)
    passed = (
        len(rows) == EXPECTED_CASE_COUNT
        and counts["SUCCESS"] == EXPECTED_CASE_COUNT
        and counts["ERROR"] == 0
        and len({str(row["case_id"]) for row in rows}) == EXPECTED_CASE_COUNT
    )
    verification_path = output_dir / "generation-verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-audio-generation-verification/v1",
                "passed": passed,
                "model_id": plan.model_id,
                "row_count": len(rows),
                "status_counts": dict(counts),
                "case_ids_unique": len({str(row["case_id"]) for row in rows}) == len(rows),
                "all_audio_finite": all(row.get("audio_finite") is True for row in rows),
                "generation_config_path": str(config_path),
                "generation_config_sha256": sha256_file(config_path),
                "generation_results_path": str(results_path),
                "generation_results_sha256": sha256_file(results_path),
                "upstream_runtime": upstream_guard.binding(),
                "upstream_validation_points": upstream_guard.validation_points,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "model_id": plan.model_id,
                "success": counts["SUCCESS"],
                "error": counts["ERROR"],
                "verification": str(verification_path),
                "verification_sha256": sha256_file(verification_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


def _create_runtime(runtime_api: Mapping[str, Any], *, checkpoint_path: Path) -> RuntimeLike:
    runtime_key = runtime_api["RuntimeKey"](
        checkpoint=str(checkpoint_path.absolute()),
        model_device="cuda",
        model_precision="bf16",
        codec_device="cuda",
        codec_precision="fp32",
        codec_deterministic_encode=True,
        codec_deterministic_decode=True,
        compile_model=False,
        compile_dynamic=False,
    )
    return cast("RuntimeLike", runtime_api["InferenceRuntime"].from_key(runtime_key))


def _write_generation_results(
    cases: Sequence[GenerationCase],
    *,
    plan: GenerationPlan,
    wav_dir: Path,
    results_path: Path,
    runtime: RuntimeLike,
    runtime_api: Mapping[str, Any],
    upstream_guard: UpstreamGuardLike,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with results_path.open("w", encoding="utf-8", newline="\n") as results_file:
        for index, case in enumerate(cases, start=1):
            previous_step = cases[index - 2].checkpoint.checkpoint_step if index > 1 else None
            checkpoint_first_case = previous_step != case.checkpoint.checkpoint_step
            if checkpoint_first_case:
                upstream_guard.verify(f"checkpoint_{case.checkpoint.checkpoint_step}_before_load")
            row = _generate_case_result(
                case,
                plan=plan,
                wav_dir=wav_dir,
                runtime=runtime,
                runtime_api=runtime_api,
            )
            if checkpoint_first_case:
                upstream_guard.verify(f"checkpoint_{case.checkpoint.checkpoint_step}_after_load")
            row["upstream_runtime"] = upstream_guard.binding()
            row["upstream_checkpoint_validation_points"] = [
                f"checkpoint_{case.checkpoint.checkpoint_step}_before_load",
                f"checkpoint_{case.checkpoint.checkpoint_step}_after_load",
            ]
            rows.append(row)
            results_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            results_file.flush()
            print(f"[{index}/{len(cases)}] {case.case_id}: {row['status']}", flush=True)
    return rows


def _generate_case_result(
    case: GenerationCase,
    *,
    plan: GenerationPlan,
    wav_dir: Path,
    runtime: RuntimeLike,
    runtime_api: Mapping[str, Any],
) -> dict[str, object]:
    wav_path = wav_dir / f"{case.case_id}.wav"
    started = time.perf_counter()
    try:
        metadata = _synthesize_case(
            case,
            wav_path=wav_path,
            runtime=runtime,
            runtime_api=runtime_api,
        )
    except Exception as error:  # noqa: BLE001 - every case must emit an auditable row.
        return _build_error_row(
            case,
            plan=plan,
            elapsed_seconds=time.perf_counter() - started,
            error=error,
        )
    return build_success_row(
        case,
        plan=plan,
        wav_path=wav_path,
        elapsed_seconds=time.perf_counter() - started,
        audio_metadata=metadata,
    )


def _synthesize_case(
    case: GenerationCase,
    *,
    wav_path: Path,
    runtime: RuntimeLike,
    runtime_api: Mapping[str, Any],
) -> dict[str, object]:
    actual_embedding_sha256 = sha256_file(case.checkpoint.embedding_path)
    if actual_embedding_sha256 != case.checkpoint.embedding_sha256:
        message = f"embedding SHA-256 mismatch: {case.checkpoint.embedding_path}"
        raise ValueError(message)
    request = runtime_api["SamplingRequest"](
        text=case.text,
        caption=case.caption,
        ref_embed=str(case.checkpoint.embedding_path),
        num_candidates=1,
        decode_mode="sequential",
        duration_scale=1.0,
        num_steps=40,
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0,
        cfg_scale_speaker=5.0,
        cfg_guidance_mode="independent",
        context_kv_cache=True,
        seed=case.seed,
        t_schedule_mode="linear",
        sway_coeff=-1.0,
    )
    result = runtime.synthesize(request, log_fn=None)
    saved = Path(runtime_api["save_wav"](str(wav_path), result.audio, result.sample_rate))
    if saved.resolve() != wav_path.resolve():
        message = f"runtime saved WAV to unexpected path: {saved}"
        raise ValueError(message)
    metadata = _audio_metadata(result.audio, wav_path)
    metadata["used_seed"] = result.used_seed
    return metadata


def _unload_runtime(runtime: object) -> None:
    unload = getattr(runtime, "unload", None)
    if callable(unload):
        unload()


def _load_runtime_api(
    upstream_root: Path,
    *,
    package_archive: Path,
) -> dict[str, Any]:
    root = validate_upstream_root(upstream_root)
    if package_archive.resolve() == root:
        message = "upstream package archive must be separate from the worktree"
        raise ValueError(message)
    sys.path.insert(0, str(package_archive.resolve()))
    runtime_module = importlib.import_module("irodori_tts.inference_runtime")
    _verify_imported_upstream_modules(package_archive.resolve())

    return {
        "InferenceRuntime": runtime_module.InferenceRuntime,
        "RuntimeKey": runtime_module.RuntimeKey,
        "SamplingRequest": runtime_module.SamplingRequest,
        "save_wav": runtime_module.save_wav,
    }


def _verify_imported_upstream_modules(package_archive: Path) -> None:
    archive_prefix = str(package_archive.resolve()).replace("\\", "/") + "/"
    imported = []
    for name, module in sys.modules.items():
        if name != "irodori_tts" and not name.startswith("irodori_tts."):
            continue
        raw_source = getattr(module, "__file__", None)
        if not isinstance(raw_source, str):
            message = f"imported upstream module has no source file: {name}"
            raise TypeError(message)
        source = raw_source.replace("\\", "/")
        if not source.startswith(archive_prefix):
            message = f"imported upstream module is not archive-bound: {name}: {raw_source}"
            raise ValueError(message)
        imported.append(name)
    if "irodori_tts" not in imported or "irodori_tts.inference_runtime" not in imported:
        message = "upstream runtime package was not imported from the package archive"
        raise ValueError(message)


def validate_upstream_root(upstream_root: Path) -> Path:
    root = upstream_root.resolve()
    if not (root / "irodori_tts" / "inference_runtime.py").is_file():
        message = f"upstream runtime is missing: {root}"
        raise ValueError(message)
    return root


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "generate"))
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint-path", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-runtime-provenance", type=Path, required=True)
    parser.add_argument("--upstream-package-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--upstream-runtime-provenance-sha256", required=True)
    parser.add_argument("--upstream-package-archive-sha256", required=True)
    args = parser.parse_args(argv)
    if args.mode == "generate" and args.output_dir is None:
        parser.error("generate mode requires --output-dir")
    if args.mode == "preflight" and args.output_dir is not None:
        parser.error("preflight mode does not accept --output-dir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_generation_plan(args.checkpoint_manifest)
    base_sha256 = validate_base_checkpoint(args.base_checkpoint_path, plan=plan)
    validate_upstream_root(args.upstream_root)
    upstream_guard = load_upstream_runtime_provenance(
        args.upstream_runtime_provenance,
        expected_sha256=args.upstream_runtime_provenance_sha256,
        upstream_root=args.upstream_root,
        package_archive=args.upstream_package_archive,
        expected_package_archive_sha256=args.upstream_package_archive_sha256,
    )
    upstream_guard.verify("preflight")
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "passed": True,
                    "model_id": plan.model_id,
                    "case_count": len(build_cases(plan)),
                    "checkpoint_manifest_sha256": plan.evaluation_manifest_sha256,
                    "base_checkpoint_sha256": base_sha256,
                    "upstream_runtime": upstream_guard.binding(),
                    "upstream_validation_points": upstream_guard.validation_points,
                    "embedding_sha256": {
                        str(candidate.checkpoint_step): candidate.embedding_sha256
                        for candidate in plan.checkpoints
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        return 0
    return generate(
        plan=plan,
        checkpoint_path=args.base_checkpoint_path,
        upstream_root=args.upstream_root,
        upstream_guard=upstream_guard,
        output_path=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
