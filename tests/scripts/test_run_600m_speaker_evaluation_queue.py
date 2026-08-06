from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import socket
import subprocess  # noqa: S404 - tests spawn a fixed no-op interpreter to obtain a dead PID.
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/run_600m_speaker_evaluation_queue.py")
MODEL_COUNT = 12
FAILED_EXIT_CODE = 17
CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
EVALUATION_CASE_COUNT = 140
RUNTIME_PAYLOAD_COUNT = 11


def _evaluation_manifest(model_id: str, *, base_checkpoint: str) -> dict[str, object]:
    checkpoints = [
        {
            "checkpoint_step": step,
            "embedding_path": f"/training/{model_id}/checkpoint_{step:07d}.speaker.safetensors",
            "embedding_sha256": f"{step:064x}",
            "training_config_sha256": "d" * 64,
            "base_checkpoint": base_checkpoint,
            "base_checkpoint_sha256": "e" * 64,
            "base_revision": "base-revision",
            "run_id": "f" * 64,
        }
        for step in CHECKPOINT_STEPS
    ]
    return {
        "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
        "models": [{"model_id": model_id, "checkpoints": checkpoints}],
        "text_ids": ["word_unko", "control"],
        "seeds": [1234, 5678],
        "styles": ["neutral", "calm"],
        "metrics_provenance": {
            "reference_wavs_sha256": "1" * 64,
            "speaker_embedding": {
                "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                "revision": "ecapa-revision",
                "source_sha256": "b" * 64,
            },
            "transcription": {
                "model_id": "openai/whisper-large-v3-turbo",
                "revision": "whisper-revision",
                "source_sha256": "c" * 64,
            },
        },
    }


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_600m_speaker_evaluation_queue",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(
    tmp_path: Path,
    *,
    bundle_name: str = "queue",
    config_name: str = "evaluation-queue.json",
) -> Path:
    config_path = tmp_path / bundle_name / config_name
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-evaluation-queue/v1",
                "training_status": "training-status.jsonl",
                "training_jobs": "training-jobs.json",
                "manifest_output_dir": "manifests",
                "base_checkpoint": {
                    "model_id": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                    "path": "base/model.safetensors",
                    "sha256": "a" * 64,
                    "revision": "base-revision",
                },
                "upstream_root": "upstream",
                "metric_models": {
                    "speaker_embedding": {
                        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                        "revision": "ecapa-revision",
                        "source_sha256": "b" * 64,
                        "source": "models/ecapa",
                        "savedir": "models/ecapa-cache",
                    },
                    "transcription": {
                        "model_id": "openai/whisper-large-v3-turbo",
                        "revision": "whisper-revision",
                        "source_sha256": "c" * 64,
                        "source": "models/whisper",
                        "device": "cuda:0",
                    },
                },
                "models": [
                    {
                        "model_id": f"model-{index:02d}",
                        "reference_wavs": f"references/model-{index:02d}.json",
                        "generation_dir": f"evaluation/model-{index:02d}/generation",
                        "analysis_dir": f"evaluation/model-{index:02d}/analysis",
                        "metrics_dir": f"evaluation/model-{index:02d}/metrics",
                        "evaluation_dir": f"evaluation/model-{index:02d}/selection",
                    }
                    for index in range(MODEL_COUNT)
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _prepare_frozen_runtime(tmp_path: Path) -> tuple[Path, Path]:
    config_path = _write_config(
        tmp_path,
        bundle_name="runtime-inputs-v1",
        config_name="evaluation-queue-runtime.json",
    )
    external = tmp_path / "external"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["base_checkpoint"]["path"] = str(external / "base/model.safetensors")
    document["manifest_output_dir"] = str(external / "manifests")
    document["upstream_root"] = str(external / "upstream")
    document["metric_models"]["speaker_embedding"]["source"] = str(external / "models/ecapa")
    document["metric_models"]["speaker_embedding"]["savedir"] = str(external / "models/ecapa-cache")
    document["metric_models"]["transcription"]["source"] = str(external / "models/whisper")
    for model in document["models"]:
        model_id = model["model_id"]
        model["reference_wavs"] = str(external / f"references/{model_id}.json")
        model["generation_dir"] = str(external / f"evaluation/{model_id}/generation")
        model["analysis_dir"] = str(external / f"evaluation/{model_id}/analysis")
        model["metrics_dir"] = str(external / f"evaluation/{model_id}/metrics")
        model["evaluation_dir"] = str(external / f"evaluation/{model_id}/selection")
    config_path.write_text(json.dumps(document), encoding="utf-8")
    _prepare_inputs(config_path)

    scripts_dir = config_path.parent / "scripts"
    for name in (
        "run_600m_speaker_evaluation_queue.py",
        "build_600m_checkpoint_evaluation_manifests.py",
        "generate_600m_checkpoint_audio_remote.py",
        "analyze_nko_beep_matrix.py",
        "compute_600m_speaker_metrics.py",
        "evaluate_600m_speaker_checkpoints.py",
    ):
        _write_file(scripts_dir / name, f"# {name}\n")
    root = config_path.parent
    payloads = sorted(
        (path.relative_to(root) for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    assert len(payloads) == RUNTIME_PAYLOAD_COUNT
    manifest = {
        "schema_version": "speaker-evaluation-runtime-inputs/v1",
        "source_inputs": {
            str((tmp_path / f"source-{index}").resolve()): f"{index:064x}" for index in range(9)
        },
        "files": {
            relative.as_posix(): {
                "sha256": _sha256(root / relative),
                "size": (root / relative).stat().st_size,
            }
            for relative in payloads
        },
    }
    manifest_path = root / "snapshot-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path, scripts_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_inputs(  # noqa: PLR0914, PLR0915 - fixture mirrors the canonical artifact contract.
    config_path: Path,
    *,
    successful_models: int = MODEL_COUNT,
    reuse_first: bool = False,
    pilot_canonical: bool = False,
) -> None:
    root = config_path.parent
    document = json.loads(config_path.read_text(encoding="utf-8"))
    base_checkpoint = root / document["base_checkpoint"]["path"]
    base_checkpoint.parent.mkdir(parents=True)
    base_checkpoint.write_bytes(b"base checkpoint")
    document["base_checkpoint"]["sha256"] = _sha256(base_checkpoint)

    jobs = [{"model_id": f"model-{index:02d}"} for index in range(MODEL_COUNT)]
    (root / document["training_jobs"]).write_text(
        json.dumps({"jobs": jobs}),
        encoding="utf-8",
    )
    (root / document["training_status"]).write_text(
        "".join(
            json.dumps(
                {
                    "event": "finished",
                    "status": "success",
                    "model_id": f"model-{index:02d}",
                }
            )
            + "\n"
            for index in range(successful_models)
        ),
        encoding="utf-8",
    )
    for model in document["models"]:
        reference_path = root / model["reference_wavs"]
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(
            json.dumps({"model_id": model["model_id"]}),
            encoding="utf-8",
        )
    (root / document["upstream_root"]).mkdir(parents=True)
    (root / "upstream-runtime-provenance.json").write_text(
        json.dumps({"schema_version": "irodori-upstream-runtime-provenance/v1"}) + "\n",
        encoding="utf-8",
    )
    (root / "upstream-runtime-package.zip").write_bytes(b"package archive")
    (root / document["metric_models"]["speaker_embedding"]["source"]).mkdir(parents=True)
    (root / document["metric_models"]["transcription"]["source"]).mkdir(parents=True)
    if reuse_first:
        model = document["models"][0]
        for field in (
            "generation_dir",
            "analysis_dir",
            "metrics_dir",
            "evaluation_dir",
        ):
            model.pop(field)
        reuse_root = root / "canonical" / model["model_id"]
        generation_dir = reuse_root / "generation"
        analysis_dir = reuse_root / "analysis"
        metrics_dir = reuse_root / "metrics"
        evaluation_dir = reuse_root / "evaluation"
        _write_file(generation_dir / "generation-results.jsonl")
        generation_sha256 = _sha256(generation_dir / "generation-results.jsonl")
        if pilot_canonical:
            _write_file(
                generation_dir / "canonicalization-report.json",
                json.dumps(
                    {
                        "schema_version": "speaker-canonicalization/v1",
                        "model_id": model["model_id"],
                        "canonical_sha256": {"generation_results": generation_sha256},
                        "counts": {"generation": 140},
                    }
                )
                + "\n",
            )
        else:
            _write_file(
                generation_dir / "generation-verification.json",
                json.dumps(
                    {
                        "schema_version": ("speaker-checkpoint-audio-generation-verification/v1"),
                        "passed": True,
                        "model_id": model["model_id"],
                        "row_count": 140,
                        "generation_results_sha256": generation_sha256,
                    }
                )
                + "\n",
            )
        _write_file(analysis_dir / "analysis-results.jsonl")
        _write_file(metrics_dir / "metrics-results.jsonl")
        _write_file(metrics_dir / "metrics-results.provenance.json")
        canonical_manifest = reuse_root / "evaluation-manifest.json"
        _write_file(
            canonical_manifest,
            json.dumps(_evaluation_manifest(model["model_id"], base_checkpoint="pilot/base"))
            + "\n",
        )
        selected_inputs = {
            "generation_results": _sha256(generation_dir / "generation-results.jsonl"),
            "analysis_results": _sha256(analysis_dir / "analysis-results.jsonl"),
            "metrics_results": _sha256(metrics_dir / "metrics-results.jsonl"),
            "metrics_provenance": _sha256(metrics_dir / "metrics-results.provenance.json"),
            "evaluation_manifest": _sha256(canonical_manifest),
        }
        canonical_models = _evaluation_manifest(model["model_id"], base_checkpoint="pilot/base")[
            "models"
        ]
        assert isinstance(canonical_models, list)
        canonical_model = canonical_models[0]
        assert isinstance(canonical_model, dict)
        canonical_checkpoints = canonical_model["checkpoints"]
        assert isinstance(canonical_checkpoints, list)
        canonical_selected = canonical_checkpoints[0]
        assert isinstance(canonical_selected, dict)
        selected = {
            **canonical_selected,
            "model_id": model["model_id"],
            "rank": 1,
        }
        selected_path = evaluation_dir / "selected-models.json"
        _write_file(
            selected_path,
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-evaluation/v1",
                    "input_sha256": selected_inputs,
                    "selections": [selected],
                }
            )
            + "\n",
        )
        _write_file(
            evaluation_dir / "evaluation-verification.json",
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-evaluation-verification/v2",
                    "status": "PASS",
                    "selected": selected,
                    "artifact_sha256": {str(selected_path): _sha256(selected_path)},
                }
            )
            + "\n",
        )
        model["reuse"] = {
            "generation_dir": str(generation_dir),
            "analysis_dir": str(analysis_dir),
            "metrics_results": str(metrics_dir / "metrics-results.jsonl"),
            "metrics_provenance": str(metrics_dir / "metrics-results.provenance.json"),
            "evaluation_manifest": str(canonical_manifest),
            "evaluation_dir": str(evaluation_dir),
        }
    config_path.write_text(json.dumps(document), encoding="utf-8")


def test_generation_stage_binds_upstream_runtime_provenance_in_fingerprint(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    scripts_dir = tmp_path / "scripts"
    for name in (
        "generate_600m_checkpoint_audio_remote.py",
        "analyze_nko_beep_matrix.py",
        "compute_600m_speaker_metrics.py",
        "evaluate_600m_speaker_checkpoints.py",
    ):
        _write_file(scripts_dir / name, f"# {name}\n")
    config = module.load_queue_config(config_path)
    manifest = config.manifest_output_dir / config.models[0].model_id / "evaluation-manifest.json"
    _write_file(manifest)

    stage = module._model_stages(  # noqa: SLF001 - producer contract regression.
        config,
        model=config.models[0],
        scripts_dir=scripts_dir,
    )[0]
    provenance = config.source_path.parent / "upstream-runtime-provenance.json"
    package_archive = config.source_path.parent / "upstream-runtime-package.zip"
    provenance_sha256 = _sha256(provenance)
    package_archive_sha256 = _sha256(package_archive)
    before = module._stage_fingerprint(stage)  # noqa: SLF001
    provenance.write_text('{"changed":true}\n', encoding="utf-8")
    after = module._stage_fingerprint(stage)  # noqa: SLF001

    assert provenance in stage.input_files
    assert package_archive in stage.input_files
    assert _argument(stage.command, "--upstream-runtime-provenance") == provenance
    assert _argument(stage.command, "--upstream-package-archive") == package_archive
    assert stage.command[stage.command.index("--upstream-runtime-provenance-sha256") + 1] == (
        provenance_sha256
    )
    assert stage.command[stage.command.index("--upstream-package-archive-sha256") + 1] == (
        package_archive_sha256
    )
    assert before != after


def test_frozen_runtime_guard_supplies_immutable_generation_hash_arguments(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path, scripts_dir = _prepare_frozen_runtime(tmp_path)
    config = module.load_queue_config(config_path)
    guard = module._load_runtime_snapshot_guard(  # noqa: SLF001 - integrity boundary.
        config,
        scripts_dir=scripts_dir,
    )
    assert guard is not None
    manifest = config.manifest_output_dir / config.models[0].model_id / "evaluation-manifest.json"
    _write_file(manifest)
    provenance = config_path.parent / "upstream-runtime-provenance.json"
    package = config_path.parent / "upstream-runtime-package.zip"
    expected_provenance_sha256 = _sha256(provenance)
    expected_package_sha256 = _sha256(package)

    provenance.write_bytes(b"x" * provenance.stat().st_size)
    stage = module._model_stages(  # noqa: SLF001 - producer contract regression.
        config,
        model=config.models[0],
        scripts_dir=scripts_dir,
        runtime_guard=guard,
    )[0]

    assert stage.command[stage.command.index("--upstream-runtime-provenance-sha256") + 1] == (
        expected_provenance_sha256
    )
    assert stage.command[stage.command.index("--upstream-package-archive-sha256") + 1] == (
        expected_package_sha256
    )


def test_queue_aborts_before_changed_component_can_execute_or_succeed(tmp_path: Path) -> None:
    module = _load_script()
    config_path, scripts_dir = _prepare_frozen_runtime(tmp_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    calls: list[tuple[str, ...]] = []
    artifact_runner = _artifact_runner(calls)
    generator = scripts_dir / "generate_600m_checkpoint_audio_remote.py"

    def mutate_after_manifest(command: tuple[str, ...], log_path: Path) -> int:
        result = artifact_runner(command, log_path)
        if Path(command[1]).name == "build_600m_checkpoint_evaluation_manifests.py":
            generator.write_bytes(b"x" * generator.stat().st_size)
        return result

    with pytest.raises(ValueError, match="runtime snapshot content changed"):
        module.run_evaluation_queue(
            config,
            status_path=status_path,
            scripts_dir=scripts_dir,
            runner=mutate_after_manifest,
        )

    assert [Path(command[1]).name for command in calls] == [
        "build_600m_checkpoint_evaluation_manifests.py"
    ]
    rows = _status_rows(status_path)
    assert not any(row["status"] == "success" for row in rows)
    assert _sha256(generator) not in status_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("target_name", "mutation"),
    [
        ("upstream-runtime-provenance.json", "same-size"),
        ("upstream-runtime-package.zip", "size"),
        ("snapshot-manifest.json", "same-size"),
    ],
)
def test_queue_reverifies_frozen_bytes_loaded_at_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    mutation: str,
) -> None:
    module = _load_script()
    config_path, scripts_dir = _prepare_frozen_runtime(tmp_path)
    config = module.load_queue_config(config_path)
    calls: list[tuple[str, ...]] = []
    original_validate = module._validate_ready_training  # noqa: SLF001

    def mutate_after_guard(loaded_config: object) -> None:
        original_validate(loaded_config)
        target = config_path.parent / target_name
        if mutation == "size":
            target.write_bytes(target.read_bytes() + b"drift")
        else:
            target.write_bytes(b"x" * target.stat().st_size)

    monkeypatch.setattr(module, "_validate_ready_training", mutate_after_guard)

    with pytest.raises(ValueError, match=r"runtime snapshot (?:manifest|content) changed"):
        module.run_evaluation_queue(
            config,
            status_path=tmp_path / "evaluation-status.jsonl",
            scripts_dir=scripts_dir,
            runner=_artifact_runner(calls),
        )

    assert calls == []


@pytest.mark.parametrize("mutation", ["extra", "missing", "symlink", "reparse", "casefold"])
def test_frozen_runtime_guard_rejects_invalid_actual_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = _load_script()
    config_path, scripts_dir = _prepare_frozen_runtime(tmp_path)
    root = config_path.parent
    manifest_path = root / "snapshot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        _write_file(root / "extra.txt", "extra\n")
        expected = "file set mismatch"
    elif mutation == "missing":
        (scripts_dir / "analyze_nko_beep_matrix.py").unlink()
        expected = "file set mismatch"
    elif mutation == "symlink":
        target = scripts_dir / "analyze_nko_beep_matrix.py"
        content = target.read_bytes()
        target.unlink()
        external = tmp_path / "alias-target.py"
        external.write_bytes(content)
        target.symlink_to(external)
        expected = "filesystem alias"
    elif mutation == "reparse":
        target = scripts_dir / "analyze_nko_beep_matrix.py"
        original_alias_check = module._is_filesystem_alias  # noqa: SLF001
        monkeypatch.setattr(
            module,
            "_is_filesystem_alias",
            lambda path: path == target or original_alias_check(path),
        )
        expected = "filesystem alias"
    else:
        relative = "Scripts/run_600m_speaker_evaluation_queue.py"
        _write_file(root / relative, "# collision\n")
        manifest["files"][relative] = {
            "sha256": _sha256(root / relative),
            "size": (root / relative).stat().st_size,
        }
        manifest["files"] = dict(sorted(manifest["files"].items()))
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = "case-insensitive collision"

    with pytest.raises(ValueError, match=expected):
        module._load_runtime_snapshot_guard(  # noqa: SLF001 - integrity boundary.
            module.load_queue_config(config_path),
            scripts_dir=scripts_dir,
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("schema", "schema mismatch"),
        ("unsorted", "inventory must be sorted"),
        ("unsafe", "unsafe frozen runtime snapshot path"),
        ("hash-type", "SHA-256 is invalid"),
        ("size-type", "size is invalid"),
    ],
)
def test_frozen_runtime_guard_strictly_validates_manifest_contract(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    module = _load_script()
    config_path, scripts_dir = _prepare_frozen_runtime(tmp_path)
    manifest_path = config_path.parent / "snapshot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    first = next(iter(files))
    if mutation == "schema":
        manifest["schema_version"] = "speaker-evaluation-runtime-inputs/v2"
    elif mutation == "unsorted":
        manifest["files"] = dict(reversed(list(files.items())))
    elif mutation == "unsafe":
        binding = files.pop(first)
        manifest["files"] = {"../escape": binding, **files}
    elif mutation == "hash-type":
        files[first]["sha256"] = 123
    else:
        files[first]["size"] = True
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=expected):
        module._load_runtime_snapshot_guard(  # noqa: SLF001 - integrity boundary.
            module.load_queue_config(config_path),
            scripts_dir=scripts_dir,
        )


def test_frozen_runtime_bundle_requires_snapshot_manifest(tmp_path: Path) -> None:
    module = _load_script()
    config_path, scripts_dir = _prepare_frozen_runtime(tmp_path)
    (config_path.parent / "snapshot-manifest.json").unlink()

    with pytest.raises(ValueError, match="manifest is missing"):
        module._load_runtime_snapshot_guard(  # noqa: SLF001 - integrity boundary.
            module.load_queue_config(config_path),
            scripts_dir=scripts_dir,
        )


def test_frozen_runtime_bundle_rejects_all_missing_snapshot_metadata_before_subprocess(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path, scripts_dir = _prepare_frozen_runtime(tmp_path)
    for name in (
        "snapshot-manifest.json",
        "upstream-runtime-provenance.json",
        "upstream-runtime-package.zip",
    ):
        (config_path.parent / name).unlink()
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="manifest is missing"):
        module.run_evaluation_queue(
            module.load_queue_config(config_path),
            status_path=tmp_path / "evaluation-status.jsonl",
            scripts_dir=scripts_dir,
            runner=_artifact_runner(calls),
        )

    assert calls == []


def _write_file(path: Path, content: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _argument(command: tuple[str, ...], flag: str) -> Path:
    return Path(command[command.index(flag) + 1])


def _artifact_runner(
    calls: list[tuple[str, ...]],
) -> Callable[[tuple[str, ...], Path], int]:
    def run(command: tuple[str, ...], _log_path: Path) -> int:
        calls.append(command)
        script = Path(command[1]).name
        if script == "build_600m_checkpoint_evaluation_manifests.py":
            output_dir = _argument(command, "--output-dir")
            for index in range(MODEL_COUNT):
                model_id = f"model-{index:02d}"
                _write_file(
                    output_dir / model_id / "evaluation-manifest.json",
                    json.dumps(
                        _evaluation_manifest(
                            model_id,
                            base_checkpoint="Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                        )
                    )
                    + "\n",
                )
            _write_file(output_dir / "manifest-index.json")
        elif script == "generate_600m_checkpoint_audio_remote.py":
            output_dir = _argument(command, "--output-dir")
            _write_file(output_dir / "generation-results.jsonl")
            _write_file(output_dir / "generation-verification.json")
        elif script == "analyze_nko_beep_matrix.py":
            output_dir = _argument(command, "--output-dir")
            _write_file(output_dir / "analysis-results.jsonl")
        elif script == "compute_600m_speaker_metrics.py":
            _write_file(_argument(command, "--output"))
            _write_file(_argument(command, "--provenance-output"))
        elif script == "evaluate_600m_speaker_checkpoints.py":
            output_dir = _argument(command, "--output-dir")
            _write_file(output_dir / "selected-models.json")
            _write_file(output_dir / "evaluation-verification.json")
        else:
            message = f"unexpected command: {command}"
            raise AssertionError(message)
        return 0

    return run


def _status_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_load_queue_config_resolves_exactly_twelve_models(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)

    config = module.load_queue_config(config_path)

    assert module.EXPECTED_CHECKPOINT_STEPS == CHECKPOINT_STEPS
    assert module.EXPECTED_GENERATION_COUNT == EVALUATION_CASE_COUNT
    assert len(config.models) == MODEL_COUNT
    assert config.training_status == config_path.parent / "training-status.jsonl"
    assert config.models[0].generation_dir == (
        config_path.parent / "evaluation/model-00/generation"
    )
    assert config.metric_models.transcription.device == "cuda:0"


def test_queue_refuses_to_start_until_all_twelve_training_jobs_succeed(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path, successful_models=MODEL_COUNT - 1)
    config = module.load_queue_config(config_path)
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="all 12 training models"):
        module.run_evaluation_queue(
            config,
            status_path=tmp_path / "evaluation-status.jsonl",
            scripts_dir=Path("scripts"),
            runner=_artifact_runner(calls),
        )

    assert calls == []
    assert not config.manifest_output_dir.exists()


def test_queue_runs_builder_then_each_model_serially_and_resumes_from_hashes(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    calls: list[tuple[str, ...]] = []

    first = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=_artifact_runner(calls),
    )

    assert len(calls) == 1 + MODEL_COUNT * 4
    assert Path(calls[0][1]).name == "build_600m_checkpoint_evaluation_manifests.py"
    assert [Path(command[1]).name for command in calls[1:5]] == [
        "generate_600m_checkpoint_audio_remote.py",
        "analyze_nko_beep_matrix.py",
        "compute_600m_speaker_metrics.py",
        "evaluate_600m_speaker_checkpoints.py",
    ]
    assert first.failed == ()
    assert len(first.succeeded) == 1 + MODEL_COUNT * 4
    success_rows = [row for row in _status_rows(status_path) if row["status"] == "success"]
    assert len(success_rows) == 1 + MODEL_COUNT * 4
    assert all(row["outputs"] for row in success_rows)

    calls.clear()
    second = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=_artifact_runner(calls),
    )

    assert calls == []
    assert len(second.skipped) == 1 + MODEL_COUNT * 4
    assert second.failed == ()


def test_queue_reuses_only_explicit_canonical_model_artifacts(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path, reuse_first=True)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    calls: list[tuple[str, ...]] = []

    result = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=_artifact_runner(calls),
    )

    assert len(calls) == 1 + (MODEL_COUNT - 1) * 4
    assert result.reused == (
        "model-00:generation",
        "model-00:analysis",
        "model-00:metrics",
        "model-00:evaluate",
    )
    model_zero_commands = [command for command in calls[1:] if "model-00" in " ".join(command)]
    assert model_zero_commands == []


def test_queue_accepts_pilot_canonicalization_report_and_resumes_per_file(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path, reuse_first=True, pilot_canonical=True)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    first_calls: list[tuple[str, ...]] = []

    first = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=_artifact_runner(first_calls),
    )

    assert first.failed == ()
    assert config.models[0].reuse is not None
    _write_file(config.models[0].reuse.generation_dir / "later-analysis-artifact.json")
    second_calls: list[tuple[str, ...]] = []
    second = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=_artifact_runner(second_calls),
    )

    assert second_calls == []
    assert "model-00:generation" in second.skipped
    assert "model-00:analysis" in second.skipped


def test_queue_rejects_tampered_pilot_canonicalization_report(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path, reuse_first=True, pilot_canonical=True)
    config = module.load_queue_config(config_path)
    assert config.models[0].reuse is not None
    report_path = config.models[0].reuse.generation_dir / "canonicalization-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["canonical_sha256"]["generation_results"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical reuse binding"):
        module.run_evaluation_queue(
            config,
            status_path=tmp_path / "evaluation-status.jsonl",
            scripts_dir=Path("scripts"),
            runner=_artifact_runner([]),
        )


def test_queue_rejects_semantic_drift_between_canonical_and_new_manifest(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path, reuse_first=True, pilot_canonical=True)
    config = module.load_queue_config(config_path)
    assert config.models[0].reuse is not None
    canonical_path = config.models[0].reuse.evaluation_manifest
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["models"][0]["checkpoints"][1]["embedding_sha256"] = "9" * 64
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    selected_path = config.models[0].reuse.evaluation_dir / "selected-models.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected["input_sha256"]["evaluation_manifest"] = _sha256(canonical_path)
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    verification_path = config.models[0].reuse.evaluation_dir / "evaluation-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["artifact_sha256"][str(selected_path)] = _sha256(selected_path)
    verification_path.write_text(json.dumps(verification), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic manifest binding"):
        module.run_evaluation_queue(
            config,
            status_path=tmp_path / "evaluation-status.jsonl",
            scripts_dir=Path("scripts"),
            runner=_artifact_runner([]),
        )


def test_manifest_semantics_rejects_noncanonical_checkpoint_order(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _evaluation_manifest("model-00", base_checkpoint="pilot/base")
    models = manifest["models"]
    assert isinstance(models, list)
    model = models[0]
    assert isinstance(model, dict)
    checkpoints = model["checkpoints"]
    assert isinstance(checkpoints, list)
    model["checkpoints"] = list(reversed(checkpoints))
    manifest_path = tmp_path / "evaluation-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint order"):
        module._manifest_semantics(  # noqa: SLF001 - exercises the fail-closed boundary.
            manifest_path,
            model_id="model-00",
        )


def test_queue_rejects_canonical_reuse_not_bound_to_current_inputs(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path, reuse_first=True)
    config = module.load_queue_config(config_path)
    assert config.models[0].reuse is not None
    (config.models[0].reuse.generation_dir / "generation-results.jsonl").write_text(
        '{"tampered":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical reuse binding"):
        module.run_evaluation_queue(
            config,
            status_path=tmp_path / "evaluation-status.jsonl",
            scripts_dir=Path("scripts"),
            runner=_artifact_runner([]),
        )


def test_main_runs_the_configured_queue_and_returns_zero(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    calls: list[tuple[str, ...]] = []

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--status-path",
            str(status_path),
            "--scripts-dir",
            "scripts",
        ],
        runner=_artifact_runner(calls),
    )

    assert exit_code == 0
    assert len(calls) == 1 + MODEL_COUNT * 4


def test_queue_fails_without_overwriting_a_colliding_stage_directory(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    assert config.models[0].generation_dir is not None
    _write_file(config.models[0].generation_dir / "do-not-overwrite.txt", "owned\n")
    status_path = tmp_path / "evaluation-status.jsonl"
    calls: list[tuple[str, ...]] = []

    with pytest.raises(FileExistsError, match="already exists"):
        module.run_evaluation_queue(
            config,
            status_path=status_path,
            scripts_dir=Path("scripts"),
            runner=_artifact_runner(calls),
        )

    assert len(calls) == 1
    assert (config.models[0].generation_dir / "do-not-overwrite.txt").read_text() == "owned\n"
    assert _status_rows(status_path)[-1]["status"] == "failed"


def test_queue_continues_later_models_after_failed_model_stage(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    successful_runner = _artifact_runner([])
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], log_path: Path) -> int:
        calls.append(command)
        if Path(
            command[1]
        ).name == "generate_600m_checkpoint_audio_remote.py" and "model-00" in " ".join(command):
            return FAILED_EXIT_CODE
        return successful_runner(command, log_path)

    result = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=runner,
    )

    assert len(calls) == 1 + 1 + (MODEL_COUNT - 1) * 4
    assert [Path(command[1]).name for command in calls[:2]] == [
        "build_600m_checkpoint_evaluation_manifests.py",
        "generate_600m_checkpoint_audio_remote.py",
    ]
    assert not any("model-00" in " ".join(command) for command in calls[2:])
    assert any("model-11" in " ".join(command) for command in calls[2:])
    assert result.failed == ("model-00:generation",)
    rows = _status_rows(status_path)
    [failed] = [row for row in rows if row["status"] == "failed"]
    assert failed["stage"] == "model-00:generation"
    assert failed["exit_code"] == FAILED_EXIT_CODE
    assert rows[-1]["stage"] == "model-11:evaluate"
    assert rows[-1]["status"] == "success"


def test_queue_reuses_unchanged_builder_after_versioning_a_failed_model_attempt(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    builder_runner = _artifact_runner([])

    def fail_generation(command: tuple[str, ...], log_path: Path) -> int:
        if Path(
            command[1]
        ).name == "generate_600m_checkpoint_audio_remote.py" and "model-00" in " ".join(command):
            output_dir = _argument(command, "--output-dir")
            _write_file(output_dir / "partial-attempt.txt")
            return FAILED_EXIT_CODE
        return builder_runner(command, log_path)

    first = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=fail_generation,
    )
    assert first.failed == ("model-00:generation",)

    document = json.loads(config_path.read_text(encoding="utf-8"))
    first_model = document["models"][0]
    for field in ("generation_dir", "analysis_dir", "metrics_dir", "evaluation_dir"):
        first_model[field] += "-v2"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    versioned_config = module.load_queue_config(config_path)
    calls: list[tuple[str, ...]] = []

    second = module.run_evaluation_queue(
        versioned_config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=_artifact_runner(calls),
    )

    assert Path(calls[0][1]).name == "generate_600m_checkpoint_audio_remote.py"
    assert "manifests" in second.skipped
    assert second.failed == ()


def test_component_script_drift_invalidates_only_its_successful_stage(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    scripts_dir = tmp_path / "component-scripts"
    script_names = (
        "build_600m_checkpoint_evaluation_manifests.py",
        "generate_600m_checkpoint_audio_remote.py",
        "analyze_nko_beep_matrix.py",
        "compute_600m_speaker_metrics.py",
        "evaluate_600m_speaker_checkpoints.py",
    )
    for script_name in script_names:
        _write_file(scripts_dir / script_name, f"# {script_name} v1\n")

    first = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=scripts_dir,
        runner=_artifact_runner([]),
    )
    assert first.failed == ()
    analyzer = scripts_dir / "analyze_nko_beep_matrix.py"
    analyzer.write_text("# analyzer v2\n", encoding="utf-8")
    for model in config.models:
        assert model.analysis_dir is not None
        shutil.rmtree(model.analysis_dir)
    calls: list[tuple[str, ...]] = []

    second = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=scripts_dir,
        runner=_artifact_runner(calls),
    )

    assert [Path(command[1]).name for command in calls] == [
        "analyze_nko_beep_matrix.py"
    ] * MODEL_COUNT
    assert "model-00:analysis" in second.succeeded
    analyzer_rows = [
        row
        for row in _status_rows(status_path)
        if row["stage"] == "model-00:analysis" and row["status"] == "success"
    ]
    assert analyzer_rows[-1]["component_script"] == {
        "path": str(analyzer.resolve()),
        "sha256": _sha256(analyzer),
    }


def test_queue_rejects_a_second_live_process_for_the_same_status_config(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"

    with module.evaluation_queue_lock(config=config, status_path=status_path):
        assert module.queue_lock_path(status_path).is_file()
        with pytest.raises(module.QueueLockedError, match="already locked"):
            module.run_evaluation_queue(
                config,
                status_path=status_path,
                scripts_dir=Path("scripts"),
                runner=_artifact_runner([]),
            )


def test_queue_removes_its_lock_when_a_stage_raises(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    lock_path = status_path.with_suffix(status_path.suffix + ".lock")

    def explode(_command: tuple[str, ...], _log_path: Path) -> int:
        assert lock_path.is_file()
        message = "injected runner failure"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="injected runner failure"):
        module.run_evaluation_queue(
            config,
            status_path=status_path,
            scripts_dir=Path("scripts"),
            runner=explode,
        )

    assert not lock_path.exists()


def test_queue_recovers_a_dead_same_host_process_lock(tmp_path: Path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path)
    _prepare_inputs(config_path)
    config = module.load_queue_config(config_path)
    status_path = tmp_path / "evaluation-status.jsonl"
    lock_path = module.queue_lock_path(status_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
    )
    assert process.wait() == 0
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-evaluation-queue-lock/v1",
                "pid": process.pid,
                "hostname": socket.gethostname(),
                "token": "stale-owner",
            }
        ),
        encoding="utf-8",
    )

    result = module.run_evaluation_queue(
        config,
        status_path=status_path,
        scripts_dir=Path("scripts"),
        runner=_artifact_runner([]),
    )

    assert result.failed == ()
    assert not lock_path.exists()
