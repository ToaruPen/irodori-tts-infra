# ruff: noqa: SLF001 - evaluator helpers are the cross-script compatibility contract.
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess  # noqa: S404 - tests create an isolated Git repository.
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/generate_600m_checkpoint_audio_remote.py")
STEPS = (1000, 1500, 2000, 2500, 3000)
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
CASE_COUNT = len(STEPS) * len(TEXT_IDS) * len(SEEDS) * len(STYLES)
FIRST_CHECKPOINT_STEP = STEPS[0]
EXPECTED_ELAPSED_SECONDS = 1.235


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_600m_checkpoint_audio_remote",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_external_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed Git executable and fixture arguments.
        ("git", "-C", str(root), *arguments),  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _upstream_provenance(tmp_path: Path) -> tuple[Path, Path, str, Path, str]:
    root = tmp_path / "Irodori-TTS"
    package = root / "irodori_tts"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    runtime = package / "inference_runtime.py"
    runtime.write_text(
        "class InferenceRuntime:\n    pass\n"
        "def RuntimeKey(**kwargs):\n    return kwargs\n"
        "class SamplingRequest:\n    pass\n"
        "def save_wav(*args):\n    return args[0]\n",
        encoding="utf-8",
    )
    helper = package / "helper.py"
    helper.write_text("HELPER = True\n", encoding="utf-8")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    provenance = tmp_path / "upstream-runtime-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "irodori-upstream-runtime-provenance/v1",
                "upstream_root": str(root.resolve()),
                "commit": _git(root, "rev-parse", "HEAD"),
                "tree": _git(root, "rev-parse", "HEAD^{tree}"),
                "package": "irodori_tts",
                "python_files": [
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": _sha256(path),
                    }
                    for path in sorted(package.glob("*.py"))
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "upstream-runtime-package.zip"
    with zipfile.ZipFile(archive, "w") as package_zip:
        for path in sorted(package.glob("*.py")):
            package_zip.write(path, path.relative_to(root).as_posix())
    return root, provenance, _sha256(provenance), archive, _sha256(archive)


def test_upstream_provenance_binds_git_identity_and_all_tracked_python_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    root, provenance, expected_sha256, archive, archive_sha256 = _upstream_provenance(tmp_path)
    monkeypatch.setattr(module, "PINNED_UPSTREAM_COMMIT", _git(root, "rev-parse", "HEAD"))

    binding = module.load_upstream_runtime_provenance(
        provenance,
        expected_sha256=expected_sha256,
        upstream_root=root,
        package_archive=archive,
        expected_package_archive_sha256=archive_sha256,
    )
    binding.verify("before_import")

    assert binding.sha256 == expected_sha256
    assert binding.validation_points == ["before_import"]
    (root / "irodori_tts" / "helper.py").write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"dirty|hash mismatch"):
        binding.verify("after_import")


def test_upstream_provenance_rejects_expected_sha_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    root, provenance, _expected_sha256, archive, archive_sha256 = _upstream_provenance(tmp_path)

    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        module.load_upstream_runtime_provenance(
            provenance,
            expected_sha256="0" * 64,
            upstream_root=root,
            package_archive=archive,
            expected_package_archive_sha256=archive_sha256,
        )


@pytest.mark.parametrize("alias_target", ["provenance", "package"])
def test_upstream_provenance_rejects_lexical_runtime_asset_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_target: str,
) -> None:
    module = _load_script()
    root, provenance, provenance_sha, archive, archive_sha = _upstream_provenance(tmp_path)
    monkeypatch.setattr(module, "PINNED_UPSTREAM_COMMIT", _git(root, "rev-parse", "HEAD"))
    if alias_target == "provenance":
        alias = tmp_path / "provenance-alias.json"
        alias.symlink_to(provenance)
        provenance = alias
    else:
        alias = tmp_path / "package-alias.zip"
        alias.symlink_to(archive)
        archive = alias

    with pytest.raises(ValueError, match="filesystem alias"):
        module.load_upstream_runtime_provenance(
            provenance,
            expected_sha256=provenance_sha,
            upstream_root=root,
            package_archive=archive,
            expected_package_archive_sha256=archive_sha,
        )


def test_runtime_api_imports_irodori_only_from_bound_package_archive(tmp_path: Path) -> None:
    module = _load_script()
    root, _provenance, _provenance_sha, archive, _archive_sha = _upstream_provenance(tmp_path)
    for name in tuple(sys.modules):
        if name == "irodori_tts" or name.startswith("irodori_tts."):
            sys.modules.pop(name)

    runtime_api = module._load_runtime_api(root, package_archive=archive)
    module._verify_imported_upstream_modules(archive)

    assert set(runtime_api) == {
        "InferenceRuntime",
        "RuntimeKey",
        "SamplingRequest",
        "save_wav",
    }
    assert (
        str(sys.modules["irodori_tts"].__file__)
        .replace("\\", "/")
        .startswith(str(archive.resolve()).replace("\\", "/") + "/")
    )


def test_upstream_guard_rejects_package_archive_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    root, provenance, provenance_sha, archive, archive_sha = _upstream_provenance(tmp_path)
    monkeypatch.setattr(module, "PINNED_UPSTREAM_COMMIT", _git(root, "rev-parse", "HEAD"))
    guard = module.load_upstream_runtime_provenance(
        provenance,
        expected_sha256=provenance_sha,
        upstream_root=root,
        package_archive=archive,
        expected_package_archive_sha256=archive_sha,
    )
    archive.write_bytes(archive.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="package archive SHA-256 mismatch"):
        guard.verify("checkpoint_1000_before_load")


def test_upstream_provenance_rejects_unpinned_commit_before_archive_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    root, provenance, provenance_sha, archive, archive_sha = _upstream_provenance(tmp_path)
    archive_checked = False

    def mark_archive_checked(_guard: object) -> None:
        nonlocal archive_checked
        archive_checked = True

    monkeypatch.setattr(module, "_verify_package_archive", mark_archive_checked)

    with pytest.raises(ValueError, match="upstream commit mismatch"):
        module.load_upstream_runtime_provenance(
            provenance,
            expected_sha256=provenance_sha,
            upstream_root=root,
            package_archive=archive,
            expected_package_archive_sha256=archive_sha,
        )

    assert not archive_checked


def _write_safetensors(path: Path, *, fill_value: float) -> None:
    values = np.full((16, 768), fill_value, dtype="<f4")
    tensor = values.tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": "F32",
                "shape": [16, 768],
                "data_offsets": [0, len(tensor)],
            },
        },
        separators=(",", ":"),
    ).encode()
    padding = b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header) + len(padding)) + header + padding + tensor)


def _manifest(tmp_path: Path, *, steps: tuple[int, ...] = STEPS) -> Path:
    checkpoints = []
    for step in steps:
        embedding = tmp_path / "embeddings" / f"checkpoint_{step:07d}.speaker.safetensors"
        embedding.parent.mkdir(parents=True, exist_ok=True)
        _write_safetensors(embedding, fill_value=step)
        checkpoints.append(
            {
                "checkpoint_step": step,
                "embedding_path": str(embedding),
                "embedding_sha256": _sha256(embedding),
                "training_config_sha256": "a" * 64,
                "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                "base_checkpoint_sha256": "b" * 64,
                "base_revision": "c" * 40,
                "run_id": "d" * 64,
            },
        )
    path = tmp_path / "checkpoint-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
                "models": [{"model_id": "kasumi", "checkpoints": checkpoints}],
                "text_ids": list(TEXT_IDS),
                "seeds": list(SEEDS),
                "styles": list(STYLES),
                "metrics_provenance": {
                    "reference_wavs_sha256": "e" * 64,
                    "speaker_embedding": {
                        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                        "revision": "ecapa-revision",
                        "source_sha256": "f" * 64,
                    },
                    "transcription": {
                        "model_id": "openai/whisper-large-v3-turbo",
                        "revision": "whisper-revision",
                        "source_sha256": "0" * 64,
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    return path


def test_load_checkpoint_manifest_builds_exact_140_case_matrix(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _manifest(tmp_path)

    plan = module.load_generation_plan(manifest)
    cases = module.build_cases(plan)

    assert plan.model_id == "kasumi"
    assert [candidate.checkpoint_step for candidate in plan.checkpoints] == list(STEPS)
    assert len(cases) == CASE_COUNT
    assert len({case.case_id for case in cases}) == CASE_COUNT
    assert cases[0].case_id == "kasumi__checkpoint-1000__word_unko__seed-1234__neutral"
    assert cases[-1].case_id == "kasumi__checkpoint-3000__control__seed-5678__calm"
    assert cases[0].caption is None
    assert cases[1].caption == "穏やかで優しい女性の声で、自然に話す。"
    assert plan.evaluation_manifest_sha256 == _sha256(manifest)


def test_generation_revalidates_upstream_around_each_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    plan = module.load_generation_plan(_manifest(tmp_path))
    cases = module.build_cases(plan)

    class Guard:
        def __init__(self) -> None:
            self.validation_points: list[str] = []

        def verify(self, point: str) -> None:
            self.validation_points.append(point)

        def binding(self) -> dict[str, object]:  # noqa: PLR6301 - guard protocol fixture.
            return {"sha256": "a" * 64}

    guard = Guard()
    monkeypatch.setattr(
        module,
        "_generate_case_result",
        lambda case, **_kwargs: {"case_id": case.case_id, "status": "SUCCESS"},
    )

    rows = module._write_generation_results(
        cases,
        plan=plan,
        wav_dir=tmp_path / "wav",
        results_path=tmp_path / "results.jsonl",
        runtime=object(),
        runtime_api={},
        upstream_guard=guard,
    )

    assert len(rows) == CASE_COUNT
    assert all(row["upstream_runtime"] == {"sha256": "a" * 64} for row in rows)
    assert rows[0]["upstream_checkpoint_validation_points"] == [
        "checkpoint_1000_before_load",
        "checkpoint_1000_after_load",
    ]
    assert guard.validation_points == [
        point
        for step in STEPS
        for point in (
            f"checkpoint_{step}_before_load",
            f"checkpoint_{step}_after_load",
        )
    ]


def test_generation_verification_records_archive_provenance_and_validation_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    manifest = _manifest(tmp_path)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"base")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for candidate in payload["models"][0]["checkpoints"]:
        candidate["base_checkpoint_sha256"] = _sha256(checkpoint)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    plan = module.load_generation_plan(manifest)
    archive = tmp_path / "upstream-runtime-package.zip"
    archive.write_bytes(b"archive")

    class Guard:
        def __init__(self) -> None:
            self.package_archive = archive
            self.validation_points: list[str] = []

        def verify(self, point: str) -> None:
            self.validation_points.append(point)

        def binding(self) -> dict[str, object]:  # noqa: PLR6301 - guard protocol fixture.
            return {"sha256": "a" * 64, "package_archive_sha256": "b" * 64}

    guard = Guard()
    monkeypatch.setattr(module, "_load_runtime_api", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "_verify_imported_upstream_modules", lambda *_args: None)
    monkeypatch.setattr(module, "_create_runtime", lambda *_args, **_kwargs: object())

    def write_results(
        _cases: object, *, results_path: Path, **_kwargs: object
    ) -> list[dict[str, object]]:
        rows = [
            {"case_id": f"case-{index}", "status": "SUCCESS", "audio_finite": True}
            for index in range(CASE_COUNT)
        ]
        results_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return rows

    monkeypatch.setattr(module, "_write_generation_results", write_results)

    assert (
        module.generate(
            plan=plan,
            checkpoint_path=checkpoint,
            upstream_root=tmp_path,
            upstream_guard=guard,
            output_path=tmp_path / "output",
        )
        == 0
    )
    verification = json.loads(
        (tmp_path / "output" / "generation-verification.json").read_text(encoding="utf-8")
    )

    assert verification["upstream_runtime"] == guard.binding()
    assert verification["upstream_validation_points"] == [
        "before_import",
        "after_import",
        "base_model_before_load",
        "base_model_after_load",
        "after_generation",
    ]


def test_load_checkpoint_manifest_rejects_noncanonical_steps(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _manifest(tmp_path, steps=(1000, 1500, 2000, 3000))

    with pytest.raises(ValueError, match="checkpoint steps"):
        module.load_generation_plan(manifest)


def test_load_checkpoint_manifest_rejects_embedding_hash_drift(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"][0]["checkpoints"][0]["embedding_sha256"] = "9" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="embedding SHA-256 mismatch"):
        module.load_generation_plan(manifest)


def test_validate_base_checkpoint_binds_local_file_to_manifest(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _manifest(tmp_path)
    plan = module.load_generation_plan(manifest)
    base_checkpoint = tmp_path / "model.safetensors"
    base_checkpoint.write_bytes(b"official-base")

    with pytest.raises(ValueError, match="base checkpoint SHA-256 mismatch"):
        module.validate_base_checkpoint(base_checkpoint, plan=plan)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    actual_sha256 = _sha256(base_checkpoint)
    for checkpoint in payload["models"][0]["checkpoints"]:
        checkpoint["base_checkpoint_sha256"] = actual_sha256
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    corrected_plan = module.load_generation_plan(manifest)

    assert module.validate_base_checkpoint(base_checkpoint, plan=corrected_plan) == actual_sha256


def test_create_runtime_preserves_absolute_symlink_checkpoint_path(tmp_path: Path) -> None:
    module = _load_script()
    target = tmp_path / "blobs" / "suffixless-checkpoint"
    target.parent.mkdir()
    target.write_bytes(b"official-base")
    checkpoint = tmp_path / "snapshots" / "revision" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(target)
    captured: dict[str, object] = {}
    runtime = object()

    def fake_runtime_key(**values: object) -> dict[str, object]:
        captured.update(values)
        return values

    class FakeInferenceRuntime:
        @staticmethod
        def from_key(key: object) -> object:
            captured["runtime_key"] = key
            return runtime

    result = module._create_runtime(
        {
            "RuntimeKey": fake_runtime_key,
            "InferenceRuntime": FakeInferenceRuntime,
        },
        checkpoint_path=checkpoint,
    )

    assert result is runtime
    assert checkpoint.resolve() == target.resolve()
    assert not checkpoint.resolve().suffix
    assert captured["checkpoint"] == str(checkpoint.absolute())
    assert Path(str(captured["checkpoint"])).suffix == ".safetensors"


def test_generation_config_preserves_absolute_symlink_checkpoint_path(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _manifest(tmp_path)
    plan = module.load_generation_plan(manifest)
    target = tmp_path / "blobs" / "suffixless-checkpoint"
    target.parent.mkdir()
    target.write_bytes(b"official-base")
    checkpoint = tmp_path / "snapshots" / "revision" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(target)

    config = module._generation_config(
        plan=plan,
        checkpoint_path=checkpoint,
        cases=module.build_cases(plan),
    )

    assert config["base_checkpoint"] == str(checkpoint.absolute())
    assert Path(str(config["base_checkpoint"])).suffix == ".safetensors"


def test_result_row_matches_metrics_identity_contract(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _manifest(tmp_path)
    plan = module.load_generation_plan(manifest)
    case = module.build_cases(plan)[0]
    wav = tmp_path / "case.wav"
    wav.write_bytes(b"wav")

    row = module.build_success_row(
        case,
        plan=plan,
        wav_path=wav,
        elapsed_seconds=1.23456,
        audio_metadata={
            "sample_rate": 48000,
            "channels": 1,
            "sample_width": 2,
            "num_frames": 48000,
            "duration_seconds": 1.0,
            "audio_finite": True,
        },
    )

    assert row["status"] == "SUCCESS"
    assert row["case_id"] == case.case_id
    assert row["model_id"] == "kasumi"
    assert row["checkpoint_step"] == FIRST_CHECKPOINT_STEP
    assert row["checkpoint"] == "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
    assert row["speaker_filename"] == case.checkpoint.embedding_path.name
    assert row["embedding_path"] == str(case.checkpoint.embedding_path)
    assert row["embedding_sha256"] == case.checkpoint.embedding_sha256
    assert row["evaluation_manifest_sha256"] == _sha256(manifest)
    assert row["base_checkpoint_sha256"] == "b" * 64
    assert row["wav_path"] == str(wav.resolve())
    assert row["wav_sha256"] == _sha256(wav)
    assert row["elapsed_seconds"] == pytest.approx(EXPECTED_ELAPSED_SECONDS)
    assert row["provenance"] == {
        "training_config_sha256": "a" * 64,
        "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        "base_revision": "c" * 40,
        "run_id": "d" * 64,
    }
    evaluator = _load_external_script(
        Path("scripts/evaluate_600m_speaker_checkpoints.py"),
        "evaluate_600m_speaker_checkpoints_for_remote_generator_test",
    )
    evaluation_manifest = evaluator._load_evaluation_manifest(manifest)
    evaluator._validate_checkpoint_contract(
        {case.case_id: row},
        manifest=evaluation_manifest,
    )


def test_reserve_output_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "evaluation"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.reserve_output(output)


def test_validate_upstream_root_requires_inference_runtime(tmp_path: Path) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="upstream runtime is missing"):
        module.validate_upstream_root(tmp_path)

    runtime = tmp_path / "irodori_tts" / "inference_runtime.py"
    runtime.parent.mkdir()
    runtime.write_text("# fixture\n", encoding="utf-8")

    assert module.validate_upstream_root(tmp_path) == tmp_path.resolve()
