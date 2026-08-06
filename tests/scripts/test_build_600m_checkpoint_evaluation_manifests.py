# ruff: noqa: SLF001 - standalone script helpers are the compatibility-test API.
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/build_600m_checkpoint_evaluation_manifests.py")
MODEL_COUNT = 12
REFERENCE_COUNT = 25
SPEAKER_TOKEN_COUNT = 16
SPEAKER_EMBEDDING_DIM = 768
CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
BASE_CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
BASE_CHECKPOINT_BYTES = b"official 600m checkpoint"
BASE_CHECKPOINT_SHA256 = hashlib.sha256(BASE_CHECKPOINT_BYTES).hexdigest()
BASE_REVISION = "base-revision"
UPSTREAM_COMMIT = "b" * 40
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "ecapa-revision"
ECAPA_SOURCE_SHA256 = "c" * 64
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "whisper-revision"
WHISPER_SOURCE_SHA256 = "d" * 64


@dataclass(slots=True)
class Fixture:
    jobs_path: Path
    status_path: Path
    base_checkpoint_path: Path
    reference_paths: list[Path]
    output_dir: Path
    jobs: dict[str, object]
    status_rows: list[dict[str, object]]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_600m_checkpoint_evaluation_manifests",
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


def _canonical_sha256(row: dict[str, object]) -> str:
    serialized = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _write_safetensors(
    path: Path,
    *,
    shape: tuple[int, ...] = (SPEAKER_TOKEN_COUNT, SPEAKER_EMBEDDING_DIM),
    dtype: str = "F32",
    finite: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_dtype = np.dtype("<f4") if dtype == "F32" else np.dtype("<f2")
    values = np.ones(shape, dtype=numpy_dtype)
    if not finite:
        values.flat[0] = np.nan
    tensor = values.tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [0, len(tensor)],
            },
        },
        separators=(",", ":"),
    ).encode()
    padding = b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header) + len(padding)) + header + padding + tensor)


def _write_fixture(tmp_path: Path) -> Fixture:  # noqa: PLR0914
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    base_checkpoint_path = queue_dir / "models" / "base.safetensors"
    base_checkpoint_path.parent.mkdir()
    base_checkpoint_path.write_bytes(BASE_CHECKPOINT_BYTES)
    job_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    reference_paths: list[Path] = []
    model_ids = ["oop77_anabel_maidgarden_sp_451488a7c1"] + [
        f"model-{index:02d}" for index in range(1, MODEL_COUNT)
    ]
    for model_id in model_ids:
        clean_manifest = queue_dir / "datasets" / model_id / "clean-manifest.jsonl"
        config = queue_dir / "configs" / f"{model_id}.json"
        output_dir = queue_dir / "training" / model_id
        clean_manifest.parent.mkdir(parents=True)
        config.parent.mkdir(parents=True, exist_ok=True)
        clean_manifest.write_text('{"source_id":"source-1"}\n', encoding="utf-8")
        config.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
        job_rows.append(
            {
                "model_id": model_id,
                "clean_manifest": str(clean_manifest.relative_to(queue_dir)),
                "config": str(config.relative_to(queue_dir)),
                "output_dir": str(output_dir.relative_to(queue_dir)),
                "command": ["python", "train.py", "--model-id", model_id],
            },
        )
        candidates = []
        for step in CHECKPOINT_STEPS:
            embedding = output_dir / f"checkpoint-{step}" / f"{model_id}.speaker.safetensors"
            _write_safetensors(embedding)
            candidates.append({"path": str(embedding), "sha256": _sha256(embedding)})
        status_rows.append(
            {
                "event": "finished",
                "status": "success",
                "model_id": model_id,
                "clean_manifest_sha256": _sha256(clean_manifest),
                "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
                "checkpoint_revision": BASE_REVISION,
                "config_sha256": _sha256(config),
                "upstream_commit": UPSTREAM_COMMIT,
                "started_at": "2026-08-01T00:00:00+00:00",
                "ended_at": "2026-08-01T01:00:00+00:00",
                "exit_code": 0,
                "candidate_checkpoints": candidates,
                "last_checkpoint": candidates[-1]["path"],
                "last_checkpoint_sha256": candidates[-1]["sha256"],
                "error": None,
            },
        )
        reference_dir = tmp_path / "references" / model_id
        reference_dir.mkdir(parents=True)
        references = []
        for index in range(REFERENCE_COUNT):
            wav_path = reference_dir / f"reference-{index:02d}.wav"
            wav_path.write_bytes(f"reference:{model_id}:{index}".encode())
            references.append(
                {
                    "reference_wav_path": wav_path.name,
                    "reference_wav_sha256": _sha256(wav_path),
                    "source_id": f"{model_id}:{index:02d}",
                },
            )
        reference_path = reference_dir / "reference-wavs.json"
        reference_path.write_text(
            json.dumps(
                {
                    "schema_version": "speaker-similarity-references/v1",
                    "model_id": model_id,
                    "all_reference_wavs_finite": True,
                    "all_selected_source_hashes_verified": True,
                    "references": references,
                },
            ),
            encoding="utf-8",
        )
        reference_paths.append(reference_path)

    jobs: dict[str, object] = {
        "schema_version": "speaker-training-queue/v1",
        "created_at_utc": "2026-08-01T00:00:00+00:00",
        "queue_policy": {"execution": "sequential"},
        "base_checkpoint_path": str(base_checkpoint_path.relative_to(queue_dir)),
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "checkpoint_revision": BASE_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
        "anabel_strategy": {"selection": "required"},
        "jobs": job_rows,
    }
    jobs_path = queue_dir / "training-jobs.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    status_path = queue_dir / "training-status.jsonl"
    _write_status(status_path, status_rows)
    return Fixture(
        jobs_path=jobs_path,
        status_path=status_path,
        base_checkpoint_path=base_checkpoint_path,
        reference_paths=reference_paths,
        output_dir=tmp_path / "evaluation-manifests",
        jobs=jobs,
        status_rows=status_rows,
    )


def _write_status(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _argv(fixture: Fixture) -> list[str]:
    args = [
        "--training-status",
        str(fixture.status_path),
        "--training-jobs",
        str(fixture.jobs_path),
        "--output-dir",
        str(fixture.output_dir),
        "--base-checkpoint",
        BASE_CHECKPOINT,
        "--base-checkpoint-sha256",
        BASE_CHECKPOINT_SHA256,
        "--base-revision",
        BASE_REVISION,
        "--speaker-embedding-model-id",
        ECAPA_MODEL_ID,
        "--speaker-embedding-revision",
        ECAPA_REVISION,
        "--speaker-embedding-source-sha256",
        ECAPA_SOURCE_SHA256,
        "--transcription-model-id",
        WHISPER_MODEL_ID,
        "--transcription-revision",
        WHISPER_REVISION,
        "--transcription-source-sha256",
        WHISPER_SOURCE_SHA256,
    ]
    for path in fixture.reference_paths:
        args.extend(("--reference-wavs", str(path)))
    return args


def test_main_atomically_builds_twelve_per_model_manifests_and_index(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)

    assert module.main(_argv(fixture)) == 0

    index_path = fixture.output_dir / "manifest-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == "speaker-checkpoint-evaluation-manifest-index/v1"
    assert len(index["manifests"]) == MODEL_COUNT
    assert index["provenance"]["builder_script"]["sha256"] == _sha256(SCRIPT_PATH)
    assert index["provenance"]["jobs_contract"] == {
        "base_checkpoint_path": str(fixture.base_checkpoint_path.resolve()),
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "checkpoint_revision": BASE_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    assert index["provenance"]["inputs"] == {
        "training_jobs": {
            "path": str(fixture.jobs_path.resolve()),
            "sha256": _sha256(fixture.jobs_path),
        },
        "training_status": {
            "path": str(fixture.status_path.resolve()),
            "sha256": _sha256(fixture.status_path),
        },
        "reference_wavs": [
            {
                "model_id": json.loads(path.read_text(encoding="utf-8"))["model_id"],
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in fixture.reference_paths
        ],
    }
    first_status = fixture.status_rows[0]
    first_entry = index["manifests"][0]
    assert first_entry["model_id"] == first_status["model_id"]
    assert first_entry["provenance"] == {
        "clean_manifest_sha256": first_status["clean_manifest_sha256"],
        "config_sha256": first_status["config_sha256"],
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "checkpoint_revision": BASE_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
        "run_id": _canonical_sha256(first_status),
    }
    assert [row["checkpoint_step"] for row in first_entry["selected_candidates"]] == list(
        CHECKPOINT_STEPS,
    )
    manifest_path = fixture.output_dir / first_entry["manifest_path"]
    assert first_entry["manifest_sha256"] == _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
        "models": [
            {
                "model_id": first_status["model_id"],
                "checkpoints": [
                    {
                        **candidate,
                        "training_config_sha256": first_status["config_sha256"],
                        "base_checkpoint": BASE_CHECKPOINT,
                        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
                        "base_revision": BASE_REVISION,
                        "run_id": _canonical_sha256(first_status),
                    }
                    for candidate in first_entry["selected_candidates"]
                ],
            },
        ],
        "text_ids": [
            "word_unko",
            "word_chinko",
            "word_manko",
            "sentence_unko",
            "sentence_chinko",
            "sentence_manko",
            "control",
        ],
        "seeds": [1234, 5678],
        "styles": ["neutral", "calm"],
        "metrics_provenance": {
            "reference_wavs_sha256": _sha256(fixture.reference_paths[0]),
            "speaker_embedding": {
                "model_id": ECAPA_MODEL_ID,
                "revision": ECAPA_REVISION,
                "source_sha256": ECAPA_SOURCE_SHA256,
            },
            "transcription": {
                "model_id": WHISPER_MODEL_ID,
                "revision": WHISPER_REVISION,
                "source_sha256": WHISPER_SOURCE_SHA256,
            },
        },
    }
    assert not list(tmp_path.glob(".evaluation-manifests.tmp-*"))


def test_actual_reference_manifest_builds_outputs_consumable_by_downstream_loaders(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    module.main(_argv(fixture))
    index = json.loads(
        (fixture.output_dir / "manifest-index.json").read_text(encoding="utf-8"),
    )
    manifest_path = fixture.output_dir / index["manifests"][0]["manifest_path"]
    generator = _load_external_script(
        Path("scripts/generate_600m_checkpoint_audio_remote.py"),
        "generate_600m_checkpoint_audio_remote_for_builder_test",
    )
    evaluator = _load_external_script(
        Path("scripts/evaluate_600m_speaker_checkpoints.py"),
        "evaluate_600m_speaker_checkpoints_for_builder_test",
    )
    metrics = _load_external_script(
        Path("scripts/compute_600m_speaker_metrics.py"),
        "compute_600m_speaker_metrics_for_builder_test",
    )

    generation_plan = generator.load_generation_plan(manifest_path)
    evaluation_manifest = evaluator._load_evaluation_manifest(manifest_path)
    references = metrics.load_reference_wavs(fixture.reference_paths[0])

    assert [candidate.checkpoint_step for candidate in generation_plan.checkpoints] == list(
        CHECKPOINT_STEPS
    )
    assert sorted(step for _, step in evaluation_manifest.checkpoints) == list(CHECKPOINT_STEPS)
    assert references.keys() == {fixture.status_rows[0]["model_id"]}
    assert len(next(iter(references.values()))) == REFERENCE_COUNT


@pytest.mark.parametrize(
    "option",
    [
        "--base-checkpoint",
        "--base-revision",
        "--speaker-embedding-model-id",
        "--speaker-embedding-revision",
        "--transcription-model-id",
        "--transcription-revision",
    ],
)
def test_cli_rejects_empty_model_identity_fields_without_publishing(
    tmp_path: Path,
    option: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    argv = _argv(fixture)
    argv[argv.index(option) + 1] = ""

    with pytest.raises(ValueError, match="nonempty string"):
        module.main(argv)

    assert not fixture.output_dir.exists()


def test_atomic_publish_cleans_temporary_directory_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    original_write_json = module._write_json
    calls = 0

    def fail_after_first_write(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        original_write_json(path, payload)
        if calls == 1:
            raise OSError

    monkeypatch.setattr(module, "_write_json", fail_after_first_write)

    with pytest.raises(OSError):
        module.main(_argv(fixture))

    assert not fixture.output_dir.exists()
    assert not list(tmp_path.glob(".evaluation-manifests.tmp-*"))


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "failed",
        "unknown_finished_model",
        "missing_anabel",
        "duplicate_job",
        "clean_current",
        "config_current",
        "upstream_jobs",
        "jobs_base_sha",
        "jobs_revision",
        "base_checkpoint_file",
    ],
)
def test_status_and_jobs_contract_fail_without_publishing(
    tmp_path: Path,
    failure: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    if failure == "partial":
        fixture.status_rows.pop()
    elif failure == "failed":
        fixture.status_rows[0]["status"] = "failed"
    elif failure == "unknown_finished_model":
        fixture.status_rows.append({**fixture.status_rows[0], "model_id": "unknown-model"})
    elif failure == "missing_anabel":
        raw_jobs = fixture.jobs["jobs"]
        assert isinstance(raw_jobs, list)
        first_job = raw_jobs[0]
        assert isinstance(first_job, dict)
        first_job["model_id"] = "model-00"
        fixture.status_rows[0]["model_id"] = "model-00"
        reference_payload = json.loads(fixture.reference_paths[0].read_text(encoding="utf-8"))
        reference_payload["model_id"] = "model-00"
        fixture.reference_paths[0].write_text(
            json.dumps(reference_payload),
            encoding="utf-8",
        )
    elif failure == "duplicate_job":
        raw_jobs = fixture.jobs["jobs"]
        assert isinstance(raw_jobs, list)
        assert isinstance(raw_jobs[0], dict)
        assert isinstance(raw_jobs[1], dict)
        raw_jobs[1]["model_id"] = raw_jobs[0]["model_id"]
    elif failure == "clean_current":
        raw_jobs = fixture.jobs["jobs"]
        assert isinstance(raw_jobs, list)
        assert isinstance(raw_jobs[0], dict)
        clean = fixture.jobs_path.parent / str(raw_jobs[0]["clean_manifest"])
        clean.write_text("changed\n", encoding="utf-8")
    elif failure == "config_current":
        raw_jobs = fixture.jobs["jobs"]
        assert isinstance(raw_jobs, list)
        assert isinstance(raw_jobs[0], dict)
        config = fixture.jobs_path.parent / str(raw_jobs[0]["config"])
        config.write_text("changed\n", encoding="utf-8")
    elif failure == "upstream_jobs":
        fixture.jobs["upstream_commit"] = "different-upstream"
    elif failure == "jobs_base_sha":
        fixture.jobs["base_checkpoint_sha256"] = "0" * 64
    elif failure == "jobs_revision":
        fixture.jobs["checkpoint_revision"] = "different-revision"
    else:
        fixture.base_checkpoint_path.write_bytes(b"tampered")
    fixture.jobs_path.write_text(json.dumps(fixture.jobs), encoding="utf-8")
    _write_status(fixture.status_path, fixture.status_rows)

    with pytest.raises((TypeError, ValueError)):
        module.main(_argv(fixture))

    assert not fixture.output_dir.exists()


@pytest.mark.parametrize(
    "field",
    [
        "clean_manifest_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_revision",
        "upstream_commit",
    ],
)
def test_each_status_provenance_field_requires_a_current_success(
    tmp_path: Path,
    field: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    fixture.status_rows[0][field] = "mismatch"
    _write_status(fixture.status_path, fixture.status_rows)

    with pytest.raises(ValueError, match="reusable successful finished status"):
        module.main(_argv(fixture))

    assert not fixture.output_dir.exists()


def test_append_only_status_selects_last_current_success_and_ignores_other_attempts(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    current_success = fixture.status_rows[0]
    past_failure = dict(current_success)
    past_failure.update(
        status="failed",
        exit_code=1,
        ended_at="2026-08-01T00:30:00+00:00",
        error="transient failure",
    )
    stale_success = dict(current_success)
    stale_success.update(
        config_sha256="0" * 64,
        ended_at="2026-08-01T00:45:00+00:00",
    )
    latest_success = dict(current_success)
    latest_success.update(
        started_at="2026-08-02T00:00:00+00:00",
        ended_at="2026-08-02T01:00:00+00:00",
    )
    final_failure = dict(latest_success)
    final_failure.update(
        status="failed",
        exit_code=1,
        ended_at="2026-08-03T01:00:00+00:00",
        error="later transient failure",
    )
    fixture.status_rows = [
        past_failure,
        stale_success,
        current_success,
        latest_success,
        final_failure,
        *fixture.status_rows[1:],
    ]
    _write_status(fixture.status_path, fixture.status_rows)

    assert module.main(_argv(fixture)) == 0

    index = json.loads(
        (fixture.output_dir / "manifest-index.json").read_text(encoding="utf-8"),
    )
    first_entry = index["manifests"][0]
    assert first_entry["provenance"]["run_id"] == _canonical_sha256(latest_success)


def test_seeded_existing_run_uses_declared_run_provenance_sha256_as_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    seeded_status = fixture.status_rows[0]
    run_provenance_path = tmp_path / "run-provenance.json"
    run_provenance_path.write_text(
        json.dumps({"model_id": seeded_status["model_id"]}),
        encoding="utf-8",
    )
    run_provenance_sha256 = _sha256(run_provenance_path)
    seeded_status["seeded_existing_run"] = {
        "run_provenance_path": str(run_provenance_path.resolve()),
        "run_provenance_sha256": run_provenance_sha256,
    }
    _write_status(fixture.status_path, fixture.status_rows)
    original_open = Path.open
    provenance_open_count = 0

    def count_provenance_open(
        path: Path,
        mode: str = "r",
        *,
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        nonlocal provenance_open_count
        if path == run_provenance_path:
            provenance_open_count += 1
        return original_open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", count_provenance_open)

    assert module.main(_argv(fixture)) == 0
    assert provenance_open_count == 1

    index = json.loads(
        (fixture.output_dir / "manifest-index.json").read_text(encoding="utf-8"),
    )
    run_id = index["manifests"][0]["provenance"]["run_id"]
    assert run_id == run_provenance_sha256
    assert run_id != _canonical_sha256(seeded_status)
    manifest_path = fixture.output_dir / index["manifests"][0]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all(
        checkpoint["run_id"] == run_provenance_sha256
        for checkpoint in manifest["models"][0]["checkpoints"]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "nested_object",
        "missing_path",
        "relative_path",
        "noncanonical_path",
        "symlink",
        "missing_file",
        "sha256_format",
        "sha256_mismatch",
        "provenance_json",
        "missing_model_id",
        "model_id_mismatch",
    ],
)
def test_seeded_existing_run_rejects_invalid_run_provenance_without_publishing(
    tmp_path: Path,
    failure: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    seeded_status = fixture.status_rows[0]
    run_provenance_path = tmp_path / "run-provenance.json"
    run_provenance_path.write_text(
        json.dumps({"model_id": seeded_status["model_id"]}),
        encoding="utf-8",
    )
    seeded_existing_run: dict[str, object] = {
        "run_provenance_path": str(run_provenance_path.resolve()),
        "run_provenance_sha256": _sha256(run_provenance_path),
    }
    seeded_status["seeded_existing_run"] = seeded_existing_run
    if failure == "nested_object":
        seeded_status["seeded_existing_run"] = "invalid"
    elif failure == "missing_path":
        seeded_existing_run.pop("run_provenance_path")
    elif failure == "relative_path":
        seeded_existing_run["run_provenance_path"] = run_provenance_path.name
    elif failure == "noncanonical_path":
        intermediate = tmp_path / "intermediate"
        intermediate.mkdir()
        seeded_existing_run["run_provenance_path"] = str(
            intermediate / ".." / run_provenance_path.name,
        )
    elif failure == "symlink":
        symlink = tmp_path / "run-provenance-link.json"
        symlink.symlink_to(run_provenance_path)
        seeded_existing_run["run_provenance_path"] = str(symlink)
    elif failure == "missing_file":
        seeded_existing_run["run_provenance_path"] = str(
            (tmp_path / "missing-run-provenance.json").resolve(),
        )
    elif failure == "sha256_format":
        seeded_existing_run["run_provenance_sha256"] = "invalid"
    elif failure == "sha256_mismatch":
        seeded_existing_run["run_provenance_sha256"] = "0" * 64
    elif failure == "provenance_json":
        run_provenance_path.write_text("[]", encoding="utf-8")
        seeded_existing_run["run_provenance_sha256"] = _sha256(run_provenance_path)
    elif failure == "missing_model_id":
        run_provenance_path.write_text("{}", encoding="utf-8")
        seeded_existing_run["run_provenance_sha256"] = _sha256(run_provenance_path)
    else:
        run_provenance_path.write_text(
            json.dumps({"model_id": "different-model"}),
            encoding="utf-8",
        )
        seeded_existing_run["run_provenance_sha256"] = _sha256(run_provenance_path)
    _write_status(fixture.status_path, fixture.status_rows)

    with pytest.raises((TypeError, ValueError)):
        module.main(_argv(fixture))

    assert not fixture.output_dir.exists()


@pytest.mark.parametrize(
    "failure",
    ["missing_step", "missing_file", "hash", "shape", "dtype", "finite"],
)
def test_checkpoint_contract_fails_without_publishing(tmp_path: Path, failure: str) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    candidates = fixture.status_rows[0]["candidate_checkpoints"]
    assert isinstance(candidates, list)
    target = candidates[-1]
    assert isinstance(target, dict)
    target_path = Path(str(target["path"]))
    if failure == "missing_step":
        candidates.pop()
    elif failure == "missing_file":
        target_path.unlink()
    elif failure == "hash":
        target["sha256"] = "0" * 64
    elif failure == "shape":
        _write_safetensors(target_path, shape=(15, SPEAKER_EMBEDDING_DIM))
        target["sha256"] = _sha256(target_path)
    elif failure == "dtype":
        _write_safetensors(target_path, dtype="F16")
        target["sha256"] = _sha256(target_path)
    else:
        _write_safetensors(target_path, finite=False)
        target["sha256"] = _sha256(target_path)
    _write_status(fixture.status_path, fixture.status_rows)

    with pytest.raises((TypeError, ValueError)):
        module.main(_argv(fixture))

    assert not fixture.output_dir.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "schema",
        "missing",
        "hash",
        "finite_flag",
        "source_hash_flag",
        "model_id",
        "count",
        "duplicate",
        "extra",
    ],
)
def test_reference_manifest_contract_fails_without_publishing(
    tmp_path: Path,
    failure: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    target = fixture.reference_paths[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    if failure == "schema":
        payload["schema_version"] = "speaker-reference-wavs/v1"
        target.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "missing":
        target.unlink()
    elif failure == "hash":
        referenced = target.parent / payload["references"][0]["reference_wav_path"]
        referenced.write_bytes(b"changed")
    elif failure == "finite_flag":
        payload["all_reference_wavs_finite"] = False
        target.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "source_hash_flag":
        payload["all_selected_source_hashes_verified"] = False
        target.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "model_id":
        payload["model_id"] = "unknown"
        target.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "count":
        payload["references"].pop()
        target.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "duplicate":
        fixture.reference_paths.append(fixture.reference_paths[0])
    else:
        extra = tmp_path / "extra-reference-wavs.json"
        extra.write_text(
            json.dumps({**payload, "model_id": "extra-model"}),
            encoding="utf-8",
        )
        fixture.reference_paths.append(extra)

    with pytest.raises((OSError, TypeError, ValueError)):
        module.main(_argv(fixture))

    assert not fixture.output_dir.exists()
