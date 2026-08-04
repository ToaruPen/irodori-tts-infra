from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType
    from typing import Protocol

    class TrainingJobLike(Protocol):
        model_id: str
        clean_manifest: Path
        config: Path
        output_dir: Path
        command: tuple[str, ...]

    class QueueProvenanceLike(Protocol):
        checkpoint: Path
        checkpoint_revision: str
        upstream_commit: str


pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/run_600m_speaker_training_queue.py")
MODEL_COUNT = 12
SPEAKER_TOKEN_COUNT = 16
SPEAKER_EMBEDDING_DIM = 768
SHA256_HEX_LENGTH = 64
FAILED_EXIT_CODE = 9


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_600m_speaker_training_queue", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    payload = values.tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    padding = b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header) + len(padding)) + header + padding + payload)


def _make_jobs(tmp_path: Path, *, count: int = MODEL_COUNT) -> tuple[TrainingJobLike, ...]:
    module = _load_script()
    jobs = []
    for index in range(count):
        model_id = f"model-{index:02d}"
        manifest = tmp_path / "datasets" / model_id / "clean-manifest.jsonl"
        config = tmp_path / "configs" / f"{model_id}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        config.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(f'{{"source_id":"{model_id}:1"}}\n', encoding="utf-8")
        config.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
        jobs.append(
            module.TrainingJob(
                model_id=model_id,
                clean_manifest=manifest,
                config=config,
                output_dir=tmp_path / "training" / model_id,
                command=("python", "train.py", "--model-id", model_id),
            )
        )
    return tuple(jobs)


def _provenance(tmp_path: Path) -> QueueProvenanceLike:
    module = _load_script()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"600m checkpoint")
    return cast(
        "QueueProvenanceLike",
        module.QueueProvenance(
            checkpoint=checkpoint,
            checkpoint_revision="e863a3a93e652e09afeff3e84823a206a0a60314",
            upstream_commit="eaf74d6a19138f743acb5b71a445fd25a57db987",
        ),
    )


def _clock() -> Iterator[str]:
    counter = 0
    while True:
        counter += 1
        yield f"2026-07-31T12:00:{counter:02d}+09:00"


def _status_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_load_training_jobs_resolves_twelve_manifest_rows_in_order(tmp_path: Path) -> None:
    module = _load_script()
    manifest_path = tmp_path / "queue" / "jobs.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "model_id": f"model-{index:02d}",
                        "clean_manifest": f"datasets/model-{index:02d}.jsonl",
                        "config": f"configs/model-{index:02d}.json",
                        "output_dir": f"training/model-{index:02d}",
                        "command": ["python", "train.py", "--model-id", f"model-{index:02d}"],
                    }
                    for index in range(MODEL_COUNT)
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = module.load_training_jobs(manifest_path)

    assert [job.model_id for job in jobs] == [f"model-{index:02d}" for index in range(MODEL_COUNT)]
    assert jobs[0].clean_manifest == manifest_path.parent / "datasets/model-00.jsonl"
    assert jobs[-1].output_dir == manifest_path.parent / "training/model-11"


def test_queue_runs_twelve_jobs_serially_and_flushes_provenance(tmp_path: Path) -> None:
    module = _load_script()
    jobs = _make_jobs(tmp_path)
    provenance = _provenance(tmp_path)
    status_path = tmp_path / "training-status.jsonl"
    calls: list[str] = []

    def runner(command: tuple[str, ...], log_path: Path) -> int:
        model_id = command[-1]
        rows = _status_rows(status_path)
        assert rows[-1]["event"] == "started"
        assert rows[-1]["model_id"] == model_id
        assert rows[-1]["ended_at"] is None
        assert rows[-1]["exit_code"] is None
        assert rows[-1]["log_path"] == str(log_path)
        calls.append(model_id)
        job = next(job for job in jobs if job.model_id == model_id)
        _write_safetensors(job.output_dir / "checkpoint-250" / f"{model_id}.speaker.safetensors")
        return 0

    timestamps = _clock()
    result = module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=runner,
        now=lambda: next(timestamps),
    )

    assert calls == [job.model_id for job in jobs]
    assert result.succeeded == tuple(calls)
    assert result.failed == ()
    assert result.skipped == ()
    rows = _status_rows(status_path)
    assert len(rows) == MODEL_COUNT * 2
    finished = rows[-1]
    assert finished["event"] == "finished"
    assert finished["status"] == "success"
    assert finished["model_id"] == calls[-1]
    assert finished["clean_manifest_sha256"] == module.sha256_file(jobs[-1].clean_manifest)
    assert finished["checkpoint_sha256"] == module.sha256_file(provenance.checkpoint)
    assert finished["checkpoint_revision"] == provenance.checkpoint_revision
    assert finished["config_sha256"] == module.sha256_file(jobs[-1].config)
    assert finished["upstream_commit"] == provenance.upstream_commit
    assert finished["started_at"] is not None
    assert finished["ended_at"] is not None
    assert finished["exit_code"] == 0
    assert str(finished["last_checkpoint"]).endswith("model-11.speaker.safetensors")
    assert len(str(finished["last_checkpoint_sha256"])) == SHA256_HEX_LENGTH


@pytest.mark.parametrize(
    "changed_field",
    [
        "clean_manifest",
        "checkpoint",
        "checkpoint_revision",
        "config",
        "upstream_commit",
    ],
)
def test_queue_skips_only_when_all_success_provenance_matches(
    tmp_path: Path,
    changed_field: str,
) -> None:
    module = _load_script()
    jobs = _make_jobs(tmp_path, count=1)
    provenance = _provenance(tmp_path)
    status_path = tmp_path / "training-status.jsonl"

    def successful_runner(_command: tuple[str, ...], _log_path: Path) -> int:
        _write_safetensors(jobs[0].output_dir / "checkpoint-250" / "model-00.speaker.safetensors")
        return 0

    timestamps = _clock()
    module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=successful_runner,
        now=lambda: next(timestamps),
    )

    skipped = module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=lambda _command, _log_path: pytest.fail("matching job must be skipped"),
        now=lambda: next(timestamps),
    )
    assert skipped.skipped == ("model-00",)

    if changed_field == "clean_manifest":
        jobs[0].clean_manifest.write_text("changed manifest\n", encoding="utf-8")
    elif changed_field == "checkpoint":
        provenance.checkpoint.write_bytes(b"changed checkpoint")
    elif changed_field == "checkpoint_revision":
        provenance = module.QueueProvenance(
            checkpoint=provenance.checkpoint,
            checkpoint_revision="different-revision",
            upstream_commit=provenance.upstream_commit,
        )
    elif changed_field == "config":
        jobs[0].config.write_text("changed config\n", encoding="utf-8")
    else:
        provenance = module.QueueProvenance(
            checkpoint=provenance.checkpoint,
            checkpoint_revision=provenance.checkpoint_revision,
            upstream_commit="different-upstream-commit",
        )

    calls = 0

    def rerun(_command: tuple[str, ...], _log_path: Path) -> int:
        nonlocal calls
        calls += 1
        return 0

    rerun_result = module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=rerun,
        now=lambda: next(timestamps),
    )

    assert calls == 1
    assert rerun_result.succeeded == ("model-00",)


def test_queue_continues_after_model_failure_without_marking_it_successful(tmp_path: Path) -> None:
    module = _load_script()
    jobs = _make_jobs(tmp_path, count=3)
    provenance = _provenance(tmp_path)
    status_path = tmp_path / "training-status.jsonl"
    calls: list[str] = []

    def runner(command: tuple[str, ...], _log_path: Path) -> int:
        model_id = command[-1]
        calls.append(model_id)
        if model_id == "model-01":
            return FAILED_EXIT_CODE
        job = next(job for job in jobs if job.model_id == model_id)
        _write_safetensors(job.output_dir / f"{model_id}.speaker.safetensors")
        return 0

    timestamps = _clock()
    result = module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=runner,
        now=lambda: next(timestamps),
    )

    assert calls == ["model-00", "model-01", "model-02"]
    assert result.succeeded == ("model-00", "model-02")
    assert result.failed == ("model-01",)
    finished = [row for row in _status_rows(status_path) if row["event"] == "finished"]
    by_model = {str(row["model_id"]): row for row in finished}
    assert by_model["model-01"]["status"] == "failed"
    assert by_model["model-01"]["exit_code"] == FAILED_EXIT_CODE
    assert by_model["model-01"]["last_checkpoint"] is None


@pytest.mark.parametrize(
    ("shape", "dtype", "finite", "expected_message"),
    [
        ((15, SPEAKER_EMBEDDING_DIM), "F32", True, "shape"),
        ((SPEAKER_TOKEN_COUNT, SPEAKER_EMBEDDING_DIM), "F16", True, "float32"),
        ((SPEAKER_TOKEN_COUNT, SPEAKER_EMBEDDING_DIM), "F32", False, "finite"),
    ],
)
def test_queue_rejects_invalid_embedding_payload_and_continues(
    tmp_path: Path,
    shape: tuple[int, ...],
    dtype: str,
    finite: bool,  # noqa: FBT001 - boolean is the parameterized safetensors property.
    expected_message: str,
) -> None:
    module = _load_script()
    jobs = _make_jobs(tmp_path, count=2)
    provenance = _provenance(tmp_path)
    status_path = tmp_path / "training-status.jsonl"

    def runner(command: tuple[str, ...], _log_path: Path) -> int:
        model_id = command[-1]
        job = next(job for job in jobs if job.model_id == model_id)
        if model_id == "model-00":
            _write_safetensors(
                job.output_dir / f"{model_id}.speaker.safetensors",
                shape=shape,
                dtype=dtype,
                finite=finite,
            )
        else:
            _write_safetensors(job.output_dir / f"{model_id}.speaker.safetensors")
        return 0

    timestamps = _clock()
    result = module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=runner,
        now=lambda: next(timestamps),
    )

    assert result.failed == ("model-00",)
    assert result.succeeded == ("model-01",)
    finished = [row for row in _status_rows(status_path) if row["event"] == "finished"]
    assert expected_message in str(finished[0]["error"])


def test_validate_embedding_accepts_missing_safetensors_metadata(tmp_path: Path) -> None:
    module = _load_script()
    embedding = tmp_path / "speaker.speaker.safetensors"
    _write_safetensors(embedding)

    result = module.validate_speaker_embedding(embedding)

    assert result.path == embedding
    assert result.sha256 == module.sha256_file(embedding)


def test_queue_does_not_skip_when_an_earlier_recorded_checkpoint_changed(tmp_path: Path) -> None:
    module = _load_script()
    jobs = _make_jobs(tmp_path, count=1)
    provenance = _provenance(tmp_path)
    status_path = tmp_path / "training-status.jsonl"
    first = jobs[0].output_dir / "checkpoint-250" / "model-00.speaker.safetensors"
    last = jobs[0].output_dir / "checkpoint-500" / "model-00.speaker.safetensors"

    def first_runner(_command: tuple[str, ...], _log_path: Path) -> int:
        _write_safetensors(first)
        _write_safetensors(last)
        return 0

    timestamps = _clock()
    module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=first_runner,
        now=lambda: next(timestamps),
    )
    first.write_bytes(b"changed")
    calls = 0

    def rerun(_command: tuple[str, ...], _log_path: Path) -> int:
        nonlocal calls
        calls += 1
        return 7

    result = module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=rerun,
        now=lambda: next(timestamps),
    )

    assert calls == 1
    assert result.skipped == ()
    assert result.failed == ("model-00",)


def test_dry_run_does_not_execute_or_write_status(tmp_path: Path) -> None:
    module = _load_script()
    jobs = _make_jobs(tmp_path, count=2)
    provenance = _provenance(tmp_path)
    status_path = tmp_path / "training-status.jsonl"

    result = module.run_training_queue(
        jobs,
        provenance=provenance,
        status_path=status_path,
        runner=lambda _command, _log_path: pytest.fail("dry-run must not execute"),
        dry_run=True,
    )

    assert result.planned == ("model-00", "model-01")
    assert not status_path.exists()
