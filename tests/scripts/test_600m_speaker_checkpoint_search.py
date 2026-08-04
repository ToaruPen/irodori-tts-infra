# ruff: noqa: ARG001, ARG005, FBT001, PLR2004, PT007, PT018, RUF043, RUF069, SLF001
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pytestmark = pytest.mark.unit

BUILDER = Path("scripts/build_600m_speaker_checkpoint_search_manifest.py")
GENERATOR = Path("scripts/generate_600m_speaker_checkpoint_search_remote.py")
EVALUATOR = Path("scripts/evaluate_600m_speaker_checkpoint_search.py")
MODEL_ID = "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
RUN_ID = "search-seed-2-lr-0.0001"
SEARCH_CASE_SCHEMA = "speaker-checkpoint-search-generation-case/v1"
SEARCH_EVALUATION_CASE_SCHEMA = "speaker-checkpoint-search-evaluation-case/v1"
SEARCH_RUN_EVIDENCE_SCHEMA = "speaker-quality-search-run-evidence/v1"
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
PRODUCTION_HASHES = {
    "scripts/build_600m_checkpoint_evaluation_manifests.py": (
        "e3f62e07f07c949fe60d4db00a7eef11dbd1ae9111a7628dd88982a2702d0e93"
    ),
    "scripts/generate_600m_checkpoint_audio_remote.py": (
        "947babd074d83b08c2c9a535f9d718cdac17a3cbfe845e430758bc1008818816"
    ),
    "scripts/evaluate_600m_speaker_checkpoints.py": (
        "9f330552c018027522457bb171aa34599e35bbb303e6c0792e2396fe266e0900"
    ),
}


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_embedding(
    path: Path,
    *,
    dtype: str = "F32",
    finite: bool = True,
) -> None:
    numpy_dtype = np.dtype("<f4") if dtype == "F32" else np.dtype("<f2")
    values = np.ones((16, 768), dtype=numpy_dtype)
    if not finite:
        values.flat[0] = np.nan
    payload = values.tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": dtype,
                "shape": [16, 768],
                "data_offsets": [0, len(payload)],
            },
        },
        separators=(",", ":"),
    ).encode()
    padding = b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header) + len(padding)) + header + padding + payload)


def _source_manifest(tmp_path: Path) -> Path:
    base_checkpoint = tmp_path / "base-checkpoint.safetensors"
    base_checkpoint.write_bytes(b"search base checkpoint")
    checkpoints = []
    for step in range(250, 3001, 250):
        embedding = tmp_path / f"source-{step}.speaker.safetensors"
        _write_embedding(embedding)
        checkpoints.append(
            {
                "checkpoint_step": step,
                "embedding_path": str(embedding),
                "embedding_sha256": _sha(embedding),
                "training_config_sha256": "a" * 64,
                "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                "base_checkpoint_sha256": _sha(base_checkpoint),
                "base_revision": "base-revision",
                "run_id": "source-run",
            }
        )
    path = tmp_path / "source-evaluation-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
                "models": [{"model_id": MODEL_ID, "checkpoints": checkpoints}],
                "text_ids": list(TEXT_IDS),
                "seeds": list(SEEDS),
                "styles": list(STYLES),
                "metrics_provenance": {
                    "reference_wavs_sha256": "c" * 64,
                    "speaker_embedding": {
                        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                        "revision": "ecapa-revision",
                        "source_sha256": "d" * 64,
                    },
                    "transcription": {
                        "model_id": "openai/whisper-large-v3-turbo",
                        "revision": "whisper-revision",
                        "source_sha256": "e" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _build_fixture(tmp_path: Path) -> tuple[ModuleType, Path, Path, Path]:
    module = _load(BUILDER, "speaker_search_builder_test")
    source = _source_manifest(tmp_path)
    run_root = tmp_path / RUN_ID
    output_dir = run_root / "outputs"
    output_dir.mkdir(parents=True)
    embedding = output_dir / "checkpoint_0000250.speaker.safetensors"
    _write_embedding(embedding)
    clean_manifest = tmp_path / "clean-manifest.jsonl"
    clean_manifest.write_text('{"source_id":"sample"}\n', encoding="utf-8")
    config = run_root / "training-config.json"
    config.write_text(
        json.dumps(
            {
                "train": {
                    "manifest_path": str(clean_manifest.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "max_steps": 250,
                    "save_every": 250,
                    "log_every": 20,
                }
            }
        ),
        encoding="utf-8",
    )
    return module, source, embedding, config


def _write_run_evidence(  # noqa: PLR0914 - mirrors the real run-evidence artifact.
    tmp_path: Path,
    *,
    embedding: Path,
    config: Path,
    step: int = 250,
) -> Path:
    del tmp_path
    run_root = config.parent
    output_dir = embedding.parent
    final = output_dir / "checkpoint_final.speaker.safetensors"
    final.write_bytes(embedding.read_bytes())
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    clean_manifest = Path(config_payload["train"]["manifest_path"])
    base_checkpoint = run_root.parent / "base-checkpoint.safetensors"
    base_sha = _sha(base_checkpoint)
    jobs = run_root / "search-jobs.json"
    job_rows = [
        {
            "model_id": MODEL_ID,
            "clean_manifest": str(clean_manifest.resolve()),
            "config": str(config.resolve()),
            "output_dir": str(output_dir.resolve()),
            "command": ["python", "train.py", "--config", str(config.resolve())],
        }
    ]
    job_rows.extend(
        {
            "model_id": f"other-model-{index:02d}",
            "clean_manifest": str(clean_manifest.resolve()),
            "config": str(run_root / f"other-config-{index:02d}.json"),
            "output_dir": str(run_root / f"other-output-{index:02d}"),
            "command": [
                "python",
                "train.py",
                "--config",
                str(run_root / f"other-config-{index:02d}.json"),
            ],
        }
        for index in range(11)
    )
    jobs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": "2026-08-02T03:00:00+00:00",
                "base_checkpoint_path": str(base_checkpoint.resolve()),
                "base_checkpoint_sha256": base_sha,
                "checkpoint_revision": "base-revision",
                "upstream_commit": "upstream-commit",
                "queue_policy": "serial_one_at_a_time",
                "anabel_strategy": "reuse_existing_fresh_3000_run",
                "jobs": job_rows,
            }
        ),
        encoding="utf-8",
    )
    prefix = (
        json.dumps(
            {"event": "finished", "status": "success", "model_id": "historical-model"},
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    status = run_root / "search-status.jsonl"
    common_status: dict[str, Any] = {
        "model_id": MODEL_ID,
        "clean_manifest_sha256": _sha(clean_manifest),
        "config_sha256": _sha(config),
        "checkpoint_sha256": base_sha,
        "checkpoint_revision": "base-revision",
        "upstream_commit": "upstream-commit",
        "started_at": "2026-08-02T03:01:00+00:00",
        "log_path": str((run_root / "training.log").resolve()),
    }
    started = common_status | {
        "event": "started",
        "status": "running",
        "ended_at": None,
        "exit_code": None,
        "last_checkpoint": None,
        "last_checkpoint_sha256": None,
        "candidate_checkpoints": [],
        "error": None,
    }
    checkpoint_bindings = [
        {"path": str(embedding.resolve()), "sha256": _sha(embedding)},
        {"path": str(final.resolve()), "sha256": _sha(final)},
    ]
    finished = common_status | {
        "event": "finished",
        "status": "success",
        "ended_at": "2026-08-02T03:05:00+00:00",
        "exit_code": 0,
        "last_checkpoint": str(embedding.resolve()),
        "last_checkpoint_sha256": _sha(embedding),
        "candidate_checkpoints": checkpoint_bindings,
        "error": None,
    }
    status.write_bytes(
        prefix
        + json.dumps(started, sort_keys=True).encode()
        + b"\n"
        + json.dumps(finished, sort_keys=True).encode()
        + b"\n"
    )
    setup = run_root / "setup-evidence.json"
    setup.write_text(json.dumps({"model_id": MODEL_ID}), encoding="utf-8")
    queue_script = run_root / "run_600m_speaker_training_queue.py"
    queue_script.write_text("# queue fixture\n", encoding="utf-8")
    log = run_root / "training.log"
    loss_steps = list(range(20, 250, 20))
    log.write_text(
        "".join(f"step={loss_step} loss=0.1\n" for loss_step in loss_steps)
        + "Training finished at step=250.\n",
        encoding="utf-8",
    )
    evidence = run_root / "search-run-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": SEARCH_RUN_EVIDENCE_SCHEMA,
                "created_at": "2026-08-02T03:06:00+00:00",
                "state": "finished",
                "model_id": MODEL_ID,
                "queue_exit_code": 0,
                "setup_evidence": {"path": str(setup.resolve()), "sha256": _sha(setup)},
                "training_jobs": {"path": str(jobs.resolve()), "sha256": _sha(jobs)},
                "training_status": {
                    "path": str(status.resolve()),
                    "before_row_count": 1,
                    "before_sha256": hashlib.sha256(prefix).hexdigest(),
                    "after_row_count": 3,
                    "after_sha256": _sha(status),
                    "new_status_row_count": 2,
                    "new_started_model_ids": [MODEL_ID],
                    "new_finished_success_model_ids": [MODEL_ID],
                },
                "queue_script": {
                    "path": str(queue_script.resolve()),
                    "sha256": _sha(queue_script),
                },
                "invocation": {
                    "recipe": "speaker-quality-search",
                    "checkpoint_revision": "base-revision",
                    "upstream_commit": "upstream-commit",
                },
                "run": {
                    "started_at": started["started_at"],
                    "ended_at": finished["ended_at"],
                    "config_path": str(config.resolve()),
                    "config_sha256": _sha(config),
                    "clean_manifest_sha256": _sha(clean_manifest),
                    "base_checkpoint_sha256": base_sha,
                    "candidate_checkpoint_count": 2,
                    "checkpoints": [
                        {
                            "name": embedding.name,
                            "path": str(embedding.resolve()),
                            "sha256": _sha(embedding),
                        },
                        {
                            "name": final.name,
                            "path": str(final.resolve()),
                            "sha256": _sha(final),
                        },
                    ],
                    "final_equals_step250": step == 250,
                    "log": {
                        "path": str(log.resolve()),
                        "sha256": _sha(log),
                        "loss_event_count": len(loss_steps),
                        "loss_steps": loss_steps,
                        "loss_all_finite": True,
                        "last_loss": 0.1,
                        "oom": False,
                        "traceback": False,
                    },
                },
                "runtime_after": {
                    "gpu_memory_used_mib": 900.0,
                    "gpu_memory_total_mib": 12282.0,
                    "gpu_utilization_percent": 0.0,
                    "gpu_power_watts": 12.0,
                    "active_training_processes": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return evidence


def _rewrite_evidence_status(
    evidence: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = cast("dict[str, Any]", json.loads(evidence.read_text(encoding="utf-8")))
    status = Path(payload["training_status"]["path"])
    status.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload["training_status"]["after_row_count"] = len(rows)
    payload["training_status"]["after_sha256"] = _sha(status)
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _evidence_status_rows(evidence: Path) -> list[dict[str, Any]]:
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    status = Path(payload["training_status"]["path"])
    return [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines()]


def _rebind_evidence_config_sha(evidence: Path, config: Path) -> None:
    config_sha = _sha(config)
    rows = _evidence_status_rows(evidence)
    for row in rows[-2:]:
        row["config_sha256"] = config_sha
    payload = _rewrite_evidence_status(evidence, rows)
    payload["run"]["config_sha256"] = config_sha
    evidence.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_evidence_log(evidence: Path, text: str) -> dict[str, Any]:
    payload = cast("dict[str, Any]", json.loads(evidence.read_text(encoding="utf-8")))
    log = Path(payload["run"]["log"]["path"])
    log.write_text(text, encoding="utf-8")
    payload["run"]["log"]["sha256"] = _sha(log)
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _build(
    module: ModuleType,
    source: Path,
    embedding: Path,
    config: Path,
    output: Path,
) -> dict[str, Any]:
    evidence = _write_run_evidence(
        output.parent,
        embedding=embedding,
        config=config,
    )
    return cast(
        "dict[str, Any]",
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=output,
        ),
    )


def _rewrite_bound_search_training_config(
    manifest: Path,
    config: Path,
    *,
    max_steps: int | bool,
) -> None:
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload["train"]["max_steps"] = max_steps
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    config_sha = _sha(config)

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    evidence = Path(manifest_payload["training_run_evidence"]["path"])
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    status = Path(evidence_payload["training_status"]["path"])
    status_rows = [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines()]
    for row in status_rows[-2:]:
        row["config_sha256"] = config_sha
    status.write_text(
        "".join(json.dumps(row) + "\n" for row in status_rows),
        encoding="utf-8",
    )
    evidence_payload["training_status"]["after_sha256"] = _sha(status)
    evidence_payload["run"]["config_sha256"] = config_sha
    evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")

    manifest_payload["checkpoint"]["training_config_sha256"] = config_sha
    manifest_payload["training_run_evidence"]["sha256"] = _sha(evidence)
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")


def test_builder_creates_dedicated_one_checkpoint_manifest(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    output = tmp_path / "search-manifest.json"

    payload = _build(module, source, embedding, config, output)

    assert payload["schema_version"] == "speaker-checkpoint-search-manifest/v1"
    assert payload["checkpoint"]["checkpoint_step"] == 250
    assert payload["source_evaluation_manifest"] == {
        "path": str(source.resolve()),
        "sha256": _sha(source),
    }
    assert payload["training_run_evidence"] == {
        "path": str((tmp_path / RUN_ID / "search-run-evidence.json").resolve()),
        "sha256": _sha(tmp_path / RUN_ID / "search-run-evidence.json"),
    }
    assert output.exists()


def test_builder_accepts_sample_equivalent_actual_run_evidence(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    jobs = json.loads(Path(evidence_payload["training_jobs"]["path"]).read_text(encoding="utf-8"))

    payload = module.build_search_manifest(
        source_manifest=source,
        source_manifest_sha256=_sha(source),
        embedding=embedding,
        embedding_sha256=_sha(embedding),
        training_config=config,
        training_config_sha256=_sha(config),
        training_run_evidence=evidence,
        model_id=MODEL_ID,
        run_id=RUN_ID,
        output=tmp_path / "actual-evidence-search-manifest.json",
    )

    assert set(evidence_payload) == {
        "schema_version",
        "created_at",
        "state",
        "model_id",
        "queue_exit_code",
        "setup_evidence",
        "training_jobs",
        "training_status",
        "queue_script",
        "invocation",
        "run",
        "runtime_after",
    }
    assert len(jobs["jobs"]) == 12
    assert payload["checkpoint"]["checkpoint_step"] == 250


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_steps", 3000),
        ("max_steps", True),
        ("save_every", 3000),
        ("save_every", True),
    ),
)
def test_builder_rejects_nonisolated_training_config(
    tmp_path: Path,
    field: str,
    value: int | bool,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload["train"][field] = value
    config.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=f"train.{field}.*250"):
        _build(module, source, embedding, config, tmp_path / "search-manifest.json")


@pytest.mark.parametrize("value", [25, True, None])
def test_builder_requires_actual_search_log_interval(
    tmp_path: Path,
    value: int | bool | None,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload["train"]["log_every"] = value
    config.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="train.log_every.*20"):
        _build(module, source, embedding, config, tmp_path / "search-manifest.json")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_manifest_sha256", "0" * 64),
        ("embedding_sha256", "1" * 64),
        ("training_config_sha256", "2" * 64),
    ),
)
def test_builder_fails_closed_on_input_hash_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    kwargs = {
        "source_manifest": source,
        "source_manifest_sha256": _sha(source),
        "embedding": embedding,
        "embedding_sha256": _sha(embedding),
        "training_config": config,
        "training_config_sha256": _sha(config),
        "training_run_evidence": evidence,
        "model_id": MODEL_ID,
        "run_id": RUN_ID,
        "output": tmp_path / "search-manifest.json",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.build_search_manifest(**kwargs)


@pytest.mark.parametrize(("dtype", "finite"), (("F16", True), ("F32", False)))
def test_builder_rejects_non_f32_or_nonfinite_embedding(
    tmp_path: Path,
    dtype: str,
    finite: bool,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    _write_embedding(embedding, dtype=dtype, finite=finite)

    with pytest.raises(ValueError, match="F32|finite"):
        _build(module, source, embedding, config, tmp_path / "search-manifest.json")


def test_builder_rejects_wrong_model_run_and_output_collision(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    with pytest.raises(ValueError, match="model_id"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id="wrong-model",
            run_id=RUN_ID,
            output=tmp_path / "wrong.json",
        )
    with pytest.raises(ValueError, match="run_id"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id="",
            output=tmp_path / "empty.json",
        )
    output = tmp_path / "search-manifest.json"
    _build(module, source, embedding, config, output)
    with pytest.raises(FileExistsError):
        _build(module, source, embedding, config, output)


def test_builder_validates_source_checkpoint_payload_hash(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["models"][0]["checkpoints"][0]["embedding_sha256"] = "f" * 64
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source checkpoint embedding SHA-256 mismatch"):
        _build(module, source, embedding, config, tmp_path / "search-manifest.json")


def test_builder_rejects_wrong_step_run_evidence(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(
        tmp_path,
        embedding=embedding,
        config=config,
        step=249,
    )

    with pytest.raises(ValueError, match="step250|step 250"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


def test_builder_binds_run_id_to_evidence_parent(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)

    with pytest.raises(ValueError, match="run_id root"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id="different-run-id",
            output=tmp_path / "search-manifest.json",
        )


def test_builder_rejects_run_evidence_status_without_exit_code(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    rows = _evidence_status_rows(evidence)
    rows[-1].pop("exit_code")
    _rewrite_evidence_status(evidence, rows)

    with pytest.raises(ValueError, match="started then success"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


@pytest.mark.parametrize("exit_code", [1, True])
def test_builder_rejects_invalid_run_evidence_exit_code(
    tmp_path: Path,
    exit_code: int | bool,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    rows = _evidence_status_rows(evidence)
    rows[-1]["exit_code"] = exit_code
    _rewrite_evidence_status(evidence, rows)

    with pytest.raises(ValueError, match="started then success"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


def test_builder_rejects_running_final_appended_status(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    rows = _evidence_status_rows(evidence)
    rows[-1].update(
        {
            "event": "started",
            "status": "running",
            "exit_code": None,
            "candidate_checkpoints": [],
        }
    )
    _rewrite_evidence_status(evidence, rows)

    with pytest.raises(ValueError, match="started then success"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


def test_builder_rejects_noncanonical_search_checkpoint_name(tmp_path: Path) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    noncanonical = tmp_path / "checkpoint-250.speaker.safetensors"
    noncanonical.write_bytes(embedding.read_bytes())
    evidence = _write_run_evidence(tmp_path, embedding=noncanonical, config=config)

    with pytest.raises(ValueError, match="checkpoint_0000250"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=noncanonical,
            embedding_sha256=_sha(noncanonical),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "jobs",
        "status",
        "checkpoint",
        "queue_exit_nonzero",
        "queue_exit_bool",
        "duplicate_job",
        "job_count",
        "status_prefix",
        "status_arrays",
        "status_after_sha",
        "run_config_path",
        "candidate_count",
        "missing_final",
    ],
)
def test_builder_rejects_run_evidence_binding_drift(  # noqa: C901, PLR0912
    tmp_path: Path,
    mutation: str,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if mutation == "model":
        payload["model_id"] = "wrong-model"
    elif mutation == "jobs":
        jobs = Path(payload["training_jobs"]["path"])
        jobs_payload = json.loads(jobs.read_text(encoding="utf-8"))
        jobs_payload["jobs"][0]["config"] = str(tmp_path / "wrong-config.json")
        jobs.write_text(json.dumps(jobs_payload), encoding="utf-8")
        payload["training_jobs"]["sha256"] = _sha(jobs)
    elif mutation == "status":
        rows = _evidence_status_rows(evidence)
        rows[-1]["status"] = "failed"
        payload = _rewrite_evidence_status(evidence, rows)
    elif mutation == "checkpoint":
        payload["run"]["checkpoints"][0]["sha256"] = "0" * 64
    elif mutation == "queue_exit_nonzero":
        payload["queue_exit_code"] = 1
    elif mutation == "queue_exit_bool":
        payload["queue_exit_code"] = False
    elif mutation in {"duplicate_job", "job_count"}:
        jobs = Path(payload["training_jobs"]["path"])
        jobs_payload = json.loads(jobs.read_text(encoding="utf-8"))
        if mutation == "duplicate_job":
            jobs_payload["jobs"][1]["model_id"] = MODEL_ID
        else:
            jobs_payload["jobs"].pop()
        jobs.write_text(json.dumps(jobs_payload), encoding="utf-8")
        payload["training_jobs"]["sha256"] = _sha(jobs)
    elif mutation == "status_prefix":
        rows = _evidence_status_rows(evidence)
        rows[0]["status"] = "failed"
        payload = _rewrite_evidence_status(evidence, rows)
    elif mutation == "status_arrays":
        payload["training_status"]["new_started_model_ids"] = []
    elif mutation == "status_after_sha":
        payload["training_status"]["after_sha256"] = "0" * 64
    elif mutation == "run_config_path":
        payload["run"]["config_path"] = str(config.parent / "wrong-config.json")
    elif mutation == "candidate_count":
        payload["run"]["candidate_checkpoint_count"] = 1
    else:
        payload["run"]["checkpoints"].pop()
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence|job|status|checkpoint"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "command_config_drift",
        "command_config_missing",
        "command_config_duplicate",
        "command_manifest_drift",
        "command_output_drift",
        "config_manifest_drift",
        "config_output_drift",
    ],
)
def test_builder_rejects_target_job_command_and_config_path_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    jobs = Path(payload["training_jobs"]["path"])
    jobs_payload = json.loads(jobs.read_text(encoding="utf-8"))
    target_job = next(job for job in jobs_payload["jobs"] if job["model_id"] == MODEL_ID)

    if mutation == "command_config_drift":
        target_job["command"][-1] = str(config.parent / "alternate-config.json")
    elif mutation == "command_config_missing":
        target_job["command"] = ["python", "train.py", "--config"]
    elif mutation == "command_config_duplicate":
        target_job["command"].extend(["--config", str(config.resolve())])
    elif mutation == "command_manifest_drift":
        target_job["command"].extend(
            ["--manifest", str(config.parent / "alternate-manifest.jsonl")]
        )
    elif mutation == "command_output_drift":
        target_job["command"].extend(["--output-dir", str(config.parent / "alternate-output")])
    else:
        config_payload = json.loads(config.read_text(encoding="utf-8"))
        field = "manifest_path" if mutation == "config_manifest_drift" else "output_dir"
        config_payload["train"][field] = str(config.parent / f"alternate-{field}")
        config.write_text(json.dumps(config_payload), encoding="utf-8")
        _rebind_evidence_config_sha(evidence, config)
        payload = json.loads(evidence.read_text(encoding="utf-8"))

    if mutation.startswith("command_"):
        jobs.write_text(json.dumps(jobs_payload), encoding="utf-8")
        payload["training_jobs"]["sha256"] = _sha(jobs)
        evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="command|config|manifest|output"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "metadata_keys",
        "metadata_count",
        "metadata_zero_count",
        "metadata_steps",
        "metadata_last_loss",
        "metadata_nonfinite_loss",
        "metadata_finite",
        "metadata_oom",
        "metadata_traceback",
        "body_nan",
        "body_oom",
        "body_cuda_oom",
        "body_traceback",
        "body_missing_finish",
        "body_step",
        "status_error",
    ],
)
def test_builder_rejects_search_training_log_drift(  # noqa: C901, PLR0912
    tmp_path: Path,
    mutation: str,
) -> None:
    module, source, embedding, config = _build_fixture(tmp_path)
    evidence = _write_run_evidence(tmp_path, embedding=embedding, config=config)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    log_metadata = payload["run"]["log"]
    log = Path(log_metadata["path"])
    log_text = log.read_text(encoding="utf-8")

    if mutation == "metadata_keys":
        log_metadata["loss_steps_exact"] = True
    elif mutation == "metadata_count":
        log_metadata["loss_event_count"] = 11
    elif mutation == "metadata_zero_count":
        log_metadata["loss_event_count"] = 0
    elif mutation == "metadata_steps":
        log_metadata["loss_steps"][-1] = 250
    elif mutation == "metadata_last_loss":
        log_metadata["last_loss"] = 0.2
    elif mutation == "metadata_nonfinite_loss":
        log_metadata["last_loss"] = float("nan")
    elif mutation == "metadata_finite":
        log_metadata["loss_all_finite"] = False
    elif mutation == "metadata_oom":
        log_metadata["oom"] = True
    elif mutation == "metadata_traceback":
        log_metadata["traceback"] = True
    elif mutation == "body_nan":
        payload = _rewrite_evidence_log(evidence, log_text.replace("loss=0.1", "loss=nan", 1))
    elif mutation == "body_oom":
        payload = _rewrite_evidence_log(evidence, log_text + "oOm while allocating tensor\n")
    elif mutation == "body_cuda_oom":
        payload = _rewrite_evidence_log(evidence, log_text + "CuDa OuT oF MeMoRy\n")
    elif mutation == "body_traceback":
        payload = _rewrite_evidence_log(evidence, log_text + "Traceback (most recent call last):\n")
    elif mutation == "body_missing_finish":
        payload = _rewrite_evidence_log(
            evidence,
            log_text.replace("Training finished at step=250.\n", ""),
        )
    elif mutation == "body_step":
        payload = _rewrite_evidence_log(evidence, log_text.replace("step=240", "step=250"))
    else:
        rows = _evidence_status_rows(evidence)
        rows[-1]["error"] = "training failed"
        payload = _rewrite_evidence_status(evidence, rows)
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="log|loss|step|OOM|Traceback|status"):
        module.build_search_manifest(
            source_manifest=source,
            source_manifest_sha256=_sha(source),
            embedding=embedding,
            embedding_sha256=_sha(embedding),
            training_config=config,
            training_config_sha256=_sha(config),
            training_run_evidence=evidence,
            model_id=MODEL_ID,
            run_id=RUN_ID,
            output=tmp_path / "search-manifest.json",
        )


def test_generator_loads_only_search_schema_and_builds_exact_28_cases(tmp_path: Path) -> None:
    builder, source, embedding, config = _build_fixture(tmp_path)
    manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, config, manifest)
    module = _load(GENERATOR, "speaker_search_generator_test")

    plan = module.load_search_plan(manifest)

    assert len(module.build_search_cases(plan)) == 28
    assert {case.checkpoint.checkpoint_step for case in module.build_search_cases(plan)} == {250}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "speaker-checkpoint-evaluation-manifest/v1"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TypeError, match="search-manifest"):
        module.load_search_plan(manifest)


def test_generator_rejects_fully_rebound_3000_step_training_config(tmp_path: Path) -> None:
    builder, source, embedding, config = _build_fixture(tmp_path)
    manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, config, manifest)
    _rewrite_bound_search_training_config(manifest, config, max_steps=3000)
    module = _load(GENERATOR, "speaker_search_generator_training_config_test")

    with pytest.raises(ValueError, match="train.max_steps.*250"):
        module.load_search_plan(manifest)


def test_generator_revalidates_search_embedding_payload(tmp_path: Path) -> None:
    builder, source, embedding, config = _build_fixture(tmp_path)
    manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, config, manifest)
    module = _load(GENERATOR, "speaker_search_generator_embedding_test")
    _write_embedding(embedding, dtype="F16")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint"]["embedding_sha256"] = _sha(embedding)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="F32"):
        module.load_search_plan(manifest)


def test_generator_revalidates_pinned_source_contract(tmp_path: Path) -> None:
    builder, source, embedding, config = _build_fixture(tmp_path)
    manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, config, manifest)
    module = _load(GENERATOR, "speaker_search_generator_source_test")
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload["schema_version"] = "wrong-source-schema"
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_evaluation_manifest"]["sha256"] = _sha(source)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source manifest schema_version"):
        module.load_search_plan(manifest)


@pytest.mark.parametrize("field", ["base_revision", "metrics_provenance"])
def test_generator_rejects_source_contract_drift(tmp_path: Path, field: str) -> None:
    builder, source, embedding, config = _build_fixture(tmp_path)
    manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, config, manifest)
    module = _load(GENERATOR, f"speaker_search_generator_drift_{field}_test")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if field == "base_revision":
        payload["checkpoint"]["base_revision"] = "drifted"
    else:
        payload["metrics_provenance"]["speaker_embedding"]["revision"] = "drifted"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source.*drift|does not match source"):
        module.load_search_plan(manifest)


def test_generator_assigns_dedicated_schema_to_every_case_row() -> None:
    module = _load(GENERATOR, "speaker_search_generator_case_schema_test")
    rows = [{"status": "SUCCESS"}, {"status": "ERROR"}]

    bound = module.bind_search_case_schema(rows)

    assert len(bound) == 2
    assert all(row["schema_version"] == SEARCH_CASE_SCHEMA for row in bound)


def test_generator_cli_rejects_dangling_output_symlink_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(GENERATOR, "speaker_search_generator_dangling_output_test")
    output = tmp_path / "generation-output"
    output.symlink_to(tmp_path / "missing-generation-target", target_is_directory=True)
    assert not output.exists()
    production = SimpleNamespace(
        validate_base_checkpoint=lambda *args, **kwargs: "0" * 64,
        validate_upstream_root=lambda *args, **kwargs: None,
        build_cases=lambda plan: tuple(range(28)),
    )
    monkeypatch.setattr(module, "load_search_plan", lambda path: SimpleNamespace())
    monkeypatch.setattr(module, "_production", lambda: production)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main(
            [
                "generate",
                "--search-manifest",
                str(tmp_path / "missing-manifest.json"),
                "--base-checkpoint-path",
                str(tmp_path / "missing-base.safetensors"),
                "--upstream-root",
                str(tmp_path / "missing-upstream"),
                "--output-dir",
                str(output),
            ]
        )


def test_generator_atomic_reserve_rejects_symlink_inserted_after_parent_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(GENERATOR, "speaker_search_generator_output_race_test")
    output = tmp_path / "generation-output"
    dangling_target = tmp_path / "dangling-generation-target"
    prepare_parent = module._prepare_output_parent

    def insert_dangling_symlink(path: Path, *, source: str) -> Path:
        prepared = cast("Path", prepare_parent(path, source=source))
        output.symlink_to(dangling_target, target_is_directory=True)
        return prepared

    monkeypatch.setattr(module, "_prepare_output_parent", insert_dangling_symlink)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.reserve_output(output)
    assert not dangling_target.exists()


def test_generator_reserve_rejects_symlink_parent_boundary(tmp_path: Path) -> None:
    module = _load(GENERATOR, "speaker_search_generator_parent_alias_test")
    actual_parent = tmp_path / "actual-generation-parent"
    actual_parent.mkdir()
    alias_parent = tmp_path / "generation-parent-alias"
    alias_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        module.reserve_output(alias_parent / "output")
    assert not (actual_parent / "output").exists()


def test_generator_emits_fully_bound_config_and_verification(tmp_path: Path) -> None:
    builder, source, embedding, training_config = _build_fixture(tmp_path)
    search_manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, training_config, search_manifest)
    module = _load(GENERATOR, "speaker_search_generator_evidence_contract_test")
    plan = module.load_search_plan(search_manifest)
    base_path = tmp_path / "base-checkpoint.safetensors"
    generation_config = module.build_generation_config(plan=plan, checkpoint_path=base_path)
    config_path = tmp_path / "generation-config.json"
    config_path.write_text(json.dumps(generation_config), encoding="utf-8")
    results_path = tmp_path / "generation-results.jsonl"
    results_path.write_text("{}\n", encoding="utf-8")
    rows = module.bind_search_case_schema(
        [{"case_id": str(index), "status": "SUCCESS", "audio_finite": True} for index in range(28)]
    )

    verification = module.build_generation_verification(
        plan=plan,
        checkpoint_path=base_path,
        config_path=config_path,
        results_path=results_path,
        rows=rows,
    )

    assert generation_config["schema_version"] == "speaker-checkpoint-search-generation/v1"
    assert generation_config["search_manifest_sha256"] == _sha(search_manifest)
    assert generation_config["base_checkpoint_model_id"] == plan.base_checkpoint
    assert generation_config["search_generator_script_sha256"] == _sha(GENERATOR)
    assert verification["schema_version"] == (
        "speaker-checkpoint-search-generation-verification/v1"
    )
    assert verification["passed"] is True
    assert verification["generation_config_sha256"] == _sha(config_path)
    assert verification["production_generator_script_sha256"] == _sha(
        Path("scripts/generate_600m_checkpoint_audio_remote.py")
    )


def _evaluation_rows(module: ModuleType, similarity: float = 0.75) -> list[dict[str, object]]:
    rows = []
    for text_id in TEXT_IDS:
        for seed in SEEDS:
            for style in STYLES:
                metric_gate = text_id in module.METRIC_GATE_TEXT_IDS
                rows.append(
                    {
                        "model_id": MODEL_ID,
                        "checkpoint_step": 250,
                        "text_id": text_id,
                        "seed": seed,
                        "style": style,
                        "metric_gate_applied": metric_gate,
                        "evaluation_status": "PASS",
                        "speaker_similarity": similarity,
                        "normalized_cer": 0.0,
                        "rejection_reasons": [],
                        "incomplete_reasons": [],
                        "review_reasons": [],
                        "audio": {
                            "duration_seconds": 1.0 if style == "neutral" else 1.02,
                            "rms": 0.2,
                        },
                    }
                )
    return rows


def test_evaluator_threshold_boundary_and_similarity_statistics() -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_threshold_test")
    rows = _evaluation_rows(module)
    metric_rows = [row for row in rows if row["metric_gate_applied"]]
    metric_rows[0]["speaker_similarity"] = 0.80
    metric_rows[1]["speaker_similarity"] = 0.76

    summary = module.summarize_search(rows)

    assert summary["status"] == "ELIGIBLE"
    assert summary["speaker_similarity_pass_count"] == 16
    assert summary["min_speaker_similarity"] == 0.75
    assert summary["second_min_speaker_similarity"] == 0.75
    assert summary["mean_speaker_similarity"] == pytest.approx(0.75375)


def test_evaluator_reviews_zero_style_contrast() -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_zero_contrast_test")
    rows = _evaluation_rows(module, similarity=0.9)
    for row in rows:
        row["audio"] = {"duration_seconds": 1.0, "rms": 0.2}

    summary = module.summarize_search(rows)

    assert summary["status"] == "REVIEW"
    assert summary["style_contrast"] == 0.0
    assert "insufficient_style_contrast" in summary["review_reasons"]


def test_evaluator_preserves_incomplete_metric_evidence() -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_incomplete_test")
    rows = _evaluation_rows(module, similarity=0.9)
    row = next(item for item in rows if item["metric_gate_applied"])
    row["speaker_similarity"] = None
    row["evaluation_status"] = "INCOMPLETE"
    row["incomplete_reasons"] = ["missing_speaker_similarity"]

    summary = module.summarize_search(rows)

    assert summary["status"] == "INCOMPLETE"
    assert summary["speaker_similarity_pass_count"] == 15
    assert summary["min_speaker_similarity"] == pytest.approx(0.9)
    assert "missing_speaker_similarity" in summary["incomplete_reasons"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("tone", "tone_candidate"),
        ("audio", "invalid_audio"),
        ("cer", "high_normalized_cer"),
        ("style", "style_similarity_not_preserved"),
    ),
)
def test_evaluator_reports_each_quality_gate(mutation: str, reason: str) -> None:
    module = _load(EVALUATOR, f"speaker_search_evaluator_{mutation}_test")
    rows = _evaluation_rows(module, similarity=0.9)
    if mutation in {"tone", "audio", "cer"}:
        rows[0]["evaluation_status"] = "REJECTED"
        rows[0]["rejection_reasons"] = [reason]
    else:
        neutral = next(
            row
            for row in rows
            if row["text_id"] == "control" and row["seed"] == 1234 and row["style"] == "neutral"
        )
        calm = next(
            row
            for row in rows
            if row["text_id"] == "control" and row["seed"] == 1234 and row["style"] == "calm"
        )
        neutral["speaker_similarity"] = 0.90
        calm["speaker_similarity"] = 0.80

    summary = module.summarize_search(rows)

    assert summary["status"] == "REJECTED"
    assert reason in summary["rejection_reasons"]


@pytest.mark.parametrize("mutation", ("duplicate", "missing", "extra"))
def test_evaluator_rejects_non_exact_case_matrix(mutation: str) -> None:
    module = _load(EVALUATOR, f"speaker_search_evaluator_matrix_{mutation}_test")
    identities = [
        (MODEL_ID, 250, text_id, seed, style)
        for text_id in TEXT_IDS
        for seed in SEEDS
        for style in STYLES
    ]
    if mutation == "duplicate":
        identities[-1] = identities[0]
    elif mutation == "missing":
        identities.pop()
    else:
        identities.append((MODEL_ID, 250, "extra", 1234, "neutral"))

    with pytest.raises(ValueError, match="duplicate|missing|unexpected"):
        module.validate_case_matrix(identities, model_id=MODEL_ID)


@pytest.mark.parametrize("mutation", ("wav", "hash", "provenance"))
def test_evaluator_rejects_swapped_artifact_identity(mutation: str) -> None:
    module = _load(EVALUATOR, f"speaker_search_evaluator_identity_{mutation}_test")
    generation = {
        "case": {
            "wav_path": "/audio/a.wav",
            "wav_sha256": "a" * 64,
            "provenance": {"run_id": RUN_ID},
        }
    }
    other = {
        "case": {
            "wav_path": "/audio/a.wav",
            "wav_sha256": "a" * 64,
            "provenance": {"run_id": RUN_ID},
        }
    }
    if mutation == "wav":
        other["case"]["wav_path"] = "/audio/b.wav"
    elif mutation == "hash":
        other["case"]["wav_sha256"] = "b" * 64
    else:
        other["case"]["provenance"] = {"run_id": "different"}

    with pytest.raises(ValueError, match="identity"):
        module.validate_artifact_identity(generation, other, source="analysis")


def test_evaluator_rejects_non_search_case_schema() -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_case_schema_test")
    rows = {
        "good": {"schema_version": SEARCH_CASE_SCHEMA},
        "bad": {"schema_version": "speaker-checkpoint-audio-generation/v1"},
    }

    with pytest.raises(ValueError, match="row schema mismatch: bad"):
        module.validate_generation_case_schemas(rows, source="generation")


def test_evaluator_overwrites_schema_for_every_evaluation_row() -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_output_schema_test")
    rows = [
        {"evaluation_schema_version": "speaker-checkpoint-evaluation/v1"},
        {"evaluation_schema_version": "speaker-checkpoint-evaluation/v1"},
    ]

    bound = module.bind_search_evaluation_case_schema(rows)

    assert len(bound) == 2
    assert all(row["evaluation_schema_version"] == SEARCH_EVALUATION_CASE_SCHEMA for row in bound)


@pytest.mark.parametrize("field", ["base_checkpoint_sha256", "metrics_provenance"])
def test_evaluator_rejects_source_contract_drift(tmp_path: Path, field: str) -> None:
    builder, source, embedding, config = _build_fixture(tmp_path)
    manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, config, manifest)
    module = _load(EVALUATOR, f"speaker_search_evaluator_drift_{field}_test")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if field == "base_checkpoint_sha256":
        payload["checkpoint"]["base_checkpoint_sha256"] = "0" * 64
    else:
        payload["metrics_provenance"]["transcription"]["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source.*drift|does not match source"):
        module.load_search_manifest(manifest)


def test_evaluator_rejects_fully_rebound_3000_step_training_config(tmp_path: Path) -> None:
    builder, source, embedding, config = _build_fixture(tmp_path)
    manifest = tmp_path / "search-manifest.json"
    _build(builder, source, embedding, config, manifest)
    _rewrite_bound_search_training_config(manifest, config, max_steps=3000)
    module = _load(EVALUATOR, "speaker_search_evaluator_training_config_test")

    with pytest.raises(ValueError, match="train.max_steps.*250"):
        module.load_search_manifest(manifest)


def _generation_evidence_fixture(
    tmp_path: Path,
) -> tuple[ModuleType, Path, Path, Path, Any]:
    builder, source, embedding, training_config = _build_fixture(tmp_path)
    search_manifest = tmp_path / "search-manifest.json"
    search_payload = _build(builder, source, embedding, training_config, search_manifest)
    evaluator = _load(EVALUATOR, "speaker_search_generation_evidence_test")
    manifest = evaluator.load_search_manifest(search_manifest)
    results = tmp_path / "generation-results.jsonl"
    results.write_text("{}\n", encoding="utf-8")
    base_path = tmp_path / "base-checkpoint.safetensors"
    config = tmp_path / "generation-config.json"
    config_payload = {
        "schema_version": "speaker-checkpoint-search-generation/v1",
        "model_id": MODEL_ID,
        "case_count": 28,
        "search_manifest_path": str(search_manifest.resolve()),
        "search_manifest_sha256": _sha(search_manifest),
        "search_generator_script": str(GENERATOR.resolve()),
        "search_generator_script_sha256": _sha(GENERATOR),
        "production_generator_script": str(
            Path("scripts/generate_600m_checkpoint_audio_remote.py").resolve()
        ),
        "production_generator_script_sha256": _sha(
            Path("scripts/generate_600m_checkpoint_audio_remote.py")
        ),
        "base_checkpoint_path": str(base_path.resolve()),
        "base_checkpoint_model_id": search_payload["checkpoint"]["base_checkpoint"],
        "base_checkpoint_sha256": search_payload["checkpoint"]["base_checkpoint_sha256"],
        "base_revision": search_payload["checkpoint"]["base_revision"],
        "text_ids": list(TEXT_IDS),
        "seeds": list(SEEDS),
        "styles": list(STYLES),
    }
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    verification = tmp_path / "generation-verification.json"
    verification.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-search-generation-verification/v1",
                "passed": True,
                "model_id": MODEL_ID,
                "case_count": 28,
                "row_count": 28,
                "status_counts": {"SUCCESS": 28},
                "case_ids_unique": True,
                "all_audio_finite": True,
                "search_manifest_path": str(search_manifest.resolve()),
                "search_manifest_sha256": _sha(search_manifest),
                "generation_config_path": str(config.resolve()),
                "generation_config_sha256": _sha(config),
                "generation_results_path": str(results.resolve()),
                "generation_results_sha256": _sha(results),
                "base_checkpoint_path": str(base_path.resolve()),
                "base_checkpoint_model_id": config_payload["base_checkpoint_model_id"],
                "base_checkpoint_sha256": config_payload["base_checkpoint_sha256"],
                "base_revision": config_payload["base_revision"],
                "search_generator_script": str(GENERATOR.resolve()),
                "search_generator_script_sha256": _sha(GENERATOR),
                "production_generator_script": config_payload["production_generator_script"],
                "production_generator_script_sha256": config_payload[
                    "production_generator_script_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )
    return evaluator, verification, results, search_manifest, manifest


def test_evaluator_accepts_fully_bound_generation_evidence(tmp_path: Path) -> None:
    module, verification, results, manifest_path, manifest = _generation_evidence_fixture(tmp_path)

    module._validate_generation_evidence(
        verification,
        generation_results=results,
        search_manifest=manifest_path,
        manifest=manifest,
    )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "model_id",
        "case_count",
        "search_manifest_sha256",
        "base_checkpoint_model_id",
        "search_generator_script_sha256",
        "production_generator_script_sha256",
        "unexpected_field",
    ],
)
def test_evaluator_rejects_generation_config_contract_drift(
    tmp_path: Path,
    field: str,
) -> None:
    module, verification, results, manifest_path, manifest = _generation_evidence_fixture(tmp_path)
    verification_payload = json.loads(verification.read_text(encoding="utf-8"))
    config = Path(verification_payload["generation_config_path"])
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload[field] = "drifted" if field != "case_count" else 27
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    verification_payload["generation_config_sha256"] = _sha(config)
    verification.write_text(json.dumps(verification_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generation config"):
        module._validate_generation_evidence(
            verification,
            generation_results=results,
            search_manifest=manifest_path,
            manifest=manifest,
        )


@pytest.mark.parametrize(
    "field",
    [
        "generation_config_path",
        "generation_config_sha256",
        "search_manifest_sha256",
        "base_checkpoint_sha256",
        "search_generator_script_sha256",
        "production_generator_script_sha256",
        "unexpected_field",
        "status_counts",
    ],
)
def test_evaluator_rejects_generation_verification_contract_drift(
    tmp_path: Path,
    field: str,
) -> None:
    module, verification, results, manifest_path, manifest = _generation_evidence_fixture(tmp_path)
    payload = json.loads(verification.read_text(encoding="utf-8"))
    if field == "status_counts":
        payload[field] = {"SUCCESS": 27, "ERROR": 1}
    else:
        payload[field] = str(tmp_path / "wrong") if field.endswith("_path") else "0" * 64
    verification.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generation (config|verification)"):
        module._validate_generation_evidence(
            verification,
            generation_results=results,
            search_manifest=manifest_path,
            manifest=manifest,
        )


def test_evaluator_reserves_output_create_only(tmp_path: Path) -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_output_test")
    output = tmp_path / "search-evaluation"

    module.reserve_output(output)

    assert output.is_dir()
    with pytest.raises(FileExistsError):
        module.reserve_output(output)


def test_evaluator_atomic_reserve_rejects_symlink_inserted_after_parent_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_output_race_test")
    output = tmp_path / "evaluation-output"
    dangling_target = tmp_path / "dangling-evaluation-target"
    prepare_parent = module._prepare_output_parent

    def insert_dangling_symlink(path: Path, *, source: str) -> Path:
        prepared = cast("Path", prepare_parent(path, source=source))
        output.symlink_to(dangling_target, target_is_directory=True)
        return prepared

    monkeypatch.setattr(module, "_prepare_output_parent", insert_dangling_symlink)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.reserve_output(output)
    assert not dangling_target.exists()


def test_evaluator_reserve_rejects_symlink_parent_boundary(tmp_path: Path) -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_parent_alias_test")
    actual_parent = tmp_path / "actual-evaluation-parent"
    actual_parent.mkdir()
    alias_parent = tmp_path / "evaluation-parent-alias"
    alias_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        module.reserve_output(alias_parent / "output")
    assert not (actual_parent / "output").exists()


def test_evaluator_cli_rejects_dangling_output_symlink_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_dangling_output_test")
    argv, _ = _stub_evaluator_cli_inputs(tmp_path, module, monkeypatch)
    output = Path(argv[-1])
    output.symlink_to(tmp_path / "missing-evaluation-target", target_is_directory=True)
    assert not output.exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main(argv)


def _stub_evaluator_cli_inputs(
    tmp_path: Path,
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], Path]:
    wav = tmp_path / "generated.wav"
    wav.write_bytes(b"generated wav bytes")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference wav bytes")
    identity = {
        "case_id": "case",
        "model_id": MODEL_ID,
        "checkpoint_step": 250,
        "text_id": "control",
        "seed": 1234,
        "style": "neutral",
        "status": "SUCCESS",
        "schema_version": SEARCH_CASE_SCHEMA,
        "wav_path": str(wav),
        "wav_sha256": _sha(wav),
        "provenance": {"run_id": RUN_ID},
    }
    paths = {
        "search_manifest": tmp_path / "search-manifest.json",
        "generation_results": tmp_path / "generation-results.jsonl",
        "generation_verification": tmp_path / "generation-verification.json",
        "analysis_results": tmp_path / "analysis-results.jsonl",
        "metrics_results": tmp_path / "metrics-results.jsonl",
        "metrics_provenance": tmp_path / "metrics-provenance.json",
    }
    paths["search_manifest"].write_text("{}\n", encoding="utf-8")
    paths["generation_verification"].write_text("{}\n", encoding="utf-8")
    for name in ("generation_results", "analysis_results", "metrics_results"):
        paths[name].write_text(json.dumps(identity) + "\n", encoding="utf-8")
    paths["metrics_provenance"].write_text(
        json.dumps(
            {
                "input_sha256": {
                    "generated_audio": {str(wav): _sha(wav)},
                    "reference_audio": {str(reference): _sha(reference)},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = SimpleNamespace(checkpoints={(MODEL_ID, 250): object()})
    production = SimpleNamespace(
        _index_rows=lambda rows, *, source: {str(row["case_id"]): row for row in rows},
        _validate_inputs=lambda *args, **kwargs: None,
        _validate_expected_matrix=lambda *args, **kwargs: None,
        _validate_checkpoint_contract=lambda *args, **kwargs: None,
        _validate_metrics_provenance=lambda *args, **kwargs: None,
        _validate_metric_rows=lambda *args, **kwargs: None,
        _validate_audio_bindings=lambda *args, **kwargs: None,
        _evaluate_case=lambda *args, **kwargs: {},
        _case_sort_key=lambda row: str(row["case_id"]),
        DEFAULT_CONFIG=object(),
    )
    monkeypatch.setattr(module, "load_search_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(module, "_validate_generation_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_production", lambda: production)
    monkeypatch.setattr(module, "validate_case_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "summarize_search",
        lambda rows: {"status": "ELIGIBLE", "hard_gate_metric_case_count": 0},
    )
    argv = [
        "--search-manifest",
        str(paths["search_manifest"]),
        "--generation-results",
        str(paths["generation_results"]),
        "--generation-verification",
        str(paths["generation_verification"]),
        "--analysis-results",
        str(paths["analysis_results"]),
        "--metrics-results",
        str(paths["metrics_results"]),
        "--metrics-provenance",
        str(paths["metrics_provenance"]),
        "--output-dir",
        str(tmp_path / "evaluation-output"),
    ]
    return argv, wav


@pytest.mark.parametrize("source", ["analysis_results", "metrics_results"])
def test_evaluator_cli_rejects_jsonl_replaced_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    module = _load(EVALUATOR, f"speaker_search_evaluator_{source}_swap_test")
    argv, _ = _stub_evaluator_cli_inputs(tmp_path, module, monkeypatch)
    original_snapshot = module._snapshot_file

    def snapshot_and_replace(path: Path, *, source: str) -> tuple[bytes, str]:
        snapshot = cast("tuple[bytes, str]", original_snapshot(path, source=source))
        if source == source_to_replace:
            path.write_text(json.dumps({"case_id": "replacement"}) + "\n", encoding="utf-8")
        return snapshot

    source_to_replace = source
    monkeypatch.setattr(module, "_snapshot_file", snapshot_and_replace)

    with pytest.raises(ValueError, match=f"{source}.*changed"):
        module.main(argv)


def test_evaluator_cli_rejects_wav_replaced_after_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_bound_wav_swap_test")
    argv, wav = _stub_evaluator_cli_inputs(tmp_path, module, monkeypatch)
    production = module._production()

    def replace_wav(*args: object, **kwargs: object) -> dict[str, object]:
        wav.write_bytes(b"replacement wav bytes")
        return {}

    production._evaluate_case = replace_wav

    with pytest.raises(ValueError, match="bound WAV.*changed"):
        module.main(argv)


@pytest.mark.parametrize("source", ["analysis", "metrics"])
def test_evaluator_parses_the_same_input_snapshot_it_hashes(
    tmp_path: Path,
    source: str,
) -> None:
    module = _load(EVALUATOR, f"speaker_search_evaluator_{source}_snapshot_test")
    path = tmp_path / f"{source}.jsonl"
    path.write_text(json.dumps({"case_id": "original"}) + "\n", encoding="utf-8")
    snapshot = module._snapshot_file(path, source=source)
    path.write_text(json.dumps({"case_id": "replacement"}) + "\n", encoding="utf-8")

    rows = module._read_jsonl_snapshot(snapshot, path=path)

    assert rows == [{"case_id": "original"}]
    with pytest.raises(ValueError, match=f"{source}.*changed"):
        module._validate_snapshot_unchanged(snapshot, path=path, source=source)


def test_evaluator_rejects_bound_wav_changed_after_binding(tmp_path: Path) -> None:
    module = _load(EVALUATOR, "speaker_search_evaluator_wav_toctou_test")
    wav = tmp_path / "case.wav"
    wav.write_bytes(b"bound wav bytes")
    bindings = {wav.resolve(): _sha(wav)}
    module._validate_bound_wavs_unchanged(bindings)
    wav.write_bytes(b"replacement wav bytes")

    with pytest.raises(ValueError, match="bound WAV.*changed"):
        module._validate_bound_wavs_unchanged(bindings)


@pytest.mark.parametrize("mutation", ["schema", "passed", "count", "hash", "path"])
def test_evaluator_rejects_invalid_generation_verification(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load(EVALUATOR, f"speaker_search_generation_verification_{mutation}_test")
    results = tmp_path / "generation-results.jsonl"
    results.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "speaker-checkpoint-search-generation-verification/v1",
        "passed": True,
        "row_count": 28,
        "generation_results_path": str(results.resolve()),
        "generation_results_sha256": _sha(results),
    }
    if mutation == "schema":
        payload["schema_version"] = "wrong"
    elif mutation == "passed":
        payload["passed"] = False
    elif mutation == "count":
        payload["row_count"] = 27
    elif mutation == "hash":
        payload["generation_results_sha256"] = "0" * 64
    else:
        payload["generation_results_path"] = str(tmp_path / "other.jsonl")

    with pytest.raises(ValueError, match="generation verification"):
        module._validate_generation_verification(
            payload,
            generation_results=results,
            generation_results_sha256=_sha(results),
        )


def test_production_evaluation_scripts_remain_byte_identical() -> None:
    assert {path: _sha(Path(path)) for path in PRODUCTION_HASHES} == PRODUCTION_HASHES
