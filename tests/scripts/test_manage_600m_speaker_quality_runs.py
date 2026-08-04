# ruff: noqa: PLR0914, PLR2004, PT007, PT018, S404, SLF001

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT = Path("scripts/manage_600m_speaker_quality_runs.py")
MODEL_ID = "target-model"


class PrepareFixture(TypedDict):
    jobs: Path
    jobs_document: dict[str, Any]
    status: Path
    status_bytes: bytes
    diagnostic: Path
    queue_script: Path
    quality_root: Path
    config: Path
    config_payload: dict[str, Any]
    manifest: Path
    base_checkpoint: Path
    predecessor_output: Path
    init_checkpoint: Path


class PreparedResult(TypedDict):
    kind: str
    model_id: str
    run_root: str
    config: str
    jobs: str
    status: str
    setup_evidence: str
    output_dir: str


class CompletedFixture(PrepareFixture):
    before_status: bytes
    started: dict[str, Any]
    finished: dict[str, Any]
    runtime: Path
    log: Path
    output: Path


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_argument(command: list[str], flag: str) -> str:
    assert command.count(flag) == 1
    index = command.index(flag) + 1
    assert index < len(command)
    return command[index]


def _write_embedding(path: Path, value: float = 1.0) -> None:
    payload = struct.pack("<f", value) * (16 * 768)
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": "F32",
                "shape": [16, 768],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    padding = b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header) + len(padding)) + header + padding + payload)


def _prepare_fixture(tmp_path: Path, *, checkpoint_step: int = 2500) -> PrepareFixture:
    root = tmp_path / "predecessor"
    root.mkdir()
    base_checkpoint = root / "base.safetensors"
    base_checkpoint.write_bytes(b"base checkpoint")
    manifest = root / "clean-manifest.jsonl"
    manifest.write_text('{"source_id":"sample"}\n', encoding="utf-8")
    predecessor_output = root / "target-output"
    predecessor_output.mkdir()
    init_checkpoint = predecessor_output / f"checkpoint_{checkpoint_step:07d}.speaker.safetensors"
    _write_embedding(init_checkpoint)
    config = root / "target-config.json"
    config_payload = {
        "model": {"name": "unchanged"},
        "train": {
            "manifest_path": str(manifest.resolve()),
            "output_dir": str(predecessor_output.resolve()),
            "adam_eps": 0.00000001,
            "learning_rate": 0.001,
            "seed": 7,
            "max_steps": 3000,
            "save_every": 250,
            "log_every": 20,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": True,
            "speaker_inversion_enabled": True,
            "speaker_inversion_tokens": 16,
            "valid_ratio": 0.0,
            "checkpoint_best_n": 0,
        },
    }
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    jobs_path = root / "training-jobs.json"
    jobs = []
    for index in range(12):
        model_id = MODEL_ID if index == 4 else f"model-{index:02d}"
        job_config = config if model_id == MODEL_ID else root / f"config-{index:02d}.json"
        job_output = predecessor_output if model_id == MODEL_ID else root / f"output-{index:02d}"
        command = ["python", "train.py", "--config", str(job_config.resolve())]
        if model_id == MODEL_ID:
            command = [
                "python",
                "-u",
                "train.py",
                "--config",
                str(job_config.resolve()),
                "--manifest",
                str(manifest.resolve()),
                "--init-checkpoint",
                str(base_checkpoint.resolve()),
                "--output-dir",
                str(job_output.resolve()),
                "--device",
                "cuda",
            ]
        jobs.append(
            {
                "model_id": model_id,
                "clean_manifest": str(manifest.resolve()),
                "config": str(job_config.resolve()),
                "output_dir": str(job_output.resolve()),
                "command": command,
            }
        )
    jobs_document = {
        "schema_version": 1,
        "created_at_utc": "2026-08-02T00:00:00+00:00",
        "base_checkpoint_path": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": _sha(base_checkpoint),
        "checkpoint_revision": "base-revision",
        "upstream_commit": "upstream-commit",
        "queue_policy": "serial_one_at_a_time",
        "anabel_strategy": "reuse_existing_fresh_3000_run",
        "jobs": jobs,
    }
    jobs_path.write_text(json.dumps(jobs_document), encoding="utf-8")
    status_path = root / "training-status.jsonl"
    status_row = {
        "model_id": MODEL_ID,
        "event": "finished",
        "status": "success",
        "clean_manifest_sha256": _sha(manifest),
        "config_sha256": _sha(config),
        "checkpoint_sha256": _sha(base_checkpoint),
        "checkpoint_revision": "base-revision",
        "upstream_commit": "upstream-commit",
        "started_at": "2026-08-02T00:01:00+00:00",
        "ended_at": "2026-08-02T01:00:00+00:00",
        "exit_code": 0,
        "log_path": str((root / "training.log").resolve()),
        "last_checkpoint": str(init_checkpoint.resolve()),
        "last_checkpoint_sha256": _sha(init_checkpoint),
        "candidate_checkpoints": [
            {"path": str(init_checkpoint.resolve()), "sha256": _sha(init_checkpoint)}
        ],
        "error": None,
    }
    status_path.write_text(json.dumps(status_row, sort_keys=True) + "\n", encoding="utf-8")
    diagnostic = root / "source-diagnostic.json"
    cases = [{"text_id": f"case-{index}", "speaker_similarity": 0.9} for index in range(16)]
    cases[-1] = {"text_id": "sentence_manko", "speaker_similarity": 0.7}
    diagnostic.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-search-evaluation/v1",
                "model_id": MODEL_ID,
                "checkpoint_step": checkpoint_step,
                "hard_gate_metric_case_count": 16,
                "speaker_similarity_pass_count": 15,
                "min_speaker_similarity": 0.7,
                "per_case_metrics": cases,
            }
        ),
        encoding="utf-8",
    )
    queue_script = root / "run_600m_speaker_training_queue.py"
    queue_script.write_text("# queue\n", encoding="utf-8")
    quality_root = tmp_path / "quality-runs"
    quality_root.mkdir()
    return PrepareFixture(
        jobs=jobs_path,
        jobs_document=jobs_document,
        status=status_path,
        status_bytes=status_path.read_bytes(),
        diagnostic=diagnostic,
        queue_script=queue_script,
        quality_root=quality_root,
        config=config,
        config_payload=config_payload,
        manifest=manifest,
        base_checkpoint=base_checkpoint,
        predecessor_output=predecessor_output,
        init_checkpoint=init_checkpoint,
    )


def test_extract_source_best_from_search_evaluator_summary(tmp_path: Path) -> None:
    module = _load("quality_runs_source_best_test")
    diagnostic = tmp_path / "search-evaluation.json"
    per_case = [
        {
            "text_id": f"case-{index}",
            "seed": 1234,
            "style": "neutral",
            "speaker_similarity": 0.9,
        }
        for index in range(16)
    ]
    per_case[7]["text_id"] = "sentence_manko"
    per_case[7]["speaker_similarity"] = 0.7
    diagnostic.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-search-evaluation/v1",
                "model_id": MODEL_ID,
                "checkpoint_step": 2500,
                "case_count": 28,
                "hard_gate_metric_case_count": 16,
                "speaker_similarity_pass_count": 15,
                "min_speaker_similarity": 0.7,
                "per_case_metrics": per_case,
            }
        ),
        encoding="utf-8",
    )

    source_best = module.extract_source_best(
        diagnostic,
        model_id=MODEL_ID,
        checkpoint_step=2500,
    )

    assert source_best == {
        "checkpoint_step": 2500,
        "hard_gate_pass_count": 15,
        "hard_gate_case_count": 16,
        "failing_case": "sentence_manko",
        "speaker_similarity": 0.7,
        "required_minimum": 0.75,
    }


@pytest.mark.parametrize("pass_count", [14, 8])
def test_extract_source_best_accepts_multiple_failures_from_summary(
    tmp_path: Path,
    pass_count: int,
) -> None:
    module = _load(f"quality_runs_source_best_multiple_{pass_count}_test")
    diagnostic = tmp_path / "search-evaluation.json"
    failure_count = 16 - pass_count
    per_case = [
        {
            "case_id": f"pass-{index}",
            "text_id": f"pass-{index}",
            "seed": 1234,
            "style": "neutral",
            "speaker_similarity": 0.9,
        }
        for index in range(pass_count)
    ]
    per_case.extend(
        {
            "case_id": f"failure-{index}",
            "text_id": f"failure-{index}",
            "seed": 1234,
            "style": "neutral",
            "speaker_similarity": 0.7,
        }
        for index in range(failure_count)
    )
    per_case[-2].update(
        {
            "case_id": "case-z",
            "text_id": "zeta",
            "seed": 5678,
            "style": "calm",
            "speaker_similarity": 0.6,
        }
    )
    per_case[-1].update(
        {
            "case_id": "case-a",
            "text_id": "alpha",
            "seed": 1234,
            "style": "neutral",
            "speaker_similarity": 0.6,
        }
    )
    diagnostic.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-search-evaluation/v1",
                "model_id": MODEL_ID,
                "checkpoint_step": 2500,
                "hard_gate_metric_case_count": 16,
                "speaker_similarity_pass_count": pass_count,
                "min_speaker_similarity": 0.6,
                "per_case_metrics": per_case,
            }
        ),
        encoding="utf-8",
    )

    source_best = module.extract_source_best(
        diagnostic,
        model_id=MODEL_ID,
        checkpoint_step=2500,
    )

    assert source_best["hard_gate_pass_count"] == pass_count
    assert source_best["hard_gate_case_count"] == 16
    assert source_best["failing_case"] == "case-a"
    assert source_best["speaker_similarity"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("declared_pass_count", "row_pass_count"),
    [
        (16, 16),
        (-1, 0),
        (15, 14),
    ],
)
def test_extract_source_best_rejects_invalid_summary_pass_boundaries(
    tmp_path: Path,
    declared_pass_count: int,
    row_pass_count: int,
) -> None:
    module = _load(
        f"quality_runs_summary_pass_boundary_{declared_pass_count}_{row_pass_count}_test"
    )
    diagnostic = tmp_path / "search-evaluation.json"
    rows = [
        {
            "case_id": f"case-{index:02d}",
            "text_id": f"case-{index:02d}",
            "seed": 1234,
            "style": "neutral",
            "speaker_similarity": 0.9 if index < row_pass_count else 0.7,
        }
        for index in range(16)
    ]
    diagnostic.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-search-evaluation/v1",
                "model_id": MODEL_ID,
                "checkpoint_step": 2500,
                "hard_gate_metric_case_count": 16,
                "speaker_similarity_pass_count": declared_pass_count,
                "min_speaker_similarity": 0.9 if row_pass_count == 16 else 0.7,
                "per_case_metrics": rows,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"pass count|at least one failing"):
        module.extract_source_best(diagnostic, model_id=MODEL_ID, checkpoint_step=2500)


def test_extract_source_best_accepts_zero_passes_from_summary(tmp_path: Path) -> None:
    module = _load("quality_runs_summary_zero_passes_test")
    diagnostic = tmp_path / "search-evaluation.json"
    rows = [
        {
            "case_id": f"case-{index:02d}",
            "text_id": f"case-{index:02d}",
            "seed": 1234,
            "style": "neutral",
            "speaker_similarity": 0.6 if index == 0 else 0.7,
        }
        for index in range(16)
    ]
    diagnostic.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-search-evaluation/v1",
                "model_id": MODEL_ID,
                "checkpoint_step": 2500,
                "hard_gate_metric_case_count": 16,
                "speaker_similarity_pass_count": 0,
                "min_speaker_similarity": 0.6,
                "per_case_metrics": rows,
            }
        ),
        encoding="utf-8",
    )

    source_best = module.extract_source_best(
        diagnostic,
        model_id=MODEL_ID,
        checkpoint_step=2500,
    )

    assert source_best["hard_gate_pass_count"] == 0
    assert source_best["failing_case"] == "case-00"


def test_extract_source_best_from_production_evaluator_results_jsonl(tmp_path: Path) -> None:
    module = _load("quality_runs_production_results_test")
    diagnostic = tmp_path / "evaluation-results.jsonl"
    rows = [
        {
            "evaluation_schema_version": "speaker-checkpoint-evaluation/v1",
            "model_id": MODEL_ID,
            "checkpoint_step": 2500,
            "text_id": "sentence_manko" if index == 9 else f"case-{index}",
            "seed": 1234,
            "style": "neutral",
            "metric_gate_applied": True,
            "speaker_similarity": 0.7 if index == 9 else 0.9,
        }
        for index in range(16)
    ]
    diagnostic.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    source_best = module.extract_source_best(
        diagnostic,
        model_id=MODEL_ID,
        checkpoint_step=2500,
    )

    assert source_best["hard_gate_pass_count"] == 15
    assert source_best["hard_gate_case_count"] == 16
    assert source_best["failing_case"] == "sentence_manko"


@pytest.mark.parametrize("pass_count", [14, 7])
def test_extract_source_best_accepts_multiple_failures_from_evaluator_jsonl(
    tmp_path: Path,
    pass_count: int,
) -> None:
    module = _load(f"quality_runs_production_multiple_{pass_count}_test")
    diagnostic = tmp_path / "evaluation-results.jsonl"
    rows = [
        {
            "evaluation_schema_version": "speaker-checkpoint-evaluation/v1",
            "model_id": MODEL_ID,
            "checkpoint_step": 2500,
            "case_id": f"case-{index:02d}",
            "text_id": f"pass-{index:02d}" if index < pass_count else f"failure-{index:02d}",
            "seed": 1234,
            "style": "neutral",
            "metric_gate_applied": True,
            "speaker_similarity": 0.9 if index < pass_count else 0.7,
        }
        for index in range(16)
    ]
    rows[-2].update(
        {
            "case_id": "case-z",
            "text_id": "zeta",
            "seed": 5678,
            "style": "calm",
            "speaker_similarity": 0.6,
        }
    )
    rows[-1].update(
        {
            "case_id": "case-a",
            "text_id": "alpha",
            "seed": 1234,
            "style": "neutral",
            "speaker_similarity": 0.6,
        }
    )
    diagnostic.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    source_best = module.extract_source_best(
        diagnostic,
        model_id=MODEL_ID,
        checkpoint_step=2500,
    )

    assert source_best["hard_gate_pass_count"] == pass_count
    assert source_best["hard_gate_case_count"] == 16
    assert source_best["failing_case"] == "case-a"
    assert source_best["speaker_similarity"] == pytest.approx(0.6)


@pytest.mark.parametrize("scenario", ["all_fail", "all_pass"])
def test_extract_source_best_jsonl_zero_failure_boundary(
    tmp_path: Path,
    scenario: str,
) -> None:
    module = _load(f"quality_runs_jsonl_zero_failure_{scenario}_test")
    diagnostic = tmp_path / "evaluation-results.jsonl"
    rows = [
        {
            "evaluation_schema_version": "speaker-checkpoint-evaluation/v1",
            "model_id": MODEL_ID,
            "checkpoint_step": 2500,
            "case_id": f"case-{index:02d}",
            "text_id": f"case-{index:02d}",
            "seed": 1234,
            "style": "neutral",
            "metric_gate_applied": True,
            "speaker_similarity": 0.9 if scenario == "all_pass" else 0.7,
        }
        for index in range(16)
    ]
    diagnostic.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    if scenario == "all_pass":
        with pytest.raises(ValueError, match="at least one failing"):
            module.extract_source_best(diagnostic, model_id=MODEL_ID, checkpoint_step=2500)
    else:
        source_best = module.extract_source_best(
            diagnostic,
            model_id=MODEL_ID,
            checkpoint_step=2500,
        )
        assert source_best["hard_gate_pass_count"] == 0
        assert source_best["failing_case"] == "case-00"


@pytest.mark.parametrize(("kind", "max_steps"), (("search", 250), ("retrain", 3000)))
def test_prepare_clones_one_target_and_publishes_create_only_artifacts(
    tmp_path: Path,
    kind: str,
    max_steps: int,
) -> None:
    module = _load(f"quality_runs_prepare_{kind}_test")
    fixture = _prepare_fixture(tmp_path)
    run_root = fixture["quality_root"] / f"{kind}-v1"
    source_jobs_bytes = fixture["jobs"].read_bytes()
    source_config_bytes = fixture["config"].read_bytes()
    base_checkpoint_bytes = fixture["base_checkpoint"].read_bytes()
    init_checkpoint_bytes = fixture["init_checkpoint"].read_bytes()

    result = cast(
        "PreparedResult",
        module.prepare_quality_run(
            kind=kind,
            predecessor_jobs=fixture["jobs"],
            predecessor_status=fixture["status"],
            source_diagnostic=fixture["diagnostic"],
            model_id=MODEL_ID,
            init_checkpoint_step=2500,
            learning_rate=0.0003,
            seed=11,
            run_root=run_root,
            queue_script=fixture["queue_script"],
            strategy="initialize from the strongest predecessor checkpoint",
        ),
    )

    config_path = Path(result["config"])
    jobs_path = Path(result["jobs"])
    status_path = Path(result["status"])
    setup_path = Path(result["setup_evidence"])
    output_dir = Path(result["output_dir"])
    assert {path.parent for path in (config_path, jobs_path, status_path, setup_path)} == {run_root}
    assert output_dir.parent == run_root
    assert output_dir.is_dir() and not tuple(output_dir.iterdir())
    assert status_path.read_bytes() == fixture["status_bytes"]

    source_config = fixture["config_payload"]
    expected_config = json.loads(json.dumps(source_config))
    expected_config["train"].update(
        {
            "manifest_path": str(Path(fixture["manifest"]).resolve()),
            "output_dir": str(output_dir.resolve()),
            "learning_rate": 0.0003,
            "seed": 11,
            "max_steps": max_steps,
            "save_every": 250,
            "speaker_inversion_init_embedding": str(Path(fixture["init_checkpoint"]).resolve()),
        }
    )
    assert json.loads(config_path.read_text(encoding="utf-8")) == expected_config
    config_text = config_path.read_text(encoding="utf-8")
    assert '"adam_eps": 0.00000001' in config_text
    assert '"adam_eps": 1e-08' not in config_text

    predecessor_jobs = fixture["jobs_document"]
    successor_jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert [row["model_id"] for row in successor_jobs["jobs"]] == [
        row["model_id"] for row in predecessor_jobs["jobs"]
    ]
    target_index = next(
        index for index, row in enumerate(predecessor_jobs["jobs"]) if row["model_id"] == MODEL_ID
    )
    for index, (before, after) in enumerate(
        zip(predecessor_jobs["jobs"], successor_jobs["jobs"], strict=True)
    ):
        if index != target_index:
            assert after == before
    target = successor_jobs["jobs"][target_index]
    assert target["config"] == str(config_path.resolve())
    assert target["output_dir"] == str(output_dir.resolve())
    assert _command_argument(target["command"], "--config") == str(config_path.resolve())
    assert _command_argument(target["command"], "--output-dir") == str(output_dir.resolve())
    assert _command_argument(target["command"], "--manifest") == str(fixture["manifest"].resolve())
    assert _command_argument(target["command"], "--init-checkpoint") == str(
        fixture["base_checkpoint"].resolve()
    )
    assert str(fixture["predecessor_output"].resolve()) not in target["command"]
    assert fixture["jobs"].read_bytes() == source_jobs_bytes
    assert fixture["config"].read_bytes() == source_config_bytes
    assert fixture["base_checkpoint"].read_bytes() == base_checkpoint_bytes
    assert fixture["init_checkpoint"].read_bytes() == init_checkpoint_bytes

    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    assert setup["schema_version"] == f"speaker-quality-{kind}-setup/v1"
    assert setup["reason"]["source_best"]["checkpoint_step"] == 2500
    assert setup["paths"]["queue_script"] == str(fixture["queue_script"].resolve())
    assert setup["sha256"]["queue_script"] == _sha(fixture["queue_script"])
    assert setup["sha256"]["status_seed"] == hashlib.sha256(fixture["status_bytes"]).hexdigest()
    with pytest.raises(FileExistsError):
        module.prepare_quality_run(
            kind=kind,
            predecessor_jobs=fixture["jobs"],
            predecessor_status=fixture["status"],
            source_diagnostic=fixture["diagnostic"],
            model_id=MODEL_ID,
            init_checkpoint_step=2500,
            learning_rate=0.0003,
            seed=11,
            run_root=run_root,
            queue_script=fixture["queue_script"],
        )


def test_prepare_rejects_symlinked_input(tmp_path: Path) -> None:
    module = _load("quality_runs_prepare_symlink_test")
    fixture = _prepare_fixture(tmp_path)
    linked = tmp_path / "linked-jobs.json"
    linked.symlink_to(fixture["jobs"])

    with pytest.raises(ValueError, match="symlink"):
        module.prepare_quality_run(
            kind="search",
            predecessor_jobs=linked,
            predecessor_status=fixture["status"],
            source_diagnostic=fixture["diagnostic"],
            model_id=MODEL_ID,
            init_checkpoint_step=2500,
            learning_rate=0.0003,
            seed=11,
            run_root=fixture["quality_root"] / "search-v1",
            queue_script=fixture["queue_script"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_output", "duplicate_output", "manifest_drift", "base_checkpoint_drift"],
)
def test_prepare_rejects_remote_command_contract_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load(f"quality_runs_prepare_command_{mutation}_test")
    fixture = _prepare_fixture(tmp_path)
    jobs = json.loads(fixture["jobs"].read_text(encoding="utf-8"))
    target = next(job for job in jobs["jobs"] if job["model_id"] == MODEL_ID)
    command = target["command"]
    if mutation == "missing_output":
        index = command.index("--output-dir")
        del command[index : index + 2]
    elif mutation == "duplicate_output":
        command.extend(["--output-dir", str(fixture["predecessor_output"])])
    elif mutation == "manifest_drift":
        command[command.index("--manifest") + 1] = str(tmp_path / "alternate-manifest.jsonl")
    else:
        command[command.index("--init-checkpoint") + 1] = str(fixture["init_checkpoint"])
    fixture["jobs"].write_text(json.dumps(jobs), encoding="utf-8")

    with pytest.raises(ValueError, match=r"target command.*argument|path mismatch"):
        module.prepare_quality_run(
            kind="search",
            predecessor_jobs=fixture["jobs"],
            predecessor_status=fixture["status"],
            source_diagnostic=fixture["diagnostic"],
            model_id=MODEL_ID,
            init_checkpoint_step=2500,
            learning_rate=0.0003,
            seed=11,
            run_root=fixture["quality_root"] / "search-v1",
            queue_script=fixture["queue_script"],
        )


@pytest.mark.parametrize(
    "derived_input",
    ["config", "manifest", "base_checkpoint", "predecessor_output", "init_checkpoint"],
)
def test_prepare_rejects_alias_components_in_derived_paths(
    tmp_path: Path,
    derived_input: str,
) -> None:
    module = _load(f"quality_runs_prepare_derived_alias_{derived_input}_test")
    fixture = _prepare_fixture(tmp_path)
    jobs = json.loads(fixture["jobs"].read_text(encoding="utf-8"))
    target = next(job for job in jobs["jobs"] if job["model_id"] == MODEL_ID)
    predecessor_root = fixture["jobs"].parent
    alias_root = tmp_path / "predecessor-alias"
    alias_root.symlink_to(predecessor_root, target_is_directory=True)
    if derived_input == "config":
        target["config"] = str(alias_root / fixture["config"].name)
    elif derived_input == "manifest":
        target["clean_manifest"] = str(alias_root / fixture["manifest"].name)
    elif derived_input == "base_checkpoint":
        jobs["base_checkpoint_path"] = str(alias_root / "base.safetensors")
    elif derived_input == "predecessor_output":
        target["output_dir"] = str(alias_root / "target-output")
    else:
        init_checkpoint = fixture["init_checkpoint"]
        actual_checkpoint = tmp_path / init_checkpoint.name
        actual_checkpoint.write_bytes(init_checkpoint.read_bytes())
        init_checkpoint.unlink()
        init_checkpoint.symlink_to(actual_checkpoint)
    fixture["jobs"].write_text(json.dumps(jobs), encoding="utf-8")

    with pytest.raises(ValueError, match=r"symlink|alias|reparse"):
        module.prepare_quality_run(
            kind="search",
            predecessor_jobs=fixture["jobs"],
            predecessor_status=fixture["status"],
            source_diagnostic=fixture["diagnostic"],
            model_id=MODEL_ID,
            init_checkpoint_step=2500,
            learning_rate=0.0003,
            seed=11,
            run_root=fixture["quality_root"] / "search-v1",
            queue_script=fixture["queue_script"],
        )


def _prepared_completed_run(
    tmp_path: Path,
    *,
    kind: str,
) -> tuple[ModuleType, PreparedResult, CompletedFixture]:
    module = _load(f"quality_runs_finalize_fixture_{kind}_{tmp_path.name}")
    fixture = _prepare_fixture(tmp_path)
    run_root = Path(fixture["quality_root"]) / f"{kind}-v1"
    prepared = cast(
        "PreparedResult",
        module.prepare_quality_run(
            kind=kind,
            predecessor_jobs=fixture["jobs"],
            predecessor_status=fixture["status"],
            source_diagnostic=fixture["diagnostic"],
            model_id=MODEL_ID,
            init_checkpoint_step=2500,
            learning_rate=0.0003,
            seed=11,
            run_root=run_root,
            queue_script=fixture["queue_script"],
        ),
    )
    output_dir = Path(prepared["output_dir"])
    steps = [250] if kind == "search" else list(range(250, 3001, 250))
    checkpoint_bindings = []
    for step in steps:
        checkpoint = output_dir / f"checkpoint_{step:07d}.speaker.safetensors"
        _write_embedding(checkpoint, float(step))
        checkpoint_bindings.append({"path": str(checkpoint), "sha256": _sha(checkpoint)})
    final = output_dir / "checkpoint_final.speaker.safetensors"
    final.write_bytes((output_dir / f"checkpoint_{steps[-1]:07d}.speaker.safetensors").read_bytes())
    checkpoint_bindings.append({"path": str(final), "sha256": _sha(final)})

    log_dir = run_root / "logs"
    log_dir.mkdir()
    log = log_dir / f"{MODEL_ID}.log"
    loss_steps = list(range(20, 250, 20)) if kind == "search" else list(range(20, 3001, 20))
    finish_step = 250 if kind == "search" else 3000
    log.write_text(
        "".join(f"step={step} loss={1 / step:.8f}\n" for step in loss_steps)
        + f"Training finished at step={finish_step}.\n",
        encoding="utf-8",
    )

    jobs_document = json.loads(Path(prepared["jobs"]).read_text(encoding="utf-8"))
    target = next(job for job in jobs_document["jobs"] if job["model_id"] == MODEL_ID)
    config = Path(target["config"])
    manifest = Path(target["clean_manifest"])
    before = Path(prepared["status"]).read_bytes()
    common = {
        "model_id": MODEL_ID,
        "clean_manifest_sha256": _sha(manifest),
        "config_sha256": _sha(config),
        "checkpoint_sha256": jobs_document["base_checkpoint_sha256"],
        "checkpoint_revision": jobs_document["checkpoint_revision"],
        "upstream_commit": jobs_document["upstream_commit"],
        "started_at": "2026-08-02T02:00:00+00:00",
        "log_path": str(log),
    }
    started = common | {
        "event": "started",
        "status": "running",
        "ended_at": None,
        "exit_code": None,
        "last_checkpoint": None,
        "last_checkpoint_sha256": None,
        "candidate_checkpoints": [],
        "error": None,
    }
    periodic = output_dir / f"checkpoint_{steps[-1]:07d}.speaker.safetensors"
    finished = common | {
        "event": "finished",
        "status": "success",
        "ended_at": "2026-08-02T03:00:00+00:00",
        "exit_code": 0,
        "last_checkpoint": str(periodic),
        "last_checkpoint_sha256": _sha(periodic),
        "candidate_checkpoints": checkpoint_bindings,
        "error": None,
    }
    with Path(prepared["status"]).open("ab") as status:
        status.write(json.dumps(started, sort_keys=True).encode() + b"\n")
        status.write(json.dumps(finished, sort_keys=True).encode() + b"\n")
    runtime = run_root / "runtime-after.json"
    runtime.write_text(
        json.dumps(
            {
                "gpu_memory_used_mib": 900.0,
                "gpu_memory_total_mib": 12282.0,
                "gpu_utilization_percent": 0.0,
                "gpu_power_watts": 12.0,
                "active_training_processes": [],
            }
        ),
        encoding="utf-8",
    )
    completed = CompletedFixture(
        **fixture,
        before_status=before,
        started=started,
        finished=finished,
        runtime=runtime,
        log=log,
        output=run_root / "run-evidence.json",
    )
    return module, prepared, completed


@pytest.mark.parametrize("kind", ("search", "retrain"))
def test_finalize_emits_downstream_compatible_exact_evidence(
    tmp_path: Path,
    kind: str,
) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind=kind)

    evidence = module.finalize_quality_run(
        setup_evidence=Path(prepared["setup_evidence"]),
        training_jobs=Path(prepared["jobs"]),
        training_status=Path(prepared["status"]),
        queue_script=Path(fixture["queue_script"]),
        queue_exit_code=0,
        runtime_after=Path(fixture["runtime"]),
        output=Path(fixture["output"]),
    )

    expected_schema = f"speaker-quality-{kind}-run-evidence/v1"
    assert evidence["schema_version"] == expected_schema
    assert evidence["training_status"] == {
        "path": str(Path(prepared["status"]).resolve()),
        "before_row_count": 1,
        "before_sha256": hashlib.sha256(fixture["before_status"]).hexdigest(),
        "after_row_count": 3,
        "after_sha256": _sha(Path(prepared["status"])),
        "new_status_row_count": 2,
        "new_started_model_ids": [MODEL_ID],
        "new_finished_success_model_ids": [MODEL_ID],
    }
    assert json.loads(Path(fixture["output"]).read_text(encoding="utf-8")) == evidence

    if kind == "search":
        builder = _load_external(
            Path("scripts/build_600m_speaker_checkpoint_search_manifest.py"),
            f"quality_runs_search_downstream_{tmp_path.name}",
        )
        output_dir = Path(prepared["output_dir"])
        builder._validate_run_evidence(
            Path(fixture["output"]),
            model_id=MODEL_ID,
            run_id=Path(prepared["run_root"]).name,
            config_path=Path(prepared["config"]),
            config_sha256=_sha(Path(prepared["config"])),
            embedding_path=output_dir / "checkpoint_0000250.speaker.safetensors",
            embedding_sha256=_sha(output_dir / "checkpoint_0000250.speaker.safetensors"),
            base_checkpoint_sha256=json.loads(Path(prepared["jobs"]).read_text(encoding="utf-8"))[
                "base_checkpoint_sha256"
            ],
        )
    else:
        verifier = _load_external(
            Path("scripts/verify_600m_speaker_retraining_completion.py"),
            f"quality_runs_retrain_downstream_{tmp_path.name}",
        )
        jobs = json.loads(Path(prepared["jobs"]).read_text(encoding="utf-8"))
        target = next(row for row in jobs["jobs"] if row["model_id"] == MODEL_ID)
        predecessor_jobs = fixture["jobs_document"]
        predecessor_target = next(
            row for row in predecessor_jobs["jobs"] if row["model_id"] == MODEL_ID
        )
        setup_path = Path(prepared["setup_evidence"])
        verifier._validate_quality_setup(
            json.loads(setup_path.read_text(encoding="utf-8")),
            model_id=MODEL_ID,
            setup_path=setup_path,
            jobs_binding=verifier.FileBinding(Path(prepared["jobs"]), _sha(Path(prepared["jobs"]))),
            queue_binding=verifier.FileBinding(
                fixture["queue_script"], _sha(fixture["queue_script"])
            ),
            status_path=Path(prepared["status"]),
            before_sha=hashlib.sha256(fixture["before_status"]).hexdigest(),
            current_job=target,
            current_jobs_base=Path(prepared["jobs"]).parent,
            predecessor_job=predecessor_target,
            predecessor_jobs_base=fixture["jobs"].parent,
        )
        verifier._validate_quality_run_payload(
            evidence["run"],
            evidence["runtime_after"],
            model_id=MODEL_ID,
            started=fixture["started"],
            finished=fixture["finished"],
            config=Path(target["config"]),
            manifest=Path(target["clean_manifest"]),
            output_dir=Path(target["output_dir"]),
            base_sha=jobs["base_checkpoint_sha256"],
            launch_gpu_baseline=1000.0,
            gpu_memory_tolerance_mib=0.0,
        )
        verifier._validate_training_config(
            Path(target["config"]),
            output_dir=Path(target["output_dir"]),
            manifest=Path(target["clean_manifest"]),
        )


def test_atomic_create_only_tolerates_windows_directory_fsync_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load("quality_runs_windows_directory_fsync_test")
    output = tmp_path / "run-evidence.json"
    real_open = os.open

    def deny_directory_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if Path(path) == tmp_path and flags == os.O_RDONLY:
            raise PermissionError
        return real_open(path, flags, mode)

    monkeypatch.setattr(module.os, "open", deny_directory_open)

    module._write_atomic_create_only(output, b"evidence\n")

    assert output.read_bytes() == b"evidence\n"
    with pytest.raises(FileExistsError, match="overwrite"):
        module._write_atomic_create_only(output, b"replacement\n")


def _load_external(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("mutation", "message"), (("runtime", "active"), ("loss", "finite")))
def test_finalize_fails_closed_on_bad_runtime_or_loss(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind="search")
    if mutation == "runtime":
        runtime = json.loads(Path(fixture["runtime"]).read_text(encoding="utf-8"))
        runtime["active_training_processes"] = [{"pid": 123}]
        Path(fixture["runtime"]).write_text(json.dumps(runtime), encoding="utf-8")
    else:
        log = Path(fixture["log"])
        log.write_text(
            log.read_text(encoding="utf-8").replace("loss=0.05000000", "loss=nan"),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=message):
        module.finalize_quality_run(
            setup_evidence=Path(prepared["setup_evidence"]),
            training_jobs=Path(prepared["jobs"]),
            training_status=Path(prepared["status"]),
            queue_script=Path(fixture["queue_script"]),
            queue_exit_code=0,
            runtime_after=Path(fixture["runtime"]),
            output=Path(fixture["output"]),
        )
    assert not Path(fixture["output"]).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_checkpoint", "missing or extra"),
        ("final_mismatch", "checkpoint"),
        ("status_drift", "append drift"),
        ("oom", "OOM"),
        ("traceback", "traceback"),
    ],
)
def test_finalize_fails_closed_on_artifact_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind="search")
    if mutation == "extra_checkpoint":
        _write_embedding(Path(prepared["output_dir"]) / "checkpoint_0000500.speaker.safetensors")
    elif mutation == "final_mismatch":
        _write_embedding(Path(prepared["output_dir"]) / "checkpoint_final.speaker.safetensors", 99)
    elif mutation == "status_drift":
        with Path(prepared["status"]).open("ab") as status:
            status.write(b'{"event":"unexpected"}\n')
    elif mutation == "oom":
        with fixture["log"].open("a", encoding="utf-8") as log:
            log.write("CUDA out of memory\n")
    else:
        with fixture["log"].open("a", encoding="utf-8") as log:
            log.write("Traceback (most recent call last):\n")

    with pytest.raises(ValueError, match=message):
        module.finalize_quality_run(
            setup_evidence=Path(prepared["setup_evidence"]),
            training_jobs=Path(prepared["jobs"]),
            training_status=Path(prepared["status"]),
            queue_script=fixture["queue_script"],
            queue_exit_code=0,
            runtime_after=fixture["runtime"],
            output=fixture["output"],
        )
    assert not fixture["output"].exists()


def test_finalize_rejects_existing_output(tmp_path: Path) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind="search")
    fixture["output"].write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        module.finalize_quality_run(
            setup_evidence=Path(prepared["setup_evidence"]),
            training_jobs=Path(prepared["jobs"]),
            training_status=Path(prepared["status"]),
            queue_script=fixture["queue_script"],
            queue_exit_code=0,
            runtime_after=fixture["runtime"],
            output=fixture["output"],
        )
    assert fixture["output"].read_text(encoding="utf-8") == "operator-owned\n"


def test_finalize_rejects_output_beneath_symlinked_parent(tmp_path: Path) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind="search")
    run_root = Path(prepared["run_root"])
    alias_root = tmp_path / "run-root-alias"
    alias_root.symlink_to(run_root, target_is_directory=True)
    aliased_output = alias_root / fixture["output"].name

    with pytest.raises(ValueError, match=r"symlink|alias|reparse"):
        module.finalize_quality_run(
            setup_evidence=Path(prepared["setup_evidence"]),
            training_jobs=Path(prepared["jobs"]),
            training_status=Path(prepared["status"]),
            queue_script=fixture["queue_script"],
            queue_exit_code=0,
            runtime_after=fixture["runtime"],
            output=aliased_output,
        )

    assert not fixture["output"].exists()


def test_finalize_rejects_same_named_substituted_queue_script(tmp_path: Path) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind="search")
    substitute_root = tmp_path / "substitute"
    substitute_root.mkdir()
    substitute = substitute_root / "run_600m_speaker_training_queue.py"
    substitute.write_bytes(fixture["queue_script"].read_bytes())

    with pytest.raises(ValueError, match="setup queue script path mismatch"):
        module.finalize_quality_run(
            setup_evidence=Path(prepared["setup_evidence"]),
            training_jobs=Path(prepared["jobs"]),
            training_status=Path(prepared["status"]),
            queue_script=substitute,
            queue_exit_code=0,
            runtime_after=fixture["runtime"],
            output=fixture["output"],
        )
    assert not fixture["output"].exists()


@pytest.mark.parametrize(
    ("kind", "layer"),
    [
        ("retrain", "top_level"),
        ("retrain", "paths"),
        ("retrain", "sha256"),
        ("retrain", "reason"),
        ("retrain", "source_best"),
        ("retrain", "changes"),
        ("retrain", "learning_rate"),
        ("retrain", "seed"),
        ("search", "candidate"),
    ],
)
def test_finalize_rejects_extra_setup_contract_fields(
    tmp_path: Path,
    kind: str,
    layer: str,
) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind=kind)
    setup_path = Path(prepared["setup_evidence"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    if layer == "top_level":
        setup["unexpected"] = True
    elif layer == "source_best":
        setup["reason"]["source_best"]["unexpected"] = True
    elif layer in {"learning_rate", "seed"}:
        setup["changes"][layer]["unexpected"] = True
    else:
        setup[layer]["unexpected"] = True
    setup_path.write_text(json.dumps(setup), encoding="utf-8")

    with pytest.raises(ValueError, match="field set mismatch"):
        module.finalize_quality_run(
            setup_evidence=setup_path,
            training_jobs=Path(prepared["jobs"]),
            training_status=Path(prepared["status"]),
            queue_script=fixture["queue_script"],
            queue_exit_code=0,
            runtime_after=fixture["runtime"],
            output=fixture["output"],
        )

    assert not fixture["output"].exists()


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        ("--config", "alternate-config.json"),
        ("--output-dir", "alternate-output"),
        ("--manifest", "alternate-manifest.jsonl"),
        ("--init-checkpoint", "alternate-base.safetensors"),
    ],
)
def test_finalize_rejects_prepared_command_path_drift(
    tmp_path: Path,
    flag: str,
    replacement: str,
) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind="search")
    jobs_path = Path(prepared["jobs"])
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    target = next(job for job in jobs["jobs"] if job["model_id"] == MODEL_ID)
    command = target["command"]
    command[command.index(flag) + 1] = str(tmp_path / replacement)
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    setup_path = Path(prepared["setup_evidence"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["sha256"]["jobs"] = _sha(jobs_path)
    setup_path.write_text(json.dumps(setup), encoding="utf-8")

    with pytest.raises(ValueError, match=r"target command.*path mismatch"):
        module.finalize_quality_run(
            setup_evidence=setup_path,
            training_jobs=jobs_path,
            training_status=Path(prepared["status"]),
            queue_script=fixture["queue_script"],
            queue_exit_code=0,
            runtime_after=fixture["runtime"],
            output=fixture["output"],
        )

    assert not fixture["output"].exists()


def test_finalize_rechecks_snapshots_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, prepared, fixture = _prepared_completed_run(tmp_path, kind="search")
    original = module._recheck_snapshots

    def mutate_then_recheck(snapshots: tuple[object, ...]) -> None:
        with fixture["runtime"].open("a", encoding="utf-8") as runtime:
            runtime.write("\n")
        original(snapshots)

    monkeypatch.setattr(module, "_recheck_snapshots", mutate_then_recheck)
    with pytest.raises(ValueError, match="changed after snapshot"):
        module.finalize_quality_run(
            setup_evidence=Path(prepared["setup_evidence"]),
            training_jobs=Path(prepared["jobs"]),
            training_status=Path(prepared["status"]),
            queue_script=fixture["queue_script"],
            queue_exit_code=0,
            runtime_after=fixture["runtime"],
            output=fixture["output"],
        )
    assert not fixture["output"].exists()


def test_prepare_cli_only_prints_result_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load("quality_runs_prepare_cli_test")
    fixture = _prepare_fixture(tmp_path)
    run_root = Path(fixture["quality_root"]) / "search-cli-v1"

    result = module.main(
        [
            "prepare",
            "--kind",
            "search",
            "--predecessor-jobs",
            str(fixture["jobs"]),
            "--predecessor-status",
            str(fixture["status"]),
            "--source-diagnostic",
            str(fixture["diagnostic"]),
            "--model-id",
            MODEL_ID,
            "--init-checkpoint-step",
            "2500",
            "--learning-rate",
            "0.0003",
            "--seed",
            "11",
            "--run-root",
            str(run_root),
            "--queue-script",
            str(fixture["queue_script"]),
        ]
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["run_root"] == str(run_root.resolve())
    assert printed["kind"] == "search"


def test_prepare_cli_executes_as_a_real_process(tmp_path: Path) -> None:
    fixture = _prepare_fixture(tmp_path)
    run_root = fixture["quality_root"] / "search-process-v1"

    completed = subprocess.run(  # noqa: S603 - fixed local script under test.
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            "--kind",
            "search",
            "--predecessor-jobs",
            str(fixture["jobs"]),
            "--predecessor-status",
            str(fixture["status"]),
            "--source-diagnostic",
            str(fixture["diagnostic"]),
            "--model-id",
            MODEL_ID,
            "--init-checkpoint-step",
            "2500",
            "--learning-rate",
            "0.0003",
            "--seed",
            "11",
            "--run-root",
            str(run_root),
            "--queue-script",
            str(fixture["queue_script"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["run_root"] == str(run_root.resolve())
