from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from threading import Event
    from types import ModuleType
    from typing import Protocol

    class BenchmarkPlan(Protocol):
        candidate_id: str


pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/apply_600m_training_speed_selection.py")
BENCHMARK_SCRIPT_PATH = Path("scripts/run_600m_training_speed_benchmark.py")
MODEL_COUNT = 12
ANABEL_MODEL_ID = "oop77_anabel_maidgarden_sp_451488a7c1"
KASUMI_MODEL_ID = "kasumi"
BENCHMARK_MODEL_ID = "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
COMPLETED_MODEL_IDS = (ANABEL_MODEL_ID, KASUMI_MODEL_ID)
SPEED_CONFIG_NAME = "training-config-speed-v1.json"
RECOMMENDED_BATCH_SIZE = 2
RECOMMENDED_ACCUMULATION_STEPS = 8
EFFECTIVE_GLOBAL_BATCH_SIZE = 16
CHANGED_MODEL_COUNT = 10
INJECTED_FAILURE_LINK_NUMBER = 3

BenchmarkMutation = Callable[[dict[str, Any]], object]


@dataclass(slots=True)
class Fixture:
    benchmark_summary: Path
    training_jobs: Path
    training_status: Path
    output_jobs: Path
    status_output: Path
    model_ids: list[str]
    original_jobs: dict[str, Any]
    original_configs: dict[str, dict[str, Any]]
    original_config_texts: dict[str, str]


FixtureMutation = Callable[[Fixture], object]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "apply_600m_training_speed_selection",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_600m_training_speed_benchmark_for_apply_test",
        BENCHMARK_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_summary(*, mean: float, maximum: float) -> dict[str, float | int]:
    return {"sample_count": 10, "minimum": mean - 1.0, "mean": mean, "maximum": maximum}


def _candidate(
    candidate_id: str,
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    optimizer_steps_per_second: float,
) -> dict[str, Any]:
    overrides = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
    }
    return {
        "id": candidate_id,
        **overrides,
        "effective_global_batch_size": batch_size * gradient_accumulation_steps,
        "overrides": overrides,
        "metrics": {
            "measured_optimizer_steps": 50,
            "steady_optimizer_steps_per_second": optimizer_steps_per_second,
            "steady_samples_per_second": optimizer_steps_per_second * 16,
            "peak_vram_mib": 9_850.5,
            "gpu_utilization_percent": _metric_summary(mean=95.0, maximum=99.0),
            "power_watts": _metric_summary(mean=180.0, maximum=205.0),
            "loss_finite": True,
            "oom": False,
            "exit_code": 0,
            "eligible": True,
            "ineligible_reasons": [],
        },
    }


def _failed_candidate() -> dict[str, Any]:
    empty_stats = {"sample_count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "id": "A",
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "effective_global_batch_size": 16,
        "overrides": {
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
        },
        "metrics": {
            "measured_optimizer_steps": 0,
            "steady_optimizer_steps_per_second": None,
            "steady_samples_per_second": None,
            "peak_vram_mib": None,
            "gpu_utilization_percent": empty_stats,
            "power_watts": empty_stats,
            "loss_finite": False,
            "oom": True,
            "exit_code": None,
            "eligible": False,
            "ineligible_reasons": ["oom"],
        },
    }


def _source_config_json(payload: dict[str, Any]) -> str:
    document = json.dumps(payload)
    replacements = {
        '"norm_eps": 1e-05': '"norm_eps": 0.00001',
        '"adam_eps": 1e-08': '"adam_eps": 0.00000001',
        '"positive_exponent_float": 1e+20': ('"positive_exponent_float": 100000000000000000000.0'),
    }
    for emitted, yaml_numeric in replacements.items():
        assert emitted in document
        document = document.replace(emitted, yaml_numeric)
    return document


def _write_fixture(tmp_path: Path) -> Fixture:  # noqa: PLR0914, PLR0915 - complete fixture.
    queue_root = tmp_path / "queue"
    queue_root.mkdir(parents=True)
    training_status = queue_root / "training-status.jsonl"
    model_ids = [
        *COMPLETED_MODEL_IDS,
        BENCHMARK_MODEL_ID,
        *(f"model-{index:02d}" for index in range(3, 12)),
    ]
    jobs: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    original_configs: dict[str, dict[str, Any]] = {}
    original_config_texts: dict[str, str] = {}
    for model_id in model_ids:
        manifest = queue_root / "datasets" / model_id / "clean-manifest.jsonl"
        config = manifest.parent / "training-config-v1.json"
        output_dir = queue_root / "training" / model_id
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"source_id":"source-1"}\n', encoding="utf-8")
        if model_id in COMPLETED_MODEL_IDS:
            output_dir.mkdir(parents=True)
        config_payload: dict[str, Any] = {
            "model": {
                "latent_dim": 32,
                "model_dim": 1024,
                "norm_eps": 1e-5,
                "positive_exponent_float": 1e20,
                "scientific_notation_label": "1e-05",
                "use_caption_condition": True,
                "use_speaker_condition": True,
            },
            "train": {
                "manifest_path": str(manifest.resolve()),
                "output_dir": str(output_dir.resolve()),
                "batch_size": 16,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "precision": "bf16",
                "allow_tf32": True,
                "adam_eps": 1e-8,
                "learning_rate": 0.01,
                "lr_scheduler": "none",
                "compile_model": False,
                "max_latent_steps": 750,
                "seed": 0,
                "max_steps": 3000,
            },
        }
        config_text = _source_config_json(config_payload)
        config.write_text(config_text, encoding="utf-8")
        original_configs[model_id] = config_payload
        original_config_texts[model_id] = config_text
        jobs.append(
            {
                "model_id": model_id,
                "clean_manifest": str(manifest.relative_to(queue_root)),
                "config": str(config.relative_to(queue_root)),
                "output_dir": str(output_dir.relative_to(queue_root)),
                "command": [
                    "python",
                    "train.py",
                    "--config",
                    str(config.resolve()),
                    "--model-id",
                    model_id,
                ],
            },
        )
        if model_id in COMPLETED_MODEL_IDS:
            checkpoint = output_dir / "checkpoint-3000" / f"{model_id}.speaker.safetensors"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(f"speaker:{model_id}".encode())
            status_base: dict[str, object] = {
                "model_id": model_id,
                "clean_manifest_sha256": _sha256(manifest),
                "checkpoint_sha256": "pending-base-checkpoint-hash",
                "checkpoint_revision": "c" * 40,
                "config_sha256": _sha256(config),
                "upstream_commit": "e" * 40,
                "started_at": "2026-08-01T00:00:00+00:00",
                "ended_at": None,
                "exit_code": None,
                "log_path": str((queue_root / "logs" / f"{model_id}.log").resolve()),
                "last_checkpoint": None,
                "last_checkpoint_sha256": None,
                "candidate_checkpoints": [],
                "error": None,
            }
            status_rows.extend(
                (
                    status_base | {"event": "started", "status": "running"},
                    status_base
                    | {
                        "event": "finished",
                        "status": "success",
                        "ended_at": "2026-08-01T00:30:00+00:00",
                        "exit_code": 0,
                        "last_checkpoint": str(checkpoint.resolve()),
                        "last_checkpoint_sha256": _sha256(checkpoint),
                        "candidate_checkpoints": [
                            {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)}
                        ],
                    },
                )
            )
    base_checkpoint = queue_root / "models" / "base.safetensors"
    base_checkpoint.parent.mkdir()
    base_checkpoint.write_bytes(b"600m base checkpoint")
    for row in status_rows:
        row["checkpoint_sha256"] = _sha256(base_checkpoint)
    training_jobs = queue_root / "training-jobs.json"
    jobs_document: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": "2026-08-02T00:00:00+00:00",
        "queue_policy": {"execution": "sequential"},
        "base_checkpoint_path": str(base_checkpoint.relative_to(queue_root)),
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "checkpoint_revision": "c" * 40,
        "upstream_commit": "e" * 40,
        "jobs": jobs,
    }
    training_jobs.write_text(json.dumps(jobs_document), encoding="utf-8")
    training_status.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in status_rows),
        encoding="utf-8",
    )
    pilot_log = queue_root / "benchmark" / "pilot.log"
    pilot_log.parent.mkdir()
    pilot_log.write_text("finite benchmark log\n", encoding="utf-8")
    benchmark_script = queue_root / "benchmark" / "benchmark.py"
    benchmark_script.write_text("# benchmark fixture\n", encoding="utf-8")
    upstream_root = queue_root / "upstream"
    upstream_root.mkdir()
    trainer_python = upstream_root / ".venv" / "Scripts" / "python.exe"
    trainer_script = upstream_root / "train.py"
    trainer_python.parent.mkdir(parents=True)
    trainer_script.parent.mkdir(parents=True, exist_ok=True)
    trainer_python.write_bytes(b"python fixture")
    trainer_script.write_text("# trainer fixture\n", encoding="utf-8")
    candidates = [
        _failed_candidate(),
        _candidate(
            "B",
            batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optimizer_steps_per_second=0.9,
        ),
        _candidate(
            "C",
            batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optimizer_steps_per_second=0.7,
        ),
        _candidate(
            "D",
            batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=False,
            optimizer_steps_per_second=1.25,
        ),
    ]
    benchmark_summary = queue_root / "benchmark" / "summary.json"
    benchmark_summary.write_text(
        json.dumps(
            {
                "schema_version": "speaker-training-speed-benchmark/v1",
                "status": "PASS",
                "recommended_candidate": candidates[3],
                "candidates": candidates,
                "provenance": {
                    "base_config": {
                        "path": str(
                            (
                                queue_root
                                / "datasets"
                                / BENCHMARK_MODEL_ID
                                / "training-config-v1.json"
                            ).resolve(),
                        ),
                        "sha256": _sha256(
                            queue_root
                            / "datasets"
                            / BENCHMARK_MODEL_ID
                            / "training-config-v1.json",
                        ),
                    },
                    "manifest": {
                        "path": str(
                            (
                                queue_root
                                / "datasets"
                                / BENCHMARK_MODEL_ID
                                / "clean-manifest.jsonl"
                            ).resolve(),
                        ),
                        "sha256": _sha256(
                            queue_root / "datasets" / BENCHMARK_MODEL_ID / "clean-manifest.jsonl",
                        ),
                    },
                    "base_checkpoint": {
                        "path": str(base_checkpoint.resolve()),
                        "sha256": _sha256(base_checkpoint),
                    },
                    "script": {
                        "path": str(benchmark_script.resolve()),
                        "sha256": _sha256(benchmark_script),
                    },
                    "training_jobs": {
                        "path": str(training_jobs.resolve()),
                        "sha256": _sha256(training_jobs),
                    },
                    "training_status": {
                        "path": str(training_status.resolve()),
                        "sha256": _sha256(training_status),
                    },
                    "upstream": {
                        "root": str(upstream_root.resolve()),
                        "commit": "e" * 40,
                        "trainer_python": {
                            "path": str(trainer_python.resolve()),
                            "sha256": _sha256(trainer_python),
                        },
                        "trainer_script": {
                            "path": str(trainer_script.resolve()),
                            "sha256": _sha256(trainer_script),
                        },
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    return Fixture(
        benchmark_summary=benchmark_summary,
        training_jobs=training_jobs,
        training_status=training_status,
        output_jobs=queue_root / "training-jobs-speed-v1.json",
        status_output=queue_root / "status" / "speed-selection-v1.json",
        model_ids=model_ids,
        original_jobs=jobs_document,
        original_configs=original_configs,
        original_config_texts=original_config_texts,
    )


def _argv(
    fixture: Fixture,
    *,
    completed_model_ids: tuple[str, str] = COMPLETED_MODEL_IDS,
) -> list[str]:
    args = [
        "--benchmark-summary",
        str(fixture.benchmark_summary),
        "--training-jobs",
        str(fixture.training_jobs),
        "--output-jobs",
        str(fixture.output_jobs),
        "--status-output",
        str(fixture.status_output),
    ]
    for model_id in completed_model_ids:
        args.extend(("--completed-model-id", model_id))
    return args


def _assert_yaml_scalar_types_preserved(
    *,
    source_path: Path,
    selected_path: Path,
    expected_source_text: str,
) -> None:
    source_text = source_path.read_text(encoding="utf-8")
    assert source_text == expected_source_text
    source_yaml = yaml.safe_load(source_text)
    selected_yaml = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    for section_name in ("model", "train"):
        for field, source_value in source_yaml[section_name].items():
            if section_name == "train" and field in {
                "batch_size",
                "gradient_accumulation_steps",
                "gradient_checkpointing",
            }:
                continue
            assert selected_yaml[section_name][field] == source_value
            assert type(selected_yaml[section_name][field]) is type(source_value)
    assert type(selected_yaml["model"]["norm_eps"]) is float
    assert type(selected_yaml["train"]["adam_eps"]) is float
    assert type(selected_yaml["model"]["positive_exponent_float"]) is float
    assert selected_yaml["model"]["scientific_notation_label"] == "1e-05"


def test_main_applies_recommended_speed_config_to_exactly_ten_pending_jobs(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)

    assert module.main(_argv(fixture)) == 0

    output_document = json.loads(fixture.output_jobs.read_text(encoding="utf-8"))
    assert [row["model_id"] for row in output_document["jobs"]] == fixture.model_ids
    assert output_document["schema_version"] == 1
    assert type(output_document["schema_version"]) is int
    output_by_id = {row["model_id"]: row for row in output_document["jobs"]}
    original_by_id = {row["model_id"]: row for row in fixture.original_jobs["jobs"]}
    for model_id in COMPLETED_MODEL_IDS:
        assert output_by_id[model_id] == original_by_id[model_id]
    for model_id in fixture.model_ids[2:]:
        source = original_by_id[model_id]
        selected = output_by_id[model_id]
        speed_path = (
            fixture.training_jobs.parent / str(source["clean_manifest"])
        ).parent / SPEED_CONFIG_NAME
        assert selected["config"] == str(speed_path.relative_to(fixture.training_jobs.parent))
        expected_command = list(source["command"])
        config_index = expected_command.index("--config") + 1
        expected_command[config_index] = str(speed_path.resolve())
        assert selected["command"] == expected_command
        assert selected["output_dir"] == source["output_dir"]
        assert speed_path.is_file()
        _assert_yaml_scalar_types_preserved(
            source_path=fixture.training_jobs.parent / str(source["config"]),
            selected_path=speed_path,
            expected_source_text=fixture.original_config_texts[model_id],
        )
        speed_config = json.loads(speed_path.read_text(encoding="utf-8"))
        assert speed_config["model"] == fixture.original_configs[model_id]["model"]
        changed_train = speed_config["train"]
        original_train = fixture.original_configs[model_id]["train"]
        assert {
            key: value
            for key, value in changed_train.items()
            if key
            not in {
                "batch_size",
                "gradient_accumulation_steps",
                "gradient_checkpointing",
            }
        } == {
            key: value
            for key, value in original_train.items()
            if key
            not in {
                "batch_size",
                "gradient_accumulation_steps",
                "gradient_checkpointing",
            }
        }
        assert changed_train["batch_size"] == RECOMMENDED_BATCH_SIZE
        assert changed_train["gradient_accumulation_steps"] == RECOMMENDED_ACCUMULATION_STEPS
        assert changed_train["gradient_checkpointing"] is False
        assert (
            changed_train["batch_size"] * changed_train["gradient_accumulation_steps"]
            == EFFECTIVE_GLOBAL_BATCH_SIZE
        )

    status = json.loads(fixture.status_output.read_text(encoding="utf-8"))
    assert status["schema_version"] == "speaker-training-speed-selection/v1"
    assert status["status"] == "PASS"
    assert status["completed_model_ids"] == list(COMPLETED_MODEL_IDS)
    assert status["changed_model_count"] == CHANGED_MODEL_COUNT
    assert [row["model_id"] for row in status["changes"]] == fixture.model_ids[2:]
    assert status["benchmark_binding"] == {
        "path": str(fixture.benchmark_summary.resolve()),
        "sha256": _sha256(fixture.benchmark_summary),
        "schema_version": "speaker-training-speed-benchmark/v1",
        "recommended_candidate_id": "D",
    }
    assert status["inputs"]["training_jobs"]["sha256"] == _sha256(fixture.training_jobs)
    assert status["inputs"]["training_status"] == {
        "path": str(fixture.training_status.resolve()),
        "sha256": _sha256(fixture.training_status),
    }
    base_checkpoint = fixture.training_jobs.parent / fixture.original_jobs["base_checkpoint_path"]
    assert status["inputs"]["base_checkpoint"] == {
        "path": str(base_checkpoint.resolve()),
        "sha256": _sha256(base_checkpoint),
    }
    assert set(status["inputs"]["configs"]) == set(fixture.model_ids)
    assert set(status["inputs"]["clean_manifests"]) == set(fixture.model_ids)
    assert status["outputs"]["training_jobs"] == {
        "path": str(fixture.output_jobs.resolve()),
        "sha256": _sha256(fixture.output_jobs),
    }
    assert set(status["outputs"]["configs"]) == set(fixture.model_ids[2:])
    assert all(
        row["changed_fields"]
        == {
            "batch_size": {"before": 16, "after": 2},
            "gradient_accumulation_steps": {"before": 1, "after": 8},
            "gradient_checkpointing": {"before": True, "after": False},
        }
        for row in status["changes"]
    )
    assert not any(fixture.output_jobs.parent.glob(".speed-selection-stage-*"))


def test_real_benchmark_summary_is_accepted_by_apply(tmp_path: Path) -> None:
    apply_module = _load_script()
    benchmark_module = _load_benchmark_script()
    fixture = _write_fixture(tmp_path)
    queue_root = fixture.training_jobs.parent
    benchmark_job = fixture.original_jobs["jobs"][2]
    base_config = queue_root / benchmark_job["config"]
    manifest = queue_root / benchmark_job["clean_manifest"]
    base_checkpoint = queue_root / fixture.original_jobs["base_checkpoint_path"]
    upstream_root = queue_root / "upstream"
    benchmark_output = queue_root / "benchmark-produced"
    base_time = datetime(2026, 8, 2, tzinfo=UTC)
    seconds_per_step = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}

    def runner(
        plan: BenchmarkPlan,
        _command: tuple[str, ...],
        emit: Callable[[str, str | None], None],
    ) -> int:
        interval = seconds_per_step[plan.candidate_id]
        for step in range(1, 61):
            emit(
                f"step={step} loss=0.5 rf=0.1 dur=0.2 dur_mae=0.3 lr=1.000e-02",
                (base_time + timedelta(seconds=step * interval)).isoformat(),
            )
        return 0

    def sampler(plan: BenchmarkPlan, path: Path, _interval: float, _stop: Event) -> None:
        peak = {"A": 9_000.0, "B": 9_100.0, "C": 9_200.0, "D": 9_300.0}[plan.candidate_id]
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(benchmark_module.GPU_HEADER)
            for sample_index in range(141):
                timestamp = base_time + timedelta(seconds=sample_index / 2)
                writer.writerow(
                    (
                        timestamp.isoformat(),
                        0,
                        95,
                        peak,
                        180,
                        65,
                        "",
                        f"0, 95, {peak}, 180, 65",
                    )
                )

    def inspect_git(_root: Path) -> dict[str, object]:
        return {
            "head": fixture.original_jobs["upstream_commit"],
            "tracked_worktree_clean": True,
            "index_clean": True,
            "untracked_files": [],
        }

    def inspect_environment() -> dict[str, object]:
        return {
            "python": {"version": "3.10.20", "executable": "python.exe"},
            "torch": {"version": "2.7.1", "cuda_version": "12.8"},
            "gpu": {
                "name": "NVIDIA GeForce RTX 4070",
                "uuid": "GPU-test",
                "driver_version": "576.80",
                "memory_total_mib": 12_282.0,
                "power_limit_w": 200.0,
                "error": None,
            },
        }

    def probe_runtime(_plan: BenchmarkPlan, _phase: str) -> dict[str, object]:
        return {
            "timestamp": base_time.isoformat(),
            "matching_processes": [],
            "gpu_memory_used_mib": 1_000.0,
            "errors": [],
        }

    benchmark = benchmark_module.run_benchmark(
        base_config=base_config,
        manifest=manifest,
        base_checkpoint=base_checkpoint,
        upstream_root=upstream_root,
        upstream_commit=fixture.original_jobs["upstream_commit"],
        training_jobs=fixture.training_jobs,
        training_status=fixture.training_status,
        output_root=benchmark_output,
        runner=runner,
        sampler=sampler,
        git_inspector=inspect_git,
        environment_inspector=inspect_environment,
        runtime_probe=probe_runtime,
        cleanup_timeout=0.0,
        cleanup_poll_interval=0.01,
    )

    assert benchmark["status"] == "PASS"
    assert benchmark["recommended_candidate"]["id"] == "D"
    argv = _argv(fixture)
    argv[argv.index("--benchmark-summary") + 1] = str(benchmark_output / "summary.json")

    assert apply_module.main(argv) == 0
    assert fixture.output_jobs.is_file()
    assert fixture.status_output.is_file()


def test_main_rejects_wrong_completed_model_ids_before_writing(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="required completed model ids"):
        module.main(
            _argv(
                fixture,
                completed_model_ids=(ANABEL_MODEL_ID, BENCHMARK_MODEL_ID),
            ),
        )

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()
    assert not any(
        path.name == SPEED_CONFIG_NAME
        for path in fixture.training_jobs.parent.rglob(SPEED_CONFIG_NAME)
    )


@pytest.mark.parametrize("schema_version", ["speaker-training-queue/v1", True])
def test_main_requires_exact_numeric_training_jobs_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    fixture.original_jobs["schema_version"] = schema_version
    fixture.training_jobs.write_text(json.dumps(fixture.original_jobs), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version 1"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(status="REJECTED"), "status PASS"),
        (
            lambda payload: payload["recommended_candidate"]["metrics"].update(
                peak_vram_mib=10_501.0,
            ),
            "peak_vram_mib",
        ),
        (
            lambda payload: payload["recommended_candidate"]["metrics"].update(
                loss_finite=False,
            ),
            "loss_finite",
        ),
        (
            lambda payload: payload["recommended_candidate"].update(
                effective_global_batch_size=8,
            ),
            "effective_global_batch_size",
        ),
    ],
)
def test_main_rejects_malformed_or_ineligible_benchmark_before_writing(
    tmp_path: Path,
    mutation: BenchmarkMutation,
    message: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    mutation(payload)
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()
    assert not any(
        path.name == SPEED_CONFIG_NAME
        for path in fixture.training_jobs.parent.rglob(SPEED_CONFIG_NAME)
    )


def test_main_rejects_benchmark_training_jobs_binding_that_differs_from_cli(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    alternate_jobs = fixture.training_jobs.with_name("alternate-training-jobs.json")
    alternate_jobs.write_bytes(fixture.training_jobs.read_bytes())
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    payload["provenance"]["training_jobs"] = {
        "path": str(alternate_jobs.resolve()),
        "sha256": _sha256(alternate_jobs),
    }
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match --training-jobs"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


@pytest.mark.parametrize("binding_name", ["training_jobs", "training_status"])
def test_main_rejects_benchmark_input_binding_hash_tampering(
    tmp_path: Path,
    binding_name: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    payload["provenance"][binding_name]["sha256"] = "0" * 64
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"benchmark {binding_name} SHA-256"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


def test_main_rejects_missing_benchmark_training_status_binding_path(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    payload["provenance"]["training_status"]["path"] = str(
        fixture.training_status.with_name("missing-training-status.jsonl"),
    )
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark training_status path does not exist"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


def test_main_recomputes_recommended_candidate_ranking(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    payload["recommended_candidate"] = payload["candidates"][1]
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fastest eligible candidate"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


def test_main_requires_complete_abcd_candidate_set(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    payload["candidates"] = [
        candidate for candidate in payload["candidates"] if candidate["id"] != "C"
    ]
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly A, B, C, D"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


def test_main_requires_each_candidate_id_to_match_its_fixed_spec(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    by_id = {candidate["id"]: candidate for candidate in payload["candidates"]}
    by_id["B"]["id"] = "C"
    by_id["C"]["id"] = "B"
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed spec"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


@pytest.mark.parametrize("telemetry_field", ["gpu_utilization_percent", "power_watts"])
def test_main_requires_ten_samples_for_eligible_telemetry(
    tmp_path: Path,
    telemetry_field: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    selected = next(candidate for candidate in payload["candidates"] if candidate["id"] == "D")
    selected["metrics"][telemetry_field]["sample_count"] = 9
    payload["recommended_candidate"] = selected
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="at least 10 samples"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


def test_main_recomputes_peak_vram_and_id_tiebreaks(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    by_id = {candidate["id"]: candidate for candidate in payload["candidates"]}
    selected = by_id["D"]
    selected["metrics"]["steady_optimizer_steps_per_second"] = 0.9
    selected["metrics"]["steady_samples_per_second"] = 14.4
    selected["metrics"]["peak_vram_mib"] = 9_000.0
    payload["recommended_candidate"] = selected
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    assert module.main(_argv(fixture)) == 0

    second_fixture = _write_fixture(tmp_path / "id-tie")
    tied_payload = json.loads(second_fixture.benchmark_summary.read_text(encoding="utf-8"))
    tied_by_id = {candidate["id"]: candidate for candidate in tied_payload["candidates"]}
    tied_selected = tied_by_id["D"]
    tied_challenger = tied_by_id["B"]
    for candidate in (tied_selected, tied_challenger):
        candidate["metrics"]["steady_optimizer_steps_per_second"] = 0.9
        candidate["metrics"]["steady_samples_per_second"] = 14.4
        candidate["metrics"]["peak_vram_mib"] = 9_000.0
    tied_payload["recommended_candidate"] = tied_selected
    second_fixture.benchmark_summary.write_text(json.dumps(tied_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fastest eligible candidate"):
        module.main(_argv(second_fixture))


def test_benchmark_config_and_manifest_must_bind_one_pending_job(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    payload = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    completed_job = fixture.original_jobs["jobs"][0]
    completed_config = fixture.training_jobs.parent / completed_job["config"]
    completed_manifest = fixture.training_jobs.parent / completed_job["clean_manifest"]
    payload["provenance"]["base_config"] = {
        "path": str(completed_config.resolve()),
        "sha256": _sha256(completed_config),
    }
    payload["provenance"]["manifest"] = {
        "path": str(completed_manifest.resolve()),
        "sha256": _sha256(completed_manifest),
    }
    fixture.benchmark_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pending model"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


def test_main_requires_source_and_speed_configs_to_share_a_directory(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    pending_job = fixture.original_jobs["jobs"][2]
    source_config = fixture.training_jobs.parent / pending_job["config"]
    relocated_config = fixture.training_jobs.parent / "configs" / "benchmark-config.json"
    relocated_config.parent.mkdir()
    relocated_config.write_bytes(source_config.read_bytes())
    pending_job["config"] = str(relocated_config.relative_to(fixture.training_jobs.parent))
    config_index = pending_job["command"].index("--config") + 1
    pending_job["command"][config_index] = str(relocated_config.resolve())
    fixture.training_jobs.write_text(json.dumps(fixture.original_jobs), encoding="utf-8")
    benchmark = json.loads(fixture.benchmark_summary.read_text(encoding="utf-8"))
    benchmark["provenance"]["base_config"] = {
        "path": str(relocated_config.resolve()),
        "sha256": _sha256(relocated_config),
    }
    benchmark["provenance"]["training_jobs"]["sha256"] = _sha256(fixture.training_jobs)
    fixture.benchmark_summary.write_text(json.dumps(benchmark), encoding="utf-8")

    with pytest.raises(ValueError, match="same directory"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()
    assert not any(
        path.name == SPEED_CONFIG_NAME
        for path in fixture.training_jobs.parent.rglob(SPEED_CONFIG_NAME)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda fixture: fixture.original_jobs["jobs"].pop(), "exactly 12"),
        (
            lambda fixture: fixture.original_jobs["jobs"][2].update(
                command=["python", "train.py", "--model-id", BENCHMARK_MODEL_ID],
            ),
            "--config",
        ),
        (
            lambda fixture: fixture.original_configs[BENCHMARK_MODEL_ID]["train"].update(
                learning_rate=0.001,
            ),
            "learning_rate",
        ),
    ],
)
def test_main_rejects_invalid_training_job_contract(
    tmp_path: Path,
    mutation: FixtureMutation,
    message: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    mutation(fixture)
    for row in fixture.original_jobs["jobs"]:
        model_id = row["model_id"]
        config_path = fixture.training_jobs.parent / str(row["config"])
        config_path.write_text(json.dumps(fixture.original_configs[model_id]), encoding="utf-8")
    fixture.training_jobs.write_text(json.dumps(fixture.original_jobs), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()


@pytest.mark.parametrize("collision", ["config", "output_jobs", "status_output"])
def test_main_preflight_collision_publishes_nothing(
    tmp_path: Path,
    collision: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    if collision == "config":
        collision_path = (
            fixture.training_jobs.parent / fixture.original_jobs["jobs"][5]["clean_manifest"]
        ).parent / SPEED_CONFIG_NAME
    else:
        collision_path = getattr(fixture, collision)
        collision_path.parent.mkdir(parents=True, exist_ok=True)
    collision_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main(_argv(fixture))

    assert collision_path.read_text(encoding="utf-8") == "existing"
    created_speed_configs = list(
        fixture.training_jobs.parent.rglob(SPEED_CONFIG_NAME),
    )
    assert created_speed_configs == ([collision_path] if collision == "config" else [])
    if collision != "output_jobs":
        assert not fixture.output_jobs.exists()
    if collision != "status_output":
        assert not fixture.status_output.exists()


def test_publish_failure_rolls_back_all_new_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    original_link = module.os.link
    calls = 0

    def fail_third_link(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == INJECTED_FAILURE_LINK_NUMBER:
            message = "injected atomic publish failure"
            raise OSError(message)
        original_link(source, target)

    monkeypatch.setattr(module.os, "link", fail_third_link)

    with pytest.raises(OSError, match="injected atomic publish failure"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()
    assert not any(
        path.name == SPEED_CONFIG_NAME
        for path in fixture.training_jobs.parent.rglob(SPEED_CONFIG_NAME)
    )
    assert not any(fixture.output_jobs.parent.glob(".speed-selection-stage-*"))


def test_input_drift_before_publish_leaves_no_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    original_write_json = module._write_json  # noqa: SLF001 - inject drift at publish boundary.
    pending_job = fixture.original_jobs["jobs"][2]
    source_config = fixture.training_jobs.parent / pending_job["config"]

    def mutate_after_status(path: Path, payload: dict[str, object]) -> None:
        original_write_json(path, payload)
        if path.name == "status.json":
            source_config.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "_write_json", mutate_after_status)

    with pytest.raises(ValueError, match="input changed during speed selection"):
        module.main(_argv(fixture))

    assert not fixture.output_jobs.exists()
    assert not fixture.status_output.exists()
    assert not any(
        path.name == SPEED_CONFIG_NAME
        for path in fixture.training_jobs.parent.rglob(SPEED_CONFIG_NAME)
    )
