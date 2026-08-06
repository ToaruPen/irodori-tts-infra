# ruff: noqa: ANN001, ANN003, ANN202, ARG001, PLR2004
# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Callable
    from threading import Event
    from types import ModuleType
    from typing import Any

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/run_600m_training_speed_benchmark.py")
UPSTREAM_COMMIT = "eaf74d6a19138f743acb5b71a445fd25a57db987"
CHECKPOINT_REVISION = "e863a3a93e652e09afeff3e84823a206a0a60314"
ANABEL_MODEL_ID = "oop77_anabel_maidgarden_sp_451488a7c1"
KASUMI_MODEL_ID = "kasumi"
NEXT_MODEL_ID = "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
MODEL_IDS = (
    ANABEL_MODEL_ID,
    KASUMI_MODEL_ID,
    NEXT_MODEL_ID,
    *(f"pending-{index:02d}" for index in range(3, 12)),
)
CANDIDATE_IDS = ("A", "B", "C", "D")
GLOBAL_BATCH_SIZE = 16
MEASUREMENT_STEPS = 50
PEAK_LIMIT_MIB = 10_500.0


def _scalar_types(
    value: object,
    path: tuple[str | int, ...] = (),
) -> dict[tuple[str | int, ...], type[Any]]:
    if isinstance(value, dict):
        result: dict[tuple[str | int, ...], type[Any]] = {}
        for key, nested in value.items():
            result.update(_scalar_types(nested, (*path, str(key))))
        return result
    if isinstance(value, list):
        result = {}
        for index, nested in enumerate(value):
            result.update(_scalar_types(nested, (*path, index)))
        return result
    return {path: type(value)}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_600m_training_speed_benchmark",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> dict[str, Path]:  # noqa: PLR0914 - complete queue fixture.
    base_config = tmp_path / "inputs" / "training-config.json"
    manifest = tmp_path / "inputs" / "clean-manifest.jsonl"
    checkpoint = tmp_path / "inputs" / "model.safetensors"
    upstream = tmp_path / "upstream"
    trainer_python = upstream / ".venv" / "Scripts" / "python.exe"
    trainer_script = upstream / "train.py"
    base_config.parent.mkdir(parents=True)
    trainer_python.parent.mkdir(parents=True)
    base_config_payload = json.dumps(
        {
            "model": {
                "model_dim": 1024,
                "layers": 24,
                "norm_eps": 0.00001,
                "positive_float": 1e20,
                "scientific_text": "1e-08",
            },
            "data": {"latent_cache": "fixed"},
            "train": {
                "manifest_path": "old-manifest.jsonl",
                "output_dir": "old-output",
                "batch_size": 16,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "allow_tf32": False,
                "precision": "fp32",
                "compile_model": True,
                "learning_rate": 0.0001,
                "adam_eps": 0.00000001,
                "lr_scheduler": "cosine",
                "max_steps": 3000,
                "log_every": 20,
                "save_every": 250,
                "max_latent_steps": 750,
                "seed": 0,
                "valid_ratio": 0.1,
                "valid_every": 50,
                "wandb_enabled": True,
                "wandb_project": "old",
                "wandb_entity": "old",
                "wandb_run_name": "old",
                "custom_train_field": "preserved",
            },
        },
        indent=2,
    )
    base_config_payload = base_config_payload.replace('"norm_eps": 1e-05', '"norm_eps": 0.00001')
    base_config_payload = base_config_payload.replace(
        '"positive_float": 1e+20',
        '"positive_float": 100000000000000000000.0',
    )
    base_config_payload = base_config_payload.replace('"adam_eps": 1e-08', '"adam_eps": 0.00000001')
    base_config.write_text(base_config_payload, encoding="utf-8")
    manifest.write_text('{"audio":"a.wav"}\n', encoding="utf-8")
    checkpoint.write_bytes(b"600m base checkpoint")
    trainer_python.write_bytes(b"python")
    trainer_script.write_text("# upstream trainer\n", encoding="utf-8")
    training_jobs = tmp_path / "inputs" / "training-jobs.json"
    training_status = tmp_path / "inputs" / "training-status.jsonl"
    jobs: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for index, model_id in enumerate(MODEL_IDS):
        if index == 2:
            job_manifest = manifest
            job_config = base_config
        else:
            job_root = tmp_path / "inputs" / "datasets" / model_id
            job_manifest = job_root / "clean-manifest.jsonl"
            job_config = job_root / "training-config-v1.json"
            job_root.mkdir(parents=True, exist_ok=True)
            job_manifest.write_text(f'{{"audio":"{model_id}.wav"}}\n', encoding="utf-8")
            config_document = json.loads(base_config.read_text(encoding="utf-8"))
            config_document["train"]["manifest_path"] = str(job_manifest.resolve())
            config_document["train"]["output_dir"] = str(
                (tmp_path / "inputs" / "training" / model_id).resolve()
            )
            job_config.write_text(json.dumps(config_document), encoding="utf-8")
        output_dir = tmp_path / "inputs" / "training" / model_id
        jobs.append(
            {
                "model_id": model_id,
                "clean_manifest": str(job_manifest.relative_to(training_jobs.parent)),
                "config": str(job_config.relative_to(training_jobs.parent)),
                "output_dir": str(output_dir.relative_to(training_jobs.parent)),
                "command": [
                    str(trainer_python),
                    "-u",
                    str(trainer_script),
                    "--config",
                    str(job_config.resolve()),
                    "--manifest",
                    str(job_manifest.resolve()),
                    "--init-checkpoint",
                    str(checkpoint.resolve()),
                    "--output-dir",
                    str(output_dir.resolve()),
                    "--device",
                    "cuda",
                ],
            }
        )
        if index >= 2:
            continue
        output_dir.mkdir(parents=True)
        candidate = output_dir / "checkpoint-3000" / f"{model_id}.speaker.safetensors"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(f"speaker:{model_id}".encode())
        base_status: dict[str, object] = {
            "model_id": model_id,
            "clean_manifest_sha256": _sha256(job_manifest),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_revision": CHECKPOINT_REVISION,
            "config_sha256": _sha256(job_config),
            "upstream_commit": UPSTREAM_COMMIT,
            "started_at": f"2026-08-01T0{index}:00:00+00:00",
            "ended_at": None,
            "exit_code": None,
            "log_path": str((training_status.parent / "logs" / f"{model_id}.log").resolve()),
            "last_checkpoint": None,
            "last_checkpoint_sha256": None,
            "candidate_checkpoints": [],
            "error": None,
        }
        status_rows.extend(
            (
                base_status | {"event": "started", "status": "running"},
                base_status
                | {
                    "event": "finished",
                    "status": "success",
                    "ended_at": f"2026-08-01T0{index}:30:00+00:00",
                    "exit_code": 0,
                    "last_checkpoint": str(candidate.resolve()),
                    "last_checkpoint_sha256": _sha256(candidate),
                    "candidate_checkpoints": [
                        {"path": str(candidate.resolve()), "sha256": _sha256(candidate)}
                    ],
                },
            )
        )
    training_jobs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": "2026-08-01T00:00:00+00:00",
                "queue_policy": {"execution": "sequential"},
                "base_checkpoint_path": str(checkpoint.relative_to(training_jobs.parent)),
                "base_checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_revision": CHECKPOINT_REVISION,
                "upstream_commit": UPSTREAM_COMMIT,
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    _write_status(training_status, status_rows)
    return {
        "base_config": base_config,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "upstream": upstream,
        "training_jobs": training_jobs,
        "training_status": training_status,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_status(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_status(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _timestamp(step: int, *, seconds_per_step: float) -> str:
    base = datetime(2026, 8, 2, tzinfo=UTC)
    return (base + timedelta(seconds=step * seconds_per_step)).isoformat()


def _runner(
    *,
    intervals: dict[str, float] | None = None,
    exits: dict[str, int] | None = None,
    nonfinite: set[str] | None = None,
    oom: set[str] | None = None,
    calls: list[str] | None = None,
) -> Callable[..., int]:
    intervals = intervals or {}
    exits = exits or {}
    nonfinite = nonfinite or set()
    oom = oom or set()

    def run(plan, command: tuple[str, ...], emit) -> int:
        if calls is not None:
            calls.append(plan.candidate_id)
        assert command[1] == "-u"
        assert command[-2:] == ("--device", "cuda")
        interval = intervals.get(plan.candidate_id, 0.5)
        for step in range(1, 61):
            loss = "nan" if plan.candidate_id in nonfinite and step == 30 else "0.5"
            emit(
                f"step={step} loss={loss} rf=0.1 dur=0.2 dur_mae=0.3 lr=1.000e-02",
                _timestamp(step, seconds_per_step=interval),
            )
        if plan.candidate_id in oom:
            emit(
                "CUDA out of memory while allocating tensor",
                _timestamp(61, seconds_per_step=interval),
            )
        return exits.get(plan.candidate_id, 0)

    return run


def _sampler(peaks: dict[str, float]) -> Callable[..., None]:
    def sample(plan, path: Path, _interval: float, _stop: Event) -> None:
        peak = peaks.get(plan.candidate_id, 8_000.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(
                (
                    "timestamp",
                    "index",
                    "utilization_gpu_percent",
                    "memory_used_mib",
                    "power_draw_w",
                    "temperature_c",
                    "error",
                    "raw_csv",
                )
            )
            for second in range(41):
                memory = peak if second == 1 else peak - 100
                writer.writerow(
                    (
                        _timestamp(second, seconds_per_step=1.0),
                        0,
                        90,
                        memory,
                        180,
                        65,
                        "",
                        f"0, 90, {memory}, 180, 65",
                    )
                )

    return sample


def _environment_inspector() -> Callable[..., dict[str, object]]:
    def inspect() -> dict[str, object]:
        return {
            "python": {"version": "3.10.11", "executable": "python.exe"},
            "torch": {"version": "2.7.1", "cuda_version": "12.8"},
            "gpu": {
                "name": "NVIDIA GeForce RTX 4070",
                "uuid": "GPU-test",
                "driver_version": "576.80",
                "memory_total_mib": 12282.0,
                "power_limit_w": 200.0,
                "error": None,
                "raw_csv": "NVIDIA GeForce RTX 4070, GPU-test, 576.80, 12282, 200",
            },
        }

    return inspect


def _git_inspector(
    *,
    head: str = UPSTREAM_COMMIT,
    tracked_worktree_clean: bool = True,
    index_clean: bool = True,
    untracked_files: tuple[str, ...] = ("speakers/local.safetensors", "remote_server.py"),
) -> Callable[..., dict[str, object]]:
    def inspect(_upstream_root: Path) -> dict[str, object]:
        return {
            "head": head,
            "tracked_worktree_clean": tracked_worktree_clean,
            "index_clean": index_clean,
            "untracked_files": list(untracked_files),
        }

    return inspect


def _runtime_probe(
    states: dict[tuple[str, str], dict[str, object]] | None = None,
    calls: list[tuple[str, str]] | None = None,
) -> Callable[..., dict[str, object]]:
    states = states or {}

    def probe(plan, phase: str) -> dict[str, object]:
        if calls is not None:
            calls.append((plan.candidate_id, phase))
        return states.get(
            (plan.candidate_id, phase),
            {
                "timestamp": _timestamp(0, seconds_per_step=1.0),
                "matching_processes": [],
                "gpu_memory_used_mib": 1_000.0,
            },
        )

    return probe


def _run(module: ModuleType, fixture: dict[str, Path], output_root: Path, **kwargs):
    return module.run_benchmark(
        base_config=fixture["base_config"],
        manifest=fixture["manifest"],
        base_checkpoint=fixture["checkpoint"],
        upstream_root=fixture["upstream"],
        upstream_commit=UPSTREAM_COMMIT,
        training_jobs=fixture["training_jobs"],
        training_status=fixture["training_status"],
        output_root=output_root,
        runner=kwargs.get("runner", _runner()),
        sampler=kwargs.get("sampler", _sampler({})),
        sampler_interval=kwargs.get("sampler_interval", 0.5),
        git_inspector=kwargs.get("git_inspector", _git_inspector()),
        runtime_probe=kwargs.get("runtime_probe", _runtime_probe()),
        cleanup_timeout=kwargs.get("cleanup_timeout", 0.0),
        cleanup_poll_interval=kwargs.get("cleanup_poll_interval", 0.01),
        environment_inspector=kwargs.get("environment_inspector", _environment_inspector()),
    )


def test_candidate_configs_preserve_fixed_inputs_and_global_batch_16(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"

    summary = _run(module, fixture, output_root)

    expected = {
        "A": (1, 16, True),
        "B": (2, 8, True),
        "C": (4, 4, True),
        "D": (2, 8, False),
    }
    configs = []
    for candidate_id, (batch_size, accumulation, checkpointing) in expected.items():
        config_path = output_root / f"candidate-{candidate_id}" / "training-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs.append(config)
        train = config["train"]
        assert train["batch_size"] == batch_size
        assert train["gradient_accumulation_steps"] == accumulation
        assert train["batch_size"] * train["gradient_accumulation_steps"] == GLOBAL_BATCH_SIZE
        assert train["gradient_checkpointing"] is checkpointing
        assert train["precision"] == "bf16"
        assert train["allow_tf32"] is True
        assert train["compile_model"] is False
        assert train["learning_rate"] == pytest.approx(0.01)
        assert train["lr_scheduler"] == "none"
        assert train["max_steps"] == 60
        assert train["log_every"] == 1
        assert train["save_every"] == 60
        assert train["max_latent_steps"] == 750
        assert train["seed"] == 0
        assert train["valid_ratio"] == pytest.approx(0.0)
        assert train["valid_every"] == 0
        assert train["wandb_enabled"] is False
        assert train["custom_train_field"] == "preserved"
        assert train["manifest_path"] == str(fixture["manifest"].resolve())
        assert train["output_dir"] == str((config_path.parent / "output").resolve())
    assert all(config["model"] == configs[0]["model"] for config in configs)
    assert all(config["data"] == configs[0]["data"] for config in configs)
    assert summary["constraints"] == {
        "candidate_parallelism": 1,
        "global_batch_size": 16,
        "measurement_optimizer_steps": 50,
        "perf_warmup_optimizer_steps": 10,
        "torch_compile": False,
        "multi_model_parallelism": False,
    }
    assert summary["provenance"]["manifest"]["row_count"] == 1
    assert summary["provenance"]["training_jobs"] == {
        "path": str(fixture["training_jobs"].resolve()),
        "sha256": _sha256(fixture["training_jobs"]),
    }
    assert summary["provenance"]["training_status"] == {
        "path": str(fixture["training_status"].resolve()),
        "sha256": _sha256(fixture["training_status"]),
    }
    assert summary["provenance"]["environment"]["gpu"]["uuid"] == "GPU-test"
    assert summary["status"] == "PASS"
    assert summary["recommended_candidate"]["id"] in CANDIDATE_IDS
    for candidate in summary["candidates"]:
        assert candidate["id"] in CANDIDATE_IDS
        assert candidate["effective_global_batch_size"] == GLOBAL_BATCH_SIZE
        assert candidate["metrics"]["measured_optimizer_steps"] == MEASUREMENT_STEPS
        assert candidate["overrides"] == {
            "batch_size": candidate["batch_size"],
            "gradient_accumulation_steps": candidate["gradient_accumulation_steps"],
            "gradient_checkpointing": candidate["gradient_checkpointing"],
        }


def test_candidate_configs_preserve_shared_scalar_types_under_pyyaml(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"
    _run(module, fixture, output_root)
    base_document = yaml.safe_load(fixture["base_config"].read_text(encoding="utf-8"))
    base_types = _scalar_types(base_document)
    overridden_train_fields = {
        "manifest_path",
        "output_dir",
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "allow_tf32",
        "precision",
        "compile_model",
        "learning_rate",
        "lr_scheduler",
        "max_steps",
        "log_every",
        "save_every",
        "valid_ratio",
        "valid_every",
        "wandb_enabled",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
    }
    shared_types = {
        path: scalar_type
        for path, scalar_type in base_types.items()
        if not (len(path) == 2 and path[0] == "train" and path[1] in overridden_train_fields)
    }

    for candidate_id in CANDIDATE_IDS:
        config_path = output_root / f"candidate-{candidate_id}" / "training-config.json"
        raw_config = config_path.read_text(encoding="utf-8")
        candidate_document = yaml.safe_load(raw_config)
        candidate_types = _scalar_types(candidate_document)
        assert {path: candidate_types[path] for path in shared_types} == shared_types
        assert type(candidate_document["model"]["norm_eps"]) is float
        assert type(candidate_document["model"]["positive_float"]) is float
        assert type(candidate_document["train"]["adam_eps"]) is float
        assert candidate_document["model"]["scientific_text"] == "1e-08"
        assert '"scientific_text": "1e-08"' in raw_config


def test_training_command_preserves_safetensors_checkpoint_symlink_alias(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint_blob = tmp_path / "hf-cache" / "blobs" / ("a" * 64)
    checkpoint_blob.parent.mkdir(parents=True)
    checkpoint_blob.write_bytes(b"checkpoint")
    checkpoint_alias = tmp_path / "hf-cache" / "snapshots" / "revision" / "model.safetensors"
    checkpoint_alias.parent.mkdir(parents=True)
    checkpoint_alias.symlink_to(checkpoint_blob)
    plan = module.CandidatePlan(
        candidate_id="A",
        spec=module.CANDIDATE_SPECS[0],
        root=tmp_path / "candidate-A",
        output_dir=tmp_path / "candidate-A" / "output",
        config_path=tmp_path / "candidate-A" / "training-config.json",
        upstream_root=tmp_path / "upstream",
    )

    command = module._training_command(  # noqa: SLF001 - regression at command boundary.
        plan,
        trainer_python=tmp_path / "upstream" / ".venv" / "Scripts" / "python.exe",
        trainer_script=tmp_path / "upstream" / "train.py",
        manifest=tmp_path / "clean-manifest.jsonl",
        base_checkpoint=checkpoint_alias,
    )

    checkpoint_argument = Path(command[command.index("--init-checkpoint") + 1])
    assert checkpoint_argument == checkpoint_alias.absolute()
    assert checkpoint_argument.suffix == ".safetensors"
    assert checkpoint_argument.is_symlink()
    assert checkpoint_argument.samefile(checkpoint_blob)


def test_output_root_collision_fails_before_any_candidate_runs(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"
    output_root.mkdir()
    calls: list[str] = []

    with pytest.raises(FileExistsError, match="already exists"):
        _run(module, fixture, output_root, runner=_runner(calls=calls))

    assert calls == []


def test_failures_oom_nonfinite_and_vram_are_preserved_and_excluded(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    calls: list[str] = []

    summary = _run(
        module,
        fixture,
        tmp_path / "benchmark-v1",
        runner=_runner(
            intervals={"A": 0.25, "B": 0.5, "C": 0.2, "D": 0.1},
            exits={"D": 1},
            nonfinite={"C"},
            oom={"D"},
            calls=calls,
        ),
        sampler=_sampler({"A": 11_000, "B": 9_000, "C": 8_000, "D": 8_000}),
    )

    assert calls == list(CANDIDATE_IDS)
    results = {row["id"]: row for row in summary["candidates"]}
    assert results["A"]["metrics"]["eligible"] is False
    assert "peak_vram_exceeds_10500_mib" in results["A"]["metrics"]["ineligible_reasons"]
    assert results["B"]["metrics"]["eligible"] is True
    assert results["B"]["metrics"]["measured_optimizer_steps"] == MEASUREMENT_STEPS
    assert results["B"]["metrics"]["steady_optimizer_steps_per_second"] == pytest.approx(2.0)
    assert results["B"]["metrics"]["steady_samples_per_second"] == pytest.approx(32.0)
    assert results["B"]["metrics"]["gpu_utilization_percent"] == {
        "sample_count": 26,
        "minimum": 90.0,
        "mean": 90.0,
        "maximum": 90.0,
    }
    assert results["C"]["metrics"]["loss_finite"] is False
    assert "nonfinite_loss" in results["C"]["metrics"]["ineligible_reasons"]
    assert results["D"]["metrics"]["oom"] is True
    assert results["D"]["metrics"]["exit_code"] == 1
    assert summary["recommended_candidate"]["id"] == "B"
    assert (tmp_path / "benchmark-v1/candidate-D/raw.log").is_file()
    assert (tmp_path / "benchmark-v1/candidate-D/result.json").is_file()


def test_equal_speed_recommends_lower_peak_vram_and_writes_timestamped_atomic_outputs(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"

    summary = _run(
        module,
        fixture,
        output_root,
        runner=_runner(intervals=dict.fromkeys(CANDIDATE_IDS, 0.5)),
        sampler=_sampler({"A": 9_000, "B": 8_000, "C": 9_500, "D": 9_750}),
    )

    assert summary["recommended_candidate"]["id"] == "B"
    events_path = output_root / "candidate-A/step-events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 60
    assert all(event["timestamp"].endswith("+00:00") for event in events)
    assert (
        (output_root / "candidate-A/nvidia-smi.csv")
        .read_text(encoding="utf-8")
        .startswith("timestamp,index,utilization_gpu_percent")
    )
    assert (output_root / "summary.json").is_file()
    assert not (output_root / "summary.json.tmp").exists()
    assert not (output_root / "candidate-A/result.json.tmp").exists()
    statuses = [
        json.loads(line)
        for line in (output_root / "status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in statuses] == ["started", "finished"] * 4
    assert summary["provenance"]["script"]["sha256"] == module.sha256_file(SCRIPT_PATH)
    assert summary["measurement_boundary"] == {
        "definition": "(60-10)/(timestamp(step=60)-timestamp(step=10))",
        "start_step": 10,
        "end_step": 60,
        "measured_optimizer_steps": 50,
    }
    first = summary["candidates"][0]["metrics"]
    assert first["peak_vram_mib"] == pytest.approx(9_000.0)
    assert first["steady_peak_vram_mib"] == pytest.approx(8_900.0)
    assert first["full_run_peak_vram_mib"] == pytest.approx(9_000.0)


def test_main_returns_one_but_keeps_summary_when_no_candidate_is_eligible(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"

    exit_code = module.main(
        [
            "--base-config",
            str(fixture["base_config"]),
            "--manifest",
            str(fixture["manifest"]),
            "--base-checkpoint",
            str(fixture["checkpoint"]),
            "--upstream-root",
            str(fixture["upstream"]),
            "--upstream-commit",
            UPSTREAM_COMMIT,
            "--training-jobs",
            str(fixture["training_jobs"]),
            "--training-status",
            str(fixture["training_status"]),
            "--output-root",
            str(output_root),
        ],
        runner=_runner(exits=dict.fromkeys(CANDIDATE_IDS, 1)),
        sampler=_sampler({}),
        git_inspector=_git_inspector(),
        runtime_probe=_runtime_probe(),
        cleanup_timeout=0.0,
        cleanup_poll_interval=0.01,
        environment_inspector=_environment_inspector(),
    )

    assert exit_code == 1
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["recommended_candidate"] is None
    assert len(summary["candidates"]) == 4


@pytest.mark.parametrize(
    ("inspector", "message"),
    [
        (_git_inspector(head="0" * 40), "HEAD does not match"),
        (_git_inspector(tracked_worktree_clean=False), "tracked worktree is dirty"),
        (_git_inspector(index_clean=False), "index is dirty"),
    ],
)
def test_upstream_head_and_tracked_state_are_verified_before_reservation(
    tmp_path: Path,
    inspector: Callable[..., dict[str, object]],
    message: str,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"
    calls: list[str] = []

    with pytest.raises(ValueError, match=message):
        _run(
            module,
            fixture,
            output_root,
            runner=_runner(calls=calls),
            git_inspector=inspector,
        )

    assert calls == []
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("kasumi_not_finished", "Kasumi current status"),
        ("next_started", "third training job must be unstarted"),
        ("next_output_exists", "third training job output"),
        ("completed_config_hash_mismatch", "config_sha256"),
        ("jobs_out_of_order", "training job order"),
        ("seed_not_zero", "seed must equal 0"),
        ("max_latent_not_750", "max_latent_steps must equal 750"),
        ("checkpoint_hash_mismatch", "base checkpoint SHA-256"),
        ("checkpoint_revision_mismatch", "checkpoint_revision"),
        ("upstream_mismatch", "upstream_commit"),
        ("later_success", "remaining training jobs must be unfinished"),
    ],
)
def test_training_queue_state_is_rejected_before_output_reservation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"
    runner_calls: list[str] = []
    jobs = json.loads(fixture["training_jobs"].read_text(encoding="utf-8"))
    rows = _read_status(fixture["training_status"])
    if mutation == "kasumi_not_finished":
        rows.append(rows[-2])
    elif mutation == "next_started":
        rows.append(
            {
                "event": "started",
                "status": "running",
                "model_id": NEXT_MODEL_ID,
            }
        )
    elif mutation == "next_output_exists":
        (fixture["training_jobs"].parent / jobs["jobs"][2]["output_dir"]).mkdir(parents=True)
    elif mutation == "completed_config_hash_mismatch":
        rows[-1]["config_sha256"] = "0" * 64
    elif mutation == "jobs_out_of_order":
        jobs["jobs"][1], jobs["jobs"][2] = jobs["jobs"][2], jobs["jobs"][1]
    elif mutation == "seed_not_zero":
        config = json.loads(fixture["base_config"].read_text(encoding="utf-8"))
        config["train"]["seed"] = 1
        fixture["base_config"].write_text(json.dumps(config), encoding="utf-8")
    elif mutation == "max_latent_not_750":
        config = json.loads(fixture["base_config"].read_text(encoding="utf-8"))
        config["train"]["max_latent_steps"] = 749
        fixture["base_config"].write_text(json.dumps(config), encoding="utf-8")
    elif mutation == "checkpoint_hash_mismatch":
        jobs["base_checkpoint_sha256"] = "0" * 64
    elif mutation == "checkpoint_revision_mismatch":
        rows[-1]["checkpoint_revision"] = "0" * 40
    elif mutation == "upstream_mismatch":
        jobs["upstream_commit"] = "0" * 40
    elif mutation == "later_success":
        rows.append(
            {
                "event": "finished",
                "status": "success",
                "model_id": MODEL_IDS[3],
                "exit_code": 0,
            }
        )
    fixture["training_jobs"].write_text(json.dumps(jobs), encoding="utf-8")
    _write_status(fixture["training_status"], rows)

    with pytest.raises((FileExistsError, TypeError, ValueError), match=message):
        _run(module, fixture, output_root, runner=_runner(calls=runner_calls))

    assert runner_calls == []
    assert not output_root.exists()


@pytest.mark.parametrize("schema_version", ["speaker-training-queue/v1", True])
def test_training_jobs_rejects_non_numeric_v1_schema_before_output_reservation(
    tmp_path: Path,
    schema_version: object,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    jobs = json.loads(fixture["training_jobs"].read_text(encoding="utf-8"))
    jobs["schema_version"] = schema_version
    fixture["training_jobs"].write_text(json.dumps(jobs), encoding="utf-8")
    output_root = tmp_path / "benchmark-v1"
    runner_calls: list[str] = []

    with pytest.raises(ValueError, match="training jobs schema_version must be numeric 1"):
        _run(module, fixture, output_root, runner=_runner(calls=runner_calls))

    assert runner_calls == []
    assert not output_root.exists()


def test_benchmark_dataset_must_be_exactly_the_third_pending_job(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    jobs = json.loads(fixture["training_jobs"].read_text(encoding="utf-8"))["jobs"]
    fixture["base_config"] = fixture["training_jobs"].parent / jobs[3]["config"]
    fixture["manifest"] = fixture["training_jobs"].parent / jobs[3]["clean_manifest"]
    output_root = tmp_path / "benchmark-v1"

    with pytest.raises(ValueError, match="third training job"):
        _run(module, fixture, output_root)

    assert not output_root.exists()


@pytest.mark.parametrize(
    "script_name",
    ["run_600m_speaker_training_queue.py", "launch_600m_training_queue_runtime.py"],
)
def test_runtime_process_matcher_blocks_old_queue_and_supervisor(
    tmp_path: Path,
    script_name: str,
) -> None:
    module = _load_script()
    plan = module.CandidatePlan(
        candidate_id="A",
        spec=module.CANDIDATE_SPECS[0],
        root=tmp_path / "candidate-A",
        output_dir=tmp_path / "candidate-A/output",
        config_path=tmp_path / "candidate-A/config.json",
        upstream_root=tmp_path / "upstream",
    )

    assert module._is_conflicting_runtime_command(  # noqa: SLF001
        f'python.exe "C:\\queue\\{script_name}" --training-jobs jobs.json',
        plan,
    )


def test_untracked_upstream_assets_are_accepted_and_bound_in_provenance(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)

    summary = _run(
        module,
        fixture,
        tmp_path / "benchmark-v1",
        git_inspector=_git_inspector(untracked_files=("z.bin", "a.bin")),
    )

    upstream = summary["provenance"]["upstream"]
    assert upstream["head"] == UPSTREAM_COMMIT
    assert upstream["tracked_worktree_clean"] is True
    assert upstream["index_clean"] is True
    assert upstream["untracked_files"] == ["a.bin", "z.bin"]
    assert upstream["untracked_count"] == 2
    assert len(upstream["untracked_files_sha256"]) == 64


def test_runtime_residue_marks_candidate_failed_and_stops_later_candidates(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    runner_calls: list[str] = []
    probe_calls: list[tuple[str, str]] = []
    residual = {
        "timestamp": _timestamp(61, seconds_per_step=0.5),
        "matching_processes": [
            {"pid": 321, "parent_pid": 123, "name": "python.exe", "command_line": "spawn"}
        ],
        "gpu_memory_used_mib": 9_000.0,
    }

    summary = _run(
        module,
        fixture,
        tmp_path / "benchmark-v1",
        runner=_runner(calls=runner_calls),
        runtime_probe=_runtime_probe({("B", "after"): residual}, probe_calls),
    )

    assert runner_calls == ["A", "B"]
    assert probe_calls == [
        ("A", "before"),
        ("A", "after"),
        ("B", "before"),
        ("B", "after"),
    ]
    assert summary["status"] == "FAILED"
    assert summary["recommended_candidate"] is None
    assert len(summary["candidates"]) == 2
    candidate = summary["candidates"][1]
    assert candidate["id"] == "B"
    assert candidate["metrics"]["eligible"] is False
    assert "runtime_not_quiescent" in candidate["metrics"]["ineligible_reasons"]
    assert candidate["runtime_guard"]["after"]["matching_processes"][0]["pid"] == 321
    assert (tmp_path / "benchmark-v1/candidate-B/runtime-guard.jsonl").is_file()


def test_step_events_require_exact_1_through_60_and_fixed_learning_rate(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)

    def malformed_runner(plan, _command: tuple[str, ...], emit) -> int:
        for step in (*range(1, 30), 29, *range(31, 61)):
            lr = "9.000e-03" if step == 40 else "1.000e-02"
            emit(
                f"step={step} loss=0.5 rf=0.1 dur=0.2 dur_mae=0.3 lr={lr}",
                _timestamp(step, seconds_per_step=0.5),
            )
        return 0

    summary = _run(
        module,
        fixture,
        tmp_path / "benchmark-v1",
        runner=malformed_runner,
    )

    first = summary["candidates"][0]["metrics"]
    assert first["observed_step_sequence_valid"] is False
    assert first["learning_rate_fixed"] is False
    assert "step_sequence_not_1_through_60" in first["ineligible_reasons"]
    assert "learning_rate_not_fixed_1e-2" in first["ineligible_reasons"]


def test_unstructured_training_finished_line_is_not_an_optimizer_event(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"

    def real_log_runner(plan, _command: tuple[str, ...], emit) -> int:
        for step in range(1, 61):
            emit(
                f"step={step} loss=0.5 rf=0.1 dur=0.2 dur_mae=0.3 lr=1.000e-02",
                _timestamp(step, seconds_per_step=0.5),
            )
        emit("Training finished at step=60.", _timestamp(61, seconds_per_step=0.5))
        return 0

    summary = _run(module, fixture, output_root, runner=real_log_runner)

    events = _read_status(output_root / "candidate-A" / "step-events.jsonl")
    raw_lines = (output_root / "candidate-A" / "raw.log").read_text(encoding="utf-8").splitlines()
    first = summary["candidates"][0]["metrics"]
    assert len(events) == 60
    assert [event["step"] for event in events] == list(range(1, 61))
    assert len(raw_lines) == 61
    assert raw_lines[-1] == "Training finished at step=60."
    assert first["observed_step_sequence_valid"] is True
    assert first["measured_optimizer_steps"] == 50
    assert first["loss_finite"] is True
    assert first["learning_rate_fixed"] is True
    assert first["eligible"] is True


def test_fewer_than_ten_steady_gpu_samples_is_ineligible(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)

    def sparse_sampler(plan, path: Path, _interval: float, _stop: Event) -> None:
        del plan
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(module.GPU_HEADER)
            for second in range(6, 15):
                writer.writerow(
                    (
                        _timestamp(second, seconds_per_step=1.0),
                        0,
                        90,
                        8_000,
                        180,
                        65,
                        "",
                        "0, 90, 8000, 180, 65",
                    )
                )

    summary = _run(
        module,
        fixture,
        tmp_path / "benchmark-v1",
        sampler=sparse_sampler,
    )

    assert summary["status"] == "FAILED"
    assert summary["recommended_candidate"] is None
    first = summary["candidates"][0]["metrics"]
    assert first["gpu_utilization_percent"]["sample_count"] == 9
    assert "steady_gpu_samples_below_10" in first["ineligible_reasons"]


def test_gpu_preflight_failure_is_explicit_and_does_not_reserve_output(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "benchmark-v1"

    def failed_environment() -> dict[str, object]:
        return {
            "python": {},
            "torch": {},
            "gpu": {"error": "nvidia-smi unavailable", "raw_csv": ""},
        }

    with pytest.raises(ValueError, match="GPU preflight failed"):
        _run(
            module,
            fixture,
            output_root,
            environment_inspector=failed_environment,
        )

    assert not output_root.exists()
