# ruff: noqa: EM101, PLR2004, PLR6301, S404, TRY003, TRY301

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import BinaryIO


SCRIPT_PATH = Path("scripts/launch_600m_speaker_training_queue_detached.py")
MODEL_IDS = tuple(f"model-{index:02d}" for index in range(12))
TARGET_MODEL_ID = MODEL_IDS[-1]
SKIPPED_MODEL_IDS = MODEL_IDS[:-1]


def _load_script() -> ModuleType:
    module_name = "_test_launch_600m_speaker_training_queue_detached"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _inputs(tmp_path: Path) -> dict[str, Any]:
    queue_script = _write(tmp_path / "bundle-v001/run_queue.py", b"# queue\n")
    checkpoint = _write(tmp_path / "weights/model.safetensors", b"checkpoint")
    python = _write(tmp_path / "upstream/.venv/Scripts/python.exe", b"python")
    upstream = tmp_path / "upstream"
    _write(upstream / "train.py", b"# tracked trainer\n")
    _write(upstream / "irodori_tts/__init__.py", b"# tracked package\n")
    commit = "a" * 40
    revision = "irodori-tts-v3-600m"
    jobs = tmp_path / "training/training-jobs.json"
    jobs.parent.mkdir(parents=True)
    job_rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        clean_manifest = _write(
            tmp_path / f"training/manifests/{model_id}.jsonl",
            (json.dumps({"audio_path": f"{model_id}.wav", "text": "test"}) + "\n").encode(),
        )
        output_dir = tmp_path / f"training/outputs/{model_id}"
        output_dir.mkdir(parents=True)
        train: dict[str, object] = {
            "manifest_path": str(clean_manifest.resolve()),
            "output_dir": str(output_dir.resolve()),
        }
        if model_id == TARGET_MODEL_ID:
            init_embedding = _write(
                tmp_path / f"training/embeddings/{model_id}.pt",
                b"speaker embedding",
            )
            train["speaker_inversion_init_embedding"] = str(init_embedding.resolve())
            train["speaker_inversion_init_embedding_sha256"] = _sha256(init_embedding)
        config = tmp_path / f"training/configs/{model_id}.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"train": train}, sort_keys=True) + "\n", encoding="utf-8")
        job_rows.append(
            {
                "model_id": model_id,
                "clean_manifest": str(clean_manifest.resolve()),
                "config": str(config.resolve()),
                "output_dir": str(output_dir.resolve()),
                "command": [
                    "python.exe",
                    "train.py",
                    "--config",
                    str(config.resolve()),
                    "--manifest",
                    str(clean_manifest.resolve()),
                    "--init-checkpoint",
                    str(checkpoint.resolve()),
                    "--output-dir",
                    str(output_dir.resolve()),
                ],
            }
        )
    jobs.write_text(
        json.dumps(
            {
                "base_checkpoint_path": str(checkpoint.resolve()),
                "base_checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_revision": revision,
                "upstream_commit": commit,
                "jobs": job_rows,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    status = tmp_path / "training/training-status.jsonl"
    status.write_text('{"status":"success"}\n', encoding="utf-8")
    return {
        "queue_script": queue_script,
        "expected_queue_sha256": _sha256(queue_script),
        "jobs_path": jobs,
        "expected_jobs_sha256": _sha256(jobs),
        "status_path": status,
        "checkpoint_path": checkpoint,
        "expected_checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_revision": revision,
        "upstream_root": upstream,
        "expected_upstream_commit": commit,
        "python_path": python,
        "evidence_root": tmp_path / "evidence/speaker-training-detached-v001",
        "target_model_id": TARGET_MODEL_ID,
    }


def _git_runner(
    commit: str,
    *,
    tracked_dirty: str = "",
    tracked_critical: str = "train.py\nirodori_tts/__init__.py\n",
    untracked_critical: str = "",
    ignored_critical: str = "",
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        command: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[-2:] == ("rev-parse", "HEAD"):
            stdout = f"{commit}\n"
            returncode = 0
        elif "--untracked-files=no" in command:
            stdout = tracked_dirty
            returncode = 0
        elif "--untracked-files=all" in command:
            stdout = "?? remote_server.py\n?? speakers/runtime.safetensors\n"
            returncode = 0
        elif "--error-unmatch" in command:
            stdout = tracked_critical
            returncode = 0 if "train.py\n" in f"{tracked_critical}\n" else 1
        elif "--ignored" in command:
            stdout = ignored_critical
            returncode = 0
        else:
            assert "--others" in command
            stdout = untracked_critical
            returncode = 0
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    return run


def _safe_snapshot(module: ModuleType) -> object:
    return module.RuntimeSnapshot(
        observed_at="2026-08-03T00:00:00+00:00",
        processes=(),
        training_processes=(),
        service_processes=(),
        gpu_total_mib=12_000.0,
        gpu_used_mib=1_000.0,
        gpu_free_mib=11_000.0,
        errors=(),
    )


def test_verify_contract_binds_all_immutable_inputs_and_mutable_status(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    contract = module.verify_contract(
        **inputs,
        detached_script=SCRIPT_PATH,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
    )

    assert contract["queue_script"] == {
        "path": str(inputs["queue_script"].resolve()),
        "sha256": inputs["expected_queue_sha256"],
        "size": inputs["queue_script"].stat().st_size,
    }
    assert contract["jobs"] == {
        "path": str(inputs["jobs_path"].resolve()),
        "sha256": inputs["expected_jobs_sha256"],
        "size": inputs["jobs_path"].stat().st_size,
    }
    assert contract["checkpoint"]["sha256"] == inputs["expected_checkpoint_sha256"]
    assert contract["checkpoint_revision"] == inputs["checkpoint_revision"]
    assert contract["upstream"] == {
        "path": str(inputs["upstream_root"].resolve()),
        "commit": inputs["expected_upstream_commit"],
        "tracked_worktree": "clean",
        "critical_paths": "tracked-clean-no-untracked-source",
    }
    assert contract["status_before"]["sha256"] == _sha256(inputs["status_path"])
    assert contract["output_identity"]["evidence_root"].endswith("speaker-training-detached-v001")
    assert contract["target_model_id"] == TARGET_MODEL_ID
    assert contract["expected_skipped_model_ids"] == list(SKIPPED_MODEL_IDS)
    assert len(contract["job_inputs"]) == 12
    target = next(row for row in contract["job_inputs"] if row["model_id"] == TARGET_MODEL_ID)
    assert target["clean_manifest"]["sha256"]
    assert target["config"]["sha256"]
    assert target["speaker_inversion_init_embedding"]["sha256"]


@pytest.mark.parametrize(
    "field",
    [
        "expected_queue_sha256",
        "expected_jobs_sha256",
        "expected_checkpoint_sha256",
    ],
)
def test_verify_contract_rejects_wrong_file_pin(tmp_path: Path, field: str) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    inputs[field] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )


def test_verify_contract_rejects_manifest_provenance_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    inputs["checkpoint_revision"] = "wrong-revision"

    with pytest.raises(ValueError, match="checkpoint revision"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )


def test_verify_contract_rejects_job_command_external_input_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    document = json.loads(inputs["jobs_path"].read_text(encoding="utf-8"))
    command = document["jobs"][-1]["command"]
    command[command.index("--init-checkpoint") + 1] = str(tmp_path / "other.safetensors")
    inputs["jobs_path"].write_text(json.dumps(document) + "\n", encoding="utf-8")
    inputs["expected_jobs_sha256"] = _sha256(inputs["jobs_path"])

    with pytest.raises(ValueError, match="command --init-checkpoint path mismatch"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )


def test_verify_contract_rejects_unpinned_upstream_head(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    with pytest.raises(ValueError, match="upstream commit mismatch"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner("b" * 40),
        )


def test_verify_contract_allows_untracked_runtime_assets_outside_critical_scope(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    _write(inputs["upstream_root"] / "remote_server.py", b"# existing runtime helper\n")
    _write(inputs["upstream_root"] / "speakers/runtime.safetensors", b"runtime speaker")
    _write(inputs["upstream_root"] / "voice_bank_speakers.toml", b"[speakers]\n")

    contract = module.verify_contract(
        **inputs,
        detached_script=SCRIPT_PATH,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
    )

    assert contract["upstream"]["tracked_worktree"] == "clean"


@pytest.mark.parametrize("tracked_dirty", [" M train.py\n", "M  train.py\n"])
def test_verify_contract_rejects_dirty_upstream_tracked_files(
    tmp_path: Path,
    tracked_dirty: str,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    with pytest.raises(ValueError, match="tracked worktree is not clean"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(
                inputs["expected_upstream_commit"],
                tracked_dirty=tracked_dirty,
            ),
        )


@pytest.mark.parametrize("listing", ["untracked_critical", "ignored_critical"])
@pytest.mark.parametrize("filename", ["evil.py", "evil.pyc", "evil.pyd"])
def test_verify_contract_rejects_untracked_source_inside_critical_package(
    tmp_path: Path,
    listing: str,
    filename: str,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    relative = f"irodori_tts/{filename}"
    _write(inputs["upstream_root"] / relative, b"untracked importable payload")

    with pytest.raises(ValueError, match="untracked source in critical upstream scope"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(
                inputs["expected_upstream_commit"],
                **{listing: f"{relative}\n"},
            ),
        )


def test_verify_contract_rejects_nested_tracked_alias_in_critical_package(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    replacement = tmp_path / "attacker/backends"
    _write(replacement / "runtime.py", b"# replacement\n")
    nested = inputs["upstream_root"] / "irodori_tts/backends"
    nested.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(ValueError, match="tracked upstream path has a symlink"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(
                inputs["expected_upstream_commit"],
                tracked_critical=(
                    "train.py\nirodori_tts/__init__.py\nirodori_tts/backends/runtime.py\n"
                ),
            ),
        )


def test_verify_contract_requires_every_tracked_critical_descendant_to_be_regular(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    with pytest.raises(ValueError, match="tracked upstream file is missing or not regular"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(
                inputs["expected_upstream_commit"],
                tracked_critical=("train.py\nirodori_tts/__init__.py\nirodori_tts/missing.py\n"),
            ),
        )


def test_verify_contract_rejects_untracked_train_path(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    with pytest.raises(ValueError, match=r"train\.py must be tracked"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(
                inputs["expected_upstream_commit"],
                tracked_critical="irodori_tts/__init__.py\n",
                untracked_critical="train.py\n",
            ),
        )


def test_verify_contract_rejects_replaced_train_path(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    train = inputs["upstream_root"] / "train.py"
    replacement = _write(tmp_path / "attacker/train.py", b"# replacement\n")
    train.unlink()
    train.symlink_to(replacement)

    with pytest.raises(ValueError, match=r"train\.py has a symlink"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )


def test_verify_contract_requires_versioned_evidence_root(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    inputs["evidence_root"] = tmp_path / "evidence/current"

    with pytest.raises(ValueError, match="versioned evidence root"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )


def test_preflight_runs_queue_dry_run_without_mutating_status(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    before = inputs["status_path"].read_bytes()
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "planned": [TARGET_MODEL_ID],
                    "succeeded": [],
                    "failed": [],
                    "skipped": list(SKIPPED_MODEL_IDS),
                }
            )
            + "\n",
            stderr="",
        )

    report = module.preflight_detached(
        **inputs,
        detached_script=SCRIPT_PATH,
        probe=lambda: _safe_snapshot(module),
        runner=runner,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
    )

    assert report["passed"] is True
    assert report["launch_performed"] is False
    assert report["queue_dry_run"]["failed"] == []
    assert inputs["status_path"].read_bytes() == before
    assert len(calls) == 1
    command = calls[0]
    assert command[:2] == (
        str(inputs["python_path"].resolve()),
        str(inputs["queue_script"].resolve()),
    )
    assert command[-1] == "--dry-run"


@pytest.mark.parametrize(
    "report",
    [
        {"planned": [], "succeeded": [], "failed": [], "skipped": list(MODEL_IDS)},
        {
            "planned": [MODEL_IDS[-2], TARGET_MODEL_ID],
            "succeeded": [],
            "failed": [],
            "skipped": list(MODEL_IDS[:-2]),
        },
        {
            "planned": [MODEL_IDS[0]],
            "succeeded": [],
            "failed": [],
            "skipped": list(MODEL_IDS[1:]),
        },
        {
            "planned": [TARGET_MODEL_ID],
            "succeeded": [],
            "failed": [],
            "skipped": list(SKIPPED_MODEL_IDS[:-1]),
        },
        {
            "planned": [TARGET_MODEL_ID],
            "succeeded": [TARGET_MODEL_ID],
            "failed": [],
            "skipped": list(SKIPPED_MODEL_IDS),
        },
    ],
)
def test_preflight_rejects_any_report_except_exact_one_pending_contract(
    tmp_path: Path,
    report: dict[str, list[str]],
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    with pytest.raises(RuntimeError, match="dry-run report is unsafe"):
        module.preflight_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: _safe_snapshot(module),
            runner=lambda command, **_kwargs: subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(report) + "\n", stderr=""
            ),
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )


@pytest.mark.parametrize("input_name", ["clean_manifest", "config", "init_embedding"])
def test_reverify_contract_rejects_bound_job_input_mutation(
    tmp_path: Path,
    input_name: str,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    contract = module.verify_contract(
        **inputs,
        detached_script=SCRIPT_PATH,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
    )
    target = next(row for row in contract["job_inputs"] if row["model_id"] == TARGET_MODEL_ID)
    binding_name = (
        "speaker_inversion_init_embedding" if input_name == "init_embedding" else input_name
    )
    Path(target[binding_name]["path"]).write_bytes(b"mutated")

    with pytest.raises((RuntimeError, ValueError), match=r"training contract changed|SHA-256"):
        module._reverify_contract(  # noqa: SLF001 - verifies the immutable boundary.
            contract,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            require_status_before=True,
        )


def test_preflight_rejects_generic_train_py_and_pinned_target_config_worker(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    jobs = json.loads(inputs["jobs_path"].read_text(encoding="utf-8"))["jobs"]
    target_config = next(row["config"] for row in jobs if row["model_id"] == TARGET_MODEL_ID)
    for command_line in (
        r"C:\Python\python.exe C:\other\train.py --config C:\other\config.json",
        f'python.exe worker.py --config "{target_config}"',
        f'python.exe worker.py --config="{target_config}"',
    ):
        snapshot = module.RuntimeSnapshot(
            observed_at="2026-08-03T00:00:00+00:00",
            processes=({"pid": 55, "command_line": command_line},),
            training_processes=(),
            service_processes=(),
            gpu_total_mib=12_000.0,
            gpu_used_mib=500.0,
            gpu_free_mib=11_500.0,
            errors=(),
        )
        with pytest.raises(RuntimeError, match="training-owned process"):
            module.preflight_detached(
                **inputs,
                detached_script=SCRIPT_PATH,
                probe=lambda snapshot=snapshot: snapshot,
                runner=_dry_run_runner,
                git_runner=_git_runner(inputs["expected_upstream_commit"]),
            )


@pytest.mark.parametrize("job_count", [11, 13])
def test_verify_contract_rejects_non_twelve_job_count(tmp_path: Path, job_count: int) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    document = json.loads(inputs["jobs_path"].read_text(encoding="utf-8"))
    document["jobs"] = (document["jobs"] * 2)[:job_count]
    inputs["jobs_path"].write_text(json.dumps(document) + "\n", encoding="utf-8")
    inputs["expected_jobs_sha256"] = _sha256(inputs["jobs_path"])

    with pytest.raises(ValueError, match="exactly 12 jobs"):
        module.verify_contract(
            **inputs,
            detached_script=SCRIPT_PATH,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )


@pytest.mark.parametrize("binding_name", ["clean_manifest", "config"])
def test_reverify_contract_rejects_non_target_input_mutation(
    tmp_path: Path,
    binding_name: str,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    contract = module.verify_contract(
        **inputs,
        detached_script=SCRIPT_PATH,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
    )
    non_target = next(row for row in contract["job_inputs"] if row["model_id"] != TARGET_MODEL_ID)
    Path(non_target[binding_name]["path"]).write_bytes(b"mutated non-target input")

    with pytest.raises(RuntimeError, match="training contract changed"):
        module._reverify_contract(  # noqa: SLF001
            contract,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            require_status_before=True,
        )


@pytest.mark.parametrize("unsafe", ["training", "service", "gpu", "probe"])
def test_preflight_fails_closed_before_dry_run_for_unsafe_runtime(
    tmp_path: Path,
    unsafe: str,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    training = ({"pid": 10, "command_line": "run_600m_speaker_training_queue.py"},)
    service = ({"pid": 11, "command_line": "uvicorn irodori_tts_infra.server"},)
    snapshot = module.RuntimeSnapshot(
        observed_at="2026-08-03T00:00:00+00:00",
        processes=(),
        training_processes=training if unsafe == "training" else (),
        service_processes=service if unsafe == "service" else (),
        gpu_total_mib=12_000.0,
        gpu_used_mib=2_000.0 if unsafe == "gpu" else 1_000.0,
        gpu_free_mib=10_000.0 if unsafe == "gpu" else 11_000.0,
        errors=("probe failed",) if unsafe == "probe" else (),
    )
    calls: list[tuple[str, ...]] = []

    with pytest.raises(RuntimeError):
        module.preflight_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: snapshot,
            runner=lambda command, **_kwargs: calls.append(command),
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
        )

    assert calls == []


def _dry_run_runner(
    command: tuple[str, ...],
    **_kwargs: object,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=json.dumps(
            {
                "planned": [TARGET_MODEL_ID],
                "succeeded": [],
                "failed": [],
                "skipped": list(SKIPPED_MODEL_IDS),
            }
        )
        + "\n",
        stderr="",
    )


def test_reserve_launch_directory_is_create_only_and_versioned(tmp_path: Path) -> None:
    module = _load_script()
    launches = tmp_path / "speaker-training-detached-v001/launches"

    first = module.reserve_launch_directory(launches)
    second = module.reserve_launch_directory(launches)

    assert first.name == "launch-v001"
    assert second.name == "launch-v002"
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        module.reserve_launch_directory(launches, "launch-v001")
    with pytest.raises(ValueError, match="invalid launch id"):
        module.reserve_launch_directory(launches, "latest")


def test_launch_creates_two_stage_handoff_and_active_locks(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    spawns: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class Process:
        pid = 321

    def popen(command: tuple[str, ...], **kwargs: object) -> Process:
        spawns.append((command, kwargs))
        launch_dir = Path(command[command.index("--launch-dir") + 1])
        token = command[command.index("--token") + 1]
        lock_path = launch_dir / "supervisor.lock"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["supervisor_pid"] = 654
        lock["launch_token"] = token
        lock_path.write_text(json.dumps(lock) + "\n", encoding="utf-8")
        queue_lock = Path(inputs["status_path"]).with_suffix(".jsonl.detached.lock")
        queue_payload = json.loads(queue_lock.read_text(encoding="utf-8"))
        queue_payload["supervisor_pid"] = 654
        queue_lock.write_text(json.dumps(queue_payload) + "\n", encoding="utf-8")
        return Process()

    result = module.launch_detached(
        **inputs,
        detached_script=SCRIPT_PATH,
        probe=lambda: _safe_snapshot(module),
        runner=_dry_run_runner,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
        popen=popen,
        launch_id="launch-v007",
    )

    assert result["status"] == "DETACHED"
    assert result["spawn_pid"] == 321
    assert result["supervisor_pid"] == 654
    launch_dir = Path(result["launch_dir"])
    reservation_path = launch_dir / "reservation-evidence.json"
    handoff_path = launch_dir / "parent-handoff-evidence.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert handoff["previous_evidence"] == {
        "path": str(reservation_path.resolve()),
        "sha256": _sha256(reservation_path),
        "size": reservation_path.stat().st_size,
    }
    assert handoff["contract"] == reservation["contract"]
    assert (launch_dir / "supervisor.lock").is_file()
    assert Path(reservation["output_identity"]["queue_lock_path"]).is_file()
    active_lock = json.loads((launch_dir / "supervisor.lock").read_text(encoding="utf-8"))
    assert active_lock["mutation_protocol"] == module.LOCK_MUTATION_PROTOCOL
    assert active_lock["launcher_sha256"] == _sha256(SCRIPT_PATH)
    assert spawns[0][1]["creationflags"] == module.WINDOWS_DETACHED_FLAGS
    assert spawns[0][1]["stdin"] is subprocess.DEVNULL
    assert spawns[0][1]["stdout"] is subprocess.DEVNULL
    assert spawns[0][1]["stderr"] is subprocess.DEVNULL
    assert spawns[0][1]["shell"] is False


def test_launch_reverifies_bound_inputs_immediately_before_handoff(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    class Process:
        pid = 321

    def popen(command: tuple[str, ...], **_kwargs: object) -> Process:
        launch_dir = Path(command[command.index("--launch-dir") + 1])
        for path in (
            launch_dir / "supervisor.lock",
            Path(inputs["status_path"]).with_suffix(".jsonl.detached.lock"),
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["supervisor_pid"] = 654
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        jobs = json.loads(inputs["jobs_path"].read_text(encoding="utf-8"))["jobs"]
        target_config = next(
            Path(row["config"]) for row in jobs if row["model_id"] == TARGET_MODEL_ID
        )
        target_config.write_bytes(b"mutated after worker registration")
        return Process()

    with pytest.raises(RuntimeError, match="training contract changed"):
        module.launch_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: _safe_snapshot(module),
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            popen=popen,
        )

    launch_dir = inputs["evidence_root"] / "launches/launch-v001"
    assert not (launch_dir / "parent-handoff-evidence.json").exists()
    assert json.loads((launch_dir / "spawn-failure-evidence.json").read_text())["status"] == (
        "SPAWN_FAILED"
    )


def test_launch_spawn_failure_archives_both_locks_without_overwrite(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    def fail_spawn(*_args: object, **_kwargs: object) -> object:
        raise OSError("spawn failed")

    with pytest.raises(OSError, match="spawn failed"):
        module.launch_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: _safe_snapshot(module),
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            popen=fail_spawn,
            launch_id="launch-v003",
        )

    launch_dir = inputs["evidence_root"] / "launches/launch-v003"
    failure = json.loads((launch_dir / "spawn-failure-evidence.json").read_text(encoding="utf-8"))
    assert failure["status"] == "SPAWN_FAILED"
    assert (launch_dir / "supervisor-lock-final.json").is_file()
    assert (launch_dir / "queue-lock-final.json").is_file()
    assert not (launch_dir / "supervisor.lock").exists()
    assert not Path(failure["output_identity"]["queue_lock_path"]).exists()
    with pytest.raises(FileExistsError):
        module._write_json_create_only(  # noqa: SLF001 - verifies artifact boundary.
            launch_dir / "spawn-failure-evidence.json",
            {"status": "overwritten"},
        )


def test_archive_lock_does_not_unlink_a_replacement_foreign_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    owner_token = "a" * 32
    lock = tmp_path / "queue.lock"
    own = module._lock_payload(  # noqa: SLF001
        schema=module.QUEUE_LOCK_SCHEMA,
        token=owner_token,
        supervisor_pid=22,
        status_path="status.jsonl",
    )
    foreign = own | {"launch_token": "foreign-token", "supervisor_pid": 33}
    module._write_json_create_only(lock, own)  # noqa: SLF001
    original_write = module._write_bytes_create_only  # noqa: SLF001

    def swap_after_archive_write(path: Path, content: bytes) -> None:
        original_write(path, content)
        lock.write_text(json.dumps(foreign) + "\n", encoding="utf-8")

    monkeypatch.setattr(module, "_write_bytes_create_only", swap_after_archive_write)
    destination = tmp_path / "queue-final.json"

    with pytest.raises(RuntimeError, match="lock ownership changed"):
        module._archive_lock(lock, destination, token=owner_token)  # noqa: SLF001

    assert json.loads(lock.read_text(encoding="utf-8")) == foreign


def test_update_lock_does_not_overwrite_a_replacement_foreign_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    owner_token = "a" * 32
    lock = tmp_path / "supervisor.lock"
    own = module._lock_payload(  # noqa: SLF001
        schema=module.LOCK_SCHEMA,
        token=owner_token,
        supervisor_pid=None,
    )
    foreign = own | {"launch_token": "foreign-token", "supervisor_pid": 33}
    module._write_json_create_only(lock, own)  # noqa: SLF001
    original_json_bytes = module._json_bytes  # noqa: SLF001
    swapped = False

    def swap_before_replace(payload: dict[str, object]) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            lock.write_text(json.dumps(foreign) + "\n", encoding="utf-8")
        return cast("bytes", original_json_bytes(payload))

    monkeypatch.setattr(module, "_json_bytes", swap_before_replace)

    with pytest.raises(RuntimeError, match="lock ownership changed"):
        module._update_lock(  # noqa: SLF001
            lock,
            schema=module.LOCK_SCHEMA,
            token=owner_token,
            supervisor_pid=22,
        )

    assert json.loads(lock.read_text(encoding="utf-8")) == foreign


@pytest.mark.parametrize("operation", ["update", "archive"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "unknown-lock/v1"),
        ("mutation_protocol", "unknown-protocol/v1"),
        ("launcher_sha256", "0" * 64),
    ],
)
def test_lock_mutation_rejects_unknown_protocol_or_launcher_binding(
    tmp_path: Path,
    operation: str,
    field: str,
    value: str,
) -> None:
    module = _load_script()
    owner_token = "a" * 32
    lock = tmp_path / "supervisor.lock"
    payload = module._lock_payload(  # noqa: SLF001
        schema=module.LOCK_SCHEMA,
        token=owner_token,
        supervisor_pid=None,
    )
    payload[field] = value
    module._write_json_create_only(lock, payload)  # noqa: SLF001

    def mutate() -> None:
        if operation == "update":
            module._update_lock(  # noqa: SLF001
                lock,
                schema=module.LOCK_SCHEMA,
                token=owner_token,
                supervisor_pid=22,
            )
        else:
            module._archive_lock(  # noqa: SLF001
                lock,
                tmp_path / "final.json",
                token=owner_token,
            )

    with pytest.raises(RuntimeError, match="lock ownership mismatch"):
        mutate()

    assert json.loads(lock.read_text(encoding="utf-8")) == payload


def test_launch_cleanup_ownership_loss_records_chain_without_touching_foreign_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    queue_lock = Path(inputs["status_path"]).with_suffix(".jsonl.detached.lock")
    foreign = {
        "schema_version": module.QUEUE_LOCK_SCHEMA,
        "launch_token": "b" * 32,
        "hostname": "foreign-host",
        "launcher_pid": 77,
        "supervisor_pid": 88,
        "updated_at": "2026-08-03T00:00:00+00:00",
        "status_path": str(inputs["status_path"].resolve()),
    }
    original_write = module._write_bytes_create_only  # noqa: SLF001

    def swap_queue_during_archive(path: Path, content: bytes) -> None:
        original_write(path, content)
        if path.name == "queue-lock-final.json":
            queue_lock.write_text(json.dumps(foreign) + "\n", encoding="utf-8")

    monkeypatch.setattr(module, "_write_bytes_create_only", swap_queue_during_archive)

    class InvalidProcess:
        pid = 0

    with pytest.raises(RuntimeError, match="invalid PID"):
        module.launch_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: _safe_snapshot(module),
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            popen=lambda *_args, **_kwargs: InvalidProcess(),
        )

    assert json.loads(queue_lock.read_text(encoding="utf-8")) == foreign
    status = module.read_status(inputs["evidence_root"])
    assert status["status"] == "SUPERVISOR_ERROR"
    assert status["evidence"]["foreign_queue_lock"]["sha256"] == _sha256(queue_lock)


def _launch_queue_lock_race(
    module: ModuleType,
    inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    queue_lock = Path(inputs["status_path"]).with_suffix(".jsonl.detached.lock")
    original_write = module._write_json_create_only  # noqa: SLF001
    foreign: dict[str, object] = {
        "schema_version": module.QUEUE_LOCK_SCHEMA,
        "launch_token": "foreign-token",
        "hostname": "foreign-host",
        "launcher_pid": 77,
        "supervisor_pid": 88,
        "updated_at": "2026-08-03T00:00:00+00:00",
        "status_path": str(inputs["status_path"].resolve()),
    }

    def racing_write(path: Path, payload: dict[str, object]) -> None:
        original_write(path, payload)
        if path.name == "supervisor.lock":
            original_write(queue_lock, foreign)

    monkeypatch.setattr(module, "_write_json_create_only", racing_write)
    with pytest.raises(FileExistsError):
        module.launch_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: _safe_snapshot(module),
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            launch_id="launch-v002",
        )
    return inputs["evidence_root"] / "launches/launch-v002", queue_lock, foreign


def test_launch_queue_lock_race_retains_foreign_lock_and_closes_reservation_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launch_dir, queue_lock, foreign = _launch_queue_lock_race(module, inputs, monkeypatch)
    assert json.loads(queue_lock.read_text(encoding="utf-8")) == foreign
    assert not (launch_dir / "supervisor.lock").exists()
    assert (launch_dir / "supervisor-lock-final.json").is_file()
    failure_path = launch_dir / "reservation-failure-evidence.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "RESERVATION_FAILED"
    snapshot_path = launch_dir / "foreign-queue-lock-snapshot.json"
    assert failure["foreign_queue_lock"] == module._file_binding(snapshot_path)  # noqa: SLF001
    assert failure["foreign_queue_lock_path"] == str(queue_lock.resolve())
    assert snapshot_path.read_bytes() == queue_lock.read_bytes()
    status = module.read_status(inputs["evidence_root"])
    assert status["status"] == "RESERVATION_FAILED"
    assert status["evidence_path"] == str(failure_path.resolve())
    failure["foreign_queue_lock"]["path"] = str(tmp_path / "attacker.lock")
    failure_path.write_text(json.dumps(failure) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="foreign lock snapshot binding mismatch"):
        module.read_status(inputs["evidence_root"])


@pytest.mark.parametrize("winner_state", ["pid_updated", "lock_deleted"])
def test_latest_reservation_failure_ignores_winner_live_lock_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner_state: str,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launch_dir, queue_lock, foreign = _launch_queue_lock_race(module, inputs, monkeypatch)
    if winner_state == "pid_updated":
        changed_foreign = foreign | {
            "supervisor_pid": 99,
            "updated_at": "2026-08-03T00:01:00+00:00",
        }
        queue_lock.write_text(json.dumps(changed_foreign) + "\n", encoding="utf-8")
    else:
        queue_lock.unlink()

    status = module.read_status(inputs["evidence_root"])
    assert status["status"] == "RESERVATION_FAILED"
    assert status["launch_dir"] == str(launch_dir.resolve())


def test_supervisor_does_not_fork_spawn_failure_evidence_chain(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)

    def fail_spawn(*_args: object, **_kwargs: object) -> object:
        raise OSError("spawn failed")

    with pytest.raises(OSError):
        module.launch_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: _safe_snapshot(module),
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            popen=fail_spawn,
        )
    launch_dir = inputs["evidence_root"] / "launches/launch-v001"
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text(encoding="utf-8"))

    module._record_bootstrap_error(  # noqa: SLF001 - simulates a late worker wakeup.
        launch_dir=launch_dir,
        token=reservation["launch_token"],
        error=RuntimeError("parent already recorded failure"),
    )

    assert not (launch_dir / "supervisor-bootstrap-error.json").exists()
    assert module.read_status(inputs["evidence_root"])["status"] == "SPAWN_FAILED"


def _detached_launch(module: ModuleType, inputs: dict[str, Any]) -> dict[str, object]:
    class Process:
        pid = 321

    def popen(command: tuple[str, ...], **_kwargs: object) -> Process:
        launch_dir = Path(command[command.index("--launch-dir") + 1])
        for path in (
            launch_dir / "supervisor.lock",
            Path(inputs["status_path"]).with_suffix(".jsonl.detached.lock"),
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["supervisor_pid"] = 654
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return Process()

    return cast(
        "dict[str, object]",
        module.launch_detached(
            **inputs,
            detached_script=SCRIPT_PATH,
            probe=lambda: _safe_snapshot(module),
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            popen=popen,
        ),
    )


def _append_target_success_status(
    inputs: dict[str, Any],
    contract: dict[str, object],
) -> None:
    job_inputs = cast("list[dict[str, Any]]", contract["job_inputs"])
    checkpoint_binding = cast("dict[str, Any]", contract["checkpoint"])
    upstream = cast("dict[str, Any]", contract["upstream"])
    target = next(row for row in job_inputs if row["model_id"] == TARGET_MODEL_ID)
    output_dir = Path(target["output_dir"])
    checkpoint = _write(output_dir / "checkpoint-step-3000.speaker.safetensors", b"result")
    base = {
        "model_id": TARGET_MODEL_ID,
        "clean_manifest_sha256": target["clean_manifest"]["sha256"],
        "checkpoint_sha256": checkpoint_binding["sha256"],
        "checkpoint_revision": contract["checkpoint_revision"],
        "config_sha256": target["config"]["sha256"],
        "upstream_commit": upstream["commit"],
        "started_at": "2026-08-03T00:01:00+00:00",
        "ended_at": None,
        "exit_code": None,
        "log_path": "target.log",
        "last_checkpoint": None,
        "last_checkpoint_sha256": None,
        "candidate_checkpoints": [],
        "error": None,
    }
    rows = (
        base | {"event": "started", "status": "running"},
        base
        | {
            "event": "finished",
            "status": "success",
            "ended_at": "2026-08-03T01:01:00+00:00",
            "exit_code": 0,
            "last_checkpoint": str(checkpoint.resolve()),
            "last_checkpoint_sha256": _sha256(checkpoint),
            "candidate_checkpoints": [
                {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)}
            ],
        },
    )
    with Path(inputs["status_path"]).open("a", encoding="utf-8") as status:
        status.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def test_supervisor_runs_queue_to_terminal_evidence_and_logs_output(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text(encoding="utf-8"))
    token = reservation["launch_token"]
    popen_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class QueueProcess:
        def wait(self) -> int:
            return 0

    def popen(command: tuple[str, ...], **kwargs: object) -> QueueProcess:
        popen_calls.append((command, kwargs))
        output = cast("BinaryIO", kwargs["stdout"])
        output.write(
            (
                "noise\n"
                + json.dumps(
                    {
                        "planned": [TARGET_MODEL_ID],
                        "succeeded": [TARGET_MODEL_ID],
                        "failed": [],
                        "skipped": list(SKIPPED_MODEL_IDS),
                    }
                )
                + "\n"
            ).encode()
        )
        _append_target_success_status(inputs, reservation["contract"])
        return QueueProcess()

    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=token,
        probe=lambda: _safe_snapshot(module),
        popen=popen,
        runner=_dry_run_runner,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
        current_pid=654,
    )

    assert exit_code == 0
    terminal_path = launch_dir / "terminal-final-evidence.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["status"] == "SUCCEEDED"
    assert terminal["queue_exit_code"] == 0
    assert terminal["queue_summary"]["failed"] == []
    assert terminal["target_status_delta"]["model_id"] == TARGET_MODEL_ID
    assert terminal["queue_log"]["sha256"] == _sha256(launch_dir / "queue.log")
    assert terminal["queue_status"]["sha256"] == _sha256(inputs["status_path"])
    assert not (launch_dir / "supervisor.lock").exists()
    assert not Path(reservation["output_identity"]["queue_lock_path"]).exists()
    assert (launch_dir / "supervisor-lock-final.json").is_file()
    assert (launch_dir / "queue-lock-final.json").is_file()
    assert popen_calls[0][1]["stderr"] is subprocess.STDOUT
    assert popen_calls[0][1]["stdin"] is subprocess.DEVNULL
    assert popen_calls[0][1]["shell"] is False
    child_env = cast("dict[str, str]", popen_calls[0][1]["env"])
    assert child_env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert child_env["PATH"] == os.environ["PATH"]
    assert "--dry-run" not in popen_calls[0][0]
    status = module.read_status(inputs["evidence_root"])
    assert status["status"] == "SUCCEEDED"
    assert status["evidence_path"] == str(terminal_path.resolve())


def test_supervisor_records_failed_terminal_for_nonzero_queue(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text(encoding="utf-8"))

    class QueueProcess:
        def wait(self) -> int:
            return 9

    def popen(_command: tuple[str, ...], **kwargs: object) -> QueueProcess:
        output = cast("BinaryIO", kwargs["stdout"])
        output.write(b'{"planned":["model-00"],"failed":["model-00"]}\n')
        return QueueProcess()

    exit_code = module.run_supervisor(
        launch_dir=launch_dir,
        token=reservation["launch_token"],
        probe=lambda: _safe_snapshot(module),
        popen=popen,
        runner=_dry_run_runner,
        git_runner=_git_runner(inputs["expected_upstream_commit"]),
        current_pid=654,
    )

    assert exit_code == 1
    terminal = json.loads((launch_dir / "terminal-final-evidence.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "FAILED"
    assert terminal["queue_exit_code"] == 9
    assert terminal["queue_summary"]["failed"] == ["model-00"]


def test_supervisor_rejects_zero_work_success(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text())

    class QueueProcess:
        def wait(self) -> int:
            return 0

    def popen(_command: tuple[str, ...], **kwargs: object) -> QueueProcess:
        cast("BinaryIO", kwargs["stdout"]).write(
            b'{"planned":[],"succeeded":[],"failed":[],"skipped":[]}\n'
        )
        return QueueProcess()

    assert (
        module.run_supervisor(
            launch_dir=launch_dir,
            token=reservation["launch_token"],
            probe=lambda: _safe_snapshot(module),
            popen=popen,
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            current_pid=654,
        )
        == 1
    )
    terminal = json.loads((launch_dir / "terminal-final-evidence.json").read_text())
    assert terminal["status"] == "FAILED"
    assert "queue completion report is unsafe" in terminal["error"]


@pytest.mark.parametrize(
    "tampering",
    ["extra_row", "target_mismatch", "provenance_mismatch", "candidate_escape", "candidate_hash"],
)
def test_supervisor_rejects_invalid_target_status_delta(
    tmp_path: Path,
    tampering: str,
) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text())

    class QueueProcess:
        def wait(self) -> int:
            return 0

    def popen(_command: tuple[str, ...], **kwargs: object) -> QueueProcess:
        cast("BinaryIO", kwargs["stdout"]).write(
            (
                json.dumps(
                    {
                        "planned": [TARGET_MODEL_ID],
                        "succeeded": [TARGET_MODEL_ID],
                        "failed": [],
                        "skipped": list(SKIPPED_MODEL_IDS),
                    }
                )
                + "\n"
            ).encode()
        )
        _append_target_success_status(inputs, reservation["contract"])
        status_path = Path(inputs["status_path"])
        rows = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
        if tampering == "extra_row":
            rows.append({"event": "unexpected"})
        elif tampering == "target_mismatch":
            rows[-2]["model_id"] = MODEL_IDS[0]
        elif tampering == "provenance_mismatch":
            rows[-1]["config_sha256"] = "0" * 64
        elif tampering == "candidate_escape":
            outside = _write(tmp_path / "outside.speaker.safetensors", b"outside")
            rows[-1]["candidate_checkpoints"] = [
                {"path": str(outside.resolve()), "sha256": _sha256(outside)}
            ]
            rows[-1]["last_checkpoint"] = str(outside.resolve())
            rows[-1]["last_checkpoint_sha256"] = _sha256(outside)
        else:
            rows[-1]["candidate_checkpoints"][-1]["sha256"] = "0" * 64
            rows[-1]["last_checkpoint_sha256"] = "0" * 64
        status_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return QueueProcess()

    assert (
        module.run_supervisor(
            launch_dir=launch_dir,
            token=reservation["launch_token"],
            probe=lambda: _safe_snapshot(module),
            popen=popen,
            runner=_dry_run_runner,
            git_runner=_git_runner(inputs["expected_upstream_commit"]),
            current_pid=654,
        )
        == 1
    )
    terminal = json.loads((launch_dir / "terminal-final-evidence.json").read_text())
    assert terminal["status"] == "FAILED"
    assert terminal["target_status_delta"] is None
    assert terminal["error"].startswith("RuntimeError: training")


def test_status_rejects_tampered_immutable_evidence(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    reservation_path = launch_dir / "reservation-evidence.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation["created_at"] = "tampered"
    reservation_path.write_text(json.dumps(reservation) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="binding mismatch"):
        module.read_status(inputs["evidence_root"])


def test_reserved_status_rejects_tampered_active_queue_lock(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    (launch_dir / "parent-handoff-evidence.json").unlink()
    queue_lock = Path(inputs["status_path"]).with_suffix(".jsonl.detached.lock")
    payload = json.loads(queue_lock.read_text(encoding="utf-8"))
    payload["status_path"] = str(tmp_path / "attacker-status.jsonl")
    queue_lock.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="lock mismatch"):
        module.read_status(inputs["evidence_root"])


def test_status_reports_not_launched_without_creating_artifacts(tmp_path: Path) -> None:
    module = _load_script()
    evidence_root = tmp_path / "speaker-training-detached-v001"

    status = module.read_status(evidence_root)

    assert status["status"] == "NOT_LAUNCHED"
    assert not evidence_root.exists()


def test_bootstrap_failure_archives_locks_and_extends_handoff_chain(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text(encoding="utf-8"))

    try:
        raise RuntimeError("bootstrap broke")
    except RuntimeError as exc:
        module._record_bootstrap_error(  # noqa: SLF001 - exercises supervisor boundary.
            launch_dir=launch_dir,
            token=reservation["launch_token"],
            error=exc,
        )

    status = module.read_status(inputs["evidence_root"])
    assert status["status"] == "SUPERVISOR_ERROR"
    evidence = status["evidence"]
    assert evidence["error"] == "RuntimeError: bootstrap broke"
    assert evidence["previous_evidence"]["path"].endswith("parent-handoff-evidence.json")
    assert not (launch_dir / "supervisor.lock").exists()
    assert not Path(reservation["output_identity"]["queue_lock_path"]).exists()
    assert (launch_dir / "supervisor-lock-final.json").is_file()
    assert (launch_dir / "queue-lock-final.json").is_file()


def test_bootstrap_failure_reuses_locks_archived_before_terminal_write(tmp_path: Path) -> None:
    module = _load_script()
    inputs = _inputs(tmp_path)
    launched = _detached_launch(module, inputs)
    launch_dir = Path(cast("str", launched["launch_dir"]))
    reservation = json.loads((launch_dir / "reservation-evidence.json").read_text(encoding="utf-8"))
    token = reservation["launch_token"]
    module._archive_lock(  # noqa: SLF001 - simulates the post-queue terminal boundary.
        launch_dir / "supervisor.lock",
        launch_dir / "supervisor-lock-final.json",
        token=token,
    )
    module._archive_lock(  # noqa: SLF001 - simulates the post-queue terminal boundary.
        Path(reservation["output_identity"]["queue_lock_path"]),
        launch_dir / "queue-lock-final.json",
        token=token,
    )

    try:
        raise RuntimeError("terminal write broke")
    except RuntimeError as exc:
        module._record_bootstrap_error(  # noqa: SLF001 - exercises recovery boundary.
            launch_dir=launch_dir,
            token=token,
            error=exc,
        )

    assert module.read_status(inputs["evidence_root"])["status"] == "SUPERVISOR_ERROR"


def test_cli_requires_explicit_pins_for_preflight() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="requires --queue-script"):
        module.main(["preflight", "--evidence-root", "speaker-training-detached-v001"])


def test_cli_status_requires_only_versioned_evidence_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    evidence_root = tmp_path / "speaker-training-detached-v001"

    assert module.main(["status", "--evidence-root", str(evidence_root)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "NOT_LAUNCHED"
