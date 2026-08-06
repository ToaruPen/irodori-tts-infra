from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v2.py")
ANABEL_MODEL_ID = "oop77_anabel_maidgarden_sp_451488a7c1"
MODEL_COUNT = 12


class QueueResult(Protocol):
    succeeded: tuple[str, ...]


class LauncherModule(Protocol):
    REMOTE_ROOT: Path
    DEFAULT_OUTPUT_ROOT: Path
    DEFAULT_CONFIG_PATH: Path
    DEFAULT_STATUS_PATH: Path
    RUNTIME_SNAPSHOT_NAME: str
    PENDING_CONFIG_SHA256: str
    EXPECTED_CONFIG_SHA256: str
    EXPECTED_SOURCE_CONFIG_SHA256: str
    EXPECTED_JOBS_SHA256: str
    EXPECTED_COMPONENT_SHA256: Mapping[str, str]

    def preflight(
        self,
        *,
        config_path: Path,
        status_path: Path,
        scripts_dir: Path,
        output_root: Path,
        expected_config_sha256: str,
        expected_jobs_path: Path,
        expected_jobs_sha256: str,
        expected_component_sha256: Mapping[str, str],
    ) -> dict[str, object]: ...

    def launch(
        self,
        *,
        config_path: Path,
        status_path: Path,
        scripts_dir: Path,
        output_root: Path,
        expected_config_sha256: str,
        expected_jobs_path: Path,
        expected_jobs_sha256: str,
        expected_component_sha256: Mapping[str, str],
    ) -> QueueResult: ...

    def prepare_speed_v4_config(
        self,
        *,
        source_path: Path,
        destination: Path,
        status_path: Path,
        output_root: Path,
        expected_source_sha256: str,
        jobs_path: Path,
        expected_jobs_sha256: str,
    ) -> dict[str, object]: ...

    def main(self, argv: Sequence[str] | None = None) -> int: ...


class Fixture(TypedDict):
    module: LauncherModule
    output_root: Path
    config_path: Path
    status_path: Path
    jobs_path: Path
    jobs_document: dict[str, object]
    training_status: Path
    model_ids: list[str]
    module_exec_log: Path
    scripts_dir: Path
    config_sha256: str
    jobs_sha256: str
    pins: dict[str, str]


class LaunchArguments(TypedDict):
    config_path: Path
    status_path: Path
    scripts_dir: Path
    output_root: Path
    expected_config_sha256: str
    expected_jobs_path: Path
    expected_jobs_sha256: str
    expected_component_sha256: Mapping[str, str]


def _load_script() -> LauncherModule:
    spec = importlib.util.spec_from_file_location(
        "launch_600m_speaker_evaluation_queue_speed_v2",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("LauncherModule", module)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_versioned_retry_defaults_target_new_speed_v4_root() -> None:
    module = _load_script()

    assert module.DEFAULT_OUTPUT_ROOT == module.REMOTE_ROOT / "evaluation_speed_v4"
    assert module.DEFAULT_CONFIG_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-queue-speed-v4.json"
    )
    assert module.DEFAULT_STATUS_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v4.jsonl"
    )
    assert module.RUNTIME_SNAPSHOT_NAME == "runtime-inputs-v1"


def test_versioned_retry_pins_reviewed_inputs_and_prepared_config() -> None:
    module = _load_script()

    assert module.EXPECTED_CONFIG_SHA256 == (
        "472fb1ec315def4423a0cc60ad9169a6c4395a6b587a21ea128b1ef9e332b5ce"
    )
    assert module.EXPECTED_CONFIG_SHA256 != module.PENDING_CONFIG_SHA256
    assert module.EXPECTED_SOURCE_CONFIG_SHA256 == (
        "33109e7ea9b62b014d59ce0673ea3d4e50c45f9e60cebf13e06463e2e5e4fd02"
    )
    assert module.EXPECTED_JOBS_SHA256 == (
        "206f8fe9d1428a5aa9426c215ee5f092e4546e44fd4bc094e92478180ef163c6"
    )
    assert module.EXPECTED_COMPONENT_SHA256 == {
        "run_600m_speaker_evaluation_queue.py": (
            "60e2456a5d51e6a4b935a64fdbd0140647f187d1599350039f51f77b7dbfef70"
        ),
        "build_600m_checkpoint_evaluation_manifests.py": (
            "e3f62e07f07c949fe60d4db00a7eef11dbd1ae9111a7628dd88982a2702d0e93"
        ),
        "generate_600m_checkpoint_audio_remote.py": (
            "9688f956fb2be1f148f4583a77f41f5774e033a93f6d25dc0db835412b099312"
        ),
        "analyze_nko_beep_matrix.py": (
            "06c23b489975843bf080b7fa70ebb41abac19c269b6cfe5bcafec019a0693dc1"
        ),
        "compute_600m_speaker_metrics.py": (
            "fa83491f0ee2f1e1f21c8d833ba90557ed67c885a2512c9c05104faf3b14a407"
        ),
        "evaluate_600m_speaker_checkpoints.py": (
            "cb28b8541956fcfdb549e1ee6e87905176ab4acf8b96809fd6da60655a10c738"
        ),
    }


@pytest.mark.parametrize("mode", ["preflight", "launch"])
def test_pending_config_pin_blocks_nonprepare_modes(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "EXPECTED_CONFIG_SHA256", module.PENDING_CONFIG_SHA256)

    with pytest.raises(ValueError, match="PENDING_REMOTE_PREPARE_CONFIG_SHA256"):
        module.main([mode])


def test_just_catalog_exposes_only_speed_v4_retry_recipes() -> None:
    catalog = Path("justfile").read_text(encoding="utf-8")

    assert "speaker-evaluation-speed-v4 *args:" in catalog
    assert "remote-speaker-evaluation-speed-v4 *args:" in catalog
    assert "speaker-evaluation-speed-v3 *args:" not in catalog
    assert "remote-speaker-evaluation-speed-v3 *args:" not in catalog
    assert "speaker-evaluation-speed-v2 *args:" not in catalog
    assert "remote-speaker-evaluation-speed-v2 *args:" not in catalog


def test_evaluation_module_executes_verified_source_instead_of_stale_pyc(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source_path = tmp_path / "pinned_component.py"
    source_path.write_text("VALUE = 'old'\n", encoding="utf-8")
    original_stat = source_path.stat()

    py_compile.compile(str(source_path), doraise=True)
    assert next((tmp_path / "__pycache__").glob("pinned_component.*.pyc")).is_file()

    source_path.write_text("VALUE = 'new'\n", encoding="utf-8")
    os.utime(
        source_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    loaded = getattr(module, "_load_evaluation_module")(source_path)  # noqa: B009

    assert loaded.VALUE == "new"


def _write_fixture(  # noqa: PLR0914 - fixture materializes the complete remote contract.
    tmp_path: Path,
    *,
    ready: bool = True,
) -> Fixture:
    output_root = tmp_path / "evaluation-speed-v4"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    evaluation_script = scripts_dir / "run_600m_speaker_evaluation_queue.py"
    module_exec_log = tmp_path / "module-executions.log"
    evaluation_source = """
import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MODULE_EXEC_LOG = Path(__MODULE_EXEC_LOG__)
with MODULE_EXEC_LOG.open('a', encoding='utf-8') as output:
    output.write('exec\\n')

@dataclass
class Config:
    source_path: Path
    source_sha256: str
    training_status: Path
    training_jobs: Path
    mutation_target: Path | None
    mutation_kind: str | None

@dataclass
class Result:
    succeeded: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    reused: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()

def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_queue_config(path):
    document = _read_json(path)
    return Config(
        source_path=path,
        source_sha256=sha256_file(path),
        training_status=Path(document['training_status']),
        training_jobs=Path(document['training_jobs']),
        mutation_target=(
            Path(document['test_mutation_target'])
            if document.get('test_mutation_target')
            else None
        ),
        mutation_kind=document.get('test_mutation_kind'),
    )

def _apply_mutation(config, mutation_kind):
    if mutation_kind == 'component':
        config.mutation_target.write_text('# mutated after verification\\n', encoding='utf-8')
    elif mutation_kind == 'status':
        with config.mutation_target.open('a', encoding='utf-8') as output:
            output.write(json.dumps({
                'model_id': 'oop77_anabel_maidgarden_sp_451488a7c1',
                'event': 'started',
                'status': 'running',
                'exit_code': None,
            }) + '\\n')

def _validate_ready_training(config):
    if config.mutation_kind in {'component', 'status'}:
        _apply_mutation(config, config.mutation_kind)

@contextlib.contextmanager
def evaluation_queue_lock(*, config, status_path):
    yield status_path.with_suffix(status_path.suffix + '.lock')

def _result(status_path, scripts_dir, config, mode):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({
        'mode': mode,
        'scripts_dir': str(scripts_dir),
        'config_path': str(config.source_path),
        'config_sha256': config.source_sha256,
        'training_status': str(config.training_status),
        'training_jobs': str(config.training_jobs),
        'analyzer_source': (scripts_dir / 'analyze_nko_beep_matrix.py').read_text(
            encoding='utf-8'
        ),
    }), encoding='utf-8')
    return Result(succeeded=('manifests',))

def run_evaluation_queue(config, *, status_path, scripts_dir):
    _validate_ready_training(config)
    return _result(status_path, scripts_dir, config, 'unisolated')

def _run_evaluation_queue_locked(
    config,
    *,
    status_path,
    scripts_dir,
    runner=None,
    now=None,
):
    if config.mutation_kind in {'during-run-component', 'during-run-status'}:
        _apply_mutation(config, config.mutation_kind.removeprefix('during-run-'))
    return _result(status_path, scripts_dir, config, 'snapshot')
""".lstrip().replace("__MODULE_EXEC_LOG__", repr(str(module_exec_log)))
    evaluation_script.write_text(
        evaluation_source,
        encoding="utf-8",
    )
    component_names = (
        "generate_600m_checkpoint_audio_remote.py",
        "analyze_nko_beep_matrix.py",
        "compute_600m_speaker_metrics.py",
        "evaluate_600m_speaker_checkpoints.py",
        "build_600m_checkpoint_evaluation_manifests.py",
    )
    for name in component_names:
        (scripts_dir / name).write_text(f"# {name}\n", encoding="utf-8")

    model_ids = [ANABEL_MODEL_ID, *(f"model-{index:02d}" for index in range(1, 12))]
    checkpoint_path = tmp_path / "base-checkpoint.safetensors"
    checkpoint_path.write_bytes(b"base checkpoint")
    jobs: list[dict[str, object]] = []
    for model_id in model_ids:
        clean_manifest = tmp_path / "manifests" / model_id / "clean-manifest.jsonl"
        config = tmp_path / "manifests" / model_id / "training-config.json"
        clean_manifest.parent.mkdir(parents=True)
        clean_manifest.write_text(f'{{"model_id":"{model_id}"}}\n', encoding="utf-8")
        config.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
        jobs.append(
            {
                "model_id": model_id,
                "clean_manifest": str(clean_manifest),
                "config": str(config),
                "output_dir": str(tmp_path / "training" / model_id / "output"),
                "command": ["python", "train.py"],
            }
        )
    jobs_path = tmp_path / "training-jobs-speed-v1.json"
    jobs_document = {
        "schema_version": 1,
        "base_checkpoint_path": str(checkpoint_path),
        "base_checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_revision": "base-revision",
        "upstream_commit": "upstream-commit",
        "jobs": jobs,
    }
    jobs_path.write_text(
        json.dumps(jobs_document),
        encoding="utf-8",
    )
    config_path = output_root / "evaluation-queue-speed-v4.json"
    config_path.parent.mkdir(parents=True)
    models: list[dict[str, object]] = [
        {
            "model_id": ANABEL_MODEL_ID,
            "reference_wavs": str(tmp_path / "anabel-references.json"),
            "reuse": {
                "generation_dir": str(tmp_path / "canonical-anabel"),
                "analysis_dir": str(tmp_path / "canonical-anabel"),
                "metrics_results": str(tmp_path / "canonical-anabel/metrics.jsonl"),
                "metrics_provenance": str(tmp_path / "canonical-anabel/provenance.json"),
                "evaluation_manifest": str(tmp_path / "canonical-anabel/manifest.json"),
                "evaluation_dir": str(tmp_path / "canonical-anabel/selection"),
            },
        }
    ]
    for model_id in model_ids[1:]:
        model_root = output_root / "models" / model_id
        models.append(
            {
                "model_id": model_id,
                "reference_wavs": str(tmp_path / f"{model_id}-references.json"),
                "generation_dir": str(model_root / "generation"),
                "analysis_dir": str(model_root / "analysis"),
                "metrics_dir": str(model_root / "metrics"),
                "evaluation_dir": str(model_root / "selection"),
            }
        )
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-evaluation-queue/v1",
                "training_status": str(tmp_path / "training-status.jsonl"),
                "training_jobs": str(jobs_path),
                "manifest_output_dir": str(output_root / "manifests"),
                "metric_models": {
                    "speaker_embedding": {"savedir": str(output_root / "runtime-cache/ecapa")}
                },
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    status_rows = [
        {
            "model_id": job["model_id"],
            "event": "finished",
            "status": "success",
            "exit_code": 0,
            "clean_manifest_sha256": _sha256(Path(str(job["clean_manifest"]))),
            "config_sha256": _sha256(Path(str(job["config"]))),
            "checkpoint_sha256": jobs_document["base_checkpoint_sha256"],
            "checkpoint_revision": jobs_document["checkpoint_revision"],
            "upstream_commit": jobs_document["upstream_commit"],
        }
        for job in jobs
    ]
    if not ready:
        status_rows.append(
            {
                "model_id": model_ids[-1],
                "event": "started",
                "status": "running",
                "exit_code": None,
            }
        )
    training_status = tmp_path / "training-status.jsonl"
    training_status.write_text(
        "".join(json.dumps(row) + "\n" for row in status_rows),
        encoding="utf-8",
    )
    pins = {path.name: _sha256(path) for path in scripts_dir.iterdir() if path.is_file()}
    return {
        "module": _load_script(),
        "output_root": output_root,
        "config_path": config_path,
        "status_path": output_root / "evaluation-status-speed-v4.jsonl",
        "jobs_path": jobs_path,
        "jobs_document": jobs_document,
        "training_status": training_status,
        "model_ids": model_ids,
        "module_exec_log": module_exec_log,
        "scripts_dir": scripts_dir,
        "config_sha256": _sha256(config_path),
        "jobs_sha256": _sha256(jobs_path),
        "pins": pins,
    }


def _patch_operational_defaults(
    monkeypatch: pytest.MonkeyPatch,
    module: LauncherModule,
    fixture: Fixture,
) -> None:
    monkeypatch.setattr(module, "DEFAULT_OUTPUT_ROOT", fixture["output_root"])
    monkeypatch.setattr(module, "DEFAULT_CONFIG_PATH", fixture["config_path"])
    monkeypatch.setattr(module, "DEFAULT_STATUS_PATH", fixture["status_path"])
    monkeypatch.setattr(module, "DEFAULT_JOBS_PATH", fixture["jobs_path"])
    monkeypatch.setattr(module, "DEFAULT_SCRIPTS_DIR", fixture["scripts_dir"])
    monkeypatch.setattr(module, "EXPECTED_CONFIG_SHA256", fixture["config_sha256"])
    monkeypatch.setattr(module, "EXPECTED_JOBS_SHA256", fixture["jobs_sha256"])
    monkeypatch.setattr(module, "EXPECTED_COMPONENT_SHA256", fixture["pins"])


def test_preflight_is_read_only_and_verifies_all_pinned_inputs(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]

    result = module.preflight(
        config_path=fixture["config_path"],
        status_path=fixture["status_path"],
        scripts_dir=fixture["scripts_dir"],
        output_root=fixture["output_root"],
        expected_config_sha256=fixture["config_sha256"],
        expected_jobs_path=fixture["jobs_path"],
        expected_jobs_sha256=fixture["jobs_sha256"],
        expected_component_sha256=fixture["pins"],
    )

    assert result["passed"] is True
    assert result["model_count"] == MODEL_COUNT
    assert result["reused_model_ids"] == [ANABEL_MODEL_ID]
    assert not fixture["status_path"].exists()


def test_preflight_refuses_pending_training_without_starting_queue(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, ready=False)
    module = fixture["module"]

    with pytest.raises(ValueError, match="latest training status"):
        module.preflight(
            config_path=fixture["config_path"],
            status_path=fixture["status_path"],
            scripts_dir=fixture["scripts_dir"],
            output_root=fixture["output_root"],
            expected_config_sha256=fixture["config_sha256"],
            expected_jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
            expected_component_sha256=fixture["pins"],
        )

    assert not fixture["status_path"].exists()


@pytest.mark.parametrize(
    ("event", "status", "exit_code"),
    [("started", "running", None), ("finished", "failed", 1)],
)
def test_preflight_rejects_a_later_non_success_after_historical_success(
    tmp_path: Path,
    event: str,
    status: str,
    exit_code: int | None,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    with fixture["training_status"].open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "model_id": fixture["model_ids"][0],
                    "event": event,
                    "status": status,
                    "exit_code": exit_code,
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="latest training status"):
        module.preflight(
            config_path=fixture["config_path"],
            status_path=fixture["status_path"],
            scripts_dir=fixture["scripts_dir"],
            output_root=fixture["output_root"],
            expected_config_sha256=fixture["config_sha256"],
            expected_jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
            expected_component_sha256=fixture["pins"],
        )


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
def test_preflight_rejects_latest_success_with_provenance_drift(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    rows = [
        json.loads(line)
        for line in fixture["training_status"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0][field] = "drifted"
    fixture["training_status"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"provenance mismatch.*{field}"):
        module.preflight(
            config_path=fixture["config_path"],
            status_path=fixture["status_path"],
            scripts_dir=fixture["scripts_dir"],
            output_root=fixture["output_root"],
            expected_config_sha256=fixture["config_sha256"],
            expected_jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
            expected_component_sha256=fixture["pins"],
        )


def test_preflight_refuses_component_drift(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    analyzer = fixture["scripts_dir"] / "analyze_nko_beep_matrix.py"
    analyzer.write_text("# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="component SHA-256 mismatch"):
        module.preflight(
            config_path=fixture["config_path"],
            status_path=fixture["status_path"],
            scripts_dir=fixture["scripts_dir"],
            output_root=fixture["output_root"],
            expected_config_sha256=fixture["config_sha256"],
            expected_jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
            expected_component_sha256=fixture["pins"],
        )


def test_launch_runs_only_after_preflight_and_uses_versioned_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, module, fixture)

    result = module.launch(
        config_path=fixture["config_path"],
        status_path=fixture["status_path"],
        scripts_dir=fixture["scripts_dir"],
        output_root=fixture["output_root"],
        expected_config_sha256=fixture["config_sha256"],
        expected_jobs_path=fixture["jobs_path"],
        expected_jobs_sha256=fixture["jobs_sha256"],
        expected_component_sha256=fixture["pins"],
    )

    assert result.succeeded == ("manifests",)
    status = json.loads(fixture["status_path"].read_text(encoding="utf-8"))
    snapshot_root = fixture["output_root"] / "runtime-inputs-v1"
    assert status["mode"] == "snapshot"
    assert status["scripts_dir"] == str(snapshot_root / "scripts")
    assert status["config_path"] == str(snapshot_root / "evaluation-queue-runtime.json")
    assert status["training_status"] == str(snapshot_root / "training-status.jsonl")
    assert status["training_jobs"] == str(snapshot_root / "training-jobs-speed-v1.json")
    assert status["analyzer_source"] == "# analyze_nko_beep_matrix.py\n"
    assert fixture["module_exec_log"].read_text(encoding="utf-8").splitlines() == ["exec"]


@pytest.mark.parametrize("mutation_kind", ["component", "status"])
def test_launch_rejects_source_mutation_after_preflight_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    config_path = fixture["config_path"]
    document = json.loads(config_path.read_text(encoding="utf-8"))
    target = (
        fixture["scripts_dir"] / "analyze_nko_beep_matrix.py"
        if mutation_kind == "component"
        else fixture["training_status"]
    )
    document["test_mutation_target"] = str(target)
    document["test_mutation_kind"] = mutation_kind
    config_path.write_text(json.dumps(document), encoding="utf-8")
    fixture["config_sha256"] = _sha256(config_path)
    _patch_operational_defaults(monkeypatch, module, fixture)

    with pytest.raises(ValueError, match="verified source changed before snapshot"):
        module.launch(
            config_path=config_path,
            status_path=fixture["status_path"],
            scripts_dir=fixture["scripts_dir"],
            output_root=fixture["output_root"],
            expected_config_sha256=fixture["config_sha256"],
            expected_jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
            expected_component_sha256=fixture["pins"],
        )

    assert not fixture["status_path"].exists()


def test_launch_refuses_to_overwrite_an_existing_invalid_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, module, fixture)
    snapshot_root = fixture["output_root"] / "runtime-inputs-v1"
    snapshot_root.mkdir()
    tampered = snapshot_root / "owned.txt"
    tampered.write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime snapshot file set mismatch"):
        module.launch(
            config_path=fixture["config_path"],
            status_path=fixture["status_path"],
            scripts_dir=fixture["scripts_dir"],
            output_root=fixture["output_root"],
            expected_config_sha256=fixture["config_sha256"],
            expected_jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
            expected_component_sha256=fixture["pins"],
        )

    assert tampered.read_text(encoding="utf-8") == "do not overwrite\n"


def test_launch_reuses_valid_runtime_snapshot_without_overwriting_any_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, module, fixture)
    launch_kwargs: LaunchArguments = {
        "config_path": fixture["config_path"],
        "status_path": fixture["status_path"],
        "scripts_dir": fixture["scripts_dir"],
        "output_root": fixture["output_root"],
        "expected_config_sha256": fixture["config_sha256"],
        "expected_jobs_path": fixture["jobs_path"],
        "expected_jobs_sha256": fixture["jobs_sha256"],
        "expected_component_sha256": fixture["pins"],
    }

    first_result = module.launch(**launch_kwargs)
    snapshot_root = fixture["output_root"] / "runtime-inputs-v1"

    def snapshot_state() -> dict[Path, tuple[bool, bytes, int]]:
        paths = (snapshot_root, *snapshot_root.rglob("*"))
        return {
            path.relative_to(snapshot_root): (
                path.is_file(),
                path.read_bytes() if path.is_file() else b"",
                path.stat().st_mtime_ns,
            )
            for path in paths
        }

    before = snapshot_state()

    def refuse_snapshot_write(_snapshot: object) -> None:
        raise AssertionError

    monkeypatch.setattr(module, "_write_new_snapshot", refuse_snapshot_write)
    second_result = module.launch(**launch_kwargs)

    assert first_result.succeeded == ("manifests",)
    assert second_result.succeeded == ("manifests",)
    assert snapshot_state() == before


@pytest.mark.parametrize("mutation_kind", ["component", "status"])
def test_launch_uses_snapshot_when_original_source_changes_during_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    config_path = fixture["config_path"]
    document = json.loads(config_path.read_text(encoding="utf-8"))
    target = (
        fixture["scripts_dir"] / "analyze_nko_beep_matrix.py"
        if mutation_kind == "component"
        else fixture["training_status"]
    )
    original = target.read_bytes()
    document["test_mutation_target"] = str(target)
    document["test_mutation_kind"] = f"during-run-{mutation_kind}"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    fixture["config_sha256"] = _sha256(config_path)
    _patch_operational_defaults(monkeypatch, module, fixture)

    module.launch(
        config_path=config_path,
        status_path=fixture["status_path"],
        scripts_dir=fixture["scripts_dir"],
        output_root=fixture["output_root"],
        expected_config_sha256=fixture["config_sha256"],
        expected_jobs_path=fixture["jobs_path"],
        expected_jobs_sha256=fixture["jobs_sha256"],
        expected_component_sha256=fixture["pins"],
    )

    snapshot_root = fixture["output_root"] / "runtime-inputs-v1"
    snapshot_path = (
        snapshot_root / "scripts" / "analyze_nko_beep_matrix.py"
        if mutation_kind == "component"
        else snapshot_root / "training-status.jsonl"
    )
    assert target.read_bytes() != original
    assert snapshot_path.read_bytes() == original
    status = json.loads(fixture["status_path"].read_text(encoding="utf-8"))
    assert status["mode"] == "snapshot"


def test_cli_preflight_refuses_an_ancestor_output_root_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, module, fixture)

    with pytest.raises(ValueError, match="fixed speed-v4 output root"):
        module.main(["preflight", "--output-root", str(tmp_path)])


def test_cli_prepare_refuses_a_legacy_output_root_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, module, fixture)
    monkeypatch.setattr(module, "DEFAULT_SOURCE_CONFIG_PATH", fixture["config_path"])
    monkeypatch.setattr(module, "EXPECTED_SOURCE_CONFIG_SHA256", fixture["config_sha256"])
    legacy_root = tmp_path / "evaluation_speed_v3"
    legacy_config = legacy_root / "evaluation-queue-speed-v3-added-by-v4.json"
    legacy_status = legacy_root / "evaluation-status-speed-v3-added-by-v4.jsonl"

    with pytest.raises(ValueError, match="fixed speed-v4 output root"):
        module.main(
            [
                "prepare",
                "--output-root",
                str(legacy_root),
                "--config",
                str(legacy_config),
                "--status-path",
                str(legacy_status),
            ]
        )

    assert not legacy_root.exists()


def test_cli_launch_refuses_v1_status_even_with_ancestor_root_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, module, fixture)
    v1_status = tmp_path / "status" / "evaluation-status-official-v1.jsonl"
    v1_status.parent.mkdir()

    with pytest.raises(ValueError, match=r"fixed speed-v4 output root|fixed speed-v4 status"):
        module.main(
            [
                "launch",
                "--output-root",
                str(tmp_path),
                "--status-path",
                str(v1_status),
            ]
        )

    assert not v1_status.exists()


def test_preflight_refuses_nonreused_outputs_outside_v4_root(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    config_path = fixture["config_path"]
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["models"][1]["generation_dir"] = str(tmp_path / "v1-output/generation")
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="outside speed-v4 output root"):
        module.preflight(
            config_path=config_path,
            status_path=fixture["status_path"],
            scripts_dir=fixture["scripts_dir"],
            output_root=fixture["output_root"],
            expected_config_sha256=_sha256(config_path),
            expected_jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
            expected_component_sha256=fixture["pins"],
        )


def test_prepare_creates_isolated_speed_v4_config_without_changing_source(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    source_path = tmp_path / "evaluation-queue-official-v1.json"
    destination = tmp_path / "prepared-v4" / "evaluation-queue-speed-v4.json"
    source = json.loads(fixture["config_path"].read_text(encoding="utf-8"))
    source["training_jobs"] = str(tmp_path / "training-jobs-official-v1.json")
    source["manifest_output_dir"] = str(tmp_path / "official-v1-manifests")
    source["metric_models"]["speaker_embedding"]["savedir"] = str(tmp_path / "official-v1-cache")
    for model in source["models"]:
        if "reuse" not in model:
            old_root = tmp_path / "official-v1" / model["model_id"]
            model["generation_dir"] = str(old_root / "generation")
            model["analysis_dir"] = str(old_root / "analysis")
            model["metrics_dir"] = str(old_root / "metrics")
            model["evaluation_dir"] = str(old_root / "selection")
    source_path.write_text(json.dumps(source), encoding="utf-8")
    source_before = source_path.read_bytes()

    result = module.prepare_speed_v4_config(
        source_path=source_path,
        destination=destination,
        status_path=destination.parent / "evaluation-status-speed-v4.jsonl",
        output_root=destination.parent,
        expected_source_sha256=_sha256(source_path),
        jobs_path=fixture["jobs_path"],
        expected_jobs_sha256=fixture["jobs_sha256"],
    )

    assert source_path.read_bytes() == source_before
    assert result["config_path"] == str(destination.resolve())
    assert result["config_sha256"] == _sha256(destination)
    assert not destination.with_name(f".{destination.name}.tmp").exists()
    prepared = json.loads(destination.read_text(encoding="utf-8"))
    assert prepared["training_jobs"] == str(fixture["jobs_path"].resolve())
    assert prepared["manifest_output_dir"] == str(destination.parent / "manifests")
    assert prepared["models"][0]["reuse"] == source["models"][0]["reuse"]
    for model in prepared["models"][1:]:
        model_root = destination.parent / "models" / model["model_id"]
        assert model["generation_dir"] == str(model_root / "generation")
        assert model["analysis_dir"] == str(model_root / "analysis")
        assert model["metrics_dir"] == str(model_root / "metrics")
        assert model["evaluation_dir"] == str(model_root / "selection")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.prepare_speed_v4_config(
            source_path=source_path,
            destination=destination,
            status_path=destination.parent / "evaluation-status-speed-v4.jsonl",
            output_root=destination.parent,
            expected_source_sha256=_sha256(source_path),
            jobs_path=fixture["jobs_path"],
            expected_jobs_sha256=fixture["jobs_sha256"],
        )
