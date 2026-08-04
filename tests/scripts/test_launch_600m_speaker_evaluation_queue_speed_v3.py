from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v3.py")
MODEL_COUNT = 12
SNAPSHOT_FILE_COUNT = 9
QUALITY_RELATIVE = (
    Path("training")
    / "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
    / "quality_retrain_init2500_lr0003_seed1_v1"
)


class QueueResult(Protocol):
    succeeded: tuple[str, ...]


class LauncherModule(Protocol):
    REMOTE_ROOT: Path
    DEFAULT_OUTPUT_ROOT: Path
    DEFAULT_CONFIG_PATH: Path
    DEFAULT_STATUS_PATH: Path
    DEFAULT_JOBS_PATH: Path
    DEFAULT_TRAINING_STATUS_PATH: Path
    RUNTIME_SNAPSHOT_NAME: str
    RUNTIME_JOBS_NAME: str
    RUNTIME_STATUS_NAME: str
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

    def prepare_speed_v5_config(
        self,
        *,
        source_path: Path,
        destination: Path,
        status_path: Path,
        output_root: Path,
        expected_source_sha256: str,
        jobs_path: Path,
        training_status_path: Path,
        expected_jobs_sha256: str,
    ) -> dict[str, object]: ...

    def main(self, argv: Sequence[str] | None = None) -> int: ...


class Fixture(TypedDict):
    module: LauncherModule
    output_root: Path
    config_path: Path
    status_path: Path
    jobs_path: Path
    training_status: Path
    source_path: Path
    model_ids: list[str]
    scripts_dir: Path
    config_sha256: str
    jobs_sha256: str
    source_sha256: str
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
        "launch_600m_speaker_evaluation_queue_speed_v3",
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


def test_v5_defaults_pin_quality_successor_and_fresh_output_root() -> None:
    module = _load_script()

    assert module.DEFAULT_OUTPUT_ROOT == module.REMOTE_ROOT / "evaluation_speed_v5"
    assert module.DEFAULT_CONFIG_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-queue-speed-v5.json"
    )
    assert module.DEFAULT_STATUS_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v5.jsonl"
    )
    assert module.DEFAULT_JOBS_PATH == (
        module.REMOTE_ROOT / QUALITY_RELATIVE / "training-jobs-init2500-lr0003-seed1-v1.json"
    )
    assert module.DEFAULT_TRAINING_STATUS_PATH == (
        module.REMOTE_ROOT / QUALITY_RELATIVE / "training-status-init2500-lr0003-seed1-v1.jsonl"
    )
    assert module.RUNTIME_SNAPSHOT_NAME == "runtime-inputs-v1"
    assert module.RUNTIME_JOBS_NAME == "training-jobs-init2500-lr0003-seed1-v1.json"
    assert module.RUNTIME_STATUS_NAME == "training-status-init2500-lr0003-seed1-v1.jsonl"


def test_v5_preserves_legacy_six_evaluation_component_pins() -> None:
    module = _load_script()

    assert module.EXPECTED_CONFIG_SHA256 == (
        "a79670dabcfd986f9b3e35b9096c8f26f672a0e14e4d2b226c3ad616caa12a84"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", module.EXPECTED_CONFIG_SHA256)
    assert module.EXPECTED_CONFIG_SHA256 != module.PENDING_CONFIG_SHA256
    assert module.EXPECTED_SOURCE_CONFIG_SHA256 == (
        "33109e7ea9b62b014d59ce0673ea3d4e50c45f9e60cebf13e06463e2e5e4fd02"
    )
    assert module.EXPECTED_JOBS_SHA256 == (
        "f6a943d87fe4d405a8830367796791363f69b034eabd4787b56a500d1c941a59"
    )
    expected_components = {
        "run_600m_speaker_evaluation_queue.py": (
            "5b8be7e8cab5684a614bd7fdbb699abb371a362468eb3887f4a3993af2174aa7"
        ),
        "build_600m_checkpoint_evaluation_manifests.py": (
            "bde9b1455e681fcdb2cd88552a2b6f61fce188c7ea22a4d5a0145b8ec9a77dc1"
        ),
        "generate_600m_checkpoint_audio_remote.py": (
            "2bac3d178bbad17f3d97185b14362b870bfc546ce866e705fca4f725fca07110"
        ),
        "analyze_nko_beep_matrix.py": (
            "06c23b489975843bf080b7fa70ebb41abac19c269b6cfe5bcafec019a0693dc1"
        ),
        "compute_600m_speaker_metrics.py": (
            "fa83491f0ee2f1e1f21c8d833ba90557ed67c885a2512c9c05104faf3b14a407"
        ),
        "evaluate_600m_speaker_checkpoints.py": (
            "fde5bf9f8fc92ec8e50b1c22532d6f9eb1b83a6cccb65985b502bbd1b2b7b8a5"
        ),
    }
    assert expected_components == module.EXPECTED_COMPONENT_SHA256


def test_just_catalog_retains_v4_and_adds_v5_recipes() -> None:
    catalog = Path("justfile").read_text(encoding="utf-8")

    assert "speaker-evaluation-speed-v4 *args:" in catalog
    assert "remote-speaker-evaluation-speed-v4 *args:" in catalog
    assert "speaker-evaluation-speed-v5 *args:" in catalog
    assert "remote-speaker-evaluation-speed-v5 *args:" in catalog
    assert "launch_600m_speaker_evaluation_queue_speed_v3.py" in catalog


def _recipe_body(catalog: str, name: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(name)} \*args:\n(?P<body>(?:    .*\n)+)",
        catalog,
    )
    assert match is not None
    return match.group("body")


def test_just_recipe_blocks_bind_v4_to_v2_and_v5_to_v3_launchers() -> None:
    catalog = Path("justfile").read_text(encoding="utf-8")

    assert _recipe_body(catalog, "speaker-evaluation-speed-v4") == (
        '    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v2.py "$@"\n'
    )
    assert _recipe_body(catalog, "speaker-evaluation-speed-v5") == (
        '    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v3.py "$@"\n'
    )
    assert (
        "remote_evaluation_speed_v4_launcher := remote_work_root + "
        "'\\scripts\\launch_600m_speaker_evaluation_queue_speed_v2.py'"
    ) in catalog
    assert (
        "remote_evaluation_speed_v5_launcher := remote_work_root + "
        "'\\scripts\\launch_600m_speaker_evaluation_queue_speed_v3.py'"
    ) in catalog
    assert _recipe_body(catalog, "remote-speaker-evaluation-speed-v4") == (
        "    just remote-python '{{ remote_evaluation_speed_v4_launcher }}' \"$@\"\n"
    )
    assert _recipe_body(catalog, "remote-speaker-evaluation-speed-v5") == (
        "    just remote-python '{{ remote_evaluation_speed_v5_launcher }}' \"$@\"\n"
    )


def _write_fixture(tmp_path: Path) -> Fixture:  # noqa: PLR0914
    output_root = tmp_path / "evaluation-speed-v5"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    module_exec_log = tmp_path / "module-executions.log"
    queue_source = """
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
                'model_id': 'model-00',
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

def _run_evaluation_queue_locked(config, *, status_path, scripts_dir, runner=None, now=None):
    if config.mutation_kind in {'during-run-component', 'during-run-status'}:
        _apply_mutation(config, config.mutation_kind.removeprefix('during-run-'))
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({
        'mode': 'snapshot',
        'scripts_dir': str(scripts_dir),
        'config_path': str(config.source_path),
        'training_status': str(config.training_status),
        'training_jobs': str(config.training_jobs),
        'analyzer_source': (scripts_dir / 'analyze_nko_beep_matrix.py').read_text(
            encoding='utf-8'
        ),
    }), encoding='utf-8')
    return Result(succeeded=('manifests',))
""".lstrip().replace("__MODULE_EXEC_LOG__", repr(str(module_exec_log)))
    (scripts_dir / "run_600m_speaker_evaluation_queue.py").write_text(
        queue_source,
        encoding="utf-8",
    )
    for name in (
        "build_600m_checkpoint_evaluation_manifests.py",
        "generate_600m_checkpoint_audio_remote.py",
        "analyze_nko_beep_matrix.py",
        "compute_600m_speaker_metrics.py",
        "evaluate_600m_speaker_checkpoints.py",
    ):
        (scripts_dir / name).write_text(f"# {name}\n", encoding="utf-8")

    model_ids = [f"model-{index:02d}" for index in range(MODEL_COUNT)]
    checkpoint = tmp_path / "base-checkpoint.safetensors"
    checkpoint.write_bytes(b"base checkpoint")
    jobs: list[dict[str, object]] = []
    for model_id in model_ids:
        model_input = tmp_path / "inputs" / model_id
        model_input.mkdir(parents=True)
        manifest = model_input / "clean-manifest.jsonl"
        config = model_input / "training-config.json"
        manifest.write_text(f'{{"model_id":"{model_id}"}}\n', encoding="utf-8")
        config.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
        jobs.append(
            {
                "model_id": model_id,
                "clean_manifest": str(manifest),
                "config": str(config),
                "output_dir": str(tmp_path / "training" / model_id),
                "command": ["python", "train.py"],
            }
        )
    jobs_document: dict[str, object] = {
        "schema_version": 1,
        "base_checkpoint_path": str(checkpoint),
        "base_checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_revision": "base-revision",
        "upstream_commit": "upstream-commit",
        "jobs": jobs,
    }
    jobs_path = tmp_path / "training-jobs-init2500-lr0003-seed1-v1.json"
    jobs_path.write_text(json.dumps(jobs_document), encoding="utf-8")
    training_status = tmp_path / "training-status-init2500-lr0003-seed1-v1.jsonl"
    training_status.write_text(
        "".join(
            json.dumps(
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
            )
            + "\n"
            for job in jobs
        ),
        encoding="utf-8",
    )
    models: list[dict[str, object]] = []
    for index, model_id in enumerate(model_ids):
        model: dict[str, object] = {
            "model_id": model_id,
            "reference_wavs": str(tmp_path / f"{model_id}-references.json"),
        }
        if index == 0:
            model["reuse"] = {
                "generation_dir": str(tmp_path / "old-generation"),
                "analysis_dir": str(tmp_path / "old-analysis"),
                "metrics_results": str(tmp_path / "old-metrics.jsonl"),
                "metrics_provenance": str(tmp_path / "old-provenance.json"),
                "evaluation_manifest": str(tmp_path / "old-manifest.json"),
                "evaluation_dir": str(tmp_path / "old-selection"),
            }
        else:
            old_root = tmp_path / "old" / model_id
            model.update(
                {
                    "generation_dir": str(old_root / "generation"),
                    "analysis_dir": str(old_root / "analysis"),
                    "metrics_dir": str(old_root / "metrics"),
                    "evaluation_dir": str(old_root / "selection"),
                }
            )
        models.append(model)
    source_path = tmp_path / "evaluation-queue-official-v1.json"
    source_path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-evaluation-queue/v1",
                "training_status": str(tmp_path / "old-training-status.jsonl"),
                "training_jobs": str(tmp_path / "old-training-jobs.json"),
                "manifest_output_dir": str(tmp_path / "old-manifests"),
                "metric_models": {"speaker_embedding": {"savedir": str(tmp_path / "old-ecapa")}},
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    config_path = output_root / "evaluation-queue-speed-v5.json"
    status_path = output_root / "evaluation-status-speed-v5.jsonl"
    module = _load_script()
    module.prepare_speed_v5_config(
        source_path=source_path,
        destination=config_path,
        status_path=status_path,
        output_root=output_root,
        expected_source_sha256=_sha256(source_path),
        jobs_path=jobs_path,
        training_status_path=training_status,
        expected_jobs_sha256=_sha256(jobs_path),
    )
    pins = {path.name: _sha256(path) for path in scripts_dir.iterdir() if path.is_file()}
    return {
        "module": module,
        "output_root": output_root,
        "config_path": config_path,
        "status_path": status_path,
        "jobs_path": jobs_path,
        "training_status": training_status,
        "source_path": source_path,
        "model_ids": model_ids,
        "scripts_dir": scripts_dir,
        "config_sha256": _sha256(config_path),
        "jobs_sha256": _sha256(jobs_path),
        "source_sha256": _sha256(source_path),
        "pins": pins,
    }


def _launch_kwargs(fixture: Fixture) -> LaunchArguments:
    return {
        "config_path": fixture["config_path"],
        "status_path": fixture["status_path"],
        "scripts_dir": fixture["scripts_dir"],
        "output_root": fixture["output_root"],
        "expected_config_sha256": fixture["config_sha256"],
        "expected_jobs_path": fixture["jobs_path"],
        "expected_jobs_sha256": fixture["jobs_sha256"],
        "expected_component_sha256": fixture["pins"],
    }


def _patch_operational_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fixture: Fixture,
) -> None:
    module = fixture["module"]
    monkeypatch.setattr(module, "DEFAULT_OUTPUT_ROOT", fixture["output_root"])
    monkeypatch.setattr(module, "DEFAULT_CONFIG_PATH", fixture["config_path"])
    monkeypatch.setattr(module, "DEFAULT_STATUS_PATH", fixture["status_path"])
    monkeypatch.setattr(module, "DEFAULT_JOBS_PATH", fixture["jobs_path"])
    monkeypatch.setattr(
        module,
        "DEFAULT_TRAINING_STATUS_PATH",
        fixture["training_status"],
    )
    monkeypatch.setattr(module, "DEFAULT_SCRIPTS_DIR", fixture["scripts_dir"])


def test_prepare_creates_all_twelve_fresh_model_stages_and_is_create_only(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    prepared = json.loads(fixture["config_path"].read_text(encoding="utf-8"))

    assert prepared["training_jobs"] == str(fixture["jobs_path"].resolve())
    assert prepared["training_status"] == str(fixture["training_status"].resolve())
    assert len(prepared["models"]) == MODEL_COUNT
    assert all("reuse" not in model for model in prepared["models"])
    for model in prepared["models"]:
        model_root = fixture["output_root"] / "models" / model["model_id"]
        assert model["generation_dir"] == str(model_root / "generation")
        assert model["analysis_dir"] == str(model_root / "analysis")
        assert model["metrics_dir"] == str(model_root / "metrics")
        assert model["evaluation_dir"] == str(model_root / "selection")

    with pytest.raises(FileExistsError, match="refusing to overwrite speed-v5 config"):
        fixture["module"].prepare_speed_v5_config(
            source_path=fixture["source_path"],
            destination=fixture["config_path"],
            status_path=fixture["status_path"],
            output_root=fixture["output_root"],
            expected_source_sha256=fixture["source_sha256"],
            jobs_path=fixture["jobs_path"],
            training_status_path=fixture["training_status"],
            expected_jobs_sha256=fixture["jobs_sha256"],
        )


def test_prepare_rejects_source_mutation_after_capture_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    source_path = fixture["source_path"]
    jobs_path = fixture["jobs_path"]
    source_before = source_path.read_bytes()
    destination = tmp_path / "prepared-after-race" / "evaluation-queue-speed-v5.json"
    original_read_bytes = Path.read_bytes
    mutation_applied = False

    def mutate_source_while_inputs_are_captured(path: Path) -> bytes:
        nonlocal mutation_applied
        content = original_read_bytes(path)
        if path.resolve() == jobs_path.resolve() and not mutation_applied:
            source_path.write_bytes(source_before + b"\n")
            mutation_applied = True
        return content

    monkeypatch.setattr(Path, "read_bytes", mutate_source_while_inputs_are_captured)

    with pytest.raises(ValueError, match="verified source changed before publish"):
        module.prepare_speed_v5_config(
            source_path=source_path,
            destination=destination,
            status_path=destination.parent / "evaluation-status-speed-v5.jsonl",
            output_root=destination.parent,
            expected_source_sha256=fixture["source_sha256"],
            jobs_path=jobs_path,
            training_status_path=fixture["training_status"],
            expected_jobs_sha256=fixture["jobs_sha256"],
        )

    assert mutation_applied is True
    assert source_path.read_bytes() != source_before
    assert not destination.exists()


def test_prepare_rejects_v5_output_root_alias_before_legacy_write(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    legacy_root = tmp_path / "evaluation-speed-v4-legacy"
    legacy_root.mkdir()
    aliased_v5_root = tmp_path / "evaluation-speed-v5-alias"
    aliased_v5_root.symlink_to(legacy_root, target_is_directory=True)
    destination = aliased_v5_root / "evaluation-queue-speed-v5.json"

    with pytest.raises(ValueError, match=r"output root.*symlink|junction|reparse alias"):
        fixture["module"].prepare_speed_v5_config(
            source_path=fixture["source_path"],
            destination=destination,
            status_path=aliased_v5_root / "evaluation-status-speed-v5.jsonl",
            output_root=aliased_v5_root,
            expected_source_sha256=fixture["source_sha256"],
            jobs_path=fixture["jobs_path"],
            training_status_path=fixture["training_status"],
            expected_jobs_sha256=fixture["jobs_sha256"],
        )

    assert list(legacy_root.iterdir()) == []
    assert not destination.exists()


def test_preflight_verifies_fresh_twelve_model_contract(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    report = fixture["module"].preflight(**_launch_kwargs(fixture))

    assert report["passed"] is True
    assert report["model_count"] == MODEL_COUNT
    assert report["model_ids"] == fixture["model_ids"]
    assert report["reused_model_ids"] == []
    assert report["training_status"] == str(fixture["training_status"].resolve())
    assert not fixture["status_path"].exists()


def test_preflight_rejects_any_reused_model(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    document = json.loads(fixture["config_path"].read_text(encoding="utf-8"))
    document["models"][0]["reuse"] = {"generation_dir": "old"}
    for field in ("generation_dir", "analysis_dir", "metrics_dir", "evaluation_dir"):
        del document["models"][0][field]
    fixture["config_path"].write_text(json.dumps(document), encoding="utf-8")
    fixture["config_sha256"] = _sha256(fixture["config_path"])

    with pytest.raises(ValueError, match="speed-v5 does not permit reused models"):
        fixture["module"].preflight(**_launch_kwargs(fixture))


def test_preflight_rejects_unpinned_training_status(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    document = json.loads(fixture["config_path"].read_text(encoding="utf-8"))
    document["training_status"] = str(tmp_path / "different-status.jsonl")
    fixture["config_path"].write_text(json.dumps(document), encoding="utf-8")
    fixture["config_sha256"] = _sha256(fixture["config_path"])

    with pytest.raises(ValueError, match="quality successor training status"):
        fixture["module"].preflight(**_launch_kwargs(fixture))


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
def test_preflight_rejects_training_provenance_drift(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _write_fixture(tmp_path)
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
        fixture["module"].preflight(**_launch_kwargs(fixture))


def test_preflight_rejects_component_drift(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    analyzer = fixture["scripts_dir"] / "analyze_nko_beep_matrix.py"
    analyzer.write_text("# drifted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="component SHA-256 mismatch"):
        fixture["module"].preflight(**_launch_kwargs(fixture))


def test_launch_uses_complete_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    _patch_operational_defaults(monkeypatch, fixture)

    result = fixture["module"].launch(**_launch_kwargs(fixture))

    assert result.succeeded == ("manifests",)
    snapshot = fixture["output_root"] / fixture["module"].RUNTIME_SNAPSHOT_NAME
    status = json.loads(fixture["status_path"].read_text(encoding="utf-8"))
    assert status["scripts_dir"] == str(snapshot / "scripts")
    assert status["config_path"] == str(snapshot / "evaluation-queue-runtime.json")
    assert status["training_jobs"] == str(snapshot / fixture["module"].RUNTIME_JOBS_NAME)
    assert status["training_status"] == str(snapshot / fixture["module"].RUNTIME_STATUS_NAME)
    manifest = json.loads((snapshot / "snapshot-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == SNAPSHOT_FILE_COUNT
    assert set((snapshot / "scripts").iterdir()) == {
        snapshot / "scripts" / name for name in fixture["pins"]
    }


def test_launch_refuses_to_overwrite_invalid_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    _patch_operational_defaults(monkeypatch, fixture)
    snapshot = fixture["output_root"] / fixture["module"].RUNTIME_SNAPSHOT_NAME
    snapshot.mkdir()
    protected = snapshot / "protected.txt"
    protected.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime snapshot file set mismatch"):
        fixture["module"].launch(**_launch_kwargs(fixture))

    assert protected.read_text(encoding="utf-8") == "preserve\n"


def test_launch_rejects_runtime_snapshot_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    _patch_operational_defaults(monkeypatch, fixture)
    outside = tmp_path / "outside-snapshot"
    nominal_snapshot = fixture["output_root"] / fixture["module"].RUNTIME_SNAPSHOT_NAME
    nominal_snapshot.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=r"runtime snapshot.*alias|symlink|reparse"):
        fixture["module"].launch(**_launch_kwargs(fixture))

    assert not outside.exists()
    assert not fixture["status_path"].exists()


def test_launch_never_replaces_destination_created_during_snapshot_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, fixture)
    snapshot = fixture["output_root"] / module.RUNTIME_SNAPSHOT_NAME
    temporary = snapshot.with_name(f".{snapshot.name}.tmp")
    original_mkdir = Path.mkdir
    collision_inode: int | None = None

    def inject_destination_collision(
        path: Path,
        mode: int = 0o777,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        nonlocal collision_inode
        if collision_inode is None and path in {snapshot, temporary}:
            original_mkdir(snapshot)
            collision_inode = snapshot.stat().st_ino
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", inject_destination_collision)

    with pytest.raises(ValueError, match="runtime snapshot file set mismatch"):
        module.launch(**_launch_kwargs(fixture))

    assert collision_inode is not None
    assert snapshot.is_dir()
    assert snapshot.stat().st_ino == collision_inode
    assert list(snapshot.iterdir()) == []
    assert not fixture["status_path"].exists()


@pytest.mark.parametrize("mutation_kind", ["component", "status"])
def test_launch_rejects_verified_source_mutation_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    document = json.loads(fixture["config_path"].read_text(encoding="utf-8"))
    target = (
        fixture["scripts_dir"] / "analyze_nko_beep_matrix.py"
        if mutation_kind == "component"
        else fixture["training_status"]
    )
    document["test_mutation_target"] = str(target)
    document["test_mutation_kind"] = mutation_kind
    fixture["config_path"].write_text(json.dumps(document), encoding="utf-8")
    fixture["config_sha256"] = _sha256(fixture["config_path"])
    _patch_operational_defaults(monkeypatch, fixture)

    with pytest.raises(ValueError, match="verified source changed before snapshot"):
        module.launch(**_launch_kwargs(fixture))

    assert not fixture["status_path"].exists()


def test_launch_reuses_valid_snapshot_without_overwriting_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, fixture)
    launch_kwargs = _launch_kwargs(fixture)

    module.launch(**launch_kwargs)
    snapshot = fixture["output_root"] / module.RUNTIME_SNAPSHOT_NAME

    def snapshot_state() -> dict[Path, tuple[bool, bytes, int]]:
        paths = (snapshot, *snapshot.rglob("*"))
        return {
            path.relative_to(snapshot): (
                path.is_file(),
                path.read_bytes() if path.is_file() else b"",
                path.stat().st_mtime_ns,
            )
            for path in paths
        }

    before = snapshot_state()

    def reject_snapshot_write(_snapshot: object) -> None:
        raise AssertionError

    monkeypatch.setattr(module, "_write_new_snapshot", reject_snapshot_write)
    result = module.launch(**launch_kwargs)

    assert result.succeeded == ("manifests",)
    assert snapshot_state() == before


@pytest.mark.parametrize("mutation_kind", ["component", "status"])
def test_launch_isolates_snapshot_when_source_changes_during_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    target = (
        fixture["scripts_dir"] / "analyze_nko_beep_matrix.py"
        if mutation_kind == "component"
        else fixture["training_status"]
    )
    original = target.read_bytes()
    document = json.loads(fixture["config_path"].read_text(encoding="utf-8"))
    document["test_mutation_target"] = str(target)
    document["test_mutation_kind"] = f"during-run-{mutation_kind}"
    fixture["config_path"].write_text(json.dumps(document), encoding="utf-8")
    fixture["config_sha256"] = _sha256(fixture["config_path"])
    _patch_operational_defaults(monkeypatch, fixture)

    module.launch(**_launch_kwargs(fixture))

    snapshot = fixture["output_root"] / module.RUNTIME_SNAPSHOT_NAME
    snapshot_path = (
        snapshot / "scripts" / "analyze_nko_beep_matrix.py"
        if mutation_kind == "component"
        else snapshot / module.RUNTIME_STATUS_NAME
    )
    assert target.read_bytes() != original
    assert snapshot_path.read_bytes() == original
    status = json.loads(fixture["status_path"].read_text(encoding="utf-8"))
    assert status["mode"] == "snapshot"
    assert status["analyzer_source"] == "# analyze_nko_beep_matrix.py\n"


def test_cli_refuses_operational_output_root_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    _patch_operational_defaults(monkeypatch, fixture)

    with pytest.raises(ValueError, match="fixed speed-v5 output root"):
        fixture["module"].main(["preflight", "--output-root", str(tmp_path)])


def test_cli_prepare_refuses_legacy_root_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    module = fixture["module"]
    _patch_operational_defaults(monkeypatch, fixture)
    monkeypatch.setattr(module, "DEFAULT_SOURCE_CONFIG_PATH", fixture["source_path"])
    monkeypatch.setattr(module, "EXPECTED_SOURCE_CONFIG_SHA256", fixture["source_sha256"])
    legacy_root = tmp_path / "evaluation_speed_v4"

    with pytest.raises(ValueError, match="fixed speed-v5 output root"):
        module.main(
            [
                "prepare",
                "--output-root",
                str(legacy_root),
                "--config",
                str(legacy_root / "evaluation-queue-speed-v5.json"),
                "--status-path",
                str(legacy_root / "evaluation-status-speed-v5.jsonl"),
            ]
        )

    assert not legacy_root.exists()


def test_cli_launch_refuses_legacy_status_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    _patch_operational_defaults(monkeypatch, fixture)
    legacy_status = tmp_path / "evaluation_speed_v4" / "evaluation-status-speed-v4.jsonl"

    with pytest.raises(
        ValueError,
        match=r"evaluation status is outside nominal speed-v5 output root|fixed speed-v5 status",
    ):
        fixture["module"].main(["launch", "--status-path", str(legacy_status)])

    assert not legacy_status.exists()


def test_preflight_rejects_model_outputs_outside_v5_root(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    document = json.loads(fixture["config_path"].read_text(encoding="utf-8"))
    document["models"][0]["generation_dir"] = str(tmp_path / "outside-generation")
    fixture["config_path"].write_text(json.dumps(document), encoding="utf-8")
    fixture["config_sha256"] = _sha256(fixture["config_path"])

    with pytest.raises(ValueError, match="outside speed-v5 output root"):
        fixture["module"].preflight(**_launch_kwargs(fixture))


@pytest.mark.parametrize("mode", ["preflight", "launch"])
def test_pending_config_pin_blocks_nonprepare_modes(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    _patch_operational_defaults(monkeypatch, fixture)
    monkeypatch.setattr(
        fixture["module"],
        "EXPECTED_CONFIG_SHA256",
        fixture["module"].PENDING_CONFIG_SHA256,
    )

    with pytest.raises(ValueError, match="PENDING_REMOTE_PREPARE_CONFIG_SHA256"):
        fixture["module"].main([mode])
