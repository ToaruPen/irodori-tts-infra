from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess  # noqa: S404 - tests create an isolated local Git repository.
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    class RuntimeSnapshotLike(Protocol):
        files: Mapping[Path, bytes]


pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v4.py")
CORE_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v3.py")
MODEL_COUNT = 12
QUALITY_RELATIVE = (
    Path("training")
    / "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
    / "quality_retrain_init2500_lr0003_seed1_v1"
)


class LauncherModule(Protocol):
    REMOTE_ROOT: Path
    QUALITY_SUCCESSOR_ROOT: Path
    DEFAULT_OUTPUT_ROOT: Path
    DEFAULT_CONFIG_PATH: Path
    DEFAULT_STATUS_PATH: Path
    DEFAULT_JOBS_PATH: Path
    DEFAULT_TRAINING_STATUS_PATH: Path
    DEFAULT_SCRIPTS_DIR: Path
    DEFAULT_SOURCE_CONFIG_PATH: Path
    PENDING_CONFIG_SHA256: str
    PENDING_COMPONENT_SHA256: str
    EXPECTED_CONFIG_SHA256: str
    EXPECTED_SOURCE_CONFIG_SHA256: str
    EXPECTED_JOBS_SHA256: str
    EXPECTED_COMPONENT_SHA256: Mapping[str, str]
    EXPECTED_SPEED_V5_CORE_SHA256: str
    RUNTIME_JOBS_NAME: str
    RUNTIME_STATUS_NAME: str
    PINNED_UPSTREAM_COMMIT: str
    UPSTREAM_RUNTIME_PROVENANCE_NAME: str

    def _load_speed_v5_core(self, path: Path, *, expected_sha256: str) -> ModuleType: ...

    def prepare_speed_v6_config(
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

    def _build_upstream_runtime_provenance(
        self,
        root: Path,
        *,
        expected_commit: str,
    ) -> bytes: ...

    def _with_upstream_runtime_provenance(
        self, snapshot: object, provenance: bytes
    ) -> RuntimeSnapshotLike: ...

    def preflight(self, **kwargs: object) -> dict[str, object]: ...


class PrepareArguments(TypedDict):
    source_path: Path
    destination: Path
    status_path: Path
    output_root: Path
    expected_source_sha256: str
    jobs_path: Path
    training_status_path: Path
    expected_jobs_sha256: str


def _load_script() -> LauncherModule:
    spec = importlib.util.spec_from_file_location(
        "launch_600m_speaker_evaluation_queue_speed_v4",
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


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed Git executable and test arguments.
        ("git", "-C", str(root), *args),  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_upstream_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "Irodori-TTS"
    package = root / "irodori_tts"
    package.mkdir(parents=True)
    (package / "z.py").write_text("Z = 1\n", encoding="utf-8")
    (package / "a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "README.md").write_text("upstream\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _write_prepare_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    jobs_path = tmp_path / "training-jobs-init2500-lr0003-seed1-v1.json"
    training_status = tmp_path / "training-status-init2500-lr0003-seed1-v1.jsonl"
    model_ids = [f"model-{index:02d}" for index in range(MODEL_COUNT)]
    jobs_path.write_text(
        json.dumps({"jobs": [{"model_id": model_id} for model_id in model_ids]}),
        encoding="utf-8",
    )
    training_status.write_text("", encoding="utf-8")
    source_path = tmp_path / "evaluation-queue-official-v1.json"
    source_path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-evaluation-queue/v1",
                "training_jobs": "old-jobs.json",
                "training_status": "old-status.jsonl",
                "manifest_output_dir": str(tmp_path / "old-manifests"),
                "metric_models": {"speaker_embedding": {"savedir": str(tmp_path / "old-cache")}},
                "models": [
                    {
                        "model_id": model_id,
                        "reuse": {"generation_dir": "old"},
                    }
                    for model_id in model_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return source_path, jobs_path, training_status


def test_v6_defaults_pin_quality_successor_and_fresh_output_root() -> None:
    module = _load_script()

    assert module.DEFAULT_OUTPUT_ROOT == module.REMOTE_ROOT / "evaluation_speed_v6"
    assert module.DEFAULT_CONFIG_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-queue-speed-v6.json"
    )
    assert module.DEFAULT_STATUS_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v6.jsonl"
    )
    assert module.QUALITY_SUCCESSOR_ROOT == module.REMOTE_ROOT / QUALITY_RELATIVE
    assert module.DEFAULT_JOBS_PATH == (
        module.QUALITY_SUCCESSOR_ROOT / "training-jobs-init2500-lr0003-seed1-v1.json"
    )
    assert module.DEFAULT_TRAINING_STATUS_PATH == (
        module.QUALITY_SUCCESSOR_ROOT / "training-status-init2500-lr0003-seed1-v1.jsonl"
    )
    assert module.DEFAULT_SOURCE_CONFIG_PATH == (
        module.REMOTE_ROOT / "status" / "evaluation-queue-official-v1.json"
    )
    assert module.PINNED_UPSTREAM_COMMIT == "eaf74d6a19138f743acb5b71a445fd25a57db987"


def test_upstream_runtime_provenance_is_deterministic_and_allows_untracked_outside_package(
    tmp_path: Path,
) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    (root / "scratch.txt").write_text("allowed\n", encoding="utf-8")

    first = module._build_upstream_runtime_provenance(  # noqa: SLF001
        root, expected_commit=head
    )
    second = module._build_upstream_runtime_provenance(  # noqa: SLF001
        root, expected_commit=head
    )
    document = json.loads(first)

    assert first == second
    assert document["schema_version"] == "irodori-upstream-runtime-provenance/v1"
    assert document["upstream_root"] == str(root.resolve())
    assert document["commit"] == head
    assert [row["path"] for row in document["python_files"]] == [
        "irodori_tts/a.py",
        "irodori_tts/z.py",
    ]
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in document["python_files"])


@pytest.mark.parametrize("mutation", ["tracked", "untracked"])
def test_upstream_runtime_provenance_rejects_dirty_package(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    if mutation == "tracked":
        (root / "irodori_tts" / "a.py").write_text("A = 2\n", encoding="utf-8")
    else:
        (root / "irodori_tts" / "new.py").write_text("NEW = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"upstream .*package.*(dirty|untracked)"):
        module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001


def test_runtime_snapshot_adds_provenance_to_exact_manifest_inventory(tmp_path: Path) -> None:
    module = _load_script()
    runtime_root = tmp_path / "runtime-inputs-v1"
    config = b"{}\n"
    snapshot = module._core.RuntimeSnapshot(  # type: ignore[attr-defined]  # noqa: SLF001
        root=runtime_root,
        scripts_dir=runtime_root / "scripts",
        config_path=runtime_root / "evaluation-queue-runtime.json",
        jobs_path=runtime_root / "training-jobs-speed-v1.json",
        training_status=runtime_root / "training-status.jsonl",
        files={
            Path("evaluation-queue-runtime.json"): config,
            Path("snapshot-manifest.json"): json.dumps(
                {
                    "schema_version": "speaker-evaluation-runtime-inputs/v1",
                    "source_inputs": {},
                    "files": {
                        "evaluation-queue-runtime.json": {
                            "sha256": hashlib.sha256(config).hexdigest(),
                            "size": len(config),
                        }
                    },
                }
            ).encode(),
        },
        config_sha256=hashlib.sha256(config).hexdigest(),
    )
    provenance = b'{"schema_version":"irodori-upstream-runtime-provenance/v1"}\n'

    augmented = module._with_upstream_runtime_provenance(snapshot, provenance)  # noqa: SLF001
    manifest = json.loads(augmented.files[Path("snapshot-manifest.json")])

    assert augmented.files[Path(module.UPSTREAM_RUNTIME_PROVENANCE_NAME)] == provenance
    assert set(manifest["files"]) == {
        "evaluation-queue-runtime.json",
        module.UPSTREAM_RUNTIME_PROVENANCE_NAME,
    }


def test_preflight_verifies_upstream_without_materializing_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    module.PINNED_UPSTREAM_COMMIT = head
    context = SimpleNamespace(
        report={"passed": True},
        queue_config=SimpleNamespace(upstream_root=root),
    )
    monkeypatch.setattr(module._core, "_verify_context", lambda **_kwargs: context)  # type: ignore[attr-defined]  # noqa: SLF001

    report = module.preflight(config_path=tmp_path / "config.json")

    assert report["passed"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", cast("str", report["upstream_runtime_provenance_sha256"]))
    assert not (tmp_path / module.UPSTREAM_RUNTIME_PROVENANCE_NAME).exists()


def test_v6_runtime_training_names_match_completion_verifier_contract() -> None:
    module = _load_script()
    verifier_source = Path("scripts/verify_600m_speaker_retraining_completion.py").read_text(
        encoding="utf-8"
    )

    assert module.RUNTIME_JOBS_NAME == "training-jobs-speed-v1.json"
    assert module.RUNTIME_STATUS_NAME == "training-status.jsonl"
    assert f'RUNTIME_JOBS_NAME = "{module.RUNTIME_JOBS_NAME}"' in verifier_source
    assert f'RUNTIME_STATUS_NAME = "{module.RUNTIME_STATUS_NAME}"' in verifier_source
    assert module.DEFAULT_SCRIPTS_DIR == (module.REMOTE_ROOT / "scripts" / "evaluation_speed_v6_v4")


def test_v6_pins_reviewed_speed_v5_core_before_loading() -> None:
    module = _load_script()

    assert re.fullmatch(r"[0-9a-f]{64}", module.EXPECTED_SPEED_V5_CORE_SHA256)
    assert _sha256(CORE_PATH) == module.EXPECTED_SPEED_V5_CORE_SHA256


def test_v6_retains_its_reviewed_legacy_component_pins() -> None:
    module = _load_script()

    assert module.EXPECTED_COMPONENT_SHA256 == {
        "run_600m_speaker_evaluation_queue.py": (
            "8c2842875bf9aa94d5249066e237396c2ecea487a0dc5c21af4dbe384b9b7414"
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
            "9f330552c018027522457bb171aa34599e35bbb303e6c0792e2396fe266e0900"
        ),
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", expected) and expected != module.PENDING_COMPONENT_SHA256
        for expected in module.EXPECTED_COMPONENT_SHA256.values()
    )


def test_v6_rejects_core_sha_mismatch_before_executing_it(tmp_path: Path) -> None:
    module = _load_script()
    marker = tmp_path / "executed.txt"
    candidate = tmp_path / "unreviewed-core.py"
    candidate.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="speed-v5 core SHA-256 mismatch"):
        module._load_speed_v5_core(  # noqa: SLF001 - explicit loader contract test.
            candidate,
            expected_sha256="0" * 64,
        )

    assert not marker.exists()


def test_prepare_creates_v6_config_once_without_reusing_outputs(tmp_path: Path) -> None:
    module = _load_script()
    source_path, jobs_path, training_status = _write_prepare_inputs(tmp_path)
    output_root = tmp_path / "evaluation_speed_v6"
    config_path = output_root / "evaluation-queue-speed-v6.json"
    status_path = output_root / "evaluation-status-speed-v6.jsonl"
    kwargs: PrepareArguments = {
        "source_path": source_path,
        "destination": config_path,
        "status_path": status_path,
        "output_root": output_root,
        "expected_source_sha256": _sha256(source_path),
        "jobs_path": jobs_path,
        "training_status_path": training_status,
        "expected_jobs_sha256": _sha256(jobs_path),
    }

    result = module.prepare_speed_v6_config(**kwargs)
    document = json.loads(config_path.read_text(encoding="utf-8"))

    assert result["prepared"] is True
    assert result["model_count"] == MODEL_COUNT
    assert document["training_jobs"] == str(jobs_path.resolve())
    assert document["training_status"] == str(training_status.resolve())
    assert all("reuse" not in model for model in document["models"])
    assert all(
        Path(model["generation_dir"]).is_relative_to(output_root.resolve())
        for model in document["models"]
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.prepare_speed_v6_config(**kwargs)


def test_v6_pins_reviewed_remote_config() -> None:
    module = _load_script()

    assert module.EXPECTED_CONFIG_SHA256 == (
        "4717517f4229dc416bfdbaf7d73d5315873c2f4e9a37488f260acce5848cc17a"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", module.EXPECTED_CONFIG_SHA256)
    assert module.EXPECTED_CONFIG_SHA256 != module.PENDING_CONFIG_SHA256


def _recipe_body(catalog: str, name: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(name)} \*args:\n(?P<body>(?:    .*\n)+)",
        catalog,
    )
    assert match is not None
    return match.group("body")


def test_just_catalog_adds_v6_without_repointing_v5() -> None:
    catalog = Path("justfile").read_text(encoding="utf-8")

    assert _recipe_body(catalog, "speaker-evaluation-speed-v5") == (
        '    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v3.py "$@"\n'
    )
    assert _recipe_body(catalog, "speaker-evaluation-speed-v6") == (
        '    uv run python scripts/launch_600m_speaker_evaluation_queue_speed_v4.py "$@"\n'
    )
    assert _recipe_body(catalog, "remote-speaker-evaluation-speed-v5") == (
        "    just remote-python '{{ remote_evaluation_speed_v5_launcher }}' \"$@\"\n"
    )
    assert _recipe_body(catalog, "remote-speaker-evaluation-speed-v6") == (
        "    just remote-python '{{ remote_evaluation_speed_v6_launcher }}' \"$@\"\n"
    )
    assert (
        "remote_evaluation_speed_v6_launcher := remote_work_root + "
        "'\\scripts\\evaluation_speed_v6_v4\\launch_600m_speaker_evaluation_queue_speed_v4.py'"
    ) in catalog


def test_agents_catalog_mentions_fresh_speed_v6_launcher() -> None:
    catalog = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "speaker-evaluation-speed-v6" in catalog
    assert "140-case" in catalog
