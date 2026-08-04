from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import subprocess  # noqa: S404 - tests create an isolated local Git repository.
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType


pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v5.py")
CORE_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v3.py")
V4_PATH = Path("scripts/launch_600m_speaker_evaluation_queue_speed_v4.py")
EXPECTED_V4_SHA256 = "91f38bb35f9b8f5dffd31a49f464ffd814404fe9cdbe10ab5ad35e2b4de7f9da"
EXPECTED_PYTHON_FILE_COUNT = 16


class RuntimeSnapshotLike(Protocol):
    files: Mapping[Path, bytes]


class LauncherModule(Protocol):
    REMOTE_ROOT: Path
    QUALITY_SUCCESSOR_ROOT: Path
    DEFAULT_OUTPUT_ROOT: Path
    DEFAULT_CONFIG_PATH: Path
    DEFAULT_STATUS_PATH: Path
    DEFAULT_SCRIPTS_BUNDLE_NAME: str
    DEFAULT_SCRIPTS_DIR: Path
    DEFAULT_JOBS_PATH: Path
    DEFAULT_TRAINING_STATUS_PATH: Path
    EXPECTED_SPEED_V7_CORE_SHA256: str
    EXPECTED_CONFIG_SHA256: str
    EXPECTED_COMPONENT_SHA256: Mapping[str, str]
    PENDING_CONFIG_SHA256: str
    PENDING_COMPONENT_SHA256: str
    PENDING_JOBS_SHA256: str
    EXPECTED_JOBS_SHA256: str
    PINNED_UPSTREAM_COMMIT: str
    EXPECTED_UPSTREAM_PYTHON_FILE_COUNT: int
    UPSTREAM_RUNTIME_PROVENANCE_NAME: str
    UPSTREAM_RUNTIME_PACKAGE_NAME: str
    ZIP_TIMESTAMP: tuple[int, int, int, int, int, int]
    ZIP_UNIX_MODE: int
    ZIP_CREATE_SYSTEM: int
    ZIP_COMPRESSION: int
    EXPECTED_RUNTIME_PAYLOAD_COUNT: int
    RUNTIME_MANIFEST_NAME: str
    RUNTIME_CONFIG_NAME: str
    RUNTIME_JOBS_NAME: str
    RUNTIME_STATUS_NAME: str
    _core: ModuleType

    def _load_speed_v7_core(self, path: Path, *, expected_sha256: str) -> ModuleType: ...

    def _build_upstream_runtime_provenance(
        self,
        root: Path,
        *,
        expected_commit: str | None = None,
    ) -> bytes: ...

    def _build_upstream_runtime_package(self, root: Path, provenance: bytes) -> bytes: ...

    def _verify_upstream_runtime_package(self, archive: bytes, provenance: bytes) -> None: ...

    def _with_upstream_runtime_assets(
        self,
        snapshot: object,
        provenance: bytes,
        archive: bytes,
    ) -> RuntimeSnapshotLike: ...

    def preflight(self, **kwargs: object) -> dict[str, object]: ...

    def main(self, argv: Sequence[str] | None = None) -> int: ...


def _load_script() -> LauncherModule:
    spec = importlib.util.spec_from_file_location(
        "launch_600m_speaker_evaluation_queue_speed_v5",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("LauncherModule", module)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed Git executable and test arguments.
        ("git", "-C", str(root), *args),  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_upstream_repo(
    tmp_path: Path,
    *,
    file_count: int = EXPECTED_PYTHON_FILE_COUNT,
) -> tuple[Path, str]:
    root = tmp_path / "Irodori-TTS"
    package = root / "irodori_tts"
    package.mkdir(parents=True)
    for index in range(file_count):
        name = "__init__.py" if index == 0 else f"module_{index:02d}.py"
        (package / name).write_text(f"VALUE_{index} = {index}\n", encoding="utf-8")
    (root / "README.md").write_text("upstream\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _resolved_pin_kwargs(module: LauncherModule, tmp_path: Path) -> dict[str, object]:
    return {
        "config_path": tmp_path / "config.json",
        "status_path": tmp_path / "status.jsonl",
        "scripts_dir": tmp_path / "scripts",
        "output_root": tmp_path / "output",
        "expected_config_sha256": "1" * 64,
        "expected_jobs_path": tmp_path / "jobs.json",
        "expected_jobs_sha256": "2" * 64,
        "expected_component_sha256": dict.fromkeys(
            module.EXPECTED_COMPONENT_SHA256,
            "3" * 64,
        ),
    }


def test_v7_defaults_target_only_the_final_five_model_successor() -> None:
    module = _load_script()

    assert module.DEFAULT_OUTPUT_ROOT == module.REMOTE_ROOT / "evaluation_speed_v7"
    assert module.DEFAULT_CONFIG_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-queue-speed-v7.json"
    )
    assert module.DEFAULT_STATUS_PATH == (
        module.DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v7.jsonl"
    )
    assert module.DEFAULT_SCRIPTS_BUNDLE_NAME == "evaluation_speed_v7_v1"
    assert module.DEFAULT_SCRIPTS_DIR == (
        module.REMOTE_ROOT / "scripts" / module.DEFAULT_SCRIPTS_BUNDLE_NAME
    )
    assert module.QUALITY_SUCCESSOR_ROOT == (
        module.REMOTE_ROOT
        / "training"
        / "oop70_osananajimi_no_iru_kurashi_sp_7195504dbb"
        / "quality_retrain_orig2000_lr00035_seed2_v1"
    )
    assert module.DEFAULT_JOBS_PATH == module.QUALITY_SUCCESSOR_ROOT / "training-jobs.json"
    assert module.DEFAULT_TRAINING_STATUS_PATH == (
        module.QUALITY_SUCCESSOR_ROOT / "training-status.jsonl"
    )
    assert module.EXPECTED_JOBS_SHA256 == module.PENDING_JOBS_SHA256
    assert module.EXPECTED_CONFIG_SHA256 == module.PENDING_CONFIG_SHA256
    assert module.EXPECTED_COMPONENT_SHA256 == {
        "run_600m_speaker_evaluation_queue.py": (
            "278e75341980286e7b37b12ba0df1e1a5090aab5de73f8e3d65b93df9214d978"
        ),
        "build_600m_checkpoint_evaluation_manifests.py": (
            "e3f62e07f07c949fe60d4db00a7eef11dbd1ae9111a7628dd88982a2702d0e93"
        ),
        "generate_600m_checkpoint_audio_remote.py": (
            "947babd074d83b08c2c9a535f9d718cdac17a3cbfe845e430758bc1008818816"
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
    assert module.PENDING_COMPONENT_SHA256 not in module.EXPECTED_COMPONENT_SHA256.values()
    assert {
        name: _sha256_file(Path("scripts") / name) for name in module.EXPECTED_COMPONENT_SHA256
    } == module.EXPECTED_COMPONENT_SHA256


def test_v7_pins_same_reviewed_v3_core_as_v4() -> None:
    module = _load_script()

    assert re.fullmatch(r"[0-9a-f]{64}", module.EXPECTED_SPEED_V7_CORE_SHA256)
    assert _sha256_file(CORE_PATH) == module.EXPECTED_SPEED_V7_CORE_SHA256
    assert module.EXPECTED_SPEED_V7_CORE_SHA256 == (
        "04d602f3e78781e7bffe6d0810555a49d7dc89577b19eaa6eebf2142102773ae"
    )


def test_v7_rejects_core_sha_mismatch_before_execution(tmp_path: Path) -> None:
    module = _load_script()
    marker = tmp_path / "executed.txt"
    candidate = tmp_path / "unreviewed.py"
    candidate.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="speed-v7 core SHA-256 mismatch"):
        module._load_speed_v7_core(candidate, expected_sha256="0" * 64)  # noqa: SLF001

    assert not marker.exists()


def test_provenance_requires_exactly_sixteen_tracked_python_files_and_is_deterministic(
    tmp_path: Path,
) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    (root / "scratch.txt").write_text("allowed outside package\n", encoding="utf-8")

    first = module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001
    second = module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001
    document = json.loads(first)

    assert first == second
    assert module.EXPECTED_UPSTREAM_PYTHON_FILE_COUNT == EXPECTED_PYTHON_FILE_COUNT
    assert len(document["python_files"]) == EXPECTED_PYTHON_FILE_COUNT
    assert [row["path"] for row in document["python_files"]] == sorted(
        row["path"] for row in document["python_files"]
    )


@pytest.mark.parametrize("file_count", [15, 17])
def test_provenance_rejects_wrong_tracked_python_file_count(
    tmp_path: Path,
    file_count: int,
) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path, file_count=file_count)

    with pytest.raises(ValueError, match="exactly 16 tracked Python files"):
        module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001


def test_runtime_package_zip_is_deterministic_and_matches_provenance(tmp_path: Path) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    provenance = module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001

    first = module._build_upstream_runtime_package(root, provenance)  # noqa: SLF001
    second = module._build_upstream_runtime_package(root, provenance)  # noqa: SLF001
    expected = json.loads(provenance)["python_files"]

    assert first == second
    module._verify_upstream_runtime_package(first, provenance)  # noqa: SLF001
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == [row["path"] for row in expected]
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.date_time == module.ZIP_TIMESTAMP
            assert info.create_system == module.ZIP_CREATE_SYSTEM
            assert info.compress_type == module.ZIP_COMPRESSION
            assert info.external_attr >> 16 == module.ZIP_UNIX_MODE
            assert info.extra == b""
            assert info.comment == b""


def test_runtime_package_rejects_tampered_entry_content(tmp_path: Path) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    provenance = module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001
    archive_bytes = module._build_upstream_runtime_package(root, provenance)  # noqa: SLF001
    tampered = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(archive_bytes)) as source,
        zipfile.ZipFile(
            tampered,
            "w",
            compression=module.ZIP_COMPRESSION,
        ) as destination,
    ):
        for index, info in enumerate(source.infolist()):
            content = b"tampered\n" if index == 0 else source.read(info)
            destination.writestr(info, content)

    with pytest.raises(ValueError, match="archive entry hash mismatch"):
        module._verify_upstream_runtime_package(tampered.getvalue(), provenance)  # noqa: SLF001


def test_runtime_package_rejects_noncanonical_trailing_bytes(tmp_path: Path) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    provenance = module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001
    archive = module._build_upstream_runtime_package(root, provenance)  # noqa: SLF001

    with pytest.raises(ValueError, match="canonical deterministic ZIP"):
        module._verify_upstream_runtime_package(archive + b"tampered", provenance)  # noqa: SLF001


def test_runtime_snapshot_manifest_has_exact_eleven_payload_files(tmp_path: Path) -> None:
    module = _load_script()
    root, head = _write_upstream_repo(tmp_path)
    provenance = module._build_upstream_runtime_provenance(root, expected_commit=head)  # noqa: SLF001
    archive = module._build_upstream_runtime_package(root, provenance)  # noqa: SLF001
    runtime_root = tmp_path / "runtime-inputs-v1"
    base_files = {
        Path(module.RUNTIME_CONFIG_NAME): b"{}\n",
        Path(module.RUNTIME_JOBS_NAME): b'{"jobs": []}\n',
        Path(module.RUNTIME_STATUS_NAME): b"",
        **{Path("scripts") / name: name.encode() for name in module.EXPECTED_COMPONENT_SHA256},
    }
    base_manifest = {
        "schema_version": "speaker-evaluation-runtime-inputs/v1",
        "source_inputs": {"source": "0" * 64},
        "files": {
            relative.as_posix(): {
                "sha256": _sha256_bytes(content),
                "size": len(content),
            }
            for relative, content in base_files.items()
        },
    }
    snapshot = module._core.RuntimeSnapshot(  # noqa: SLF001
        root=runtime_root,
        scripts_dir=runtime_root / "scripts",
        config_path=runtime_root / module.RUNTIME_CONFIG_NAME,
        jobs_path=runtime_root / module.RUNTIME_JOBS_NAME,
        training_status=runtime_root / module.RUNTIME_STATUS_NAME,
        files={
            **base_files,
            Path(module.RUNTIME_MANIFEST_NAME): (
                json.dumps(base_manifest, sort_keys=True) + "\n"
            ).encode(),
        },
        config_sha256=_sha256_bytes(base_files[Path(module.RUNTIME_CONFIG_NAME)]),
    )

    augmented = module._with_upstream_runtime_assets(snapshot, provenance, archive)  # noqa: SLF001
    manifest = json.loads(augmented.files[Path(module.RUNTIME_MANIFEST_NAME)])
    payloads = {
        relative: content
        for relative, content in augmented.files.items()
        if relative != Path(module.RUNTIME_MANIFEST_NAME)
    }

    assert len(payloads) == module.EXPECTED_RUNTIME_PAYLOAD_COUNT
    assert set(manifest["files"]) == {path.as_posix() for path in payloads}
    assert set(payloads) == {
        Path(module.RUNTIME_CONFIG_NAME),
        Path(module.RUNTIME_JOBS_NAME),
        Path(module.RUNTIME_STATUS_NAME),
        Path(module.UPSTREAM_RUNTIME_PROVENANCE_NAME),
        Path(module.UPSTREAM_RUNTIME_PACKAGE_NAME),
        *(Path("scripts") / name for name in module.EXPECTED_COMPONENT_SHA256),
    }
    for relative, content in payloads.items():
        assert manifest["files"][relative.as_posix()] == {
            "sha256": _sha256_bytes(content),
            "size": len(content),
        }


def test_preflight_reports_provenance_and_archive_hashes_without_materializing(
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
    monkeypatch.setattr(
        module._core,  # noqa: SLF001
        "_verify_context",
        lambda **_kwargs: context,
    )

    report = module.preflight(**_resolved_pin_kwargs(module, tmp_path))

    assert re.fullmatch(
        r"[0-9a-f]{64}",
        cast("str", report["upstream_runtime_provenance_sha256"]),
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        cast("str", report["upstream_runtime_package_sha256"]),
    )
    assert report["upstream_runtime_python_file_count"] == EXPECTED_PYTHON_FILE_COUNT
    assert not (tmp_path / module.UPSTREAM_RUNTIME_PROVENANCE_NAME).exists()
    assert not (tmp_path / module.UPSTREAM_RUNTIME_PACKAGE_NAME).exists()


@pytest.mark.parametrize("pending_kind", ["config", "component", "jobs"])
def test_preflight_rejects_pending_pins_before_reading_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending_kind: str,
) -> None:
    module = _load_script()
    kwargs = _resolved_pin_kwargs(module, tmp_path)
    if pending_kind == "config":
        kwargs["expected_config_sha256"] = module.PENDING_CONFIG_SHA256
    elif pending_kind == "jobs":
        kwargs["expected_jobs_sha256"] = module.PENDING_JOBS_SHA256
    else:
        components = cast("dict[str, str]", kwargs["expected_component_sha256"])
        components[next(iter(components))] = module.PENDING_COMPONENT_SHA256
    monkeypatch.setattr(
        module._core,  # noqa: SLF001
        "_verify_context",
        lambda **_kwargs: pytest.fail("pending pins must fail before source verification"),
    )

    with pytest.raises(ValueError, match="PENDING_REMOTE_SPEED_V7"):
        module.preflight(**kwargs)


def test_prepare_requires_an_explicit_final_jobs_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module._core,  # noqa: SLF001 - isolate fixed-path validation in CLI unit test.
        "_validate_operational_paths",
        lambda **_kwargs: None,
    )

    def prepare(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"prepared": True}

    monkeypatch.setattr(module, "prepare_speed_v7_config", prepare)

    with pytest.raises(SystemExit):
        module.main(["prepare"])
    with pytest.raises(ValueError, match="jobs SHA-256 pin is not finalized"):
        module.main(
            [
                "prepare",
                "--expected-jobs-sha256",
                module.PENDING_JOBS_SHA256,
            ]
        )

    assert (
        module.main(
            [
                "prepare",
                "--expected-jobs-sha256",
                "4" * 64,
            ]
        )
        == 0
    )
    assert captured["jobs_path"] == module.DEFAULT_JOBS_PATH
    assert captured["training_status_path"] == module.DEFAULT_TRAINING_STATUS_PATH
    assert captured["expected_jobs_sha256"] == "4" * 64


def test_existing_v4_wrapper_is_unchanged() -> None:
    assert _sha256_file(V4_PATH) == EXPECTED_V4_SHA256
