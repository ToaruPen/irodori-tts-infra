# ruff: noqa: ANN401, EM101, EM102, TRY003 - operational errors retain exact artifact context.

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess  # noqa: S404 - fixed read-only Git provenance commands.
import sys
import zipfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


EXPECTED_SPEED_V7_CORE_SHA256 = "04d602f3e78781e7bffe6d0810555a49d7dc89577b19eaa6eebf2142102773ae"
SPEED_V7_CORE_PATH = Path(__file__).with_name("launch_600m_speaker_evaluation_queue_speed_v3.py")


def _load_speed_v7_core(path: Path, *, expected_sha256: str) -> ModuleType:
    source = path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "speed-v7 core SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}, path={path}"
        )
    module_name = f"_speaker_evaluation_speed_v7_core_{actual_sha256[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    code = compile(source, str(path), "exec")
    exec(code, module.__dict__)  # noqa: S102 - executes reviewed, SHA-pinned source.
    return module


_core = _load_speed_v7_core(
    SPEED_V7_CORE_PATH,
    expected_sha256=EXPECTED_SPEED_V7_CORE_SHA256,
)

EXPECTED_MODEL_COUNT = cast("int", _core.EXPECTED_MODEL_COUNT)
REMOTE_ROOT = cast("Path", _core.REMOTE_ROOT)
QUALITY_SUCCESSOR_ROOT = (
    REMOTE_ROOT
    / "training"
    / "oop70_osananajimi_no_iru_kurashi_sp_7195504dbb"
    / "quality_retrain_orig2000_lr00035_seed2_v1"
)
DEFAULT_OUTPUT_ROOT = REMOTE_ROOT / "evaluation_speed_v7"
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-queue-speed-v7.json"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v7.jsonl"
DEFAULT_JOBS_PATH = QUALITY_SUCCESSOR_ROOT / "training-jobs.json"
DEFAULT_TRAINING_STATUS_PATH = QUALITY_SUCCESSOR_ROOT / "training-status.jsonl"
DEFAULT_SCRIPTS_BUNDLE_NAME = "evaluation_speed_v7_v1"
DEFAULT_SCRIPTS_DIR = REMOTE_ROOT / "scripts" / DEFAULT_SCRIPTS_BUNDLE_NAME
DEFAULT_SOURCE_CONFIG_PATH = cast("Path", _core.DEFAULT_SOURCE_CONFIG_PATH)

RUNTIME_SNAPSHOT_NAME = cast("str", _core.RUNTIME_SNAPSHOT_NAME)
RUNTIME_CONFIG_NAME = cast("str", _core.RUNTIME_CONFIG_NAME)
RUNTIME_JOBS_NAME = "training-jobs-speed-v1.json"
RUNTIME_STATUS_NAME = "training-status.jsonl"
RUNTIME_MANIFEST_NAME = cast("str", _core.RUNTIME_MANIFEST_NAME)
RUNTIME_SNAPSHOT_SCHEMA = cast("str", _core.RUNTIME_SNAPSHOT_SCHEMA)
UPSTREAM_RUNTIME_PROVENANCE_NAME = "upstream-runtime-provenance.json"
UPSTREAM_RUNTIME_PACKAGE_NAME = "upstream-runtime-package.zip"
UPSTREAM_RUNTIME_PROVENANCE_SCHEMA = "irodori-upstream-runtime-provenance/v1"
PINNED_UPSTREAM_COMMIT = "eaf74d6a19138f743acb5b71a445fd25a57db987"
EXPECTED_UPSTREAM_PYTHON_FILE_COUNT = 16

ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_UNIX_MODE = 0o100644
ZIP_CREATE_SYSTEM = 3
ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
ZIP_COMPRESSLEVEL = 9
EXPECTED_RUNTIME_PAYLOAD_COUNT = 11

PENDING_CONFIG_SHA256 = "PENDING_REMOTE_SPEED_V7_CONFIG_SHA256"
PENDING_JOBS_SHA256 = "PENDING_REMOTE_SPEED_V7_JOBS_SHA256"
PENDING_COMPONENT_SHA256 = "PENDING_REMOTE_SPEED_V7_COMPONENT_SHA256"
EXPECTED_CONFIG_SHA256 = PENDING_CONFIG_SHA256
EXPECTED_SOURCE_CONFIG_SHA256 = cast("str", _core.EXPECTED_SOURCE_CONFIG_SHA256)
EXPECTED_JOBS_SHA256 = PENDING_JOBS_SHA256
EXPECTED_COMPONENT_SHA256 = {
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

# The reviewed core resolves these values dynamically. This wrapper substitutes its
# final-successor inputs plus fresh, versioned output and runtime-input contract.
for _name, _value in {
    "QUALITY_SUCCESSOR_ROOT": QUALITY_SUCCESSOR_ROOT,
    "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
    "DEFAULT_CONFIG_PATH": DEFAULT_CONFIG_PATH,
    "DEFAULT_STATUS_PATH": DEFAULT_STATUS_PATH,
    "DEFAULT_JOBS_PATH": DEFAULT_JOBS_PATH,
    "DEFAULT_TRAINING_STATUS_PATH": DEFAULT_TRAINING_STATUS_PATH,
    "DEFAULT_SCRIPTS_DIR": DEFAULT_SCRIPTS_DIR,
    "EXPECTED_CONFIG_SHA256": EXPECTED_CONFIG_SHA256,
    "EXPECTED_JOBS_SHA256": EXPECTED_JOBS_SHA256,
    "EXPECTED_COMPONENT_SHA256": EXPECTED_COMPONENT_SHA256,
    "PENDING_CONFIG_SHA256": PENDING_CONFIG_SHA256,
    "RUNTIME_JOBS_NAME": RUNTIME_JOBS_NAME,
    "RUNTIME_STATUS_NAME": RUNTIME_STATUS_NAME,
}.items():
    setattr(_core, _name, _value)

sha256_file = cast("Any", _core.sha256_file)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _assert_resolved_pins(
    *,
    expected_config_sha256: str,
    expected_jobs_sha256: str,
    expected_component_sha256: Mapping[str, str],
) -> None:
    if expected_config_sha256 == PENDING_CONFIG_SHA256:
        raise ValueError(
            f"EXPECTED_CONFIG_SHA256 is {PENDING_CONFIG_SHA256}; prepare and review "
            "the create-only speed-v7 config before pinning it"
        )
    if not _is_sha256(expected_config_sha256):
        raise ValueError("speed-v7 config pin must be a lowercase SHA-256")
    _assert_final_jobs_sha256(expected_jobs_sha256)
    if set(expected_component_sha256) != set(EXPECTED_COMPONENT_SHA256):
        raise ValueError("speed-v7 component pin inventory mismatch")
    for name, expected in expected_component_sha256.items():
        if expected == PENDING_COMPONENT_SHA256:
            raise ValueError(
                f"speed-v7 component {name} is {PENDING_COMPONENT_SHA256}; "
                "materialize and review a versioned remote bundle before pinning it"
            )
        if not _is_sha256(expected):
            raise ValueError(f"speed-v7 component pin must be a lowercase SHA-256: {name}")


def _assert_final_jobs_sha256(expected_jobs_sha256: str) -> None:
    if expected_jobs_sha256 == PENDING_JOBS_SHA256 or not _is_sha256(expected_jobs_sha256):
        raise ValueError(f"speed-v7 jobs SHA-256 pin is not finalized: {expected_jobs_sha256}")


def _git_output(root: Path, *arguments: str) -> bytes:
    process = subprocess.run(  # noqa: S603 - fixed Git executable, no shell.
        ("git", "-C", str(root), *arguments),  # noqa: S607
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"upstream Git provenance command failed ({' '.join(arguments)}): {stderr}"
        )
    return process.stdout


def _tracked_upstream_python_paths(root: Path) -> tuple[str, ...]:
    tracked = _git_output(root, "ls-files", "-z", "--", "irodori_tts")
    paths = tuple(
        sorted(
            path for path in tracked.decode("utf-8").split("\0") if path and path.endswith(".py")
        )
    )
    if len(paths) != EXPECTED_UPSTREAM_PYTHON_FILE_COUNT:
        raise ValueError(
            "upstream irodori_tts package must contain exactly "
            f"{EXPECTED_UPSTREAM_PYTHON_FILE_COUNT} tracked Python files; "
            f"actual={len(paths)}"
        )
    for relative in paths:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or pure.parts[0] != "irodori_tts" or ".." in pure.parts:
            raise ValueError(f"invalid upstream tracked Python path: {relative}")
    return paths


def _build_upstream_runtime_provenance(
    root: Path,
    *,
    expected_commit: str | None = None,
) -> bytes:
    expected_commit = expected_commit or PINNED_UPSTREAM_COMMIT
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"upstream root is not a directory: {resolved}")
    top_level = Path(
        _git_output(resolved, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    if top_level != resolved:
        raise ValueError(f"upstream root is not the Git worktree root: {resolved}")
    head = _git_output(resolved, "rev-parse", "HEAD").decode("ascii").strip()
    if head != expected_commit:
        raise ValueError(
            f"upstream commit mismatch: expected={expected_commit}, actual={head}, root={resolved}"
        )
    package_status = _git_output(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "irodori_tts",
    ).decode("utf-8", errors="replace")
    if package_status:
        kind = (
            "untracked"
            if any(line.startswith("?? ") for line in package_status.splitlines())
            else "dirty"
        )
        raise ValueError(f"upstream irodori_tts package is {kind}: {package_status.strip()}")
    paths = _tracked_upstream_python_paths(resolved)
    python_files: list[dict[str, str]] = []
    for relative in paths:
        path = resolved / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"upstream package Python file is missing or symlinked: {path}")
        python_files.append({"path": relative, "sha256": sha256_file(path)})
    document = {
        "schema_version": UPSTREAM_RUNTIME_PROVENANCE_SCHEMA,
        "upstream_root": str(resolved),
        "commit": head,
        "tree": _git_output(resolved, "rev-parse", "HEAD^{tree}").decode("ascii").strip(),
        "package": "irodori_tts",
        "python_files": python_files,
    }
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _provenance_bindings(provenance: bytes) -> tuple[tuple[str, str], ...]:
    document = json.loads(provenance)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "upstream_root",
        "commit",
        "tree",
        "package",
        "python_files",
    }:
        raise ValueError("upstream runtime provenance contract mismatch")
    if (
        document.get("schema_version") != UPSTREAM_RUNTIME_PROVENANCE_SCHEMA
        or document.get("package") != "irodori_tts"
    ):
        raise ValueError("upstream runtime provenance schema or package mismatch")
    raw_files = document.get("python_files")
    if not isinstance(raw_files, list) or len(raw_files) != EXPECTED_UPSTREAM_PYTHON_FILE_COUNT:
        raise ValueError(
            "upstream runtime provenance requires exactly "
            f"{EXPECTED_UPSTREAM_PYTHON_FILE_COUNT} Python file bindings"
        )
    bindings: list[tuple[str, str]] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise ValueError("upstream runtime Python file binding contract mismatch")
        relative = raw.get("path")
        digest = raw.get("sha256")
        if not isinstance(relative, str) or not _is_sha256(digest):
            raise ValueError("upstream runtime Python file binding is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.parts[0] != "irodori_tts"
            or ".." in pure.parts
            or pure.suffix != ".py"
        ):
            raise ValueError("upstream runtime Python file path is invalid")
        bindings.append((relative, cast("str", digest)))
    if bindings != sorted(bindings) or len({path for path, _digest in bindings}) != len(bindings):
        raise ValueError("upstream runtime Python file inventory must be sorted and unique")
    return tuple(bindings)


def _build_upstream_runtime_package(root: Path, provenance: bytes) -> bytes:
    resolved = root.resolve()
    document = json.loads(provenance)
    if not isinstance(document, dict) or document.get("upstream_root") != str(resolved):
        raise ValueError("upstream runtime provenance root mismatch")
    bindings = _provenance_bindings(provenance)
    contents: list[tuple[str, bytes]] = []
    for relative, expected_sha256 in bindings:
        path = resolved / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"upstream package Python file is missing or symlinked: {path}")
        content = path.read_bytes()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "upstream Python file changed after provenance: "
                f"path={path}, expected={expected_sha256}, actual={actual_sha256}"
            )
        contents.append((relative, content))

    result = _serialize_upstream_runtime_package(contents)
    _verify_upstream_runtime_package(result, provenance)
    return result


def _serialize_upstream_runtime_package(contents: Sequence[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=ZIP_COMPRESSION,
        compresslevel=ZIP_COMPRESSLEVEL,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for relative, content in contents:
            info = zipfile.ZipInfo(relative, date_time=ZIP_TIMESTAMP)
            info.create_system = ZIP_CREATE_SYSTEM
            info.compress_type = ZIP_COMPRESSION
            info.external_attr = ZIP_UNIX_MODE << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                content,
                compress_type=ZIP_COMPRESSION,
                compresslevel=ZIP_COMPRESSLEVEL,
            )
    return output.getvalue()


def _verify_upstream_runtime_package(archive: bytes, provenance: bytes) -> None:
    bindings = _provenance_bindings(provenance)
    expected_names = [path for path, _digest in bindings]
    expected_sha256 = dict(bindings)
    contents: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as package:
            infos = package.infolist()
            names = [info.filename for info in infos]
            if names != expected_names or len(set(names)) != len(names):
                raise ValueError("upstream runtime archive entry inventory mismatch")
            if package.comment:
                raise ValueError("upstream runtime archive comment must be empty")
            for info in infos:
                metadata = (
                    info.is_dir(),
                    info.date_time,
                    info.create_system,
                    info.compress_type,
                    info.external_attr >> 16,
                    info.extra,
                    info.comment,
                )
                expected_metadata = (
                    False,
                    ZIP_TIMESTAMP,
                    ZIP_CREATE_SYSTEM,
                    ZIP_COMPRESSION,
                    ZIP_UNIX_MODE,
                    b"",
                    b"",
                )
                if metadata != expected_metadata:
                    raise ValueError(
                        f"upstream runtime archive entry metadata mismatch: {info.filename}"
                    )
                content = package.read(info)
                actual_sha256 = hashlib.sha256(content).hexdigest()
                if actual_sha256 != expected_sha256[info.filename]:
                    raise ValueError(
                        f"upstream runtime archive entry hash mismatch: {info.filename}"
                    )
                contents.append((info.filename, content))
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("upstream runtime package is not a valid deterministic ZIP") from error
    if archive != _serialize_upstream_runtime_package(contents):
        raise ValueError("upstream runtime package is not the canonical deterministic ZIP")


def _expected_base_runtime_payloads() -> set[Path]:
    return {
        Path(RUNTIME_CONFIG_NAME),
        Path(RUNTIME_JOBS_NAME),
        Path(RUNTIME_STATUS_NAME),
        *(Path("scripts") / name for name in EXPECTED_COMPONENT_SHA256),
    }


def _with_upstream_runtime_assets(
    snapshot: Any,
    provenance: bytes,
    archive: bytes,
) -> Any:
    _verify_upstream_runtime_package(archive, provenance)
    files = {
        relative: content
        for relative, content in snapshot.files.items()
        if relative != Path(RUNTIME_MANIFEST_NAME)
    }
    expected_base = _expected_base_runtime_payloads()
    if set(files) != expected_base:
        raise ValueError(
            "base runtime snapshot payload inventory mismatch: "
            f"expected={sorted(map(str, expected_base))}, actual={sorted(map(str, files))}"
        )
    files[Path(UPSTREAM_RUNTIME_PROVENANCE_NAME)] = provenance
    files[Path(UPSTREAM_RUNTIME_PACKAGE_NAME)] = archive
    old_manifest = json.loads(snapshot.files[Path(RUNTIME_MANIFEST_NAME)])
    if (
        not isinstance(old_manifest, dict)
        or old_manifest.get("schema_version") != RUNTIME_SNAPSHOT_SCHEMA
        or not isinstance(old_manifest.get("source_inputs"), dict)
    ):
        raise ValueError("base runtime snapshot manifest contract mismatch")
    manifest = {
        "schema_version": RUNTIME_SNAPSHOT_SCHEMA,
        "source_inputs": old_manifest["source_inputs"],
        "files": {
            relative.as_posix(): {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
            for relative, content in sorted(files.items(), key=lambda item: item[0].as_posix())
        },
    }
    if len(manifest["files"]) != EXPECTED_RUNTIME_PAYLOAD_COUNT:
        raise ValueError(
            "speed-v7 runtime snapshot must contain exactly "
            f"{EXPECTED_RUNTIME_PAYLOAD_COUNT} payload files"
        )
    files[Path(RUNTIME_MANIFEST_NAME)] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return replace(snapshot, files=files)


def _verified_context_and_assets(**kwargs: Any) -> tuple[Any, bytes, bytes]:
    _assert_resolved_pins(
        expected_config_sha256=kwargs["expected_config_sha256"],
        expected_jobs_sha256=kwargs["expected_jobs_sha256"],
        expected_component_sha256=kwargs["expected_component_sha256"],
    )
    context = _core._verify_context(**kwargs)  # noqa: SLF001 - reviewed pinned core contract.
    root = cast("Path", context.queue_config.upstream_root)
    provenance = _build_upstream_runtime_provenance(root)
    archive = _build_upstream_runtime_package(root, provenance)
    return context, provenance, archive


def preflight(**kwargs: Any) -> dict[str, object]:
    context, provenance, archive = _verified_context_and_assets(**kwargs)
    report = dict(context.report)
    report["upstream_runtime_provenance_sha256"] = hashlib.sha256(provenance).hexdigest()
    report["upstream_runtime_package_sha256"] = hashlib.sha256(archive).hexdigest()
    report["upstream_runtime_python_file_count"] = len(_provenance_bindings(provenance))
    return report


def launch(**kwargs: Any) -> Any:
    _core._validate_operational_paths(  # noqa: SLF001 - reviewed pinned core contract.
        config_path=kwargs["config_path"],
        status_path=kwargs["status_path"],
        output_root=kwargs["output_root"],
    )
    context, provenance, archive = _verified_context_and_assets(**kwargs)
    snapshot = _with_upstream_runtime_assets(
        _core._runtime_snapshot(context, output_root=kwargs["output_root"]),  # noqa: SLF001
        provenance,
        archive,
    )
    runtime_config = replace(
        context.queue_config,
        source_path=snapshot.config_path,
        source_sha256=snapshot.config_sha256,
        training_status=snapshot.training_status,
        training_jobs=snapshot.jobs_path,
    )
    with context.queue_module.evaluation_queue_lock(
        config=runtime_config,
        status_path=kwargs["status_path"],
    ):
        _core._assert_verified_sources_unchanged(  # noqa: SLF001
            context.sources,
            operation="speed-v7 snapshot materialization",
        )
        live_provenance = _build_upstream_runtime_provenance(runtime_config.upstream_root)
        if live_provenance != provenance:
            raise ValueError("upstream runtime provenance changed before snapshot materialization")
        live_archive = _build_upstream_runtime_package(
            runtime_config.upstream_root,
            live_provenance,
        )
        if live_archive != archive:
            raise ValueError("upstream runtime package changed before snapshot materialization")
        _core._materialize_runtime_snapshot(snapshot)  # noqa: SLF001
        return context.queue_module._run_evaluation_queue_locked(  # noqa: SLF001
            runtime_config,
            status_path=kwargs["status_path"],
            scripts_dir=snapshot.scripts_dir,
            runner=None,
            now=None,
        )


def prepare_speed_v7_config(
    *,
    source_path: Path,
    destination: Path,
    status_path: Path,
    output_root: Path,
    expected_source_sha256: str,
    jobs_path: Path,
    training_status_path: Path,
    expected_jobs_sha256: str,
) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        _core.prepare_speed_v5_config(
            source_path=source_path,
            destination=destination,
            status_path=status_path,
            output_root=output_root,
            expected_source_sha256=expected_source_sha256,
            jobs_path=jobs_path,
            training_status_path=training_status_path,
            expected_jobs_sha256=expected_jobs_sha256,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "prepare":
        parser = argparse.ArgumentParser()
        parser.add_argument("mode", choices=("prepare",))
        parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG_PATH)
        parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
        parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--expected-jobs-sha256", required=True)
        args = parser.parse_args(arguments)
        _assert_final_jobs_sha256(args.expected_jobs_sha256)
        _core._validate_operational_paths(  # noqa: SLF001 - reviewed pinned core contract.
            config_path=args.config,
            status_path=args.status_path,
            output_root=args.output_root,
        )
        result = prepare_speed_v7_config(
            source_path=args.source_config,
            destination=args.config,
            status_path=args.status_path,
            output_root=args.output_root,
            expected_source_sha256=EXPECTED_SOURCE_CONFIG_SHA256,
            jobs_path=DEFAULT_JOBS_PATH,
            training_status_path=DEFAULT_TRAINING_STATUS_PATH,
            expected_jobs_sha256=args.expected_jobs_sha256,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    setattr(_core, "preflight", preflight)  # noqa: B010 - dynamic pinned module API.
    setattr(_core, "launch", launch)  # noqa: B010 - dynamic pinned module API.
    return cast("int", _core.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
