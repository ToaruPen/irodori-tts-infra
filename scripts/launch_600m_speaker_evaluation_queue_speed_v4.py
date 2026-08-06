# ruff: noqa: ANN401, EM101, EM102, TRY003 - operational errors retain exact artifact context.

from __future__ import annotations

import hashlib
import json
import subprocess  # noqa: S404 - fixed read-only Git provenance commands.
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


EXPECTED_SPEED_V5_CORE_SHA256 = "04d602f3e78781e7bffe6d0810555a49d7dc89577b19eaa6eebf2142102773ae"
SPEED_V5_CORE_PATH = Path(__file__).with_name("launch_600m_speaker_evaluation_queue_speed_v3.py")


def _load_speed_v5_core(path: Path, *, expected_sha256: str) -> ModuleType:
    source = path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "speed-v5 core SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}, path={path}"
        )
    module_name = f"_speaker_evaluation_speed_v5_core_{actual_sha256[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    code = compile(source, str(path), "exec")
    exec(code, module.__dict__)  # noqa: S102 - executes reviewed, SHA-pinned source.
    return module


_core = _load_speed_v5_core(
    SPEED_V5_CORE_PATH,
    expected_sha256=EXPECTED_SPEED_V5_CORE_SHA256,
)

EXPECTED_MODEL_COUNT = cast("int", _core.EXPECTED_MODEL_COUNT)
REMOTE_ROOT = cast("Path", _core.REMOTE_ROOT)
QUALITY_SUCCESSOR_ROOT = cast("Path", _core.QUALITY_SUCCESSOR_ROOT)
DEFAULT_OUTPUT_ROOT = REMOTE_ROOT / "evaluation_speed_v6"
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-queue-speed-v6.json"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "evaluation-status-speed-v6.jsonl"
DEFAULT_JOBS_PATH = cast("Path", _core.DEFAULT_JOBS_PATH)
DEFAULT_TRAINING_STATUS_PATH = cast("Path", _core.DEFAULT_TRAINING_STATUS_PATH)
DEFAULT_SCRIPTS_DIR = REMOTE_ROOT / "scripts" / "evaluation_speed_v6_v4"
DEFAULT_SOURCE_CONFIG_PATH = cast("Path", _core.DEFAULT_SOURCE_CONFIG_PATH)

RUNTIME_SNAPSHOT_NAME = cast("str", _core.RUNTIME_SNAPSHOT_NAME)
RUNTIME_CONFIG_NAME = cast("str", _core.RUNTIME_CONFIG_NAME)
RUNTIME_JOBS_NAME = "training-jobs-speed-v1.json"
RUNTIME_STATUS_NAME = "training-status.jsonl"
RUNTIME_MANIFEST_NAME = cast("str", _core.RUNTIME_MANIFEST_NAME)
RUNTIME_SNAPSHOT_SCHEMA = cast("str", _core.RUNTIME_SNAPSHOT_SCHEMA)
UPSTREAM_RUNTIME_PROVENANCE_NAME = "upstream-runtime-provenance.json"
UPSTREAM_RUNTIME_PROVENANCE_SCHEMA = "irodori-upstream-runtime-provenance/v1"
PINNED_UPSTREAM_COMMIT = "eaf74d6a19138f743acb5b71a445fd25a57db987"

PENDING_CONFIG_SHA256 = cast("str", _core.PENDING_CONFIG_SHA256)
PENDING_COMPONENT_SHA256 = "PENDING_REMOTE_SPEED_V6_COMPONENT_SHA256"
EXPECTED_CONFIG_SHA256 = "4717517f4229dc416bfdbaf7d73d5315873c2f4e9a37488f260acce5848cc17a"
EXPECTED_SOURCE_CONFIG_SHA256 = cast("str", _core.EXPECTED_SOURCE_CONFIG_SHA256)
EXPECTED_JOBS_SHA256 = cast("str", _core.EXPECTED_JOBS_SHA256)
EXPECTED_COMPONENT_SHA256 = {
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

# The reviewed core resolves these values dynamically. Override only the versioned
# output contract and pins; its validation, snapshot, and launch behavior stays intact.
for _name, _value in {
    "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
    "DEFAULT_CONFIG_PATH": DEFAULT_CONFIG_PATH,
    "DEFAULT_STATUS_PATH": DEFAULT_STATUS_PATH,
    "DEFAULT_SCRIPTS_DIR": DEFAULT_SCRIPTS_DIR,
    "EXPECTED_CONFIG_SHA256": EXPECTED_CONFIG_SHA256,
    "EXPECTED_COMPONENT_SHA256": EXPECTED_COMPONENT_SHA256,
    "RUNTIME_JOBS_NAME": RUNTIME_JOBS_NAME,
    "RUNTIME_STATUS_NAME": RUNTIME_STATUS_NAME,
}.items():
    setattr(_core, _name, _value)

sha256_file = cast("Any", _core.sha256_file)


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
    tracked = _git_output(resolved, "ls-files", "-z", "--", "irodori_tts")
    paths = sorted(
        path for path in tracked.decode("utf-8").split("\0") if path and path.endswith(".py")
    )
    if not paths:
        raise ValueError("upstream irodori_tts package has no tracked Python files")
    python_files = []
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


def _with_upstream_runtime_provenance(
    snapshot: Any,
    provenance: bytes,
) -> Any:
    files = {
        relative: content
        for relative, content in snapshot.files.items()
        if relative != Path(RUNTIME_MANIFEST_NAME)
    }
    files[Path(UPSTREAM_RUNTIME_PROVENANCE_NAME)] = provenance
    old_manifest = json.loads(snapshot.files[Path(RUNTIME_MANIFEST_NAME)])
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
    files[Path(RUNTIME_MANIFEST_NAME)] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return replace(snapshot, files=files)


def _verified_context_and_provenance(
    **kwargs: Any,
) -> tuple[Any, bytes]:
    context = _core._verify_context(**kwargs)  # noqa: SLF001 - reviewed pinned core contract.
    provenance = _build_upstream_runtime_provenance(
        cast("Path", context.queue_config.upstream_root),
    )
    return context, provenance


def preflight(**kwargs: Any) -> dict[str, object]:
    context, provenance = _verified_context_and_provenance(**kwargs)
    report = dict(context.report)
    report["upstream_runtime_provenance_sha256"] = hashlib.sha256(provenance).hexdigest()
    return report


def launch(**kwargs: Any) -> Any:
    _core._validate_operational_paths(  # noqa: SLF001 - reviewed pinned core contract.
        config_path=kwargs["config_path"],
        status_path=kwargs["status_path"],
        output_root=kwargs["output_root"],
    )
    context, provenance = _verified_context_and_provenance(**kwargs)
    snapshot = _with_upstream_runtime_provenance(
        _core._runtime_snapshot(context, output_root=kwargs["output_root"]),  # noqa: SLF001
        provenance,
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
        _core._assert_verified_sources_unchanged(context.sources)  # noqa: SLF001
        current = _build_upstream_runtime_provenance(runtime_config.upstream_root)
        if current != provenance:
            raise ValueError("upstream runtime provenance changed before snapshot materialization")
        _core._materialize_runtime_snapshot(snapshot)  # noqa: SLF001
        return context.queue_module._run_evaluation_queue_locked(  # noqa: SLF001
            runtime_config,
            status_path=kwargs["status_path"],
            scripts_dir=snapshot.scripts_dir,
            runner=None,
            now=None,
        )


def prepare_speed_v6_config(
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
    setattr(_core, "preflight", preflight)  # noqa: B010 - dynamic pinned module API.
    setattr(_core, "launch", launch)  # noqa: B010 - dynamic pinned module API.
    return cast("int", _core.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
