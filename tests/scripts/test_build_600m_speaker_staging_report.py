from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/build_600m_speaker_staging_report.py")
MODEL_COUNT = 12


@dataclass(slots=True)
class Fixture:
    training_jobs: Path
    evaluation_dirs: list[Path]
    voice_bank_baseline: Path
    voice_bank_root: Path
    staging_root: Path
    output: Path


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_600m_speaker_staging_report",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> Fixture:
    voice_bank_root = tmp_path / "active-voice-bank"
    speaker_dir = voice_bank_root / "speakers"
    speaker_dir.mkdir(parents=True)
    manifest = voice_bank_root / "voice_bank_speakers.toml"
    manifest.write_text('[narrator]\nref_embed = "speakers/current.speaker.safetensors"\n')
    current_speaker = speaker_dir / "current.speaker.safetensors"
    current_speaker.write_bytes(b"active-speaker")
    voice_bank_baseline = tmp_path / "voice-bank-baseline.json"
    voice_bank_baseline.write_text(
        json.dumps(
            {
                "schema_version": "voice-bank-snapshot/v1",
                "voice_bank_root": str(voice_bank_root),
                "manifest": {
                    "path": str(manifest),
                    "sha256": _sha256(manifest),
                    "size": manifest.stat().st_size,
                },
                "speaker_count": 1,
                "speakers": [
                    {
                        "path": str(current_speaker),
                        "name": current_speaker.name,
                        "sha256": _sha256(current_speaker),
                        "size": current_speaker.stat().st_size,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    evaluation_dirs: list[Path] = []
    job_rows: list[dict[str, object]] = []
    for index in range(MODEL_COUNT):
        model_id = f"model-{index:02d}"
        job_rows.append({"model_id": model_id})
        embedding = tmp_path / "training" / model_id / "checkpoint_0001000.speaker.safetensors"
        embedding.parent.mkdir(parents=True)
        embedding.write_bytes(f"embedding:{model_id}".encode())
        selected = {
            "model_id": model_id,
            "checkpoint_step": 1000,
            "embedding_path": str(embedding),
            "embedding_sha256": _sha256(embedding),
            "training_config_sha256": "a" * 64,
            "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
            "base_checkpoint_sha256": "b" * 64,
            "base_revision": "c" * 40,
            "run_id": "d" * 64,
            "rank": 1,
        }
        evaluation_dir = tmp_path / "evaluation" / model_id
        evaluation_dir.mkdir(parents=True)
        selected_path = evaluation_dir / "selected-models.json"
        selected_path.write_text(
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-evaluation/v1",
                    "selections": [selected],
                },
            ),
            encoding="utf-8",
        )
        (evaluation_dir / "evaluation-verification.json").write_text(
            json.dumps(
                {
                    "schema_version": "speaker-checkpoint-evaluation-verification/v2",
                    "status": "PASS",
                    "selected": selected,
                    "artifact_sha256": {str(selected_path): _sha256(selected_path)},
                },
            ),
            encoding="utf-8",
        )
        evaluation_dirs.append(evaluation_dir)
    training_jobs = tmp_path / "training-jobs.json"
    training_jobs.write_text(
        json.dumps(
            {
                "schema_version": "speaker-training-jobs/v1",
                "base_checkpoint_sha256": "b" * 64,
                "checkpoint_revision": "c" * 40,
                "upstream_commit": "e" * 40,
                "jobs": job_rows,
            },
        ),
        encoding="utf-8",
    )
    return Fixture(
        training_jobs=training_jobs,
        evaluation_dirs=evaluation_dirs,
        voice_bank_baseline=voice_bank_baseline,
        voice_bank_root=voice_bank_root,
        staging_root=tmp_path / "proposed-staging",
        output=tmp_path / "staging-report.json",
    )


def _argv(fixture: Fixture) -> list[str]:
    args = [
        "--training-jobs",
        str(fixture.training_jobs),
        "--voice-bank-baseline",
        str(fixture.voice_bank_baseline),
        "--voice-bank-root",
        str(fixture.voice_bank_root),
        "--staging-root",
        str(fixture.staging_root),
        "--output",
        str(fixture.output),
    ]
    for path in fixture.evaluation_dirs:
        args.extend(("--evaluation-dir", str(path)))
    return args


def test_main_builds_non_destructive_twelve_model_staging_report(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)

    assert module.main(_argv(fixture)) == 0

    report = json.loads(fixture.output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "speaker-model-staging-report/v1"
    assert report["status"] == "PASS"
    assert report["deployment_performed"] is False
    assert report["active_voice_bank_unchanged"] is True
    assert report["active_voice_bank_snapshot_sha256"] == _sha256(
        fixture.voice_bank_baseline,
    )
    assert report["training_jobs_sha256"] == _sha256(fixture.training_jobs)
    assert len(report["selections"]) == MODEL_COUNT
    assert [row["model_id"] for row in report["selections"]] == [
        f"model-{index:02d}" for index in range(MODEL_COUNT)
    ]
    for row in report["selections"]:
        assert row["embedding_verified"] is True
        assert row["evaluation_verified"] is True
        assert row["proposed_staging_path"].startswith(str(fixture.staging_root.resolve()))
    assert not fixture.staging_root.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "voice_bank_manifest_drift",
        "voice_bank_speaker_drift",
        "embedding_drift",
        "evaluation_failed",
        "selection_mismatch",
        "duplicate_evaluation",
        "missing_evaluation",
    ],
)
def test_contract_drift_fails_without_report(
    tmp_path: Path,
    failure: str,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    first = fixture.evaluation_dirs[0]
    if failure == "voice_bank_manifest_drift":
        (fixture.voice_bank_root / "voice_bank_speakers.toml").write_text("changed\n")
    elif failure == "voice_bank_speaker_drift":
        next((fixture.voice_bank_root / "speakers").iterdir()).write_bytes(b"changed")
    elif failure == "embedding_drift":
        selected = json.loads((first / "selected-models.json").read_text(encoding="utf-8"))
        Path(selected["selections"][0]["embedding_path"]).write_bytes(b"changed")
    elif failure == "evaluation_failed":
        path = first / "evaluation-verification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "FAIL"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "selection_mismatch":
        path = first / "evaluation-verification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selected"]["checkpoint_step"] = 1500
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "duplicate_evaluation":
        fixture.evaluation_dirs[-1] = fixture.evaluation_dirs[0]
    else:
        fixture.evaluation_dirs.pop()

    with pytest.raises((OSError, TypeError, ValueError)):
        module.main(_argv(fixture))

    assert not fixture.output.exists()
    assert not fixture.staging_root.exists()


def test_output_and_staging_root_must_be_separate_from_active_voice_bank(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    fixture.staging_root = fixture.voice_bank_root / "staging"

    with pytest.raises(ValueError, match="active voice bank"):
        module.main(_argv(fixture))

    assert not fixture.output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _write_fixture(tmp_path)
    fixture.output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main(_argv(fixture))

    assert fixture.output.read_text(encoding="utf-8") == "keep"
