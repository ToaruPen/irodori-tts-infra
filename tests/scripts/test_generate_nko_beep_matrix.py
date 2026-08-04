# ruff: noqa: SLF001 - script helpers are the unit-test API.
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/generate_nko_beep_matrix.py")
EXPECTED_TEXT_CASES = 7
EXPECTED_SEEDS = 2
FILTER_SEED = 5678
SHA256_HEX_LENGTH = 64
BASE_CHECKPOINT_SHA256 = "e" * SHA256_HEX_LENGTH
BASE_REVISION = "f" * 40
OFFICIAL_CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
OFFICIAL_REVISION = "e863a3a93e652e09afeff3e84823a206a0a60314"
OFFICIAL_SHA256 = "93c1f8356857ab4297073f452d01c29015e0db5c83c62109800f8566900f4497"
CUSTOM_REVISION = "1" * 40
CUSTOM_SHA256 = "2" * SHA256_HEX_LENGTH
CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
EVALUATION_CASE_COUNT = 140


def _checkpoint_manifest(tmp_path: Path, *, model_id: str = "miu") -> Path:
    checkpoints: list[dict[str, object]] = []
    for step in CHECKPOINT_STEPS:
        embedding = tmp_path / "run" / f"step-{step}" / f"{model_id}.speaker.safetensors"
        embedding.parent.mkdir(parents=True, exist_ok=True)
        embedding.write_bytes(f"embedding-{step}".encode())
        checkpoints.append(
            {
                "checkpoint_step": step,
                "embedding_path": str(embedding.relative_to(tmp_path)),
                "embedding_sha256": hashlib.sha256(embedding.read_bytes()).hexdigest(),
                "training_config_sha256": "a" * SHA256_HEX_LENGTH,
                "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
                "base_revision": BASE_REVISION,
                "run_id": "miu-run",
            },
        )
    path = tmp_path / "checkpoint-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
                "models": [{"model_id": model_id, "checkpoints": checkpoints}],
                "text_ids": [
                    "word_unko",
                    "word_chinko",
                    "word_manko",
                    "sentence_unko",
                    "sentence_chinko",
                    "sentence_manko",
                    "control",
                ],
                "seeds": [1234, 5678],
                "styles": ["neutral", "calm"],
                "metrics_provenance": {
                    "reference_wavs_sha256": "b" * SHA256_HEX_LENGTH,
                    "speaker_embedding": {
                        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                        "revision": "ecapa-revision",
                        "source_sha256": "c" * SHA256_HEX_LENGTH,
                    },
                    "transcription": {
                        "model_id": "openai/whisper-large-v3-turbo",
                        "revision": "whisper-revision",
                        "source_sha256": "d" * SHA256_HEX_LENGTH,
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    return path


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_nko_beep_matrix", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_cases_creates_complete_unique_neutral_matrix() -> None:
    module = _load_script()

    cases = module.build_cases(
        speaker_paths=(
            Path("a.speaker.safetensors"),
            Path("b.speaker.safetensors"),
        ),
        text_cases=module.TEXT_CASES,
        seeds=module.SEEDS,
    )

    assert len(cases) == 2 * EXPECTED_TEXT_CASES * EXPECTED_SEEDS
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.style for case in cases} == {"neutral"}


def test_deployed_speaker_case_id_is_unchanged() -> None:
    module = _load_script()

    (case,) = module.build_cases(
        speaker_paths=(Path("miu.speaker.safetensors"),),
        text_cases=(module.TEXT_CASES[0],),
        seeds=(1234,),
    )

    assert case.case_id == "miu__word_unko__seed-1234__neutral"


def test_checkpoint_manifest_expands_model_steps_without_case_id_collisions(
    tmp_path: Path,
) -> None:
    module = _load_script()
    manifest_path = _checkpoint_manifest(tmp_path)

    candidates = module.load_checkpoint_manifest(manifest_path)
    cases = module.build_checkpoint_cases(
        candidates=candidates,
        text_cases=(module.TEXT_CASES[0],),
        seeds=(1234,),
    )

    assert [candidate.model_id for candidate in candidates] == ["miu"] * len(CHECKPOINT_STEPS)
    assert [candidate.checkpoint_step for candidate in candidates] == list(CHECKPOINT_STEPS)
    assert (
        candidates[0].speaker_path == (tmp_path / "run/step-1000/miu.speaker.safetensors").resolve()
    )
    assert (
        candidates[0].embedding_sha256
        == hashlib.sha256(
            candidates[0].speaker_path.read_bytes(),
        ).hexdigest()
    )
    assert candidates[0].base_checkpoint_sha256 == BASE_CHECKPOINT_SHA256
    assert (
        candidates[0].evaluation_manifest_sha256
        == hashlib.sha256(
            manifest_path.read_bytes(),
        ).hexdigest()
    )
    assert {case.case_id for case in cases} == {
        f"miu__checkpoint-{step}__word_unko__seed-1234__neutral" for step in CHECKPOINT_STEPS
    }
    assert len({case.case_id for case in cases}) == len(cases)


def test_filter_cases_applies_all_repeatable_filters() -> None:
    module = _load_script()
    cases = module.build_cases(
        speaker_paths=(
            Path("a.speaker.safetensors"),
            Path("b.speaker.safetensors"),
        ),
        text_cases=module.TEXT_CASES,
        seeds=module.SEEDS,
    )

    filtered = module.filter_cases(
        cases,
        speakers=frozenset({"a"}),
        text_ids=frozenset({"sentence_manko"}),
        seeds=frozenset({FILTER_SEED}),
    )

    assert len(filtered) == 1
    assert filtered[0].speaker_path == Path("a.speaker.safetensors")
    assert filtered[0].text_case.text_id == "sentence_manko"
    assert filtered[0].seed == FILTER_SEED


def test_filter_cases_uses_all_cases_for_empty_filters() -> None:
    module = _load_script()
    cases = module.build_cases(
        speaker_paths=(Path("a.speaker.safetensors"),),
        text_cases=module.TEXT_CASES,
        seeds=module.SEEDS,
    )

    filtered = module.filter_cases(
        cases,
        speakers=frozenset(),
        text_ids=frozenset(),
        seeds=frozenset(),
    )

    assert filtered == cases


def test_deployed_mode_defaults_to_pinned_official_checkpoint(tmp_path: Path) -> None:
    module = _load_script()
    speakers_dir = tmp_path / "speakers"
    speakers_dir.mkdir()
    (speakers_dir / "miu.speaker.safetensors").write_bytes(b"embedding")
    args = module._parse_args(
        ["--speakers-dir", str(speakers_dir), "--output-dir", str(tmp_path / "output")],
    )

    _, settings = module._generation_plan(args)

    assert settings.checkpoint == OFFICIAL_CHECKPOINT
    assert settings.checkpoint_revision == OFFICIAL_REVISION
    assert settings.checkpoint_sha256 == OFFICIAL_SHA256


def test_deployed_mode_passes_complete_custom_checkpoint_contract(tmp_path: Path) -> None:
    module = _load_script()
    speakers_dir = tmp_path / "speakers"
    speakers_dir.mkdir()
    (speakers_dir / "miu.speaker.safetensors").write_bytes(b"embedding")
    args = module._parse_args(
        [
            "--speakers-dir",
            str(speakers_dir),
            "--output-dir",
            str(tmp_path / "output"),
            "--checkpoint",
            "org/custom-model",
            "--checkpoint-revision",
            CUSTOM_REVISION,
            "--checkpoint-sha256",
            CUSTOM_SHA256,
        ],
    )

    _, settings = module._generation_plan(args)

    assert settings.checkpoint == "org/custom-model"
    assert settings.checkpoint_revision == CUSTOM_REVISION
    assert settings.checkpoint_sha256 == CUSTOM_SHA256


@pytest.mark.parametrize(
    "checkpoint_options",
    [
        ["--checkpoint", "org/custom-model"],
        [
            "--checkpoint",
            "org/custom-model",
            "--checkpoint-revision",
            CUSTOM_REVISION,
        ],
        ["--checkpoint", "org/custom-model", "--checkpoint-sha256", CUSTOM_SHA256],
    ],
)
def test_custom_checkpoint_requires_revision_and_sha256(
    checkpoint_options: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module._parse_args(
            [
                "--speakers-dir",
                "speakers",
                "--output-dir",
                "output",
                *checkpoint_options,
            ],
        )

    assert "custom --checkpoint requires both" in capsys.readouterr().err


@pytest.mark.parametrize(
    "checkpoint_options",
    [
        ["--checkpoint", "org/custom-model"],
        ["--checkpoint-revision", CUSTOM_REVISION],
        ["--checkpoint-sha256", CUSTOM_SHA256],
    ],
)
def test_manifest_mode_rejects_checkpoint_overrides(
    checkpoint_options: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module._parse_args(
            [
                "--checkpoint-manifest",
                "manifest.json",
                "--output-dir",
                "output",
                *checkpoint_options,
            ],
        )

    assert "checkpoint overrides are not allowed" in capsys.readouterr().err


def test_checkpoint_manifest_requires_complete_evaluation_provenance(tmp_path: Path) -> None:
    module = _load_script()
    manifest_path = _checkpoint_manifest(tmp_path)
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["models"][0]["checkpoints"][0]["training_config_sha256"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training_config_sha256"):
        module.load_checkpoint_manifest(manifest_path)


def test_checkpoint_manifest_requires_base_checkpoint_sha256(tmp_path: Path) -> None:
    module = _load_script()
    manifest_path = _checkpoint_manifest(tmp_path)
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["models"][0]["checkpoints"][0]["base_checkpoint_sha256"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="base_checkpoint_sha256"):
        module.load_checkpoint_manifest(manifest_path)


def test_checkpoint_result_rows_bind_identity_provenance_and_wav(tmp_path: Path) -> None:
    module = _load_script()
    candidate = module.load_checkpoint_manifest(_checkpoint_manifest(tmp_path))[0]
    case = module.build_checkpoint_cases(
        candidates=(candidate,),
        text_cases=(module.TEXT_CASES[0],),
        seeds=(1234,),
    )[0]
    wav_path = tmp_path / "relative-output" / "wav" / "case.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"fixture-wav")
    settings = module.IrodoriRuntimeSettings(checkpoint=candidate.base_checkpoint)

    success = module._result_row(
        case,
        settings=settings,
        status="SUCCESS",
        elapsed_seconds=0.1,
        wav_path=wav_path,
    )
    error = module._result_row(
        case,
        settings=settings,
        status="ERROR",
        elapsed_seconds=0.1,
        wav_path=None,
        exc=RuntimeError("failed"),
    )

    expected_identity = {
        "model_id": "miu",
        "checkpoint_step": 1000,
        "checkpoint": candidate.base_checkpoint,
        "speaker_filename": candidate.speaker_path.name,
        "embedding_path": str(candidate.speaker_path.resolve()),
        "embedding_sha256": candidate.embedding_sha256,
        "evaluation_manifest_sha256": candidate.evaluation_manifest_sha256,
        "base_checkpoint_sha256": candidate.base_checkpoint_sha256,
        "text_id": "word_unko",
        "seed": 1234,
        "style": "neutral",
        "provenance": {
            "training_config_sha256": candidate.training_config_sha256,
            "base_checkpoint": candidate.base_checkpoint,
            "base_revision": candidate.base_revision,
            "run_id": candidate.run_id,
        },
    }
    assert {field: success[field] for field in expected_identity} == expected_identity
    assert {field: error[field] for field in expected_identity} == expected_identity
    assert success["wav_path"] == str(wav_path.resolve())
    assert success["wav_sha256"] == hashlib.sha256(wav_path.read_bytes()).hexdigest()
    assert error["wav_path"] is None
    assert error["wav_sha256"] is None


def test_generate_case_rechecks_embedding_sha_before_synthesis(tmp_path: Path) -> None:
    module = _load_script()
    candidate = module.load_checkpoint_manifest(_checkpoint_manifest(tmp_path))[0]
    case = module.build_checkpoint_cases(
        candidates=(candidate,),
        text_cases=(module.TEXT_CASES[0],),
        seeds=(1234,),
    )[0]
    candidate.speaker_path.write_bytes(b"tampered")
    synthesis_calls: list[object] = []

    class _Backend:
        @staticmethod
        def synthesize(request: object) -> object:
            synthesis_calls.append(request)
            return SimpleNamespace(wav_bytes=b"fixture-wav")

    row = module._generate_case(
        _Backend(),
        case=case,
        wav_dir=tmp_path,
        settings=module.IrodoriRuntimeSettings(
            checkpoint=candidate.base_checkpoint,
            checkpoint_revision=candidate.base_revision,
            checkpoint_sha256=candidate.base_checkpoint_sha256,
        ),
    )

    assert row["status"] == "ERROR"
    assert "embedding SHA-256 mismatch" in str(row["exception_message"])
    assert synthesis_calls == []


def test_checkpoint_main_relative_output_dir_emits_absolute_single_prefix_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    manifest_path = _checkpoint_manifest(tmp_path)

    class _Backend:
        @staticmethod
        def synthesize(_request: object) -> object:
            return SimpleNamespace(wav_bytes=b"fixture-wav")

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(module, "create_irodori_backend", lambda _settings: _Backend())
    monkeypatch.chdir(tmp_path)

    exit_code = module.main(
        [
            "--checkpoint-manifest",
            manifest_path.name,
            "--output-dir",
            "artifacts/evaluation",
        ],
    )

    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "artifacts/evaluation/generation-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == len(CHECKPOINT_STEPS) * 7 * 2 * 2
    assert len(rows) == EVALUATION_CASE_COUNT
    assert {row["style"] for row in rows} == {"neutral", "calm"}
    assert all(Path(row["wav_path"]).is_absolute() for row in rows)
    assert all(Path(row["wav_path"]).is_file() for row in rows)
