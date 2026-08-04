# ruff: noqa: SLF001 - script helpers are the unit-test API.
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

    class EvaluationModule(ModuleType):
        REQUIRED_CHECKPOINT_STEPS: tuple[int, ...]
        REQUIRED_HARD_GATE_METRIC_CASE_COUNT: int
        REQUIRED_TEXT_IDS: tuple[str, ...]
        REQUIRED_SEEDS: tuple[int, ...]
        REQUIRED_STYLES: tuple[str, ...]


pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/evaluate_600m_speaker_checkpoints.py")
SAMPLE_RATE = 16_000
SILENCE_RATIO_LIMIT = 0.01
CASES_PER_CHECKPOINT = 4
CHECKPOINT_COUNT = 2
STYLE_PAIR_COUNT = 2
PRODUCTION_CASES_PER_CHECKPOINT = 28
PRODUCTION_HARD_GATE_METRIC_CASE_COUNT = 16
PRODUCTION_WORD_DIAGNOSTIC_CASE_COUNT = 12
PRODUCTION_STYLE_PAIR_COUNT = 14
PRODUCTION_CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
SHA256_HEX_LENGTH = 64
SPEAKER_TOKEN_COUNT = 16
SPEAKER_EMBEDDING_DIM = 768
BASE_CHECKPOINT_SHA256 = "b" * SHA256_HEX_LENGTH
GENERATION_PROVENANCE = {
    "training_config_sha256": "a" * SHA256_HEX_LENGTH,
    "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
    "base_revision": "base-revision",
    "run_id": "d" * SHA256_HEX_LENGTH,
}
HARD_GATE_TEXT_IDS = {
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
}
FULL_TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)


def _load_script() -> EvaluationModule:
    spec = importlib.util.spec_from_file_location(
        "evaluate_600m_speaker_checkpoints",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("EvaluationModule", module)


def _load_staging_script() -> ModuleType:
    path = Path("scripts/build_600m_speaker_staging_report.py")
    spec = importlib.util.spec_from_file_location(
        "build_600m_speaker_staging_report_for_evaluator_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wav(
    path: Path,
    *,
    duration: float = 1.0,
    amplitude: float = 0.2,
    clipped: bool = False,
) -> None:
    sample_count = round(duration * SAMPLE_RATE)
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    samples = np.sin(2.0 * np.pi * 220.0 * time) * amplitude
    pcm = np.round(samples * 32_767.0).astype("<i2")
    if clipped:
        pcm[:] = 32_767
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm.tobytes())


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_safetensors(path: Path, *, finite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ones((SPEAKER_TOKEN_COUNT, SPEAKER_EMBEDDING_DIM), dtype="<f4")
    if not finite:
        values.flat[0] = np.nan
    tensor = values.tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": "F32",
                "shape": [SPEAKER_TOKEN_COUNT, SPEAKER_EMBEDDING_DIM],
                "data_offsets": [0, len(tensor)],
            },
        },
        separators=(",", ":"),
    ).encode()
    padding = b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header) + len(padding)) + header + padding + tensor)


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(file_path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _case_rows(
    tmp_path: Path,
    *,
    model_id: str,
    step: int,
    similarity: float | None = 0.9,
    cer: float | None = 0.03,
    analysis_status: str = "CLEAR",
    clipped: bool = False,
    text_ids: tuple[str, ...] = ("control",),
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    generation: list[dict[str, object]] = []
    analysis: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for text_id in text_ids:
        for seed in (1, 2):
            for style, duration, amplitude in (
                ("neutral", 1.0, 0.20),
                ("calm", 1.15, 0.16),
            ):
                case_id = f"{model_id}__checkpoint-{step}__{text_id}__seed-{seed}__{style}"
                wav_path = tmp_path / "wav" / f"{case_id}.wav"
                _write_wav(
                    wav_path,
                    duration=duration,
                    amplitude=amplitude,
                    clipped=clipped,
                )
                wav_sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()
                identity: dict[str, object] = {
                    "case_id": case_id,
                    "model_id": model_id,
                    "checkpoint_step": step,
                    "checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                    "speaker_filename": f"checkpoint-{step}.speaker.safetensors",
                    "text_id": text_id,
                    "seed": seed,
                    "style": style,
                    "wav_path": str(wav_path),
                    "wav_sha256": wav_sha256,
                    "provenance": GENERATION_PROVENANCE,
                }
                generation.append({**identity, "status": "SUCCESS"})
                analysis.append(
                    {**identity, "analysis_status": analysis_status, "intervals": []},
                )
                metric_row = dict(identity)
                metric_row["metrics_status"] = (
                    "COMPLETE" if similarity is not None and cer is not None else "INCOMPLETE"
                )
                if metric_row["metrics_status"] == "INCOMPLETE":
                    metric_row["incomplete_reason"] = "metric unavailable"
                if similarity is not None:
                    metric_row["speaker_similarity"] = similarity
                if cer is not None:
                    metric_row["normalized_cer"] = cer
                metrics.append(metric_row)
    return generation, analysis, metrics


def _embedding_contract(tmp_path: Path, *, model_id: str, step: int) -> dict[str, object]:
    path = tmp_path / "embeddings" / model_id / f"checkpoint-{step}.speaker.safetensors"
    _write_safetensors(path)
    return {
        "checkpoint_step": step,
        "embedding_path": str(path.relative_to(tmp_path)),
        "embedding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        **GENERATION_PROVENANCE,
    }


def _reference_audio(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "references" / "reference.wav"
    if not path.exists():
        _write_wav(path)
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()}


def _generation_wav_path(tmp_path: Path, row: dict[str, object]) -> Path:
    path = Path(str(row["wav_path"]))
    return path if path.is_absolute() else tmp_path / path


def _run(  # noqa: PLR0914 - fixture assembles all evaluator inputs.
    module: EvaluationModule,
    tmp_path: Path,
    generation: list[dict[str, object]],
    analysis: list[dict[str, object]],
    metrics: list[dict[str, object]] | None,
    *,
    matrix_text_ids: tuple[str, ...] = ("control",),
) -> tuple[int, Path]:
    generation_path = tmp_path / "generation-results.jsonl"
    analysis_path = tmp_path / "analysis-results.jsonl"
    metrics_path = tmp_path / "metrics-results.jsonl"
    output_dir = tmp_path / "evaluation"
    checkpoint_steps = sorted({int(str(row["checkpoint_step"])) for row in generation})
    text_ids = list(matrix_text_ids)
    seeds = sorted({int(str(row["seed"])) for row in generation})
    styles = sorted({str(row["style"]) for row in generation})
    module.REQUIRED_CHECKPOINT_STEPS = tuple(checkpoint_steps)
    module.REQUIRED_TEXT_IDS = tuple(text_ids)
    module.REQUIRED_SEEDS = tuple(seeds)
    module.REQUIRED_STYLES = tuple(styles)
    module.REQUIRED_HARD_GATE_METRIC_CASE_COUNT = (
        len(HARD_GATE_TEXT_IDS.intersection(text_ids)) * len(seeds) * len(styles)
    )
    speaker_source = tmp_path / "models" / "ecapa"
    whisper_source = tmp_path / "models" / "whisper"
    speaker_source.mkdir(parents=True, exist_ok=True)
    whisper_source.mkdir(parents=True, exist_ok=True)
    (speaker_source / "model.bin").write_bytes(b"ecapa")
    (whisper_source / "model.bin").write_bytes(b"whisper")
    manifest: dict[str, Any] = {
        "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
        "models": [
            {
                "model_id": model_id,
                "checkpoints": [
                    _embedding_contract(tmp_path, model_id=model_id, step=step)
                    for step in sorted(
                        {
                            int(str(row["checkpoint_step"]))
                            for row in generation
                            if row["model_id"] == model_id
                        },
                    )
                ],
            }
            for model_id in sorted({str(row["model_id"]) for row in generation})
        ],
        "text_ids": text_ids,
        "seeds": seeds,
        "styles": styles,
        "metrics_provenance": {
            "reference_wavs_sha256": "c" * SHA256_HEX_LENGTH,
            "speaker_embedding": {
                "model_id": "speechbrain/spkrec-ecapa-voxceleb",
                "revision": "ecapa-revision",
                "source_sha256": _tree_sha256(speaker_source),
            },
            "transcription": {
                "model_id": "openai/whisper-large-v3-turbo",
                "revision": "whisper-revision",
                "source_sha256": _tree_sha256(whisper_source),
            },
        },
    }
    manifest_path = tmp_path / "evaluation-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    contracts = {
        (model["model_id"], checkpoint["checkpoint_step"]): checkpoint
        for model in manifest["models"]
        for checkpoint in model["checkpoints"]
    }
    generation_steps = {row["case_id"]: row["checkpoint_step"] for row in generation}
    for collection in (generation, analysis, metrics or []):
        for row in collection:
            contract = contracts[row["model_id"], generation_steps[row["case_id"]]]
            embedding_path = tmp_path / str(contract["embedding_path"])
            row.update(
                embedding_path=str(embedding_path.resolve()),
                embedding_sha256=contract["embedding_sha256"],
                evaluation_manifest_sha256=manifest_sha256,
                base_checkpoint_sha256=contract["base_checkpoint_sha256"],
            )
    _write_jsonl(generation_path, generation)
    _write_jsonl(analysis_path, analysis)
    generation_sha256 = hashlib.sha256(generation_path.read_bytes()).hexdigest()
    argv = [
        "--generation-results",
        str(generation_path),
        "--analysis-results",
        str(analysis_path),
        "--evaluation-manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
    ]
    if metrics is not None:
        for row in metrics:
            row["generation_results_sha256"] = generation_sha256
        _write_jsonl(metrics_path, metrics)
        metrics_provenance_path = tmp_path / "metrics-results.provenance.json"
        metrics_provenance_path.write_text(
            json.dumps(
                {
                    "schema_version": "speaker-metrics-extraction/v1",
                    "models": {
                        "speaker_embedding": manifest["metrics_provenance"]["speaker_embedding"],
                        "transcription": {
                            **manifest["metrics_provenance"]["transcription"],
                            "device": "cpu",
                            "dtype": "float32",
                        },
                    },
                    "input_sha256": {
                        "generation_results": generation_sha256,
                        "reference_wavs": manifest["metrics_provenance"]["reference_wavs_sha256"],
                        "generated_audio": {
                            str(_generation_wav_path(tmp_path, row).resolve()): row["wav_sha256"]
                            for row in generation
                            if row["status"] == "SUCCESS"
                        },
                        "reference_audio": _reference_audio(tmp_path),
                    },
                    "case_count": len(metrics),
                    "complete_count": sum(row["metrics_status"] == "COMPLETE" for row in metrics),
                    "incomplete_count": sum(row["metrics_status"] != "COMPLETE" for row in metrics),
                },
            ),
            encoding="utf-8",
        )
        argv.extend(
            (
                "--metrics-results",
                str(metrics_path),
                "--metrics-provenance",
                str(metrics_provenance_path),
            ),
        )
    return module.main(argv), output_dir


def test_analyze_wav_computes_pcm_quality_statistics(tmp_path: Path) -> None:
    module = _load_script()
    wav_path = tmp_path / "speech.wav"
    _write_wav(wav_path, duration=0.75, amplitude=0.25)

    stats = module.analyze_wav(wav_path)

    assert stats.duration_seconds == pytest.approx(0.75)
    assert stats.sample_rate == SAMPLE_RATE
    assert stats.sample_count == round(0.75 * SAMPLE_RATE)
    assert stats.finite_samples is True
    assert stats.rms == pytest.approx(0.25 / np.sqrt(2.0), rel=1e-3)
    assert stats.silence_ratio < SILENCE_RATIO_LIMIT
    assert stats.clipping_ratio == pytest.approx(0.0)


def test_production_matrix_contract_is_fixed() -> None:
    module = _load_script()

    assert module.REQUIRED_CHECKPOINT_STEPS == PRODUCTION_CHECKPOINT_STEPS
    assert module.REQUIRED_TEXT_IDS == (
        "word_unko",
        "word_chinko",
        "word_manko",
        "sentence_unko",
        "sentence_chinko",
        "sentence_manko",
        "control",
    )
    assert module.REQUIRED_SEEDS == (1234, 5678)
    assert module.REQUIRED_STYLES == ("neutral", "calm")
    assert module.REQUIRED_HARD_GATE_METRIC_CASE_COUNT == PRODUCTION_HARD_GATE_METRIC_CASE_COUNT


def test_cli_requires_metrics_results_and_provenance() -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module._parse_args(
            [
                "--generation-results",
                "generation.jsonl",
                "--analysis-results",
                "analysis.jsonl",
                "--evaluation-manifest",
                "manifest.json",
                "--output-dir",
                "output",
            ],
        )


def test_main_ranks_eligible_checkpoints_and_emits_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_script()
    first = _case_rows(
        tmp_path,
        model_id="anabel",
        step=1000,
        similarity=0.92,
        cer=0.02,
    )
    second = _case_rows(
        tmp_path,
        model_id="anabel",
        step=1500,
        similarity=0.92,
        cer=0.02,
    )

    exit_code, output_dir = _run(
        module,
        tmp_path,
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    )

    assert exit_code == 0
    evaluation = _read_jsonl(output_dir / "evaluation-results.jsonl")
    assert len(evaluation) == CASES_PER_CHECKPOINT * CHECKPOINT_COUNT
    assert {row["evaluation_status"] for row in evaluation} == {"PASS"}
    assert all(row["audio"]["finite_samples"] for row in evaluation)

    summaries = _read_jsonl(output_dir / "checkpoint-summary.jsonl")
    assert [row["checkpoint_step"] for row in summaries] == [1000, 1500]
    assert [row["rank"] for row in summaries] == [1, 2]
    assert all(row["status"] == "ELIGIBLE" for row in summaries)
    assert summaries[0]["style_pair_count"] == STYLE_PAIR_COUNT
    assert summaries[0]["mean_speaker_similarity"] == pytest.approx(0.92)
    assert summaries[0]["mean_normalized_cer"] == pytest.approx(0.02)
    assert summaries[0]["style_contrast"] > 0.0

    selected = json.loads((output_dir / "selected-models.json").read_text(encoding="utf-8"))
    assert selected["schema_version"] == "speaker-checkpoint-evaluation/v1"
    assert selected["selections"] == [
        {
            "checkpoint_step": 1000,
            "embedding_path": str(
                (tmp_path / "embeddings/anabel/checkpoint-1000.speaker.safetensors").resolve(),
            ),
            "embedding_sha256": hashlib.sha256(
                (tmp_path / "embeddings/anabel/checkpoint-1000.speaker.safetensors").read_bytes(),
            ).hexdigest(),
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "model_id": "anabel",
            "rank": 1,
            **GENERATION_PROVENANCE,
        },
    ]
    assert set(selected["input_sha256"]) == {
        "analysis_results",
        "generation_results",
        "metrics_results",
        "metrics_provenance",
        "evaluation_manifest",
    }
    assert all(len(digest) == SHA256_HEX_LENGTH for digest in selected["input_sha256"].values())

    config = json.loads((output_dir / "evaluation-config.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == "speaker-checkpoint-thresholds/v1"
    assert config["thresholds"] == module.DEFAULT_CONFIG.to_dict()["thresholds"]
    assert _read_jsonl(output_dir / "review-candidates.jsonl") == []


def test_main_emits_atomic_verification_accepted_by_staging_contract(
    tmp_path: Path,
) -> None:
    module = _load_script()
    rows = [
        _case_rows(
            tmp_path,
            model_id="kasumi",
            step=step,
            similarity=0.92,
            cer=0.02,
            text_ids=FULL_TEXT_IDS,
        )
        for step in PRODUCTION_CHECKPOINT_STEPS
    ]
    for generation, analysis, _ in rows[2:]:
        tone_case_id = generation[0]["case_id"]
        next(row for row in analysis if row["case_id"] == tone_case_id)["analysis_status"] = (
            "CANDIDATE"
        )

    exit_code, output_dir = _run(
        module,
        tmp_path,
        [row for group in rows for row in group[0]],
        [row for group in rows for row in group[1]],
        [row for group in rows for row in group[2]],
        matrix_text_ids=FULL_TEXT_IDS,
    )

    assert exit_code == 0
    selected_path = output_dir / "selected-models.json"
    verification_path = output_dir / "evaluation-verification.json"
    selected_document = json.loads(selected_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    [selected] = selected_document["selections"]
    expected_artifacts = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_dir.rglob("*")
        if path.is_file() and path != verification_path
    }

    assert verification == {
        "schema_version": "speaker-checkpoint-evaluation-verification/v2",
        "status": "PASS",
        "selected": selected,
        "input_sha256": selected_document["input_sha256"],
        "artifact_sha256": expected_artifacts,
        "checkpoint_count": 5,
        "evaluation_case_count": 140,
        "hard_gate_metric_case_count_per_checkpoint": 16,
        "diagnostic_word_case_count_per_checkpoint": 12,
        "eligible_steps": [1000, 1500],
        "rejected_tone_steps": list(PRODUCTION_CHECKPOINT_STEPS[2:]),
        "review_candidate_count": 3,
        "review_packet_reference_count": 3,
        "review_packet_unique_audio_count": 3,
        "review_packet_spectrogram_count": 0,
        "evaluator_script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        "metrics_results_sha256": selected_document["input_sha256"]["metrics_results"],
        "metrics_provenance_sha256": selected_document["input_sha256"]["metrics_provenance"],
    }
    assert str(selected_path.resolve()) in verification["artifact_sha256"]
    assert str(verification_path.resolve()) not in verification["artifact_sha256"]
    assert not (output_dir / "evaluation-verification.json.tmp").exists()

    staging = _load_staging_script()
    staged = staging._validate_evaluation(
        selected_document,
        verification=verification,
        selected_path=selected_path.resolve(),
        verification_path=verification_path.resolve(),
        evaluation_dir=output_dir.resolve(),
        staging_root=(tmp_path / "proposed-staging").resolve(),
    )
    assert staged["model_id"] == "kasumi"
    assert staged["evaluation_verified"] is True


def test_main_emits_failed_verification_when_no_checkpoint_is_selected(
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(
        tmp_path,
        model_id="kasumi",
        step=1000,
        analysis_status="CANDIDATE",
    )

    exit_code, output_dir = _run(module, tmp_path, generation, analysis, metrics)

    assert exit_code == 1
    verification_path = output_dir / "evaluation-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["schema_version"] == ("speaker-checkpoint-evaluation-verification/v2")
    assert verification["status"] == "FAIL"
    assert verification["selected"] is None
    assert verification["eligible_steps"] == []
    assert verification["rejected_tone_steps"] == [1000]
    assert str(verification_path.resolve()) not in verification["artifact_sha256"]


def test_word_metrics_are_nonblocking_diagnostics_and_hard_gate_metrics_drive_summary(
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(
        tmp_path,
        model_id="anabel",
        step=1000,
        similarity=0.90,
        cer=0.03,
        text_ids=FULL_TEXT_IDS,
    )
    for row in metrics:
        if str(row["text_id"]).startswith("word_"):
            row["normalized_cer"] = 0.90
            row["speaker_similarity"] = 0.50 if row["style"] == "calm" else 0.95

    exit_code, output_dir = _run(
        module,
        tmp_path,
        generation,
        analysis,
        metrics,
        matrix_text_ids=FULL_TEXT_IDS,
    )

    assert exit_code == 0
    evaluation = _read_jsonl(output_dir / "evaluation-results.jsonl")
    word_rows = [row for row in evaluation if str(row["text_id"]).startswith("word_")]
    hard_gate_rows = [row for row in evaluation if row["text_id"] in HARD_GATE_TEXT_IDS]
    assert len(word_rows) == PRODUCTION_WORD_DIAGNOSTIC_CASE_COUNT
    assert len(hard_gate_rows) == PRODUCTION_HARD_GATE_METRIC_CASE_COUNT
    assert all(row["metric_gate_applied"] is False for row in word_rows)
    assert all(row["evaluation_status"] == "PASS" for row in word_rows)
    assert {flag for row in word_rows for flag in row["diagnostic_flags"]} == {
        "high_normalized_cer",
        "low_speaker_similarity",
    }
    assert all(row["rejection_reasons"] == [] for row in word_rows)
    assert all(row["metric_gate_applied"] is True for row in hard_gate_rows)
    assert all(row["diagnostic_flags"] == [] for row in hard_gate_rows)

    [summary] = _read_jsonl(output_dir / "checkpoint-summary.jsonl")
    assert summary["status"] == "ELIGIBLE"
    assert summary["case_count"] == PRODUCTION_CASES_PER_CHECKPOINT
    assert summary["hard_gate_metric_case_count"] == PRODUCTION_HARD_GATE_METRIC_CASE_COUNT
    assert summary["mean_speaker_similarity"] == pytest.approx(0.90)
    assert summary["mean_normalized_cer"] == pytest.approx(0.03)
    assert summary["style_pair_count"] == PRODUCTION_STYLE_PAIR_COUNT
    assert summary["rejection_reasons"] == []
    assert _read_jsonl(output_dir / "review-candidates.jsonl") == []


def test_only_missing_hard_gate_metrics_make_checkpoint_incomplete_and_enter_review(
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(
        tmp_path,
        model_id="anabel",
        step=1000,
        text_ids=FULL_TEXT_IDS,
    )
    word_metric = next(row for row in metrics if row["text_id"] == "word_unko")
    hard_gate_metric = next(row for row in metrics if row["text_id"] == "control")
    for row in (word_metric, hard_gate_metric):
        row["metrics_status"] = "INCOMPLETE"
        row["incomplete_reason"] = "metric unavailable"
        row.pop("speaker_similarity")
        row.pop("normalized_cer")

    exit_code, output_dir = _run(
        module,
        tmp_path,
        generation,
        analysis,
        metrics,
        matrix_text_ids=FULL_TEXT_IDS,
    )

    assert exit_code == 1
    evaluation = _read_jsonl(output_dir / "evaluation-results.jsonl")
    word_evaluation = next(row for row in evaluation if row["case_id"] == word_metric["case_id"])
    hard_evaluation = next(
        row for row in evaluation if row["case_id"] == hard_gate_metric["case_id"]
    )
    assert word_evaluation["evaluation_status"] == "PASS"
    assert word_evaluation["diagnostic_flags"] == [
        "metrics_incomplete",
        "missing_normalized_cer",
        "missing_speaker_similarity",
    ]
    assert hard_evaluation["evaluation_status"] == "INCOMPLETE"

    [summary] = _read_jsonl(output_dir / "checkpoint-summary.jsonl")
    assert summary["hard_gate_metric_case_count"] == PRODUCTION_HARD_GATE_METRIC_CASE_COUNT - 1
    assert summary["status"] == "INCOMPLETE"
    assert "hard_gate_metric_case_count" in summary["incomplete_reasons"]
    review = _read_jsonl(output_dir / "review-candidates.jsonl")
    assert {row["case_id"] for row in review} == {hard_gate_metric["case_id"]}


def test_one_word_tone_candidate_still_rejects_checkpoint(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(
        tmp_path,
        model_id="anabel",
        step=1000,
        text_ids=FULL_TEXT_IDS,
    )
    word_analysis = next(row for row in analysis if row["text_id"] == "word_unko")
    word_analysis["analysis_status"] = "CANDIDATE"

    exit_code, output_dir = _run(
        module,
        tmp_path,
        generation,
        analysis,
        metrics,
        matrix_text_ids=FULL_TEXT_IDS,
    )

    assert exit_code == 1
    [summary] = _read_jsonl(output_dir / "checkpoint-summary.jsonl")
    assert summary["status"] == "REJECTED"
    assert summary["rejection_reasons"] == ["tone_candidate"]
    [review] = _read_jsonl(output_dir / "review-candidates.jsonl")
    assert review["case_id"] == word_analysis["case_id"]


def test_low_similarity_rejections_enter_self_contained_review_packet(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(
        tmp_path,
        model_id="oop69",
        step=1000,
        similarity=0.50,
    )

    exit_code, output_dir = _run(module, tmp_path, generation, analysis, metrics)

    assert exit_code == 1
    review = _read_jsonl(output_dir / "review-candidates.jsonl")
    assert len(review) == CASES_PER_CHECKPOINT
    assert all(row["evaluation_status"] == "REJECTED" for row in review)
    assert all(row["review_reasons"] == ["low_speaker_similarity"] for row in review)
    packet_root = output_dir / "review_packet"
    packet = json.loads((packet_root / "manifest.json").read_text(encoding="utf-8"))
    assert len(packet["review_candidates"]) == CASES_PER_CHECKPOINT
    for candidate in packet["review_candidates"]:
        assert candidate["evaluation_status"] == "REJECTED"
        assert candidate["review_reasons"] == ["low_speaker_similarity"]
        copied = packet_root / candidate["wav"]["path"]
        assert copied.is_file()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == candidate["wav"]["sha256"]


def test_main_rejects_tones_and_clipping_and_marks_missing_metrics_incomplete(
    tmp_path: Path,
) -> None:
    module = _load_script()
    tone = _case_rows(
        tmp_path,
        model_id="miu",
        step=1000,
        analysis_status="CANDIDATE",
    )
    incomplete = _case_rows(
        tmp_path,
        model_id="miu",
        step=1500,
        similarity=None,
        cer=None,
    )
    clipped = _case_rows(
        tmp_path,
        model_id="miu",
        step=2000,
        similarity=None,
        clipped=True,
    )

    exit_code, output_dir = _run(
        module,
        tmp_path,
        tone[0] + incomplete[0] + clipped[0],
        tone[1] + incomplete[1] + clipped[1],
        tone[2] + incomplete[2] + clipped[2],
    )

    assert exit_code == 1
    summaries = _read_jsonl(output_dir / "checkpoint-summary.jsonl")
    assert [(row["checkpoint_step"], row["status"]) for row in summaries] == [
        (1000, "REJECTED"),
        (1500, "INCOMPLETE"),
        (2000, "REJECTED"),
    ]
    assert summaries[0]["rejection_reasons"] == ["tone_candidate"]
    assert summaries[1]["incomplete_reasons"] == [
        "hard_gate_metric_case_count",
        "metrics_incomplete",
        "missing_normalized_cer",
        "missing_speaker_similarity",
    ]
    assert summaries[2]["rejection_reasons"] == ["clipped_audio"]

    selected = json.loads((output_dir / "selected-models.json").read_text(encoding="utf-8"))
    assert selected["selections"] == []
    review = _read_jsonl(output_dir / "review-candidates.jsonl")
    assert {row["checkpoint_step"] for row in review} == {1000, 1500, 2000}
    assert all(row["review_reasons"] for row in review)


def test_main_routes_ambiguous_tone_rows_to_review(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(
        tmp_path,
        model_id="sora",
        step=1000,
        analysis_status="AMBIGUOUS",
    )

    exit_code, output_dir = _run(module, tmp_path, generation, analysis, metrics)

    assert exit_code == 1
    summary = _read_jsonl(output_dir / "checkpoint-summary.jsonl")[0]
    assert summary["status"] == "REVIEW"
    assert summary["review_reasons"] == ["tone_ambiguous"]
    assert len(_read_jsonl(output_dir / "review-candidates.jsonl")) == CASES_PER_CHECKPOINT


def test_main_rejects_generation_analysis_identity_or_provenance_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    analysis[0]["checkpoint_step"] = 999
    analysis[1]["provenance"] = {**GENERATION_PROVENANCE, "run_id": "different"}

    with pytest.raises(ValueError) as exc_info:
        _run(module, tmp_path, generation, analysis, metrics)
    assert "inconsistent checkpoint_step" in str(exc_info.value)
    assert "inconsistent provenance" in str(exc_info.value)


def test_main_rejects_empty_generation_results() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="generation results contain no rows"):
        module._validate_inputs({}, {}, {})


def test_main_rejects_missing_duplicate_and_extra_expected_matrix_cases(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    duplicate = {**generation[0], "case_id": "duplicate-case"}
    generation.append(duplicate)
    analysis.append({**analysis[0], "case_id": "duplicate-case"})
    metrics.append({**metrics[0], "case_id": "duplicate-case"})

    with pytest.raises(ValueError, match="duplicate expected matrix case"):
        _run(module, tmp_path, generation, analysis, metrics)

    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    generation.pop()
    analysis.pop()
    metrics.pop()
    with pytest.raises(ValueError, match="generation missing expected matrix case"):
        _run(module, tmp_path, generation, analysis, metrics)

    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    extra_identity = {
        **generation[0],
        "case_id": "unexpected-case",
        "text_id": "unexpected",
    }
    generation.append(extra_identity)
    analysis.append({**analysis[0], "case_id": "unexpected-case", "text_id": "unexpected"})
    metrics.append({**metrics[0], "case_id": "unexpected-case", "text_id": "unexpected"})
    with pytest.raises(ValueError, match="generation has unexpected matrix case"):
        _run(module, tmp_path, generation, analysis, metrics)


def test_main_requires_complete_strictly_identified_metrics_and_matching_sidecar(
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    metrics[0]["metrics_status"] = "INCOMPLETE"
    metrics[0]["incomplete_reason"] = "generation status is ERROR"
    metrics[0].pop("speaker_similarity")
    metrics[0].pop("normalized_cer")

    exit_code, output_dir = _run(module, tmp_path, generation, analysis, metrics)
    assert exit_code == 1
    [summary] = _read_jsonl(output_dir / "checkpoint-summary.jsonl")
    assert summary["status"] == "INCOMPLETE"
    assert "metrics_incomplete" in summary["incomplete_reasons"]

    metrics[0].pop("checkpoint")
    with pytest.raises(ValueError, match="metrics row requires checkpoint"):
        _run(module, tmp_path, generation, analysis, metrics)


def test_main_rejects_metrics_provenance_generation_reference_and_revision_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    _run(module, tmp_path, generation, analysis, metrics)
    sidecar = tmp_path / "metrics-results.provenance.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["input_sha256"]["generation_results"] = "d" * SHA256_HEX_LENGTH
    payload["input_sha256"]["reference_wavs"] = "e" * SHA256_HEX_LENGTH
    payload["models"]["transcription"]["revision"] = "wrong-revision"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    argv = [
        "--generation-results",
        str(tmp_path / "generation-results.jsonl"),
        "--analysis-results",
        str(tmp_path / "analysis-results.jsonl"),
        "--metrics-results",
        str(tmp_path / "metrics-results.jsonl"),
        "--metrics-provenance",
        str(sidecar),
        "--evaluation-manifest",
        str(tmp_path / "evaluation-manifest.json"),
        "--output-dir",
        str(tmp_path / "evaluation-two"),
    ]
    with pytest.raises(ValueError) as exc_info:
        module.main(argv)
    message = str(exc_info.value)
    assert "generation_results SHA" in message
    assert "reference_wavs SHA" in message
    assert "transcription revision" in message


def test_manifest_rejects_missing_changed_or_nonfinite_embedding(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    _run(module, tmp_path, generation, analysis, metrics)
    manifest_path = tmp_path / "evaluation-manifest.json"
    embedding = tmp_path / "embeddings/anabel/checkpoint-1000.speaker.safetensors"

    embedding.unlink()
    with pytest.raises(ValueError, match="embedding does not exist"):
        module._load_evaluation_manifest(manifest_path)

    _write_safetensors(embedding)
    embedding.write_bytes(embedding.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="embedding SHA-256 mismatch"):
        module._load_evaluation_manifest(manifest_path)

    _write_safetensors(embedding, finite=False)
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["models"][0]["checkpoints"][0]["embedding_sha256"] = hashlib.sha256(
        embedding.read_bytes(),
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only finite"):
        module._load_evaluation_manifest(manifest_path)


def test_manifest_requires_base_checkpoint_sha256(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    _run(module, tmp_path, generation, analysis, metrics)
    manifest_path = tmp_path / "evaluation-manifest.json"
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["models"][0]["checkpoints"][0]["base_checkpoint_sha256"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="base_checkpoint_sha256"):
        module._load_evaluation_manifest(manifest_path)


@pytest.mark.parametrize(
    "field",
    [
        "embedding_path",
        "embedding_sha256",
        "evaluation_manifest_sha256",
        "base_checkpoint_sha256",
    ],
)
def test_generation_checkpoint_identity_must_match_manifest(tmp_path: Path, field: str) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    _run(module, tmp_path, generation, analysis, metrics)
    manifest = module._load_evaluation_manifest(tmp_path / "evaluation-manifest.json")
    indexed = module._index_rows(
        module._read_jsonl(tmp_path / "generation-results.jsonl"),
        source="generation",
    )
    indexed[next(iter(indexed))][field] = "0" * SHA256_HEX_LENGTH

    with pytest.raises(ValueError, match=field):
        module._validate_checkpoint_contract(indexed, manifest=manifest)


def test_manifest_cannot_self_declare_a_smaller_matrix(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    _run(module, tmp_path, generation, analysis, metrics)
    module.REQUIRED_CHECKPOINT_STEPS = PRODUCTION_CHECKPOINT_STEPS
    module.REQUIRED_TEXT_IDS = (
        "word_unko",
        "word_chinko",
        "word_manko",
        "sentence_unko",
        "sentence_chinko",
        "sentence_manko",
        "control",
    )
    module.REQUIRED_SEEDS = (1234, 5678)
    module.REQUIRED_STYLES = ("neutral", "calm")

    with pytest.raises(ValueError, match="must exactly match"):
        module._load_evaluation_manifest(tmp_path / "evaluation-manifest.json")


def test_main_rejects_generated_audio_and_tone_evidence_after_wav_replacement(
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    _run(module, tmp_path, generation, analysis, metrics)
    Path(str(generation[0]["wav_path"])).write_bytes(b"replacement")

    with pytest.raises(ValueError) as exc_info:
        module.main(
            [
                "--generation-results",
                str(tmp_path / "generation-results.jsonl"),
                "--analysis-results",
                str(tmp_path / "analysis-results.jsonl"),
                "--metrics-results",
                str(tmp_path / "metrics-results.jsonl"),
                "--metrics-provenance",
                str(tmp_path / "metrics-results.provenance.json"),
                "--evaluation-manifest",
                str(tmp_path / "evaluation-manifest.json"),
                "--output-dir",
                str(tmp_path / "evaluation-replaced"),
            ],
        )
    message = str(exc_info.value)
    assert "generation wav_sha256" in message
    assert "analysis wav_sha256" in message
    assert "metrics provenance generated_audio" in message


def test_main_marks_zero_style_contrast_review(tmp_path: Path) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    for row in generation:
        _write_wav(Path(str(row["wav_path"])), duration=1.0, amplitude=0.2)
        digest = hashlib.sha256(Path(str(row["wav_path"])).read_bytes()).hexdigest()
        row["wav_sha256"] = digest
        case_id = row["case_id"]
        next(candidate for candidate in analysis if candidate["case_id"] == case_id)[
            "wav_sha256"
        ] = digest
        next(candidate for candidate in metrics if candidate["case_id"] == case_id)[
            "wav_sha256"
        ] = digest

    exit_code, output_dir = _run(module, tmp_path, generation, analysis, metrics)

    assert exit_code == 1
    [summary] = _read_jsonl(output_dir / "checkpoint-summary.jsonl")
    assert summary["status"] == "REVIEW"
    assert summary["style_contrast"] == pytest.approx(0.0)
    assert summary["review_reasons"] == ["insufficient_style_contrast"]


def test_review_candidate_preserves_tone_evidence_and_paired_control_ids(  # noqa: PLR0914
    tmp_path: Path,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(
        tmp_path,
        model_id="anabel",
        step=1000,
        analysis_status="CANDIDATE",
    )
    target = next(row for row in generation if row["seed"] == 1 and row["style"] == "calm")
    control = next(row for row in generation if row["seed"] == 1 and row["style"] == "neutral")
    spectrogram = tmp_path / "spectrograms" / "target.png"
    spectrogram.parent.mkdir()
    spectrogram.write_bytes(b"spectrogram")
    target["control"] = False
    control["control"] = True
    for row in analysis:
        if row["case_id"] == target["case_id"]:
            row["control"] = False
            row["intervals"] = [{"start_seconds": 0.2, "end_seconds": 0.4}]
            row["spectrogram_path"] = "spectrograms/target.png"
        elif row["case_id"] == control["case_id"]:
            row["control"] = True
    for row in metrics:
        if row["case_id"] == target["case_id"]:
            row["control"] = False
        elif row["case_id"] == control["case_id"]:
            row["control"] = True

    _, output_dir = _run(module, tmp_path, generation, analysis, metrics)
    review = _read_jsonl(output_dir / "review-candidates.jsonl")
    target_review = next(row for row in review if row["case_id"] == target["case_id"])
    assert target_review["tone_evidence"]["intervals"] == [
        {"start_seconds": 0.2, "end_seconds": 0.4},
    ]
    assert target_review["tone_evidence"]["spectrogram_path"] == "spectrograms/target.png"
    assert target_review["paired_control_case_ids"] == [control["case_id"]]

    packet_root = output_dir / "review_packet"
    packet = json.loads((packet_root / "manifest.json").read_text(encoding="utf-8"))
    assert packet["schema_version"] == "speaker-checkpoint-review-packet/v1"
    packet_target = next(
        row for row in packet["review_candidates"] if row["case_id"] == target["case_id"]
    )
    for asset in (packet_target["wav"], packet_target["spectrogram"]):
        relative_path = Path(asset["path"])
        assert not relative_path.is_absolute()
        copied = packet_root / relative_path
        assert copied.is_file()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == asset["sha256"]
    [paired_control] = packet_target["paired_controls"]
    paired_path = packet_root / paired_control["wav"]["path"]
    assert paired_control["case_id"] == control["case_id"]
    assert paired_path.is_file()
    assert hashlib.sha256(paired_path.read_bytes()).hexdigest() == paired_control["wav"]["sha256"]
    assert packet_target["wav"]["source_path"] == target["wav_path"]
    assert packet_target["spectrogram"]["source_path"] == "spectrograms/target.png"


def test_review_packet_paths_prevent_traversal_and_filename_collisions(tmp_path: Path) -> None:
    module = _load_script()
    packet_root = tmp_path / "review_packet"

    first = module._packet_relative_path(
        packet_root,
        category="audio",
        case_id="../same",
        suffix=".wav",
    )
    second = module._packet_relative_path(
        packet_root,
        category="audio",
        case_id="same",
        suffix=".wav",
    )

    assert first != second
    assert not first.is_absolute()
    assert ".." not in first.parts
    assert (packet_root / first).resolve().is_relative_to(packet_root.resolve())


def test_review_packet_rejects_relative_source_path_traversal(tmp_path: Path) -> None:
    module = _load_script()
    base = tmp_path / "analysis"
    base.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="escapes base directory"):
        module._resolve_packet_source("../outside.png", base=base)


def test_main_resolves_relative_wav_paths_from_generation_jsonl_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    generation, analysis, metrics = _case_rows(tmp_path, model_id="anabel", step=1000)
    for collection in (generation, analysis, metrics):
        for row in collection:
            row["wav_path"] = str(Path(str(row["wav_path"])).relative_to(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    exit_code, output_dir = _run(module, tmp_path, generation, analysis, metrics)

    assert exit_code == 0
    assert {
        row["evaluation_status"]
        for row in _read_jsonl(
            output_dir / "evaluation-results.jsonl",
        )
    } == {"PASS"}
