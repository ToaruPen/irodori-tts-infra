# ruff: noqa: PLR0914, PLR2004, PLR6301, RUF001, SLF001
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import wave
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict, cast

import numpy as np
import pytest
from typing_extensions import override

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/audit_600m_speaker_reference_centroid.py")
MODEL_ID = "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
MODEL_PREFIX = "oop69"
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "fixture-revision"


def _load_script() -> ModuleType:
    name = "audit_600m_speaker_reference_centroid"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, *, value: float, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(frames, value, dtype=np.float64)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32_767.0).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(pcm.tobytes())


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class _Embedder:
    model_id = ECAPA_MODEL_ID
    revision = ECAPA_REVISION

    def __init__(self, source_sha256: str) -> None:
        self.source_sha256 = source_sha256

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        assert sample_rate == 16_000
        mean = float(np.mean(samples))
        return np.array([1.0, mean * 4.0, mean * mean * 2.0], dtype=np.float64)


class _MutatingEmbedder(_Embedder):
    def __init__(self, source_sha256: str, *, target: Path, replacement: bytes) -> None:
        super().__init__(source_sha256)
        self.target = target
        self.replacement = replacement
        self.call_count = 0

    @override
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.call_count += 1
        if self.call_count == 26:
            self.target.write_bytes(self.replacement)
        return super().embed(samples, sample_rate)


class Fixture(TypedDict):
    module: ModuleType
    embedder: _Embedder
    ecapa_source: Path
    clean_manifest: Path
    reference_manifest: Path
    generation_results: Path
    metrics_results: Path
    metrics_provenance: Path
    output: Path


def _fixture(tmp_path: Path) -> Fixture:
    ecapa_source = tmp_path / "ecapa"
    ecapa_source.mkdir(parents=True)
    (ecapa_source / "model.bin").write_bytes(b"fixture-ecapa")

    clean_rows: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for index in range(25):
        source_id = f"{MODEL_PREFIX}:{index:08d}"
        audio_path = tmp_path / "source" / f"source-{index:02d}.wav"
        reference_path = tmp_path / "references" / f"reference-{index:02d}.wav"
        frames = 800 + index * 40
        value = 0.01 + index * 0.001
        _write_wav(audio_path, value=value, frames=frames)
        _write_wav(reference_path, value=value, frames=frames)
        audio_sha = _sha(audio_path)
        pcm_sha = hashlib.sha256(f"pcm-{index}".encode()).hexdigest()
        text = "（吐息）" if index == 0 else f"参照音声{index}"
        clean_rows.append(
            {
                "source_id": source_id,
                "audio_sha256": audio_sha,
                "pcm_sha256": pcm_sha,
                "text": text,
            },
        )
        references.append(
            {
                "audio_path": str(audio_path),
                "audio_sha256": audio_sha,
                "clipped_fraction": 0.0,
                "duration_quantile": index // 5 + 1,
                "duration_seconds": frames / 16_000,
                "num_frames": frames,
                "pcm_sha256": pcm_sha,
                "quality_decision": "KEEP",
                "quality_reasons": [],
                "reference_wav_channels": 1,
                "reference_wav_finite": True,
                "reference_wav_num_frames": frames,
                "reference_wav_path": str(reference_path),
                "reference_wav_sample_rate": 16_000,
                "reference_wav_sha256": _sha(reference_path),
                "rms_dbfs": -30.0,
                "sample_rate": 16_000,
                "selection_order_within_quantile": index % 5 + 1,
                "source_id": source_id,
                "text": text,
                "tone_intervals": [],
            },
        )
    clean_manifest = tmp_path / "clean-dataset.jsonl"
    _write_jsonl(clean_manifest, clean_rows)
    reference_manifest = tmp_path / "reference-wavs.json"
    reference_payload = {
        "schema_version": "speaker-similarity-references/v1",
        "created_at_utc": "2026-08-03T00:00:00+00:00",
        "model_id": MODEL_ID,
        "healthy_population_count": len(clean_rows),
        "selected_count": 25,
        "all_reference_wavs_finite": True,
        "all_selected_source_hashes_verified": True,
        "quantiles": [
            {
                "quantile": quantile,
                "population_count": 5,
                "population_min_seconds": (800 + (quantile - 1) * 5 * 40) / 16_000,
                "population_max_seconds": (800 + ((quantile - 1) * 5 + 4) * 40) / 16_000,
                "selected_count": 5,
            }
            for quantile in range(1, 6)
        ],
        "selection_strategy": {
            "duration_quantiles": 5,
            "duration_stratification": "equal_population",
            "health_filter": "healthy_only",
            "references_per_quantile": 5,
            "selection_within_each_quantile": "deterministic",
            "source": "training_clean_manifest",
        },
        "source_hashes": {
            "round4_clean_dataset_sha256": "1" * 64,
            "round4_decisions_sha256": "2" * 64,
            "round4_summary_sha256": "3" * 64,
            "training_clean_manifest_sha256": _sha(clean_manifest),
            "training_latent_provenance_sha256": "4" * 64,
            "training_provenance_sha256": "5" * 64,
        },
        "references": references,
    }
    reference_manifest.write_text(json.dumps(reference_payload), encoding="utf-8")

    generation_rows: list[dict[str, object]] = []
    for text_id in ("sentence_unko", "sentence_chinko", "sentence_manko", "control"):
        for seed in (1234, 5678):
            for style in ("neutral", "calm"):
                case_id = f"{MODEL_ID}__250__{text_id}__{seed}__{style}"
                wav_path = tmp_path / "generated" / f"{case_id}.wav"
                _write_wav(wav_path, value=0.02, frames=1200)
                generation_rows.append(
                    {
                        "schema_version": "speaker-checkpoint-search-generation-case/v1",
                        "case_id": case_id,
                        "model_id": MODEL_ID,
                        "checkpoint_step": 250,
                        "checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                        "speaker_filename": "checkpoint_0000250.speaker.safetensors",
                        "embedding_path": str(tmp_path / "embedding.safetensors"),
                        "embedding_sha256": "a" * 64,
                        "evaluation_manifest_sha256": "b" * 64,
                        "base_checkpoint_sha256": "c" * 64,
                        "text_id": text_id,
                        "text": "固定検証文",
                        "seed": seed,
                        "style": style,
                        "wav_path": str(wav_path),
                        "wav_sha256": _sha(wav_path),
                        "status": "SUCCESS",
                        "provenance": {
                            "training_config_sha256": "d" * 64,
                            "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                            "base_revision": "fixture",
                            "run_id": "fixture-run",
                        },
                    },
                )
    generation_results = tmp_path / "generation-results.jsonl"
    _write_jsonl(generation_results, generation_rows)

    module = _load_script()
    embedder = _Embedder(module.metrics.sha256_tree(ecapa_source))
    reference_embeddings = []
    for reference in references:
        samples, rate = module.metrics.read_wav(Path(str(reference["reference_wav_path"])))
        reference_embeddings.append(embedder.embed(samples, rate))
    centroid = module.metrics.aggregate_reference_centroid(reference_embeddings)
    metrics_rows: list[dict[str, object]] = []
    for row in generation_rows:
        samples, rate = module.metrics.read_wav(Path(str(row["wav_path"])))
        similarity = module.metrics.normalized_cosine_similarity(
            embedder.embed(samples, rate),
            centroid,
        )
        metrics_rows.append(
            {
                **{field: row.get(field) for field in module.metrics.IDENTITY_FIELDS},
                "generation_results_sha256": _sha(generation_results),
                "metrics_status": "COMPLETE",
                "speaker_similarity": similarity,
                "normalized_cer": 0.0,
                "transcript": "固定検証文",
            },
        )
    metrics_results = tmp_path / "metrics-results.jsonl"
    _write_jsonl(metrics_results, metrics_rows)
    metrics_provenance = tmp_path / "metrics-results.provenance.json"
    metrics_provenance.write_text(
        json.dumps(
            {
                "schema_version": "speaker-metrics-extraction/v1",
                "models": {
                    "speaker_embedding": {
                        "model_id": ECAPA_MODEL_ID,
                        "revision": ECAPA_REVISION,
                        "source_sha256": embedder.source_sha256,
                    },
                    "transcription": {
                        "model_id": "openai/whisper-large-v3-turbo",
                        "revision": "fixture",
                        "source_sha256": "e" * 64,
                    },
                },
                "input_sha256": {
                    "generation_results": _sha(generation_results),
                    "reference_wavs": _sha(reference_manifest),
                    "generated_audio": {
                        str(row["wav_path"]): row["wav_sha256"] for row in generation_rows
                    },
                    "reference_audio": {
                        str(row["reference_wav_path"]): row["reference_wav_sha256"]
                        for row in references
                    },
                },
                "case_count": 16,
                "complete_count": 16,
                "incomplete_count": 0,
            },
        ),
        encoding="utf-8",
    )
    return {
        "module": module,
        "embedder": embedder,
        "ecapa_source": ecapa_source,
        "clean_manifest": clean_manifest,
        "reference_manifest": reference_manifest,
        "generation_results": generation_results,
        "metrics_results": metrics_results,
        "metrics_provenance": metrics_provenance,
        "output": tmp_path / "reference-centroid-audit.json",
    }


def _run(fixture: Fixture, *, checkpoint_step: int | None = None) -> dict[str, Any]:
    module = fixture["module"]
    assert isinstance(module, ModuleType)
    module.run_audit(
        reference_wavs_path=Path(fixture["reference_manifest"]),
        clean_manifest_path=Path(fixture["clean_manifest"]),
        generation_results_path=Path(fixture["generation_results"]),
        metrics_results_path=Path(fixture["metrics_results"]),
        metrics_provenance_path=Path(fixture["metrics_provenance"]),
        output_path=Path(fixture["output"]),
        ecapa_source=Path(fixture["ecapa_source"]),
        embedder=fixture["embedder"],
        checkpoint_step=checkpoint_step,
    )
    raw_report: Any = json.loads(fixture["output"].read_text(encoding="utf-8"))
    return cast("dict[str, Any]", raw_report)


def _convert_to_full_generation_matrix(fixture: Fixture) -> None:
    generation_path = fixture["generation_results"]
    source_rows = [
        json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()
    ]
    source_by_case = {(row["text_id"], row["seed"], row["style"]): row for row in source_rows}
    text_ids = (
        "word_unko",
        "word_chinko",
        "word_manko",
        "sentence_unko",
        "sentence_chinko",
        "sentence_manko",
        "control",
    )
    full_rows: list[dict[str, object]] = []
    for checkpoint_step in (1000, 1500, 2000, 2500, 3000):
        for text_id in text_ids:
            template_text_id = (
                text_id
                if text_id
                in {
                    "sentence_unko",
                    "sentence_chinko",
                    "sentence_manko",
                    "control",
                }
                else "sentence_unko"
            )
            for seed in (1234, 5678):
                for style in ("neutral", "calm"):
                    row = deepcopy(source_by_case[template_text_id, seed, style])
                    row.pop("schema_version")
                    row["case_id"] = (
                        f"{MODEL_ID}__checkpoint-{checkpoint_step}__{text_id}__seed-{seed}__{style}"
                    )
                    row["checkpoint_step"] = checkpoint_step
                    row["speaker_filename"] = (
                        f"checkpoint_{checkpoint_step:07d}.speaker.safetensors"
                    )
                    row["embedding_path"] = str(
                        generation_path.parent
                        / f"checkpoint_{checkpoint_step:07d}.speaker.safetensors"
                    )
                    row["embedding_sha256"] = hashlib.sha256(
                        f"embedding-{checkpoint_step}".encode()
                    ).hexdigest()
                    row["text_id"] = text_id
                    full_rows.append(row)
    _write_jsonl(generation_path, full_rows)

    metric_path = fixture["metrics_results"]
    source_metrics = [
        json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines()
    ]
    similarity_by_case = {row["case_id"]: row["speaker_similarity"] for row in source_metrics}
    original_case_by_triplet = {
        (row["text_id"], row["seed"], row["style"]): row["case_id"] for row in source_rows
    }
    generation_sha = _sha(generation_path)
    full_metrics: list[dict[str, object]] = []
    module = fixture["module"]
    for row in full_rows:
        template_text_id = (
            str(row["text_id"])
            if row["text_id"]
            in {
                "sentence_unko",
                "sentence_chinko",
                "sentence_manko",
                "control",
            }
            else "sentence_unko"
        )
        original_case_id = original_case_by_triplet[template_text_id, row["seed"], row["style"]]
        full_metrics.append(
            {
                **{field: row.get(field) for field in module.metrics.IDENTITY_FIELDS},
                "generation_results_sha256": generation_sha,
                "metrics_status": "COMPLETE",
                "speaker_similarity": similarity_by_case[original_case_id],
                "normalized_cer": 0.0,
                "transcript": "固定検証文",
            }
        )
    _write_jsonl(metric_path, full_metrics)

    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["generation_results"] = generation_sha
    provenance["input_sha256"]["generated_audio"] = {
        str(row["wav_path"]): row["wav_sha256"] for row in full_rows
    }
    provenance["case_count"] = 140
    provenance["complete_count"] = 140
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")


def test_full_generation_matrix_requires_explicit_checkpoint_step(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _convert_to_full_generation_matrix(fixture)

    with pytest.raises(ValueError, match="requires --checkpoint-step"):
        _run(fixture)


def test_full_generation_matrix_selects_explicit_checkpoint_step(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _convert_to_full_generation_matrix(fixture)

    report = _run(fixture, checkpoint_step=1500)

    assert report["summary"]["checkpoint_run_identity"]["checkpoint_step"] == 1500
    assert report["generation_selection"] == {
        "mode": "full_matrix_explicit_checkpoint",
        "requested_checkpoint_step": 1500,
        "selected_checkpoint_step": 1500,
        "input_case_count": 140,
        "selected_case_count": 28,
        "metric_gate_case_count": 16,
    }


def test_full_generation_matrix_rejects_explicit_null_schema_version(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _convert_to_full_generation_matrix(fixture)
    rows = [
        json.loads(line)
        for line in fixture["generation_results"].read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["schema_version"] = None

    with pytest.raises(ValueError, match="mixed or unsupported case schemas"):
        fixture["module"]._validate_generation_matrix(
            rows,
            model_id=MODEL_ID,
            checkpoint_step=1500,
        )


def test_full_generation_matrix_rejects_non_gate_checkpoint_identity_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _convert_to_full_generation_matrix(fixture)
    rows = [
        json.loads(line)
        for line in fixture["generation_results"].read_text(encoding="utf-8").splitlines()
    ]
    drift = next(
        row for row in rows if row["checkpoint_step"] == 1500 and row["text_id"] == "word_unko"
    )
    drift["embedding_sha256"] = "9" * 64
    drift["provenance"]["run_id"] = "drift-run"

    with pytest.raises(ValueError, match="checkpoint/run identity mismatch"):
        fixture["module"]._validate_generation_matrix(
            rows,
            model_id=MODEL_ID,
            checkpoint_step=1500,
        )


def test_audit_reports_reference_identity_centroid_stability_and_generated_deltas(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    report = _run(fixture)

    assert report["schema_version"] == "speaker-reference-centroid-audit/v1"
    assert report["model_identity"] == {
        "model_id": MODEL_ID,
        "model_prefix": MODEL_PREFIX,
        "clean_manifest_row_count": 25,
        "reference_count": 25,
        "source_identity_consistent": True,
    }
    assert report["summary"]["identity_consistent"] is True
    assert report["identity_verification"]["clean_manifest_join_fields"] == [
        "source_id",
        "audio_sha256",
        "pcm_sha256",
        "text",
    ]
    assert report["identity_verification"]["reference_source_audio_current_hash_verified"]
    assert report["identity_verification"]["reference_wav_path_and_sha256_unique"]
    assert report["identity_verification"]["source_audio_and_pcm_identity_unique"]
    assert report["identity_verification"]["duration_quantile_contract_verified"]
    assert report["summary"]["centroid_stable"] is (
        report["summary"]["reference_outlier_count"] == 0
    )
    assert report["reference_analysis"]["pairwise_similarity"]["count"] == 300
    assert len(report["reference_analysis"]["per_reference"]) == 25
    assert report["reference_analysis"]["per_reference"][0]["source_audio_sha256"]
    assert Path(
        report["reference_analysis"]["per_reference"][0]["source_audio_path"],
    ).is_absolute()
    assert report["reference_analysis"]["per_reference"][0]["pcm_sha256"]
    assert len(report["reference_analysis"]["leave_one_out"]) == 25
    assert report["reference_analysis"]["outlier_rule"]["method"] == "median_minus_3_mad"
    assert (
        "strictly below median" in report["reference_analysis"]["outlier_rule"]["mad_zero_behavior"]
    )
    assert report["reference_analysis"]["text_split"]["gate_effect"] == "diagnostic_only"
    assert report["reference_analysis"]["text_split"]["expressive_nonverbal_count"] == 1
    assert len(report["reference_analysis"]["duration_quantile_centroids"]) == 5
    assert len(report["reference_analysis"]["duration_quantile_mutual_similarities"]) == 10
    assert report["generated_analysis"]["metric_gate_case_count"] == 16
    assert report["generated_analysis"]["metrics_similarity_verified"] is True
    gate_identity = report["generated_analysis"]["checkpoint_run_identity"]
    assert gate_identity["checkpoint_step"] == 250
    assert gate_identity["run_id"] == "fixture-run"
    assert len(report["generated_analysis"]["cases"]) == 16
    assert all(
        case["checkpoint_run_identity"] == gate_identity
        for case in report["generated_analysis"]["cases"]
    )
    assert Path(report["generated_analysis"]["cases"][0]["wav_path"]).is_absolute()
    assert report["generated_analysis"]["cases"][0]["wav_sha256"]
    assert len(report["generated_analysis"]["cases"][0]["leave_one_out"]) == 25
    assert report["provenance"]["inputs"]["clean_manifest"]["sha256"] == _sha(
        Path(fixture["clean_manifest"]),
    )
    assert report["provenance"]["script"]["sha256"] == _sha(SCRIPT_PATH)
    assert report["provenance"]["publication_reverification"] == {
        "status": "passed_before_atomic_publication",
        "explicit_input_count": 5,
        "script_count": 2,
        "bound_audio_count": 66,
        "ecapa_tree_verified": True,
    }


def test_model_identity_accepts_explicit_alias_for_one_consistent_source_prefix() -> None:
    module = _load_script()
    clean_rows = [
        {"source_id": "oop55_aikagi_3_sp_683c9895cc:00000001"},
        {"source_id": "oop55_aikagi_3_sp_683c9895cc:00000002"},
    ]
    reference_rows = [
        {"source_id": "oop55_aikagi_3_sp_683c9895cc:00000001"},
    ]

    model_prefix = module._validate_model_identity(
        model_id="miu",
        clean_rows=clean_rows,
        reference_rows=reference_rows,
    )

    assert model_prefix == "oop55_aikagi_3_sp_683c9895cc"


def test_audit_rejects_reference_source_identity_mismatch_and_duplicate_clean_source(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    reference_path = Path(fixture["reference_manifest"])
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["references"][0]["text"] = "改変"
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"source identity mismatch.*text"):
        _run(fixture)

    fixture = _fixture(tmp_path / "duplicate")
    clean_path = Path(fixture["clean_manifest"])
    first = clean_path.read_text(encoding="utf-8").splitlines()[0]
    with clean_path.open("a", encoding="utf-8") as stream:
        stream.write(first + "\n")
    reference_path = Path(fixture["reference_manifest"])
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["source_hashes"]["training_clean_manifest_sha256"] = _sha(clean_path)
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate clean manifest source_id"):
        _run(fixture)


def test_audit_rejects_metrics_similarity_and_provenance_identity_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    metrics_path = Path(fixture["metrics_results"])
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["speaker_similarity"] -= 0.01
    _write_jsonl(metrics_path, rows)

    with pytest.raises(ValueError, match="speaker_similarity does not match recomputed"):
        _run(fixture)

    fixture = _fixture(tmp_path / "provenance")
    provenance_path = Path(fixture["metrics_provenance"])
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["models"]["speaker_embedding"]["revision"] = "wrong"
    provenance_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ECAPA revision"):
        _run(fixture)


def test_create_only_output_rejects_existing_file_and_symlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(fixture)

    fixture = _fixture(tmp_path / "symlink")
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    output = Path(fixture["output"])
    output.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        _run(fixture)
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert not list(output.parent.glob("*.tmp"))


def test_parse_args_requires_no_whisper_options() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--reference-wavs",
            "references.json",
            "--clean-manifest",
            "clean.jsonl",
            "--generation-results",
            "generation.jsonl",
            "--checkpoint-step",
            "1500",
            "--metrics-results",
            "metrics.jsonl",
            "--metrics-provenance",
            "metrics.provenance.json",
            "--output",
            "audit.json",
            "--ecapa-source",
            "ecapa",
            "--ecapa-savedir",
            "ecapa-cache",
            "--ecapa-model-id",
            ECAPA_MODEL_ID,
            "--ecapa-revision",
            ECAPA_REVISION,
        ],
    )

    assert args.ecapa_model_id == ECAPA_MODEL_ID
    assert args.checkpoint_step == 1500
    assert not hasattr(args, "whisper_model")


def test_audit_resolves_reference_paths_relative_to_reference_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reference_path = Path(fixture["reference_manifest"])
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    for row in payload["references"]:
        row["audio_path"] = str(Path(row["audio_path"]).relative_to(tmp_path))
        row["reference_wav_path"] = str(
            Path(row["reference_wav_path"]).relative_to(tmp_path),
        )
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = Path(fixture["metrics_provenance"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = _run(fixture)

    assert report["summary"]["identity_consistent"] is True


def test_audit_resolves_generated_paths_relative_to_generation_results(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    generation_path = Path(fixture["generation_results"])
    generation = [
        json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()
    ]
    relative_by_case = {}
    for row in generation:
        relative = str(Path(row["wav_path"]).relative_to(tmp_path))
        row["wav_path"] = relative
        relative_by_case[row["case_id"]] = relative
    _write_jsonl(generation_path, generation)

    metrics_path = Path(fixture["metrics_results"])
    metric_rows = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in metric_rows:
        row["wav_path"] = relative_by_case[row["case_id"]]
        row["generation_results_sha256"] = _sha(generation_path)
    _write_jsonl(metrics_path, metric_rows)
    provenance_path = Path(fixture["metrics_provenance"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["generation_results"] = _sha(generation_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = _run(fixture)

    assert report["generated_analysis"]["metric_gate_case_count"] == 16


def test_healthy_population_may_be_a_verified_subset_of_clean_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["healthy_population_count"] = 24
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="healthy_population_count"):
        _run(fixture)

    fixture = _fixture(tmp_path / "valid-subset")
    clean_path = fixture["clean_manifest"]
    with clean_path.open("a", encoding="utf-8") as stream:
        for index in range(25, 30):
            stream.write(
                json.dumps(
                    {
                        "source_id": f"{MODEL_PREFIX}:{index:08d}",
                        "audio_sha256": hashlib.sha256(f"audio-{index}".encode()).hexdigest(),
                        "pcm_sha256": hashlib.sha256(f"pcm-{index}".encode()).hexdigest(),
                        "text": f"追加音声{index}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["healthy_population_count"] = 30
    payload["source_hashes"]["training_clean_manifest_sha256"] = _sha(clean_path)
    for quantile in payload["quantiles"]:
        quantile["population_count"] = 6
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = _run(fixture)

    assert report["summary"]["identity_consistent"] is True


def test_optional_clean_audio_path_must_match_reference_source_audio(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    clean_path = fixture["clean_manifest"]
    clean_rows = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines()]
    clean_rows[0]["audio_path"] = str(tmp_path / "source/source-01.wav")
    _write_jsonl(clean_path, clean_rows)
    reference_path = fixture["reference_manifest"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["source_hashes"]["training_clean_manifest_sha256"] = _sha(clean_path)
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="clean audio_path does not match"):
        _run(fixture)


def test_manifest_replacement_during_first_generated_embedding_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    original_sha = _sha(fixture["reference_manifest"])
    fixture["embedder"] = _MutatingEmbedder(
        fixture["embedder"].source_sha256,
        target=fixture["reference_manifest"],
        replacement=b"{}\n",
    )

    with pytest.raises(ValueError, match="reference_wavs changed during audit"):
        _run(fixture)

    assert original_sha != _sha(fixture["reference_manifest"])
    assert not fixture["output"].exists()


def test_output_rejects_symlink_in_existing_lexical_parent_chain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    real_parent = tmp_path / "real-parent"
    child = real_parent / "existing-child"
    child.mkdir(parents=True)
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    fixture["output"] = alias / "existing-child" / "audit.json"

    with pytest.raises(ValueError, match=r"symbolic link|reparse point"):
        _run(fixture)

    assert not (child / "audit.json").exists()


@pytest.mark.parametrize("drift", ["checkpoint_step", "embedding", "run_id"])
def test_metric_gate_rejects_checkpoint_or_run_identity_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    fixture = _fixture(tmp_path)
    generation_path = fixture["generation_results"]
    generation = [
        json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()
    ]
    metrics_path = fixture["metrics_results"]
    metric_rows = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    if drift == "checkpoint_step":
        generation[0]["checkpoint_step"] = 500
        metric_rows[0]["checkpoint_step"] = 500
    elif drift == "embedding":
        generation[0]["embedding_path"] = str(tmp_path / "drift.safetensors")
        generation[0]["embedding_sha256"] = "9" * 64
        metric_rows[0]["embedding_path"] = generation[0]["embedding_path"]
        metric_rows[0]["embedding_sha256"] = generation[0]["embedding_sha256"]
    else:
        generation[0]["provenance"]["run_id"] = "drift-run"
        metric_rows[0]["provenance"]["run_id"] = "drift-run"
    _write_jsonl(generation_path, generation)
    generation_sha = _sha(generation_path)
    for row in metric_rows:
        row["generation_results_sha256"] = generation_sha
    _write_jsonl(metrics_path, metric_rows)
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["generation_results"] = generation_sha
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="selected generation checkpoint/run identity mismatch"):
        _run(fixture)


def test_reference_manifest_rejects_duplicate_resolved_reference_wav(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["references"][1]["reference_wav_path"] = payload["references"][0]["reference_wav_path"]
    payload["references"][1]["reference_wav_sha256"] = payload["references"][0][
        "reference_wav_sha256"
    ]
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate resolved reference WAV"):
        _run(fixture)


def test_reference_manifest_rejects_duplicate_reference_wav_hash_at_distinct_path(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    original_second = Path(payload["references"][1]["reference_wav_path"])
    duplicate_path = tmp_path / "references/duplicate-bytes.wav"
    duplicate_path.write_bytes(
        Path(payload["references"][0]["reference_wav_path"]).read_bytes(),
    )
    payload["references"][1]["reference_wav_path"] = str(duplicate_path)
    payload["references"][1]["reference_wav_sha256"] = payload["references"][0][
        "reference_wav_sha256"
    ]
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    del provenance["input_sha256"]["reference_audio"][str(original_second)]
    provenance["input_sha256"]["reference_audio"][str(duplicate_path)] = _sha(
        duplicate_path,
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate reference_wav_sha256"):
        _run(fixture)


@pytest.mark.parametrize(
    ("identity_kind", "expected_message"),
    [
        ("source_audio", "duplicate resolved source audio"),
        ("pcm_sha256", "duplicate pcm_sha256"),
    ],
)
def test_reference_manifest_rejects_duplicate_source_audio_or_pcm_identity(
    tmp_path: Path,
    identity_kind: str,
    expected_message: str,
) -> None:
    fixture = _fixture(tmp_path)
    clean_path = fixture["clean_manifest"]
    clean_rows = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines()]
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if identity_kind == "source_audio":
        payload["references"][1]["audio_path"] = payload["references"][0]["audio_path"]
        payload["references"][1]["audio_sha256"] = payload["references"][0]["audio_sha256"]
        clean_rows[1]["audio_sha256"] = clean_rows[0]["audio_sha256"]
    else:
        payload["references"][1]["pcm_sha256"] = payload["references"][0]["pcm_sha256"]
        clean_rows[1]["pcm_sha256"] = clean_rows[0]["pcm_sha256"]
    _write_jsonl(clean_path, clean_rows)
    payload["source_hashes"]["training_clean_manifest_sha256"] = _sha(clean_path)
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        _run(fixture)


def test_reference_manifest_rejects_duration_quantile_swap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["references"][0]["duration_quantile"] = 5
    payload["references"][20]["duration_quantile"] = 1
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="outside declared duration range"):
        _run(fixture)


def test_reference_manifest_allows_equal_adjacent_quantile_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["quantiles"][1]["population_max_seconds"] = payload["quantiles"][2][
        "population_min_seconds"
    ]
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = _run(fixture)

    assert report["identity_verification"]["duration_quantile_contract_verified"]


def test_reference_manifest_rejects_true_adjacent_quantile_overlap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_manifest"]
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["quantiles"][1]["population_max_seconds"] = (
        payload["quantiles"][2]["population_min_seconds"] + 0.001
    )
    reference_path.write_text(json.dumps(payload), encoding="utf-8")
    provenance_path = fixture["metrics_provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["reference_wavs"] = _sha(reference_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="duration ranges overlap or are unordered"):
        _run(fixture)


def test_robust_outlier_rule_uses_strict_median_minus_three_mad() -> None:
    module = _load_script()

    threshold, flags = module._robust_outliers([0.4, 0.7, 0.8, 0.9, 1.0])
    assert threshold == pytest.approx(0.5)
    assert flags == (True, False, False, False, False)

    zero_mad_threshold, zero_mad_flags = module._robust_outliers([0.0, 1.0, 1.0, 1.0, 2.0])
    assert zero_mad_threshold == pytest.approx(1.0)
    assert zero_mad_flags == (True, False, False, False, False)
