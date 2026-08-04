from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SCHEMA_VERSION = "speaker-checkpoint-evaluation/v1"
CONFIG_SCHEMA_VERSION = "speaker-checkpoint-thresholds/v1"
MANIFEST_SCHEMA_VERSION = "speaker-checkpoint-evaluation-manifest/v1"
METRICS_PROVENANCE_SCHEMA_VERSION = "speaker-metrics-extraction/v1"
VERIFICATION_SCHEMA_VERSION = "speaker-checkpoint-evaluation-verification/v2"
PCM16_SAMPLE_WIDTH = 2
PCM16_SCALE = 32_768.0
MIN_STABILITY_SAMPLES = 2
SHA256_HEX_LENGTH = 64
REQUIRED_CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
REQUIRED_TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
REQUIRED_SEEDS = (1234, 5678)
REQUIRED_STYLES = ("neutral", "calm")
METRIC_GATE_TEXT_IDS = frozenset(
    {
        "sentence_unko",
        "sentence_chinko",
        "sentence_manko",
        "control",
    },
)
REQUIRED_HARD_GATE_METRIC_CASE_COUNT = 16
EXPECTED_EMBEDDING_SHAPE = (16, 768)
SAFETENSORS_HEADER_LENGTH_BYTES = 8
SAFETENSORS_OFFSET_COUNT = 2
IDENTITY_FIELDS = (
    "model_id",
    "checkpoint_step",
    "checkpoint",
    "speaker_filename",
    "embedding_path",
    "embedding_sha256",
    "evaluation_manifest_sha256",
    "base_checkpoint_sha256",
    "text_id",
    "seed",
    "style",
    "wav_path",
    "wav_sha256",
    "provenance",
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    min_duration_seconds: float = 0.2
    max_duration_seconds: float = 20.0
    min_rms: float = 0.005
    silence_amplitude: float = 0.001
    max_silence_ratio: float = 0.98
    clipping_amplitude: float = 32_767.0 / PCM16_SCALE
    max_clipping_ratio: float = 0.01
    min_speaker_similarity: float = 0.75
    max_normalized_cer: float = 0.20
    max_style_similarity_drop: float = 0.08
    min_style_pair_count: int = 1
    min_style_contrast: float = 0.01

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "thresholds": asdict(self),
        }


@dataclass(frozen=True, slots=True)
class AudioStats:
    duration_seconds: float
    sample_rate: int
    sample_count: int
    finite_samples: bool
    rms: float
    silence_ratio: float
    clipping_ratio: float


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    model_id: str
    checkpoint_step: int
    embedding_path: Path
    embedding_sha256: str
    training_config_sha256: str
    base_checkpoint: str
    base_checkpoint_sha256: str
    base_revision: str
    run_id: str
    evaluation_manifest_sha256: str

    def selection(self, *, rank: int) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "checkpoint_step": self.checkpoint_step,
            "rank": rank,
            "embedding_path": str(self.embedding_path),
            "embedding_sha256": self.embedding_sha256,
            "training_config_sha256": self.training_config_sha256,
            "base_checkpoint": self.base_checkpoint,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_revision": self.base_revision,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    checkpoints: dict[tuple[str, int], CheckpointContract]
    text_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    styles: tuple[str, ...]
    reference_wavs_sha256: str
    speaker_embedding_model_id: str
    speaker_embedding_revision: str
    speaker_embedding_source_sha256: str
    transcription_model_id: str
    transcription_revision: str
    transcription_source_sha256: str


DEFAULT_CONFIG = EvaluationConfig()


def analyze_wav(path: Path, config: EvaluationConfig = DEFAULT_CONFIG) -> AudioStats:
    with wave.open(str(path), "rb") as reader:
        if reader.getcomptype() != "NONE":
            message = f"compressed WAV is unsupported: {path}"
            raise ValueError(message)
        if reader.getsampwidth() != PCM16_SAMPLE_WIDTH:
            message = f"WAV must use PCM16 samples: {path}"
            raise ValueError(message)
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        frames = reader.readframes(frame_count)
    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        message = f"WAV has invalid audio dimensions: {path}"
        raise ValueError(message)

    pcm = np.frombuffer(frames, dtype="<i2")
    if pcm.size != frame_count * channels:
        message = f"WAV payload is truncated: {path}"
        raise ValueError(message)
    samples = pcm.astype(np.float64).reshape(-1, channels).mean(axis=1) / PCM16_SCALE
    finite_samples = bool(np.all(np.isfinite(samples)))
    rms = float(np.sqrt(np.mean(samples**2))) if finite_samples else math.nan
    return AudioStats(
        duration_seconds=round(frame_count / sample_rate, 6),
        sample_rate=sample_rate,
        sample_count=frame_count,
        finite_samples=finite_samples,
        rms=round(rms, 8),
        silence_ratio=round(float(np.mean(np.abs(samples) <= config.silence_amplitude)), 8),
        clipping_ratio=round(float(np.mean(np.abs(samples) >= config.clipping_amplitude)), 8),
    )


def main(  # noqa: PLR0914 - orchestration keeps all verified artifact inputs explicit.
    argv: Sequence[str] | None = None,
) -> int:
    args = _parse_args(argv)
    verification_path = args.output_dir / "evaluation-verification.json"
    verification_path.unlink(missing_ok=True)
    manifest = _load_evaluation_manifest(args.evaluation_manifest)
    input_paths = {
        "generation_results": args.generation_results,
        "analysis_results": args.analysis_results,
        "evaluation_manifest": args.evaluation_manifest,
        "metrics_results": args.metrics_results,
        "metrics_provenance": args.metrics_provenance,
    }
    input_sha256 = {name: _sha256(path) for name, path in input_paths.items()}

    generation_rows = _read_jsonl(args.generation_results)
    analysis_rows = _read_jsonl(args.analysis_results)
    metric_rows = _read_jsonl(args.metrics_results)
    generation = _index_rows(generation_rows, source="generation")
    analysis = _index_rows(analysis_rows, source="analysis")
    metrics = _index_rows(metric_rows, source="metrics")
    _validate_inputs(generation, analysis, metrics)
    _validate_expected_matrix(generation, manifest=manifest)
    _validate_checkpoint_contract(generation, manifest=manifest)
    metrics_provenance = _read_json(args.metrics_provenance)
    _validate_metrics_provenance(
        metrics_provenance,
        generation_results_sha256=input_sha256["generation_results"],
        manifest=manifest,
    )
    _validate_metric_rows(
        metrics,
        generation_results_sha256=input_sha256["generation_results"],
    )
    _validate_audio_bindings(
        generation,
        analysis,
        metrics_provenance=metrics_provenance,
        generation_base=args.generation_results.parent,
        analysis_base=args.analysis_results.parent,
        provenance_base=args.metrics_provenance.parent,
    )

    evaluations = [
        _evaluate_case(
            generation[case_id],
            analysis[case_id],
            metrics.get(case_id),
            generation_base=args.generation_results.parent,
            config=DEFAULT_CONFIG,
        )
        for case_id in sorted(
            generation,
            key=lambda value: _case_sort_key(generation[value]),
        )
    ]
    summaries = _summarize_checkpoints(evaluations, input_sha256=input_sha256)
    _assign_ranks(summaries)
    selections = [_selection(row, manifest=manifest) for row in summaries if row["rank"] == 1]

    review_rows = [
        _review_row(row, evaluations=evaluations)
        for row in evaluations
        if row["evaluation_status"] != "PASS"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "evaluation-results.jsonl", evaluations)
    _write_jsonl(args.output_dir / "checkpoint-summary.jsonl", summaries)
    _write_jsonl(
        args.output_dir / "review-candidates.jsonl",
        review_rows,
    )
    _write_review_packet(
        args.output_dir / "review_packet",
        review_rows,
        generation_base=args.generation_results.parent,
        analysis_base=args.analysis_results.parent,
    )
    _write_json(args.output_dir / "evaluation-config.json", DEFAULT_CONFIG.to_dict())
    selected_document = {
        "schema_version": SCHEMA_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "selections": selections,
    }
    _write_json(args.output_dir / "selected-models.json", selected_document)
    model_ids = {str(row["model_id"]) for row in summaries}
    passed = len(model_ids) == 1 and len(selections) == 1
    packet_counts = _review_packet_counts(args.output_dir / "review_packet")
    _write_json_atomic(
        verification_path,
        {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "status": "PASS" if passed else "FAIL",
            "selected": selections[0] if passed else None,
            "input_sha256": input_sha256,
            "artifact_sha256": _evaluation_artifact_sha256(args.output_dir),
            "checkpoint_count": len(summaries),
            "evaluation_case_count": len(evaluations),
            "hard_gate_metric_case_count_per_checkpoint": (REQUIRED_HARD_GATE_METRIC_CASE_COUNT),
            "diagnostic_word_case_count_per_checkpoint": (
                len(set(manifest.text_ids) - METRIC_GATE_TEXT_IDS)
                * len(manifest.seeds)
                * len(manifest.styles)
            ),
            "eligible_steps": sorted(
                {
                    _required_int(row, "checkpoint_step")
                    for row in summaries
                    if row["status"] == "ELIGIBLE"
                },
            ),
            "rejected_tone_steps": sorted(
                {
                    _required_int(row, "checkpoint_step")
                    for row in summaries
                    if _has_reason(row, "rejection_reasons", "tone_candidate")
                },
            ),
            "review_candidate_count": len(review_rows),
            **packet_counts,
            "evaluator_script_sha256": _sha256(Path(__file__).resolve()),
            "metrics_results_sha256": input_sha256["metrics_results"],
            "metrics_provenance_sha256": input_sha256["metrics_provenance"],
        },
    )
    print(
        f"evaluation complete: {len(evaluations)} cases, "
        f"{len(summaries)} checkpoints, {len(selections)} selected",
    )
    return 0 if passed else 1


def _evaluate_case(
    generation: Mapping[str, object],
    analysis: Mapping[str, object],
    metrics: Mapping[str, object] | None,
    *,
    generation_base: Path,
    config: EvaluationConfig,
) -> dict[str, object]:
    rejection_reasons: list[str] = []
    incomplete_reasons: list[str] = []
    review_reasons: list[str] = []
    diagnostic_flags: list[str] = []
    audio: AudioStats | None = None
    metric_gate_applied = generation.get("text_id") in METRIC_GATE_TEXT_IDS

    if generation.get("status") != "SUCCESS":
        rejection_reasons.append("generation_error")
    analysis_status = analysis.get("analysis_status")
    analysis_rejections, analysis_reviews = _analysis_reasons(analysis_status)
    rejection_reasons.extend(analysis_rejections)
    review_reasons.extend(analysis_reviews)

    if generation.get("status") == "SUCCESS":
        try:
            wav_path = _resolve_wav_path(generation, base=generation_base)
            audio = analyze_wav(wav_path, config)
            rejection_reasons.extend(_audio_rejection_reasons(audio, config=config))
        except (OSError, ValueError, wave.Error):
            rejection_reasons.append("invalid_audio")

    metrics_complete = metrics is not None and metrics.get("metrics_status") == "COMPLETE"
    if metrics is not None and not metrics_complete:
        (incomplete_reasons if metric_gate_applied else diagnostic_flags).append(
            "metrics_incomplete",
        )
    similarity = _metric(metrics, "speaker_similarity") if metrics_complete else None
    cer = _metric(metrics, "normalized_cer") if metrics_complete else None
    if similarity is None:
        (incomplete_reasons if metric_gate_applied else diagnostic_flags).append(
            "missing_speaker_similarity",
        )
    elif similarity < config.min_speaker_similarity:
        (rejection_reasons if metric_gate_applied else diagnostic_flags).append(
            "low_speaker_similarity",
        )
    if cer is None:
        (incomplete_reasons if metric_gate_applied else diagnostic_flags).append(
            "missing_normalized_cer",
        )
    elif cer > config.max_normalized_cer:
        (rejection_reasons if metric_gate_applied else diagnostic_flags).append(
            "high_normalized_cer",
        )

    status = _case_status(rejection_reasons, incomplete_reasons, review_reasons)
    return {
        **generation,
        "evaluation_schema_version": SCHEMA_VERSION,
        "evaluation_status": status,
        "audio": asdict(audio) if audio is not None else None,
        "speaker_similarity": similarity,
        "normalized_cer": cer,
        "metric_gate_applied": metric_gate_applied,
        "diagnostic_flags": sorted(set(diagnostic_flags)),
        "rejection_reasons": sorted(set(rejection_reasons)),
        "incomplete_reasons": sorted(set(incomplete_reasons)),
        "review_reasons": sorted(set(review_reasons)),
        "tone_evidence": {
            "analysis_status": analysis_status,
            "intervals": analysis.get("intervals", []),
            "spectrogram_path": analysis.get("spectrogram_path"),
            "spectrogram_error": analysis.get("spectrogram_error"),
            "detector_config": analysis.get("detector_config"),
        },
    }


def _analysis_reasons(status: object) -> tuple[list[str], list[str]]:
    if status == "ERROR":
        return ["analysis_error"], []
    if status == "CANDIDATE":
        return ["tone_candidate"], []
    if status == "AMBIGUOUS":
        return [], ["tone_ambiguous"]
    if status != "CLEAR":
        return ["invalid_analysis_status"], []
    return [], []


def _audio_rejection_reasons(
    audio: AudioStats,
    *,
    config: EvaluationConfig,
) -> list[str]:
    reasons: list[str] = []
    if not audio.finite_samples:
        reasons.append("nonfinite_audio")
    if not config.min_duration_seconds <= audio.duration_seconds <= config.max_duration_seconds:
        reasons.append("extreme_duration")
    if audio.rms < config.min_rms or audio.silence_ratio > config.max_silence_ratio:
        reasons.append("near_silent_audio")
    if audio.clipping_ratio > config.max_clipping_ratio:
        reasons.append("clipped_audio")
    return reasons


def _summarize_checkpoints(
    evaluations: Sequence[dict[str, object]],
    *,
    input_sha256: Mapping[str, str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in evaluations:
        key = (str(row["model_id"]), _required_int(row, "checkpoint_step"))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (model_id, step), rows in sorted(grouped.items()):
        rejection_reasons = _collect_reasons(rows, "rejection_reasons")
        incomplete_reasons = _collect_reasons(rows, "incomplete_reasons")
        review_reasons = _collect_reasons(rows, "review_reasons")
        metric_gate_rows = [row for row in rows if row.get("metric_gate_applied") is True]
        hard_gate_metric_case_count = sum(
            isinstance(row.get("speaker_similarity"), float)
            and isinstance(row.get("normalized_cer"), float)
            for row in metric_gate_rows
        )
        if hard_gate_metric_case_count != REQUIRED_HARD_GATE_METRIC_CASE_COUNT:
            incomplete_reasons.append("hard_gate_metric_case_count")
        style_pairs = _style_pairs(rows)
        if len(style_pairs) < DEFAULT_CONFIG.min_style_pair_count:
            incomplete_reasons.append("missing_style_pair")
        comparable_style_pairs = [
            (neutral_similarity, calm_similarity)
            for _, _, neutral_similarity, calm_similarity in _style_pairs(metric_gate_rows)
            if neutral_similarity is not None and calm_similarity is not None
        ]
        if any(
            calm_similarity < neutral_similarity - DEFAULT_CONFIG.max_style_similarity_drop
            for neutral_similarity, calm_similarity in comparable_style_pairs
        ):
            rejection_reasons.append("style_similarity_not_preserved")
        style_contrast = _style_contrast(style_pairs)
        if style_pairs and style_contrast < DEFAULT_CONFIG.min_style_contrast:
            review_reasons.append("insufficient_style_contrast")
        status = _checkpoint_status(rejection_reasons, incomplete_reasons, review_reasons)
        similarities = _present_metric(metric_gate_rows, "speaker_similarity")
        cer_values = _present_metric(metric_gate_rows, "normalized_cer")
        summaries.append(
            {
                "evaluation_schema_version": SCHEMA_VERSION,
                "model_id": model_id,
                "checkpoint_step": step,
                "status": status,
                "rank": None,
                "case_count": len(rows),
                "hard_gate_metric_case_count": hard_gate_metric_case_count,
                "pass_count": sum(row["evaluation_status"] == "PASS" for row in rows),
                "stability_score": _stability_score(rows),
                "mean_normalized_cer": _rounded_mean(cer_values),
                "mean_speaker_similarity": _rounded_mean(similarities),
                "style_pair_count": len(style_pairs),
                "style_contrast": style_contrast,
                "rejection_reasons": sorted(set(rejection_reasons)),
                "incomplete_reasons": sorted(set(incomplete_reasons)),
                "review_reasons": sorted(set(review_reasons)),
                "input_sha256": dict(input_sha256),
            },
        )
    return summaries


def _style_pairs(
    rows: Sequence[dict[str, object]],
) -> list[tuple[AudioStats, AudioStats, float | None, float | None]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("text_id")), _required_int(row, "seed"))
        grouped.setdefault(key, {})[str(row.get("style"))] = row
    pairs = []
    for styles in grouped.values():
        neutral = styles.get("neutral")
        calm = styles.get("calm")
        if neutral is None or calm is None:
            continue
        neutral_audio = _audio_from_row(neutral)
        calm_audio = _audio_from_row(calm)
        neutral_similarity = neutral.get("speaker_similarity")
        calm_similarity = calm.get("speaker_similarity")
        if neutral_audio is not None and calm_audio is not None:
            pairs.append(
                (
                    neutral_audio,
                    calm_audio,
                    neutral_similarity if isinstance(neutral_similarity, float) else None,
                    calm_similarity if isinstance(calm_similarity, float) else None,
                ),
            )
    return pairs


def _style_contrast(
    pairs: Sequence[tuple[AudioStats, AudioStats, float | None, float | None]],
) -> float:
    values = [
        abs(math.log(calm.duration_seconds / neutral.duration_seconds))
        + abs(20.0 * math.log10(calm.rms / neutral.rms)) / 20.0
        for neutral, calm, _, _ in pairs
        if neutral.duration_seconds > 0.0 and neutral.rms > 0.0 and calm.rms > 0.0
    ]
    return _rounded_mean(values) or 0.0


def _stability_score(rows: Sequence[dict[str, object]]) -> float:
    groups: dict[tuple[str, str], list[AudioStats]] = {}
    for row in rows:
        audio = _audio_from_row(row)
        if audio is not None:
            key = (str(row.get("text_id")), str(row.get("style")))
            groups.setdefault(key, []).append(audio)
    variations: list[float] = []
    for audio_rows in groups.values():
        if len(audio_rows) < MIN_STABILITY_SAMPLES:
            continue
        durations = np.array([audio.duration_seconds for audio in audio_rows])
        rms_values = np.array([audio.rms for audio in audio_rows])
        variations.extend(
            (
                float(np.std(durations) / np.mean(durations)),
                float(np.std(rms_values) / np.mean(rms_values)),
            ),
        )
    return round(1.0 / (1.0 + (float(np.mean(variations)) if variations else 0.0)), 8)


def _assign_ranks(summaries: list[dict[str, object]]) -> None:
    model_ids = sorted({str(row["model_id"]) for row in summaries})
    for model_id in model_ids:
        eligible = [
            row for row in summaries if row["model_id"] == model_id and row["status"] == "ELIGIBLE"
        ]
        eligible.sort(key=_ranking_key)
        for rank, row in enumerate(eligible, start=1):
            row["rank"] = rank


def _ranking_key(row: Mapping[str, object]) -> tuple[float, float, float, float, int]:
    return (
        -_required_float(row, "stability_score"),
        _required_float(row, "mean_normalized_cer"),
        -_required_float(row, "mean_speaker_similarity"),
        -_required_float(row, "style_contrast"),
        _required_int(row, "checkpoint_step"),
    )


def _selection(
    row: Mapping[str, object],
    *,
    manifest: EvaluationManifest,
) -> dict[str, object]:
    key = (str(row["model_id"]), _required_int(row, "checkpoint_step"))
    return manifest.checkpoints[key].selection(rank=_required_int(row, "rank"))


def _load_evaluation_manifest(path: Path) -> EvaluationManifest:
    payload = _read_json(path)
    evaluation_manifest_sha256 = _sha256(path)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        message = f"evaluation manifest requires schema_version {MANIFEST_SCHEMA_VERSION}"
        raise ValueError(message)
    _validate_manifest_dimensions(payload)
    checkpoints: dict[tuple[str, int], CheckpointContract] = {}
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        message = "evaluation manifest models must be a list"
        raise TypeError(message)
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            message = "evaluation manifest model entries must be objects"
            raise TypeError(message)
        model_id = _required_string(raw_model, "model_id", source="evaluation manifest model")
        raw_checkpoints = raw_model.get("checkpoints")
        if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
            message = f"evaluation manifest checkpoints must be a nonempty list for {model_id}"
            raise ValueError(message)
        for raw_checkpoint in raw_checkpoints:
            if not isinstance(raw_checkpoint, dict):
                message = f"evaluation manifest checkpoint entries must be objects for {model_id}"
                raise TypeError(message)
            step = _required_manifest_int(raw_checkpoint, "checkpoint_step")
            key = (model_id, step)
            if key in checkpoints:
                message = f"duplicate evaluation manifest checkpoint: {model_id} step {step}"
                raise ValueError(message)
            raw_embedding_path = _required_string(
                raw_checkpoint,
                "embedding_path",
                source=f"evaluation manifest checkpoint {model_id} step {step}",
            )
            embedding_path = Path(raw_embedding_path)
            if not embedding_path.is_absolute():
                embedding_path = path.parent / embedding_path
            checkpoints[key] = CheckpointContract(
                model_id=model_id,
                checkpoint_step=step,
                embedding_path=embedding_path.resolve(),
                embedding_sha256=_required_sha256(raw_checkpoint, "embedding_sha256"),
                training_config_sha256=_required_sha256(
                    raw_checkpoint,
                    "training_config_sha256",
                ),
                base_checkpoint=_required_string(
                    raw_checkpoint,
                    "base_checkpoint",
                    source=f"evaluation manifest checkpoint {model_id} step {step}",
                ),
                base_checkpoint_sha256=_required_sha256(
                    raw_checkpoint,
                    "base_checkpoint_sha256",
                ),
                base_revision=_required_string(
                    raw_checkpoint,
                    "base_revision",
                    source=f"evaluation manifest checkpoint {model_id} step {step}",
                ),
                run_id=_required_string(
                    raw_checkpoint,
                    "run_id",
                    source=f"evaluation manifest checkpoint {model_id} step {step}",
                ),
                evaluation_manifest_sha256=evaluation_manifest_sha256,
            )
            _validate_embedding(checkpoints[key])
    metrics_provenance = payload.get("metrics_provenance")
    if not isinstance(metrics_provenance, dict):
        message = "evaluation manifest metrics_provenance must be an object"
        raise TypeError(message)
    speaker_embedding = metrics_provenance.get("speaker_embedding")
    transcription = metrics_provenance.get("transcription")
    if not isinstance(speaker_embedding, dict) or not isinstance(transcription, dict):
        message = "evaluation manifest metric models must be objects"
        raise TypeError(message)
    return EvaluationManifest(
        checkpoints=checkpoints,
        text_ids=_required_string_tuple(payload, "text_ids"),
        seeds=_required_int_tuple(payload, "seeds"),
        styles=_required_string_tuple(payload, "styles"),
        reference_wavs_sha256=_required_sha256(
            metrics_provenance,
            "reference_wavs_sha256",
        ),
        speaker_embedding_model_id=_required_string(
            speaker_embedding,
            "model_id",
            source="evaluation manifest speaker_embedding",
        ),
        speaker_embedding_revision=_required_string(
            speaker_embedding,
            "revision",
            source="evaluation manifest speaker_embedding",
        ),
        speaker_embedding_source_sha256=_required_sha256(
            speaker_embedding,
            "source_sha256",
        ),
        transcription_model_id=_required_string(
            transcription,
            "model_id",
            source="evaluation manifest transcription",
        ),
        transcription_revision=_required_string(
            transcription,
            "revision",
            source="evaluation manifest transcription",
        ),
        transcription_source_sha256=_required_sha256(
            transcription,
            "source_sha256",
        ),
    )


def _validate_manifest_dimensions(payload: Mapping[str, object]) -> None:
    expected_dimensions: tuple[tuple[str, tuple[object, ...]], ...] = (
        ("text_ids", REQUIRED_TEXT_IDS),
        ("seeds", REQUIRED_SEEDS),
        ("styles", REQUIRED_STYLES),
    )
    for field, expected in expected_dimensions:
        value = payload.get(field)
        if not isinstance(value, list) or tuple(value) != expected:
            message = f"evaluation manifest {field} must exactly match {expected}"
            raise ValueError(message)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            continue
        raw_checkpoints = raw_model.get("checkpoints")
        if not isinstance(raw_checkpoints, list):
            continue
        steps = tuple(
            checkpoint.get("checkpoint_step")
            for checkpoint in raw_checkpoints
            if isinstance(checkpoint, dict)
        )
        if steps != REQUIRED_CHECKPOINT_STEPS:
            message = (
                "evaluation manifest checkpoint steps must exactly match "
                f"{REQUIRED_CHECKPOINT_STEPS}"
            )
            raise ValueError(message)


def _validate_embedding(contract: CheckpointContract) -> None:
    path = contract.embedding_path
    if not path.is_file():
        message = f"evaluation embedding does not exist: {path}"
        raise ValueError(message)
    if _sha256(path) != contract.embedding_sha256:
        message = f"evaluation embedding SHA-256 mismatch: {path}"
        raise ValueError(message)
    try:
        with path.open("rb") as source:
            raw_length = source.read(SAFETENSORS_HEADER_LENGTH_BYTES)
            if len(raw_length) != SAFETENSORS_HEADER_LENGTH_BYTES:
                message = "safetensors header length is truncated"
                raise ValueError(message)
            header_length = struct.unpack("<Q", raw_length)[0]
            header: Any = json.loads(source.read(header_length))
            tensor = header.get("speaker_embedding") if isinstance(header, dict) else None
            if not isinstance(tensor, dict):
                message = "speaker_embedding tensor is missing"
                raise TypeError(message)
            if tensor.get("dtype") != "F32":
                message = "speaker_embedding must be F32"
                raise ValueError(message)
            if tensor.get("shape") != list(EXPECTED_EMBEDDING_SHAPE):
                message = f"speaker_embedding shape must be {EXPECTED_EMBEDDING_SHAPE}"
                raise ValueError(message)
            offsets = tensor.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != SAFETENSORS_OFFSET_COUNT:
                message = "speaker_embedding data_offsets are invalid"
                raise ValueError(message)
            start, end = offsets
            if not isinstance(start, int) or not isinstance(end, int):
                message = "speaker_embedding data_offsets are invalid"
                raise TypeError(message)
            source.seek(SAFETENSORS_HEADER_LENGTH_BYTES + header_length + start)
            payload = source.read(end - start)
            values = np.frombuffer(payload, dtype="<f4")
            if values.size != math.prod(EXPECTED_EMBEDDING_SHAPE):
                message = "speaker_embedding payload size is invalid"
                raise ValueError(message)
            if not np.isfinite(values).all():
                message = "speaker_embedding must contain only finite values"
                raise ValueError(message)
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        message = f"invalid evaluation embedding {path}: {exc}"
        raise ValueError(message) from exc


def _validate_expected_matrix(
    generation: Mapping[str, Mapping[str, object]],
    *,
    manifest: EvaluationManifest,
) -> None:
    expected = {
        (model_id, step, text_id, seed, style)
        for model_id, step in manifest.checkpoints
        for text_id in manifest.text_ids
        for seed in manifest.seeds
        for style in manifest.styles
    }
    actual: dict[tuple[str, int, str, int, str], list[str]] = {}
    for case_id, row in generation.items():
        key = (
            _required_string(row, "model_id", source=f"generation row {case_id}"),
            _required_int(row, "checkpoint_step"),
            _required_string(row, "text_id", source=f"generation row {case_id}"),
            _required_int(row, "seed"),
            _required_string(row, "style", source=f"generation row {case_id}"),
        )
        actual.setdefault(key, []).append(case_id)
    errors = [
        f"duplicate expected matrix case: {key} ({', '.join(case_ids)})"
        for key, case_ids in sorted(actual.items())
        if len(case_ids) > 1
    ]
    missing = sorted(expected - actual.keys())
    extra = sorted(actual.keys() - expected)
    if missing:
        errors.append(f"generation missing expected matrix case: {missing[0]}")
    if extra:
        errors.append(f"generation has unexpected matrix case: {extra[0]}")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_checkpoint_contract(
    generation: Mapping[str, Mapping[str, object]],
    *,
    manifest: EvaluationManifest,
) -> None:
    errors: list[str] = []
    for case_id, row in generation.items():
        key = (str(row.get("model_id")), _required_int(row, "checkpoint_step"))
        contract = manifest.checkpoints.get(key)
        if contract is None:
            continue
        if row.get("speaker_filename") != contract.embedding_path.name:
            errors.append(f"{case_id}: speaker_filename does not match evaluation manifest")
        if row.get("checkpoint") != contract.base_checkpoint:
            errors.append(f"{case_id}: checkpoint does not match evaluation manifest")
        expected_identity = {
            "embedding_path": str(contract.embedding_path),
            "embedding_sha256": contract.embedding_sha256,
            "evaluation_manifest_sha256": contract.evaluation_manifest_sha256,
            "base_checkpoint_sha256": contract.base_checkpoint_sha256,
        }
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                errors.append(f"{case_id}: {field} does not match evaluation manifest")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{case_id}: generation provenance must be an object")
            continue
        expected_provenance = {
            "training_config_sha256": contract.training_config_sha256,
            "base_checkpoint": contract.base_checkpoint,
            "base_revision": contract.base_revision,
            "run_id": contract.run_id,
        }
        for field, expected in expected_provenance.items():
            if provenance.get(field) != expected:
                errors.append(f"{case_id}: provenance {field} does not match evaluation manifest")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_metrics_provenance(
    payload: Mapping[str, object],
    *,
    generation_results_sha256: str,
    manifest: EvaluationManifest,
) -> None:
    errors: list[str] = []
    if payload.get("schema_version") != METRICS_PROVENANCE_SCHEMA_VERSION:
        errors.append(
            f"metrics provenance schema_version must be {METRICS_PROVENANCE_SCHEMA_VERSION}"
        )
    input_sha256 = payload.get("input_sha256")
    models = payload.get("models")
    if not isinstance(input_sha256, dict):
        errors.append("metrics provenance input_sha256 must be an object")
        input_sha256 = {}
    if not isinstance(models, dict):
        errors.append("metrics provenance models must be an object")
        models = {}
    if input_sha256.get("generation_results") != generation_results_sha256:
        errors.append("metrics provenance generation_results SHA does not match input")
    if input_sha256.get("reference_wavs") != manifest.reference_wavs_sha256:
        errors.append("metrics provenance reference_wavs SHA does not match evaluation manifest")
    expected_models = {
        "speaker_embedding": (
            manifest.speaker_embedding_model_id,
            manifest.speaker_embedding_revision,
            manifest.speaker_embedding_source_sha256,
        ),
        "transcription": (
            manifest.transcription_model_id,
            manifest.transcription_revision,
            manifest.transcription_source_sha256,
        ),
    }
    for name, (
        expected_model_id,
        expected_revision,
        expected_source_sha256,
    ) in expected_models.items():
        model = models.get(name)
        if not isinstance(model, dict):
            errors.append(f"metrics provenance {name} must be an object")
            continue
        if model.get("model_id") != expected_model_id:
            errors.append(f"metrics provenance {name} model_id does not match evaluation manifest")
        if model.get("revision") != expected_revision:
            errors.append(f"metrics provenance {name} revision does not match evaluation manifest")
        if model.get("source_sha256") != expected_source_sha256:
            errors.append(
                f"metrics provenance {name} source_sha256 does not match evaluation manifest"
            )
    if errors:
        raise ValueError("; ".join(errors))


def _validate_audio_bindings(
    generation: Mapping[str, Mapping[str, object]],
    analysis: Mapping[str, Mapping[str, object]],
    *,
    metrics_provenance: Mapping[str, object],
    generation_base: Path,
    analysis_base: Path,
    provenance_base: Path,
) -> None:
    errors: list[str] = []
    expected_generated: dict[Path, str] = {}
    for case_id, row in generation.items():
        if row.get("status") != "SUCCESS":
            continue
        generation_path = _resolved_artifact_path(row.get("wav_path"), base=generation_base)
        analysis_path = _resolved_artifact_path(
            analysis[case_id].get("wav_path"), base=analysis_base
        )
        actual_sha256 = _sha256(generation_path) if generation_path.is_file() else None
        if actual_sha256 != row.get("wav_sha256"):
            errors.append(f"{case_id}: generation wav_sha256 does not match current WAV")
        if analysis_path != generation_path or actual_sha256 != analysis[case_id].get("wav_sha256"):
            errors.append(f"{case_id}: analysis wav_sha256 does not match current WAV")
        if isinstance(row.get("wav_sha256"), str):
            expected_generated[generation_path.resolve()] = str(row["wav_sha256"])

    input_sha256 = metrics_provenance.get("input_sha256")
    if not isinstance(input_sha256, dict):
        errors.append("metrics provenance input_sha256 must be an object")
    else:
        generated_audio = _artifact_hashes(
            input_sha256.get("generated_audio"),
            base=provenance_base,
            source="generated_audio",
            errors=errors,
        )
        reference_audio = _artifact_hashes(
            input_sha256.get("reference_audio"),
            base=provenance_base,
            source="reference_audio",
            errors=errors,
        )
        if generated_audio != expected_generated:
            errors.append("metrics provenance generated_audio does not match generation rows")
        if not reference_audio:
            errors.append("metrics provenance reference_audio must be nonempty")
    if errors:
        raise ValueError("; ".join(errors))


def _artifact_hashes(
    value: object,
    *,
    base: Path,
    source: str,
    errors: list[str],
) -> dict[Path, str]:
    if not isinstance(value, dict) or not value:
        errors.append(f"metrics provenance {source} must be a nonempty object")
        return {}
    resolved: dict[Path, str] = {}
    for raw_path, expected_sha256 in value.items():
        if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
            errors.append(f"metrics provenance {source} entries must map paths to SHA-256")
            continue
        path = _resolved_artifact_path(raw_path, base=base).resolve()
        if not path.is_file() or _sha256(path) != expected_sha256:
            errors.append(f"metrics provenance {source} hash mismatch: {path}")
        resolved[path] = expected_sha256
    return resolved


def _resolved_artifact_path(value: object, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        return base / "__missing_artifact__"
    path = Path(value)
    return path if path.is_absolute() else base / path


def _validate_metric_rows(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    generation_results_sha256: str,
) -> None:
    errors: list[str] = []
    for case_id, row in metrics.items():
        status = row.get("metrics_status")
        if status not in {"COMPLETE", "INCOMPLETE"}:
            errors.append(f"{case_id}: metrics_status must be COMPLETE or INCOMPLETE")
        if row.get("generation_results_sha256") != generation_results_sha256:
            errors.append(
                f"{case_id}: metrics provenance binding does not match generation results"
            )
        if status == "COMPLETE":
            errors.extend(
                f"{case_id}: COMPLETE metrics row requires valid {field}"
                for field in ("speaker_similarity", "normalized_cer")
                if _metric(row, field) is None
            )
    if errors:
        raise ValueError("; ".join(errors))


def _validate_inputs(
    generation: Mapping[str, Mapping[str, object]],
    analysis: Mapping[str, Mapping[str, object]],
    metrics: Mapping[str, Mapping[str, object]],
) -> None:
    errors: list[str] = []
    if not generation:
        errors.append("generation results contain no rows")
    missing_analysis = sorted(generation.keys() - analysis.keys())
    missing_metrics = sorted(generation.keys() - metrics.keys())
    extra_analysis = sorted(analysis.keys() - generation.keys())
    extra_metrics = sorted(metrics.keys() - generation.keys())
    if missing_analysis:
        errors.append(f"analysis missing case_id: {', '.join(missing_analysis)}")
    if missing_metrics:
        errors.append(f"metrics missing case_id: {', '.join(missing_metrics)}")
    if extra_analysis:
        errors.append(f"analysis has unknown case_id: {', '.join(extra_analysis)}")
    if extra_metrics:
        errors.append(f"metrics has unknown case_id: {', '.join(extra_metrics)}")
    for case_id in sorted(generation.keys() & analysis.keys()):
        errors.extend(_identity_errors(case_id, generation[case_id], analysis[case_id], "analysis"))
    for case_id in sorted(generation.keys() & metrics.keys()):
        errors.extend(
            _identity_errors(
                case_id,
                generation[case_id],
                metrics[case_id],
                "metrics",
            ),
        )
    if errors:
        raise ValueError("; ".join(errors))


def _identity_errors(
    case_id: str,
    generation: Mapping[str, object],
    other: Mapping[str, object],
    source: str,
) -> list[str]:
    errors = []
    for field in IDENTITY_FIELDS:
        if field not in other:
            errors.append(f"{case_id}: {source} row requires {field}")
            continue
        if generation.get(field) != other.get(field):
            errors.append(f"{case_id}: inconsistent {field} in {source}")
    return errors


def _index_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
) -> dict[str, dict[str, object]]:
    indexed = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            message = f"{source} row requires a nonempty case_id"
            raise ValueError(message)
        if case_id in indexed:
            message = f"duplicate case_id in {source}: {case_id}"
            raise ValueError(message)
        indexed[case_id] = row
    return indexed


def _metric(metrics: Mapping[str, object] | None, field: str) -> float | None:
    if metrics is None:
        return None
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


def _resolve_wav_path(row: Mapping[str, object], *, base: Path) -> Path:
    raw_path = row.get("wav_path")
    if not isinstance(raw_path, str) or not raw_path:
        message = f"missing wav_path for {row.get('case_id')}"
        raise ValueError(message)
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base / path


def _audio_from_row(row: Mapping[str, object]) -> AudioStats | None:
    value = row.get("audio")
    if not isinstance(value, dict):
        return None
    return AudioStats(**value)


def _case_status(
    rejected: Sequence[str],
    incomplete: Sequence[str],
    review: Sequence[str],
) -> str:
    if rejected:
        return "REJECTED"
    if incomplete:
        return "INCOMPLETE"
    if review:
        return "REVIEW"
    return "PASS"


def _checkpoint_status(
    rejected: Sequence[str],
    incomplete: Sequence[str],
    review: Sequence[str],
) -> str:
    if rejected:
        return "REJECTED"
    if incomplete:
        return "INCOMPLETE"
    if review:
        return "REVIEW"
    return "ELIGIBLE"


def _collect_reasons(rows: Sequence[Mapping[str, object]], field: str) -> list[str]:
    return sorted(
        {reason for row in rows for reason in _reasons(row, field)},
    )


def _present_metric(rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, float):
            values.append(value)
    return values


def _rounded_mean(values: Sequence[float]) -> float | None:
    return round(float(np.mean(values)), 8) if values else None


def _review_row(
    row: Mapping[str, object],
    *,
    evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reasons = sorted(
        {
            *_reasons(row, "incomplete_reasons"),
            *_reasons(row, "review_reasons"),
            *_reasons(row, "rejection_reasons"),
        },
    )
    paired_controls = sorted(
        (
            {
                "case_id": candidate["case_id"],
                "text_id": candidate.get("text_id"),
                "seed": candidate.get("seed"),
                "style": candidate.get("style"),
                "wav_path": candidate.get("wav_path"),
                "wav_sha256": candidate.get("wav_sha256"),
            }
            for candidate in evaluations
            if candidate.get("case_id") != row.get("case_id")
            and candidate.get("model_id") == row.get("model_id")
            and candidate.get("checkpoint_step") == row.get("checkpoint_step")
            and candidate.get("seed") == row.get("seed")
            and candidate.get("control") is True
        ),
        key=lambda candidate: str(candidate["case_id"]),
    )
    return {
        "case_id": row["case_id"],
        "model_id": row["model_id"],
        "checkpoint_step": row["checkpoint_step"],
        "wav_path": row.get("wav_path"),
        "wav_sha256": row.get("wav_sha256"),
        "evaluation_status": row["evaluation_status"],
        "review_reasons": reasons,
        "tone_evidence": row["tone_evidence"],
        "paired_control_case_ids": [candidate["case_id"] for candidate in paired_controls],
        "paired_controls": paired_controls,
    }


def _write_review_packet(
    packet_root: Path,
    review_rows: Sequence[Mapping[str, object]],
    *,
    generation_base: Path,
    analysis_base: Path,
) -> None:
    packet_root.mkdir(parents=True, exist_ok=True)
    owners: dict[Path, tuple[Path, str]] = {}
    candidates = []
    for row in review_rows:
        case_id = str(row["case_id"])
        wav = _copy_packet_asset(
            packet_root,
            raw_source=row.get("wav_path"),
            declared_sha256=row.get("wav_sha256"),
            source_base=generation_base,
            relative=_packet_relative_path(
                packet_root,
                category="audio",
                case_id=case_id,
                suffix=".wav",
            ),
            owners=owners,
        )
        tone_evidence = row.get("tone_evidence")
        tone = tone_evidence if isinstance(tone_evidence, dict) else {}
        spectrogram = _copy_packet_asset(
            packet_root,
            raw_source=tone.get("spectrogram_path"),
            declared_sha256=None,
            source_base=analysis_base,
            relative=_packet_relative_path(
                packet_root,
                category="spectrograms",
                case_id=case_id,
                suffix=".png",
            ),
            owners=owners,
            allow_missing=tone.get("spectrogram_error") is not None,
        )
        raw_controls = row.get("paired_controls")
        controls = raw_controls if isinstance(raw_controls, list) else []
        paired_controls = []
        for control in controls:
            if not isinstance(control, dict):
                message = f"review packet paired control must be an object for {case_id}"
                raise TypeError(message)
            control_id = str(control.get("case_id"))
            control_wav = _copy_packet_asset(
                packet_root,
                raw_source=control.get("wav_path"),
                declared_sha256=control.get("wav_sha256"),
                source_base=generation_base,
                relative=_packet_relative_path(
                    packet_root,
                    category="audio",
                    case_id=control_id,
                    suffix=".wav",
                ),
                owners=owners,
            )
            paired_controls.append({"case_id": control_id, "wav": control_wav})
        candidates.append(
            {
                "case_id": case_id,
                "evaluation_status": row.get("evaluation_status"),
                "review_reasons": row.get("review_reasons"),
                "wav": wav,
                "spectrogram": spectrogram,
                "paired_controls": paired_controls,
            },
        )
    _write_json(
        packet_root / "manifest.json",
        {
            "schema_version": "speaker-checkpoint-review-packet/v1",
            "review_candidates": candidates,
        },
    )


def _copy_packet_asset(
    packet_root: Path,
    *,
    raw_source: object,
    declared_sha256: object,
    source_base: Path,
    relative: Path,
    owners: dict[Path, tuple[Path, str]],
    allow_missing: bool = False,
) -> dict[str, str] | None:
    if not isinstance(raw_source, str) or not raw_source:
        return None
    source = _resolve_packet_source(raw_source, base=source_base)
    if not source.is_file():
        if allow_missing:
            return None
        message = f"review packet source does not exist: {source}"
        raise ValueError(message)
    source_sha256 = _sha256(source)
    if declared_sha256 is not None and declared_sha256 != source_sha256:
        message = f"review packet source SHA-256 mismatch: {source}"
        raise ValueError(message)
    owner = owners.get(relative)
    source_identity = (source, source_sha256)
    if owner is not None and owner != source_identity:
        message = f"review packet filename collision: {relative}"
        raise ValueError(message)
    destination = packet_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if owner is None:
        shutil.copy2(source, destination)
        owners[relative] = source_identity
    if _sha256(destination) != source_sha256:
        message = f"review packet copied SHA-256 mismatch: {destination}"
        raise ValueError(message)
    return {
        "path": relative.as_posix(),
        "sha256": source_sha256,
        "source_path": raw_source,
    }


def _resolve_packet_source(raw_path: str, *, base: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    resolved_base = base.resolve()
    resolved = (resolved_base / path).resolve()
    if not resolved.is_relative_to(resolved_base):
        message = f"review packet source escapes base directory: {raw_path}"
        raise ValueError(message)
    return resolved


def _packet_relative_path(
    packet_root: Path,
    *,
    category: str,
    case_id: str,
    suffix: str,
) -> Path:
    if category not in {"audio", "spectrograms"}:
        message = f"unsupported review packet category: {category}"
        raise ValueError(message)
    if not suffix.startswith(".") or Path(suffix).name != suffix:
        message = f"unsafe review packet suffix: {suffix}"
        raise ValueError(message)
    filename = hashlib.sha256(case_id.encode()).hexdigest() + suffix
    relative = Path(category) / filename
    if not (packet_root / relative).resolve().is_relative_to(packet_root.resolve()):
        message = f"review packet destination escapes packet root: {relative}"
        raise ValueError(message)
    return relative


def _case_sort_key(row: Mapping[str, object]) -> tuple[str, int, str, int, str, str]:
    return (
        str(row.get("model_id")),
        _required_int(row, "checkpoint_step"),
        str(row.get("text_id")),
        _required_int(row, "seed"),
        str(row.get("style")),
        str(row.get("case_id")),
    )


def _required_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{field} must be an integer for {row.get('case_id')}"
        raise TypeError(message)
    return value


def _required_float(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{field} must be numeric"
        raise TypeError(message)
    return float(value)


def _required_string(row: Mapping[str, object], field: str, *, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        message = f"{source} requires nonempty string {field}"
        raise ValueError(message)
    return value


def _required_manifest_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"evaluation manifest requires integer {field}"
        raise TypeError(message)
    return value


def _required_sha256(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        message = f"{field} must be a lowercase SHA-256 hex digest"
        raise ValueError(message)
    return value


def _required_string_tuple(row: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = row.get(field)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        message = f"evaluation manifest {field} must be a nonempty string list"
        raise ValueError(message)
    if len(value) != len(set(value)):
        message = f"evaluation manifest {field} contains duplicates"
        raise ValueError(message)
    return tuple(value)


def _required_int_tuple(row: Mapping[str, object], field: str) -> tuple[int, ...]:
    value = row.get(field)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        message = f"evaluation manifest {field} must be a nonempty integer list"
        raise ValueError(message)
    if len(value) != len(set(value)):
        message = f"evaluation manifest {field} contains duplicates"
        raise ValueError(message)
    return tuple(value)


def _reasons(row: Mapping[str, object], field: str) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        message = f"{field} must be a string list"
        raise TypeError(message)
    return value


def _has_reason(row: Mapping[str, object], field: str, reason: str) -> bool:
    return reason in _reasons(row, field)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value: Any = json.loads(line)
        if not isinstance(value, dict):
            message = f"JSONL row must be an object: {path}:{line_number}"
            raise TypeError(message)
        rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, object]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"JSON document must be an object: {path}"
        raise TypeError(message)
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        _write_json(temporary, payload)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _evaluation_artifact_sha256(output_dir: Path) -> dict[str, str]:
    root_artifacts = (
        "evaluation-results.jsonl",
        "checkpoint-summary.jsonl",
        "review-candidates.jsonl",
        "evaluation-config.json",
        "selected-models.json",
    )
    paths = [output_dir / name for name in root_artifacts]
    packet_root = output_dir / "review_packet"
    paths.extend(path for path in packet_root.rglob("*") if path.is_file())
    return {
        str(path.resolve()): _sha256(path)
        for path in sorted(paths, key=lambda candidate: candidate.as_posix())
    }


def _review_packet_counts(packet_root: Path) -> dict[str, int]:
    manifest = _read_json(packet_root / "manifest.json")
    raw_candidates = manifest.get("review_candidates")
    if not isinstance(raw_candidates, list):
        message = "review packet candidates must be a list"
        raise TypeError(message)
    reference_count = 0
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            message = "review packet candidates must be objects"
            raise TypeError(message)
        if raw_candidate.get("wav") is not None:
            reference_count += 1
        raw_controls = raw_candidate.get("paired_controls")
        if not isinstance(raw_controls, list):
            message = "review packet paired_controls must be a list"
            raise TypeError(message)
        reference_count += sum(
            isinstance(control, dict) and control.get("wav") is not None for control in raw_controls
        )
    return {
        "review_packet_reference_count": reference_count,
        "review_packet_unique_audio_count": sum(
            path.is_file() for path in (packet_root / "audio").glob("*")
        ),
        "review_packet_spectrogram_count": sum(
            path.is_file() for path in (packet_root / "spectrograms").glob("*")
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-results", type=Path, required=True)
    parser.add_argument("--analysis-results", type=Path, required=True)
    parser.add_argument("--metrics-results", type=Path, required=True)
    parser.add_argument("--metrics-provenance", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
