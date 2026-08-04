from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import unicodedata
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np

TARGET_SAMPLE_RATE = 16_000
PROVENANCE_SCHEMA_VERSION = "speaker-metrics-extraction/v1"
SHA256_HEX_LENGTH = 64
PCM_WIDTH_16_BIT = 2
PCM_WIDTH_24_BIT = 3
PCM_WIDTH_32_BIT = 4
RESAMPLE_FILTER_TAPS = 161
RESAMPLE_CUTOFF_FRACTION = 0.94
IDENTITY_FIELDS = (
    "case_id",
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


class SpeakerEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def source_sha256(self) -> str: ...

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


class Transcriber(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...

    @property
    def torchcodec_mode(self) -> str: ...

    @property
    def source_sha256(self) -> str: ...

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str: ...


@dataclass(frozen=True, slots=True)
class SpeechBrainECAPA:
    source: Path
    savedir: Path
    model_id: str
    revision: str
    source_sha256: str
    _classifier: object

    @classmethod
    def load(
        cls,
        *,
        source: Path,
        savedir: Path,
        model_id: str,
        revision: str,
    ) -> SpeechBrainECAPA:
        if not source.is_dir():
            message = f"ECAPA source must be an existing local directory: {source}"
            raise ValueError(message)
        try:
            from speechbrain.inference.classifiers import (  # type: ignore[import-not-found] # noqa: PLC0415
                EncoderClassifier,
            )
        except ImportError:
            try:
                from speechbrain.inference.speaker import (  # type: ignore[import-not-found] # noqa: PLC0415
                    SpeakerRecognition as EncoderClassifier,
                )
            except ImportError as exc:
                message = "SpeechBrain is required to compute speaker embeddings"
                raise RuntimeError(message) from exc
        savedir.mkdir(parents=True, exist_ok=True)
        classifier = EncoderClassifier.from_hparams(
            source=str(source),
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
        return cls(
            source=source,
            savedir=savedir,
            model_id=model_id,
            revision=revision,
            source_sha256=sha256_tree(source),
            _classifier=classifier,
        )

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        normalized = resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)
        try:
            import torch  # type: ignore[import-not-found] # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            message = "PyTorch is required to compute speaker embeddings"
            raise RuntimeError(message) from exc
        waveform = torch.from_numpy(normalized.astype(np.float32, copy=False)).unsqueeze(0)
        encode_batch = getattr(self._classifier, "encode_batch", None)
        if not callable(encode_batch):
            message = "SpeechBrain classifier does not provide encode_batch"
            raise TypeError(message)
        with torch.no_grad():
            embedding = encode_batch(waveform)
        return np.asarray(embedding.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


@dataclass(frozen=True, slots=True)
class WhisperTranscriber:
    source: Path
    model_id: str
    revision: str
    device: str
    dtype: str
    torchcodec_mode: str
    source_sha256: str
    _pipeline: object

    @classmethod
    def load(
        cls,
        *,
        source: Path,
        model_id: str,
        revision: str,
        device: str,
    ) -> WhisperTranscriber:
        if not source.is_dir():
            message = f"Whisper source must be an existing local directory: {source}"
            raise ValueError(message)
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # type: ignore[import-not-found] # noqa: PLC0415
                pipeline,
            )

            asr_module = importlib.import_module(
                "transformers.pipelines.automatic_speech_recognition",
            )
        except ImportError as exc:
            message = "Transformers is required to transcribe generated audio"
            raise RuntimeError(message) from exc
        is_torchcodec_available = getattr(asr_module, "is_torchcodec_available", None)
        if not callable(is_torchcodec_available):
            message = "Transformers ASR module does not report torchcodec availability"
            raise TypeError(message)
        torchcodec_mode = "unavailable"
        if is_torchcodec_available():
            try:
                importlib.import_module("torchcodec")
            except (ImportError, OSError, RuntimeError):
                asr_module.is_torchcodec_available = _torchcodec_unavailable  # type: ignore[attr-defined]
                torchcodec_mode = "disabled_import_failure"
            else:
                torchcodec_mode = "available"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        dtype_name = "float16" if device.startswith("cuda") else "float32"
        asr = pipeline(
            task="automatic-speech-recognition",
            model=str(source),
            revision=revision,
            device=device,
            dtype=dtype,
        )
        return cls(
            model_id=model_id,
            source=source,
            revision=revision,
            device=device,
            dtype=dtype_name,
            torchcodec_mode=torchcodec_mode,
            source_sha256=sha256_tree(source),
            _pipeline=asr,
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        normalized = resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)
        if not callable(self._pipeline):
            message = "Transformers ASR pipeline is not callable"
            raise TypeError(message)
        result = self._pipeline(
            {"raw": normalized.astype(np.float32, copy=False), "sampling_rate": TARGET_SAMPLE_RATE},
            generate_kwargs={"language": "ja", "task": "transcribe"},
        )
        if not isinstance(result, Mapping):
            message = "Whisper returned a non-object result"
            raise TypeError(message)
        transcript = result.get("text")
        if not isinstance(transcript, str):
            message = "Whisper result does not contain a string transcript"
            raise TypeError(message)
        return transcript


def _torchcodec_unavailable() -> bool:
    return False


def normalize_japanese_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        chr(ord(character) - 0x60) if "ァ" <= character <= "ヶ" else character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_character != hypothesis_character),
                ),
            )
        previous = current
    return previous[-1]


def normalized_cer(reference: str, transcript: str) -> float:
    normalized_reference = normalize_japanese_text(reference)
    normalized_transcript = normalize_japanese_text(transcript)
    if not normalized_reference:
        message = "reference text is empty after normalization"
        raise ValueError(message)
    if not normalized_transcript:
        message = "transcript is empty"
        raise ValueError(message)
    distance = levenshtein_distance(normalized_reference, normalized_transcript)
    return min(1.0, distance / len(normalized_reference))


def normalized_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = _validated_embedding(left, name="embedding")
    right_vector = _validated_embedding(right, name="reference embedding")
    if left_vector.shape != right_vector.shape:
        message = "embeddings must have the same shape"
        raise ValueError(message)
    cosine = float(np.dot(left_vector, right_vector))
    return min(1.0, max(0.0, (cosine + 1.0) / 2.0))


def aggregate_reference_centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    if not embeddings:
        message = "at least one reference embedding is required"
        raise ValueError(message)
    normalized = [_validated_embedding(embedding, name="embedding") for embedding in embeddings]
    expected_shape = normalized[0].shape
    if any(embedding.shape != expected_shape for embedding in normalized[1:]):
        message = "reference embeddings must have the same shape"
        raise ValueError(message)
    centroid = np.mean(np.stack(normalized), axis=0)
    return _validated_embedding(centroid, name="reference centroid")


def _validated_embedding(embedding: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        message = f"{name} must be a non-empty one-dimensional array"
        raise ValueError(message)
    if not np.all(np.isfinite(vector)):
        message = f"{name} contains non-finite values"
        raise ValueError(message)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        message = f"{name} has zero norm"
        raise ValueError(message)
    return vector / norm


def load_generation_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            message = f"blank JSONL row at line {line_number}"
            raise ValueError(message)
        try:
            raw_row: object = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"invalid JSON at line {line_number}: {exc.msg}"
            raise ValueError(message) from exc
        row = validate_generation_row(raw_row, line_number=line_number)
        case_id = cast("str", row["case_id"])
        if case_id in seen_case_ids:
            message = f"duplicate case_id at line {line_number}: {case_id}"
            raise ValueError(message)
        seen_case_ids.add(case_id)
        rows.append(row)
    if not rows:
        message = f"generation results contain no rows: {path}"
        raise ValueError(message)
    return tuple(rows)


def validate_generation_row(raw_row: object, *, line_number: int = 1) -> dict[str, object]:
    if not isinstance(raw_row, dict):
        message = f"generation row {line_number} must be an object"
        raise TypeError(message)
    row = cast("dict[str, object]", raw_row)
    _validate_generation_identity(row, line_number=line_number)
    if row["status"] not in {"SUCCESS", "ERROR"}:
        message = f"generation row {line_number} has unsupported status: {row['status']}"
        raise ValueError(message)
    wav_path = row.get("wav_path")
    wav_sha256 = row.get("wav_sha256")
    if row["status"] == "SUCCESS" and (not isinstance(wav_path, str) or not wav_path.strip()):
        message = f"generation row {line_number} SUCCESS requires non-empty string wav_path"
        raise ValueError(message)
    if row["status"] == "ERROR" and wav_path is not None and not isinstance(wav_path, str):
        message = f"generation row {line_number} ERROR wav_path must be a string or null"
        raise ValueError(message)
    if row["status"] == "SUCCESS" and not _is_sha256(wav_sha256):
        message = f"generation row {line_number} SUCCESS requires wav_sha256"
        raise ValueError(message)
    if row["status"] == "ERROR" and wav_sha256 is not None:
        message = f"generation row {line_number} ERROR wav_sha256 must be null"
        raise ValueError(message)
    for field in ("checkpoint_step", "seed"):
        if not isinstance(row.get(field), int) or isinstance(row.get(field), bool):
            message = f"generation row {line_number} requires integer {field}"
            raise TypeError(message)
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or not all(isinstance(key, str) for key in provenance):
        message = f"generation row {line_number} provenance must be an object"
        raise ValueError(message)
    validated = dict(row)
    validated.setdefault("wav_path", None)
    validated.setdefault("wav_sha256", None)
    return validated


def _validate_generation_identity(row: Mapping[str, object], *, line_number: int) -> None:
    for field in (
        "case_id",
        "model_id",
        "checkpoint",
        "speaker_filename",
        "embedding_path",
        "text_id",
        "text",
        "style",
        "status",
    ):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            message = f"generation row {line_number} requires non-empty string {field}"
            raise ValueError(message)
    for field in (
        "embedding_sha256",
        "evaluation_manifest_sha256",
        "base_checkpoint_sha256",
    ):
        if not _is_sha256(row.get(field)):
            message = f"generation row {line_number} requires {field}"
            raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def load_reference_wavs(path: Path) -> dict[str, tuple[Path, ...]]:
    raw_payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict) or not raw_payload:
        message = "reference WAV mapping must be a non-empty object"
        raise ValueError(message)
    if "references" in raw_payload:
        return _load_rich_reference_manifest(path, raw_payload)
    raw_mapping = raw_payload
    references: dict[str, tuple[Path, ...]] = {}
    for model_id, raw_paths in raw_mapping.items():
        if not isinstance(model_id, str) or not model_id:
            message = "reference WAV model ids must be non-empty strings"
            raise ValueError(message)
        if not isinstance(raw_paths, list) or not raw_paths:
            message = f"reference WAVs for {model_id} must be a non-empty list"
            raise ValueError(message)
        resolved: list[Path] = []
        seen: set[str] = set()
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path:
                message = f"reference WAV paths for {model_id} must be non-empty strings"
                raise ValueError(message)
            if raw_path in seen:
                message = f"duplicate reference WAV for {model_id}: {raw_path}"
                raise ValueError(message)
            seen.add(raw_path)
            wav_path = Path(raw_path)
            if not wav_path.is_absolute():
                wav_path = path.parent / wav_path
            resolved.append(wav_path)
        missing_paths = [wav_path for wav_path in resolved if not wav_path.is_file()]
        if missing_paths:
            message = f"reference WAV does not exist for {model_id}: {missing_paths[0]}"
            raise ValueError(message)
        references[model_id] = tuple(resolved)
    return references


def _load_rich_reference_manifest(
    path: Path,
    payload: Mapping[str, object],
) -> dict[str, tuple[Path, ...]]:
    model_id = payload.get("model_id")
    raw_references = payload.get("references")
    if not isinstance(model_id, str) or not model_id:
        message = "rich reference WAV manifest requires non-empty string model_id"
        raise ValueError(message)
    if not isinstance(raw_references, list) or not raw_references:
        message = "rich reference WAV manifest references must be a non-empty list"
        raise ValueError(message)
    _validate_rich_reference_flags(payload)
    resolved: list[Path] = []
    seen_paths: set[str] = set()
    for index, raw_reference in enumerate(raw_references, start=1):
        raw_path, expected_sha256 = _rich_reference_identity(raw_reference, index=index)
        if raw_path in seen_paths:
            message = f"duplicate reference WAV for {model_id}: {raw_path}"
            raise ValueError(message)
        seen_paths.add(raw_path)
        wav_path = Path(raw_path)
        if not wav_path.is_absolute():
            wav_path = path.parent / wav_path
        if not wav_path.is_file():
            message = f"reference WAV does not exist for {model_id}: {wav_path}"
            raise ValueError(message)
        if sha256_file(wav_path) != expected_sha256:
            message = f"reference WAV SHA-256 mismatch for {model_id}: {wav_path}"
            raise ValueError(message)
        resolved.append(wav_path)
    return {model_id: tuple(resolved)}


def _validate_rich_reference_flags(payload: Mapping[str, object]) -> None:
    if payload.get("all_reference_wavs_finite") is not True:
        message = "rich reference WAV manifest requires all_reference_wavs_finite=true"
        raise ValueError(message)
    if payload.get("all_selected_source_hashes_verified") is not True:
        message = "rich reference WAV manifest requires all_selected_source_hashes_verified=true"
        raise ValueError(message)


def _rich_reference_identity(raw_reference: object, *, index: int) -> tuple[str, str]:
    if not isinstance(raw_reference, dict):
        message = f"rich reference WAV entry {index} must be an object"
        raise TypeError(message)
    raw_path = raw_reference.get("reference_wav_path")
    expected_sha256 = raw_reference.get("reference_wav_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        message = f"rich reference WAV entry {index} requires reference_wav_path"
        raise ValueError(message)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        message = f"rich reference WAV entry {index} requires reference_wav_sha256"
        raise ValueError(message)
    return raw_path, expected_sha256


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            if reader.getcomptype() != "NONE":
                message = f"compressed WAV is unsupported: {path}"
                raise ValueError(message)
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (EOFError, wave.Error) as exc:
        message = f"invalid WAV file: {path}"
        raise ValueError(message) from exc
    if channels <= 0 or sample_rate <= 0:
        message = f"invalid WAV metadata: {path}"
        raise ValueError(message)
    samples = _decode_pcm(frames, sample_width)
    if samples.size % channels != 0:
        message = f"WAV frame data is misaligned: {path}"
        raise ValueError(message)
    mono = samples.reshape(-1, channels).mean(axis=1)
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        message = f"WAV contains no finite audio: {path}"
        raise ValueError(message)
    return mono, sample_rate


def _decode_pcm(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == PCM_WIDTH_16_BIT:
        return np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32_768.0
    if sample_width == PCM_WIDTH_24_BIT:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8_388_608.0
    if sample_width == PCM_WIDTH_32_BIT:
        return np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2_147_483_648.0
    message = f"unsupported PCM sample width: {sample_width}"
    raise ValueError(message)


def resample_audio(samples: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    vector = np.asarray(samples, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        message = "audio must be a non-empty finite one-dimensional array"
        raise ValueError(message)
    if sample_rate <= 0 or target_rate <= 0:
        message = "sample rates must be positive"
        raise ValueError(message)
    if sample_rate == target_rate:
        return vector
    if target_rate < sample_rate:
        vector = _lowpass_for_downsampling(
            vector,
            sample_rate=sample_rate,
            target_rate=target_rate,
        )
    output_size = max(1, round(vector.size * target_rate / sample_rate))
    source_positions = np.arange(vector.size, dtype=np.float64)
    target_positions = np.arange(output_size, dtype=np.float64) * sample_rate / target_rate
    return np.interp(target_positions, source_positions, vector)


def _lowpass_for_downsampling(
    samples: np.ndarray,
    *,
    sample_rate: int,
    target_rate: int,
) -> np.ndarray:
    half_width = RESAMPLE_FILTER_TAPS // 2
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    cutoff = RESAMPLE_CUTOFF_FRACTION * target_rate / (2.0 * sample_rate)
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets)
    kernel *= np.blackman(RESAMPLE_FILTER_TAPS)
    kernel /= np.sum(kernel)
    pad_mode = "reflect" if samples.size > 1 else "edge"
    padded = np.pad(samples, (half_width, half_width), mode=pad_mode)
    return np.convolve(padded, kernel, mode="valid")


def run_extraction(
    *,
    generation_results: Path,
    reference_wavs_path: Path,
    output_path: Path,
    provenance_path: Path,
    embedder: SpeakerEmbedder,
    transcriber: Transcriber,
) -> int:
    generation_results = generation_results.resolve()
    reference_wavs_path = reference_wavs_path.resolve()
    generation_rows = load_generation_rows(generation_results)
    _validate_generation_audio(generation_rows, base=generation_results.parent)
    references = load_reference_wavs(reference_wavs_path)
    _validate_reference_coverage(generation_rows, references)
    reference_centroids, reference_errors = _reference_centroids(references, embedder=embedder)
    generation_results_sha256 = sha256_file(generation_results)
    metric_rows = tuple(
        _compute_metric_row(
            row,
            generation_dir=generation_results.parent,
            reference_centroids=reference_centroids,
            reference_errors=reference_errors,
            embedder=embedder,
            transcriber=transcriber,
            generation_results_sha256=generation_results_sha256,
        )
        for row in generation_rows
    )
    _write_jsonl(output_path, metric_rows)
    provenance = _build_provenance(
        generation_results=generation_results,
        reference_wavs_path=reference_wavs_path,
        generation_rows=generation_rows,
        references=references,
        metric_rows=metric_rows,
        embedder=embedder,
        transcriber=transcriber,
    )
    _write_json(provenance_path, provenance)
    return 1 if any(row["metrics_status"] == "INCOMPLETE" for row in metric_rows) else 0


def _validate_reference_coverage(
    generation_rows: Sequence[Mapping[str, object]],
    references: Mapping[str, Sequence[Path]],
) -> None:
    missing = sorted(
        {
            cast("str", row["model_id"])
            for row in generation_rows
            if cast("str", row["model_id"]) not in references
        },
    )
    if missing:
        message = f"reference WAV mapping is missing model ids: {', '.join(missing)}"
        raise ValueError(message)


def _reference_centroids(
    references: Mapping[str, Sequence[Path]],
    *,
    embedder: SpeakerEmbedder,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    centroids: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for model_id, paths in references.items():
        try:
            embeddings = []
            for path in paths:
                samples, sample_rate = read_wav(path)
                normalized = resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)
                embeddings.append(embedder.embed(normalized, TARGET_SAMPLE_RATE))
            centroids[model_id] = aggregate_reference_centroid(embeddings)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors[model_id] = f"reference embedding: {exc}"
    return centroids, errors


def _compute_metric_row(
    generation_row: Mapping[str, object],
    *,
    generation_dir: Path,
    reference_centroids: Mapping[str, np.ndarray],
    reference_errors: Mapping[str, str],
    embedder: SpeakerEmbedder,
    transcriber: Transcriber,
    generation_results_sha256: str,
) -> dict[str, object]:
    metric_row = {field: generation_row.get(field) for field in IDENTITY_FIELDS}
    metric_row["generation_results_sha256"] = generation_results_sha256
    if generation_row["status"] != "SUCCESS":
        metric_row.update(
            metrics_status="INCOMPLETE",
            incomplete_reason=f"generation status is {generation_row['status']}",
        )
        return metric_row
    model_id = cast("str", generation_row["model_id"])
    if model_id in reference_errors:
        metric_row.update(
            metrics_status="INCOMPLETE",
            incomplete_reason=reference_errors[model_id],
        )
        return metric_row
    raw_wav_path = cast("str", generation_row["wav_path"])
    wav_path = Path(raw_wav_path)
    if not wav_path.is_absolute():
        wav_path = generation_dir / wav_path
    try:
        samples, sample_rate = read_wav(wav_path)
        normalized = resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)
        embedding = embedder.embed(normalized, TARGET_SAMPLE_RATE)
        transcript = transcriber.transcribe(normalized, TARGET_SAMPLE_RATE)
        similarity = normalized_cosine_similarity(
            embedding,
            reference_centroids[model_id],
        )
        cer = normalized_cer(cast("str", generation_row["text"]), transcript)
        _validate_metric(similarity, name="speaker_similarity")
        _validate_metric(cer, name="normalized_cer")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        metric_row.update(metrics_status="INCOMPLETE", incomplete_reason=str(exc))
        return metric_row
    metric_row.update(
        metrics_status="COMPLETE",
        speaker_similarity=similarity,
        normalized_cer=cer,
        transcript=transcript,
    )
    return metric_row


def _validate_metric(value: float, *, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        message = f"{name} must be finite and within [0, 1]"
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        message = f"model source must be an existing directory: {path}"
        raise ValueError(message)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        message = f"model source contains no files: {path}"
        raise ValueError(message)
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_generation_audio(
    rows: Sequence[Mapping[str, object]],
    *,
    base: Path,
) -> None:
    errors: list[str] = []
    for row in rows:
        if row.get("status") != "SUCCESS":
            continue
        raw_path = cast("str", row["wav_path"])
        path = Path(raw_path)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            errors.append(f"generation WAV does not exist for {row['case_id']}: {path}")
        elif sha256_file(path) != row.get("wav_sha256"):
            errors.append(f"generation wav_sha256 mismatch for {row['case_id']}: {path}")
    if errors:
        raise ValueError("; ".join(errors))


def _build_provenance(
    *,
    generation_results: Path,
    reference_wavs_path: Path,
    generation_rows: Sequence[Mapping[str, object]],
    references: Mapping[str, Sequence[Path]],
    metric_rows: Sequence[Mapping[str, object]],
    embedder: SpeakerEmbedder,
    transcriber: Transcriber,
) -> dict[str, object]:
    generated_audio: dict[str, str] = {}
    for row in generation_rows:
        raw_path = row.get("wav_path")
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = generation_results.parent / path
        if path.is_file():
            generated_audio[str(path)] = sha256_file(path)
    reference_audio = {
        str(path): sha256_file(path) for paths in references.values() for path in paths
    }
    complete_count = sum(row["metrics_status"] == "COMPLETE" for row in metric_rows)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "models": {
            "speaker_embedding": {
                "model_id": embedder.model_id,
                "revision": embedder.revision,
                "source_sha256": embedder.source_sha256,
            },
            "transcription": {
                "model_id": transcriber.model_id,
                "revision": transcriber.revision,
                "device": transcriber.device,
                "dtype": transcriber.dtype,
                "torchcodec_mode": transcriber.torchcodec_mode,
                "source_sha256": transcriber.source_sha256,
            },
        },
        "input_sha256": {
            "generation_results": sha256_file(generation_results),
            "reference_wavs": sha256_file(reference_wavs_path),
            "generated_audio": generated_audio,
            "reference_audio": reference_audio,
        },
        "case_count": len(metric_rows),
        "complete_count": complete_count,
        "incomplete_count": len(metric_rows) - complete_count,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_provenance_path(output_path: Path) -> Path:
    return output_path.with_suffix(".provenance.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-results", type=Path, required=True)
    parser.add_argument("--reference-wavs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--ecapa-source", type=Path, required=True)
    parser.add_argument("--ecapa-savedir", type=Path, required=True)
    parser.add_argument("--ecapa-model-id", required=True)
    parser.add_argument("--ecapa-revision", required=True)
    parser.add_argument("--whisper-model", required=True)
    parser.add_argument("--whisper-source", type=Path, required=True)
    parser.add_argument("--whisper-revision", required=True)
    parser.add_argument("--whisper-device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    if args.provenance_output is None:
        args.provenance_output = _default_provenance_path(args.output)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    transcriber = WhisperTranscriber.load(
        source=args.whisper_source,
        model_id=args.whisper_model,
        revision=args.whisper_revision,
        device=args.whisper_device,
    )
    embedder = SpeechBrainECAPA.load(
        source=args.ecapa_source,
        savedir=args.ecapa_savedir,
        model_id=args.ecapa_model_id,
        revision=args.ecapa_revision,
    )
    exit_code = run_extraction(
        generation_results=args.generation_results,
        reference_wavs_path=args.reference_wavs,
        output_path=args.output,
        provenance_path=args.provenance_output,
        embedder=embedder,
        transcriber=transcriber,
    )
    print(f"speaker metrics written to {args.output}")
    print(f"provenance written to {args.provenance_output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
