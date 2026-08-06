from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from typing_extensions import override

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/compute_600m_speaker_metrics.py")
TARGET_SAMPLE_RATE = 16_000
EMBEDDING_PATH = "/canonical/checkpoint-1000.speaker.safetensors"
EMBEDDING_SHA256 = "b" * 64
EVALUATION_MANIFEST_SHA256 = "c" * 64
BASE_CHECKPOINT_SHA256 = "d" * 64
MAX_ALIAS_RMS = 0.01


def _generation_row(
    *,
    case_id: str = "case-1",
    status: str = "SUCCESS",
    wav_path: str | None = "generated.wav",
    wav_sha256: str | None = "f" * 64,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "model_id": "anabel",
        "checkpoint_step": 1000,
        "checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        "speaker_filename": "checkpoint-1000.speaker.safetensors",
        "embedding_path": EMBEDDING_PATH,
        "embedding_sha256": EMBEDDING_SHA256,
        "evaluation_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "text_id": "hello",
        "text": "こんにちは",
        "seed": 1,
        "style": "neutral",
        "wav_path": wav_path,
        "wav_sha256": wav_sha256,
        "status": status,
        "provenance": {
            "training_config_sha256": "a" * 64,
            "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
            "base_revision": "base-revision",
            "run_id": "anabel-run",
        },
    }


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compute_600m_speaker_metrics", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 8_000) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32_767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class _Embedder:
    model_id = "speechbrain/spkrec-ecapa-voxceleb"
    revision = "local-fixture"
    source_sha256 = "1" * 64

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        assert self.model_id
        assert sample_rate == TARGET_SAMPLE_RATE
        if float(np.mean(samples)) >= 0:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([0.0, 1.0], dtype=np.float64)


class _Transcriber:
    model_id = "openai/whisper-large-v3-turbo"
    revision = "fixture-revision"
    device = "cpu"
    dtype = "float32"
    torchcodec_mode = "unavailable"
    source_sha256 = "2" * 64

    def __init__(self, transcript: str = "こんにちは 世界") -> None:
        self.transcript = transcript

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        assert samples.ndim == 1
        assert sample_rate == TARGET_SAMPLE_RATE
        return self.transcript


def test_japanese_text_normalization_and_character_cer() -> None:
    module = _load_script()

    assert module.normalize_japanese_text(" Ｈｅｌｌｏ、\t世界！\n") == "Hello世界"  # noqa: RUF001
    expected_distance = 2
    assert module.levenshtein_distance("こんにちは", "こんばんは") == expected_distance
    assert module.normalized_cer("こんにちは。", "こんにちわ") == pytest.approx(0.2)
    assert module.normalized_cer("abc", "xyzxyz") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="reference text is empty"):
        module.normalized_cer("。 \t", "text")
    with pytest.raises(ValueError, match="transcript is empty"):
        module.normalized_cer("text", "。 \t")


@pytest.mark.parametrize(
    ("katakana", "hiragana"),
    [("ウンコ", "うんこ"), ("チンコ", "ちんこ"), ("マンコ", "まんこ")],
)
def test_japanese_text_normalization_folds_katakana_to_hiragana(
    katakana: str,
    hiragana: str,
) -> None:
    module = _load_script()

    assert module.normalize_japanese_text(f" {katakana}、\n") == hiragana
    assert module.normalized_cer(katakana, hiragana) == pytest.approx(0.0)


def test_embedding_helpers_normalize_cosine_and_centroid() -> None:
    module = _load_script()

    assert module.normalized_cosine_similarity(
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    ) == pytest.approx(1.0)
    assert module.normalized_cosine_similarity(
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
    ) == pytest.approx(0.0)
    assert module.normalized_cosine_similarity(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ) == pytest.approx(0.5)

    centroid = module.aggregate_reference_centroid(
        [np.array([3.0, 0.0]), np.array([0.0, 4.0])],
    )
    assert centroid == pytest.approx(np.array([2**-0.5, 2**-0.5]))

    with pytest.raises(ValueError, match="non-finite"):
        module.normalized_cosine_similarity(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(ValueError, match="zero norm"):
        module.aggregate_reference_centroid([np.zeros(2)])
    with pytest.raises(ValueError, match="same shape"):
        module.aggregate_reference_centroid([np.ones(2), np.ones(3)])


def test_resample_24k_to_16k_attenuates_10khz_alias() -> None:
    module = _load_script()
    sample_rate = 24_000
    duration_seconds = 1.0
    time = np.arange(round(sample_rate * duration_seconds), dtype=np.float64) / sample_rate
    samples = np.sin(2.0 * np.pi * 10_000.0 * time)

    resampled = module.resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)

    assert resampled.size == TARGET_SAMPLE_RATE
    assert float(np.sqrt(np.mean(resampled**2))) < MAX_ALIAS_RMS


def test_resample_24k_to_16k_preserves_voice_band_rms() -> None:
    module = _load_script()
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    samples = np.sin(2.0 * np.pi * 1_000.0 * time)

    resampled = module.resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)

    input_rms = float(np.sqrt(np.mean(samples**2)))
    output_rms = float(np.sqrt(np.mean(resampled**2)))
    assert output_rms == pytest.approx(input_rms, rel=0.02)


def test_sha256_tree_binds_sorted_relative_paths_and_file_bytes(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "snapshot"
    (source / "nested").mkdir(parents=True)
    (source / "z.bin").write_bytes(b"z")
    (source / "nested/a.bin").write_bytes(b"a")

    first = module.sha256_tree(source)
    (source / "nested/a.bin").write_bytes(b"changed")

    assert len(first) == module.SHA256_HEX_LENGTH
    assert module.sha256_tree(source) != first


def test_loaders_reject_duplicate_cases_and_invalid_reference_mapping(tmp_path: Path) -> None:
    module = _load_script()
    generation_path = tmp_path / "generation.jsonl"
    valid = _generation_row(case_id="anabel__case", wav_path="case.wav")
    _write_jsonl(generation_path, [valid, valid])

    with pytest.raises(ValueError, match=r"duplicate case_id.*anabel__case"):
        module.load_generation_rows(generation_path)

    references_path = tmp_path / "references.json"
    references_path.write_text(json.dumps({"anabel": ["one.wav", "one.wav"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate reference WAV"):
        module.load_reference_wavs(references_path)

    references_path.write_text(json.dumps({"anabel": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        module.load_reference_wavs(references_path)

    references_path.write_text(json.dumps({"anabel": ["missing.wav"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="reference WAV does not exist"):
        module.load_reference_wavs(references_path)


@pytest.mark.parametrize(
    "field",
    [
        "embedding_path",
        "embedding_sha256",
        "evaluation_manifest_sha256",
        "base_checkpoint_sha256",
    ],
)
def test_generation_rows_require_checkpoint_identity(field: str) -> None:
    module = _load_script()
    row = _generation_row()
    del row[field]

    with pytest.raises(ValueError, match=field):
        module.validate_generation_row(row)


def test_load_reference_wavs_accepts_rich_manifest_and_verifies_hash(tmp_path: Path) -> None:
    module = _load_script()
    wav_path = tmp_path / "references" / "anabel.wav"
    _write_wav(wav_path, np.full(80, 0.2, dtype=np.float64))
    digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "reference-wavs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-reference-wavs/v1",
                "model_id": "anabel",
                "all_reference_wavs_finite": True,
                "all_selected_source_hashes_verified": True,
                "references": [
                    {
                        "reference_wav_path": "references/anabel.wav",
                        "reference_wav_sha256": digest,
                        "source_id": "anabel-clean-001",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert module.load_reference_wavs(manifest_path) == {"anabel": (wav_path,)}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["references"][0]["reference_wav_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reference WAV SHA-256 mismatch"):
        module.load_reference_wavs(manifest_path)


def test_run_extraction_writes_complete_metrics_and_hashed_provenance(tmp_path: Path) -> None:
    module = _load_script()
    generation_path = tmp_path / "generation-results.jsonl"
    references_path = tmp_path / "reference-wavs.json"
    output_path = tmp_path / "metrics-results.jsonl"
    provenance_path = tmp_path / "metrics-results.provenance.json"

    positive = np.full(80, 0.2, dtype=np.float64)
    generated_wav = tmp_path / "generated.wav"
    reference_wav = tmp_path / "reference.wav"
    _write_wav(generated_wav, positive)
    _write_wav(reference_wav, positive)
    generation_row = {
        "case_id": "anabel__checkpoint-1000__hello__seed-1__neutral",
        "model_id": "anabel",
        "checkpoint_step": 1000,
        "checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        "speaker_filename": "checkpoint-1000.speaker.safetensors",
        "embedding_path": EMBEDDING_PATH,
        "embedding_sha256": EMBEDDING_SHA256,
        "evaluation_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "text_id": "hello",
        "text": "こんにちは、世界。",
        "seed": 1,
        "style": "neutral",
        "wav_path": str(generated_wav),
        "wav_sha256": hashlib.sha256(generated_wav.read_bytes()).hexdigest(),
        "status": "SUCCESS",
        "provenance": {"training_config_sha256": "abc123"},
    }
    _write_jsonl(generation_path, [generation_row])
    references_path.write_text(
        json.dumps({"anabel": [reference_wav.name]}, ensure_ascii=False),
        encoding="utf-8",
    )

    exit_code = module.run_extraction(
        generation_results=generation_path,
        reference_wavs_path=references_path,
        output_path=output_path,
        provenance_path=provenance_path,
        embedder=_Embedder(),
        transcriber=_Transcriber(),
    )

    assert exit_code == 0
    [row] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert row == {
        "case_id": generation_row["case_id"],
        "checkpoint": generation_row["checkpoint"],
        "checkpoint_step": 1000,
        "embedding_path": EMBEDDING_PATH,
        "embedding_sha256": EMBEDDING_SHA256,
        "evaluation_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "metrics_status": "COMPLETE",
        "generation_results_sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
        "model_id": "anabel",
        "normalized_cer": 0.0,
        "provenance": generation_row["provenance"],
        "seed": 1,
        "speaker_filename": generation_row["speaker_filename"],
        "speaker_similarity": 1.0,
        "style": "neutral",
        "text_id": "hello",
        "transcript": "こんにちは 世界",
        "wav_path": str(generated_wav),
        "wav_sha256": generation_row["wav_sha256"],
    }

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "speaker-metrics-extraction/v1"
    assert provenance["models"] == {
        "speaker_embedding": {
            "model_id": _Embedder.model_id,
            "revision": _Embedder.revision,
            "source_sha256": _Embedder.source_sha256,
        },
        "transcription": {
            "device": "cpu",
            "dtype": "float32",
            "model_id": _Transcriber.model_id,
            "revision": _Transcriber.revision,
            "source_sha256": _Transcriber.source_sha256,
            "torchcodec_mode": _Transcriber.torchcodec_mode,
        },
    }
    assert provenance["input_sha256"] == {
        "generation_results": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
        "reference_wavs": hashlib.sha256(references_path.read_bytes()).hexdigest(),
        "generated_audio": {
            str(generated_wav): hashlib.sha256(generated_wav.read_bytes()).hexdigest(),
        },
        "reference_audio": {
            str(reference_wav): hashlib.sha256(reference_wav.read_bytes()).hexdigest(),
        },
    }
    assert provenance["case_count"] == 1
    assert provenance["complete_count"] == 1
    assert provenance["incomplete_count"] == 0


@pytest.mark.parametrize(
    ("transcriber", "embedder", "reason"),
    [
        (_Transcriber(" 。 \t"), _Embedder(), "transcript is empty"),
        (_Transcriber(), None, "embedding has zero norm"),
    ],
)
def test_run_extraction_marks_invalid_runtime_results_incomplete(
    tmp_path: Path,
    transcriber: _Transcriber,
    embedder: _Embedder | None,
    reason: str,
) -> None:
    module = _load_script()
    generation_path = tmp_path / "generation.jsonl"
    references_path = tmp_path / "references.json"
    output_path = tmp_path / "metrics.jsonl"
    generated_wav = tmp_path / "generated.wav"
    reference_wav = tmp_path / "reference.wav"
    samples = np.full(80, 0.2, dtype=np.float64)
    _write_wav(generated_wav, samples)
    _write_wav(reference_wav, samples)
    _write_jsonl(
        generation_path,
        [
            _generation_row(
                wav_path=str(generated_wav),
                wav_sha256=hashlib.sha256(generated_wav.read_bytes()).hexdigest(),
            ),
        ],
    )
    references_path.write_text(json.dumps({"anabel": [str(reference_wav)]}), encoding="utf-8")

    runtime_embedder = embedder or _ZeroEmbedder()
    exit_code = module.run_extraction(
        generation_results=generation_path,
        reference_wavs_path=references_path,
        output_path=output_path,
        provenance_path=tmp_path / "provenance.json",
        embedder=runtime_embedder,
        transcriber=transcriber,
    )

    assert exit_code == 1
    [row] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert row == {
        "case_id": "case-1",
        "checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        "checkpoint_step": 1000,
        "embedding_path": EMBEDDING_PATH,
        "embedding_sha256": EMBEDDING_SHA256,
        "evaluation_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "generation_results_sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
        "metrics_status": "INCOMPLETE",
        "incomplete_reason": reason,
        "model_id": "anabel",
        "provenance": _generation_row()["provenance"],
        "seed": 1,
        "speaker_filename": "checkpoint-1000.speaker.safetensors",
        "style": "neutral",
        "text_id": "hello",
        "wav_path": str(generated_wav),
        "wav_sha256": hashlib.sha256(generated_wav.read_bytes()).hexdigest(),
    }
    assert "speaker_similarity" not in row
    assert "normalized_cer" not in row


def test_run_extraction_preserves_failed_generation_as_incomplete(tmp_path: Path) -> None:
    module = _load_script()
    generation_path = tmp_path / "generation.jsonl"
    references_path = tmp_path / "references.json"
    reference_wav = tmp_path / "reference.wav"
    _write_wav(reference_wav, np.full(80, 0.2, dtype=np.float64))
    failed_row = _generation_row(
        case_id="failed-case",
        status="ERROR",
        wav_path=None,
        wav_sha256=None,
    )
    del failed_row["wav_path"]
    _write_jsonl(generation_path, [failed_row])
    references_path.write_text(json.dumps({"anabel": [str(reference_wav)]}), encoding="utf-8")

    output_path = tmp_path / "metrics.jsonl"
    exit_code = module.run_extraction(
        generation_results=generation_path,
        reference_wavs_path=references_path,
        output_path=output_path,
        provenance_path=tmp_path / "provenance.json",
        embedder=_Embedder(),
        transcriber=_Transcriber(),
    )

    assert exit_code == 1
    [row] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert row["metrics_status"] == "INCOMPLETE"
    assert row["incomplete_reason"] == "generation status is ERROR"
    assert row["wav_path"] is None
    assert set(row) >= {
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
        "provenance",
        "generation_results_sha256",
    }
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["case_count"] == 1
    assert provenance["complete_count"] == 0
    assert provenance["incomplete_count"] == 1
    assert provenance["input_sha256"]["generated_audio"] == {}


def test_run_extraction_marks_invalid_reference_embedding_incomplete(tmp_path: Path) -> None:
    module = _load_script()
    generation_path = tmp_path / "generation.jsonl"
    references_path = tmp_path / "references.json"
    generated_wav = tmp_path / "generated.wav"
    reference_wav = tmp_path / "reference.wav"
    samples = np.full(80, 0.2, dtype=np.float64)
    _write_wav(generated_wav, samples)
    _write_wav(reference_wav, samples)
    _write_jsonl(
        generation_path,
        [
            _generation_row(
                wav_path=str(generated_wav),
                wav_sha256=hashlib.sha256(generated_wav.read_bytes()).hexdigest(),
            ),
        ],
    )
    references_path.write_text(json.dumps({"anabel": [str(reference_wav)]}), encoding="utf-8")

    output_path = tmp_path / "metrics.jsonl"
    exit_code = module.run_extraction(
        generation_results=generation_path,
        reference_wavs_path=references_path,
        output_path=output_path,
        provenance_path=tmp_path / "provenance.json",
        embedder=_AlwaysZeroEmbedder(),
        transcriber=_Transcriber(),
    )

    assert exit_code == 1
    [row] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert row["metrics_status"] == "INCOMPLETE"
    assert row["incomplete_reason"] == "reference embedding: embedding has zero norm"
    assert "speaker_similarity" not in row


def test_run_extraction_resolves_relative_wav_from_generation_jsonl_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    generated_wav = input_dir / "audio" / "generated.wav"
    reference_wav = input_dir / "reference.wav"
    samples = np.full(80, 0.2, dtype=np.float64)
    _write_wav(generated_wav, samples)
    _write_wav(reference_wav, samples)
    generation_path = input_dir / "generation.jsonl"
    references_path = input_dir / "references.json"
    _write_jsonl(
        generation_path,
        [
            _generation_row(
                wav_path="audio/generated.wav",
                wav_sha256=hashlib.sha256(generated_wav.read_bytes()).hexdigest(),
            ),
        ],
    )
    references_path.write_text(json.dumps({"anabel": ["reference.wav"]}), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    exit_code = module.run_extraction(
        generation_results=generation_path,
        reference_wavs_path=references_path,
        output_path=tmp_path / "metrics.jsonl",
        provenance_path=tmp_path / "metrics.provenance.json",
        embedder=_Embedder(),
        transcriber=_Transcriber(),
    )

    assert exit_code == 0


@pytest.mark.parametrize(
    ("device", "expected_dtype"),
    [("cpu", "float32"), ("cuda:0", "float16")],
)
def test_whisper_loader_selects_explicit_device_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_dtype: str,
) -> None:
    module = _load_script()
    source = tmp_path / "whisper"
    source.mkdir()
    (source / "model.bin").write_bytes(b"whisper")
    calls: list[dict[str, object]] = []
    transformers = ModuleType("transformers")
    asr_module = ModuleType("transformers.pipelines.automatic_speech_recognition")
    asr_module.is_torchcodec_available = lambda: False  # type: ignore[attr-defined]

    def fake_pipeline(**kwargs: object) -> object:
        calls.append(kwargs)
        return lambda *_args, **_kwargs: {"text": "ok"}

    transformers.pipeline = fake_pipeline  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers.pipelines.automatic_speech_recognition",
        asr_module,
    )

    transcriber = module.WhisperTranscriber.load(
        source=source,
        model_id="openai/whisper-large-v3-turbo",
        revision="fixture-revision",
        device=device,
    )

    assert calls == [
        {
            "task": "automatic-speech-recognition",
            "model": str(source),
            "revision": "fixture-revision",
            "device": device,
            "dtype": expected_dtype,
        },
    ]
    assert transcriber.dtype == expected_dtype
    assert transcriber.torchcodec_mode == "unavailable"
    assert transcriber.source_sha256 == module.sha256_tree(source)


@pytest.mark.parametrize(
    "import_error", [ImportError("missing"), OSError("broken DLL"), RuntimeError("ABI mismatch")]
)
def test_whisper_loader_disables_only_asr_torchcodec_check_when_import_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    import_error: Exception,
) -> None:
    module = _load_script()
    source = tmp_path / "whisper"
    source.mkdir()
    (source / "model.bin").write_bytes(b"whisper")
    transformers = ModuleType("transformers")
    transformers_utils = ModuleType("transformers.utils")
    asr_module = ModuleType("transformers.pipelines.automatic_speech_recognition")

    def global_is_torchcodec_available() -> bool:
        return True

    def asr_is_torchcodec_available() -> bool:
        return True

    pipeline_modes: list[bool] = []

    def fake_pipeline(**_kwargs: object) -> object:
        pipeline_modes.append(asr_module.is_torchcodec_available())
        return lambda *_args, **_kwargs: {"text": "ok"}

    transformers.pipeline = fake_pipeline  # type: ignore[attr-defined]
    transformers_utils.is_torchcodec_available = global_is_torchcodec_available  # type: ignore[attr-defined]
    asr_module.is_torchcodec_available = asr_is_torchcodec_available  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[attr-defined]
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "transformers.pipelines.automatic_speech_recognition":
            return asr_module
        if name == "torchcodec":
            raise import_error
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)
    monkeypatch.setitem(sys.modules, "torch", torch)

    transcriber = module.WhisperTranscriber.load(
        source=source,
        model_id="openai/whisper-large-v3-turbo",
        revision="fixture-revision",
        device="cpu",
    )

    assert pipeline_modes == [False]
    assert transcriber.torchcodec_mode == "disabled_import_failure"
    assert asr_module.is_torchcodec_available() is False
    assert transformers_utils.is_torchcodec_available is global_is_torchcodec_available


def test_whisper_loader_keeps_asr_torchcodec_check_when_import_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source = tmp_path / "whisper"
    source.mkdir()
    (source / "model.bin").write_bytes(b"whisper")
    transformers = ModuleType("transformers")
    asr_module = ModuleType("transformers.pipelines.automatic_speech_recognition")

    def asr_is_torchcodec_available() -> bool:
        return True

    transformers.pipeline = lambda **_kwargs: (  # type: ignore[attr-defined]
        lambda *_args, **_inner_kwargs: {"text": "ok"}
    )
    asr_module.is_torchcodec_available = asr_is_torchcodec_available  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[attr-defined]
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "transformers.pipelines.automatic_speech_recognition":
            return asr_module
        if name == "torchcodec":
            return ModuleType("torchcodec")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    transcriber = module.WhisperTranscriber.load(
        source=source,
        model_id="openai/whisper-large-v3-turbo",
        revision="fixture-revision",
        device="cpu",
    )

    assert transcriber.torchcodec_mode == "available"
    assert asr_module.is_torchcodec_available is asr_is_torchcodec_available


def test_whisper_loader_does_not_import_or_patch_unavailable_torchcodec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source = tmp_path / "whisper"
    source.mkdir()
    (source / "model.bin").write_bytes(b"whisper")
    transformers = ModuleType("transformers")
    asr_module = ModuleType("transformers.pipelines.automatic_speech_recognition")

    def asr_is_torchcodec_available() -> bool:
        return False

    transformers.pipeline = lambda **_kwargs: (  # type: ignore[attr-defined]
        lambda *_args, **_inner_kwargs: {"text": "ok"}
    )
    asr_module.is_torchcodec_available = asr_is_torchcodec_available  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[attr-defined]
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "transformers.pipelines.automatic_speech_recognition":
            return asr_module
        if name == "torchcodec":
            pytest.fail("torchcodec must not be imported when Transformers reports it unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    transcriber = module.WhisperTranscriber.load(
        source=source,
        model_id="openai/whisper-large-v3-turbo",
        revision="fixture-revision",
        device="cpu",
    )

    assert transcriber.torchcodec_mode == "unavailable"
    assert asr_module.is_torchcodec_available is asr_is_torchcodec_available


def test_main_loads_whisper_before_speechbrain(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    calls: list[str] = []
    transcriber = _Transcriber()
    embedder = _Embedder()
    args = SimpleNamespace(
        generation_results=Path("generation.jsonl"),
        reference_wavs=Path("references.json"),
        output=Path("metrics.jsonl"),
        provenance_output=Path("metrics.provenance.json"),
        ecapa_source=Path("ecapa"),
        ecapa_savedir=Path("ecapa-cache"),
        ecapa_model_id=_Embedder.model_id,
        ecapa_revision=_Embedder.revision,
        whisper_model=_Transcriber.model_id,
        whisper_source=Path("whisper"),
        whisper_revision=_Transcriber.revision,
        whisper_device=_Transcriber.device,
    )

    def fake_whisper_load(**_kwargs: object) -> _Transcriber:
        calls.append("whisper")
        return transcriber

    def fake_ecapa_load(**_kwargs: object) -> _Embedder:
        calls.append("speechbrain")
        return embedder

    def fake_run_extraction(**kwargs: object) -> int:
        assert kwargs["transcriber"] is transcriber
        assert kwargs["embedder"] is embedder
        return 0

    monkeypatch.setattr(module, "parse_args", lambda _argv: args)
    monkeypatch.setattr(module.WhisperTranscriber, "load", staticmethod(fake_whisper_load))
    monkeypatch.setattr(module.SpeechBrainECAPA, "load", staticmethod(fake_ecapa_load))
    monkeypatch.setattr(module, "run_extraction", fake_run_extraction)

    assert module.main([]) == 0
    assert calls == ["whisper", "speechbrain"]


class _ZeroEmbedder(_Embedder):
    def __init__(self) -> None:
        self.calls = 0

    @override
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        assert samples.ndim == 1
        assert sample_rate == TARGET_SAMPLE_RATE
        self.calls += 1
        if self.calls == 1:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.zeros(2, dtype=np.float64)


class _AlwaysZeroEmbedder(_Embedder):
    @override
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        assert samples.ndim == 1
        assert sample_rate == TARGET_SAMPLE_RATE
        return np.zeros(2, dtype=np.float64)


def test_cli_requires_pinned_model_configuration() -> None:
    module = _load_script()

    args = module.parse_args(
        [
            "--generation-results",
            "generation.jsonl",
            "--reference-wavs",
            "references.json",
            "--output",
            "metrics.jsonl",
            "--ecapa-source",
            "/models/ecapa/snapshot",
            "--ecapa-savedir",
            "/models/ecapa/cache",
            "--ecapa-model-id",
            "speechbrain/spkrec-ecapa-voxceleb",
            "--ecapa-revision",
            "7f0c",
            "--whisper-model",
            "openai/whisper-large-v3-turbo",
            "--whisper-source",
            "/models/whisper/snapshot",
            "--whisper-revision",
            "abcdef",
            "--whisper-device",
            "cuda:0",
        ],
    )

    assert args.ecapa_source == Path("/models/ecapa/snapshot")
    assert args.ecapa_savedir == Path("/models/ecapa/cache")
    assert args.ecapa_model_id == "speechbrain/spkrec-ecapa-voxceleb"
    assert args.ecapa_revision == "7f0c"
    assert args.whisper_model == "openai/whisper-large-v3-turbo"
    assert args.whisper_source == Path("/models/whisper/snapshot")
    assert args.whisper_revision == "abcdef"
    assert args.whisper_device == "cuda:0"
    assert args.provenance_output == Path("metrics.provenance.json")
