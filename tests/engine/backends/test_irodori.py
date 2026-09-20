from __future__ import annotations

import hashlib
import importlib
import io
import os
import subprocess  # noqa: S404
import sys
import tempfile
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from irodori_tts_infra.config.settings import IrodoriRuntimeSettings
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.backends.irodori import (
    INSTALL_HINT,
    IrodoriBaseBackend,
    _encode_wav_bytes,  # noqa: PLC2701
    _runtime_factory,  # noqa: PLC2701
    _runtime_key_cls,  # noqa: PLC2701
    _sampling_request_cls,  # noqa: PLC2701
    create_irodori_backend,
)
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import ResolvedSynthesisRequest, SynthesizedAudio
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.text.models import Segment, SegmentKind
from irodori_tts_infra.voice_bank import CharacterVoice, SpeakerEmbeddingProfile, VoiceProfile

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from irodori_tts_infra.engine.backends.irodori import (
        _InferenceRuntimeModule,
    )
    from irodori_tts_infra.engine.protocols import Synthesizer

pytestmark = pytest.mark.unit

FAKE_WAV_BYTES = b"RIFF\x08\x00\x00\x00WAVEfake"
DEFAULT_SAMPLE_RATE = 24_000
PCM_16_SAMPLE_WIDTH_BYTES = 2
PCM_16_MAX = 32_767
PCM_16_MIN = -32_768
DEFAULT_NUM_STEPS = 40
DEFAULT_CFG_SCALE_TEXT = 3.0
DEFAULT_CFG_SCALE_CAPTION = 3.0
DEFAULT_CFG_SCALE_SPEAKER = 5.0
CUSTOM_STEPS = 12
CUSTOM_CFG_TEXT = 2.25
CUSTOM_CFG_CAPTION = 2.75
CUSTOM_CFG_SPEAKER = 4.25
CUSTOM_SEED = 123
CUSTOM_DURATION_SCALE = 1.25
CUSTOM_NUM_CANDIDATES = 2
CUSTOM_SWAY_COEFF = -0.5
WARMUP_STEPS = 5
CUSTOM_REVISION = "a" * 40
DEFAULT_REF_EMBED = "speakers/mika.speaker.safetensors"
NARRATOR_REF_EMBED = "speakers/narrator.speaker.safetensors"


@dataclass(frozen=True, slots=True)
class FakeRuntimeResult:
    audio: object = b"audio"
    sample_rate: int = DEFAULT_SAMPLE_RATE


class FakeSamplingRequest:
    text: str
    caption: str | None
    ref_embed: str
    num_steps: int
    cfg_scale_text: float
    cfg_scale_caption: float
    cfg_scale_speaker: float
    cfg_guidance_mode: str
    seed: int | None
    duration_scale: float
    num_candidates: int
    t_schedule_mode: str
    sway_coeff: float
    decode_mode: str
    context_kv_cache: bool

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.text = cast("str", kwargs.get("text", ""))
        self.caption = cast("str | None", kwargs.get("caption"))
        self.ref_embed = cast("str", kwargs.get("ref_embed", ""))
        self.num_steps = cast("int", kwargs.get("num_steps", 0))
        self.cfg_scale_text = cast("float", kwargs.get("cfg_scale_text", 0.0))
        self.cfg_scale_caption = cast("float", kwargs.get("cfg_scale_caption", 0.0))
        self.cfg_scale_speaker = cast("float", kwargs.get("cfg_scale_speaker", 0.0))
        self.cfg_guidance_mode = cast("str", kwargs.get("cfg_guidance_mode", ""))
        self.seed = cast("int | None", kwargs.get("seed"))
        self.duration_scale = cast("float", kwargs.get("duration_scale", 0.0))
        self.num_candidates = cast("int", kwargs.get("num_candidates", 0))
        self.t_schedule_mode = cast("str", kwargs.get("t_schedule_mode", ""))
        self.sway_coeff = cast("float", kwargs.get("sway_coeff", 0.0))
        self.decode_mode = cast("str", kwargs.get("decode_mode", ""))
        self.context_kv_cache = cast("bool", kwargs.get("context_kv_cache", False))


class FakeRuntimeKey:
    checkpoint: str
    model_device: str
    model_precision: str
    codec_device: str
    codec_precision: str
    compile_model: bool

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.checkpoint = cast("str", kwargs.get("checkpoint", ""))
        self.model_device = cast("str", kwargs.get("model_device", ""))
        self.model_precision = cast("str", kwargs.get("model_precision", ""))
        self.codec_device = cast("str", kwargs.get("codec_device", ""))
        self.codec_precision = cast("str", kwargs.get("codec_precision", ""))
        self.compile_model = cast("bool", kwargs.get("compile_model", False))


class FakeRuntime:
    def __init__(self, result: FakeRuntimeResult | None = None) -> None:
        self.result = result or FakeRuntimeResult()
        self.calls: list[FakeSamplingRequest] = []
        self.unload_count = 0

    def synthesize(self, request: object) -> FakeRuntimeResult:
        assert isinstance(request, FakeSamplingRequest)
        self.calls.append(request)
        return self.result

    def unload(self) -> None:
        self.unload_count += 1


class UnloadFailingRuntime:
    def __init__(self) -> None:
        self.calls: list[FakeSamplingRequest] = []
        self.unload_count = 0

    def synthesize(self, request: object) -> FakeRuntimeResult:
        assert isinstance(request, FakeSamplingRequest)
        self.calls.append(request)
        return FakeRuntimeResult()

    def unload(self) -> None:
        self.unload_count += 1
        msg = "unload failed"
        raise RuntimeError(msg)


class RuntimeWithoutUnload:
    def __init__(self) -> None:
        self.calls: list[FakeSamplingRequest] = []

    def synthesize(self, request: object) -> FakeRuntimeResult:
        assert isinstance(request, FakeSamplingRequest)
        self.calls.append(request)
        return FakeRuntimeResult()


class TensorLikeAudio:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.calls: list[object] = []
        self.mono_samples = object()

    def detach(self) -> TensorLikeAudio:
        self.calls.append("detach")
        return self

    def to(self, *, device: str, dtype: object) -> TensorLikeAudio:
        self.calls.append(("to", device, dtype))
        return self

    def squeeze(self, dim: int) -> TensorNumpyView:
        self.calls.append(("squeeze", dim))
        return TensorNumpyView(self.calls, self.mono_samples)


class TensorNumpyView:
    def __init__(self, calls: list[object], samples: object) -> None:
        self._calls = calls
        self._samples = samples

    def numpy(self) -> object:
        self._calls.append("numpy")
        return self._samples


class _FakeInferenceRuntime:
    RuntimeKey = object
    SamplingRequest = object

    class InferenceRuntime:
        @staticmethod
        def from_key(_key: object) -> object:
            return object()


def fake_inference_runtime_module() -> _InferenceRuntimeModule:
    return cast("_InferenceRuntimeModule", _FakeInferenceRuntime)


def runtime_settings(**overrides: object) -> IrodoriRuntimeSettings:
    data: dict[str, object] = {
        "checkpoint": "org/model",
        "checkpoint_tokenizer_json_sha256": None,
        "checkpoint_tokenizer_config_sha256": None,
        "num_steps": DEFAULT_NUM_STEPS,
        "cfg_scale_text": DEFAULT_CFG_SCALE_TEXT,
        "cfg_scale_caption": DEFAULT_CFG_SCALE_CAPTION,
        "cfg_scale_speaker": DEFAULT_CFG_SCALE_SPEAKER,
        "model_device": "cuda",
        "model_precision": "bf16",
        "codec_device": "cuda",
        "codec_precision": "fp32",
        "warmup_num_steps": DEFAULT_NUM_STEPS,
        "warmup_text": "テスト",
        "warmup_style": "calm",
        "decode_mode": "batch",
        "context_kv_cache": True,
        "compile_model": False,
    }
    data.update(overrides)
    return IrodoriRuntimeSettings.model_validate(data)


def synthesis_request(**overrides: object) -> ResolvedSynthesisRequest:
    data: dict[str, object] = {
        "text": "本文です。",
        "ref_embed": DEFAULT_REF_EMBED,
        "num_steps": CUSTOM_STEPS,
        "cfg_scale_text": CUSTOM_CFG_TEXT,
        "cfg_scale_caption": CUSTOM_CFG_CAPTION,
        "cfg_scale_speaker": CUSTOM_CFG_SPEAKER,
        "style": "clear",
        "seed": CUSTOM_SEED,
        "duration_scale": CUSTOM_DURATION_SCALE,
        "num_candidates": CUSTOM_NUM_CANDIDATES,
        "t_schedule_mode": "sway",
        "sway_coeff": CUSTOM_SWAY_COEFF,
    }
    data.update(overrides)
    return ResolvedSynthesisRequest.model_validate(data)


def fake_encode_wav(_audio: object, _sample_rate: int) -> bytes:
    return FAKE_WAV_BYTES


class NumpyFakeTorch:
    float32 = np.float32


def patch_encoder_imports(monkeypatch: pytest.MonkeyPatch, modules: dict[str, object]) -> None:
    """Replace the encoder's lazy imports. An exception value is raised; other names import."""
    real_import_module = importlib.import_module

    def import_module(name: str) -> object:
        if name not in modules:
            return real_import_module(name)
        module = modules[name]
        if isinstance(module, Exception):
            raise module
        return module

    monkeypatch.setattr(
        "irodori_tts_infra.engine.backends.irodori.importlib.import_module",
        import_module,
    )


def make_backend(
    runtime: FakeRuntime | RuntimeWithoutUnload | UnloadFailingRuntime | None = None,
    *,
    settings: IrodoriRuntimeSettings | None = None,
    encode_wav_fn: Callable[[object, int], bytes] = fake_encode_wav,
) -> IrodoriBaseBackend:
    return IrodoriBaseBackend(
        runtime=runtime or FakeRuntime(),
        settings=settings or runtime_settings(),
        encode_wav_fn=encode_wav_fn,
        sampling_request_cls=FakeSamplingRequest,
    )


def make_profile() -> VoiceProfile:
    return VoiceProfile(
        characters={
            "ミカ": CharacterVoice(
                name="ミカ",
                speaker=SpeakerEmbeddingProfile(DEFAULT_REF_EMBED),  # type: ignore[arg-type]
            ),
        },
        narrator=SpeakerEmbeddingProfile(NARRATOR_REF_EMBED),  # type: ignore[arg-type]
    )


def test_backend_implements_synthesizer_protocol() -> None:
    synth: Synthesizer = IrodoriBaseBackend(
        runtime=FakeRuntime(),
        settings=runtime_settings(),
        encode_wav_fn=fake_encode_wav,
        sampling_request_cls=FakeSamplingRequest,
    )

    assert callable(synth.synthesize)


def test_synthesize_forwards_sampling_request_fields() -> None:
    runtime = FakeRuntime()
    settings = runtime_settings(decode_mode="sequential", context_kv_cache=False)
    backend = make_backend(runtime, settings=settings)

    backend.synthesize(synthesis_request())

    call = runtime.calls[0]
    assert call.text == "本文です。"
    assert call.ref_embed == DEFAULT_REF_EMBED
    assert call.num_steps == CUSTOM_STEPS
    assert call.cfg_scale_text == pytest.approx(CUSTOM_CFG_TEXT)
    assert call.caption == "子どもに伝わるように、ゆっくり明瞭な女性の声で話す。"
    assert call.cfg_scale_caption == pytest.approx(CUSTOM_CFG_CAPTION)
    assert call.cfg_scale_speaker == pytest.approx(CUSTOM_CFG_SPEAKER)
    assert call.cfg_guidance_mode == "independent"
    assert call.seed == CUSTOM_SEED
    assert call.duration_scale == pytest.approx(CUSTOM_DURATION_SCALE)
    assert call.num_candidates == CUSTOM_NUM_CANDIDATES
    assert call.t_schedule_mode == "sway"
    assert call.sway_coeff == pytest.approx(CUSTOM_SWAY_COEFF)
    assert call.decode_mode == "sequential"
    assert call.context_kv_cache is False


def test_synthesize_normalizes_ref_embed_before_sampling_request() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)

    backend.synthesize(synthesis_request(ref_embed=f"  {DEFAULT_REF_EMBED}  "))

    assert runtime.calls[0].ref_embed == DEFAULT_REF_EMBED


def test_synthesize_neutral_style_omits_caption() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)

    backend.synthesize(synthesis_request(style="neutral"))

    assert runtime.calls[0].caption is None


def test_synthesize_forwards_freeform_delivery_caption_without_preset() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)

    backend.synthesize(
        synthesis_request(
            style="neutral",
            delivery_caption="雨の夜、耳元で囁くように話す。",
        )
    )

    assert runtime.calls[0].caption == "雨の夜、耳元で囁くように話す。"


def test_synthesize_uses_in_memory_encoder() -> None:
    runtime_audio = object()
    encoder_calls: list[tuple[object, int]] = []

    def encode_wav(audio: object, sample_rate: int) -> bytes:
        encoder_calls.append((audio, sample_rate))
        return FAKE_WAV_BYTES

    backend = make_backend(
        FakeRuntime(FakeRuntimeResult(audio=runtime_audio, sample_rate=48_000)),
        encode_wav_fn=encode_wav,
    )

    audio = backend.synthesize(synthesis_request())

    assert encoder_calls == [(runtime_audio, 48_000)]
    assert audio == SynthesizedAudio(wav_bytes=FAKE_WAV_BYTES, sample_rate=48_000)


def test_encoder_failure_propagates() -> None:
    def encode_wav(_audio: object, _sample_rate: int) -> bytes:
        msg = "encoder failed"
        raise RuntimeError(msg)

    backend = make_backend(encode_wav_fn=encode_wav)

    with pytest.raises(RuntimeError, match="encoder failed"):
        backend.synthesize(synthesis_request())


def test_warm_up_requires_ref_embed() -> None:
    backend = make_backend()

    with pytest.raises(BackendUnavailableError, match="warmup ref_embed is required"):
        backend.warm_up()


def test_warm_up_uses_warmup_settings_and_ref_embed() -> None:
    runtime = FakeRuntime()
    settings = runtime_settings(
        warmup_text="準備です。",
        warmup_num_steps=WARMUP_STEPS,
    )
    backend = make_backend(runtime, settings=settings)

    backend.warm_up(ref_embed=NARRATOR_REF_EMBED)

    call = runtime.calls[0]
    assert call.text == "準備です。"
    assert call.ref_embed == NARRATOR_REF_EMBED
    assert call.num_steps == WARMUP_STEPS
    assert call.caption == "穏やかで優しい女性の声で、自然に話す。"
    assert call.cfg_scale_caption == pytest.approx(DEFAULT_CFG_SCALE_CAPTION)
    assert call.cfg_scale_speaker == pytest.approx(DEFAULT_CFG_SCALE_SPEAKER)
    assert call.cfg_guidance_mode == "independent"


def test_warm_up_normalizes_ref_embed_before_sampling_request() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)

    backend.warm_up(ref_embed=f"  {NARRATOR_REF_EMBED}  ")

    assert runtime.calls[0].ref_embed == NARRATOR_REF_EMBED


def test_close_marks_backend_unavailable() -> None:
    backend = make_backend()

    backend.close()

    with pytest.raises(BackendUnavailableError, match="backend is closed"):
        backend.synthesize(synthesis_request())


def test_warm_up_after_close_raises_backend_unavailable() -> None:
    backend = make_backend()

    backend.close()

    with pytest.raises(BackendUnavailableError, match="backend is closed"):
        backend.warm_up()


def test_close_marks_backend_closed_when_unload_raises() -> None:
    backend = make_backend(UnloadFailingRuntime())

    with pytest.raises(RuntimeError, match="unload failed"):
        backend.close()

    with pytest.raises(BackendUnavailableError, match="backend is closed"):
        backend.synthesize(synthesis_request())
    with pytest.raises(BackendUnavailableError, match="backend is closed"):
        backend.warm_up()


def test_close_calls_runtime_unload_once() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)

    backend.close()
    backend.close()

    assert runtime.unload_count == 1


def test_close_tolerates_runtime_without_unload() -> None:
    backend = make_backend(RuntimeWithoutUnload())

    backend.close()

    with pytest.raises(BackendUnavailableError, match="backend is closed"):
        backend.synthesize(synthesis_request())
    with pytest.raises(BackendUnavailableError, match="backend is closed"):
        backend.warm_up()


def test_factory_uses_injected_download_and_runtime_factory(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    checkpoint = snapshot / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    settings = runtime_settings(
        checkpoint_revision=CUSTOM_REVISION,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    runtime = FakeRuntime()
    download_calls: list[dict[str, object]] = []
    runtime_keys: list[FakeRuntimeKey] = []

    def snapshot_download_fn(**kwargs: object) -> str:
        download_calls.append(kwargs)
        return str(snapshot)

    def runtime_factory(key: object) -> FakeRuntime:
        assert isinstance(key, FakeRuntimeKey)
        runtime_keys.append(key)
        return runtime

    backend = create_irodori_backend(
        settings,
        snapshot_download_fn=snapshot_download_fn,
        runtime_factory=runtime_factory,
        runtime_key_cls=FakeRuntimeKey,
        encode_wav_fn=fake_encode_wav,
        sampling_request_cls=FakeSamplingRequest,
    )

    assert isinstance(backend, IrodoriBaseBackend)
    assert download_calls == [
        {
            "repo_id": "org/model",
            "revision": CUSTOM_REVISION,
            "allow_patterns": ["model.safetensors", "tokenizer/*"],
        },
    ]
    assert runtime_keys[0].checkpoint == str(checkpoint)
    assert runtime_keys[0].model_device == "cuda"
    assert runtime_keys[0].model_precision == "bf16"
    assert runtime_keys[0].codec_device == "cuda"
    assert runtime_keys[0].codec_precision == "fp32"


def test_factory_rejects_checkpoint_sha_mismatch_before_runtime_creation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"unexpected checkpoint")
    runtime_keys: list[object] = []

    def runtime_factory(key: object) -> FakeRuntime:
        runtime_keys.append(key)
        return FakeRuntime()

    with pytest.raises(BackendUnavailableError, match="checkpoint SHA-256 mismatch"):
        create_irodori_backend(
            runtime_settings(
                checkpoint_revision=CUSTOM_REVISION,
                checkpoint_sha256="0" * 64,
            ),
            snapshot_download_fn=lambda **_kwargs: str(tmp_path),
            runtime_factory=runtime_factory,
            runtime_key_cls=FakeRuntimeKey,
            encode_wav_fn=fake_encode_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert runtime_keys == []


@pytest.mark.parametrize("failure", ["missing", "mismatch"])
def test_factory_rejects_invalid_bundled_tokenizer_before_runtime_creation(
    tmp_path: Path,
    failure: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    tokenizer_dir = snapshot / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    checkpoint = snapshot / "model.safetensors"
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    checkpoint.write_bytes(b"v4 checkpoint")
    tokenizer_json.write_bytes(b"v4 tokenizer")
    tokenizer_config.write_bytes(b"v4 tokenizer config")
    settings = runtime_settings(
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_tokenizer_json_sha256=hashlib.sha256(tokenizer_json.read_bytes()).hexdigest(),
        checkpoint_tokenizer_config_sha256=hashlib.sha256(
            tokenizer_config.read_bytes()
        ).hexdigest(),
    )
    runtime_keys: list[FakeRuntimeKey] = []
    if failure == "missing":
        tokenizer_json.unlink()
        expected = "bundled tokenizer.json is missing or unreadable"
    else:
        tokenizer_json.write_bytes(b"tampered tokenizer")
        expected = "bundled tokenizer.json SHA-256 mismatch"

    def runtime_factory(key: object) -> FakeRuntime:
        runtime_keys.append(cast("FakeRuntimeKey", key))
        return FakeRuntime()

    with pytest.raises(BackendUnavailableError, match=expected):
        create_irodori_backend(
            settings,
            snapshot_download_fn=lambda **_kwargs: str(snapshot),
            runtime_factory=runtime_factory,
            runtime_key_cls=FakeRuntimeKey,
            encode_wav_fn=fake_encode_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert runtime_keys == []


def test_runtime_key_cls_falls_back_to_module_attr() -> None:
    module = fake_inference_runtime_module()

    assert _runtime_key_cls(None, module) is _FakeInferenceRuntime.RuntimeKey


def test_runtime_factory_falls_back_to_module_attr() -> None:
    module = fake_inference_runtime_module()
    resolved = _runtime_factory(None, module)

    assert resolved is module.InferenceRuntime.from_key


def test_sampling_request_cls_falls_back_to_module_attr() -> None:
    module = fake_inference_runtime_module()

    assert _sampling_request_cls(None, module) is _FakeInferenceRuntime.SamplingRequest


def test_factory_raises_backend_unavailable_on_missing_optional_deps() -> None:
    code = (
        "import sys\n"
        "sys.modules['huggingface_hub'] = None\n"
        "from irodori_tts_infra.engine.backends.irodori import create_irodori_backend\n"
        "from irodori_tts_infra.config.settings import IrodoriRuntimeSettings\n"
        "try:\n"
        "    create_irodori_backend(IrodoriRuntimeSettings())\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
        "    print(str(exc))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )

    assert "BackendUnavailableError" in result.stdout
    assert "pip install" in result.stdout or "irodori" in result.stdout.lower()


def test_install_hint_lists_packages_without_nonexistent_extra() -> None:
    assert "irodori-tts-infra[irodori]" not in INSTALL_HINT
    assert "irodori-tts" in INSTALL_HINT
    assert "huggingface-hub" in INSTALL_HINT
    assert "torch" in INSTALL_HINT
    assert "soundfile" in INSTALL_HINT


def test_factory_wraps_snapshot_download_failure() -> None:
    error = OSError("network down")

    def download_fn(**_kwargs: object) -> str:
        raise error

    with pytest.raises(BackendUnavailableError, match="Failed to create Irodori") as exc_info:
        create_irodori_backend(
            runtime_settings(),
            snapshot_download_fn=download_fn,
            runtime_factory=lambda _key: FakeRuntime(),
            runtime_key_cls=FakeRuntimeKey,
            encode_wav_fn=fake_encode_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value.__cause__ is error


def test_factory_wraps_runtime_factory_failure(tmp_path: Path) -> None:
    error = RuntimeError("runtime failed")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")

    def runtime_factory(_key: object) -> FakeRuntime:
        raise error

    with pytest.raises(BackendUnavailableError, match="Failed to create Irodori") as exc_info:
        create_irodori_backend(
            runtime_settings(
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            ),
            snapshot_download_fn=lambda **_kwargs: str(tmp_path),
            runtime_factory=runtime_factory,
            runtime_key_cls=FakeRuntimeKey,
            encode_wav_fn=fake_encode_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value.__cause__ is error


def test_factory_does_not_wrap_runtime_factory_type_error(tmp_path: Path) -> None:
    error = TypeError("wrong runtime factory signature")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")

    def runtime_factory(_key: object) -> FakeRuntime:
        raise error

    with pytest.raises(TypeError, match="wrong runtime factory signature") as exc_info:
        create_irodori_backend(
            runtime_settings(
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            ),
            snapshot_download_fn=lambda **_kwargs: str(tmp_path),
            runtime_factory=runtime_factory,
            runtime_key_cls=FakeRuntimeKey,
            encode_wav_fn=fake_encode_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value is error


@pytest.mark.parametrize("missing_module", ["torch", "soundfile"])
def test_factory_rejects_missing_default_encoder_dependency_before_loading_model(
    missing_module: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ModuleNotFoundError(f"No module named {missing_module!r}")

    def download_fn(**_kwargs: object) -> str:
        pytest.fail("the model must not be downloaded when the encoder cannot work")

    patch_encoder_imports(
        monkeypatch,
        {"torch": object(), "soundfile": object(), missing_module: error},
    )

    with pytest.raises(BackendUnavailableError) as exc_info:
        create_irodori_backend(
            runtime_settings(),
            snapshot_download_fn=download_fn,
            runtime_factory=lambda _key: FakeRuntime(),
            runtime_key_cls=FakeRuntimeKey,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert str(exc_info.value) == INSTALL_HINT
    assert exc_info.value.__cause__ is error


def test_factory_does_not_wrap_injected_download_import_error() -> None:
    error = ImportError("transitive import failed")

    def download_fn(**_kwargs: object) -> str:
        raise error

    with pytest.raises(ImportError, match="transitive import failed") as exc_info:
        create_irodori_backend(
            runtime_settings(),
            snapshot_download_fn=download_fn,
            runtime_factory=lambda _key: FakeRuntime(),
            runtime_key_cls=FakeRuntimeKey,
            encode_wav_fn=fake_encode_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value is error


def test_importing_irodori_backend_is_lightweight() -> None:
    code = (
        "import sys\n"
        "import irodori_tts_infra.engine.backends.irodori\n"
        "blocked = {'irodori_tts', 'huggingface_hub', 'torch', 'soundfile'}\n"
        "loaded = blocked & set(sys.modules)\n"
        "assert not loaded, f'heavy modules loaded: {loaded}'\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("field", "consumer"),
    [
        ("checkpoint", "snapshot_download"),
        ("model_device", "runtime_key"),
        ("model_precision", "runtime_key"),
        ("codec_device", "runtime_key"),
        ("codec_precision", "runtime_key"),
        ("compile_model", "runtime_key"),
        ("decode_mode", "sampling_request"),
        ("context_kv_cache", "sampling_request"),
        ("num_steps", "sampling_request"),
        ("cfg_scale_text", "sampling_request"),
        ("cfg_scale_speaker", "sampling_request"),
        ("seed", "sampling_request"),
        ("duration_scale", "sampling_request"),
        ("num_candidates", "sampling_request"),
        ("t_schedule_mode", "sampling_request"),
        ("sway_coeff", "sampling_request"),
        ("warmup_num_steps", "warmup_request"),
        ("warmup_text", "warmup_request"),
    ],
)
def test_all_runtime_settings_reach_expected_consumer(
    field: str,
    consumer: str,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "custom.bin"
    checkpoint.write_bytes(b"custom checkpoint")
    settings = runtime_settings(
        checkpoint="custom/repo",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        model_device="cpu",
        model_precision="fp32",
        codec_device="mps",
        codec_precision="fp16",
        compile_model=True,
        decode_mode="sequential",
        context_kv_cache=False,
        warmup_num_steps=7,
        warmup_text="ウォームアップ本文。",
    )
    download_calls: list[dict[str, object]] = []
    runtime_keys: list[FakeRuntimeKey] = []
    runtime = FakeRuntime()

    def download_fn(**kwargs: object) -> str:
        download_calls.append(kwargs)
        return str(tmp_path)

    def runtime_factory(key: object) -> FakeRuntime:
        assert isinstance(key, FakeRuntimeKey)
        runtime_keys.append(key)
        return runtime

    backend = create_irodori_backend(
        settings,
        checkpoint_filename="custom.bin",
        snapshot_download_fn=download_fn,
        runtime_factory=runtime_factory,
        runtime_key_cls=FakeRuntimeKey,
        encode_wav_fn=fake_encode_wav,
        sampling_request_cls=FakeSamplingRequest,
    )
    backend.synthesize(
        synthesis_request(
            num_steps=11,
            cfg_scale_text=1.5,
            cfg_scale_speaker=2.5,
            seed=987,
            duration_scale=1.5,
            num_candidates=3,
            t_schedule_mode="sway",
            sway_coeff=-0.25,
        ),
    )
    backend.warm_up(ref_embed=NARRATOR_REF_EMBED)

    expected = {
        "checkpoint": ("custom/repo", download_calls[0]["repo_id"]),
        "model_device": ("cpu", runtime_keys[0].model_device),
        "model_precision": ("fp32", runtime_keys[0].model_precision),
        "codec_device": ("mps", runtime_keys[0].codec_device),
        "codec_precision": ("fp16", runtime_keys[0].codec_precision),
        "compile_model": (True, runtime_keys[0].compile_model),
        "decode_mode": ("sequential", runtime.calls[0].decode_mode),
        "context_kv_cache": (False, runtime.calls[0].context_kv_cache),
        "num_steps": (11, runtime.calls[0].num_steps),
        "cfg_scale_text": (1.5, runtime.calls[0].cfg_scale_text),
        "cfg_scale_speaker": (2.5, runtime.calls[0].cfg_scale_speaker),
        "seed": (987, runtime.calls[0].seed),
        "duration_scale": (1.5, runtime.calls[0].duration_scale),
        "num_candidates": (3, runtime.calls[0].num_candidates),
        "t_schedule_mode": ("sway", runtime.calls[0].t_schedule_mode),
        "sway_coeff": (-0.25, runtime.calls[0].sway_coeff),
        "warmup_num_steps": (7, runtime.calls[1].num_steps),
        "warmup_text": ("ウォームアップ本文。", runtime.calls[1].text),
    }

    assert consumer in {"snapshot_download", "runtime_key", "sampling_request", "warmup_request"}
    assert expected[field][1] == expected[field][0]


def test_contract_mapping_defaults_round_trip() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)

    backend.synthesize(ResolvedSynthesisRequest(text="本文", ref_embed=DEFAULT_REF_EMBED))

    call = runtime.calls[0]
    assert call.num_steps == DEFAULT_NUM_STEPS
    assert call.cfg_scale_text == pytest.approx(DEFAULT_CFG_SCALE_TEXT)
    assert call.cfg_scale_speaker == pytest.approx(DEFAULT_CFG_SCALE_SPEAKER)
    assert call.seed is None
    assert call.duration_scale == pytest.approx(1.0)
    assert call.num_candidates == 1
    assert call.t_schedule_mode == "linear"
    assert call.sway_coeff == pytest.approx(-1.0)


def test_request_mapping_is_deterministic() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)
    request = synthesis_request()

    backend.synthesize(request)
    backend.synthesize(request)

    assert runtime.calls[0].kwargs == runtime.calls[1].kwargs


def test_pipeline_can_swap_irodori_backend_and_fake_backend() -> None:
    segment = Segment(kind=SegmentKind.NARRATION, text="地の文です。")
    irodori_pipeline = SynthesisPipeline(make_backend(), make_profile())
    fake_pipeline = SynthesisPipeline(FakeSynthesizer(), make_profile())

    irodori_result = irodori_pipeline.synthesize_batch([segment])
    fake_result = fake_pipeline.synthesize_batch([segment])

    assert len(irodori_result.results) == 1
    assert len(fake_result.results) == 1
    assert irodori_result.results[0].wav_bytes
    assert fake_result.results[0].wav_bytes


def test_multiple_backends_coexist_independently() -> None:
    first_runtime = FakeRuntime()
    second_runtime = FakeRuntime()
    first = make_backend(first_runtime, settings=runtime_settings(checkpoint="first/repo"))
    second = make_backend(second_runtime, settings=runtime_settings(checkpoint="second/repo"))

    first.synthesize(synthesis_request(text="一つ目"))
    second.synthesize(synthesis_request(text="二つ目"))
    first.close()
    second.synthesize(synthesis_request(text="三つ目"))

    assert first_runtime.calls[0].text == "一つ目"
    assert [call.text for call in second_runtime.calls] == ["二つ目", "三つ目"]


def test_default_encoder_writes_normalized_samples_to_wav_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    float32 = object()
    audio = TensorLikeAudio((1, 3))
    writes: list[tuple[object, object, int, str, str]] = []

    class FakeTorch:
        float32: object

    FakeTorch.float32 = float32

    class FakeSoundFile:
        @staticmethod
        def write(
            file: object,
            data: object,
            samplerate: int,
            *,
            format: str,  # noqa: A002 - matches soundfile.write's keyword
            subtype: str,
        ) -> None:
            assert isinstance(file, io.BytesIO)
            writes.append((file, data, samplerate, format, subtype))
            file.write(FAKE_WAV_BYTES)

    patch_encoder_imports(monkeypatch, {"torch": FakeTorch, "soundfile": FakeSoundFile})

    result = _encode_wav_bytes(audio, DEFAULT_SAMPLE_RATE)

    samples = audio.mono_samples
    assert result == FAKE_WAV_BYTES
    assert writes == [(writes[0][0], samples, DEFAULT_SAMPLE_RATE, "WAV", "PCM_16")]
    assert audio.calls == [
        "detach",
        ("to", "cpu", float32),
        ("squeeze", 0),
        "numpy",
    ]


def test_backend_default_encoder_returns_real_in_memory_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel_first_samples = np.asarray(
        [[-0.5, 0.0, 0.5]],
        dtype=np.float32,
    )
    audio = TensorLikeAudio(channel_first_samples.shape)
    audio.mono_samples = channel_first_samples[0]

    def fail_named_temporary_file(*_args: object, **_kwargs: object) -> None:
        pytest.fail("synthesis must not create a temporary WAV file")

    patch_encoder_imports(monkeypatch, {"torch": NumpyFakeTorch})
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_named_temporary_file)
    backend = IrodoriBaseBackend(
        runtime=FakeRuntime(
            FakeRuntimeResult(audio=audio, sample_rate=DEFAULT_SAMPLE_RATE),
        ),
        settings=runtime_settings(),
        sampling_request_cls=FakeSamplingRequest,
    )

    result = backend.synthesize(synthesis_request())

    assert result.wav_bytes[:4] == b"RIFF"
    assert result.wav_bytes[8:12] == b"WAVE"
    with wave.open(io.BytesIO(result.wav_bytes), "rb") as reader:
        assert reader.getframerate() == DEFAULT_SAMPLE_RATE
        assert reader.getnframes() == channel_first_samples.shape[1]
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == PCM_16_SAMPLE_WIDTH_BYTES


def test_default_encoder_saturates_out_of_range_samples_instead_of_wrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = TensorLikeAudio((1, 4))
    audio.mono_samples = np.asarray([1.02, 1.5, -1.02, -1.5], dtype=np.float32)
    patch_encoder_imports(monkeypatch, {"torch": NumpyFakeTorch})

    wav_bytes = _encode_wav_bytes(audio, DEFAULT_SAMPLE_RATE)

    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        frames = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    assert frames.tolist() == [PCM_16_MAX, PCM_16_MAX, PCM_16_MIN, PCM_16_MIN]


@pytest.mark.parametrize("shape", [(3,), (1, 1, 3), (2, 24_000), (4, 24_000), (24_000, 1)])
def test_default_encoder_rejects_audio_that_is_not_one_mono_candidate(
    shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = TensorLikeAudio(shape)
    patch_encoder_imports(monkeypatch, {"torch": NumpyFakeTorch, "soundfile": object()})

    with pytest.raises(BackendUnavailableError, match=r"\(1, samples\)") as exc_info:
        _encode_wav_bytes(audio, DEFAULT_SAMPLE_RATE)

    assert str(shape) in str(exc_info.value)
    assert "numpy" not in audio.calls


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'soundfile'"),
        OSError("sndfile library not found"),
    ],
)
@pytest.mark.parametrize("missing_module", ["torch", "soundfile"])
def test_default_encoder_reports_missing_dependency_as_backend_unavailable(
    missing_module: str,
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_encoder_imports(
        monkeypatch,
        {"torch": object(), "soundfile": object(), missing_module: error},
    )

    with pytest.raises(BackendUnavailableError) as exc_info:
        _encode_wav_bytes(TensorLikeAudio((1, 3)), DEFAULT_SAMPLE_RATE)

    assert str(exc_info.value) == INSTALL_HINT
    assert exc_info.value.__cause__ is error
