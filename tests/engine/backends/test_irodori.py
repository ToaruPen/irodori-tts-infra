from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from irodori_tts_infra.config.settings import IrodoriRuntimeSettings
from irodori_tts_infra.contracts.synthesis import SynthesisRequest
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer
from irodori_tts_infra.engine.backends.irodori import (
    INSTALL_HINT,
    IrodoriVoiceDesignBackend,
    _runtime_factory,  # noqa: PLC2701
    _runtime_key_cls,  # noqa: PLC2701
    _sampling_request_cls,  # noqa: PLC2701
    _save_wav_fn,  # noqa: PLC2701
    create_irodori_backend,
)
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import SynthesizedAudio
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.text.models import Segment, SegmentKind
from irodori_tts_infra.voice_bank import CharacterVoice, VoiceProfile

if TYPE_CHECKING:
    from collections.abc import Callable

    from irodori_tts_infra.engine.backends.irodori import (
        _InferenceRuntimeModule,
    )
    from irodori_tts_infra.engine.protocols import Synthesizer

pytestmark = pytest.mark.unit

FAKE_WAV_BYTES = b"RIFF\x08\x00\x00\x00WAVEfake"
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_NUM_STEPS = 30
DEFAULT_CFG_SCALE_TEXT = 3.0
DEFAULT_CFG_SCALE_CAPTION = 3.5
CUSTOM_STEPS = 12
CUSTOM_CFG_TEXT = 2.25
CUSTOM_CFG_CAPTION = 4.25
WARMUP_STEPS = 5
MODEL_PATH = "downloaded/model.safetensors"
CUSTOM_MODEL_PATH = "downloaded/custom.safetensors"


@dataclass(frozen=True, slots=True)
class FakeRuntimeResult:
    audio: object = b"audio"
    sample_rate: int = DEFAULT_SAMPLE_RATE


class FakeSamplingRequest:
    text: str
    caption: str
    no_ref: bool
    num_steps: int
    cfg_scale_text: float
    cfg_scale_caption: float
    decode_mode: str
    context_kv_cache: bool

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.text = cast("str", kwargs.get("text", ""))
        self.caption = cast("str", kwargs.get("caption", ""))
        self.no_ref = cast("bool", kwargs.get("no_ref", False))
        self.num_steps = cast("int", kwargs.get("num_steps", 0))
        self.cfg_scale_text = cast("float", kwargs.get("cfg_scale_text", 0.0))
        self.cfg_scale_caption = cast("float", kwargs.get("cfg_scale_caption", 0.0))
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
    def __init__(self) -> None:
        self.detach_count = 0
        self.cpu_count = 0

    def detach(self) -> TensorLikeAudio:
        self.detach_count += 1
        return self

    def cpu(self) -> TensorLikeAudio:
        self.cpu_count += 1
        return self


class _FakeInferenceRuntime:
    RuntimeKey = object
    SamplingRequest = object

    class InferenceRuntime:
        @staticmethod
        def from_key(_key: object) -> object:
            return object()

    @staticmethod
    def save_wav(_path: str, _audio: object, _sample_rate: int) -> None:
        return None


def fake_inference_runtime_module() -> _InferenceRuntimeModule:
    return cast("_InferenceRuntimeModule", _FakeInferenceRuntime)


def runtime_settings(**overrides: object) -> IrodoriRuntimeSettings:
    data: dict[str, object] = {
        "checkpoint": "org/model",
        "num_steps": DEFAULT_NUM_STEPS,
        "cfg_scale_text": DEFAULT_CFG_SCALE_TEXT,
        "cfg_scale_caption": DEFAULT_CFG_SCALE_CAPTION,
        "model_device": "cuda",
        "model_precision": "bf16",
        "codec_device": "cuda",
        "codec_precision": "fp32",
        "warmup_num_steps": DEFAULT_NUM_STEPS,
        "warmup_text": "テスト",
        "warmup_caption": "女性が話している。",
        "decode_mode": "batch",
        "context_kv_cache": True,
        "compile_model": False,
    }
    data.update(overrides)
    return IrodoriRuntimeSettings.model_validate(data)


def synthesis_request(**overrides: object) -> SynthesisRequest:
    data: dict[str, object] = {
        "text": "本文です。",
        "caption": "女性が自然に話している。",
        "num_steps": CUSTOM_STEPS,
        "cfg_scale_text": CUSTOM_CFG_TEXT,
        "cfg_scale_caption": CUSTOM_CFG_CAPTION,
        "no_ref": False,
    }
    data.update(overrides)
    return SynthesisRequest.model_validate(data)


def fake_save_wav(path: str, _audio: object, _sample_rate: int) -> None:
    Path(path).write_bytes(FAKE_WAV_BYTES)


def make_backend(
    runtime: FakeRuntime | RuntimeWithoutUnload | UnloadFailingRuntime | None = None,
    *,
    settings: IrodoriRuntimeSettings | None = None,
    save_wav_fn: Callable[[str, object, int], None] = fake_save_wav,
) -> IrodoriVoiceDesignBackend:
    return IrodoriVoiceDesignBackend(
        runtime=runtime or FakeRuntime(),
        settings=settings or runtime_settings(),
        save_wav_fn=save_wav_fn,
        sampling_request_cls=FakeSamplingRequest,
    )


def make_profile() -> VoiceProfile:
    return VoiceProfile(
        characters={
            "ミカ": CharacterVoice(
                name="ミカ",
                caption="若い女性が明るく話している。",
            ),
        },
        narrator_caption="落ち着いた声で読み上げている。",
        generic_dialogue_caption="自然な口調で話している。",
    )


def test_backend_implements_synthesizer_protocol() -> None:
    synth: Synthesizer = IrodoriVoiceDesignBackend(
        runtime=FakeRuntime(),
        settings=runtime_settings(),
        save_wav_fn=fake_save_wav,
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
    assert call.caption == "女性が自然に話している。"
    assert call.num_steps == CUSTOM_STEPS
    assert call.cfg_scale_text == pytest.approx(CUSTOM_CFG_TEXT)
    assert call.cfg_scale_caption == pytest.approx(CUSTOM_CFG_CAPTION)
    assert call.no_ref is False
    assert call.decode_mode == "sequential"
    assert call.context_kv_cache is False


def test_synthesize_produces_audio_with_fake_save_wav_bytes() -> None:
    backend = make_backend(FakeRuntime(FakeRuntimeResult(sample_rate=48_000)))

    audio = backend.synthesize(synthesis_request())

    assert audio == SynthesizedAudio(wav_bytes=FAKE_WAV_BYTES, sample_rate=48_000)


def test_temp_wav_file_is_deleted_after_synthesize() -> None:
    paths: list[Path] = []

    def save_wav(path: str, _audio: object, _sample_rate: int) -> None:
        wav_path = Path(path)
        paths.append(wav_path)
        wav_path.write_bytes(FAKE_WAV_BYTES)

    backend = make_backend(save_wav_fn=save_wav)

    backend.synthesize(synthesis_request())

    assert paths
    assert not paths[0].exists()


def test_temp_wav_file_is_deleted_when_save_wav_raises() -> None:
    paths: list[Path] = []

    def save_wav(path: str, _audio: object, _sample_rate: int) -> None:
        wav_path = Path(path)
        paths.append(wav_path)
        wav_path.write_bytes(b"partial")
        msg = "encoder failed"
        raise RuntimeError(msg)

    backend = make_backend(save_wav_fn=save_wav)

    with pytest.raises(RuntimeError, match="encoder failed"):
        backend.synthesize(synthesis_request())

    assert paths
    assert not paths[0].exists()


def test_warm_up_uses_warmup_settings() -> None:
    runtime = FakeRuntime()
    settings = runtime_settings(
        warmup_text="準備です。",
        warmup_caption="落ち着いて話している。",
        warmup_num_steps=WARMUP_STEPS,
    )
    backend = make_backend(runtime, settings=settings)

    backend.warm_up()

    call = runtime.calls[0]
    assert call.text == "準備です。"
    assert call.caption == "落ち着いて話している。"
    assert call.num_steps == WARMUP_STEPS


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


def test_factory_uses_injected_download_and_runtime_factory() -> None:
    settings = runtime_settings()
    runtime = FakeRuntime()
    download_calls: list[dict[str, object]] = []
    runtime_keys: list[FakeRuntimeKey] = []

    def download_fn(**kwargs: object) -> str:
        download_calls.append(kwargs)
        return MODEL_PATH

    def runtime_factory(key: object) -> FakeRuntime:
        assert isinstance(key, FakeRuntimeKey)
        runtime_keys.append(key)
        return runtime

    backend = create_irodori_backend(
        settings,
        hf_hub_download_fn=download_fn,
        runtime_factory=runtime_factory,
        runtime_key_cls=FakeRuntimeKey,
        save_wav_fn=fake_save_wav,
        sampling_request_cls=FakeSamplingRequest,
    )

    assert isinstance(backend, IrodoriVoiceDesignBackend)
    assert download_calls == [{"repo_id": "org/model", "filename": "model.safetensors"}]
    assert runtime_keys[0].checkpoint == MODEL_PATH
    assert runtime_keys[0].model_device == "cuda"
    assert runtime_keys[0].model_precision == "bf16"
    assert runtime_keys[0].codec_device == "cuda"
    assert runtime_keys[0].codec_precision == "fp32"


def test_runtime_key_cls_falls_back_to_module_attr() -> None:
    module = fake_inference_runtime_module()

    assert _runtime_key_cls(None, module) is _FakeInferenceRuntime.RuntimeKey


def test_runtime_factory_falls_back_to_module_attr() -> None:
    resolved = _runtime_factory(None, fake_inference_runtime_module())

    assert callable(resolved)


def test_save_wav_fn_falls_back_to_module_attr() -> None:
    module = fake_inference_runtime_module()

    assert _save_wav_fn(None, module) is _FakeInferenceRuntime.save_wav


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


def test_factory_wraps_hf_download_failure() -> None:
    error = OSError("network down")

    def download_fn(**_kwargs: object) -> str:
        raise error

    with pytest.raises(BackendUnavailableError, match="Failed to create Irodori") as exc_info:
        create_irodori_backend(
            runtime_settings(),
            hf_hub_download_fn=download_fn,
            runtime_factory=lambda _key: FakeRuntime(),
            runtime_key_cls=FakeRuntimeKey,
            save_wav_fn=fake_save_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value.__cause__ is error


def test_factory_wraps_runtime_factory_failure() -> None:
    error = RuntimeError("runtime failed")

    def runtime_factory(_key: object) -> FakeRuntime:
        raise error

    with pytest.raises(BackendUnavailableError, match="Failed to create Irodori") as exc_info:
        create_irodori_backend(
            runtime_settings(),
            hf_hub_download_fn=lambda **_kwargs: MODEL_PATH,
            runtime_factory=runtime_factory,
            runtime_key_cls=FakeRuntimeKey,
            save_wav_fn=fake_save_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value.__cause__ is error


def test_factory_does_not_wrap_runtime_factory_type_error() -> None:
    error = TypeError("wrong runtime factory signature")

    def runtime_factory(_key: object) -> FakeRuntime:
        raise error

    with pytest.raises(TypeError, match="wrong runtime factory signature") as exc_info:
        create_irodori_backend(
            runtime_settings(),
            hf_hub_download_fn=lambda **_kwargs: MODEL_PATH,
            runtime_factory=runtime_factory,
            runtime_key_cls=FakeRuntimeKey,
            save_wav_fn=fake_save_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value is error


def test_factory_does_not_wrap_injected_download_import_error() -> None:
    error = ImportError("transitive import failed")

    def download_fn(**_kwargs: object) -> str:
        raise error

    with pytest.raises(ImportError, match="transitive import failed") as exc_info:
        create_irodori_backend(
            runtime_settings(),
            hf_hub_download_fn=download_fn,
            runtime_factory=lambda _key: FakeRuntime(),
            runtime_key_cls=FakeRuntimeKey,
            save_wav_fn=fake_save_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert exc_info.value is error


def test_importing_irodori_backend_is_lightweight() -> None:
    code = (
        "import sys\n"
        "import irodori_tts_infra.engine.backends.irodori\n"
        "blocked = {'irodori_tts', 'huggingface_hub', 'torch'}\n"
        "loaded = blocked & set(sys.modules)\n"
        "print(loaded)\n"
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
        ("checkpoint", "hf_hub_download"),
        ("model_device", "runtime_key"),
        ("model_precision", "runtime_key"),
        ("codec_device", "runtime_key"),
        ("codec_precision", "runtime_key"),
        ("compile_model", "runtime_key"),
        ("decode_mode", "sampling_request"),
        ("context_kv_cache", "sampling_request"),
        ("num_steps", "sampling_request"),
        ("cfg_scale_text", "sampling_request"),
        ("cfg_scale_caption", "sampling_request"),
        ("no_ref", "sampling_request"),
        ("warmup_num_steps", "warmup_request"),
        ("warmup_text", "warmup_request"),
        ("warmup_caption", "warmup_request"),
    ],
)
def test_all_runtime_settings_reach_expected_consumer(field: str, consumer: str) -> None:
    settings = runtime_settings(
        checkpoint="custom/repo",
        model_device="cpu",
        model_precision="fp32",
        codec_device="mps",
        codec_precision="fp16",
        compile_model=True,
        decode_mode="sequential",
        context_kv_cache=False,
        warmup_num_steps=7,
        warmup_text="ウォームアップ本文。",
        warmup_caption="ウォームアップ声。",
    )
    download_calls: list[dict[str, object]] = []
    runtime_keys: list[FakeRuntimeKey] = []
    runtime = FakeRuntime()

    def download_fn(**kwargs: object) -> str:
        download_calls.append(kwargs)
        return CUSTOM_MODEL_PATH

    def runtime_factory(key: object) -> FakeRuntime:
        assert isinstance(key, FakeRuntimeKey)
        runtime_keys.append(key)
        return runtime

    backend = create_irodori_backend(
        settings,
        checkpoint_filename="custom.bin",
        hf_hub_download_fn=download_fn,
        runtime_factory=runtime_factory,
        runtime_key_cls=FakeRuntimeKey,
        save_wav_fn=fake_save_wav,
        sampling_request_cls=FakeSamplingRequest,
    )
    backend.synthesize(
        synthesis_request(
            num_steps=11,
            cfg_scale_text=1.5,
            cfg_scale_caption=2.5,
            no_ref=True,
        ),
    )
    backend.warm_up()

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
        "cfg_scale_caption": (2.5, runtime.calls[0].cfg_scale_caption),
        "no_ref": (True, runtime.calls[0].no_ref),
        "warmup_num_steps": (7, runtime.calls[1].num_steps),
        "warmup_text": ("ウォームアップ本文。", runtime.calls[1].text),
        "warmup_caption": ("ウォームアップ声。", runtime.calls[1].caption),
    }

    assert consumer in {"hf_hub_download", "runtime_key", "sampling_request", "warmup_request"}
    assert expected[field][1] == expected[field][0]


def test_contract_mapping_defaults_round_trip() -> None:
    runtime = FakeRuntime()
    backend = make_backend(runtime)

    backend.synthesize(SynthesisRequest(text="本文", caption="声"))

    call = runtime.calls[0]
    assert call.num_steps == DEFAULT_NUM_STEPS
    assert call.cfg_scale_text == pytest.approx(DEFAULT_CFG_SCALE_TEXT)
    assert call.cfg_scale_caption == pytest.approx(DEFAULT_CFG_SCALE_CAPTION)
    assert call.no_ref is True


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


def test_tensor_like_audio_is_passed_to_save_wav_verbatim() -> None:
    audio = TensorLikeAudio()
    saved_audio: list[object] = []

    def save_wav(path: str, captured_audio: object, _sample_rate: int) -> None:
        saved_audio.append(captured_audio)
        Path(path).write_bytes(FAKE_WAV_BYTES)

    backend = make_backend(
        FakeRuntime(FakeRuntimeResult(audio=audio)),
        save_wav_fn=save_wav,
    )

    backend.synthesize(synthesis_request())

    assert saved_audio == [audio]
    assert audio.detach_count == 0
    assert audio.cpu_count == 0
