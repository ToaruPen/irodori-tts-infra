# ruff: noqa: FBT001, PLR2004, RUF029, SLF001 - explicit values and async seams keep the script contract readable.
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import math
import struct
import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from irodori_tts_infra.client import AsyncIrodoriClient
from irodori_tts_infra.contracts import (
    CapabilitiesResponse,
    SynthesisRequest,
    SynthesisResult,
    VoiceCapability,
)

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/benchmark_v4_inference.py")


class _FakeBenchmarkClient:
    def __init__(
        self,
        capabilities: CapabilitiesResponse,
        *,
        wav_bytes: bytes,
    ) -> None:
        self.capability_response = capabilities
        self.wav_bytes = wav_bytes
        self.requests: list[SynthesisRequest] = []

    async def capabilities(self) -> CapabilitiesResponse:
        return self.capability_response

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        return SynthesisResult(
            segment_index=0,
            wav_bytes=self.wav_bytes,
            elapsed_seconds=request.num_steps / 100,
        )


def _capabilities(
    *voices: VoiceCapability,
    ready: bool = True,
) -> CapabilitiesResponse:
    return CapabilitiesResponse(
        generation="opaque-runtime-generation",
        ready=ready,
        readiness="ready" if ready else "model_not_loaded",
        voices=voices,
    )


def _load_script() -> ModuleType:
    assert SCRIPT_PATH.is_file(), "v4 inference benchmark script is missing"
    spec = importlib.util.spec_from_file_location("benchmark_v4_inference", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wav_bytes(samples: tuple[int, ...], *, sample_rate: int = 24_000) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return payload.getvalue()


def test_parse_condition_accepts_bounded_unique_profile() -> None:
    module = _load_script()

    condition = module.parse_condition("candidate-16-sway:16:sway")

    assert condition.name == "candidate-16-sway"
    assert condition.num_steps == 16
    assert condition.schedule == "sway"


def test_parse_condition_rejects_steps_above_server_budget() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="steps"):
        module.parse_condition(f"candidate:{module.MAX_NUM_STEPS + 1}:sway")


def test_benchmark_samples_cover_varied_speech_structures_without_exact_fixtures() -> None:
    module = _load_script()
    samples = module.V4_INFERENCE_SAMPLES

    assert len(samples) == 6
    assert all(sample == sample.strip() for sample in samples)
    assert len({sample.strip() for sample in samples}) == 6
    assert min(map(len, samples)) <= 10
    assert max(map(len, samples)) >= 50
    assert any("?" in sample for sample in samples)
    assert any("「" in sample and "」" in sample for sample in samples)
    assert any("、" in sample for sample in samples)
    assert any(
        any(character.isascii() and character.isalnum() for character in sample)
        and any(character.isdigit() for character in sample)
        for sample in samples
    )


@pytest.mark.parametrize(
    "raw",
    [
        "missing-fields",
        "UPPER:16:sway",
        "candidate:0:sway",
        "candidate:true:sway",
        "candidate:16:unknown",
    ],
)
def test_parse_condition_rejects_invalid_profile(raw: str) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="condition"):
        module.parse_condition(raw)


def test_analyze_wav_returns_content_free_technical_metrics() -> None:
    module = _load_script()
    wav_bytes = _wav_bytes((0, 1000, -1000, 32767))

    metrics = module.analyze_wav(wav_bytes)

    assert metrics.sample_rate == 24_000
    assert metrics.channels == 1
    assert metrics.frames == 4
    assert metrics.duration_ms == pytest.approx(4 / 24_000 * 1000)
    assert not hasattr(metrics, "finite_samples")
    assert metrics.rms == pytest.approx(
        math.sqrt(sum((sample / 32768.0) ** 2 for sample in (0, 1000, -1000, 32767)) / 4),
    )
    assert metrics.silence_ratio == pytest.approx(0.25)
    assert metrics.clipping_ratio == pytest.approx(0.25)
    assert not hasattr(metrics, "wav_bytes")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not-wave", "invalid_wav"),
        (_wav_bytes((), sample_rate=24_000), "empty_wav"),
    ],
)
def test_analyze_wav_rejects_invalid_or_empty_audio(payload: bytes, match: str) -> None:
    module = _load_script()

    with pytest.raises(module.BenchmarkError, match=match):
        module.analyze_wav(payload)


def test_analyze_wav_rejects_oversized_payload_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    wav_bytes = _wav_bytes((1000,) * 24)
    monkeypatch.setattr(module, "_MAX_WAV_BYTES", len(wav_bytes) - 1)

    with pytest.raises(module.BenchmarkError, match="wav_too_large"):
        module.analyze_wav(wav_bytes)


def test_analyze_wav_rejects_excessive_audio_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_MAX_AUDIO_DURATION_SECONDS", 0.5)

    with pytest.raises(module.BenchmarkError, match="audio_too_long"):
        module.analyze_wav(_wav_bytes((1000,) * 24_000))


def test_nearest_rank_uses_observed_value_and_rejects_empty_input() -> None:
    module = _load_script()

    assert module.nearest_rank([4.0, 1.0, 3.0, 2.0], 50) == pytest.approx(2.0)
    assert module.nearest_rank([4.0, 1.0, 3.0, 2.0], 95) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="percentile"):
        module.nearest_rank([], 95)


def test_summarize_condition_contains_only_bounded_aggregate_values() -> None:
    module = _load_script()
    audio = module.analyze_wav(_wav_bytes((0, 1000, -1000, 2000)))
    measurements = [
        module.Measurement(
            client_elapsed_ms=120.0,
            server_elapsed_ms=100.0,
            audio=audio,
        ),
        module.Measurement(
            client_elapsed_ms=240.0,
            server_elapsed_ms=200.0,
            audio=audio,
        ),
    ]

    summary = module.summarize_condition(
        module.Condition(name="baseline", num_steps=24, schedule="linear"),
        measurements,
        reference_condition=module.Condition(
            name="baseline",
            num_steps=24,
            schedule="linear",
        ),
        reference_measurements=measurements,
    )
    encoded = json.dumps(summary, ensure_ascii=False)

    assert summary == {
        "name": "baseline",
        "num_steps": 24,
        "schedule": "linear",
        "samples": 2,
        "is_reference": True,
        "client_elapsed_ms_p50": 120.0,
        "client_elapsed_ms_p95": 240.0,
        "client_elapsed_ms_mean": 180.0,
        "server_elapsed_ms_p50": 100.0,
        "server_elapsed_ms_p95": 200.0,
        "server_elapsed_ms_mean": 150.0,
        "realtime_factor_p50": 600.0,
        "realtime_factor_p95": 1200.0,
        "realtime_factor_mean": 900.0,
        "realtime_factor_max": 1200.0,
        "realtime_factor_violation_count": 2,
        "realtime_factor_regression_count": 0,
        "realtime_factor_pass": False,
        "duration_ms_p50": 0.167,
        "duration_ms_p95": 0.167,
        "duration_ms_mean": 0.167,
        "duration_drift_percent_max": 0.0,
        "duration_drift_violation_count": 0,
        "duration_drift_pass": True,
        "rms_p50": pytest.approx(audio.rms, abs=0.001),
        "rms_p95": pytest.approx(audio.rms, abs=0.001),
        "rms_mean": pytest.approx(audio.rms, abs=0.001),
        "technical_audio_pass": True,
        "server_latency_reduction_percent": 0.0,
        "client_latency_reduction_percent": 0.0,
        "latency_reduction_pass": False,
        "request_technical_gate_pass": False,
    }
    assert "本文" not in encoded
    assert "voice" not in encoded
    assert "generation" not in encoded
    assert "wav" not in encoded


def test_summarize_condition_reports_clip_level_sampling_gate_failures() -> None:
    module = _load_script()
    reference_audio = module.analyze_wav(_wav_bytes((1000,) * 24_000))
    shorter_audio = module.analyze_wav(_wav_bytes((1000,) * 18_000))
    reference = [
        module.Measurement(500.0, 500.0, reference_audio),
        module.Measurement(500.0, 500.0, reference_audio),
    ]
    candidate = [
        module.Measurement(200.0, 200.0, reference_audio),
        module.Measurement(900.0, 900.0, shorter_audio),
    ]

    summary = module.summarize_condition(
        module.Condition(name="candidate", num_steps=12, schedule="sway"),
        candidate,
        reference_condition=module.Condition(
            name="baseline",
            num_steps=24,
            schedule="linear",
        ),
        reference_measurements=reference,
    )

    assert summary["realtime_factor_max"] == pytest.approx(1.2)
    assert summary["realtime_factor_violation_count"] == 1
    assert summary["realtime_factor_regression_count"] == 1
    assert summary["realtime_factor_pass"] is False
    assert summary["duration_drift_percent_max"] == pytest.approx(25.0)
    assert summary["duration_drift_violation_count"] == 1
    assert summary["duration_drift_pass"] is False
    assert summary["request_technical_gate_pass"] is False


def test_summarize_condition_requires_paired_reference_measurements() -> None:
    module = _load_script()
    audio = module.analyze_wav(_wav_bytes((1000,) * 24_000))
    measurement = module.Measurement(500.0, 500.0, audio)

    with pytest.raises(ValueError, match="paired"):
        module.summarize_condition(
            module.Condition(name="candidate", num_steps=12, schedule="sway"),
            [measurement],
            reference_condition=module.Condition(
                name="baseline",
                num_steps=24,
                schedule="linear",
            ),
            reference_measurements=[measurement, measurement],
        )


@pytest.mark.parametrize(
    ("server_elapsed_ms", "client_elapsed_ms", "expected_pass"),
    [
        (850.0, 850.0, True),
        (850.0, 851.0, False),
        (851.0, 850.0, False),
        (851.0, 851.0, False),
    ],
)
def test_request_technical_gate_requires_both_latency_reductions(
    server_elapsed_ms: float,
    client_elapsed_ms: float,
    expected_pass: bool,
) -> None:
    module = _load_script()
    audio = module.analyze_wav(_wav_bytes((1000,) * 48_000))
    reference = [module.Measurement(1000.0, 1000.0, audio)]
    candidate = [
        module.Measurement(client_elapsed_ms, server_elapsed_ms, audio),
    ]

    summary = module.summarize_condition(
        module.Condition(name="candidate", num_steps=12, schedule="sway"),
        candidate,
        reference_condition=module.Condition(
            name="baseline",
            num_steps=24,
            schedule="linear",
        ),
        reference_measurements=reference,
    )

    expected_server_reduction = (1000.0 - server_elapsed_ms) / 10.0
    expected_client_reduction = (1000.0 - client_elapsed_ms) / 10.0
    assert summary["server_latency_reduction_percent"] == pytest.approx(
        expected_server_reduction,
    )
    assert summary["client_latency_reduction_percent"] == pytest.approx(
        expected_client_reduction,
    )
    assert summary["latency_reduction_pass"] is expected_pass
    assert summary["request_technical_gate_pass"] is expected_pass


def test_select_voice_id_uses_dynamic_default_or_explicit_alias() -> None:
    module = _load_script()
    first = VoiceCapability(id="opaque-a", label="Dynamic A", aliases=("alias-a",))
    second = VoiceCapability(id="opaque-b", label="Dynamic B", default=True)
    capabilities = _capabilities(first, second)

    assert module.select_voice_id(capabilities, None) == "opaque-b"
    assert module.select_voice_id(capabilities, "alias-a") == "opaque-a"


def test_select_voice_id_fails_closed_without_default_or_match() -> None:
    module = _load_script()
    capabilities = _capabilities(
        VoiceCapability(id="opaque-a", label="Dynamic A"),
    )

    with pytest.raises(module.BenchmarkError, match="default_voice_missing"):
        module.select_voice_id(capabilities, None)
    with pytest.raises(module.BenchmarkError, match="voice_not_found"):
        module.select_voice_id(capabilities, "missing")


@pytest.mark.asyncio
async def test_run_benchmark_rotates_conditions_and_binds_generation() -> None:
    module = _load_script()
    client = _FakeBenchmarkClient(
        _capabilities(
            VoiceCapability(id="opaque-a", label="Dynamic A", default=True),
        ),
        wav_bytes=_wav_bytes((1000,) * 24_000),
    )
    baseline = module.Condition(name="baseline", num_steps=24, schedule="linear")
    candidate = module.Condition(name="candidate", num_steps=16, schedule="sway")

    summary = await module.run_benchmark(
        client,
        conditions=(baseline, candidate),
        samples=("非機密の一文目です。", "非機密の二文目です。"),
        seeds=(101,),
    )

    assert [request.num_steps for request in client.requests] == [24, 16, 16, 24]
    assert [request.t_schedule_mode for request in client.requests] == [
        "linear",
        "sway",
        "sway",
        "linear",
    ]
    assert all(request.voice_id == "opaque-a" for request in client.requests)
    assert all(request.if_generation == "opaque-runtime-generation" for request in client.requests)
    assert all(request.style == "neutral" for request in client.requests)
    assert all(request.seed == 101 for request in client.requests)
    assert summary["schema_version"] == "irodori-v4-inference-benchmark/v1"
    assert summary["reference_condition"] == "baseline"
    assert summary["sample_count"] == 2
    assert summary["seed_count"] == 1
    assert [condition["name"] for condition in summary["conditions"]] == [
        "baseline",
        "candidate",
    ]
    encoded = json.dumps(summary, ensure_ascii=False)
    assert "非機密" not in encoded
    assert "opaque-a" not in encoded
    assert "opaque-runtime-generation" not in encoded


def test_safe_error_code_uses_an_explicit_allowlist() -> None:
    module = _load_script()

    assert module._safe_error_code("runtime_not_ready") == "runtime_not_ready"
    assert module._safe_error_code("sensitive-voice-alias") == "client_error"
    assert module._safe_error_code("private_content") == "client_error"


@pytest.mark.asyncio
async def test_run_benchmark_rejects_unready_runtime_and_duplicate_conditions() -> None:
    module = _load_script()
    voice = VoiceCapability(id="opaque-a", label="Dynamic A", default=True)
    unready = _FakeBenchmarkClient(
        _capabilities(voice, ready=False),
        wav_bytes=_wav_bytes((1000,) * 24_000),
    )
    baseline = module.Condition(name="baseline", num_steps=24, schedule="linear")

    with pytest.raises(module.BenchmarkError, match="runtime_not_ready"):
        await module.run_benchmark(
            unready,
            conditions=(baseline,),
            samples=("非機密です。",),
            seeds=(101,),
        )

    ready = _FakeBenchmarkClient(
        _capabilities(voice),
        wav_bytes=_wav_bytes((1000,) * 24_000),
    )
    candidate = module.Condition(name="candidate", num_steps=12, schedule="sway")
    with pytest.raises(ValueError, match="reference"):
        await module.run_benchmark(
            ready,
            conditions=(candidate, baseline),
            samples=("非機密です。",),
            seeds=(101,),
        )

    with pytest.raises(ValueError, match="condition names"):
        await module.run_benchmark(
            ready,
            conditions=(baseline, baseline),
            samples=("非機密です。",),
            seeds=(101,),
        )


@pytest.mark.asyncio
async def test_run_catalog_benchmark_uses_every_dynamic_voice_anonymously() -> None:
    module = _load_script()
    client = _FakeBenchmarkClient(
        _capabilities(
            VoiceCapability(id="opaque-z", label="Dynamic Z"),
            VoiceCapability(id="opaque-a", label="Dynamic A"),
        ),
        wav_bytes=_wav_bytes((1000,) * 24_000),
    )
    conditions = (
        module.Condition(name="baseline", num_steps=24, schedule="linear"),
        module.Condition(name="candidate", num_steps=12, schedule="sway"),
    )

    summary = await module.run_catalog_benchmark(
        client,
        conditions=conditions,
        samples=("非機密です。",),
        seeds=(101,),
    )

    assert [request.voice_id for request in client.requests] == [
        "opaque-z",
        "opaque-z",
        "opaque-a",
        "opaque-a",
    ]
    assert summary["voice_count"] == 2
    assert all(condition["samples"] == 2 for condition in summary["conditions"])
    encoded = json.dumps(summary, ensure_ascii=False)
    assert "opaque-z" not in encoded
    assert "opaque-a" not in encoded
    assert "Dynamic" not in encoded


@pytest.mark.asyncio
async def test_run_catalog_benchmark_rejects_excessive_dynamic_voice_count() -> None:
    module = _load_script()
    voices = tuple(
        VoiceCapability(id=f"opaque-{index}", label=f"Dynamic {index}")
        for index in range(module._MAX_CATALOG_VOICES + 1)
    )
    client = _FakeBenchmarkClient(
        _capabilities(*voices),
        wav_bytes=_wav_bytes((1000,) * 24_000),
    )

    with pytest.raises(module.BenchmarkError, match="benchmark_workload_too_large"):
        await module.run_catalog_benchmark(
            client,
            conditions=(module.Condition(name="baseline", num_steps=24, schedule="linear"),),
            samples=("非機密です。",),
            seeds=(101,),
        )

    assert not client.requests


def test_workload_validation_caps_condition_seed_and_step_budgets() -> None:
    module = _load_script()
    baseline = module.Condition(name="baseline", num_steps=24, schedule="linear")
    extra_conditions = tuple(
        module.Condition(name=f"candidate-{index}", num_steps=12, schedule="sway")
        for index in range(module._MAX_CONDITIONS)
    )
    with pytest.raises(module.BenchmarkError, match="benchmark_workload_too_large"):
        module._validate_run_inputs(
            (baseline, *extra_conditions),
            module.V4_INFERENCE_SAMPLES,
            (101,),
        )
    with pytest.raises(module.BenchmarkError, match="benchmark_workload_too_large"):
        module._validate_run_inputs(
            (baseline,),
            module.V4_INFERENCE_SAMPLES,
            tuple(range(module._MAX_SEEDS + 1)),
        )

    seeds = tuple(range(module._MAX_SEEDS))
    step_units_per_voice = baseline.num_steps * len(module.V4_INFERENCE_SAMPLES) * len(seeds)
    voice_count = module._MAX_STEP_UNITS // step_units_per_voice + 1
    assert voice_count <= module._MAX_CATALOG_VOICES
    with pytest.raises(module.BenchmarkError, match="benchmark_workload_too_large"):
        module._validate_workload(
            tuple(f"opaque-{index}" for index in range(voice_count)),
            conditions=(baseline,),
            samples=module.V4_INFERENCE_SAMPLES,
            seeds=seeds,
        )


@pytest.mark.asyncio
async def test_run_cli_uses_fixed_samples_and_default_candidate_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    captured: dict[str, object] = {}

    async def fake_execute_benchmark(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "schema_version": "irodori-v4-inference-benchmark/v1",
            "sample_count": 2,
            "seed_count": 1,
            "conditions": [],
        }

    monkeypatch.setattr(module, "execute_benchmark", fake_execute_benchmark)

    args = module._parse_args(
        (
            "--base-url",
            "http://127.0.0.1:18924",
            "--sample-limit",
            "2",
            "--seed",
            "707",
        ),
    )
    exit_code = await module._run_cli(args)

    assert exit_code == 0
    assert captured["base_url"] == "http://127.0.0.1:18924"
    assert captured["all_voices"] is False
    assert captured["samples"] == module.V4_INFERENCE_SAMPLES[:2]
    assert captured["seeds"] == (707,)
    conditions = captured["conditions"]
    assert isinstance(conditions, tuple)
    assert [condition.name for condition in conditions] == [
        "baseline-24-linear",
        "candidate-24-sway",
        "candidate-20-sway",
        "candidate-16-sway",
        "candidate-16-linear",
        "candidate-12-sway",
    ]
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["schema_version"] == "irodori-v4-inference-benchmark/v1"
    assert not output.err


@pytest.mark.asyncio
async def test_run_cli_accepts_explicit_conditions_and_omits_sensitive_failure_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()

    async def fake_execute_benchmark(**_kwargs: object) -> dict[str, object]:
        message = "runtime_not_ready"
        raise module.BenchmarkError(message)

    monkeypatch.setattr(module, "execute_benchmark", fake_execute_benchmark)

    args = module._parse_args(
        (
            "--voice",
            "sensitive-voice-alias",
            "--condition",
            "baseline:24:linear",
            "--sample-limit",
            "1",
        ),
    )
    exit_code = await module._run_cli(args)

    assert exit_code != 0
    output = capsys.readouterr()
    payload = json.loads(output.err)
    assert payload == {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "status": "failed",
        "code": "runtime_not_ready",
    }
    assert not output.out
    assert "sensitive-voice-alias" not in output.err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition_args",
    [
        ("--condition", "candidate:12:sway"),
        (
            "--condition",
            "baseline:24:linear",
            "--condition",
            "baseline:12:sway",
        ),
    ],
)
async def test_run_cli_maps_invalid_condition_collection_to_stable_failure_json(
    condition_args: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()

    async def validate_inputs(**kwargs: object) -> dict[str, object]:
        module._validate_run_inputs(
            kwargs["conditions"],
            kwargs["samples"],
            kwargs["seeds"],
        )
        message = "unreachable"
        raise AssertionError(message)

    monkeypatch.setattr(module, "execute_benchmark", validate_inputs)
    args = module._parse_args((*condition_args, "--sample-limit", "1"))

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "status": "failed",
        "code": "invalid_input",
    }
    assert not output.out


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_endpoint", ["/capabilities", "/synthesize"])
async def test_run_cli_maps_contract_validation_errors_to_private_stable_json(
    malformed_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    capabilities = _capabilities(
        VoiceCapability(id="opaque-a", label="Dynamic A", default=True),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == malformed_endpoint:
            return httpx.Response(
                200,
                json={
                    "sensitive-voice-id": "must-not-leak",
                    "unexpected": "private-content",
                },
            )
        assert request.url.path == "/capabilities"
        return httpx.Response(200, json=capabilities.model_dump(mode="json"))

    client = AsyncIrodoriClient(
        base_url="http://benchmark.invalid",
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(module, "AsyncIrodoriClient", lambda **_kwargs: client)
    args = module._parse_args(
        ("--condition", "baseline:24:linear", "--sample-limit", "1"),
    )

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "status": "failed",
        "code": "invalid_response",
    }
    assert not output.out
    assert "sensitive-voice-id" not in output.err
    assert "private-content" not in output.err


@pytest.mark.asyncio
async def test_run_cli_enforces_an_outer_wall_clock_deadline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()

    async def never_finishes(**_kwargs: object) -> dict[str, object]:
        await asyncio.Event().wait()
        message = "unreachable"
        raise AssertionError(message)

    monkeypatch.setattr(module, "execute_benchmark", never_finishes)
    monkeypatch.setattr(module, "_MAX_RUN_SECONDS", 0.001)
    args = module._parse_args(
        ("--condition", "baseline:24:linear", "--sample-limit", "1"),
    )

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "status": "failed",
        "code": "benchmark_timeout",
    }
    assert not output.out


@pytest.mark.asyncio
@pytest.mark.parametrize("excessive_input", ["conditions", "seeds"])
async def test_run_cli_maps_input_resource_caps_to_stable_failure_json(
    excessive_input: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()

    async def validate_inputs(**kwargs: object) -> dict[str, object]:
        module._validate_run_inputs(
            kwargs["conditions"],
            kwargs["samples"],
            kwargs["seeds"],
        )
        message = "unreachable"
        raise AssertionError(message)

    monkeypatch.setattr(module, "execute_benchmark", validate_inputs)
    args = module._parse_args(
        ("--condition", "baseline:24:linear", "--sample-limit", "1"),
    )
    if excessive_input == "conditions":
        args.conditions = (
            *args.conditions,
            *(
                module.Condition(name=f"candidate-{index}", num_steps=12, schedule="sway")
                for index in range(module._MAX_CONDITIONS)
            ),
        )
    else:
        args.seed = list(range(module._MAX_SEEDS + 1))

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "status": "failed",
        "code": "benchmark_workload_too_large",
    }
    assert not output.out


@pytest.mark.asyncio
async def test_run_cli_enables_anonymous_catalog_sweep(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    captured: dict[str, object] = {}

    async def fake_execute_benchmark(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "schema_version": "irodori-v4-inference-benchmark/v1",
            "voice_count": 7,
            "conditions": [],
        }

    monkeypatch.setattr(module, "execute_benchmark", fake_execute_benchmark)
    args = module._parse_args(("--all-voices", "--sample-limit", "1"))

    assert await module._run_cli(args) == 0
    assert captured["all_voices"] is True
    assert captured["voice_selector"] is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["voice_count"] == 7


def test_run_cli_rejects_voice_selector_with_catalog_sweep() -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module._parse_args(("--all-voices", "--voice", "dynamic-alias"))


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.invalid:8924",
        "http://127.0.0.1.example.invalid:8924",
        "http://user:password@127.0.0.1:8924",
        "http://127.0.0.1:8924/private",
        "http://127.0.0.1:abc",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:-1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:",
        "http://localhost:8924",
    ],
)
def test_run_cli_rejects_non_loopback_or_ambiguous_base_urls(base_url: str) -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module._parse_args(("--base-url", base_url))


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.2:8924", "http://[::1]:8924"],
)
def test_run_cli_accepts_loopback_base_urls(base_url: str) -> None:
    module = _load_script()

    assert module._parse_args(("--base-url", base_url)).base_url == base_url


@pytest.mark.asyncio
async def test_execute_benchmark_uses_a_direct_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    client = _FakeBenchmarkClient(
        _capabilities(
            VoiceCapability(id="opaque-a", label="Dynamic A", default=True),
        ),
        wav_bytes=_wav_bytes((1000,) * 24_000),
    )
    captured: dict[str, object] = {}

    class ClientContext:
        async def __aenter__(self) -> _FakeBenchmarkClient:
            return client

        async def __aexit__(self, *_args: object) -> None:
            return None

    def client_factory(**kwargs: object) -> ClientContext:
        captured.update(kwargs)
        return ClientContext()

    monkeypatch.setattr(module, "AsyncIrodoriClient", client_factory)
    await module.execute_benchmark(
        base_url="http://127.0.0.1:8924",
        timeout_seconds=1.0,
        conditions=(module.Condition(name="baseline", num_steps=24, schedule="linear"),),
        samples=("非機密です。",),
        seeds=(101,),
        voice_selector=None,
        all_voices=False,
    )

    transport = captured["transport"]
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    assert captured["max_response_bytes"] == module._MAX_HTTP_RESPONSE_BYTES
    await transport.aclose()


@pytest.mark.asyncio
async def test_run_cli_rejects_nonfinite_server_elapsed_as_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    capabilities = _capabilities(
        VoiceCapability(id="opaque-a", label="Dynamic A", default=True),
    )
    synthesis_json = SynthesisResult(
        segment_index=0,
        wav_bytes=_wav_bytes((1000,) * 24_000),
        elapsed_seconds=0.1,
    ).model_dump_json()
    synthesis_json = synthesis_json.replace('"elapsed_seconds":0.1', '"elapsed_seconds":1e999')

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/capabilities":
            return httpx.Response(200, json=capabilities.model_dump(mode="json"))
        assert request.url.path == "/synthesize"
        return httpx.Response(
            200,
            content=synthesis_json,
            headers={"content-type": "application/json"},
        )

    client = AsyncIrodoriClient(
        base_url="http://benchmark.invalid",
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(module, "AsyncIrodoriClient", lambda **_kwargs: client)
    args = module._parse_args(
        ("--condition", "baseline:24:linear", "--sample-limit", "1"),
    )

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "status": "failed",
        "code": "invalid_response",
    }
    assert not output.out


@pytest.mark.parametrize("sample_limit", ["0", "999"])
def test_run_cli_rejects_out_of_range_sample_limit(sample_limit: str) -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module._parse_args(("--sample-limit", sample_limit))
