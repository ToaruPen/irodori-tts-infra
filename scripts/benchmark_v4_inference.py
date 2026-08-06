from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import re
import struct
import sys
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Literal, Protocol, cast
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

from irodori_tts_infra.client import AsyncIrodoriClient, ClientError
from irodori_tts_infra.contracts import (
    MAX_NUM_STEPS,
    CapabilitiesResponse,
    SynthesisRequest,
    SynthesisResult,
)
from irodori_tts_infra.evaluation_samples import V4_INFERENCE_SAMPLES

ConditionSchedule = Literal["linear", "sway"]

_CONDITION_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CONDITION_PARTS = 3
_PCM16_BYTES = 2
_PCM16_SCALE = 32768.0
_SILENCE_AMPLITUDE = 0.001
_MAX_SILENCE_RATIO = 0.98
_CLIPPING_AMPLITUDE = 32767.0 / _PCM16_SCALE
_MAX_CLIPPING_RATIO = 0.01
_MAX_DURATION_DRIFT_PERCENT = 10.0
_MIN_LATENCY_REDUCTION_PERCENT = 15.0
_MAX_PERCENTILE = 100
_MAX_WAV_BYTES = 4 * 1024 * 1024
_MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_AUDIO_DURATION_SECONDS = 60.0
_MAX_CATALOG_VOICES = 128
_MAX_CONDITIONS = 8
_MAX_SEEDS = 8
_MAX_TRIALS = 2_048
_MAX_STEP_UNITS = 32_768
_MAX_RUN_SECONDS = 900.0
_REFERENCE_NUM_STEPS = 24
_REFERENCE_SCHEDULE: ConditionSchedule = "linear"
_DEFAULT_BASE_URL = "http://127.0.0.1:8924"
_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_SEEDS = (101, 202, 303)
_DEFAULT_CONDITION_SPECS = (
    "baseline-24-linear:24:linear",
    "candidate-24-sway:24:sway",
    "candidate-20-sway:20:sway",
    "candidate-16-sway:16:sway",
    "candidate-16-linear:16:linear",
    "candidate-12-sway:12:sway",
)
_SAFE_FAILURE_CODES = frozenset(
    {
        "backpressure",
        "audio_too_long",
        "backend_unavailable",
        "benchmark_timeout",
        "benchmark_workload_too_large",
        "client_error",
        "default_voice_missing",
        "empty_batch",
        "empty_wav",
        "http_error",
        "invalid_input",
        "invalid_response",
        "invalid_wav",
        "model_not_loaded",
        "protocol_error",
        "response_too_large",
        "runtime_generation_mismatch",
        "runtime_not_ready",
        "timeout",
        "transport_error",
        "truncated_wav",
        "unsupported_wav",
        "voice_bank_invalid",
        "voice_catalog_ambiguous",
        "voice_not_found",
        "wav_too_large",
    },
)


class BenchmarkError(RuntimeError):
    pass


class BenchmarkClient(Protocol):
    async def capabilities(self) -> CapabilitiesResponse: ...

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    num_steps: int
    schedule: ConditionSchedule


@dataclass(frozen=True, slots=True)
class AudioMetrics:
    sample_rate: int
    channels: int
    frames: int
    duration_ms: float
    rms: float
    silence_ratio: float
    clipping_ratio: float


@dataclass(frozen=True, slots=True)
class Measurement:
    client_elapsed_ms: float
    server_elapsed_ms: float
    audio: AudioMetrics


def parse_condition(raw: str) -> Condition:
    parts = raw.split(":")
    if len(parts) != _CONDITION_PARTS:
        message = "condition must use name:steps:schedule"
        raise ValueError(message)
    name, steps_text, schedule = parts
    if _CONDITION_NAME_RE.fullmatch(name) is None:
        message = "condition name must be a lowercase slug"
        raise ValueError(message)
    if not steps_text.isdecimal():
        message = "condition steps must be a positive integer"
        raise ValueError(message)
    num_steps = int(steps_text)
    if num_steps <= 0:
        message = "condition steps must be a positive integer"
        raise ValueError(message)
    if num_steps > MAX_NUM_STEPS:
        message = f"condition steps must not exceed {MAX_NUM_STEPS}"
        raise ValueError(message)
    if schedule not in {"linear", "sway"}:
        message = "condition schedule must be linear or sway"
        raise ValueError(message)
    return Condition(
        name=name,
        num_steps=num_steps,
        schedule=cast("ConditionSchedule", schedule),
    )


def analyze_wav(wav_bytes: bytes) -> AudioMetrics:
    if len(wav_bytes) > _MAX_WAV_BYTES:
        message = "wav_too_large"
        raise BenchmarkError(message)
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != _PCM16_BYTES:
                message = "unsupported_wav"
                raise BenchmarkError(message)
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            pcm_bytes = wav_file.readframes(frames)
    except (EOFError, wave.Error) as error:
        message = "invalid_wav"
        raise BenchmarkError(message) from error
    if channels <= 0 or sample_rate <= 0:
        message = "invalid_wav"
        raise BenchmarkError(message)
    if frames <= 0:
        message = "empty_wav"
        raise BenchmarkError(message)
    duration_seconds = frames / sample_rate
    if duration_seconds > _MAX_AUDIO_DURATION_SECONDS:
        message = "audio_too_long"
        raise BenchmarkError(message)
    sample_count = frames * channels
    if len(pcm_bytes) != sample_count * _PCM16_BYTES:
        message = "truncated_wav"
        raise BenchmarkError(message)
    samples = struct.unpack(f"<{sample_count}h", pcm_bytes)
    normalized = tuple(sample / _PCM16_SCALE for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in normalized) / sample_count)
    silence_ratio = sum(abs(sample) <= _SILENCE_AMPLITUDE for sample in normalized) / sample_count
    clipping_ratio = sum(abs(sample) >= _CLIPPING_AMPLITUDE for sample in normalized) / sample_count
    return AudioMetrics(
        sample_rate=sample_rate,
        channels=channels,
        frames=frames,
        duration_ms=duration_seconds * 1000.0,
        rms=rms,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
    )


def nearest_rank(values: list[float], percentile: int) -> float:
    if not values or not 1 <= percentile <= _MAX_PERCENTILE:
        message = "percentile requires values and an integer from 1 to 100"
        raise ValueError(message)
    ordered = sorted(values)
    rank = math.ceil(percentile / _MAX_PERCENTILE * len(ordered))
    return ordered[rank - 1]


def summarize_condition(  # noqa: PLR0914 - explicit gate inputs aid auditability.
    condition: Condition,
    measurements: list[Measurement],
    *,
    reference_condition: Condition,
    reference_measurements: Sequence[Measurement],
) -> dict[str, str | int | float | bool]:
    if not measurements:
        message = "condition summary requires at least one measurement"
        raise ValueError(message)
    if len(measurements) != len(reference_measurements):
        message = "condition summary requires paired reference measurements"
        raise ValueError(message)
    client_elapsed = [measurement.client_elapsed_ms for measurement in measurements]
    server_elapsed = [measurement.server_elapsed_ms for measurement in measurements]
    durations = [measurement.audio.duration_ms for measurement in measurements]
    rms_values = [measurement.audio.rms for measurement in measurements]
    client_elapsed_p95 = nearest_rank(client_elapsed, 95)
    server_elapsed_p95 = nearest_rank(server_elapsed, 95)
    reference_client_elapsed_p95 = nearest_rank(
        [measurement.client_elapsed_ms for measurement in reference_measurements],
        95,
    )
    reference_server_elapsed_p95 = nearest_rank(
        [measurement.server_elapsed_ms for measurement in reference_measurements],
        95,
    )
    realtime_factors = [
        measurement.server_elapsed_ms / measurement.audio.duration_ms
        for measurement in measurements
    ]
    reference_realtime_factors = [
        measurement.server_elapsed_ms / measurement.audio.duration_ms
        for measurement in reference_measurements
    ]
    duration_drifts = [
        abs(measurement.audio.duration_ms - reference.audio.duration_ms)
        / reference.audio.duration_ms
        * 100.0
        for measurement, reference in zip(measurements, reference_measurements, strict=True)
    ]
    realtime_factor_violation_count = sum(value >= 1.0 for value in realtime_factors)
    realtime_factor_regression_count = sum(
        value > reference
        for value, reference in zip(
            realtime_factors,
            reference_realtime_factors,
            strict=True,
        )
    )
    duration_drift_violation_count = sum(
        value > _MAX_DURATION_DRIFT_PERCENT for value in duration_drifts
    )
    realtime_factor_pass = (
        realtime_factor_violation_count == 0 and realtime_factor_regression_count == 0
    )
    duration_drift_pass = duration_drift_violation_count == 0
    is_reference = condition.name == reference_condition.name
    server_latency_reduction_percent = _percent_reduction(
        reference_server_elapsed_p95,
        server_elapsed_p95,
    )
    client_latency_reduction_percent = _percent_reduction(
        reference_client_elapsed_p95,
        client_elapsed_p95,
    )
    latency_reduction_pass = not is_reference and (
        server_latency_reduction_percent >= _MIN_LATENCY_REDUCTION_PERCENT
        and client_latency_reduction_percent >= _MIN_LATENCY_REDUCTION_PERCENT
    )
    technical_audio_pass = all(
        measurement.audio.rms > 0
        and measurement.audio.silence_ratio <= _MAX_SILENCE_RATIO
        and measurement.audio.clipping_ratio <= _MAX_CLIPPING_RATIO
        for measurement in measurements
    )
    return {
        "name": condition.name,
        "num_steps": condition.num_steps,
        "schedule": condition.schedule,
        "samples": len(measurements),
        "is_reference": is_reference,
        "client_elapsed_ms_p50": _rounded(nearest_rank(client_elapsed, 50)),
        "client_elapsed_ms_p95": _rounded(client_elapsed_p95),
        "client_elapsed_ms_mean": _rounded(_mean(client_elapsed)),
        "server_elapsed_ms_p50": _rounded(nearest_rank(server_elapsed, 50)),
        "server_elapsed_ms_p95": _rounded(server_elapsed_p95),
        "server_elapsed_ms_mean": _rounded(_mean(server_elapsed)),
        "realtime_factor_p50": _rounded(nearest_rank(realtime_factors, 50)),
        "realtime_factor_p95": _rounded(nearest_rank(realtime_factors, 95)),
        "realtime_factor_mean": _rounded(_mean(realtime_factors)),
        "realtime_factor_max": _rounded(max(realtime_factors)),
        "realtime_factor_violation_count": realtime_factor_violation_count,
        "realtime_factor_regression_count": realtime_factor_regression_count,
        "realtime_factor_pass": realtime_factor_pass,
        "duration_ms_p50": _rounded(nearest_rank(durations, 50)),
        "duration_ms_p95": _rounded(nearest_rank(durations, 95)),
        "duration_ms_mean": _rounded(_mean(durations)),
        "duration_drift_percent_max": _rounded(max(duration_drifts)),
        "duration_drift_violation_count": duration_drift_violation_count,
        "duration_drift_pass": duration_drift_pass,
        "rms_p50": _rounded(nearest_rank(rms_values, 50)),
        "rms_p95": _rounded(nearest_rank(rms_values, 95)),
        "rms_mean": _rounded(_mean(rms_values)),
        "technical_audio_pass": technical_audio_pass,
        "server_latency_reduction_percent": _rounded(
            server_latency_reduction_percent,
        ),
        "client_latency_reduction_percent": _rounded(
            client_latency_reduction_percent,
        ),
        "latency_reduction_pass": latency_reduction_pass,
        "request_technical_gate_pass": (
            latency_reduction_pass
            and technical_audio_pass
            and realtime_factor_pass
            and duration_drift_pass
        ),
    }


def select_voice_id(capabilities: CapabilitiesResponse, selector: str | None) -> str:
    if selector is None:
        defaults = [voice.id for voice in capabilities.voices if voice.default]
        if len(defaults) != 1:
            message = "default_voice_missing"
            raise BenchmarkError(message)
        return defaults[0]
    for voice in capabilities.voices:
        if selector == voice.id or selector in voice.aliases:
            return voice.id
    message = "voice_not_found"
    raise BenchmarkError(message)


async def run_benchmark(
    client: BenchmarkClient,
    *,
    conditions: Sequence[Condition],
    samples: Sequence[str],
    seeds: Sequence[int],
    voice_selector: str | None = None,
) -> dict[str, object]:
    _validate_run_inputs(conditions, samples, seeds)
    capabilities = await client.capabilities()
    _require_ready(capabilities)
    voice_id = select_voice_id(capabilities, voice_selector)
    return await _run_for_voice_ids(
        client,
        capabilities=capabilities,
        voice_ids=(voice_id,),
        conditions=conditions,
        samples=samples,
        seeds=seeds,
    )


async def run_catalog_benchmark(
    client: BenchmarkClient,
    *,
    conditions: Sequence[Condition],
    samples: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, object]:
    _validate_run_inputs(conditions, samples, seeds)
    capabilities = await client.capabilities()
    _require_ready(capabilities)
    voice_ids = tuple(voice.id for voice in capabilities.voices)
    if not voice_ids or len(voice_ids) != len(set(voice_ids)):
        message = "voice_catalog_ambiguous"
        raise BenchmarkError(message)
    summary = await _run_for_voice_ids(
        client,
        capabilities=capabilities,
        voice_ids=voice_ids,
        conditions=conditions,
        samples=samples,
        seeds=seeds,
    )
    summary["voice_count"] = len(voice_ids)
    return summary


async def _run_for_voice_ids(
    client: BenchmarkClient,
    *,
    capabilities: CapabilitiesResponse,
    voice_ids: Sequence[str],
    conditions: Sequence[Condition],
    samples: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, object]:
    _validate_workload(voice_ids, conditions, samples, seeds)
    measurements: dict[str, list[Measurement]] = {condition.name: [] for condition in conditions}
    trial_index = 0
    for voice_id in voice_ids:
        for sample in samples:
            for seed in seeds:
                offset = trial_index % len(conditions)
                ordered_conditions = (*conditions[offset:], *conditions[:offset])
                for condition in ordered_conditions:
                    request = SynthesisRequest(
                        text=sample,
                        voice_id=voice_id,
                        if_generation=capabilities.generation,
                        num_steps=condition.num_steps,
                        style="neutral",
                        seed=seed,
                        t_schedule_mode=condition.schedule,
                    )
                    measurements[condition.name].append(
                        await _measure_condition(client, request),
                    )
                trial_index += 1
    reference_condition = conditions[0]
    reference_measurements = measurements[reference_condition.name]
    return {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "sample_count": len(samples),
        "seed_count": len(seeds),
        "reference_condition": conditions[0].name,
        "conditions": [
            summarize_condition(
                condition,
                measurements[condition.name],
                reference_condition=reference_condition,
                reference_measurements=reference_measurements,
            )
            for condition in conditions
        ],
    }


async def _measure_condition(
    client: BenchmarkClient,
    request: SynthesisRequest,
) -> Measurement:
    started_ns = time.perf_counter_ns()
    result = await client.synthesize(request)
    client_elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    server_elapsed_ms = result.elapsed_seconds * 1000.0
    if not math.isfinite(server_elapsed_ms) or server_elapsed_ms < 0:
        message = "invalid_response"
        raise BenchmarkError(message)
    return Measurement(
        client_elapsed_ms=client_elapsed_ms,
        server_elapsed_ms=server_elapsed_ms,
        audio=analyze_wav(result.wav_bytes),
    )


def _require_ready(capabilities: CapabilitiesResponse) -> None:
    if not capabilities.ready or capabilities.readiness != "ready":
        message = "runtime_not_ready"
        raise BenchmarkError(message)


def _validate_run_inputs(
    conditions: Sequence[Condition],
    samples: Sequence[str],
    seeds: Sequence[int],
) -> None:
    if not conditions:
        message = "benchmark requires at least one condition"
        raise ValueError(message)
    if len(conditions) > _MAX_CONDITIONS:
        message = "benchmark_workload_too_large"
        raise BenchmarkError(message)
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        message = "condition names must be unique"
        raise ValueError(message)
    reference = conditions[0]
    if reference.num_steps != _REFERENCE_NUM_STEPS or reference.schedule != _REFERENCE_SCHEDULE:
        message = "first condition must be the 24-step linear reference"
        raise ValueError(message)
    if not samples or any(not sample.strip() for sample in samples):
        message = "benchmark samples must be non-blank"
        raise ValueError(message)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        message = "benchmark seeds must be integers"
        raise ValueError(message)
    if len(seeds) > _MAX_SEEDS:
        message = "benchmark_workload_too_large"
        raise BenchmarkError(message)


def _validate_workload(
    voice_ids: Sequence[str],
    conditions: Sequence[Condition],
    samples: Sequence[str],
    seeds: Sequence[int],
) -> None:
    if len(voice_ids) > _MAX_CATALOG_VOICES:
        message = "benchmark_workload_too_large"
        raise BenchmarkError(message)
    base_trials = len(voice_ids) * len(samples) * len(seeds)
    trial_count = base_trials * len(conditions)
    step_units = base_trials * sum(condition.num_steps for condition in conditions)
    if trial_count > _MAX_TRIALS or step_units > _MAX_STEP_UNITS:
        message = "benchmark_workload_too_large"
        raise BenchmarkError(message)


def _rounded(value: float) -> float:
    return round(value, 3)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percent_reduction(reference: float, candidate: float) -> float:
    if reference <= 0:
        return 0.0
    return (reference - candidate) / reference * 100.0


async def execute_benchmark(
    *,
    base_url: str,
    timeout_seconds: float,
    conditions: Sequence[Condition],
    samples: Sequence[str],
    seeds: Sequence[int],
    voice_selector: str | None,
    all_voices: bool,
) -> dict[str, object]:
    transport = httpx.AsyncHTTPTransport()
    async with AsyncIrodoriClient(
        base_url=base_url,
        timeout=timeout_seconds,
        transport=transport,
        max_response_bytes=_MAX_HTTP_RESPONSE_BYTES,
    ) as client:
        if all_voices:
            return await run_catalog_benchmark(
                client,
                conditions=conditions,
                samples=samples,
                seeds=seeds,
            )
        return await run_benchmark(
            client,
            conditions=conditions,
            samples=samples,
            seeds=seeds,
            voice_selector=voice_selector,
        )


def run_cli(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run_cli(_parse_args(argv)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.sample_limit <= len(V4_INFERENCE_SAMPLES):
        parser.error(f"--sample-limit must be between 1 and {len(V4_INFERENCE_SAMPLES)}")
    try:
        conditions = tuple(
            parse_condition(spec) for spec in (args.condition or _DEFAULT_CONDITION_SPECS)
        )
    except ValueError as error:
        parser.error(str(error))
    args.conditions = conditions
    return args


async def _run_cli(args: argparse.Namespace) -> int:
    try:
        async with asyncio.timeout(_MAX_RUN_SECONDS):
            summary = await execute_benchmark(
                base_url=args.base_url,
                timeout_seconds=args.timeout_seconds,
                conditions=args.conditions,
                samples=V4_INFERENCE_SAMPLES[: args.sample_limit],
                seeds=tuple(args.seed or _DEFAULT_SEEDS),
                voice_selector=args.voice,
                all_voices=args.all_voices,
            )
    except TimeoutError:
        _write_failure("benchmark_timeout")
        return 2
    except BenchmarkError as error:
        _write_failure(str(error))
        return 2
    except ClientError as error:
        _write_failure(error.code)
        return 2
    except ValidationError:
        _write_failure("invalid_response")
        return 2
    except ValueError:
        _write_failure("invalid_input")
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Irodori-TTS v4 inference without retaining audio or text.",
    )
    parser.add_argument(
        "--base-url",
        type=_loopback_base_url,
        default=_DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=_DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--sample-limit", type=int, default=len(V4_INFERENCE_SAMPLES))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--condition", action="append")
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument("--voice")
    voice_group.add_argument("--all-voices", action="store_true")
    return parser


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        message = "value must be a positive finite number"
        raise argparse.ArgumentTypeError(message)
    return value


def _loopback_base_url(raw: str) -> str:
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        port = parsed.port
        is_loopback = hostname is not None and ipaddress.ip_address(hostname).is_loopback
    except ValueError as error:
        message = "base URL must be an unambiguous loopback HTTP URL"
        raise argparse.ArgumentTypeError(message) from error
    valid_parts = (
        parsed.scheme in {"http", "https"},
        is_loopback,
        parsed.username is None,
        parsed.password is None,
        port is None or port > 0,
        not parsed.netloc.endswith(":"),
        parsed.path in {"", "/"},
        not parsed.query,
        not parsed.fragment,
    )
    if not all(valid_parts):
        message = "base URL must be an unambiguous loopback HTTP URL"
        raise argparse.ArgumentTypeError(message)
    return raw


def _safe_error_code(code: str) -> str:
    return code if code in _SAFE_FAILURE_CODES else "client_error"


def _write_failure(code: str) -> None:
    payload = {
        "schema_version": "irodori-v4-inference-benchmark/v1",
        "status": "failed",
        "code": _safe_error_code(code),
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run_cli())
