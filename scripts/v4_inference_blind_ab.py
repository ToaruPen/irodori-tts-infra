from __future__ import annotations

import argparse
import asyncio
import ctypes
import errno
import hashlib
import ipaddress
import json
import math
import os
import random
import re
import secrets
import shutil
import stat
import struct
import subprocess  # noqa: S404 - fixed argv launcher with isolated standard streams.
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from irodori_tts_infra.client import AsyncIrodoriClient, ClientError
from irodori_tts_infra.contracts import (
    CapabilitiesResponse,
    HealthResponse,
    SynthesisRequest,
    SynthesisResult,
)
from irodori_tts_infra.evaluation_samples import V4_INFERENCE_SAMPLES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

__all__ = ["V4_INFERENCE_SAMPLES"]

ConditionName = Literal["baseline", "candidate"]
Side = Literal["a", "b"]
Choice = Literal["a", "b", "same", "unsure"]
Reason = Literal["reading", "voice", "noise", "prosody", "emotion"]
Outcome = Literal["no_detected_degradation", "degraded", "inconclusive"]
ScoreBucket = Literal["candidate_wins", "baseline_wins", "same", "unsure"]

_BLIND_SEEDS = (101, 202)
_PAIR_COUNT = 12
_SAMPLE_COUNT = 6
_OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_WAV_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_WAV_BYTES = 96 * 1024 * 1024
_MAX_AUDIO_DURATION_SECONDS = 60.0
_MAX_METADATA_BYTES = 1024 * 1024
_UI_COPY_CHUNK_BYTES = 64 * 1024
_MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_RUN_SECONDS = 900.0
_BROWSER_LAUNCH_TIMEOUT_SECONDS = 10.0
_DEFAULT_BASE_URL = "http://127.0.0.1:8924"
_WAV_HEADER_BYTES = 12
_PCM16_BYTES = 2
_CHUNK_HEADER_BYTES = 8
_CANONICAL_FMT_BYTES = 16
_LIST_TYPE_BYTES = 4
_MACOS_AT_FDCWD = -2
_MACOS_RENAME_EXCL = 0x00000004
_LINUX_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 1
_MANIFEST_SCHEMA = "irodori-v4-inference-blind-ab-manifest/v1"
_ANSWER_KEY_SCHEMA = "irodori-v4-inference-blind-ab-answer-key/v1"
_RESULTS_SCHEMA = "irodori-v4-inference-blind-ab-results/v1"
_MANIFEST_PREFIX = "window.IRODORI_BLIND_AB_MANIFEST="
_UI_ASSET_NAMES = ("index.html", "review.js")
_REASONS = ("reading", "voice", "noise", "prosody", "emotion")
_REQUEST_ORDERS = (("baseline", "candidate"), ("candidate", "baseline"))
_UNSURE_THRESHOLD = 4
_SIGNIFICANCE_LEVEL = 0.05
_ASSET_ROOT = Path(__file__).parent / "assets/v4_inference_blind_ab"
_SAFE_FAILURE_CODES = frozenset(
    {
        "runtime_not_ready",
        "default_voice_unavailable",
        "runtime_generation_mismatch",
        "response_too_large",
        "invalid_wav",
        "audio_too_large",
        "audio_too_long",
        "blind_ab_timeout",
        "browser_open_failed",
        "output_exists",
        "unsafe_output_path",
        "packet_integrity_error",
        "invalid_results",
        "client_error",
    },
)


class BlindAbError(RuntimeError):
    """Stable blind AB contract validation error."""


class BlindAbClient(Protocol):
    async def health(self) -> HealthResponse: ...

    async def capabilities(self) -> CapabilitiesResponse: ...

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...


class ResultAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pair_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    choice: Choice
    reasons: tuple[Reason, ...] = ()

    @field_validator("reasons", mode="before")
    @classmethod
    def _tuple_reasons(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            message = "reasons must be an array"
            raise ValueError(message)  # noqa: TRY004 - Pydantic does not wrap validator TypeError.
        return tuple(value)

    @field_validator("reasons")
    @classmethod
    def _unique_reasons(cls, value: tuple[Reason, ...]) -> tuple[Reason, ...]:
        if len(value) != len(set(value)):
            message = "reasons must be unique"
            raise ValueError(message)
        return value


class ResultsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["irodori-v4-inference-blind-ab-results/v1"]
    packet_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answers: tuple[ResultAnswer, ...]

    @field_validator("answers", mode="before")
    @classmethod
    def _tuple_answers(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            message = "answers must be an array"
            raise ValueError(message)  # noqa: TRY004 - Pydantic does not wrap validator TypeError.
        return tuple(value)


@dataclass(frozen=True, slots=True)
class Condition:
    name: ConditionName
    num_steps: int
    schedule: Literal["linear", "sway"]


@dataclass(frozen=True, slots=True)
class PairPlan:
    pair_id: str
    sample_index: int
    seed: int
    baseline_side: Side
    request_order: tuple[ConditionName, ConditionName]
    a_audio_id: str
    b_audio_id: str


@dataclass(frozen=True, slots=True)
class ScoreDecision:
    p_value: float
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class WavMetadata:
    channels: int
    sample_rate: int
    frame_count: int
    duration_seconds: float


BASELINE = Condition(name="baseline", num_steps=24, schedule="linear")
CANDIDATE = Condition(name="candidate", num_steps=12, schedule="sway")


def _is_opaque_id(value: object) -> bool:
    return isinstance(value, str) and _OPAQUE_ID_RE.fullmatch(value) is not None


def _is_nonblank_stripped_string(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def new_opaque_id() -> str:
    return secrets.token_hex(16)


def build_request(
    *,
    text: str,
    voice_id: str,
    generation: str,
    seed: int,
    condition: Condition,
) -> SynthesisRequest:
    return SynthesisRequest(
        text=text,
        speaker=None,
        voice_id=voice_id,
        if_generation=generation,
        num_steps=condition.num_steps,
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0,
        cfg_scale_speaker=5.0,
        style="neutral",
        seed=seed,
        duration_scale=1.0,
        num_candidates=1,
        t_schedule_mode=condition.schedule,
        sway_coeff=-1.0,
    )


def validate_wav(wav_bytes: bytes) -> WavMetadata:
    audio_too_large = "audio_too_large"
    invalid_wav = "invalid_wav"
    if len(wav_bytes) > _MAX_WAV_BYTES:
        raise BlindAbError(audio_too_large)
    if len(wav_bytes) < _WAV_HEADER_BYTES or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise BlindAbError(invalid_wav)
    (riff_size,) = struct.unpack_from("<I", wav_bytes, 4)
    if riff_size + _CHUNK_HEADER_BYTES != len(wav_bytes):
        raise BlindAbError(invalid_wav)

    fmt_payload, data_size = _scan_wav_chunks(wav_bytes)

    format_tag, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", fmt_payload
    )
    expected_block_align = channels * _PCM16_BYTES
    if format_tag != 1 or channels <= 0 or sample_rate <= 0 or bits_per_sample != _PCM16_BYTES * 8:
        raise BlindAbError(invalid_wav)
    if (
        block_align != expected_block_align
        or byte_rate != sample_rate * block_align
        or data_size % block_align != 0
    ):
        raise BlindAbError(invalid_wav)
    frame_count = data_size // block_align
    if frame_count <= 0:
        raise BlindAbError(invalid_wav)
    duration_seconds = frame_count / sample_rate
    if duration_seconds > _MAX_AUDIO_DURATION_SECONDS:
        audio_too_long = "audio_too_long"
        raise BlindAbError(audio_too_long)
    return WavMetadata(
        channels=channels,
        sample_rate=sample_rate,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )


def _scan_wav_chunks(wav_bytes: bytes) -> tuple[bytes, int]:
    invalid_wav = "invalid_wav"
    offset = _WAV_HEADER_BYTES
    fmt_payload: bytes | None = None
    data_size: int | None = None
    while offset < len(wav_bytes):
        if offset + _CHUNK_HEADER_BYTES > len(wav_bytes):
            raise BlindAbError(invalid_wav)
        chunk_id = wav_bytes[offset : offset + 4]
        (chunk_size,) = struct.unpack_from("<I", wav_bytes, offset + 4)
        payload_start = offset + _CHUNK_HEADER_BYTES
        payload_end = payload_start + chunk_size
        padded_end = payload_end + (chunk_size % 2)
        if payload_end > len(wav_bytes) or padded_end > len(wav_bytes):
            raise BlindAbError(invalid_wav)
        if chunk_id == b"fmt ":
            if (
                fmt_payload is not None
                or data_size is not None
                or chunk_size != _CANONICAL_FMT_BYTES
            ):
                raise BlindAbError(invalid_wav)
            fmt_payload = wav_bytes[payload_start:payload_end]
        elif chunk_id == b"LIST":
            if (
                fmt_payload is None
                or data_size is not None
                or chunk_size < _LIST_TYPE_BYTES
                or chunk_size > _MAX_METADATA_BYTES
            ):
                raise BlindAbError(invalid_wav)
        elif chunk_id == b"data":
            if (
                fmt_payload is None
                or data_size is not None
                or chunk_size <= 0
                or padded_end != len(wav_bytes)
            ):
                raise BlindAbError(invalid_wav)
            data_size = chunk_size
        else:
            raise BlindAbError(invalid_wav)
        offset = padded_end
    if fmt_payload is None or data_size is None:
        raise BlindAbError(invalid_wav)
    return fmt_payload, data_size


def _remove_owned_temporary(temporary: Path) -> None:
    if temporary.is_symlink() or temporary.is_file():
        temporary.unlink()
    elif temporary.is_dir():
        shutil.rmtree(temporary)


def _reject_existing_output(final: Path) -> None:
    if os.path.lexists(final):
        output_exists = "output_exists"
        raise BlindAbError(output_exists)


def _raise_rename_error(error_number: int, destination: Path) -> None:
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)  # noqa: PTH104 - Windows rename is create-only.
        return

    if sys.platform == "darwin":
        symbol = "renameatx_np"
        directory_fd = _MACOS_AT_FDCWD
        flags = _MACOS_RENAME_EXCL
    elif sys.platform.startswith("linux"):
        symbol = "renameat2"
        directory_fd = _LINUX_AT_FDCWD
        flags = _LINUX_RENAME_NOREPLACE
    else:
        unsupported = errno.ENOTSUP
        raise OSError(unsupported, os.strerror(unsupported), destination)
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, symbol, None)
    if rename is None:
        unavailable = errno.ENOSYS
        raise OSError(unavailable, os.strerror(unavailable), destination)
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(
        directory_fd,
        os.fsencode(source),
        directory_fd,
        os.fsencode(destination),
        flags,
    )
    if result != 0:
        _raise_rename_error(ctypes.get_errno(), destination)


@contextmanager
def atomic_output_directory(destination: Path) -> Iterator[Path]:
    unsafe_output_path = "unsafe_output_path"
    raw = destination.expanduser()
    if raw.name in {"", ".", ".."} or raw == Path(raw.anchor):
        raise BlindAbError(unsafe_output_path)
    expanded = raw.absolute()
    try:
        parent = expanded.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise BlindAbError(unsafe_output_path) from error
    if not parent.is_dir():
        raise BlindAbError(unsafe_output_path)
    final = parent / expanded.name
    _reject_existing_output(final)
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=parent))
    except OSError as error:
        raise BlindAbError(unsafe_output_path) from error
    try:
        yield temporary
        _reject_existing_output(final)
        try:
            _rename_noreplace(temporary, final)
        except FileExistsError as error:
            output_exists = "output_exists"
            raise BlindAbError(output_exists) from error
        except OSError as error:
            raise BlindAbError(unsafe_output_path) from error
    except BaseException:
        _remove_owned_temporary(temporary)
        raise


async def _require_runtime(client: BlindAbClient) -> tuple[str, str]:
    runtime_not_ready = "runtime_not_ready"
    health = await client.health()
    if health.status != "ok" or not health.model_loaded:
        raise BlindAbError(runtime_not_ready)
    capabilities = await client.capabilities()
    if not capabilities.ready or capabilities.readiness != "ready":
        raise BlindAbError(runtime_not_ready)
    defaults = [voice.id for voice in capabilities.voices if voice.default]
    if len(defaults) != 1:
        default_voice_unavailable = "default_voice_unavailable"
        raise BlindAbError(default_voice_unavailable)
    return defaults[0], capabilities.generation


async def _generate_wavs(
    client: BlindAbClient,
    *,
    plans: tuple[PairPlan, ...],
    samples: tuple[str, ...],
    voice_id: str,
    generation: str,
) -> dict[str, bytes]:
    condition_by_name = {BASELINE.name: BASELINE, CANDIDATE.name: CANDIDATE}
    wav_by_audio_id: dict[str, bytes] = {}
    total_wav_bytes = 0
    for plan in plans:
        result_by_condition: dict[ConditionName, bytes] = {}
        for condition_name in plan.request_order:
            request = build_request(
                text=samples[plan.sample_index],
                voice_id=voice_id,
                generation=generation,
                seed=plan.seed,
                condition=condition_by_name[condition_name],
            )
            result = await client.synthesize(request)
            if (
                result.segment_index != 0
                or result.content_type != "audio/wav"
                or not math.isfinite(result.elapsed_seconds)
                or result.elapsed_seconds < 0
            ):
                invalid_response = "invalid_response"
                raise BlindAbError(invalid_response)
            validate_wav(result.wav_bytes)
            total_wav_bytes += len(result.wav_bytes)
            if total_wav_bytes > _MAX_TOTAL_WAV_BYTES:
                audio_too_large = "audio_too_large"
                raise BlindAbError(audio_too_large)
            result_by_condition[condition_name] = result.wav_bytes
        baseline_audio = result_by_condition["baseline"]
        candidate_audio = result_by_condition["candidate"]
        wav_by_audio_id[plan.a_audio_id] = (
            baseline_audio if plan.baseline_side == "a" else candidate_audio
        )
        wav_by_audio_id[plan.b_audio_id] = (
            baseline_audio if plan.baseline_side == "b" else candidate_audio
        )
    return wav_by_audio_id


def _resolved_destination(destination: Path) -> Path:
    return destination.expanduser().absolute().resolve(strict=True)


async def prepare_packet(
    client: BlindAbClient,
    *,
    destination: Path,
    samples: tuple[str, ...] = V4_INFERENCE_SAMPLES,
    seeds: tuple[int, ...] = _BLIND_SEEDS,
    randomization_seed: int | None = None,
    id_factory: Callable[[], str] = new_opaque_id,
) -> Path:
    effective_seed = secrets.randbits(256) if randomization_seed is None else randomization_seed
    packet_id = id_factory()
    if not _is_opaque_id(packet_id):
        invalid_artifact = "invalid_artifact"
        raise BlindAbError(invalid_artifact)
    plans = build_pair_plans(
        samples=samples,
        seeds=seeds,
        randomization_seed=effective_seed,
        id_factory=id_factory,
    )
    with atomic_output_directory(destination) as temporary:
        voice_id, generation = await _require_runtime(client)
        wav_by_audio_id = await _generate_wavs(
            client,
            plans=plans,
            samples=samples,
            voice_id=voice_id,
            generation=generation,
        )
        ui_sha256 = _copy_ui_assets(temporary)
        manifest_wrapper, answer_key = build_artifact_payloads(
            packet_id=packet_id,
            plans=plans,
            samples=samples,
            randomization_seed=effective_seed,
            voice_id=voice_id,
            generation=generation,
            wav_by_audio_id=wav_by_audio_id,
            ui_sha256=ui_sha256,
        )
        _write_packet(
            temporary,
            manifest_wrapper=manifest_wrapper,
            answer_key=answer_key,
            wav_by_audio_id=wav_by_audio_id,
        )
    return _resolved_destination(destination)


def _write_packet(
    root: Path,
    *,
    manifest_wrapper: dict[str, object],
    answer_key: dict[str, object],
    wav_by_audio_id: dict[str, bytes],
) -> None:
    try:
        packet = root / "packet"
        audio = packet / "audio"
        private = root / "private"
        audio.mkdir()
        private.mkdir()
        manifest_bytes = _MANIFEST_PREFIX.encode() + canonical_json_bytes(manifest_wrapper) + b";\n"
        (packet / "manifest.js").write_bytes(manifest_bytes)
        for audio_id, wav_bytes in wav_by_audio_id.items():
            (audio / f"{audio_id}.wav").write_bytes(wav_bytes)
        (private / "answer-key.json").write_bytes(canonical_json_bytes(answer_key) + b"\n")
    except OSError as error:
        client_error = "client_error"
        raise BlindAbError(client_error) from error


def _copy_ui_assets(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    try:
        packet = root / "packet"
        packet.mkdir()
        for name in _UI_ASSET_NAMES:
            hashes[name] = _copy_ui_asset(_ASSET_ROOT / name, packet / name)
    except (OSError, ValueError) as error:
        client_error = "client_error"
        raise BlindAbError(client_error) from error
    return hashes


def _copy_ui_asset(source: Path, destination: Path) -> str:
    descriptor = _open_regular_fd(source)
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with destination.open("xb") as sink:
            while chunk := os.read(descriptor, _UI_COPY_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > _MAX_METADATA_BYTES:
                    raise ValueError
                if sink.write(chunk) != len(chunk):
                    raise OSError
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def build_pair_plans(
    *,
    samples: Sequence[str],
    seeds: Sequence[int],
    randomization_seed: int,
    id_factory: Callable[[], str] = new_opaque_id,
) -> tuple[PairPlan, ...]:
    combinations = [(sample_index, seed) for sample_index in range(len(samples)) for seed in seeds]
    if len(combinations) != _PAIR_COUNT:
        message = "blind AB requires exactly 12 sample/seed pairs"
        raise ValueError(message)

    rng = random.Random(randomization_seed)  # noqa: S311 - deterministic local randomization
    sides_per_condition = _PAIR_COUNT // 2
    baseline_sides: list[Side] = [
        "a" if index < sides_per_condition else "b" for index in range(_PAIR_COUNT)
    ]
    rng.shuffle(baseline_sides)
    plans: list[PairPlan] = []
    seen_ids: set[str] = set()

    for (sample_index, seed), baseline_side in zip(combinations, baseline_sides, strict=True):
        request_order: tuple[ConditionName, ConditionName] = (
            ("baseline", "candidate") if rng.getrandbits(1) == 0 else ("candidate", "baseline")
        )
        pair_id, a_audio_id, b_audio_id = (id_factory(), id_factory(), id_factory())
        generated_ids = (pair_id, a_audio_id, b_audio_id)
        if any(not _is_opaque_id(identifier) for identifier in generated_ids):
            message = "opaque IDs must be 128-bit lowercase hex"
            raise ValueError(message)
        if len(set(generated_ids)) != len(generated_ids) or seen_ids.intersection(generated_ids):
            message = "opaque IDs must be unique"
            raise ValueError(message)
        seen_ids.update(generated_ids)
        plans.append(
            PairPlan(
                pair_id=pair_id,
                sample_index=sample_index,
                seed=seed,
                baseline_side=baseline_side,
                request_order=request_order,
                a_audio_id=a_audio_id,
                b_audio_id=b_audio_id,
            ),
        )

    rng.shuffle(plans)
    return tuple(plans)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_artifact_semantics(
    *,
    plans: tuple[PairPlan, ...],
    samples: tuple[str, ...],
    voice_id: str,
    generation: str,
) -> None:
    invalid_artifact = "invalid_artifact"
    if (
        not isinstance(samples, tuple)
        or len(samples) != _SAMPLE_COUNT
        or any(not _is_nonblank_stripped_string(sample) for sample in samples)
    ):
        raise BlindAbError(invalid_artifact)
    if not _is_nonblank_stripped_string(voice_id) or not _is_nonblank_stripped_string(generation):
        raise BlindAbError(invalid_artifact)
    if len(plans) != _PAIR_COUNT:
        raise BlindAbError(invalid_artifact)
    if any(
        not isinstance(plan.sample_index, int)
        or isinstance(plan.sample_index, bool)
        or not 0 <= plan.sample_index < len(samples)
        for plan in plans
    ):
        raise BlindAbError(invalid_artifact)
    if any(
        not isinstance(plan.seed, int)
        or isinstance(plan.seed, bool)
        or (plan.seed != _BLIND_SEEDS[0] and plan.seed != _BLIND_SEEDS[1])
        for plan in plans
    ):
        raise BlindAbError(invalid_artifact)
    if any(
        not isinstance(plan.baseline_side, str) or plan.baseline_side not in {"a", "b"}
        for plan in plans
    ):
        raise BlindAbError(invalid_artifact)
    if sum(plan.baseline_side == "a" for plan in plans) != _PAIR_COUNT // 2:
        raise BlindAbError(invalid_artifact)
    if any(
        not isinstance(plan.request_order, tuple)
        or any(not isinstance(condition, str) for condition in plan.request_order)
        or plan.request_order not in _REQUEST_ORDERS
        for plan in plans
    ):
        raise BlindAbError(invalid_artifact)

    expected_sample_seeds = {
        (sample_index, seed) for sample_index in range(len(samples)) for seed in _BLIND_SEEDS
    }
    actual_sample_seeds = {(plan.sample_index, plan.seed) for plan in plans}
    if actual_sample_seeds != expected_sample_seeds:
        raise BlindAbError(invalid_artifact)


def _validate_artifact_ids_and_audio(
    *,
    packet_id: str,
    plans: tuple[PairPlan, ...],
    randomization_seed: int,
    wav_by_audio_id: dict[str, bytes],
) -> tuple[list[str], list[str]]:
    pair_ids = [plan.pair_id for plan in plans]
    audio_ids = [audio_id for plan in plans for audio_id in (plan.a_audio_id, plan.b_audio_id)]
    invalid_artifact = "invalid_artifact"
    if not _is_opaque_id(packet_id):
        raise BlindAbError(invalid_artifact)
    if any(not _is_opaque_id(pair_id) for pair_id in pair_ids):
        raise BlindAbError(invalid_artifact)
    if len(set(pair_ids)) != _PAIR_COUNT:
        raise BlindAbError(invalid_artifact)
    if any(not _is_opaque_id(audio_id) for audio_id in audio_ids):
        raise BlindAbError(invalid_artifact)
    if len(set(audio_ids)) != _PAIR_COUNT * 2:
        raise BlindAbError(invalid_artifact)
    if set(pair_ids).intersection(audio_ids):
        raise BlindAbError(invalid_artifact)
    if (
        not isinstance(randomization_seed, int)
        or isinstance(randomization_seed, bool)
        or not 0 <= randomization_seed < 2**256
    ):
        raise BlindAbError(invalid_artifact)
    if set(wav_by_audio_id) != set(audio_ids) or any(
        not isinstance(value, bytes) for value in wav_by_audio_id.values()
    ):
        raise BlindAbError(invalid_artifact)
    return pair_ids, audio_ids


def build_artifact_payloads(
    *,
    packet_id: str,
    plans: tuple[PairPlan, ...],
    samples: tuple[str, ...],
    randomization_seed: int,
    voice_id: str,
    generation: str,
    wav_by_audio_id: dict[str, bytes],
    ui_sha256: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    _validate_artifact_semantics(
        plans=plans,
        samples=samples,
        voice_id=voice_id,
        generation=generation,
    )
    _, audio_ids = _validate_artifact_ids_and_audio(
        packet_id=packet_id,
        plans=plans,
        randomization_seed=randomization_seed,
        wav_by_audio_id=wav_by_audio_id,
    )
    if (
        not isinstance(ui_sha256, dict)
        or set(ui_sha256) != set(_UI_ASSET_NAMES)
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in ui_sha256.items()
        )
    ):
        invalid_artifact = "invalid_artifact"
        raise BlindAbError(invalid_artifact)

    public_pairs = [
        {
            "pair_id": plan.pair_id,
            "text": samples[plan.sample_index],
            "a_audio": f"audio/{plan.a_audio_id}.wav",
            "b_audio": f"audio/{plan.b_audio_id}.wav",
        }
        for plan in plans
    ]
    private_pairs = [
        {
            "pair_id": plan.pair_id,
            "sample_index": plan.sample_index,
            "seed": plan.seed,
            "baseline_side": plan.baseline_side,
            "request_order": list(plan.request_order),
        }
        for plan in plans
    ]
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "packet_id": packet_id,
        "pairs": public_pairs,
        "reasons": list(_REASONS),
    }
    manifest_sha256 = sha256_hex(canonical_json_bytes(manifest))
    manifest_wrapper: dict[str, object] = {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
    }
    answer_key: dict[str, object] = {
        "schema_version": _ANSWER_KEY_SCHEMA,
        "packet_id": packet_id,
        "manifest_sha256": manifest_sha256,
        "audio_sha256": {
            f"audio/{audio_id}.wav": sha256_hex(wav_by_audio_id[audio_id])
            for audio_id in sorted(audio_ids)
        },
        "pairs": private_pairs,
        "randomization_seed": f"{randomization_seed:064x}",
        "runtime": {
            "voice_id_sha256": sha256_hex(voice_id.encode()),
            "generation_sha256": sha256_hex(generation.encode()),
        },
        "ui_sha256": dict(ui_sha256),
    }
    return manifest_wrapper, answer_key


def validate_results(
    value: object,
    *,
    expected_pair_ids: set[str],
) -> ResultsPayload:
    invalid_results = "invalid_results"
    try:
        results = ResultsPayload.model_validate(value)
    except ValidationError as error:
        raise BlindAbError(invalid_results) from error

    actual_pair_ids = [answer.pair_id for answer in results.answers]
    if (
        len(actual_pair_ids) != _PAIR_COUNT
        or len(set(actual_pair_ids)) != _PAIR_COUNT
        or set(actual_pair_ids) != expected_pair_ids
    ):
        raise BlindAbError(invalid_results)
    return results


def exact_baseline_preference_p_value(*, baseline_wins: int, decisive: int) -> float:
    if (
        not isinstance(baseline_wins, int)
        or isinstance(baseline_wins, bool)
        or not isinstance(decisive, int)
        or isinstance(decisive, bool)
    ):
        message = "invalid binomial counts"
        raise ValueError(message)  # noqa: TRY004 - invalid count values share one error contract.
    if baseline_wins < 0 or decisive < 0 or baseline_wins > decisive:
        message = "invalid binomial counts"
        raise ValueError(message)
    if decisive == 0:
        return 1.0
    numerator = sum(math.comb(decisive, value) for value in range(baseline_wins, decisive + 1))
    denominator = 1 << decisive
    return numerator / denominator


def classify_score(
    *,
    candidate_wins: int,
    baseline_wins: int,
    same: int,
    unsure: int,
) -> ScoreDecision:
    counts = (candidate_wins, baseline_wins, same, unsure)
    if (
        any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts)
        or sum(counts) != _PAIR_COUNT
    ):
        message = "score counts must be non-negative and sum to 12"
        raise ValueError(message)
    p_value = exact_baseline_preference_p_value(
        baseline_wins=baseline_wins,
        decisive=candidate_wins + baseline_wins,
    )
    if unsure >= _UNSURE_THRESHOLD:
        outcome: Outcome = "inconclusive"
    elif baseline_wins > candidate_wins and p_value <= _SIGNIFICANCE_LEVEL:
        outcome = "degraded"
    else:
        outcome = "no_detected_degradation"
    return ScoreDecision(p_value=p_value, outcome=outcome)


def summarize_answers(
    results: ResultsPayload,
    *,
    baseline_side_by_pair: dict[str, Side],
) -> dict[str, object]:
    result_pair_ids = {answer.pair_id for answer in results.answers}
    if set(baseline_side_by_pair) != result_pair_ids or any(
        not isinstance(side, str) or side not in {"a", "b"}
        for side in baseline_side_by_pair.values()
    ):
        message = "invalid_results"
        raise BlindAbError(message)

    counts: dict[ScoreBucket, int] = {
        "candidate_wins": 0,
        "baseline_wins": 0,
        "same": 0,
        "unsure": 0,
    }
    reasons: dict[str, dict[ScoreBucket, int]] = {
        reason: {
            "candidate_wins": 0,
            "baseline_wins": 0,
            "same": 0,
            "unsure": 0,
        }
        for reason in _REASONS
    }
    for answer in results.answers:
        if answer.choice == "same":
            bucket: ScoreBucket = "same"
        elif answer.choice == "unsure":
            bucket = "unsure"
        elif answer.choice == baseline_side_by_pair[answer.pair_id]:
            bucket = "baseline_wins"
        else:
            bucket = "candidate_wins"
        counts[bucket] += 1
        for reason in answer.reasons:
            reasons[reason][bucket] += 1

    decision = classify_score(
        candidate_wins=counts["candidate_wins"],
        baseline_wins=counts["baseline_wins"],
        same=counts["same"],
        unsure=counts["unsure"],
    )
    return {
        "candidate_wins": counts["candidate_wins"],
        "baseline_wins": counts["baseline_wins"],
        "same": counts["same"],
        "unsure": counts["unsure"],
        "decisive": counts["candidate_wins"] + counts["baseline_wins"],
        "p_value": decision.p_value,
        "outcome": decision.outcome,
        "reason_breakdown": reasons,
    }


def _secure_open_flags(*, directory: bool) -> int:
    required_names = ["O_CLOEXEC", "O_NOFOLLOW"]
    if directory:
        required_names.append("O_DIRECTORY")
    else:
        required_names.append("O_NONBLOCK")
    if (
        os.name == "nt"
        or os.open not in os.supports_dir_fd
        or os.scandir not in os.supports_fd
        or any(not hasattr(os, name) for name in required_names)
    ):
        raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP))
    return os.O_RDONLY | sum(cast("int", getattr(os, name)) for name in required_names)


@contextmanager
def _open_directory_fd(
    path: str | Path,
    *,
    dir_fd: int | None = None,
) -> Iterator[int]:
    flags = _secure_open_flags(directory=True)
    descriptor = os.open(path, flags) if dir_fd is None else os.open(path, flags, dir_fd=dir_fd)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise ValueError
        yield descriptor
    finally:
        os.close(descriptor)


def _open_regular_fd(path: str | Path, *, dir_fd: int | None = None) -> int:
    flags = _secure_open_flags(directory=False)
    descriptor = os.open(path, flags) if dir_fd is None else os.open(path, flags, dir_fd=dir_fd)
    try:
        is_regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
    except BaseException:
        os.close(descriptor)
        raise
    if not is_regular:
        os.close(descriptor)
        raise ValueError
    return descriptor


def _read_bounded_regular(
    path: str | Path,
    *,
    limit: int,
    dir_fd: int | None = None,
) -> bytes:
    descriptor = _open_regular_fd(path, dir_fd=dir_fd)
    try:
        opened_stat = os.fstat(descriptor)
        if opened_stat.st_size > limit:
            raise ValueError
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(value) > limit:
        raise ValueError
    return value


def _parse_json_bytes(value: bytes) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError
            result[key] = item
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError

    return json.loads(
        value.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _require_exact_entries(directory_fd: int, expected: set[str]) -> None:
    seen: set[str] = set()
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            if len(seen) == len(expected):
                raise ValueError
            seen.add(entry.name)
    if seen != expected:
        raise ValueError


def _require_exact_keys(value: object, expected: set[str]) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or any(not isinstance(key, str) for key in value)
    ):
        raise ValueError
    return cast("dict[str, object]", value)


def _require_list(value: object, *, length: int) -> list[object]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError
    return value


def _require_hex(value: object, *, length: int) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError
    return value


def _parse_manifest(packet_fd: int) -> tuple[dict[str, object], str]:
    raw = _read_bounded_regular(
        "manifest.js",
        limit=_MAX_METADATA_BYTES,
        dir_fd=packet_fd,
    )
    prefix = _MANIFEST_PREFIX.encode()
    suffix = b";\n"
    if not raw.startswith(prefix) or not raw.endswith(suffix):
        raise ValueError
    wrapper = _require_exact_keys(
        _parse_json_bytes(raw[len(prefix) : -len(suffix)]),
        {"manifest", "manifest_sha256"},
    )
    manifest = _require_exact_keys(
        wrapper["manifest"],
        {"schema_version", "packet_id", "pairs", "reasons"},
    )
    if manifest["schema_version"] != _MANIFEST_SCHEMA:
        raise ValueError
    _require_hex(manifest["packet_id"], length=32)
    manifest_digest = _require_hex(wrapper["manifest_sha256"], length=64)
    if sha256_hex(canonical_json_bytes(manifest)) != manifest_digest:
        raise ValueError
    if manifest["reasons"] != list(_REASONS):
        raise ValueError

    pairs = _require_list(manifest["pairs"], length=_PAIR_COUNT)
    pair_ids: set[str] = set()
    audio_ids: set[str] = set()
    for pair_value in pairs:
        pair = _require_exact_keys(pair_value, {"pair_id", "text", "a_audio", "b_audio"})
        pair_id = _require_hex(pair["pair_id"], length=32)
        if pair_id in pair_ids or not _is_nonblank_stripped_string(pair["text"]):
            raise ValueError
        pair_ids.add(pair_id)
        for side in ("a", "b"):
            audio_path = pair[f"{side}_audio"]
            if not isinstance(audio_path, str):
                raise TypeError
            match = re.fullmatch(r"audio/([0-9a-f]{32})\.wav", audio_path)
            if match is None or match.group(1) in audio_ids:
                raise ValueError
            audio_ids.add(match.group(1))
    if len(pair_ids) != _PAIR_COUNT or len(audio_ids) != _PAIR_COUNT * 2:
        raise ValueError
    return manifest, manifest_digest


def _validate_answer_key(  # noqa: PLR0914, PLR0915 - explicit private checks remain auditable.
    private_fd: int,
    *,
    manifest: dict[str, object],
    manifest_digest: str,
) -> tuple[dict[str, Side], dict[str, str], dict[str, str]]:
    answer_key = _require_exact_keys(
        _parse_json_bytes(
            _read_bounded_regular(
                "answer-key.json",
                limit=_MAX_METADATA_BYTES,
                dir_fd=private_fd,
            ),
        ),
        {
            "schema_version",
            "packet_id",
            "manifest_sha256",
            "audio_sha256",
            "pairs",
            "randomization_seed",
            "runtime",
            "ui_sha256",
        },
    )
    if answer_key["schema_version"] != _ANSWER_KEY_SCHEMA:
        raise ValueError
    packet_id = _require_hex(answer_key["packet_id"], length=32)
    if packet_id != manifest["packet_id"]:
        raise ValueError
    if _require_hex(answer_key["manifest_sha256"], length=64) != manifest_digest:
        raise ValueError
    randomization_seed = int(_require_hex(answer_key["randomization_seed"], length=64), 16)
    runtime = _require_exact_keys(
        answer_key["runtime"],
        {"voice_id_sha256", "generation_sha256"},
    )
    _require_hex(runtime["voice_id_sha256"], length=64)
    _require_hex(runtime["generation_sha256"], length=64)

    public_pair_values = cast("list[object]", manifest["pairs"])
    public_pairs = {
        cast("dict[str, object]", pair)["pair_id"]: cast("dict[str, object]", pair)
        for pair in public_pair_values
    }
    private_pairs = _require_list(answer_key["pairs"], length=_PAIR_COUNT)
    baseline_side_by_pair: dict[str, Side] = {}
    sample_seeds: set[tuple[int, int]] = set()
    actual_randomized_metadata: list[tuple[int, int, str, list[object]]] = []
    private_pair_ids: list[str] = []
    for pair_value in private_pairs:
        pair = _require_exact_keys(
            pair_value,
            {"pair_id", "sample_index", "seed", "baseline_side", "request_order"},
        )
        pair_id = _require_hex(pair["pair_id"], length=32)
        sample_index = pair["sample_index"]
        seed = pair["seed"]
        baseline_side = pair["baseline_side"]
        request_order = pair["request_order"]
        if (
            pair_id not in public_pairs  # noqa: PLR0916 - one fail-closed record contract.
            or pair_id in baseline_side_by_pair
            or not isinstance(sample_index, int)
            or isinstance(sample_index, bool)
            or not 0 <= sample_index < _SAMPLE_COUNT
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed not in _BLIND_SEEDS
            or not isinstance(baseline_side, str)
            or baseline_side not in {"a", "b"}
            or request_order not in [list(order) for order in _REQUEST_ORDERS]
            or public_pairs[pair_id]["text"] != V4_INFERENCE_SAMPLES[sample_index]
        ):
            raise ValueError
        baseline_side_by_pair[pair_id] = cast("Side", baseline_side)
        sample_seeds.add((sample_index, seed))
        private_pair_ids.append(pair_id)
        actual_randomized_metadata.append(
            (sample_index, seed, baseline_side, cast("list[object]", request_order)),
        )
    generated_ids = iter(f"{value:032x}" for value in range(1, _PAIR_COUNT * 3 + 1))
    expected_plans = build_pair_plans(
        samples=V4_INFERENCE_SAMPLES,
        seeds=_BLIND_SEEDS,
        randomization_seed=randomization_seed,
        id_factory=lambda: next(generated_ids),
    )
    expected_randomized_metadata = [
        (
            plan.sample_index,
            plan.seed,
            plan.baseline_side,
            list(plan.request_order),
        )
        for plan in expected_plans
    ]
    public_pair_ids = [
        cast("str", cast("dict[str, object]", pair)["pair_id"]) for pair in public_pair_values
    ]
    expected_sample_seeds = {
        (sample_index, seed) for sample_index in range(_SAMPLE_COUNT) for seed in _BLIND_SEEDS
    }
    metadata_matches = (
        private_pair_ids == public_pair_ids
        and actual_randomized_metadata == expected_randomized_metadata
    )
    coverage_matches = (
        set(baseline_side_by_pair) == set(public_pairs)
        and sample_seeds == expected_sample_seeds
        and sum(side == "a" for side in baseline_side_by_pair.values()) == _PAIR_COUNT // 2
    )
    if not metadata_matches or not coverage_matches:
        raise ValueError

    audio_hash_value = answer_key["audio_sha256"]
    if not isinstance(audio_hash_value, dict):
        raise TypeError
    audio_hashes = cast("dict[object, object]", audio_hash_value)
    expected_paths = {
        cast("str", pair[path_key])
        for pair in public_pairs.values()
        for path_key in ("a_audio", "b_audio")
    }
    if set(audio_hashes) != expected_paths:
        raise ValueError
    validated_hashes: dict[str, str] = {}
    for path, digest in audio_hashes.items():
        if not isinstance(path, str):
            raise TypeError
        validated_hashes[path] = _require_hex(digest, length=64)
    ui_hash_value = _require_exact_keys(answer_key["ui_sha256"], set(_UI_ASSET_NAMES))
    ui_hashes = {name: _require_hex(ui_hash_value[name], length=64) for name in _UI_ASSET_NAMES}
    return baseline_side_by_pair, validated_hashes, ui_hashes


def _validate_packet(packet_root: Path) -> tuple[dict[str, object], str, dict[str, Side]]:
    trusted_ui_assets = {
        name: _read_bounded_regular(_ASSET_ROOT / name, limit=_MAX_METADATA_BYTES)
        for name in _UI_ASSET_NAMES
    }
    with _open_directory_fd(packet_root) as root_fd:
        _require_exact_entries(root_fd, {"packet", "private"})
        with (
            _open_directory_fd("packet", dir_fd=root_fd) as packet_fd,
            _open_directory_fd("private", dir_fd=root_fd) as private_fd,
            _open_directory_fd("audio", dir_fd=packet_fd) as audio_fd,
        ):
            _require_exact_entries(
                packet_fd,
                {"index.html", "review.js", "manifest.js", "audio"},
            )
            _require_exact_entries(private_fd, {"answer-key.json"})
            manifest, manifest_digest = _parse_manifest(packet_fd)
            baseline_sides, audio_hashes, ui_hashes = _validate_answer_key(
                private_fd,
                manifest=manifest,
                manifest_digest=manifest_digest,
            )
            for name, expected_digest in ui_hashes.items():
                value = _read_bounded_regular(
                    name,
                    limit=_MAX_METADATA_BYTES,
                    dir_fd=packet_fd,
                )
                if value != trusted_ui_assets[name] or sha256_hex(value) != expected_digest:
                    raise ValueError
            expected_names = {Path(path).name for path in audio_hashes}
            _require_exact_entries(audio_fd, expected_names)
            for relative_path, expected_digest in audio_hashes.items():
                wav_bytes = _read_bounded_regular(
                    Path(relative_path).name,
                    limit=_MAX_WAV_BYTES,
                    dir_fd=audio_fd,
                )
                validate_wav(wav_bytes)
                if sha256_hex(wav_bytes) != expected_digest:
                    raise ValueError
    return manifest, manifest_digest, baseline_sides


def _require_packet_integrity(
    packet_root: Path,
) -> tuple[dict[str, object], str, dict[str, Side]]:
    try:
        return _validate_packet(packet_root)
    except Exception as error:
        packet_integrity_error = "packet_integrity_error"
        raise BlindAbError(packet_integrity_error) from error


def score_packet(packet_root: Path, results_path: Path) -> dict[str, object]:
    manifest, manifest_digest, baseline_sides = _require_packet_integrity(packet_root)

    invalid_results = "invalid_results"
    try:
        results_value = _parse_json_bytes(
            _read_bounded_regular(results_path, limit=_MAX_METADATA_BYTES),
        )
        results = validate_results(results_value, expected_pair_ids=set(baseline_sides))
        if results.packet_id != manifest["packet_id"] or results.manifest_sha256 != manifest_digest:
            raise BlindAbError(invalid_results)  # noqa: TRY301 - stable boundary code.
        summary = summarize_answers(results, baseline_side_by_pair=baseline_sides)
    except Exception as error:
        raise BlindAbError(invalid_results) from error
    return {
        "schema_version": "irodori-v4-inference-blind-ab-score/v1",
        "status": "complete",
        **summary,
    }


def _launch_browser(
    file_uri: str,
    timeout_seconds: float = _BROWSER_LAUNCH_TIMEOUT_SECONDS,
) -> bool:
    if sys.platform == "darwin":
        argv = ["open", file_uri]
    elif sys.platform.startswith("linux"):
        argv = ["xdg-open", file_uri]
    else:
        return False
    try:
        result = subprocess.run(  # noqa: S603 - fixed platform executable and opaque URI argv.
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception:  # noqa: BLE001 - launcher failures must not expose platform details.
        return False
    return result.returncode == 0


async def execute_prepare(
    *,
    base_url: str,
    output_dir: Path,
    open_browser: bool = False,
) -> Path:
    transport = httpx.AsyncHTTPTransport()
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        async with asyncio.timeout(_MAX_RUN_SECONDS):
            async with AsyncIrodoriClient(
                base_url=base_url,
                timeout=None,
                transport=transport,
                max_response_bytes=_MAX_HTTP_RESPONSE_BYTES,
            ) as client:
                packet_root = await prepare_packet(client, destination=output_dir)
            if open_browser:
                _require_packet_integrity(packet_root)
                remaining = _MAX_RUN_SECONDS - (loop.time() - started)
                if remaining <= 0:
                    raise TimeoutError
                launch_timeout = min(_BROWSER_LAUNCH_TIMEOUT_SECONDS, remaining)
                opened = await asyncio.to_thread(
                    _launch_browser,
                    (packet_root / "packet/index.html").as_uri(),
                    launch_timeout,
                )
                if not opened:
                    browser_open_failed = "browser_open_failed"
                    raise BlindAbError(browser_open_failed)
            return packet_root
    finally:
        await transport.aclose()


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
        "?" not in raw,
        "#" not in raw,
        not parsed.query,
        not parsed.fragment,
    )
    if not all(valid_parts):
        message = "base URL must be an unambiguous loopback HTTP URL"
        raise argparse.ArgumentTypeError(message)
    return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or score a fixed-profile local blind AB listening packet.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base-url", type=_loopback_base_url, default=_DEFAULT_BASE_URL)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--open", action="store_true", dest="open_browser")
    score = subparsers.add_parser("score")
    score.add_argument("--packet-root", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _safe_error_code(code: str) -> str:
    return code if code in _SAFE_FAILURE_CODES else "client_error"


def _write_failure(code: str) -> None:
    payload = {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": _safe_error_code(code),
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


def _prepare_success(packet_root: Path) -> dict[str, object]:
    return {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "complete",
        "packet_root": str(packet_root),
        "answer_key": str(packet_root / "private/answer-key.json"),
        "pair_count": _PAIR_COUNT,
    }


def _write_score_summary(score: dict[str, object]) -> None:
    print(
        f"candidate={score['candidate_wins']} baseline={score['baseline_wins']} "
        f"same={score['same']} unsure={score['unsure']} outcome={score['outcome']}",
        file=sys.stderr,
    )


async def _run_cli(args: argparse.Namespace) -> int:
    try:
        if args.command == "prepare":
            packet_root = await execute_prepare(
                base_url=args.base_url,
                output_dir=args.output_dir,
                open_browser=args.open_browser,
            )
            print(json.dumps(_prepare_success(packet_root), ensure_ascii=False, sort_keys=True))
            return 0

        score = score_packet(args.packet_root, args.results)
        print(json.dumps(score, ensure_ascii=False, sort_keys=True))
        _write_score_summary(score)
        return 0  # noqa: TRY300 - both subcommands complete inside the shared error boundary.
    except TimeoutError:
        _write_failure("blind_ab_timeout")
    except BlindAbError as error:
        code = _safe_error_code(str(error))
        _write_failure(code)
        if code == "browser_open_failed":
            print(f"packet_root: {args.output_dir.expanduser().absolute()}", file=sys.stderr)
    except ClientError as error:
        _write_failure(error.code)
    except ValidationError:
        _write_failure("client_error")
    except Exception:  # noqa: BLE001 - unknown messages must collapse at the CLI boundary.
        _write_failure("client_error")
    return 2


def run_cli(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run_cli(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(run_cli())
