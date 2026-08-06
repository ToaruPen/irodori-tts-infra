# ruff: noqa: ASYNC240, PLR0914, PLR0917, PLR2004, PT012, SLF001
from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import errno
import hashlib
import importlib.util
import io
import itertools
import json
import os
import queue
import random
import secrets
import shutil
import struct
import subprocess  # noqa: S404 - bounded local CLI contract test.
import sys
import threading
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import pytest
from typing_extensions import override

from irodori_tts_infra.contracts import (
    CapabilitiesResponse,
    HealthResponse,
    SynthesisRequest,
    SynthesisResult,
    VoiceCapability,
)
from irodori_tts_infra.evaluation_samples import V4_INFERENCE_SAMPLES

if TYPE_CHECKING:
    import argparse
    import re
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from typing import Self

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/v4_inference_blind_ab.py")
ASSET_ROOT = Path("scripts/assets/v4_inference_blind_ab")
_SAMPLES = tuple(f"評価文{index}" for index in range(6))
_SEEDS = (101, 202)
_MAX_AUDIO_DURATION_SECONDS = 60.0
_UI_SHA256 = {"index.html": "a" * 64, "review.js": "b" * 64}


class _Condition(Protocol):
    name: Literal["baseline", "candidate"]
    num_steps: int
    schedule: Literal["linear", "sway"]


class _PairPlan(Protocol):
    pair_id: str
    sample_index: int
    seed: int
    baseline_side: Literal["a", "b"]
    request_order: tuple[Literal["baseline", "candidate"], Literal["baseline", "candidate"]]
    a_audio_id: str
    b_audio_id: str


class _ResultAnswer(Protocol):
    pair_id: str
    choice: Literal["a", "b", "same", "unsure"]
    reasons: tuple[str, ...]


class _ResultsPayload(Protocol):
    answers: tuple[_ResultAnswer, ...]


class _ScoreDecision(Protocol):
    p_value: float
    outcome: Literal["no_detected_degradation", "degraded", "inconclusive"]


class _WavMetadata(Protocol):
    channels: int
    sample_rate: int
    frame_count: int
    duration_seconds: float


class _BlindAbClient(Protocol):
    async def health(self) -> HealthResponse: ...

    async def capabilities(self) -> CapabilitiesResponse: ...

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...


class _ShutilModule(Protocol):
    copyfile: Callable[..., str]


class _Secrets(Protocol):
    def token_hex(self, nbytes: int | None = None) -> str: ...

    def randbits(self, k: int) -> int: ...


class _FakeRenameFunction:
    def __init__(self, *, result: int = 0, error_number: int = 0) -> None:
        self.result = result
        self.error_number = error_number
        self.argtypes: list[object] | None = None
        self.restype: object | None = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        ctypes.set_errno(self.error_number)
        return self.result


@dataclasses.dataclass
class _FakeLibc:
    renameatx_np: _FakeRenameFunction | None = None
    renameat2: _FakeRenameFunction | None = None


def _pcm16_wav(samples: tuple[int, ...], *, sample_rate: int = 24_000) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return payload.getvalue()


def _wav_with_width(samples: bytes, *, sample_width: int) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(24_000)
        wav_file.writeframes(samples)
    return payload.getvalue()


def _fmt_chunk_payload(
    *,
    format_tag: int = 1,
    channels: int = 1,
    sample_rate: int = 24_000,
    byte_rate: int = 48_000,
    block_align: int = 2,
    bits_per_sample: int = 16,
) -> bytes:
    return struct.pack(
        "<HHIIHH",
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )


def _riff_chunk(chunk_id: bytes, payload: bytes, *, include_padding: bool = True) -> bytes:
    padding = b"\x00" if include_padding and len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def _riff_wave(
    *chunks: bytes,
    declared_size: int | None = None,
    suffix: bytes = b"",
) -> bytes:
    body = b"WAVE" + b"".join(chunks) + suffix
    riff_size = len(body) if declared_size is None else declared_size
    return b"RIFF" + struct.pack("<I", riff_size) + body


def _capabilities(
    *,
    default_id: str | None,
    ready: bool = True,
    generation: str = "dynamic-generation",
) -> CapabilitiesResponse:
    voices: tuple[VoiceCapability, ...] = ()
    if default_id is not None:
        voices = (VoiceCapability(id=default_id, label="Dynamic", default=True),)
    return CapabilitiesResponse(
        generation=generation,
        ready=ready,
        readiness="ready" if ready else "model_loading",
        voices=voices,
    )


class _FakeBlindClient:
    def __init__(
        self,
        *,
        health: HealthResponse | None = None,
        capabilities: CapabilitiesResponse | None = None,
        wav_bytes: bytes | None = None,
        synthesize_hook: Callable[[SynthesisRequest, int], SynthesisResult] | None = None,
    ) -> None:
        self.health_response = health or HealthResponse(status="ok", model_loaded=True)
        self.capabilities_response = capabilities or _capabilities(
            default_id="voice-" + secrets.token_hex(8),
        )
        self.wav_bytes = wav_bytes or _pcm16_wav((1000, -1000) * 120)
        self.synthesize_hook = synthesize_hook
        self.health_calls = 0
        self.capabilities_calls = 0
        self.requests: list[SynthesisRequest] = []

    async def health(self) -> HealthResponse:
        self.health_calls += 1
        return self.health_response

    async def capabilities(self) -> CapabilitiesResponse:
        self.capabilities_calls += 1
        return self.capabilities_response

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        if self.synthesize_hook is not None:
            return self.synthesize_hook(request, len(self.requests) - 1)
        return SynthesisResult(segment_index=0, wav_bytes=self.wav_bytes, elapsed_seconds=0.01)


def _install_test_assets(
    module: _BlindAbModule,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    asset_root = root / "assets"
    asset_root.mkdir()
    (asset_root / "index.html").write_bytes(b"<!doctype html><title>Blind AB</title>")
    (asset_root / "review.js").write_bytes(b'"use strict";\n')
    monkeypatch.setattr(module, "_ASSET_ROOT", asset_root)
    return asset_root


class _BlindAbModule(Protocol):
    V4_INFERENCE_SAMPLES: tuple[str, ...]
    _BLIND_SEEDS: tuple[int, int]
    _PAIR_COUNT: int
    _OPAQUE_ID_RE: re.Pattern[str]
    _MAX_WAV_BYTES: int
    _MAX_TOTAL_WAV_BYTES: int
    _MAX_AUDIO_DURATION_SECONDS: float
    _MANIFEST_PREFIX: str
    _ASSET_ROOT: Path
    Condition: Callable[
        [Literal["baseline", "candidate"], int, Literal["linear", "sway"]], _Condition
    ]
    PairPlan: type[_PairPlan]
    BASELINE: _Condition
    CANDIDATE: _Condition
    BlindAbError: type[RuntimeError]
    ResultAnswer: Any
    ResultsPayload: Any
    asyncio: Any
    httpx: Any
    os: Any
    secrets: _Secrets
    shutil: _ShutilModule
    subprocess: Any
    sys: Any

    def new_opaque_id(self) -> str: ...

    def build_request(
        self,
        *,
        text: str,
        voice_id: str,
        generation: str,
        seed: int,
        condition: _Condition,
    ) -> SynthesisRequest: ...

    def validate_wav(self, wav_bytes: bytes) -> _WavMetadata: ...

    def _reject_existing_output(self, final: Path) -> None: ...

    def _rename_noreplace(self, source: Path, destination: Path) -> None: ...

    def atomic_output_directory(self, destination: Path) -> AbstractContextManager[Path]: ...

    def _open_directory_fd(
        self,
        path: str | Path,
        *,
        dir_fd: int | None = ...,
    ) -> AbstractContextManager[int]: ...

    def _read_bounded_regular(
        self,
        path: str | Path,
        *,
        limit: int,
        dir_fd: int | None = ...,
    ) -> bytes: ...

    def _require_exact_entries(self, directory_fd: int, expected: set[str]) -> None: ...

    async def prepare_packet(
        self,
        client: _BlindAbClient,
        *,
        destination: Path,
        samples: tuple[str, ...] = ...,
        seeds: tuple[int, ...] = ...,
        randomization_seed: int | None = ...,
        id_factory: Callable[[], str] = ...,
    ) -> Path: ...

    def build_pair_plans(
        self,
        *,
        samples: tuple[str, ...],
        seeds: tuple[int, ...],
        randomization_seed: int,
        id_factory: Callable[[], str] = ...,
    ) -> tuple[_PairPlan, ...]: ...

    def canonical_json_bytes(self, value: object) -> bytes: ...

    def sha256_hex(self, value: bytes) -> str: ...

    def build_artifact_payloads(
        self,
        *,
        packet_id: str,
        plans: tuple[_PairPlan, ...],
        samples: tuple[str, ...],
        randomization_seed: int,
        voice_id: str,
        generation: str,
        wav_by_audio_id: dict[str, bytes],
        ui_sha256: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def validate_results(
        self,
        value: object,
        *,
        expected_pair_ids: set[str],
    ) -> _ResultsPayload: ...

    def exact_baseline_preference_p_value(
        self,
        *,
        baseline_wins: int,
        decisive: int,
    ) -> float: ...

    def classify_score(
        self,
        *,
        candidate_wins: int,
        baseline_wins: int,
        same: int,
        unsure: int,
    ) -> _ScoreDecision: ...

    def summarize_answers(
        self,
        results: _ResultsPayload,
        *,
        baseline_side_by_pair: dict[str, Literal["a", "b"]],
    ) -> dict[str, Any]: ...

    def score_packet(self, packet_root: Path, results_path: Path) -> dict[str, object]: ...

    async def execute_prepare(
        self,
        *,
        base_url: str,
        output_dir: Path,
        open_browser: bool = ...,
    ) -> Path: ...

    def _launch_browser(self, file_uri: str) -> bool: ...

    def _parse_args(self, argv: tuple[str, ...]) -> argparse.Namespace: ...

    async def _run_cli(self, args: argparse.Namespace) -> int: ...


def _load_script() -> _BlindAbModule:
    assert SCRIPT_PATH.is_file(), "blind AB script is missing"
    spec = importlib.util.spec_from_file_location("v4_inference_blind_ab", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("_BlindAbModule", module)


def _opaque_id_factory() -> Callable[[], str]:
    identifiers = (f"{value:032x}" for value in itertools.count(1))
    return lambda: next(identifiers)


def _build_plans(
    module: _BlindAbModule,
    *,
    randomization_seed: int = 7,
) -> tuple[_PairPlan, ...]:
    return module.build_pair_plans(
        samples=_SAMPLES,
        seeds=_SEEDS,
        randomization_seed=randomization_seed,
        id_factory=_opaque_id_factory(),
    )


def _wav_by_audio_id(plans: tuple[_PairPlan, ...]) -> dict[str, bytes]:
    return {
        audio_id: f"wave:{audio_id}".encode()
        for plan in plans
        for audio_id in (plan.a_audio_id, plan.b_audio_id)
    }


def _build_artifacts(
    module: _BlindAbModule,
    *,
    packet_id: str = "f" * 32,
    plans: tuple[_PairPlan, ...] | None = None,
    samples: tuple[str, ...] = _SAMPLES,
    randomization_seed: int = 7,
    voice_id: str = "runtime-selected-voice",
    generation: str = "runtime-generation",
    wav_by_audio_id: dict[str, bytes] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_plans = plans if plans is not None else _build_plans(module)
    resolved_wavs = (
        wav_by_audio_id
        if wav_by_audio_id is not None
        else _wav_by_audio_id(
            resolved_plans,
        )
    )
    return module.build_artifact_payloads(
        packet_id=packet_id,
        plans=resolved_plans,
        samples=samples,
        randomization_seed=randomization_seed,
        voice_id=voice_id,
        generation=generation,
        wav_by_audio_id=resolved_wavs,
        ui_sha256=_UI_SHA256,
    )


def _valid_results_payload() -> dict[str, Any]:
    return {
        "schema_version": "irodori-v4-inference-blind-ab-results/v1",
        "packet_id": "c" * 32,
        "manifest_sha256": "d" * 64,
        "answers": tuple(
            {"pair_id": f"{index:032x}", "choice": "same", "reasons": ()} for index in range(1, 13)
        ),
    }


def _expected_result_pair_ids() -> set[str]:
    return {f"{index:032x}" for index in range(1, 13)}


def _replace_plan(plan: _PairPlan, **changes: object) -> _PairPlan:
    return cast("_PairPlan", dataclasses.replace(cast("Any", plan), **changes))


def test_build_pair_plans_is_balanced_opaque_and_reproducible() -> None:
    module = _load_script()

    plans = _build_plans(module)

    assert len(plans) == 12
    assert sum(plan.baseline_side == "a" for plan in plans) == 6
    assert sum(plan.baseline_side == "b" for plan in plans) == 6
    assert {(plan.sample_index, plan.seed) for plan in plans} == set(
        itertools.product(range(6), _SEEDS),
    )
    assert all(
        module._OPAQUE_ID_RE.fullmatch(identifier)
        for plan in plans
        for identifier in (
            plan.pair_id,
            plan.a_audio_id,
            plan.b_audio_id,
        )
    )
    identifiers = {
        identifier
        for plan in plans
        for identifier in (plan.pair_id, plan.a_audio_id, plan.b_audio_id)
    }
    assert len(identifiers) == 36
    assert all(set(plan.request_order) == {"baseline", "candidate"} for plan in plans)
    assert [plan.sample_index for plan in plans] != sorted(plan.sample_index for plan in plans)
    assert plans == _build_plans(module)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plans[0].seed = 0


def test_fixed_contract_uses_only_shared_samples_and_profiles() -> None:
    module = _load_script()

    assert module.V4_INFERENCE_SAMPLES is V4_INFERENCE_SAMPLES
    assert module._BLIND_SEEDS == (101, 202)
    assert module._PAIR_COUNT == 12
    assert module._OPAQUE_ID_RE.pattern == r"^[0-9a-f]{32}$"
    assert module._MAX_WAV_BYTES == 4 * 1024 * 1024
    assert module._MAX_TOTAL_WAV_BYTES == 96 * 1024 * 1024
    assert pytest.approx(_MAX_AUDIO_DURATION_SECONDS) == module._MAX_AUDIO_DURATION_SECONDS
    assert module.Condition("baseline", 24, "linear") == module.BASELINE
    assert module.Condition("candidate", 12, "sway") == module.CANDIDATE
    with pytest.raises(dataclasses.FrozenInstanceError):
        module.BASELINE.num_steps = 1


def test_new_opaque_id_requests_sixteen_random_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    calls: list[int | None] = []

    def fake_token_hex(nbytes: int | None = None) -> str:
        calls.append(nbytes)
        return "f" * 32

    monkeypatch.setattr(module.secrets, "token_hex", fake_token_hex)

    identifier = module.new_opaque_id()

    assert identifier == "f" * 32
    assert calls == [16]


def test_build_pair_plans_default_id_factory_allocates_fresh_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    identifiers = iter(f"{value:032x}" for value in itertools.count(1))
    calls: list[int | None] = []

    def fake_token_hex(nbytes: int | None = None) -> str:
        calls.append(nbytes)
        return next(identifiers)

    monkeypatch.setattr(module.secrets, "token_hex", fake_token_hex)

    plans = module.build_pair_plans(
        samples=_SAMPLES,
        seeds=_SEEDS,
        randomization_seed=7,
    )

    allocated_ids = {
        identifier
        for plan in plans
        for identifier in (plan.pair_id, plan.a_audio_id, plan.b_audio_id)
    }
    assert allocated_ids == {f"{value:032x}" for value in range(1, 37)}
    assert calls == [16] * 36


def test_build_pair_plans_uses_only_local_random_state() -> None:
    module = _load_script()
    original_state = random.getstate()
    try:
        random.seed(8675309)
        expected_state = random.getstate()

        _build_plans(module)

        assert random.getstate() == expected_state
    finally:
        random.setstate(original_state)


def test_randomization_seed_changes_display_sides_and_request_orders() -> None:
    module = _load_script()

    first = _build_plans(module, randomization_seed=7)
    second = _build_plans(module, randomization_seed=8)

    assert [(plan.sample_index, plan.seed) for plan in first] != [
        (plan.sample_index, plan.seed) for plan in second
    ]
    assert sorted((plan.sample_index, plan.seed, plan.baseline_side) for plan in first) != sorted(
        (plan.sample_index, plan.seed, plan.baseline_side) for plan in second
    )
    assert sorted((plan.sample_index, plan.seed, plan.request_order) for plan in first) != sorted(
        (plan.sample_index, plan.seed, plan.request_order) for plan in second
    )


@pytest.mark.parametrize(
    ("samples", "seeds"),
    [
        (_SAMPLES[:-1], _SEEDS),
        (_SAMPLES, _SEEDS[:1]),
        (_SAMPLES, (101, 202, 303)),
    ],
)
def test_build_pair_plans_rejects_any_count_other_than_twelve(
    samples: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="exactly 12"):
        module.build_pair_plans(
            samples=samples,
            seeds=seeds,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )


def test_build_pair_plans_requires_twelve_pairs_independent_of_owner_tuple_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_BLIND_SEEDS", (101,))

    with pytest.raises(ValueError, match="exactly 12"):
        module.build_pair_plans(
            samples=_SAMPLES,
            seeds=(101,),
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )


@pytest.mark.parametrize("bad_identifier", ["a" * 31, "A" * 32, "g" * 32])
def test_build_pair_plans_rejects_malformed_opaque_ids(bad_identifier: str) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="128-bit lowercase hex"):
        module.build_pair_plans(
            samples=_SAMPLES,
            seeds=_SEEDS,
            randomization_seed=7,
            id_factory=lambda: bad_identifier,
        )


def test_build_pair_plans_rejects_duplicate_ids_within_one_pair() -> None:
    module = _load_script()
    identifiers = iter(("1" * 32, "1" * 32, "2" * 32))

    with pytest.raises(ValueError, match="unique"):
        module.build_pair_plans(
            samples=_SAMPLES,
            seeds=_SEEDS,
            randomization_seed=7,
            id_factory=lambda: next(identifiers),
        )


def test_build_pair_plans_rejects_duplicate_ids_across_pairs() -> None:
    module = _load_script()
    identifiers = iter(("1" * 32, "2" * 32, "3" * 32, "1" * 32, "4" * 32, "5" * 32))

    with pytest.raises(ValueError, match="unique"):
        module.build_pair_plans(
            samples=_SAMPLES,
            seeds=_SEEDS,
            randomization_seed=7,
            id_factory=lambda: next(identifiers),
        )


def test_canonical_json_bytes_is_sorted_compact_utf8_and_rejects_nan() -> None:
    module = _load_script()

    encoded = module.canonical_json_bytes({"z": "日本語", "a": {"d": 2, "c": 1}})

    assert encoded == b'{"a":{"c":1,"d":2},"z":"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"}'
    assert module.sha256_hex(encoded) == hashlib.sha256(encoded).hexdigest()
    assert len(module.sha256_hex(encoded)) == 64
    with pytest.raises(ValueError, match="JSON compliant"):
        module.canonical_json_bytes({"value": float("nan")})


def test_build_artifacts_has_exact_public_private_contract_and_payload_digest() -> None:
    module = _load_script()
    plans = _build_plans(module)
    wav_by_audio_id = _wav_by_audio_id(plans)

    manifest_wrapper, answer_key = _build_artifacts(
        module,
        plans=plans,
        wav_by_audio_id=wav_by_audio_id,
    )

    assert set(manifest_wrapper) == {"manifest", "manifest_sha256"}
    manifest = manifest_wrapper["manifest"]
    assert set(manifest) == {"schema_version", "packet_id", "pairs", "reasons"}
    assert manifest["schema_version"] == "irodori-v4-inference-blind-ab-manifest/v1"
    assert manifest["packet_id"] == "f" * 32
    assert manifest["reasons"] == ["reading", "voice", "noise", "prosody", "emotion"]
    assert manifest_wrapper["manifest_sha256"] == module.sha256_hex(
        module.canonical_json_bytes(manifest),
    )
    assert manifest_wrapper["manifest_sha256"] != module.sha256_hex(
        module.canonical_json_bytes(manifest_wrapper),
    )

    public_pairs = manifest["pairs"]
    assert len(public_pairs) == 12
    assert [pair["pair_id"] for pair in public_pairs] == [plan.pair_id for plan in plans]
    assert all(set(pair) == {"pair_id", "text", "a_audio", "b_audio"} for pair in public_pairs)
    assert all(
        pair["a_audio"] == f"audio/{plan.a_audio_id}.wav"
        and pair["b_audio"] == f"audio/{plan.b_audio_id}.wav"
        and pair["text"] == _SAMPLES[plan.sample_index]
        for pair, plan in zip(public_pairs, plans, strict=True)
    )

    assert set(answer_key) == {
        "schema_version",
        "packet_id",
        "manifest_sha256",
        "audio_sha256",
        "pairs",
        "randomization_seed",
        "runtime",
        "ui_sha256",
    }
    assert answer_key["schema_version"] == "irodori-v4-inference-blind-ab-answer-key/v1"
    assert answer_key["packet_id"] == "f" * 32
    assert answer_key["manifest_sha256"] == manifest_wrapper["manifest_sha256"]
    assert answer_key["randomization_seed"] == f"{7:064x}"
    assert set(answer_key["runtime"]) == {"voice_id_sha256", "generation_sha256"}
    assert answer_key["runtime"] == {
        "voice_id_sha256": hashlib.sha256(b"runtime-selected-voice").hexdigest(),
        "generation_sha256": hashlib.sha256(b"runtime-generation").hexdigest(),
    }
    assert answer_key["ui_sha256"] == _UI_SHA256

    expected_paths = {
        f"audio/{audio_id}.wav" for plan in plans for audio_id in (plan.a_audio_id, plan.b_audio_id)
    }
    assert set(answer_key["audio_sha256"]) == expected_paths
    assert len(answer_key["audio_sha256"]) == 24
    assert all(
        answer_key["audio_sha256"][f"audio/{audio_id}.wav"]
        == hashlib.sha256(wav_by_audio_id[audio_id]).hexdigest()
        for plan in plans
        for audio_id in (plan.a_audio_id, plan.b_audio_id)
    )
    assert answer_key["pairs"] == [
        {
            "pair_id": plan.pair_id,
            "sample_index": plan.sample_index,
            "seed": plan.seed,
            "baseline_side": plan.baseline_side,
            "request_order": list(plan.request_order),
        }
        for plan in plans
    ]

    public_json = json.dumps(manifest_wrapper, ensure_ascii=False)
    all_json = public_json + json.dumps(answer_key, ensure_ascii=False)
    for secret in ("runtime-selected-voice", "runtime-generation"):
        assert secret not in all_json
    for private_value in ("baseline", "candidate", "linear", "sway"):
        assert private_value not in public_json
    for private_key in (
        "seed",
        "baseline_side",
        "request_order",
        "randomization_seed",
        "runtime",
    ):
        assert private_key not in public_json


@pytest.mark.parametrize("bad_packet_id", ["f" * 31, "F" * 32, "g" * 32])
def test_build_artifacts_rejects_malformed_packet_id(bad_packet_id: str) -> None:
    module = _load_script()

    with pytest.raises(module.BlindAbError, match="invalid_artifact"):
        _build_artifacts(module, packet_id=bad_packet_id)


@pytest.mark.parametrize("bad_packet_id", [None, 1, b"f" * 32])
def test_build_artifacts_rejects_non_string_packet_id(bad_packet_id: object) -> None:
    module = _load_script()

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, packet_id=cast("Any", bad_packet_id))


def test_build_artifacts_rejects_wrong_plan_count_and_duplicate_pair_ids() -> None:
    module = _load_script()
    plans = _build_plans(module)

    duplicate_pair = _replace_plan(plans[-1], pair_id=plans[0].pair_id)
    for invalid_plans in (plans[:-1], (*plans[:-1], duplicate_pair)):
        with pytest.raises(module.BlindAbError, match="invalid_artifact"):
            _build_artifacts(module, plans=invalid_plans)


@pytest.mark.parametrize("field", ["pair_id", "a_audio_id", "b_audio_id"])
@pytest.mark.parametrize("bad_identifier", [None, 1, b"1" * 32])
def test_build_artifacts_rejects_non_string_plan_ids(
    field: str,
    bad_identifier: object,
) -> None:
    module = _load_script()
    plans = _build_plans(module)
    invalid_plans = (
        _replace_plan(plans[0], **{field: cast("Any", bad_identifier)}),
        *plans[1:],
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(
            module,
            plans=invalid_plans,
            wav_by_audio_id=_wav_by_audio_id(invalid_plans),
        )


@pytest.mark.parametrize("sample_index", [-1, len(_SAMPLES)])
def test_build_artifacts_rejects_invalid_sample_indexes(sample_index: int) -> None:
    module = _load_script()
    plans = _build_plans(module)
    invalid_plans = (_replace_plan(plans[0], sample_index=sample_index), *plans[1:])

    with pytest.raises(module.BlindAbError, match="invalid_artifact"):
        _build_artifacts(module, plans=invalid_plans)


@pytest.mark.parametrize("randomization_seed", [-1, 2**256])
def test_build_artifacts_rejects_out_of_range_randomization_seed(
    randomization_seed: int,
) -> None:
    module = _load_script()

    with pytest.raises(module.BlindAbError, match="invalid_artifact"):
        _build_artifacts(module, randomization_seed=randomization_seed)


def test_build_artifacts_rejects_missing_unknown_or_non_bytes_audio() -> None:
    module = _load_script()
    plans = _build_plans(module)
    valid_wavs = _wav_by_audio_id(plans)
    first_audio_id = plans[0].a_audio_id
    missing = {key: value for key, value in valid_wavs.items() if key != first_audio_id}
    unknown = {**valid_wavs, "f" * 32: b"unknown"}
    non_bytes: dict[str, bytes] = {**valid_wavs, first_audio_id: cast("Any", "not-bytes")}

    for invalid_wavs in (missing, unknown, non_bytes):
        with pytest.raises(module.BlindAbError, match="invalid_artifact"):
            _build_artifacts(module, plans=plans, wav_by_audio_id=invalid_wavs)


def test_build_artifacts_rejects_duplicate_audio_ids_across_plans() -> None:
    module = _load_script()
    plans = _build_plans(module)
    duplicate_audio = _replace_plan(plans[-1], b_audio_id=plans[0].a_audio_id)
    invalid_plans = (*plans[:-1], duplicate_audio)

    with pytest.raises(module.BlindAbError, match="invalid_artifact"):
        _build_artifacts(
            module,
            plans=invalid_plans,
            wav_by_audio_id=_wav_by_audio_id(invalid_plans),
        )


@pytest.mark.parametrize(
    "bad_samples",
    [
        _SAMPLES[:-1],
        (*_SAMPLES, "追加"),
        (cast("Any", None), *_SAMPLES[1:]),
        ("", *_SAMPLES[1:]),
        (" ", *_SAMPLES[1:]),
        (" 前後空白", *_SAMPLES[1:]),
        ("前後空白 ", *_SAMPLES[1:]),
    ],
)
def test_build_artifacts_rejects_invalid_sample_structure(
    bad_samples: tuple[str, ...],
) -> None:
    module = _load_script()

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, samples=bad_samples)


def test_build_artifacts_rejects_duplicate_or_missing_sample_seed_combinations() -> None:
    module = _load_script()
    plans = _build_plans(module)
    invalid_plans = (
        _replace_plan(
            plans[0],
            sample_index=plans[1].sample_index,
            seed=plans[1].seed,
        ),
        *plans[1:],
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, plans=invalid_plans)


@pytest.mark.parametrize("bad_seed", [True, None, 101.0, 303, [], {}])
def test_build_artifacts_rejects_invalid_plan_seed(bad_seed: object) -> None:
    module = _load_script()
    plans = _build_plans(module)
    invalid_plans = (
        _replace_plan(plans[0], seed=cast("Any", bad_seed)),
        *plans[1:],
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, plans=invalid_plans)


@pytest.mark.parametrize("bad_side", [None, True, "A", ["a"], {"side": "a"}])
def test_build_artifacts_rejects_invalid_baseline_side(bad_side: object) -> None:
    module = _load_script()
    plans = _build_plans(module)
    invalid_plans = (
        _replace_plan(plans[0], baseline_side=cast("Any", bad_side)),
        *plans[1:],
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, plans=invalid_plans)


@pytest.mark.parametrize("baseline_a_count", [12, 7])
def test_build_artifacts_requires_balanced_baseline_sides(baseline_a_count: int) -> None:
    module = _load_script()
    plans = _build_plans(module)
    invalid_plans = tuple(
        _replace_plan(plan, baseline_side="a" if index < baseline_a_count else "b")
        for index, plan in enumerate(plans)
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, plans=invalid_plans)


@pytest.mark.parametrize(
    "bad_order",
    [
        ["baseline", "candidate"],
        ("baseline",),
        ("baseline", "candidate", "baseline"),
        ("baseline", "baseline"),
        ("candidate", "candidate"),
        ("baseline", "unknown"),
        ([], "candidate"),
        None,
    ],
)
def test_build_artifacts_rejects_invalid_request_order(bad_order: object) -> None:
    module = _load_script()
    plans = _build_plans(module)
    invalid_plans = (
        _replace_plan(plans[0], request_order=cast("Any", bad_order)),
        *plans[1:],
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, plans=invalid_plans)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("voice_id", None),
        ("voice_id", []),
        ("voice_id", ""),
        ("voice_id", " "),
        ("voice_id", " voice"),
        ("voice_id", "voice "),
        ("generation", None),
        ("generation", {}),
        ("generation", ""),
        ("generation", " "),
        ("generation", " generation"),
        ("generation", "generation "),
    ],
)
def test_build_artifacts_rejects_invalid_runtime_identifiers(
    field: str,
    bad_value: object,
) -> None:
    module = _load_script()
    kwargs = {field: cast("Any", bad_value)}

    with pytest.raises(module.BlindAbError, match=r"^invalid_artifact$"):
        _build_artifacts(module, **kwargs)


def test_validate_results_accepts_complete_valid_payload_and_defaults_reasons() -> None:
    module = _load_script()
    payload = _valid_results_payload()
    payload["answers"] = tuple(
        {"pair_id": answer["pair_id"], "choice": answer["choice"]} for answer in payload["answers"]
    )

    results = module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())

    assert len(results.answers) == 12
    assert all(answer.reasons == () for answer in results.answers)


def test_results_models_are_strict_but_accept_json_shaped_arrays() -> None:
    module = _load_script()
    payload = _valid_results_payload()
    payload["answers"] = [
        {**answer, "reasons": list(answer["reasons"])} for answer in payload["answers"]
    ]

    results = module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())

    assert module.ResultAnswer.model_config["strict"] is True
    assert module.ResultsPayload.model_config["strict"] is True
    assert isinstance(results.answers, tuple)
    assert all(isinstance(answer.reasons, tuple) for answer in results.answers)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", b"irodori-v4-inference-blind-ab-results/v1"),
        ("packet_id", b"c" * 32),
        ("manifest_sha256", b"d" * 64),
        ("answers", iter(_valid_results_payload()["answers"])),
    ],
)
def test_validate_results_rejects_coercible_envelope_types(field: str, value: object) -> None:
    module = _load_script()
    payload = {**_valid_results_payload(), field: value}

    with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
        module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pair_id", b"1" * 32),
        ("choice", b"same"),
        ("reasons", {"voice"}),
        ("reasons", [b"voice"]),
    ],
)
def test_validate_results_rejects_coercible_answer_types(field: str, value: object) -> None:
    module = _load_script()
    valid = _valid_results_payload()
    first, *rest = valid["answers"]
    payload = {**valid, "answers": ({**first, field: value}, *rest)}

    with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
        module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "irodori-v4-inference-blind-ab-results/v2"),
        ("packet_id", "c" * 31),
        ("packet_id", "C" * 32),
        ("manifest_sha256", "d" * 63),
        ("manifest_sha256", "D" * 64),
    ],
)
def test_validate_results_rejects_unknown_schema_and_malformed_envelope(
    field: str,
    value: str,
) -> None:
    module = _load_script()
    payload = {**_valid_results_payload(), field: value}

    with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
        module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())


def test_validate_results_rejects_extra_envelope_and_answer_fields() -> None:
    module = _load_script()
    valid = _valid_results_payload()
    first, *rest = valid["answers"]
    payloads = (
        {**valid, "extra": True},
        {**valid, "answers": ({**first, "extra": True}, *rest)},
    )

    for payload in payloads:
        with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
            module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pair_id", "1" * 31),
        ("pair_id", "A" * 32),
        ("choice", "baseline"),
        ("reason", "speed"),
        ("reason", ("voice", "voice")),
    ],
)
def test_validate_results_rejects_malformed_answers(field: str, value: object) -> None:
    module = _load_script()
    valid = _valid_results_payload()
    first, *rest = valid["answers"]
    if field == "reason":
        reasons = value if isinstance(value, tuple) else (value,)
        changed = {**first, "reasons": reasons}
    else:
        changed = {**first, field: value}
    payload = {**valid, "answers": (changed, *rest)}

    with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
        module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())


def test_validate_results_rejects_missing_duplicate_unknown_and_wrong_total() -> None:
    module = _load_script()
    valid = _valid_results_payload()
    answers = valid["answers"]
    missing = {**valid, "answers": answers[:-1]}
    duplicate = {**valid, "answers": (*answers[:-1], answers[0])}
    unknown = {
        **valid,
        "answers": (*answers[:-1], {**answers[-1], "pair_id": "e" * 32}),
    }
    extra = {
        **valid,
        "answers": (*answers, {"pair_id": "e" * 32, "choice": "same"}),
    }

    for payload in (missing, duplicate, unknown, extra):
        with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
            module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())


@pytest.mark.parametrize(
    ("baseline_wins", "decisive", "expected"),
    [
        (0, 0, 1.0),
        (0, 6, 1.0),
        (3, 6, 42 / 64),
        (5, 6, 7 / 64),
        (6, 6, 1 / 64),
        (9, 10, 11 / 1024),
    ],
)
def test_exact_binomial_matches_known_one_sided_values(
    baseline_wins: int,
    decisive: int,
    expected: float,
) -> None:
    module = _load_script()

    assert module.exact_baseline_preference_p_value(
        baseline_wins=baseline_wins,
        decisive=decisive,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("baseline_wins", "decisive"),
    [(-1, 0), (0, -1), (2, 1)],
)
def test_exact_binomial_rejects_invalid_parameters(baseline_wins: int, decisive: int) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="invalid binomial counts"):
        module.exact_baseline_preference_p_value(
            baseline_wins=baseline_wins,
            decisive=decisive,
        )


@pytest.mark.parametrize(
    ("candidate_wins", "baseline_wins", "same", "unsure", "outcome"),
    [
        (6, 0, 6, 0, "no_detected_degradation"),
        (0, 6, 6, 0, "degraded"),
        (1, 5, 6, 0, "no_detected_degradation"),
        (4, 4, 0, 4, "inconclusive"),
    ],
)
def test_classify_score_uses_significance_and_unsure_boundaries(
    candidate_wins: int,
    baseline_wins: int,
    same: int,
    unsure: int,
    outcome: str,
) -> None:
    module = _load_script()

    decision = module.classify_score(
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        same=same,
        unsure=unsure,
    )

    assert decision.outcome == outcome
    assert 0.0 <= decision.p_value <= 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.outcome = "degraded"


@pytest.mark.parametrize(
    ("candidate_wins", "baseline_wins", "same", "unsure"),
    [(-1, 1, 12, 0), (0, 0, 11, 0), (0, 0, 13, 0)],
)
def test_classify_score_rejects_negative_or_non_twelve_counts(
    candidate_wins: int,
    baseline_wins: int,
    same: int,
    unsure: int,
) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="score counts"):
        module.classify_score(
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            same=same,
            unsure=unsure,
        )


def test_summarize_answers_translates_sides_and_reports_reason_breakdown() -> None:
    module = _load_script()
    payload = _valid_results_payload()
    answers = list(payload["answers"])
    answers[:6] = [
        {**answers[0], "choice": "a", "reasons": ("reading", "voice")},
        {**answers[1], "choice": "a", "reasons": ("reading",)},
        {**answers[2], "choice": "b", "reasons": ("noise",)},
        {**answers[3], "choice": "b", "reasons": ("prosody",)},
        {**answers[4], "choice": "same", "reasons": ("emotion",)},
        {**answers[5], "choice": "unsure", "reasons": ("voice",)},
    ]
    payload["answers"] = tuple(answers)
    results = module.validate_results(payload, expected_pair_ids=_expected_result_pair_ids())
    baseline_sides: dict[str, Literal["a", "b"]] = {
        answer.pair_id: "a" if index % 2 else "b"
        for index, answer in enumerate(results.answers, start=1)
    }

    summary = module.summarize_answers(results, baseline_side_by_pair=baseline_sides)

    assert summary == {
        "candidate_wins": 2,
        "baseline_wins": 2,
        "same": 7,
        "unsure": 1,
        "decisive": 4,
        "p_value": 11 / 16,
        "outcome": "no_detected_degradation",
        "reason_breakdown": {
            "reading": {"candidate_wins": 1, "baseline_wins": 1, "same": 0, "unsure": 0},
            "voice": {"candidate_wins": 0, "baseline_wins": 1, "same": 0, "unsure": 1},
            "noise": {"candidate_wins": 1, "baseline_wins": 0, "same": 0, "unsure": 0},
            "prosody": {"candidate_wins": 0, "baseline_wins": 1, "same": 0, "unsure": 0},
            "emotion": {"candidate_wins": 0, "baseline_wins": 0, "same": 1, "unsure": 0},
        },
    }
    json.dumps(summary, allow_nan=False)


def test_summarize_answers_rejects_missing_or_unknown_baseline_mapping() -> None:
    module = _load_script()
    results = module.validate_results(
        _valid_results_payload(),
        expected_pair_ids=_expected_result_pair_ids(),
    )
    valid_mapping: dict[str, Literal["a", "b"]] = {
        answer.pair_id: "a" for answer in results.answers
    }
    invalid_mappings = (
        {key: value for index, (key, value) in enumerate(valid_mapping.items()) if index != 0},
        {**valid_mapping, "e" * 32: "b"},
    )

    for mapping in invalid_mappings:
        with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
            module.summarize_answers(results, baseline_side_by_pair=mapping)


@pytest.mark.parametrize("bad_side", [None, 1, b"a", ["a"], {"side": "a"}])
def test_summarize_answers_rejects_non_string_or_nonhashable_sides(
    bad_side: object,
) -> None:
    module = _load_script()
    results = module.validate_results(
        _valid_results_payload(),
        expected_pair_ids=_expected_result_pair_ids(),
    )
    mapping: dict[str, Literal["a", "b"]] = {answer.pair_id: "a" for answer in results.answers}
    mapping[results.answers[0].pair_id] = cast("Any", bad_side)

    with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
        module.summarize_answers(results, baseline_side_by_pair=mapping)


def test_build_request_sets_every_shared_and_condition_value_explicitly() -> None:
    module = _load_script()

    request = module.build_request(
        text="動的な評価文",
        voice_id="runtime-default",
        generation="runtime-generation",
        seed=202,
        condition=module.CANDIDATE,
    )

    assert request.model_dump() == {
        "text": "動的な評価文",
        "speaker": None,
        "voice_id": "runtime-default",
        "if_generation": "runtime-generation",
        "num_steps": 12,
        "cfg_scale_text": 3.0,
        "cfg_scale_caption": 3.0,
        "cfg_scale_speaker": 5.0,
        "style": "neutral",
        "seed": 202,
        "duration_scale": 1.0,
        "num_candidates": 1,
        "t_schedule_mode": "sway",
        "sway_coeff": -1.0,
    }


@pytest.mark.parametrize(
    ("health", "capabilities", "expected_code", "expected_capability_calls"),
    [
        (
            HealthResponse(status="degraded", model_loaded=True),
            _capabilities(default_id="v"),
            "runtime_not_ready",
            0,
        ),
        (
            HealthResponse(status="ok", model_loaded=False),
            _capabilities(default_id="v"),
            "runtime_not_ready",
            0,
        ),
        (
            HealthResponse(status="ok", model_loaded=True),
            _capabilities(default_id="v", ready=False),
            "runtime_not_ready",
            1,
        ),
        (
            HealthResponse(status="ok", model_loaded=True),
            _capabilities(default_id=None),
            "default_voice_unavailable",
            1,
        ),
    ],
)
@pytest.mark.asyncio
async def test_prepare_fails_closed_on_readiness_or_default_voice(
    tmp_path: Path,
    health: HealthResponse,
    capabilities: CapabilitiesResponse,
    expected_code: str,
    expected_capability_calls: int,
) -> None:
    module = _load_script()
    client = _FakeBlindClient(health=health, capabilities=capabilities)
    destination = tmp_path / "packet"

    with pytest.raises(module.BlindAbError, match=rf"^{expected_code}$"):
        await module.prepare_packet(client, destination=destination)

    assert client.health_calls == 1
    assert client.capabilities_calls == expected_capability_calls
    assert client.requests == []
    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


def test_validate_wav_accepts_pcm16_without_altering_bytes() -> None:
    module = _load_script()
    wav_bytes = _pcm16_wav((1000, -1000, 500, -500), sample_rate=16_000)
    original = bytes(wav_bytes)

    metadata = module.validate_wav(wav_bytes)

    assert wav_bytes == original
    assert {
        "channels": metadata.channels,
        "sample_rate": metadata.sample_rate,
        "frame_count": metadata.frame_count,
        "duration_seconds": metadata.duration_seconds,
    } == {
        "channels": 1,
        "sample_rate": 16_000,
        "frame_count": 4,
        "duration_seconds": 4 / 16_000,
    }


def test_validate_wav_accepts_torchaudio_pcm16_with_odd_list_metadata_chunk() -> None:
    module = _load_script()
    wav_bytes = _riff_wave(
        _riff_chunk(b"fmt ", _fmt_chunk_payload(sample_rate=16_000, byte_rate=32_000)),
        _riff_chunk(b"LIST", b"INFOx"),
        _riff_chunk(b"data", b"\xe8\x03\x18\xfc"),
    )

    metadata = module.validate_wav(wav_bytes)

    assert metadata.channels == 1
    assert metadata.sample_rate == 16_000
    assert metadata.frame_count == 2
    assert metadata.duration_seconds == pytest.approx(2 / 16_000)


def test_validate_wav_rejects_list_metadata_over_bounded_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_MAX_METADATA_BYTES", 4)
    wav_bytes = _riff_wave(
        _riff_chunk(b"fmt ", _fmt_chunk_payload()),
        _riff_chunk(b"LIST", b"INFOx"),
        _riff_chunk(b"data", b"\x01\x00"),
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_wav$"):
        module.validate_wav(wav_bytes)


def test_validate_wav_rejects_partial_pcm16_frame_data() -> None:
    module = _load_script()
    pcm_bytes = b"\x01\x00\x02\x00\xff"
    wav_bytes = (
        b"RIFF"
        + struct.pack("<I", 42)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 24_000, 48_000, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm_bytes))
        + pcm_bytes
        + b"\x00"
    )

    with pytest.raises(module.BlindAbError, match=r"^invalid_wav$"):
        module.validate_wav(wav_bytes)


@pytest.mark.parametrize(
    "wav_bytes",
    [
        _pcm16_wav((1, 2)) + b"TAIL",
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"data", b"\x01\x00\x02\x00"),
            declared_size=42,
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"JUNK", b"x"),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"LIST", b"INFO"),
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"data", b"\x01\x00"),
            _riff_chunk(b"data", b"\x02\x00"),
        ),
        _riff_wave(_riff_chunk(b"data", b"\x01\x00")),
        _riff_wave(_riff_chunk(b"fmt ", _fmt_chunk_payload())),
        _riff_wave(
            _riff_chunk(b"data", b"\x01\x00"),
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload() + b"\x00\x00"),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload(format_tag=3)),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload(channels=0, byte_rate=0, block_align=0)),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload(sample_rate=0, byte_rate=0)),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload(bits_per_sample=8)),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload(block_align=4)),
            _riff_chunk(b"data", b"\x01\x00\x02\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload(byte_rate=1)),
            _riff_chunk(b"data", b"\x01\x00"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"data", b"\x01\x00\x02\x00\xff"),
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"data", b"\x01\x00\x02\x00"),
            suffix=b"\x00",
        ),
        _riff_wave(
            _riff_chunk(b"fmt ", _fmt_chunk_payload()),
            _riff_chunk(b"data", b"\x01\x00\x02\x00"),
            suffix=b"TAIL",
        ),
    ],
    ids=(
        "bytes-beyond-riff",
        "declared-riff-truncated",
        "junk-chunk",
        "list-chunk",
        "duplicate-fmt",
        "duplicate-data",
        "missing-fmt",
        "missing-data",
        "data-before-fmt",
        "noncanonical-fmt-size",
        "non-pcm-format",
        "zero-channels",
        "zero-sample-rate",
        "non-16-bit",
        "wrong-block-align",
        "wrong-byte-rate",
        "partial-data-frame",
        "padding-after-data",
        "bytes-after-data",
    ),
)
def test_validate_wav_rejects_noncanonical_riff_layout(wav_bytes: bytes) -> None:
    module = _load_script()

    with pytest.raises(module.BlindAbError, match=r"^invalid_wav$"):
        module.validate_wav(wav_bytes)


@pytest.mark.parametrize(
    "wav_bytes",
    [
        b"not a wave",
        _pcm16_wav((1, 2, 3))[:-1],
        bytes(
            bytearray(_pcm16_wav((1, 2, 3)))[:20]
            + b"\x03\x00"
            + bytearray(_pcm16_wav((1, 2, 3)))[22:]
        ),
        _wav_with_width(b"\x01\x02\x03", sample_width=1),
        _pcm16_wav(()),
    ],
    ids=("unreadable", "truncated", "compressed", "wrong-width", "empty"),
)
def test_validate_wav_rejects_invalid_audio_structure(wav_bytes: bytes) -> None:
    module = _load_script()

    with pytest.raises(module.BlindAbError, match=r"^invalid_wav$"):
        module.validate_wav(wav_bytes)


def test_validate_wav_enforces_per_file_size_and_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    wav_bytes = _pcm16_wav((1, 2, 3, 4))

    monkeypatch.setattr(module, "_MAX_WAV_BYTES", len(wav_bytes) - 1)
    with pytest.raises(module.BlindAbError, match=r"^audio_too_large$"):
        module.validate_wav(wav_bytes)

    monkeypatch.setattr(module, "_MAX_WAV_BYTES", len(wav_bytes))
    monkeypatch.setattr(module, "_MAX_AUDIO_DURATION_SECONDS", 3 / 24_000)
    with pytest.raises(module.BlindAbError, match=r"^audio_too_long$"):
        module.validate_wav(wav_bytes)


def test_atomic_output_directory_finalizes_complete_owned_directory(tmp_path: Path) -> None:
    module = _load_script()
    destination = tmp_path / "packet"

    with module.atomic_output_directory(destination) as temporary:
        assert temporary.parent == tmp_path.resolve()
        (temporary / "complete.txt").write_text("complete", encoding="utf-8")
        assert not destination.exists()

    assert destination.resolve() == destination
    assert (destination / "complete.txt").read_text(encoding="utf-8") == "complete"
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_atomic_output_directory_rejects_existing_destination(
    tmp_path: Path,
    kind: str,
) -> None:
    module = _load_script()
    destination = tmp_path / "packet"
    if kind == "file":
        destination.write_text("sentinel", encoding="utf-8")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing-target")

    with (
        pytest.raises(module.BlindAbError, match=r"^output_exists$"),
        module.atomic_output_directory(destination),
    ):
        pytest.fail("existing output must fail before yielding")

    assert destination.exists() or destination.is_symlink()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.parametrize(
    "destination_factory",
    [
        lambda _root: Path("/"),
        lambda root: root / "missing-parent" / "packet",
        lambda root: root / "parent-file" / "packet",
        lambda root: root / "..",
    ],
    ids=("root", "missing-parent", "parent-file", "dotdot"),
)
def test_atomic_output_directory_rejects_unsafe_path(
    tmp_path: Path,
    destination_factory: Callable[[Path], Path],
) -> None:
    module = _load_script()
    (tmp_path / "parent-file").write_text("sentinel", encoding="utf-8")

    with (
        pytest.raises(module.BlindAbError, match=r"^unsafe_output_path$"),
        module.atomic_output_directory(destination_factory(tmp_path)),
    ):
        pytest.fail("unsafe output must fail before yielding")


@pytest.mark.parametrize(
    ("platform", "symbol", "directory_fd", "flags"),
    [
        ("darwin", "renameatx_np", -2, 4),
        ("linux", "renameat2", -100, 1),
    ],
)
def test_rename_noreplace_uses_platform_exclusive_libc_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    symbol: str,
    directory_fd: int,
    flags: int,
) -> None:
    module = _load_script()
    runtime = cast("Any", module)
    rename = _FakeRenameFunction()
    libc = _FakeLibc(
        renameatx_np=rename if symbol == "renameatx_np" else None,
        renameat2=rename if symbol == "renameat2" else None,
    )
    cdll_calls: list[tuple[object, bool]] = []

    def fake_cdll(library: object, *, use_errno: bool) -> _FakeLibc:
        cdll_calls.append((library, use_errno))
        return libc

    monkeypatch.setattr(runtime.os, "name", "posix")
    monkeypatch.setattr(runtime.sys, "platform", platform)
    monkeypatch.setattr(runtime.ctypes, "CDLL", fake_cdll)
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    module._rename_noreplace(source, destination)

    assert cdll_calls == [(None, True)]
    assert rename.argtypes == [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    assert rename.restype is ctypes.c_int
    assert rename.calls == [
        (
            directory_fd,
            os.fsencode(source),
            directory_fd,
            os.fsencode(destination),
            flags,
        )
    ]


def test_rename_noreplace_windows_delegates_to_os_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    runtime = cast("Any", module)
    calls: list[tuple[Path, Path]] = []

    def fake_rename(source: Path, destination: Path) -> None:
        calls.append((source, destination))

    monkeypatch.setattr(runtime.os, "name", "nt")
    monkeypatch.setattr(runtime.os, "rename", fake_rename)
    monkeypatch.setattr(
        runtime.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: pytest.fail("CDLL must not be called"),
    )
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    module._rename_noreplace(source, destination)

    assert calls == [(source, destination)]


def test_rename_noreplace_unknown_platform_fails_without_loading_libc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    runtime = cast("Any", module)
    monkeypatch.setattr(runtime.os, "name", "posix")
    monkeypatch.setattr(runtime.sys, "platform", "unsupported")
    monkeypatch.setattr(
        runtime.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: pytest.fail("CDLL must not be called"),
    )
    destination = tmp_path / "destination"

    with pytest.raises(OSError) as caught:
        module._rename_noreplace(tmp_path / "source", destination)

    assert caught.value.errno == errno.ENOTSUP
    assert caught.value.filename == destination


def test_rename_noreplace_missing_libc_symbol_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    runtime = cast("Any", module)
    monkeypatch.setattr(runtime.os, "name", "posix")
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *_args, **_kwargs: _FakeLibc())
    destination = tmp_path / "destination"

    with pytest.raises(OSError) as caught:
        module._rename_noreplace(tmp_path / "source", destination)

    assert caught.value.errno == errno.ENOSYS
    assert caught.value.filename == destination


@pytest.mark.parametrize(
    ("error_number", "expected_type"),
    [(errno.EEXIST, FileExistsError), (errno.EACCES, OSError)],
)
def test_rename_noreplace_converts_libc_errno(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    expected_type: type[OSError],
) -> None:
    module = _load_script()
    runtime = cast("Any", module)
    rename = _FakeRenameFunction(result=-1, error_number=error_number)
    monkeypatch.setattr(runtime.os, "name", "posix")
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(
        runtime.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: _FakeLibc(renameatx_np=rename),
    )
    destination = tmp_path / "destination"

    with pytest.raises(expected_type) as caught:
        module._rename_noreplace(tmp_path / "source", destination)

    assert caught.value.errno == error_number
    assert caught.value.filename == destination
    if error_number != errno.EEXIST:
        assert not isinstance(caught.value, FileExistsError)


def test_rename_noreplace_preserves_empty_existing_destination(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "ours.txt").write_text("ours", encoding="utf-8")
    destination.mkdir()

    with pytest.raises(FileExistsError):
        module._rename_noreplace(source, destination)

    assert (source / "ours.txt").read_text(encoding="utf-8") == "ours"
    assert list(destination.iterdir()) == []


def test_atomic_output_directory_preserves_empty_destination_created_after_final_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    destination = tmp_path / "packet"
    original_reject = module._reject_existing_output
    reject_calls = 0

    def create_racer_after_check(final: Path) -> None:
        nonlocal reject_calls
        original_reject(final)
        reject_calls += 1
        if reject_calls == 2:
            destination.mkdir()

    monkeypatch.setattr(module, "_reject_existing_output", create_racer_after_check)

    with (
        pytest.raises(module.BlindAbError, match=r"^output_exists$"),
        module.atomic_output_directory(destination) as temporary,
    ):
        (temporary / "ours.txt").write_text("ours", encoding="utf-8")

    assert reject_calls == 2
    assert list(destination.iterdir()) == []
    assert not (destination / "ours.txt").exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


def test_atomic_output_directory_cleans_only_its_owned_temp_on_base_exception(
    tmp_path: Path,
) -> None:
    module = _load_script()
    destination = tmp_path / "packet"
    sibling = tmp_path / ".packet.tmp-sentinel"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")
    owned: Path | None = None

    with (
        pytest.raises(KeyboardInterrupt),
        module.atomic_output_directory(
            destination,
        ) as temporary,
    ):
        owned = temporary
        (temporary / "partial.txt").write_text("partial", encoding="utf-8")
        raise KeyboardInterrupt

    assert owned is not None
    assert not owned.exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not destination.exists()


def test_public_ui_source_assets_are_regular_repository_files() -> None:
    for name in ("index.html", "review.js"):
        asset = ASSET_ROOT / name
        assert asset.exists(), f"missing public UI asset: {asset}"
        assert asset.is_file()
        assert not asset.is_symlink()


def test_public_ui_markup_is_self_contained_accessible_and_symmetric() -> None:
    index = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")

    assert "<!doctype html>" in index.lower()
    assert '<html lang="ja">' in index
    assert '<meta charset="utf-8">' in index
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in index
    for element_id in (
        "progress",
        "sample-text",
        "audio-a",
        "audio-b",
        "choices",
        "reasons",
        "previous",
        "next",
        "download",
        "remaining",
    ):
        assert f'id="{element_id}"' in index
    assert index.index('<script src="manifest.js"></script>') < index.index(
        '<script src="review.js"></script>'
    )
    assert index.count('class="channel-card"') == 2
    assert '<section class="channel-card" aria-labelledby="channel-a-title">' in index
    assert '<section class="channel-card" aria-labelledby="channel-b-title">' in index
    assert '<audio id="audio-a" controls preload="metadata"></audio>' in index
    assert '<audio id="audio-b" controls preload="metadata"></audio>' in index
    assert "autoplay" not in index.lower()
    assert "://" not in index
    assert 'src="//' not in index
    assert 'href="//' not in index
    assert "<link" not in index.lower()


def test_public_ui_has_strict_offline_csp_and_honest_integrity_scope() -> None:
    index = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    expected_policy = (
        "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; "
        "media-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'none'; "
        "form-action 'none'; frame-src 'none'; child-src 'none'; worker-src 'none'"
    )

    assert '<meta http-equiv="Content-Security-Policy" content="' + expected_policy + '">' in index
    assert "偶発的な破損や部分的な改変" in index
    assert "同じユーザー権限でpacket全体を改変" in index
    assert "外部seal" in index
    assert "http:" not in index.lower()
    assert "https:" not in index.lower()


def test_public_ui_css_encodes_approved_console_tokens_and_responsive_accessibility() -> None:
    index = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")

    for color in ("#18212b", "#eaf2f5", "#ffffff", "#176c78", "#c77a2b", "#b8c7cd"):
        assert color in index.lower()
    assert "Avenir Next" in index
    assert "Hiragino Sans" in index
    assert "Yu Gothic" in index
    assert "SFMono-Regular" in index
    assert "Roboto Mono" in index
    assert ":focus-visible" in index
    assert "@media (prefers-reduced-motion: reduce)" in index
    assert "@media (max-width:" in index
    assert ".channel-card" in index
    assert "linear-gradient" not in index
    assert "radial-gradient" not in index


def test_public_ui_script_has_static_state_navigation_and_download_contract() -> None:
    script = (ASSET_ROOT / "review.js").read_text(encoding="utf-8")

    assert script.startswith('"use strict";')
    for token in (
        "window.IRODORI_BLIND_AB_MANIFEST",
        "localStorage.getItem",
        "localStorage.setItem",
        "Number.isInteger",
        "validPairIds",
        "validChoices",
        "validReasons",
        "createElement",
        "createTextNode",
        "textContent",
        "audio.pause()",
        "audio.currentTime = 0",
        "pair.a_audio",
        "pair.b_audio",
        'schema_version: "irodori-v4-inference-blind-ab-results/v1"',
        "packet_id: manifest.packet_id",
        "manifest_sha256: wrapper.manifest_sha256",
        "manifest.pairs.map",
        'download = "irodori-blind-ab-results.json"',
        "URL.createObjectURL",
        "URL.revokeObjectURL",
        "setTimeout",
    ):
        assert token in script
    for choice, label in (
        ("a", "Aが良い"),
        ("b", "Bが良い"),
        ("same", "同等"),
        ("unsure", "判断できない"),
    ):
        assert f'["{choice}", "{label}"]' in script
    for reason, label in (
        ("reading", "読み"),
        ("voice", "声"),
        ("noise", "ノイズ"),
        ("prosody", "自然さ・韻律"),
        ("emotion", "感情"),
    ):
        assert f'{reason}: "{label}"' in script
    assert script.count("try {") >= 2
    assert script.count("catch (") >= 2
    assert "answeredCount() !== manifest.pairs.length" in script
    assert "innerHTML" not in script
    assert "document.write" not in script
    assert "fetch(" not in script
    assert "eval(" not in script
    assert "crypto" not in script.lower()


def test_public_ui_assets_contain_no_condition_or_runtime_clues() -> None:
    public_source = "\n".join(
        (ASSET_ROOT / name).read_text(encoding="utf-8") for name in ("index.html", "review.js")
    ).lower()

    for forbidden in (
        "baseline",
        "candidate",
        "linear",
        "sway",
        "voice_id",
        "voice-id",
        "generation",
        "steps",
        "profile",
    ):
        assert forbidden not in public_source


@pytest.mark.asyncio
async def test_prepare_with_repository_assets_yields_complete_condition_blind_packet(
    tmp_path: Path,
) -> None:
    module = _load_script()
    assert module._ASSET_ROOT.resolve() == ASSET_ROOT.resolve()
    destination = tmp_path / "blind-packet"
    raw_voice_id = "voice-" + secrets.token_hex(8)
    raw_generation = "generation-" + secrets.token_hex(8)
    client = _FakeBlindClient(
        capabilities=_capabilities(default_id=raw_voice_id, generation=raw_generation)
    )

    await module.prepare_packet(
        client,
        destination=destination,
        samples=_SAMPLES,
        seeds=_SEEDS,
        randomization_seed=7,
        id_factory=_opaque_id_factory(),
    )

    packet_files = {
        path.relative_to(destination / "packet").as_posix()
        for path in (destination / "packet").rglob("*")
        if path.is_file()
    }
    audio_files = {name for name in packet_files if name.startswith("audio/")}
    assert len(audio_files) == 24
    assert all(name.endswith(".wav") for name in audio_files)
    assert packet_files == {"index.html", "review.js", "manifest.js", *audio_files}
    assert {
        path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_dir()
    } == {"packet", "packet/audio", "private"}
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == {
        "packet/index.html",
        "packet/review.js",
        "packet/manifest.js",
        "private/answer-key.json",
        *(f"packet/{name}" for name in audio_files),
    }

    public_text_by_name = {
        name: (destination / "packet" / name).read_text(encoding="utf-8")
        for name in ("index.html", "review.js", "manifest.js")
    }
    public_surface = "\n".join((*public_text_by_name.values(), *sorted(packet_files))).lower()
    for forbidden in (
        "baseline",
        "candidate",
        "linear",
        "sway",
        raw_voice_id.lower(),
        raw_generation.lower(),
    ):
        assert forbidden not in public_surface

    public_scripts = "\n".join(
        public_text_by_name[name] for name in ("manifest.js", "review.js")
    ).lower()
    for request_field in (
        "num_steps",
        "t_schedule_mode",
        "request_order",
        "if_generation",
        "generation",
        "seed",
        "speaker",
        "style",
        "cfg_scale",
    ):
        assert request_field not in public_scripts

    answer_key_text = (destination / "private/answer-key.json").read_text(encoding="utf-8")
    answer_key = json.loads(answer_key_text)
    assert answer_key["runtime"] == {
        "voice_id_sha256": hashlib.sha256(raw_voice_id.encode()).hexdigest(),
        "generation_sha256": hashlib.sha256(raw_generation.encode()).hexdigest(),
    }
    assert raw_voice_id not in answer_key_text
    assert raw_generation not in answer_key_text


@pytest.mark.asyncio
async def test_prepare_generates_exact_blind_packet_from_pair_plans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)
    default_id = "voice-" + secrets.token_hex(8)
    generation = "generation-" + secrets.token_hex(8)
    returned_wavs: list[bytes] = []

    def synthesize_hook(_request: SynthesisRequest, index: int) -> SynthesisResult:
        wav_bytes = _pcm16_wav((index + 1, -(index + 1)) * 4)
        returned_wavs.append(wav_bytes)
        return SynthesisResult(segment_index=0, wav_bytes=wav_bytes, elapsed_seconds=0.01)

    client = _FakeBlindClient(
        capabilities=_capabilities(default_id=default_id, generation=generation),
        synthesize_hook=synthesize_hook,
    )
    destination = tmp_path / "blind-packet"
    id_factory = _opaque_id_factory()

    result = await module.prepare_packet(
        client,
        destination=destination,
        samples=_SAMPLES,
        seeds=_SEEDS,
        randomization_seed=7,
        id_factory=id_factory,
    )

    expected_factory = _opaque_id_factory()
    packet_id = expected_factory()
    plans = module.build_pair_plans(
        samples=_SAMPLES,
        seeds=_SEEDS,
        randomization_seed=7,
        id_factory=expected_factory,
    )
    condition_by_name = {"baseline": module.BASELINE, "candidate": module.CANDIDATE}
    expected_requests = [
        (
            _SAMPLES[plan.sample_index],
            plan.seed,
            condition_by_name[name].num_steps,
            condition_by_name[name].schedule,
        )
        for plan in plans
        for name in plan.request_order
    ]
    actual_requests = [
        (request.text, request.seed, request.num_steps, request.t_schedule_mode)
        for request in client.requests
    ]
    assert result == destination.resolve()
    assert actual_requests == expected_requests
    assert len(client.requests) == 24
    assert all(request.voice_id == default_id for request in client.requests)
    assert all(request.if_generation == generation for request in client.requests)
    assert all(
        request.speaker is None and request.style == "neutral" for request in client.requests
    )
    assert all(
        (
            request.cfg_scale_text,
            request.cfg_scale_caption,
            request.cfg_scale_speaker,
            request.duration_scale,
            request.num_candidates,
            request.sway_coeff,
        )
        == (3.0, 3.0, 5.0, 1.0, 1, -1.0)
        for request in client.requests
    )

    expected_wav_by_id: dict[str, bytes] = {}
    offset = 0
    for plan in plans:
        by_condition = dict(
            zip(plan.request_order, returned_wavs[offset : offset + 2], strict=True)
        )
        offset += 2
        expected_wav_by_id[plan.a_audio_id] = by_condition[
            "baseline" if plan.baseline_side == "a" else "candidate"
        ]
        expected_wav_by_id[plan.b_audio_id] = by_condition[
            "baseline" if plan.baseline_side == "b" else "candidate"
        ]
    actual_wav_paths = tuple(sorted((destination / "packet/audio").glob("*.wav")))
    assert len(actual_wav_paths) == 24
    assert {path.stem: path.read_bytes() for path in actual_wav_paths} == expected_wav_by_id
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == {
        "packet/index.html",
        "packet/review.js",
        "packet/manifest.js",
        "private/answer-key.json",
        *(f"packet/audio/{audio_id}.wav" for audio_id in expected_wav_by_id),
    }

    manifest_js = (destination / "packet/manifest.js").read_bytes()
    assert manifest_js.startswith(b"window.IRODORI_BLIND_AB_MANIFEST=")
    assert manifest_js.endswith(b";\n")
    wrapper = json.loads(manifest_js.removeprefix(module._MANIFEST_PREFIX.encode())[:-2])
    answer_key_bytes = (destination / "private/answer-key.json").read_bytes()
    assert answer_key_bytes.endswith(b"\n")
    answer_key = json.loads(answer_key_bytes)
    assert wrapper["manifest"]["packet_id"] == packet_id
    assert answer_key["packet_id"] == packet_id
    assert answer_key["audio_sha256"] == {
        f"audio/{audio_id}.wav": hashlib.sha256(wav_bytes).hexdigest()
        for audio_id, wav_bytes in sorted(expected_wav_by_id.items())
    }
    public_bytes = b"\n".join(
        (destination / "packet" / name).read_bytes()
        for name in ("index.html", "review.js", "manifest.js")
    )
    assert all(
        secret not in public_bytes for secret in (b"baseline", b"candidate", b"linear", b"sway")
    )


@pytest.mark.parametrize(
    "result",
    [
        SynthesisResult.model_construct(
            segment_index=1,
            wav_bytes=_pcm16_wav((1, 2)),
            elapsed_seconds=0.1,
            content_type="audio/wav",
        ),
        SynthesisResult.model_construct(
            segment_index=0,
            wav_bytes=_pcm16_wav((1, 2)),
            elapsed_seconds=float("nan"),
            content_type="audio/wav",
        ),
        SynthesisResult.model_construct(
            segment_index=0,
            wav_bytes=_pcm16_wav((1, 2)),
            elapsed_seconds=-1.0,
            content_type="audio/wav",
        ),
        SynthesisResult.model_construct(
            segment_index=0,
            wav_bytes=_pcm16_wav((1, 2)),
            elapsed_seconds=0.1,
            content_type="text/plain",
        ),
    ],
    ids=("segment", "nan-elapsed", "negative-elapsed", "content-type"),
)
@pytest.mark.asyncio
async def test_prepare_rejects_invalid_result_metadata_and_cleans_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: SynthesisResult,
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)
    client = _FakeBlindClient(synthesize_hook=lambda _request, _index: result)
    destination = tmp_path / "packet"

    with pytest.raises(module.BlindAbError, match=r"^invalid_response$"):
        await module.prepare_packet(
            client,
            destination=destination,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


class _GenerationMismatchSentinelError(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_prepare_propagates_generation_mismatch_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)
    sentinel = _GenerationMismatchSentinelError("runtime_generation_mismatch")

    def fail(_request: SynthesisRequest, _index: int) -> SynthesisResult:
        raise sentinel

    client = _FakeBlindClient(synthesize_hook=fail)
    destination = tmp_path / "packet"

    with pytest.raises(_GenerationMismatchSentinelError) as caught:
        await module.prepare_packet(
            client,
            destination=destination,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )

    assert caught.value is sentinel
    assert len(client.requests) == 1
    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.asyncio
async def test_prepare_enforces_total_wav_cap_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)
    wav_bytes = _pcm16_wav((1, 2, 3, 4))
    monkeypatch.setattr(module, "_MAX_TOTAL_WAV_BYTES", len(wav_bytes) * 24 - 1)
    client = _FakeBlindClient(wav_bytes=wav_bytes)
    destination = tmp_path / "packet"

    with pytest.raises(module.BlindAbError, match=r"^audio_too_large$"):
        await module.prepare_packet(
            client,
            destination=destination,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )

    assert len(client.requests) == 24
    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.parametrize(
    ("limit_name", "limit", "wav_bytes", "expected_code"),
    [
        (None, None, b"invalid", "invalid_wav"),
        ("_MAX_WAV_BYTES", 1, _pcm16_wav((1, 2)), "audio_too_large"),
        ("_MAX_AUDIO_DURATION_SECONDS", 0.0, _pcm16_wav((1, 2)), "audio_too_long"),
    ],
    ids=("invalid", "oversize", "too-long"),
)
@pytest.mark.asyncio
async def test_prepare_cleans_output_for_wav_validation_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str | None,
    limit: float | None,
    wav_bytes: bytes,
    expected_code: str,
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)
    if limit_name is not None:
        monkeypatch.setattr(module, limit_name, limit)
    client = _FakeBlindClient(wav_bytes=wav_bytes)
    destination = tmp_path / "packet"

    with pytest.raises(module.BlindAbError, match=rf"^{expected_code}$"):
        await module.prepare_packet(
            client,
            destination=destination,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )

    assert len(client.requests) == 1
    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.parametrize("bad_asset", ["missing", "symlink"])
@pytest.mark.asyncio
async def test_prepare_rejects_missing_or_symlinked_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_asset: str,
) -> None:
    module = _load_script()
    asset_root = _install_test_assets(module, tmp_path, monkeypatch)
    review = asset_root / "review.js"
    review.unlink()
    if bad_asset == "symlink":
        review.symlink_to(asset_root / "index.html")
    client = _FakeBlindClient()
    destination = tmp_path / "packet"

    with pytest.raises(module.BlindAbError, match=r"^client_error$"):
        await module.prepare_packet(
            client,
            destination=destination,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )

    assert len(client.requests) == 24
    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.parametrize("asset_name", ["index.html", "review.js"])
@pytest.mark.asyncio
async def test_prepare_rejects_oversized_ui_asset_with_bounded_chunk_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset_name: str,
) -> None:
    module = _load_script()
    asset_root = _install_test_assets(module, tmp_path, monkeypatch)
    limit = 128 * 1024
    (asset_root / asset_name).write_bytes(b"x" * (limit + 1))
    monkeypatch.setattr(module, "_MAX_METADATA_BYTES", limit)
    real_read = os.read
    read_sizes: list[int] = []
    bytes_read = 0

    def bounded_read(descriptor: int, size: int) -> bytes:
        nonlocal bytes_read
        assert size <= 64 * 1024
        value = real_read(descriptor, size)
        read_sizes.append(size)
        bytes_read += len(value)
        return value

    monkeypatch.setattr(module.os, "read", bounded_read)
    destination = tmp_path / "packet"

    with pytest.raises(module.BlindAbError, match=r"^client_error$"):
        await module.prepare_packet(
            _FakeBlindClient(),
            destination=destination,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )

    assert read_sizes
    assert bytes_read <= limit + 64 * 1024
    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.asyncio
async def test_prepare_maps_asset_copy_resource_failure_and_cleans_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)

    def fail_copy(_source: Path, _destination: Path) -> str:
        raise OSError

    monkeypatch.setattr(module, "_copy_ui_asset", fail_copy)
    client = _FakeBlindClient()
    destination = tmp_path / "packet"

    with pytest.raises(module.BlindAbError, match=r"^client_error$"):
        await module.prepare_packet(
            client,
            destination=destination,
            randomization_seed=7,
            id_factory=_opaque_id_factory(),
        )

    assert len(client.requests) == 24
    assert not destination.exists()
    assert list(tmp_path.glob(".packet.tmp-*")) == []


@pytest.mark.asyncio
async def test_prepare_uses_256_bit_csprng_seed_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)
    calls: list[int] = []
    chosen_seed = (1 << 255) + 123

    def fake_randbits(bits: int) -> int:
        calls.append(bits)
        return chosen_seed

    monkeypatch.setattr(module.secrets, "randbits", fake_randbits)
    destination = tmp_path / "packet"

    await module.prepare_packet(
        _FakeBlindClient(),
        destination=destination,
        id_factory=_opaque_id_factory(),
    )

    answer_key = json.loads((destination / "private/answer-key.json").read_bytes())
    assert calls == [256]
    assert answer_key["randomization_seed"] == f"{chosen_seed:064x}"


async def _prepare_scoring_fixture(
    module: _BlindAbModule,
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    packet_root = tmp_path / "packet-root"
    await module.prepare_packet(
        _FakeBlindClient(),
        destination=packet_root,
        randomization_seed=7,
        id_factory=_opaque_id_factory(),
    )
    manifest_bytes = (packet_root / "packet/manifest.js").read_bytes()
    manifest_wrapper = json.loads(
        manifest_bytes.removeprefix(module._MANIFEST_PREFIX.encode())[:-2],
    )
    answer_key = json.loads((packet_root / "private/answer-key.json").read_bytes())
    baseline_by_pair = {pair["pair_id"]: pair["baseline_side"] for pair in answer_key["pairs"]}
    answers = []
    for index, pair in enumerate(manifest_wrapper["manifest"]["pairs"]):
        baseline_side = baseline_by_pair[pair["pair_id"]]
        if index < 6:
            choice = baseline_side
            reasons = ["voice"]
        elif index < 8:
            choice = "b" if baseline_side == "a" else "a"
            reasons = ["prosody"]
        elif index < 10:
            choice = "same"
            reasons = ["noise"]
        else:
            choice = "unsure"
            reasons = []
        answers.append({"pair_id": pair["pair_id"], "choice": choice, "reasons": reasons})
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": "irodori-v4-inference-blind-ab-results/v1",
                "packet_id": manifest_wrapper["manifest"]["packet_id"],
                "manifest_sha256": manifest_wrapper["manifest_sha256"],
                "answers": answers,
            },
        ),
        encoding="utf-8",
    )
    return packet_root, results_path, manifest_wrapper, answer_key


@pytest.mark.asyncio
async def test_score_packet_validates_real_packet_and_returns_public_machine_score(
    tmp_path: Path,
) -> None:
    module = _load_script()
    packet_root, results_path, _manifest_wrapper, _answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )

    score = module.score_packet(packet_root, results_path)

    assert score == {
        "schema_version": "irodori-v4-inference-blind-ab-score/v1",
        "status": "complete",
        "candidate_wins": 2,
        "baseline_wins": 6,
        "same": 2,
        "unsure": 2,
        "decisive": 8,
        "p_value": 37 / 256,
        "outcome": "no_detected_degradation",
        "reason_breakdown": {
            "reading": {"candidate_wins": 0, "baseline_wins": 0, "same": 0, "unsure": 0},
            "voice": {"candidate_wins": 0, "baseline_wins": 6, "same": 0, "unsure": 0},
            "noise": {"candidate_wins": 0, "baseline_wins": 0, "same": 2, "unsure": 0},
            "prosody": {"candidate_wins": 2, "baseline_wins": 0, "same": 0, "unsure": 0},
            "emotion": {"candidate_wins": 0, "baseline_wins": 0, "same": 0, "unsure": 0},
        },
    }
    serialized = json.dumps(score, allow_nan=False)
    for private_value in (
        "sample_index",
        "baseline_side",
        "request_order",
        "voice_id_sha256",
        "generation_sha256",
        "audio/",
    ):
        assert private_value not in serialized


@pytest.mark.parametrize("asset_name", ["index.html", "review.js"])
@pytest.mark.asyncio
async def test_score_packet_rejects_one_byte_public_ui_tamper(
    tmp_path: Path,
    asset_name: str,
) -> None:
    module = _load_script()
    packet_root, results_path, _wrapper, answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    asset_path = packet_root / "packet" / asset_name
    original = asset_path.read_bytes()

    assert answer_key["ui_sha256"][asset_name] == hashlib.sha256(original).hexdigest()
    asset_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(module.BlindAbError, match=r"^packet_integrity_error$"):
        module.score_packet(packet_root, results_path)


@pytest.mark.parametrize("asset_name", ["index.html", "review.js"])
@pytest.mark.asyncio
async def test_score_rejects_packet_ui_and_answer_key_digest_changed_together(
    tmp_path: Path,
    asset_name: str,
) -> None:
    module = _load_script()
    packet_root, results_path, _wrapper, answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    asset_path = packet_root / "packet" / asset_name
    tampered = asset_path.read_bytes() + b"\nmodified"
    asset_path.write_bytes(tampered)
    answer_key["ui_sha256"][asset_name] = hashlib.sha256(tampered).hexdigest()
    (packet_root / "private/answer-key.json").write_bytes(
        module.canonical_json_bytes(answer_key) + b"\n",
    )

    with pytest.raises(module.BlindAbError, match=r"^packet_integrity_error$"):
        module.score_packet(packet_root, results_path)


@pytest.mark.parametrize(
    "tamper",
    ["randomization-seed", "display-order", "baseline-side-swap", "request-order"],
)
@pytest.mark.asyncio
async def test_score_packet_reconstructs_randomization_metadata_and_rejects_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    module = _load_script()
    packet_root, results_path, _wrapper, answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    private_pairs = answer_key["pairs"]
    if tamper == "randomization-seed":
        answer_key["randomization_seed"] = f"{8:064x}"
    elif tamper == "display-order":
        private_pairs[0], private_pairs[1] = private_pairs[1], private_pairs[0]
    elif tamper == "baseline-side-swap":
        first = next(pair for pair in private_pairs if pair["baseline_side"] == "a")
        second = next(pair for pair in private_pairs if pair["baseline_side"] == "b")
        first["baseline_side"], second["baseline_side"] = (
            second["baseline_side"],
            first["baseline_side"],
        )
        assert sum(pair["baseline_side"] == "a" for pair in private_pairs) == 6
    else:
        private_pairs[0]["request_order"] = list(
            reversed(private_pairs[0]["request_order"]),
        )
    (packet_root / "private/answer-key.json").write_bytes(
        module.canonical_json_bytes(answer_key) + b"\n",
    )

    with pytest.raises(module.BlindAbError, match=r"^packet_integrity_error$"):
        module.score_packet(packet_root, results_path)


@pytest.mark.asyncio
async def test_score_packet_rejects_reordered_manifest_pairs_with_stale_canonical_digest(
    tmp_path: Path,
) -> None:
    module = _load_script()
    packet_root, results_path, wrapper, answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    manifest = wrapper["manifest"]
    original_pairs = tuple(manifest["pairs"])
    original_manifest_bytes = module.canonical_json_bytes(manifest)
    original_digest = wrapper["manifest_sha256"]
    original_key_digest = answer_key["manifest_sha256"]

    manifest["pairs"] = list(reversed(original_pairs))
    reordered_manifest_bytes = module.canonical_json_bytes(manifest)
    (packet_root / "packet/manifest.js").write_bytes(
        module._MANIFEST_PREFIX.encode() + module.canonical_json_bytes(wrapper) + b";\n",
    )

    assert {module.canonical_json_bytes(pair) for pair in original_pairs} == {
        module.canonical_json_bytes(pair) for pair in manifest["pairs"]
    }
    assert reordered_manifest_bytes != original_manifest_bytes
    assert module.sha256_hex(reordered_manifest_bytes) != original_digest
    assert wrapper["manifest_sha256"] == original_digest
    assert answer_key["manifest_sha256"] == original_key_digest == original_digest
    with pytest.raises(module.BlindAbError, match=r"^packet_integrity_error$"):
        module.score_packet(packet_root, results_path)


@pytest.mark.parametrize("corruption", ["manifest-wrapper", "answer-key", "audio"])
@pytest.mark.asyncio
async def test_score_packet_maps_packet_corruption_to_integrity_error(
    tmp_path: Path,
    corruption: str,
) -> None:
    module = _load_script()
    packet_root, results_path, manifest_wrapper, answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    if corruption == "manifest-wrapper":
        (packet_root / "packet/manifest.js").write_text(
            "window.WRONG=" + json.dumps(manifest_wrapper) + ";\n",
            encoding="utf-8",
        )
    elif corruption == "answer-key":
        answer_key["manifest_sha256"] = "0" * 64
        (packet_root / "private/answer-key.json").write_text(
            json.dumps(answer_key),
            encoding="utf-8",
        )
    else:
        first_audio = next((packet_root / "packet/audio").glob("*.wav"))
        first_audio.write_bytes(first_audio.read_bytes() + b"tampered")

    with pytest.raises(module.BlindAbError, match=r"^packet_integrity_error$"):
        module.score_packet(packet_root, results_path)


@pytest.mark.parametrize("corruption", ["json", "packet-id", "extra-answer"])
@pytest.mark.asyncio
async def test_score_packet_maps_results_corruption_to_invalid_results(
    tmp_path: Path,
    corruption: str,
) -> None:
    module = _load_script()
    packet_root, results_path, _manifest_wrapper, _answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    if corruption == "json":
        results_path.write_text("not-json", encoding="utf-8")
    else:
        payload = json.loads(results_path.read_bytes())
        if corruption == "packet-id":
            payload["packet_id"] = "f" * 32
        else:
            payload["answers"].append(payload["answers"][0])
        results_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
        module.score_packet(packet_root, results_path)


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.2:8924", "https://[::1]:443", "http://[::1]"],
)
def test_blind_ab_parser_accepts_only_numeric_loopback_roots(base_url: str, tmp_path: Path) -> None:
    module = _load_script()

    args = module._parse_args(
        ("prepare", "--base-url", base_url, "--output-dir", str(tmp_path / "packet")),
    )

    assert args.command == "prepare"
    assert args.base_url == base_url
    assert args.output_dir == tmp_path / "packet"


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://127.0.0.1:8924",
        "http://localhost:8924",
        "http://example.invalid:8924",
        "http://user:secret@127.0.0.1:8924",
        "http://127.0.0.1:",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "http://127.0.0.1/path",
        "http://127.0.0.1:8924?",
        "http://127.0.0.1:8924#",
        "http://127.0.0.1:8924/?",
        "http://127.0.0.1:8924/#",
        "http://127.0.0.1/?query=1",
        "http://127.0.0.1/#fragment",
    ],
)
def test_blind_ab_parser_rejects_non_loopback_or_ambiguous_urls(
    base_url: str,
    tmp_path: Path,
) -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module._parse_args(
            ("prepare", "--base-url", base_url, "--output-dir", str(tmp_path / "packet")),
        )


def test_blind_ab_parser_has_prepare_and_score_contract(tmp_path: Path) -> None:
    module = _load_script()

    prepare = module._parse_args(("prepare", "--output-dir", str(tmp_path / "packet"), "--open"))
    score = module._parse_args(
        (
            "score",
            "--packet-root",
            str(tmp_path / "packet"),
            "--results",
            str(tmp_path / "results.json"),
        ),
    )

    assert prepare.base_url == "http://127.0.0.1:8924"
    assert prepare.open_browser is True
    assert score.command == "score"
    assert score.packet_root == tmp_path / "packet"
    assert score.results == tmp_path / "results.json"


@pytest.mark.asyncio
async def test_execute_prepare_uses_direct_transport_no_client_timeout_and_global_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    captured: dict[str, object] = {}
    events: list[str] = []

    class FakeTransport:
        async def aclose(self) -> None:  # noqa: PLR6301
            events.append("transport-close")

    transport = FakeTransport()

    class ClientContext:
        async def __aenter__(self) -> _FakeBlindClient:
            events.append("client-enter")
            return _FakeBlindClient()

        async def __aexit__(self, *_args: object) -> None:
            events.append("client-exit")

    def client_factory(**kwargs: object) -> ClientContext:
        captured.update(kwargs)
        return ClientContext()

    class TimeoutContext:
        async def __aenter__(self) -> None:
            events.append("timeout-enter")

        async def __aexit__(self, *_args: object) -> None:
            events.append("timeout-exit")

    async def fake_prepare(  # noqa: RUF029
        _client: _FakeBlindClient,
        *,
        destination: Path,
    ) -> Path:
        events.append("prepare")
        return destination.resolve()

    monkeypatch.setattr(module.httpx, "AsyncHTTPTransport", lambda: transport)
    monkeypatch.setattr(module, "AsyncIrodoriClient", client_factory)

    def timeout_factory(seconds: float) -> TimeoutContext:
        captured["deadline"] = seconds
        return TimeoutContext()

    monkeypatch.setattr(module.asyncio, "timeout", timeout_factory)
    monkeypatch.setattr(module, "prepare_packet", fake_prepare)

    def fake_validate(packet_root: Path) -> object:
        events.append("validate")
        assert packet_root == (tmp_path / "packet").resolve()
        return object()

    monkeypatch.setattr(module, "_validate_packet", fake_validate)

    async def fake_to_thread(  # noqa: RUF029
        function: Callable[[str, float], bool],
        file_uri: str,
        timeout_seconds: float,
    ) -> bool:
        events.append("to-thread")
        return function(file_uri, timeout_seconds)

    def fake_launch(file_uri: str, timeout_seconds: float) -> bool:
        events.append("browser-launch")
        assert file_uri == (tmp_path / "packet/packet/index.html").resolve().as_uri()
        assert 0 < timeout_seconds <= 10.0
        return True

    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(module, "_launch_browser", fake_launch)

    result = await module.execute_prepare(
        base_url="http://127.0.0.1:8924",
        output_dir=tmp_path / "packet",
        open_browser=True,
    )

    assert result == (tmp_path / "packet").resolve()
    assert captured == {
        "deadline": 900.0,
        "base_url": "http://127.0.0.1:8924",
        "timeout": None,
        "transport": transport,
        "max_response_bytes": 8 * 1024 * 1024,
    }
    assert events == [
        "timeout-enter",
        "client-enter",
        "prepare",
        "client-exit",
        "validate",
        "to-thread",
        "browser-launch",
        "timeout-exit",
        "transport-close",
    ]


@pytest.mark.asyncio
async def test_execute_prepare_rejects_pre_open_packet_tamper_without_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    packet_root, _results, _wrapper, _answer_key = await _prepare_scoring_fixture(
        module,
        fixture_root,
    )
    launch_calls: list[str] = []

    class FakeTransport:
        async def aclose(self) -> None:  # noqa: PLR6301
            return None

    class ClientContext:
        async def __aenter__(self) -> _FakeBlindClient:
            return _FakeBlindClient()

        async def __aexit__(self, *_args: object) -> None:
            return None

    async def fake_prepare(  # noqa: RUF029
        _client: _FakeBlindClient,
        *,
        destination: Path,
    ) -> Path:
        assert destination == tmp_path / "packet"
        review = packet_root / "packet/review.js"
        review.write_bytes(review.read_bytes() + b"\nmodified-before-open")
        return packet_root

    async def fake_to_thread(  # noqa: RUF029
        function: Callable[[str, float], bool],
        file_uri: str,
        timeout_seconds: float,
    ) -> bool:
        return function(file_uri, timeout_seconds)

    def fake_launch(file_uri: str, _timeout_seconds: float) -> bool:
        launch_calls.append(file_uri)
        return True

    monkeypatch.setattr(module.httpx, "AsyncHTTPTransport", FakeTransport)
    monkeypatch.setattr(module, "AsyncIrodoriClient", lambda **_kwargs: ClientContext())
    monkeypatch.setattr(module, "prepare_packet", fake_prepare)
    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(module, "_launch_browser", fake_launch)

    with pytest.raises(module.BlindAbError, match=r"^packet_integrity_error$"):
        await module.execute_prepare(
            base_url="http://127.0.0.1:8924",
            output_dir=tmp_path / "packet",
            open_browser=True,
        )

    assert launch_calls == []


@pytest.mark.parametrize(
    ("platform", "executable"),
    [("darwin", "open"), ("linux", "xdg-open")],
)
def test_browser_launcher_uses_argv_isolated_stdio_and_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    executable: str,
) -> None:
    module = _load_script()
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> object:
        captured["argv"] = argv
        captured.update(kwargs)
        return module.subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._launch_browser("file:///tmp/listening/index.html") is True
    assert captured == {
        "argv": [executable, "file:///tmp/listening/index.html"],
        "stdin": module.subprocess.DEVNULL,
        "stdout": module.subprocess.DEVNULL,
        "stderr": module.subprocess.DEVNULL,
        "check": False,
        "timeout": 10.0,
    }
    assert captured.get("shell", False) is False


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "exception"])
def test_browser_launcher_fails_closed_for_nonzero_and_launch_timeout(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = _load_script()

    def fake_run(argv: list[str], **_kwargs: object) -> object:
        if failure == "timeout":
            raise module.subprocess.TimeoutExpired(argv, 10.0)
        if failure == "exception":
            private_failure = "private launcher failure"
            raise RuntimeError(private_failure)
        return module.subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._launch_browser("file:///tmp/listening/index.html") is False


def test_browser_launcher_fails_closed_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not launch"),
    )

    assert module._launch_browser("file:///tmp/listening/index.html") is False


@pytest.mark.asyncio
async def test_prepare_cli_opens_final_index_before_printing_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    destination = tmp_path / "chosen-output"
    index = destination / "packet/index.html"
    events: list[str] = []

    async def fake_execute_prepare(**kwargs: object) -> Path:  # noqa: RUF029
        index.parent.mkdir(parents=True)
        index.write_text("ready", encoding="utf-8")
        assert kwargs["open_browser"] is True
        output_before_completion = capsys.readouterr()
        assert not output_before_completion.out
        assert not output_before_completion.err
        events.append("prepared-and-opened")
        return destination.resolve()

    monkeypatch.setattr(module, "execute_prepare", fake_execute_prepare)
    args = module._parse_args(("prepare", "--output-dir", str(destination), "--open"))

    assert await module._run_cli(args) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "complete",
        "packet_root": str(destination.resolve()),
        "answer_key": str((destination / "private/answer-key.json").resolve()),
        "pair_count": 12,
    }
    assert not output.err
    assert events == ["prepared-and-opened"]


@pytest.mark.asyncio
async def test_prepare_cli_browser_failure_preserves_packet_and_prints_manual_open_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    destination = tmp_path / "user-chosen-packet"
    index = destination / "packet/index.html"

    async def fake_execute_prepare(**_kwargs: object) -> Path:  # noqa: RUF029
        index.parent.mkdir(parents=True)
        index.write_text("ready", encoding="utf-8")
        browser_open_failed = "browser_open_failed"
        raise module.BlindAbError(browser_open_failed)

    monkeypatch.setattr(module, "execute_prepare", fake_execute_prepare)
    args = module._parse_args(("prepare", "--output-dir", str(destination), "--open"))

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert not output.out
    error_line, packet_root_line = output.err.splitlines()
    assert json.loads(error_line) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": "browser_open_failed",
    }
    assert packet_root_line == f"packet_root: {destination.absolute()}"
    assert destination.is_dir()
    assert "private detail" not in output.err


@pytest.mark.asyncio
async def test_score_cli_separates_machine_json_and_human_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    expected = {
        "schema_version": "irodori-v4-inference-blind-ab-score/v1",
        "status": "complete",
        "candidate_wins": 5,
        "baseline_wins": 3,
        "same": 2,
        "unsure": 2,
        "decisive": 8,
        "p_value": 0.85546875,
        "outcome": "no_detected_degradation",
        "reason_breakdown": {},
    }
    monkeypatch.setattr(module, "score_packet", lambda *_args: expected)
    args = module._parse_args(
        (
            "score",
            "--packet-root",
            str(tmp_path / "packet"),
            "--results",
            str(tmp_path / "results.json"),
        ),
    )

    assert await module._run_cli(args) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == expected
    assert output.err == (
        "candidate=5 baseline=3 same=2 unsure=2 outcome=no_detected_degradation\n"
    )


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (TimeoutError("private timeout"), "blind_ab_timeout"),
        (RuntimeError("remote response with secret"), "client_error"),
    ],
)
@pytest.mark.asyncio
async def test_prepare_cli_maps_failures_without_leaking_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    expected_code: str,
) -> None:
    module = _load_script()

    async def fail(**_kwargs: object) -> Path:  # noqa: RUF029
        raise error

    monkeypatch.setattr(module, "execute_prepare", fail)
    args = module._parse_args(("prepare", "--output-dir", str(tmp_path / "packet")))

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": expected_code,
    }
    assert "private" not in output.err


@pytest.mark.parametrize("error_kind", ["blind-error", "validation-error"])
@pytest.mark.asyncio
async def test_prepare_cli_collapses_internal_invalid_response_to_client_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_kind: str,
) -> None:
    module = _load_script()
    if error_kind == "blind-error":
        error: Exception = module.BlindAbError("invalid_response")
    else:
        try:
            module.ResultsPayload.model_validate({})
        except Exception as caught:  # noqa: BLE001 - captures the script's Pydantic version.
            error = caught
        else:
            raise AssertionError

    async def fail(**_kwargs: object) -> Path:  # noqa: RUF029
        raise error

    monkeypatch.setattr(module, "execute_prepare", fail)
    args = module._parse_args(("prepare", "--output-dir", str(tmp_path / "packet")))

    assert await module._run_cli(args) == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": "client_error",
    }


def test_fd_directory_enumeration_stops_after_expected_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    @dataclasses.dataclass
    class Entry:
        name: str

    class EndlessScandir:
        def __init__(self) -> None:
            self.yielded = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> EndlessScandir:
            return self

        def __next__(self) -> Entry:
            self.yielded += 1
            return Entry(f"entry-{self.yielded}")

    entries = EndlessScandir()
    monkeypatch.setattr(module.os, "scandir", lambda _fd: entries)

    with pytest.raises(ValueError):
        module._require_exact_entries(7, {"expected"})

    assert entries.yielded == 2


def test_directory_fd_remains_anchored_after_path_is_replaced_with_symlink(
    tmp_path: Path,
) -> None:
    module = _load_script()
    root = tmp_path / "root"
    packet = root / "packet"
    packet.mkdir(parents=True)
    (packet / "manifest.js").write_bytes(b"anchored")
    replacement = tmp_path / "replacement"
    replacement_packet = replacement / "packet"
    replacement_packet.mkdir(parents=True)
    (replacement_packet / "manifest.js").write_bytes(b"redirected")
    moved = tmp_path / "moved-root"

    with module._open_directory_fd(root) as root_fd:
        root.rename(moved)
        root.symlink_to(replacement, target_is_directory=True)
        with module._open_directory_fd("packet", dir_fd=root_fd) as packet_fd:
            value = module._read_bounded_regular("manifest.js", limit=32, dir_fd=packet_fd)

    assert value == b"anchored"


@pytest.mark.parametrize("target", ["root", "manifest", "results"])
@pytest.mark.asyncio
async def test_score_rejects_symlinked_untrusted_directories_and_files(
    tmp_path: Path,
    target: str,
) -> None:
    module = _load_script()
    packet_root, results_path, _manifest, _key = await _prepare_scoring_fixture(module, tmp_path)
    expected_code = "packet_integrity_error"
    if target == "root":
        real_root = tmp_path / "real-root"
        packet_root.rename(real_root)
        packet_root.symlink_to(real_root, target_is_directory=True)
    elif target == "manifest":
        manifest_path = packet_root / "packet/manifest.js"
        manifest_path.unlink()
        manifest_path.symlink_to(packet_root / "packet/review.js")
    else:
        expected_code = "invalid_results"
        real_results = tmp_path / "real-results.json"
        results_path.rename(real_results)
        results_path.symlink_to(real_results)

    with pytest.raises(module.BlindAbError, match=rf"^{expected_code}$"):
        module.score_packet(packet_root, results_path)


def _unblock_fifo_reader(fifo_path: Path, *, deadline_seconds: float) -> None:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        try:
            descriptor = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as error:
            if error.errno not in {errno.ENXIO, errno.ENOENT}:
                return
            time.sleep(0.001)
            continue
        try:
            os.write(descriptor, b"\n")
        except BrokenPipeError:
            pass
        finally:
            os.close(descriptor)
        return


def _run_fifo_call_bounded(
    operation: Callable[[], object],
    *,
    fifo_path: Path,
    deadline_seconds: float = 0.1,
) -> object:
    outcomes: queue.Queue[tuple[Literal["result", "error"], object]] = queue.Queue(maxsize=1)
    finished = threading.Event()

    def run() -> None:
        try:
            outcomes.put(("result", operation()))
        except BaseException as error:  # noqa: BLE001 - propagate worker outcome on the test thread.
            outcomes.put(("error", error))
        finally:
            finished.set()

    worker = threading.Thread(target=run, name="bounded-fifo-test-call", daemon=True)
    worker.start()
    if not finished.wait(deadline_seconds):
        _unblock_fifo_reader(fifo_path, deadline_seconds=0.25)
        worker.join(timeout=0.25)
        if worker.is_alive():
            message = "FIFO worker did not join after deterministic unblock"
            raise AssertionError(message)
        message = "operation exceeded FIFO deadline"
        raise AssertionError(message)

    worker.join(timeout=0.25)
    if worker.is_alive():
        message = "finished FIFO worker did not join"
        raise AssertionError(message)
    kind, value = outcomes.get_nowait()
    if kind == "error":
        raise cast("BaseException", value)
    return value


@pytest.mark.parametrize("target", ["manifest", "results"])
@pytest.mark.asyncio
async def test_score_rejects_fifo_inputs_without_blocking(
    tmp_path: Path,
    target: str,
) -> None:
    module = _load_script()
    packet_root, results_path, _manifest, _key = await _prepare_scoring_fixture(module, tmp_path)
    fifo_path = packet_root / "packet/manifest.js" if target == "manifest" else results_path
    fifo_path.unlink()
    os.mkfifo(fifo_path)
    expected_code = "packet_integrity_error" if target == "manifest" else "invalid_results"

    started = time.monotonic()
    with pytest.raises(module.BlindAbError, match=rf"^{expected_code}$"):
        _run_fifo_call_bounded(
            lambda: module.score_packet(packet_root, results_path),
            fifo_path=fifo_path,
        )

    assert time.monotonic() - started < 1.0


def test_bounded_fifo_call_fails_promptly_and_joins_a_blocked_reader(tmp_path: Path) -> None:
    fifo_path = tmp_path / "blocking-reader"
    os.mkfifo(fifo_path)

    started = time.monotonic()
    with pytest.raises(AssertionError, match=r"exceeded FIFO deadline"):
        _run_fifo_call_bounded(
            fifo_path.read_bytes,
            fifo_path=fifo_path,
            deadline_seconds=0.02,
        )

    assert time.monotonic() - started < 1.0
    assert all(thread.name != "bounded-fifo-test-call" for thread in threading.enumerate())


def _rewrite_manifest(
    module: _BlindAbModule,
    packet_root: Path,
    wrapper: dict[str, Any],
    answer_key: dict[str, Any],
) -> None:
    wrapper["manifest_sha256"] = module.sha256_hex(
        module.canonical_json_bytes(wrapper["manifest"]),
    )
    answer_key["manifest_sha256"] = wrapper["manifest_sha256"]
    (packet_root / "packet/manifest.js").write_bytes(
        module._MANIFEST_PREFIX.encode() + module.canonical_json_bytes(wrapper) + b";\n",
    )
    (packet_root / "private/answer-key.json").write_bytes(
        module.canonical_json_bytes(answer_key) + b"\n",
    )


def _mutate_packet_case(  # noqa: C901, PLR0911, PLR0912, PLR0915 - explicit tamper catalog.
    module: _BlindAbModule,
    packet_root: Path,
    wrapper: dict[str, Any],
    answer_key: dict[str, Any],
    case: str,
) -> None:
    packet = packet_root / "packet"
    private = packet_root / "private"
    audio = packet / "audio"
    manifest_path = packet / "manifest.js"
    key_path = private / "answer-key.json"
    manifest = wrapper["manifest"]
    public_pair = manifest["pairs"][0]
    private_pair = answer_key["pairs"][0]
    first_audio = audio / Path(public_pair["a_audio"]).name

    if case in {"root-missing", "packet-missing", "private-missing", "audio-missing-dir"}:
        target = {
            "root-missing": packet_root,
            "packet-missing": packet,
            "private-missing": private,
            "audio-missing-dir": audio,
        }[case]
        shutil.rmtree(target)
        return
    if case in {"root-extra", "packet-extra", "private-extra", "audio-extra"}:
        parent = {
            "root-extra": packet_root,
            "packet-extra": packet,
            "private-extra": private,
            "audio-extra": audio,
        }[case]
        (parent / "unexpected").write_bytes(b"x")
        return
    if case in {"packet-symlink", "private-symlink", "audio-dir-symlink"}:
        target = {
            "packet-symlink": packet,
            "private-symlink": private,
            "audio-dir-symlink": audio,
        }[case]
        replacement = packet_root.parent / f"real-{case}"
        target.rename(replacement)
        target.symlink_to(replacement, target_is_directory=True)
        return
    if case == "audio-file-symlink":
        replacement = packet_root.parent / "real-audio.wav"
        first_audio.rename(replacement)
        first_audio.symlink_to(replacement)
        return
    if case in {"packet-fifo", "private-fifo", "audio-dir-fifo", "audio-file-fifo"}:
        target = {
            "packet-fifo": packet,
            "private-fifo": private,
            "audio-dir-fifo": audio,
            "audio-file-fifo": first_audio,
        }[case]
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        os.mkfifo(target)
        return
    if case.startswith("manifest-"):
        if case == "manifest-wrong-prefix":
            manifest_path.write_bytes(
                b"window.WRONG=" + module.canonical_json_bytes(wrapper) + b";\n"
            )
        elif case == "manifest-wrong-suffix":
            manifest_path.write_bytes(
                module._MANIFEST_PREFIX.encode() + module.canonical_json_bytes(wrapper) + b"\n",
            )
        elif case == "manifest-trailing-script":
            manifest_path.write_bytes(manifest_path.read_bytes() + b"alert(1);\n")
        elif case == "manifest-malformed-utf8":
            manifest_path.write_bytes(module._MANIFEST_PREFIX.encode() + b"\xff;\n")
        elif case == "manifest-malformed-json":
            manifest_path.write_bytes(module._MANIFEST_PREFIX.encode() + b"{;\n")
        elif case == "manifest-duplicate-wrapper-key":
            raw = module.canonical_json_bytes(wrapper)
            manifest_path.write_bytes(
                module._MANIFEST_PREFIX.encode() + b'{"manifest":null,' + raw[1:] + b";\n",
            )
        elif case == "manifest-duplicate-nested-key":
            raw = module.canonical_json_bytes(wrapper)
            raw = raw.replace(
                b'"schema_version":', b'"schema_version":"duplicate","schema_version":', 1
            )
            manifest_path.write_bytes(module._MANIFEST_PREFIX.encode() + raw + b";\n")
        elif case in {"manifest-wrapper-missing-key", "manifest-wrapper-extra-key"}:
            if case.endswith("missing-key"):
                wrapper.pop("manifest_sha256")
            else:
                wrapper["extra"] = True
            manifest_path.write_bytes(
                module._MANIFEST_PREFIX.encode() + module.canonical_json_bytes(wrapper) + b";\n",
            )
        elif case == "manifest-unknown-schema":
            manifest["schema_version"] = "unknown"
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case in {"manifest-missing-key", "manifest-extra-key"}:
            if case.endswith("missing-key"):
                manifest.pop("reasons")
            else:
                manifest["extra"] = True
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-nonstring-packet-id":
            manifest["packet_id"] = 1
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-packet-id-mismatch":
            manifest["packet_id"] = "e" * 32
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-digest-mismatch":
            wrapper["manifest_sha256"] = "0" * 64
            manifest_path.write_bytes(
                module._MANIFEST_PREFIX.encode() + module.canonical_json_bytes(wrapper) + b";\n",
            )
        elif case == "manifest-reasons-changed":
            manifest["reasons"][0] = "changed"
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-reasons-order":
            manifest["reasons"].reverse()
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-reasons-duplicate":
            manifest["reasons"][1] = manifest["reasons"][0]
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-pair-missing":
            manifest["pairs"].pop()
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-pair-duplicate":
            manifest["pairs"][-1]["pair_id"] = public_pair["pair_id"]
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-pair-unknown-key":
            public_pair["extra"] = True
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-blank-text":
            public_pair["text"] = " "
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case == "manifest-text-mismatch":
            public_pair["text"] = "別の評価文"
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        elif case.startswith("manifest-audio-"):
            public_pair["a_audio"] = {
                "manifest-audio-traversal": "audio/../secret.wav",
                "manifest-audio-absolute": "/tmp/secret.wav",  # noqa: S108 - hostile fixture.
                "manifest-audio-backslash": r"audio\\secret.wav",
                "manifest-audio-duplicate": public_pair["b_audio"],
            }[case]
            _rewrite_manifest(module, packet_root, wrapper, answer_key)
        return
    if case.startswith("key-"):
        if case == "key-malformed-utf8":
            key_path.write_bytes(b"\xff")
        elif case == "key-malformed-json":
            key_path.write_bytes(b"{")
        elif case == "key-duplicate-key":
            raw = module.canonical_json_bytes(answer_key)
            key_path.write_bytes(b'{"schema_version":"duplicate",' + raw[1:])
        elif case == "key-duplicate-nested-key":
            raw = module.canonical_json_bytes(answer_key)
            raw = raw.replace(b'"pair_id":', b'"pair_id":"duplicate","pair_id":', 1)
            key_path.write_bytes(raw)
        elif case == "key-schema":
            answer_key["schema_version"] = "unknown"
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case in {"key-missing-key", "key-extra-key"}:
            if case.endswith("missing-key"):
                answer_key.pop("runtime")
            else:
                answer_key["extra"] = True
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-packet":
            answer_key["packet_id"] = "e" * 32
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-digest":
            answer_key["manifest_sha256"] = "0" * 64
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case in {"key-pair-missing", "key-pair-extra"}:
            if case.endswith("missing"):
                answer_key["pairs"].pop()
            else:
                answer_key["pairs"].append(dict(private_pair))
                answer_key["pairs"][-1]["pair_id"] = "e" * 32
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-pair-duplicate":
            answer_key["pairs"][-1]["pair_id"] = private_pair["pair_id"]
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-pair-unknown":
            private_pair["pair_id"] = "e" * 32
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-sample-index":
            private_pair["sample_index"] = 6
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-seed":
            private_pair["seed"] = 303
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-cartesian":
            private_pair["sample_index"] = answer_key["pairs"][1]["sample_index"]
            private_pair["seed"] = answer_key["pairs"][1]["seed"]
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-side-balance":
            side = private_pair["baseline_side"]
            replacement = next(
                pair for pair in answer_key["pairs"] if pair["baseline_side"] != side
            )
            replacement["baseline_side"] = side
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-side":
            private_pair["baseline_side"] = "c"
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-request-order":
            private_pair["request_order"] = ["baseline", "baseline"]
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-random-seed":
            answer_key["randomization_seed"] = "x"
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-runtime-keys":
            answer_key["runtime"]["extra"] = True
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-runtime-value":
            answer_key["runtime"]["voice_id_sha256"] = "x"
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-audio-hash-keys":
            answer_key["audio_sha256"]["audio/extra.wav"] = "0" * 64
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        elif case == "key-audio-hash-value":
            answer_key["audio_sha256"][public_pair["a_audio"]] = "x"
            key_path.write_bytes(module.canonical_json_bytes(answer_key))
        return
    if case == "audio-missing":
        first_audio.unlink()
    elif case == "audio-tampered-hash":
        first_audio.write_bytes(first_audio.read_bytes()[:-2] + b"xx")
    elif case == "audio-invalid-wav":
        first_audio.write_bytes(b"not wav")
    elif case == "audio-oversized":
        first_audio.write_bytes(b"x" * (module._MAX_WAV_BYTES + 1))


_PACKET_TAMPER_CASES = (
    "root-missing",
    "packet-missing",
    "private-missing",
    "audio-missing-dir",
    "root-extra",
    "packet-extra",
    "private-extra",
    "audio-extra",
    "packet-symlink",
    "private-symlink",
    "audio-dir-symlink",
    "audio-file-symlink",
    "packet-fifo",
    "private-fifo",
    "audio-dir-fifo",
    "audio-file-fifo",
    "manifest-wrong-prefix",
    "manifest-wrong-suffix",
    "manifest-trailing-script",
    "manifest-malformed-utf8",
    "manifest-malformed-json",
    "manifest-duplicate-wrapper-key",
    "manifest-duplicate-nested-key",
    "manifest-wrapper-missing-key",
    "manifest-wrapper-extra-key",
    "manifest-unknown-schema",
    "manifest-missing-key",
    "manifest-extra-key",
    "manifest-nonstring-packet-id",
    "manifest-packet-id-mismatch",
    "manifest-digest-mismatch",
    "manifest-reasons-changed",
    "manifest-reasons-order",
    "manifest-reasons-duplicate",
    "manifest-pair-missing",
    "manifest-pair-duplicate",
    "manifest-pair-unknown-key",
    "manifest-blank-text",
    "manifest-text-mismatch",
    "manifest-audio-traversal",
    "manifest-audio-absolute",
    "manifest-audio-backslash",
    "manifest-audio-duplicate",
    "key-malformed-utf8",
    "key-malformed-json",
    "key-duplicate-key",
    "key-duplicate-nested-key",
    "key-schema",
    "key-missing-key",
    "key-extra-key",
    "key-packet",
    "key-digest",
    "key-pair-missing",
    "key-pair-extra",
    "key-pair-duplicate",
    "key-pair-unknown",
    "key-sample-index",
    "key-seed",
    "key-cartesian",
    "key-side-balance",
    "key-side",
    "key-request-order",
    "key-random-seed",
    "key-runtime-keys",
    "key-runtime-value",
    "key-audio-hash-keys",
    "key-audio-hash-value",
    "audio-missing",
    "audio-tampered-hash",
    "audio-invalid-wav",
    "audio-oversized",
)

_PACKET_FIFO_CASES = frozenset(
    {"packet-fifo", "private-fifo", "audio-dir-fifo", "audio-file-fifo"},
)


def _packet_fifo_path(case_root: Path, wrapper: dict[str, Any], case: str) -> Path:
    return {
        "packet-fifo": case_root / "packet",
        "private-fifo": case_root / "private",
        "audio-dir-fifo": case_root / "packet/audio",
        "audio-file-fifo": case_root
        / "packet/audio"
        / Path(wrapper["manifest"]["pairs"][0]["a_audio"]).name,
    }[case]


@pytest.mark.parametrize("case", _PACKET_TAMPER_CASES)
@pytest.mark.asyncio
async def test_score_rejects_complete_packet_tamper_matrix_at_stable_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_MAX_WAV_BYTES", 1024)
    (tmp_path / "source").mkdir()
    source_root, source_results, wrapper, answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path / "source",
    )
    case_root = tmp_path / "case"
    shutil.copytree(source_root, case_root)
    case_results = tmp_path / "case-results.json"
    shutil.copyfile(source_results, case_results)
    _mutate_packet_case(module, case_root, wrapper, answer_key, case)

    with pytest.raises(module.BlindAbError, match=r"^packet_integrity_error$"):
        if case in _PACKET_FIFO_CASES:
            _run_fifo_call_bounded(
                lambda: module.score_packet(case_root, case_results),
                fifo_path=_packet_fifo_path(case_root, wrapper, case),
            )
        else:
            module.score_packet(case_root, case_results)

    args = module._parse_args(
        ("score", "--packet-root", str(case_root), "--results", str(case_results)),
    )
    if case in _PACKET_FIFO_CASES:
        cli_code = _run_fifo_call_bounded(
            lambda: asyncio.run(module._run_cli(args)),
            fifo_path=_packet_fifo_path(case_root, wrapper, case),
        )
    else:
        cli_code = await module._run_cli(args)
    assert cli_code == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": "packet_integrity_error",
    }
    for sensitive in ("評価文", "voice-", "dynamic-generation", "baseline_side", "remote error"):
        assert sensitive not in output.err


def _mutate_results_case(  # noqa: C901, PLR0912, PLR0915 - explicit tamper catalog.
    results_path: Path,
    case: str,
) -> None:
    payload = json.loads(results_path.read_bytes())
    first_answer = payload["answers"][0]
    if case == "results-missing-file":
        results_path.unlink()
    elif case == "results-symlink":
        target = results_path.with_name("real-results.json")
        results_path.rename(target)
        results_path.symlink_to(target)
    elif case == "results-fifo":
        results_path.unlink()
        os.mkfifo(results_path)
    elif case == "results-oversized":
        results_path.write_bytes(b" " * (1024 * 1024 + 1))
    elif case == "results-malformed-utf8":
        results_path.write_bytes(b"\xff")
    elif case == "results-malformed-json":
        results_path.write_bytes(b"{")
    elif case == "results-duplicate-key":
        raw = json.dumps(payload).encode()
        results_path.write_bytes(b'{"schema_version":"duplicate",' + raw[1:])
    elif case == "results-duplicate-nested-key":
        raw = json.dumps(payload).encode()
        raw = raw.replace(b'"pair_id":', b'"pair_id":"duplicate","pair_id":', 1)
        results_path.write_bytes(raw)
    elif case == "results-unknown-schema":
        payload["schema_version"] = "unknown"
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-missing-schema":
        payload.pop("schema_version")
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-extra-key":
        payload["extra"] = True
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-packet-mismatch":
        payload["packet_id"] = "e" * 32
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-digest-mismatch":
        payload["manifest_sha256"] = "0" * 64
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-answer-missing":
        payload["answers"].pop()
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-answer-duplicate":
        payload["answers"][-1] = dict(first_answer)
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-answer-unknown":
        first_answer["pair_id"] = "e" * 32
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-pair-id-malformed":
        first_answer["pair_id"] = "not-an-id"
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-choice-unknown":
        first_answer["choice"] = "candidate"
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-choice-nonstring":
        first_answer["choice"] = 1
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-reason-unknown":
        first_answer["reasons"] = ["other"]
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-reason-nonstring":
        first_answer["reasons"] = [1]
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-reason-duplicate":
        first_answer["reasons"] = ["voice", "voice"]
        results_path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "results-answer-extra-key":
        first_answer["extra"] = True
        results_path.write_text(json.dumps(payload), encoding="utf-8")


_RESULTS_TAMPER_CASES = (
    "results-missing-file",
    "results-symlink",
    "results-fifo",
    "results-oversized",
    "results-malformed-utf8",
    "results-malformed-json",
    "results-duplicate-key",
    "results-duplicate-nested-key",
    "results-unknown-schema",
    "results-missing-schema",
    "results-extra-key",
    "results-packet-mismatch",
    "results-digest-mismatch",
    "results-answer-missing",
    "results-answer-duplicate",
    "results-answer-unknown",
    "results-pair-id-malformed",
    "results-choice-unknown",
    "results-choice-nonstring",
    "results-reason-unknown",
    "results-reason-nonstring",
    "results-reason-duplicate",
    "results-answer-extra-key",
)


@pytest.mark.parametrize("case", _RESULTS_TAMPER_CASES)
@pytest.mark.asyncio
async def test_score_rejects_complete_results_tamper_matrix_at_stable_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    module = _load_script()
    packet_root, results_path, _wrapper, _answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    _mutate_results_case(results_path, case)

    with pytest.raises(module.BlindAbError, match=r"^invalid_results$"):
        if case == "results-fifo":
            _run_fifo_call_bounded(
                lambda: module.score_packet(packet_root, results_path),
                fifo_path=results_path,
            )
        else:
            module.score_packet(packet_root, results_path)

    args = module._parse_args(
        ("score", "--packet-root", str(packet_root), "--results", str(results_path)),
    )
    if case == "results-fifo":
        cli_code = _run_fifo_call_bounded(
            lambda: asyncio.run(module._run_cli(args)),
            fifo_path=results_path,
        )
    else:
        cli_code = await module._run_cli(args)
    assert cli_code == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": "invalid_results",
    }
    for sensitive in ("評価文", "voice-", "dynamic-generation", "baseline_side", "remote error"):
        assert sensitive not in output.err


@pytest.mark.asyncio
async def test_prepare_actual_timeout_cleans_atomic_output_and_closes_all_async_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    _install_test_assets(module, tmp_path, monkeypatch)
    events: list[str] = []
    destination = tmp_path / "timed-out-packet"
    raw_voice_id = "fake-sensitive-voice-id"
    raw_generation = "fake-sensitive-generation"

    class AwaitingClient(_FakeBlindClient):
        def __init__(self) -> None:
            super().__init__(
                capabilities=_capabilities(
                    default_id=raw_voice_id,
                    generation=raw_generation,
                ),
            )

        @override
        async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
            self.requests.append(request)
            try:
                await asyncio.Event().wait()
            finally:
                events.append("synthesize-cancelled")
            raise AssertionError

    class ClientContext:
        async def __aenter__(self) -> AwaitingClient:
            events.append("client-enter")
            return AwaitingClient()

        async def __aexit__(self, *_args: object) -> None:
            events.append("client-close")

    class FakeTransport:
        async def aclose(self) -> None:  # noqa: PLR6301 - protocol-shaped fake.
            events.append("transport-close")

    monkeypatch.setattr(module, "_MAX_RUN_SECONDS", 0.001)
    monkeypatch.setattr(module, "AsyncIrodoriClient", lambda **_kwargs: ClientContext())
    monkeypatch.setattr(module.httpx, "AsyncHTTPTransport", FakeTransport)
    args = module._parse_args(("prepare", "--output-dir", str(destination)))

    assert await module._run_cli(args) == 2

    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": "blind_ab_timeout",
    }
    assert events == [
        "client-enter",
        "synthesize-cancelled",
        "client-close",
        "transport-close",
    ]
    assert not destination.exists()
    assert list(tmp_path.glob(".timed-out-packet.tmp-*")) == []
    for sensitive in (raw_voice_id, raw_generation, "評価文", "remote error"):
        assert sensitive not in output.err


@pytest.mark.asyncio
@pytest.mark.integration
async def test_score_subprocess_success_writes_machine_stdout_and_human_stderr(
    tmp_path: Path,
) -> None:
    module = _load_script()
    packet_root, results_path, _wrapper, _answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    # Limited unit contract: the child uses only pytest's current interpreter/project.
    argv = [
        sys.executable,
        str(SCRIPT_PATH.resolve()),
        "score",
        "--packet-root",
        str(packet_root),
        "--results",
        str(results_path),
    ]

    success = await asyncio.to_thread(
        subprocess.run,
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert success.returncode == 0
    assert json.loads(success.stdout) == module.score_packet(packet_root, results_path)
    assert success.stdout.count("\n") == 1
    assert success.stderr == (
        "candidate=2 baseline=6 same=2 unsure=2 outcome=no_detected_degradation\n"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_score_subprocess_failure_writes_only_sanitized_stderr(
    tmp_path: Path,
) -> None:
    module = _load_script()
    packet_root, results_path, _wrapper, _answer_key = await _prepare_scoring_fixture(
        module,
        tmp_path,
    )
    # Limited unit contract: the child uses only pytest's current interpreter/project.
    argv = [
        sys.executable,
        str(SCRIPT_PATH.resolve()),
        "score",
        "--packet-root",
        str(packet_root),
        "--results",
        str(results_path),
    ]
    sensitive = (
        "評価文 fake-sensitive-voice-id fake-sensitive-generation baseline_side remote error"
    )
    results_path.write_text("{" + sensitive, encoding="utf-8")
    failure = await asyncio.to_thread(
        subprocess.run,
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert failure.returncode == 2
    assert not failure.stdout
    first_line, *remaining = failure.stderr.splitlines()
    assert json.loads(first_line) == {
        "schema_version": "irodori-v4-inference-blind-ab/v1",
        "status": "failed",
        "code": "invalid_results",
    }
    assert remaining == []
    for private_value in sensitive.split():
        assert private_value not in failure.stderr
