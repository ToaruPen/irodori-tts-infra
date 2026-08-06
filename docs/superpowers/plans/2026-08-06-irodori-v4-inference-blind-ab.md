# Irodori-TTS v4 推論設定 Blind AB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 動的default voiceを使い、`24 steps / linear / neutral`と`12 steps / sway / neutral`を条件名なしで12組聴き比べ、改ざん検出付きで集計できるcreate-onlyローカル評価packetを作る。

**Architecture:** Irodori infraの独立CLIが、既存benchmarkと共有する固定評価文、固定2 seed、runtimeの動的default voiceから24 WAVを生成する。公開packetは固定HTML/JavaScript、canonical manifest payload、opaque audioだけを持ち、条件対応とruntimeのhashはpacket外のprivate answer keyだけが持つ。score subcommandは両artifactとresultsのhash・schema・集合一致を検証してからexact binomialを計算し、production設定には触れない。

**Tech Stack:** Python 3.11、asyncio、httpx、Pydantic v2、標準ライブラリ`wave`/`hashlib`/`secrets`/`random`、vanilla HTML/JavaScript、pytest、Ruff、mypy、just。

---

## Execution constraints

- `/Users/sankenbisha/Dev/irodori-tts-infra/AGENTS.md`に従い、Red → Green → Refactorで進める。
- 既存のdirty worktreeを保存し、この計画のfile以外をrevertまたは上書きしない。
- repository instructionがskillのcommit例より優先する。明示依頼がないため各taskでcommitしない。
- standard service、voice bank、runtime設定、moco設定を変更しない。
- live smokeは実装とlocal gateが完了してから明示的に実行し、promotionやdeploymentを行わない。

## File map

- Create: `src/irodori_tts_infra/evaluation_samples.py` — 2つのv4評価toolが共有する6件の評価文の副作用のない単一owner。
- Modify: `scripts/benchmark_v4_inference.py:98` — inline評価文をshared module importへ置換。
- Modify: `tests/scripts/test_benchmark_v4_inference.py:82` — shared sample contractを構造だけで検証。
- Create: `scripts/v4_inference_blind_ab.py` — prepare/score CLI、randomization、synthesis、artifact integrity、resource/path guard、集計。
- Create: `scripts/assets/v4_inference_blind_ab/index.html` — `file://`で開く固定UI shell。
- Create: `scripts/assets/v4_inference_blind_ab/review.js` — playback、回答state、localStorage、results download。
- Create: `tests/scripts/test_v4_inference_blind_ab.py` — deterministic unit/contract testとthin CLI test。
- Modify: `justfile:93` — `v4-inference-blind-ab` recipeを追加。
- Modify: `AGENTS.md:33` — 新recipeの安全境界をcommand catalogへ追加。

### Task 1: 評価文を単一ownerへ抽出する

**Files:**
- Create: `src/irodori_tts_infra/evaluation_samples.py`
- Modify: `scripts/benchmark_v4_inference.py:98-106`
- Modify: `tests/scripts/test_benchmark_v4_inference.py:82-96`

- [ ] **Step 1: shared sample contractのfailing testを書く**

`tests/scripts/test_benchmark_v4_inference.py`のsample testを、productionの文面そのものではなく構造を検証する形へ変更する。

```python
def test_v4_inference_samples_cover_varied_speech_structures_without_exact_fixtures() -> None:
    module = _load_script()
    samples = module.V4_INFERENCE_SAMPLES

    assert len(samples) == 6
    assert len(set(samples)) == len(samples)
    assert all(sample.strip() == sample for sample in samples)
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
```

- [ ] **Step 2: testがshared symbol不在でfailすることを確認する**

Run: `uv run pytest -q tests/scripts/test_benchmark_v4_inference.py::test_v4_inference_samples_cover_varied_speech_structures_without_exact_fixtures`

Expected: `AttributeError`でFAILし、test collectionや既存importでは失敗しない。

- [ ] **Step 3: shared moduleを作りbenchmarkを切り替える**

`src/irodori_tts_infra/evaluation_samples.py`へ現在の6文を移す。

```python
from __future__ import annotations

V4_INFERENCE_SAMPLES = (
    "了解しました。",
    "準備ができました。次の操作を始めてもよろしいでしょうか?",
    "彼女は「確認できました」と答え、静かに画面を閉じました。",
    "状況を整理すると、通信は復旧していますが、未送信の処理が残っているため、完了するまで少しお待ちください。",
    "東京都千代田区一ツ橋二丁目で、二〇二六年八月五日の午後七時三十分に再試行します。",
    "APIの応答はHTTP 429でした。三・一四秒後に、もう一度接続してください。",
)
```

`scripts/benchmark_v4_inference.py`ではinline tupleを削除し、次をimportする。全`BENCHMARK_SAMPLES`参照を`V4_INFERENCE_SAMPLES`へ置換する。

```python
from irodori_tts_infra.evaluation_samples import V4_INFERENCE_SAMPLES
```

- [ ] **Step 4: narrow testとidentifier scanを通す**

Run: `uv run pytest -q tests/scripts/test_benchmark_v4_inference.py`

Expected: 全test PASS。

Run: `rg -n "BENCHMARK_SAMPLES|V4_INFERENCE_SAMPLES" src scripts tests/scripts`

Expected: inline ownerは`evaluation_samples.py`だけで、両toolは同じsymbolをimportする。`BENCHMARK_SAMPLES`は0件。

### Task 2: Blind ABの固定契約とbalanced randomizationを作る

**Files:**
- Create: `scripts/v4_inference_blind_ab.py`
- Create: `tests/scripts/test_v4_inference_blind_ab.py`

- [ ] **Step 1: module loader、opaque ID、pair plan、fixed conditionのfailing testを書く**

test moduleの先頭にloaderとdynamic fake dataを置く。productionのvoice名・件数・順序はfixtureへ固定しない。

```python
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit
SCRIPT_PATH = Path("scripts/v4_inference_blind_ab.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v4_inference_blind_ab", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_pair_plans_is_balanced_opaque_and_reproducible() -> None:
    module = _load_script()
    ids = (f"{value:032x}" for value in itertools.count(1))

    plans = module.build_pair_plans(
        samples=tuple(f"評価文{index}" for index in range(6)),
        seeds=(101, 202),
        randomization_seed=7,
        id_factory=lambda: next(ids),
    )

    assert len(plans) == 12
    assert sum(plan.baseline_side == "a" for plan in plans) == 6
    assert sum(plan.baseline_side == "b" for plan in plans) == 6
    assert {plan.seed for plan in plans} == {101, 202}
    assert {plan.sample_index for plan in plans} == set(range(6))
    assert all(len(plan.pair_id) == 32 for plan in plans)
    assert all(set(plan.request_order) == {"baseline", "candidate"} for plan in plans)
    assert [plan.sample_index for plan in plans] != sorted(plan.sample_index for plan in plans)
```

- [ ] **Step 2: missing moduleでfailすることを確認する**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py::test_build_pair_plans_is_balanced_opaque_and_reproducible`

Expected: script不在によるFAIL。

- [ ] **Step 3: fixed constants、types、pair plannerを最小実装する**

`scripts/v4_inference_blind_ab.py`へ次の公開shapeを作る。

```python
from __future__ import annotations

import random
import re
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from irodori_tts_infra.evaluation_samples import V4_INFERENCE_SAMPLES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ConditionName = Literal["baseline", "candidate"]
Side = Literal["a", "b"]
_BLIND_SEEDS = (101, 202)
_OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_WAV_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_WAV_BYTES = 96 * 1024 * 1024
_MAX_AUDIO_DURATION_SECONDS = 60.0


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


BASELINE = Condition("baseline", 24, "linear")
CANDIDATE = Condition("candidate", 12, "sway")


def new_opaque_id() -> str:
    return secrets.token_hex(16)


def build_pair_plans(
    *,
    samples: Sequence[str],
    seeds: Sequence[int],
    randomization_seed: int,
    id_factory: Callable[[], str] = new_opaque_id,
) -> tuple[PairPlan, ...]:
    combinations = [(sample_index, seed) for sample_index in range(len(samples)) for seed in seeds]
    if len(combinations) != 12:
        raise ValueError("blind AB requires exactly 12 sample/seed pairs")
    rng = random.Random(randomization_seed)
    sides: list[Side] = ["a"] * 6 + ["b"] * 6
    rng.shuffle(sides)
    plans: list[PairPlan] = []
    seen_ids: set[str] = set()
    for (sample_index, seed), baseline_side in zip(combinations, sides, strict=True):
        request_order: tuple[ConditionName, ConditionName] = (
            ("baseline", "candidate")
            if rng.getrandbits(1) == 0
            else ("candidate", "baseline")
        )
        generated_ids = (id_factory(), id_factory(), id_factory())
        if any(_OPAQUE_ID_RE.fullmatch(value) is None for value in generated_ids):
            raise ValueError("opaque IDs must be 128-bit lowercase hex")
        if len(set(generated_ids)) != len(generated_ids) or seen_ids.intersection(generated_ids):
            raise ValueError("opaque IDs must be unique")
        seen_ids.update(generated_ids)
        plans.append(
            PairPlan(
                pair_id=generated_ids[0],
                sample_index=sample_index,
                seed=seed,
                baseline_side=baseline_side,
                request_order=request_order,
                a_audio_id=generated_ids[1],
                b_audio_id=generated_ids[2],
            )
        )
    rng.shuffle(plans)
    return tuple(plans)
```

- [ ] **Step 4: pair planner testをpassさせる**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py::test_build_pair_plans_is_balanced_opaque_and_reproducible`

Expected: PASS。

### Task 3: canonical manifest、results validation、scoreをTDDで作る

**Files:**
- Modify: `scripts/v4_inference_blind_ab.py`
- Modify: `tests/scripts/test_v4_inference_blind_ab.py`

- [ ] **Step 1: canonical digestとpublic/private分離のfailing testを書く**

```python
def test_build_artifacts_keeps_conditions_and_runtime_metadata_private() -> None:
    module = _load_script()
    plans = _deterministic_plans(module)
    wav_by_audio_id = {
        audio_id: _wav_bytes((1000, -1000))
        for plan in plans
        for audio_id in (plan.a_audio_id, plan.b_audio_id)
    }

    manifest_wrapper, answer_key = module.build_artifact_payloads(
        packet_id="f" * 32,
        plans=plans,
        samples=tuple(f"評価文{index}" for index in range(6)),
        randomization_seed=7,
        voice_id="runtime-selected-voice",
        generation="runtime-generation",
        wav_by_audio_id=wav_by_audio_id,
    )

    public_json = module.canonical_json_bytes(manifest_wrapper).decode()
    private_json = module.canonical_json_bytes(answer_key).decode()
    assert manifest_wrapper["manifest_sha256"] == module.sha256_hex(
        module.canonical_json_bytes(manifest_wrapper["manifest"])
    )
    assert "baseline" not in public_json
    assert "candidate" not in public_json
    assert "linear" not in public_json
    assert "sway" not in public_json
    assert "runtime-selected-voice" not in public_json + private_json
    assert "runtime-generation" not in public_json + private_json
    assert len(answer_key["audio_sha256"]) == 24
```

test helperは次のdeterministic seamにする。

```python
def _opaque_id_factory() -> Callable[[], str]:
    values = (f"{value:032x}" for value in itertools.count(1))
    return lambda: next(values)


def _deterministic_plans(module: ModuleType) -> tuple[object, ...]:
    return module.build_pair_plans(
        samples=tuple(f"評価文{index}" for index in range(6)),
        seeds=(101, 202),
        randomization_seed=7,
        id_factory=_opaque_id_factory(),
    )


def _wav_bytes(samples: tuple[int, ...], *, sample_rate: int = 24_000) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return payload.getvalue()
```

- [ ] **Step 2: artifact builder不在でfailすることを確認する**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py::test_build_artifacts_keeps_conditions_and_runtime_metadata_private`

Expected: `AttributeError`でFAIL。

- [ ] **Step 3: canonical JSON、hash、manifest/private builderを実装する**

実装する契約は次のshapeとする。

```python
import hashlib
import json
from typing import Any

_MANIFEST_SCHEMA = "irodori-v4-inference-blind-ab-manifest/v1"
_ANSWER_KEY_SCHEMA = "irodori-v4-inference-blind-ab-answer-key/v1"
_REASONS = ("reading", "voice", "noise", "prosody", "emotion")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_artifact_payloads(
    *,
    packet_id: str,
    plans: tuple[PairPlan, ...],
    samples: tuple[str, ...],
    randomization_seed: int,
    voice_id: str,
    generation: str,
    wav_by_audio_id: dict[str, bytes],
) -> tuple[dict[str, object], dict[str, object]]:
    public_pairs = []
    private_pairs = []
    for plan in plans:
        public_pairs.append(
            {
                "pair_id": plan.pair_id,
                "text": samples[plan.sample_index],
                "a_audio": f"audio/{plan.a_audio_id}.wav",
                "b_audio": f"audio/{plan.b_audio_id}.wav",
            }
        )
        private_pairs.append(
            {
                "pair_id": plan.pair_id,
                "sample_index": plan.sample_index,
                "seed": plan.seed,
                "baseline_side": plan.baseline_side,
                "request_order": list(plan.request_order),
            }
        )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "packet_id": packet_id,
        "pairs": public_pairs,
        "reasons": list(_REASONS),
    }
    manifest_sha256 = sha256_hex(canonical_json_bytes(manifest))
    wrapper = {"manifest": manifest, "manifest_sha256": manifest_sha256}
    answer_key = {
        "schema_version": _ANSWER_KEY_SCHEMA,
        "packet_id": packet_id,
        "manifest_sha256": manifest_sha256,
        "audio_sha256": {
            f"audio/{audio_id}.wav": sha256_hex(wav_bytes)
            for audio_id, wav_bytes in sorted(wav_by_audio_id.items())
        },
        "pairs": private_pairs,
        "randomization_seed": f"{randomization_seed:064x}",
        "runtime": {
            "voice_id_sha256": sha256_hex(voice_id.encode()),
            "generation_sha256": sha256_hex(generation.encode()),
        },
    }
    return wrapper, answer_key
```

実装では`packet_id`、全pair/audio ID、件数、集合一致も検証し、`dict[str, Any]`を外部境界以外へ広げない。

- [ ] **Step 4: results schemaとscore境界のfailing testを書く**

```python
@pytest.mark.parametrize(
    ("candidate_wins", "baseline_wins", "same", "unsure", "outcome"),
    [
        (6, 0, 6, 0, "no_detected_degradation"),
        (0, 6, 6, 0, "degraded"),
        (3, 5, 4, 0, "no_detected_degradation"),
        (3, 3, 2, 4, "inconclusive"),
    ],
)
def test_classify_score_uses_exact_one_sided_rule(
    candidate_wins: int,
    baseline_wins: int,
    same: int,
    unsure: int,
    outcome: str,
) -> None:
    module = _load_script()
    result = module.classify_score(
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        same=same,
        unsure=unsure,
    )
    assert result.outcome == outcome
    assert 0.0 <= result.p_value <= 1.0


def test_validate_results_rejects_missing_duplicate_unknown_and_tampered_answers() -> None:
    module = _load_script()
    valid = _valid_results_payload(module)
    expected_pair_ids = {answer["pair_id"] for answer in valid["answers"]}
    assert len(module.validate_results(valid, expected_pair_ids=expected_pair_ids).answers) == 12

    missing = {**valid, "answers": valid["answers"][:-1]}
    duplicate = {**valid, "answers": [*valid["answers"][:-1], valid["answers"][0]]}
    unknown = {
        **valid,
        "answers": [*valid["answers"][:-1], {**valid["answers"][-1], "pair_id": "e" * 32}],
    }
    for payload in (missing, duplicate, unknown):
        with pytest.raises(module.BlindAbError, match="invalid_results"):
            module.validate_results(payload, expected_pair_ids=expected_pair_ids)
```

- [ ] **Step 5: exact binomial、Pydantic results model、score translationを実装する**

```python
import math
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Choice = Literal["a", "b", "same", "unsure"]
Reason = Literal["reading", "voice", "noise", "prosody", "emotion"]
Outcome = Literal["no_detected_degradation", "degraded", "inconclusive"]


class ResultAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    choice: Choice
    reasons: tuple[Reason, ...] = ()

    @field_validator("reasons")
    @classmethod
    def _unique_reasons(cls, value: tuple[Reason, ...]) -> tuple[Reason, ...]:
        if len(value) != len(set(value)):
            raise ValueError("reasons must be unique")
        return value


class ResultsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["irodori-v4-inference-blind-ab-results/v1"]
    packet_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answers: tuple[ResultAnswer, ...]


def validate_results(
    value: object,
    *,
    expected_pair_ids: set[str],
) -> ResultsPayload:
    try:
        results = ResultsPayload.model_validate(value)
    except ValidationError as error:
        raise BlindAbError("invalid_results") from error
    actual_pair_ids = [answer.pair_id for answer in results.answers]
    if (
        len(actual_pair_ids) != 12
        or len(actual_pair_ids) != len(set(actual_pair_ids))
        or set(actual_pair_ids) != expected_pair_ids
    ):
        raise BlindAbError("invalid_results")
    return results


@dataclass(frozen=True, slots=True)
class ScoreDecision:
    p_value: float
    outcome: Outcome


def exact_baseline_preference_p_value(*, baseline_wins: int, decisive: int) -> float:
    if decisive == 0:
        return 1.0
    numerator = sum(math.comb(decisive, value) for value in range(baseline_wins, decisive + 1))
    return numerator / (2**decisive)


def classify_score(
    *, candidate_wins: int, baseline_wins: int, same: int, unsure: int
) -> ScoreDecision:
    if min(candidate_wins, baseline_wins, same, unsure) < 0:
        raise ValueError("score counts must be non-negative")
    if candidate_wins + baseline_wins + same + unsure != 12:
        raise ValueError("score requires exactly 12 answers")
    p_value = exact_baseline_preference_p_value(
        baseline_wins=baseline_wins,
        decisive=candidate_wins + baseline_wins,
    )
    if unsure >= 4:
        outcome: Outcome = "inconclusive"
    elif baseline_wins > candidate_wins and p_value <= 0.05:
        outcome = "degraded"
    else:
        outcome = "no_detected_degradation"
    return ScoreDecision(p_value=p_value, outcome=outcome)


def summarize_answers(
    results: ResultsPayload,
    *,
    baseline_side_by_pair: dict[str, Side],
) -> dict[str, object]:
    counts = {"candidate_wins": 0, "baseline_wins": 0, "same": 0, "unsure": 0}
    reasons = {
        reason: {"candidate_wins": 0, "baseline_wins": 0, "same": 0, "unsure": 0}
        for reason in _REASONS
    }
    for answer in results.answers:
        if answer.choice in {"same", "unsure"}:
            bucket = answer.choice
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
        **counts,
        "decisive": counts["candidate_wins"] + counts["baseline_wins"],
        "p_value": decision.p_value,
        "outcome": decision.outcome,
        "reason_breakdown": reasons,
    }
```

`score_packet`はmanifest wrapper、answer key、resultsを読み、canonical manifest digest、packet ID、全audio hashとexpected file集合を確認してからA/Bを戻す。読み取り順はmanifest → key → audio → resultsとし、前段のintegrityが通るまでresultsを集計しない。manifest parserは固定prefix/suffix以外を拒否し、audio集合はkeyのrelative path集合と`packet/audio/*.wav`集合の完全一致を要求する。

- [ ] **Step 6: artifact/score testsを通す**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py -k "artifact or results or score or binomial"`

Expected: 全対象test PASS。

### Task 4: synthesisとcreate-only packet writerをTDDで作る

**Files:**
- Modify: `scripts/v4_inference_blind_ab.py`
- Modify: `tests/scripts/test_v4_inference_blind_ab.py`

- [ ] **Step 1: dynamic default voiceとrequest固定条件のfailing testを書く**

fake clientはcallごとにruntime catalogを注入し、production voice fixtureを持たない。

```python
@pytest.mark.asyncio
async def test_prepare_uses_only_runtime_default_and_fixed_neutral_conditions(tmp_path: Path) -> None:
    module = _load_script()
    default_id = "voice-" + secrets.token_hex(8)
    client = _FakeBlindClient(
        capabilities=_capabilities_with_dynamic_default(default_id),
        wav_bytes=_wav_bytes((1000, -1000) * 120),
    )
    destination = tmp_path / "blind-packet"

    await module.prepare_packet(
        client,
        destination=destination,
        samples=module.V4_INFERENCE_SAMPLES,
        seeds=(101, 202),
        randomization_seed=7,
        id_factory=_opaque_id_factory(),
    )

    assert len(client.requests) == 24
    assert {(request.num_steps, request.t_schedule_mode) for request in client.requests} == {
        (24, "linear"),
        (12, "sway"),
    }
    assert all(request.voice_id == default_id for request in client.requests)
    assert all(request.if_generation == client.capabilities_response.generation for request in client.requests)
    assert all(request.style == "neutral" for request in client.requests)
    assert all(request.num_candidates == 1 for request in client.requests)
    assert {request.seed for request in client.requests} == {101, 202}
    assert (destination / "packet/index.html").is_file()
    assert len(tuple((destination / "packet/audio").glob("*.wav"))) == 24
```

- [ ] **Step 2: prepare_packet不在でfailすることを確認する**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py::test_prepare_uses_only_runtime_default_and_fixed_neutral_conditions`

Expected: `AttributeError`でFAIL。

- [ ] **Step 3: readiness、default voice、WAV resource guard testsを追加する**

次のtest名と期待codeを追加する。

```python
@pytest.mark.parametrize(
    ("health_loaded", "capability_ready", "default_count", "code"),
    [
        (False, True, 1, "runtime_not_ready"),
        (True, False, 1, "runtime_not_ready"),
        (True, True, 0, "default_voice_unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_prepare_fails_closed_on_readiness_or_default_voice(
    tmp_path: Path,
    health_loaded: bool,
    capability_ready: bool,
    default_count: int,
    code: str,
) -> None:
    module = _load_script()
    client = _configured_fake_client(
        health_loaded=health_loaded,
        capability_ready=capability_ready,
        default_count=default_count,
    )
    with pytest.raises(module.BlindAbError, match=code):
        await module.prepare_packet(client, destination=tmp_path / "packet")
    assert not (tmp_path / "packet").exists()
```

加えて`invalid_wav`、`audio_too_large`、`audio_too_long`、aggregate 96 MiB、generation mismatchの各failureでdestinationとowned tempが残らないtestを加える。制限値は`monkeypatch`で小さくし、大容量fixtureは作らない。

- [ ] **Step 4: client protocol、request builder、WAV validatorを実装する**

```python
import struct
import wave
from io import BytesIO
from typing import Protocol

from irodori_tts_infra.contracts import (
    CapabilitiesResponse,
    HealthResponse,
    SynthesisRequest,
    SynthesisResult,
)


class BlindAbError(RuntimeError):
    pass


class BlindAbClient(Protocol):
    async def health(self) -> HealthResponse: ...
    async def capabilities(self) -> CapabilitiesResponse: ...
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...


def build_request(
    *, text: str, voice_id: str, generation: str, seed: int, condition: Condition
) -> SynthesisRequest:
    return SynthesisRequest(
        text=text,
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


def validate_wav(wav_bytes: bytes) -> None:
    if len(wav_bytes) > _MAX_WAV_BYTES:
        raise BlindAbError("audio_too_large")
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
                raise BlindAbError("invalid_wav")
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            payload = wav_file.readframes(frames)
    except (EOFError, wave.Error) as error:
        raise BlindAbError("invalid_wav") from error
    if channels <= 0 or sample_rate <= 0 or frames <= 0:
        raise BlindAbError("invalid_wav")
    if frames / sample_rate > _MAX_AUDIO_DURATION_SECONDS:
        raise BlindAbError("audio_too_long")
    if len(payload) != frames * channels * 2:
        raise BlindAbError("invalid_wav")
```

`prepare_packet`は`health.status == "ok"`、`health.model_loaded`、`capabilities.ready`、`readiness == "ready"`、default voiceちょうど1件を要求する。各pairで`request_order`どおり逐次synthesizeし、sideに対応するopaque filenameへ未加工bytesを書き、合計sizeを96 MiB以下に保つ。runtime codeが`runtime_generation_mismatch`ならfallbackせずそのまま失敗させる。

公開signatureとCSPRNG defaultは次に固定する。

```python
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
    plans = build_pair_plans(
        samples=samples,
        seeds=seeds,
        randomization_seed=effective_seed,
        id_factory=id_factory,
    )
    return await _generate_atomic_packet(
        client,
        destination=destination,
        packet_id=packet_id,
        plans=plans,
        samples=samples,
        randomization_seed=effective_seed,
    )
```

- [ ] **Step 5: create-only atomic output guardを実装する**

destination final componentと既存/symlinkを先に検証し、同じresolved parentで`tempfile.mkdtemp`したdirectoryだけをcleanup対象にする。

```python
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator


@contextmanager
def atomic_output_directory(destination: Path) -> Iterator[Path]:
    expanded = destination.expanduser().absolute()
    if expanded.name in {"", ".", ".."}:
        raise BlindAbError("unsafe_output_path")
    parent = expanded.parent.resolve(strict=True)
    final = parent / expanded.name
    if final.exists() or final.is_symlink():
        raise BlindAbError("output_exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=parent))
    try:
        yield temporary
        if final.exists() or final.is_symlink():
            raise BlindAbError("output_exists")
        os.replace(temporary, final)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
```

実装ではparent不在、非directory、permission errorを`unsafe_output_path`または`client_error`のstable codeへ変換する。`rmtree`対象が作成済み`temporary`と一致することをtestする。

- [ ] **Step 6: prepare testsを通す**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py -k "prepare or wav or output or generation"`

Expected: 全対象test PASS、test終了後に`.tmp-` directoryが0件。

### Task 5: file URL対応の評価UIを作る

**Files:**
- Create: `scripts/assets/v4_inference_blind_ab/index.html`
- Create: `scripts/assets/v4_inference_blind_ab/review.js`
- Modify: `scripts/v4_inference_blind_ab.py`
- Modify: `tests/scripts/test_v4_inference_blind_ab.py`

- [ ] **Step 1: UI assetとmanifest wrapperのfailing contract testを書く**

```python
def test_public_ui_is_file_url_self_contained_and_condition_blind(tmp_path: Path) -> None:
    module = _load_script()
    destination = _prepare_with_fake_client(module, tmp_path)
    index = (destination / "packet/index.html").read_text()
    script = (destination / "packet/review.js").read_text()
    manifest_js = (destination / "packet/manifest.js").read_text()
    public_text = "\n".join((index, script, manifest_js))

    assert '<script src="manifest.js"></script>' in index
    assert '<script src="review.js"></script>' in index
    assert "fetch(" not in public_text
    assert "innerHTML" not in public_text
    assert "localStorage" in script
    assert "baseline" not in public_text
    assert "candidate" not in public_text
    assert "linear" not in public_text
    assert "sway" not in public_text
    assert "manifest_sha256" in manifest_js
```

- [ ] **Step 2: missing assetsでfailすることを確認する**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py::test_public_ui_is_file_url_self_contained_and_condition_blind`

Expected: asset不在またはpacket copy不在でFAIL。

- [ ] **Step 3: accessibleな固定HTML shellを作る**

`index.html`はUTF-8、`lang="ja"`、responsive viewportを指定し、次の固定IDを持つ。

```html
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Irodori Blind AB</title>
  </head>
  <body>
    <main>
      <p id="progress" aria-live="polite"></p>
      <p id="sample-text"></p>
      <section aria-label="音声A"><h2>A</h2><audio id="audio-a" controls preload="metadata"></audio></section>
      <section aria-label="音声B"><h2>B</h2><audio id="audio-b" controls preload="metadata"></audio></section>
      <fieldset id="choices"><legend>どちらが良いですか</legend></fieldset>
      <fieldset id="reasons"><legend>理由（任意）</legend></fieldset>
      <nav>
        <button id="previous" type="button">前へ</button>
        <button id="next" type="button">次へ</button>
      </nav>
      <button id="download" type="button" disabled>結果を保存</button>
      <p id="remaining" aria-live="polite"></p>
    </main>
    <script src="manifest.js"></script>
    <script src="review.js"></script>
  </body>
</html>
```

CSSは同fileへinlineで置き、condition名やprofile値を含めない。autoplay属性を付けない。

- [ ] **Step 4: review.jsへstate、navigation、downloadを実装する**

`window.IRODORI_BLIND_AB_MANIFEST`を同期的に読み、DOM挿入は`textContent`とproperty assignmentだけを使う。回答keyはpair ID、storage keyはpacket IDとする。

```javascript
"use strict";

const wrapper = window.IRODORI_BLIND_AB_MANIFEST;
const manifest = wrapper.manifest;
const storageKey = `irodori-blind-ab:${manifest.packet_id}`;
const choices = [["a", "Aが良い"], ["b", "Bが良い"], ["same", "同等"], ["unsure", "判断できない"]];
const reasonLabels = {reading: "読み", voice: "声", noise: "ノイズ", prosody: "自然さ・韻律", emotion: "感情"};

function loadState() {
  try {
    const restored = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (restored && Number.isInteger(restored.index) && typeof restored.answers === "object") {
      restored.index = Math.max(0, Math.min(manifest.pairs.length - 1, restored.index));
      return restored;
    }
  } catch (error) {
    localStorage.removeItem(storageKey);
  }
  return {index: 0, answers: {}};
}

const state = loadState();

function save() {
  localStorage.setItem(storageKey, JSON.stringify(state));
}

function answeredCount() {
  return manifest.pairs.filter((pair) => state.answers[pair.pair_id]?.choice).length;
}

function render() {
  const pair = manifest.pairs[state.index];
  document.getElementById("sample-text").textContent = pair.text;
  document.getElementById("audio-a").src = pair.a_audio;
  document.getElementById("audio-b").src = pair.b_audio;
  document.getElementById("progress").textContent = `${state.index + 1} / ${manifest.pairs.length}`;
  document.getElementById("remaining").textContent = `未回答 ${manifest.pairs.length - answeredCount()} 件`;
  document.getElementById("previous").disabled = state.index === 0;
  document.getElementById("next").disabled = state.index === manifest.pairs.length - 1;
  document.getElementById("download").disabled = answeredCount() !== manifest.pairs.length;
  renderSelection(pair.pair_id);
}

function downloadResults() {
  const answers = manifest.pairs.map((pair) => ({pair_id: pair.pair_id, ...state.answers[pair.pair_id]}));
  const result = {
    schema_version: "irodori-v4-inference-blind-ab-results/v1",
    packet_id: manifest.packet_id,
    manifest_sha256: wrapper.manifest_sha256,
    answers,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: "application/json"}));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "irodori-blind-ab-results.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function renderSelection(pairId) {
  const current = state.answers[pairId] || {choice: null, reasons: []};
  const choiceRoot = document.getElementById("choices");
  const reasonRoot = document.getElementById("reasons");
  choiceRoot.querySelectorAll("label").forEach((element) => element.remove());
  reasonRoot.querySelectorAll("label").forEach((element) => element.remove());
  choices.forEach(([value, labelText]) => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "choice";
    input.value = value;
    input.checked = current.choice === value;
    input.addEventListener("change", () => {
      state.answers[pairId] = {choice: value, reasons: current.reasons};
      save();
      render();
    });
    label.append(input, document.createTextNode(labelText));
    choiceRoot.append(label);
  });
  manifest.reasons.forEach((reason) => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = reason;
    input.checked = current.reasons.includes(reason);
    input.addEventListener("change", () => {
      const selected = new Set(current.reasons);
      input.checked ? selected.add(reason) : selected.delete(reason);
      state.answers[pairId] = {choice: current.choice, reasons: [...selected]};
      save();
      render();
    });
    label.append(input, document.createTextNode(reasonLabels[reason]));
    reasonRoot.append(label);
  });
}

function move(delta) {
  state.index = Math.max(0, Math.min(manifest.pairs.length - 1, state.index + delta));
  save();
  render();
}

document.getElementById("previous").addEventListener("click", () => move(-1));
document.getElementById("next").addEventListener("click", () => move(1));
document.getElementById("download").addEventListener("click", () => {
  if (answeredCount() === manifest.pairs.length) {
    downloadResults();
  }
});
render();
```

- [ ] **Step 5: prepare writerがassetsとmanifest.jsを配置する**

manifest wrapperを次の固定prefix/suffixで書き、score parserも同じconstantsを使う。

```python
_MANIFEST_PREFIX = "window.IRODORI_BLIND_AB_MANIFEST="
_MANIFEST_SUFFIX = ";\n"


def encode_manifest_js(wrapper: dict[str, object]) -> bytes:
    return _MANIFEST_PREFIX.encode() + canonical_json_bytes(wrapper) + _MANIFEST_SUFFIX.encode()
```

assetsは`Path(__file__).parent / "assets/v4_inference_blind_ab"`から`packet/`へcopyする。source assetがregular fileでない場合は`client_error`でfailし、partial destinationを残さない。

- [ ] **Step 6: UI contract testを通す**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py -k "public_ui or manifest"`

Expected: 全対象test PASS。

### Task 6: CLI、stable error、just recipeを接続する

**Files:**
- Modify: `scripts/v4_inference_blind_ab.py`
- Modify: `tests/scripts/test_v4_inference_blind_ab.py`
- Modify: `justfile:93`
- Modify: `AGENTS.md:33`

- [ ] **Step 1: parserとloopback restrictionのfailing testsを書く**

```python
@pytest.mark.parametrize("base_url", ["http://127.0.0.1:18924", "http://[::1]:18924"])
def test_prepare_cli_accepts_numeric_loopback_only(base_url: str) -> None:
    module = _load_script()
    args = module.parse_args(("prepare", "--base-url", base_url, "--output-dir", "/tmp/blind"))
    assert args.base_url == base_url


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:18924", "http://192.0.2.1:18924", "https://user@127.0.0.1:18924", "http://127.0.0.1:18924/path"],
)
def test_prepare_cli_rejects_non_numeric_or_non_loopback_url(base_url: str) -> None:
    module = _load_script()
    with pytest.raises(SystemExit):
        module.parse_args(("prepare", "--base-url", base_url, "--output-dir", "/tmp/blind"))
```

- [ ] **Step 2: parser testがfailすることを確認する**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py -k "prepare_cli"`

Expected: `parse_args`不在でFAIL。

- [ ] **Step 3: prepare/score subcommandとoverall timeoutを実装する**

```python
import argparse
import asyncio
import ipaddress
import sys
import webbrowser
from urllib.parse import urlsplit

from irodori_tts_infra.client import AsyncIrodoriClient, ClientError

_DEFAULT_BASE_URL = "http://127.0.0.1:8924"
_MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_RUN_SECONDS = 900.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or score an Irodori v4 blind AB packet.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base-url", type=loopback_base_url, default=_DEFAULT_BASE_URL)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--open", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--packet-root", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


async def execute_prepare(*, base_url: str, output_dir: Path) -> Path:
    transport = httpx.AsyncHTTPTransport()
    async with AsyncIrodoriClient(
        base_url=base_url,
        timeout=None,
        transport=transport,
        max_response_bytes=_MAX_HTTP_RESPONSE_BYTES,
    ) as client:
        async with asyncio.timeout(_MAX_RUN_SECONDS):
            return await prepare_packet(client, destination=output_dir)
```

`loopback_base_url`は既存benchmarkと同じnumeric loopback検証を使う。`run_cli`はprepare成功時にschema/status/packet path/private key path/pair countをstdout JSONへ出す。`--open`はatomic rename後だけ`packet/index.html`のabsolute `file:` URIを`webbrowser.open`へ渡す。open失敗時は`browser_open_failed`をstderr JSONへ出し、完成packetは保持する。scoreは人間summaryをstderr、machine JSONをstdoutへ出す。

- [ ] **Step 4: stable failure allowlistとCLI error mapping testsを書く**

allowlistへ設計書の全codeを列挙し、未知messageが`client_error`へ縮退するtest、timeoutが`blind_ab_timeout`になるtest、remote error text/voice ID/textがstderrへ出ないtestを追加する。failure JSONは次だけを持つ。

```json
{"schema_version":"irodori-v4-inference-blind-ab/v1","status":"failed","code":"runtime_not_ready"}
```

- [ ] **Step 5: justfileとAGENTS command catalogを更新する**

`justfile`へ追加する。

```just
# Prepare or score a local blind AB packet for fixed v4 inference profiles.
v4-inference-blind-ab *args:
    uv run python scripts/v4_inference_blind_ab.py "$@"
```

`AGENTS.md`のv4 benchmark説明直後へ英語で追加する。

```markdown
- `just v4-inference-blind-ab {prepare,score} [ARGS]`: create or score a
  create-only local listening packet for the fixed v4 baseline/candidate profiles;
  it never changes runtime or production settings.
```

- [ ] **Step 6: CLIとjust expansionを検証する**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py -k "cli or error or timeout"`

Expected: 全対象test PASS。

Run: `just --dry-run v4-inference-blind-ab score --packet-root /tmp/example --results /tmp/results.json`

Expected: `uv run python scripts/v4_inference_blind_ab.py score --packet-root /tmp/example --results /tmp/results.json`へ正しく展開される。

### Task 7: security/integrity回帰とrepository gateを収束させる

**Files:**
- Modify: `tests/scripts/test_v4_inference_blind_ab.py`
- Modify only if tests expose a defect: files listed in Tasks 1-6

- [ ] **Step 1: tamper matrixを完成させる**

次の各caseを独立testにし、すべて`packet_integrity_error`または`invalid_results`でfail closedすること、stdoutが空であることを検証する。

- manifest payload 1 byte変更
- manifest digest変更
- answer key packet ID変更
- answer key pair mapping欠落/重複
- WAV 1 byte変更
- WAV欠落
- extra WAV追加
- results packet ID/digest変更
- results answer欠落/重複/未知ID
- unknown choice/reason、reason重複
- malformed JSON、未知schema version

- [ ] **Step 2: full narrow testを実行する**

Run: `uv run pytest -q tests/scripts/test_v4_inference_blind_ab.py tests/scripts/test_benchmark_v4_inference.py`

Expected: 全test PASS。

- [ ] **Step 3: changed Python/asset static checksを実行する**

Run: `uv run ruff check src/irodori_tts_infra/evaluation_samples.py scripts/benchmark_v4_inference.py scripts/v4_inference_blind_ab.py tests/scripts/test_benchmark_v4_inference.py tests/scripts/test_v4_inference_blind_ab.py`

Expected: PASS。

Run: `uv run ruff format --check src/irodori_tts_infra/evaluation_samples.py scripts/benchmark_v4_inference.py scripts/v4_inference_blind_ab.py tests/scripts/test_benchmark_v4_inference.py tests/scripts/test_v4_inference_blind_ab.py`

Expected: PASS。

Run: `uv run mypy`

Expected: PASS。

- [ ] **Step 4: repository full gateを実行する**

Run: `just check`

Expected: Ruff lint、format、mypy、vulture、default pytestがすべてPASS。GPU/network/SSH testはdefault marker設定により実行されない。

Run: `git diff --check`

Expected: 出力なし、exit 0。

- [ ] **Step 5: repeated identifierとdirty-worktree scopeを確認する**

Run: `rg -n "v4-inference-blind-ab|V4_INFERENCE_SAMPLES|manifest_sha256|browser_open_failed" AGENTS.md justfile scripts tests docs/superpowers`

Expected: identifierはこの機能のowner/helper/test/design/planだけに現れ、unrelated subsystemへ漏れない。

Run: `git status --short && git diff --stat`

Expected: 既存dirty changesが保持され、Blind ABの変更はFile mapの範囲だけ。commitは作成されていない。

### Task 8: polishmentとAI slop reviewをサブエージェントで通す

**Files:**
- Review: Blind AB implementation diff and its interaction with the existing dirty worktree
- Modify only for verified findings: files listed in Tasks 1-7

- [ ] **Step 1: polishment skillを読み、reviewer subagentへread-only reviewを依頼する**

主agentが`/Users/sankenbisha/Dev/dotfiles/home/.codex/skills/polishment/SKILL.md`を全て読み、skillのreview scopeとrequired evidenceをpromptへ反映する。reviewerにはBlind ABのfile map、設計書、計画書、`git diff`、narrow/full gate結果を渡し、既存dirty changeをrevertせず、codeを編集せず、correctness・clarity・test adequacyのactionable findingだけをseverity付きで返すよう依頼する。

- [ ] **Step 2: polishment findingを検証して必要なものだけ直す**

各findingを該当codeとtestで再現する。behavior defectはfailing testを先に追加し、最小修正後にnarrow testを通す。根拠のないstyle preferenceは採用しない。修正があれば`just check`と`git diff --check`を再実行し、両方PASSさせる。

- [ ] **Step 3: ai-slop-cleaner skillを読み、別reviewer subagentへread-only reviewを依頼する**

主agentが`/Users/sankenbisha/Dev/dotfiles/home/.codex/skills/ai-slop-cleaner/SKILL.md`を全て読み、polishment反映後のdiffを別subagentへ渡す。reviewerには重複、dead code、過剰なabstraction、不要なcompatibility、曖昧なhelper ownershipを探し、behaviorを変えない削減だけを提案し、codeを編集しないよう依頼する。

- [ ] **Step 4: AI slop findingを検証してbehavior-preserving cleanupを行う**

採用するfindingごとに既存testが守るobservable contractを列挙し、削減後にnarrow testを実行する。新しいarchitecture layerやproduction compatibility shimは追加しない。修正があれば`just check`と`git diff --check`を再実行し、両方PASSさせる。

- [ ] **Step 5: review収束を確認する**

両reviewerのactionable findingが0件、または全findingが修正済み/根拠付き却下であることをreview logへまとめる。最終`git status --short`で既存dirty changesが保持され、commitが作成されていないことを確認する。

### Task 9: 明示的なlive smokeとユーザー向けblind packetを作る

**Files:**
- Runtime only: user-selected output directory outside Git
- No source modifications unless smoke exposes a reproduced defect

- [ ] **Step 1: read-only readiness preflightを行う**

SSH tunnelが既に用意されたloopback endpointへ`/health`と`/capabilities`を問い合わせ、`status=ok`、`model_loaded=true`、`ready=true`、`readiness=ready`、default voice 1件を確認する。responseはterminalへraw保存せず、voice名/ID/catalogを報告へ転記しない。

- [ ] **Step 2: create-only packetを生成する**

Run:

```bash
just v4-inference-blind-ab prepare \
  --base-url http://127.0.0.1:18924 \
  --output-dir /tmp/irodori-v4-12-sway-blind-ab \
  --open
```

Expected: 15分以内に24 WAV、public packet、private answer keyがatomicに完成し、browserでpacketが開く。既存destinationがある場合は上書きせず別のexplicit pathを選ぶ。

- [ ] **Step 3: browser manual smokeを行う**

次を手動確認する。

- A/Bが両方再生でき、autoplayされない。
- 前/次、任意順再生、回答変更が機能する。
- reload後に回答と現在位置が復元される。
- 11件以下ではdownload不可、12件回答後だけdownload可能。
- downloaded JSONにtext、voice、generation、condition名がない。

- [ ] **Step 4: downloaded resultsをscoreする**

Run:

```bash
just v4-inference-blind-ab score \
  --packet-root /tmp/irodori-v4-12-sway-blind-ab \
  --results ~/Downloads/irodori-blind-ab-results.json
```

Expected: stderrに短い勝敗summary、stdoutにscore schema、12回答、exact p-value、3状態outcomeが出る。scoreはruntime/moco設定を変更しない。

- [ ] **Step 5: non-mutationを確認して引き渡す**

標準service PID/port、health generation hash、voice-bank hash、mocoのinference設定をpreflight値と比較する。差分がないことを報告し、packet path、results path、score outcomeだけをユーザーへ渡す。`no_detected_degradation`でも12/swayへ自動昇格しない。
