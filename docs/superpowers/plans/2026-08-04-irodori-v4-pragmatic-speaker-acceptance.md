# Irodori-TTS v4 Pragmatic Speaker Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the immutable strict evaluation while selecting exactly one already-evaluated v4 Speaker Inversion checkpoint under the approved outlier-tolerant acceptance contract.

**Architecture:** Build one standalone, versioned Python selector in a temporary local tool root, verify it test-first, then copy the exact hashed files to the pinned Windows runtime. The selector audits all 26 immutable evaluation result sets, derives slot metrics from the raw hard-gate rows, ranks eligible checkpoints deterministically, and writes one create-only acceptance decision without deploying it.

**Tech Stack:** Python 3.10/3.11, standard-library JSON/hash/path handling, pytest, Ruff, existing Irodori-TTS v4 evaluation JSONL evidence, Windows SSH through the repository `just remote-python` recipe

---

## Scope boundary

This plan covers only the `oop53_aibeya_sp_f7269f5ffc` pragmatic acceptance pilot. The pilot succeeded and the remaining 11 models continue under `2026-08-04-irodori-v4-multi-speaker-inversion.md`; their training is not owned by this plan.

## Files and runtime roots

- Create: `/private/tmp/irodori-v4-si-pragmatic-acceptance-tools-v001/test_select_pragmatic_acceptance.py`
- Create: `/private/tmp/irodori-v4-si-pragmatic-acceptance-tools-v001/select_pragmatic_acceptance.py`
- Copy unchanged to: `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_pragmatic_acceptance_v001\`
- Create remotely: `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_token_data_v001\acceptance\pragmatic_v001\acceptance-decision.json`
- Modify after execution: `docs/superpowers/specs/2026-08-04-irodori-v4-pragmatic-speaker-acceptance-design.md`

### Task 1: Implement the pure acceptance policy test-first

- [x] **Step 1: Write boundary tests before production code**

Create the test file with these first tests:

```python
from select_pragmatic_acceptance import Policy, evaluate_similarities


POLICY = Policy(
    strict_threshold=0.75,
    outlier_floor=0.72,
    max_outliers=2,
    min_mean=0.765,
    hard_case_count=16,
)


def test_accepts_fourteen_strict_cases_and_two_bounded_outliers() -> None:
    values = [0.77] * 14 + [0.72, 0.74]
    result = evaluate_similarities(values, policy=POLICY, other_failure_count=0)
    assert result["eligible"] is True
    assert result["strict_pass_count"] == 14
    assert result["outlier_count"] == 2


def test_rejects_three_outliers() -> None:
    values = [0.78] * 13 + [0.74, 0.74, 0.74]
    result = evaluate_similarities(values, policy=POLICY, other_failure_count=0)
    assert result["eligible"] is False
    assert "too_many_outliers" in result["reasons"]


def test_rejects_a_value_below_the_floor() -> None:
    values = [0.78] * 15 + [0.719999]
    result = evaluate_similarities(values, policy=POLICY, other_failure_count=0)
    assert result["eligible"] is False
    assert "outlier_below_floor" in result["reasons"]


def test_rejects_low_mean_or_other_hard_gate_failure() -> None:
    low_mean = evaluate_similarities(
        [0.765] * 14 + [0.72, 0.72], policy=POLICY, other_failure_count=0
    )
    other_failure = evaluate_similarities(
        [0.80] * 16, policy=POLICY, other_failure_count=1
    )
    assert "mean_below_minimum" in low_mean["reasons"]
    assert "other_hard_gate_failure" in other_failure["reasons"]
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-si-pragmatic-acceptance-tools-v001/test_select_pragmatic_acceptance.py
```

Expected: collection fails because `select_pragmatic_acceptance` does not exist.

- [x] **Step 3: Implement only the pure policy**

Create `select_pragmatic_acceptance.py` with a frozen `Policy` dataclass and this contract:

```python
@dataclass(frozen=True, slots=True)
class Policy:
    strict_threshold: float
    outlier_floor: float
    max_outliers: int
    min_mean: float
    hard_case_count: int


POLICY = Policy(
    strict_threshold=0.75,
    outlier_floor=0.72,
    max_outliers=2,
    min_mean=0.765,
    hard_case_count=16,
)


def evaluate_similarities(
    values: Sequence[float], *, policy: Policy, other_failure_count: int
) -> dict[str, object]:
    if len(values) != policy.hard_case_count:
        raise ValueError("hard-case similarity count mismatch")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("speaker similarities must be finite")
    strict_pass_count = sum(value >= policy.strict_threshold for value in values)
    outliers = [value for value in values if value < policy.strict_threshold]
    mean = statistics.fmean(values)
    reasons = []
    if len(outliers) > policy.max_outliers:
        reasons.append("too_many_outliers")
    if outliers and min(outliers) < policy.outlier_floor:
        reasons.append("outlier_below_floor")
    if mean < policy.min_mean:
        reasons.append("mean_below_minimum")
    if other_failure_count:
        reasons.append("other_hard_gate_failure")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "strict_pass_count": strict_pass_count,
        "outlier_count": len(outliers),
        "minimum_speaker_similarity": min(values),
        "mean_speaker_similarity": mean,
        "other_hard_gate_failure_count": other_failure_count,
    }
```

- [x] **Step 4: Run the narrow test and verify GREEN**

Run the Task 1 pytest command. Expected: four tests pass.

### Task 2: Add deterministic ranking test-first

- [x] **Step 1: Add ranking tests**

Append tests that construct eligible dictionaries and require strict candidates first, then fewer failures, higher minimum, higher mean, candidate name, and step:

```python
from select_pragmatic_acceptance import choose_candidate


def test_ranking_prefers_the_highest_minimum_after_failure_count() -> None:
    rows = [
        {"candidate": "cosine", "checkpoint_step": 1000, "strict_eligible": False,
         "similarity_failure_count": 2, "minimum_speaker_similarity": 0.72516,
         "mean_speaker_similarity": 0.77039},
        {"candidate": "echo", "checkpoint_step": 750, "strict_eligible": False,
         "similarity_failure_count": 2, "minimum_speaker_similarity": 0.72559,
         "mean_speaker_similarity": 0.76944},
    ]
    assert choose_candidate(rows)["candidate"] == "echo"


def test_ranking_always_prefers_a_strict_candidate() -> None:
    relaxed = {"candidate": "a", "checkpoint_step": 1, "strict_eligible": False,
               "similarity_failure_count": 1, "minimum_speaker_similarity": 0.74,
               "mean_speaker_similarity": 0.80}
    strict = {"candidate": "z", "checkpoint_step": 9, "strict_eligible": True,
              "similarity_failure_count": 0, "minimum_speaker_similarity": 0.75,
              "mean_speaker_similarity": 0.76}
    assert choose_candidate([relaxed, strict])["candidate"] == "z"
```

- [x] **Step 2: Verify RED**

Run the narrow pytest command. Expected: import fails for missing `choose_candidate`.

- [x] **Step 3: Implement the deterministic key**

```python
def selection_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        0 if row["strict_eligible"] else 1,
        int(row["similarity_failure_count"]),
        -float(row["minimum_speaker_similarity"]),
        -float(row["mean_speaker_similarity"]),
        str(row["candidate"]),
        int(row["checkpoint_step"]),
    )


def choose_candidate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("no eligible pragmatic candidates")
    return min(rows, key=selection_key)
```

- [x] **Step 4: Verify GREEN**

Run the narrow pytest command. Expected: all Task 1 and Task 2 tests pass.

### Task 3: Audit immutable evaluation evidence and write create-only output

- [x] **Step 1: Add an end-to-end fixture test**

Add this deterministic fixture builder and the rejection tests. `write_jsonl` writes each mapping as one
sorted JSON object followed by `\n`; `sha256_file` is imported from the production module.

```python
import hashlib
import json
from pathlib import Path

import pytest

from select_pragmatic_acceptance import select_pragmatic_candidate


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_contract(
    root: Path, *, duplicate: bool = False, terminal_status: str = "COMPLETE"
) -> tuple[Path, Path]:
    evaluations = root / "evaluations"
    selected_embedding = root / "embeddings" / "candidate-01-step-500.safetensors"
    for candidate_index in range(26):
        candidate = f"candidate-{candidate_index:02d}"
        evaluation = evaluations / candidate
        write_json(evaluation / "terminal-evidence.json", {"status": terminal_status})
        results = []
        summaries = []
        decisions = []
        for step in (250, 500, 750, 1000, 1500):
            embedding = root / "embeddings" / f"{candidate}-step-{step}.safetensors"
            embedding.parent.mkdir(parents=True, exist_ok=True)
            embedding.write_bytes(f"{candidate}:{step}".encode())
            embedding_sha256 = hashlib.sha256(embedding.read_bytes()).hexdigest()
            if candidate_index == 0 and step == 250:
                similarities = [0.78] * 14 + [0.7252, 0.74]
            elif candidate_index == 1 and step == 500:
                similarities = [0.78] * 14 + [0.7256, 0.73]
            else:
                similarities = [0.78] * 13 + [0.74] * 3
            for case_index in range(28):
                metric_gate = case_index < 16
                similarity = similarities[case_index] if metric_gate else 0.80
                case_id = f"{candidate}:{step}:{case_index}"
                results.append(
                    {
                        "case_id": case_id,
                        "checkpoint_step": step,
                        "metric_gate_applied": metric_gate,
                        "speaker_similarity": similarity,
                        "status": "SUCCESS",
                        "evaluation_status": (
                            "REJECTED" if metric_gate and similarity < 0.75 else "PASS"
                        ),
                        "rejection_reasons": (
                            ["low_speaker_similarity"]
                            if metric_gate and similarity < 0.75
                            else []
                        ),
                        "incomplete_reasons": [],
                        "review_reasons": [],
                        "embedding_path": str(embedding),
                        "embedding_sha256": embedding_sha256,
                    }
                )
            failure_count = sum(value < 0.75 for value in similarities)
            summaries.append(
                {
                    "checkpoint_step": step,
                    "status": "REJECTED" if failure_count else "ELIGIBLE",
                    "rejection_reasons": ["low_speaker_similarity"] if failure_count else [],
                    "incomplete_reasons": [],
                    "review_reasons": [],
                }
            )
            decisions.append(
                {
                    "checkpoint_step": step,
                    "similarity_failure_count": failure_count,
                    "other_hard_gate_failure_count": 0,
                }
            )
        if duplicate and candidate_index == 0:
            results[-1]["case_id"] = results[0]["case_id"]
        selection = evaluation / "v4" / "selection"
        write_jsonl(selection / "evaluation-results.jsonl", results)
        write_jsonl(selection / "checkpoint-summary.jsonl", summaries)
        write_json(
            evaluation / "candidate-decision.json",
            {"status": "REJECTED", "checkpoint_results": decisions},
        )
    return evaluations, selected_embedding


def test_selector_writes_one_create_only_decision(tmp_path: Path) -> None:
    evaluations, _ = build_contract(tmp_path)
    output = tmp_path / "acceptance"
    decision = select_pragmatic_candidate(evaluations, output)
    assert decision["status"] == "PRAGMATICALLY_ELIGIBLE"
    assert decision["selected"]["candidate"] == "candidate-01"
    assert decision["selected"]["checkpoint_step"] == 500
    assert decision["deployment_performed"] is False
    assert (output / "acceptance-decision.json").is_file()


def test_selector_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    evaluations, _ = build_contract(tmp_path, duplicate=True)
    with pytest.raises(RuntimeError, match="duplicate case_id"):
        select_pragmatic_candidate(evaluations, tmp_path / "acceptance")


def test_selector_rejects_non_complete_terminal(tmp_path: Path) -> None:
    evaluations, _ = build_contract(tmp_path, terminal_status="FAIL")
    with pytest.raises(RuntimeError, match="terminal status is not COMPLETE"):
        select_pragmatic_candidate(evaluations, tmp_path / "acceptance")


def test_selector_rejects_embedding_hash_mismatch(tmp_path: Path) -> None:
    evaluations, selected_embedding = build_contract(tmp_path)
    selected_embedding.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="selected embedding SHA-256 mismatch"):
        select_pragmatic_candidate(evaluations, tmp_path / "acceptance")


def test_selector_refuses_existing_output_root(tmp_path: Path) -> None:
    evaluations, _ = build_contract(tmp_path)
    output = tmp_path / "acceptance"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output root already exists"):
        select_pragmatic_candidate(evaluations, output)
```

- [x] **Step 2: Verify RED**

Run the narrow pytest command. Expected: `select_pragmatic_candidate` is missing.

- [x] **Step 3: Implement the evidence scanner**

The scanner must:

1. Discover exactly 26 `*/v4/selection/evaluation-results.jsonl` files.
2. Require 140 unique case IDs and terminal status `COMPLETE` for every evaluation root.
3. Group the 16 `metric_gate_applied` rows per checkpoint.
4. Accept only low-speaker-similarity rejection reasons; count all other rejection or incomplete reasons as other hard-gate failures.
5. Cross-check `checkpoint-summary.jsonl` and any available `candidate-decision.json`.
6. Bind every results, summary, terminal, and decision file by SHA-256.
7. Require a total of 130 checkpoint slots; rank any strict eligible slot first and otherwise apply the relaxed contract.
8. Verify the selected embedding file against the identical `embedding_sha256` present in all 28 rows for that checkpoint.
9. Rank only eligible slots with `choose_candidate`.
10. Create the output directory with `exist_ok=False` and write `acceptance-decision.json` with mode `x`.

Expose one entry point with an injectable policy only for deterministic tests:

```python
def select_pragmatic_candidate(
    evaluations_root: Path,
    output_root: Path,
    *,
    policy: Policy = POLICY,
) -> dict[str, object]:
    """Audit immutable evidence, select one slot, and write one create-only decision."""
```

The top-level result must use:

```python
{
    "schema_version": "irodori-v4-si-pragmatic-acceptance/v1",
    "status": "PRAGMATICALLY_ELIGIBLE",
    "policy": asdict(POLICY),
    "audit": {"evaluation_count": 26, "checkpoint_slot_count": 130},
    "selected": selected,
    "eligible_candidates": sorted_eligible,
    "bindings": bindings,
    "deployment_performed": False,
    "active_voice_bank_unchanged": True,
}
```

- [x] **Step 4: Verify GREEN and refactor**

Run:

```bash
uv run ruff format /private/tmp/irodori-v4-si-pragmatic-acceptance-tools-v001
uv run ruff check --ignore PGH004 /private/tmp/irodori-v4-si-pragmatic-acceptance-tools-v001
uv run pytest -q /private/tmp/irodori-v4-si-pragmatic-acceptance-tools-v001/test_select_pragmatic_acceptance.py
uv run python -m py_compile /private/tmp/irodori-v4-si-pragmatic-acceptance-tools-v001/*.py
```

Expected: formatting succeeds, lint succeeds, all tests pass, and both files compile.

### Task 4: Pin and run the selector remotely

- [x] **Step 1: Hash the local tool files**

Run `sha256sum` on both files and retain the exact digests in the execution report.

- [x] **Step 2: Create the remote tool root once and copy the files**

Use `just remote-python -c` to create `_tools_token_data_pragmatic_acceptance_v001` with `exist_ok=False`, then copy both files over SSH. Do not reuse an existing root.

- [x] **Step 3: Verify remote identity**

Use `just remote-python -c` to compile both remote files and print their SHA-256 values. Require byte-for-byte equality with Step 1.

- [x] **Step 4: Execute one create-only selection**

Invoke the remote selector with:

```text
EVALUATIONS_ROOT = C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_token_data_v001\evaluations
OUTPUT_ROOT = C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_oop53_token_data_v001\acceptance\pragmatic_v001
```

Expected: exit 0, status `PRAGMATICALLY_ELIGIBLE`, one selected checkpoint, no deployment.

### Task 5: Independently verify and document the result

- [x] **Step 1: Recompute the selected metrics independently**

Use a separate read-only `just remote-python -c` script to reload the selected checkpoint's 16 hard-gate rows and verify strict pass count, outlier values, minimum, mean, other failure count, embedding path, and embedding SHA-256 against the decision.

- [x] **Step 2: Verify no owned processes or deployment changes remain**

Confirm all prior training/evaluation PIDs are absent, `deployment_performed` is false, and `active_voice_bank_unchanged` is true.

- [x] **Step 3: Append the execution result to the design document**

Record policy, selected candidate/step, exact metrics, decision/tool hashes, source binding counts, and the explicit statement that strict status remains `REJECTED` while pragmatic status is accepted. Run `git diff --check` on the document.

- [ ] **Step 4: Transition to the conditional multi-speaker phase**

Read the existing 12-model clean-data catalog and remote immutable manifests. Create a separate v4 multi-speaker design and implementation plan for the remaining 11 models, reusing this acceptance policy only after each model completes its own unchanged 140-case evaluation. Do not deploy any model without a separate explicit instruction.
