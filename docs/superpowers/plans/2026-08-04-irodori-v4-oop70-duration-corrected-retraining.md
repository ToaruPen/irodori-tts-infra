# Irodori-TTS v4 Manual Override and oop70 Duration-Corrected Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use subagents only if the user explicitly requests delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 3話者を監査可能な手動実用合格として記録し、duration/text-length補正済み800件manifestからoop70 Speaker Inversionを再学習・評価して12話者のoperator dispositionを閉じる。

**Architecture:** 既存の自動判定と`final-verification-v001.json`をimmutable入力として保持し、手動override、residual manifest、oop70 scratch、条件付きecho、operator verificationを新しいcreate-only rootへ保存する。訓練・評価の実処理は検証済みsupervisorをhash-boundで再利用し、新規toolは選択、契約差分の限定注入、分岐、集約を担当する。

**Tech Stack:** local Python 3.11、pinned remote Python 3.10、NumPy、PyTorch safetensors、Irodori-TTS v4-Small、SpeechBrain ECAPA-TDNN、Whisper large-v3-turbo、pytest、Ruff、Windows OpenSSH/PowerShell

---

## File map

Local runtime root: `/private/tmp/irodori-v4-oop70-duration-v001/`

- `manual_override.py`: 3話者のoperator overrideをimmutable自動decisionへ結び付ける。
- `duration_residual_manifest.py`: OLS residual、duration五分位、800件manifestを作る。
- `verify_duration_residual_manifest.py`: builderをimportせず、選択と全bindingを再計算検証する。
- `prepare_oop70_scratch.py`: 既存scratch configを検証し、新manifest用config/setupを作る。
- `run_oop70_echo_training.py`: generic token supervisorをhash検証してduration-residual echo契約へ限定する。
- `decide_oop70_branch.py`: scratch合格、echo実行、改善不足停止を決定する。
- `run_oop70_queue.py`: scratch訓練、評価、条件付きechoを直列実行する。
- `select_oop70_pragmatic_acceptance.py`: 1話者のraw評価行へ既存policyを適用するadapter。
- `launch_oop70_queue_detached.py`: fail-closed preflight後にqueueをdetached起動する。
- `verify_operator_disposition.py`: override、oop70結果、v3/voice bank不変を集約検証する。
- `test_manual_override.py`: overrideのRed/Green contract test。
- `test_duration_residual_manifest.py`: OLS、bin、選択、manifestのcontract test。
- `test_verify_duration_residual_manifest.py`: 独立manifest verifierの改変検知test。
- `test_prepare_oop70_scratch.py`: config/setupのcontract test。
- `test_decide_oop70_branch.py`: 3分岐のcontract test。
- `test_run_oop70_queue.py`: queue状態遷移と再開のcontract test。
- `test_verify_operator_disposition.py`: 12話者dispositionのcontract test。
- `test_publish_runtime_bundle.py`: chunked upload、remote copy、exclusive publishのcontract test。
- `publish_runtime_bundle.py`: 新規toolをchunk転送し、既存toolをremote側でbyte copyしてbundleをpublishする。

Remote tool root:
`C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v009`

Remote run root:
`C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001`

Remote queue root:
`C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-queue-v005`

## Pinned remote inputs

実装中に次のpathを探索・置換しない。各fileは記載したSHA-256、各directory treeは記載した
file countとtree SHA-256をpreflightで要求する。

| Input | Path | Expected identity |
|---|---|---|
| Phase 1 root | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001` | immutable input root |
| Final verification | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\final-verification-v001.json` | `e0411dbf6d90567ee146a05a8cdf28e4def5e117c09672f8e7fdf5e39f597c03` |
| v3 baseline | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\v3-runtime-baseline-v001.json` | `b058a26b06e111ac66300db3fd8686803143348c3587831bbf5a1d207f01394e` |
| oop70 source manifest | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\manifests\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\clean-manifest.jsonl` | `da85ddb21c8bdbaecbdc8269d3bfe9673128ea683c24a92d770de368c56174a4` |
| oop70 ECAPA audit | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\models\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\refinement\central-q50-audit-v001\training-ecapa-results.jsonl` | `2f289542b141c0739f342406e142f6ddc0c7780faf919e2d3e6218ed82048fac` |
| Phase 1 oop70 scratch config | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\models\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\scratch\config.yaml` | `d63cfe4cf39d325c489828ebab190aa661050eff96103548975b93bed59aba15` |
| v3 evaluation manifest | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\evaluation_speed_v6\manifests\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\evaluation-manifest.json` | `a147807c63e72d47f3f5266770232a74e51b08436bd51956fde7097e60cbdf75` |
| oop70 reference manifest | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\training\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\evaluation_assets\reference-wavs.json` | `dee1267c0c7b7b7f653d9e07dbe913ea5e5d649dffed8879015b200dad59456b` |
| v4 upstream | `C:\Users\takut\Dev\Irodori-TTS-v4-8ca3acb` | commit `8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71`, tracked clean |
| v4 Python | `C:\Users\takut\Dev\Irodori-TTS-v4-8ca3acb\.venv\Scripts\python.exe` | file exists |
| queue/metrics Python | `C:\Users\takut\Dev\Irodori-TTS\.venv\Scripts\python.exe` | file exists |
| v4 base checkpoint | `D:\hf_cache\hub\models--Aratako--Irodori-TTS-v4-Small\snapshots\e4aaac4df355ff560dcd35e0dae272c3a759317b\model.safetensors` | `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593` |
| ECAPA source | `D:\hf_cache\hub\models--speechbrain--spkrec-ecapa-voxceleb\snapshots\0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` | 5 files, tree `3c7366fdcb1e9c1b2f36e8280df884cc476a07c20858ea961a210f6eaa621738` |
| ECAPA savedir | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\evaluation_speed_v6\runtime-cache\ecapa` | pre-existing cache only |
| Whisper source | `D:\hf_cache\hub\models--openai--whisper-large-v3-turbo\snapshots\41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | tree `3a11028cd81359a5f51120ad0eac67f76b6072a41603bc53ff6bff0381d78705` |
| Metrics script | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\evaluation_speed_v6\runtime-inputs-v1\scripts\compute_600m_speaker_metrics.py` | `fa83491f0ee2f1e1f21c8d833ba90557ed67c885a2512c9c05104faf3b14a407` |

再利用する既存toolもsource pathとSHA-256を固定する。

| Tool | Source path | SHA-256 |
|---|---|---|
| Training supervisor | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_v001\run_token_training_supervisor.py` | `e7913be263521cc5defed7767e388fe8261271facb2b52807c5402a459d7fc3b` |
| Training supervisor contract | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_v001\token_search_contract.py` | `b470477c3fb0f2966426778d3e0d49051044490b569430a4cb53fd73f0ff0700` |
| Training supervisor constants | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_v001\prepare_token_search.py` | `3c69980370cf8663c483811942181c6e1b2cfe5fc43375f1f0619f09aa8f784e` |
| Scratch training wrapper | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_multi_speaker_v003\run_multispeaker_training.py` | `ab3b4cf82593f4b32a30bc98e79ff616504879c31cf83e9d55a5a820b0dcbdbd` |
| Scratch training contract | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_multi_speaker_v003\multispeaker_contract.py` | `22c31a542b9da5e5568ea279e47a16e69767d715d7ba69048f7bbb3d1f23e146` |
| Scratch evaluation preparer | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_multi_speaker_v003\prepare_multispeaker_evaluation.py` | `c87c848619857ce2dfcf655eb1550e894fa22618d3e518dfb9d57d037f944ffe` |
| Scratch evaluation wrapper | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_multi_speaker_v003\run_multispeaker_evaluation.py` | `75bbd2c0158389e4b6bb86e63cfe3071e34927c02f766251e6dbe43af74ae7f7` |
| Scratch generator | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v3_v4_oop53_comparison_v001\v4\runtime\generate_v4_checkpoint_audio_remote.py` | `a7974544140a8376a6b1203645a7c64f7bac6a6f750d95319f2b5082bc9e9d54` |
| Refinement generator | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_stage3_v003\generate_refinement_checkpoint_audio_remote.py` | `37f0636d41ee28d6d47f96cf0653311bdf3b55b1130ecae7b310b0db935c7d2c` |
| Evaluation supervisor | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools\run_irodori_v4_candidate_evaluation_supervisor.py` | `6c404eefdf4a8db534885fa526d071ce1b70bef61c7c47caeeb3d2c78c7e3ee0` |
| Refinement evaluator entrypoint | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_stage4_v001\refinement_evaluator_entrypoint.py` | `dab18dbe535e024d278300e88053e3bb4dab75292edcbd774152ef76df8ed0d7` |
| Pragmatic acceptance | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_pragmatic_acceptance_v002\select_pragmatic_acceptance.py` | `9b68357088672472ca5e6c71afae434b1a45f9f8d36109bf98d5eaf3d5c4a736` |
| Candidate decision | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools\decide_irodori_v4_si_candidate.py` | `59349ce3b4a85e34c1fdd8fdb0d7b639fe44cd221fd62edd494f6e2dbd4055cb` |
| Multi-speaker echo evaluation adapter | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_multi_speaker_refinement_echo_recovery_v004\prepare_phase3_evaluation.py` | `bc032073102b60ec59b789c97380f7577d5ad4fab6ca0fc105fdd771302faeed` |
| Echo evaluation preparer | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_token_data_stage3_v001\prepare_refinement_evaluation.py` | `219b23a2d9b819deeaf3285d103ddf4b953e715e810bbfe6a42645765639c921` |
| Echo evaluation wrapper | `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_multi_speaker_refinement_echo_recovery_v004\run_phase3_evaluation.py` | `5fd3b4dc0e7d81cdbe36666306c92d6177ecf1c2fd92c11aa4357e32b14721bf` |

旧echo evaluation preparerは評価manifestの話者IDを`oop53`へ固定するため、直接は呼ばない。
上記adapterが学習setupに束縛された`oop70`へmanifest/evidenceを適応し、旧preparerのhashも検証してから実行する。

Repository documentation only:

- Modify: `docs/superpowers/plans/2026-08-04-irodori-v4-multi-speaker-inversion.md`
- Modify: `docs/superpowers/specs/2026-08-04-irodori-v4-oop70-duration-corrected-retraining-design.md`

既存のリポジトリ変更はユーザー所有として保持する。コミットは明示依頼がないため行わない。

## Task 1: 手動override contractをTDDで作る

**Files:**

- Create: `/private/tmp/irodori-v4-oop70-duration-v001/test_manual_override.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/manual_override.py`

- [ ] **Step 1: 失敗するoverride testを書く**

```python
from pathlib import Path

import pytest

from manual_override import OVERRIDES, build_manual_override


def test_override_inventory_is_fixed() -> None:
    assert OVERRIDES == {
        "oop176_natsu_no_owari_sp_dcec9a11d3": 750,
        "oop52_aibeya_2_sp_5d544fe890": 750,
        "miu": 1500,
    }


def test_refuses_non_pass_final_verification(tmp_path: Path) -> None:
    source = tmp_path / "final.json"
    source.write_text('{"status":"FAIL"}', encoding="utf-8")
    with pytest.raises(ValueError, match="final verification must be PASS"):
        build_manual_override(source, tmp_path / "override.json")


def test_refuses_existing_output(valid_final: Path, tmp_path: Path) -> None:
    output = tmp_path / "override.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_manual_override(valid_final, output)
```

- [ ] **Step 2: testがRedになることを確認する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_manual_override.py
```

Expected: `ModuleNotFoundError: No module named 'manual_override'`

- [ ] **Step 3: override builderの最小実装を書く**

```python
OVERRIDES = {
    "oop176_natsu_no_owari_sp_dcec9a11d3": 750,
    "oop52_aibeya_2_sp_5d544fe890": 750,
    "miu": 1500,
}


def build_manual_override(source: Path, output: Path) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output}")
    final = read_json(source)
    if final.get("status") != "PASS":
        raise ValueError("final verification must be PASS")
    by_id = {row["model_id"]: row for row in final["models"]}
    phase1_root = source.parent
    rows = []
    for model_id, checkpoint_step in OVERRIDES.items():
        model = by_id[model_id]
        if model.get("final_status") != "REJECTED_EXHAUSTED":
            raise ValueError(f"override source must be rejected: {model_id}")
        echo = model["echo"]
        acceptance_path = (
            phase1_root
            / "models"
            / model_id
            / "refinement"
            / "central-q50-echo-lr0001-v001-acceptance"
            / "acceptance-decision.json"
        )
        training_terminal = Path(echo["training"]["terminal"]["path"])
        checkpoint = checkpoint_from_terminal(training_terminal, checkpoint_step)
        rows.append(
            build_override_row(
                model_id=model_id,
                checkpoint_step=checkpoint_step,
                checkpoint=checkpoint,
                acceptance_path=acceptance_path,
                training_terminal=training_terminal,
                evaluation_terminal=Path(echo["evaluation"]["terminal"]["path"]),
            )
        )
    payload = {
        "schema_version": "irodori-v4-manual-acceptance-override/v1",
        "status": "COMPLETE",
        "source_final_verification": file_binding(source),
        "approval_basis": "user_auditory_judgement",
        "overrides": rows,
        "deployment_performed": False,
        "active_voice_bank_unchanged": True,
    }
    write_json_exclusive(output, payload)
    return payload
```

`build_override_row`は、導出した元acceptance pathの内容がfinal verification内の
`echo.acceptance` summaryと一致し、statusが`NO_PRAGMATIC_CANDIDATE`であることを検証する。
選択embeddingはtraining terminalのcheckpoint inventoryから指定stepをexact lookupし、
`F32[16,768]`かつfiniteであること、training/evaluation terminalが成功していることを検証して、
全pathとSHA-256を出力する。final verificationにはacceptance decision pathが入っていないため、
存在しない`echo.acceptance.decision_path`を参照してはならない。

- [ ] **Step 4: 正常系、hash mismatch、shape mismatchのtestを追加する**

```python
def test_builds_three_hash_bound_manual_acceptances(valid_final: Path, tmp_path: Path) -> None:
    result = build_manual_override(valid_final, tmp_path / "override.json")
    assert result["status"] == "COMPLETE"
    assert [row["model_id"] for row in result["overrides"]] == list(OVERRIDES)
    assert all(row["operator_status"] == "MANUAL_ACCEPTED" for row in result["overrides"])
    assert all(row["automated_status"] == "NO_PRAGMATIC_CANDIDATE" for row in result["overrides"])
    assert all(row["embedding"]["shape"] == [16, 768] for row in result["overrides"])


def test_rejects_changed_bound_decision(valid_final: Path, tmp_path: Path) -> None:
    mutate_bound_acceptance(valid_final)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_manual_override(valid_final, tmp_path / "override.json")
```

- [ ] **Step 5: override testsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_manual_override.py
```

Expected: all tests pass。

## Task 2: duration-residual manifest builderをTDDで作る

**Files:**

- Create: `/private/tmp/irodori-v4-oop70-duration-v001/test_duration_residual_manifest.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/duration_residual_manifest.py`

- [ ] **Step 1: OLSとbin選択の失敗testを書く**

```python
import numpy as np
import pytest

from duration_residual_manifest import fit_residuals, select_balanced_rows


def test_ols_removes_duration_and_text_effect() -> None:
    rows = make_rows(count=1000, duration_effect=0.08, text_effect=0.03)
    result = fit_residuals(rows)
    assert result.design_columns == ["intercept", "log1p_duration", "log1p_text_length"]
    assert abs(np.corrcoef(result.residuals, result.log_durations)[0, 1]) < 1e-10
    assert abs(np.corrcoef(result.residuals, result.log_text_lengths)[0, 1]) < 1e-10


def test_selects_exactly_160_from_each_duration_bin() -> None:
    rows = make_rows(count=1000, minimum_similarity=0.75)
    selected = select_balanced_rows(rows, per_bin=160, floor=0.72)
    assert len(selected.rows) == 800
    assert selected.bin_counts == {"0": 160, "1": 160, "2": 160, "3": 160, "4": 160}
    assert len({row["source_id"] for row in selected.rows}) == 800


def test_fails_when_one_bin_has_fewer_than_160_eligible_rows() -> None:
    rows = make_rows(count=1000, low_similarity_bin=3)
    with pytest.raises(ValueError, match="duration bin 3 has fewer than 160 eligible rows"):
        select_balanced_rows(rows, per_bin=160, floor=0.72)
```

- [ ] **Step 2: testがRedになることを確認する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_duration_residual_manifest.py
```

Expected: import failure。

- [ ] **Step 3: 決定的OLSと五分位選択を実装する**

```python
def fit_residuals(rows: list[dict[str, object]]) -> ResidualFit:
    durations = np.asarray([float(row["duration_seconds"]) for row in rows], dtype=np.float64)
    text_lengths = np.asarray([int(row["text_length"]) for row in rows], dtype=np.float64)
    similarities = np.asarray([float(row["speaker_similarity"]) for row in rows], dtype=np.float64)
    if not np.isfinite(durations).all() or not np.isfinite(similarities).all():
        raise ValueError("audit rows contain non-finite values")
    if (durations <= 0).any() or (text_lengths < 0).any():
        raise ValueError("duration/text length contract mismatch")
    log_durations = np.log1p(durations)
    log_text_lengths = np.log1p(text_lengths)
    design = np.column_stack((np.ones(len(rows)), log_durations, log_text_lengths))
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, similarities, rcond=None)
    if rank != 3:
        raise ValueError("OLS design matrix is rank deficient")
    predicted = design @ coefficients
    residuals = similarities - predicted
    boundaries = np.quantile(durations, [0.2, 0.4, 0.6, 0.8], method="linear")
    bins = np.searchsorted(boundaries, durations, side="right")
    return ResidualFit(coefficients, singular_values, predicted, residuals, bins, boundaries)


def select_balanced_rows(
    rows: list[dict[str, object]], *, per_bin: int = 160, floor: float = 0.72
) -> BalancedSelection:
    fit = fit_residuals(rows)
    ranked: dict[int, list[tuple[float, str, dict[str, object]]]] = {index: [] for index in range(5)}
    for row, residual, bin_index in zip(rows, fit.residuals, fit.bins, strict=True):
        if float(row["speaker_similarity"]) >= floor:
            ranked[int(bin_index)].append((-float(residual), str(row["source_id"]), row))
    chosen = []
    for bin_index in range(5):
        candidates = sorted(ranked[bin_index])
        if len(candidates) < per_bin:
            raise ValueError(f"duration bin {bin_index} has fewer than {per_bin} eligible rows")
        chosen.extend(item[2] for item in candidates[:per_bin])
    return BalancedSelection(rows=chosen, fit=fit, bin_counts={str(i): per_bin for i in range(5)})
```

- [ ] **Step 4: manifest join、latent rebase、quality preflight testを書く**

```python
def test_build_manifest_preserves_source_order_and_rebases_latents(tmp_path: Path) -> None:
    result = build_manifest(audit_path, source_manifest, tmp_path / "output")
    rows = read_jsonl(result.manifest_path)
    assert len(rows) == 800
    assert [row["source_id"] for row in rows] == sorted_by_source_manifest_order(rows)
    assert all(Path(row["latent_path"]).is_absolute() for row in rows)
    assert all(Path(row["latent_path"]).is_file() for row in rows)


def test_rejects_selection_below_source_mean(tmp_path: Path) -> None:
    lower_selected_similarity(audit_path)
    with pytest.raises(ValueError, match="selected similarity mean is below source mean"):
        build_manifest(audit_path, source_manifest, tmp_path / "output")
```

- [ ] **Step 5: create-only manifest outputを実装する**

`build_manifest`はaudit rowsを`source_id`でmanifest rowsへjoinし、選択800件を元manifest index順に
並べる。`latent_path`は`source_manifest.parent / latent_path`から解決したabsolute pathへ置換する。
出力は次の3ファイルとする。

```text
duration-residual-manifest-v001/
  clean-manifest.jsonl
  selection.json
  provenance.json
```

`provenance.json`へ元manifest、audit、builder、出力manifestのSHA-256、OLS係数、五分位境界、
各bin件数、source/selected similarity平均、selected residual平均を保存する。

- [ ] **Step 6: manifest testsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_duration_residual_manifest.py
```

Expected: all tests pass。

- [ ] **Step 7: 独立verifierの失敗testを書く**

`test_verify_duration_residual_manifest.py`は正常なbuilder出力を別process相当で読み直す正常系と、
selected source ID、順序、OLS係数、latent hashをそれぞれ1件変えた失敗系を書く。verifierは
`duration_residual_manifest`をimportせず、共通化による同一バグの見逃しを避ける。

- [ ] **Step 8: 独立verifierを実装する**

CLIは`AUDIT_JSONL SOURCE_MANIFEST MANIFEST_ROOT OUTPUT_JSON`の4引数とする。2209 audit rowsから
OLS、五分位、floor、各bin上位160件を再計算し、800件manifestのsource ID、元順序、全latent、
`selection.json`、`provenance.json`のpath/hashを照合する。`OUTPUT_JSON`はcreate-onlyで、成功時だけ
`status: PASS`を保存する。

- [ ] **Step 9: builderと独立verifierのtestsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_duration_residual_manifest.py \
  /private/tmp/irodori-v4-oop70-duration-v001/test_verify_duration_residual_manifest.py
```

Expected: all tests pass。

## Task 3: scratch preparerをTDDで作る

**Files:**

- Create: `/private/tmp/irodori-v4-oop70-duration-v001/test_prepare_oop70_scratch.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/prepare_oop70_scratch.py`

- [ ] **Step 1: config差分を固定するtestを書く**

```python
def test_only_manifest_and_output_change_from_phase1_config(tmp_path: Path) -> None:
    result = prepare_scratch(
        baseline_config=baseline_config,
        manifest=manifest,
        output_root=tmp_path / "scratch",
    )
    baseline = yaml.safe_load(baseline_config.read_text())
    actual = yaml.safe_load(result.config.read_text())
    expected = copy.deepcopy(baseline)
    expected["train"]["manifest_path"] = str(manifest)
    expected["train"]["output_dir"] = str(tmp_path / "scratch" / "training")
    assert actual == expected
    assert actual["train"]["seed"] == 2
    assert actual["train"]["max_steps"] == 3000
    assert actual["train"]["rf_loss_mode"] == "utterance_mean"
    assert actual["train"]["speaker_inversion_init_embedding"] is None
```

- [ ] **Step 2: testがRedになることを確認する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_prepare_oop70_scratch.py
```

Expected: import failure。

- [ ] **Step 3: exclusive config/setup作成を実装する**

```python
ALLOWED_CHANGED_KEYS = {"train.manifest_path", "train.output_dir"}


def prepare_scratch(*, baseline_config: Path, manifest: Path, output_root: Path) -> ScratchSetup:
    refuse_existing(output_root)
    baseline_setup_path = baseline_config.parent / "setup-evidence.json"
    baseline_setup = read_json(baseline_setup_path)
    validate_phase1_setup(baseline_setup, baseline_config)
    baseline = yaml.safe_load(baseline_config.read_text(encoding="utf-8"))
    validate_phase1_contract(baseline)
    config = copy.deepcopy(baseline)
    config["train"]["manifest_path"] = str(manifest)
    config["train"]["output_dir"] = str(output_root / "training")
    assert_only_changed(baseline, config, ALLOWED_CHANGED_KEYS)
    output_root.mkdir(parents=True, exist_ok=False)
    write_yaml_exclusive(output_root / "config.yaml", config)
    setup = {
        "schema_version": "irodori-v4-multi-speaker-training-setup/v1",
        "state": "prepared",
        "model_id": "oop70_osananajimi_no_iru_kurashi_sp_7195504dbb",
        "dataset_id": "oop70_osananajimi_no_iru_kurashi_sp_7195504dbb",
        "config": file_binding(output_root / "config.yaml"),
        "manifest": {**file_binding(manifest), "rows": 800},
        "reference_wavs": baseline_setup["reference_wavs"],
        "source_evaluation_manifest": baseline_setup["source_evaluation_manifest"],
        "template": baseline_setup["template"],
        "source_baseline": {
            "setup": file_binding(baseline_setup_path),
            "config": file_binding(baseline_config),
        },
        "training_contract": extract_training_contract(config["train"]),
        "deployment_performed": False,
        "active_voice_bank_unchanged": True,
    }
    write_json_exclusive(output_root / "setup-evidence.json", setup)
    return ScratchSetup(output_root / "config.yaml", output_root / "setup-evidence.json")
```

同梱する既存`run_multispeaker_training.py`との互換性を保つため、setup schemaと
`training_contract`の部分集合は既存Phase 1 contractを維持する。追加provenanceは
`source_baseline`に分離し、既存wrapperが読むキーの意味を変えない。

- [ ] **Step 4: config testsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_prepare_oop70_scratch.py
```

Expected: all tests pass。

- [ ] **Step 5: 条件付きecho setupのtestを追加する**

```python
def test_echo_changes_only_the_bounded_continuation_fields(tmp_path: Path) -> None:
    result = prepare_echo(
        scratch_config=scratch_config,
        manifest=manifest,
        init_embedding=init_embedding,
        output_root=tmp_path / "echo",
    )
    scratch = yaml.safe_load(scratch_config.read_text())
    actual = yaml.safe_load(result.config.read_text())
    expected = copy.deepcopy(scratch)
    expected["train"].update(
        {
            "learning_rate": 0.0001,
            "max_steps": 1500,
            "output_dir": str(tmp_path / "echo" / "training"),
            "rf_loss_mode": "echo",
            "speaker_inversion_init_embedding": str(init_embedding),
        }
    )
    assert actual == expected
```

`prepare_echo`はscratchの`improvement-decision-v003.json`が`RUN_ECHO`であり、そこにbindingされた
best exact checkpointと引数のembeddingがpath/hash/stepまで一致するときだけrootを作る。setupは
schema `irodori-v4-multi-speaker-refinement-training-setup/v1`で、manifest 800件、LR `0.0001`、
seed 2、最大1500 steps、echo loss、warm start、oop70 reference、v3 evaluation manifest、v4 upstreamを
記録する。

- [ ] **Step 6: scratch/echo preparer testsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_prepare_oop70_scratch.py
```

Expected: all tests pass。

## Task 4: 改善分岐をTDDで作る

**Files:**

- Create: `/private/tmp/irodori-v4-oop70-duration-v001/test_decide_oop70_branch.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/decide_oop70_branch.py`

- [ ] **Step 1: 3分岐のtestを書く**

```python
from decide_oop70_branch import decide_branch


def test_accepts_when_pragmatic_candidate_exists() -> None:
    assert decide_branch(acceptance_status="PRAGMATICALLY_ELIGIBLE", best_mean=0.770).action == "ACCEPT"


def test_runs_echo_at_exact_improvement_boundary() -> None:
    assert decide_branch(
        acceptance_status="NO_PRAGMATIC_CANDIDATE", best_mean=0.73898819
    ).action == "RUN_ECHO"


def test_stops_below_improvement_boundary() -> None:
    assert decide_branch(
        acceptance_status="NO_PRAGMATIC_CANDIDATE", best_mean=0.73898818
    ).action == "STOP_NO_IMPROVEMENT"
```

- [ ] **Step 2: testがRedになることを確認する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_decide_oop70_branch.py
```

Expected: import failure。

- [ ] **Step 3: 分岐とbest exact checkpoint bindingを実装する**

```python
OLD_BEST_MEAN = 0.71898819
MIN_IMPROVEMENT = 0.02
ECHO_BOUNDARY = OLD_BEST_MEAN + MIN_IMPROVEMENT


def decide_branch(*, acceptance_status: str, best_mean: float) -> BranchDecision:
    if acceptance_status in {"STRICTLY_ELIGIBLE", "PRAGMATICALLY_ELIGIBLE"}:
        return BranchDecision("ACCEPT", best_mean, ECHO_BOUNDARY)
    if acceptance_status != "NO_PRAGMATIC_CANDIDATE":
        raise ValueError(f"unexpected scratch acceptance status: {acceptance_status}")
    action = "RUN_ECHO" if best_mean >= ECHO_BOUNDARY else "STOP_NO_IMPROVEMENT"
    return BranchDecision(action, best_mean, ECHO_BOUNDARY)
```

CLIはscratch acceptance、candidate decision、training terminalを読み、best checkpoint embeddingの
path/hash/shapeと評価成果物をbindingした`improvement-decision-v003.json`をexclusive保存する。

- [ ] **Step 4: 分岐testsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_decide_oop70_branch.py
```

Expected: all tests pass。

## Task 5: queueとdetached launcherをTDDで作る

**Files:**

- Create: `/private/tmp/irodori-v4-oop70-duration-v001/test_run_oop70_queue.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/run_oop70_queue.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/run_oop70_echo_training.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/launch_oop70_queue_detached.py`

- [ ] **Step 1: queue状態遷移testを書く**

```python
def test_queue_runs_scratch_and_skips_echo_after_acceptance(fake_tools: FakeTools) -> None:
    fake_tools.acceptance_status = "PRAGMATICALLY_ELIGIBLE"
    result = run_queue(fake_tools.context())
    assert result.status == "COMPLETE"
    assert result.stages == [
        "scratch_training",
        "scratch_evaluation_prepare",
        "scratch_evaluation",
        "scratch_candidate_decision",
        "scratch_acceptance",
        "improvement_decision",
    ]
    assert result.echo_state == "SKIPPED_SCRATCH_ACCEPTED"


def test_queue_runs_echo_only_after_improvement(fake_tools: FakeTools) -> None:
    fake_tools.acceptance_status = "NO_PRAGMATIC_CANDIDATE"
    fake_tools.best_mean = 0.74
    result = run_queue(fake_tools.context())
    assert result.echo_state == "EVALUATED"


def test_queue_stops_without_echo_below_boundary(fake_tools: FakeTools) -> None:
    fake_tools.acceptance_status = "NO_PRAGMATIC_CANDIDATE"
    fake_tools.best_mean = 0.738
    result = run_queue(fake_tools.context())
    assert result.echo_state == "SKIPPED_NO_IMPROVEMENT"


def test_echo_wrapper_rejects_unbound_init_embedding(echo_fixture: EchoFixture) -> None:
    echo_fixture.replace_init_embedding()
    with pytest.raises(ValueError, match="init embedding SHA-256 mismatch"):
        load_echo_contract(echo_fixture.context())
```

- [ ] **Step 2: testがRedになることを確認する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_run_oop70_queue.py
```

Expected: import failure。

- [ ] **Step 3: 検証済みsupervisorを呼ぶ直列queueを実装する**

queueは各stageを`subprocess.run(..., shell=False)`で実行し、各command、return code、log path/hashを
`queue-status.jsonl`へ追記する。使用する既存toolは起動時にpath/hashを固定する。

```python
def run_queue(context: QueueContext) -> QueueTerminal:
    write_start_evidence(context)
    run_stage("scratch_training", context.training_command())
    run_stage("scratch_evaluation_prepare", context.evaluation_prepare_command())
    run_stage("scratch_evaluation", context.evaluation_command())
    run_stage(
        "scratch_candidate_decision",
        context.candidate_decision_command(),
        allowed_exit_codes={0, 1},
    )
    run_stage(
        "scratch_acceptance", context.acceptance_command(), allowed_exit_codes={0, 1}
    )
    run_stage("improvement_decision", context.branch_command())
    branch = read_branch_decision(context.branch_decision)
    if branch["action"] == "RUN_ECHO":
        run_stage("echo_prepare", context.echo_prepare_command())
        run_stage("echo_training", context.echo_training_command())
        run_stage("echo_evaluation_prepare", context.echo_evaluation_prepare_command())
        run_stage("echo_evaluation", context.echo_evaluation_command())
        run_stage(
            "echo_candidate_decision",
            context.echo_candidate_decision_command(),
            allowed_exit_codes={0, 1},
        )
        run_stage(
            "echo_acceptance",
            context.echo_acceptance_command(),
            allowed_exit_codes={0, 1},
        )
        echo_state = "EVALUATED"
    elif branch["action"] == "ACCEPT":
        echo_state = "SKIPPED_SCRATCH_ACCEPTED"
    else:
        echo_state = "SKIPPED_NO_IMPROVEMENT"
    return write_terminal_evidence(context, echo_state=echo_state, status="COMPLETE")
```

`decide_irodori_v4_si_candidate.py`と`select_pragmatic_acceptance.py`は不合格を有効なdecision fileと
exit code 1で表現するため、この2 stageだけ0/1を許可し、必須decisionのschema/path/hashを検証する。
それ以外はexit code 0のみを成功とする。成功済みstageはterminal/setup/input hashが一致するときだけ
skipする。FAIL rootは再利用せず、新しいversioned rootを要求する。

scratch trainingは既存`run_multispeaker_training.py`を用いる。echo trainingは
`run_oop70_echo_training.py`が既存`run_token_training_supervisor.py`をSHA-256検証後にimportし、
`EXPECTED_MANIFEST_SHA256`、warm-start exact embedding、LR `0.0001`、1500 steps、echo lossだけを
setup evidenceから注入する。scratch evaluationは既存`run_multispeaker_evaluation.py`、echo evaluationは
既存`run_phase3_evaluation.py`を用い、いずれもoop70 reference manifestをhash検証してからbase evaluation
supervisorの固定referenceを差し替える。

- [ ] **Step 4: launcherのownership/preflight testを書く**

```python
def test_launcher_refuses_existing_root(tmp_path: Path) -> None:
    root = tmp_path / "queue"
    root.mkdir()
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        launch(queue_root=root, context=fake_context())


def test_launcher_binds_directory_tree_for_ecapa(tmp_path: Path) -> None:
    reservation = launch_preflight(fake_context(ecapa_source=ecapa_tree(tmp_path)))
    assert reservation["bindings"]["ecapa_source"]["file_count"] == 5
    assert len(reservation["bindings"]["ecapa_source"]["tree_sha256"]) == 64
```

- [ ] **Step 5: detached launcherを実装する**

launcherは競合train/queue/service process、GPU free 10,500 MiB未満、v4 commit mismatch、dirty tracked
worktree、base/tokenizer/script hash mismatch、既存run rootでfail-closed停止する。起動後30秒以内に
start evidenceまたはterminal evidenceを要求し、parent handoffをexclusive保存する。

- [ ] **Step 6: queue testsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_run_oop70_queue.py
```

Expected: all tests pass。

## Task 6: operator disposition verifierをTDDで作る

**Files:**

- Create: `/private/tmp/irodori-v4-oop70-duration-v001/test_verify_operator_disposition.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/verify_operator_disposition.py`

- [ ] **Step 1: 12話者inventoryとstatus分離testを書く**

```python
def test_operator_disposition_keeps_automated_and_manual_status_separate(
    complete_fixture: Fixture,
) -> None:
    result = verify_operator_disposition(complete_fixture.context())
    assert result["model_count"] == 12
    assert result["automated_accepted_count"] == 8
    assert result["manual_accepted_count"] == 3
    assert result["models_by_id"]["miu"]["automated_status"] == "NO_PRAGMATIC_CANDIDATE"
    assert result["models_by_id"]["miu"]["operator_status"] == "MANUAL_ACCEPTED"


def test_rejects_changed_original_final_verification(complete_fixture: Fixture) -> None:
    complete_fixture.mutate_original_final()
    with pytest.raises(ValueError, match="source final verification SHA-256 mismatch"):
        verify_operator_disposition(complete_fixture.context())
```

- [ ] **Step 2: testがRedになることを確認する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_verify_operator_disposition.py
```

Expected: import failure。

- [ ] **Step 3: final verifierを実装する**

```python
def classify_oop70(final_acceptance_status: str) -> str:
    if final_acceptance_status in {"STRICTLY_ELIGIBLE", "PRAGMATICALLY_ELIGIBLE"}:
        return "AUTOMATED_ACCEPTED"
    if final_acceptance_status == "NO_PRAGMATIC_CANDIDATE":
        return "REJECTED_AFTER_DURATION_CORRECTED_RETRAINING"
    raise ValueError(f"unexpected oop70 final status: {final_acceptance_status}")


def verify_operator_disposition(context: VerificationContext) -> dict[str, object]:
    original = verify_original_final(context.original_final)
    override = verify_manual_override(context.manual_override)
    manifest = verify_residual_manifest(context.manifest_root)
    queue = verify_queue(context.queue_root)
    oop70 = verify_oop70_stage(context)
    models = merge_statuses(original, override, oop70)
    if len(models) != 12 or len({row["model_id"] for row in models}) != 12:
        raise ValueError("operator disposition must contain exactly 12 unique models")
    runtime = verify_runtime_idle()
    v3 = verify_v3_and_voice_bank_unchanged(context.v3_baseline)
    automated_count = sum(row["operator_status"] == "AUTOMATED_ACCEPTED" for row in models)
    manual_count = sum(row["operator_status"] == "MANUAL_ACCEPTED" for row in models)
    rejected_count = sum(
        row["operator_status"] == "REJECTED_AFTER_DURATION_CORRECTED_RETRAINING"
        for row in models
    )
    if (automated_count, manual_count, rejected_count) not in {(9, 3, 0), (8, 3, 1)}:
        raise ValueError("operator disposition count contract mismatch")
    return {
        "schema_version": "irodori-v4-operator-disposition-verification/v1",
        "status": "PASS",
        "model_count": 12,
        "automated_accepted_count": automated_count,
        "manual_accepted_count": manual_count,
        "rejected_count": rejected_count,
        "models": models,
        "manifest": manifest,
        "queue": queue,
        "runtime": runtime,
        "v3_runtime": v3,
        "deployment_performed": False,
        "active_voice_bank_unchanged": True,
    }
```

oop70が自動合格した場合はautomated 9 + manual 3、不合格ならautomated 8 + manual 3 + rejected 1を
集計する。元final verificationの8自動合格は変えず、oop70の新規自動判定だけを加算する。

- [ ] **Step 4: verifier testsをGreenにする**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001/test_verify_operator_disposition.py
```

Expected: all tests pass。

## Task 7: runtime bundleを検証・転送する

**Files:**

- Create: `/private/tmp/irodori-v4-oop70-duration-v001/test_publish_runtime_bundle.py`
- Create: `/private/tmp/irodori-v4-oop70-duration-v001/publish_runtime_bundle.py`
- Create remotely: `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007.uploading`
- Rename remotely after verification: `C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007`

`v001.uploading`は8192-byte payloadがWindowsのPowerShell command-line上限を超えて空rootのまま
転送に失敗した証跡として保持する。`v002.uploading`は全tool転送後、最終provenance JSONを1引数で
渡したため同じ上限で停止した。どちらも再利用せず、4096-byte chunkで最終provenanceも転送する
`v003`へ発行した。その後、実final verificationではcandidate decision要約が本文全体ではなく
`status`/`best_failed`/`checkpoint_results`と`binding_count`の射影であることが判明したため、本文hashと
binding件数を個別検証する回帰修正を加え、`v003`を再利用せず`v004`へ発行した。実データではさらに
`best_failed`がcandidate本文の独立fieldではなく`checkpoint_results`からの導出値だったため、選択stepの
元行と全field一致を検証する回帰修正を加え、`v004`を再利用せず`v005`へ発行した。最初の
`scratch-v001/queue-v001`は既存trainerが要求する厳密なbindingから余分な`size` fieldを含んだため、
学習process開始前にFAILで閉じた。失敗rootを再利用せず、既存trainerと完全一致するbindingへ修正した
`scratch-v002/queue-v002`とtool bundle `v006`を使用したが、queue worker登録前に親がhandoffを返し、
Windows OpenSSH job終了時にworkerも巻き取られた。学習step/checkpointは未作成である。失敗rootを
再利用せず、実績あるdetached launcherと同じ`CREATE_BREAKAWAY_FROM_JOB`を含むflagsへ修正した
`scratch-v003/queue-v003`とtool bundle `v007`を使用した。scratchは3000 step、評価は140/140を完走したが、
集合用acceptance toolが26 evaluation setを要求したため`queue-v003`は評価後にFAILで閉じた。既存scratch
証跡を再利用するpost-scratch経路と1話者adapterをTDDで追加し、`v008/queue-v004`で再開したところ、
candidate decisionとraw row再計算におけるtone failureの扱いの差をfail-closedで検出した。raw rowsを
正本、candidate decisionをadvisoryなhash-bound sourceと明記する回帰修正を加え、失敗rootを再利用せず
`v009/queue-v005`で`COMPLETE`した。

- [ ] **Step 1: publisherの失敗testを書く**

subprocess runnerをfake化し、final/staging rootの既存時は転送前に停止すること、local fileを8192-byte
base64 chunkに分割すること、既存remote toolを上表のsource path/hashからbyte copyすること、1件でも
hash/compile mismatchならrenameしないことをtestする。

- [ ] **Step 2: create-only publisherを実装する**

`publish_runtime_bundle.py`は`.env`を直接解釈せず、全remote操作を
`just remote-python`のargv配列で実行する。新規runtime fileはbase64を8192-byte chunkで
`.uploading`配下へ送り、各chunk index/totalを検証してからexclusive decodeする。既存toolは
Pinned remote inputs表のsourceからremote側で`xb` copyする。全fileのSHA-256、`py_compile`、file inventoryを
`bundle-manifest.json`へ保存後、final root不在を再確認してdirectory renameする。

- [ ] **Step 3: local full test gateを実行する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001
uv run ruff check --isolated --select E9,F63,F7,F82 /private/tmp/irodori-v4-oop70-duration-v001
rg --files -g '*.py' /private/tmp/irodori-v4-oop70-duration-v001 | xargs uv run python -m py_compile
```

Expected: all tests pass、Ruff `All checks passed!`、compile exit 0。

- [ ] **Step 4: bundleをpublishする**

Run:

```bash
uv run python /private/tmp/irodori-v4-oop70-duration-v001/publish_runtime_bundle.py \
  --repo-root /Users/sankenbisha/Dev/irodori-tts-infra \
  --local-root /private/tmp/irodori-v4-oop70-duration-v001 \
  --staging-root 'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007.uploading' \
  --final-root 'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007'
```

Expected: 全file hash一致、remote `py_compile` exit 0、staging root消滅、final tool root存在。queue start
evidenceは新規fileと再利用toolをすべて個別bindingとして列挙する。

## Task 8: overrideとresidual manifestを作成する

- [ ] **Step 1: remote operator rootをexclusive予約する**

Run:

```bash
just remote-python -c 'from pathlib import Path; import sys; root=Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=False); (root / "oop70").mkdir(exist_ok=False)' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001'
```

Expected: operator rootと直下の`oop70`だけが新規作成される。

- [ ] **Step 2: manual overrideをexclusive作成する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007\manual_override.py' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\final-verification-v001.json' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\manual-acceptance-override-v001.json'
```

Expected: `COMPLETE`、override count 3、selected steps 750/750/1500、deployment false。

- [ ] **Step 3: residual manifestをexclusive作成する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007\duration_residual_manifest.py' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\models\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\refinement\central-q50-audit-v001\training-ecapa-results.jsonl' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\manifests\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\clean-manifest.jsonl' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-manifest-v001'
```

Expected: row count 800、5 bins × 160、floor violation 0、missing latent 0、selected mean >=
`0.7829642909888729`、selected residual mean > 0。

- [ ] **Step 4: 別processでmanifestを再計算検証する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007\verify_duration_residual_manifest.py' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\models\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\refinement\central-q50-audit-v001\training-ecapa-results.jsonl' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\manifests\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\clean-manifest.jsonl' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-manifest-v001' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\manifest-verification-v001.json'
```

Expected: status `PASS`、OLS係数、境界、source ID、順序、latent、全hash一致。

## Task 9: oop70 scratch queueを起動・監視する

- [ ] **Step 1: scratch setupをexclusive作成する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007\prepare_oop70_scratch.py' scratch \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\models\oop70_osananajimi_no_iru_kurashi_sp_7195504dbb\scratch\config.yaml' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-manifest-v001\clean-manifest.jsonl' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-scratch-v003'
```

Expected: setup schemaは既存multispeaker trainer互換、config差分はmanifest/outputだけ、max steps 3000、
seed 2、LR `0.01`、warm-start null。

- [ ] **Step 2: detached queueのpreflightを実行する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007\launch_oop70_queue_detached.py' preflight \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-queue-v003' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007'
```

Expected: final verification/input/tool hash、v4 commitとtracked clean、base/tokenizer、ECAPA/Whisper tree、
manifest verification PASS、GPU free >= 10,500 MiB、training/queue/service process 0。preflightはqueue rootを
まだ作らない。

- [ ] **Step 3: detached queueをfail-closed起動する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007\launch_oop70_queue_detached.py' launch \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-queue-v003' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007'
```

Expected: queue rootをexclusive予約し、reservation/start/parent-handoff evidenceを作る。30秒以内にstartまたは
terminal evidenceがなければlaunchをFAILで閉じる。配備は行わない。

- [ ] **Step 4: queueを50秒以内の間隔で監視する**

Run on each monitor turn:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007\launch_oop70_queue_detached.py' status \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-queue-v003' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v007'
```

3000-step scratchはstep、GPU、log size、terminalを確認する。続くscratch評価は5 checkpoints × 28 cases、
140 SUCCESSを要求する。自動合格ならechoを作らず終了する。不合格かつbest mean >= `0.73898819`の
場合だけ、同じ800件manifestとbest exact scratch checkpointから1500-step echoを作成・評価する。
それ未満なら`SKIPPED_NO_IMPROVEMENT`で閉じる。OOM、非finite、競合process、停止はそのrootをFAILで
閉じ、同じrootを再利用しない。

## Task 10: 最終verificationと文書化を完了する

- [x] **Step 1: operator verifierをfresh実行する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v009\verify_operator_disposition.py' verify \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\final-verification-v001.json' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\manual-acceptance-override-v001.json' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-manifest-v001' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\oop70\duration-residual-queue-v005' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_multi_v001\v3-runtime-baseline-v001.json' \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\operator-disposition-v001.json'
```

Expected: status PASS、model count 12、manual accepted 3。oop70合格時はautomated 9/rejected 0、未達時は
automated 8/rejected 1。v3/voice bank unchanged、runtime idle、deployment false。

- [x] **Step 2: 別processで正本と全bindingを再監査する**

Run:

```bash
just remote-python \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\_tools_oop70_duration_corrected_v009\verify_operator_disposition.py' audit \
  'C:\Users\takut\Dev\ooppeenn_training\600m_retraining_20260731\v4_compatibility\v4_speaker_inversion_operator_v001\operator-disposition-v001.json'
```

Expected: fresh `PASS`をstdoutへ出し、全file/directory binding、model/status count、GPU/process idleを再計算。
正本のSHA-256を記録し、既存fileは変更しない。

- [x] **Step 3: 計画・設計文書へ実行結果を追記する**

Modify:

- `docs/superpowers/plans/2026-08-04-irodori-v4-multi-speaker-inversion.md`
- `docs/superpowers/specs/2026-08-04-irodori-v4-oop70-duration-corrected-retraining-design.md`
- `docs/superpowers/plans/2026-08-04-irodori-v4-oop70-duration-corrected-retraining.md`

manual override 3話者のstep/hash、OLS係数とbin境界、各bin件数と平均、oop70各checkpointのhard 16件、
echo実行またはskip理由、oop70最終判定、operator disposition path/hash、v3/voice bank不変、runtime idleを
実測値で記録する。

- [x] **Step 4: completion gateを実行する**

Run:

```bash
uv run pytest -q /private/tmp/irodori-v4-oop70-duration-v001
uv run ruff check --isolated --select E9,F63,F7,F82 /private/tmp/irodori-v4-oop70-duration-v001
rg --files -g '*.py' /private/tmp/irodori-v4-oop70-duration-v001 | xargs uv run python -m py_compile
```

Expected: tests 0 failures、Ruff 0 errors、compile exit 0。remote verifierのfresh PASSと合わせてのみ完了を
宣言する。コミットはユーザーから明示依頼がないため作成しない。

## 実行記録（2026-08-04）

- manual override: 3話者、SHA-256 `c1710ecd22a99ad402f2634805d5128a3d79e5203ef1f59c4283c5fb216fbde5`
- residual manifest: 800件、各bin 160件、manifest SHA-256
  `eff0bebf5884b0c65d65998560c27352ed975edb838ccd14163d133279752562`、独立verify `PASS`
- scratch: `duration-residual-scratch-v003`、3000/3000 step、terminal `PASS`、最大GPU 6410 MiB
- evaluation: 5 checkpoint × 28 case、140/140成功。平均はstep順に
  `0.70438040 / 0.71095193 / 0.70733061 / 0.71592508 / 0.71197509`
- branch: 最高0.71592508 < 旧最高0.71898819 < echo閾値0.73898819のため
  `STOP_NO_IMPROVEMENT`、echoは未実施
- final queue: `duration-residual-queue-v005`、`COMPLETE`、
  terminal SHA-256 `8960953d95813428ef997e08f091640cdbefe5afdb2e5d52374a72eaa704e30d`
- operator disposition: 自動8、手動3、再学習後不合格1、fresh verifier/auditとも`PASS`、SHA-256
  `8d36021f1c45fe2a3a9a987cae2e321ac5a7dfdf2db351b2dded52aaa11935a1`
- 最終runtime: GPU free 11,078 MiB、utilization 0%、training/queue process 0。v3 Git、標準設定、
  voice bank不変、配備なし
- completion gate: runtime `60 passed`、Ruff critical rules `PASS`、全Python `py_compile` `PASS`。
  repository全体は`1548 passed / 1 failed / 3 deselected`で、失敗は開始前から同一の
  `test_production_evaluation_scripts_remain_byte_identical`のみ。対象scriptの実SHA
  `947babd074d83b08c2c9a535f9d718cdac17a3cbfe845e430758bc1008818816`に対し、fixture期待SHAが
  `5ca4f4b13e92a4f283f04c22898d2dabcc4f2982ceac4565c25f9fc671a590fb`である既知のdirty-worktree差を
  本作業では変更していない

失敗したtool/queue root（`v001`〜`v008`、`queue-v001`〜`queue-v004`）は上書き・再利用せず、
それぞれのfail-closed証跡として保持した。最終tool bundleは
`_tools_oop70_duration_corrected_v009`、bundle manifest SHA-256
`a37463f95c226abab5aa0f1bb63e75fc9c63d20c71b5c4681e34704959645c13`である。

## 後続採用記録（2026-08-05）

ユーザー指示により、oop70の自動不合格を監査履歴として保持したまま、最高到達checkpointを
operator手動採用した。選択はcandidate decisionの5行から平均類似度を再計算し、step 2500
（平均`0.7159250827723578`、最小`0.688714710818511`）に決定した。embedding SHA-256は
`bfadf21004c16b0a71ed44e0884ab53342e36ed2f8efbbeedbc8073373ccbe9b`、`F32[16,768]`、finiteである。

- tool bundle: `_tools_oop70_duration_corrected_v010`、27 files、manifest SHA-256
  `7b1290123bd8f478cfff56cbbcf8385bc37d37b10dda52cf382db510dedfac65`
- manual adoption: `oop70-manual-acceptance-override-v001.json`、SHA-256
  `5e91c1960a3f74373aa50412e374b5247a2de5f172efda0753ab0136db34acb1`
- latest disposition: `operator-disposition-v002.json`、SHA-256
  `e014e64ad311ca4fc994eb245c272d13e3c65099af4db388cc394022d7299c34`
- latest counts: automated 8、manual 4、rejected 0
- create時audit / 別process audit: `PASS`
- runtime: GPU idle、training/queue process 0、v3/voice bank unchanged、deployment false

runtime contract testは追加3件を含む`63 passed`。元の`operator-disposition-v001.json`と自動判定policyは
変更していない。
