# ruff: noqa: PLR0914, PLR2004, SLF001
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

BUILDER = Path("scripts/derive_600m_speaker_midpoint_diagnostic.py")
GENERATOR = Path("scripts/generate_600m_speaker_midpoint_diagnostic_remote.py")
EVALUATOR = Path("scripts/evaluate_600m_speaker_midpoint_diagnostic.py")
SEARCH_TESTS = Path("tests/scripts/test_600m_speaker_checkpoint_search.py")
MODEL_ID = "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd"
TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
SEEDS = (1234, 5678)
STYLES = ("neutral", "calm")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_embedding(path: Path, values: np.ndarray, *, dtype: str = "F32") -> None:
    payload = values.astype("<f4" if dtype == "F32" else "<f2").tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": dtype,
                "shape": list(values.shape),
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _write_embedding_with_metadata(
    path: Path,
    values: np.ndarray,
    metadata: object,
) -> None:
    payload = values.astype("<f4").tobytes()
    header = json.dumps(
        {
            "__metadata__": metadata,
            "speaker_embedding": {
                "dtype": "F32",
                "shape": list(values.shape),
                "data_offsets": [0, len(payload)],
            },
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _source_manifest(tmp_path: Path) -> Path:
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    source_embedding = tmp_path / "source.speaker.safetensors"
    _write_embedding(source_embedding, np.ones((16, 768), dtype=np.float32))
    checkpoints = [
        {
            "checkpoint_step": step,
            "embedding_path": str(source_embedding),
            "embedding_sha256": _sha(source_embedding),
            "training_config_sha256": "a" * 64,
            "base_checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
            "base_checkpoint_sha256": _sha(base),
            "base_revision": "revision",
            "run_id": "source-run",
        }
        for step in range(250, 3001, 250)
    ]
    path = tmp_path / "source-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "speaker-checkpoint-evaluation-manifest/v1",
                "models": [{"model_id": MODEL_ID, "checkpoints": checkpoints}],
                "text_ids": list(TEXT_IDS),
                "seeds": list(SEEDS),
                "styles": list(STYLES),
                "metrics_provenance": {
                    "reference_wavs_sha256": "b" * 64,
                    "speaker_embedding": {
                        "model_id": "ecapa",
                        "revision": "ecapa-rev",
                        "source_sha256": "c" * 64,
                    },
                    "transcription": {
                        "model_id": "whisper",
                        "revision": "whisper-rev",
                        "source_sha256": "d" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _fixture(
    tmp_path: Path,
) -> tuple[ModuleType, dict[str, Any], np.ndarray, np.ndarray]:
    module = _load(BUILDER, f"midpoint_builder_{tmp_path.name}")
    search_tests = _load(SEARCH_TESTS, f"midpoint_search_fixture_{tmp_path.name}")
    search_builder, parent_source, candidate, config = search_tests._build_fixture(tmp_path)
    parent_payload = json.loads(parent_source.read_text(encoding="utf-8"))
    parent_row = next(
        row for row in parent_payload["models"][0]["checkpoints"] if row["checkpoint_step"] == 1000
    )
    parent = tmp_path / "checkpoint_0001000.speaker.safetensors"
    p = np.ones((16, 768), dtype=np.float32)
    f = np.full((16, 768), 6.0, dtype=np.float32)
    _write_embedding(parent, p)
    _write_embedding(candidate, f)
    parent_row["embedding_path"] = str(parent.resolve())
    parent_row["embedding_sha256"] = _sha(parent)
    parent_source.write_text(json.dumps(parent_payload), encoding="utf-8")
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload["train"]["speaker_inversion_init_embedding"] = str(parent.resolve())
    config_payload["train"]["speaker_inversion_init_embedding_sha256"] = _sha(parent)
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    search_manifest = tmp_path / "candidate-f-search-manifest.json"
    search_tests._build(search_builder, parent_source, candidate, config, search_manifest)
    search_payload = json.loads(search_manifest.read_text(encoding="utf-8"))
    evidence = Path(search_payload["training_run_evidence"]["path"])
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    setup = Path(evidence_payload["setup_evidence"]["path"])
    setup.write_text(
        json.dumps(
            {
                "schema_version": "speaker-quality-search-setup/v1",
                "candidate": {
                    "model_id": MODEL_ID,
                    "init_label": "original1000",
                    "speaker_inversion_init_embedding": str(parent.resolve()),
                    "speaker_inversion_init_embedding_sha256": _sha(parent),
                },
            }
        ),
        encoding="utf-8",
    )
    evidence_payload["setup_evidence"]["sha256"] = _sha(setup)
    evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")
    search_payload["training_run_evidence"]["sha256"] = _sha(evidence)
    search_manifest.write_text(json.dumps(search_payload), encoding="utf-8")
    output_root = tmp_path / "derived"
    output_root.mkdir()
    kwargs = {
        "source_manifest": search_manifest,
        "source_manifest_sha256": _sha(search_manifest),
        "original_embedding": parent,
        "original_embedding_sha256": _sha(parent),
        "candidate_f_embedding": candidate,
        "candidate_f_embedding_sha256": _sha(candidate),
        "model_id": MODEL_ID,
        "output_root": output_root,
        "version": "midpoint-v1",
    }
    return module, kwargs, p, f


def test_builder_derives_exact_cpu_midpoint_without_normalization(tmp_path: Path) -> None:
    module, kwargs, p, f = _fixture(tmp_path)
    manifest = module.derive_midpoint(**kwargs)
    output = kwargs["output_root"] / kwargs["version"]
    values = module.read_embedding(output / "derived-midpoint.speaker.safetensors")

    np.testing.assert_array_equal(values, p * np.float32(0.5) + f * np.float32(0.5))
    assert float(np.linalg.norm(values)) != pytest.approx(1.0)
    assert manifest["schema_version"] == "speaker-derived-diagnostic-manifest/v1"
    assert manifest["diagnostic_kind"] == "derived diagnostic embedding"
    assert manifest["checkpoint_step"] == 0
    assert "training_config_sha256" not in manifest["derived_embedding"]
    derivation = json.loads((output / "derivation.json").read_text())
    assert derivation["alpha"] == pytest.approx(0.5)
    assert derivation["formula"] == "M = 0.5 * P + 0.5 * F"
    assert derivation["normalization"] == "none"


def test_builder_accepts_init_bound_p_when_nested_evaluation_step1000_differs(
    tmp_path: Path,
) -> None:
    module, kwargs, p, f = _fixture(tmp_path)
    search_manifest = kwargs["source_manifest"]
    search_payload = json.loads(search_manifest.read_text(encoding="utf-8"))
    nested_manifest = Path(search_payload["source_evaluation_manifest"]["path"])
    nested_payload = json.loads(nested_manifest.read_text(encoding="utf-8"))
    nested_step1000 = next(
        row for row in nested_payload["models"][0]["checkpoints"] if row["checkpoint_step"] == 1000
    )
    unrelated = tmp_path / "different-quality-run-step1000.speaker.safetensors"
    _write_embedding(unrelated, np.full((16, 768), 9.0, dtype=np.float32))
    nested_step1000["embedding_path"] = str(unrelated.resolve())
    nested_step1000["embedding_sha256"] = _sha(unrelated)
    nested_manifest.write_text(json.dumps(nested_payload), encoding="utf-8")
    search_payload["source_evaluation_manifest"]["sha256"] = _sha(nested_manifest)
    search_manifest.write_text(json.dumps(search_payload), encoding="utf-8")
    kwargs["source_manifest_sha256"] = _sha(search_manifest)

    manifest = module.derive_midpoint(**kwargs)

    values = module.read_embedding(Path(manifest["derived_embedding"]["path"]))
    np.testing.assert_array_equal(values, p * np.float32(0.5) + f * np.float32(0.5))


@pytest.mark.parametrize(
    "which",
    [
        "source_manifest_sha256",
        "original_embedding_sha256",
        "candidate_f_embedding_sha256",
    ],
)
def test_builder_rejects_hash_tampering(tmp_path: Path, which: str) -> None:
    module, kwargs, _p, _f = _fixture(tmp_path)
    kwargs[which] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.derive_midpoint(**kwargs)


@pytest.mark.parametrize("parent", ["original", "candidate_f"])
def test_builder_rejects_self_consistent_parent_outside_verified_lineage(
    tmp_path: Path, parent: str
) -> None:
    module, kwargs, _p, _f = _fixture(tmp_path)
    replacement = tmp_path / f"replacement-{parent}.speaker.safetensors"
    _write_embedding(replacement, np.full((16, 768), 3.0, dtype=np.float32))
    kwargs[f"{parent}_embedding"] = replacement
    kwargs[f"{parent}_embedding_sha256"] = _sha(replacement)
    with pytest.raises(ValueError, match=r"original1000|setup evidence|search manifest checkpoint"):
        module.derive_midpoint(**kwargs)


@pytest.mark.parametrize("failure", ["shape", "dtype", "finite"])
def test_builder_rejects_invalid_parent_embedding(tmp_path: Path, failure: str) -> None:
    module, kwargs, _p, _f = _fixture(tmp_path)
    path = kwargs["candidate_f_embedding"]
    values = np.ones((15, 768) if failure == "shape" else (16, 768), dtype=np.float32)
    if failure == "finite":
        values.flat[0] = np.nan
    _write_embedding(path, values, dtype="F16" if failure == "dtype" else "F32")
    kwargs["candidate_f_embedding_sha256"] = _sha(path)
    with pytest.raises((TypeError, ValueError)):
        module.derive_midpoint(**kwargs)


def test_builder_rejects_symlink_parent_and_existing_output(tmp_path: Path) -> None:
    module, kwargs, _p, _f = _fixture(tmp_path)
    alias = tmp_path / "alias.safetensors"
    alias.symlink_to(kwargs["original_embedding"])
    kwargs["original_embedding"] = alias
    with pytest.raises(ValueError, match=r"symlink|alias|reparse"):
        module.derive_midpoint(**kwargs)

    second = tmp_path / "second"
    second.mkdir()
    kwargs = _fixture(second)[1]
    module.derive_midpoint(**kwargs)
    with pytest.raises(FileExistsError, match="overwrite"):
        module.derive_midpoint(**kwargs)


def test_builder_detects_parent_change_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, kwargs, _p, _f = _fixture(tmp_path)
    original_write = module._write_exclusive
    call_count = 0

    def mutate_after_first_write(path: Path, payload: bytes) -> None:
        nonlocal call_count
        original_write(path, payload)
        call_count += 1
        if call_count == 1:
            kwargs["original_embedding"].write_bytes(b"changed")

    monkeypatch.setattr(module, "_write_exclusive", mutate_after_first_write)
    with pytest.raises(ValueError, match="changed after input snapshot"):
        module.derive_midpoint(**kwargs)


def test_builder_rejects_embedding_with_extra_tensor(tmp_path: Path) -> None:
    module, kwargs, _p, _f = _fixture(tmp_path)
    path = kwargs["candidate_f_embedding"]
    values = np.ones((16, 768), dtype="<f4")
    raw = values.tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": "F32",
                "shape": [16, 768],
                "data_offsets": [0, len(raw)],
            },
            "unexpected": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    with pytest.raises(ValueError, match="exactly one"):
        module.read_embedding(path)


@pytest.mark.parametrize("metadata", [{}, {"format": "pt"}])
def test_builder_accepts_reserved_safetensors_metadata(
    tmp_path: Path, metadata: dict[str, str]
) -> None:
    module = _load(BUILDER, f"midpoint_builder_metadata_{len(metadata)}")
    path = tmp_path / "real-form.speaker.safetensors"
    values = np.arange(16 * 768, dtype=np.float32).reshape(16, 768)
    _write_embedding_with_metadata(path, values, metadata)

    np.testing.assert_array_equal(module.read_embedding(path), values)


@pytest.mark.parametrize("metadata", [None, [], "", {"format": 1}, {"format": None}])
def test_builder_rejects_malformed_reserved_safetensors_metadata(
    tmp_path: Path, metadata: object
) -> None:
    module = _load(BUILDER, f"midpoint_builder_bad_metadata_{type(metadata).__name__}")
    path = tmp_path / "bad-metadata.speaker.safetensors"
    _write_embedding_with_metadata(path, np.ones((16, 768), dtype=np.float32), metadata)

    with pytest.raises((TypeError, ValueError), match="metadata"):
        module.read_embedding(path)


def test_builder_rejects_unknown_safetensors_top_level_key(tmp_path: Path) -> None:
    module = _load(BUILDER, "midpoint_builder_unknown_top_level")
    path = tmp_path / "unknown-key.speaker.safetensors"
    values = np.ones((16, 768), dtype="<f4")
    raw = values.tobytes()
    header = json.dumps(
        {
            "speaker_embedding": {
                "dtype": "F32",
                "shape": [16, 768],
                "data_offsets": [0, len(raw)],
            },
            "unknown": "value",
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)

    with pytest.raises(ValueError, match="exactly one"):
        module.read_embedding(path)


def test_generator_builds_28_step_zero_cases_with_derived_provenance(tmp_path: Path) -> None:
    builder, kwargs, _p, _f = _fixture(tmp_path)
    manifest = builder.derive_midpoint(**kwargs)
    generator = _load(GENERATOR, f"midpoint_generator_{tmp_path.name}")
    plan = generator.load_derived_plan(Path(manifest["manifest_path"]))
    cases = generator.build_derived_cases(plan)
    assert len(cases) == 28
    assert {case.checkpoint.checkpoint_step for case in cases} == {0}

    row = generator.bind_derived_case_provenance(
        {
            "provenance": {"training_config_sha256": "x", "run_id": "x"},
            "checkpoint_step": 0,
        },
        plan=plan,
    )
    assert row["schema_version"] == "speaker-derived-diagnostic-generation-case/v1"
    assert set(row["provenance"]) == {
        "derivation_sha256",
        "original_embedding_sha256",
        "candidate_f_embedding_sha256",
        "base_checkpoint",
        "base_revision",
        "diagnostic_kind",
    }
    assert "training_config_sha256" not in row["provenance"]


@pytest.mark.parametrize(
    ("status", "minimum", "expected"),
    [
        ("ELIGIBLE", 0.753, "STRONG_DIAGNOSTIC"),
        ("ELIGIBLE", 0.752999, "BOUNDARY_DIAGNOSTIC"),
        ("ELIGIBLE", 0.750, "BOUNDARY_DIAGNOSTIC"),
        ("ELIGIBLE", 0.749999, "FAILED_DIAGNOSTIC"),
        ("REJECTED", 0.9, "FAILED_DIAGNOSTIC"),
    ],
)
def test_midpoint_predeclared_decision(status: str, minimum: float, expected: str) -> None:
    module = _load(EVALUATOR, f"midpoint_evaluator_{status}_{minimum}")
    assert module.midpoint_decision(status=status, min_similarity=minimum) == expected


def test_generator_rejects_tampered_derivation(tmp_path: Path) -> None:
    builder, kwargs, _p, _f = _fixture(tmp_path)
    manifest = builder.derive_midpoint(**kwargs)
    derivation = Path(manifest["derivation"]["path"])
    derivation.write_text("{}", encoding="utf-8")
    generator = _load(GENERATOR, f"midpoint_generator_tamper_{tmp_path.name}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generator.load_derived_plan(Path(manifest["manifest_path"]))


def test_generator_detects_dependency_change_after_module_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder, kwargs, _p, _f = _fixture(tmp_path)
    manifest = builder.derive_midpoint(**kwargs)
    generator = _load(GENERATOR, f"midpoint_generator_dependency_{tmp_path.name}")
    plan = generator.load_derived_plan(Path(manifest["manifest_path"]))
    original_snapshot = generator._snapshot

    def changed(path: Path) -> tuple[bytes, str]:
        snapshot = cast("tuple[bytes, str]", original_snapshot(path))
        if path.resolve() == GENERATOR.resolve():
            return snapshot[0] + b"changed", "0" * 64
        return snapshot

    monkeypatch.setattr(generator, "_snapshot", changed)
    with pytest.raises(ValueError, match="dependency changed"):
        generator._validate_plan_unchanged(plan)


def test_generator_output_reservation_is_create_only(tmp_path: Path) -> None:
    module = _load(GENERATOR, "midpoint_generator_reservation")
    output = tmp_path / "generation-v1"
    module.reserve_output(output)
    with pytest.raises(FileExistsError, match="overwrite"):
        module.reserve_output(output)


def test_evaluator_validates_step_zero_matrix() -> None:
    module = _load(EVALUATOR, "midpoint_evaluator_matrix")
    identities = [
        (MODEL_ID, 0, text_id, seed, style)
        for text_id in TEXT_IDS
        for seed in SEEDS
        for style in STYLES
    ]
    module.validate_case_matrix(identities, model_id=MODEL_ID)
    bad = list(identities)
    bad[0] = (MODEL_ID, 250, *bad[0][2:])
    with pytest.raises(ValueError, match=r"unexpected|missing"):
        module.validate_case_matrix(bad, model_id=MODEL_ID)


def test_evaluator_detects_dependency_change_after_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(EVALUATOR, "midpoint_evaluator_dependency")
    snapshots = module._snapshot_dependencies()
    original_snapshot = module._snapshot

    def changed(path: Path, *, source: str) -> tuple[bytes, str]:
        snapshot = cast("tuple[bytes, str]", original_snapshot(path, source=source))
        if path.resolve() == EVALUATOR.resolve():
            return snapshot[0] + b"changed", "0" * 64
        return snapshot

    monkeypatch.setattr(module, "_snapshot", changed)
    with pytest.raises(ValueError, match="dependency changed"):
        module._validate_dependencies_unchanged(snapshots)


def test_existing_search_contract_remains_unchanged() -> None:
    search_builder = _load(
        Path("scripts/build_600m_speaker_checkpoint_search_manifest.py"),
        "unchanged_search_builder",
    )
    search_generator = _load(
        Path("scripts/generate_600m_speaker_checkpoint_search_remote.py"),
        "unchanged_search_generator",
    )
    search_evaluator = _load(
        Path("scripts/evaluate_600m_speaker_checkpoint_search.py"),
        "unchanged_search_evaluator",
    )
    assert search_builder.SEARCH_SCHEMA == "speaker-checkpoint-search-manifest/v1"
    assert search_generator.SEARCH_STEP == 250
    assert search_evaluator.SEARCH_STEP == 250
