from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/build_clean_speaker_datasets.py")
SAMPLE_RATE = 48_000
RULE_VERSION = "2026-07-31.v1"
EXPECTED_EXPRESSIVE_ROWS = 2


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_clean_speaker_datasets", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_inventory_hashes_duplicate_audio_once(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"same encoded audio")
    index_path = _write_index(
        tmp_path,
        [
            {"Speaker": "ミウ", "Text": "ひとつめ", "FilePath": audio_path.name},
            {"Speaker": "ミウ", "Text": "ふたつめ", "FilePath": audio_path.name},
        ],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_voice_decoder,
    )

    assert [record.decision for record in records] == ["KEEP", "EXCLUDE_DUPLICATE"]
    assert records[0].audio_sha256 == records[1].audio_sha256
    assert records[1].duplicate_of == records[0].record_id


def test_build_inventory_excludes_same_decoded_pcm_with_different_file_bytes(
    tmp_path: Path,
) -> None:
    module = _load_script()
    first_path = tmp_path / "first.ogg"
    second_path = tmp_path / "second.ogg"
    first_path.write_bytes(b"encoding one")
    second_path.write_bytes(b"encoding two")
    index_path = _write_index(
        tmp_path,
        [
            {"Speaker": "ミウ", "Text": "ひとつめ", "FilePath": first_path.name},
            {"Speaker": "ミウ", "Text": "ふたつめ", "FilePath": second_path.name},
        ],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_voice_decoder,
    )

    assert records[0].audio_sha256 != records[1].audio_sha256
    assert records[0].pcm_sha256 == records[1].pcm_sha256
    assert records[1].decision == "EXCLUDE_DUPLICATE"


def test_build_inventory_records_decode_failure_as_invalid_audio(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "broken.ogg"
    audio_path.write_bytes(b"broken")
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "本文", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_failing_decoder,
    )

    assert records[0].decision == "EXCLUDE_INVALID_AUDIO"
    assert "decode_error:RuntimeError" in records[0].reasons


def test_build_inventory_resolves_file_path_from_explicit_audio_root(tmp_path: Path) -> None:
    module = _load_script()
    index_dir = tmp_path / "indexes"
    audio_root = tmp_path / "extracted"
    speaker_dir = audio_root / "ミウ"
    index_dir.mkdir()
    speaker_dir.mkdir(parents=True)
    audio_path = speaker_dir / "voice.ogg"
    audio_path.write_bytes(b"voice")
    index_path = _write_index(
        index_dir,
        [{"Speaker": "ミウ", "Text": "本文", "FilePath": "ミウ/voice.ogg"}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
        audio_root=audio_root,
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_voice_decoder,
    )

    assert records[0].audio_path == audio_path


def test_build_inventory_keeps_ambiguous_source_mapping_in_review(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"voice")
    index_path = _write_index(
        tmp_path,
        [
            {
                "Speaker": "藍子",
                "Text": "重複caption",
                "FilePath": audio_path.name,
                "MappingDecision": "REVIEW",
                "MappingCandidateMetadataIndices": [10, 11],
            }
        ],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="kasumi",
        output_model_id="kasumi",
        index_json=index_path,
        speaker="藍子",
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_voice_decoder,
    )

    assert records[0].decision == "REVIEW"
    assert "source_mapping_ambiguous" in records[0].reasons


def test_build_inventory_resolves_voice_stem_schema(tmp_path: Path) -> None:
    module = _load_script()
    audio_root = tmp_path / "audio"
    speaker_dir = audio_root / "ama"
    speaker_dir.mkdir(parents=True)
    audio_path = speaker_dir / "fem_ama_00006.ogg"
    audio_path.write_bytes(b"voice")
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ama", "Text": "本文", "Voice": "fem_ama_00006"}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="narrator_toshiue_ama",
        output_model_id="narrator_sayoko",
        index_json=index_path,
        speaker="ama",
        audio_root=audio_root,
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_voice_decoder,
    )

    assert records[0].audio_path == audio_path


def test_build_clean_dataset_keeps_expressive_voice_and_repairs_caption(
    tmp_path: Path,
) -> None:
    module = _load_script()
    breath_path = tmp_path / "breath.ogg"
    censored_path = tmp_path / "censored.ogg"
    breath_path.write_bytes(b"breath")
    censored_path.write_bytes(b"spoken word")
    index_path = _write_index(
        tmp_path,
        [
            {"Speaker": "ミウ", "Text": "はぁ、んっ", "FilePath": breath_path.name},
            {
                "Speaker": "ミウ",
                "Text": "おち◯ちんです",
                "FilePath": censored_path.name,
            },
        ],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )
    rules = (
        module.CaptionRule(
            rule_id="ochinchin-nasal",
            source="おち◯ちん",
            replacement="おちんちん",
        ),
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=rules,
        rule_version=RULE_VERSION,
        decoder=_distinct_voice_decoder,
    )
    result = module.build_clean_dataset(records)

    assert result.summary["total"] == EXPECTED_EXPRESSIVE_ROWS
    assert result.summary["kept"] == EXPECTED_EXPRESSIVE_ROWS
    assert result.summary["keep_recaptioned"] == 1
    assert [row["text"] for row in result.rows] == ["はぁ、んっ", "おちんちんです"]
    assert records[1].decision == "KEEP_RECAPTIONED"
    assert records[1].original_text == "おち◯ちんです"
    assert records[1].caption_rule_id == "ochinchin-nasal"


def test_build_inventory_applies_voice_label_to_low_level_breath(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "quiet.ogg"
    audio_bytes = b"quiet breath"
    audio_path.write_bytes(audio_bytes)
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "はぁ", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )
    labels = {
        hashlib.sha256(audio_bytes).hexdigest(): module.ReviewLabel(
            label="VOICE",
            reviewer="user",
            note="audible breath",
        ),
    }

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256=labels,
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_quiet_decoder,
    )

    assert records[0].decision == "KEEP"
    assert records[0].review_label == "VOICE"
    assert "user_label:VOICE" in records[0].reasons


def test_build_inventory_keeps_tone_label_authoritative_over_caption_review(
    tmp_path: Path,
) -> None:
    module = _load_script()
    audio_path = tmp_path / "tone.ogg"
    audio_bytes = b"user-confirmed tone"
    audio_path.write_bytes(audio_bytes)
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "未知◯語", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )
    labels = {
        hashlib.sha256(audio_bytes).hexdigest(): module.ReviewLabel(
            label="TONE",
            reviewer="user",
            note="listening review",
        ),
    }

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256=labels,
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_voice_decoder,
    )
    result = module.build_clean_dataset(records)

    assert records[0].decision == "EXCLUDE_CONFIRMED_TONE"
    assert "user_label:TONE" in records[0].reasons
    assert "valid_voice_audio" not in records[0].reasons
    assert "caption_repair_required" not in records[0].reasons
    assert result.summary["review"] == 0


def test_build_inventory_voice_label_does_not_bypass_caption_review(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "quiet.ogg"
    audio_bytes = b"quiet breath"
    audio_path.write_bytes(audio_bytes)
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "未知◯語", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )
    labels = {
        hashlib.sha256(audio_bytes).hexdigest(): module.ReviewLabel(
            label="VOICE",
            reviewer="user",
            note="audible breath",
        ),
    }

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256=labels,
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_quiet_decoder,
    )

    assert records[0].decision == "REVIEW"
    assert "user_label:VOICE" in records[0].reasons
    assert "caption_repair_required" in records[0].reasons


def test_write_clean_dataset_rejects_unresolved_review(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "quiet.ogg"
    audio_path.write_bytes(b"quiet breath")
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "はぁ", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )
    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_quiet_decoder,
    )

    with pytest.raises(ValueError, match="unresolved REVIEW"):
        module.write_clean_dataset(tmp_path / "clean-dataset.jsonl", records)


def test_inventory_records_rule_version_and_reasons(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"voice")
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "本文", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )

    records = module.build_inventory(
        entry,
        labels_by_audio_sha256={},
        caption_rules=(),
        rule_version=RULE_VERSION,
        decoder=_voice_decoder,
    )

    assert records[0].rule_version == RULE_VERSION
    assert records[0].reasons


def test_process_dataset_writes_review_artifacts_but_not_clean_dataset(
    tmp_path: Path,
) -> None:
    module = _load_script()
    audio_path = tmp_path / "quiet.ogg"
    audio_path.write_bytes(b"quiet breath")
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "はぁ", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )
    candidate_paths: list[Path] = []
    output_dir = tmp_path / "output"
    rules = module.DatasetRules(
        labels_by_audio_sha256={},
        caption_rules=(),
        confirmed_signatures=(),
        rule_version=RULE_VERSION,
    )

    summary = module.process_dataset(
        entry,
        output_dir=output_dir,
        rules=rules,
        decoder=_quiet_decoder,
        candidate_writer=lambda path, _samples, _sample_rate: candidate_paths.append(path),
    )

    assert summary["review"] == 1
    assert (output_dir / "source-inventory.jsonl").is_file()
    assert (output_dir / "decisions.jsonl").is_file()
    assert (output_dir / "review-candidates.jsonl").is_file()
    assert (output_dir / "summary.json").is_file()
    assert not (output_dir / "clean-dataset.jsonl").exists()
    assert candidate_paths == [output_dir / "candidate-audio" / "oop55_00000000.wav"]


def test_process_dataset_emits_clean_dataset_after_review_resolution(tmp_path: Path) -> None:
    module = _load_script()
    audio_path = tmp_path / "quiet.ogg"
    audio_bytes = b"quiet breath"
    audio_path.write_bytes(audio_bytes)
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "はぁ", "FilePath": audio_path.name}],
    )
    entry = module.DatasetCatalogEntry(
        dataset_id="oop55",
        output_model_id="miu",
        index_json=index_path,
        speaker="ミウ",
    )
    labels = {
        hashlib.sha256(audio_bytes).hexdigest(): module.ReviewLabel(
            label="VOICE",
            reviewer="user",
            note="breath",
        )
    }
    output_dir = tmp_path / "resolved"
    rules = module.DatasetRules(
        labels_by_audio_sha256=labels,
        caption_rules=(),
        confirmed_signatures=(),
        rule_version=RULE_VERSION,
    )

    summary = module.process_dataset(
        entry,
        output_dir=output_dir,
        rules=rules,
        decoder=_quiet_decoder,
        candidate_writer=lambda *_args: None,
    )

    assert summary["review"] == 0
    assert (output_dir / "clean-dataset.jsonl").is_file()


def test_main_processes_catalog_into_per_dataset_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"voice")
    index_path = _write_index(
        tmp_path,
        [{"Speaker": "ミウ", "Text": "本文", "FilePath": audio_path.name}],
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "rule_version": RULE_VERSION,
                "datasets": [
                    {
                        "dataset_id": "oop55",
                        "output_model_id": "miu",
                        "index_json": index_path.name,
                        "speaker": "ミウ",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("", encoding="utf-8")
    rules_path = tmp_path / "caption-rules.json"
    rules_path.write_text("[]", encoding="utf-8")
    output_root = tmp_path / "results"
    monkeypatch.setattr(module, "decode_audio", _voice_decoder)

    exit_code = module.main(
        [
            "--catalog-json",
            str(catalog_path),
            "--labels-jsonl",
            str(labels_path),
            "--caption-rules-json",
            str(rules_path),
            "--output-root",
            str(output_root),
            "--progress-every",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_root / "oop55" / "clean-dataset.jsonl").is_file()


def _write_index(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "index.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def _voice_decoder(_path: Path) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.04, SAMPLE_RATE), SAMPLE_RATE


def _distinct_voice_decoder(path: Path) -> tuple[np.ndarray, int]:
    seed = sum(path.name.encode("utf-8"))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.04, SAMPLE_RATE), SAMPLE_RATE


def _quiet_decoder(_path: Path) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 0.0001, SAMPLE_RATE), SAMPLE_RATE


def _failing_decoder(_path: Path) -> tuple[np.ndarray, int]:
    message = "cannot decode"
    raise RuntimeError(message)
