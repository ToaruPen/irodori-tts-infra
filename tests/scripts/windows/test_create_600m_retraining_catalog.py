# ruff: noqa: INP001
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/windows/create_600m_retraining_catalog.py")
RULE_VERSION = "2026-07-31.v1"
EXPECTED_DATASET_COUNT = 12
SHA256_HEX_LENGTH = 64
KASUMI_FIXTURE_ROWS = 2

DIRECT_FIXTURES = (
    (
        "oop176_natsu_no_owari_sp_dcec9a11d3",
        "oop176_natsu_no_owari_sp_dcec9a11d3",
        "ミオ",
        "galgame_cube_natsu_no_owari_7z",
        "CUBE_Natsu no Owari",
    ),
    (
        "oop52_aibeya_2_sp_5d544fe890",
        "oop52_aibeya_2_sp_5d544fe890",
        "朔",
        "galgame_azarashi_soft_aibeya_2_7z",
        "Azarashi Soft_Aibeya 2",
    ),
    (
        "oop53_aibeya_sp_f7269f5ffc",
        "oop53_aibeya_sp_f7269f5ffc",
        "綾希",
        "galgame_azarashi_soft_aibeya_7z",
        "Azarashi Soft_Aibeya",
    ),
    (
        "oop54_aikagi_2_sp_85dded42a7",
        "oop54_aikagi_2_sp_85dded42a7",
        "綾乃",
        "galgame_azarashi_soft_aikagi_2_7z",
        "Azarashi Soft_Aikagi 2",
    ),
    (
        "oop55_aikagi_3_sp_683c9895cc",
        "miu",
        "アイ",
        "galgame_azarashi_soft_aikagi_3_7z",
        "Azarashi Soft_Aikagi 3",
    ),
    (
        "oop68_maid_san_no_iru_kurashi_s_sp_e4da3225a4",
        "oop68_maid_san_no_iru_kurashi_s_sp_e4da3225a4",
        "エレナ",
        "galgame_azarashi_soft_maid_san_no_iru_kurashi_s_7z",
        "Azarashi Soft_Maid-san no Iru Kurashi S",
    ),
    (
        "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd",
        "oop69_maid_san_no_iru_kurashi_sp_07497b0fbd",
        "イヴ",
        "galgame_azarashi_soft_maid_san_no_iru_kurashi_7z",
        "Azarashi Soft_Maid-san no Iru Kurashi",
    ),
    (
        "oop70_osananajimi_no_iru_kurashi_sp_7195504dbb",
        "oop70_osananajimi_no_iru_kurashi_sp_7195504dbb",
        "舞雪",
        "galgame_azarashi_soft_osananajimi_no_iru_kurashi_7z",
        "Azarashi Soft_Osananajimi no Iru Kurashi",
    ),
    (
        "oop73_toshishita_kanojo_sp_6b50dbf844",
        "oop73_toshishita_kanojo_sp_6b50dbf844",
        "絢音",
        "galgame_azarashi_soft_toshishita_kanojo_7z",
        "Azarashi Soft_Toshishita Kanojo",
    ),
    (
        "oop77_anabel_maidgarden_sp_451488a7c1",
        "oop77_anabel_maidgarden_sp_451488a7c1",
        "アナベル",
        "galgame_barista_lab_anabel_maidgarden_7z",
        "Barista Lab_Anabel Maidgarden",
    ),
    (
        "narrator_toshiue_ama",
        "narrator_sayoko",
        "ama",
        "galgame_azarashi_soft_toshiue_kanojo_7z",
        "Azarashi Soft_Toshiue Kanojo",
    ),
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "create_600m_retraining_catalog",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_create_catalog_maps_twelve_sources_and_merges_oop55_into_miu(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training_root = tmp_path / "ooppeenn_training"
    expected_indexes = _write_direct_sources(training_root)
    _write_kasumi_sources(training_root, ambiguous=False)
    output_root = tmp_path / "output"

    payload = module.create_catalog(
        training_root=training_root,
        output_root=output_root,
        rule_version=RULE_VERSION,
        expected_kasumi_rows=KASUMI_FIXTURE_ROWS,
    )

    assert payload["rule_version"] == RULE_VERSION
    assert len(payload["datasets"]) == EXPECTED_DATASET_COUNT
    by_output = {row["output_model_id"]: row for row in payload["datasets"]}
    assert "miu" in by_output
    assert "oop55_aikagi_3_sp_683c9895cc" not in by_output
    assert by_output["miu"]["dataset_id"] == "oop55_aikagi_3_sp_683c9895cc"
    assert by_output["narrator_sayoko"]["dataset_id"] == "narrator_toshiue_ama"
    assert (
        Path(by_output["oop53_aibeya_sp_f7269f5ffc"]["index_json"])
        == (expected_indexes["oop53_aibeya_sp_f7269f5ffc"])
    )
    assert by_output["oop53_aibeya_sp_f7269f5ffc"]["source_speaker_rows"] == 1
    assert by_output["oop53_aibeya_sp_f7269f5ffc"]["missing_audio_count"] == 0
    assert len(by_output["oop53_aibeya_sp_f7269f5ffc"]["source_index_sha256"]) == SHA256_HEX_LENGTH
    assert json.loads((output_root / "catalog.json").read_text(encoding="utf-8")) == payload


def test_kasumi_duplicate_caption_mapping_is_explicitly_reviewable(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training_root = tmp_path / "ooppeenn_training"
    _write_direct_sources(training_root)
    kasumi_audio = _write_kasumi_sources(training_root, ambiguous=True)
    output_root = tmp_path / "output"

    payload = module.create_catalog(
        training_root=training_root,
        output_root=output_root,
        rule_version=RULE_VERSION,
        expected_kasumi_rows=KASUMI_FIXTURE_ROWS,
    )

    kasumi = next(row for row in payload["datasets"] if row["dataset_id"] == "kasumi")
    assert kasumi["mapping_review_count"] == 1
    normalized = json.loads(Path(kasumi["index_json"]).read_text(encoding="utf-8"))
    assert len(normalized) == KASUMI_FIXTURE_ROWS
    assert normalized[0]["MappingDecision"] == "REVIEW"
    assert normalized[0]["MappingCandidateMetadataIndices"] == [0, 1]
    assert Path(normalized[0]["FilePath"]) == kasumi_audio[0]
    audit_rows = [
        json.loads(line)
        for line in Path(kasumi["mapping_audit_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert audit_rows[0]["decision"] == "REVIEW"
    assert [Path(path) for path in audit_rows[0]["candidate_audio_paths"]] == kasumi_audio[:2]
    assert audit_rows[1]["decision"] == "MAPPED"


def test_kasumi_duplicate_caption_mapping_uses_exact_latent_frame_match(
    tmp_path: Path,
) -> None:
    module = _load_script()
    training_root = tmp_path / "ooppeenn_training"
    _write_direct_sources(training_root)
    kasumi_audio = _write_kasumi_sources(training_root, ambiguous=True)
    manifest_path = training_root / "0006_aiko" / "manifest.jsonl"
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest_rows[0]["num_frames"] = 50
    manifest_rows[1]["num_frames"] = 25
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    metadata_path = training_root / "0006_aiko_check" / "metadata.csv"
    with metadata_path.open(encoding="utf-8", newline="") as source:
        metadata_rows = list(csv.DictReader(source))
    with metadata_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("audio", "text", "speaker", "duration"),
        )
        writer.writeheader()
        for row, duration in zip(metadata_rows, ("1.0", "2.0", "1.0"), strict=True):
            writer.writerow({**row, "duration": duration})

    payload = module.create_catalog(
        training_root=training_root,
        output_root=tmp_path / "output",
        rule_version=RULE_VERSION,
        expected_kasumi_rows=KASUMI_FIXTURE_ROWS,
    )

    kasumi = next(row for row in payload["datasets"] if row["dataset_id"] == "kasumi")
    normalized = json.loads(Path(kasumi["index_json"]).read_text(encoding="utf-8"))
    assert kasumi["mapping_review_count"] == 0
    assert normalized[0]["MappingDecision"] == "MAPPED"
    assert normalized[0]["MappingCandidateMetadataIndices"] == [1]
    assert Path(normalized[0]["FilePath"]) == kasumi_audio[1]


def test_create_catalog_rejects_missing_referenced_audio(tmp_path: Path) -> None:
    module = _load_script()
    training_root = tmp_path / "ooppeenn_training"
    _write_direct_sources(training_root)
    _write_kasumi_sources(training_root, ambiguous=False)
    missing = (
        training_root
        / "top_speakers_20260529"
        / "extracted"
        / "galgame_cube_natsu_no_owari_7z"
        / "CUBE_Natsu no Owari"
        / "ミオ"
        / "voice.ogg"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=r"oop176.*missing audio"):
        module.create_catalog(
            training_root=training_root,
            output_root=tmp_path / "output",
            rule_version=RULE_VERSION,
            expected_kasumi_rows=KASUMI_FIXTURE_ROWS,
        )


def test_kasumi_manifest_must_be_an_ordered_metadata_subsequence(tmp_path: Path) -> None:
    module = _load_script()
    training_root = tmp_path / "ooppeenn_training"
    _write_direct_sources(training_root)
    _write_kasumi_sources(training_root, ambiguous=False, second_manifest_text="missing")

    with pytest.raises(ValueError, match=r"kasumi manifest row 1.*metadata"):
        module.create_catalog(
            training_root=training_root,
            output_root=tmp_path / "output",
            rule_version=RULE_VERSION,
            expected_kasumi_rows=KASUMI_FIXTURE_ROWS,
        )


def _write_direct_sources(training_root: Path) -> dict[str, Path]:
    indexes: dict[str, Path] = {}
    for dataset_id, _output_model_id, speaker, slug, game in DIRECT_FIXTURES:
        index_path = (
            training_root / "top_speakers_20260529" / "indexes" / slug / game / "index.json"
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        source_row = (
            {"Speaker": speaker, "Text": "本文", "Voice": "voice"}
            if dataset_id == "narrator_toshiue_ama"
            else {"Speaker": speaker, "Text": "本文", "FilePath": f"{speaker}/voice.ogg"}
        )
        index_path.write_text(
            json.dumps([source_row], ensure_ascii=False),
            encoding="utf-8",
        )
        if dataset_id == "narrator_toshiue_ama":
            audio_root = (
                training_root
                / "narrator_toshiue_ama_20260530"
                / "extracted"
                / "Azarashi Soft_Toshiue Kanojo"
            )
        else:
            audio_root = training_root / "top_speakers_20260529" / "extracted" / slug / game
        audio_path = audio_root / speaker / "voice.ogg"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(dataset_id.encode())
        indexes[dataset_id] = index_path.resolve()
    return indexes


def _write_kasumi_sources(
    training_root: Path,
    *,
    ambiguous: bool,
    second_manifest_text: str = "最後",
) -> list[Path]:
    manifest = training_root / "0006_aiko" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            json.dumps({"text": text}, ensure_ascii=False)
            for text in ("重複", second_manifest_text)
        )
        + "\n",
        encoding="utf-8",
    )
    audio_root = training_root / "0006_aiko_check" / "speaker_audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    names = ["first.ogg", "alternative.ogg", "last.ogg"] if ambiguous else ["first.ogg", "last.ogg"]
    audio_paths = [(audio_root / name).resolve() for name in names]
    for path in audio_paths:
        path.write_bytes(path.name.encode())
    metadata_path = training_root / "0006_aiko_check" / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    texts = ["重複", "重複", "最後"] if ambiguous else ["重複", "最後"]
    with metadata_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("audio", "text", "speaker"))
        writer.writeheader()
        for path, text in zip(audio_paths, texts, strict=True):
            writer.writerow({"audio": path, "text": text, "speaker": "藍子"})
    return audio_paths
