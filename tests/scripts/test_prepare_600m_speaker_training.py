from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/prepare_600m_speaker_training.py")
EXPECTED_BATCH_SIZE = 16
EXPECTED_SPEAKER_TOKENS = 16
EXPECTED_SAVE_EVERY = 250
EXPECTED_MAX_TEXT_LEN = 256
EXPECTED_MAX_CAPTION_LEN = 512
EXPECTED_TARGET_LATENT_STEPS = 750
EXPECTED_LEARNING_RATE = 0.01
EXPECTED_MAX_STEPS = 3000
EXPECTED_ADAM_EPS = 0.00000001
EXPECTED_NUM_WORKERS = 8
EXPECTED_PREFETCH_FACTOR = 4
EXPECTED_LOG_EVERY = 20

MODEL_CONFIG = {
    "latent_dim": 32,
    "latent_patch_size": 1,
    "model_dim": 1024,
    "use_caption_condition": True,
    "use_speaker_condition": True,
    "duration_architecture": "token_sum_dual_adarn_zero_no_aux",
    "max_text_len": EXPECTED_MAX_TEXT_LEN,
    "max_caption_len": EXPECTED_MAX_CAPTION_LEN,
    "fixed_target_latent_steps": EXPECTED_TARGET_LATENT_STEPS,
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_600m_speaker_training", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_safetensors_model_config_extracts_config_json(tmp_path: Path) -> None:
    module = _load_script()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test checkpoint")
    calls: list[Path] = []

    def metadata_reader(path: Path) -> Mapping[str, str]:
        calls.append(path)
        return {"config_json": json.dumps(MODEL_CONFIG)}

    result = module.read_safetensors_model_config(checkpoint, metadata_reader=metadata_reader)

    assert result == MODEL_CONFIG
    assert calls == [checkpoint]


def test_read_safetensors_model_config_reads_file_metadata(tmp_path: Path) -> None:
    module = _load_script()
    checkpoint = tmp_path / "model.safetensors"
    header = json.dumps(
        {"__metadata__": {"config_json": json.dumps(MODEL_CONFIG)}},
        separators=(",", ":"),
    ).encode()
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)

    assert module.read_safetensors_model_config(checkpoint) == MODEL_CONFIG


def test_read_safetensors_model_config_rejects_missing_config_json(tmp_path: Path) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="config_json"):
        module.read_safetensors_model_config(
            tmp_path / "model.safetensors",
            metadata_reader=lambda _path: {},
        )


def test_build_training_config_preserves_600m_model_and_moves_train_lengths(
    tmp_path: Path,
) -> None:
    module = _load_script()
    manifest = tmp_path / "speaker" / "clean-manifest.jsonl"
    output = tmp_path / "speaker" / "outputs_600m_speaker_inversion"
    metadata = {"config_json": json.dumps(MODEL_CONFIG)}

    config = module.build_training_config(metadata, manifest=manifest, output=output)

    assert config["model"] == {
        key: value
        for key, value in MODEL_CONFIG.items()
        if key not in {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}
    }
    assert config["model"]["use_caption_condition"] is True
    assert config["model"]["use_speaker_condition"] is True
    assert config["model"]["duration_architecture"] == "token_sum_dual_adarn_zero_no_aux"
    assert config["train"]["manifest_path"] == str(manifest)
    assert config["train"]["output_dir"] == str(output)
    assert config["train"]["max_text_len"] == EXPECTED_MAX_TEXT_LEN
    assert config["train"]["max_caption_len"] == EXPECTED_MAX_CAPTION_LEN
    assert config["train"]["fixed_target_latent_steps"] is None


def test_build_training_config_from_checkpoint_reads_its_metadata(tmp_path: Path) -> None:
    module = _load_script()
    checkpoint = tmp_path / "model.safetensors"
    manifest = tmp_path / "clean-manifest.jsonl"
    output = tmp_path / "output"

    config = module.build_training_config_from_checkpoint(
        checkpoint,
        manifest=manifest,
        output=output,
        metadata_reader=lambda path: (
            {"config_json": json.dumps(MODEL_CONFIG)} if path == checkpoint else {}
        ),
    )

    assert config["model"]["model_dim"] == MODEL_CONFIG["model_dim"]
    assert config["train"]["manifest_path"] == str(manifest)


def test_build_training_config_uses_new_16_token_speaker_inversion(tmp_path: Path) -> None:
    module = _load_script()

    config = module.build_training_config(
        {"config_json": json.dumps(MODEL_CONFIG)},
        manifest=tmp_path / "clean-manifest.jsonl",
        output=tmp_path / "output",
    )

    assert config["train"]["speaker_inversion_enabled"] is True
    assert config["train"]["speaker_inversion_tokens"] == EXPECTED_SPEAKER_TOKENS
    assert config["train"]["speaker_inversion_init_embedding"] is None
    assert config["train"]["batch_size"] == EXPECTED_BATCH_SIZE
    assert config["train"]["precision"] == "bf16"
    assert config["train"]["gradient_checkpointing"] is True
    assert config["train"]["save_every"] == EXPECTED_SAVE_EVERY
    assert config["train"]["train_mode"] == "rf"
    assert config["train"]["caption_warmup"] is False
    assert config["train"]["optimizer"] == "adamw"
    assert config["train"]["lora_enabled"] is False
    assert config["train"]["valid_ratio"] == pytest.approx(0.0)


def test_build_training_config_uses_upstream_speaker_inversion_optimizer_defaults(
    tmp_path: Path,
) -> None:
    module = _load_script()

    config = module.build_training_config(
        {"config_json": json.dumps(MODEL_CONFIG)},
        manifest=tmp_path / "clean-manifest.jsonl",
        output=tmp_path / "output",
    )

    assert config["train"]["learning_rate"] == pytest.approx(EXPECTED_LEARNING_RATE)
    assert config["train"]["weight_decay"] == pytest.approx(0.0)
    assert config["train"]["adam_beta1"] == pytest.approx(0.9)
    assert config["train"]["adam_beta2"] == pytest.approx(0.999)
    assert config["train"]["adam_eps"] == pytest.approx(EXPECTED_ADAM_EPS)
    assert config["train"]["lr_scheduler"] == "none"
    assert config["train"]["max_steps"] == EXPECTED_MAX_STEPS


def test_build_training_config_uses_upstream_speaker_inversion_loss_and_io_defaults(
    tmp_path: Path,
) -> None:
    module = _load_script()

    train = module.build_training_config(
        {"config_json": json.dumps(MODEL_CONFIG)},
        manifest=tmp_path / "clean-manifest.jsonl",
        output=tmp_path / "output",
    )["train"]

    assert train["gradient_accumulation_steps"] == 1
    assert train["num_workers"] == EXPECTED_NUM_WORKERS
    assert train["dataloader_persistent_workers"] is True
    assert train["dataloader_prefetch_factor"] == EXPECTED_PREFETCH_FACTOR
    assert train["allow_tf32"] is True
    assert train["compile_model"] is False
    assert train["text_condition_dropout"] == pytest.approx(0.0)
    assert train["speaker_condition_dropout"] == pytest.approx(0.0)
    assert train["caption_condition_dropout"] == pytest.approx(0.0)
    assert train["timestep_stratified"] is True
    assert train["max_latent_steps"] == EXPECTED_TARGET_LATENT_STEPS
    assert train["fixed_target_full_mask"] is False
    assert train["rf_loss_mode"] == "utterance_mean"
    assert train["duration_loss_weight"] == pytest.approx(0.1)
    assert train["duration_speaker_dropout"] == pytest.approx(0.0)
    assert train["duration_huber_delta"] == pytest.approx(0.1)
    assert train["log_every"] == EXPECTED_LOG_EVERY
    assert train["checkpoint_best_n"] == 0
    assert train["valid_every"] == 0
    assert train["wandb_enabled"] is False
    assert train["wandb_project"] is None
    assert train["wandb_entity"] is None
    assert train["wandb_run_name"] is None
    assert train["wandb_mode"] == "online"
    assert train["ddp_find_unused_parameters"] is False
    assert train["seed"] == 0


def test_to_training_manifest_row_forces_empty_caption() -> None:
    module = _load_script()
    clean_row = {
        "text": "はぁ、んっ",
        "caption": "既存の任意captionは使わない",
        "latent_path": "latents/0001.pt",
        "num_frames": 42,
        "source_id": "oop55:00000001",
        "audio_sha256": "a" * 64,
    }

    result = module.to_training_manifest_row(clean_row)

    assert result == clean_row | {"caption": ""}


def test_latent_reuse_ignores_transcript_when_source_and_audio_are_stable() -> None:
    module = _load_script()
    clean_row = {
        "source_id": "oop55:00000001",
        "audio_sha256": "a" * 64,
        "text": "修正後のcaption",
    }
    provenance = {
        "source_id": "oop55:00000001",
        "audio_sha256": "a" * 64,
        "text": "修正前のcaption",
    }
    latent = np.ones((42, 32), dtype=np.float32)

    assert module.can_reuse_latent(clean_row, provenance=provenance, latent=latent) is True


@pytest.mark.parametrize(
    ("provenance"),
    [
        {"source_id": "oop55:00000002", "audio_sha256": "a" * 64},
        {"source_id": "oop55:00000001", "audio_sha256": "b" * 64},
        {"audio_sha256": "a" * 64},
        {"source_id": "oop55:00000001"},
    ],
)
def test_latent_reuse_rejects_unstable_source_or_audio(provenance: dict[str, str]) -> None:
    module = _load_script()
    clean_row = {"source_id": "oop55:00000001", "audio_sha256": "a" * 64}

    assert (
        module.can_reuse_latent(
            clean_row,
            provenance=provenance,
            latent=np.ones((1, 32), dtype=np.float32),
        )
        is False
    )


@pytest.mark.parametrize(
    "latent",
    [
        np.ones((0, 32), dtype=np.float32),
        np.ones((2, 31), dtype=np.float32),
        np.ones((32,), dtype=np.float32),
        np.array([[float("nan")] * 32], dtype=np.float32),
        np.array([[float("inf")] * 32], dtype=np.float32),
    ],
)
def test_latent_reuse_rejects_invalid_latent(latent: np.ndarray) -> None:
    module = _load_script()
    clean_row = {"source_id": "oop55:00000001", "audio_sha256": "a" * 64}

    assert module.can_reuse_latent(clean_row, provenance=clean_row, latent=latent) is False
