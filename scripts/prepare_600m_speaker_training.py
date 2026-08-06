from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

MetadataReader = Callable[[Path], Mapping[str, str]]

_CONFIG_METADATA_KEY = "config_json"
_SAFETENSORS_METADATA_KEY = "__metadata__"
_SAFETENSORS_LENGTH_BYTES = 8
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_LATENT_RANK = 2
_LATENT_DIM = 32
_TRAIN_FIELDS_FROM_CHECKPOINT = {
    "max_text_len": 256,
    "max_caption_len": 512,
}
_MODEL_FIELDS_UNUSED_FOR_SPEAKER_TRAINING = ("fixed_target_latent_steps",)


def read_safetensors_model_config(
    checkpoint: Path,
    *,
    metadata_reader: MetadataReader | None = None,
) -> dict[str, object]:
    reader = metadata_reader or _read_safetensors_metadata
    return _model_config_from_metadata(reader(checkpoint))


def build_training_config(
    checkpoint_metadata: Mapping[str, str],
    *,
    manifest: Path,
    output: Path,
) -> dict[str, dict[str, object]]:
    model = _model_config_from_metadata(checkpoint_metadata)
    train: dict[str, object] = {
        "manifest_path": str(manifest),
        "output_dir": str(output),
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "num_workers": 8,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 4,
        "allow_tf32": True,
        "compile_model": False,
        "precision": "bf16",
        "gradient_checkpointing": True,
        "train_mode": "rf",
        "optimizer": "adamw",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 0.00000001,
        "lr_scheduler": "none",
        "max_steps": 3000,
        "caption_warmup": False,
        "caption_warmup_steps": 0,
        "text_condition_dropout": 0.0,
        "speaker_condition_dropout": 0.0,
        "caption_condition_dropout": 0.0,
        "timestep_stratified": True,
        "max_latent_steps": 750,
        "fixed_target_latent_steps": None,
        "fixed_target_full_mask": False,
        "rf_loss_mode": "utterance_mean",
        "duration_loss_weight": 0.1,
        "duration_speaker_dropout": 0.0,
        "duration_huber_delta": 0.1,
        "log_every": 20,
        "save_every": 250,
        "checkpoint_best_n": 0,
        "valid_ratio": 0.0,
        "valid_every": 0,
        "wandb_enabled": False,
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_run_name": None,
        "wandb_mode": "online",
        "ddp_find_unused_parameters": False,
        "seed": 0,
        "lora_enabled": False,
        "speaker_inversion_enabled": True,
        "speaker_inversion_tokens": 16,
        "speaker_inversion_init_std": 0.02,
        "speaker_inversion_init_embedding": None,
    }
    for key, default in _TRAIN_FIELDS_FROM_CHECKPOINT.items():
        train[key] = model.pop(key, default)
    for key in _MODEL_FIELDS_UNUSED_FOR_SPEAKER_TRAINING:
        model.pop(key, None)
    return {"model": model, "train": train}


def build_training_config_from_checkpoint(
    checkpoint: Path,
    *,
    manifest: Path,
    output: Path,
    metadata_reader: MetadataReader | None = None,
) -> dict[str, dict[str, object]]:
    reader = metadata_reader or _read_safetensors_metadata
    return build_training_config(reader(checkpoint), manifest=manifest, output=output)


def to_training_manifest_row(clean_row: Mapping[str, object]) -> dict[str, object]:
    return dict(clean_row) | {"caption": ""}


def can_reuse_latent(
    clean_row: Mapping[str, object],
    *,
    provenance: Mapping[str, object],
    latent: np.ndarray,
) -> bool:
    source_id = clean_row.get("source_id")
    audio_sha256 = clean_row.get("audio_sha256")
    if not isinstance(source_id, str) or not source_id:
        return False
    if not isinstance(audio_sha256, str) or not audio_sha256:
        return False
    if provenance.get("source_id") != source_id:
        return False
    if provenance.get("audio_sha256") != audio_sha256:
        return False
    return bool(
        latent.ndim == _LATENT_RANK
        and latent.shape[0] > 0
        and latent.shape[1] == _LATENT_DIM
        and np.isfinite(latent).all()
    )


def _model_config_from_metadata(metadata: Mapping[str, str]) -> dict[str, object]:
    raw_config = metadata.get(_CONFIG_METADATA_KEY)
    if raw_config is None:
        message = f"safetensors metadata is missing {_CONFIG_METADATA_KEY!r}"
        raise ValueError(message)
    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        message = f"safetensors {_CONFIG_METADATA_KEY!r} is not valid JSON"
        raise ValueError(message) from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        message = f"safetensors {_CONFIG_METADATA_KEY!r} must contain a JSON object"
        raise ValueError(message)
    return dict(parsed)


def _read_safetensors_metadata(checkpoint: Path) -> Mapping[str, str]:
    with checkpoint.open("rb") as file:
        raw_length = file.read(_SAFETENSORS_LENGTH_BYTES)
        if len(raw_length) != _SAFETENSORS_LENGTH_BYTES:
            message = f"invalid safetensors header length: {checkpoint}"
            raise ValueError(message)
        header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
        if header_length <= 0 or header_length > _MAX_SAFETENSORS_HEADER_BYTES:
            message = f"invalid safetensors header size {header_length}: {checkpoint}"
            raise ValueError(message)
        raw_header = file.read(header_length)
    if len(raw_header) != header_length:
        message = f"truncated safetensors header: {checkpoint}"
        raise ValueError(message)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"invalid safetensors JSON header: {checkpoint}"
        raise ValueError(message) from exc
    if not isinstance(header, dict):
        message = f"safetensors header must be a JSON object: {checkpoint}"
        raise TypeError(message)
    metadata = header.get(_SAFETENSORS_METADATA_KEY)
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        message = f"safetensors metadata must be a string mapping: {checkpoint}"
        raise ValueError(message)
    return metadata
