from __future__ import annotations

from irodori_tts_infra.deploy.remote.bootstrap import bootstrap_remote
from irodori_tts_infra.deploy.remote.service import (
    start_service,
    status_service,
    stop_service,
)
from irodori_tts_infra.deploy.remote.sync import sync_project
from irodori_tts_infra.deploy.remote.voice_bank import verify_voice_bank

__all__ = [
    "bootstrap_remote",
    "start_service",
    "status_service",
    "stop_service",
    "sync_project",
    "verify_voice_bank",
]
