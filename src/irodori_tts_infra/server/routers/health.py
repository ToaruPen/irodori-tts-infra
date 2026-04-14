from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from irodori_tts_infra.contracts import HealthResponse
from irodori_tts_infra.server.dependencies import get_health_response

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(
    health_response: Annotated[HealthResponse, Depends(get_health_response)],
) -> HealthResponse:
    return health_response
