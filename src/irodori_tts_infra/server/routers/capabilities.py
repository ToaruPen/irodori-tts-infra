from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from irodori_tts_infra.contracts import CapabilitiesResponse
from irodori_tts_infra.server.dependencies import get_capabilities_response

router = APIRouter()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(
    response: Annotated[CapabilitiesResponse, Depends(get_capabilities_response)],
) -> CapabilitiesResponse:
    return response
