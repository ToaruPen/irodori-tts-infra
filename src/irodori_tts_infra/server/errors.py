from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi.responses import JSONResponse

from irodori_tts_infra.contracts import ErrorPayload
from irodori_tts_infra.engine.errors import (
    BackendUnavailableError,
    BackpressureError,
    EmptyBatchError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request


def _handle_backend_unavailable(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request
    return _error_response(503, "backend_unavailable", str(cast("BackendUnavailableError", exc)))


def _handle_backpressure(request: Request, exc: Exception) -> JSONResponse:
    del request
    return _error_response(429, "backpressure", str(cast("BackpressureError", exc)))


def _handle_empty_batch(request: Request, exc: Exception) -> JSONResponse:
    del request
    return _error_response(422, "empty_batch", str(cast("EmptyBatchError", exc)))


def add_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(BackendUnavailableError, _handle_backend_unavailable)
    app.add_exception_handler(BackpressureError, _handle_backpressure)
    app.add_exception_handler(EmptyBatchError, _handle_empty_batch)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorPayload(code=code, message=message).model_dump(mode="json"),
    )
