from __future__ import annotations

from typing import TYPE_CHECKING, cast

import structlog
from fastapi.responses import JSONResponse

from irodori_tts_infra.contracts import ErrorPayload
from irodori_tts_infra.engine.errors import (
    BackendUnavailableError,
    BackpressureError,
    EmptyBatchError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

logger = structlog.get_logger()


def _handle_backend_unavailable(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    error = cast("BackendUnavailableError", exc)
    logger.error(
        "backend_unavailable",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(503, "backend_unavailable", str(error))


def _handle_backpressure(request: Request, exc: Exception) -> JSONResponse:
    error = cast("BackpressureError", exc)
    logger.error(
        "backpressure",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(429, "backpressure", str(error))


def _handle_empty_batch(request: Request, exc: Exception) -> JSONResponse:
    error = cast("EmptyBatchError", exc)
    logger.error(
        "empty_batch",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(422, "empty_batch", str(error))


def add_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(BackendUnavailableError, _handle_backend_unavailable)
    app.add_exception_handler(BackpressureError, _handle_backpressure)
    app.add_exception_handler(EmptyBatchError, _handle_empty_batch)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorPayload(code=code, message=message).model_dump(mode="json"),
    )
