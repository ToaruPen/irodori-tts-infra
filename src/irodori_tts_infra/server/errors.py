from __future__ import annotations

from typing import TYPE_CHECKING, cast

import structlog
from fastapi.responses import JSONResponse

from irodori_tts_infra.contracts import ErrorPayload
from irodori_tts_infra.engine.errors import (
    BackendUnavailableError,
    BackpressureError,
    EmptyBatchError,
    ModelNotLoadedError,
    RuntimeGenerationMismatchError,
    VoiceBankInvalidError,
    VoiceNotFoundError,
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
    return _error_response(
        503,
        "backend_unavailable",
        "Synthesis backend is unavailable",
    )


def _handle_backpressure(request: Request, exc: Exception) -> JSONResponse:
    error = cast("BackpressureError", exc)
    logger.error(
        "backpressure",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(429, "backpressure", "Synthesis backend is busy")


def _handle_empty_batch(request: Request, exc: Exception) -> JSONResponse:
    error = cast("EmptyBatchError", exc)
    logger.error(
        "empty_batch",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(422, "empty_batch", "Synthesis batch is empty")


def _handle_voice_not_found(request: Request, exc: Exception) -> JSONResponse:
    error = cast("VoiceNotFoundError", exc)
    logger.error(
        "voice_not_found",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(404, "voice_not_found", "Requested voice is not available")


def _handle_generation_mismatch(request: Request, exc: Exception) -> JSONResponse:
    error = cast("RuntimeGenerationMismatchError", exc)
    logger.error(
        "runtime_generation_mismatch",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(
        409,
        "runtime_generation_mismatch",
        "Requested runtime generation is no longer active",
    )


def _handle_model_not_loaded(request: Request, exc: Exception) -> JSONResponse:
    error = cast("ModelNotLoadedError", exc)
    logger.error(
        "model_not_loaded",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(503, "model_not_loaded", "Synthesis model is not loaded")


def _handle_voice_bank_invalid(request: Request, exc: Exception) -> JSONResponse:
    error = cast("VoiceBankInvalidError", exc)
    logger.error(
        "voice_bank_invalid",
        method=request.method,
        path=request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(503, "voice_bank_invalid", "Voice catalog is unavailable")


def add_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(BackendUnavailableError, _handle_backend_unavailable)
    app.add_exception_handler(BackpressureError, _handle_backpressure)
    app.add_exception_handler(EmptyBatchError, _handle_empty_batch)
    app.add_exception_handler(VoiceNotFoundError, _handle_voice_not_found)
    app.add_exception_handler(RuntimeGenerationMismatchError, _handle_generation_mismatch)
    app.add_exception_handler(ModelNotLoadedError, _handle_model_not_loaded)
    app.add_exception_handler(VoiceBankInvalidError, _handle_voice_bank_invalid)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorPayload(code=code, message=message).model_dump(mode="json"),
    )
