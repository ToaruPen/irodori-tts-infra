from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from irodori_tts_infra.contracts import ErrorPayload


class ClientError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str = "client_error",
        details: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}
        self.endpoint = endpoint


class ClientTimeoutError(ClientError):
    pass


class ClientBackpressureError(ClientError):
    pass


class ClientUnavailableError(ClientError):
    pass


def build_timeout_error(exc: httpx.TimeoutException, *, endpoint: str) -> ClientTimeoutError:
    return ClientTimeoutError(
        str(exc),
        code="timeout",
        details={"error": str(exc)},
        endpoint=endpoint,
    )


def build_transport_error(exc: httpx.TransportError, *, endpoint: str) -> ClientUnavailableError:
    return ClientUnavailableError(
        str(exc),
        code="transport_error",
        details={"error": str(exc)},
        endpoint=endpoint,
    )


def build_response_error(response: httpx.Response, *, endpoint: str) -> ClientError:
    payload = _response_error_payload(response)
    error_type = _error_type_for_status(response.status_code)
    return error_type(
        payload.message,
        status_code=response.status_code,
        code=payload.code,
        details=dict(payload.details),
        endpoint=endpoint,
    )


def _error_type_for_status(status_code: int) -> type[ClientError]:
    if status_code == httpx.codes.REQUEST_TIMEOUT:
        return ClientTimeoutError
    if status_code == httpx.codes.TOO_MANY_REQUESTS:
        return ClientBackpressureError
    if status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
        return ClientUnavailableError
    return ClientError


def _response_error_payload(response: httpx.Response) -> ErrorPayload:
    try:
        data = response.json()
    except ValueError:
        return ErrorPayload(
            code="http_error",
            message=response.text or response.reason_phrase,
            details={"status_code": response.status_code},
        )

    try:
        return ErrorPayload.model_validate(data)
    except ValidationError:
        detail = data.get("detail") if isinstance(data, dict) else None
        message = str(detail) if detail is not None else response.reason_phrase
        return ErrorPayload(
            code="http_error",
            message=message,
            details={"status_code": response.status_code},
        )
