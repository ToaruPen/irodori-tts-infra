from __future__ import annotations

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import ErrorPayload

pytestmark = pytest.mark.unit


def test_error_payload_rejects_blank_code() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        ErrorPayload(code=" ", message="ok")
