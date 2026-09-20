from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_server_extra_declares_default_wav_encoder_dependency() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    server_requirements = project["optional-dependencies"]["server"]

    assert any(requirement.startswith("soundfile") for requirement in server_requirements)
