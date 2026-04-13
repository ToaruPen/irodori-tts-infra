from __future__ import annotations

from irodori_tts_infra import __version__


def test_package_importable() -> None:
    assert isinstance(__version__, str)
    assert __version__.strip()
