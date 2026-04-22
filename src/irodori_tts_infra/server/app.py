from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog
from fastapi import FastAPI

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.server.errors import add_exception_handlers
from irodori_tts_infra.server.routers.health import router as health_router
from irodori_tts_infra.server.routers.synthesis import router as synthesis_router

_logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline


@runtime_checkable
class _WarmableBackend(Protocol):
    def warm_up(self) -> None: ...


@runtime_checkable
class _ClosableBackend(Protocol):
    def close(self) -> None: ...


def create_app(pipeline: SynthesisPipeline) -> FastAPI:
    backend = pipeline.backend
    voice_converter = pipeline.voice_converter

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_loaded = True
        app.state.health_detail = None
        for component in (backend, voice_converter):
            if component is None or not isinstance(component, _WarmableBackend):
                continue
            try:
                await asyncio.to_thread(component.warm_up)
            except BackendUnavailableError as exc:
                app.state.model_loaded = False
                app.state.health_detail = str(exc)
                break

        try:
            yield
        finally:
            for component in (voice_converter, backend):
                if component is None or not isinstance(component, _ClosableBackend):
                    continue
                try:
                    component.close()
                except Exception:
                    _logger.exception(
                        "pipeline component close failed",
                        component=type(component).__name__,
                    )

    app = FastAPI(lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.backend = backend
    app.state.health_detail = None
    app.state.max_chunk_size = MAX_CHUNK_SIZE_BYTES
    app.state.model_loaded = False
    add_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(synthesis_router)
    return app
