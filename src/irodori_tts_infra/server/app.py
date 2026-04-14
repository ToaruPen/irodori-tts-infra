from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fastapi import FastAPI

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.server.errors import add_exception_handlers
from irodori_tts_infra.server.routers.health import router as health_router
from irodori_tts_infra.server.routers.synthesis import router as synthesis_router

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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if isinstance(backend, _WarmableBackend):
            try:
                await asyncio.to_thread(backend.warm_up)
            except BackendUnavailableError as exc:
                app.state.model_loaded = False
                app.state.health_detail = str(exc)
            else:
                app.state.model_loaded = True
        else:
            app.state.model_loaded = True

        try:
            yield
        finally:
            if isinstance(backend, _ClosableBackend):
                backend.close()

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
