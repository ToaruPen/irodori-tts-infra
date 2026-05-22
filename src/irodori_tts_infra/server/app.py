from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.server.errors import add_exception_handlers
from irodori_tts_infra.server.routers.health import router as health_router
from irodori_tts_infra.server.routers.synthesis import router as synthesis_router

_logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline

    PipelineFactory = Callable[[], SynthesisPipeline]


def create_app(pipeline: SynthesisPipeline) -> FastAPI:
    return create_app_from_factory(lambda: pipeline, initial_pipeline=pipeline)


def create_app_from_factory(
    pipeline_factory: PipelineFactory,
    *,
    initial_pipeline: SynthesisPipeline | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pipeline = app.state.pipeline
        if pipeline is None and initial_pipeline is None:
            pipeline = await asyncio.to_thread(pipeline_factory)
            app.state.pipeline = pipeline

        if pipeline is None:
            app.state.model_loaded = False
            app.state.health_detail = "Synthesis pipeline is not configured"
            yield
            return

        backend = pipeline.backend
        app.state.model_loaded = True
        app.state.health_detail = None

        warm_up = getattr(backend, "warm_up", None)
        if callable(warm_up):
            warmup_ref_embed = str(pipeline.voice_profile.narrator.ref_embed)
            try:
                await asyncio.to_thread(warm_up, ref_embed=warmup_ref_embed)
            except BackendUnavailableError as exc:
                app.state.model_loaded = False
                app.state.health_detail = str(exc)

        try:
            yield
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    await asyncio.to_thread(close)
                except (BackendUnavailableError, OSError):
                    _logger.exception(
                        "pipeline component close failed",
                        component=type(backend).__name__,
                    )
            if initial_pipeline is None:
                app.state.pipeline = None
                app.state.model_loaded = False
                app.state.health_detail = None

    app = FastAPI(lifespan=lifespan)
    app.state.pipeline = initial_pipeline
    app.state.health_detail = None
    app.state.max_chunk_size = MAX_CHUNK_SIZE_BYTES
    app.state.model_loaded = False
    add_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(synthesis_router)
    return app
