from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES, Readiness
from irodori_tts_infra.engine.errors import (
    BackendUnavailableError,
    VoiceBankInvalidError,
)
from irodori_tts_infra.server.errors import add_exception_handlers
from irodori_tts_infra.server.routers.capabilities import router as capabilities_router
from irodori_tts_infra.server.routers.health import router as health_router
from irodori_tts_infra.server.routers.synthesis import router as synthesis_router

_logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from irodori_tts_infra.engine.pipeline import SynthesisPipeline

    PipelineFactory = Callable[[], SynthesisPipeline]


def create_app(pipeline: SynthesisPipeline) -> FastAPI:
    return create_app_from_factory(
        lambda: pipeline,
        initial_pipeline=pipeline,
        generation=pipeline.generation,
    )


def create_app_from_factory(
    pipeline_factory: PipelineFactory,
    *,
    initial_pipeline: SynthesisPipeline | None = None,
    generation: str = "unconfigured",
    emoji_conditioning_supported: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if initial_pipeline is not None:
            await _start_initial_pipeline(app, initial_pipeline)
            try:
                yield
            finally:
                await _close_pipeline(initial_pipeline)
            return

        if app.state.generation == "unconfigured":
            _set_failure_state(app, "voice_bank_invalid")
            yield
            return

        _set_loading_state(app)
        load_task = asyncio.create_task(_load_pipeline(app, pipeline_factory))
        app.state.load_task = load_task
        try:
            yield
        finally:
            if not load_task.done():
                load_task.cancel()
            with suppress(asyncio.CancelledError):
                await load_task
            pipeline = app.state.pipeline
            if pipeline is not None:
                await _close_pipeline(pipeline)
            _reset_factory_state(app)

    app = FastAPI(lifespan=lifespan)
    app.state.pipeline = initial_pipeline
    app.state.generation = generation
    app.state.emoji_conditioning_supported = emoji_conditioning_supported
    app.state.health_detail = None
    app.state.max_chunk_size = MAX_CHUNK_SIZE_BYTES
    app.state.model_loaded = initial_pipeline is not None
    app.state.readiness = "ready" if initial_pipeline is not None else "model_loading"
    app.state.load_task = None
    add_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(capabilities_router)
    app.include_router(synthesis_router)
    return app


async def _start_initial_pipeline(app: FastAPI, pipeline: SynthesisPipeline) -> None:
    try:
        await _warm_up_pipeline(pipeline)
    except Exception:
        _logger.exception("pipeline warm-up failed")
        _set_failure_state(app, "model_not_loaded")
        return
    _publish_pipeline(app, pipeline)


async def _load_pipeline(app: FastAPI, pipeline_factory: PipelineFactory) -> None:
    pipeline: SynthesisPipeline | None = None
    factory_task = asyncio.create_task(asyncio.to_thread(pipeline_factory))
    try:
        pipeline = await asyncio.shield(factory_task)
        _require_matching_generation(pipeline, app.state.generation)
        await _warm_up_pipeline(pipeline)
        _publish_pipeline(app, pipeline)
    except asyncio.CancelledError:
        if pipeline is None:
            with suppress(Exception):
                pipeline = await factory_task
        if pipeline is not None and pipeline is not app.state.pipeline:
            await _close_pipeline(pipeline)
        raise
    except VoiceBankInvalidError:
        _logger.exception("voice bank load failed")
        _set_failure_state(app, "voice_bank_invalid")
        if pipeline is not None:
            await _close_pipeline(pipeline)
    except Exception:
        _logger.exception("pipeline load failed")
        _set_failure_state(app, "model_not_loaded")
        if pipeline is not None:
            await _close_pipeline(pipeline)


async def _warm_up_pipeline(pipeline: SynthesisPipeline) -> None:
    warm_up = getattr(pipeline.backend, "warm_up", None)
    if not callable(warm_up):
        return
    warmup_ref_embed = str(pipeline.voice_profile.narrator.ref_embed)
    warmup_task = asyncio.create_task(asyncio.to_thread(warm_up, ref_embed=warmup_ref_embed))
    try:
        await asyncio.shield(warmup_task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await warmup_task
        raise


async def _close_pipeline(pipeline: SynthesisPipeline) -> None:
    close = getattr(pipeline.backend, "close", None)
    if not callable(close):
        return
    try:
        await asyncio.to_thread(close)
    except (BackendUnavailableError, OSError):
        _logger.exception(
            "pipeline component close failed",
            component=type(pipeline.backend).__name__,
        )


def _publish_pipeline(app: FastAPI, pipeline: SynthesisPipeline) -> None:
    app.state.pipeline = pipeline
    app.state.model_loaded = True
    app.state.health_detail = None
    app.state.readiness = "ready"


def _require_matching_generation(
    pipeline: SynthesisPipeline,
    advertised_generation: str,
) -> None:
    if pipeline.generation != advertised_generation:
        msg = "loaded pipeline generation does not match advertised generation"
        raise BackendUnavailableError(msg)


def _set_loading_state(app: FastAPI) -> None:
    app.state.pipeline = None
    app.state.model_loaded = False
    app.state.health_detail = "Synthesis model is loading"
    app.state.readiness = "model_loading"


def _set_failure_state(app: FastAPI, readiness: Readiness) -> None:
    app.state.pipeline = None
    app.state.model_loaded = False
    app.state.readiness = readiness
    app.state.health_detail = (
        "Voice catalog is unavailable"
        if readiness == "voice_bank_invalid"
        else "Synthesis model is not loaded"
    )


def _reset_factory_state(app: FastAPI) -> None:
    app.state.pipeline = None
    app.state.model_loaded = False
    app.state.health_detail = None
    app.state.readiness = "model_loading"
    app.state.load_task = None
