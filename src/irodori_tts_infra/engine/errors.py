from __future__ import annotations


class EngineError(Exception):
    pass


class BackendUnavailableError(EngineError):
    pass


class BackpressureError(EngineError):
    pass


class EmptyBatchError(EngineError):
    pass


class VoiceNotFoundError(EngineError):
    pass


class RuntimeGenerationMismatchError(EngineError):
    pass


class ModelNotLoadedError(EngineError):
    pass


class VoiceBankInvalidError(EngineError):
    pass
