# engine/

Windows GPU synthesis boundary. No FastAPI, no HTTP client, no cache policy.
Optional heavy imports (torch, irodori_tts) stay inside adapter modules.
Imports: config, voice_bank, text, audio, metrics.
Must not import: server, client.
