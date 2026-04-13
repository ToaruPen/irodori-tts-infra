# server/

FastAPI HTTP boundary. Routers translate HTTP <-> contracts and call lower packages.
No synthesis internals in route files. Keep handlers thin.
Imports: contracts, config, engine, cache, voice_bank, text, audio, metrics.
