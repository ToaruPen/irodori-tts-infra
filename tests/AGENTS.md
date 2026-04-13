# tests/

Tests mirror src structure. Prefer unit tests first, then thin integration tests
around FastAPI TestClient and real cache store.
Mark GPU/SSH/integration tests with appropriate pytest markers.
