"""Application factory for the FastAPI service."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .core import OUTPUT_DIR
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Chatterbox TTS Extended API", version="1.0.0")
    app.mount("/media", StaticFiles(directory=str(OUTPUT_DIR)), name="media")
    app.include_router(router)
    return app


__all__ = ["create_app"]
