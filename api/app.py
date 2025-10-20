"""Application factory for the FastAPI service."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core import OUTPUT_DIR, BASE_DIR
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Chatterbox TTS Extended API", version="1.0.0")
    app.mount("/media", StaticFiles(directory=str(OUTPUT_DIR)), name="media")
    app.include_router(router)

    ui_dist = BASE_DIR / "ui" / "dist"
    assets_dir = ui_dist / "assets"
    if assets_dir.exists():
        app.mount("/ui/assets", StaticFiles(directory=str(assets_dir)), name="ui-assets")

        @app.get("/ui", include_in_schema=False)
        async def serve_ui_index() -> FileResponse:
            return FileResponse(ui_dist / "index.html")

        @app.get("/ui/{path:path}", include_in_schema=False)
        async def serve_ui_spa(path: str) -> FileResponse:
            return FileResponse(ui_dist / "index.html")

    return app


__all__ = ["create_app"]
