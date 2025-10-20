"""Core environment and shared state for the FastAPI service."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import nltk

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "chatterbox" / "src"
for path in (BASE_DIR, SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# -----------------------------------------------------------------------------
# Optional dependency shims
# -----------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when Hugging Face "spaces" missing
    import spaces  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback context
    import types

    def _gpu_decorator(func=None, *args, **kwargs):
        if func is None:
            return lambda real_func: real_func
        return func

    spaces_stub = types.ModuleType("spaces")
    spaces_stub.GPU = _gpu_decorator
    sys.modules["spaces"] = spaces_stub

# Import core functionality from the reusable service module
from chatterbox.service import (  # type: ignore
    DEVICE,
    WHISPER_MODEL_MAP as whisper_model_map,
    default_settings,
    generate_batch_tts,
    load_settings,
    save_settings,
    voice_conversion as chatter_voice_conversion,
)

LOGGER = logging.getLogger("chatterbox.fastapi")

if Path.cwd() != BASE_DIR:
    os.chdir(BASE_DIR)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = BASE_DIR / "api_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CHATTER_DEFAULTS = default_settings()
GENERATION_LOCK = asyncio.Lock()


def _ensure_nltk_resource(resource: str, package: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            LOGGER.info("Downloading NLTK resource '%s' (package '%s')...", resource, package)
            nltk.download(package, quiet=True)
        except Exception as exc:  # pragma: no cover - network failure path
            raise RuntimeError(
                f"Unable to download NLTK resource '{resource}'. "
                "Install it manually with `python -m nltk.downloader punkt punkt_tab`."
            ) from exc


for resource, package in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]:
    try:
        _ensure_nltk_resource(resource, package)
    except RuntimeError as exc:  # pragma: no cover
        LOGGER.warning(str(exc))

__all__ = [
    "BASE_DIR",
    "CHATTER_DEFAULTS",
    "DEVICE",
    "GENERATION_LOCK",
    "OUTPUT_DIR",
    "TEMP_DIR",
    "LOGGER",
    "chatter_voice_conversion",
    "generate_batch_tts",
    "load_settings",
    "save_settings",
    "whisper_model_map",
]
