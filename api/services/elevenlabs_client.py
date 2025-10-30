"""Thin wrapper around the ElevenLabs SDK used for speech-to-speech cloning."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import HTTPException

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

from ..config import settings


_elevenlabs_client: Optional[ElevenLabs] = None


def _get_client() -> ElevenLabs:
    global _elevenlabs_client
    if _elevenlabs_client is not None:
        return _elevenlabs_client

    api_key = settings.elevenlabs_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="ElevenLabs API key is not configured")

    _elevenlabs_client = ElevenLabs(api_key=api_key)
    return _elevenlabs_client


def generate_clone_bytes(
    *,
    source_path: Path,
    voice_id: str,
    model_id: str,
    voice_settings: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Submit an audio file to ElevenLabs speech-to-speech and return the rendered bytes."""
    client = _get_client()

    settings_payload: Optional[str] = None
    if voice_settings:
        try:
            settings_payload = VoiceSettings(**voice_settings).json(exclude_none=True)
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=400, detail=f"Invalid ElevenLabs voice settings: {exc}") from exc

    try:
        with source_path.open("rb") as handle:
            audio_tuple = (source_path.name, handle, "audio/wav")
            stream = client.speech_to_speech.convert(
                voice_id=voice_id,
                model_id=model_id,
                audio=audio_tuple,  # type: ignore[arg-type]
                voice_settings=settings_payload,
            )
            chunks = bytearray()
            for chunk in stream:
                chunks.extend(chunk)
    except ApiError as exc:
        body = exc.body if isinstance(exc.body, str) else repr(exc.body)
        raise HTTPException(status_code=502, detail=f"ElevenLabs request failed: {body}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"ElevenLabs network error: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=502, detail=f"Unexpected ElevenLabs error: {exc}") from exc

    return bytes(chunks)


def detect_audio_extension(data: bytes) -> str:
    """Best-effort detection of audio container based on magic bytes."""
    if data.startswith(b"RIFF"):
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data.startswith(b"OggS"):
        return ".ogg"
    if data[:3] == b"ID3" or data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return ".mp3"
    return ".mp3"
