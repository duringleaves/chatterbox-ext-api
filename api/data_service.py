"""Helpers for reading station formats, scripts, and voice assets."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from .config import settings

IGNORED_FILES = {".DS_Store"}


def _relative_to_data_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(settings.resolved_data_dir))
    except ValueError:
        return str(path)


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def _tag_variants(tag: str) -> List[str]:
    normalized = tag.strip().lower()
    if not normalized:
        return []
    candidates = [normalized]
    candidates.append(normalized.replace(" ", "_"))
    candidates.append(normalized.replace("-", "_"))
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if slug:
        candidates.append(slug)
    # Preserve order while removing duplicates
    seen: set[str] = set()
    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _safe_listdir(directory: Path) -> List[Path]:
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    return sorted(p for p in directory.iterdir() if p.name not in IGNORED_FILES)


def _ensure_inside(base: Path, target: Path) -> Path:
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    return target


def list_station_formats() -> List[Dict[str, str]]:
    base = settings.resolved_data_dir / "script_templates"
    results = []
    for path in _safe_listdir(base):
        if path.suffix.lower() != ".json":
            continue
        results.append({
            "id": path.stem,
            "filename": path.name,
        })
    return results


def load_station_template(template_id: str) -> Dict:
    base = settings.resolved_data_dir / "script_templates"
    path = _ensure_inside(base, (base / f"{template_id}.json").resolve())
    if not path.exists():
        raise HTTPException(status_code=404, detail="Station template not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_sample_stations() -> List[Dict[str, str]]:
    base = settings.resolved_data_dir / "sample_stations"
    results = []
    for path in _safe_listdir(base):
        if path.suffix.lower() != ".json":
            continue
        results.append({
            "id": path.stem,
            "filename": path.name,
        })
    return results


def load_sample_station(station_id: str) -> Dict:
    base = settings.resolved_data_dir / "sample_stations"
    path = _ensure_inside(base, (base / f"{station_id}.json").resolve())
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample station not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_sample_scripts() -> List[Dict[str, str]]:
    base = settings.resolved_data_dir / "sample_scripts"
    results = []
    for path in _safe_listdir(base):
        if path.suffix.lower() != ".txt":
            continue
        results.append({
            "id": path.stem,
            "filename": path.name,
        })
    return results


def load_sample_script(script_id: str) -> str:
    base = settings.resolved_data_dir / "sample_scripts"
    path = _ensure_inside(base, (base / f"{script_id}.txt").resolve())
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample script not found")
    return path.read_text(encoding="utf-8")


def list_reference_voices() -> List[Dict]:
    base = settings.resolved_data_dir / "reference_voices"
    voices = []
    for voice_dir in _safe_listdir(base):
        if not voice_dir.is_dir():
            continue
        styles = []
        for style_dir in _safe_listdir(voice_dir):
            if not style_dir.is_dir():
                continue
            audio_files = [p.name for p in _safe_listdir(style_dir) if p.suffix.lower() in {".wav", ".mp3", ".flac"}]
            settings_map: Dict[str, Dict] = {}
            default_settings: Optional[Dict] = None
            for json_path in _safe_listdir(style_dir):
                if json_path.suffix.lower() != ".json":
                    continue
                name = json_path.stem
                if name == "chatterbox_settings":
                    default_settings = json.loads(json_path.read_text(encoding="utf-8"))
                elif name.startswith("chatterbox_settings_"):
                    tag = name.replace("chatterbox_settings_", "").lower()
                    settings_map[tag] = json.loads(json_path.read_text(encoding="utf-8"))
            styles.append({
                "name": style_dir.name,
                "audio_files": audio_files,
                "default_settings": default_settings,
                "tag_settings": settings_map,
            })
        voices.append({
            "name": voice_dir.name,
            "styles": styles,
        })
    return voices


def resolve_chatterbox_settings(
    voice: str,
    style: str,
    tag: Optional[str] = None,
) -> Dict[str, Optional[Any]]:
    base = settings.resolved_data_dir / "reference_voices"
    style_dir = (base / voice / style).resolve()
    try:
        style_dir = _ensure_inside(base, style_dir)
    except HTTPException:
        return {
            "default_settings": None,
            "default_path": None,
            "tag_settings": None,
            "tag_path": None,
            "resolved_settings": None,
        }

    default_path = (style_dir / "chatterbox_settings.json").resolve()
    try:
        default_path = _ensure_inside(base, default_path)
    except HTTPException:
        default_data = None
    else:
        default_data = _read_json_if_exists(default_path)

    tag_data = None
    tag_path: Optional[Path] = None
    if tag:
        for variant in _tag_variants(tag):
            candidate = (style_dir / f"chatterbox_settings_{variant}.json").resolve()
            try:
                candidate = _ensure_inside(base, candidate)
            except HTTPException:
                continue
            tag_data = _read_json_if_exists(candidate)
            if tag_data is not None:
                tag_path = candidate
                break

    resolved: Optional[Dict[str, Any]] = None
    if default_data or tag_data:
        resolved = {}
        if default_data:
            resolved.update(default_data)
        if tag_data:
            resolved.update(tag_data)

    return {
        "default_settings": default_data,
        "default_path": _relative_to_data_root(default_path) if default_data else None,
        "tag_settings": tag_data,
        "tag_path": _relative_to_data_root(tag_path) if tag_path else None,
        "resolved_settings": resolved,
    }


def get_reference_audio_path(voice: str, style: str, filename: str) -> Path:
    base = settings.resolved_data_dir / "reference_voices"
    path = (base / voice / style / filename).resolve()
    return _ensure_inside(base, path)


def _elevenlabs_clone_dir() -> Path:
    return settings.resolved_data_dir / "clone_voices" / "elevenlabs"


def list_clone_voices() -> List[Dict[str, Any]]:
    base = _elevenlabs_clone_dir()
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"ElevenLabs voices directory not found: {base}")

    result: List[Dict[str, Any]] = []
    for voice_file in _safe_listdir(base):
        if voice_file.suffix.lower() != ".json":
            continue
        try:
            content = json.loads(voice_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for key, payload in content.items():
            voice_id = payload.get("voice_id")
            if not voice_id:
                continue
            name = payload.get("name") or key
            description = payload.get("description") or ""
            voice_settings = payload.get("voice_settings") or {}
            result.append({
                "id": key,
                "name": name,
                "description": description,
                "voice_id": voice_id,
                "voice_settings": voice_settings,
                "source_file": str(_relative_to_data_root(voice_file)),
            })
    result.sort(key=lambda item: item["name"].lower())
    return result


def get_clone_voice_config(voice_key: str) -> Dict[str, Any]:
    voices = list_clone_voices()
    for voice in voices:
        if voice["id"] == voice_key:
            return voice
    raise HTTPException(status_code=404, detail="Clone voice not found")


def get_clone_voice_path(voice: str, filename: str) -> Path:
    base = settings.resolved_data_dir / "clone_voices"
    path = (base / voice / filename).resolve()
    return _ensure_inside(base, path)
