"""Helpers for reading station formats, scripts, and voice assets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from .config import settings

IGNORED_FILES = {".DS_Store"}


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


def get_reference_audio_path(voice: str, style: str, filename: str) -> Path:
    base = settings.resolved_data_dir / "reference_voices"
    path = (base / voice / style / filename).resolve()
    return _ensure_inside(base, path)


def list_clone_voices() -> List[Dict[str, str]]:
    base = settings.resolved_data_dir / "clone_voices"
    result = []
    for voice_dir in _safe_listdir(base):
        if voice_dir.is_dir():
            files = [p.name for p in _safe_listdir(voice_dir) if p.suffix.lower() in {".wav", ".mp3", ".flac"}]
            result.append({
                "name": voice_dir.name,
                "files": files,
            })
    return result


def get_clone_voice_path(voice: str, filename: str) -> Path:
    base = settings.resolved_data_dir / "clone_voices"
    path = (base / voice / filename).resolve()
    return _ensure_inside(base, path)
