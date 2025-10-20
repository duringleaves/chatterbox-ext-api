"""Utility helpers for the FastAPI service."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List

import soundfile as sf
from fastapi import HTTPException

from .core import BASE_DIR, OUTPUT_DIR, whisper_model_map
from .schemas import FileResult, TTSOptions


def to_file_result(path: Path, include_base64: bool = False) -> FileResult:
    if not path.exists():  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Expected output {path} is missing")

    try:
        relative = path.resolve().relative_to(BASE_DIR)
    except ValueError:  # pragma: no cover
        relative = path.name

    url = None
    try:
        rel_to_output = path.resolve().relative_to(OUTPUT_DIR)
        url = f"/media/{rel_to_output.as_posix()}"
    except ValueError:
        pass

    try:
        info = sf.info(str(path))
        duration = getattr(info, "duration", None)
    except Exception:
        duration = None

    file_result = FileResult(
        path=str(relative),
        url=url,
        size_bytes=path.stat().st_size,
        duration_seconds=duration,
    )

    if include_base64:
        file_result.base64 = base64.b64encode(path.read_bytes()).decode("utf-8")

    return file_result


def resolve_whisper_label(name: str) -> str:
    if name in whisper_model_map:
        return name
    inverse = {code: label for label, code in whisper_model_map.items()}
    if name in inverse:
        return inverse[name]
    raise HTTPException(status_code=400, detail=f"Unknown Whisper model '{name}'")


def build_sound_words_text(options: TTSOptions) -> str:
    if options.sound_words_text is not None:
        return options.sound_words_text
    if not options.sound_words:
        return ""
    fragments = []
    for item in options.sound_words:
        if item.replacement:
            fragments.append(f"{item.pattern}=>{item.replacement}")
        else:
            fragments.append(item.pattern)
    return "\n".join(fragments)


def safe_cleanup(paths: List[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def collect_settings_files(output_paths: List[str]) -> List[FileResult]:
    seen = set()
    results: List[FileResult] = []
    for path_str in output_paths:
        base = Path(path_str)
        stem = base.with_suffix("")
        if stem in seen:
            continue
        seen.add(stem)
        json_path = stem.with_suffix(".settings.json")
        csv_path = stem.with_suffix(".settings.csv")
        for candidate in [json_path, csv_path]:
            if candidate.exists():
                results.append(to_file_result(candidate))
    return results


__all__ = [
    "build_sound_words_text",
    "collect_settings_files",
    "resolve_whisper_label",
    "safe_cleanup",
    "to_file_result",
]
