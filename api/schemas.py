"""Pydantic models used by the FastAPI endpoints."""
from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator, validator

from .core import CHATTER_DEFAULTS


class Base64File(BaseModel):
    """Payload carrier for file-like content.

    Callers may supply **either** a base64 payload via ``data`` or a server-side
    filesystem path via ``path`` during local testing. Exactly one of these
    fields must be set.
    """

    filename: str
    data: Optional[str] = None
    path: Optional[str] = None

    @field_validator("path", mode="before")
    def _expand_path(cls, value: Optional[str]) -> Optional[str]:
        return os.path.expanduser(value) if value else value

    @model_validator(mode="after")
    def _validate_choice(cls, model: "Base64File") -> "Base64File":
        if bool(model.data) == bool(model.path):
            raise ValueError("provide exactly one of 'data' or 'path'")
        return model

    def write_to(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        suffix = Path(self.filename).suffix
        target_path = directory / f"{uuid.uuid4().hex}{suffix}"

        if self.data:
            try:
                decoded = base64.b64decode(self.data)
            except Exception as exc:  # pragma: no cover - validation guard
                raise HTTPException(status_code=400, detail=f"Failed to decode base64 for {self.filename}: {exc}")
            target_path.write_bytes(decoded)
            return target_path

        if self.path:
            src = Path(self.path)
            if not src.exists():
                raise HTTPException(status_code=400, detail=f"File path not found: {self.path}")
            target_path.write_bytes(src.read_bytes())
            return target_path

        raise HTTPException(status_code=400, detail="No input data or path provided")


class SoundWordReplacement(BaseModel):
    pattern: str
    replacement: Optional[str] = ""

    @validator("pattern")
    def validate_pattern(cls, value: str) -> str:
        if not value:
            raise ValueError("pattern may not be empty")
        return value


class TTSOptions(BaseModel):
    exaggeration: float = Field(default=CHATTER_DEFAULTS["exaggeration_slider"], ge=0.0, le=1.0)
    temperature: float = Field(default=CHATTER_DEFAULTS["temp_slider"], ge=0.0)
    seed: int = Field(default=CHATTER_DEFAULTS["seed_input"])
    cfg_weight: float = Field(default=CHATTER_DEFAULTS["cfg_weight_slider"], ge=0.0)
    use_pyrnnoise: bool = Field(default=CHATTER_DEFAULTS["use_pyrnnoise_checkbox"])
    use_auto_editor: bool = Field(default=CHATTER_DEFAULTS["use_auto_editor_checkbox"])
    auto_editor_threshold: float = Field(default=CHATTER_DEFAULTS["threshold_slider"], ge=0.0)
    auto_editor_margin: float = Field(default=CHATTER_DEFAULTS["margin_slider"], ge=0.0)
    export_formats: List[str] = Field(default_factory=lambda: list(CHATTER_DEFAULTS["export_format_checkboxes"]))
    enable_batching: bool = Field(default=CHATTER_DEFAULTS["enable_batching_checkbox"])
    smart_batch_short_sentences: bool = Field(default=CHATTER_DEFAULTS["smart_batch_short_sentences_checkbox"])
    to_lowercase: bool = Field(default=CHATTER_DEFAULTS["to_lowercase_checkbox"])
    normalize_spacing: bool = Field(default=CHATTER_DEFAULTS["normalize_spacing_checkbox"])
    fix_dot_letters: bool = Field(default=CHATTER_DEFAULTS["fix_dot_letters_checkbox"])
    remove_reference_numbers: bool = Field(default=CHATTER_DEFAULTS["remove_reference_numbers_checkbox"])
    keep_original_wav: bool = Field(default=CHATTER_DEFAULTS["keep_original_checkbox"])
    disable_watermark: bool = Field(default=CHATTER_DEFAULTS["disable_watermark_checkbox"])
    num_generations: int = Field(default=CHATTER_DEFAULTS["num_generations_input"], ge=1)
    normalize_audio: bool = Field(default=CHATTER_DEFAULTS["normalize_audio_checkbox"])
    normalize_method: str = Field(default=CHATTER_DEFAULTS["normalize_method_dropdown"])
    normalize_level: float = Field(default=CHATTER_DEFAULTS["normalize_level_slider"])
    normalize_true_peak: float = Field(default=CHATTER_DEFAULTS["normalize_tp_slider"])
    normalize_lra: float = Field(default=CHATTER_DEFAULTS["normalize_lra_slider"])
    num_candidates: int = Field(default=CHATTER_DEFAULTS["num_candidates_slider"], ge=1)
    max_attempts: int = Field(default=CHATTER_DEFAULTS["max_attempts_slider"], ge=1)
    bypass_whisper: bool = Field(default=CHATTER_DEFAULTS["bypass_whisper_checkbox"])
    whisper_model: str = Field(default=CHATTER_DEFAULTS["whisper_model_dropdown"])
    use_faster_whisper: bool = Field(default=CHATTER_DEFAULTS["use_faster_whisper_checkbox"])
    enable_parallel: bool = Field(default=CHATTER_DEFAULTS["enable_parallel_checkbox"])
    num_parallel_workers: int = Field(default=CHATTER_DEFAULTS["num_parallel_workers_slider"], ge=1)
    use_longest_transcript_on_fail: bool = Field(default=CHATTER_DEFAULTS["use_longest_transcript_on_fail_checkbox"])
    reference_audio_path: Optional[str] = None
    sound_words: Optional[List[SoundWordReplacement]] = None
    sound_words_text: Optional[str] = None
    generate_separate_audio_files: bool = False

    @validator("normalize_method")
    def validate_normalize_method(cls, value: str) -> str:
        value_lower = value.lower()
        if value_lower not in {"ebu", "peak"}:
            raise ValueError("normalize_method must be either 'ebu' or 'peak'")
        return value_lower

    @validator("export_formats", each_item=True)
    def standardize_export_formats(cls, value: str) -> str:
        fmt = value.strip().lower()
        if fmt not in {"wav", "mp3", "flac"}:
            raise ValueError("export_formats can only contain 'wav', 'mp3', or 'flac'")
        return fmt

    @validator("export_formats")
    def ensure_export_formats_not_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one export format must be specified")
        seen = set()
        unique = []
        for item in value:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique


class TTSRequest(BaseModel):
    text: Optional[str] = None
    text_files: Optional[List[Base64File]] = None
    reference_audio: Optional[Base64File] = None
    options: TTSOptions = Field(default_factory=TTSOptions)
    return_audio_base64: bool = False
    include_settings_files: bool = True


class FileResult(BaseModel):
    path: str
    url: Optional[str]
    size_bytes: int
    duration_seconds: Optional[float] = None
    base64: Optional[str] = None


class TTSResponse(BaseModel):
    outputs: List[FileResult]
    settings: List[FileResult]


class VoiceConversionRequest(BaseModel):
    input_audio: Base64File
    target_voice_audio: Base64File
    chunk_seconds: float = 60.0
    overlap_seconds: float = 0.1
    disable_watermark: bool = True
    pitch_shift: int = 0
    export_formats: List[str] = Field(default_factory=lambda: ["wav"])
    return_audio_base64: bool = False

    @validator("export_formats", each_item=True)
    def validate_vc_export_format(cls, value: str) -> str:
        fmt = value.strip().lower()
        if fmt not in {"wav", "mp3", "flac"}:
            raise ValueError("export_formats can only contain 'wav', 'mp3', or 'flac'")
        return fmt

    @validator("export_formats")
    def ensure_vc_format_not_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one export format must be specified")
        seen = set()
        ordered = []
        for fmt in value:
            if fmt not in seen:
                ordered.append(fmt)
                seen.add(fmt)
        return ordered


class VoiceConversionResponse(BaseModel):
    outputs: List[FileResult]


__all__ = [
    "Base64File",
    "FileResult",
    "SoundWordReplacement",
    "TTSOptions",
    "TTSRequest",
    "TTSResponse",
    "VoiceConversionRequest",
    "VoiceConversionResponse",
]
