"""Utilities for single-line TTS generation and optional cloning."""
from __future__ import annotations

import datetime
import math
import random
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
from fastapi import HTTPException
from pydub import AudioSegment

from .core import LOGGER, generate_batch_tts
from .data_service import (
    get_clone_voice_path,
    get_reference_audio_path,
    list_clone_voices,
    resolve_chatterbox_settings,
)
from .schemas import LineGenerationRequest, LineGenerationResponse, TTSOptions
from .utils import build_sound_words_text, resolve_whisper_label, to_file_result


def _tts_options_to_args(options: TTSOptions, sound_words_field: str) -> Tuple:
    whisper_label = resolve_whisper_label(options.whisper_model)
    return (
        options.exaggeration,
        options.temperature,
        options.seed,
        options.cfg_weight,
        options.use_pyrnnoise,
        options.use_auto_editor,
        options.auto_editor_threshold,
        options.auto_editor_margin,
        options.export_formats,
        options.enable_batching,
        options.to_lowercase,
        options.normalize_spacing,
        options.fix_dot_letters,
        options.remove_reference_numbers,
        options.keep_original_wav,
        options.smart_batch_short_sentences,
        options.disable_watermark,
        options.num_generations,
        options.normalize_audio,
        options.normalize_method,
        options.normalize_level,
        options.normalize_true_peak,
        options.normalize_lra,
        options.num_candidates,
        options.max_attempts,
        options.bypass_whisper,
        whisper_label,
        options.enable_parallel,
        options.num_parallel_workers,
        options.use_longest_transcript_on_fail,
        sound_words_field,
        options.use_faster_whisper,
        options.generate_separate_audio_files,
    )


def ensure_wav_in_exports(exports: List[str]) -> List[str]:
    lowered = [fmt.lower() for fmt in exports]
    if "wav" not in lowered:
        return exports + ["wav"]
    return exports


def _slugify_component(value: Optional[str], fallback: str, max_len: int = 60) -> str:
    base = value or ""
    normalized = unicodedata.normalize("NFKD", base)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"\s+", " ", ascii_only).strip()
    if max_len > 0 and len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or fallback


def _extract_index_from_line_id(line_id: Optional[str]) -> Optional[int]:
    if not line_id:
        return None
    matches = re.findall(r"(\d+)", line_id)
    if matches:
        return int(matches[-1])
    return None


def _detect_seed_from_paths(paths: List[Path], fallback: int) -> int:
    for path in paths:
        match = re.search(r"_seed(\d+)", path.stem)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return fallback


def _unique_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _rename_generated_files(paths: List[Path], base_name: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not base_name:
        return {str(path): path for path in paths}
    seen_ext: Dict[str, int] = {}
    for original in paths:
        ext = original.suffix.lower()
        seen_ext[ext] = seen_ext.get(ext, 0) + 1
        suffix = "" if seen_ext[ext] == 1 else f"_{seen_ext[ext]:02d}"
        candidate = original.with_name(f"{base_name}{suffix}{ext}")
        if candidate.exists() and candidate != original:
            try:
                candidate.unlink()
            except Exception:
                pass
        if candidate != original:
            original.rename(candidate)
        mapping[str(original)] = candidate
    return mapping


def generate_line_audio(payload: LineGenerationRequest) -> LineGenerationResponse:
    options = TTSOptions(**payload.options.model_dump())
    options.export_formats = ensure_wav_in_exports(options.export_formats)

    chatterbox_info = resolve_chatterbox_settings(
        payload.reference_voice,
        payload.reference_style,
        payload.tag,
    )

    apply_reference_defaults = bool(getattr(options, "force_reference_defaults", True))

    initial_settings = {
        "temperature": options.temperature,
        "cfg_weight": options.cfg_weight,
        "exaggeration": options.exaggeration,
    }

    expected_settings = {}
    resolved = chatterbox_info.get("resolved_settings") or {}
    overrides_applied = {}
    for key in ("temperature", "cfg_weight", "exaggeration"):
        value = resolved.get(key)
        if isinstance(value, (int, float)):
            expected_value = float(value)
            expected_settings[key] = expected_value
            current_value = float(getattr(options, key))
            if apply_reference_defaults and not math.isclose(current_value, expected_value, rel_tol=1e-6, abs_tol=1e-6):
                overrides_applied[key] = {
                    "requested": current_value,
                    "config": expected_value,
                }
                setattr(options, key, expected_value)

    applied_settings = {
        "temperature": options.temperature,
        "cfg_weight": options.cfg_weight,
        "exaggeration": options.exaggeration,
    }

    requested_str = ", ".join(f"{key}={value:.3f}" for key, value in initial_settings.items())
    applied_str = ", ".join(f"{key}={value:.3f}" for key, value in applied_settings.items())
    expected_str = (
        ", ".join(f"{key}={value:.3f}" for key, value in expected_settings.items())
        if expected_settings
        else "n/a"
    )
    sources = []
    tag_path = chatterbox_info.get("tag_path")
    default_path = chatterbox_info.get("default_path")
    if tag_path:
        sources.append(f"tag:{tag_path}")
    if default_path:
        sources.append(f"default:{default_path}")
    source_str = ", ".join(sources) if sources else "not found"

    LOGGER.info(
        "Line %s using %s/%s (tag=%s) | requested [%s] | applied [%s] | config [%s] | sources=%s",
        payload.line_id,
        payload.reference_voice,
        payload.reference_style,
        payload.tag or "default",
        requested_str,
        applied_str,
        expected_str,
        source_str,
    )

    if overrides_applied:
        diff_str = "; ".join(
            f"{key}: {vals['requested']:.3f} -> {vals['config']:.3f}" for key, vals in overrides_applied.items()
        )
        LOGGER.warning(
            "Line %s overrides applied to match chatterbox settings: %s",
            payload.line_id,
            diff_str,
        )

    reference_path = get_reference_audio_path(
        payload.reference_voice,
        payload.reference_style,
        payload.reference_audio,
    )

    sound_words_field = payload.sound_words_field or build_sound_words_text(options)

    args = _tts_options_to_args(options, sound_words_field)

    output_paths = generate_batch_tts(
        payload.text,
        None,
        str(reference_path),
        *args,
    )

    raw_outputs = [Path(path) for path in output_paths]

    wav_candidates = [path for path in raw_outputs if path.suffix.lower() == ".wav"]
    if not wav_candidates:
        raise HTTPException(status_code=500, detail="TTS pipeline did not produce a WAV output")
    base_wav = wav_candidates[0]

    final_files: List[Path] = []

    clone_file = payload.clone_audio
    if payload.clone_voice:
        available_clones = {entry["name"]: entry["files"] for entry in list_clone_voices()}
        if payload.clone_voice not in available_clones:
            raise HTTPException(status_code=404, detail="Clone voice not found")
        if not clone_file:
            clone_file = random.choice(available_clones[payload.clone_voice])
        clone_path = get_clone_voice_path(payload.clone_voice, clone_file)

        from chatterbox.service import voice_conversion  # local import to avoid circular

        sr, audio = voice_conversion(
            str(base_wav),
            str(clone_path),
            pitch_shift=int(payload.clone_pitch or 0),
        )
        clone_wav = base_wav.with_name(f"{base_wav.stem}_clone.wav")
        sf.write(clone_wav, audio, sr)
        final_files.append(clone_wav)

        mp3_path = clone_wav.with_suffix(".mp3")
        segment = AudioSegment.from_wav(clone_wav)
        segment.export(mp3_path, format="mp3", bitrate="320k")
        final_files.append(mp3_path)
        # keep raw wav so UI can inspect waveform
        final_files.append(base_wav)
    else:
        mp3_path = base_wav.with_suffix(".mp3")
        segment = AudioSegment.from_wav(base_wav)
        segment.export(mp3_path, format="mp3", bitrate="320k")
        final_files.extend([base_wav, mp3_path])

    requested_queue_position = payload.queue_position or _extract_index_from_line_id(payload.line_id) or 0
    seed_used = _detect_seed_from_paths(raw_outputs, int(options.seed or 0))

    text_component = _slugify_component((payload.text or "")[:80], "line", 60)
    reference_source = Path(payload.reference_audio).stem if payload.reference_audio else "reference"
    reference_component = _slugify_component(reference_source, "reference", 40)
    components = []
    if requested_queue_position:
        components.append(f"{requested_queue_position:03d}")
    components.extend([text_component, reference_component, f"seed{seed_used}"])
    base_name = "_".join(filter(None, components))
    if len(base_name) > 180:
        base_name = base_name[:180]

    combined_paths = _unique_paths(raw_outputs + final_files)
    rename_map = _rename_generated_files(combined_paths, base_name)
    raw_outputs = [rename_map.get(str(path), path) for path in raw_outputs]
    final_files = [rename_map.get(str(path), path) for path in final_files]

    raw_file_results = [to_file_result(path) for path in raw_outputs if path.exists()]
    final_results = [to_file_result(path) for path in final_files if path.exists()]

    metadata = {
        "reference_voice": payload.reference_voice,
        "reference_style": payload.reference_style,
        "reference_audio": payload.reference_audio,
        "tag": payload.tag or "",
        "reference_defaults_applied": "true" if apply_reference_defaults else "false",
    }

    metadata.update(
        {
            "tts_settings_request": requested_str,
            "tts_settings_applied": applied_str,
            "tts_temperature": f"{applied_settings['temperature']:.3f}",
            "tts_cfg_weight": f"{applied_settings['cfg_weight']:.3f}",
            "tts_exaggeration": f"{applied_settings['exaggeration']:.3f}",
            "tts_settings_expected": expected_str,
            "tts_settings_source": source_str,
            "tts_settings_match": "false" if overrides_applied else "true",
            "tts_seed_used": str(seed_used),
            "queue_position": str(requested_queue_position) if requested_queue_position else "",
            "output_basename": base_name,
        }
    )

    if default_path:
        metadata["tts_default_settings_file"] = str(default_path)
    if tag_path:
        metadata["tts_tag_settings_file"] = str(tag_path)
    if overrides_applied:
        metadata["tts_settings_overrides"] = "; ".join(
            f"{key}:{vals['requested']:.3f}->{vals['config']:.3f}" for key, vals in overrides_applied.items()
        )

    generation_id = uuid.uuid4().hex
    generated_at = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    metadata["generation_id"] = generation_id
    metadata["generated_at"] = generated_at

    if payload.clone_voice:
        metadata.update({
            "clone_voice": payload.clone_voice,
            "clone_audio": clone_file,
        })

    return LineGenerationResponse(
        line_id=payload.line_id,
        raw_outputs=raw_file_results,
        final_outputs=final_results,
        metadata=metadata,
    )
