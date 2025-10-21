"""Utilities for single-line TTS generation and optional cloning."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from fastapi import HTTPException
from pydub import AudioSegment
import soundfile as sf

from .core import generate_batch_tts
from .data_service import get_clone_voice_path, get_reference_audio_path, list_clone_voices
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


def generate_line_audio(payload: LineGenerationRequest) -> LineGenerationResponse:
    options = TTSOptions(**payload.options.model_dump())
    options.export_formats = ensure_wav_in_exports(options.export_formats)

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
    raw_file_results = [to_file_result(path) for path in raw_outputs]

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

    final_results = [to_file_result(path) for path in final_files if path.exists()]

    metadata = {
        "reference_voice": payload.reference_voice,
        "reference_style": payload.reference_style,
        "reference_audio": payload.reference_audio,
        "tag": payload.tag or "",
    }

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
