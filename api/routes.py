"""Route definitions for the FastAPI service."""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import List

from fastapi import APIRouter, HTTPException
import soundfile as sf
from pydub import AudioSegment
from starlette.concurrency import run_in_threadpool

from .core import (
    CHATTER_DEFAULTS,
    DEVICE,
    GENERATION_LOCK,
    OUTPUT_DIR,
    TEMP_DIR,
    chatter_voice_conversion,
    generate_batch_tts,
    load_settings,
    save_settings,
    whisper_model_map,
)
from .schemas import (
    FileResult,
    TTSRequest,
    TTSResponse,
    VoiceConversionRequest,
    VoiceConversionResponse,
)
from .utils import (
    build_sound_words_text,
    collect_settings_files,
    resolve_whisper_label,
    safe_cleanup,
    to_file_result,
)

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE}


@router.get("/tts/defaults")
def get_default_settings() -> dict:
    return dict(CHATTER_DEFAULTS)


@router.get("/tts/settings")
def get_saved_settings() -> dict:
    return load_settings()


@router.post("/tts/settings")
def update_settings(settings_payload: dict) -> dict:
    save_settings(settings_payload)
    return {"status": "saved"}


@router.post("/tts/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest) -> TTSResponse:
    if not request.text and not request.text_files:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'text_files'.")

    temp_paths: List[Path] = []
    try:
        text_files_objects: List[SimpleNamespace] | None = None
        if request.text_files:
            text_files_objects = []
            for payload in request.text_files:
                path = payload.write_to(TEMP_DIR)
                temp_paths.append(path)
                text_files_objects.append(SimpleNamespace(name=str(path)))

        audio_prompt_path = request.options.reference_audio_path
        if request.reference_audio:
            audio_path = request.reference_audio.write_to(TEMP_DIR)
            temp_paths.append(audio_path)
            audio_prompt_path = str(audio_path)

        whisper_label = resolve_whisper_label(request.options.whisper_model)
        sound_words_field = build_sound_words_text(request.options)
        export_formats = request.options.export_formats

        async with GENERATION_LOCK:
            output_paths: List[str] = await run_in_threadpool(
                generate_batch_tts,
                request.text or "",
                text_files_objects,
                audio_prompt_path,
                request.options.exaggeration,
                request.options.temperature,
                request.options.seed,
                request.options.cfg_weight,
                request.options.use_pyrnnoise,
                request.options.use_auto_editor,
                request.options.auto_editor_threshold,
                request.options.auto_editor_margin,
                export_formats,
                request.options.enable_batching,
                request.options.to_lowercase,
                request.options.normalize_spacing,
                request.options.fix_dot_letters,
                request.options.remove_reference_numbers,
                request.options.keep_original_wav,
                request.options.smart_batch_short_sentences,
                request.options.disable_watermark,
                request.options.num_generations,
                request.options.normalize_audio,
                request.options.normalize_method,
                request.options.normalize_level,
                request.options.normalize_true_peak,
                request.options.normalize_lra,
                request.options.num_candidates,
                request.options.max_attempts,
                request.options.bypass_whisper,
                whisper_label,
                request.options.enable_parallel,
                request.options.num_parallel_workers,
                request.options.use_longest_transcript_on_fail,
                sound_words_field,
                request.options.use_faster_whisper,
                request.options.generate_separate_audio_files,
            )

        outputs = [to_file_result(Path(path), include_base64=request.return_audio_base64) for path in output_paths]
        settings_files = collect_settings_files(output_paths) if request.include_settings_files else []
        return TTSResponse(outputs=outputs, settings=settings_files)
    finally:
        safe_cleanup(temp_paths)


@router.post("/voice/convert", response_model=VoiceConversionResponse)
async def convert_voice(request: VoiceConversionRequest) -> VoiceConversionResponse:
    temp_paths: List[Path] = []
    try:
        input_path = request.input_audio.write_to(TEMP_DIR)
        target_path = request.target_voice_audio.write_to(TEMP_DIR)
        temp_paths.extend([input_path, target_path])

        def _run_conversion() -> tuple[int, List[Path]]:
            sr, audio = chatter_voice_conversion(
                str(input_path),
                str(target_path),
                chunk_sec=request.chunk_seconds,
                overlap_sec=request.overlap_seconds,
                disable_watermark=request.disable_watermark,
                pitch_shift=request.pitch_shift,
            )

            timestamp = uuid.uuid4().hex
            base_name = f"vc_{timestamp}"
            wav_path = OUTPUT_DIR / f"{base_name}.wav"
            sf.write(str(wav_path), audio, sr)

            generated_paths = [wav_path]
            if "wav" not in request.export_formats:
                generated_paths = []

            for fmt in request.export_formats:
                if fmt == "wav":
                    generated_paths.append(wav_path)
                else:
                    segment = AudioSegment.from_wav(wav_path)
                    extra_path = OUTPUT_DIR / f"{base_name}.{fmt}"
                    export_kwargs = {"bitrate": "320k"} if fmt == "mp3" else {}
                    segment.export(extra_path, format=fmt, **export_kwargs)
                    generated_paths.append(extra_path)

            if "wav" not in request.export_formats:
                try:
                    wav_path.unlink()
                except Exception:
                    pass

            return sr, generated_paths

        _, generated_paths = await run_in_threadpool(_run_conversion)
        outputs = [to_file_result(path, include_base64=request.return_audio_base64) for path in generated_paths]
        return VoiceConversionResponse(outputs=outputs)
    finally:
        safe_cleanup(temp_paths)


@router.get("/whisper/models")
def list_whisper_models() -> List[dict]:
    return [{"label": label, "code": code} for label, code in whisper_model_map.items()]
