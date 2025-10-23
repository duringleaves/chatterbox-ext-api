"""Route definitions for the FastAPI service."""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import List

import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydub import AudioSegment
from starlette.concurrency import run_in_threadpool

from .auth import require_api_key
from .config import settings
from .core import (
    CHATTER_DEFAULTS,
    DEVICE,
    GENERATION_LOCK,
    LOGGER,
    OUTPUT_DIR,
    TEMP_DIR,
    chatter_voice_conversion,
    generate_batch_tts,
    load_settings,
    save_settings,
    whisper_model_map,
)
from .data_service import (
    get_clone_voice_path,
    get_reference_audio_path,
    list_clone_voices,
    list_reference_voices,
    list_sample_scripts,
    list_sample_stations,
    list_station_formats,
    load_sample_script,
    load_sample_station,
    load_station_template,
)
from .jobs import job_manager
from .line_processing import generate_line_audio
from .schemas import (
    AnalyzeScriptRequest,
    AnalyzeScriptResponse,
    BatchCreateRequest,
    BatchJobStatus,
    FileResult,
    LineGenerationRequest,
    LineGenerationResponse,
    TTSRequest,
    TTSResponse,
    VoiceConversionRequest,
    VoiceConversionResponse,
)
from .services.text_processing import TextProcessingError, text_preprocessor
from .utils import (
    build_sound_words_text,
    collect_settings_files,
    resolve_whisper_label,
    safe_cleanup,
    to_file_result,
)

router = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE}


@protected.get("/tts/defaults")
def get_default_settings() -> dict:
    return dict(CHATTER_DEFAULTS)


@protected.get("/tts/settings")
def get_saved_settings() -> dict:
    return load_settings()


@protected.post("/tts/settings")
def update_settings(settings_payload: dict) -> dict:
    save_settings(settings_payload)
    return {"status": "saved"}


@protected.post("/tts/generate", response_model=TTSResponse)
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

        LOGGER.info(
            "/tts/generate | prompt=%s | temperature=%.3f cfg_weight=%.3f exaggeration=%.3f | text_chars=%d | text_files=%d | include_settings=%s",
            audio_prompt_path or request.options.reference_audio_path or "none",
            request.options.temperature,
            request.options.cfg_weight,
            request.options.exaggeration,
            len(request.text or ""),
            len(request.text_files or []),
            request.include_settings_files,
        )

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


@protected.post("/voice/convert", response_model=VoiceConversionResponse)
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


@protected.get("/whisper/models")
def list_whisper_models() -> List[dict]:
    return [{"label": label, "code": code} for label, code in whisper_model_map.items()]


# -----------------------------------------------------------------------------
# Line-level generation
# -----------------------------------------------------------------------------


@protected.post("/lines/generate", response_model=LineGenerationResponse)
async def generate_line(payload: LineGenerationRequest) -> LineGenerationResponse:
    response = await run_in_threadpool(generate_line_audio, payload)
    zip_file = await job_manager.apply_line_update(payload, response)
    if zip_file:
        return response.model_copy(update={"zip_file": zip_file})
    return response


# -----------------------------------------------------------------------------
# Batch job orchestration
# -----------------------------------------------------------------------------


@protected.post("/jobs", status_code=202)
async def create_batch_job(request: BatchCreateRequest) -> dict:
    job_id = await job_manager.create_job(request)
    return {"job_id": job_id}


@protected.get("/jobs/{job_id}", response_model=BatchJobStatus)
async def get_batch_job(job_id: str) -> BatchJobStatus:
    return await job_manager.get_job(job_id)


@protected.post("/jobs/{job_id}/cancel")
async def cancel_batch_job(job_id: str) -> dict:
    await job_manager.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@protected.get("/jobs/{job_id}/zip")
async def download_job_zip(job_id: str) -> FileResponse:
    path = await job_manager.get_job_zip_path(job_id)
    return FileResponse(path, filename=path.name)


# -----------------------------------------------------------------------------
# Data discovery endpoints
# -----------------------------------------------------------------------------


@protected.get("/data/station-formats")
def get_station_formats() -> List[dict]:
    return list_station_formats()


@protected.get("/data/station-formats/{template_id}")
def get_station_format(template_id: str) -> dict:
    return load_station_template(template_id)


@protected.get("/data/sample-stations")
def get_sample_station_list() -> List[dict]:
    return list_sample_stations()


@protected.get("/data/sample-stations/{station_id}")
def get_sample_station(station_id: str) -> dict:
    return load_sample_station(station_id)


@protected.get("/data/sample-scripts")
def get_sample_script_list() -> List[dict]:
    return list_sample_scripts()


@protected.get("/data/sample-scripts/{script_id}")
def get_sample_script(script_id: str) -> dict:
    return {"id": script_id, "content": load_sample_script(script_id)}


@protected.get("/voices/reference")
def get_reference_voice_metadata() -> List[dict]:
    return list_reference_voices()


@protected.get("/voices/reference/{voice}/{style}/{filename:path}")
def stream_reference_audio(voice: str, style: str, filename: str) -> FileResponse:
    path = get_reference_audio_path(voice, style, filename)
    return FileResponse(path)


@protected.get("/voices/clone")
def get_clone_voice_metadata() -> List[dict]:
    return list_clone_voices()


@protected.get("/voices/clone/{voice}/{filename:path}")
def stream_clone_audio(voice: str, filename: str) -> FileResponse:
    path = get_clone_voice_path(voice, filename)
    return FileResponse(path)


# -----------------------------------------------------------------------------
# ChatGPT-driven analysis
# -----------------------------------------------------------------------------


@protected.post("/scripts/analyze", response_model=AnalyzeScriptResponse)
async def analyze_script(payload: AnalyzeScriptRequest) -> AnalyzeScriptResponse:
    if text_preprocessor is None:
        LOGGER.warning("/scripts/analyze requested but OpenAI integration is not configured")
        raise HTTPException(status_code=503, detail="OpenAI integration is not configured")
    LOGGER.info("/scripts/analyze processing %s lines", len(payload.lines))
    try:
        processed = await text_preprocessor.process_batch(payload.lines)
    except TextProcessingError as exc:
        LOGGER.error("/scripts/analyze failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AnalyzeScriptResponse(processed_lines=processed)


router.include_router(protected)
