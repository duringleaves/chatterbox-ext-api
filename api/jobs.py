"""Batch job management for multi-line TTS generation."""
from __future__ import annotations

import asyncio
import datetime
import re
import shutil
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException
from pydub import AudioSegment

from .core import BASE_DIR, OUTPUT_DIR
from .line_processing import generate_line_audio
from .schemas import (
    BatchCreateRequest,
    BatchJobStatus,
    BatchLineStatus,
    FileResult,
    JobState,
    LineGenerationRequest,
    LineGenerationResponse,
    LineStatus,
)
from .utils import to_file_result

JOBS_ROOT = OUTPUT_DIR / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _resolve_path(file_result) -> Path:
    path = Path(file_result.path)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def _first_wav(file_results) -> Optional[Path]:
    for item in file_results or []:
        path = _resolve_path(item)
        if path.suffix.lower() == ".wav" and path.exists():
            return path
    return None


def _slugify(value: Optional[str], fallback: str) -> str:
    base = value or fallback
    normalized = unicodedata.normalize("NFKD", base)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"\s+", " ", ascii_only).strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")
    return slug or fallback


@dataclass
class LineState:
    payload: LineGenerationRequest
    status: LineStatus = LineStatus.pending
    result: Optional[LineGenerationResponse] = None
    error: Optional[str] = None
    order: int = 0


@dataclass
class BatchJob:
    id: str
    lines: Dict[str, LineState]
    name: Optional[str] = None
    state: JobState = JobState.pending
    zip_path: Optional[Path] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    dir: Path = field(default_factory=lambda: JOBS_ROOT / uuid.uuid4().hex)


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, BatchJob] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, request: BatchCreateRequest) -> str:
        if not request.lines:
            raise HTTPException(status_code=400, detail="No lines provided")
        job_id = uuid.uuid4().hex
        job_dir = JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        lines: Dict[str, LineState] = {}
        for index, line in enumerate(request.lines, start=1):
            payload = line.model_copy(update={"job_id": job_id, "queue_position": line.queue_position or index})
            lines[line.line_id] = LineState(payload=payload, order=payload.queue_position or index)
        job = BatchJob(id=job_id, lines=lines, name=request.job_name, dir=job_dir)
        async with self._lock:
            self.jobs[job_id] = job
        job.task = asyncio.create_task(self._run_job(job))
        return job_id

    async def _run_job(self, job: BatchJob) -> None:
        job.state = JobState.running
        for line_id, line_state in job.lines.items():
            if job.cancel_event.is_set():
                line_state.status = LineStatus.cancelled
                continue
            line_state.status = LineStatus.processing
            try:
                response = await asyncio.to_thread(generate_line_audio, line_state.payload)
                line_state.result = response
                line_state.status = LineStatus.completed
                self._persist_line_outputs(job, line_id, response)
            except Exception as exc:  # pragma: no cover - runtime failures
                line_state.status = LineStatus.failed
                line_state.error = str(exc)
        if job.cancel_event.is_set():
            job.state = JobState.cancelled
            return
        if any(state.status == LineStatus.failed for state in job.lines.values()):
            job.state = JobState.failed
            return
        job.state = JobState.completed

    def _persist_line_outputs(
        self,
        job: BatchJob,
        line_id: str,
        response: LineGenerationResponse,
        *,
        replace_existing: bool = False,
    ) -> None:
        line_state = job.lines.get(line_id)
        order = line_state.order if line_state else None
        folder_label = f"{order:03d}_{line_id}" if order else line_id

        final_dir = job.dir / "final" / folder_label
        raw_dir = job.dir / "raw" / folder_label
        if replace_existing and final_dir.exists():
            shutil.rmtree(final_dir)
        if replace_existing and raw_dir.exists():
            shutil.rmtree(raw_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        def copy_audio(src_path: Path, target_dir: Path) -> None:
            if src_path.suffix.lower() != ".wav" or not src_path.exists():
                return
            dest_path = target_dir / src_path.name
            counter = 2
            while dest_path.exists():
                dest_path = target_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
                counter += 1
            shutil.copy2(src_path, dest_path)

        for dest_dir, outputs in ((final_dir, response.final_outputs), (raw_dir, response.raw_outputs)):
            for item in outputs:
                src = _resolve_path(item)
                copy_audio(src, dest_dir)

    def _build_zip(self, job: BatchJob, zip_path: Path) -> None:
        silence_gap = AudioSegment.silent(duration=500)
        concatenated_final_path = job.dir / "concatenated_final.wav"
        concatenated_raw_path = job.dir / "concatenated_raw.wav"

        if zip_path.exists():
            zip_path.unlink()

        ordered_states = sorted(
            job.lines.items(),
            key=lambda item: ((item[1].order or 0), item[0]),
        )

        include_final = any(
            state.result and state.result.metadata.get("clone_voice")
            for _, state in ordered_states
        )

        final_segments = []
        raw_segments = []

        for _, line_state in ordered_states:
            if line_state.status != LineStatus.completed or not line_state.result:
                continue
            final_source = _first_wav(line_state.result.final_outputs) or _first_wav(line_state.result.raw_outputs)
            raw_source = _first_wav(line_state.result.raw_outputs)
            if include_final and final_source and final_source.exists():
                final_segments.append(AudioSegment.from_wav(final_source))
            if raw_source and raw_source.exists():
                raw_segments.append(AudioSegment.from_wav(raw_source))

        if include_final and final_segments:
            combined_final = final_segments[0]
            for segment in final_segments[1:]:
                combined_final += silence_gap + segment
            combined_final.export(concatenated_final_path, format="wav")
        else:
            if concatenated_final_path.exists():
                concatenated_final_path.unlink()

        if raw_segments:
            combined_raw = raw_segments[0]
            for segment in raw_segments[1:]:
                combined_raw += silence_gap + segment
            combined_raw.export(concatenated_raw_path, format="wav")
        elif concatenated_raw_path.exists():
            concatenated_raw_path.unlink()

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if include_final and concatenated_final_path.exists():
                zf.write(concatenated_final_path, arcname="concatenated/final.wav")
            if concatenated_raw_path.exists():
                zf.write(concatenated_raw_path, arcname="concatenated/raw.wav")
            raw_folder = job.dir / "raw"
            if raw_folder.exists():
                for path in sorted(raw_folder.rglob("*.wav")):
                    if path.is_file():
                        arcname = path.relative_to(job.dir)
                        zf.write(path, arcname=str(arcname))
            if include_final:
                final_folder = job.dir / "final"
                if final_folder.exists():
                    for path in sorted(final_folder.rglob("*.wav")):
                        if path.is_file():
                            arcname = path.relative_to(job.dir)
                            zf.write(path, arcname=str(arcname))

        job.zip_path = zip_path

    async def get_job(self, job_id: str) -> BatchJobStatus:
        async with self._lock:
            job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        total = len(job.lines)
        completed = sum(1 for state in job.lines.values() if state.status == LineStatus.completed)
        failed = sum(1 for state in job.lines.values() if state.status == LineStatus.failed)
        line_statuses = []
        for line_id, state in job.lines.items():
            line_statuses.append(
                BatchLineStatus(
                    line_id=line_id,
                    status=state.status,
                    error=state.error,
                    raw_outputs=state.result.raw_outputs if state.result else None,
                    final_outputs=state.result.final_outputs if state.result else None,
                )
            )
        progress = completed / total if total else 0.0
        zip_file = to_file_result(job.zip_path) if job.zip_path and job.zip_path.exists() else None
        message = job.error
        return BatchJobStatus(
            job_id=job.id,
            state=job.state,
            progress=progress,
            total_lines=total,
            completed_lines=completed,
            failed_lines=failed,
            lines=line_statuses,
            zip_file=zip_file,
            message=message,
        )

    async def cancel_job(self, job_id: str) -> None:
        async with self._lock:
            job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.state in {JobState.completed, JobState.cancelled, JobState.failed}:
            return
        job.cancel_event.set()
        if job.task:
            await job.task
        job.state = JobState.cancelled

    async def apply_line_update(
        self,
        payload: LineGenerationRequest,
        response: LineGenerationResponse,
    ) -> Optional[FileResult]:
        job_id = payload.job_id
        if not job_id:
            return None

        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            line_state = job.lines.get(payload.line_id)
            if not line_state:
                order = payload.queue_position or len(job.lines) + 1
                payload = payload.model_copy(update={"job_id": job_id, "queue_position": order})
                line_state = LineState(payload=payload, order=order)
                job.lines[payload.line_id] = line_state
            else:
                order = line_state.order or payload.queue_position
                queue_position = payload.queue_position or order
                payload = payload.model_copy(update={"job_id": job_id, "queue_position": queue_position})
                line_state.payload = payload
                if queue_position:
                    line_state.order = queue_position

            line_state.result = response
            line_state.status = LineStatus.completed
            line_state.error = None

            self._persist_line_outputs(job, payload.line_id, response, replace_existing=True)
            if job.zip_path and job.zip_path.exists():
                try:
                    job.zip_path.unlink()
                except FileNotFoundError:
                    pass
            job.zip_path = None
            return None

    async def get_job_zip_path(self, job_id: str) -> Path:
        async with self._lock:
            job = self.jobs.get(job_id)
        if not job or not job.zip_path or not job.zip_path.exists():
            raise HTTPException(status_code=404, detail="Zip file not available")
        return job.zip_path

    async def build_zip(self, job_id: str) -> FileResult:
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            completed_lines = [state for state in job.lines.values() if state.status == LineStatus.completed and state.result]
            if not completed_lines:
                raise HTTPException(status_code=400, detail="No completed lines to bundle")

            station_slug = _slugify(job.name, "bundle")
            reference_voice = None
            for state in completed_lines:
                if state.payload.reference_voice:
                    reference_voice = state.payload.reference_voice
                    break
            voice_slug = _slugify(reference_voice, "voice")
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            zip_filename = f"{station_slug}_{voice_slug}_{timestamp}.zip"
            zip_path = job.dir / zip_filename

            if job.zip_path and job.zip_path.exists() and job.zip_path != zip_path:
                try:
                    job.zip_path.unlink()
                except FileNotFoundError:
                    pass

            try:
                self._build_zip(job, zip_path)
            except Exception as exc:  # pragma: no cover - defensive
                raise HTTPException(status_code=500, detail=f"Failed to build ZIP: {exc}") from exc

            return to_file_result(zip_path)


job_manager = JobManager()
