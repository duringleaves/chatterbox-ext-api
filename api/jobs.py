"""Batch job management for multi-line TTS generation."""
from __future__ import annotations

import asyncio
import shutil
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


@dataclass
class LineState:
    payload: LineGenerationRequest
    status: LineStatus = LineStatus.pending
    result: Optional[LineGenerationResponse] = None
    error: Optional[str] = None


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
        lines = {line.line_id: LineState(payload=line) for line in request.lines}
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
        try:
            self._build_zip(job)
        except Exception as exc:  # pragma: no cover - zip failure
            job.state = JobState.failed
            job.error = f"Failed to build ZIP: {exc}"

    def _persist_line_outputs(self, job: BatchJob, line_id: str, response: LineGenerationResponse) -> None:
        line_dir_final = job.dir / "final" / line_id
        line_dir_raw = job.dir / "raw" / line_id
        line_dir_final.mkdir(parents=True, exist_ok=True)
        line_dir_raw.mkdir(parents=True, exist_ok=True)

        for dest_dir, outputs in ((line_dir_final, response.final_outputs), (line_dir_raw, response.raw_outputs)):
            for item in outputs:
                src = _resolve_path(item)
                if not src.exists():
                    continue
                shutil.copy2(src, dest_dir / src.name)

    def _build_zip(self, job: BatchJob) -> None:
        concatenated_path = job.dir / f"{job.id}_concatenated.wav"
        segments = []
        for line_state in job.lines.values():
            if line_state.status != LineStatus.completed or not line_state.result:
                continue
            source_path = _first_wav(line_state.result.final_outputs) or _first_wav(line_state.result.raw_outputs)
            if source_path and source_path.exists():
                segments.append(AudioSegment.from_wav(source_path))
        if segments:
            combined = segments[0]
            for seg in segments[1:]:
                combined += seg
            combined.export(concatenated_path, format="wav")

        zip_path = job.dir / f"{job.id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if concatenated_path.exists():
                zf.write(concatenated_path, arcname=f"concatenated/{concatenated_path.name}")
            for folder in [job.dir / "final", job.dir / "raw"]:
                if folder.exists():
                    for path in folder.rglob("*"):
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

    async def get_job_zip_path(self, job_id: str) -> Path:
        async with self._lock:
            job = self.jobs.get(job_id)
        if not job or not job.zip_path or not job.zip_path.exists():
            raise HTTPException(status_code=404, detail="Zip file not available")
        return job.zip_path


job_manager = JobManager()
