"""In-process registry for background ingestion jobs.

Deliberately simple: a dict behind a lock, plus a bounded worker pool. That is
the right size for a single-node deployment. A multi-worker or multi-host setup
would swap this for Redis/Celery -- the API surface below is the seam where
that change would land.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from multimodal_rag.api.schemas import IngestionResponse, JobResponse, JobStatus
from multimodal_rag.logging_config import get_logger

logger = get_logger(__name__)

# Keep the most recent jobs so the UI can still show a finished run.
_MAX_JOBS = 200


@dataclass
class Job:
    job_id: str
    filename: str
    path: Path
    status: JobStatus = "pending"
    processed: int = 0
    total: int = 0
    result: IngestionResponse | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_response(self) -> JobResponse:
        with self._lock:
            return JobResponse(
                job_id=self.job_id,
                status=self.status,
                filename=self.filename,
                processed=self.processed,
                total=self.total,
                result=self.result,
                error=self.error,
            )

    def set_progress(self, processed: int, total: int) -> None:
        with self._lock:
            self.processed = processed
            self.total = total

    def mark(
        self,
        status: JobStatus,
        *,
        result: IngestionResponse | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.status = status
            if result is not None:
                self.result = result
            if error is not None:
                self.error = error


class JobRegistry:
    """Tracks ingestion jobs and runs them off the request thread."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        # Ingestion is API-bound and token-hungry; more workers just multiply
        # rate-limit collisions.
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")

    def submit(self, filename: str, path: Path, runner) -> Job:
        """Register a job and start it. ``runner`` takes ``(job)``."""
        job = Job(job_id=uuid.uuid4().hex[:12], filename=filename, path=path)
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > _MAX_JOBS:
                self._jobs.popitem(last=False)

        def execute() -> None:
            job.mark("running")
            try:
                runner(job)
            except Exception as error:
                logger.exception("Ingestion job %s failed", job.job_id)
                job.mark("failed", error=str(error))

        self._pool.submit(execute)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
