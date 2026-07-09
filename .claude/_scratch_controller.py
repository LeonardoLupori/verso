from __future__ import annotations

from _scratch_jobs import BackgroundJob, DeepSliceWorker


class Controller:
    def __init__(self) -> None:
        self._deepslice_job: BackgroundJob[DeepSliceWorker] | None = None
