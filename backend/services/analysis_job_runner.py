import asyncio
import contextlib
import logging

from backend.llm.base import LLMProvider
from backend.services.narrative_analysis import (
    execute_narrative_analysis_job,
    mark_narrative_analysis_job_failed,
    recover_analysis_jobs,
)
from backend.storage.database import DatabasePath


_logger = logging.getLogger(__name__)


class AnalysisJobRunner:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[
            tuple[int, DatabasePath | None, LLMProvider | None]
        ] = asyncio.Queue()
        self._queued_run_ids: set[int] = set()
        self._worker_task: asyncio.Task[None] | None = None
        self._database_path: DatabasePath | None = None

    async def start(
        self,
        *,
        database_path: DatabasePath | None = None,
    ) -> None:
        if self._worker_task is not None:
            return

        self._database_path = database_path
        self._worker_task = asyncio.create_task(
            self._worker_loop()
        )

        for run_id in recover_analysis_jobs(
            database_path=self._database_path,
        ):
            await self.enqueue(
                run_id,
                database_path=self._database_path,
            )

    async def stop(self) -> None:
        if self._worker_task is None:
            return

        self._worker_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task

        self._worker_task = None
        self._queued_run_ids.clear()

    async def enqueue(
        self,
        run_id: int,
        *,
        database_path: DatabasePath | None = None,
        provider: LLMProvider | None = None,
    ) -> bool:
        if run_id in self._queued_run_ids:
            return False

        self._queued_run_ids.add(run_id)
        await self._queue.put(
            (
                run_id,
                database_path,
                provider,
            )
        )
        return True

    async def _worker_loop(self) -> None:
        while True:
            run_id, database_path, provider = await self._queue.get()

            try:
                await execute_narrative_analysis_job(
                    run_id=run_id,
                    database_path=database_path or self._database_path,
                    provider=provider,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _logger.exception(
                    "Narrative analysis job failed: run_id=%s",
                    run_id,
                )
                mark_narrative_analysis_job_failed(
                    run_id=run_id,
                    error_message=str(error),
                    database_path=database_path or self._database_path,
                )
            finally:
                if provider is not None:
                    await provider.close()

                self._queued_run_ids.discard(run_id)
                self._queue.task_done()


analysis_job_runner = AnalysisJobRunner()
