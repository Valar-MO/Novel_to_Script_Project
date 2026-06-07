import asyncio
import contextlib
import json
import logging

from backend.llm.base import LLMProvider
from backend.services.script_generation import (
    execute_script_generation_job,
    mark_script_generation_job_failed,
    recover_script_generation_jobs,
)
from backend.storage.database import DatabasePath
from backend.storage.database import database_session


_logger = logging.getLogger(__name__)


def _create_provider_for_run(
    *,
    run_id: int,
    database_path: DatabasePath | None,
) -> LLMProvider:
    from dataclasses import replace

    from backend.config import get_llm_settings
    from backend.llm.factory import create_llm_provider

    with database_session(database_path=database_path) as connection:
        row = connection.execute(
            "SELECT request_json FROM script_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    request_payload = {}
    if row and row["request_json"]:
        try:
            request_payload = json.loads(row["request_json"])
        except json.JSONDecodeError:
            request_payload = {}

    provider_name = str(
        request_payload.get("provider") or "deepseek"
    ).strip().lower()
    model_name = str(
        request_payload.get("model") or "deepseek-v4-pro"
    ).strip()
    thinking_enabled = bool(
        request_payload.get("thinking_enabled", False)
    )

    settings = get_llm_settings()
    if provider_name == "deepseek":
        settings = replace(
            settings,
            provider="deepseek",
            cloud_api_base_url=(
                settings.cloud_api_base_url
                or "https://api.deepseek.com"
            ),
            cloud_api_model=model_name,
            cloud_api_reasoning_effort=None,
            cloud_api_thinking_enabled=thinking_enabled,
        )
    elif provider_name == "cloud_api":
        settings = replace(
            settings,
            provider="cloud_api",
            cloud_api_model=model_name,
            cloud_api_reasoning_effort=None,
            cloud_api_thinking_enabled=thinking_enabled,
        )

    return create_llm_provider(settings=settings)


class ScriptGenerationRunner:
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
        self._worker_task = asyncio.create_task(self._worker_loop())

        for run_id in recover_script_generation_jobs(
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
        await self._queue.put((run_id, database_path, provider))
        return True

    async def _worker_loop(self) -> None:
        while True:
            run_id, database_path, provider = await self._queue.get()

            try:
                if provider is None:
                    provider = _create_provider_for_run(
                        run_id=run_id,
                        database_path=database_path or self._database_path,
                    )

                await execute_script_generation_job(
                    run_id=run_id,
                    database_path=database_path or self._database_path,
                    provider=provider,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _logger.exception(
                    "Script generation job failed: run_id=%s",
                    run_id,
                )
                mark_script_generation_job_failed(
                    run_id=run_id,
                    error_message=str(error),
                    database_path=database_path or self._database_path,
                )
            finally:
                if provider is not None:
                    await provider.close()

                self._queued_run_ids.discard(run_id)
                self._queue.task_done()


script_generation_runner = ScriptGenerationRunner()
