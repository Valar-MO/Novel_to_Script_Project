import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMProviderUnavailableError,
)
from backend.services.narrative_analysis import (
    ActiveNarrativeAnalysisError,
    create_narrative_analysis_job,
    get_active_narrative_analysis_run,
    get_narrative_analysis_run,
    resume_narrative_analysis_job,
    retry_failed_narrative_analysis_job,
)
from backend.services.analysis_job_runner import analysis_job_runner
from backend.services.project_storage import ProjectNotFoundError
from backend.storage.database import DatabasePath


router = APIRouter(tags=["narrative-analysis"])


class NarrativeAnalysisStartRequest(BaseModel):
    max_chunks: int | None = Field(
        default=None,
        ge=1,
    )
    previous_context_chars: int = Field(
        default=500,
        ge=0,
        le=5000,
    )
    next_context_chars: int = Field(
        default=0,
        ge=0,
        le=5000,
    )
    force_reanalyze: bool = False


class NarrativeAnalysisStartResponse(BaseModel):
    run_id: int
    project_id: str
    status: str
    total_chunks: int
    processed_chunks: int = 0
    successful_chunks: int
    failed_chunks: int
    partial_chunks: int = 0
    cached_chunks: int
    cached_layers: int = 0


class NarrativeAnalysisUnitResponse(BaseModel):
    id: int
    chunk_database_id: int
    chunk_id: str
    text_hash: str
    analysis_input_hash: str | None
    status: str
    cache_hit: bool
    cache_source_unit_id: int | None
    result: dict[str, Any] | None
    validated_result: dict[str, Any] | None
    error_message: str | None
    attempt_count: int = 0
    last_started_at: str | None = None
    last_finished_at: str | None = None
    created_at: str
    updated_at: str


class NarrativeAnalysisRunResponse(BaseModel):
    id: int
    project_id: str
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    status: str
    error_message: str | None
    total_chunks: int = 0
    processed_chunks: int = 0
    successful_chunks: int = 0
    partial_chunks: int = 0
    failed_chunks: int = 0
    cached_chunks: int = 0
    cached_layers: int = 0
    current_chunk_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    created_at: str
    updated_at: str
    units: list[NarrativeAnalysisUnitResponse]


def get_analysis_provider() -> LLMProvider:
    from backend.llm.factory import create_llm_provider

    return create_llm_provider()


def get_analysis_database_path() -> DatabasePath | None:
    return None


def _parse_json_object(
    value: str | None,
) -> dict[str, Any] | None:
    if not value:
        return None

    parsed = json.loads(value)

    if isinstance(parsed, dict):
        return parsed

    return {
        "value": parsed,
    }


def _to_start_response(
    result: Any,
) -> NarrativeAnalysisStartResponse:
    return NarrativeAnalysisStartResponse(
        run_id=result.run_id,
        project_id=result.project_id,
        status=result.status,
        total_chunks=result.total_chunks,
        processed_chunks=result.processed_chunks,
        successful_chunks=result.successful_chunks,
        failed_chunks=result.failed_chunks,
        partial_chunks=result.partial_chunks,
        cached_chunks=result.cached_chunks,
        cached_layers=result.cached_layers,
    )


@router.post(
    "/api/projects/{project_id}/narrative-analysis",
    response_model=NarrativeAnalysisStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_narrative_analysis(
    project_id: str,
    request: NarrativeAnalysisStartRequest,
    provider: LLMProvider = Depends(get_analysis_provider),
    database_path: DatabasePath | None = Depends(
        get_analysis_database_path
    ),
) -> NarrativeAnalysisStartResponse:
    provider_enqueued = False

    try:
        health = await provider.health_check()

        if not health.available:
            raise LLMProviderUnavailableError(
                health.detail
            )

        result = create_narrative_analysis_job(
            project_id=project_id,
            max_chunks=request.max_chunks,
            previous_context_chars=request.previous_context_chars,
            next_context_chars=request.next_context_chars,
            force_reanalyze=request.force_reanalyze,
            provider=provider,
            database_path=database_path,
        )
        provider_enqueued = await analysis_job_runner.enqueue(
            result.run_id,
            database_path=database_path,
            provider=provider,
        )
    except ProjectNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在。",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except ActiveNarrativeAnalysisError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(error),
                "active_run_id": error.active_run_id,
            },
        ) from error
    except LLMProviderUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except LLMProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error
    finally:
        if not provider_enqueued:
            await provider.close()

    return NarrativeAnalysisStartResponse(
        run_id=result.run_id,
        project_id=result.project_id,
        status=result.status,
        total_chunks=result.total_chunks,
        processed_chunks=result.processed_chunks,
        successful_chunks=result.successful_chunks,
        failed_chunks=result.failed_chunks,
        partial_chunks=result.partial_chunks,
        cached_chunks=result.cached_chunks,
        cached_layers=result.cached_layers,
    )


def _to_run_response(
    run_data: dict[str, Any],
) -> NarrativeAnalysisRunResponse:
    units = [
        NarrativeAnalysisUnitResponse(
            id=unit["id"],
            chunk_database_id=unit["chunk_database_id"],
            chunk_id=unit["chunk_id"],
            text_hash=unit["text_hash"],
            analysis_input_hash=unit.get("analysis_input_hash"),
            status=unit["status"],
            cache_hit=unit.get("cache_hit", False),
            cache_source_unit_id=unit.get("cache_source_unit_id"),
            result=_parse_json_object(unit["result_json"]),
            validated_result=_parse_json_object(
                unit["validated_result_json"]
            ),
            error_message=unit["error_message"],
            attempt_count=unit.get("attempt_count", 0),
            last_started_at=unit.get("last_started_at"),
            last_finished_at=unit.get("last_finished_at"),
            created_at=unit["created_at"],
            updated_at=unit["updated_at"],
        )
        for unit in run_data["units"]
    ]

    return NarrativeAnalysisRunResponse(
        id=run_data["id"],
        project_id=run_data["project_id"],
        provider=run_data["provider"],
        model=run_data["model"],
        prompt_version=run_data["prompt_version"],
        schema_version=run_data["schema_version"],
        status=run_data["status"],
        error_message=run_data["error_message"],
        total_chunks=run_data.get("total_chunks", 0),
        processed_chunks=run_data.get("processed_chunks", 0),
        successful_chunks=run_data.get("successful_chunks", 0),
        partial_chunks=run_data.get("partial_chunks", 0),
        failed_chunks=run_data.get("failed_chunks", 0),
        cached_chunks=run_data.get("cached_chunks", 0),
        cached_layers=run_data.get("cached_layers", 0),
        current_chunk_id=run_data.get("current_chunk_id"),
        started_at=run_data.get("started_at"),
        finished_at=run_data.get("finished_at"),
        heartbeat_at=run_data.get("heartbeat_at"),
        created_at=run_data["created_at"],
        updated_at=run_data["updated_at"],
        units=units,
    )


@router.get(
    "/api/projects/{project_id}/narrative-analysis/active",
    response_model=NarrativeAnalysisRunResponse | None,
    status_code=status.HTTP_200_OK,
)
def read_active_narrative_analysis_run(
    project_id: str,
    database_path: DatabasePath | None = Depends(
        get_analysis_database_path
    ),
) -> NarrativeAnalysisRunResponse | None:
    try:
        run_data = get_active_narrative_analysis_run(
            project_id=project_id,
            database_path=database_path,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    if run_data is None:
        return None

    return _to_run_response(run_data)


@router.post(
    "/api/narrative-analysis/{run_id}/resume",
    response_model=NarrativeAnalysisStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume_narrative_analysis(
    run_id: int,
    database_path: DatabasePath | None = Depends(
        get_analysis_database_path
    ),
) -> NarrativeAnalysisStartResponse:
    try:
        result = resume_narrative_analysis_job(
            run_id=run_id,
            database_path=database_path,
        )
        await analysis_job_runner.enqueue(
            run_id,
            database_path=database_path,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析批次不存在。",
        ) from error

    return _to_start_response(result)


@router.post(
    "/api/narrative-analysis/{run_id}/retry-failed",
    response_model=NarrativeAnalysisStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_failed_narrative_analysis(
    run_id: int,
    database_path: DatabasePath | None = Depends(
        get_analysis_database_path
    ),
) -> NarrativeAnalysisStartResponse:
    try:
        result = retry_failed_narrative_analysis_job(
            run_id=run_id,
            database_path=database_path,
        )
        await analysis_job_runner.enqueue(
            run_id,
            database_path=database_path,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析批次不存在。",
        ) from error

    return _to_start_response(result)


@router.get(
    "/api/narrative-analysis/{run_id}",
    response_model=NarrativeAnalysisRunResponse,
    status_code=status.HTTP_200_OK,
)
def read_narrative_analysis_run(
    run_id: int,
    include_units: bool = True,
    database_path: DatabasePath | None = Depends(
        get_analysis_database_path
    ),
) -> NarrativeAnalysisRunResponse:
    try:
        run_data = get_narrative_analysis_run(
            run_id=run_id,
            database_path=database_path,
            include_units=include_units,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析批次不存在。",
        ) from error

    return _to_run_response(run_data)
