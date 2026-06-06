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
    analyze_project_narrative,
    get_narrative_analysis_run,
)
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


@router.post(
    "/api/projects/{project_id}/narrative-analysis",
    response_model=NarrativeAnalysisStartResponse,
    status_code=status.HTTP_200_OK,
)
async def start_narrative_analysis(
    project_id: str,
    request: NarrativeAnalysisStartRequest,
    provider: LLMProvider = Depends(get_analysis_provider),
    database_path: DatabasePath | None = Depends(
        get_analysis_database_path
    ),
) -> NarrativeAnalysisStartResponse:
    try:
        health = await provider.health_check()

        if not health.available:
            raise LLMProviderUnavailableError(
                health.detail
            )

        result = await analyze_project_narrative(
            project_id=project_id,
            max_chunks=request.max_chunks,
            previous_context_chars=request.previous_context_chars,
            next_context_chars=request.next_context_chars,
            force_reanalyze=request.force_reanalyze,
            provider=provider,
            database_path=database_path,
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
        await provider.close()

    return NarrativeAnalysisStartResponse(
        run_id=result.run_id,
        project_id=result.project_id,
        status=result.status,
        total_chunks=result.total_chunks,
        successful_chunks=result.successful_chunks,
        failed_chunks=result.failed_chunks,
        partial_chunks=result.partial_chunks,
        cached_chunks=result.cached_chunks,
        cached_layers=result.cached_layers,
    )


@router.get(
    "/api/narrative-analysis/{run_id}",
    response_model=NarrativeAnalysisRunResponse,
    status_code=status.HTTP_200_OK,
)
def read_narrative_analysis_run(
    run_id: int,
    database_path: DatabasePath | None = Depends(
        get_analysis_database_path
    ),
) -> NarrativeAnalysisRunResponse:
    try:
        run_data = get_narrative_analysis_run(
            run_id=run_id,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析批次不存在。",
        ) from error

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
        created_at=run_data["created_at"],
        updated_at=run_data["updated_at"],
        units=units,
    )
