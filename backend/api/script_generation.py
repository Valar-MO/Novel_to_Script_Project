from typing import Any, Literal
from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.config import get_llm_settings
from backend.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMProviderUnavailableError,
)
from backend.services.script_generation import (
    ActiveScriptGenerationError,
    ScriptGenerationJobResult,
    create_script_generation_job,
    get_latest_script_generation_run,
    get_project_script_generation_state,
    get_script_generation_run,
    get_script_generation_scenes,
    regenerate_script_scene,
    request_script_generation_cancel,
    resume_script_generation_job,
    update_script_scene,
)
from backend.services.script_generation_runner import script_generation_runner
from backend.storage.database import DatabasePath


router = APIRouter(tags=["script-generation"])


class ScriptGenerationStartRequest(BaseModel):
    scope: Literal["all", "pending", "selected"] = "all"
    chunk_ids: list[str] = Field(default_factory=list)
    max_chunks: int | None = Field(default=None, ge=1)
    generation_style: str = "standard"
    adaptation_mode: str = "faithful"
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    thinking_enabled: bool = False


class ScriptGenerationStartResponse(BaseModel):
    run_id: int
    project_id: str
    status: str
    total_chunks: int
    processed_chunks: int = 0
    successful_chunks: int = 0
    partial_chunks: int = 0
    failed_chunks: int = 0
    scene_count: int = 0


class ScriptGenerationUnitResponse(BaseModel):
    id: int
    chunk_database_id: int
    chunk_id: str
    chunk_order: int
    status: str
    warnings: list[str]
    error_message: str | None
    attempt_count: int
    last_started_at: str | None
    last_finished_at: str | None
    created_at: str
    updated_at: str


class ScriptGenerationRunResponse(BaseModel):
    id: int
    project_id: str
    source_character_run_id: int | None
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    status: str
    error_message: str | None
    total_chunks: int
    processed_chunks: int
    successful_chunks: int
    partial_chunks: int
    failed_chunks: int
    scene_count: int
    current_chunk_id: str | None
    started_at: str | None
    finished_at: str | None
    heartbeat_at: str | None
    cancel_requested_at: str | None
    cancelled_at: str | None
    created_at: str
    updated_at: str
    units: list[ScriptGenerationUnitResponse]


class ScriptGenerationStateResponse(BaseModel):
    project_id: str
    has_generated_scenes: bool
    latest_run: ScriptGenerationRunResponse | None
    pending_script_chunk_count: int
    pending_script_chunk_ids: list[str]
    never_attempted_chunk_count: int
    retryable_chunk_count: int
    suggested_action: str


class ScriptSceneSourceResponse(BaseModel):
    id: int
    chunk_database_id: int
    chunk_id: str
    start_offset: int
    end_offset: int
    evidence_text: str


class ScriptSceneResponse(BaseModel):
    id: int
    scene_number: int
    heading: str
    interior_exterior: str
    location: str
    time_of_day: str
    characters: list[dict[str, Any]]
    script_text: str
    scene_summary: str
    generation_status: str
    is_user_edited: bool
    warnings: list[str]
    adaptation_notes: list[str]
    source_spans: list[ScriptSceneSourceResponse]
    created_at: str
    updated_at: str


class ScriptGenerationScenesResponse(BaseModel):
    run_id: int
    project_id: str
    status: str
    scenes: list[ScriptSceneResponse]


class ScriptSceneUpdateRequest(BaseModel):
    heading: str | None = None
    interior_exterior: str | None = None
    location: str | None = None
    time_of_day: str | None = None
    script_text: str | None = None
    characters: list[dict[str, Any]] | None = None
    warnings: list[str] | None = None


class ScriptSceneRegenerateRequest(BaseModel):
    instruction: str = ""


def _create_script_generation_provider(
    *,
    provider_name: str = "deepseek",
    model_name: str = "deepseek-v4-pro",
    thinking_enabled: bool = False,
) -> LLMProvider:
    from backend.llm.factory import create_llm_provider

    settings = get_llm_settings()
    normalized_provider = (provider_name or "deepseek").strip().lower()
    normalized_model = (model_name or "deepseek-v4-pro").strip()

    if normalized_provider == "deepseek":
        settings = replace(
            settings,
            provider="deepseek",
            cloud_api_base_url=(
                settings.cloud_api_base_url
                or "https://api.deepseek.com"
            ),
            cloud_api_model=normalized_model,
            cloud_api_reasoning_effort=None,
            cloud_api_thinking_enabled=thinking_enabled,
        )
    elif normalized_provider == "cloud_api":
        settings = replace(
            settings,
            provider="cloud_api",
            cloud_api_model=normalized_model,
            cloud_api_reasoning_effort=None,
            cloud_api_thinking_enabled=thinking_enabled,
        )
    return create_llm_provider(settings=settings)


def get_script_generation_provider() -> LLMProvider:
    return _create_script_generation_provider()


def get_script_generation_database_path() -> DatabasePath | None:
    return None


def _to_start_response(
    result: ScriptGenerationJobResult,
) -> ScriptGenerationStartResponse:
    return ScriptGenerationStartResponse(
        run_id=result.run_id,
        project_id=result.project_id,
        status=result.status,
        total_chunks=result.total_chunks,
        processed_chunks=result.processed_chunks,
        successful_chunks=result.successful_chunks,
        partial_chunks=result.partial_chunks,
        failed_chunks=result.failed_chunks,
        scene_count=result.scene_count,
    )


def _to_run_response(
    data: dict[str, Any],
) -> ScriptGenerationRunResponse:
    return ScriptGenerationRunResponse(
        **{
            **data,
            "units": [
                ScriptGenerationUnitResponse(**unit)
                for unit in data.get("units", [])
            ],
        }
    )


@router.post(
    "/api/projects/{project_id}/script-generation",
    response_model=ScriptGenerationStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_script_generation(
    project_id: str,
    request: ScriptGenerationStartRequest,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptGenerationStartResponse:
    provider_enqueued = False
    provider = _create_script_generation_provider(
        provider_name=request.provider,
        model_name=request.model,
        thinking_enabled=request.thinking_enabled,
    )

    try:
        health = await provider.health_check()
        if not health.available:
            raise LLMProviderUnavailableError(health.detail)

        result = create_script_generation_job(
            project_id=project_id,
            provider=provider,
            database_path=database_path,
            chunk_ids=request.chunk_ids,
            max_chunks=request.max_chunks,
            generation_style=request.generation_style,
            adaptation_mode=request.adaptation_mode,
            requested_provider=request.provider,
            requested_model=request.model,
            thinking_enabled=request.thinking_enabled,
            scope=request.scope,
        )
        provider_enqueued = await script_generation_runner.enqueue(
            result.run_id,
            database_path=database_path,
            provider=provider,
        )
    except ActiveScriptGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(error),
                "active_run_id": error.active_run_id,
            },
        ) from error
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
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
        if not provider_enqueued:
            await provider.close()

    return _to_start_response(result)


@router.get(
    "/api/projects/{project_id}/script-generation/state",
    response_model=ScriptGenerationStateResponse,
)
def read_project_script_generation_state(
    project_id: str,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptGenerationStateResponse:
    try:
        data = get_project_script_generation_state(
            project_id=project_id,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    latest_run = data.get("latest_run")
    return ScriptGenerationStateResponse(
        project_id=data["project_id"],
        has_generated_scenes=data["has_generated_scenes"],
        latest_run=(
            _to_run_response(latest_run)
            if latest_run is not None
            else None
        ),
        pending_script_chunk_count=data["pending_script_chunk_count"],
        pending_script_chunk_ids=data["pending_script_chunk_ids"],
        never_attempted_chunk_count=data["never_attempted_chunk_count"],
        retryable_chunk_count=data["retryable_chunk_count"],
        suggested_action=data["suggested_action"],
    )


@router.post(
    "/api/script-generation/{run_id}/cancel",
    response_model=ScriptGenerationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def cancel_script_generation(
    run_id: int,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptGenerationRunResponse:
    try:
        data = request_script_generation_cancel(
            run_id=run_id,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return _to_run_response(data)


@router.post(
    "/api/script-generation/{run_id}/resume",
    response_model=ScriptGenerationStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume_script_generation(
    run_id: int,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptGenerationStartResponse:
    try:
        result = resume_script_generation_job(
            run_id=run_id,
            database_path=database_path,
        )
        await script_generation_runner.enqueue(
            run_id,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except ActiveScriptGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(error),
                "active_run_id": error.active_run_id,
            },
        ) from error

    return _to_start_response(result)


@router.get(
    "/api/script-generation/{run_id}",
    response_model=ScriptGenerationRunResponse,
)
def read_script_generation_run(
    run_id: int,
    include_units: bool = True,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptGenerationRunResponse:
    try:
        data = get_script_generation_run(
            run_id=run_id,
            database_path=database_path,
            include_units=include_units,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return _to_run_response(data)


@router.get(
    "/api/projects/{project_id}/script-generation/latest",
    response_model=ScriptGenerationRunResponse | None,
)
def read_latest_script_generation_run(
    project_id: str,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptGenerationRunResponse | None:
    data = get_latest_script_generation_run(
        project_id=project_id,
        database_path=database_path,
    )
    if data is None:
        return None
    return _to_run_response(data)


@router.get(
    "/api/script-generation/{run_id}/scenes",
    response_model=ScriptGenerationScenesResponse,
)
def read_script_generation_scenes(
    run_id: int,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptGenerationScenesResponse:
    try:
        data = get_script_generation_scenes(
            run_id=run_id,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return ScriptGenerationScenesResponse(**data)


@router.patch(
    "/api/script-scenes/{scene_id}",
    response_model=ScriptSceneResponse,
)
def patch_script_scene(
    scene_id: int,
    request: ScriptSceneUpdateRequest,
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptSceneResponse:
    try:
        data = update_script_scene(
            scene_id=scene_id,
            heading=request.heading,
            interior_exterior=request.interior_exterior,
            location=request.location,
            time_of_day=request.time_of_day,
            script_text=request.script_text,
            characters=request.characters,
            warnings=request.warnings,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return ScriptSceneResponse(**data)


@router.post(
    "/api/script-scenes/{scene_id}/regenerate",
    response_model=ScriptSceneResponse,
)
async def post_regenerate_script_scene(
    scene_id: int,
    request: ScriptSceneRegenerateRequest,
    provider: LLMProvider = Depends(get_script_generation_provider),
    database_path: DatabasePath | None = Depends(
        get_script_generation_database_path
    ),
) -> ScriptSceneResponse:
    try:
        health = await provider.health_check()
        if not health.available:
            raise LLMProviderUnavailableError(health.detail)

        data = await regenerate_script_scene(
            scene_id=scene_id,
            provider=provider,
            instruction=request.instruction,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
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

    return ScriptSceneResponse(**data)
