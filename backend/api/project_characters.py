from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.services.project_characters import (
    build_project_characters,
    get_latest_project_character_run,
    get_project_character_run,
    suppress_ordinary_character,
    update_character_pin,
)
from backend.storage.database import DatabasePath


router = APIRouter(tags=["project-characters"])


class ProjectCharacterBuildRequest(BaseModel):
    narrative_run_id: int | None = None


class ProjectCharacterResponse(BaseModel):
    id: int
    character_id: str
    canonical_name: str
    aliases: list[str]
    references: list[str]
    mention_ids: list[str]
    source_candidate_ids: list[str]
    evidence_count: int
    is_user_pinned: bool
    input_quality: dict[str, Any]


class ProjectCharacterPinRequest(BaseModel):
    is_user_pinned: bool


class ProjectCharacterMergeDecisionResponse(BaseModel):
    id: int
    left_candidate_id: str
    right_candidate_id: str
    decision: str
    merge_score: float
    evidence: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]


class ProjectCharacterInputGapResponse(BaseModel):
    id: int
    source_unit_id: int | None
    chunk_id: str | None
    unit_status: str | None
    layer_name: str | None
    reason: str


class ProjectCharacterRunResponse(BaseModel):
    id: int
    project_id: str
    narrative_run_id: int
    status: str
    error_message: str | None
    total_units: int
    used_units: int
    skipped_units: int
    total_candidates: int
    merged_characters: int
    ambiguous_pairs: int
    created_at: str
    updated_at: str
    characters: list[ProjectCharacterResponse]
    merge_decisions: list[ProjectCharacterMergeDecisionResponse]
    input_gaps: list[ProjectCharacterInputGapResponse]


def get_character_database_path() -> DatabasePath | None:
    return None


def _to_response(
    data: dict[str, Any],
) -> ProjectCharacterRunResponse:
    return ProjectCharacterRunResponse(
        id=data["id"],
        project_id=data["project_id"],
        narrative_run_id=data["narrative_run_id"],
        status=data["status"],
        error_message=data["error_message"],
        total_units=data["total_units"],
        used_units=data["used_units"],
        skipped_units=data["skipped_units"],
        total_candidates=data["total_candidates"],
        merged_characters=data["merged_characters"],
        ambiguous_pairs=data["ambiguous_pairs"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        characters=[
            ProjectCharacterResponse(**character)
            for character in data["characters"]
        ],
        merge_decisions=[
            ProjectCharacterMergeDecisionResponse(**decision)
            for decision in data["merge_decisions"]
        ],
        input_gaps=[
            ProjectCharacterInputGapResponse(**gap)
            for gap in data["input_gaps"]
        ],
    )


@router.post(
    "/api/projects/{project_id}/characters/build",
    response_model=ProjectCharacterRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def build_project_character_table(
    project_id: str,
    request: ProjectCharacterBuildRequest,
    database_path: DatabasePath | None = Depends(
        get_character_database_path
    ),
) -> ProjectCharacterRunResponse:
    try:
        data = build_project_characters(
            project_id=project_id,
            narrative_run_id=request.narrative_run_id,
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

    return _to_response(data)


@router.get(
    "/api/project-character-runs/{character_run_id}",
    response_model=ProjectCharacterRunResponse,
    status_code=status.HTTP_200_OK,
)
def read_project_character_run(
    character_run_id: int,
    database_path: DatabasePath | None = Depends(
        get_character_database_path
    ),
) -> ProjectCharacterRunResponse:
    try:
        data = get_project_character_run(
            character_run_id=character_run_id,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return _to_response(data)


@router.get(
    "/api/projects/{project_id}/characters/latest",
    response_model=ProjectCharacterRunResponse | None,
    status_code=status.HTTP_200_OK,
)
def read_latest_project_character_run(
    project_id: str,
    database_path: DatabasePath | None = Depends(
        get_character_database_path
    ),
) -> ProjectCharacterRunResponse | None:
    data = get_latest_project_character_run(
        project_id=project_id,
        database_path=database_path,
    )

    if data is None:
        return None

    return _to_response(data)


@router.patch(
    "/api/project-characters/{character_row_id}/pin",
    response_model=ProjectCharacterRunResponse,
    status_code=status.HTTP_200_OK,
)
def update_project_character_pin(
    character_row_id: int,
    request: ProjectCharacterPinRequest,
    database_path: DatabasePath | None = Depends(
        get_character_database_path
    ),
) -> ProjectCharacterRunResponse:
    try:
        data = update_character_pin(
            character_row_id=character_row_id,
            is_user_pinned=request.is_user_pinned,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return _to_response(data)


@router.delete(
    "/api/project-characters/{character_row_id}",
    response_model=ProjectCharacterRunResponse,
    status_code=status.HTTP_200_OK,
)
def delete_ordinary_project_character(
    character_row_id: int,
    database_path: DatabasePath | None = Depends(
        get_character_database_path
    ),
) -> ProjectCharacterRunResponse:
    try:
        data = suppress_ordinary_character(
            character_row_id=character_row_id,
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

    return _to_response(data)
