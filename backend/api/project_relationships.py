from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.services.project_relationships import (
    build_project_relationships,
    create_project_relationship,
    delete_project_relationship,
    get_project_relationships,
    update_project_relationship,
)
from backend.storage.database import DatabasePath


router = APIRouter(tags=["project-relationships"])


class ProjectRelationshipBuildRequest(BaseModel):
    character_run_id: int | None = None


class ProjectRelationshipCreateRequest(BaseModel):
    source_character_id: str
    source_character_name: str
    target_character_id: str
    target_character_name: str
    relation_label: str
    relation_description: str = ""
    evidence_text: str = ""
    source_chunk_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class ProjectRelationshipUpdateRequest(BaseModel):
    relation_label: str | None = None
    relation_description: str | None = None
    evidence_text: str | None = None
    source_chunk_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class ProjectRelationshipResponse(BaseModel):
    id: int
    project_id: str
    source_character_id: str
    source_character_name: str
    target_character_id: str
    target_character_name: str
    relation_label: str
    relation_description: str
    source_type: str
    is_user_edited: bool
    evidence_text: str
    source_chunk_id: str | None
    start_offset: int | None
    end_offset: int | None
    evidence_count: int
    created_at: str
    updated_at: str


class ProjectRelationshipCollectionResponse(BaseModel):
    project_id: str
    core_characters: list[dict[str, Any]]
    relationships: list[ProjectRelationshipResponse]
    core_relationships: list[ProjectRelationshipResponse]


def get_relationship_database_path() -> DatabasePath | None:
    return None


def _collection_response(
    payload: dict[str, Any],
) -> ProjectRelationshipCollectionResponse:
    return ProjectRelationshipCollectionResponse(
        project_id=payload["project_id"],
        core_characters=payload["core_characters"],
        relationships=[
            ProjectRelationshipResponse(**relationship)
            for relationship in payload["relationships"]
        ],
        core_relationships=[
            ProjectRelationshipResponse(**relationship)
            for relationship in payload["core_relationships"]
        ],
    )


@router.post(
    "/api/projects/{project_id}/relationships/build",
    response_model=ProjectRelationshipCollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
def build_relationships(
    project_id: str,
    request: ProjectRelationshipBuildRequest,
    database_path: DatabasePath | None = Depends(
        get_relationship_database_path
    ),
) -> ProjectRelationshipCollectionResponse:
    try:
        payload = build_project_relationships(
            project_id=project_id,
            character_run_id=request.character_run_id,
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

    return _collection_response(payload)


@router.get(
    "/api/projects/{project_id}/relationships",
    response_model=ProjectRelationshipCollectionResponse,
)
def read_relationships(
    project_id: str,
    database_path: DatabasePath | None = Depends(
        get_relationship_database_path
    ),
) -> ProjectRelationshipCollectionResponse:
    payload = get_project_relationships(
        project_id=project_id,
        database_path=database_path,
    )
    return _collection_response(payload)


@router.post(
    "/api/projects/{project_id}/relationships",
    response_model=ProjectRelationshipResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_relationship(
    project_id: str,
    request: ProjectRelationshipCreateRequest,
    database_path: DatabasePath | None = Depends(
        get_relationship_database_path
    ),
) -> ProjectRelationshipResponse:
    try:
        relationship = create_project_relationship(
            project_id=project_id,
            source_character_id=request.source_character_id,
            source_character_name=request.source_character_name,
            target_character_id=request.target_character_id,
            target_character_name=request.target_character_name,
            relation_label=request.relation_label,
            relation_description=request.relation_description,
            evidence_text=request.evidence_text,
            source_chunk_id=request.source_chunk_id,
            start_offset=request.start_offset,
            end_offset=request.end_offset,
            database_path=database_path,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return ProjectRelationshipResponse(**relationship)


@router.patch(
    "/api/project-relationships/{relationship_id}",
    response_model=ProjectRelationshipResponse,
)
def patch_relationship(
    relationship_id: int,
    request: ProjectRelationshipUpdateRequest,
    database_path: DatabasePath | None = Depends(
        get_relationship_database_path
    ),
) -> ProjectRelationshipResponse:
    try:
        relationship = update_project_relationship(
            relationship_id=relationship_id,
            relation_label=request.relation_label,
            relation_description=request.relation_description,
            evidence_text=request.evidence_text,
            source_chunk_id=request.source_chunk_id,
            start_offset=request.start_offset,
            end_offset=request.end_offset,
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

    return ProjectRelationshipResponse(**relationship)


@router.delete(
    "/api/project-relationships/{relationship_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_relationship(
    relationship_id: int,
    database_path: DatabasePath | None = Depends(
        get_relationship_database_path
    ),
) -> None:
    try:
        delete_project_relationship(
            relationship_id=relationship_id,
            database_path=database_path,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
