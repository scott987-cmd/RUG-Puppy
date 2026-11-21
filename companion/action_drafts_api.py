"""Parent-only review. Child proposals enter through the shared chat flow."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .action_drafts import DraftError


class DraftVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = Field(ge=1)


class DraftEdit(DraftVersion):
    name: str = Field(min_length=1, max_length=32)
    steps: list[dict] = Field(min_length=1, max_length=6)


def create_action_drafts_router(service, require_parent):
    router = APIRouter(prefix="/api/action-drafts", dependencies=[Depends(require_parent)])

    async def call(operation):
        try:
            return await operation
        except DraftError as error:
            raise HTTPException(error.status_code, str(error)) from error

    @router.get("/devices/{device_id}")
    async def drafts(device_id: str):
        return await call(service.list(device_id))

    @router.get("/{draft_id}")
    async def draft(draft_id: str):
        return await call(service.get(draft_id))

    @router.patch("/{draft_id}")
    async def edit(draft_id: str, request: DraftEdit):
        return await call(service.edit(draft_id, request.version, request.name, request.steps))

    @router.post("/{draft_id}/approve")
    async def approve(draft_id: str, request: DraftVersion):
        return await call(service.approve(draft_id, request.version))

    @router.post("/{draft_id}/reject")
    async def reject(draft_id: str, request: DraftVersion):
        return await call(service.reject(draft_id, request.version))

    @router.post("/{draft_id}/revoke")
    async def revoke(draft_id: str, request: DraftVersion):
        return await call(service.revoke(draft_id, request.version))

    @router.delete("/{draft_id}")
    async def delete(draft_id: str, request: DraftVersion):
        return await call(service.delete(draft_id, request.version))

    return router
