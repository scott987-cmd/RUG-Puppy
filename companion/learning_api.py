"""Parent settings/status routes; turn accounting is an internal gateway interface.

app.include_router(create_learning_router(learning_store, require_parent))
No endpoint invokes a model, records audio, captures a photo, or changes volume.
"""
import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .learning import LearningStore


class LearningPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    daily_minutes: int | None = Field(default=None, strict=True, ge=1, le=20)
    theme: Literal["animals", "colors", "greetings", "objects", "body_parts"] | None = None
    save_progress: StrictBool | None = None
    photo_learning: StrictBool | None = None


def create_learning_router(store: LearningStore, require_parent) -> APIRouter:
    router = APIRouter(prefix="/api/learning", dependencies=[Depends(require_parent)])

    async def invoke(method, *args, **kwargs):
        try:
            return await asyncio.to_thread(method, *args, **kwargs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/{device_id}")
    async def defaults(device_id: str):
        return await invoke(store.status, device_id)

    @router.patch("/{device_id}")
    async def set_defaults(device_id: str, settings: LearningPatch):
        return await invoke(store.set_settings, device_id, patch=settings.model_dump(exclude_unset=True))

    @router.post("/{device_id}/stop")
    async def stop_guest(device_id: str):
        return await invoke(store.end_session, device_id)

    @router.get("/{device_id}/members/{person_id}")
    async def member(device_id: str, person_id: str):
        return await invoke(store.status, device_id, person_id)

    @router.patch("/{device_id}/members/{person_id}")
    async def set_member(device_id: str, person_id: str, settings: LearningPatch):
        return await invoke(store.set_settings, device_id, person_id, patch=settings.model_dump(exclude_unset=True))

    @router.post("/{device_id}/members/{person_id}/stop")
    async def stop_member(device_id: str, person_id: str):
        return await invoke(store.end_session, device_id, person_id)

    return router
