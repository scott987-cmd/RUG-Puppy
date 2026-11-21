"""Parent-only selection and explicit sync/retry API; no action execution route."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .offline_library import LibraryError, MAX_VERSION


class LibrarySelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    names: list[str] = Field(max_length=3)


class LibraryRetry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = Field(ge=1, le=MAX_VERSION)


def create_offline_library_router(service, require_parent):
    router = APIRouter(prefix="/api/offline-library", dependencies=[Depends(require_parent)])

    async def call(operation):
        try:
            return await operation
        except LibraryError as error:
            raise HTTPException(error.status_code, str(error)) from error

    @router.get("/devices/{device_id}")
    async def status(device_id: str):
        return await call(service.status(device_id))

    @router.post("/devices/{device_id}/sync")
    async def synchronize(device_id: str, request: LibrarySelection):
        return await call(service.sync(device_id, request.names))

    @router.post("/devices/{device_id}/retry")
    async def retry(device_id: str, request: LibraryRetry):
        return await call(service.retry(device_id, request.version))

    return router
