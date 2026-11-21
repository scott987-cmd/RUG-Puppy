"""Parent-only API for the stationary patrol service.

app.include_router(create_patrol_router(patrol, require_parent))
Serve patrol_panel.js as a static JS asset, then mountPatrolPanel(container,
    {fetch: authenticatedFetch, getDeviceId: () => selectedDeviceId}).
Start/close the service in the application's lifespan. Enabling a plan schedules
a future capture; there is deliberately no endpoint for immediate capture.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from .patrol import PatrolConfig


class PatrolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = False
    interval_minutes: int = Field(default=60, ge=15, le=1440)
    window_start: str = "09:00"
    window_end: str = "21:00"
    timezone: str = "Asia/Shanghai"
    describe: bool = False
    retention_days: int = Field(default=7, ge=1, le=30)


def create_patrol_router(service, require_parent):
    router = APIRouter(prefix="/api/patrol", dependencies=[Depends(require_parent)])

    @router.get("/devices/{device_id}")
    async def configuration(device_id: str):
        try:
            return service.configuration(device_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.put("/devices/{device_id}")
    async def configure(device_id: str, request: PatrolInput):
        try:
            return service.configure(device_id, PatrolConfig(**request.model_dump()))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.get("/devices/{device_id}/runs")
    async def history(device_id: str):
        try:
            return {"runs": service.history(device_id)}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.get("/photos/{photo_id}")
    async def photo(photo_id: str):
        try:
            data = service.read_photo(photo_id)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise HTTPException(404, "巡看照片不存在或已清理") from error
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

    @router.post("/cleanup")
    async def cleanup():
        return service.cleanup()

    return router
