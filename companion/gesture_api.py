"""Mount create_gesture_router(service, require_parent) on the gateway.

All endpoints require parent authorization, including read-only status.
This API cannot remap gestures, set confidence thresholds or persist enabling.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class GestureEnable(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seconds: int = Field(default=60, ge=1, le=60)
    consent: bool  # Must be true on each explicit enable request.


def create_gesture_router(service, require_parent):
    router = APIRouter(prefix="/api/gestures", dependencies=[Depends(require_parent)])

    @router.get("/devices/{device_id}")
    async def status(device_id: str):
        try:
            return service.status(device_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.post("/devices/{device_id}/enable")
    async def enable(device_id: str, request: GestureEnable):
        if request.consent is not True:
            raise HTTPException(422, "请家长明确同意本次短时拍照和手势动作")
        try:
            return await service.enable(device_id, request.seconds)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/devices/{device_id}/disable")
    async def disable(device_id: str):
        try:
            return await service.disable(device_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return router
