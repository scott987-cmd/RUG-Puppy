"""Authenticated parent editing; none of these routes invokes a model or device."""

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .pet_profile import PetProfile, RevisionConflict
from .routine import RoutineService


class VersionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(strict=True, ge=0)


class ProfileInput(VersionInput):
    style: Literal["gentle", "lively", "patient"]
    greeting_action: Literal["none", "wave", "sit", "stretch", "hug"]


class RoutineInput(VersionInput):
    title: StrictStr = Field(min_length=1, max_length=40)
    steps: list[Annotated[StrictStr, Field(min_length=1, max_length=80)]] = Field(min_length=1, max_length=8)


class TransitionInput(VersionInput):
    operation: Literal["start", "advance", "pause", "resume", "reset"]
    expected_step_index: int = Field(strict=True, ge=0, le=8)


def create_routine_router(store, require_parent, *, require_member=None):
    """Optionally pass REGISTRATIONS.member to reject unregistered member IDs."""
    router = APIRouter(dependencies=[Depends(require_parent)])
    profile, routines = PetProfile(store), RoutineService(store)

    async def invoke(method, *args, person_id=None, **kwargs):
        def operation():
            if person_id is not None and require_member is not None:
                require_member(person_id)
            return method(*args, **kwargs)
        try:
            return await asyncio.to_thread(operation)
        except RevisionConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "没有找到属于此成员的流程") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/api/pet-profile/{device_id}")
    async def get_profile(device_id: str):
        return await invoke(profile.get, device_id)

    @router.put("/api/pet-profile/{device_id}")
    async def set_profile(device_id: str, value: ProfileInput):
        return await invoke(profile.set, device_id, **value.model_dump())

    base = "/api/routines/{device_id}/members/{person_id}"

    @router.get(base)
    async def list_routines(device_id: str, person_id: str):
        return {"routines": await invoke(routines.list, device_id, person_id, person_id=person_id)}

    @router.get(base + "/{routine_id}")
    async def get_routine(device_id: str, person_id: str, routine_id: str):
        return await invoke(routines.get, device_id, person_id, routine_id, person_id=person_id)

    @router.put(base + "/{routine_id}")
    async def save_routine(device_id: str, person_id: str, routine_id: str, value: RoutineInput):
        return await invoke(routines.put, device_id, person_id, routine_id, person_id=person_id, **value.model_dump())

    @router.delete(base + "/{routine_id}")
    async def delete_routine(device_id: str, person_id: str, routine_id: str, value: VersionInput):
        return await invoke(routines.delete, device_id, person_id, routine_id, person_id=person_id, **value.model_dump())

    @router.post(base + "/{routine_id}/transition")
    async def transition(device_id: str, person_id: str, routine_id: str, value: TransitionInput):
        return await invoke(routines.transition, device_id, person_id, routine_id, person_id=person_id, **value.model_dump())

    return router
