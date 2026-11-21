from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .gateway_config import GatewayConfigError, MAX_VERSION


class ConfigureGateway(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1, max_length=320)
    token: SecretStr | None = Field(default=None, min_length=24, max_length=256)


class RetryGateway(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = Field(ge=1, le=MAX_VERSION)


class PrivateValidationRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private_handler(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # Pydantic errors include raw inputs, even for rejected SecretStr values.
                raise HTTPException(422, "连接设置格式不正确，请检查地址、令牌和版本。") from None

        return private_handler


def create_gateway_config_router(service, require_parent):
    router = APIRouter(prefix="/api/gateway", dependencies=[Depends(require_parent)],
                       route_class=PrivateValidationRoute)

    async def call(operation):
        try:
            return await operation
        except GatewayConfigError as error:
            raise HTTPException(error.status_code, str(error)) from error

    @router.get("/devices/{device}")
    async def status(device: str):
        return await call(service.status(device))

    @router.post("/devices/{device}/configure")
    async def configure(device: str, request: ConfigureGateway):
        token = request.token.get_secret_value() if request.token is not None else None
        return await call(service.configure(device, request.url, token))

    @router.post("/devices/{device}/retry")
    async def retry(device: str, request: RetryGateway):
        return await call(service.retry(device, request.version))

    return router
