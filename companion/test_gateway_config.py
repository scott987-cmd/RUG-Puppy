"""Gateway provisioning tests use temporary keys/SQLite and a fake device only."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from .gateway_config import GatewayConfigError, GatewayConfigService, validate_configuration


TOKEN = "test-only-pairing-token-abcdefghijk"
URL = "ws://192.0.2.78:8000/v1/audio"


class ConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-gateway-config-")
        self.database = Path(self.temp.name) / "private" / "config.sqlite3"
        self.device = {"gateway_config_version": 0, "gateway_config_override": False,
                       "gateway_config_url": URL, "gateway_config_restart_required": False}
        self.commands = []
        self.block = False
        self.started = asyncio.Event()
        async def command(device, kind, **payload):
            self.commands.append((device, kind, dict(payload)))
            self.started.set()
            if self.block:
                await asyncio.Event().wait()
            self.device.update(gateway_config_version=payload["version"], gateway_config_override=True,
                               gateway_config_restart_required=True, gateway_config_url=payload["url"])
            return {"ok": True, "phase": "saved", "version": payload["version"],
                    "executed": False, "restart_required": True}
        self.service = GatewayConfigService(self.database, command=command, get_status=lambda device: self.device,
                                           current_token=lambda: TOKEN, timeout=.04)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_status_is_read_only_and_never_returns_tokens(self):
        self.assertIsNone((await self.service.status("puppy"))["latest"])
        result = await self.service.configure("puppy", URL)
        self.assertEqual(result["latest"]["state"], "saved")
        self.assertNotIn(TOKEN, json.dumps(result))
        self.assertNotIn("payload", result["latest"])
        self.assertFalse(result["save_restarts_device"])
        self.assertEqual([item[1] for item in self.commands], ["set_gateway_config"])
        self.assertEqual(self.database.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.database.parent.stat().st_mode & 0o777, 0o700)

    async def test_timeout_retry_preserves_version_token_and_url(self):
        self.block = True
        first = await self.service.configure("puppy", URL)
        self.assertEqual(first["latest"]["state"], "unknown")
        self.service.current_token = lambda: "different-but-valid-token-abcdefgh"
        self.block = False
        second = await self.service.retry("puppy", 1)
        self.assertEqual(second["latest"]["version"], 1)
        self.assertEqual(second["latest"]["attempts"], 2)
        self.assertEqual(self.commands[0][2], self.commands[1][2])

    async def test_cancelled_send_does_not_restart_or_replay(self):
        self.block = True
        task = asyncio.create_task(self.service.configure("puppy", URL))
        await self.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        result = await self.service.status("puppy")
        self.assertEqual(result["latest"]["state"], "unknown")
        self.assertEqual(len(self.commands), 1)

    async def test_explicit_new_configuration_advances_high_water(self):
        self.device["gateway_config_version"] = 7
        self.block = True
        first = await self.service.configure("puppy", URL)
        self.block = False
        second = await self.service.configure("puppy", "wss://puppy.local:8443/v1/audio")
        self.assertEqual(first["latest"]["version"], 8)
        self.assertEqual(second["latest"]["version"], 9)
        with self.assertRaises(GatewayConfigError):
            await self.service.retry("puppy", 8)

    async def test_higher_device_version_rejects_old_retry(self):
        await self.service.configure("puppy", URL)
        self.device["gateway_config_version"] = 2
        with self.assertRaises(GatewayConfigError):
            await self.service.retry("puppy", 1)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual((await self.service.status("puppy"))["latest"]["state"], "conflict")

    async def test_offline_and_legacy_devices_receive_nothing(self):
        for value in (None, {}, {"gateway_config_version": True}):
            self.device = value
            with self.assertRaises(GatewayConfigError):
                await self.service.configure("puppy", URL)
        self.assertEqual(self.commands, [])

    async def test_unsuitable_or_mismatched_receipt_remains_unknown_without_secret_echo(self):
        for bad in ({"executed": True}, {"version": True}, {"restart_required": False}, {"phase": "accepted"}):
            async def command(device, kind, **payload):
                return {"ok":True,"phase":"saved","executed":False,"restart_required":True,
                        "version":payload["version"],"token":TOKEN,**bad}
            self.service.command = command
            result = await self.service.configure("puppy", URL)
            self.assertEqual(result["latest"]["state"], "unknown")
            self.assertNotIn(TOKEN, json.dumps(result))

    async def test_peer_error_cannot_echo_a_secret(self):
        async def fail(*args, **kwargs):
            raise RuntimeError(TOKEN)
        self.service.command = fail
        result = await self.service.configure("puppy", URL)
        self.assertEqual(result["latest"]["state"], "unknown")
        self.assertNotIn(TOKEN, json.dumps(result))

    async def test_parallel_new_choices_are_serialized(self):
        first, second = await asyncio.gather(self.service.configure("puppy", URL), self.service.configure("puppy", URL))
        self.assertEqual([item[2]["version"] for item in self.commands], [1, 2])
        self.assertEqual((await self.service.status("puppy"))["latest"]["version"], 2)

    async def test_strict_parent_routes(self):
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.testclient import TestClient
        from .gateway_config_api import create_gateway_config_router
        def parent(authorization: str = Header(default="")):
            if authorization != "Bearer test-parent":
                raise HTTPException(401)
        app = FastAPI()
        app.include_router(create_gateway_config_router(self.service, parent))
        with TestClient(app) as client:
            base = "/api/gateway/devices/puppy"
            self.assertEqual(client.get(base).status_code, 401)
            headers = {"Authorization": "Bearer test-parent"}
            for token in ("short-secret", "secret"*50, 123456789, {"secret": TOKEN}, [TOKEN]):
                reply = client.post(base+"/configure",json={"url":URL,"token":token},headers=headers)
                self.assertEqual(reply.status_code,422)
                self.assertEqual(reply.json(), {"detail":"连接设置格式不正确，请检查地址、令牌和版本。"})
                self.assertNotIn(str(token), reply.text)
            reply = client.post(base+"/configure",json={"url":URL,"unexpected":TOKEN},headers=headers)
            self.assertEqual(reply.status_code,422)
            self.assertNotIn(TOKEN, reply.text)
            reply = client.post(base+"/configure",content='{"token":"'+TOKEN+'",',headers=headers)
            self.assertEqual(reply.status_code,422)
            self.assertNotIn(TOKEN, reply.text)
            reply = client.post(base+"/configure",json={"url":URL,"token":TOKEN},headers=headers)
            self.assertEqual(reply.status_code,200)
            self.assertNotIn(TOKEN, reply.text)
            self.assertEqual(client.post(base+"/retry",json={"version":True},headers=headers).status_code,422)


class ConfigurationValidationTests(unittest.TestCase):
    def test_invalid_urls_are_rejected_before_any_save(self):
        for url in ("https://host:8000/v1/audio", "ws://host/v1/audio", "ws://user:pass@host:8000/v1/audio",
                    URL+"?token=secret", URL+"#secret", URL+"?", URL+"#", URL+"\n", "ws://[::1]:8000/v1/audio",
                    "ws://host:0/v1/audio", "ws://host:65536/v1/audio", "ws://host:8000/v1/control",
                    "ws://bad_host:8000/v1/audio", "ws://999.1.1.1:8000/v1/audio", "ws://-bad:8000/v1/audio",
                    "ws://host:000001/v1/audio"):
            with self.subTest(url=url), self.assertRaises(GatewayConfigError):
                validate_configuration(url, TOKEN)

    def test_tokens_are_bounded_without_leaking_them_in_error(self):
        for token in ("short", "x"*257, TOKEN+" ", TOKEN+"\n", "令"*24, None):
            with self.assertRaises(GatewayConfigError) as caught:
                validate_configuration(URL, token)
            if isinstance(token,str) and token:
                self.assertNotIn(token, str(caught.exception))
        for url in (URL, "wss://puppy-gateway.local:8443/v1/audio"):
            self.assertEqual(validate_configuration(url, TOKEN)["url"], url)


if __name__ == "__main__":
    unittest.main()
