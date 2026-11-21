"""Identity switch tests use temporary SQLite and fake features only.

No microphone, audio file, speaker embedding model, synthesis, playback, camera,
real family sample, or device connection is used.
"""
import asyncio
import concurrent.futures
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from companion.identity import IdentityService, IdentityStore, Observation
from companion.identity_api import IdentityRegistration, create_identity_router


class FakeEngine:
    model_id = "settings-test-features"

    def __init__(self, modality):
        self.modality, self.calls = modality, 0

    def extract(self, data, *, captured_at):
        if self.modality != "face":
            raise AssertionError("These settings tests must not process audio")
        self.calls += 1
        return Observation("face", self.model_id, [1, .1], {}, captured_at)


def registration(path, **kwargs):
    service = IdentityService(IdentityStore(path), FakeEngine("face"), FakeEngine("voice"))
    return IdentityRegistration(service, **kwargs)


def application(registrations):
    app = FastAPI()

    def parent(authorization: str = Header(default="")):
        if authorization != "Bearer settings-test-parent":
            raise HTTPException(401, "parent required")

    app.include_router(create_identity_router(registrations, parent))
    return app


AUTH = {"Authorization": "Bearer settings-test-parent"}
URL = "/api/identity/devices/puppy-1/settings"


class IdentitySettingsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="identity-settings-")
        self.path = Path(self.temp.name) / "identity.sqlite3"
        self.reg = registration(self.path)
        self.client = TestClient(application(self.reg))

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_new_device_defaults_off_and_other_device_is_isolated(self):
        self.assertEqual(self.client.get(URL, headers=AUTH).json(),
                         {"device_id": "puppy-1", "enabled": False, "revision": 0})
        self.assertFalse(self.reg.identity_allowed("puppy-1"))
        changed = self.client.put(URL, json={"enabled": True}, headers=AUTH).json()
        self.assertEqual(changed, {"device_id": "puppy-1", "enabled": True, "revision": 1})
        self.assertEqual(self.reg.identity_settings("puppy-2"),
                         {"device_id": "puppy-2", "enabled": False, "revision": 0})
        self.assertEqual(self.reg.service.face_engine.calls, 0)
        self.assertEqual(self.reg.service.voice_engine.calls, 0)

    def test_both_settings_endpoints_require_parent(self):
        self.assertEqual(self.client.get(URL).status_code, 401)
        self.assertEqual(self.client.put(URL, json={"enabled": True}).status_code, 401)
        self.assertEqual(self.reg.identity_settings("puppy-1")["revision"], 0)

    def test_strict_payload_and_invalid_device_do_not_mutate_settings(self):
        for value in ({}, {"enabled": None}, {"enabled": "false"}, {"enabled": 1},
                      {"enabled": False, "revision": 77}, {"enabled": []}):
            with self.subTest(value=value):
                self.assertEqual(self.client.put(URL, json=value, headers=AUTH).status_code, 422)
        for device in ("has%20space", "x" * 65):
            self.assertEqual(self.client.get(f"/api/identity/devices/{device}/settings", headers=AUTH).status_code, 422)
        self.assertEqual(self.reg.identity_settings("puppy-1")["revision"], 0)

    def test_revision_survives_restart_and_old_results_stay_invalid_after_reenable(self):
        for enabled, expected in ((True, 1), (False, 2), (True, 3), (True, 4)):
            value = self.client.put(URL, json={"enabled": enabled}, headers=AUTH).json()
            self.assertEqual(value["revision"], expected)
        reopened = registration(self.path)
        self.assertTrue(reopened.identity_allowed("puppy-1", 4))
        for stale in (0, 1, 2, 3, 99, True, "4", -1):
            self.assertFalse(reopened.identity_allowed("puppy-1", stale))

    def test_enrollment_never_enables_and_disabling_keeps_samples(self):
        self.reg.save_member("child-1", "测试成员")
        # The fake extractor receives a marker, never image/audio media.
        asyncio.run(self.reg.enroll_data("child-1", "face", b"fake-feature-marker"))
        samples = self.reg.service.store.list_samples("child-1")
        self.assertEqual(len(samples), 1)
        self.assertFalse(self.reg.identity_allowed("puppy-1"))
        self.client.put(URL, json={"enabled": True}, headers=AUTH)
        self.client.put(URL, json={"enabled": False}, headers=AUTH)
        self.assertEqual(self.reg.service.store.list_samples("child-1"), samples)
        self.assertEqual(self.reg.member("child-1")["display_name"], "测试成员")
        self.assertFalse(registration(self.path).identity_allowed("puppy-1"))
        asyncio.run(self.reg.revoke_person("child-1"))
        self.assertFalse(self.reg.identity_allowed("puppy-1"))
        self.assertEqual(self.reg.service.voice_engine.calls, 0)

    def test_distinct_instances_serialize_revision_in_sqlite(self):
        def change(index):
            return asyncio.run(registration(self.path).set_identity_settings("puppy-1", bool(index % 2)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            changes = list(workers.map(change, range(12)))
        self.assertEqual(sorted(value["revision"] for value in changes), list(range(1, 13)))
        last = max(changes, key=lambda item: item["revision"])
        self.assertEqual(self.reg.identity_settings("puppy-1"), last)


class IdentitySettingsGuardTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="identity-settings-guard-")
        self.guard = asyncio.Lock()
        self.reg = registration(Path(self.temp.name) / "identity.sqlite3", settings_guard=self.guard)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=application(self.reg)), base_url="http://test")
        self.initial = await self.reg.set_identity_settings("puppy-1", True)

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temp.cleanup()

    async def test_disable_response_waits_for_prior_personal_commit_then_blocks_new_turn(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        events = []

        async def personal_turn():
            async with self.guard:
                self.assertTrue(self.reg.identity_allowed("puppy-1", self.initial["revision"]))
                entered.set()
                await finish.wait()
                events.append("personal-memory-committed")

        async def disable():
            response = await self.client.put(URL, json={"enabled": False}, headers=AUTH)
            self.assertEqual(response.status_code, 200)
            events.append("disable-returned")
            return response.json()

        turn = asyncio.create_task(personal_turn())
        await entered.wait()
        closing = asyncio.create_task(disable())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertTrue(self.reg.identity_allowed("puppy-1"))
        finish.set()
        await turn
        result = await closing
        self.assertFalse(result["enabled"])
        self.assertEqual(events, ["personal-memory-committed", "disable-returned"])
        async with self.guard:
            self.assertFalse(self.reg.identity_allowed("puppy-1", self.initial["revision"]))
        await self.reg.set_identity_settings("puppy-1", True)
        async with self.guard:
            self.assertFalse(self.reg.identity_allowed("puppy-1", self.initial["revision"]))

    async def test_cancelled_waiting_change_never_commits_later(self):
        async with self.guard:
            changing = asyncio.create_task(self.reg.set_identity_settings("puppy-1", False))
            await asyncio.sleep(0)
            changing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await changing
        self.assertEqual(self.reg.identity_settings("puppy-1"), self.initial)
        self.assertTrue(self.reg.identity_allowed("puppy-1", self.initial["revision"]))


if __name__ == "__main__":
    unittest.main()
