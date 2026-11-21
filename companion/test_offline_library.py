"""Protocol/ledger/API fault tests: fake command, temporary SQLite, no robot."""
import asyncio
from copy import deepcopy
import os
import sqlite3
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from .offline_library import LibraryError, MAX_VERSION, OfflineLibraryService, normalize_action


class OfflineLibraryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="offline-library-test-")
        self.database = Path(self.temp.name) / "library.sqlite3"
        self.saved = {"开心舞": [{"action": "wave", "wait_ms": 0}, {"action": "sit", "wait_ms": 100}],
                      "小抱抱": [{"action": "hug", "wait_ms": 0}]}
        self.device = {"puppy": {"offline_library_version": 0, "offline_library_count": 0},
                       "other": {"offline_library_version": 0, "offline_library_count": 0}}
        self.sent = []
        self.fail = None
        self.override_receipt = None
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.hold = False
        self.service = self.make_service()

    async def asyncTearDown(self):
        self.release.set()
        self.temp.cleanup()

    def make_service(self):
        return OfflineLibraryService(self.database, command=self.command,
                                     get_status=lambda device: deepcopy(self.device.get(device)),
                                     approved_actions_snapshot=lambda: deepcopy(self.saved), timeout=.1)

    async def command(self, device, kind, **payload):
        self.sent.append((device, kind, deepcopy(payload)))
        self.entered.set()
        if self.hold and device == "puppy":
            await self.release.wait()
        if self.fail:
            raise self.fail
        if self.override_receipt is not None:
            return deepcopy(self.override_receipt)
        self.device[device] = {"offline_library_version": payload["version"], "offline_library_count": len(payload["actions"])}
        return {"ok": True, "phase": "saved", "executed": False,
                "version": payload["version"], "action_count": len(payload["actions"])}

    async def test_status_and_restart_never_sync_or_execute(self):
        self.assertIsNone((await self.service.status("puppy"))["latest"])
        other = self.make_service()
        await other.status("puppy")
        self.assertEqual(self.sent, [])

    async def test_explicit_selection_snapshot_saves_and_only_sends_library(self):
        state = await self.service.sync("puppy", ["开心舞", "小抱抱"])
        self.assertEqual(state["latest"]["state"], "saved")
        self.assertEqual(state["latest"]["version"], 1)
        self.assertFalse(state["latest"]["receipt"]["executed"])
        self.assertEqual(self.sent[0][1], "set_offline_actions")
        self.assertEqual(self.sent[0][2]["actions"][0]["steps"][0], {"action": "wave"})
        self.assertEqual(len(self.sent), 1)

    async def test_timeout_retry_uses_identical_version_and_snapshot_after_edit(self):
        self.hold = True
        state = await self.service.sync("puppy", ["开心舞"])
        self.assertEqual(state["latest"]["state"], "unknown")
        original = deepcopy(self.sent[0][2])
        self.saved["开心舞"] = [{"action": "hug"}]
        self.hold = False
        state = await self.make_service().retry("puppy", 1)
        self.assertEqual(self.sent[1][2], original)
        self.assertEqual(state["latest"]["state"], "saved")
        self.assertEqual(state["latest"]["attempts"], 2)

    async def test_cancelled_request_remains_unknown_and_can_retry_after_restart(self):
        self.hold = True
        task = asyncio.create_task(self.service.sync("puppy", ["开心舞"]))
        await self.entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.hold = False
        restarted = self.make_service()
        self.assertEqual((await restarted.status("puppy"))["latest"]["state"], "unknown")
        self.assertEqual((await restarted.retry("puppy", 1))["latest"]["state"], "saved")

    async def test_persisted_sending_is_unknown_after_restart_without_auto_replay(self):
        await self.service.sync("puppy", ["开心舞"])
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE library_versions SET state='sending'")
        count = len(self.sent)
        status = await self.make_service().status("puppy")
        self.assertEqual(status["latest"]["state"], "unknown")
        self.assertEqual(len(self.sent), count)

    async def test_device_higher_version_is_baseline_and_conflicts_with_old_retry(self):
        self.device["puppy"]["offline_library_version"] = 12
        await self.service.sync("puppy", ["开心舞"])
        self.assertEqual(self.sent[-1][2]["version"], 13)
        self.device["puppy"]["offline_library_version"] = 20
        with self.assertRaises(LibraryError):
            await self.service.retry("puppy", 13)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual((await self.service.status("puppy"))["latest"]["state"], "conflict")
        await self.service.sync("puppy", ["小抱抱"])
        self.assertEqual(self.sent[-1][2]["version"], 21)

    async def test_local_highwater_survives_device_reset_and_service_restart(self):
        await self.service.sync("puppy", ["开心舞"])
        self.device["puppy"]["offline_library_version"] = 0
        await self.make_service().sync("puppy", ["小抱抱"])
        self.assertEqual([p[2]["version"] for p in self.sent], [1, 2])

    async def test_unknown_can_be_explicitly_superseded_or_cleared_but_history_stays(self):
        self.fail = TimeoutError()
        await self.service.sync("puppy", ["开心舞"])
        self.fail = None
        state = await self.service.sync("puppy", [])
        self.assertEqual(self.sent[-1][2], {"version": 2, "actions": []})
        self.assertEqual([row["state"] for row in state["history"]], ["saved", "unknown"])
        self.assertEqual(state["history"][1]["actions"][0]["name"], "开心舞")
        with self.assertRaises(LibraryError):
            await self.service.retry("puppy", 1)

    async def test_offline_old_firmware_and_exhausted_versions_do_not_send(self):
        for state in (None, {}, {"offline_library_version": True, "offline_library_count": 0},
                      {"offline_library_version": MAX_VERSION, "offline_library_count": 0}):
            with self.subTest(state=state):
                self.device["puppy"] = state
                with self.assertRaises(LibraryError):
                    await self.service.sync("puppy", ["开心舞"])
        self.assertEqual(self.sent, [])

    async def test_invalid_selection_and_stored_steps_do_not_send(self):
        for names in (["不存在"], ["开心舞", "开心舞"], ["开心舞"] * 4, [None], "开心舞"):
            with self.assertRaises(LibraryError):
                await self.service.sync("puppy", names)
        invalid = [[], [{"action": "stop"}], [{"action": "wave", "angle": 90}],
                   [{"wait_ms": 0}], [{"action": None}], [{"wait_ms": True}],
                   [{"wait_ms": -1}], [{"wait_ms": 3001}], [{"action": "wave"}] * 7]
        for steps in invalid:
            self.saved["坏动作"] = steps
            with self.assertRaises(LibraryError):
                await self.service.sync("puppy", ["坏动作"])
        self.assertEqual(self.sent, [])

    async def test_strict_receipts_do_not_claim_saved_or_executed(self):
        valid = {"ok": True, "phase": "saved", "executed": False, "version": 1, "action_count": 1}
        for patch in ({"ok": False}, {"ok": 1}, {"phase": "accepted"}, {"executed": True},
                      {"executed": 0}, {"version": True}, {"version": 99}, {"action_count": 0}):
            self.override_receipt = {**valid, **patch}
            state = await self.service.sync("puppy", ["开心舞"])
            self.assertEqual(state["latest"]["state"], "unknown")
            self.assertIsNone(state["latest"]["receipt"]["executed"])
            valid["version"] += 1

    async def test_different_devices_do_not_block_each_other(self):
        self.hold = True
        self.service.timeout = 2
        first = asyncio.create_task(self.service.sync("puppy", ["开心舞"]))
        await self.entered.wait()
        second = await asyncio.wait_for(self.service.sync("other", ["小抱抱"]), .5)
        self.assertEqual(second["latest"]["state"], "saved")
        self.assertFalse(first.done())
        self.release.set()
        await first

    async def test_same_device_serializes_new_versions(self):
        self.hold = True
        self.service.timeout = 2
        first = asyncio.create_task(self.service.sync("puppy", ["开心舞"]))
        await self.entered.wait()
        second = asyncio.create_task(self.service.sync("puppy", ["小抱抱"]))
        await asyncio.sleep(.02)
        self.assertEqual(len(self.sent), 1)
        self.release.set()
        await asyncio.gather(first, second)
        self.assertEqual([p[2]["version"] for p in self.sent], [1, 2])

    async def test_database_work_does_not_block_event_loop(self):
        entered, release = threading.Event(), threading.Event()
        original = self.service._allocate
        def slow(*args):
            entered.set()
            release.wait(2)
            return original(*args)
        self.service._allocate = slow
        task = asyncio.create_task(self.service.sync("puppy", ["开心舞"]))
        try:
            async with asyncio.timeout(.5):
                while not entered.is_set():
                    await asyncio.sleep(.002)
            await asyncio.sleep(.01)
            self.assertFalse(task.done())
        finally:
            release.set()
            await task

    async def test_sqlite_version_allocation_is_atomic_across_instances(self):
        second = self.make_service()
        await asyncio.gather(self.service.sync("puppy", ["开心舞"]), second.sync("puppy", ["小抱抱"]))
        history = (await self.service.status("puppy"))["history"]
        self.assertEqual([item["version"] for item in history], [2, 1])

    async def test_cancelled_database_write_keeps_lock_until_it_settles(self):
        await self.check_cancelled_database_write(1)

    async def test_repeated_cancellation_keeps_device_lock_until_write_finishes(self):
        await self.check_cancelled_database_write(3)

    async def check_cancelled_database_write(self, cancellations):
        entered, release = threading.Event(), threading.Event()
        original = self.service._allocate
        first_write = True
        def slow(*args):
            nonlocal first_write
            if first_write:
                first_write = False
                entered.set()
                release.wait(2)
            return original(*args)
        self.service._allocate = slow
        first = asyncio.create_task(self.service.sync("puppy", ["开心舞"]))
        async with asyncio.timeout(.5):
            while not entered.is_set():
                await asyncio.sleep(.002)
        first.cancel()
        second = asyncio.create_task(self.service.sync("puppy", ["小抱抱"]))
        try:
            await asyncio.sleep(.02)
            for _ in range(cancellations - 1):
                first.cancel()
                await asyncio.sleep(.01)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            self.assertEqual(self.sent, [])
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
            result = await second
        self.assertEqual(result["latest"]["version"], 2)
        self.assertEqual(result["latest"]["actions"][0]["name"], "小抱抱")
        self.assertEqual([p[2]["version"] for p in self.sent], [2])

    async def test_database_created_private_even_under_permissive_umask(self):
        self.service.database = Path(self.temp.name) / "private-library" / "library.sqlite3"
        previous = os.umask(0)
        try:
            await self.service.status("puppy")
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(self.service.database.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.service.database.parent.stat().st_mode), 0o700)
        self.service.database.chmod(0o644)
        await self.service.status("puppy")
        self.assertEqual(stat.S_IMODE(self.service.database.stat().st_mode), 0o600)

    async def test_api_strict_parent_fields_and_version_binding(self):
        import httpx
        from fastapi import FastAPI, Header, HTTPException
        from .offline_library_api import create_offline_library_router
        async def parent(authorization: str | None = Header(default=None)):
            if authorization != "Bearer test":
                raise HTTPException(401)
        app = FastAPI()
        app.include_router(create_offline_library_router(self.service, parent))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            path = "/api/offline-library/devices/puppy"
            self.assertEqual((await client.get(path)).status_code, 401)
            self.assertEqual((await client.post(path + "/sync", json={"names": []})).status_code, 401)
            client.headers["Authorization"] = "Bearer test"
            for body in ({"names": ["开心舞"], "version": 900}, {"names": [], "steps": []}, {"names": [1]}):
                self.assertEqual((await client.post(path + "/sync", json=body)).status_code, 422)
            self.assertEqual((await client.post(path + "/sync", json={"names": ["开心舞"]})).status_code, 200)
            self.assertEqual((await client.post(path + "/retry", json={"version": True})).status_code, 422)
            self.assertEqual((await client.post(path + "/retry", json={"version": 2})).status_code, 409)
            self.assertEqual(len(self.sent), 1)


class NormalizationTests(unittest.TestCase):
    def test_utf8_byte_limit_controls_and_wait_only_normalization(self):
        for name in ("", "x\n", "\x00", "\ud800", "狗" * 22):
            with self.assertRaises(ValueError):
                normalize_action(name, [{"action": "wave"}])
        result = normalize_action("狗" * 21, [{"action": None, "wait_ms": 300}, {"action": "sit", "wait_ms": 0}])
        self.assertEqual(result["steps"], [{"wait_ms": 300}, {"action": "sit"}])


if __name__ == "__main__":
    unittest.main()
