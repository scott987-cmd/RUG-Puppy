"""Stationary patrol tests: fake capture, generated fixture JPEGs, temp dirs only.

python -m unittest companion.test_patrol -v
Never imports the gateway, opens hardware, loads a model, or uses audio services.
"""

import asyncio
from datetime import datetime
import io
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .patrol import PatrolConfig, PatrolService


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


class Clock:
    def __init__(self):
        self.now = timestamp("2026-09-24T10:00:00+08:00")

    def __call__(self):
        return self.now


def fixture_jpeg():
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(output, format="JPEG")
    return output.getvalue()


class PatrolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="puppy-patrol-test-")
        # /var and /tmp aliases on macOS are canonicalized by the test caller;
        # the service itself refuses symlinks in a supplied storage path.
        self.root = Path(self.temporary.name).resolve()
        self.clock = Clock()
        self.calls = []
        self.services = []

        async def capture(device):
            self.calls.append(device)
            return fixture_jpeg()

        self.capture = capture
        self.service = self.create_service()

    def create_service(self, **kwargs):
        service = PatrolService(self.root / "patrol", capture_handler=kwargs.pop("capture_handler", self.capture),
                                clock=self.clock, **kwargs)
        self.services.append(service)
        return service

    async def asyncTearDown(self):
        for service in self.services:
            await service.close()
        self.temporary.cleanup()

    def enable(self, **fields):
        return self.service.configure("puppy", PatrolConfig(enabled=True, interval_minutes=15, **fields))

    async def settle(self):
        tasks = list(self.service._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0)

    async def run_due(self):
        self.clock.now = self.service.configuration("puppy")["next_due"]
        await self.service.tick()
        await self.settle()

    async def test_default_disabled_and_save_only_schedules_future(self):
        self.assertFalse(self.service.configuration("puppy")["enabled"])
        await self.service.tick()
        self.assertFalse(self.calls)
        config = self.enable()
        self.assertEqual(config["next_due"], self.clock.now + 900)
        await self.service.tick()
        self.assertFalse(self.calls)
        await self.run_due()
        self.assertEqual(self.calls, ["puppy"])
        run = self.service.history("puppy")[0]
        self.assertEqual(run["state"], "completed")
        self.assertEqual(self.service.read_photo(run["photo_id"]), fixture_jpeg())
        await self.service.tick()
        self.assertEqual(len(self.calls), 1)

    async def test_database_connections_are_closed_after_context(self):
        with self.service.connect() as db:
            db.execute("SELECT 1")
        with self.assertRaises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")

    async def test_scheduler_recovers_from_storage_failure_without_replaying_due_slot(self):
        self.enable()
        self.clock.now = self.service.configuration("puppy")["next_due"]
        original = self.service._claim_due
        attempts = 0

        def interrupted(now):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("fake disk failure")
            return original(now)

        self.service._claim_due = interrupted
        self.service.poll_seconds = .005
        self.service.start()
        async with asyncio.timeout(1):
            while attempts < 2:
                await asyncio.sleep(.001)
        self.assertFalse(self.calls)
        self.assertEqual(self.service.history("puppy")[0]["state"], "missed")
        self.assertTrue(self.service.status()["running"])
        self.assertEqual(self.service.status()["last_error"]["type"], "OSError")

    async def test_restart_and_late_ticks_never_catch_up(self):
        self.enable()
        due = self.service.configuration("puppy")["next_due"]
        await self.service.close()
        self.clock.now = due + 5000
        self.service = self.create_service()
        await self.service.tick()
        self.assertFalse(self.calls)
        self.assertGreater(self.service.configuration("puppy")["next_due"], self.clock.now)
        self.clock.now = self.service.configuration("puppy")["next_due"] + 31
        await self.service.tick()
        self.assertFalse(self.calls)
        self.assertEqual(self.service.history("puppy")[0]["state"], "missed")

    async def test_offline_failure_is_one_attempt_and_not_replayed(self):
        async def offline(device):
            self.calls.append(device)
            raise ConnectionError("offline")
        self.service.capture_handler = offline
        self.enable()
        await self.run_due()
        self.assertEqual(self.calls, ["puppy"])
        self.assertEqual(self.service.history("puppy")[0]["state"], "failed")
        await self.service.tick()
        self.assertEqual(self.calls, ["puppy"])

    async def test_overlapping_ticks_and_due_slots_cannot_overlap_capture(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(device):
            self.calls.append(device)
            started.set()
            await release.wait()
            return fixture_jpeg()

        self.service.capture_handler = slow
        self.enable()
        self.clock.now = self.service.configuration("puppy")["next_due"]
        await asyncio.gather(self.service.tick(), self.service.tick())
        await started.wait()
        self.clock.now += 900
        await self.service.tick()
        self.assertEqual(self.calls, ["puppy"])
        self.assertEqual(self.service.history("puppy")[0]["state"], "skipped_busy")
        release.set()
        await self.settle()

    async def test_disable_cancels_capture_and_drops_late_callback_result(self):
        started = asyncio.Event()

        async def late(device):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return fixture_jpeg()

        self.service.capture_handler = late
        self.enable()
        self.clock.now = self.service.configuration("puppy")["next_due"]
        await self.service.tick()
        await started.wait()
        self.service.configure("puppy", PatrolConfig(enabled=False))
        await self.settle()
        self.assertEqual(self.service.history("puppy")[0]["state"], "cancelled")
        self.assertFalse(list((self.root / "patrol/photos").iterdir()))

    async def test_disable_before_task_starts_closes_its_claim(self):
        self.enable()
        self.clock.now = self.service.configuration("puppy")["next_due"]
        await self.service.tick()
        self.service.configure("puppy", PatrolConfig(enabled=False))
        await self.settle()
        self.assertFalse(self.calls)
        self.assertEqual(self.service.history("puppy")[0]["state"], "cancelled")

    async def test_description_failure_keeps_owned_photo(self):
        async def describe(path):
            self.assertTrue(path.is_file())
            raise RuntimeError("fake local model unavailable")
        self.service.describe_handler = describe
        self.enable(describe=True)
        await self.run_due()
        run = self.service.history("puppy")[0]
        self.assertEqual(run["state"], "completed")
        self.assertEqual(run["error"], "description_RuntimeError")
        self.assertTrue(self.service.read_photo(run["photo_id"]))

    async def test_retention_only_removes_registered_files(self):
        self.enable(retention_days=7)
        await self.run_due()
        run = self.service.history("puppy")[0]
        unrelated = self.root / "patrol/photos/manual-photo.jpg"
        unrelated.write_bytes(b"leave me")
        self.clock.now += 7 * 86400 - 1
        self.assertEqual(self.service.cleanup()["deleted"], 0)
        self.clock.now += 1
        result = self.service.cleanup()
        self.assertEqual(result["deleted"], 1)
        self.assertFalse(result["icloud_deleted"])
        self.assertTrue(unrelated.exists())
        with self.assertRaises(FileNotFoundError):
            self.service.read_photo(run["photo_id"])
        self.assertEqual(self.service.cleanup()["deleted"], 0)

    async def test_symlink_replacement_never_follows_or_deletes_target(self):
        self.enable(retention_days=1)
        await self.run_due()
        run = self.service.history("puppy")[0]
        photo = self.root / "patrol/photos" / (run["photo_id"] + ".jpg")
        external = self.root / "external.jpg"
        external.write_bytes(b"protected outside photo directory")
        photo.unlink()
        photo.symlink_to(external)
        self.clock.now += 86400
        self.assertEqual(self.service.cleanup()["blocked"], 1)
        self.assertTrue(photo.is_symlink())
        self.assertEqual(external.read_bytes(), b"protected outside photo directory")
        with self.assertRaises(ValueError):
            self.service.read_photo(run["photo_id"])

    async def test_corrupted_registered_path_cannot_escape_directory(self):
        self.enable(retention_days=1)
        await self.run_due()
        protected = self.root / "patrol/protected.jpg"
        protected.write_bytes(b"protected")
        with self.service.connect() as db:
            db.execute("UPDATE photos SET filename='../protected.jpg'")
        self.clock.now += 86400
        self.assertEqual(self.service.cleanup()["blocked"], 1)
        self.assertEqual(protected.read_bytes(), b"protected")

    async def test_storage_singleton_and_symlink_roots_are_rejected(self):
        with self.assertRaises(BlockingIOError):
            self.create_service()
        link = self.root / "alias"
        link.symlink_to(self.root / "patrol", target_is_directory=True)
        with self.assertRaises(OSError):
            PatrolService(link / "escape", capture_handler=self.capture)
        self.assertFalse((self.root / "patrol/escape").exists())

    async def test_cloud_or_unrelated_project_storage_is_rejected_before_creation(self):
        for name in ("LLMQuota", "Mobile Documents", "CloudStorage", "iCloud Drive"):
            with self.assertRaises(ValueError):
                PatrolService(self.root / name / "patrol", capture_handler=self.capture)
            self.assertFalse((self.root / name).exists())

    async def test_process_interruption_record_is_not_reexecuted(self):
        self.enable()
        with self.service.connect() as db:
            db.execute("""INSERT INTO runs (run_id,device_id,revision,scheduled_at,state)
                       VALUES ('interrupted-test','puppy','old',?,'running')""", (self.clock.now,))
        await self.service.close()
        self.service = self.create_service()
        self.assertEqual(self.service.history("puppy")[0]["state"], "interrupted")
        await self.service.tick()
        self.assertFalse(self.calls)


class PatrolConfigurationTests(unittest.TestCase):
    def test_bounds_and_windows(self):
        for fields in ({"interval_minutes": 14}, {"interval_minutes": 1441}, {"interval_minutes": True},
                       {"retention_days": 0}, {"retention_days": 31}, {"window_start": "24:00"},
                       {"timezone": "Not/AZone"}, {"enabled": 1}):
            with self.assertRaises(ValueError):
                PatrolConfig(**fields)
        config = PatrolConfig(window_start="22:00", window_end="06:00", interval_minutes=15)
        self.assertTrue(config.contains(timestamp("2026-09-24T23:00:00+08:00")))
        self.assertTrue(config.contains(timestamp("2026-09-25T02:00:00+08:00")))
        self.assertFalse(config.contains(timestamp("2026-09-25T06:00:00+08:00")))
        self.assertEqual(config.next_due(timestamp("2026-09-24T12:00:00+08:00")),
                         timestamp("2026-09-24T22:00:00+08:00"))
        self.assertTrue(PatrolConfig(window_start="00:00", window_end="00:00").contains(0))


class PatrolApiTests(unittest.TestCase):
    def test_parent_auth_and_default_disabled(self):
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.testclient import TestClient
        from .patrol_api import create_patrol_router

        async def capture(device):
            raise AssertionError("API configuration must not take a photo")

        def require_parent(authorization: str = Header(default="")):
            if authorization != "Bearer test-only":
                raise HTTPException(401, "parent required")

        with tempfile.TemporaryDirectory(prefix="puppy-patrol-api-") as directory:
            service = PatrolService(Path(directory).resolve() / "patrol", capture_handler=capture)
            app = FastAPI()
            app.include_router(create_patrol_router(service, require_parent))
            try:
                with TestClient(app) as client:
                    headers = {"Authorization": "Bearer test-only"}
                    path = "/api/patrol/devices/puppy"
                    self.assertEqual(client.get(path).status_code, 401)
                    self.assertEqual(client.post("/api/patrol/cleanup").status_code, 401)
                    self.assertFalse(client.get(path, headers=headers).json()["enabled"])
                    self.assertEqual(client.put(path, headers=headers, json={"enabled": True, "interval_minutes": 1}).status_code, 422)
                    self.assertEqual(client.put(path, headers=headers, json={"enabled": True, "icloud_delete": True}).status_code, 422)
                    result = client.put(path, headers=headers, json={"enabled": True, "interval_minutes": 15})
                    self.assertEqual(result.status_code, 200)
                    self.assertGreater(result.json()["next_due"], service.clock())
                    self.assertEqual(client.get(path + "/runs", headers=headers).json(), {"runs": []})
            finally:
                asyncio.run(service.close())


if __name__ == "__main__":
    unittest.main()
