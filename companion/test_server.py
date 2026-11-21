"""Gateway failure tests; no hardware, speech or model is invoked."""

import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

# Import-time storage must also stay outside the family's database.
_IMPORT_DIR = tempfile.TemporaryDirectory(prefix="puppy-server-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT_DIR.name}):
    from . import server
from .store import Store


class FakeSocket:
    def __init__(self, handler):
        self.handler = handler
        self.sent = []

    async def send_json(self, packet):
        self.sent.append(packet)
        await self.handler(self, packet)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="puppy-gateway-test-")
        self.root = Path(self.directory.name)
        self.store = Store(self.root / "test.sqlite3")
        self.patches = [patch.object(server, "STORE", self.store), patch.object(server, "DATA_DIR", self.root)]
        for item in self.patches:
            item.start()
        self.hub = server.DeviceHub(command_timeout=.05, photo_timeout=.05, action_timeout=.05, retry_delay=0)

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def connect(self, device_id, handler):
        socket = FakeSocket(handler)
        self.hub.connections[device_id] = socket
        self.hub.send_locks[device_id] = asyncio.Lock()
        self.hub.status[device_id] = {"motion_protocol_version": 2}
        return socket

    def save_image(self, device_id, request_id):
        output = io.BytesIO()
        Image.new("RGB", (16, 16), color="green").save(output, format="JPEG")
        server.save_photo(output.getvalue(), device_id, request_id)

    async def test_receipt_is_bound_to_device_and_connection(self):
        async def handler(socket, packet):
            receipt = {"id": packet["id"], "ok": True}
            self.hub.receipt("other-device", socket, receipt)
            self.hub.receipt("puppy", object(), receipt)
            self.assertFalse(self.hub.pending[packet["id"]].future.done())
            self.hub.receipt("puppy", socket, receipt)
        self.connect("puppy", handler)
        result = await self.hub.command("puppy", "stop")
        self.assertTrue(result["ok"])
        self.assertEqual(self.hub.pending, {})

    async def test_confirmed_motion_recovers_unresponsive_control_before_any_action_is_sent(self):
        async def no_reply(socket, packet):
            pass

        async def respond(socket, packet):
            self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "phase": "completed"})

        original = self.connect("puppy", no_reply)
        replacement = None

        async def close(code):
            nonlocal replacement
            self.assertEqual(code, 1012)
            replacement = self.connect("puppy", respond)

        original.close = close
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {}), \
             patch.object(server, "ACTION_JOBS", {}), patch.object(server, "STOPPING", {}):
            receipt = await server.execute_safe_action("puppy", "stretch", confirmed_by_voice=True)
        self.assertTrue(receipt["ok"])
        self.assertEqual([p["type"] for p in original.sent], ["status"])
        self.assertEqual([p["type"] for p in replacement.sent], ["status", "action", "stop"])
        self.assertEqual(replacement.sent[1]["name"], "stretch")

    async def test_photo_retries_failed_transfer_and_returns_decodable_file(self):
        async def handler(socket, packet):
            if len(socket.sent) == 1:
                # A claimed success without a saved image must not be accepted.
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "photo_id": packet["id"]})
            else:
                self.save_image("puppy", packet["id"])
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "photo_id": packet["id"]})
        socket = self.connect("puppy", handler)
        result = await self.hub.command("puppy", "photo")
        self.assertEqual(len(socket.sent), 2)
        self.assertNotEqual(socket.sent[0]["id"], socket.sent[1]["id"])
        with Image.open(self.store.photo_file(result["photo_id"])) as image:
            image.load()
            self.assertEqual(image.size, (16, 16))
        self.assertEqual(self.store.photos("other-device"), [])

    async def test_identity_photo_never_enters_gallery_or_filesystem(self):
        output = io.BytesIO()
        Image.new("RGB", (16, 16), color="green").save(output, format="JPEG")
        data = output.getvalue()
        async def handler(socket, packet):
            self.hub.receive_photo("puppy", socket, packet["id"], data)
            self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "photo_id": packet["id"]})
        self.connect("puppy", handler)
        result = await self.hub.command("puppy", "photo", retain_photo=False)
        self.assertEqual(result["_photo_bytes"], data)
        self.assertGreater(result["captured_at"], 0)
        self.assertEqual(self.store.photos("puppy"), [])
        self.assertFalse((self.root / "photos").exists())
        self.assertEqual(self.hub.pending, {})

    async def test_legacy_firmware_cannot_receive_motion(self):
        async def handler(socket, packet):
            self.fail("Legacy firmware must be rejected before sending an action")
        socket = self.connect("puppy", handler)
        self.hub.status["puppy"] = {}
        with self.assertRaises(server.HTTPException) as caught:
            await self.hub.command("puppy", "action", name="wave")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(socket.sent, [])

    async def test_lost_photo_receipt_uses_saved_image_without_recapture(self):
        async def handler(socket, packet):
            self.save_image("puppy", packet["id"])
        socket = self.connect("puppy", handler)
        result = await self.hub.command("puppy", "photo")
        self.assertTrue(result["ok"])
        self.assertEqual(len(socket.sent), 1)
        self.assertEqual(self.hub.pending, {})

    async def test_movement_timeout_never_retries(self):
        async def handler(socket, packet):
            pass
        socket = self.connect("puppy", handler)
        with self.assertRaises(server.HTTPException) as caught:
            await self.hub.command("puppy", "action", name="wave")
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(len(socket.sent), 1)
        self.assertEqual(self.hub.pending, {})

    async def test_action_acceptance_waits_for_device_completion(self):
        accepted = asyncio.Event()
        async def handler(socket, packet):
            self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "phase": "accepted"})
            accepted.set()
        socket = self.connect("puppy", handler)
        self.hub.action_timeout = 1
        task = asyncio.create_task(self.hub.command("puppy", "action", name="wave"))
        await accepted.wait()
        self.assertFalse(task.done())
        self.hub.receipt("puppy", socket, {"id": socket.sent[0]["id"], "ok": True, "phase": "completed"})
        self.assertEqual((await task)["phase"], "completed")

    async def test_custom_action_waits_for_each_terminal_receipt(self):
        pending = asyncio.Queue()
        async def handler(socket, packet):
            if packet["type"] == "action":
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "phase": "accepted"})
                await pending.put(packet)
            else:
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True})
        socket = self.connect("puppy", handler)
        self.hub.action_timeout = 1
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {}), patch.object(server, "ACTION_JOBS", {}):
            job = server.start_action_job("puppy", "dance", [{"action": "wave"}, {"action": "sit"}])
            for name in ("wave", "sit"):
                packet = await asyncio.wait_for(pending.get(), 1)
                self.assertEqual(packet["name"], name)
                self.assertEqual(server.ACTION_JOBS["puppy"]["state"], "running")
                self.assertFalse(job.done())
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "phase": "completed"})
            self.assertTrue((await job)["ok"])
            self.assertEqual(server.ACTION_JOBS["puppy"]["state"], "completed")
            self.assertEqual([p["type"] for p in socket.sent], ["action", "action", "stop"])

    async def test_physical_stop_receipt_marks_job_cancelled(self):
        async def handler(socket, packet):
            if packet["type"] == "action":
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "phase": "accepted"})
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": False,
                                                   "phase": "cancelled", "error": "cancelled by stop"})
            else:
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True})
        socket = self.connect("puppy", handler)
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {}), patch.object(server, "ACTION_JOBS", {}):
            result = await server.start_action_job("puppy", "dance", [{"action": "wave"}, {"action": "sit"}])
            self.assertEqual(result["phase"], "cancelled")
            self.assertEqual(server.ACTION_JOBS["puppy"]["state"], "cancelled")
            self.assertEqual([p["type"] for p in socket.sent], ["action", "stop"])

    async def test_disconnect_releases_command_without_waiting_for_timeout(self):
        async def handler(socket, packet):
            self.hub.disconnected("puppy", socket)
        self.connect("puppy", handler)
        self.hub.command_timeout = 60
        with self.assertRaises(server.HTTPException):
            await asyncio.wait_for(self.hub.command("puppy", "stop"), timeout=.2)
        self.assertEqual(self.hub.pending, {})

    async def test_send_backpressure_and_send_lock_are_inside_stop_deadline(self):
        async def blocked(socket, packet):
            await asyncio.Event().wait()
        self.connect("puppy", blocked)
        for lock_held in (False, True):
            if lock_held:
                await self.hub.send_locks["puppy"].acquire()
            try:
                with self.assertRaises(server.HTTPException) as caught:
                    await asyncio.wait_for(self.hub.command("puppy", "stop"), timeout=.3)
                self.assertEqual(caught.exception.status_code, 504)
                self.assertEqual(self.hub.pending, {})
            finally:
                if lock_held:
                    self.hub.send_locks["puppy"].release()

    async def test_revoke_photo_after_queueing_prevents_the_actual_send(self):
        async def handler(socket, packet):
            self.fail("Revoked permission must prevent sending the camera request")
        socket = self.connect("puppy", handler)
        for lock in (self.hub.photo_locks.setdefault("puppy", asyncio.Lock()), self.hub.send_locks["puppy"]):
            allowed = True
            await lock.acquire()
            task = asyncio.create_task(self.hub.command("puppy", "photo", before_send=lambda: allowed))
            await asyncio.sleep(0)
            allowed = False
            lock.release()
            with self.assertRaises(server.HTTPException) as caught:
                await task
            self.assertEqual(caught.exception.status_code, 409)
            self.assertEqual(self.hub.pending, {})
        self.assertEqual(socket.sent, [])

    async def test_total_photo_deadline_includes_waiting_for_previous_photo(self):
        async def handler(socket, packet):
            self.fail("Expired photo must not be sent later")
        socket = self.connect("puppy", handler)
        lock = self.hub.photo_locks.setdefault("puppy", asyncio.Lock())
        await lock.acquire()
        try:
            with self.assertRaises(server.HTTPException) as caught:
                await self.hub.command("puppy", "photo", deadline=asyncio.get_running_loop().time() + .02)
            self.assertEqual(caught.exception.status_code, 504)
        finally:
            lock.release()
        await asyncio.sleep(0)
        self.assertEqual(socket.sent, [])
        self.assertEqual(self.hub.pending, {})

    async def test_cancelled_action_caller_does_not_wait_for_stop_cleanup(self):
        started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def handler(socket, packet):
            if packet["type"] == "action":
                started.set()
            else:
                stopping.set()
                await release.wait()
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True})
        self.connect("puppy", handler)
        self.hub.command_timeout = 1
        self.hub.action_timeout = 1
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {}), patch.object(server, "ACTION_JOBS", {}):
            caller = asyncio.create_task(server.execute_safe_action("puppy", "wave"))
            await started.wait()
            caller.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(caller, .2)
            await stopping.wait()
            self.assertFalse(server.RUNNING["puppy"].done())
            with self.assertRaises(server.HTTPException):
                server.start_action_job("puppy", "sit", [{"action": "sit"}])
            release.set()
            await asyncio.gather(server.RUNNING["puppy"], return_exceptions=True)

    async def test_explicit_stop_retains_ownership_and_does_not_cancel_cleanup_twice(self):
        started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def handler(socket, packet):
            if packet["type"] == "action":
                started.set()
            else:
                stopping.set()
                await release.wait()
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True})
        self.connect("puppy", handler)
        self.hub.command_timeout = self.hub.action_timeout = 1
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {}), \
                patch.object(server, "ACTION_JOBS", {}), patch.object(server, "STOPPING", {}):
            job = server.start_action_job("puppy", "wave", [{"action": "wave"}])
            await started.wait()
            job.cancel()
            await stopping.wait()
            self.assertEqual(job.cancelling(), 1)
            caller = asyncio.create_task(server.stop("puppy"))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(job.cancelling(), 1)
            caller.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await caller
            with self.assertRaises(server.HTTPException):
                server.start_action_job("puppy", "sit", [{"action": "sit"}])
            self.assertIn("puppy", server.STOPPING)
            release.set()
            await asyncio.gather(job, *server.STOPPING.values(), return_exceptions=True)

    async def test_gesture_job_cannot_send_after_permission_revoked(self):
        async def handler(socket, packet):
            self.assertEqual(packet["type"], "stop")
            self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True})
        socket = self.connect("puppy", handler)
        lock = self.hub.send_locks["puppy"]
        allowed = True
        await lock.acquire()
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {}), patch.object(server, "ACTION_JOBS", {}):
            task = asyncio.create_task(server.execute_gesture_action("puppy", "wave", before_send=lambda: allowed,
                deadline=asyncio.get_running_loop().time() + 1))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            allowed = False
            lock.release()
            result = await task
            self.assertFalse(result["ok"])
            self.assertEqual([p["type"] for p in socket.sent], ["stop"])

    async def test_parent_stop_upgrades_expired_gesture_stop_behind_send_lock(self):
        async def handler(socket, packet):
            self.assertEqual(packet["type"], "stop")
            self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True})
        socket = self.connect("puppy", handler)
        lock = self.hub.send_locks["puppy"]
        allowed = True
        await lock.acquire()
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {}), \
                patch.object(server, "STOPPING", {}), patch.object(server, "STOP_REQUESTS", {}):
            gesture = asyncio.create_task(server.execute_gesture_action("puppy", "stop", before_send=lambda: allowed,
                deadline=asyncio.get_running_loop().time() + 1))
            for _ in range(3):
                await asyncio.sleep(0)
            allowed = False
            parent = asyncio.create_task(server.stop("puppy"))
            await asyncio.sleep(0)
            lock.release()
            results = await asyncio.gather(gesture, parent)
            self.assertTrue(all(result["ok"] for result in results))
            self.assertEqual(len(socket.sent), 1)

    async def test_expired_gesture_stop_does_not_cancel_another_owner(self):
        async def handler(socket, packet):
            self.fail("Expired stop must not reach the device")
        self.connect("puppy", handler)
        owned = asyncio.create_task(asyncio.Event().wait())
        checks = 0
        def permission():
            nonlocal checks
            checks += 1
            return checks == 1  # Revoked before the background task starts.
        with patch.object(server, "HUB", self.hub), patch.object(server, "RUNNING", {"puppy": owned}), \
                patch.object(server, "STOPPING", {}), patch.object(server, "STOP_REQUESTS", {}):
            try:
                with self.assertRaises(server.HTTPException):
                    await server.execute_gesture_action("puppy", "stop", before_send=permission,
                        deadline=asyncio.get_running_loop().time() + 1)
                self.assertEqual(owned.cancelling(), 0)
            finally:
                owned.cancel()
                await asyncio.gather(owned, return_exceptions=True)

    async def test_photo_serialization_does_not_block_stop(self):
        first_started = asyncio.Event()
        release = asyncio.Event()
        async def handler(socket, packet):
            if packet["type"] == "photo":
                first_started.set()
                async def finish():
                    await release.wait()
                    self.save_image("puppy", packet["id"])
                    self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True, "photo_id": packet["id"]})
                asyncio.create_task(finish())
            else:
                self.hub.receipt("puppy", socket, {"id": packet["id"], "ok": True})
        socket = self.connect("puppy", handler)
        self.hub.photo_timeout = 1
        first = asyncio.create_task(self.hub.command("puppy", "photo"))
        await first_started.wait()
        second = asyncio.create_task(self.hub.command("puppy", "photo"))
        self.assertTrue((await self.hub.command("puppy", "stop"))["ok"])
        self.assertEqual([packet["type"] for packet in socket.sent], ["photo", "stop"])
        release.set()
        results = await asyncio.gather(first, second)
        self.assertNotEqual(results[0]["photo_id"], results[1]["photo_id"])
        self.assertEqual(len(self.store.photos("puppy")), 2)


class ParentApiTests(unittest.TestCase):
    def test_parent_audio_stop_requires_auth_and_never_sends_motion(self):
        class FakeAudioSession:
            calls = 0

            async def cancel_and_close(self):
                self.calls += 1
                return True

        active = FakeAudioSession()
        with patch.object(server, "PARENT_TOKEN", "test-parent"), \
                patch.object(server, "AUDIO_SESSIONS", {"puppy": {active}}), \
                patch.dict(os.environ, {"PUPPY_REMOTE_CONFIG": ""}), TestClient(server.app) as client:
            path = "/api/devices/puppy/audio/stop"
            self.assertEqual(client.post(path).status_code, 401)
            self.assertEqual(active.calls, 0)
            self.assertEqual(client.post(path, headers={"Authorization": "Bearer test-parent"}).json(),
                             {"ok": True, "phase": "completed"})
            self.assertEqual(active.calls, 1)
            self.assertEqual(client.post("/api/devices/other/audio/stop",
                                         headers={"Authorization": "Bearer test-parent"}).json(),
                             {"ok": True, "phase": "idle"})

    def test_parent_auth_and_memory_isolation(self):
        with tempfile.TemporaryDirectory(prefix="puppy-api-") as directory:
            store = Store(Path(directory) / "test.sqlite3")
            store.remember("puppy", "child-a", "preference", "恐龙")
            store.remember("puppy", "child-b", "preference", "熊猫")
            with patch.object(server, "STORE", store), patch.object(server, "PARENT_TOKEN", "test-parent"), \
                    patch.dict(os.environ, {"PUPPY_REMOTE_CONFIG": ""}), TestClient(server.app) as client:
                headers = {"Authorization": "Bearer test-parent"}
                self.assertEqual(client.get("/api/memory/puppy/child-a").status_code, 401)
                response = client.get("/api/memory/puppy/child-a", headers=headers)
                self.assertEqual([row["value"] for row in response.json()["memories"]], ["恐龙"])
                self.assertEqual(client.delete("/api/memory/puppy/child-a/preference", headers=headers).status_code, 200)
                self.assertEqual(store.memories("puppy", "child-a"), {})
                self.assertEqual(store.memories("puppy", "child-b"), {"preference": "熊猫"})
                for invalid in (True, 61, -1, 0.5):
                    self.assertEqual(client.post("/api/devices/puppy/volume", json={"volume": invalid},
                                                 headers=headers).status_code, 422)


if __name__ == "__main__":
    unittest.main()
