"""Member revocation ordering: fake text model and temporary stores only."""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException

from .identity import Observation
from .identity_api import IdentityRegistration, create_identity_router
from .inference import InferenceQueue
from .learning import LearningStore
from .store import Store
from .personal_access import CLASSIFIER
from .test_identity_api import setup_service

_IMPORT = tempfile.TemporaryDirectory(prefix="puppy-revocation-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT.name}):
    from . import server


def _new_operation_guard(system, user):
    """Fixture-only consent; guard checks do not repeat main-model side effects."""
    if system.startswith("你只核验") and json.loads(user).get("operation") in {
            "action", "rename_pet", "remember_preference", "forget_memory"}:
        return '{"explicit":true}'
    return None


class MemberRevocationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-member-revocation-")
        root = Path(self.temp.name)
        self.store = Store(root / "memory.sqlite3")
        self.learning = LearningStore(root / "learning.sqlite3")
        self.queue = InferenceQueue()
        identity, _, _ = setup_service(root)
        self.reg = IdentityRegistration(identity, settings_guard=self.queue.lock)
        self.reg.save_member("child", "虚构成员")
        self.enabled = await self.reg.set_identity_settings("puppy", True)
        self.store.remember_preference("puppy", "child", "仅该成员的三角龙偏好")
        self.store.append_conversation("puppy", "child", "旧对话", "旧的私人回复", "chat", "session")
        self.prompts = []
        self.patches = [patch.object(server, key, value) for key, value in {
            "STORE": self.store, "LEARNING": self.learning, "INFERENCE": self.queue,
            "MODEL_LOCK": self.queue.lock, "REGISTRATIONS": self.reg,
            "HUB": server.DeviceHub(), "MODEL": self.model,
            "PARENT_TOKEN": "fictional-revocation-parent",
        }.items()]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        await self.queue.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def model(self, system, user):
        if reply := _new_operation_guard(system, user):
            return reply
        self.prompts.append(system)
        if system == CLASSIFIER:
            return '{"requires_personal_context":true}'
        return json.dumps({"reply": "好的。", "intent": "remember_preference",
                           "value": "熊猫", "action": ""}, ensure_ascii=False)

    def request(self, *, mode="chat", revision=None):
        request = server.ChatRequest(device_id="puppy", person_id="child", text="请记住我喜欢熊猫。",
                                     mode=mode, session_id="session")
        request._identity_revision = self.enabled["revision"] if revision is None else revision
        return request

    def snapshot(self):
        with self.store.connect() as db:
            return {table: [tuple(row) for row in db.execute("SELECT * FROM " + table + " ORDER BY rowid")]
                    for table in ("memories", "conversations", "english_progress", "stories", "pets")}

    def assert_anonymous(self, request, before):
        self.assertIsNone(request.person_id)
        # A rejected device result must retain its source marker.
        self.assertIsNotNone(request._identity_revision)
        self.assertNotIn("仅该成员的三角龙偏好", self.prompts[-1])
        self.assertNotIn("旧的私人回复", self.prompts[-1])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(server.HUB.pending, {})

    async def test_revoked_member_old_request_cannot_read_or_write_in_any_mode(self):
        await self.reg.revoke_person("child")
        with self.assertRaises(HTTPException):
            self.reg.member("child")
        self.assertFalse(self.reg.identity_allowed("puppy", self.enabled["revision"]))
        for mode in ("chat", "english", "story"):
            with self.subTest(mode=mode):
                before = self.snapshot()
                request = self.request(mode=mode)
                with patch.object(self.learning, "before_turn", wraps=self.learning.before_turn) as before_turn:
                    answer = await server.chat(request)
                self.assertEqual(answer["intent"], "chat")
                self.assert_anonymous(request, before)
                if mode == "english":
                    before_turn.assert_not_called()

    async def test_missing_member_with_current_device_revision_is_still_anonymous(self):
        await self.reg.revoke_person("child")
        current_revision = self.reg.identity_settings("puppy")["revision"]
        self.assertTrue(self.reg.identity_allowed("puppy", current_revision))
        before = self.snapshot()
        request = self.request(revision=current_revision)
        await server.chat(request)
        self.assert_anonymous(request, before)

    async def test_prior_turn_finishes_before_revocation_then_old_request_is_anonymous(self):
        entered, release = threading.Event(), threading.Event()
        revoke_entered = asyncio.Event()
        at_revoke_return = []

        def blocked_model(system, user):
            if reply := _new_operation_guard(system, user):
                return reply
            entered.set()
            if not release.wait(5):
                raise RuntimeError("fake model was not released")
            return self.model(system, user)

        async def revoke():
            revoke_entered.set()
            result = await self.reg.revoke_person("child")
            at_revoke_return.extend(self.store.memory_entries("puppy", "child"))
            return result

        with patch.object(server, "MODEL", blocked_model):
            turn = asyncio.create_task(server.chat(self.request()))
            closing = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                closing = asyncio.create_task(revoke())
                await revoke_entered.wait()
                self.assertFalse(closing.done())
                self.assertTrue(self.queue.lock.locked())
                self.assertEqual(self.reg.identity_settings("puppy"), self.enabled)
                self.assertEqual(self.reg.member("child")["display_name"], "虚构成员")
                release.set()
                answer = await asyncio.wait_for(turn, 2)
                await asyncio.wait_for(closing, 2)
                self.assertEqual(answer["intent"], "remember_preference")
            finally:
                release.set()
                await asyncio.gather(turn, *(tuple([closing]) if closing else ()), return_exceptions=True)
        self.assertIn("请记住我喜欢熊猫。", [entry["value"] for entry in at_revoke_return])
        before = self.snapshot()
        old = self.request()
        await server.chat(old)
        self.assert_anonymous(old, before)

    async def test_delete_recreate_same_id_does_not_revive_old_device_identity(self):
        old_generation = self.reg.member("child")["generation"]
        await self.reg.revoke_person("child")
        self.reg.save_member("child", "重建的虚构成员")
        self.assertNotEqual(self.reg.member("child")["generation"], old_generation)
        old = self.request()
        before = self.snapshot()
        await server.chat(old)
        self.assert_anonymous(old, before)
        # A genuinely new recognition using the current revision still works.
        fresh = self.request(revision=self.reg.identity_settings("puppy")["revision"])
        answer = await server.chat(fresh)
        self.assertEqual(answer["intent"], "remember_preference")
        self.assertEqual(fresh.person_id, "child")
        self.assertIn("仅该成员的三角龙偏好", self.prompts[-1])

    async def cancel_while_waiting(self, guard):
        ticket = await self.reg.create_ticket("puppy", "child")
        member = self.reg.member("child")
        async with guard:
            task = asyncio.create_task(self.reg.revoke_person("child"))
            await asyncio.sleep(0)  # Let the task reach its lock await.
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.reg.member("child"), member)
        self.assertEqual(self.reg.identity_settings("puppy"), self.enabled)
        self.assertEqual(self.reg.ticket_status(ticket["ticket_id"])["state"], "waiting")
        self.assertFalse(self.queue.lock.locked())
        self.assertFalse(self.reg.lock.locked())

    async def test_cancel_while_waiting_for_inference_never_revokes_later(self):
        await self.cancel_while_waiting(self.queue.lock)

    async def test_cancel_while_waiting_for_registration_releases_inference_guard(self):
        await self.cancel_while_waiting(self.reg.lock)

    async def test_revocation_atomically_deletes_samples_and_invalidates_all_existing_devices(self):
        self.reg.service.store.enroll("child", Observation("face", "fake-features", [1, .1]),
                                     parent_confirmed=True)
        self.reg.save_member("other", "其他虚构成员")
        self.reg.service.store.enroll("other", Observation("face", "fake-features", [.1, 1]),
                                     parent_confirmed=True)
        await self.reg.set_identity_settings("other-puppy", True)
        await self.reg.set_identity_settings("disabled-puppy", False)
        devices = ("puppy", "other-puppy", "disabled-puppy")
        before = {device: self.reg.identity_settings(device) for device in devices}
        result = await self.reg.revoke_person("child")
        self.assertEqual(result, {"person_id": "child", "revoked_samples": 1})
        self.assertEqual(self.reg.service.store.list_samples("child"), [])
        self.assertEqual(len(self.reg.service.store.list_samples("other")), 1)
        for device in devices:
            current = self.reg.identity_settings(device)
            self.assertEqual(current["revision"], before[device]["revision"] + 1)
            self.assertEqual(current["enabled"], before[device]["enabled"])
        after = {device: self.reg.identity_settings(device) for device in devices}
        with self.assertRaises(HTTPException):
            await self.reg.revoke_person("child")
        self.assertEqual({device: self.reg.identity_settings(device) for device in devices}, after)

    async def test_parent_http_manual_selection_keeps_existing_authority_and_private_fields_are_ignored(self):
        await self.reg.revoke_person("child")
        await self.reg.set_identity_settings("puppy", False)
        body = {"device_id": "puppy", "person_id": "child", "text": "请记住我喜欢熊猫。",
                "session_id": "session", "_identity_revision": self.enabled["revision"]}
        self.assertIsNone(server.ChatRequest.model_validate(body)._identity_revision)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                     base_url="http://isolated.test") as client:
            unauthorized = await client.post("/api/chat", json=body)
            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(self.prompts, [])
            response = await client.post("/api/chat", json=body,
                headers={"Authorization": "Bearer fictional-revocation-parent"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["intent"], "remember_preference")
        self.assertIn("仅该成员的三角龙偏好", self.prompts[-1])
        self.assertIn("请记住我喜欢熊猫。", [entry["value"] for entry in self.store.memory_entries("puppy", "child")])

    async def test_revision_failure_rolls_back_member_and_sample_deletion(self):
        member = self.reg.member("child")
        self.reg.service.store.enroll("child", Observation("face", "fake-features", [1, .1]),
                                     parent_confirmed=True)
        samples = self.reg.service.store.list_samples("child")
        ticket = await self.reg.create_ticket("puppy", "child")
        with self.reg.service.store.connect() as db:
            db.execute("""CREATE TRIGGER fail_revision BEFORE UPDATE OF revision
                ON identity_device_settings BEGIN SELECT RAISE(ABORT, 'test rollback'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            await self.reg.revoke_person("child")
        self.assertEqual(self.reg.member("child"), member)
        self.assertEqual(self.reg.service.store.list_samples("child"), samples)
        self.assertEqual(self.reg.identity_settings("puppy"), self.enabled)
        self.assertEqual(self.reg.ticket_status(ticket["ticket_id"])["state"], "waiting")
        self.assertFalse(self.queue.lock.locked())
        self.assertFalse(self.reg.lock.locked())

    async def test_sample_delete_invalidates_old_recognition_but_keeps_member(self):
        sample = self.reg.service.store.enroll(
            "child", Observation("face", "fake-features", [1, .1]), parent_confirmed=True)
        self.reg.save_member("other", "其他虚构成员")
        app = FastAPI()
        app.include_router(create_identity_router(self.reg, server.require_parent))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://isolated.test",
                                     headers={"Authorization": "Bearer fictional-revocation-parent"}) as client:
            wrong_owner = await client.delete("/api/identity/members/other/samples/" + sample["sample_id"])
            self.assertEqual(wrong_owner.status_code, 404)
            self.assertEqual(self.reg.identity_settings("puppy"), self.enabled)
            entered = asyncio.Event()
            original_revoke = self.reg.revoke_sample

            async def observed_revoke(person_id, sample_id):
                entered.set()
                return await original_revoke(person_id, sample_id)

            with patch.object(self.reg, "revoke_sample", observed_revoke):
                async with self.queue.lock:
                    waiting = asyncio.create_task(client.delete(
                        "/api/identity/members/child/samples/" + sample["sample_id"]))
                    await asyncio.wait_for(entered.wait(), 2)
                    self.assertFalse(waiting.done())
                    waiting.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await waiting
                self.assertEqual(self.reg.identity_settings("puppy"), self.enabled)
                self.assertEqual(len(self.reg.service.store.list_samples("child")), 1)
            response = await client.delete("/api/identity/members/child/samples/" + sample["sample_id"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"revoked": True})
        self.assertEqual(self.reg.member("child")["display_name"], "虚构成员")
        self.assertEqual(self.reg.service.store.list_samples("child"), [])
        before = self.snapshot()
        request = self.request()
        await server.chat(request)
        self.assert_anonymous(request, before)
        self.assertFalse(self.reg.identity_allowed("puppy", self.enabled["revision"]))


if __name__ == "__main__":
    unittest.main()
