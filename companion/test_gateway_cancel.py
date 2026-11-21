"""Gateway cancellation regression: fake text model, temporary databases, no audio."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from .identity import IdentityService, IdentityStore
from .identity_api import IdentityRegistration
from .inference import InferenceQueue
from .learning import LearningStore
from .store import Store
from .turn_commit import TurnCancelled


# Importing the gateway must never create or migrate the family's real stores.
_IMPORT_DIR = tempfile.TemporaryDirectory(prefix="puppy-cancel-gateway-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT_DIR.name,
                             "PUPPY_PARENT_TOKEN": "test-cancel-parent",
                             "PUPPY_DEVICE_TOKEN": "test-cancel-device"}):
    from . import server


class GatewayCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="puppy-gateway-cancel-")
        root = Path(self.directory.name)
        self.store = Store(root / "memory.sqlite3")
        self.learning = LearningStore(root / "learning.sqlite3")
        self.queue = InferenceQueue()
        self.registrations = IdentityRegistration(
            IdentityService(IdentityStore(root / "identity.sqlite3")), settings_guard=self.queue.lock)
        self.hub = server.DeviceHub()
        self.store.remember_preference("puppy", "child", "仅这个孩子的恐龙偏好")
        self.store.append_conversation("puppy", "child", "先前聊过", "已有历史", "chat", "session")
        self.patches = [patch.object(server, key, value) for key, value in {
            "STORE": self.store, "LEARNING": self.learning, "INFERENCE": self.queue,
            "MODEL_LOCK": self.queue.lock, "REGISTRATIONS": self.registrations, "HUB": self.hub,
        }.items()]
        for item in self.patches:
            item.start()
        self.loop = asyncio.get_running_loop()
        self.previous_handler = self.loop.get_exception_handler()
        self.background_errors = []
        self.loop.set_exception_handler(lambda loop, context: self.background_errors.append(context))

    async def asyncTearDown(self):
        try:
            await self.queue.close()
            await asyncio.sleep(0)
            # Python 3.14 may explicitly report a retrieved exception from an
            # abandoned shield. No other asynchronous error is expected.
            self.assertTrue(all(isinstance(item.get("exception"), TurnCancelled)
                                for item in self.background_errors), self.background_errors)
        finally:
            self.loop.set_exception_handler(self.previous_handler)
            for item in reversed(self.patches):
                item.stop()
            self.directory.cleanup()

    def snapshot(self):
        with self.store.connect() as db:
            return {table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                    for table in ("pets", "memories", "conversations", "english_progress", "stories")}

    @staticmethod
    def request(text="你好", mode="chat"):
        return server.ChatRequest(device_id="puppy", person_id="child", text=text,
                                  mode=mode, session_id="session")

    async def cancel_while_generating(self, request, response):
        entered, release = threading.Event(), threading.Event()
        before = self.snapshot()

        def model(system, user):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test model was not released")
            return json.dumps(response, ensure_ascii=False)

        with patch.object(server, "MODEL", model):
            task = asyncio.create_task(server.chat(request))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(self.queue.running)  # The model has not finished.
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                # Wait for the actual shielded worker before inspecting SQLite.
                await self.queue.close()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.hub.pending, {})

    async def test_cancelled_name_confirmation_does_not_change_name_or_history(self):
        self.store.propose_name("puppy", "豆豆", "child", "session")
        await self.cancel_while_generating(self.request("好"),
                                          {"reply": "以后我叫豆豆啦。", "intent": "chat"})
        self.assertEqual(self.store.pet("puppy"), {"name": "小狗", "pending_name": "豆豆"})

    async def test_cancelled_english_preference_does_not_write_personal_memory(self):
        await self.cancel_while_generating(self.request("我喜欢熊猫", "english"),
            {"reply": "记住你喜欢熊猫。", "intent": "remember_preference", "value": "熊猫"})
        self.assertFalse(self.learning.status("puppy", "child")["active"])

    async def test_cancelled_english_lesson_does_not_write_progress_or_history(self):
        await self.cancel_while_generating(self.request("dog", "english"),
            {"reply": "小狗的英文是 dog，一起说 dog。", "intent": "chat"})
        self.assertEqual(self.store.english_progress("puppy", "child"), [])
        self.assertFalse(self.learning.status("puppy", "child")["active"])

    async def test_completed_english_lesson_commits_progress_and_history(self):
        with patch.object(server, "MODEL", lambda s, u: '{"reply":"小狗的英文是 dog，一起说 dog。","intent":"chat"}'):
            answer = await server.chat(self.request("dog", "english"))
        self.assertEqual(answer["lesson"]["target_word"], "dog")
        progress = self.store.english_progress("puppy", "child")
        self.assertEqual(progress[0]["presentations"], 1)
        self.assertEqual(progress[0]["text_matches"], 1)
        self.assertEqual(len(self.store.conversation("puppy", "child", "english", "session")), 1)

    async def stale_identity_is_anonymous(self, *, reenable):
        enabled = await self.registrations.set_identity_settings("puppy", True)
        request = self.request("我喜欢熊猫")
        request._identity_revision = enabled["revision"]
        await self.registrations.set_identity_settings("puppy", False)
        if reenable:
            await self.registrations.set_identity_settings("puppy", True)
        before = self.snapshot()
        prompts = []

        def model(system, user):
            prompts.append(system)
            from .personal_access import CLASSIFIER
            if system == CLASSIFIER:
                return '{"requires_personal_context":true}'
            return '{"reply":"我记住了","intent":"remember_preference","value":"熊猫"}'

        with patch.object(server, "MODEL", model):
            answer = await server.chat(request)
        self.assertIsNone(request.person_id)
        self.assertEqual(answer["intent"], "chat")
        self.assertEqual(len(prompts), 1)
        from .personal_access import CLASSIFIER
        self.assertEqual(prompts[0], CLASSIFIER)
        self.assertEqual(answer["personal_access"]["reason"], "stale_identity")
        self.assertNotIn("仅这个孩子的恐龙偏好", prompts[0])
        self.assertNotIn("已有历史", prompts[0])
        self.assertEqual(self.snapshot(), before)

    async def test_disabled_identity_cannot_read_or_write_member_memory(self):
        await self.stale_identity_is_anonymous(reenable=False)

    async def test_reenabled_identity_cannot_reuse_an_old_revision(self):
        await self.stale_identity_is_anonymous(reenable=True)


if __name__ == "__main__":
    unittest.main()
