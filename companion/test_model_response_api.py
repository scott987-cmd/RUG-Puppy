"""Real chat route and memory transaction on invalid model output; no hardware/audio."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

_IMPORT = tempfile.TemporaryDirectory(prefix="puppy-model-api-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT.name}):
    from . import server
from .inference import InferenceQueue
from .action_drafts import ActionDraftService
from .agent import ModelResponseError
from .store import Store
from .personal_access import CLASSIFIER


class ModelResponseApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="puppy-model-api-")
        self.store = Store(Path(self.directory.name) / "store.sqlite3")
        self.queue = InferenceQueue()
        self.calls = 0
        self.valid = False
        self.patches = [patch.object(server, key, value) for key, value in {
            "STORE": self.store, "MODEL": self.model, "INFERENCE": self.queue,
            "HUB": server.DeviceHub(), "PARENT_TOKEN": "isolated-test-parent",
            "LEARNING": SimpleNamespace(end_session=lambda *args: None),
        }.items()]
        for item in self.patches:
            item.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
            base_url="http://isolated.test", headers={"Authorization": "Bearer isolated-test-parent"})

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.queue.close()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def model(self, system, user):
        if system == CLASSIFIER:
            return '{"requires_personal_context":false}'
        # Counts answer attempts; classifier failure/recovery has separate tests.
        self.calls += 1
        if not self.valid:
            return '{"reply":"string","intent":"chat","private_debug":"never-echo-this"}'
        return json.dumps({"reply": "你好，我们一起玩吧。", "intent": "chat"}, ensure_ascii=False)

    async def ask(self, text="你好"):
        return await self.client.post("/api/chat", json={
            "device_id": "test-puppy", "text": text, "session_id": "test-session"})

    async def test_invalid_model_returns_safe_retryable_response_and_queue_recovers(self):
        response = await self.ask()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "本地模型暂未生成合适的回答，请再试一次"})
        self.assertNotIn("never-echo-this", response.text)
        self.assertEqual(self.calls, 2)
        self.assertFalse(self.queue.lock.locked())
        self.assertEqual(server.HUB.pending, {})
        self.valid = True
        retry = await self.ask()
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["reply"], "你好，我们一起玩吧。")

    async def test_rejected_confirmation_keeps_original_name_and_pending_proposal(self):
        old_name = self.store.pet("test-puppy")["name"]
        self.store.propose_name("test-puppy", "星星", session_id="test-session")
        response = await self.ask("好")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.store.pet("test-puppy")["name"], old_name)
        self.assertEqual(self.store.pending_name("test-puppy", session_id="test-session"), "星星")
        self.assertEqual(server.HUB.pending, {})
        self.valid = True
        self.assertEqual((await self.ask("好")).status_code, 200)
        self.assertEqual(self.store.pet("test-puppy")["name"], "星星")
        self.assertIsNone(self.store.pending_name("test-puppy", session_id="test-session"))

    async def test_photo_generation_failure_is_retryable_without_exposing_model_text(self):
        self.store.add_photo("synthetic-photo", "test-puppy", str(Path(self.directory.name) / "unused.jpg"))

        def describe(path, question):
            if not self.valid:
                raise ModelResponseError("never-echo-this")
            return {"reply": "一个红色方块。", "uncertain": False}

        with patch.object(server, "MODEL", SimpleNamespace(describe=describe)):
            response = await self.client.post("/api/photo/synthetic-photo/describe", json={"question": "这是什么？"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": server.MODEL_RETRY_DETAIL})
            self.assertNotIn("never-echo-this", response.text)
            self.assertFalse(self.queue.lock.locked())
            self.assertEqual(server.HUB.pending, {})
            self.valid = True
            retry = await self.client.post("/api/photo/synthetic-photo/describe", json={"question": "这是什么？"})
            self.assertEqual(retry.status_code, 200)
            self.assertEqual(retry.json()["reply"], "一个红色方块。")

    async def test_truncated_draft_cannot_save_or_execute_and_retry_can_create_one_draft(self):
        service = ActionDraftService(Path(self.directory.name) / "drafts.sqlite3",
            local_model_generate=server.generate_action_draft, approved_actions_snapshot=self.store.actions,
            save_approved=server.save_approved_draft, delete_approved=server.delete_approved_draft)
        draft_calls = []

        def model(system, user):
            if system == CLASSIFIER:
                return '{"requires_personal_context":false}'
            if system.startswith("你只核验"):
                return '{"explicit":true}'
            if user == "请输出这次新把戏的草稿 JSON。":
                draft_calls.append(user)
                if not self.valid:
                    raise ModelResponseError("never-echo-this")
                return '{"name":"星星舞","steps":[{"action":"wave"}]}'
            return '{"reply":"我会请爸爸妈妈看看星星舞的建议。","intent":"action_draft"}'

        with patch.object(server, "MODEL", model), patch.object(server, "ACTION_DRAFTS", service):
            response = await self.ask("教你一个叫星星舞的新把戏，先挥手")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": server.MODEL_RETRY_DETAIL})
            self.assertNotIn("never-echo-this", response.text)
            self.assertEqual((await service.list("test-puppy"))["drafts"], [])
            self.assertFalse(self.queue.lock.locked())
            self.assertEqual(self.store.actions(), {})
            self.assertEqual(server.HUB.pending, {})
            self.valid = True
            retry = await self.ask("教你一个叫星星舞的新把戏，先挥手")
            self.assertEqual(retry.status_code, 200)
            self.assertFalse(retry.json()["executed"])
            self.assertEqual(retry.json()["draft"]["state"], "draft")
            self.assertEqual(len((await service.list("test-puppy"))["drafts"]), 1)
            self.assertEqual(len(draft_calls), 2)
            self.assertEqual(self.store.actions(), {})
            self.assertEqual(server.HUB.pending, {})


if __name__ == "__main__":
    unittest.main()
