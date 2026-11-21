"""Integration with real temporary stores and fake model/device; never audio."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

_IMPORT = tempfile.TemporaryDirectory(prefix="puppy-workflows-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT.name}):
    from . import server
from .action_drafts import ActionDraftService, ApprovedActionConflict, DraftError
from .inference import InferenceQueue
from .pet_profile import PetProfile
from .store import Store
from .personal_access import CLASSIFIER


class PetWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-workflows-")
        self.store = Store(Path(self.temp.name) / "store.sqlite3")
        self.queue = InferenceQueue()
        self.hub = server.DeviceHub(command_timeout=.2, action_timeout=.2, retry_delay=0)
        self.sent = []
        self.answer = {"reply": "请爸爸妈妈一起看看我的新把戏。", "intent": "action_draft"}
        self.socket = SimpleNamespace(send_json=self.send)
        self.hub.connections["puppy"] = self.socket
        self.hub.send_locks["puppy"] = asyncio.Lock()
        self.hub.status["puppy"] = {"motion_protocol_version": 2}
        self.drafts = ActionDraftService(Path(self.temp.name) / "drafts.sqlite3",
            local_model_generate=self.generate_draft, approved_actions_snapshot=self.store.actions,
            save_approved=server.save_approved_draft, delete_approved=server.delete_approved_draft)
        self.patches = [patch.object(server, key, value) for key, value in {
            "STORE": self.store, "MODEL": self.model, "INFERENCE": self.queue,
            "HUB": self.hub, "RUNNING": {}, "ACTION_JOBS": {}, "STOPPING": {}, "STOP_REQUESTS": {},
            "ACTION_DRAFTS": self.drafts, "PARENT_TOKEN": "test-parent",
            "LEARNING": SimpleNamespace(end_session=lambda *args: None),
        }.items()]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        pending = list(server.RUNNING.values()) + list(server.STOPPING.values())
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self.queue.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def send(self, packet):
        self.sent.append(packet)
        self.hub.receipt("puppy", self.socket, {"id": packet["id"], "ok": True, "phase": "completed"})

    def model(self, system, user):
        if system == CLASSIFIER:
            return '{"requires_personal_context":false}'
        return json.dumps({"explicit": True} if system.startswith("你只核验") else self.answer, ensure_ascii=False)

    async def generate_draft(self, prompt):
        return '{"name":"星星舞","steps":[{"action":"wave"},{"wait_ms":10},{"action":"sit"}]}'

    def request(self, text):
        return server.ChatRequest(device_id="puppy", text=text, session_id="test-session")

    async def propose(self):
        return (await self.drafts.propose("puppy", "教你一个叫星星舞的新把戏"))["draft"]

    async def test_chat_creates_only_draft_and_keeps_model_reply(self):
        result = await server.chat(self.request("教你一个叫星星舞的新把戏，挥手再坐下"))
        self.assertEqual(result["reply"], self.answer["reply"])
        self.assertEqual(result["draft"]["state"], "draft")
        self.assertFalse(result["executed"])
        self.assertEqual(self.store.actions(), {})
        self.assertEqual(self.sent, [])

    async def test_real_store_approval_canonicalization_and_revoke(self):
        draft = await self.propose()
        approved = await self.drafts.approve(draft["draft_id"], draft["version"])
        self.assertEqual(approved["state"], "approved")
        self.assertEqual(self.store.actions()["星星舞"][0], {"action": "wave", "wait_ms": 0})
        self.assertEqual(self.store.actions()["星星舞"][1], {"action": None, "wait_ms": 10})
        revoked = await self.drafts.revoke(approved["draft_id"], approved["version"])
        self.assertEqual(revoked["state"], "revoked")
        self.assertEqual(self.store.actions(), {})
        self.assertEqual(self.sent, [])

    async def test_revoke_cannot_remove_same_content_recreated_or_updated(self):
        for operation in ("recreate", "same_update"):
            with self.subTest(operation=operation):
                draft = await self.propose()
                approved = await self.drafts.approve(draft["draft_id"], draft["version"])
                steps = self.store.actions()["星星舞"]
                if operation == "recreate":
                    server.delete_saved_action("星星舞", steps)
                self.store.save_action("星星舞", steps)
                refused = await self.drafts.revoke(approved["draft_id"], approved["version"])
                self.assertNotEqual(refused["state"], "revoked")
                self.assertEqual(self.store.actions()["星星舞"], steps)
                with self.assertRaises(ApprovedActionConflict):
                    server.delete_approved_draft("星星舞", steps, ownership_receipt=approved["ownership_receipt"])
                server.delete_saved_action("星星舞", steps)

    async def test_unknown_approval_cannot_remove_independent_same_content(self):
        draft = await self.propose()
        with self.drafts._db() as db:
            db.execute("UPDATE action_drafts SET state='approving' WHERE draft_id=?", (draft["draft_id"],))
            db.commit()
        self.store.save_action(draft["name"], server.canonical_action_steps(draft["steps"]))
        unknown = await self.drafts.get(draft["draft_id"])
        self.assertEqual(unknown["state"], "approval_unknown")
        with self.assertRaises(DraftError):
            await self.drafts.revoke(unknown["draft_id"], unknown["version"])
        self.assertIn(draft["name"], self.store.actions())

    async def test_two_drafts_with_same_name_save_only_one_owner(self):
        first, second = await self.propose(), await self.propose()
        results = await asyncio.gather(self.drafts.approve(first["draft_id"], 1),
                                       self.drafts.approve(second["draft_id"], 1), return_exceptions=True)
        self.assertEqual(sum(isinstance(item, dict) and item["state"] == "approved" for item in results), 1)
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM action_ownership").fetchone()[0], 1)
        self.assertEqual(self.sent, [])

    async def test_sequence_executes_in_order_without_entering_library(self):
        self.answer = {"reply": "我来挥手，然后坐下。", "intent": "action_sequence", "actions": ["wave", "sit"]}
        result = await server.chat(self.request("挥手再坐下"))
        self.assertTrue(result["device_receipt"]["ok"])
        self.assertEqual([(item["type"], item.get("name")) for item in self.sent],
                         [("action", "wave"), ("action", "sit"), ("stop", None)])
        self.assertEqual(self.store.actions(), {})

    async def test_greeting_change_while_waiting_for_socket_prevents_motion(self):
        profile = PetProfile(self.store)
        initial = profile.set("puppy", style="gentle", greeting_action="wave", expected_revision=0)
        self.answer = {"reply": "我们来打招呼。", "intent": "greet"}
        started = asyncio.Event()
        original = server.start_action_job
        def start(*args, **kwargs):
            result = original(*args, **kwargs)
            started.set()
            return result
        lock = self.hub.send_locks["puppy"]
        await lock.acquire()
        with patch.object(server, "start_action_job", side_effect=start):
            task = asyncio.create_task(server.chat(self.request("按约定打招呼")))
            try:
                await asyncio.wait_for(started.wait(), 1)
                profile.set("puppy", style="gentle", greeting_action="none", expected_revision=initial["revision"])
            finally:
                lock.release()
            result = await task
        self.assertFalse(result["device_receipt"]["ok"])
        self.assertFalse(any(item["type"] == "action" for item in self.sent))

    async def test_parent_delete_rejects_stale_snapshot_and_requires_auth(self):
        old = [{"action": "wave", "wait_ms": 0}]
        self.store.save_action("星星舞", [{"action": "sit", "wait_ms": 0}])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            path = "/api/actions/星星舞"
            self.assertEqual((await client.request("DELETE", path, json={"expected_steps": old})).status_code, 401)
            headers = {"Authorization": "Bearer test-parent"}
            self.assertEqual((await client.request("DELETE", path, json={"expected_steps": old}, headers=headers)).status_code, 409)
            current = self.store.actions()["星星舞"]
            self.assertEqual((await client.request("DELETE", path, json={"expected_steps": current}, headers=headers)).status_code, 200)
        self.assertEqual(self.store.actions(), {})


if __name__ == "__main__":
    unittest.main()
