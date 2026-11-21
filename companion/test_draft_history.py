"""Two-stage draft history with real temporary stores and a fake text model only."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

_IMPORT = tempfile.TemporaryDirectory(prefix="puppy-draft-history-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT.name}):
    from . import server
from .action_drafts import ActionDraftService
from .agent import ModelResponseError, PetAgent
from .identity import IdentityService, IdentityStore
from .identity_api import IdentityRegistration
from .inference import InferenceQueue
from .learning import LearningStore
from .store import Store


class DraftHistoryTests(unittest.IsolatedAsyncioTestCase):
    TEXT = "教你一个叫星星舞的新把戏，先挥手"
    REPLY = "我会请爸爸妈妈看看星星舞的建议。"

    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="puppy-draft-history-")
        path = Path(self.directory.name)
        self.store = Store(path / "store.sqlite3")
        self.queue = InferenceQueue()
        self.learning = LearningStore(path / "learning.sqlite3")
        self.registrations = IdentityRegistration(IdentityService(IdentityStore(path / "identity.sqlite3")),
                                                  settings_guard=self.queue.lock)
        self.registrations.save_member("child", "测试成员")
        self.identity_revision = (await self.registrations.set_identity_settings("puppy", True))["revision"]
        self.hub = server.DeviceHub()
        self.draft_failed = False
        self.block_draft = False
        self.block_first = False
        self.first_entered = threading.Event()
        self.first_release = threading.Event()
        self.draft_entered = threading.Event()
        self.draft_release = threading.Event()
        self.tasks = []
        self.held_test_lock = False
        self.drafts = ActionDraftService(path / "drafts.sqlite3",
            local_model_generate=server.generate_action_draft, approved_actions_snapshot=self.store.actions,
            save_approved=server.save_approved_draft, delete_approved=server.delete_approved_draft)
        self.patches = [patch.object(server, key, value) for key, value in {
            "STORE": self.store, "MODEL": self.model, "INFERENCE": self.queue,
            "MODEL_LOCK": self.queue.lock, "REGISTRATIONS": self.registrations,
            "LEARNING": self.learning, "ACTION_DRAFTS": self.drafts, "HUB": self.hub,
        }.items()]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        self.first_release.set()
        self.draft_release.set()
        if self.held_test_lock:
            self.queue.lock.release()
            self.held_test_lock = False
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.queue.close()
        self.assertEqual(self.hub.pending, {})
        self.assertEqual(self.store.actions(), {})
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def model(self, system, user):
        if system.startswith("你只核验"):
            return '{"explicit":true}'
        if user == "请输出这次新把戏的草稿 JSON。":
            self.draft_entered.set()
            if self.block_draft and not self.draft_release.wait(3):
                raise AssertionError("test did not release fake draft model")
            if self.draft_failed:
                raise ModelResponseError("synthetic truncated draft")
            return '{"name":"星星舞","steps":[{"action":"wave"}]}'
        self.first_entered.set()
        if self.block_first and not self.first_release.wait(3):
            raise AssertionError("test did not release fake first model")
        return json.dumps({"reply": self.REPLY, "intent": "action_draft"}, ensure_ascii=False)

    def request(self, mode="chat", *, automatic_identity=False):
        request = server.ChatRequest(device_id="puppy", person_id="child", text=self.TEXT,
                                     session_id="test-session", mode=mode)
        if automatic_identity:
            request._identity_revision = self.identity_revision
        return request

    def history(self, mode="chat"):
        return self.store.conversation("puppy", "child", mode, "test-session")

    async def test_confirmed_member_failure_leaves_no_history_or_draft(self):
        self.draft_failed = True
        with self.assertRaises(ModelResponseError):
            await server.chat(self.request())
        self.assertEqual(self.history(), [])
        self.assertEqual((await self.drafts.list("puppy"))["drafts"], [])
        self.assertFalse(self.queue.lock.locked())

    async def test_success_records_exactly_one_delivered_reply(self):
        for mode in ("chat", "english"):
            with self.subTest(mode=mode):
                answer = await server.chat(self.request(mode))
                self.assertEqual(answer["reply"], self.REPLY)
                self.assertFalse(answer["executed"])
                self.assertEqual(self.history(mode), [{"user": self.TEXT, "assistant": self.REPLY}])
        self.assertEqual(len((await self.drafts.list("puppy"))["drafts"]), 2)

    async def test_english_draft_failure_does_not_commit_first_stage_history(self):
        self.draft_failed = True
        with self.assertRaises(ModelResponseError):
            await server.chat(self.request("english"))
        self.assertEqual(self.history("english"), [])

    async def test_cancelled_second_generation_never_adds_history_after_worker_finishes(self):
        self.block_draft = True
        task = asyncio.create_task(server.chat(self.request()))
        self.tasks.append(task)
        self.assertTrue(await asyncio.to_thread(self.draft_entered.wait, 2))
        self.assertEqual(self.history(), [])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.queue.lock.locked())
        self.draft_release.set()
        await self.queue.close()
        self.assertEqual(self.history(), [])
        self.assertEqual((await self.drafts.list("puppy"))["drafts"], [])

    async def test_cancel_after_draft_save_keeps_unapproved_draft_without_history(self):
        original = self.drafts.propose
        saved = asyncio.Event()

        async def save_then_hold_commit_lock(*args, **kwargs):
            created = await original(*args, **kwargs)
            await self.queue.lock.acquire()
            self.held_test_lock = True
            saved.set()
            return created

        with patch.object(self.drafts, "propose", save_then_hold_commit_lock):
            task = asyncio.create_task(server.chat(self.request()))
            self.tasks.append(task)
            await asyncio.wait_for(saved.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.history(), [])
        drafts = (await self.drafts.list("puppy"))["drafts"]
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["state"], "draft")

    async def test_english_saving_disabled_prevents_history(self):
        self.learning.set_settings("puppy", "child", patch={"save_progress": False})
        await server.chat(self.request("english"))
        self.assertEqual(self.history("english"), [])
        self.assertEqual(self.store.english_progress("puppy", "child"), [])
        self.assertEqual(len((await self.drafts.list("puppy"))["drafts"]), 1)

    async def test_english_setting_revoked_during_draft_is_rechecked(self):
        original = self.drafts.propose

        async def save_then_disable(*args, **kwargs):
            created = await original(*args, **kwargs)
            self.learning.set_settings("puppy", "child", patch={"save_progress": False})
            return created

        with patch.object(self.drafts, "propose", save_then_disable):
            await server.chat(self.request("english"))
        self.assertEqual(self.history("english"), [])

    async def test_enabling_english_saving_later_does_not_retroactively_record(self):
        self.learning.set_settings("puppy", "child", patch={"save_progress": False})
        original = self.drafts.propose

        async def save_then_enable(*args, **kwargs):
            created = await original(*args, **kwargs)
            self.learning.set_settings("puppy", "child", patch={"save_progress": True})
            return created

        with patch.object(self.drafts, "propose", save_then_enable):
            await server.chat(self.request("english"))
        self.assertEqual(self.history("english"), [])

    async def test_enabling_saving_during_first_model_cannot_replace_initial_permission(self):
        self.learning.set_settings("puppy", "child", patch={"save_progress": False})
        self.block_first = True
        task = asyncio.create_task(server.chat(self.request("english")))
        self.tasks.append(task)
        self.assertTrue(await asyncio.to_thread(self.first_entered.wait, 2))
        self.assertTrue(self.queue.lock.locked())
        # The settings writer can finish while inference still owns MODEL_LOCK.
        await asyncio.wait_for(asyncio.to_thread(self.learning.set_settings,
            "puppy", "child", patch={"save_progress": True}), 1)
        self.first_release.set()
        answer = await task
        self.assertTrue(answer["learning"]["settings"]["save_progress"])
        self.assertEqual(self.history("english"), [])
        self.assertEqual(len((await self.drafts.list("puppy"))["drafts"]), 1)

    async def test_identity_disable_reenable_invalidates_deferred_history(self):
        original = self.drafts.propose

        async def save_then_revoke_revision(*args, **kwargs):
            created = await original(*args, **kwargs)
            await self.registrations.set_identity_settings("puppy", False)
            await self.registrations.set_identity_settings("puppy", True)
            return created

        with patch.object(self.drafts, "propose", save_then_revoke_revision):
            await server.chat(self.request(automatic_identity=True))
        self.assertEqual(self.history(), [])

    async def test_deleted_and_recreated_member_cannot_receive_old_reply(self):
        original = self.drafts.propose
        old_generation = self.registrations.member("child")["generation"]

        async def save_then_replace_member(*args, **kwargs):
            created = await original(*args, **kwargs)
            await self.registrations.revoke_person("child")
            self.registrations.save_member("child", "另一位测试成员")
            return created

        with patch.object(self.drafts, "propose", save_then_replace_member):
            await server.chat(self.request())
        self.assertNotEqual(self.registrations.member("child")["generation"], old_generation)
        self.assertEqual(self.history(), [])

    async def test_agent_alone_never_commits_draft_history_in_either_mode(self):
        agent = PetAgent(self.store, self.model)
        for mode in ("chat", "english"):
            with self.subTest(mode=mode):
                answer = agent.reply("puppy", self.TEXT, "child", mode, "test-session")
                self.assertEqual(answer["intent"], "action_draft")
                self.assertEqual(self.history(mode), [])


class LearningHistoryCommitTests(unittest.TestCase):
    def test_current_permission_and_history_append_are_one_short_commit(self):
        with tempfile.TemporaryDirectory(prefix="puppy-learning-commit-") as directory:
            learning = LearningStore(Path(directory) / "learning.sqlite3")
            store = Store(Path(directory) / "store.sqlite3")
            entered, release, writer_started = threading.Event(), threading.Event(), threading.Event()

            def commit():
                entered.set()
                if not release.wait(2):
                    raise AssertionError("test did not release history commit")
                store.append_conversation("puppy", "child", "fake", "fake reply", "english", "session")

            def disable():
                writer_started.set()
                return learning.set_settings("puppy", patch={"save_progress": False})

            with ThreadPoolExecutor(max_workers=2) as pool:
                accepted = pool.submit(learning.commit_if_saving, "puppy", "child", commit)
                try:
                    self.assertTrue(entered.wait(1))
                    disabled = pool.submit(disable)
                    self.assertTrue(writer_started.wait(1))
                    with self.assertRaises(FutureTimeout):
                        disabled.result(timeout=0.1)
                finally:
                    release.set()
                self.assertTrue(accepted.result(timeout=1))
                self.assertFalse(disabled.result(timeout=1)["settings"]["save_progress"])
            self.assertEqual(len(store.conversation("puppy", "child", "english", "session")), 1)
            later = []
            self.assertFalse(learning.commit_if_saving("puppy", "child", lambda: later.append("wrong")))
            self.assertEqual(later, [])

    def test_disable_completed_before_commit_prevents_callback(self):
        with tempfile.TemporaryDirectory(prefix="puppy-learning-commit-") as directory:
            learning = LearningStore(Path(directory) / "learning.sqlite3")
            learning.set_settings("puppy", "child", patch={"save_progress": False})
            called = []
            self.assertFalse(learning.commit_if_saving("puppy", "child", lambda: called.append("wrong")))
            self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
