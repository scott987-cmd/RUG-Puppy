"""English permission/commit races with temporary databases and fake text only."""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

_IMPORT = tempfile.TemporaryDirectory(prefix="puppy-learning-commit-import-")
with patch.dict(os.environ, {"PUPPY_DATA_DIR": _IMPORT.name}):
    from . import server
from .inference import InferenceQueue
from .learning import LearningStore
from .store import Store
from .turn_commit import TurnCommit


def _new_operation_guard(system, user):
    """Fixture-only consent; guard checks do not repeat main-model side effects."""
    if system.startswith("你只核验") and json.loads(user).get("operation") in {
            "action", "rename_pet", "remember_preference", "forget_memory"}:
        return '{"explicit":true}'
    return None


class LearningCommitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="puppy-learning-commit-")
        root = Path(self.directory.name)
        self.store = Store(root / "store.sqlite3")
        self.now = 1790190000.0
        self.learning = LearningStore(root / "learning.sqlite3", clock=lambda: self.now)
        self.queue = InferenceQueue()
        self.model_entered, self.model_release = threading.Event(), threading.Event()
        self.commit_entered, self.commit_release = threading.Event(), threading.Event()
        self.block_model = False
        self.block_commit = False
        self.remember = False
        self.advance_in_model = 0.0
        self.calls = []
        self.tasks = []
        owner = self

        class ControlledTurnCommit(TurnCommit):
            def commit(self, operation):
                owner.commit_entered.set()
                if owner.block_commit and not owner.commit_release.wait(3):
                    raise AssertionError("test did not release fake commit boundary")
                return super().commit(operation)

        self.patches = [patch.object(server, name, value) for name, value in {
            "STORE": self.store, "LEARNING": self.learning, "MODEL": self.model,
            "INFERENCE": self.queue, "MODEL_LOCK": self.queue.lock,
            "TurnCommit": ControlledTurnCommit, "HUB": server.DeviceHub(),
        }.items()]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        self.model_release.set()
        self.commit_release.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.queue.close()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def model(self, system, user):
        if reply := _new_operation_guard(system, user):
            return reply
        self.calls.append(system)
        # Neither normal generation nor a time-limit farewell may own a SQLite
        # write transaction. A separate connection must be able to acquire it.
        with sqlite3.connect(self.store.database, timeout=0.05) as db:
            db.execute("BEGIN IMMEDIATE")
        if "本次英语练习已经结束" in system:
            permit_free = []

            def check_settings_gate():
                acquired = self.learning._settings_commit_lock.acquire(timeout=0.1)
                permit_free.append(acquired)
                if acquired:
                    self.learning._settings_commit_lock.release()

            worker = threading.Thread(target=check_settings_gate)
            worker.start()
            worker.join(1)
            self.assertEqual(permit_free, [True], "farewell must release the settings gate")
            return '{"reply":"我们休息一下，明天再玩英语吧。"}'
        self.model_entered.set()
        if self.block_model and not self.model_release.wait(3):
            raise AssertionError("test did not release fake text model")
        self.now += self.advance_in_model
        if self.remember:
            return '{"reply":"我记住你喜欢火车","intent":"remember_preference","value":"火车"}'
        return '{"reply":"小狗的英文是 dog，我们一起说 dog。","intent":"chat"}'

    def request(self):
        text = "请记住我喜欢火车" if self.remember else "一起学英语"
        return server.ChatRequest(device_id="puppy", person_id="child", text=text,
                                  session_id="session", mode="english")

    def history(self):
        return self.store.conversation("puppy", "child", "english", "session")

    def progress(self):
        return self.store.english_progress("puppy", "child")

    def start_chat(self):
        task = asyncio.create_task(server.chat(self.request()))
        self.tasks.append(task)
        return task

    async def test_initial_false_then_enabled_during_generation_does_not_save(self):
        self.learning.set_settings("puppy", "child", patch={"save_progress": False})
        self.block_model = True
        task = self.start_chat()
        self.assertTrue(await asyncio.to_thread(self.model_entered.wait, 1))
        await asyncio.wait_for(asyncio.to_thread(self.learning.set_settings,
            "puppy", "child", patch={"save_progress": True}), 1)
        self.model_release.set()
        answer = await task
        self.assertTrue(answer["learning"]["settings"]["save_progress"])
        self.assertEqual((self.progress(), self.history()), ([], []))

    async def test_disable_after_after_turn_before_final_commit_prevents_save(self):
        original = self.learning.after_turn

        def finish_then_disable(*args, **kwargs):
            state = original(*args, **kwargs)
            self.learning.set_settings("puppy", "child", patch={"save_progress": False})
            return state

        with patch.object(self.learning, "after_turn", finish_then_disable):
            answer = await server.chat(self.request())
        self.assertTrue(answer["learning"]["settings"]["save_progress"])
        self.assertFalse(self.learning.get_settings("puppy", "child")["save_progress"])
        self.assertEqual((self.progress(), self.history()), ([], []))

    async def test_enabled_turn_saves_one_presentation_and_one_reply(self):
        answer = await server.chat(self.request())
        self.assertEqual(answer["lesson"]["target_word"], "dog")
        self.assertEqual(answer["lesson"]["activity"], "introduce")
        self.assertEqual(len(self.progress()), 1)
        self.assertEqual(self.progress()[0]["presentations"], 1)
        self.assertEqual(self.store.english_lesson("puppy", "child")["stage"], "repeat")
        self.assertEqual(len(self.history()), 1)

    async def test_disable_waits_until_accepted_sqlite_commit_is_finished(self):
        self.block_commit = True
        chat = self.start_chat()
        self.assertTrue(await asyncio.to_thread(self.commit_entered.wait, 1))
        writer_started = threading.Event()

        def disable_and_read():
            writer_started.set()
            self.learning.set_settings("puppy", patch={"save_progress": False})
            return self.progress()

        disable = asyncio.create_task(asyncio.to_thread(disable_and_read))
        self.tasks.append(disable)
        self.assertTrue(await asyncio.to_thread(writer_started.wait, 1))
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(disable), 0.1)
        finally:
            self.commit_release.set()
        await chat
        # A successful disable response cannot precede this accepted DB commit.
        seen = await asyncio.wait_for(disable, 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["presentations"], 1)
        self.assertFalse(self.learning.get_settings("puppy", "child")["save_progress"])

    async def test_cancel_before_final_commit_rolls_back_learning_and_releases_gate(self):
        self.block_commit = True
        task = self.start_chat()
        self.assertTrue(await asyncio.to_thread(self.commit_entered.wait, 1))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.commit_release.set()
        await self.queue.close()
        self.assertEqual((self.progress(), self.history()), ([], []))
        await asyncio.wait_for(asyncio.to_thread(self.learning.set_settings,
            "puppy", patch={"save_progress": False}), 1)

    async def test_disabled_progress_does_not_discard_explicit_business_memory(self):
        self.learning.set_settings("puppy", "child", patch={"save_progress": False})
        self.remember = True
        answer = await server.chat(self.request())
        self.assertEqual(answer["intent"], "remember_preference")
        self.assertEqual([item["value"] for item in self.store.memory_entries("puppy", "child")], ["请记住我喜欢火车"])
        self.assertEqual((self.progress(), self.history()), ([], []))

    async def test_cancel_rolls_back_business_memory_with_the_same_turn(self):
        self.remember = True
        self.block_commit = True
        task = self.start_chat()
        self.assertTrue(await asyncio.to_thread(self.commit_entered.wait, 1))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.commit_release.set()
        await self.queue.close()
        self.assertEqual(self.store.memory_entries("puppy", "child"), [])
        self.assertEqual((self.progress(), self.history()), ([], []))

    async def test_record_failure_rolls_back_and_releases_settings_gate(self):
        original = self.store.record_english

        def write_then_fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("synthetic history persistence failure")

        with patch.object(self.store, "record_english", write_then_fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                await server.chat(self.request())
        self.assertEqual((self.progress(), self.history()), ([], []))
        await asyncio.wait_for(asyncio.to_thread(self.learning.set_settings,
            "puppy", patch={"save_progress": False}), 1)

    async def test_limit_farewell_holds_neither_settings_gate_nor_memory_transaction(self):
        self.learning.set_settings("puppy", "child", patch={"daily_minutes": 1})
        await server.chat(self.request())
        old_progress, old_history = self.progress(), self.history()
        self.now += 40
        self.advance_in_model = 21
        self.remember = True
        answer = await server.chat(self.request())
        self.assertTrue(answer["learning_closed"])
        self.assertEqual((self.progress(), self.history()), (old_progress, old_history))
        self.assertEqual(self.store.memory_entries("puppy", "child"), [])


if __name__ == "__main__":
    unittest.main()
