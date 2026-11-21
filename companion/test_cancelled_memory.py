"""Cancellation at the persistence boundary; fake text model and temporary DB only."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from .agent import PetAgent
from .inference import InferenceQueue
from .store import Store
from .turn_commit import TurnCancelled, TurnCommit


def _new_operation_guard(system, user):
    """Fixture-only consent for newly added guards; main replies keep their sequence."""
    if system.startswith("你只核验") and json.loads(user).get("operation") in {
            "action", "rename_pet", "remember_preference", "forget_memory"}:
        return '{"explicit":true}'
    return None


class CancelledMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-cancelled-memory-")
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "memory.sqlite3")
        self.key = self.store.remember_preference("puppy", "child", "恐龙")
        self.store.append_conversation("puppy", "child", "我喜欢恐龙", "记住了", "chat", "session")
        self.store.propose_name("puppy", "豆豆", "child", "session")

    def snapshot(self):
        with self.store.connect() as db:
            return {name: [tuple(row) for row in db.execute(f"SELECT * FROM {name} ORDER BY rowid")]
                    for name in ("pets", "memories", "conversations", "english_progress", "stories")}

    @staticmethod
    def output(intent="chat", **fields):
        return json.dumps({"reply": "我们一起看看新朋友吧。", "intent": intent, **fields}, ensure_ascii=False)

    def test_cancel_during_generation_preserves_every_kind_of_memory(self):
        cases = (
            ("propose_name", "以后叫团团", "chat", self.output("rename_pet", value="团团")),
            ("confirm_name", "好", "chat", self.output()),
            ("reject_name", "不要", "chat", self.output()),
            ("preference", "我喜欢熊猫", "chat", self.output("remember_preference", value="熊猫")),
            ("forget", "忘掉恐龙", "chat", self.output("forget_memory", value="恐龙")),
            ("story", "讲故事", "story", self.output(story_title="新朋友")),
            ("english", "dog", "english", self.output(reply="小狗的英文是 dog，我们一起说 dog。")),
            ("history", "你好", "chat", self.output()),
        )
        for label, user, mode, answer in cases:
            with self.subTest(label):
                before = self.snapshot()
                entered, release = threading.Event(), threading.Event()
                guard = TurnCommit()

                def model(system, text):
                    if reply := _new_operation_guard(system, text):
                        return reply
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("test model was not released")
                    return answer

                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(PetAgent(self.store, model).reply, "puppy", user,
                                         "child", mode, "session", turn_commit=guard)
                    try:
                        self.assertTrue(entered.wait(2))
                        self.assertTrue(guard.cancel())
                    finally:
                        release.set()
                    with self.assertRaises(TurnCancelled):
                        future.result(timeout=3)
                self.assertEqual(self.snapshot(), before)

    def test_normal_guard_commits_preference_and_history(self):
        guard = TurnCommit()
        result = PetAgent(self.store, lambda s, u: _new_operation_guard(s, u) or self.output("remember_preference", value="熊猫")).reply(
            "puppy", "我喜欢熊猫", "child", session_id="session", turn_commit=guard)
        self.assertEqual(self.store.memories("puppy", "child")[result["memory_key"]], "我喜欢熊猫")
        self.assertEqual(self.store.conversation("puppy", "child", session_id="session")[-1]["assistant"], result["reply"])
        before = self.snapshot()
        self.assertFalse(guard.cancel())
        self.assertEqual(self.snapshot(), before)

    def test_normal_guard_confirms_name(self):
        PetAgent(self.store, lambda s, u: self.output()).reply(
            "puppy", "好", "child", session_id="session", turn_commit=TurnCommit())
        self.assertEqual(self.store.pet("puppy"), {"name": "豆豆", "pending_name": None})

    def test_normal_guard_commits_story_and_history(self):
        result = PetAgent(self.store, lambda s, u: self.output(story_title="新朋友")).reply(
            "puppy", "讲故事", "child", "story", "session", turn_commit=TurnCommit())
        self.assertEqual(self.store.story("puppy", "child")["chapter"], 1)
        self.assertEqual(result["story"]["chapter"], 1)
        self.assertEqual(len(self.store.conversation("puppy", "child", "story", "session")), 1)

    def test_error_after_first_write_rolls_back_the_whole_turn(self):
        cases = (("我喜欢熊猫", "chat", self.output("remember_preference", value="熊猫")),
                 ("好", "chat", self.output()),
                 ("讲故事", "story", self.output(story_title="新朋友")))
        for user, mode, answer in cases:
            with self.subTest(user):
                before = self.snapshot()
                with patch.object(self.store, "append_conversation", side_effect=RuntimeError("storage fault")):
                    with self.assertRaisesRegex(RuntimeError, "storage fault"):
                        PetAgent(self.store, lambda s, u: _new_operation_guard(s, u) or answer).reply(
                            "puppy", user, "child", mode, "session", turn_commit=TurnCommit())
                self.assertEqual(self.snapshot(), before)

    def test_outer_scope_cancellation_rolls_back_agent_and_gateway_writes(self):
        before = self.snapshot()
        prepared, release = threading.Event(), threading.Event()
        guard = TurnCommit()

        def infer():
            with self.store.commit_scope(guard):
                PetAgent(self.store, lambda s, u: _new_operation_guard(s, u) or self.output("remember_preference", value="熊猫")).reply(
                    "puppy", "我喜欢熊猫", "child", "english", "session", record_learning=False,
                    turn_commit=guard)
                self.store.record_english("puppy", "child", "dog", True)
                self.store.append_conversation("puppy", "child", "dog", "小狗 dog", "english", "session")
                prepared.set()
                if not release.wait(3):
                    raise RuntimeError("test commit was not released")

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(infer)
            try:
                self.assertTrue(prepared.wait(2))
                self.assertEqual(self.snapshot(), before)  # Other threads see no partial writes.
                self.assertTrue(guard.cancel())
            finally:
                release.set()
            with self.assertRaises(TurnCancelled):
                future.result(timeout=3)
        self.assertEqual(self.snapshot(), before)

    def test_outer_scope_success_commits_gateway_progress(self):
        guard = TurnCommit()
        with self.store.commit_scope(guard):
            answer = PetAgent(self.store, lambda s, u: self.output(reply="小狗的英文是 dog，一起说 dog。")).reply(
                "puppy", "dog", "child", "english", "session", record_learning=False, turn_commit=guard)
            self.store.record_english("puppy", "child", answer["lesson"]["target_word"], True)
            self.store.append_conversation("puppy", "child", "dog", answer["reply"], "english", "session")
        self.assertEqual(self.store.english_progress("puppy", "child")[0]["presentations"], 1)
        self.assertEqual(len(self.store.conversation("puppy", "child", "english", "session")), 1)

    def test_generation_holds_no_sqlite_transaction_in_outer_scope(self):
        entered, release = threading.Event(), threading.Event()
        guard = TurnCommit()

        def model(system, text):
            if reply := _new_operation_guard(system, text):
                return reply
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test model was not released")
            return self.output()

        def infer():
            with self.store.commit_scope(guard):
                return PetAgent(self.store, model).reply("puppy", "你好", "child", turn_commit=guard)

        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(infer)
            try:
                self.assertTrue(entered.wait(2))
                pool.submit(self.store.remember_preference, "puppy", "another-child", "积木").result(timeout=1)
                guard.cancel()
            finally:
                release.set()
            with self.assertRaises(TurnCancelled):
                future.result(timeout=3)
        self.assertIn("积木", self.store.memories("puppy", "another-child").values())

    def test_cancel_is_prompt_while_sqlite_waits_for_another_writer(self):
        before = self.snapshot()
        entered, release, writing = threading.Event(), threading.Event(), threading.Event()
        guard = TurnCommit()
        begin = self.store.begin_commit

        def model(system, text):
            if reply := _new_operation_guard(system, text):
                return reply
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test model was not released")
            return self.output("remember_preference", value="熊猫")

        def start_writing():
            writing.set()
            begin()

        with ThreadPoolExecutor(max_workers=1) as pool, patch.object(self.store, "begin_commit", start_writing):
            future = pool.submit(PetAgent(self.store, model).reply, "puppy", "我喜欢熊猫", "child", turn_commit=guard)
            blocker = sqlite3.connect(self.store.database)
            try:
                self.assertTrue(entered.wait(2))
                blocker.execute("BEGIN IMMEDIATE")
                release.set()
                self.assertTrue(writing.wait(2))
                started = time.monotonic()
                self.assertTrue(guard.cancel())
                self.assertLess(time.monotonic() - started, .1)
            finally:
                release.set()
                blocker.rollback()
                blocker.close()
            with self.assertRaises(TurnCancelled):
                future.result(timeout=3)
        self.assertEqual(self.snapshot(), before)


class TurnCommitTests(unittest.TestCase):
    def test_accepted_commit_is_not_undone_and_cancel_does_not_wait_for_it(self):
        guard = TurnCommit()
        entered, release = threading.Event(), threading.Event()

        def commit():
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test commit was not released")
            return "committed"

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(guard.commit, commit)
            try:
                self.assertTrue(entered.wait(2))
                started = time.monotonic()
                self.assertFalse(guard.cancel())
                self.assertLess(time.monotonic() - started, .1)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=3), "committed")
        self.assertFalse(guard.cancel())
        with self.assertRaises(RuntimeError):
            guard.commit(lambda: None)


class CancelledInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_shielded_inference_discards_its_later_memory_writes(self):
        with tempfile.TemporaryDirectory(prefix="puppy-cancelled-inference-") as directory:
            store = Store(Path(directory) / "memory.sqlite3")
            queue, guard = InferenceQueue(), TurnCommit()
            entered, release = threading.Event(), threading.Event()
            loop = asyncio.get_running_loop()
            previous_handler = loop.get_exception_handler()
            abandoned = []
            # Python 3.14 reports even retrieved exceptions from a cancelled
            # shield. Record and verify that only the expected cancellation occurs.
            loop.set_exception_handler(lambda loop, context: abandoned.append(context))

            def model(system, text):
                if reply := _new_operation_guard(system, text):
                    return reply
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test model was not released")
                return '{"reply":"记住啦","intent":"remember_preference","value":"熊猫"}'

            def infer():
                with store.commit_scope(guard):
                    return PetAgent(store, model).reply("puppy", "我喜欢熊猫", "child", turn_commit=guard)

            async def request():
                try:
                    return await queue.run(infer)
                except asyncio.CancelledError:
                    guard.cancel()
                    raise

            task = asyncio.create_task(request())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(queue.running)  # The synchronous model is still alive.
            finally:
                release.set()
                try:
                    await queue.close()
                    await asyncio.sleep(0)
                finally:
                    loop.set_exception_handler(previous_handler)
            self.assertTrue(all(isinstance(item.get("exception"), TurnCancelled) for item in abandoned))
            self.assertEqual(store.memories("puppy", "child"), {})
            self.assertEqual(store.conversation("puppy", "child"), [])


if __name__ == "__main__":
    unittest.main()
