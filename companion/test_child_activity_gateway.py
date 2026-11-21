"""Child activity routing and temporary lessons with isolated gateway data."""

import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from .test_server import server
from .inference import InferenceQueue
from .learning import LearningStore
from .store import Store


class ChildActivityGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-child-activity-")
        root = Path(self.temp.name)
        self.store = Store(root / "pet.sqlite3")
        self.learning = LearningStore(root / "learning.sqlite3")
        self.queue = InferenceQueue()

        def model(system, _user):
            if "只判断本轮原话是否需要读取或改变说话者的个人记录" in system:
                return '{"requires_personal_context":false}'
            word = re.search(r"practice 的本轮英语目标：([a-z]+)", system)
            if word:
                return json.dumps({"reply": f"我们一起玩 {word.group(1)}。", "intent": "chat",
                                   "english_kind": "practice"}, ensure_ascii=False)
            return '{"reply":"我们聊聊天吧。","intent":"chat"}'

        self.patches = [patch.object(server, "STORE", self.store),
                        patch.object(server, "LEARNING", self.learning),
                        patch.object(server, "INFERENCE", self.queue),
                        patch.object(server, "MODEL", model)]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        await self.queue.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def turn(self, text, person_id=None):
        mode = self.store.child_sessions.route("puppy", "voice", text,
                                               self.store.conversation_mode("puppy"))
        return await server.chat(server.ChatRequest(device_id="puppy", session_id="voice",
                                                    person_id=person_id, text=text, mode=mode))

    async def test_anonymous_child_completes_one_word_then_exits(self):
        for text, activity in [("教我英语", "introduce"), ("dog", "repeat"),
                               ("继续", "play"), ("找到啦", "respond")]:
            with self.subTest(text=text):
                result = await self.turn(text)
                self.assertEqual(result["lesson"]["activity"], activity)
        self.assertEqual(self.store.english_progress("puppy", None), [])
        self.assertEqual((await self.turn("不玩了"))["reply"], "我们聊聊天吧。")
        self.assertEqual(self.store.child_sessions.route("puppy", "voice", "继续", "chat"), "english")

    async def test_homophone_word_request_enters_anonymous_lesson(self):
        result = await self.turn("叫我一個英文單詞")
        self.assertEqual(result["lesson"]["activity"], "introduce")
        self.assertEqual(self.store.english_progress("puppy", None), [])

    async def test_disabled_saving_keeps_session_lesson_without_private_progress(self):
        self.learning.set_settings("puppy", "child", patch={"save_progress": False})
        self.assertEqual((await self.turn("学英语", "child"))["lesson"]["activity"], "introduce")
        self.assertEqual((await self.turn("继续", "child"))["lesson"]["activity"], "repeat")
        self.assertEqual(self.store.english_progress("puppy", "child"), [])
        self.assertIsNone(self.store.english_lesson("puppy", "child"))

    async def test_guest_replay_preserves_stage(self):
        await self.turn("学英语")
        replay = await self.turn("我没听清")
        self.assertFalse(replay["lesson"]["advance"])
        self.assertEqual(self.store.child_sessions.lesson("puppy", "voice", None)["stage"], "repeat")


if __name__ == "__main__":
    unittest.main()
