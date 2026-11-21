"""Integration of parent learning settings with fake text models; no audio."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .test_server import server
from .store import Store
from .learning import LearningStore
from .inference import InferenceQueue


def _new_operation_guard(system, user):
    """Fixture-only consent; guard checks do not repeat main-model side effects."""
    if system.startswith("你只核验") and json.loads(user).get("operation") in {
            "action", "rename_pet", "remember_preference", "forget_memory"}:
        return '{"explicit":true}'
    return None


class LearningGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-learning-gateway-")
        self.root = Path(self.temp.name)
        self.now = 1790190000.0
        self.learning = LearningStore(self.root / "learning.sqlite3", clock=lambda: self.now)
        self.store = Store(self.root / "store.sqlite3")
        self.calls = []
        def model(system, user):
            if reply := _new_operation_guard(system, user):
                return reply
            self.calls.append(system)
            if "本次英语练习已经结束" in system:
                return '{"reply":"我们休息一下，明天再玩英语吧。"}'
            word = "red" if "本轮英语目标：red" in system else "dog"
            return json.dumps({"reply": "我们一起说 " + word, "intent": "chat"})
        self.queue = InferenceQueue()
        self.patches = [patch.object(server, "STORE", self.store), patch.object(server, "LEARNING", self.learning),
                        patch.object(server, "MODEL", model), patch.object(server, "INFERENCE", self.queue)]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        await self.queue.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def request(self, text="一起学习"):
        return server.ChatRequest(device_id="test-puppy", person_id="test-child", session_id="s", text=text, mode="english")

    async def test_theme_and_no_progress_setting_reach_agent(self):
        self.learning.set_settings("test-puppy", "test-child", patch={"theme": "colors", "save_progress": False})
        result = await server.chat(self.request())
        self.assertEqual(result["lesson"]["target_word"], "red")
        self.assertEqual(self.store.english_progress("test-puppy", "test-child"), [])
        self.assertEqual(self.store.conversation("test-puppy", "test-child", "english", "s"), [])

    async def test_panel_body_parts_theme_round_trips_and_invalid_theme_preserves_settings(self):
        import httpx
        from fastapi import FastAPI
        from .learning_api import create_learning_router

        app = FastAPI()
        app.include_router(create_learning_router(self.learning, server.require_parent))
        panel_payload = {"daily_minutes": 7, "theme": "body_parts",
                         "save_progress": False, "photo_learning": False}
        with patch.object(server, "PARENT_TOKEN", "fictional-learning-parent"):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://isolated.test") as client:
                for path in ("/api/learning/test-puppy", "/api/learning/test-puppy/members/test-child"):
                    with self.subTest(path=path):
                        headers = {"Authorization": "Bearer fictional-learning-parent"}
                        # Match the panel's PATCH method and complete settings body.
                        saved = await client.patch(path, json=panel_payload, headers=headers)
                        self.assertEqual(saved.status_code, 200, saved.text)
                        self.assertEqual(saved.json()["settings"], panel_payload)
                        loaded = await client.get(path, headers=headers)
                        self.assertEqual(loaded.status_code, 200, loaded.text)
                        self.assertEqual(loaded.json()["settings"], panel_payload)
                        invalid = await client.patch(path, json={**panel_payload,
                            "theme": "unknown_theme", "daily_minutes": 1, "save_progress": True}, headers=headers)
                        self.assertEqual(invalid.status_code, 422, invalid.text)
                        unchanged = await client.get(path, headers=headers)
                        self.assertEqual(unchanged.status_code, 200, unchanged.text)
                        self.assertEqual(unchanged.json()["settings"], panel_payload)
        self.assertEqual(self.calls, [])

    async def test_limit_generates_model_closing_without_advancing_words(self):
        self.learning.set_settings("test-puppy", "test-child", patch={"daily_minutes": 1})
        await server.chat(self.request())
        self.now += 40
        await server.chat(self.request())
        self.now += 25
        before = self.store.english_progress("test-puppy", "test-child")
        result = await server.chat(self.request())
        self.assertTrue(result["learning_closed"])
        self.assertIn("本次英语练习已经结束", self.calls[-1])
        self.assertEqual(self.store.english_progress("test-puppy", "test-child"), before)

    async def test_parent_end_during_model_turn_discards_reply_and_progress(self):
        def ending_model(system, user):
            if reply := _new_operation_guard(system, user):
                return reply
            self.learning.end_session("test-puppy", "test-child")
            return '{"reply":"我们一起说 dog","intent":"chat"}'
        with patch.object(server, "MODEL", ending_model):
            with self.assertRaises(server.HTTPException) as caught:
                await server.chat(self.request())
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self.store.english_progress("test-puppy", "test-child"), [])
        self.assertEqual(self.store.conversation("test-puppy", "test-child", "english", "s"), [])

    async def test_limit_during_generation_rolls_back_before_farewell_model(self):
        self.learning.set_settings("test-puppy", "test-child", patch={"daily_minutes": 1})
        await server.chat(self.request())
        self.now += 40
        before = self.store.english_progress("test-puppy", "test-child")
        history = self.store.conversation("test-puppy", "test-child", "english", "s")
        def model(system, user):
            if reply := _new_operation_guard(system, user):
                return reply
            if "本次英语练习已经结束" in system:
                # A separate writer can acquire immediately during the farewell.
                import sqlite3
                with sqlite3.connect(self.store.database, timeout=.02) as db:
                    db.execute("BEGIN IMMEDIATE")
                return '{"reply":"我们休息一下吧。"}'
            self.now += 21
            return '{"reply":"我记住你喜欢火车","intent":"remember_preference","value":"火车"}'
        with patch.object(server, "MODEL", model):
            result = await server.chat(self.request("请记住我喜欢火车"))
        self.assertTrue(result["learning_closed"])
        self.assertEqual(self.store.english_progress("test-puppy", "test-child"), before)
        self.assertEqual(self.store.conversation("test-puppy", "test-child", "english", "s"), history)
        self.assertEqual(self.store.memory_entries("test-puppy", "test-child"), [])

    async def test_leaving_english_ends_active_device_sessions(self):
        await server.chat(self.request())
        await server.set_mode("test-puppy", server.ModeRequest(mode="story"))
        self.assertEqual(Store(self.root / "store.sqlite3").conversation_mode("test-puppy"), "story")
        self.assertFalse(self.learning.status("test-puppy", "test-child")["active"])
