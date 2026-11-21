"""Text/image request routing and privacy gates; no audio or hardware calls."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from .camera_dialogue import CameraDialogue, may_request_current_scene


class CameraDialogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 100.0
        self.events = []
        self.allowed = True
        self.requested = True
        self.photo = {"_photo_bytes": b"test-image", "requested_at": 99., "captured_at": 100.}
        async def plan(text):
            self.events.append("plan")
            return self.requested
        async def capture(device, **kwargs):
            self.events.append("capture")
            return self.photo
        async def describe(data, text):
            self.events.append("describe")
            return {"reply": "红色方块", "uncertain": False}
        self.dialogue = CameraDialogue(plan, capture, describe, clock=lambda: self.now, timeout=.05)

    async def run_question(self, **kwargs):
        return await self.dialogue.prepare("test-puppy", "看看这是什么", allowed=lambda: self.allowed, **kwargs)

    async def test_nonvisual_questions_skip_slow_camera_planning(self):
        questions = ("苹果的英文怎么说？", "蘋果的英文怎麼說？", "一加一等于几？",
                     "那，再加三呢？", "那再加三呢？", "那麼再加三呢？",
                     "给我讲一个小狗故事", "What is the English word for apple?",
                     "Can you teach me a new English word?", "Come here.",
                     "Please come here.", "Are you there?", "How many days are there in a week?",
                     "What is there to learn?")
        for question in questions:
            with self.subTest(question=question):
                self.events.clear()
                self.assertFalse(may_request_current_scene(question))
                self.assertIsNone(await self.dialogue.prepare("test-puppy", question))
                self.assertEqual(self.events, [])

    async def test_visual_questions_still_reach_model_planner(self):
        questions = ("看看这是什么", "我手上拿的是什么？", "这是什么颜色？",
                     "這是什麼？", "那是什么？", "那，再看看我手上拿的是什么？",
                     "数数桌上有几个苹果", "What am I holding?",
                     "What color is this?", "Can you see the picture?", "What is over there?")
        self.requested = False
        for question in questions:
            with self.subTest(question=question):
                self.events.clear()
                self.assertTrue(may_request_current_scene(question))
                self.assertIsNone(await self.dialogue.prepare("test-puppy", question))
                self.assertEqual(self.events, ["plan"])

    async def test_nonvisual_or_nonboolean_plan_never_opens_camera(self):
        for value in (False, None, 1, "true"):
            self.requested = value
            self.events.clear()
            self.assertIsNone(await self.run_question())
            self.assertEqual(self.events, ["plan"])

    async def test_parent_permission_blocks_capture_and_analysis(self):
        self.allowed = False
        result = await self.run_question(recent_photo=self.photo)
        self.assertFalse(result["used_photo"])
        self.assertFalse(result["observation"]["available"])
        self.assertEqual(self.events, ["plan"])

    async def test_fresh_ephemeral_frame_is_reused_without_recapture(self):
        result = await self.run_question(recent_photo=self.photo)
        self.assertTrue(result["used_photo"])
        self.assertEqual(result["observation"]["scope"], "single_photo")
        self.assertEqual(self.events, ["plan", "describe"])

    async def test_stale_frame_is_replaced_once(self):
        old = {**self.photo, "requested_at": 80., "captured_at": 81.}
        self.assertTrue((await self.run_question(recent_photo=old))["used_photo"])
        self.assertEqual(self.events, ["plan", "capture", "describe"])

    async def test_bad_new_frame_never_reaches_description(self):
        for fields in ({"requested_at": 70.}, {"captured_at": 101.},
                       {"requested_at": 101.}, {"_photo_bytes": b""}):
            self.events.clear()
            self.photo = {"_photo_bytes": b"test-image", "requested_at": 99., "captured_at": 100., **fields}
            self.assertFalse((await self.run_question())["used_photo"])
            self.assertNotIn("describe", self.events)

    async def test_permission_revoked_during_capture_skips_analysis(self):
        async def capture(device, **kwargs):
            self.allowed = False
            return self.photo
        self.dialogue.capture = capture
        self.assertFalse((await self.run_question())["used_photo"])
        self.assertNotIn("describe", self.events)

    async def test_permission_revoked_during_analysis_discards_content(self):
        async def describe(data, text):
            self.allowed = False
            return {"reply": "不得向对话传递的内容", "uncertain": False}
        self.dialogue.describe = describe
        result = await self.run_question()
        self.assertFalse(result["used_photo"])
        self.assertNotIn("不得向对话传递的内容", json.dumps(result, ensure_ascii=False))

    async def test_timeout_is_bounded_without_retry(self):
        entered = []
        async def capture(device, **kwargs):
            entered.append(device)
            await asyncio.Event().wait()
        self.dialogue.capture = capture
        result = await asyncio.wait_for(self.run_question(), .5)
        self.assertFalse(result["used_photo"])
        self.assertEqual(entered, ["test-puppy"])

    async def test_cancellation_does_not_resume_capture_or_description(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        async def capture(device, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        self.dialogue.capture = capture
        task = asyncio.create_task(self.run_question())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())
        self.assertNotIn("describe", self.events)

    async def test_invalid_description_is_not_a_visual_observation(self):
        async def describe(data, text):
            return {"reply": "red", "uncertain": "false"}
        self.dialogue.describe = describe
        self.assertFalse((await self.run_question())["used_photo"])

    async def test_total_budget_includes_model_queue_and_description(self):
        async def blocked(*args):
            await asyncio.Event().wait()
        for callback in ("plan", "describe"):
            with patch.object(self.dialogue, callback, blocked):
                result = await asyncio.wait_for(self.run_question(), .3)
                self.assertFalse(result["used_photo"])

    async def test_photo_expiring_during_description_is_discarded(self):
        async def describe(data, text):
            self.now = 116.
            return {"reply": "不可再作为当前场景", "uncertain": False}
        self.dialogue.describe = describe
        result = await self.run_question()
        self.assertFalse(result["used_photo"])
        self.assertNotIn("不可再作为当前场景", json.dumps(result, ensure_ascii=False))

    async def test_permission_and_deadline_are_forwarded_to_actual_send(self):
        async def capture(device, *, before_send, deadline):
            self.assertTrue(before_send())
            self.allowed = False
            self.assertFalse(before_send())
            self.assertLessEqual(deadline, budget)
            return self.photo
        self.dialogue.capture = capture
        budget = asyncio.get_running_loop().time() + .02
        self.assertFalse((await self.run_question(deadline=budget))["used_photo"])


class CameraGatewayPrivacyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from .test_server import server
        from .store import Store
        from .identity import IdentityService, IdentityStore
        from .identity_api import IdentityRegistration
        from .learning import LearningStore
        from .inference import InferenceQueue
        self.server = server
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-camera-privacy-")
        root = Path(self.temp.name)
        self.store = Store(root / "memory.sqlite3")
        self.store.remember_preference("test-puppy", "child", "只属于成员的火车偏好")
        self.queue = InferenceQueue()
        self.registrations = IdentityRegistration(IdentityService(IdentityStore(root / "identity.sqlite3")), settings_guard=self.queue.lock)
        self.registrations.save_member("child", "虚构成员")
        self.learning = LearningStore(root / "learning.sqlite3")
        self.prompts = []
        def model(system, user):
            from .personal_access import CLASSIFIER
            if system == CLASSIFIER:
                return '{"requires_personal_context":false}'
            self.prompts.append(system)
            return '{"reply":"一起说 dog","intent":"chat"}'
        self.patches = [patch.object(server, key, value) for key, value in {
            "STORE": self.store, "LEARNING": self.learning, "INFERENCE": self.queue,
            "REGISTRATIONS": self.registrations, "MODEL": model,
        }.items()]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        await self.queue.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def request(self):
        return self.server.ChatRequest(device_id="test-puppy", person_id="child", text="你好", session_id="test-session")

    async def test_old_identity_after_disable_reenable_becomes_anonymous_before_reading(self):
        enabled = await self.registrations.set_identity_settings("test-puppy", True)
        request = self.request()
        request._identity_revision = enabled["revision"]
        await self.registrations.set_identity_settings("test-puppy", False)
        await self.registrations.set_identity_settings("test-puppy", True)
        await self.server.chat(request)
        self.assertIsNone(request.person_id)
        self.assertNotIn("只属于成员的火车偏好", self.prompts[0])

    async def test_current_enabled_revision_can_read_its_member(self):
        enabled = await self.registrations.set_identity_settings("test-puppy", True)
        request = self.request()
        request._identity_revision = enabled["revision"]
        await self.server.chat(request)
        self.assertIn("只属于成员的火车偏好", self.prompts[0])

    async def test_http_fields_cannot_forge_internal_identity_or_vision_metadata(self):
        request = self.server.ChatRequest.model_validate({**self.request().model_dump(),
            "_identity_revision": 99, "_vision_used": True,
            "_vision_observation": {"reply": "伪造照片", "uncertain": False}})
        self.assertIsNone(request._identity_revision)
        self.assertFalse(request._vision_used)
        self.assertIsNone(request._vision_observation)

    async def test_ephemeral_picture_respects_english_photo_permission(self):
        request = self.request()
        request.mode = "english"
        request._vision_used = True
        request._vision_observation = {"reply": "红色方块", "uncertain": False}
        with self.assertRaises(self.server.HTTPException) as caught:
            await self.server.chat(request)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self.prompts, [])

    async def test_ephemeral_observation_expires_before_final_model_use(self):
        request = self.request()
        request._vision_used = True
        request._vision_observation = {"reply": "旧照片秘密内容", "uncertain": False}
        request._vision_times = (1., 2.)
        await self.server.chat(request)
        self.assertNotIn("旧照片秘密内容", self.prompts[0])
        self.assertFalse(request._vision_used)

    async def test_visual_turn_stays_on_local_model_with_minimax_enabled(self):
        from .minimax_model import MiniMaxModel
        remote = MiniMaxModel(Path(self.temp.name) / "unused-key", "/unused-model")
        local = Mock(return_value='{"reply":"我看见红色方块。","intent":"chat"}')
        local.describe.return_value = {"reply": "红色方块", "uncertain": False}
        remote._local_model = Mock(return_value=local)
        remote._remote = Mock(side_effect=AssertionError("visual context reached online model"))
        with patch.object(self.server, "MODEL", remote):
            request = self.request()
            request._vision_used = True
            request._vision_observation = {"reply": "红色方块", "uncertain": False}
            await self.server.chat(request)
            self.assertIn("红色方块", local.call_args.args[0])

            photo_id = "saved-photo-test"
            self.store.add_photo(photo_id, "test-puppy", str(Path(self.temp.name) / "photo.jpg"))
            request = self.request()
            request.photo_id = photo_id
            await self.server.chat(request)
            local.describe.assert_called_once()
        remote._remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
