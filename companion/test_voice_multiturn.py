"""Regressions from M2's actual quiet multi-turn failures; no hardware calls."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from .agent import PetAgent
from .camera_dialogue import CameraDialogue
from .personal_access import classify_personal_request
from . import test_child_activity_gateway as fixtures


class PublicFollowupTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_crosscheck_cannot_turn_a_clear_start_into_stop(self):
        from .asr_disagreement import prefer_clear_activity_control
        from .child_activity import ChildActivitySessions
        self.assertEqual(prefer_clear_activity_control("先顾讲故事了", "先不讲故事了。"), "先不讲故事了。")
        for first in ("继续讲故事", "先去讲故事了", "先不讲故事了，记住我的爱好", "给我讲个故事"):
            self.assertEqual(prefer_clear_activity_control(first, "先不讲故事了。"), first)
        sessions = ChildActivitySessions()
        self.assertEqual(sessions.route("puppy", "voice", "學英語", "chat"), "english")
        self.assertEqual(sessions.route("puppy", "voice", "先不學英語了。", "chat"), "chat")

    async def test_public_followups_never_enter_camera_planner(self):
        plan, capture, describe = AsyncMock(return_value=True), AsyncMock(), AsyncMock()
        camera = CameraDialogue(plan, capture, describe)
        for text in ("那白天也能看见它吗？", "那小水滴起什么作用呀？", "它为什么要借伞呀？",
                     "那它后来怎么样啦？", "这个词是什么意思呀？", "为什么小猫会呼噜呀？",
                     "小猫的英文是什么样？"):
            with self.subTest(text=text):
                self.assertIsNone(await camera.prepare("puppy", text))
        plan.assert_not_awaited()
        capture.assert_not_awaited()
        from .personal_access import is_public_why_question
        self.assertTrue(is_public_why_question("为什么天空是蓝色的呀？"))
        self.assertFalse(is_public_why_question("它为什么要借伞呀？"))
        self.assertTrue(is_public_why_question("它为什么要借伞呀？", {"mode": "story"}))
        for text in ("为什么我最喜欢小猫？", "为什么月亮发光，请挥手", "为什么月亮发光然后坐下", "为什么我家小猫不吃饭？"):
            self.assertFalse(is_public_why_question(text))

    async def test_meaning_and_repeat_need_a_word_in_this_connection(self):
        previous = {"mode": "english", "vocabulary": ("apple", "苹果")}
        for text in ("这个词是什么意思呀？", "Apple.", "...Apple."):
            self.assertEqual(PetAgent.english_followup(text, previous), ("apple", "苹果"))
            self.assertIsNone(PetAgent.english_followup(text, None))
        for text in ("我昨天学的这个词是什么意思？", "这个词是什么意思，记住我的爱好", "Cat."):
            self.assertIsNone(PetAgent.english_followup(text, previous))

    async def test_story_asr_preserves_hero_and_personal_request_boundaries(self):
        from .asr_disagreement import prefer_quiet_story_request
        second = "给我讲个小熊借雨伞的故事。"
        self.assertEqual(prefer_quiet_story_request("給我講這小熊介於反的故事", second), second)
        for first in ("给我讲个小兔子借雨伞的故事", "给我讲我家小熊昨天的故事", "不要讲故事"):
            self.assertEqual(prefer_quiet_story_request(first, second), first)


class VoiceFollowupGatewayTests(fixtures.ChildActivityGatewayTests):
    async def test_confirmed_voice_word_practice_keeps_saved_and_session_stages_together(self):
        server = fixtures.server
        async def turn(text):
            request = server.ChatRequest(device_id="puppy", session_id="named-voice", person_id="child",
                                         text=text, mode="english")
            request._from_device, request._identity_revision = True, 1
            return await server.chat(request)
        with patch.object(server.REGISTRATIONS, "identity_allowed", return_value=True), \
             patch.object(server.REGISTRATIONS, "member", return_value={"generation": 0, "display_name": "虚构孩子"}):
            await turn("苹果用英语怎么说呀？")
            self.assertEqual(self.store.english_progress("puppy", "child"), [])
            answer = await turn("Apple.")
            self.assertEqual(answer["english_kind"], "practice")
            self.assertEqual(answer["lesson"]["target_word"], "apple")
            for current in (self.store.english_lesson("puppy", "child"),
                            self.store.child_sessions.lesson("puppy", "named-voice", "child")):
                self.assertEqual((current["word"], current["stage"]), ("apple", "play"))
    async def test_plain_why_uses_two_calls_but_private_why_keeps_access_check(self):
        calls = []
        def model(system, user):
            calls.append(system)
            if system.startswith("只判断本轮原话"):
                return '{"requires_personal_context":true}'
            return '{"reply":"空气把阳光里的蓝光向四周散开，所以天空看起来是蓝色的。","intent":"chat"}'
        with patch.object(fixtures.server, "MODEL", model):
            answer = await self.voice_turn("为什么天空是蓝色的呀？", mode="chat")
            self.assertIn("蓝光", answer["reply"])
            self.assertEqual(len(calls), 2)
            answer = await self.voice_turn("为什么我最喜欢小猫？", mode="chat")
        self.assertEqual(len(calls), 3)
        self.assertEqual(answer["business_status"]["code"], "personal_access_unavailable")
    async def voice_turn(self, text, previous=None, mode="english", alternatives=()):
        request = fixtures.server.ChatRequest(device_id="puppy", session_id="voice", text=text, mode=mode)
        request._from_device = True
        request._previous_guest_turn = previous
        request._heard_alternatives = alternatives
        return await fixtures.server.chat(request)

    async def test_english_word_meaning_and_repeat_do_not_revert_to_theme_word(self):
        first = await self.voice_turn("苹果用英语怎么说呀？")
        self.assertIn("apple", first["reply"])
        previous = {"user": "苹果用英语怎么说呀？", "assistant": first["reply"],
                    "mode": "english", "vocabulary": first.get("vocabulary")}
        for text in ("这个词是什么意思呀？", "Apple."):
            result = await self.voice_turn(text, previous)
            self.assertIn("apple", result["reply"])
            self.assertIn("苹果", result["reply"])
            self.assertNotIn("dog", result["reply"])
            self.assertNotIn("business_status", result)
        self.assertEqual(self.store.child_sessions.lesson("puppy", "voice", None)["word"], "apple")
        self.assertEqual(self.store.child_sessions.lesson("puppy", "voice", None)["stage"], "play")
        await self.voice_turn("这个词是什么意思呀？", previous)
        self.assertEqual(self.store.child_sessions.lesson("puppy", "voice", None)["stage"], "play")
        # Explicit question/meaning turns are not scored as successful practice.
        self.assertEqual(self.store.english_progress("puppy", None), [])

    async def test_word_question_with_homophone_ending_still_teaches_cat(self):
        result = await self.voice_turn("小猫的英文是什么样？")
        self.assertIn("cat", result["reply"])
        self.assertEqual(result["reply_source"], "vocabulary")

    async def test_public_followup_correction_replaces_wrong_draft_before_reply(self):
        previous = {"user": "为什么月亮发光", "assistant": "月亮反射太阳的光。", "mode": "chat"}
        def model(system, user):
            if system.startswith("只判断本轮原话"):
                return '{"requires_personal_context":false}'
            if system.startswith("复核给四岁孩子"):
                data = json.loads(user)
                self.assertIn("不能", data["draft_answer"])
                self.assertIn("月亮", str(data["previous_public_turns"]))
                return '{"reply":"有时白天也能看到月亮，只是没有夜里那么显眼。"}'
            return '{"reply":"白天不能看到月亮。","intent":"chat"}'
        with patch.object(fixtures.server, "MODEL", model):
            result = await self.voice_turn("那白天也能看见它吗？", previous, "chat")
        self.assertIn("有时白天也能", result["reply"])
        self.assertNotIn("不能", result["reply"])

    async def test_story_correction_receives_both_recognizers_and_public_context(self):
        previous = {"user": "讲个小熊借雨伞的故事", "assistant": "下雨了，小熊向小兔借伞。", "mode": "story"}
        alternatives = ("它为什么要见散鸭", "它为什么要借伞呀")
        calls = []
        def model(system, user):
            calls.append(system)
            if system.startswith("只判断本轮原话"):
                return '{"requires_personal_context":false}'
            if system.startswith("复核给四岁孩子的故事"):
                data = json.loads(user)
                self.assertEqual(data["heard_alternatives"], list(alternatives))
                self.assertIn("借伞", str(data["previous_turns"]))
                return '{"reply":"因为下雨了，小熊借伞是为了不淋湿自己。"}'
            return '{"reply":"故事里没有见散鸭。","intent":"chat","story_title":"借伞"}'
        with patch.object(fixtures.server, "MODEL", model):
            result = await self.voice_turn(alternatives[0], previous, "story", alternatives)
        self.assertIn("不淋湿", result["reply"])
        self.assertNotIn("鸭", result["reply"])
        self.assertEqual(len(calls), 2)  # public question: draft and one correction; no rewrite loop

    async def test_current_story_reaches_classifier_and_keeps_more_than_last_pair(self):
        previous = {"user": "它为什么要借伞呀？", "assistant": "因为下雨了，小熊不想淋湿。",
                    "mode": "story", "story_request": "讲个小熊借雨伞的故事，两句就好。",
                    "history": [{"user": "讲个小熊借雨伞的故事，两句就好。", "assistant": "小熊向小兔借了一把伞。"},
                                {"user": "它为什么要借伞呀？", "assistant": "因为下雨了，小熊不想淋湿。"}]}
        def model(system, user):
            if system.startswith("只判断本轮原话"):
                data = json.loads(user)
                return json.dumps({"requires_personal_context": "current_public_conversation" not in data})
            if system.startswith("复核给四岁孩子的故事"):
                self.assertIn("还给小兔", json.loads(user)["proposed_reply"])
                return json.dumps({"reply": json.loads(user)["proposed_reply"]}, ensure_ascii=False)
            if "借了一把伞" in system and "最初故事要求" in system:
                return '{"reply":"小熊撑着借来的伞走回家，把伞晾干后还给小兔。","intent":"chat","story_title":"借伞"}'
            return '{"reply":"小鸟换了一顶帽子，准备去看电视。","intent":"chat","story_title":"帽子"}'
        with patch.object(fixtures.server, "MODEL", model):
            result = await self.voice_turn("那它后来怎么样啦？", previous, "story")
        self.assertIn("还给小兔", result["reply"])
        self.assertNotIn("business_status", result)
        self.assertIsNone(self.store.story("puppy", None))

    async def test_public_context_does_not_authorize_saved_personal_records(self):
        previous = {"user": "讲个小熊的故事", "assistant": "小熊借了伞。", "mode": "story"}
        def model(system, user):
            if system.startswith("只判断本轮原话"):
                return '{"requires_personal_context":true}'
            self.fail("A personal-history request must not reach answer generation")
        with patch.object(fixtures.server, "MODEL", model):
            answer = await self.voice_turn("你还记得我昨天喜欢的故事吗？", previous, "story")
        self.assertEqual(answer["business_status"]["code"], "personal_access_unavailable")

    async def test_limited_guest_cannot_bypass_learning_limit_with_vocabulary(self):
        self.learning.set_settings("puppy", None, patch={"daily_minutes": 1})
        with patch.object(self.learning, "before_turn", return_value={
                "allowed": False, "reason": "daily_limit", "settings": {"save_progress": True}}), \
             patch.object(PetAgent, "finish_learning", return_value={"reply": "先休息一下。", "learning_closed": True}) as finish:
            answer = await self.voice_turn("苹果用英语怎么说呀？")
        finish.assert_called_once()
        self.assertNotIn("apple", answer["reply"])
        self.assertIsNone(self.store.child_sessions.lesson("puppy", "voice", None))

    async def test_cat_audio_followup_uses_english_asr_and_retains_word(self):
        server = fixtures.server
        answers, events, languages = [], [], []
        class Socket:
            headers = {"authorization": "Bearer test", "device-id": "puppy"}
            accept = AsyncMock()
        class Speech:
            repeat = "舅舅"
            def __init__(self, _model):
                self.turn = 0
            async def transcribe_async(self, pcm, *, language=None):
                languages.append(language)
                if language == "en":
                    return "Cat."
                self.turn += 1
                return ("小猫的英文是什么呀？" if self.turn == 1 else
                        "用这个词说一句话" if self.turn == 2 else self.repeat if self.turn == 3 else "Dog.")
        class Session:
            def __init__(self, socket, **callbacks):
                self.callbacks = callbacks
            async def run(self):
                for _ in range(4):
                    pcm = b"\x80\0" * 16000
                    await self.callbacks["prepare_reply"](pcm)
                    text = await self.callbacks["transcribe"](pcm)
                    answers.append(await self.callbacks["chat"](pcm, text))
                    self.callbacks["on_event"]("tts_sent", {})
        def model(system, user):
            if system.startswith("只判断本轮原话"):
                return '{"requires_personal_context":false}'
            if "一句话" in user:
                return '{"reply":"The cats are small. 小猫们很小。","intent":"chat","english_kind":"question"}'
            return '{"reply":"你好呀","intent":"chat","english_kind":"question"}'
        with patch.object(server, "MODEL", model), \
             patch.object(server, "device_audio_conversation_id", side_effect=["trial-" + str(n) for n in range(4)]), \
             patch.object(server, "DEVICE_TOKEN", "test"), \
             patch.object(server, "APPLE_ASR_BINARY", "/probe"), \
             patch.object(server, "SpeechServices", Speech), \
             patch.object(server, "AudioSession", Session), \
             patch.object(server, "OpusCodec", object), \
             patch.object(server, "apple_transcribe_pcm", AsyncMock(return_value="看。")), \
             patch.object(server.REGISTRATIONS, "consume_enrollment", AsyncMock(return_value=None)), \
             patch.object(server.REGISTRATIONS, "identity_settings", return_value={"enabled": False}), \
             patch.object(self.store, "conversation_mode", return_value="english"), \
             patch.object(server.DIALOGUE_CAMERA, "prepare", AsyncMock(return_value=None)):
            for repeat in ("舅舅", "...cab.", "Cobbook", "tad ah ah"):
                Speech.repeat = repeat
                await server.device_audio(Socket())
        self.assertEqual(len(answers), 16)
        for start in range(0, 16, 4):
            self.assertTrue(all("cat" in answer["reply"] for answer in answers[start:start+3]))
            self.assertIn("小狗", answers[start+3]["reply"])
            self.assertIn("小猫", answers[start+3]["reply"])
        self.assertIn("en", languages)
        for n in range(4):
            self.assertEqual(self.store.child_sessions.lesson("puppy", "trial-" + str(n), None)["stage"], "play")


if __name__ == "__main__":
    unittest.main()
