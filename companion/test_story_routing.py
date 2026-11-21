import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from companion.agent import PetAgent
from companion.store import Store
from companion.story_routing import pauses_story, replacement_character, requests_story, short_story_followup


class StoryRoutingTests(unittest.TestCase):
    def test_short_story_followup_excludes_compound_or_personal_question(self):
        self.assertTrue(short_story_followup("然後呢？"))
        self.assertTrue(short_story_followup("接下来呢"))
        self.assertFalse(short_story_followup("然后呢？你记得我叫什么吗？"))
        self.assertFalse(short_story_followup("后来我说了什么？"))

    def test_explicit_story_request_is_one_narrative_turn(self):
        for text in ("讲个故事", "讲一个小猫和星星的短故事", "小狗，给我讲个故事",
                     "继续讲故事吧", "接着讲刚才的故事", "給我講個小熊的故事吧，講兩句就好。",
                     "给我讲个小熊的故事吧讲两句就好"):
            with self.subTest(text=text):
                self.assertTrue(requests_story(text))
        for text in ("不要讲故事", "讲故事有什么好处？", "故事里的小猫叫什么？",
                     "蓝色的英语怎么说？", "先不讲故事了。"):
            with self.subTest(text=text):
                self.assertFalse(requests_story(text))

    def test_character_change_applies_to_story_and_cannot_propose_rename(self):
        text = "哎，我說錯啦。不要小熊，換成小兔子。"
        previous = {"user": "讲个小熊的故事", "assistant": "小熊捡到一颗亮星星。", "mode": "story"}
        self.assertEqual(replacement_character(text, previous["assistant"]), "小兔子")
        self.assertEqual(replacement_character("不要小熊换成小兔子", previous["assistant"]), "小兔子")
        self.assertIsNone(replacement_character(text))
        self.assertIsNone(replacement_character("以后你叫小兔子", previous["assistant"]))
        self.assertEqual(replacement_character("把主角换成小兔子吧。"), "小兔子")
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "story.sqlite3")
            rename = {"reply": "你想以后叫我小兔子，对吗？", "intent": "rename_pet", "value": "小兔子"}
            story = {"reply": "小兔子捡到一颗星星，把它送回了夜空，星星眨眨眼向它道谢。",
                     "intent": "chat", "story_title": "小兔子和星星"}
            prompts = []
            def model(system, _user):
                prompts.append(system)
                return json.dumps(rename if len(prompts) == 1 else story, ensure_ascii=False)

            result = PetAgent(store, model).reply(
                "puppy", text, mode="story", session_id="mic", personal_request=False,
                previous_guest_turn=previous)
            self.assertEqual(result["reply"], story["reply"])
            self.assertIsNone(store.pending_name("puppy", None, "mic"))
            self.assertIn("孩子本轮把故事主角换成", prompts[0])

    def test_public_guest_followup_uses_only_transient_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "story.sqlite3")
            store.append_conversation("puppy", "child", "我的秘密", "不应该进入匿名上下文", session_id="mic")
            previous = {"user": "七加五等于多少？", "assistant": "等于十二。", "mode": "chat"}
            def model(system, _user):
                self.assertIn("等于十二", system)
                self.assertNotIn("我的秘密", system)
                return '{"reply":"再加三就是十五。","intent":"chat"}'

            result = PetAgent(store, model).reply(
                "puppy", "那再加三呢？", session_id="mic", personal_request=False,
                previous_guest_turn=previous)
            self.assertEqual(result["reply"], "再加三就是十五。")
            self.assertEqual(store.conversation("puppy", None, session_id="mic"), [])

    def test_only_leading_explicit_pause_routes_to_dialogue(self):
        for text in (
            "先停一下故事，三加二等于几？",
            "先不续故事，蓝色的英语怎么说？",
            "先不讲故事了。我喜欢消防车。",
            "暂停故事，你记得我喜欢什么？",
            "先停一下，2加3等于几？",
            "暂停，鼻子的英文是什么？",
        ):
            with self.subTest(text=text):
                self.assertTrue(pauses_story(text))
        for text in (
            "接着讲故事", "故事里小兔子说‘先停一下故事’", "讲一个不讲故事的小狗",
            "先停一下故事里的小车", "三加二等于几？",
        ):
            with self.subTest(text=text):
                self.assertFalse(pauses_story(text))

    def test_pause_answers_question_without_advancing_chapter(self):
        with tempfile.TemporaryDirectory(prefix="puppy-story-routing-") as directory:
            store = Store(Path(directory) / "story.sqlite3")
            store.set_conversation_mode("puppy", "story")
            first = lambda _system, _user: json.dumps({
                "reply": "小兔子找到了红色的球，和小熊一起把球放进篮子里。",
                "intent": "chat", "story_title": "红球"}, ensure_ascii=False)
            PetAgent(store, first).reply("puppy", "讲个故事", "child", "story", "chapter-1")
            before = store.story("puppy", "child")
            seen = []

            def answer(system, _user):
                seen.append(system)
                return json.dumps({"reply": "三加二等于五。", "intent": "chat"}, ensure_ascii=False)

            result = PetAgent(store, answer).reply(
                "puppy", "先停一下，三加二等于几？", "child", "story", "pause-1")
            self.assertEqual(result["reply"], "三加二等于五。")
            self.assertNotIn("story", result)
            self.assertEqual(store.story("puppy", "child"), before)
            self.assertEqual(store.conversation_mode("puppy"), "story")
            self.assertIn("当前模式：chat", seen[0])
            self.assertNotIn("现在直接给四岁孩子讲故事", seen[0])

    def test_chat_mode_story_request_saves_chapter_then_returns_to_chat(self):
        with tempfile.TemporaryDirectory(prefix="puppy-story-request-") as directory:
            store = Store(Path(directory) / "story.sqlite3")
            seen = []

            def story_model(system, _user):
                seen.append(system)
                return json.dumps({"reply": "小猫在窗边看见一颗亮星星，拿来小板凳坐好，开心地向星星挥手。",
                                   "intent": "chat", "story_title": "窗边的星星"}, ensure_ascii=False)

            result = PetAgent(store, story_model).reply(
                "puppy", "讲一个小猫和星星的短故事", "child", "chat", "story-1")
            self.assertEqual(result["story"]["chapter"], 1)
            self.assertEqual(store.conversation_mode("puppy"), "chat")
            self.assertIn("现在直接给四岁孩子讲故事", seen[0])

            answer = PetAgent(store, lambda system, _user: json.dumps(
                {"reply": "苹果的英文是 apple。", "intent": "chat"}, ensure_ascii=False))
            question = answer.reply("puppy", "苹果的英文怎么说？", "child", "chat", "question-1")
            self.assertEqual(question["reply"], "苹果的英文是 apple。")
            self.assertEqual(store.story("puppy", "child")["chapter"], 1)


class StoryGatewayRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_uses_chat_for_explicit_pause_without_changing_saved_mode(self):
        with tempfile.TemporaryDirectory(prefix="puppy-story-gateway-") as directory:
            with patch.dict(os.environ, {"PUPPY_DATA_DIR": directory}):
                try:
                    from companion import server
                except ModuleNotFoundError as error:
                    if error.name == "fastapi":
                        self.skipTest("Gateway dependencies are not installed locally")
                    raise
            store = Store(Path(directory) / "test.sqlite3")
            store.set_conversation_mode("synthetic", "story")
            with patch.object(server, "STORE", store), \
                 patch.object(server.INFERENCE, "run", new_callable=AsyncMock,
                              return_value={"reply": "三加二等于五。", "intent": "chat", "action": ""}):
                request = server.ChatRequest(device_id="synthetic", mode="story",
                                             text="先停一下，三加二等于几？")
                response = await server.chat(request)
                self.assertEqual(response["reply"], "三加二等于五。")
                self.assertEqual(request.mode, "chat")
                self.assertEqual(store.conversation_mode("synthetic"), "story")

                store.set_conversation_mode("synthetic", "chat")
                story_request = server.ChatRequest(device_id="synthetic", mode="chat",
                                                   text="讲一个小猫和星星的短故事")
                await server.chat(story_request)
                self.assertEqual(story_request.mode, "story")
                self.assertEqual(store.conversation_mode("synthetic"), "chat")


if __name__ == "__main__":
    unittest.main()
