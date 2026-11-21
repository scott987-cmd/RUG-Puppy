"""Story consistency boundaries; these tests do not invoke audio or a model.

Semantic quality still needs actual local-model transcript review. Deterministic
checks only promise grounded excerpts, stable titles and rejection of literal replay.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from companion.agent import PetAgent
from companion.store import Store


class StoryQualityTests(unittest.TestCase):
    def test_rejected_causal_story_is_rewritten_before_any_story_commit(self):
        drafts = iter(["小熊借来雨伞，把大家带进大雨里淋湿。", "小熊借来雨伞，大家躲在伞下，没有淋湿。"])
        checks = []
        def model(system, user):
            if system.startswith("复核给四岁孩子的故事"):
                proposed = json.loads(user)["proposed_reply"]
                checks.append(proposed)
                bad = "淋湿" in proposed and "没有" not in proposed
                return json.dumps({"reply": "小熊借来雨伞，大家躲在伞下，没有淋湿。" if bad else proposed})
            return json.dumps({"reply": next(drafts), "intent": "chat", "story_title": "借伞"})
        result = PetAgent(self.store, model).reply("puppy", "讲个小熊借雨伞的故事", "child", "story", review_reply=True)
        self.assertIn("没有淋湿", result["reply"])
        self.assertEqual(len(checks), 1)
        saved = self.store.story("puppy", "child")
        self.assertEqual(saved["chapter"], 1)
        self.assertNotIn("带进大雨", saved["summary"])

    def test_unavailable_story_review_never_publishes_unchecked_draft(self):
        def model(system, user):
            if system.startswith("复核给四岁孩子的故事"):
                return '{"reply":null}'
            return '{"reply":"小熊借到一把伞，和朋友一起回到了家。","intent":"chat","story_title":"借伞"}'
        with self.assertRaises(ValueError):
            PetAgent(self.store, model).reply("puppy", "讲个小熊的故事", "child", "story", review_reply=True)
        self.assertIsNone(self.store.story("puppy", "child"))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-story-quality-")
        self.path = Path(self.temp.name) / "story.sqlite3"
        self.store = Store(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def model(self, reply, title="小兔子的朋友", summary="模型偷偷加入了没讲过的飞船情节。"):
        return lambda system, user: json.dumps({"reply": reply, "intent": "chat",
                                               "story_title": title, "story_summary": summary}, ensure_ascii=False)

    def test_only_actual_reply_enters_persistent_excerpt(self):
        spoken = "小恐龙从草丛里找回球，轻轻放进小兔子的篮子里。"
        result = PetAgent(self.store, self.model(spoken)).reply("puppy", "讲个故事", "child", "story")
        self.assertEqual(result["story_summary"], PetAgent.STORY_EXCERPT_PREFIX + spoken)
        self.assertEqual(self.store.story("puppy", "child")["summary"], result["story_summary"])
        self.assertNotIn("飞船", result["story_summary"])

    def test_three_chapters_keep_title_and_excerpts_after_restart_and_next_day(self):
        spoken = ["小恐龙从草丛里找回球，轻轻放进小兔子的篮子里。",
                  "小兔子把篮子放好，和小恐龙用树叶拼出一个笑脸。",
                  "小恐龙带来一根胡萝卜，小兔子切成小块，和朋友一起分享。"]
        title = None
        for index, reply in enumerate(spoken):
            with patch("companion.store.time.time", return_value=1900000000 + (86400 if index == 2 else index * 60)):
                result = PetAgent(Store(self.path), self.model(reply, f"模型标题{index}"))\
                    .reply("puppy", "接着讲", "child", "story", f"session-{index}")
            title = title or result["story_title"]
            self.assertEqual(result["story_title"], title)
            self.assertEqual(result["story"]["chapter"], index + 1)
            for excerpt in spoken[:index + 1]:
                self.assertIn(excerpt, result["story_summary"])
        self.assertIsNone(Store(self.path).story("puppy", "other-child"))
        self.assertIsNone(Store(self.path).story("puppy", None))

    def test_literal_replay_retries_without_advancing_twice(self):
        old = "小兔子把红色的球放进篮子，坐在树下休息。"
        new = "小兔子醒来后，和小恐龙一起把树叶拼成一只蝴蝶。"
        PetAgent(self.store, self.model(old)).reply("puppy", "开始", "child", "story")
        outputs = iter([self.model(old), self.model(new)])
        seen = []
        def model(system, user):
            seen.append((system, user))
            return next(outputs)(system, user)
        result = PetAgent(self.store, model).reply("puppy", "继续", "child", "story")
        self.assertEqual(result["reply"], new)
        self.assertEqual(result["story"]["chapter"], 2)
        self.assertEqual(len(seen), 2)
        self.assertIn("已解决的问题保持解决", seen[0][0])
        self.assertNotIn("浅坑", seen[0][0])

    def test_invalid_or_repeated_reply_does_not_advance_story(self):
        spoken = "小恐龙找回小兔子的球，两位朋友在树下开心地玩。"
        PetAgent(self.store, self.model(spoken)).reply("puppy", "开始", "child", "story")
        prior = self.store.story("puppy", "child")
        for invalid in [spoken, "好的，我们继续！", "出发啦！"]:
            with self.assertRaises(ValueError):
                PetAgent(self.store, self.model(invalid)).reply("puppy", "继续", "child", "story")
            self.assertEqual(self.store.story("puppy", "child"), prior)

    def test_excerpt_window_keeps_complete_recent_replies_within_300_chars(self):
        replies = []
        for index in range(5):
            spoken = f"第{index}段，小兔子和朋友开始了新的游戏。" + "它们把不同颜色的树叶轻轻放在石头旁边，摆成了一条小路。" * 3
            result = PetAgent(self.store, self.model(spoken)).reply("puppy", "继续", "child", "story")
            replies.append(result["reply"])
            self.assertLessEqual(len(result["story_summary"]), 300)
            self.assertTrue(result["story_summary"].endswith(result["reply"]))
        saved = self.store.story("puppy", "child")["summary"]
        self.assertNotIn(replies[0], saved)
        self.assertIn(replies[-2], saved)
        self.assertIn(replies[-1], saved)

    def test_legacy_summary_is_context_but_not_relabelled_as_actual_words(self):
        self.store.save_story("puppy", "child", "旧标题", "旧模型摘要包含未讲的飞船。", 0)
        seen = []
        spoken = "小兔子在草地上摆好两把椅子，和朋友一起坐下看白云。"
        def model(system, user):
            seen.append(system)
            return self.model(spoken)(system, user)
        result = PetAgent(self.store, model).reply("puppy", "继续", "child", "story")
        self.assertIn("旧模型摘要", seen[0])
        self.assertEqual(result["story_title"], "旧标题")
        self.assertEqual(result["story_summary"], PetAgent.STORY_EXCERPT_PREFIX + spoken)

    def test_colors_and_record_learning_options_keep_existing_behavior(self):
        result = PetAgent(self.store, lambda system, user: '{"reply":"红色的英文是 red，一起说 red。","intent":"chat"}')\
            .reply("puppy", "学颜色", "child", "english", learning_settings={"theme": "colors", "save_progress": True},
                   record_learning=False, person_name="小朋友", vision_context={"reply": "有一个红色杯子。"})
        self.assertEqual(result["lesson"]["target_word"], "red")
        self.assertEqual(self.store.english_progress("puppy", "child"), [])


if __name__ == "__main__":
    unittest.main()
