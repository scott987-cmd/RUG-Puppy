"""Durable four-step English practice without scoring a child's pronunciation."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from companion.agent import PetAgent
from companion.store import Store


class EnglishCurriculumTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.directory.name) / "pet.sqlite3")
        self.settings = {"theme": "body_parts", "save_progress": True}

    def tearDown(self):
        self.directory.cleanup()

    def turn(self, text, *, person="child", settings=None, record_learning=True):
        def model(system, user):
            if "不学英语了" in user or "不想学了" in user:
                return json.dumps({"reply": "好呀，我们休息一下，想学时再叫我。", "intent": "chat"}, ensure_ascii=False)
            if "鼻子的英文" in user:
                return json.dumps({"reply": "鼻子的英文是 nose。", "intent": "chat",
                                   "english_kind": "question"}, ensure_ascii=False)
            word = re.search(r"practice 的本轮英语目标：([a-z]+)", system).group(1)
            return json.dumps({"reply": f"{word} 的中文意思，我们一起玩 {word}。", "intent": "chat",
                               "english_kind": "practice"}, ensure_ascii=False)
        return PetAgent(self.store, model).reply(
            "puppy", text, person, mode="english", session_id="session",
            learning_settings=settings or self.settings, record_learning=record_learning)

    def test_intro_repeat_game_and_response_survive_new_agent_instances(self):
        expected = (("一起学英语", "introduce", "repeat"),
                    ("hand", "repeat", "play"),
                    ("继续", "play", "respond"),
                    ("在这里", "respond", "done"))
        for text, activity, next_stage in expected:
            answer = self.turn(text)
            self.assertEqual(answer["lesson"]["target_word"], "hand")
            self.assertEqual(answer["lesson"]["activity"], activity)
            self.assertEqual(self.store.english_lesson("puppy", "child")["stage"], next_stage)
            if activity == "introduce":
                question = self.turn("鼻子的英文是什么？")
                self.assertEqual(question["english_kind"], "question")
                self.assertNotIn("lesson", question)
                self.assertEqual(self.store.english_lesson("puppy", "child")["stage"], "repeat")
        self.assertEqual(self.store.english_progress("puppy", "child")[0]["presentations"], 4)
        next_word = self.turn("继续")
        self.assertEqual((next_word["lesson"]["target_word"], next_word["lesson"]["activity"]),
                         ("eye", "introduce"))

    def test_explicit_next_word_skips_remaining_stage(self):
        self.turn("一起学英语")
        answer = self.turn("下一个")
        self.assertEqual((answer["lesson"]["target_word"], answer["lesson"]["activity"]),
                         ("eye", "introduce"))
        self.assertEqual(self.store.english_lesson("puppy", "child")["word"], "eye")

    def test_explicit_pause_keeps_stage_until_child_resumes(self):
        self.turn("一起学英语")
        before = self.store.english_lesson("puppy", "child")
        paused = self.turn("先不学英语了")
        self.assertEqual(paused["reply"], "好呀，我们休息一下，想学时再叫我。")
        self.assertNotIn("lesson", paused)
        self.assertEqual(self.store.english_lesson("puppy", "child"), before)
        paused_again = self.turn("我不想学了")
        self.assertNotIn("lesson", paused_again)
        self.assertEqual(self.store.english_lesson("puppy", "child"), before)
        resumed = self.turn("继续")
        self.assertEqual(resumed["lesson"]["activity"], "repeat")

    def test_replay_does_not_advance_or_claim_learning(self):
        self.turn("一起学英语")
        replay = self.turn("我没听清")
        self.assertEqual(replay["lesson"]["activity"], "repeat")
        self.assertFalse(replay["lesson"]["advance"])
        self.assertEqual(self.store.english_lesson("puppy", "child")["stage"], "repeat")
        self.assertEqual(self.store.english_progress("puppy", "child")[0]["text_matches"], 0)

    def test_stage_survives_store_reopen_and_theme_change_starts_allowed_word(self):
        self.turn("一起学英语")
        self.store = Store(Path(self.directory.name) / "pet.sqlite3")
        self.assertEqual(self.turn("继续")["lesson"]["activity"], "repeat")
        changed = self.turn("继续", settings={"theme": "animals", "save_progress": True})
        self.assertEqual((changed["lesson"]["target_word"], changed["lesson"]["activity"]),
                         ("dog", "introduce"))
        self.assertEqual(self.store.english_lesson("puppy", "child")["word"], "dog")

    def test_due_review_returns_to_old_word_once_before_new_word(self):
        with patch("companion.store.time.time", return_value=1000000.0):
            for text in ("一起学英语", "hand", "继续", "找到啦"):
                self.turn(text)
        with patch("companion.store.time.time", return_value=1000000.0 + 24 * 3600):
            review = self.turn("继续")
            self.assertEqual(review["lesson"]["target_word"], "hand")
            self.assertEqual(review["lesson"]["review_checkpoint"], "next_day")
            for text in ("hand", "继续", "找到啦"):
                self.turn(text)
            following = self.turn("继续")
            self.assertEqual(following["lesson"]["target_word"], "eye")

    def test_guest_and_disabled_progress_do_not_create_private_stage(self):
        self.turn("一起学英语", person=None)
        self.turn("一起学英语", settings={"theme": "body_parts", "save_progress": False})
        self.turn("一起学英语", record_learning=False)
        self.assertIsNone(self.store.english_lesson("puppy", "child"))
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_forget_member_removes_course_stage_with_progress(self):
        self.turn("一起学英语")
        self.assertIsNotNone(self.store.english_lesson("puppy", "child"))
        self.store.forget_person("puppy", "child")
        self.assertIsNone(self.store.english_lesson("puppy", "child"))
        self.assertEqual(self.store.english_progress("puppy", "child"), [])


if __name__ == "__main__":
    unittest.main()
