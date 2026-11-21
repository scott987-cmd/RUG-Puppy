import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .agent import PetAgent
from .store import Store


class StoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-story-")
        self.path = Path(self.temp.name) / "store.sqlite3"
        self.store = Store(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_next_day_and_new_session_receive_correct_story_only(self):
        first = {"reply": "小恐龙来到蓝色门前。要敲门吗？", "intent": "chat", "story_title": "小恐龙探险",
                 "story_summary": "小恐龙来到蓝色门前，还没有敲门。"}
        PetAgent(self.store, lambda s, u: json.dumps(first)).reply("puppy", "讲恐龙故事", "child-a", "story", "session-1")
        seen = []
        def next_chapter(system, user):
            seen.append(system)
            return json.dumps({"reply": "小恐龙敲敲门，小兔子打开门递来一杯水。", "intent": "chat",
                               "story_title": "小恐龙探险", "story_summary": "小恐龙敲开蓝色门，见到小兔子并得到一杯水。"})
        import time
        next_day = time.time() + 86400
        with patch("companion.store.time.time", return_value=next_day):
            result = PetAgent(Store(self.path), next_chapter).reply("puppy", "敲门，接着讲", "child-a", "story", "new-session")
        self.assertEqual(result["story"]["chapter"], 2)
        self.assertIn("小恐龙来到蓝色门前。要敲门吗？", seen[0])
        self.assertNotIn("还没有敲门", seen[0])  # Only the actual reply becomes persisted progress.
        self.assertEqual(Store(self.path).story("puppy", "child-a")["chapter"], 2)
        self.assertIsNone(self.store.story("puppy", "child-b"))
        self.assertIsNone(self.store.story("puppy", None))

    def test_bad_model_result_does_not_advance_story(self):
        self.store.save_story("puppy", "child", "探险", "小兔子还在门口。", 0)
        model = lambda s, u: json.dumps({"reply": "小兔子在门口。", "intent": "chat"})
        with self.assertRaises(ValueError):
            PetAgent(self.store, model).reply("puppy", "继续", "child", "story")
        self.assertEqual(self.store.story("puppy", "child")["chapter"], 1)

    def test_invalid_action_in_story_mode_does_not_create_story(self):
        model = lambda s, u: json.dumps({"reply": "不能这样做。", "intent": "action", "action": "laser"})
        result = PetAgent(self.store, model).reply("puppy", "开激光", "child", "story")
        self.assertEqual(result["intent"], "chat")
        self.assertIsNone(self.store.story("puppy", "child"))

    def test_forget_person_clears_persistent_story_and_stale_writer_is_rejected(self):
        self.store.save_story("puppy", "child", "探险", "小兔子在门口。", 0)
        self.store.forget_person("puppy", "child")
        self.assertIsNone(self.store.story("puppy", "child"))
        with self.assertRaises(ValueError):
            self.store.save_story("puppy", "child", "探险", "打开门。", 1)


if __name__ == "__main__":
    unittest.main()
