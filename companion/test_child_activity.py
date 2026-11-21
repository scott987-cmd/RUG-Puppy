import unittest

from .child_activity import ChildActivitySessions, requests_english, requests_chat, stops_activity


class ChildActivityTests(unittest.TestCase):
    def test_child_can_start_pause_resume_and_leave_english(self):
        sessions = ChildActivitySessions()
        key = ("puppy", "voice-session")
        self.assertEqual(sessions.route(*key, "教我英语", "chat"), "english")
        self.assertEqual(sessions.route(*key, "叫我一个英文单词。", "chat"), "english")
        self.assertEqual(sessions.route(*key, "叫我一個英文單詞", "chat"), "english")
        self.assertEqual(sessions.route(*key, "继续", "chat"), "english")
        self.assertEqual(sessions.route(*key, "不玩了", "chat"), "chat")
        self.assertEqual(sessions.route(*key, "继续", "chat"), "english")
        self.assertEqual(sessions.route(*key, "陪我聊天吧", "chat"), "chat")
        self.assertEqual(sessions.route(*key, "继续", "chat"), "chat")

    def test_guest_lesson_is_session_local_and_expires(self):
        now = [100.0]
        sessions = ChildActivitySessions(clock=lambda: now[0])
        sessions.record_lesson("puppy", "one", None, "dog", "introduce")
        self.assertEqual(sessions.lesson("puppy", "one", None), {"word": "dog", "stage": "repeat"})
        self.assertIsNone(sessions.lesson("puppy", "two", None))
        self.assertIsNone(sessions.lesson("puppy", "one", "child"))
        now[0] += ChildActivitySessions.TTL_SECONDS + 1
        self.assertIsNone(sessions.lesson("puppy", "one", None))

    def test_routing_does_not_mistake_questions_or_quoted_words(self):
        for text in ("英语有什么好处？", "你能教我英语吗？", "不要学英语", "叫我团团"):
            self.assertFalse(requests_english(text))
        self.assertTrue(requests_english("小狗，我们一起学英语吧"))
        self.assertTrue(requests_english("玩英语游戏"))
        self.assertTrue(requests_english("教我一个英文单词。"))
        self.assertTrue(requests_english("叫我一個英文單詞"))
        self.assertTrue(requests_chat("聊天吧"))
        self.assertTrue(stops_activity("先不学英语了"))
        self.assertTrue(stops_activity("不讲故事了"))
        self.assertFalse(stops_activity("故事里小熊说不玩了"))


if __name__ == "__main__":
    unittest.main()
