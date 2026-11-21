"""Completed model replies must fit intact before any operation is committed.

Only fake stop results and temporary SQLite; no model construction or device I/O.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .agent import LocalModel, ModelResponseError, PetAgent
from .store import Store


LONG_STORY = (
    "小兔子把红球踢进了草丛里，急得转来转去。小恐龙先请它坐在大树旁休息，再沿着小路仔细找。"
    "它们翻开宽大的树叶，看过低低的花丛，又问了正在采蜜的小蜜蜂。小蜜蜂指向河边的一块石头，"
    "小恐龙绕到石头后面，终于看见红球安安静静地躺在那里。它用前爪轻轻把球拨出来，交还给小兔子。"
    "小兔子抱紧红球，开心地说谢谢，两个好朋友一起回家了。"
)
LONG_ENGLISH = (
    "我们可以先看看小动物的图片，想一想它会怎样走路，再用轻轻的声音学一个英文词。"
    "不用着急，也不用一下记住很多词，你可以先听一听，再选择要不要跟着说。"
    "如果暂时不想开口，也可以指一指图片，或者和小狗一起做一个找图片的小小游戏。"
    "今天我们只认识一个最熟悉的小动物，家里有时会见到它，它喜欢摇尾巴。小狗的英文是dog，一起说dog吧。"
)


def answer(reply, intent="chat", **fields):
    return {"reply": reply, "intent": intent, "value": "", "action": "", **fields}


class ReplyBoundaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="puppy-reply-boundary-")
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "memory.sqlite3")
        self.store.pet("puppy")

    def snapshot(self):
        with self.store.connect() as db:
            return {name: [tuple(row) for row in db.execute(f"SELECT * FROM {name} ORDER BY rowid")]
                    for name in ("pets", "memories", "conversations", "english_progress", "stories", "actions", "routines")}

    def model(self, outputs, before_second=None):
        pending = iter(outputs)
        calls = []
        def generate(system, user):
            if calls and before_second:
                before_second()
            calls.append((system, user))
            raw = json.dumps(next(pending), ensure_ascii=False)
            # A full JSON response with normal EOS can still exceed the UI limit.
            return LocalModel._completed_text(SimpleNamespace(text=raw, finish_reason="stop"))
        return generate, calls

    def test_short_and_exact_boundary_keep_the_complete_stripped_reply(self):
        for reply in ("小兔子把红球还给了朋友。", "字" * 119 + "。"):
            with self.subTest(length=len(reply)):
                model, calls = self.model([answer("  " + reply + "\n")])
                result = PetAgent(self.store, model).reply("puppy", "你好", "child")
                self.assertEqual(result["reply"], reply)
                self.assertEqual(len(calls), 1)

    def test_long_completed_story_retries_and_only_saves_the_complete_short_story(self):
        self.assertGreater(len(LONG_STORY), 120)
        short = "小恐龙在河边找到红球，轻轻拨出草丛，交还给小兔子。小兔子抱住球说谢谢，朋友们一起回家。"
        before = self.snapshot()
        model, calls = self.model([
            answer(LONG_STORY, story_title="找回红球"),
            answer(short, story_title="找回红球"),
        ], before_second=lambda: self.assertEqual(self.snapshot(), before))
        result = PetAgent(self.store, model).reply("puppy", "讲找回球的故事", "child", "story", "session")
        self.assertEqual(len(calls), 2)
        self.assertIn("120字以内的完整短答", calls[1][1])
        self.assertEqual(result["reply"], short)
        self.assertEqual(result["story"]["chapter"], 1)
        self.assertEqual(self.store.story("puppy", "child")["summary"], PetAgent.STORY_EXCERPT_PREFIX + short)
        self.assertEqual(self.store.conversation("puppy", "child", "story", "session")[-1]["assistant"], short)
        self.assertNotIn(LONG_STORY[:120], result["story_summary"])

    def test_two_long_stories_do_not_advance_existing_chapter_or_history(self):
        self.store.save_story("puppy", "child", "原故事", "小兔子在门口等朋友。", 0)
        self.store.append_conversation("puppy", "child", "开始故事", "小兔子在门口等朋友。", "story", "session")
        before = self.snapshot()
        model, calls = self.model([answer(LONG_STORY, story_title="新标题")] * 2)
        with self.assertRaises(ModelResponseError):
            PetAgent(self.store, model).reply("puppy", "继续", "child", "story", "session")
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.snapshot(), before)

    def test_two_long_replies_do_not_commit_memory_name_or_return_action(self):
        self.store.remember_preference("puppy", "child", "恐龙")
        self.store.propose_name("puppy", "团团", "child", "session")
        variants = [
            ("我喜欢海豚，请记住", answer(LONG_STORY, "remember_preference", value="海豚")),
            ("忘记恐龙", answer(LONG_STORY, "forget_memory", value="恐龙")),
            ("以后你叫豆豆", answer(LONG_STORY, "rename_pet", value="豆豆")),
            ("好", answer(LONG_STORY)),
            ("请挥手", answer(LONG_STORY, "action", action="wave")),
        ]
        for text, payload in variants:
            with self.subTest(text=text):
                before = self.snapshot()
                model, calls = self.model([payload, payload])
                with self.assertRaises(ModelResponseError):
                    PetAgent(self.store, model).reply("puppy", text, "child", session_id="session")
                self.assertEqual(len(calls), 2)
                self.assertEqual(self.snapshot(), before)

    def test_long_memory_or_action_first_draft_is_not_submitted_when_retry_is_chat(self):
        for intent, fields, text in (("remember_preference", {"value": "海豚"}, "我喜欢海豚"),
                                    ("action", {"action": "wave"}, "请挥手")):
            with self.subTest(intent=intent):
                before = self.snapshot()
                model, calls = self.model([answer(LONG_STORY, intent, **fields), answer("我们先聊一聊吧。")],
                    before_second=lambda: self.assertEqual(self.snapshot(), before))
                result = PetAgent(self.store, model).reply("puppy", text, "child")
                self.assertEqual(len(calls), 2)
                self.assertEqual((result["intent"], result["action"]), ("chat", ""))
                self.assertEqual(self.store.memory_entries("puppy", "child"), [])
                self.assertEqual(self.snapshot()["actions"], before["actions"])

    def test_length_check_precedes_existing_operation_confirmation_calls(self):
        model, calls = self.model([
            answer(LONG_STORY, "action_sequence", actions=["wave", "sit"]),
            answer("我们先聊一聊吧。"),
        ])
        result = PetAgent(self.store, model).reply("puppy", "请挥手再坐下", "child")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(len(calls), 2)  # No operation-verification call for an invalid long draft.

    def test_english_long_reply_retries_without_cutting_target_or_saving_partial_text(self):
        self.assertNotIn("dog", LONG_ENGLISH[:120])
        short = "小狗的英文是dog，我们一起说dog吧。"
        before = self.snapshot()
        model, calls = self.model([answer(LONG_ENGLISH), answer(short)],
            before_second=lambda: self.assertEqual(self.snapshot(), before))
        result = PetAgent(self.store, model).reply("puppy", "教我小狗的英文", "child", "english", "session")
        self.assertEqual(result["reply"], short)
        self.assertEqual(len(calls), 2)
        self.assertIn("120字以内的完整短答", calls[1][1])
        self.assertEqual(result["lesson"]["target_word"], "dog")
        progress = self.store.english_progress("puppy", "child")
        self.assertEqual([(row["word"], row["presentations"]) for row in progress], [("dog", 1)])
        self.assertEqual(self.store.conversation("puppy", "child", "english", "session")[-1]["assistant"], short)

    def test_english_even_visible_target_does_not_make_a_cut_sentence_acceptable(self):
        long = "小狗的英文是dog。" + LONG_ENGLISH
        model, calls = self.model([answer(long), answer(long)])
        before = self.snapshot()
        with self.assertRaises(ModelResponseError):
            PetAgent(self.store, model).reply("puppy", "教我小狗的英文", "child", "english", "session")
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.snapshot(), before)

    def test_farewell_retries_long_reply_and_keeps_the_complete_short_sentence(self):
        short = "今天先到这里，我们休息一下，明天再玩英语吧。"
        model, calls = self.model([answer(LONG_ENGLISH), answer(short)])
        before = self.snapshot()
        result = PetAgent(self.store, model).finish_learning({"reason": "daily_limit"})
        self.assertEqual(result["reply"], short)
        self.assertTrue(result["learning_closed"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.snapshot(), before)

    def test_farewell_two_invalid_replies_raise_typed_error_without_any_writes(self):
        for payload in (answer(LONG_ENGLISH), answer("string"), {"reply": None}):
            with self.subTest(payload=payload):
                model, calls = self.model([payload, payload])
                before = self.snapshot()
                with self.assertRaises(ModelResponseError):
                    PetAgent(self.store, model).finish_learning({"reason": "daily_limit"})
                self.assertEqual(len(calls), 2)
                self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
