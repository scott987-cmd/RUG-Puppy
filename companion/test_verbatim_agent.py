"""Persistent original statements and reference boundaries with scripted models.

The explicit decisions are fixtures, not evidence of model understanding. There
is no audio, device, identity engine or production database in these tests.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from companion.agent import PetAgent
from companion.store import Store
from companion.test_operation_replies import ScriptedModel, answer
from companion.turn_commit import TurnCancelled, TurnCommit


class VerbatimAgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="puppy-verbatim-agent-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.sqlite3"
        self.store = Store(self.path)

    def reply(self, model, text, person="child", session="first", **kwargs):
        return PetAgent(self.store, model).reply(
            "puppy", text, person, session_id=session, **kwargs)

    def test_model_shortened_or_empty_value_cannot_change_the_saved_utterance(self):
        text = "  记得我喜欢在晴天的阳台上，看小风车慢慢转；大风时我不喜欢。\n"
        for label in ("", "喜欢看风车", "喜欢大风", "另一个孩子喜欢恐龙"):
            with self.subTest(label=label):
                model = ScriptedModel(answer("好的。", "remember_preference", value=label),
                                      {"explicit": True})
                result = self.reply(model, text)
                self.assertEqual(result["value"], text)
                self.assertEqual(result["reply_source"], "business")
                self.assertLessEqual(len(result["reply"]), 120)
                if label:
                    self.assertNotIn(label, result["reply"])
                self.assertEqual(self.store.memories("puppy", "child"), {result["memory_key"]: text})
                context = json.loads(model.calls[1][2])
                self.assertEqual(context["current_utterance"], text)
                self.assertEqual(context["context"]["value"], text)
                self.assertEqual(len(model.calls), 2)

    def test_full_300_character_record_survives_without_a_long_spoken_receipt(self):
        text = "我喜欢" + "蓝色小风车" * 58 + "在晴天。\n  "
        text = (text + "！" * 300)[:300]
        model = ScriptedModel(answer("好的。", "remember_preference"), {"explicit": True})
        result = self.reply(model, text)
        self.assertEqual(self.store.memories("puppy", "child")[result["memory_key"]], text)
        self.assertEqual(len(result["value"]), 300)
        self.assertLessEqual(len(result["reply"]), 120)
        with self.assertRaises(ValueError):
            self.reply(ScriptedModel(), text + "！")

    def test_restart_other_session_and_expired_history_keep_the_exact_memory_in_both_prompts(self):
        original = "我只喜欢在傍晚和爸爸一起折蓝色纸星星，请记住。"
        with patch("companion.store.time.time", return_value=1000):
            saved = self.reply(ScriptedModel(answer("好的。", "remember_preference"),
                                             {"explicit": True}), original)
        self.store = Store(self.path)
        for session in ("first", "another-session"):
            with self.subTest(session=session), patch("companion.store.time.time", return_value=2801):
                # Force the text-only recovery path as well as the main prompt.
                model = ScriptedModel(answer("我来坐下。", "action", action="sit"),
                                      {"explicit": False}, answer("你喜欢在傍晚和爸爸折蓝色纸星星。"))
                self.reply(model, "你记得我什么时候喜欢和谁一起折什么吗？", session=session)
                for index in (0, 2):
                    self.assertIn(original, model.calls[index][1])
                    self.assertIn(saved["memory_key"], model.calls[index][1])
                self.assertIn('"history": []', model.calls[2][1])

    def test_conflicting_or_foreign_reference_cannot_fall_back_to_a_different_row(self):
        first = self.store.remember_preference_statement("puppy", "child", "我喜欢纸星星。")
        self.store.remember_preference_statement("puppy", "child", "我喜欢木陀螺。")
        other = self.store.remember_preference_statement("puppy", "other-child", "我喜欢布小熊。")
        before = self.store.memories("puppy", "child")
        for key, value in ((first, "我喜欢木陀螺。"), (other, "我喜欢纸星星。"),
                           ("missing", "我喜欢纸星星。"), ([], "我喜欢纸星星。")):
            with self.subTest(key=key):
                model = ScriptedModel(answer("好的。", "forget_memory", memory_key=key, value=value),
                                      answer("请说明要忘记哪一条。"))
                result = self.reply(model, "请忘记纸星星那条。")
                self.assertEqual(result["intent"], "chat")
                self.assertEqual(len(model.calls), 2)
                self.assertEqual(self.store.memories("puppy", "child"), before)
        self.assertEqual(self.store.memories("puppy", "other-child"), {other: "我喜欢布小熊。"})

    def test_ambiguous_or_partial_record_request_reaches_complete_record_context_and_cannot_delete(self):
        first = self.store.remember_preference_statement("puppy", "child", "我喜欢白纸星星，也喜欢红纸船。")
        second = self.store.remember_preference_statement("puppy", "child", "我喜欢用蓝纸折星星。")
        before = self.store.memories("puppy", "child")
        for text in ("忘记我喜欢折星星。", "只删白纸星星，保留红纸船。"):
            with self.subTest(text=text):
                model = ScriptedModel(answer("好的。", "forget_memory", memory_key=first),
                                      {"explicit": False}, answer("你想忘记哪一整条记录？"))
                self.reply(model, text)
                context = json.loads(model.calls[1][2])["context"]
                self.assertEqual(context["memory_key"], first)
                self.assertEqual({row["memory_key"]: row["value"] for row in context["records"]}, before)
                self.assertIn(second, {row["memory_key"] for row in context["records"]})
                self.assertEqual(self.store.memories("puppy", "child"), before)

    def test_delete_entire_record_leaves_no_original_in_history_after_restart(self):
        removed = "我喜欢白纸星星，也喜欢红纸船。"
        kept = "我喜欢拼木头房子。"
        first = self.reply(ScriptedModel(answer("好的。", "remember_preference"), {"explicit": True}), removed)
        second = self.reply(ScriptedModel(answer("好的。", "remember_preference"), {"explicit": True}), kept)
        self.reply(ScriptedModel(answer("好的。", "forget_memory", memory_key=first["memory_key"]),
                                {"explicit": True}), "请删除白纸星星和红纸船那整条记录。")
        self.store = Store(self.path)
        self.assertEqual(self.store.memories("puppy", "child"), {second["memory_key"]: kept})
        self.assertEqual(self.store.conversation("puppy", "child", session_id="first"), [])
        model = ScriptedModel(answer("你还喜欢拼木头房子。"))
        self.reply(model, "还记得哪条喜好吗？", session="later")
        self.assertNotIn(removed, model.calls[0][1])
        self.assertIn(kept, model.calls[0][1])

    def test_exact_whole_record_quote_recovers_missing_model_key(self):
        removed = "请记住：晴天我喜欢看橙色缆车；周六我喜欢画绿色潜水艇。"
        kept = "我喜欢在雨天看蓝色缆车。"
        first = self.store.remember_preference_statement("puppy", "child", removed)
        second = self.store.remember_preference_statement("puppy", "child", kept)
        text = f"麻烦把这一整条记录删掉，里面两项都不要保留：“{removed}”"
        model = ScriptedModel(answer("好的。", "forget_memory"), {"explicit": True})
        result = self.reply(model, text)
        self.assertEqual(result["intent"], "forget_memory")
        self.assertEqual(result["memory_key"], first)
        self.assertEqual(self.store.memories("puppy", "child"), {second: kept})
        self.assertEqual(json.loads(model.calls[1][2])["context"]["memory_key"], first)
        self.assertEqual(len(model.calls), 2)

    def test_quote_fallback_never_selects_partial_negative_or_conflicting_target(self):
        original = "我喜欢白纸星星，也喜欢红纸船。"
        key = self.store.remember_preference_statement("puppy", "child", original)
        before = self.store.memories("puppy", "child")
        for text, value in (("只删白纸星星，保留红纸船。", ""),
                            (f"不要整条删除：“{original}”", ""),
                            (f"这句话是什么意思：“{original}”", ""),
                            (f"请整条删除：“{original}”", "另一个喜好")):
            with self.subTest(text=text, value=value):
                model = ScriptedModel(answer("好的。", "forget_memory", value=value),
                                      answer("请说清楚要删哪一整条。"))
                result = self.reply(model, text)
                self.assertEqual(result["intent"], "chat")
                self.assertEqual(self.store.memories("puppy", "child"), before)
        self.assertIn(key, before)

    def test_identical_legacy_and_original_rows_are_not_silently_chosen(self):
        original = "我喜欢红色纸船。"
        self.store.remember_preference("puppy", "child", original)
        self.store.remember_preference_statement("puppy", "child", original)
        before = self.store.memories("puppy", "child")
        model = ScriptedModel(answer("好的。", "forget_memory"),
                              answer("请指出要删除哪一条记录。"))
        result = self.reply(model, f"请整条删除：“{original}”")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(self.store.memories("puppy", "child"), before)

    def test_delete_words_inside_saved_quote_are_not_a_new_delete_request(self):
        original = "童话里有人说：请整条删掉纸船。"
        key = self.store.remember_preference_statement("puppy", "child", original)
        model = ScriptedModel(answer("好的。", "forget_memory"),
                              answer("这句话是故事人物说的。"))
        result = self.reply(model, f"这句话是什么意思：“{original}”")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(self.store.memories("puppy", "child"), {key: original})
        self.assertEqual(len(model.calls), 2)

    def test_anonymous_other_member_and_stale_instruction_do_not_read_or_replay_original(self):
        original = "我喜欢蓝色纸船，请记住；现在先挥手。"
        self.store.remember_preference_statement("puppy", "child", original)
        for person in (None, "other-child"):
            model = ScriptedModel(answer("你好。"))
            self.reply(model, "你好", person=person)
            self.assertNotIn(original, model.calls[0][1])
        model = ScriptedModel(answer("我来挥手。", "action", action="wave"),
                              {"explicit": False}, answer("你好。"))
        result = self.reply(model, "你好")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(result["action"], "")
        self.assertEqual(json.loads(model.calls[1][2])["current_utterance"], "你好")

    def test_cancel_after_semantic_check_cannot_leave_original_or_receipt_history(self):
        guard = TurnCommit()
        def cancel():
            guard.cancel()
            return {"explicit": True}
        model = ScriptedModel(answer("好的。", "remember_preference"), cancel)
        with self.assertRaises(TurnCancelled):
            self.reply(model, "我喜欢在窗边画画，请记住。", turn_commit=guard)
        self.assertEqual(self.store.memories("puppy", "child"), {})
        self.assertEqual(self.store.conversation("puppy", "child", session_id="first"), [])


if __name__ == "__main__":
    unittest.main()
