"""Verbatim statement persistence in temporary SQLite databases; no inference."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from companion.store import Store
from companion.turn_commit import TurnCancelled, TurnCommit


class MemoryStatementTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="puppy-memory-statements-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.sqlite3"
        self.store = Store(self.path)

    def record(self, text, device="puppy", person="child"):
        return self.store.remember_preference_statement(device, person, text)

    def snapshot(self):
        with self.store.connect() as db:
            return {table: [tuple(row) for row in db.execute("SELECT * FROM " + table + " ORDER BY rowid")]
                    for table in ("memories", "conversations", "english_progress", "stories")}

    def test_300_character_statement_is_saved_without_truncation(self):
        text = "我只在下雨时喜欢看窗外，" + "呀" * (300 - len("我只在下雨时喜欢看窗外，"))
        key = self.record(text)
        self.assertEqual(self.store.memories("puppy", "child"), {key: text})
        self.assertEqual(key, "preference_statement:" + hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertLessEqual(len(key), 96)
        self.assertEqual(Store(self.path).memory_entries("puppy", "child")[0]["value"], text)

    def test_invalid_text_and_unconfirmed_owner_leave_database_unchanged(self):
        self.record("已有原话。")
        before = self.snapshot()
        for text in (None, b"text", 1, True, [], {}, "", " \t\n", "字" * 301):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.record(text)
                self.assertEqual(self.snapshot(), before)
        for device, person in (("puppy", None), ("puppy", ""), ("puppy", 1),
                               (None, "child"), ("", "child"), (1, "child")):
            with self.subTest(device=device, person=person):
                with self.assertRaises(ValueError):
                    self.record("我喜欢画画。", device, person)
                self.assertEqual(self.snapshot(), before)

    def test_case_whitespace_negation_and_conditions_are_preserved_exactly(self):
        statements = ("  我不喜欢跑步，但下雨时喜欢听雨声。\n",
                      "我只在公园喜欢骑车。", "I like DOG, not cats.\t",
                      "I like dog, not cats.\t", "I like DOG, not cats.",
                      "第一行\n第二行  中间两个空格")
        records = {self.record(text): text for text in statements}
        self.assertEqual(len(records), len(statements))
        self.assertEqual(self.store.memories("puppy", "child"), records)

    def test_same_statement_is_idempotent_without_overwriting_other_records(self):
        with patch("companion.store.time.time", return_value=100):
            key = self.record("我喜欢拼图。")
            other = self.record("我喜欢画画。")
        before = self.store.memory_entries("puppy", "child")
        with patch("companion.store.time.time", return_value=200):
            self.assertEqual(self.record("我喜欢拼图。"), key)
        self.assertEqual(self.store.memory_entries("puppy", "child"), before)
        variant = self.record("我喜欢拼图！")
        self.assertNotEqual(key, variant)
        self.assertEqual(self.store.memories("puppy", "child"),
                         {key: "我喜欢拼图。", other: "我喜欢画画。", variant: "我喜欢拼图！"})

    def test_legacy_limits_normalization_and_keys_remain_compatible(self):
        legacy = self.store.remember_preference("puppy", "child", "  DOG\n  blocks  ")
        self.assertTrue(legacy.startswith("preference:"))
        self.assertEqual(self.store.memories("puppy", "child")[legacy], "DOG blocks")
        self.assertEqual(self.store.remember_preference("puppy", "child", "dog blocks"), legacy)
        original = self.record("dog blocks")
        self.assertNotEqual(original, legacy)
        long_legacy = self.store.remember_preference("puppy", "child", "好" * 81)
        self.assertEqual(self.store.memories("puppy", "child")[long_legacy], "好" * 80)
        self.store.remember("puppy", "child", "legacy-240", "字" * 240)
        with self.assertRaises(ValueError):
            self.store.remember("puppy", "child", "legacy-too-long", "字" * 241)
        self.assertEqual(self.store.memories("puppy", "child")[original], "dog blocks")
        self.assertEqual(self.store.memories("puppy", "child")[legacy], "dog blocks")

    def test_same_text_in_different_members_and_devices_stays_isolated(self):
        text = "请记住，我喜欢搭高高的积木。"
        key = self.record(text)
        self.assertEqual(self.record(text, person="other-child"), key)
        self.assertEqual(self.record(text, device="other-puppy"), key)
        self.assertEqual(self.store.memory_entries("puppy", None), [])
        self.assertEqual(self.store.memories("puppy", None), {})
        self.store.forget("puppy", "child", key)
        self.assertEqual(self.store.memories("puppy", "child"), {})
        self.assertEqual(self.store.memories("puppy", "other-child"), {key: text})
        self.assertEqual(self.store.memories("other-puppy", "child"), {key: text})

    def test_reopened_store_deletes_one_whole_statement_and_own_history_only(self):
        first = self.record("我喜欢纸飞机，也喜欢小船。")
        second = self.record("我喜欢画太阳。")
        legacy = self.store.remember_preference("puppy", "child", "画画")
        for device, person in (("puppy", "child"), ("puppy", "other-child"),
                               ("other-puppy", "child")):
            for mode in ("chat", "story"):
                self.store.append_conversation(device, person, "原来的对话", "回应", mode, "session")
        self.store.record_english("puppy", "child", "dog", True)
        self.store.save_story("puppy", "child", "纸船", "小船来到岸边。", 0)
        self.store = Store(self.path)
        progress = self.store.english_progress("puppy", "child")
        story = self.store.story("puppy", "child")
        self.store.forget("puppy", "child", first)
        self.assertEqual(self.store.memories("puppy", "child"),
                         {second: "我喜欢画太阳。", legacy: "画画"})
        for mode in ("chat", "story"):
            self.assertEqual(self.store.conversation("puppy", "child", mode, "session"), [])
            self.assertEqual(len(self.store.conversation("puppy", "other-child", mode, "session")), 1)
            self.assertEqual(len(self.store.conversation("other-puppy", "child", mode, "session")), 1)
        self.assertEqual(self.store.english_progress("puppy", "child"), progress)
        self.assertEqual(self.store.story("puppy", "child"), story)

    def test_forget_person_removes_new_and_legacy_memories_only_in_its_scope(self):
        key = self.record("我喜欢坐小火车。")
        self.store.remember_preference("puppy", "child", "火车")
        self.record("我喜欢坐小火车。", person="other-child")
        self.record("我喜欢坐小火车。", device="other-puppy")
        self.store.append_conversation("puppy", "child", "旧话", "回应")
        self.store.record_english("puppy", "child", "dog", False)
        self.store.save_story("puppy", "child", "小火车", "小火车到了花园。", 0)
        self.store.forget_person("puppy", "child")
        self.assertEqual(self.store.memory_entries("puppy", "child"), [])
        self.assertEqual(self.store.conversation("puppy", "child"), [])
        self.assertEqual(self.store.english_progress("puppy", "child"), [])
        self.assertIsNone(self.store.story("puppy", "child"))
        self.assertEqual(self.store.memories("puppy", "other-child"), {key: "我喜欢坐小火车。"})
        self.assertEqual(self.store.memories("other-puppy", "child"), {key: "我喜欢坐小火车。"})

    def test_outer_transaction_rolls_back_statement_and_history_together(self):
        self.record("已经提交的原话。")
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "later write failed"):
            with self.store.commit_scope(TurnCommit()):
                self.store.begin_commit()
                self.record("本轮新原话，不能残留。")
                self.store.append_conversation("puppy", "child", "本轮新原话，不能残留。", "记下了")
                raise RuntimeError("later write failed")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(Store(self.path).memories("puppy", "child"), self.store.memories("puppy", "child"))

    def test_cancelled_outer_transaction_cannot_commit_but_success_is_durable(self):
        before = self.snapshot()
        guard = TurnCommit()
        with self.assertRaises(TurnCancelled):
            with self.store.commit_scope(guard):
                self.store.begin_commit()
                self.record("取消这轮后不能保存。")
                guard.cancel()
        self.assertEqual(self.snapshot(), before)
        successful = TurnCommit()
        with self.store.commit_scope(successful):
            self.store.begin_commit()
            key = self.record("这轮已经成功保存。")
        self.assertFalse(successful.cancel())
        self.assertEqual(Store(self.path).memories("puppy", "child"), {key: "这轮已经成功保存。"})


if __name__ == "__main__":
    unittest.main()
