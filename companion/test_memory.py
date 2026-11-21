import json
import sqlite3
import stat
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from companion.agent import PetAgent
from companion.store import Store


def _new_operation_guard(system, user):
    """Fixture-only consent for newly added guards; main replies keep their sequence."""
    if system.startswith("你只核验") and json.loads(user).get("operation") in {
            "action", "rename_pet", "remember_preference", "forget_memory"}:
        return '{"explicit":true}'
    return None


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "pet.db"
        self.store = Store(self.database)

    def tearDown(self):
        self.temp.cleanup()

    def test_database_permissions_do_not_change_existing_parent(self):
        existing = Path(self.temp.name) / "existing"
        existing.mkdir(mode=0o755)
        existing.chmod(0o755)
        database = existing / "profile.sqlite3"
        database.touch(mode=0o644)
        database.chmod(0o644)
        Store(database)
        self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(existing.stat().st_mode), 0o755)
        fresh = existing / "private"
        Store(fresh / "profile.sqlite3")
        self.assertEqual(stat.S_IMODE(fresh.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((fresh / "profile.sqlite3").stat().st_mode), 0o600)

    def test_name_confirmation_is_bound_to_member_session_and_expiry(self):
        with patch("companion.store.time.time", return_value=1000):
            self.store.propose_name("puppy", "豆豆", "child-1", "session-a")
        for person, session in (("child-2", "session-a"), ("child-1", "session-b"), (None, "session-a")):
            with patch("companion.store.time.time", return_value=1001):
                with self.assertRaises(ValueError):
                    self.store.confirm_name("puppy", True, person, session)
        with patch("companion.store.time.time", return_value=1120):
            with self.assertRaises(ValueError):
                self.store.confirm_name("puppy", True, "child-1", "session-a")
            self.assertIsNone(self.store.pending_name("puppy", "child-1", "session-a"))
            self.assertEqual(self.store.pet("puppy")["name"], "小狗")

    def test_anonymous_rename_requires_isolated_session(self):
        answers = iter([
            '{"reply":"好的","intent":"rename_pet","value":"豆豆"}',
            '{"reply":"好的","intent":"rename_pet","value":"豆豆"}',
            '{"reply":"我在听","intent":"chat"}',
            '{"reply":"我是豆豆啦","intent":"chat"}',
        ])
        agent = PetAgent(self.store, lambda system, user: _new_operation_guard(system, user) or next(answers))
        agent.reply("puppy", "以后叫豆豆")
        self.assertIsNone(self.store.pet("puppy")["pending_name"])
        agent.reply("puppy", "以后叫豆豆", session_id="session-a")
        agent.reply("puppy", "好", session_id="session-b")
        self.assertEqual(self.store.pet("puppy")["name"], "小狗")
        agent.reply("puppy", "好", session_id="session-a")
        self.assertEqual(self.store.pet("puppy")["name"], "豆豆")
        self.assertEqual(self.store.conversation("puppy", None, session_id="session-a"), [])

    def test_old_database_keeps_saved_name_but_drops_unscoped_proposal(self):
        legacy = Path(self.temp.name) / "legacy.db"
        with closing(sqlite3.connect(legacy)) as db, db:
            db.execute("""CREATE TABLE pets (device_id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '小狗',
                          pending_name TEXT, updated_at REAL NOT NULL)""")
            db.execute("INSERT INTO pets VALUES ('puppy', '团团', '豆豆', 1000)")
        migrated = Store(legacy)
        self.assertEqual(migrated.pet("puppy"), {"name": "团团", "pending_name": None})
        migrated.propose_name("puppy", "花花", "child-1")
        migrated.confirm_name("puppy", True, "child-1")
        self.assertEqual(Store(legacy).pet("puppy")["name"], "花花")

    def test_failed_generation_does_not_commit_name_change(self):
        self.store.propose_name("puppy", "豆豆", "child-1")
        model = lambda system, user: "invalid model answer"
        with self.assertRaises(ValueError):
            PetAgent(self.store, model).reply("puppy", "好", "child-1")
        self.assertEqual(self.store.pet("puppy")["name"], "小狗")
        self.assertEqual(self.store.pending_name("puppy", "child-1"), "豆豆")

    def test_confirmation_cannot_apply_replaced_proposal(self):
        self.store.propose_name("puppy", "豆豆", "child-1")

        def model(system, user):
            self.store.propose_name("puppy", "团团", "child-1")
            return '{"reply":"我是豆豆啦","intent":"chat"}'

        with self.assertRaises(ValueError):
            PetAgent(self.store, model).reply("puppy", "好", "child-1")
        self.assertEqual(self.store.pet("puppy")["name"], "小狗")
        self.assertEqual(self.store.pending_name("puppy", "child-1"), "团团")

    def test_multiple_preferences_are_deduplicated_and_member_scoped(self):
        values = iter(["恐龙", "熊猫", "恐龙"])

        def model(system, user):
            if reply := _new_operation_guard(system, user):
                return reply
            return json.dumps({"reply": "好的", "intent": "remember_preference", "value": next(values)})

        agent = PetAgent(self.store, model)
        for text in ("我喜欢恐龙", "我也喜欢熊猫", "我喜欢恐龙"):
            agent.reply("puppy", text, "child-1")
        memories = Store(self.database).memories("puppy", "child-1")
        self.assertEqual(set(memories.values()), {"我喜欢恐龙", "我也喜欢熊猫"})
        self.assertEqual(len(memories), 2)
        self.assertEqual(self.store.memory_entries("puppy", "child-2"), [])
        self.assertEqual(self.store.memory_entries("other-puppy", "child-1"), [])
        self.assertEqual(self.store.memory_entries("puppy", None), [])

    def test_conversation_survives_restart_but_is_scoped_bounded_and_expires(self):
        with patch("companion.store.time.time", return_value=1000):
            for index in range(8):
                self.store.append_conversation("puppy", "child-1", f"word-{index}", "嗯", "story", "s1")
            new_store = Store(self.database)
            turns = new_store.conversation("puppy", "child-1", "story", "s1")
            self.assertEqual(len(turns), 6)
            self.assertEqual(turns[0]["user"], "word-2")
            self.assertEqual(new_store.conversation("puppy", "child-2", "story", "s1"), [])
            self.assertEqual(new_store.conversation("puppy", "child-1", "chat", "s1"), [])
            self.assertEqual(new_store.conversation("puppy", "child-1", "story", "s2"), [])
            self.assertEqual(new_store.conversation("puppy", None, "story", "s1"), [])
        with patch("companion.store.time.time", return_value=2800):
            self.assertEqual(self.store.conversation("puppy", "child-1", "story", "s1"), [])

    def test_agent_gets_prior_turns_after_restart_and_deletion_clears_them(self):
        first = PetAgent(self.store, lambda system, user: '{"reply":"恐龙找到了一个蛋","intent":"chat","story_title":"恐龙的蛋","story_summary":"恐龙找到蛋，还不知道里面是什么。"}')
        first.reply("puppy", "开始讲恐龙故事", "child-1", "story")
        seen = []

        def model(system, user):
            seen.append(system)
            return '{"reply":"蛋里有一只小恐龙","intent":"chat","story_title":"恐龙的蛋","story_summary":"蛋破开了，里面有一只小恐龙。"}'

        PetAgent(Store(self.database), model).reply("puppy", "然后呢", "child-1", "story")
        self.assertIn("恐龙找到了一个蛋", seen[-1])
        key = self.store.remember_preference("puppy", "child-1", "恐龙")
        self.store.forget("puppy", "child-1", key)
        self.assertEqual(self.store.memories("puppy", "child-1"), {})
        self.assertEqual(self.store.conversation("puppy", "child-1", "story"), [])

    def test_forget_intent_cannot_delete_another_members_memory(self):
        key = self.store.remember_preference("puppy", "child-1", "恐龙")
        self.store.append_conversation("puppy", "child-1", "喜欢恐龙", "记住了")
        model = lambda system, user: _new_operation_guard(system, user) or json.dumps({"reply": "忘掉了", "intent": "forget_memory", "memory_key": key})
        agent = PetAgent(self.store, model)
        agent.reply("puppy", "忘掉这个", "child-2")
        self.assertEqual(self.store.memories("puppy", "child-1")[key], "恐龙")
        agent.reply("puppy", "忘掉我喜欢恐龙", "child-1")
        self.assertEqual(self.store.memories("puppy", "child-1"), {})
        self.assertEqual(self.store.conversation("puppy", "child-1"), [])

    def test_forget_resolves_unique_full_value_when_model_omits_key(self):
        self.store.remember_preference("puppy", "child-1", "恐龙")
        remaining = self.store.remember_preference("puppy", "child-1", "熊猫")
        model = lambda system, user: _new_operation_guard(system, user) or '{"reply":"好的","intent":"forget_memory","value":"恐龙"}'
        answer = PetAgent(self.store, model).reply("puppy", "忘掉我喜欢恐龙", "child-1")
        self.assertEqual(answer["intent"], "forget_memory")
        self.assertEqual(self.store.memories("puppy", "child-1"), {remaining: "熊猫"})

    def test_forget_does_not_guess_partial_or_ambiguous_values(self):
        self.store.remember("puppy", "child-1", "preference:a", "恐龙绘本")
        self.store.remember("puppy", "child-1", "preference:b", "恐龙绘本")
        for value in ("恐龙", "恐龙绘本"):
            model = lambda system, user: _new_operation_guard(system, user) or json.dumps({"reply":"好的", "intent":"forget_memory", "value":value})
            answer = PetAgent(self.store, model).reply("puppy", "忘掉这个", "child-1")
            self.assertEqual(answer["intent"], "chat")
            self.assertEqual(len(self.store.memories("puppy", "child-1")), 2)

    def test_english_progress_requires_target_and_does_not_claim_pronunciation(self):
        seen = []

        def model(system, user):
            seen.append(system)
            word = "cat" if "本轮英语目标：cat" in system else "dog"
            return json.dumps({"reply": f"一起说 {word}，好吗？", "intent": "chat"})

        agent = PetAgent(self.store, model)
        answer = agent.reply("puppy", "一起学英语", "child-1", "english")
        self.assertEqual(answer["lesson"]["target_word"], "dog")
        agent.reply("puppy", "hotdog", "child-1", "english")
        self.assertEqual(self.store.english_progress("puppy", "child-1")[0]["text_matches"], 0)
        agent.reply("puppy", "dog", "child-1", "english")
        answer = PetAgent(Store(self.database), model).reply("puppy", "下一个", "child-1", "english")
        self.assertEqual(answer["lesson"]["target_word"], "cat")
        progress = {row["word"]: row for row in self.store.english_progress("puppy", "child-1")}
        self.assertEqual(progress["dog"]["presentations"], 3)
        self.assertEqual(progress["dog"]["text_matches"], 1)
        self.assertNotIn("mastered", progress["dog"])
        self.assertIn("不能证明发音正确", seen[0])
        self.assertEqual(self.store.english_progress("puppy", "child-2"), [])

    def test_invalid_english_model_response_never_records_progress(self):
        model = lambda system, user: '{"reply":"跟我学英语","intent":"chat"}'
        with self.assertRaises(ValueError):
            PetAgent(self.store, model).reply("puppy", "一起学习", "child-1", "english")
        self.assertEqual(self.store.english_progress("puppy", "child-1"), [])

    def test_english_retries_answer_without_chinese_guidance(self):
        outputs = iter(['{"reply":"Good dog!","intent":"chat"}',
                        '{"reply":"小狗的英文是 dog，一起说 dog。","intent":"chat"}'])
        answer = PetAgent(self.store, lambda system, user: next(outputs)).reply(
            "puppy", "学英文", "child-1", "english")
        self.assertIn("小狗", answer["reply"])
        self.assertEqual(self.store.english_progress("puppy", "child-1")[0]["presentations"], 1)

    def test_unknown_speaker_has_no_personal_history_or_learning_record(self):
        self.store.append_conversation("puppy", "child-1", "私人故事", "私人细节")
        seen = []

        def model(system, user):
            seen.append(system)
            return '{"reply":"试着说 dog","intent":"chat"}'

        PetAgent(self.store, model).reply("puppy", "dog", mode="english", session_id="anonymous")
        self.assertNotIn("私人", seen[0])
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM english_progress").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0], 1)

    def test_forget_person_clears_profile_context_learning_and_pending_change(self):
        for person in ("child-1", "child-2"):
            self.store.remember_preference("puppy", person, "恐龙")
            self.store.append_conversation("puppy", person, "dog", "dog")
            self.store.record_english("puppy", person, "dog", True)
        self.store.propose_name("puppy", "豆豆", "child-1")
        self.store.forget_person("puppy", "child-1")
        self.assertEqual(self.store.memories("puppy", "child-1"), {})
        self.assertEqual(self.store.conversation("puppy", "child-1"), [])
        self.assertEqual(self.store.english_progress("puppy", "child-1"), [])
        self.assertIsNone(self.store.pet("puppy")["pending_name"])
        self.assertTrue(self.store.memories("puppy", "child-2"))
        self.assertTrue(self.store.english_progress("puppy", "child-2"))


if __name__ == "__main__":
    unittest.main()
