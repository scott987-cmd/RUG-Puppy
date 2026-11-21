"""English question/practice contracts with a fake text model and temporary DB."""
import json
import tempfile
import unittest
from pathlib import Path

from companion.agent import ModelResponseError, PetAgent
from companion.store import Store


class EnglishQuestionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "pet.sqlite3")
        self.settings = {"theme": "body_parts", "save_progress": True}
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def reply(self, result, text="鼻子的英语怎么说？", person="child", **kwargs):
        def model(system, user):
            self.calls.append((system, user))
            return json.dumps(result, ensure_ascii=False)
        return PetAgent(self.store, model).reply(
            "puppy", text, person, mode=kwargs.pop("mode", "english"),
            session_id="session", learning_settings=kwargs.pop("learning_settings", self.settings), **kwargs)

    def test_question_answers_requested_word_without_curriculum_progress(self):
        for person in ("child", None):
            for record in (True, False):
                with self.subTest(person=person, record_learning=record):
                    answer = self.reply({"reply": "鼻子的英文是 nose，一起说 nose。", "intent": "chat",
                                         "english_kind": "question"}, person=person, record_learning=record)
                    self.assertEqual(answer["reply"], "鼻子的英文是 nose，一起说 nose。")
                    self.assertEqual(answer["english_kind"], "question")
                    self.assertNotIn("lesson", answer)
                    self.assertEqual(self.store.english_progress("puppy", "child"), [])
        self.assertEqual(len(self.calls), 4)

    def test_question_mentioning_curriculum_word_cannot_write_or_forge_lesson(self):
        answer = self.reply({"reply": "hand 指手，不是鼻子。鼻子的英文是 nose。", "intent": "chat",
                             "english_kind": "question", "lesson": {"target_word": "hand", "text_match": True}},
                            text="hand 是鼻子吗？")
        self.assertNotIn("lesson", answer)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_explicit_and_legacy_practice_keep_program_selected_word_and_progress(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                result = {"reply": "手的英文是 hand，一起说 hand。", "intent": "chat",
                          "lesson": {"target_word": "forged", "meaning": "伪造", "text_match": False}}
                if explicit:
                    result["english_kind"] = "practice"
                answer = self.reply(result, text="hand")
                self.assertEqual(answer["lesson"], {"target_word": "hand", "meaning": "手",
                                                    "activity": "introduce", "text_match": True})
                self.assertEqual(self.store.english_progress("puppy", "child")[0]["presentations"], 1)
                self.assertEqual(self.store.english_progress("puppy", "child")[0]["text_matches"], 1)
                with self.store.connect() as db:
                    db.execute("DELETE FROM english_progress")
                    db.execute("DELETE FROM english_lessons")

    def test_practice_still_rejects_reply_without_parent_theme_word(self):
        for kind in (None, "practice"):
            with self.subTest(kind=kind):
                result = {"reply": "鼻子的英文是 nose。", "intent": "chat"}
                if kind:
                    result["english_kind"] = kind
                before = len(self.calls)
                with self.assertRaises(ModelResponseError):
                    self.reply(result)
                self.assertEqual(len(self.calls) - before, 2)
                self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_explicit_invalid_kind_is_rejected_with_bounded_attempts(self):
        for kind in (None, True, 1, [], {}, "", "QUESTION", "other"):
            with self.subTest(kind=kind):
                before = len(self.calls)
                with self.assertRaises(ModelResponseError):
                    self.reply({"reply": "手的英文是 hand。", "intent": "chat", "english_kind": kind})
                self.assertEqual(len(self.calls) - before, 2)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_kind_is_only_valid_for_original_english_chat(self):
        for mode, intent in (("chat", "chat"), ("story", "chat"), ("english", "action"),
                             ("english", "greet"), ("english", "action_draft"),
                             ("english", "action_sequence"), ("english", "routine"),
                             ("english", "rename_pet"), ("english", "remember_preference"),
                             ("english", "forget_memory")):
            with self.subTest(mode=mode, intent=intent):
                before = len(self.calls)
                with self.assertRaises(ModelResponseError):
                    self.reply({"reply": "手的英文是 hand。", "intent": intent, "english_kind": "question",
                                "value": "", "action": "wave" if intent == "action" else ""}, mode=mode)
                self.assertEqual(len(self.calls) - before, 2)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])
        self.assertEqual(self.store.memories("puppy", "child"), {})

    def test_model_lesson_in_other_mode_is_discarded(self):
        answer = self.reply({"reply": "你好！", "intent": "chat",
                             "lesson": {"target_word": "forged", "text_match": True}}, mode="chat")
        self.assertNotIn("lesson", answer)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_english_subkind_in_intent_normalizes_only_plain_text_reply(self):
        for kind in ("question", "practice"):
            for redundant in (False, True):
                with self.subTest(kind=kind, redundant=redundant):
                    payload = {"reply": "手的英文是 hand。", "intent": kind, "value": "", "action": ""}
                    if redundant:
                        payload["english_kind"] = kind
                    answer = self.reply(payload, person=None)
                    self.assertEqual(answer["intent"], "chat")
                    self.assertEqual(answer["english_kind"], kind)
                    self.assertEqual("lesson" in answer, kind == "practice")
        self.assertEqual(len(self.calls), 4)

    def test_english_alias_cannot_hide_operations_or_conflicting_kind(self):
        invalid = [{"action": "wave"}, {"actions": ["wave"]}, {"value": "豆豆"},
                   {"routine": {}}, {"lesson": {"target_word": "hand"}},
                   {"english_kind": "question"}, {"english_kind": None}]
        for extra in invalid:
            with self.subTest(extra=extra):
                with self.assertRaises(ModelResponseError):
                    self.reply({"reply": "手的英文是 hand。", "intent": "practice", **extra})
        for mode in ("chat", "story"):
            with self.subTest(mode=mode):
                with self.assertRaises(ModelResponseError):
                    self.reply({"reply": "手的英文是 hand。", "intent": "practice"}, mode=mode)
        self.assertEqual(len(self.calls), 18)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_non_string_intent_uses_bounded_rejection_for_known_and_guest(self):
        for person in ("child", None):
            for intent in ([], {}, None, 1):
                with self.subTest(person=person, intent=intent):
                    with self.assertRaises(ModelResponseError):
                        self.reply({"reply": "手的英文是 hand。", "intent": intent}, person=person)
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_anonymous_private_intent_still_returns_clean_access_status(self):
        for intent in ("remember_preference", "forget_memory", "routine"):
            with self.subTest(intent=intent):
                answer = self.reply({"reply": "我记好了", "intent": intent, "value": "风车",
                                     "english_kind": "question", "lesson": {"target_word": "forged"}},
                                    person=None)
                self.assertEqual(answer["business_status"]["code"], "personal_access_unavailable")
                self.assertNotIn("lesson", answer)
                self.assertNotIn("english_kind", answer)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_curriculum_fields_do_not_authorize_action_after_field_removal(self):
        for field in ({"english_kind": "question"}, {"lesson": {"target_word": "hand", "text_match": True}}):
            with self.subTest(field=field):
                before = len(self.calls)
                with self.assertRaises(ModelResponseError):
                    self.reply({"reply": "我来挥手。", "intent": "action", "action": "wave", "value": "", **field},
                               text="挥手")
                self.assertEqual(len(self.calls) - before, 2)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_greeting_converted_to_chat_is_not_a_lesson(self):
        def model(system, user):
            if system.startswith("你只核验当前这句话是否明确要求"):
                return '{"explicit":true}'
            return '{"reply":"你好，hand 是手。","intent":"greet","value":"","action":""}'
        answer = PetAgent(self.store, model).reply(
            "puppy", "请按约定打个招呼", "child", mode="english", learning_settings=self.settings)
        self.assertEqual(answer["intent"], "chat")
        self.assertNotIn("lesson", answer)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_resolved_rename_is_not_a_lesson(self):
        self.store.propose_name("puppy", "栗栗", "child", "session")
        answer = self.reply({"reply": "我现在叫栗栗啦！hand 是手。", "intent": "chat", "english_kind": "practice",
                             "lesson": {"target_word": "forged", "text_match": True}}, text="确认")
        self.assertEqual(self.store.pet("puppy")["name"], "栗栗")
        self.assertNotIn("lesson", answer)
        self.assertNotIn("english_kind", answer)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])

    def test_save_permission_and_gateway_owned_recording_remain_effective(self):
        for record, save in ((False, True), (True, False)):
            for kind in ("practice", "question"):
                with self.subTest(record_learning=record, save_progress=save, kind=kind):
                    answer = self.reply({"reply": "手的英文是 hand。", "intent": "chat", "english_kind": kind},
                                        record_learning=record, learning_settings={"theme": "body_parts", "save_progress": save})
                    self.assertEqual("lesson" in answer, kind == "practice")
                    self.assertEqual(self.store.english_progress("puppy", "child"), [])
                    self.assertEqual(self.store.conversation("puppy", "child", "english", "session"), [])

    def test_format_retry_does_not_force_curriculum_word_into_question(self):
        replies = iter(("not JSON", json.dumps({"reply": "鼻子的英文是 nose。", "intent": "chat",
                                               "english_kind": "question"}, ensure_ascii=False)))
        def model(system, user):
            self.calls.append((system, user))
            return next(replies)
        answer = PetAgent(self.store, model).reply(
            "puppy", "鼻子的英文怎么说？", "child", mode="english", learning_settings=self.settings)
        self.assertEqual(answer["reply"], "鼻子的英文是 nose。")
        self.assertEqual(len(self.calls), 2)
        self.assertNotIn("回答必须包含英文词 hand", self.calls[1][1])
        self.assertNotIn("lesson", answer)


if __name__ == "__main__":
    unittest.main()
