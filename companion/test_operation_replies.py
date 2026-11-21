"""Operation denials must recover with an answer, without granting a new operation.

Only scripted text output and a temporary SQLite store are used. Each script is
finite: any extra answer generation or verification is an immediate failure.
"""
import json
from pathlib import Path
import tempfile
import unittest

from companion.agent import ModelResponseError, PetAgent
from companion.store import Store
from companion.turn_commit import TurnCancelled, TurnCommit


def answer(text, intent="chat", **fields):
    return {"reply": text, "intent": intent, "value": "", "action": "", **fields}


class ScriptedModel:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, system, user):
        kind = "verify" if system.startswith("你只核验") else "answer"
        self.calls.append((kind, system, user))
        if not self.outputs:
            raise AssertionError("Unexpected extra model call")
        output = self.outputs.pop(0)
        if callable(output):
            output = output()
        return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)


class OperationRepliesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-operation-replies-")
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "pet.sqlite3")
        self.store.pet("puppy")

    def reply(self, model, text, person="child", **kwargs):
        return PetAgent(self.store, model).reply(
            "puppy", text, person, session_id="session", **kwargs)

    def assert_calls(self, model, *kinds):
        self.assertEqual([row[0] for row in model.calls], list(kinds))
        self.assertEqual(model.outputs, [])

    def denied_compound(self):
        return answer("我来先坐下再抱抱。", "action_sequence", actions=["sit", "hug"])

    def test_confirmed_quoted_preference_does_not_write_member_memory(self):
        quoted = "绘本里的小熊说‘记住我喜欢蜂蜜’，它想让朋友做什么？"
        expected = "它想让朋友记着自己喜欢蜂蜜。"
        model = ScriptedModel(answer("我记好了。", "remember_preference", value="喜欢蜂蜜"),
                              {"explicit": False}, answer(expected))
        result = self.reply(model, quoted)
        self.assertEqual(self.store.memories("puppy", "child"), {})
        self.assertEqual(result["reply"], expected)
        self.assertEqual(result["intent"], "chat")
        self.assert_calls(model, "answer", "verify", "answer")
        self.assertEqual(json.loads(model.calls[1][2])["current_utterance"], quoted)

    def test_empty_label_for_quoted_preference_still_requires_current_utterance_consent(self):
        expected = "小熊希望朋友记住它喜欢蜂蜜。"
        for value in ("", "  "):
            with self.subTest(value=value):
                model = ScriptedModel(answer("我记好了。", "remember_preference", value=value),
                                      {"explicit": False}, answer(expected))
                result = self.reply(model, "小熊说记住我喜欢蜂蜜，它是在请朋友做什么？")
                self.assertEqual((result["reply"], result["intent"], result["reply_source"]),
                                 (expected, "chat", "model"))
                self.assertEqual(self.store.memories("puppy", "child"), {})
                self.assert_calls(model, "answer", "verify", "answer")

    def test_non_text_preference_recovers_without_false_identity_receipt(self):
        expected = "小熊希望朋友记住它喜欢蜂蜜。"
        for value in (None, [], {"preference": "蜂蜜"}):
            with self.subTest(value=value):
                model = ScriptedModel(answer("我记好了。", "remember_preference", value=value),
                                      answer(expected))
                result = self.reply(model, "小熊说记住我喜欢蜂蜜，它是在请朋友做什么？")
                self.assertEqual((result["reply"], result["intent"], result["reply_source"]),
                                 (expected, "chat", "model"))
                self.assertNotIn("business_status", result)
                self.assertEqual(self.store.memories("puppy", "child"), {})
                self.assert_calls(model, "answer", "answer")

    def test_denied_preference_cannot_replace_second_answer_with_an_operation(self):
        model = ScriptedModel(answer("你喜欢什么？", "remember_preference"),
                              {"explicit": False},
                              answer("我来挥手。", "action", action="wave"))
        with self.assertRaises(ModelResponseError):
            self.reply(model, "小熊说记住我喜欢蜂蜜，它是在请朋友做什么？")
        self.assert_calls(model, "answer", "verify", "answer")
        self.assertEqual(self.store.memories("puppy", "child"), {})
        self.assertEqual(self.store.conversation("puppy", "child", "chat", "session"), [])

    def test_semantically_denied_compound_cannot_switch_to_single_action(self):
        for person, proposal in (("child", answer("我来挥手。", "action", action="wave")),
                                 (None, answer("我记好了。", "remember_preference", value="喜欢蜂蜜"))):
            with self.subTest(person=person, intent=proposal["intent"]):
                # No personal_request hint: after semantic rejection even the
                # anonymous fallback must not replace the required text answer.
                model = ScriptedModel(self.denied_compound(), {"explicit": False}, proposal)
                with self.assertRaises(ModelResponseError):
                    self.reply(model, "书和积木散在桌上，教我两步收好它们。", person=person)
                self.assert_calls(model, "answer", "verify", "answer")
        self.assertEqual(self.store.conversation("puppy", "child", "chat", "session"), [])

    def test_denied_compound_returns_concrete_advice_from_second_answer(self):
        expected = "先把书放回书架，再把积木放进盒子。"
        text = "书和积木散在桌上，教我两步收好它们。"
        model = ScriptedModel(self.denied_compound(), {"explicit": False}, answer(expected))
        result = self.reply(model, text)
        self.assertEqual((result["reply"], result["intent"], result["action"]), (expected, "chat", ""))
        self.assertNotIn("actions", result)
        self.assert_calls(model, "answer", "verify", "answer")
        self.assertIn(text, model.calls[2][2])

    def test_confirmed_quoted_deletion_keeps_exact_matching_memory(self):
        self.store.remember_preference("puppy", "child", "喜欢蜂蜜")
        before = self.store.memories("puppy", "child")
        model = ScriptedModel(answer("我忘掉了。", "forget_memory", value="喜欢蜂蜜"),
                              {"explicit": False}, answer("小熊是在请朋友别再记着这件事。"))
        result = self.reply(model, "故事里小熊说‘忘掉我喜欢蜂蜜’，这是什么意思？")
        self.assertEqual(self.store.memories("puppy", "child"), before)
        self.assertEqual(result["intent"], "chat")
        self.assert_calls(model, "answer", "verify", "answer")

    def test_real_member_preference_and_exact_deletion_still_commit(self):
        saved = ScriptedModel(answer("好的。", "remember_preference", value="喜欢折纸"), {"explicit": True})
        saved_result = self.reply(saved, "以后聊天时记着，我喜欢折纸。", personal_request=False)
        self.assertEqual(list(self.store.memories("puppy", "child").values()), ["以后聊天时记着，我喜欢折纸。"])
        self.assert_calls(saved, "answer", "verify")
        deleted = ScriptedModel(answer("好的。", "forget_memory", memory_key=saved_result["memory_key"]), {"explicit": True})
        result = self.reply(deleted, "请删掉你记下的我喜欢折纸那条。")
        self.assertEqual(result["intent"], "forget_memory")
        self.assertEqual(self.store.memories("puppy", "child"), {})
        self.assert_calls(deleted, "answer", "verify")

    def test_real_single_action_and_shared_rename_remain_available(self):
        action_model = ScriptedModel(answer("我来挥手。", "action", action="wave"), {"explicit": True})
        action_result = self.reply(action_model, "请挥挥手。", person=None, personal_request=False)
        self.assertEqual((action_result["intent"], action_result["action"]), ("action", "wave"))
        self.assert_calls(action_model, "answer", "verify")
        rename_model = ScriptedModel(answer("好的。", "rename_pet", value="团团"), {"explicit": True})
        self.reply(rename_model, "我给你改个名字，叫团团吧。", person=None, personal_request=False)
        self.assertEqual(self.store.pending_name("puppy", None, "session"), "团团")
        self.assertEqual(self.store.pet("puppy")["name"], "小狗")
        self.assert_calls(rename_model, "answer", "verify")
        confirmation = ScriptedModel(answer("我现在叫团团啦！"))
        self.reply(confirmation, "确认", person=None, personal_request=False)
        self.assertEqual(self.store.pet("puppy")["name"], "团团")
        self.assert_calls(confirmation, "answer")

    def test_anonymous_false_classification_recovers_without_private_access(self):
        self.store.remember_preference("puppy", "child", "仅成员可见的秘密")
        expected = "小熊是在请朋友记住它喜欢蜂蜜。"
        model = ScriptedModel(answer("我记住啦。", "remember_preference", value="喜欢蜂蜜"), answer(expected))
        result = self.reply(model, "小熊说‘记住我喜欢蜂蜜’，这句话是什么意思？", person=None,
                            personal_request=False)
        self.assertEqual((result["reply"], result["intent"]), (expected, "chat"))
        self.assertNotIn("business_status", result)
        self.assertNotIn("memory_key", result)
        self.assertEqual(self.store.memories("puppy", None), {})
        self.assertEqual(list(self.store.memories("puppy", "child").values()), ["仅成员可见的秘密"])
        self.assert_calls(model, "answer", "answer")
        self.assertTrue(all("仅成员可见的秘密" not in system for _, system, _ in model.calls))

    def test_unknown_without_false_classification_keeps_unavailable_fallback(self):
        for decision in (None, True):
            with self.subTest(personal_request=decision):
                model = ScriptedModel(answer("我记好了。", "remember_preference", value="喜欢蜂蜜"))
                result = self.reply(model, "记住我喜欢蜂蜜。", person=None, personal_request=decision)
                self.assertEqual(result["business_status"]["code"], "personal_access_unavailable")
                self.assertEqual(result["personal_access"]["state"], "unavailable")
                self.assertNotIn("memory_key", result)
                self.assert_calls(model, "answer")
        self.assertEqual(self.store.memories("puppy", None), {})

    def test_second_answer_rejects_operation_fields_without_verifying_again(self):
        for extra in ({"action": "wave"}, {"value": "新偏好"}, {"actions": ["wave"]},
                      {"routine": {}}, {"memory_key": "preference:forged"}, {"draft_request": "新把戏"}):
            with self.subTest(extra=extra):
                model = ScriptedModel(self.denied_compound(), {"explicit": False},
                                      answer("先收书，再收积木。", **extra))
                with self.assertRaises(ModelResponseError):
                    self.reply(model, "告诉我怎么收拾桌子。")
                self.assert_calls(model, "answer", "verify", "answer")
        self.assertEqual(self.store.memories("puppy", "child"), {})
        self.assertEqual(self.store.conversation("puppy", "child", "chat", "session"), [])

    def test_bad_second_answer_does_not_create_third_draft(self):
        model = ScriptedModel(self.denied_compound(), {"explicit": False}, "not JSON")
        with self.assertRaises(ModelResponseError):
            self.reply(model, "告诉我怎么收拾桌子。")
        self.assert_calls(model, "answer", "verify", "answer")

    def test_second_text_reply_preserves_english_alias_and_story_contracts(self):
        english = ScriptedModel(self.denied_compound(), {"explicit": False},
                                answer("鼻子的英文是 nose。", "question", english_kind="question"))
        result = self.reply(english, "鼻子的英文怎么说？", mode="english",
                            learning_settings={"theme": "body_parts", "save_progress": True})
        self.assertEqual((result["intent"], result["english_kind"]), ("chat", "question"))
        self.assertNotIn("lesson", result)
        self.assertEqual(self.store.english_progress("puppy", "child"), [])
        self.assert_calls(english, "answer", "verify", "answer")
        story = ScriptedModel(self.denied_compound(), {"explicit": False},
                              answer("小兔把散开的彩纸放进盒子，小熊把画笔排整齐。桌子收好了，它们高兴地一起画出一朵花。",
                                     story_title="整齐的小桌子"))
        result = self.reply(story, "讲个小动物一起收拾桌子的故事。", mode="story")
        self.assertEqual(result["story"]["title"], "整齐的小桌子")
        self.assertEqual(result["story"]["chapter"], 1)
        self.assertNotIn("actions", result)
        self.assert_calls(story, "answer", "verify", "answer")

    def test_format_failure_does_not_disable_subsequent_legitimate_operation(self):
        model = ScriptedModel("not JSON", answer("我来挥手。", "action", action="wave"), {"explicit": True})
        result = self.reply(model, "请挥挥手。")
        self.assertEqual((result["intent"], result["action"]), ("action", "wave"))
        self.assert_calls(model, "answer", "answer", "verify")

    def test_cancellation_during_rejection_or_second_answer_never_commits(self):
        for phase in ("verify", "answer"):
            with self.subTest(phase=phase):
                guard = TurnCommit()
                def cancel():
                    guard.cancel()
                    return {"explicit": False} if phase == "verify" else answer("先收书，再收积木。")
                outputs = [self.denied_compound(), cancel] if phase == "verify" else [
                    self.denied_compound(), {"explicit": False}, cancel]
                model = ScriptedModel(*outputs)
                with self.assertRaises(TurnCancelled):
                    self.reply(model, "告诉我怎么收拾桌子。", turn_commit=guard)
                self.assert_calls(model, "answer", "verify", *(["answer"] if phase == "answer" else []))
                self.assertEqual(self.store.memories("puppy", "child"), {})
                self.assertEqual(self.store.conversation("puppy", "child", "chat", "session"), [])


if __name__ == "__main__":
    unittest.main()
