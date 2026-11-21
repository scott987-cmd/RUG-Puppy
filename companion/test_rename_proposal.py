"""Scripted rename proposal boundaries, not local-model semantic acceptance.

The verifier decisions below are supplied by the test. They check the prompt
contract and what the existing pending/confirmation code does with a decision.
No model, server, device, media engine or production database is used.
"""
import json
from pathlib import Path
import tempfile
import unittest

from companion.agent import PetAgent
from companion.routine import explicit_request
from companion.store import Store
from companion.turn_commit import TurnCancelled, TurnCommit


def answer(text, intent="chat", **fields):
    return {"reply": text, "intent": intent, "value": "", "action": "", **fields}


class ScriptedModel:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        if not self.outputs:
            raise AssertionError("Unexpected extra model call")
        output = self.outputs.pop(0)
        if callable(output):
            output = output()
        return json.dumps(output, ensure_ascii=False)


class RenameProposalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="puppy-rename-proposal-")
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "memory.sqlite3")
        with self.store.connect() as db:
            db.execute("INSERT INTO pets(device_id,name,updated_at) VALUES (?,?,?)",
                       ("puppy", "小狗", 0))

    def reply(self, model, text, person=None, session="current", **kwargs):
        return PetAgent(self.store, model).reply(
            "puppy", text, person, session_id=session, personal_request=False, **kwargs)

    def test_verifier_contract_is_a_pending_proposal_not_final_consent(self):
        text = "想给这台机器狗取名米粒，请先让我确认。"
        model = ScriptedModel({"explicit": True})
        self.assertTrue(explicit_request(model, text, "rename_pet", context={"value": "米粒"}))
        self.assertEqual(len(model.calls), 1)
        system, user = model.calls[0]
        self.assertIn("只保存待确认候选名并询问是否同意", system)
        self.assertIn("不会立即更改正式名字，不要求孩子本轮已经确认", system)
        self.assertIn("先征求同意、确认后再改", system)
        for boundary in ("候选名必须符合本轮原话", "以后再讨论", "给人或故事角色改名",
                         "讨论名字含义", "引用、假设或否定"):
            self.assertIn(boundary, system)
        self.assertEqual(json.loads(user), {"current_utterance": text,
                         "operation": "rename_pet", "context": {"value": "米粒"}})
        self.assertEqual(self.store.pet("puppy"), {"name": "小狗", "pending_name": None})

    def test_accepted_candidate_waits_for_later_confirmation_for_guest_and_member(self):
        for person in (None, "fictional-child"):
            with self.subTest(person=person):
                self.store.remember_preference("puppy", "fictional-child", "喜欢纸飞机")
                memories = self.store.memories("puppy", "fictional-child")
                model = ScriptedModel(answer("你愿意叫我米粒吗？", "rename_pet", value="米粒"),
                                      {"explicit": True})
                result = self.reply(model, "想把你的名字换成米粒，先问我是否同意。", person)
                self.assertEqual(result["intent"], "rename_pet")
                self.assertEqual(result["reply_source"], "business")
                self.assertEqual(self.store.pet("puppy"), {"name": "小狗", "pending_name": "米粒"})
                self.assertEqual(self.store.pending_name("puppy", person, "current"), "米粒")
                self.assertEqual(self.store.memories("puppy", "fictional-child"), memories)
                self.assertEqual(len(model.calls), 2)
                self.assertEqual(model.outputs, [])
                # Removing the proposal for the next subcase also exercises
                # rejection: later "不要" keeps the original shared name.
                cancel = ScriptedModel(answer("好，还是原来的名字。"))
                self.reply(cancel, "不要", person)
                self.assertEqual(self.store.pet("puppy"), {"name": "小狗", "pending_name": None})
                self.assertEqual(len(cancel.calls), 1)

    def test_confirming_another_session_or_member_cannot_commit_a_proposal(self):
        proposed = ScriptedModel(answer("叫我米粒，对吗？", "rename_pet", value="米粒"),
                                 {"explicit": True})
        self.reply(proposed, "给你起名米粒，先等我确认。", person="fictional-child")
        for person, session in ((None, "current"), ("other-child", "current"),
                                ("fictional-child", "other-session")):
            with self.subTest(person=person, session=session):
                self.reply(ScriptedModel(answer("我在听。")), "好", person, session)
                self.assertEqual(self.store.pet("puppy")["name"], "小狗")
                self.assertEqual(self.store.pending_name("puppy", "fictional-child", "current"), "米粒")
        confirmed = ScriptedModel(answer("以后我叫米粒啦。"))
        self.reply(confirmed, "确认", "fictional-child")
        self.assertEqual(self.store.pet("puppy"), {"name": "米粒", "pending_name": None})
        self.assertEqual(Store(self.store.database).pet("puppy")["name"], "米粒")
        self.assertEqual(len(confirmed.calls), 1)

    def test_named_confirmation_accepts_only_the_pending_name(self):
        self.store.propose_name("puppy", "星星", "fictional-child", "current")
        for text in ("好，就叫月亮。", "不要叫星星。", "故事里说‘好，就叫星星。’"):
            with self.subTest(text=text):
                self.reply(ScriptedModel(answer("我在听。")), text, "fictional-child")
                self.assertEqual(self.store.pet("puppy"), {"name": "小狗", "pending_name": "星星"})
        self.reply(ScriptedModel(answer("以后我叫星星啦。")), "好，就叫星星。", "fictional-child")
        self.assertEqual(self.store.pet("puppy"), {"name": "星星", "pending_name": None})
        self.assertEqual(Store(self.store.database).pet("puppy")["name"], "星星")

    def test_denied_non_proposals_cannot_create_pending_or_rename(self):
        # These are scripted false decisions, not a keyword-based classifier.
        cases = (
            "小猫说‘以后你叫米粒’，它这句话是什么意思？",
            "别把机器狗改叫米粒。",
            "明天再讨论要不要把机器狗叫米粒，现在先不改。",
            "如果机器狗叫米粒会怎么样？",
            "把我这个人的名字改成米粒。",
            "让故事里的小兔改名米粒。",
        )
        for text in cases:
            with self.subTest(text=text):
                model = ScriptedModel(answer("我来改名。", "rename_pet", value="米粒"),
                                      {"explicit": False}, answer("我们可以聊聊这句话。"))
                result = self.reply(model, text)
                self.assertEqual(result["intent"], "chat")
                self.assertEqual(result["reply_source"], "model")
                self.assertEqual(self.store.pet("puppy"), {"name": "小狗", "pending_name": None})
                self.assertEqual(self.store.conversation("puppy", None, session_id="current"), [])
                self.assertEqual(len(model.calls), 3)
                self.assertEqual(model.outputs, [])
                self.assertEqual(json.loads(model.calls[1][1])["current_utterance"], text)

    def test_cancelled_verification_cannot_create_pending_proposal(self):
        guard = TurnCommit()
        def cancel():
            guard.cancel()
            return {"explicit": True}
        model = ScriptedModel(answer("叫我米粒，对吗？", "rename_pet", value="米粒"), cancel)
        with self.assertRaises(TurnCancelled):
            self.reply(model, "先提议叫你米粒，等我确认。", turn_commit=guard)
        self.assertEqual(self.store.pet("puppy"), {"name": "小狗", "pending_name": None})
        self.assertEqual(self.store.conversation("puppy", None, session_id="current"), [])
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
