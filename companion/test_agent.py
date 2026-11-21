import json
import tempfile
import unittest
from pathlib import Path

from companion.agent import PetAgent
from companion.store import Store


def _new_operation_guard(system, user):
    """Fixture-only consent for newly added guards; main replies keep their sequence."""
    if system.startswith("你只核验") and json.loads(user).get("operation") in {
            "action", "rename_pet", "remember_preference", "forget_memory"}:
        return '{"explicit":true}'
    return None


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "pet.db"
        self.store = Store(self.database)

    def tearDown(self):
        self.temp.cleanup()

    def test_rename_survives_new_process_and_requires_confirmation(self):
        model = lambda system, user: _new_operation_guard(system, user) or (
            '{"reply":"好的","intent":"rename_pet","value":"豆豆"}'
            if "以后" in user else
            '{"reply":"我是豆豆","intent":"chat"}'
        )
        agent = PetAgent(self.store, model)
        self.assertIn("对吗", agent.reply("puppy-1", "以后叫豆豆", "child-1")["reply"])
        self.assertEqual(self.store.pet("puppy-1")["name"], "小狗")
        agent = PetAgent(Store(self.database), model)
        agent.reply("puppy-1", "好", "child-1")
        new_process_store = Store(self.database)
        self.assertEqual(new_process_store.pet("puppy-1")["name"], "豆豆")
        self.assertEqual(PetAgent(new_process_store, model).reply("puppy-1", "你叫什么")["reply"], "我是豆豆")

    def test_unknown_speaker_never_reads_or_writes_child_memory(self):
        self.store.remember("puppy-1", "child-1", "preference", "仅成员可见的秘密偏好")
        seen = []

        def model(system, user):
            seen.append(system)
            return '{"reply":"我听到了","intent":"remember_preference","value":"熊猫"}'

        PetAgent(self.store, model).reply("puppy-1", "我喜欢熊猫")
        self.assertNotIn("仅成员可见的秘密偏好", seen[-1])
        self.assertEqual(self.store.memories("puppy-1", "child-1")["preference"], "仅成员可见的秘密偏好")
        self.assertEqual(self.store.memories("puppy-1", None), {})

    def test_unapproved_action_is_not_returned_as_action(self):
        model = lambda system, user: '{"reply":"好的","intent":"action","action":"calibrate"}'
        answer = PetAgent(self.store, model).reply("puppy-1", "标定")
        self.assertEqual(answer["intent"], "chat")
        self.assertEqual(answer["action"], "")


if __name__ == "__main__":
    unittest.main()
