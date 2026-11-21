from pathlib import Path
import json
import tempfile
import unittest

from companion.agent import PetAgent
from companion.audio_identity import device_audio_conversation_id
from companion.store import Store


class AudioIdentityTests(unittest.TestCase):
    def test_key_is_stable_bounded_and_device_scoped(self):
        first = device_audio_conversation_id("puppy-a")
        self.assertEqual(first, device_audio_conversation_id("puppy-a"))
        self.assertNotEqual(first, device_audio_conversation_id("puppy-b"))
        self.assertLessEqual(len(device_audio_conversation_id("x" * 64)), 64)
        with self.assertRaises(ValueError):
            device_audio_conversation_id("")

    def test_next_authenticated_audio_connection_can_confirm_same_members_rename(self):
        with tempfile.TemporaryDirectory(prefix="puppy-audio-identity-") as directory:
            store = Store(Path(directory) / "pet.sqlite3")
            first_connection = device_audio_conversation_id("puppy-a")
            store.propose_name("puppy-a", "团团", "child-a", first_connection)
            next_connection = device_audio_conversation_id("puppy-a")
            self.assertEqual(store.pending_name("puppy-a", "child-a", next_connection), "团团")
            for person, session in (("child-b", next_connection),
                                    ("child-a", device_audio_conversation_id("puppy-b"))):
                with self.assertRaises(ValueError):
                    store.confirm_name("puppy-a", True, person, session)
            store.confirm_name("puppy-a", True, "child-a", next_connection)
            self.assertEqual(store.pet("puppy-a")["name"], "团团")

    def test_anonymous_confirmation_stays_on_one_authenticated_device(self):
        with tempfile.TemporaryDirectory(prefix="puppy-audio-anon-") as directory:
            store = Store(Path(directory) / "pet.sqlite3")
            store.propose_name("puppy-a", "豆豆", None, device_audio_conversation_id("puppy-a"))
            with self.assertRaises(ValueError):
                store.confirm_name("puppy-a", True, None, device_audio_conversation_id("puppy-b"))
            store.confirm_name("puppy-a", True, None, device_audio_conversation_id("puppy-a"))
            self.assertEqual(store.pet("puppy-a")["name"], "豆豆")

    def test_agent_rename_survives_a_new_audio_websocket(self):
        with tempfile.TemporaryDirectory(prefix="puppy-audio-agent-") as directory:
            store = Store(Path(directory) / "pet.sqlite3")
            replies = iter((
                '{"reply":"以后叫我团团，对吗？","intent":"rename_pet","value":"团团"}',
                '{"reply":"好呀，我是团团！","intent":"chat"}',
            ))

            def model(system, user):
                if system.startswith("你只核验") and json.loads(user).get("operation") == "rename_pet":
                    return '{"explicit":true}'
                return next(replies)

            first_key = device_audio_conversation_id("puppy-a")
            PetAgent(store, model).reply("puppy-a", "以后你叫团团", None, "chat", first_key)
            self.assertEqual(store.pending_name("puppy-a", None, first_key), "团团")
            self.assertEqual(store.pet("puppy-a")["name"], "小狗")
            # A new WebSocket gets a new trace ID, but the authenticated
            # device conversation key remains the same during the TTL.
            second_key = device_audio_conversation_id("puppy-a")
            result = PetAgent(store, model).reply("puppy-a", "好", None, "chat", second_key)
            self.assertEqual(result["reply"], "好呀，我是团团！")
            self.assertEqual(store.pet("puppy-a")["name"], "团团")


if __name__ == "__main__":
    unittest.main()
