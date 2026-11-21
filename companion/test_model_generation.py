"""Real LocalModel methods with fake SDK results; no model, media or device I/O."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .agent import LocalModel, ModelResponseError, PetAgent
from .store import Store


def result(text, reason="stop", tokens=1):
    return types.SimpleNamespace(text=text, finish_reason=reason,
                                 generation_tokens=tokens)


class ModelGenerationTests(unittest.TestCase):
    def setUp(self):
        self.processor = types.SimpleNamespace(
            apply_chat_template=Mock(return_value="synthetic rendered prompt"))
        self.sdk_model = types.SimpleNamespace(config=object())
        self.sdk = types.ModuleType("mlx_vlm")
        self.sdk.load = Mock(return_value=(self.sdk_model, self.processor))
        self.sdk.generate = Mock()
        self.prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
        self.prompt_utils.apply_chat_template = Mock(return_value="synthetic vision prompt")
        self.pil = types.ModuleType("PIL")
        self.pil.Image = types.SimpleNamespace(
            open=Mock(side_effect=AssertionError("No image I/O in generation tests")))
        self.modules = patch.dict(sys.modules, {
            "mlx_vlm": self.sdk, "mlx_vlm.prompt_utils": self.prompt_utils, "PIL": self.pil,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.model = LocalModel("synthetic-model-not-loaded")
        self.image = object()  # Passed only to the fake generate function.

    def tearDown(self):
        self.pil.Image.open.assert_not_called()
        self.sdk.load.assert_called_once_with("synthetic-model-not-loaded", trust_remote_code=False)
        for call in self.sdk.generate.call_args_list:
            self.assertNotIn("audio", call.kwargs)
            self.assertNotIn("video", call.kwargs)
            if "image" in call.kwargs:
                self.assertEqual(call.kwargs["image"], [self.image])

    def store(self):
        directory = tempfile.TemporaryDirectory(prefix="puppy-generation-")
        self.addCleanup(directory.cleanup)
        return Store(Path(directory.name) / "store.sqlite3")

    def test_text_length_rejects_complete_json_without_nested_retry(self):
        private_text = '{"reply":"test-only-private-content","intent":"action","action":"wave"}'
        self.sdk.generate.return_value = result(private_text, "length", 240)
        with self.assertRaises(ModelResponseError) as error:
            self.model("system", "user")
        self.assertNotIn("test-only-private-content", str(error.exception))
        self.sdk.generate.assert_called_once()

    def test_text_stop_at_token_limit_is_valid_and_template_options_survive(self):
        raw = '{"reply":"你好","intent":"chat"}'
        self.sdk.generate.return_value = result(raw, "stop", 240)
        self.assertEqual(self.model("system", "user"), raw)
        self.processor.apply_chat_template.assert_called_once_with(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(self.sdk.generate.call_args.kwargs, {
            "max_tokens": 240, "temperature": 0.2, "enable_thinking": False, "verbose": False,
        })

    def test_explicit_length_reason_rejects_even_when_count_is_below_limit(self):
        self.sdk.generate.return_value = result('{"reply":"你好"}', "length", 1)
        with self.assertRaises(ModelResponseError):
            self.model("system", "user")

    def test_camera_length_cannot_authorize_capture_from_valid_prefix(self):
        self.sdk.generate.return_value = result('{"use_camera":true}\ntruncated tail', "length", 40)
        self.assertFalse(self.model.requests_camera("看看我手里是什么"))
        self.sdk.generate.assert_called_once()
        self.assertEqual(self.sdk.generate.call_args.kwargs["max_tokens"], 40)
        self.assertTrue(self.processor.apply_chat_template.call_args.kwargs["add_generation_prompt"])
        self.assertIs(self.processor.apply_chat_template.call_args.kwargs["enable_thinking"], False)

    def test_camera_stop_at_limit_still_accepts_explicit_true(self):
        self.sdk.generate.return_value = result('{"use_camera":true}', "stop", 40)
        self.assertTrue(self.model.requests_camera("看看我手里是什么"))
        self.sdk.generate.assert_called_once()

    def test_camera_normal_false_or_malformed_remains_false(self):
        for raw in ('{"use_camera":false}', '{"use_camera":', '{"use_camera":1}'):
            with self.subTest(raw=raw):
                self.sdk.generate.reset_mock()
                self.sdk.generate.return_value = result(raw)
                self.assertFalse(self.model.requests_camera("你好"))
                self.sdk.generate.assert_called_once()

    def test_vision_retries_length_once_and_uses_only_completed_result(self):
        self.sdk.generate.side_effect = [
            result('{"reply":"discarded truncated answer","uncertain":false}', "length", 240),
            result('{"reply":"看不清。","uncertain":true}', "stop", 240),
        ]
        self.assertEqual(self.model.describe(self.image, "synthetic question"),
                         {"reply": "看不清。", "uncertain": True})
        self.assertEqual(self.sdk.generate.call_count, 2)
        self.prompt_utils.apply_chat_template.assert_called_once()
        self.assertEqual(self.prompt_utils.apply_chat_template.call_args.kwargs,
                         {"num_images": 1, "enable_thinking": False})

    def test_vision_two_lengths_raise_model_error_without_third_attempt(self):
        self.sdk.generate.return_value = result('{"reply":"discarded","uncertain":false}', "length", 240)
        with self.assertRaises(ModelResponseError):
            self.model.describe(self.image, "synthetic question")
        self.assertEqual(self.sdk.generate.call_count, 2)

    def test_vision_two_malformed_results_use_same_typed_failure(self):
        self.sdk.generate.return_value = result('{"reply":')
        with self.assertRaises(ModelResponseError):
            self.model.describe(self.image, "synthetic question")
        self.assertEqual(self.sdk.generate.call_count, 2)

    def test_vision_completed_long_reply_is_retried_without_cutting_a_sentence(self):
        self.sdk.generate.side_effect = [
            result(json.dumps({"reply": "看见的画面" * 40 + "。", "uncertain": False})),
            result('{"reply":"只能看清一个红色盒子。","uncertain":true}'),
        ]
        self.assertEqual(self.model.describe(self.image, "synthetic question"),
                         {"reply": "只能看清一个红色盒子。", "uncertain": True})
        self.assertEqual(self.sdk.generate.call_count, 2)

    def test_vision_two_completed_long_replies_fail_without_cutting(self):
        self.sdk.generate.return_value = result(json.dumps({"reply": "画" * 181, "uncertain": False}))
        with self.assertRaises(ModelResponseError):
            self.model.describe(self.image, "synthetic question")
        self.assertEqual(self.sdk.generate.call_count, 2)

    def test_farewell_recovers_after_sdk_truncation_with_two_calls(self):
        self.sdk.generate.side_effect = [result('{"reply":"坏的完整前缀"}', "length", 240),
                                         result('{"reply":"我们休息一下吧。"}')]
        answer = PetAgent(self.store(), self.model).finish_learning({"reason": "daily_limit"})
        self.assertEqual(answer["reply"], "我们休息一下吧。")
        self.assertEqual(self.sdk.generate.call_count, 2)

    def test_vision_sdk_error_is_not_swallowed_as_a_parse_retry(self):
        self.sdk.generate.side_effect = ValueError("synthetic SDK failure")
        with self.assertRaisesRegex(ValueError, "synthetic SDK failure"):
            self.model.describe(self.image, "synthetic question")
        self.sdk.generate.assert_called_once()

    def test_pet_agent_rejects_truncated_action_and_memory_after_two_attempts(self):
        for payload in (
            {"reply": "好的", "intent": "action", "action": "wave", "value": ""},
            {"reply": "记住啦", "intent": "remember_preference", "value": "喜欢苹果"},
        ):
            with self.subTest(intent=payload["intent"]):
                store = self.store()
                self.sdk.generate.reset_mock()
                self.sdk.generate.return_value = result(json.dumps(payload, ensure_ascii=False), "length", 240)
                with self.assertRaises(ModelResponseError):
                    PetAgent(store, self.model).reply("puppy", "请挥手，我喜欢苹果", "child")
                self.assertEqual(self.sdk.generate.call_count, 2)
                self.assertEqual(store.memory_entries("puppy", "child"), [])
                self.assertEqual(store.conversation("puppy", "child", "chat", None), [])

    def test_pet_agent_can_recover_on_second_completed_response(self):
        store = self.store()
        self.sdk.generate.side_effect = [
            result('{"reply":"记住啦","intent":"remember_preference","value":"不应保存"}', "length", 240),
            result('{"reply":"我们一起玩吧。","intent":"chat"}', "stop", 240),
        ]
        answer = PetAgent(store, self.model).reply("puppy", "你好", "child")
        self.assertEqual(answer["reply"], "我们一起玩吧。")
        self.assertEqual(answer["intent"], "chat")
        self.assertEqual(self.sdk.generate.call_count, 2)
        self.assertEqual(store.memory_entries("puppy", "child"), [])

    def test_truncated_name_confirmation_does_not_commit_pending_name(self):
        store = self.store()
        original = store.pet("puppy")["name"]
        store.propose_name("puppy", "星星", session_id="session")
        self.sdk.generate.return_value = result('{"reply":"我叫星星啦","intent":"chat"}', "length", 240)
        with self.assertRaises(ModelResponseError):
            PetAgent(store, self.model).reply("puppy", "好", session_id="session")
        self.assertEqual(self.sdk.generate.call_count, 2)
        self.assertEqual(store.pet("puppy")["name"], original)
        self.assertEqual(store.pending_name("puppy", session_id="session"), "星星")


if __name__ == "__main__":
    unittest.main()
