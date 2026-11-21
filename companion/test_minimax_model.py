"""Bounded MiniMax adapter checks without network calls or personal data."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from companion.agent import ModelResponseError
from companion.minimax_model import MiniMaxModel, voice_remote_only


class MiniMaxModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-minimax-test-")
        self.addCleanup(self.temp.cleanup)
        self.key_file = Path(self.temp.name) / ".minimax-token"
        self.key_file.write_text("fixture-minimax-token-for-local-tests-only")
        self.key_file.chmod(0o600)
        self.model = MiniMaxModel(self.key_file, "/test-only/local-model")

    def test_remote_text_uses_m31_low_effort_and_keeps_images_local(self):
        def fake_open(request, timeout):
            self.assertEqual(request.full_url, "https://api.minimax.cn/v1/chat/completions")
            self.assertEqual(timeout, 8)
            self.assertEqual(request.get_header("Authorization"),
                             "Bearer fixture-minimax-token-for-local-tests-only")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "MiniMax-M3.1-Flash-Preview")
            self.assertEqual(payload["reasoning_effort"], "low")
            self.assertNotIn("thinking", payload)
            self.assertEqual(payload["max_completion_tokens"], 1024)
            self.assertEqual(payload["messages"], [
                {"role": "system", "content": "规则"}, {"role": "user", "content": "三加二等于几？"}])
            return io.BytesIO(json.dumps({"base_resp": {"status_code": 0},
                "choices": [{"finish_reason": "stop", "message": {"content": "三加二等于五。"}}]}).encode())

        with patch("companion.minimax_model.urlopen", side_effect=fake_open), \
             patch("companion.minimax_model.LocalModel") as local:
            self.assertEqual(self.model("规则", "三加二等于几？"), "三加二等于五。")
            local.assert_not_called()

    def test_network_failure_uses_local_model_without_exposing_prompt(self):
        local = Mock(return_value="本地回答")
        with patch("companion.minimax_model.urlopen", side_effect=TimeoutError), \
             patch("companion.minimax_model.LocalModel", return_value=local) as factory:
            self.assertEqual(self.model("private-system", "private-user"), "本地回答")
            factory.assert_called_once_with("/test-only/local-model")
            local.assert_called_once_with("private-system", "private-user")

    def test_live_voice_failure_skips_slow_local_fallback(self):
        with patch("companion.minimax_model.urlopen", side_effect=TimeoutError), \
             patch("companion.minimax_model.LocalModel") as local:
            with voice_remote_only(), self.assertRaises(ModelResponseError):
                self.model("private-system", "private-user")
            local.assert_not_called()
            # The restriction ends with the voice turn; normal fallback remains.
            local.return_value.return_value = "本地回答"
            self.assertEqual(self.model("规则", "问题"), "本地回答")

    def test_key_permissions_must_be_owner_only(self):
        self.key_file.chmod(0o644)
        with patch("companion.minimax_model.urlopen") as open_remote, \
             patch("companion.minimax_model.LocalModel", return_value=Mock(return_value="本地回答")):
            self.assertEqual(self.model("规则", "问题"), "本地回答")
            open_remote.assert_not_called()

    def test_camera_photo_is_never_sent_to_remote(self):
        local = Mock()
        local.describe_bytes.return_value = {"reply": "看见球", "uncertain": False}
        with patch("companion.minimax_model.LocalModel", return_value=local), \
             patch("companion.minimax_model.urlopen") as open_remote:
            self.assertEqual(self.model.describe_bytes(b"jpeg-test-only", "这是什么"),
                             {"reply": "看见球", "uncertain": False})
            open_remote.assert_not_called()

    def test_visual_planning_and_followup_use_local_text_model(self):
        local = Mock(return_value='{"use_camera":true}')
        with patch("companion.minimax_model.LocalModel", return_value=local), \
             patch("companion.minimax_model.urlopen") as open_remote:
            self.assertTrue(self.model.requests_camera("看看我家桌上是什么"))
            self.assertIs(self.model.private_context_model(), local)
            self.assertEqual(self.model.private_context_model()("场景", "桌上有玩具"),
                             '{"use_camera":true}')
            open_remote.assert_not_called()
