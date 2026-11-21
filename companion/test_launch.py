"""Isolated launch configuration checks; no real credentials or process starts."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from companion import launch
from companion.agent import LocalModel


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-launch-")
        self.root = Path(self.temp.name)
        self.default_model = self.root / "models/Qwen3.5-4B-4bit"
        self.default_model.mkdir(parents=True)
        # These are explicit test strings, not host credentials.
        for name in (".device-token", ".parent-token"):
            path = self.root / name
            path.write_text("test-only-token-never-used-for-auth")
            path.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def configure(self, value):
        (self.root / "gateway-config.json").write_text(json.dumps(value))

    def invoke(self, environment=None):
        with patch.object(sys, "argv", ["launch", "--root", str(self.root)]), \
             patch.dict(os.environ, environment or {}, clear=True), \
             patch("companion.launch.os.chdir") as chdir, \
             patch("companion.launch.os.execv") as execute:
            launch.main()
            chdir.assert_called_once_with(self.root.resolve())
            execute.assert_called_once_with("/usr/bin/caffeinate", ["caffeinate", "-i", "-s", sys.executable, "-m", "companion.server"])
            return dict(os.environ)

    def assert_rejected(self):
        with patch.object(sys, "argv", ["launch", "--root", str(self.root)]), \
             patch.dict(os.environ, {}, clear=True), \
             patch("companion.launch.os.chdir") as chdir, \
             patch("companion.launch.os.execv") as execute:
            with self.assertRaises(SystemExit):
                launch.main()
            execute.assert_not_called()
            chdir.assert_not_called()  # Bad configuration fails before credential loading.

    def test_default_keeps_4b_and_forces_offline_environment(self):
        env = self.invoke({"PUPPY_MODEL_PATH": "remote/inherited-model", "HF_HUB_OFFLINE": "0",
                           "TRANSFORMERS_OFFLINE": "0", "HF_HUB_DISABLE_TELEMETRY": "0"})
        self.assertEqual(env["PUPPY_MODEL_PATH"], str(self.default_model.resolve()))
        for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            self.assertEqual(env[variable], "1")
        self.assertEqual(env["PUPPY_MODEL_PRELOAD"], "1")
        self.assertNotIn("PUPPY_APPLE_ASR_BINARY", env)

    def test_apple_asr_requires_explicit_flag_and_local_executable(self):
        binary = self.root / "bin/apple-speech-probe"
        binary.parent.mkdir()
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o700)
        self.assertNotIn("PUPPY_APPLE_ASR_BINARY", self.invoke({"PUPPY_APPLE_ASR_BINARY": "/inherited"}))
        self.configure({"PUPPY_APPLE_ASR_ENABLED": True})
        self.assertEqual(self.invoke()["PUPPY_APPLE_ASR_BINARY"], str(binary.resolve()))
        binary.chmod(0o600)
        self.assert_rejected()

    def test_hermes_requires_explicit_enablement_and_the_worker(self):
        inherited = {"PUPPY_HERMES_ENABLED": "1", "PUPPY_HERMES_TASK_DIR": "/inherited"}
        env = self.invoke(inherited)
        self.assertEqual(env["PUPPY_HERMES_ENABLED"], "0")
        self.assertEqual(env["PUPPY_HERMES_TASK_DIR"], str(self.root.resolve() / "hermes-voice-tasks"))
        self.configure({"PUPPY_HERMES_ENABLED": True})
        self.assert_rejected()
        worker = self.root / "scripts/run_hermes_voice_task.py"
        worker.parent.mkdir()
        worker.write_text("# isolated worker placeholder for launch validation\n")
        self.assertEqual(self.invoke()["PUPPY_HERMES_ENABLED"], "1")

    def test_hermes_rejects_nonboolean_config(self):
        for flag in (None, "true", 1, []):
            with self.subTest(flag=flag):
                self.configure({"PUPPY_HERMES_ENABLED": flag})
                self.assert_rejected()

    def test_stationary_play_disables_hermes_even_when_configured(self):
        self.configure({"PUPPY_STATIONARY_PLAY": True, "PUPPY_HERMES_ENABLED": True})
        env = self.invoke()
        self.assertEqual(env["PUPPY_STATIONARY_PLAY"], "1")
        self.assertEqual(env["PUPPY_HERMES_ENABLED"], "0")
        self.configure({"PUPPY_STATIONARY_PLAY": "true"})
        self.assert_rejected()

    def test_apple_asr_rejects_bad_flag_missing_file_or_symlink(self):
        for value in (None, 1, "true", [], {}):
            with self.subTest(value=value):
                self.configure({"PUPPY_APPLE_ASR_ENABLED": value})
                self.assert_rejected()
        self.configure({"PUPPY_APPLE_ASR_ENABLED": True})
        self.assert_rejected()
        target = self.root / "real-probe"
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o700)
        binary = self.root / "bin/apple-speech-probe"
        binary.parent.mkdir()
        binary.symlink_to(target)
        self.assert_rejected()

    def test_preload_opt_out_comes_from_local_config(self):
        self.configure({"PUPPY_MODEL_PRELOAD": False})
        self.assertEqual(self.invoke()["PUPPY_MODEL_PRELOAD"], "0")

    def test_minimax_requires_private_key_and_explicit_provider(self):
        self.configure({"PUPPY_TEXT_PROVIDER": "minimax", "PUPPY_MINIMAX_REGION": "cn",
                        "PUPPY_MODEL_PRELOAD": False})
        self.assert_rejected()
        key = self.root / ".minimax-token"
        key.write_text("fixture-minimax-token-for-local-tests-only")
        key.chmod(0o644)
        self.assert_rejected()
        key.chmod(0o600)
        env = self.invoke()
        self.assertEqual(env["PUPPY_TEXT_PROVIDER"], "minimax")
        self.assertEqual(env["PUPPY_MINIMAX_API_KEY_FILE"], str(key.resolve()))
        self.assertEqual(env["PUPPY_MINIMAX_REGION"], "cn")
        self.assertEqual(env["PUPPY_MODEL_PRELOAD"], "0")
        self.configure({"PUPPY_TEXT_PROVIDER": "local"})
        self.assertNotIn("PUPPY_MINIMAX_API_KEY_FILE", self.invoke({"PUPPY_MINIMAX_API_KEY_FILE": "inherited"}))

    def test_preload_flag_must_be_boolean(self):
        for value in (None, 0, "false", [], {}):
            with self.subTest(value=value):
                self.configure({"PUPPY_MODEL_PRELOAD": value})
                self.assert_rejected()

    def test_relative_model_path_uses_root_and_existing_config_fields_survive(self):
        target = self.root / "alternate models/Qwen-9B"
        target.mkdir(parents=True)
        self.configure({"PUPPY_MODEL_PATH": "alternate models/Qwen-9B", "PUPPY_PORT": 18789,
                        "PUPPY_REMOTE_CONFIG": "/test-only/remote.json"})
        env = self.invoke()
        self.assertEqual(env["PUPPY_MODEL_PATH"], str(target.resolve()))
        self.assertEqual(env["PUPPY_PORT"], "18789")
        self.assertEqual(env["PUPPY_REMOTE_CONFIG"], "/test-only/remote.json")

    def test_absolute_directory_is_allowed_and_resolved(self):
        with tempfile.TemporaryDirectory(prefix="puppy-local-model-") as model:
            self.configure({"PUPPY_MODEL_PATH": model})
            self.assertEqual(self.invoke()["PUPPY_MODEL_PATH"], str(Path(model).resolve()))

    def test_existing_local_model_symlink_resolves_to_local_directory(self):
        alias = self.root / "selected-model"
        alias.symlink_to(self.default_model, target_is_directory=True)
        self.configure({"PUPPY_MODEL_PATH": "selected-model"})
        self.assertEqual(self.invoke()["PUPPY_MODEL_PATH"], str(self.default_model.resolve()))

    def test_missing_default_directory_is_rejected(self):
        self.default_model.rmdir()
        self.assert_rejected()

    def test_missing_remote_file_and_dangling_paths_are_rejected(self):
        plain_file = self.root / "model-file"
        plain_file.write_text("not a model directory")
        (self.root / "dangling").symlink_to(self.root / "missing", target_is_directory=True)
        for value in ("missing", "mlx-community/NoLocalModel", "https://example.invalid/model",
                      "model-file", "dangling", "\x00"):
            with self.subTest(value=value):
                self.configure({"PUPPY_MODEL_PATH": value})
                self.assert_rejected()

    def test_model_path_requires_nonempty_string(self):
        for value in (None, True, 1, [], {}, "", "   "):
            with self.subTest(value=value):
                self.configure({"PUPPY_MODEL_PATH": value})
                self.assert_rejected()

    def test_malformed_and_nonobject_configuration_is_rejected(self):
        for raw in ("{bad json", "[]", "null", '"text"'):
            with self.subTest(raw=raw):
                (self.root / "gateway-config.json").write_text(raw)
                self.assert_rejected()

    def test_model_load_explicitly_disables_remote_code(self):
        loader = Mock(return_value=(object(), object()))
        module = types.ModuleType("mlx_vlm")
        module.load = loader
        with patch.dict(sys.modules, {"mlx_vlm": module}):
            LocalModel(str(self.default_model))
        loader.assert_called_once_with(str(self.default_model), trust_remote_code=False)


if __name__ == "__main__":
    unittest.main()
