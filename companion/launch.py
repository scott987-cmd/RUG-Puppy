"""Launch the home gateway without putting credentials in a launchd plist."""

import argparse
import json
import os
import sys
from pathlib import Path

from .tls import tls_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    config = {}
    config_path = root / "gateway-config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (OSError, ValueError) as exc:
            raise SystemExit("Unable to read a valid gateway-config.json") from exc
        if not isinstance(config, dict):
            raise SystemExit("gateway-config.json must contain a JSON object")
    model_value = config.get("PUPPY_MODEL_PATH", "models/Qwen3.5-4B-4bit")
    if not isinstance(model_value, str) or not model_value.strip():
        raise SystemExit("PUPPY_MODEL_PATH must be an existing local model directory")
    model_preload = config.get("PUPPY_MODEL_PRELOAD", True)
    if type(model_preload) is not bool:
        raise SystemExit("PUPPY_MODEL_PRELOAD must be a boolean")
    text_provider = config.get("PUPPY_TEXT_PROVIDER", "local")
    if text_provider not in ("local", "minimax"):
        raise SystemExit("PUPPY_TEXT_PROVIDER must be local or minimax")
    minimax_region = config.get("PUPPY_MINIMAX_REGION", "cn")
    if minimax_region not in ("cn", "global"):
        raise SystemExit("PUPPY_MINIMAX_REGION must be cn or global")
    minimax_key_file = root / ".minimax-token"
    if text_provider == "minimax":
        if (minimax_key_file.is_symlink() or not minimax_key_file.is_file()
                or minimax_key_file.stat().st_uid != os.getuid()
                or minimax_key_file.stat().st_mode & 0o077
                or len(minimax_key_file.read_text().strip()) < 24):
            raise SystemExit("MiniMax key file must exist with owner-only permissions")
    apple_asr_enabled = config.get("PUPPY_APPLE_ASR_ENABLED", False)
    if type(apple_asr_enabled) is not bool:
        raise SystemExit("PUPPY_APPLE_ASR_ENABLED must be a boolean")
    hermes_enabled = config.get("PUPPY_HERMES_ENABLED", False)
    if type(hermes_enabled) is not bool:
        raise SystemExit("PUPPY_HERMES_ENABLED must be a boolean")
    stationary_play = config.get("PUPPY_STATIONARY_PLAY", False)
    if type(stationary_play) is not bool:
        raise SystemExit("PUPPY_STATIONARY_PLAY must be a boolean")
    if hermes_enabled and not stationary_play and not (root / "scripts/run_hermes_voice_task.py").is_file():
        raise SystemExit("Enabled Hermes requires the isolated voice task worker")
    apple_asr_binary = root / "bin/apple-speech-probe"
    if apple_asr_enabled and (apple_asr_binary.is_symlink() or not apple_asr_binary.is_file()
                              or not os.access(apple_asr_binary, os.X_OK)):
        raise SystemExit("Enabled Apple ASR requires an executable bin/apple-speech-probe")
    try:
        model_path = Path(model_value).expanduser()
        if not model_path.is_absolute():
            model_path = root / model_path
        model_path = model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise ValueError("not a directory")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit("PUPPY_MODEL_PATH must be an existing local model directory") from exc
    try:
        tls = tls_options(config, root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit("TLS configuration is invalid; the server will not fall back to HTTP") from exc
    os.chdir(root)
    for filename, variable in ((".device-token", "PUPPY_DEVICE_TOKEN"),
                               (".parent-token", "PUPPY_PARENT_TOKEN")):
        path = root / filename
        if path.stat().st_mode & 0o077:
            raise SystemExit(f"Restrict {filename} permissions to its owner")
        token = path.read_text().strip()
        if len(token) < 24:
            raise SystemExit(f"Missing or invalid {filename}")
        os.environ[variable] = token
    os.environ["PUPPY_DATA_DIR"] = str(root / "data")
    os.environ["PUPPY_STATIONARY_PLAY"] = "1" if stationary_play else "0"
    os.environ["PUPPY_HERMES_ENABLED"] = "1" if hermes_enabled and not stationary_play else "0"
    os.environ["PUPPY_HERMES_TASK_DIR"] = str(root / "hermes-voice-tasks")
    os.environ["PUPPY_MODEL_PATH"] = str(model_path)
    os.environ["PUPPY_TEXT_PROVIDER"] = text_provider
    if text_provider == "minimax":
        os.environ["PUPPY_MINIMAX_API_KEY_FILE"] = str(minimax_key_file)
        os.environ["PUPPY_MINIMAX_REGION"] = minimax_region
    else:
        os.environ.pop("PUPPY_MINIMAX_API_KEY_FILE", None)
        os.environ.pop("PUPPY_MINIMAX_REGION", None)
    # The MacBook stays at home: load the local model in the background as soon
    # as the gateway starts, while leaving control/status sockets responsive.
    os.environ["PUPPY_MODEL_PRELOAD"] = "1" if model_preload else "0"
    for variable, option in (("PUPPY_TLS_CERT_PATH", "ssl_certfile"), ("PUPPY_TLS_KEY_PATH", "ssl_keyfile")):
        if tls:
            os.environ[variable] = tls[option]
        else:
            os.environ.pop(variable, None)
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[variable] = "1"
    os.environ["PUPPY_WHISPER_PATH"] = str(root / "models/whisper-small-mlx")
    if apple_asr_enabled:
        os.environ["PUPPY_APPLE_ASR_BINARY"] = str(apple_asr_binary)
    else:
        os.environ.pop("PUPPY_APPLE_ASR_BINARY", None)
    if (root / "models/identity").is_dir():
        os.environ["PUPPY_IDENTITY_MODELS"] = str(root / "models/identity")
    gesture_python = root / ".gesture-venv/bin/python"
    gesture_model = root / "models/gestures/gesture_recognizer-float16-v1.task"
    if gesture_python.is_file() and gesture_model.is_file():
        os.environ["PUPPY_GESTURE_PYTHON"] = str(gesture_python)
        os.environ["PUPPY_GESTURE_MODEL"] = str(gesture_model)
    for key in ("PUPPY_REMOTE_CONFIG", "PUPPY_PORT"):
        if key in config:
            os.environ[key] = str(config[key])
    # Assertions follow the gateway process lifetime. This cannot override a
    # physical lid-close sleep; clamshell use still needs a supported setup.
    os.execv("/usr/bin/caffeinate", ["caffeinate", "-i", "-s", sys.executable, "-m", "companion.server"])


if __name__ == "__main__":
    main()
