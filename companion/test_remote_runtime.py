import base64
import json
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image

from .remote import atomic_write, json_bytes, sign_payload, verify_payload
from .remote_runtime import RemoteRuntime
from .store import Store


def keys():
    key = Ed25519PrivateKey.generate()
    return {"private_key": base64.b64encode(key.private_bytes_raw()).decode(),
            "public_key": base64.b64encode(key.public_key().public_bytes_raw()).decode()}


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-runtime-")
        self.root = Path(self.temp.name)
        self.server_key, self.parent_key = keys(), keys()
        key_path = self.root / "server-key.json"
        atomic_write(key_path, json_bytes(self.server_key))
        self.config = {"server_id": "home", "device_ids": ["puppy"],
                       "parent_public_keys": {"parent": self.parent_key["public_key"]},
                       "server_private_key_file": str(key_path), "local_root": str(self.root / "spool")}
        config_path = self.root / "config.json"
        atomic_write(config_path, json_bytes(self.config))
        self.hub = type("Hub", (), {})()
        self.hub.connections = {"puppy": object()}
        self.hub.status = {"puppy": {"audio_speaker": {"volume": 0}, "network": {"ssid": "private"},
                                     "observed_at": 123, "pet_name": "豆豆"}}
        self.hub.command = AsyncMock()
        self.stop, self.action = AsyncMock(return_value={"ok": True}), AsyncMock(return_value={"ok": True})
        self.store = Store(self.root / "test.sqlite3")
        self.runtime = RemoteRuntime(str(config_path), self.hub, self.store, self.stop, self.action)

    async def asyncTearDown(self):
        await self.runtime.close()
        self.temp.cleanup()

    async def submit(self, kind, params=None):
        command = {"version": 1, "request_id": str(uuid.uuid4()), "source_id": "parent", "server_id": "home",
                   "device_id": "puppy", "kind": kind, "params": params or {}, "created_at": int(time.time()),
                   "expires_at": int(time.time()) + 120}
        atomic_write(self.runtime.root / "inbox" / (command["request_id"] + ".json"),
                     json_bytes(sign_payload(command, self.parent_key["private_key"])))
        await self.runtime.bridge.process_once()
        envelope = json.loads((self.runtime.root / "outbox/receipts" / (command["request_id"] + ".json")).read_bytes())
        return verify_payload(envelope, self.server_key["public_key"])

    async def test_signed_status_preserves_silence_and_omits_private_network(self):
        receipt = await self.submit("status")
        self.assertEqual(receipt["state"], "succeeded")
        self.assertEqual(receipt["result"]["status"]["volume"], 0)
        self.assertNotIn("private", json.dumps(receipt))
        self.hub.command.assert_not_awaited()

    async def test_stop_uses_shared_cancel_handler(self):
        receipt = await self.submit("stop")
        self.assertEqual(receipt["state"], "succeeded")
        self.stop.assert_awaited_once_with("puppy")
        self.hub.command.assert_not_awaited()

    async def test_fixed_patrol_exports_decodable_saved_photo_and_no_motion(self):
        image_path = self.root / "image.jpg"
        Image.new("RGB", (16, 16), "blue").save(image_path)
        self.store.add_photo("photo-1", "puppy", str(image_path))
        self.hub.command.return_value = {"ok": True, "photo_id": "photo-1"}
        self.runtime.describe_handler = AsyncMock(return_value={"reply": "蓝色画面。", "uncertain": False})
        receipt = await self.submit("patrol_once")
        result = receipt["result"]
        self.assertEqual(receipt["state"], "succeeded")
        self.assertEqual(result["scope"], "fixed_position_single_photo")
        self.assertEqual(result["description"]["reply"], "蓝色画面。")
        self.assertEqual(result["time_source"], "gateway_receive")
        path = self.runtime.root / "outbox" / result["media"]["file"]
        with Image.open(path) as picture:
            picture.load()
            self.assertEqual(picture.size, (16, 16))
        self.hub.command.assert_awaited_once_with("puppy", "photo")
        self.action.assert_not_awaited()

    async def test_description_failure_preserves_requested_photo(self):
        image_path = self.root / "image.jpg"
        Image.new("RGB", (16, 16), "blue").save(image_path)
        self.store.add_photo("photo-2", "puppy", str(image_path))
        self.hub.command.return_value = {"ok": True, "photo_id": "photo-2"}
        self.runtime.describe_handler = AsyncMock(side_effect=RuntimeError("local model unavailable"))
        receipt = await self.submit("patrol_once")
        self.assertEqual(receipt["state"], "succeeded")
        self.assertEqual(receipt["result"]["description"]["error"], "description_unavailable")
        self.assertTrue((self.runtime.root / "outbox" / receipt["result"]["media"]["file"]).is_file())

    async def test_remote_media_retention_requires_explicit_valid_days_and_cloud(self):
        path = self.root / "retention-config.json"
        for value in (True, 0, 31, "7"):
            config = {**self.config, "local_root": str(self.root / "other-spool"),
                      "remote_media_retention_days": value}
            atomic_write(path, json_bytes(config))
            with self.assertRaises(ValueError):
                RemoteRuntime(str(path), self.hub, self.store, self.stop, self.action)
        config = {**self.config, "local_root": str(self.root / "other-spool"),
                  "remote_media_retention_days": 7}
        atomic_write(path, json_bytes(config))
        with self.assertRaisesRegex(ValueError, "cloud_root"):
            RemoteRuntime(str(path), self.hub, self.store, self.stop, self.action)
        cloud = self.root / "PuppyCompanion"
        cloud.mkdir()
        config["cloud_root"] = str(cloud)
        atomic_write(path, json_bytes(config))
        candidate = RemoteRuntime(str(path), self.hub, self.store, self.stop, self.action)
        try:
            self.assertEqual(candidate.status()["remote_media_retention_days"], 7)
        finally:
            await candidate.close()


if __name__ == "__main__":
    unittest.main()
