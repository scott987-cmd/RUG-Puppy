"""Connect the opt-in file bridge to the existing local gateway."""

import asyncio
import json
import logging
import time
from pathlib import Path

from .remote import RemoteBridge, mirror_once, read_regular


class RemoteRuntime:
    def __init__(self, config_path: str, hub, store, stop_handler, action_handler, describe_handler=None):
        config = json.loads(read_regular(Path(config_path).expanduser(), 65536))
        retention_days = config.get("remote_media_retention_days")
        if retention_days is not None and (type(retention_days) is not int or not 1 <= retention_days <= 30):
            raise ValueError("remote_media_retention_days must be 1–30 or omitted")
        self.media_retention_days = retention_days
        self.next_media_retention_at = 0.0
        self.root = Path(config["local_root"]).expanduser().resolve()
        self.cloud_root = Path(config["cloud_root"]).expanduser().resolve() if config.get("cloud_root") else None
        if retention_days is not None and self.cloud_root is None:
            raise ValueError("remote_media_retention_days requires cloud_root")
        key_file = Path(config["server_private_key_file"]).expanduser().resolve()
        if self.cloud_root and any(path == self.cloud_root or self.cloud_root in path.parents
                                   for path in (self.root, key_file)):
            raise ValueError("Local command ledger and private key must stay outside iCloud")
        if "Mobile Documents" in key_file.parts:
            raise ValueError("Private key must stay outside iCloud")
        if key_file.stat().st_mode & 0o077:
            raise ValueError("Server signing key must be accessible only to its owner")
        key_data = json.loads(read_regular(key_file, 4096))
        self.hub, self.store = hub, store
        self.stop_handler, self.action_handler = stop_handler, action_handler
        self.describe_handler = describe_handler
        self.bridge = RemoteBridge(self.root, config["server_id"], set(config["device_ids"]),
                                   config["parent_public_keys"], key_data["private_key"], self.execute)
        self.tasks = []
        self.last_error = None
        self.logger = logging.getLogger("uvicorn.error")

    def device_status(self, device_id):
        status = self.hub.status.get(device_id, {})
        return {"online": device_id in self.hub.connections,
                "observed_at": status.get("observed_at"),
                "volume": status.get("audio_speaker", {}).get("volume"),
                "battery": status.get("battery"), "pet_name": status.get("pet_name")}

    async def execute(self, command):
        if command.kind == "status":
            return {"ok": True, "status": self.device_status(command.device_id)}
        if command.kind == "action":
            return await self.action_handler(command.device_id, command.params["name"])
        if command.kind == "stop":
            return await self.stop_handler(command.device_id)
        if command.kind in {"photo", "patrol_once"}:
            receipt = await self.hub.command(command.device_id, "photo")
            if not receipt.get("ok"):
                return receipt
            photo_id = receipt["photo_id"]
            # Hub only confirms photos after JPEG decode and durable local save.
            path = self.store.photo_file(photo_id)
            if not path:
                return {"ok": False, "error": "photo_not_saved"}
            description = None
            if command.kind == "patrol_once" and self.describe_handler is not None:
                try:
                    description = await self.describe_handler(photo_id, "请简短描述此刻照片里看得清的环境，看不清就说明。")
                except Exception as exc:
                    self.logger.warning("Patrol description unavailable: %s", type(exc).__name__)
                    description = {"ok": False, "error": "description_unavailable"}
            media = self.bridge.export_photo(command, read_regular(Path(path), 512000))
            saved = next((p for p in self.store.photos(command.device_id) if p["photo_id"] == photo_id), None)
            return {"ok": True, "photo_id": photo_id, "media": media,
                    "captured_at": saved["captured_at"] if saved else None,
                    "time_source": "gateway_receive",
                    "scope": "fixed_position_single_photo", "description": description}
        return {"ok": False, "error": "unsupported_command"}

    async def consume(self):
        while True:
            try:
                await self.bridge.process_once()
                for device_id in self.bridge.device_ids:
                    self.bridge.publish_status(device_id, self.device_status(device_id))
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = type(exc).__name__
                self.logger.warning("Remote queue iteration failed: %s", self.last_error)
            await asyncio.sleep(2)

    async def mirror(self):
        delay = 5
        while True:
            try:
                retention_days = (self.media_retention_days if time.time() >= self.next_media_retention_at
                                  else None)
                result = await mirror_once(self.root, self.cloud_root, retention_days=retention_days)
                if retention_days is not None and "retention" in result:
                    self.next_media_retention_at = time.time() + 3600
                unavailable = {"mirror_timeout", "mirror_worker_failed", "commands_unreadable",
                               "mirror_worker_invalid_output"}
                stalled = any(error in unavailable for error in result.get("errors", []))
                delay = min(300, max(30, delay * 2)) if stalled else 5
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.warning("Remote mirror iteration failed: %s", type(exc).__name__)
                delay = min(300, max(30, delay * 2))
            await asyncio.sleep(delay)

    def start(self):
        self.tasks.append(asyncio.create_task(self.consume()))
        if self.cloud_root:
            self.tasks.append(asyncio.create_task(self.mirror()))

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.bridge.close()

    def status(self):
        try:
            mirror = json.loads(read_regular(self.root / "mirror-state.json", 65536))
        except (OSError, ValueError):
            mirror = None
        return {"enabled": True, "server_id": self.bridge.server_id,
                "cloud_configured": self.cloud_root is not None,
                "remote_media_retention_days": self.media_retention_days,
                "last_error": self.last_error, "mirror": mirror, "checked_at": time.time()}
