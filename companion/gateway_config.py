"""Explicit parent gateway provisioning; saving never restarts the device."""
import asyncio
from contextlib import contextmanager
import ipaddress
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlsplit


MAX_VERSION = 2_147_483_647


class GatewayConfigError(Exception):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


def validate_configuration(url, token):
    if not isinstance(url, str) or not 1 <= len(url) <= 320 or not url.isascii():
        raise GatewayConfigError("网关地址须为不超过 320 字符的 ASCII URL", 422)
    try:
        parsed = urlsplit(url)
        if (not url.startswith(("ws://", "wss://")) or parsed.scheme not in {"ws", "wss"}
                or parsed.username is not None or parsed.password is not None
                or parsed.path != "/v1/audio" or parsed.query or parsed.fragment
                or any(ord(c) <= 32 or ord(c) == 127 for c in url)
                or "?" in url or "#" in url or not parsed.hostname or parsed.port is None
                or not 1 <= parsed.port <= 65535 or ":" in parsed.hostname
                or not re.fullmatch(r"[0-9]{1,5}", parsed.netloc.rsplit(":", 1)[-1])):
            raise ValueError()
        host = parsed.hostname
        if len(host) > 253 or not all(re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                                     for label in host.split(".")):
            raise ValueError()
        if re.fullmatch(r"[0-9.]+", host):
            ipaddress.IPv4Address(host)
    except ValueError as error:
        raise GatewayConfigError("地址格式须为 ws://主机:端口/v1/audio 或 wss://主机:端口/v1/audio；不能含凭据、查询或 IPv6", 422) from error
    if not isinstance(token, str) or not 24 <= len(token) <= 256 or any(not 33 <= ord(c) <= 126 for c in token):
        raise GatewayConfigError("配对令牌须为 24 至 256 个无空格的可见 ASCII 字符", 422)
    return {"url": url, "token": token}


class GatewayConfigService:
    def __init__(self, database, *, command, get_status, current_token, timeout=35):
        self.database = Path(database)
        self.command, self.get_status, self.current_token = command, get_status, current_token
        self.timeout = timeout
        self._locks = {}
        self._sending = set()

    @staticmethod
    def _device(device):
        if not isinstance(device, str) or not re.fullmatch(r"[A-Za-z0-9:_.-]{1,64}", device):
            raise GatewayConfigError("无效设备 ID", 422)
        return device

    @contextmanager
    def _db(self):
        self.database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.database.touch(mode=0o600, exist_ok=True)
        self.database.chmod(0o600)
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS gateway_configurations (
                device TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL, attempts INTEGER NOT NULL, updated_at REAL NOT NULL)""")
            db.commit()
            yield db
        finally:
            db.close()

    async def _write(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        return task.result()

    def _read(self, device):
        with self._db() as db:
            row = db.execute("SELECT * FROM gateway_configurations WHERE device=?", (device,)).fetchone()
        return dict(row) if row else None

    def _state(self, device, *, required=False):
        raw = self.get_status(device)
        version = raw.get("gateway_config_version") if isinstance(raw, dict) else None
        supported = type(version) is int and 0 <= version <= MAX_VERSION
        if required and raw is None:
            raise GatewayConfigError("Puppy 当前不在线，尚未发送配置", 503)
        if required and not supported:
            raise GatewayConfigError("请先升级支持网关配置的固件并等待状态", 409)
        # Read only the documented non-secret state fields.
        return {"online": raw is not None, "supported": supported,
                "version": version if supported else None,
                "url": raw.get("gateway_config_url") if supported else None,
                "override": raw.get("gateway_config_override") is True if supported else False,
                "restart_required": raw.get("gateway_config_restart_required") is True if supported else False}

    async def status(self, device):
        self._device(device)
        row = await asyncio.to_thread(self._read, device)
        latest = None
        if row:
            payload = json.loads(row["payload"])
            state = row["state"]
            if state == "sending" and device not in self._sending:
                state = "unknown"
            latest = {"version": row["version"], "url": payload["url"], "state": state,
                      "attempts": row["attempts"], "has_token": True}
        return {"device_id": device, "device": self._state(device), "latest": latest,
                "save_restarts_device": False}

    def _allocate(self, device, baseline, configuration):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version FROM gateway_configurations WHERE device=?", (device,)).fetchone()
            version = max(baseline, row[0] if row else 0) + 1
            if version > MAX_VERSION:
                raise GatewayConfigError("配置版本已达上限")
            payload = {"version": version, **configuration}
            db.execute("""INSERT INTO gateway_configurations VALUES (?,?,?,'prepared',0,?)
                ON CONFLICT(device) DO UPDATE SET version=excluded.version,payload=excluded.payload,
                state='prepared',attempts=0,updated_at=excluded.updated_at""",
                (device, version, json.dumps(payload), time.time()))
            db.commit()
        return payload

    def _record(self, device, version, state):
        with self._db() as db:
            with db:
                db.execute("""UPDATE gateway_configurations SET state=?,attempts=attempts+?,updated_at=?
                    WHERE device=? AND version=?""", (state, int(state == "sending"), time.time(), device, version))

    async def configure(self, device, url, token=None):
        self._device(device)
        configuration = validate_configuration(url, self.current_token() if token is None else token)
        async with self._locks.setdefault(device, asyncio.Lock()):
            state = self._state(device, required=True)
            payload = await self._write(self._allocate, device, state["version"], configuration)
            await self._send(device, payload)
        return await self.status(device)

    async def retry(self, device, version):
        self._device(device)
        if type(version) is not int or not 1 <= version <= MAX_VERSION:
            raise GatewayConfigError("无效配置版本", 422)
        async with self._locks.setdefault(device, asyncio.Lock()):
            row = await asyncio.to_thread(self._read, device)
            if row is None or row["version"] != version:
                raise GatewayConfigError("配置版本已经变化，请刷新")
            await self._send(device, json.loads(row["payload"]))
        return await self.status(device)

    async def _send(self, device, payload):
        version = payload["version"]
        state = self._state(device, required=True)
        if state["version"] > version:
            await self._write(self._record, device, version, "conflict")
            raise GatewayConfigError("设备已有更新配置，请明确保存新配置")
        await self._write(self._record, device, version, "sending")
        self._sending.add(device)
        try:
            receipt = await asyncio.wait_for(self.command(device, "set_gateway_config", **payload), self.timeout)
            saved = (isinstance(receipt, dict) and receipt.get("ok") is True
                     and receipt.get("phase") == "saved" and receipt.get("executed") is False
                     and receipt.get("restart_required") is True
                     and type(receipt.get("version")) is int and receipt["version"] == version)
            await self._write(self._record, device, version, "saved" if saved else "unknown")
        except asyncio.CancelledError:
            await self._write(self._record, device, version, "unknown")
            raise
        except Exception:
            # Do not reflect a peer's receipt or exception, which may echo secrets.
            await self._write(self._record, device, version, "unknown")
        finally:
            self._sending.discard(device)
