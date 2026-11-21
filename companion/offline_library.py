"""Parent-selected immutable offline libraries. Synchronization never executes.

Callbacks: async command(device_id, 'set_offline_actions', version=..., actions=...)
sync get_status(device_id) -> connected device status copy, or None when offline;
sync approved_actions_snapshot() -> {saved_name: saved_steps}. No polling/replay.
All SQLite and action-catalog reads run off the event loop.
"""
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
import time
import unicodedata


MAX_VERSION = 2_147_483_647
ALLOWED_ACTIONS = frozenset({"wave", "sit", "stretch", "hug"})


class LibraryError(Exception):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


def _device(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9:_.-]{1,64}", value):
        raise LibraryError("无效设备 ID", 422)
    return value


def _name(name):
    if not isinstance(name, str) or not name or any(unicodedata.category(c) == "Cc" for c in name):
        raise ValueError("动作名称须为非空文字，不能含控制字符")
    try:
        if len(name.encode("utf-8")) > 64:
            raise ValueError("动作名称的 UTF-8 长度不能超过 64 字节")
    except UnicodeEncodeError as error:
        raise ValueError("动作名称必须是有效 UTF-8") from error
    return name


def normalize_action(name, raw):
    """Convert parent Store's optional None/zero fields to firmware canonical data."""
    _name(name)
    if not isinstance(raw, list) or not 1 <= len(raw) <= 6:
        raise ValueError("离线动作须为 1 至 6 步")
    steps = []
    for source in raw:
        if not isinstance(source, dict) or set(source) - {"action", "wait_ms"}:
            raise ValueError("动作步骤含未知参数")
        action, wait = source.get("action"), source.get("wait_ms", 0)
        if action is not None and (not isinstance(action, str) or action not in ALLOWED_ACTIONS):
            raise ValueError("离线库只支持挥手、坐下、伸懒腰和抱抱")
        if type(wait) is not int or not 0 <= wait <= 3000:
            raise ValueError("每步等待须为 0 至 3000 毫秒整数")
        if action is None and wait == 0:
            raise ValueError("只等待的步骤必须大于 0 毫秒")
        step = {}
        if action is not None:
            step["action"] = action
        if wait:
            step["wait_ms"] = wait
        steps.append(step)
    return {"name": name, "steps": steps}


def _canonical(version, actions):
    payload = {"version": version, "actions": actions}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 4096:
        raise LibraryError("离线动作库内容过大", 422)
    envelope = {"id": "x" * 64, "type": "set_offline_actions", **payload}
    if len(json.dumps(envelope, ensure_ascii=False).encode("utf-8")) > 8192:
        raise LibraryError("离线动作库消息过大", 422)
    return encoded


class OfflineLibraryService:
    def __init__(self, database, *, command, get_status, approved_actions_snapshot,
                 timeout=35, clock=time.time):
        self.database = Path(database).expanduser().absolute()
        self.command, self.get_status = command, get_status
        self.approved_actions_snapshot = approved_actions_snapshot
        self.timeout, self.clock = timeout, clock
        if timeout <= 0:
            raise ValueError("同步超时必须为正数")
        self._locks = {}
        self._sending = set()

    async def _write(self, function, *args, **kwargs):
        # SQLite work cannot be killed midway. Keep the device lock until an
        # outstanding write has settled, including caller cancellation.
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                # Retrieve the finished task's result below, preserving either
                # caller cancellation or the database error.
                break
        if cancelled:
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        return task.result()

    @contextmanager
    def _db(self):
        self.database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.database.exists():
            self.database.touch(mode=0o600)
        self.database.chmod(0o600)
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS library_devices (
                    device_id TEXT PRIMARY KEY, high_water INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS library_versions (
                    device_id TEXT NOT NULL, version INTEGER NOT NULL,
                    payload TEXT NOT NULL, state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, error TEXT, receipt TEXT,
                    PRIMARY KEY(device_id, version));
            """)
            yield db
        finally:
            db.close()

    def _catalog(self):
        approved = self.approved_actions_snapshot()
        if not isinstance(approved, dict):
            raise LibraryError("家长动作列表暂不可用", 503)
        catalog, valid = [], {}
        for name, steps in approved.items():
            try:
                action = normalize_action(name, steps)
                valid[name] = action
                catalog.append({"name": name, "eligible": True, "step_count": len(action["steps"])})
            except ValueError as error:
                # Corrupt local names aren't reflected into a public response.
                if isinstance(name, str) and len(name) <= 128 and not any(unicodedata.category(c) == "Cc" for c in name):
                    try:
                        name.encode("utf-8")
                    except UnicodeEncodeError:
                        continue
                    catalog.append({"name": name, "eligible": False, "reason": str(error)})
        return catalog, valid

    def _device_status(self, device_id, *, required=False):
        status = self.get_status(device_id)
        if status is None:
            if required:
                raise LibraryError("Puppy 当前不在线，尚未发送同步请求", 503)
            return {"online": False, "supported": False, "version": None, "action_count": None}
        if not isinstance(status, dict):
            raise LibraryError("设备状态暂不可用", 503)
        version, count = status.get("offline_library_version"), status.get("offline_library_count")
        valid = (type(version) is int and 0 <= version <= MAX_VERSION
                 and type(count) is int and 0 <= count <= 3)
        if required and not valid:
            raise LibraryError("请先升级支持离线动作库的固件并等待设备状态", 409)
        return {"online": True, "supported": valid, "version": version if valid else None,
                "action_count": count if valid else None}

    def _history(self, device_id):
        with self._db() as db:
            rows = db.execute("SELECT * FROM library_versions WHERE device_id=? ORDER BY version DESC LIMIT 10",
                              (device_id,)).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["actions"] = json.loads(item.pop("payload"))["actions"]
            item["receipt"] = json.loads(item["receipt"]) if item["receipt"] else None
            # A process restart never replays; a formerly sending record is uncertain.
            if item["state"] == "sending" and (device_id, item["version"]) not in self._sending:
                item.update(state="unknown", error="服务中断，保存结果未确认；可重试同一版本")
            output.append(item)
        return output

    async def status(self, device_id):
        _device(device_id)
        device = self._device_status(device_id)
        (catalog, _), history = await asyncio.gather(asyncio.to_thread(self._catalog),
                                                    asyncio.to_thread(self._history, device_id))
        return {"device_id": device_id, "device": device, "catalog": catalog,
                "latest": history[0] if history else None, "history": history,
                "sync_executes_actions": False}

    def _allocate(self, device_id, baseline, actions):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT high_water FROM library_devices WHERE device_id=?", (device_id,)).fetchone()
            version = max(baseline, row[0] if row else 0) + 1
            if version > MAX_VERSION:
                raise LibraryError("设备离线库版本已达上限，不能继续分配")
            payload = _canonical(version, actions)
            now = self.clock()
            db.execute("INSERT INTO library_devices VALUES (?, ?) ON CONFLICT(device_id) DO UPDATE SET high_water=excluded.high_water",
                       (device_id, version))
            db.execute("INSERT INTO library_versions VALUES (?, ?, ?, 'prepared', 0, ?, ?, NULL, NULL)",
                       (device_id, version, payload, now, now))
            db.commit()
        return json.loads(payload)

    def _retry_payload(self, device_id, version):
        with self._db() as db:
            row = db.execute("SELECT version,payload FROM library_versions WHERE device_id=? ORDER BY version DESC LIMIT 1",
                             (device_id,)).fetchone()
        if row is None or row["version"] != version:
            raise LibraryError("待重试版本已变化，请刷新；不会发送其他版本")
        return json.loads(row["payload"])

    def _record(self, device_id, version, state, *, error=None, receipt=None, attempt=False):
        with self._db() as db:
            with db:
                db.execute("""UPDATE library_versions SET state=?, error=?, receipt=?, updated_at=?,
                           attempts=attempts+? WHERE device_id=? AND version=?""",
                           (state, error, json.dumps(receipt) if receipt else None, self.clock(),
                            int(attempt), device_id, version))

    async def sync(self, device_id, names):
        _device(device_id)
        if (not isinstance(names, list) or len(names) > 3 or any(not isinstance(n, str) for n in names)
                or len(set(names)) != len(names)):
            raise LibraryError("请选择 0 至 3 个不同的已保存动作", 422)
        async with self._locks.setdefault(device_id, asyncio.Lock()):
            _, valid = await asyncio.to_thread(self._catalog)
            if any(name not in valid for name in names):
                raise LibraryError("所选动作不存在或不符合离线库限制，请刷新家长动作列表", 422)
            # The selected definitions become immutable. Later edits affect only
            # a new explicit sync, never this request or its retry.
            actions = deepcopy([valid[name] for name in names])
            device = self._device_status(device_id, required=True)
            payload = await self._write(self._allocate, device_id, device["version"], actions)
            await self._send(device_id, payload)
        return await self.status(device_id)

    async def retry(self, device_id, version):
        _device(device_id)
        if type(version) is not int or not 1 <= version <= MAX_VERSION:
            raise LibraryError("需要有效的待重试版本", 422)
        async with self._locks.setdefault(device_id, asyncio.Lock()):
            payload = await asyncio.to_thread(self._retry_payload, device_id, version)
            device = self._device_status(device_id, required=True)
            if device["version"] > version:
                await self._write(self._record, device_id, version, "conflict",
                                  error="设备已有更高版本，请家长重新选择并显式同步")
                raise LibraryError("设备已有更高版本，拒绝重试旧版；请显式同步新选择")
            await self._send(device_id, payload)
        return await self.status(device_id)

    async def _send(self, device_id, payload):
        version = payload["version"]
        await self._write(self._record, device_id, version, "sending", attempt=True)
        self._sending.add((device_id, version))
        try:
            # Recheck the current connection immediately before the only command.
            device = self._device_status(device_id, required=True)
            if device["version"] > version:
                await self._write(self._record, device_id, version, "conflict",
                                  error="设备状态已更新为更高版本，本次未发送；请显式同步新选择")
                return
            receipt = await asyncio.wait_for(self.command(device_id, "set_offline_actions", **deepcopy(payload)), self.timeout)
            saved = (isinstance(receipt, dict) and receipt.get("ok") is True
                     and receipt.get("phase") == "saved" and receipt.get("executed") is False
                     and type(receipt.get("version")) is int and receipt["version"] == version
                     and type(receipt.get("action_count")) is int and receipt["action_count"] == len(payload["actions"]))
            safe_receipt = {"ok": saved, "version": version, "executed": False if saved else None}
            if saved:
                await self._write(self._record, device_id, version, "saved", receipt=safe_receipt)
            else:
                # Even a failed flash commit can leave persistence uncertain.
                await self._write(self._record, device_id, version, "unknown",
                                  error="设备未确认保存成功；可重试同版本同内容", receipt=safe_receipt)
        except asyncio.CancelledError:
            await self._write(self._record, device_id, version, "unknown",
                              error="同步已中断，保存结果未确认；可重试同一版本")
            raise
        except Exception:
            await self._write(self._record, device_id, version, "unknown",
                              error="同步超时或连接中断，保存结果未确认；可重试同一版本")
        finally:
            self._sending.discard((device_id, version))
