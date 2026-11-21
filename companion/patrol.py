"""Opt-in, stationary local patrol scheduling and owned-photo retention.

PatrolService(directory, capture_handler=async_device_to_jpeg,
              describe_handler=optional_async_path_to_text_or_dict)
owns a private SQLite ledger and photos/ directory. Capture must be a SINGLE
attempt, cancellable, and must not move a device or play sound. Description
must use a local model. Neither callback is called until a parent enables a
future schedule. This module never accesses or deletes iCloud files.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import fcntl
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image


@dataclass(frozen=True)
class PatrolConfig:
    enabled: bool = False
    interval_minutes: int = 60
    window_start: str = "09:00"
    window_end: str = "21:00"
    timezone: str = "Asia/Shanghai"
    describe: bool = False
    retention_days: int = 7

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.describe) is not bool:
            raise ValueError("开关必须为布尔值")
        if type(self.interval_minutes) is not int or not 15 <= self.interval_minutes <= 1440:
            raise ValueError("巡看间隔须为 15 至 1440 分钟")
        if type(self.retention_days) is not int or not 1 <= self.retention_days <= 30:
            raise ValueError("照片保留须为 1 至 30 天")
        for value in (self.window_start, self.window_end):
            if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
                raise ValueError("时间窗请使用 HH:MM")
        try:
            if not isinstance(self.timezone, str) or len(self.timezone) > 64:
                raise ValueError("无效时区")
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("需要有效的 IANA 时区") from error

    def contains(self, timestamp: float) -> bool:
        current = datetime.fromtimestamp(timestamp, ZoneInfo(self.timezone)).strftime("%H:%M")
        if self.window_start == self.window_end:
            return True  # Explicitly configured all-day window.
        if self.window_start < self.window_end:
            return self.window_start <= current < self.window_end
        return current >= self.window_start or current < self.window_end

    def next_due(self, after: float) -> float:
        candidate = after + self.interval_minutes * 60
        if self.contains(candidate):
            return candidate
        local = datetime.fromtimestamp(candidate, ZoneInfo(self.timezone))
        hour, minute = map(int, self.window_start.split(":"))
        opening = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if opening <= local:
            opening += timedelta(days=1)
        return opening.timestamp()


def _device_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9:_.-]{1,64}", value):
        raise ValueError("无效设备 ID")
    return value


def _open_directory(path: Path) -> int:
    """Walk directory components without following any symlink, including parents."""
    if ".." in path.parts:
        raise ValueError("巡看目录不能含上级路径")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class PatrolService:
    def __init__(self, directory, *, capture_handler, describe_handler=None,
                 clock=time.time, poll_seconds=15, lateness_seconds=30,
                 capture_timeout=40, describe_timeout=60):
        self.directory = Path(directory).expanduser().absolute()
        # Local retention must never become an implicit iCloud deletion policy,
        # nor use the unrelated LLMQuota project's files as patrol storage.
        excluded = {"mobile documents", "cloudstorage", "icloud drive", "llmquota"}
        if any(part.casefold() in excluded for part in self.directory.parts):
            raise ValueError("巡看存储须使用独立的本地目录，不能放入 iCloud 或 LLMQuota")
        self.capture_handler, self.describe_handler = capture_handler, describe_handler
        self.clock, self.poll_seconds, self.lateness_seconds = clock, poll_seconds, lateness_seconds
        self.capture_timeout, self.describe_timeout = capture_timeout, describe_timeout
        if min(poll_seconds, lateness_seconds, capture_timeout, describe_timeout) <= 0:
            raise ValueError("巡看运行上限必须为正数")
        self._directory_fd = self._photos_fd = self._owner_fd = None
        self._loop = None
        self._tasks = {}
        self._closed = False
        self._last_cleanup = float("-inf")
        self._skip_due_after_error = False
        self.last_error = None
        self.last_success_at = None
        try:
            self._directory_fd = _open_directory(self.directory)
            self._owner_fd = os.open("scheduler.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                                     0o600, dir_fd=self._directory_fd)
            fcntl.flock(self._owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                os.mkdir("photos", mode=0o700, dir_fd=self._directory_fd)
            except FileExistsError:
                pass
            self._photos_fd = os.open("photos", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                      dir_fd=self._directory_fd)
            self.database = self.directory / "patrol.sqlite3"
            if self.database.is_symlink():
                raise ValueError("巡看账本不能是符号链接")
            self._initialize()
        except BaseException:
            self._close_descriptors()
            raise

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        now = self.clock()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schedules (
                    device_id TEXT PRIMARY KEY, config TEXT NOT NULL,
                    revision TEXT NOT NULL, next_due REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, device_id TEXT NOT NULL,
                    revision TEXT NOT NULL, scheduled_at REAL NOT NULL,
                    started_at REAL, finished_at REAL, state TEXT NOT NULL,
                    error TEXT, photo_id TEXT, description TEXT,
                    UNIQUE(device_id, revision, scheduled_at));
                CREATE TABLE IF NOT EXISTS photos (
                    photo_id TEXT PRIMARY KEY, device_id TEXT NOT NULL,
                    filename TEXT NOT NULL, created_at REAL NOT NULL,
                    file_device INTEGER NOT NULL, file_inode INTEGER NOT NULL,
                    deleted_at REAL, cleanup_error TEXT);
            """)
            # A process restart is not permission to replay an earlier capture.
            db.execute("UPDATE runs SET state='interrupted', finished_at=? WHERE state='running'", (now,))
            for row in db.execute("SELECT * FROM schedules").fetchall():
                config = PatrolConfig(**json.loads(row["config"]))
                if row["next_due"] <= now:
                    db.execute("UPDATE schedules SET next_due=? WHERE device_id=?",
                               (config.next_due(now), row["device_id"]))
        os.chmod(self.database, 0o600)

    def configuration(self, device_id):
        _device_id(device_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM schedules WHERE device_id=?", (device_id,)).fetchone()
        config = PatrolConfig(**json.loads(row["config"])) if row else PatrolConfig()
        return {"device_id": device_id, **asdict(config),
                "next_due": row["next_due"] if row and config.enabled else None,
                "description_available": self.describe_handler is not None,
                "cleanup_scope": "local_patrol_photos_only", "icloud_deletion": False}

    def configure(self, device_id, config: PatrolConfig):
        _device_id(device_id)
        if self._closed:
            raise RuntimeError("巡看服务已关闭")
        if config.describe and self.describe_handler is None:
            raise ValueError("本地照片描述暂不可用")
        with self.connect() as db:
            db.execute("""INSERT INTO schedules VALUES (?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET config=excluded.config,
                    revision=excluded.revision, next_due=excluded.next_due""",
                (device_id, json.dumps(asdict(config)), str(uuid.uuid4()), config.next_due(self.clock())))
        # Keep the old task registered until its cleanup actually finishes.
        task = self._tasks.get(device_id)
        if task is not None:
            task.cancel()
        return self.configuration(device_id)

    def history(self, device_id, limit=50):
        _device_id(device_id)
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs WHERE device_id=? ORDER BY scheduled_at DESC LIMIT ?",
                              (device_id, min(max(int(limit), 1), 100))).fetchall()
        return [dict(row) for row in rows]

    def _claim_due(self, now):
        jobs = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT * FROM schedules WHERE next_due<=?", (now,)).fetchall():
                config = PatrolConfig(**json.loads(row["config"]))
                if not config.enabled:
                    continue
                state = "running"
                if self._skip_due_after_error or now - row["next_due"] > self.lateness_seconds:
                    state = "missed"
                elif not config.contains(now):
                    state = "outside_window"
                elif row["device_id"] in self._tasks:
                    state = "skipped_busy"
                run_id = str(uuid.uuid4())
                db.execute("UPDATE schedules SET next_due=? WHERE device_id=?",
                           (config.next_due(now), row["device_id"]))
                inserted = db.execute("""INSERT OR IGNORE INTO runs
                    (run_id,device_id,revision,scheduled_at,started_at,finished_at,state)
                    VALUES (?,?,?,?,?,?,?)""", (run_id, row["device_id"], row["revision"], row["next_due"],
                    now if state == "running" else None, None if state == "running" else now, state))
                if inserted.rowcount and state == "running":
                    jobs.append((run_id, row["device_id"], row["revision"], config))
        self._skip_due_after_error = False
        return jobs

    def _authorized(self, device_id, revision):
        with self.connect() as db:
            row = db.execute("SELECT config,revision FROM schedules WHERE device_id=?", (device_id,)).fetchone()
        return bool(row and row["revision"] == revision and json.loads(row["config"])["enabled"])

    def _save_photo(self, run_id, device_id, data):
        if not isinstance(data, bytes) or not 0 < len(data) <= 512000:
            raise ValueError("invalid_photo")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or not 0 < min(image.size) <= max(image.size) <= 1920:
                raise ValueError("invalid_photo")
            image.verify()
        filename = run_id + ".jpg"
        fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self._photos_fd)
        info = os.fstat(fd)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            with self.connect() as db:
                db.execute("INSERT INTO photos VALUES (?,?,?,?,?,?,NULL,NULL)",
                           (run_id, device_id, filename, self.clock(), info.st_dev, info.st_ino))
                db.execute("UPDATE runs SET photo_id=? WHERE run_id=?", (run_id, run_id))
        except BaseException:
            current = os.stat(filename, dir_fd=self._photos_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                os.unlink(filename, dir_fd=self._photos_fd)
            raise
        return self.directory / "photos" / filename

    async def _execute(self, run_id, device_id, revision, config):
        state, error, description = "failed", None, None
        try:
            async with asyncio.timeout(self.capture_timeout):
                data = await self.capture_handler(device_id)  # Exactly one attempt; no retry.
            if not self._authorized(device_id, revision):
                raise asyncio.CancelledError
            path = self._save_photo(run_id, device_id, data)
            state = "completed"
            if config.describe:
                try:
                    async with asyncio.timeout(self.describe_timeout):
                        result = await self.describe_handler(path)
                    description = (result if isinstance(result, str) else json.dumps(result, ensure_ascii=False))[:2000]
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A description failure does not invalidate the captured photo.
                    error = "description_" + type(exc).__name__
        except asyncio.CancelledError:
            state = "cancelled"
            raise
        except Exception as exc:
            error = type(exc).__name__
        finally:
            with self.connect() as db:
                db.execute("UPDATE runs SET state=?,error=?,description=?,finished_at=? WHERE run_id=?",
                           (state, error, description, self.clock(), run_id))

    async def tick(self):
        if self._closed:
            return
        now = self.clock()
        # No awaits between durable claim and task ownership publication.
        for run_id, device_id, revision, config in self._claim_due(now):
            task = asyncio.create_task(self._execute(run_id, device_id, revision, config), name="puppy-patrol")
            self._tasks[device_id] = task

            def finished(done, owner=device_id, claimed_run=run_id):
                if self._tasks.get(owner) is done:
                    self._tasks.pop(owner, None)
                try:
                    if done.cancelled():
                        # Cancellation can precede the coroutine's first instruction,
                        # in which case its try/finally never got a chance to run.
                        with self.connect() as db:
                            db.execute("UPDATE runs SET state='cancelled',finished_at=? WHERE run_id=? AND state='running'",
                                       (self.clock(), claimed_run))
                    else:
                        error = done.exception()
                        if error is not None:
                            self._record_error(error)
                except Exception as error:
                    self._record_error(error)

            task.add_done_callback(finished)
        if now - self._last_cleanup >= 3600:
            self.cleanup()
            self._last_cleanup = now

    @staticmethod
    def _filename(row):
        if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", row["photo_id"]):
            raise ValueError("invalid_registered_path")
        if row["filename"] != row["photo_id"] + ".jpg":
            raise ValueError("invalid_registered_path")
        return row["filename"]

    def _file_info(self, row):
        name = self._filename(row)
        info = os.stat(name, dir_fd=self._photos_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (row["file_device"], row["file_inode"]):
            raise ValueError("registered_file_changed")
        return name

    def read_photo(self, photo_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM photos WHERE photo_id=? AND deleted_at IS NULL", (photo_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("photo_not_found")
        name = self._file_info(row)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._photos_fd)
        with os.fdopen(fd, "rb") as file:
            info = os.fstat(file.fileno())
            if (info.st_dev, info.st_ino) != (row["file_device"], row["file_inode"]):
                raise ValueError("registered_file_changed")
            data = file.read(512001)
        if len(data) > 512000:
            raise ValueError("registered_file_changed")
        return data

    def cleanup(self):
        now = self.clock()
        result = {"deleted": 0, "missing": 0, "blocked": 0, "scope": "local_only", "icloud_deleted": False}
        with self.connect() as db:
            rows = db.execute("""SELECT p.*,s.config FROM photos p JOIN schedules s USING(device_id)
                WHERE p.deleted_at IS NULL""").fetchall()
            for row in rows:
                retention = PatrolConfig(**json.loads(row["config"])).retention_days
                if now - row["created_at"] < retention * 86400:
                    continue
                try:
                    name = self._file_info(row)
                    # Single-component unlink relative to a pinned directory:
                    # even a last-instant symlink replacement cannot be followed.
                    os.unlink(name, dir_fd=self._photos_fd)
                    result["deleted"] += 1
                except FileNotFoundError:
                    result["missing"] += 1
                except (OSError, ValueError) as exc:
                    result["blocked"] += 1
                    db.execute("UPDATE photos SET cleanup_error=? WHERE photo_id=?", (type(exc).__name__, row["photo_id"]))
                    continue
                db.execute("UPDATE photos SET deleted_at=?,cleanup_error=NULL WHERE photo_id=?", (now, row["photo_id"]))
        return result

    def start(self):
        if self._closed:
            raise RuntimeError("巡看服务已关闭")
        if self._loop is None:
            async def run():
                while True:
                    try:
                        await self.tick()
                        self.last_success_at = self.clock()
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        self._record_error(error)
                    await asyncio.sleep(self.poll_seconds)
            self._loop = asyncio.create_task(run(), name="puppy-patrol-scheduler")

    def _record_error(self, error):
        self.last_error = {"type": type(error).__name__, "observed_at": self.clock()}
        # Recovery may advance a due slot, never replay one failed during a
        # scheduler/storage outage, even if it is inside the lateness allowance.
        self._skip_due_after_error = True

    def status(self):
        return {"running": self._loop is not None and not self._loop.done(),
                "active_devices": list(self._tasks), "last_error": self.last_error,
                "last_success_at": self.last_success_at, "icloud_deletion": False}

    def _close_descriptors(self):
        for name in ("_photos_fd", "_owner_fd", "_directory_fd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    async def close(self):
        self._closed = True
        tasks = [task for task in (self._loop, *self._tasks.values()) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._close_descriptors()
