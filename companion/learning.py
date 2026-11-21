"""Persistent English practice settings and elapsed-time limits, without model/audio work.

One ledger belongs to (device, person, local date). person=None is a separate guest
ledger and reads only device defaults. A model turn receives at most 30 seconds;
a successful reply leaves a 45-second activity window for the child's next turn.
The window expires silently and is capped by remaining daily time and local midnight.
Restarts account the persisted window up to its deadline; they never grant a new
budget. Switching away from learning should call end_session() immediately.

The caller must use after_turn() before delivering a model response. A stale turn
cannot extend a newer session. save_progress controls curriculum records in the
caller; usage counters always persist so disabling progress cannot bypass limits.
"""
import math
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, time as daytime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

THEMES = {"animals", "colors", "greetings", "objects", "body_parts"}
DEFAULT_SETTINGS = {"daily_minutes": 5, "theme": "animals", "save_progress": True,
                    "photo_learning": False}
TURN_SECONDS = 30.0
IDLE_SECONDS = 45.0


class LearningStore:
    def __init__(self, database: str | Path, *, timezone: str = "Asia/Shanghai", clock=time.time):
        path = Path(database)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not path.exists():
            path.touch(mode=0o600)
        path.chmod(0o600)
        self.database, self.zone, self.clock = str(path), ZoneInfo(timezone), clock
        self._settings_commit_lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS learning_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS learning_settings (
                    device_id TEXT NOT NULL, subject TEXT NOT NULL,
                    daily_minutes INTEGER NOT NULL, theme TEXT NOT NULL,
                    save_progress INTEGER NOT NULL, photo_learning INTEGER NOT NULL,
                    PRIMARY KEY(device_id, subject));
                CREATE TABLE IF NOT EXISTS learning_usage (
                    device_id TEXT NOT NULL, subject TEXT NOT NULL, local_day TEXT NOT NULL,
                    used_seconds REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(device_id, subject, local_day));
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    device_id TEXT NOT NULL, subject TEXT NOT NULL, session_id TEXT,
                    accounted_at REAL NOT NULL, active_until REAL NOT NULL,
                    turn_id TEXT, last_seen REAL NOT NULL,
                    PRIMARY KEY(device_id, subject));
            """)
            prior = db.execute("SELECT value FROM learning_meta WHERE key='timezone'").fetchone()
            if prior and prior["value"] != timezone:
                raise ValueError("Changing the learning timezone requires a ledger migration")
            db.execute("INSERT OR IGNORE INTO learning_meta VALUES ('timezone', ?)", (timezone,))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _key(device_id, person_id):
        if not isinstance(device_id, str) or not device_id.strip() or len(device_id) > 128:
            raise ValueError("invalid_device_id")
        if person_id is not None and (not isinstance(person_id, str) or not person_id.strip() or len(person_id) > 128):
            raise ValueError("invalid_person_id")
        return device_id, person_id or ""

    def _now(self):
        value = float(self.clock())
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid_clock")
        return value

    def _day(self, timestamp):
        return datetime.fromtimestamp(timestamp, self.zone).date().isoformat()

    def _midnight(self, timestamp):
        next_day = datetime.fromtimestamp(timestamp, self.zone).date() + timedelta(days=1)
        return datetime.combine(next_day, daytime.min, self.zone).timestamp()

    @staticmethod
    def _settings(db, device, subject):
        row = db.execute("SELECT daily_minutes, theme, save_progress, photo_learning FROM learning_settings WHERE device_id=? AND subject=?",
                         (device, subject)).fetchone()
        inherited = False
        if row is None and subject:
            row = db.execute("SELECT daily_minutes, theme, save_progress, photo_learning FROM learning_settings WHERE device_id=? AND subject=''",
                             (device,)).fetchone()
            inherited = True
        value = dict(row) if row else DEFAULT_SETTINGS.copy()
        value["save_progress"], value["photo_learning"] = bool(value["save_progress"]), bool(value["photo_learning"])
        return value, inherited

    def get_settings(self, device_id: str, person_id: str | None = None):
        device, subject = self._key(device_id, person_id)
        with self.connect() as db:
            settings, inherited = self._settings(db, device, subject)
        return {**settings, "source": "device_default" if not subject or inherited else "member"}

    @contextmanager
    def saving_permission(self, device_id, person_id):
        """Hold current permission through a short synchronous SQLite commit.

        A completed disable prevents later commits; a commit that acquired this
        lock first finishes before disable returns. This does not undo history.
        The scope must include the actual commit/rollback, not only staged SQL.
        It must not await, run a model, or acquire a model queue lock.
        """
        with self._settings_commit_lock:
            yield self.get_settings(device_id, person_id)["save_progress"]

    def commit_if_saving(self, device_id, person_id, commit):
        """Run a short synchronous callback under the settings writer lock."""
        with self.saving_permission(device_id, person_id) as allowed:
            if not allowed:
                return False
            commit()
            return True

    @staticmethod
    def _validate_patch(patch):
        if not isinstance(patch, dict) or not patch or set(patch) - set(DEFAULT_SETTINGS):
            raise ValueError("invalid_learning_settings")
        if "daily_minutes" in patch and (type(patch["daily_minutes"]) is not int or not 1 <= patch["daily_minutes"] <= 20):
            raise ValueError("daily_minutes_must_be_1_to_20")
        if "theme" in patch and (not isinstance(patch["theme"], str) or patch["theme"] not in THEMES):
            raise ValueError("invalid_learning_theme")
        if any(key in patch and type(patch[key]) is not bool for key in ("save_progress", "photo_learning")):
            raise ValueError("learning_switch_must_be_boolean")

    def set_settings(self, device_id: str, person_id: str | None = None, *, patch: dict):
        self._validate_patch(patch)
        device, subject = self._key(device_id, person_id)
        with self._settings_commit_lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state, now = self._advance(db, device, subject)
            settings, _ = self._settings(db, device, subject)
            settings.update(patch)
            db.execute("""INSERT INTO learning_settings VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_id, subject) DO UPDATE SET daily_minutes=excluded.daily_minutes,
                theme=excluded.theme, save_progress=excluded.save_progress,
                photo_learning=excluded.photo_learning""", (device, subject, settings["daily_minutes"],
                settings["theme"], int(settings["save_progress"]), int(settings["photo_learning"])))
            remaining = max(0, settings["daily_minutes"] * 60 - self._used(db, device, subject, self._day(now)))
            state["active_until"] = min(state["active_until"], now + remaining)
            if remaining <= 0:
                state["turn_id"], state["session_id"] = None, None
            self._save_state(db, device, subject, state)
            return self._view(db, device, subject, state, now)

    @staticmethod
    def _used(db, device, subject, day):
        row = db.execute("SELECT used_seconds FROM learning_usage WHERE device_id=? AND subject=? AND local_day=?",
                         (device, subject, day)).fetchone()
        return row["used_seconds"] if row else 0.0

    @staticmethod
    def _save_state(db, device, subject, state):
        db.execute("""INSERT INTO learning_sessions VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id, subject) DO UPDATE SET session_id=excluded.session_id,
            accounted_at=excluded.accounted_at, active_until=excluded.active_until,
            turn_id=excluded.turn_id, last_seen=excluded.last_seen""",
                   (device, subject, state["session_id"], state["accounted_at"], state["active_until"],
                    state["turn_id"], state["last_seen"]))

    def _advance(self, db, device, subject):
        row = db.execute("SELECT * FROM learning_sessions WHERE device_id=? AND subject=?", (device, subject)).fetchone()
        clock_now = self._now()
        state = dict(row) if row else {"session_id": None, "accounted_at": clock_now,
                                      "active_until": clock_now, "turn_id": None, "last_seen": clock_now}
        # A backward wall-clock adjustment cannot reopen an earlier date or refund time.
        now = max(clock_now, state["last_seen"])
        state["clock_adjusted"] = clock_now < state["last_seen"]
        cursor, end = state["accounted_at"], min(now, state["active_until"])
        while cursor < end:
            edge = min(end, self._midnight(cursor))
            db.execute("""INSERT INTO learning_usage VALUES (?, ?, ?, ?)
                ON CONFLICT(device_id, subject, local_day) DO UPDATE SET
                used_seconds=used_seconds+excluded.used_seconds""",
                       (device, subject, self._day(cursor), edge - cursor))
            cursor = edge
        state["accounted_at"], state["last_seen"] = now, now
        if state["clock_adjusted"]:
            state["active_until"] = now
        if state["active_until"] <= now:
            state["turn_id"], state["session_id"] = None, None
        self._save_state(db, device, subject, state)
        return state, now

    def _view(self, db, device, subject, state, now):
        settings, inherited = self._settings(db, device, subject)
        day = self._day(now)
        used = self._used(db, device, subject, day)
        limit = settings["daily_minutes"] * 60
        return {"device_id": device, "person_id": subject or None, "local_day": day,
                "timezone": self.zone.key, "settings": settings,
                "settings_source": "device_default" if not subject or inherited else "member",
                "used_seconds": used, "limit_seconds": limit,
                "remaining_seconds": max(0, limit - used), "server_time": now,
                "clock_adjusted": state.get("clock_adjusted", False),
                "active": state["active_until"] > now, "busy": state["turn_id"] is not None,
                "session_id": state["session_id"], "active_until": state["active_until"]}

    @staticmethod
    def _decision(view, allowed, reason, turn_id=None):
        stop = None
        if not allowed and reason != "turn_in_progress":
            stop = {"kind": "finish_learning", "reason": reason, "local_day": view["local_day"],
                    "used_seconds": view["used_seconds"], "limit_seconds": view["limit_seconds"]}
        return {**view, "allowed": allowed, "reason": reason, "turn_id": turn_id, "stop": stop}

    def status(self, device_id: str, person_id: str | None = None):
        device, subject = self._key(device_id, person_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state, now = self._advance(db, device, subject)
            return self._view(db, device, subject, state, now)

    def before_turn(self, device_id: str, person_id: str | None = None, *,
                    session_id: str, use_photo: bool = False):
        device, subject = self._key(device_id, person_id)
        if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
            raise ValueError("invalid_session_id")
        if type(use_photo) is not bool:
            raise ValueError("invalid_photo_flag")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state, now = self._advance(db, device, subject)
            view = self._view(db, device, subject, state, now)
            if view["clock_adjusted"]:
                return self._decision(view, False, "clock_changed")
            if view["remaining_seconds"] <= 0:
                return self._decision(view, False, "daily_limit")
            if state["turn_id"]:
                return self._decision(view, False, "turn_in_progress")
            if use_photo and not view["settings"]["photo_learning"]:
                return self._decision(view, False, "photo_learning_disabled")
            state.update(session_id=session_id, turn_id=str(uuid.uuid4()), accounted_at=now,
                         active_until=min(now + TURN_SECONDS, now + view["remaining_seconds"], self._midnight(now)))
            self._save_state(db, device, subject, state)
            return self._decision(self._view(db, device, subject, state, now), True, "ready", state["turn_id"])

    def after_turn(self, device_id: str, person_id: str | None = None, *, turn_id: str, success: bool = True):
        device, subject = self._key(device_id, person_id)
        if not isinstance(turn_id, str) or not turn_id or type(success) is not bool:
            raise ValueError("invalid_turn_result")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state, now = self._advance(db, device, subject)
            view = self._view(db, device, subject, state, now)
            if view["clock_adjusted"]:
                return self._decision(view, False, "clock_changed")
            if state["turn_id"] != turn_id:
                reason = "daily_limit" if view["remaining_seconds"] <= 0 else "stale_turn"
                return self._decision(view, False, reason)
            state["turn_id"] = None
            if not success or view["remaining_seconds"] <= 0:
                state["session_id"], state["active_until"] = None, now
                reason = "turn_failed" if not success else "daily_limit"
                self._save_state(db, device, subject, state)
                return self._decision(self._view(db, device, subject, state, now), False, reason)
            state["active_until"] = min(now + IDLE_SECONDS, now + view["remaining_seconds"], self._midnight(now))
            self._save_state(db, device, subject, state)
            return self._decision(self._view(db, device, subject, state, now), True, "continue")

    def end_session(self, device_id: str, person_id: str | None = None):
        device, subject = self._key(device_id, person_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state, now = self._advance(db, device, subject)
            state.update(session_id=None, turn_id=None, active_until=now)
            self._save_state(db, device, subject, state)
            return self._decision(self._view(db, device, subject, state, now), False, "session_ended")

    def end_device_sessions(self, device_id: str):
        with self.connect() as db:
            subjects = [row["subject"] for row in db.execute(
                "SELECT subject FROM learning_sessions WHERE device_id=?", (device_id,)).fetchall()]
        for subject in subjects:
            self.end_session(device_id, subject or None)
