import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .turn_commit import TurnCommit
from .child_activity import ChildActivitySessions


class Store:
    NAME_CONFIRMATION_SECONDS = 120
    CONVERSATION_SECONDS = 30 * 60
    CONVERSATION_TURNS = 6

    def __init__(self, database: str | Path):
        self.database = str(database)
        self._local = threading.local()
        self.child_sessions = ChildActivitySessions()
        database_path = Path(self.database)
        database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not database_path.exists():
            database_path.touch(mode=0o600)
        database_path.chmod(0o600)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS pets (
                    device_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '小狗',
                    pending_name TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS device_modes (
                    device_id TEXT PRIMARY KEY, mode TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    device_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id, memory_key)
                );
                CREATE TABLE IF NOT EXISTS actions (
                    name TEXT PRIMARY KEY,
                    steps TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS action_ownership (
                    name TEXT PRIMARY KEY, draft_id TEXT NOT NULL,
                    ownership_nonce TEXT NOT NULL UNIQUE
                );
                CREATE TRIGGER IF NOT EXISTS clear_action_owner_insert AFTER INSERT ON actions
                    BEGIN DELETE FROM action_ownership WHERE name=NEW.name; END;
                CREATE TRIGGER IF NOT EXISTS clear_action_owner_update AFTER UPDATE ON actions
                    BEGIN DELETE FROM action_ownership WHERE name=OLD.name OR name=NEW.name; END;
                CREATE TRIGGER IF NOT EXISTS clear_action_owner_delete AFTER DELETE ON actions
                    BEGIN DELETE FROM action_ownership WHERE name=OLD.name; END;
                CREATE TABLE IF NOT EXISTS photos (
                    photo_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    captured_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    device_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    turns TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id, session_id, mode)
                );
                CREATE TABLE IF NOT EXISTS english_progress (
                    device_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    word TEXT NOT NULL,
                    presentations INTEGER NOT NULL DEFAULT 0,
                    text_matches INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id, word)
                );
                CREATE TABLE IF NOT EXISTS english_checks (
                    device_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    word TEXT NOT NULL,
                    checkpoint TEXT NOT NULL,
                    result TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id, word, checkpoint)
                );
                CREATE TABLE IF NOT EXISTS english_review_prompts (
                    device_id TEXT NOT NULL, person_id TEXT NOT NULL,
                    word TEXT NOT NULL, checkpoint TEXT NOT NULL,
                    prompted_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id, word, checkpoint)
                );
                CREATE TABLE IF NOT EXISTS english_lessons (
                    device_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    word TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id)
                );
                CREATE TABLE IF NOT EXISTS stories (
                    device_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    chapter INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id)
                );
                CREATE TABLE IF NOT EXISTS pet_profiles (
                    device_id TEXT PRIMARY KEY, style TEXT NOT NULL,
                    greeting_action TEXT NOT NULL, revision INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS routines (
                    device_id TEXT NOT NULL, person_id TEXT NOT NULL,
                    routine_id TEXT NOT NULL, title TEXT NOT NULL, steps TEXT NOT NULL,
                    state TEXT NOT NULL, step_index INTEGER NOT NULL,
                    revision INTEGER NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY (device_id, person_id, routine_id)
                );
                CREATE TABLE IF NOT EXISTS routine_clock (
                    id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO routine_clock VALUES(1, 0);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(pets)")}
            if "pending_actor" not in columns:
                db.execute("ALTER TABLE pets ADD COLUMN pending_actor TEXT")
            if "pending_expires_at" not in columns:
                db.execute("ALTER TABLE pets ADD COLUMN pending_expires_at REAL")
            progress_columns = {row["name"] for row in db.execute("PRAGMA table_info(english_progress)")}
            if "review_anchor_at" not in progress_columns:
                db.execute("ALTER TABLE english_progress ADD COLUMN review_anchor_at REAL")
                # Existing rows only contain their latest contact time. Use it as
                # a conservative review anchor; never claim it was first exposure.
                db.execute("UPDATE english_progress SET review_anchor_at=updated_at WHERE review_anchor_at IS NULL")
            # Old unscoped proposals must not be confirmed by a different speaker.
            db.execute("""UPDATE pets SET pending_name=NULL, pending_actor=NULL,
                          pending_expires_at=NULL WHERE pending_name IS NOT NULL
                          AND (pending_actor IS NULL OR pending_expires_at IS NULL)""")

    @contextmanager
    def connect(self):
        scope = getattr(self._local, "commit_scope", None)
        if scope is not None and scope["connection"] is not None:
            yield scope["connection"]
            return
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def commit_scope(self, turn_commit: TurnCommit | None = None):
        """Group a turn's writes, without holding a DB transaction during inference.

        The agent calls ``begin_commit`` only after generation. An optional outer
        scope lets the gateway include its learning/history writes in that same
        transaction. All scopes and database work stay on the inference thread.
        """
        active = getattr(self._local, "commit_scope", None)
        if active is not None:
            if turn_commit is not None and active["guard"] is not turn_commit:
                raise ValueError("嵌套提交必须使用同一个本轮提交对象")
            yield
            return
        scope = {"guard": turn_commit, "connection": None}
        self._local.commit_scope = scope
        try:
            if turn_commit is not None:
                turn_commit.check()
            yield
            db = scope["connection"]
            if turn_commit is not None:
                turn_commit.commit(db.commit if db is not None else lambda: None)
            elif db is not None:
                db.commit()
        except BaseException:
            if scope["connection"] is not None:
                scope["connection"].rollback()
            raise
        finally:
            if scope["connection"] is not None:
                scope["connection"].close()
            del self._local.commit_scope

    def begin_commit(self):
        """Start the short write transaction after the model has finished."""
        scope = getattr(self._local, "commit_scope", None)
        if scope is None:
            raise RuntimeError("记忆提交需要 commit_scope")
        guard = scope["guard"]
        if guard is not None:
            guard.check()
        if scope["connection"] is None:
            db = sqlite3.connect(self.database, timeout=5)
            db.row_factory = sqlite3.Row
            scope["connection"] = db
            db.execute("BEGIN IMMEDIATE")
        # Cancellation can arrive while SQLite waits for another writer.
        if guard is not None:
            guard.check()

    @staticmethod
    def _begin_immediate(db):
        if not db.in_transaction:
            db.execute("BEGIN IMMEDIATE")

    def pet(self, device_id: str) -> dict:
        with self.connect() as db:
            self._expire_name(db, device_id)
            row = db.execute("SELECT name, pending_name FROM pets WHERE device_id=?", (device_id,)).fetchone()
        return dict(row) if row else {"name": "小狗", "pending_name": None}

    def conversation_mode(self, device_id: str) -> str:
        with self.connect() as db:
            row = db.execute("SELECT mode FROM device_modes WHERE device_id=?", (device_id,)).fetchone()
        return row["mode"] if row else "chat"

    def set_conversation_mode(self, device_id: str, mode: str):
        if mode not in {"chat", "english", "story"}:
            raise ValueError("不支持的对话模式")
        with self.connect() as db:
            db.execute("INSERT INTO device_modes VALUES (?,?) ON CONFLICT(device_id) DO UPDATE SET mode=excluded.mode",
                       (device_id, mode))

    @staticmethod
    def _actor(person_id: str | None, session_id: str | None) -> str | None:
        if not person_id and not session_id:
            return None
        return json.dumps([person_id or None, session_id or None], ensure_ascii=False)

    @staticmethod
    def _expire_name(db, device_id: str):
        db.execute("""UPDATE pets SET pending_name=NULL, pending_actor=NULL,
                      pending_expires_at=NULL WHERE device_id=?
                      AND pending_expires_at<=?""", (device_id, time.time()))

    def pending_name(self, device_id: str, person_id: str | None = None,
                     session_id: str | None = None) -> str | None:
        actor = self._actor(person_id, session_id)
        if actor is None:
            return None
        with self.connect() as db:
            self._expire_name(db, device_id)
            row = db.execute("SELECT pending_name FROM pets WHERE device_id=? AND pending_actor=?",
                             (device_id, actor)).fetchone()
        return row["pending_name"] if row else None

    def propose_name(self, device_id: str, name: str, person_id: str | None = None,
                     session_id: str | None = None):
        actor = self._actor(person_id, session_id)
        if actor is None:
            raise ValueError("改名确认必须关联成员或独立会话")
        if not name or len(name) > 12:
            raise ValueError("名字须为 1 至 12 字")
        now = time.time()
        with self.connect() as db:
            db.execute("""INSERT INTO pets(device_id, pending_name, pending_actor,
                          pending_expires_at, updated_at) VALUES (?, ?, ?, ?, ?)
                          ON CONFLICT(device_id) DO UPDATE SET
                          pending_name=excluded.pending_name, pending_actor=excluded.pending_actor,
                          pending_expires_at=excluded.pending_expires_at,
                          updated_at=excluded.updated_at""",
                       (device_id, name, actor, now + self.NAME_CONFIRMATION_SECONDS, now))

    def confirm_name(self, device_id: str, accepted: bool, person_id: str | None = None,
                     session_id: str | None = None, expected_name: str | None = None) -> str:
        actor = self._actor(person_id, session_id)
        with self.connect() as db:
            self._begin_immediate(db)
            row = db.execute("SELECT * FROM pets WHERE device_id=?", (device_id,)).fetchone()
            if (actor is None or row is None or not row["pending_name"]
                    or row["pending_actor"] != actor or row["pending_expires_at"] <= time.time()
                    or (expected_name is not None and row["pending_name"] != expected_name)):
                raise ValueError("没有属于当前会话且未过期的待确认名字")
            name = row["pending_name"] if accepted else row["name"]
            db.execute("""UPDATE pets SET name=?, pending_name=NULL, pending_actor=NULL,
                          pending_expires_at=NULL, updated_at=? WHERE device_id=?""",
                       (name, time.time(), device_id))
            return name

    def memories(self, device_id: str, person_id: str | None) -> dict[str, str]:
        if not person_id:
            return {}
        with self.connect() as db:
            rows = db.execute("SELECT memory_key, value FROM memories WHERE device_id=? AND person_id=?",
                              (device_id, person_id)).fetchall()
        return {row["memory_key"]: row["value"] for row in rows}

    def remember(self, device_id: str, person_id: str, key: str, value: str):
        if not person_id or not key or not value:
            raise ValueError("记忆必须关联已确认的家庭成员")
        if len(key) > 96 or len(value) > 240:
            raise ValueError("记忆字段过长")
        with self.connect() as db:
            db.execute("""INSERT INTO memories VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(device_id, person_id, memory_key) DO UPDATE SET
                value=excluded.value, updated_at=excluded.updated_at""",
                (device_id, person_id, key, value, time.time()))

    def remember_preference(self, device_id: str, person_id: str, value: str) -> str:
        value = " ".join(value.split())[:80]
        key = "preference:" + hashlib.sha256(value.casefold().encode()).hexdigest()[:16]
        self.remember(device_id, person_id, key, value)
        return key

    def remember_preference_statement(self, device_id: str, person_id: str, text: str) -> str:
        """Preserve one user statement verbatim; interpretation belongs elsewhere.

        Keep the legacy preference writer's normalization and limits unchanged.
        Using connect also joins an outer turn's active write transaction.
        """
        if not isinstance(device_id, str) or not device_id or not isinstance(person_id, str) or not person_id:
            raise ValueError("原话记忆必须关联设备与已确认的家庭成员")
        if not isinstance(text, str) or not text or text.isspace() or len(text) > 300:
            raise ValueError("原话须为 1 至 300 字的非空文本")
        key = "preference_statement:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.connect() as db:
            db.execute("""INSERT INTO memories (device_id, person_id, memory_key, value, updated_at)
                          VALUES (?, ?, ?, ?, ?)
                          ON CONFLICT(device_id, person_id, memory_key) DO NOTHING""",
                       (device_id, person_id, key, text, time.time()))
        return key

    def memory_entries(self, device_id: str, person_id: str | None) -> list[dict]:
        if not person_id:
            return []
        with self.connect() as db:
            rows = db.execute("""SELECT memory_key, value, updated_at FROM memories
                                 WHERE device_id=? AND person_id=? ORDER BY updated_at DESC""",
                              (device_id, person_id)).fetchall()
        return [dict(row) for row in rows]

    def forget(self, device_id: str, person_id: str, key: str):
        if not person_id:
            raise ValueError("删除记忆必须关联已确认的家庭成员")
        with self.connect() as db:
            db.execute("DELETE FROM memories WHERE device_id=? AND person_id=? AND memory_key=?",
                       (device_id, person_id, key))
            # Otherwise the next prompt could recover the deleted preference from history.
            db.execute("DELETE FROM conversations WHERE device_id=? AND person_id=?",
                       (device_id, person_id))

    def forget_person(self, device_id: str, person_id: str):
        if not person_id:
            raise ValueError("删除资料必须关联已确认的家庭成员")
        with self.connect() as db:
            for table in ("memories", "conversations", "english_progress", "english_checks", "english_review_prompts", "english_lessons", "stories", "routines"):
                db.execute(f"DELETE FROM {table} WHERE device_id=? AND person_id=?",
                           (device_id, person_id))
            row = db.execute("SELECT pending_actor FROM pets WHERE device_id=?", (device_id,)).fetchone()
            if row and row["pending_actor"] and json.loads(row["pending_actor"])[0] == person_id:
                db.execute("""UPDATE pets SET pending_name=NULL, pending_actor=NULL,
                              pending_expires_at=NULL WHERE device_id=?""", (device_id,))

    def conversation(self, device_id: str, person_id: str | None, mode: str = "chat",
                     session_id: str | None = None) -> list[dict]:
        if not person_id:
            return []
        with self.connect() as db:
            db.execute("DELETE FROM conversations WHERE updated_at<=?",
                       (time.time() - self.CONVERSATION_SECONDS,))
            row = db.execute("""SELECT turns FROM conversations WHERE device_id=?
                                 AND person_id=? AND session_id=? AND mode=?""",
                              (device_id, person_id, session_id or "", mode)).fetchone()
        return json.loads(row["turns"]) if row else []

    def append_conversation(self, device_id: str, person_id: str | None, user: str,
                            reply: str, mode: str = "chat", session_id: str | None = None):
        if not person_id:
            return
        with self.connect() as db:
            self._begin_immediate(db)
            row = db.execute("""SELECT turns, updated_at FROM conversations WHERE device_id=?
                                 AND person_id=? AND session_id=? AND mode=?""",
                              (device_id, person_id, session_id or "", mode)).fetchone()
            turns = (json.loads(row["turns"]) if row
                     and row["updated_at"] > time.time() - self.CONVERSATION_SECONDS else [])
            turns.append({"user": user[:300], "assistant": reply[:120]})
            db.execute("""INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)
                          ON CONFLICT(device_id, person_id, session_id, mode) DO UPDATE SET
                          turns=excluded.turns, updated_at=excluded.updated_at""",
                       (device_id, person_id, session_id or "", mode,
                        json.dumps(turns[-self.CONVERSATION_TURNS:], ensure_ascii=False), time.time()))

    def english_progress(self, device_id: str, person_id: str | None) -> list[dict]:
        if not person_id:
            return []
        with self.connect() as db:
            rows = db.execute("""SELECT word, presentations, text_matches, updated_at, review_anchor_at
                                 FROM english_progress WHERE device_id=? AND person_id=?
                                 ORDER BY updated_at, word""", (device_id, person_id)).fetchall()
        return [dict(row) for row in rows]

    def english_checks(self, device_id: str, person_id: str | None) -> list[dict]:
        if not person_id:
            return []
        with self.connect() as db:
            rows = db.execute("""SELECT word, checkpoint, result, modality, observed_at
                                 FROM english_checks WHERE device_id=? AND person_id=?
                                 ORDER BY word, observed_at""", (device_id, person_id)).fetchall()
        return [{**dict(row), "source": "parent_report"} for row in rows]

    def record_english_check(self, device_id: str, person_id: str, word: str,
                             checkpoint: str, result: str, modality: str):
        if not person_id or not isinstance(word, str) or not 2 <= len(word) <= 24 or not word.isascii() or not word.isalpha():
            raise ValueError("请选择已确认成员和英文目标词")
        if checkpoint not in {"baseline", "next_day", "week_later"}:
            raise ValueError("无效的英语检查时间点")
        if result not in {"independent", "prompted", "not_yet"}:
            raise ValueError("无效的英语观察结果")
        if modality not in {"pointed", "spoken", "used_in_context"}:
            raise ValueError("无效的英语观察方式")
        word = word.lower()
        now = time.time()
        with self.connect() as db:
            self._begin_immediate(db)
            progress = db.execute("""SELECT review_anchor_at FROM english_progress
                                   WHERE device_id=? AND person_id=? AND word=?""",
                                  (device_id, person_id, word)).fetchone()
            existing = db.execute("""SELECT 1 FROM english_checks
                                   WHERE device_id=? AND person_id=? AND word=? AND checkpoint=?""",
                                  (device_id, person_id, word, checkpoint)).fetchone()
            if checkpoint == "baseline":
                if progress and not existing:
                    raise ValueError("起点观察须在这个词第一次练习前记录")
            elif not progress:
                raise ValueError("请先完成这个词的第一次练习")
            else:
                delay = 24 * 3600 if checkpoint == "next_day" else 7 * 24 * 3600
                if now < progress["review_anchor_at"] + delay:
                    raise ValueError("尚未到达这个词的延迟检查时间")
            db.execute("""INSERT INTO english_checks VALUES (?,?,?,?,?,?,?)
                          ON CONFLICT(device_id,person_id,word,checkpoint) DO UPDATE SET
                          result=excluded.result, modality=excluded.modality,
                          observed_at=excluded.observed_at""",
                       (device_id, person_id, word.lower(), checkpoint, result, modality, now))
        return {"word": word.lower(), "checkpoint": checkpoint, "result": result,
                "modality": modality, "observed_at": now, "source": "parent_report"}

    def english_review_plan(self, device_id: str, person_id: str | None) -> list[dict]:
        if not person_id:
            return []
        checks = {(row["word"], row["checkpoint"]): row for row in self.english_checks(device_id, person_id)}
        now = time.time()
        plan = []
        for row in self.english_progress(device_id, person_id):
            first = row["review_anchor_at"] or row["updated_at"]
            for checkpoint, delay in (("next_day", 24 * 3600), ("week_later", 7 * 24 * 3600)):
                due = first + delay
                plan.append({"word": row["word"], "checkpoint": checkpoint,
                             "due_at": due, "status": "recorded" if (row["word"], checkpoint) in checks
                             else "due" if now >= due else "upcoming"})
        return plan

    def english_due_review(self, device_id: str, person_id: str | None,
                           allowed_words: tuple[str, ...]) -> dict | None:
        if not person_id:
            return None
        allowed = set(allowed_words)
        now = time.time()
        with self.connect() as db:
            prompted = {(row["word"], row["checkpoint"]): row["prompted_at"] for row in
                        db.execute("""SELECT word, checkpoint, prompted_at FROM english_review_prompts
                                      WHERE device_id=? AND person_id=?""", (device_id, person_id))}
        due = [item for item in self.english_review_plan(device_id, person_id)
               if item["word"] in allowed and item["status"] == "due"
               and now - prompted.get((item["word"], item["checkpoint"]), 0) >= 24 * 3600]
        return min(due, key=lambda item: (item["due_at"], item["word"])) if due else None

    def english_lesson(self, device_id: str, person_id: str | None) -> dict | None:
        if not person_id:
            return None
        with self.connect() as db:
            row = db.execute("""SELECT word, stage, updated_at FROM english_lessons
                                WHERE device_id=? AND person_id=?""",
                             (device_id, person_id)).fetchone()
        return dict(row) if row else None

    def story(self, device_id: str, person_id: str | None) -> dict | None:
        if not person_id:
            return None
        with self.connect() as db:
            row = db.execute("SELECT title,chapter,summary,updated_at FROM stories WHERE device_id=? AND person_id=?",
                             (device_id, person_id)).fetchone()
        return dict(row) if row else None

    def save_story(self, device_id: str, person_id: str, title: str, summary: str, previous_chapter: int) -> dict:
        if not person_id or not 1 <= len(title) <= 40 or not 1 <= len(summary) <= 300:
            raise ValueError("故事需要已确认成员、简短标题与进度")
        with self.connect() as db:
            self._begin_immediate(db)
            row = db.execute("SELECT chapter FROM stories WHERE device_id=? AND person_id=?",
                             (device_id, person_id)).fetchone()
            if (row["chapter"] if row else 0) != previous_chapter:
                raise ValueError("故事进度已改变，请重新续讲")
            db.execute("""INSERT INTO stories VALUES(?,?,?,?,?,?) ON CONFLICT(device_id,person_id)
                          DO UPDATE SET title=excluded.title,chapter=excluded.chapter,
                          summary=excluded.summary,updated_at=excluded.updated_at""",
                       (device_id, person_id, title, previous_chapter + 1, summary, time.time()))
        return self.story(device_id, person_id)

    def forget_story(self, device_id: str, person_id: str):
        with self.connect() as db:
            db.execute("DELETE FROM stories WHERE device_id=? AND person_id=?", (device_id, person_id))
            db.execute("DELETE FROM conversations WHERE device_id=? AND person_id=? AND mode='story'",
                       (device_id, person_id))

    def record_english(self, device_id: str, person_id: str, word: str, text_match: bool,
                       activity: str | None = None, advance: bool = True,
                       review_checkpoint: str | None = None):
        if not person_id:
            raise ValueError("学习进度必须关联已确认的家庭成员")
        next_stage = {"introduce": "repeat", "repeat": "play",
                      "play": "respond", "respond": "done"}
        if activity is not None and activity not in next_stage:
            raise ValueError("无效的英语课程阶段")
        if review_checkpoint is not None and (activity != "introduce" or not advance
                or review_checkpoint not in {"next_day", "week_later"}):
            raise ValueError("无效的英语复习记录")
        with self.connect() as db:
            now = time.time()
            db.execute("""INSERT INTO english_progress
                          (device_id, person_id, word, presentations, text_matches, updated_at, review_anchor_at)
                          VALUES (?, ?, ?, 1, ?, ?, ?)
                          ON CONFLICT(device_id, person_id, word) DO UPDATE SET
                          presentations=presentations+1,
                          text_matches=text_matches+excluded.text_matches,
                          updated_at=excluded.updated_at""",
                       (device_id, person_id, word, int(text_match), now, now))
            if activity is not None and advance:
                db.execute("""INSERT INTO english_lessons VALUES (?, ?, ?, ?, ?)
                              ON CONFLICT(device_id, person_id) DO UPDATE SET
                              word=excluded.word, stage=excluded.stage,
                              updated_at=excluded.updated_at""",
                           (device_id, person_id, word, next_stage[activity], time.time()))
            if review_checkpoint is not None:
                db.execute("""INSERT INTO english_review_prompts VALUES (?,?,?,?,?)
                              ON CONFLICT(device_id,person_id,word,checkpoint) DO UPDATE SET
                              prompted_at=excluded.prompted_at""",
                           (device_id, person_id, word, review_checkpoint, now))

    def save_action(self, name: str, steps: list[dict]):
        with self.connect() as db:
            db.execute("""INSERT INTO actions VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET steps=excluded.steps, updated_at=excluded.updated_at""",
                (name, json.dumps(steps, ensure_ascii=False), time.time()))

    def actions(self) -> dict[str, list[dict]]:
        with self.connect() as db:
            rows = db.execute("SELECT name, steps FROM actions ORDER BY name").fetchall()
        return {row["name"]: json.loads(row["steps"]) for row in rows}

    def add_photo(self, photo_id: str, device_id: str, file_path: str):
        with self.connect() as db:
            db.execute("INSERT INTO photos VALUES (?, ?, ?, ?)",
                       (photo_id, device_id, file_path, time.time()))

    def photos(self, device_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""SELECT photo_id, device_id, captured_at FROM photos
                                 WHERE device_id=? ORDER BY captured_at DESC LIMIT 20""",
                              (device_id,)).fetchall()
        return [dict(row) for row in rows]

    def photo_file(self, photo_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT file_path FROM photos WHERE photo_id=?", (photo_id,)).fetchone()
        return row["file_path"] if row else None
