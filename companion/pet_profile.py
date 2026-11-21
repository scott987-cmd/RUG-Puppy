"""Parent-owned shared pet style and an explicit greeting convention."""

import re
import time

from .store import Store

STYLES = {"gentle": "温和", "lively": "活泼", "patient": "耐心"}
GREETING_ACTIONS = frozenset({"none", "wave", "sit", "stretch", "hug"})


class RevisionConflict(ValueError):
    pass


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w:.-]{1,64}", value):
        raise ValueError("设备、成员和流程 ID 须为 1–64 个字母、数字或分隔符")
    return value


def revision(value):
    if type(value) is not int or value < 0:
        raise ValueError("版本必须为非负整数")
    return value


class PetProfile:
    def __init__(self, store: Store):
        self.store = store

    def get(self, device_id):
        device_id = identifier(device_id)
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM pet_profiles WHERE device_id=?", (device_id,)).fetchone()
        return dict(row) if row else {"device_id": device_id, "style": "gentle",
                                     "greeting_action": "none", "revision": 0, "updated_at": None}

    def set(self, device_id, *, style, greeting_action, expected_revision):
        device_id = identifier(device_id)
        revision(expected_revision)
        if (not isinstance(style, str) or not isinstance(greeting_action, str)
                or style not in STYLES or greeting_action not in GREETING_ACTIONS):
            raise ValueError("请选择已有的性格和招呼动作")
        with self.store.connect() as db:
            self.store._begin_immediate(db)
            current = db.execute("SELECT revision FROM pet_profiles WHERE device_id=?", (device_id,)).fetchone()
            if (current["revision"] if current else 0) != expected_revision:
                raise RevisionConflict("宠物资料已改变，请刷新后重试")
            db.execute("""INSERT INTO pet_profiles VALUES(?,?,?,?,?) ON CONFLICT(device_id)
                          DO UPDATE SET style=excluded.style,greeting_action=excluded.greeting_action,
                          revision=excluded.revision,updated_at=excluded.updated_at""",
                       (device_id, style, greeting_action, expected_revision + 1, time.time()))
            return dict(db.execute("SELECT * FROM pet_profiles WHERE device_id=?", (device_id,)).fetchone())
