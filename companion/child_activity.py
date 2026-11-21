"""Small, private voice-session state for child-selected activities.

This state holds only an activity name and one English word/stage. It expires
without disk writes and never grants access to a member's saved progress.
"""

import re
import threading
import time

from .story_routing import requests_story


_ENGLISH_START = re.compile(
    r"\s*(?:(?:小狗|你)[，,\s]*)?(?:请|我们|给我|跟我)?(?:一起|开始|继续|再)?"
    r"(?:学|玩|练)(?:点|一点)?英语(?:游戏|单词)?(?:吧|呀|啦)?[。.!！\s]*$"
)
_TEACH_ENGLISH = re.compile(r"\s*(?:(?:小狗|你)[，,\s]*)?(?:请)?教我(?:一点|点)?英语(?:吧|呀|啦)?[。.!！\s]*$")
_TEACH_WORD = re.compile(
    r"\s*(?:(?:小狗|你)[，,\s]*)?(?:请)?[教叫]我(?:一[个個]|[个個])?英文[单單][词詞]"
    r"(?:吧|呀|啦)?[。.!！\s]*$"
)
_CHAT_START = re.compile(r"\s*(?:(?:小狗|你)[，,\s]*)?(?:我们|跟我|陪我|换成)?(?:聊聊天|聊天|说说话)(?:吧|呀|啦)?[。.!！\s]*$")
_STOP = re.compile(
    r"\s*(?:我)?(?:先)?(?:不学(?:英语)?了|不要学(?:英语)?了|不想学(?:英语)?了|不玩(?:英语)?了|不想玩了|"
    r"不想听了|不讲故事了|别讲了|英语先停一下|英语暂停|英语先暂停|停|停一下|暂停|不玩了|结束(?:英语|故事|聊天)?|先休息一下)"
    r"[。.!！\s]*$"
)
_CONTINUE = re.compile(r"\s*(?:继续|接着|再来)(?:吧|呀|啦)?[。.!！\s]*$")
_SPOKEN_FORMS = str.maketrans({"學": "学", "語": "语", "講": "讲", "們": "们", "請": "请",
                              "個": "个", "這": "这", "換": "换", "結": "结", "續": "续",
                              "別": "别", "練": "练", "點": "点"})


def requests_english(text: str) -> bool:
    text = text.translate(_SPOKEN_FORMS)
    return bool(_ENGLISH_START.fullmatch(text) or _TEACH_ENGLISH.fullmatch(text)
                or _TEACH_WORD.fullmatch(text))


def requests_chat(text: str) -> bool:
    text = text.translate(_SPOKEN_FORMS)
    return bool(_CHAT_START.fullmatch(text))


def stops_activity(text: str) -> bool:
    text = text.translate(_SPOKEN_FORMS)
    return bool(_STOP.fullmatch(text))


class ChildActivitySessions:
    TTL_SECONDS = 30 * 60

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._lock = threading.RLock()
        self._sessions = {}

    def _get(self, device_id: str, session_id: str, *, create=False):
        now = self.clock()
        for key, value in tuple(self._sessions.items()):
            if now - value["at"] >= self.TTL_SECONDS:
                del self._sessions[key]
        key = (device_id, session_id)
        value = self._sessions.get(key)
        if value is None and create:
            value = {"mode": None, "resume": None, "lessons": {}, "at": now}
            self._sessions[key] = value
        if value is not None:
            value["at"] = now
        return value

    def route(self, device_id: str, session_id: str, text: str, default_mode: str) -> str:
        text = text.translate(_SPOKEN_FORMS)
        with self._lock:
            state = self._get(device_id, session_id, create=True)
            current = state["mode"] or default_mode
            if stops_activity(text):
                state["resume"] = current if current in {"english", "story"} else None
                state["mode"] = "chat"
            elif requests_english(text):
                state["mode"], state["resume"] = "english", None
            elif requests_story(text):
                state["mode"], state["resume"] = "story", None
            elif requests_chat(text):
                state["mode"], state["resume"] = "chat", None
            elif _CONTINUE.fullmatch(text) and state["resume"]:
                state["mode"], state["resume"] = state["resume"], None
            return state["mode"] or default_mode

    def lesson(self, device_id: str, session_id: str, person_id: str | None):
        with self._lock:
            state = self._get(device_id, session_id)
            if state is None:
                return None
            value = state["lessons"].get(person_id or "")
            return dict(value) if value else None

    def record_lesson(self, device_id: str, session_id: str, person_id: str | None,
                      word: str, activity: str, *, advance: bool = True):
        next_stage = {"introduce": "repeat", "repeat": "play",
                      "play": "respond", "respond": "done"}
        if activity not in next_stage:
            raise ValueError("无效的英语课程阶段")
        with self._lock:
            state = self._get(device_id, session_id, create=True)
            state["lessons"][person_id or ""] = {
                "word": word, "stage": next_stage[activity] if advance else activity}

    def finish(self, device_id: str, session_id: str):
        with self._lock:
            state = self._get(device_id, session_id, create=True)
            state["mode"], state["resume"] = "chat", None

    def clear_device(self, device_id: str):
        with self._lock:
            for key in tuple(self._sessions):
                if key[0] == device_id:
                    del self._sessions[key]
