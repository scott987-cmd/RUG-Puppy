"""An explicit visual question may request one ephemeral camera observation.

This module accepts text and image callbacks only. It neither records audio nor
stores photographs; the caller supplies the current parent permission check.
"""
import asyncio
import re
import time


# Over-include possible references to the current scene. The model still decides
# whether to open the camera; this gate only saves a model call for plainly
# nonvisual questions such as arithmetic, English vocabulary or storytelling.
_SCENE_CUES_ZH = (
    "看看", "瞧瞧", "你看", "帮我看", "幫我看", "拍", "摄", "攝", "鏡", "镜",
    "眼前", "面前", "前面", "后面", "後面", "旁边", "旁邊",
    "周围", "周圍", "四周", "这里", "這裡", "那里", "那裡", "手里", "手裡", "手上",
    "拿着", "拿著", "举着", "舉著", "穿着", "穿著", "衣服", "桌上", "桌子",
    "地上", "房间", "房間",
)
_SCENE_CUES_EN = re.compile(
    r"\b(?:look|show|photo|picture|image|camera|colou?r|"
    r"hold|holding|wear|wearing|shirt|dress|table|floor|room|around|"
    r"behind|beside|object|identify|finger|hand)\b|\bin front\b", re.I)
_DEICTIC_SCENE_EN = re.compile(
    r"\b(?:what(?:'s| is)|which (?:object|one)(?: is)?)\s+"
    r"(?:(?:over|right)\s+)?(?:here|there)[?.!\s]*$", re.I)
_DISCOURSE_THEN = re.compile(
    r"(?:^|(?<=[，,。.!！?？\s]))那(?:[么麼]|(?=[，,。.!！?？\s]|再|就|接[着著]|如果|[为為]什))"
)
_DEICTIC_OBJECT = re.compile(
    r"^(?:这|這|那)(?:个|個|些|边|邊)?(?:东西|東西|物体|物體)?"
    r"(?:是(?:什[么麼]|啥)|有(?:几|幾|多少)|什[么麼][颜顏]色)"
    r"[^。！？!?]{0,12}[。！？!?\s]*$"
)


def may_request_current_scene(text: str) -> bool:
    # A conversational "then" does not point at something in the room.
    text = _DISCOURSE_THEN.sub("", text)
    return (any(cue in text for cue in _SCENE_CUES_ZH)
            or bool(_SCENE_CUES_EN.search(text)) or bool(_DEICTIC_SCENE_EN.search(text))
            or bool(_DEICTIC_OBJECT.fullmatch(text.strip())))


class CameraDialogue:
    def __init__(self, plan, capture, describe, *, clock=time.time, timeout=12,
                 max_age=15):
        self.plan, self.capture, self.describe = plan, capture, describe
        self.clock, self.timeout, self.max_age = clock, timeout, max_age

    def fresh_times(self, times):
        if not isinstance(times, (list, tuple)) or len(times) != 2:
            return False
        start, end = times
        return (type(start) in (int, float) and type(end) in (int, float)
                and self.clock() - self.max_age <= start <= end <= self.clock())

    def fresh(self, photo):
        if not isinstance(photo, dict) or not isinstance(photo.get("_photo_bytes"), bytes):
            return False
        return (self.fresh_times((photo.get("requested_at"), photo.get("captured_at")))
                and 0 < len(photo["_photo_bytes"]) <= 512000)

    @staticmethod
    def unavailable(reason):
        return {"observation": {"available": False, "reason": reason,
                                "uncertain": True}, "used_photo": False}

    async def prepare(self, device_id, text, *, allowed=lambda: True, recent_photo=None, deadline=None):
        if not may_request_current_scene(text):
            return None
        budget = asyncio.get_running_loop().time() + self.timeout
        deadline = min(budget, deadline) if deadline is not None else budget
        try:
            async with asyncio.timeout_at(deadline):
                return await self._prepare(device_id, text, allowed, recent_photo, deadline)
        except (TimeoutError, ValueError, RuntimeError, OSError):
            return self.unavailable("本轮摄像头或图像分析暂不可用，不能判断眼前物品")

    async def _prepare(self, device_id, text, allowed, recent_photo, deadline):
        # Only the local model decides whether this utterance explicitly asks
        # to look at the present scene. This is not a general background camera.
        wants_camera = await self.plan(text)
        if wants_camera is not True:
            return None
        if not allowed():
            return self.unavailable("家长尚未允许本轮看图学习；没有拍照")
        photo = recent_photo if self.fresh(recent_photo) else None
        if photo is None:
            # The gateway evaluates permission again after both of its locks,
            # immediately before sending. Checking only here is insufficient.
            photo = await self.capture(device_id, before_send=allowed, deadline=deadline)
        if not allowed():
            return self.unavailable("家长已关闭看图学习；不要使用照片内容")
        if not self.fresh(photo):
            return self.unavailable("本轮没有获得足够新鲜的照片，不能判断眼前物品")
        observation = await self.describe(photo["_photo_bytes"], text)
        if not allowed():
            return self.unavailable("家长已关闭看图学习；不要使用照片内容")
        if not self.fresh(photo):
            return self.unavailable("分析期间照片已过期，不能用它判断眼前物品")
        if (not isinstance(observation, dict) or not isinstance(observation.get("reply"), str)
                or not observation["reply"].strip() or type(observation.get("uncertain")) is not bool):
            return self.unavailable("这次图像分析未成功，不能编造照片内容")
        return {"observation": {**observation, "available": True, "scope": "single_photo"},
                "used_photo": True, "times": (photo["requested_at"], photo["captured_at"])}
