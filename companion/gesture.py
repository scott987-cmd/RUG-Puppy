"""Short, parent-enabled gesture sessions; no camera/device/model opened on import.

Capture must make a fresh, cancellable request and honor retain_photo=False.
Recognize is async (the provided subprocess adapter is isolated and killable).
Execute is the gateway's safe-action coroutine. It accepts before_send/deadline,
checks them immediately before sending, and on cancellation drains its own job
cleanup before returning. All callbacks run only after enable(); sessions are
never persisted.
"""
import asyncio
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import time
import uuid


MAPPING = {"Open_Palm": "stop", "Thumb_Up": "wave", "Closed_Fist": "sit"}
MAX_JPEG_BYTES = 2_000_000


@dataclass(frozen=True)
class GestureObservation:
    kind: str  # gesture, absent, uncertain
    label: str = ""
    score: float = 0.0
    reason: str = ""

    def __post_init__(self):
        if self.kind not in {"gesture", "absent", "uncertain"}:
            raise ValueError("invalid gesture result")
        if (type(self.score) not in {int, float} or not math.isfinite(self.score)
                or not 0 <= self.score <= 1):
            raise ValueError("invalid gesture confidence")
        if self.kind == "gesture" and (self.label not in MAPPING or self.score < .85):
            raise ValueError("uncertain gesture cannot authorize movement")
        if not isinstance(self.reason, str) or len(self.reason) > 80:
            raise ValueError("invalid gesture reason")


class SubprocessRecognizer:
    """One image per isolated process; no imports of MediaPipe in the gateway."""
    def __init__(self, python_path, model_path, *, timeout=8):
        self.python_path = str(Path(python_path).expanduser().absolute())
        self.model_path = str(Path(model_path).expanduser().absolute())
        self.worker = str(Path(__file__).with_name("gesture_worker.py"))
        self.timeout = timeout
        self.busy = False

    @property
    def available(self):
        return all(Path(p).is_file() for p in (self.python_path, self.model_path, self.worker))

    async def __call__(self, jpeg):
        if self.busy:
            raise RuntimeError("gesture worker busy")
        if not self.available or not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG_BYTES:
            raise ValueError("gesture input unavailable")
        self.busy = True
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                self.python_path, self.worker, self.model_path,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)

            async def exchange():
                process.stdin.write(jpeg)
                await process.stdin.drain()
                process.stdin.close()
                # The result contains no image/landmarks; excessive output is an error.
                result = bytearray()
                while chunk := await process.stdout.read(4096):
                    result.extend(chunk)
                    if len(result) > 16384:
                        raise ValueError("gesture worker output too large")
                await process.wait()
                if process.returncode:
                    raise RuntimeError("gesture worker failed")
                return GestureObservation(**json.loads(result))

            return await asyncio.wait_for(exchange(), self.timeout)
        finally:
            if process is None:
                self.busy = False
            else:
                async def reap():
                    try:
                        if process.returncode is None:
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass  # It exited between the flag read and kill.
                        await process.wait()
                    finally:
                        self.busy = False

                cleanup = asyncio.create_task(reap())
                interrupted = False
                while True:
                    try:
                        await asyncio.shield(cleanup)
                        break
                    except asyncio.CancelledError:
                        if cleanup.cancelled():
                            raise
                        # Stage timeout and the session watchdog can cancel the
                        # same caller. Neither may abandon process reaping or
                        # allow a new worker before the old one has exited.
                        interrupted = True
                if interrupted:
                    raise asyncio.CancelledError


@dataclass
class _Session:
    device_id: str
    deadline: float
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enabled: bool = True
    reason: str = "enabled"
    candidate: str = ""
    confirmations: int = 0
    absent_frames: int = 0
    latched: bool = False
    stop_latched: bool = False
    movement_count: int = 0
    cooldown_until: float = 0
    last_observation: str = "waiting"
    last_action: str | None = None
    last_result: dict | None = None
    loop: asyncio.Task | None = None
    watchdog: asyncio.Task | None = None
    action: asyncio.Task | None = None
    cleanup: asyncio.Task | None = None
    action_cleanup: asyncio.Task | None = None


def _device(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9:_.-]{1,64}", value):
        raise ValueError("无效设备 ID")
    return value


class GestureService:
    """A single in-memory session, one capture/recognition at a time.

Only motion (wave/sit) consumes the maximum-three budget. Recognized stop is
always available during the enabled session, even while cooling down or at the
motion limit. A parent's ordinary emergency-stop button remains independent.
"""
    def __init__(self, *, capture_handler, recognize_handler, execute_handler,
                 online_handler, clock=time.monotonic, poll_seconds=.8,
                 stage_timeout=8, cooldown_seconds=3, watchdog_seconds=.1):
        self.capture, self.recognize, self.execute = capture_handler, recognize_handler, execute_handler
        self.online = online_handler
        self.clock = clock
        self.poll_seconds, self.stage_timeout = poll_seconds, stage_timeout
        self.cooldown_seconds, self.watchdog_seconds = cooldown_seconds, watchdog_seconds
        if min(poll_seconds, stage_timeout, cooldown_seconds, watchdog_seconds) <= 0:
            raise ValueError("手势运行上限必须为正数")
        self._session = None
        self._closed = False

    @property
    def available(self):
        return self.recognize is not None and getattr(self.recognize, "available", True)

    def _valid(self, session):
        return (self._session is session and session.enabled and not self._closed
                and self.clock() < session.deadline and self.online(session.device_id))

    def status(self, device_id):
        _device(device_id)
        session = self._session
        result = {"device_id": device_id, "available": bool(self.available), "enabled": False,
                  "remaining_seconds": 0, "movement_count": 0, "movement_limit": 3,
                  "max_seconds": 60, "mapping": dict(MAPPING), "reason": "disabled",
                  "photos_retained": False, "busy": False}
        if session and session.device_id == device_id:
            enabled = self._valid(session)
            result.update(enabled=bool(enabled), session_id=session.session_id,
                          remaining_seconds=max(0, math.ceil(session.deadline - self.clock())) if enabled else 0,
                          movement_count=session.movement_count, reason=session.reason,
                          last_observation=session.last_observation, last_action=session.last_action,
                          last_result=session.last_result,
                          busy=bool(session.cleanup and not session.cleanup.done()))
        elif session and (session.enabled or (session.cleanup and not session.cleanup.done())):
            result.update(busy=True, reason="other_device_busy")
        return result

    async def enable(self, device_id, seconds=60):
        _device(device_id)
        if type(seconds) is not int or not 1 <= seconds <= 60:
            raise ValueError("手势开启时长须为 1 至 60 秒")
        if self._closed or not self.available:
            raise RuntimeError("本地手势模型暂不可用")
        if not self.online(device_id):
            raise RuntimeError("Puppy 当前不在线")
        old = self._session
        if old and (old.enabled or (old.cleanup and not old.cleanup.done())):
            raise RuntimeError("请先关闭当前手势会话并等待清理完成")
        session = self._session = _Session(device_id, self.clock() + seconds)
        session.loop = asyncio.create_task(self._run(session))
        session.watchdog = asyncio.create_task(self._watch(session))
        return self.status(device_id)

    def _end(self, session, reason):
        # Revoke permission synchronously before cancellation/cleanup can yield.
        if session.enabled:
            session.enabled, session.reason = False, reason
        if session.cleanup is None:
            session.cleanup = asyncio.create_task(self._cleanup(session))
        return session.cleanup

    async def disable(self, device_id):
        _device(device_id)
        session = self._session
        if session and session.device_id == device_id:
            await asyncio.shield(self._end(session, "parent_disabled"))
        return self.status(device_id)

    async def close(self):
        self._closed = True
        if self._session:
            await asyncio.shield(self._end(self._session, "service_closed"))

    async def _cleanup(self, session):
        # Start cancelling our action before joining camera/recognizer cleanup.
        # A slow or cancellation-resistant worker must not extend permission
        # for a motor job while its process is being reaped.
        action_cleanup = asyncio.create_task(self._cancel_owned_action(session))
        for task in (session.loop, session.watchdog):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(action_cleanup, *(t for t in (session.loop, session.watchdog) if t),
                             return_exceptions=True)

    async def _cancel_owned_action(self, session):
        # Cancel only our own pending safe-action call. That executor owns its
        # job cleanup. Never issue an unconditional stop that could hit another job.
        if session.action and not session.action.done():
            if session.action_cleanup is None or session.action_cleanup.done():
                action = session.action
                if not action.cancelling():
                    action.cancel()

                async def drain():
                    await asyncio.gather(action, return_exceptions=True)
                session.action_cleanup = asyncio.create_task(drain())
            await asyncio.shield(session.action_cleanup)

    async def _watch(self, session):
        while session.enabled:
            if self.clock() >= session.deadline:
                self._end(session, "expired")
                return
            if not self.online(session.device_id):
                self._end(session, "disconnected")
                return
            await asyncio.sleep(self.watchdog_seconds)

    async def _stage(self, session, operation):
        remaining = session.deadline - self.clock()
        return await asyncio.wait_for(operation, max(.001, min(self.stage_timeout, remaining)))

    async def _run(self, session):
        try:
            while self._valid(session):
                try:
                    jpeg = await self._stage(session, self.capture(session.device_id, retain_photo=False))
                    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG_BYTES:
                        raise ValueError("invalid JPEG")
                except Exception:
                    self._end(session, "capture_failed")
                    return
                if not self._valid(session):
                    return
                try:
                    observation = await self._stage(session, self.recognize(jpeg))
                    if not isinstance(observation, GestureObservation):
                        raise ValueError("invalid observation")
                except Exception:
                    self._end(session, "recognition_failed")
                    return
                finally:
                    jpeg = None  # No photos retained in service state or logs.
                if not self._valid(session):
                    return
                await self._observe(session, observation)
                await asyncio.sleep(self.poll_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._end(session, "session_failed")

    async def _observe(self, session, result):
        if not self._valid(session):
            return
        session.last_observation = result.label if result.kind == "gesture" else result.kind
        if result.kind != "gesture":
            session.candidate, session.confirmations = "", 0
            session.absent_frames = session.absent_frames + 1 if result.kind == "absent" else 0
            # Low confidence, unsupported poses and multiple hands do not rearm.
            if session.absent_frames >= 2:
                session.latched = session.stop_latched = False
            await self._cancel_owned_action(session)
            return
        session.absent_frames = 0
        if result.label != session.candidate:
            session.candidate, session.confirmations = result.label, 1
        else:
            session.confirmations += 1
        if session.confirmations < 2:
            return
        action = MAPPING[result.label]
        if action == "stop":
            if session.stop_latched:
                return
            session.stop_latched = session.latched = True
            await self._cancel_owned_action(session)
        else:
            if session.latched:
                return
            session.latched = True  # Busy/cooling gestures aren't queued for later.
            if (session.movement_count >= 3 or self.clock() < session.cooldown_until
                    or (session.action and not session.action.done())):
                return
        # Recheck after any cancellation wait, immediately before creating work.
        if self._valid(session):
            session.action = asyncio.create_task(self._execute(session, action))

    async def _execute(self, session, action):
        if not self._valid(session):
            return
        session.last_action = action
        if action != "stop":
            session.movement_count += 1
            session.cooldown_until = self.clock() + self.cooldown_seconds
        session.last_result = {"state": "executing"}
        try:
            # The injected clock may be a test clock. Translate its remaining
            # budget to the event loop clock used by DeviceHub.timeout_at().
            deadline = asyncio.get_running_loop().time() + max(0, session.deadline - self.clock())
            receipt = await self.execute(session.device_id, action,
                                         before_send=lambda: self._valid(session), deadline=deadline)
            if not isinstance(receipt, dict) or receipt.get("ok") is not True:
                session.last_result = {"state": "failed"}
                self._end(session, "action_failed")
            elif action != "stop" and receipt.get("phase") != "completed":
                session.last_result = {"state": "unknown"}
                self._end(session, "action_unknown")
            else:
                session.last_result = {"state": "completed"}
        except asyncio.CancelledError:
            # Cancellation is not a claim that the physical robot has stopped.
            session.last_result = {"state": "cancelled", "physical_stop_confirmed": False}
            raise
        except Exception:
            session.last_result = {"state": "failed"}
            self._end(session, "action_failed")
