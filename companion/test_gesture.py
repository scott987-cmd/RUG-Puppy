"""No real photographs, cameras, devices or audio in these regressions."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from .gesture import GestureObservation as O, GestureService, SubprocessRecognizer


class GestureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 100.
        self.connected = True
        self.frames = asyncio.Queue()
        self.captured = []
        self.observed = 0
        self.actions = []
        self.cancelled = []
        self.hold_action = False
        self.hold_recognition = False
        self.recognition_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.service = self.new_service()

    def new_service(self):
        return GestureService(capture_handler=self.capture, recognize_handler=self.recognize,
                              execute_handler=self.execute, online_handler=lambda d: self.connected,
                              clock=lambda: self.now, poll_seconds=.002, watchdog_seconds=.002,
                              stage_timeout=.3, cooldown_seconds=3)

    async def asyncTearDown(self):
        self.release.set()
        await self.service.close()

    async def capture(self, device, *, retain_photo):
        value = await self.frames.get()
        self.captured.append(retain_photo)
        if isinstance(value, Exception):
            raise value
        self.current = value
        return b"fake-jpeg" if value is not None else b""

    async def recognize(self, jpeg):
        self.recognition_entered.set()
        if self.hold_recognition:
            await self.release.wait()
        self.observed += 1
        return self.current

    async def execute(self, device, action, *, before_send, deadline):
        self.assertTrue(before_send())
        self.assertGreater(deadline, asyncio.get_running_loop().time())
        self.actions.append(action)
        try:
            if self.hold_action and action != "stop":
                await self.release.wait()
            return {"ok": True, "phase": "completed"}
        except asyncio.CancelledError:
            self.cancelled.append(action)
            raise

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(.002)

    async def feed(self, *frames):
        expected = self.observed + len(frames)
        for frame in frames:
            self.frames.put_nowait(frame)
        await self.until(lambda: self.observed >= expected)
        await asyncio.sleep(.006)

    async def gesture(self, label="Thumb_Up"):
        await self.feed(O("gesture", label, .95), O("gesture", label, .96))

    async def reset_hand(self):
        await self.feed(O("absent"), O("absent"))

    async def test_default_disabled_restart_and_no_capture(self):
        self.frames.put_nowait(O("gesture", "Thumb_Up", .99))
        await asyncio.sleep(.01)
        self.assertFalse(self.service.status("puppy")["enabled"])
        self.assertEqual(self.captured, [])
        await self.service.enable("puppy")
        fresh = self.new_service()
        self.assertFalse(fresh.status("puppy")["enabled"])
        await fresh.close()

    async def test_enable_is_explicit_bounded_and_cannot_extend_live_session(self):
        for seconds in (0, 61, True, "60"):
            with self.assertRaises(ValueError):
                await self.service.enable("puppy", seconds)
        self.connected = False
        with self.assertRaises(RuntimeError):
            await self.service.enable("puppy")
        self.connected = True
        await self.service.enable("puppy", 20)
        with self.assertRaises(RuntimeError):
            await self.service.enable("puppy", 60)
        with self.assertRaises(RuntimeError):
            await self.service.enable("other", 60)
        self.assertEqual(self.service.status("puppy")["remaining_seconds"], 20)

    async def test_two_distinct_capture_results_single_pose_and_no_retention(self):
        await self.service.enable("puppy")
        await self.feed(O("gesture", "Thumb_Up", .95))
        self.assertEqual(self.actions, [])
        await self.feed(O("gesture", "Closed_Fist", .95))
        self.assertEqual(self.actions, [])
        await self.feed(O("gesture", "Closed_Fist", .95))
        self.assertEqual(self.actions, ["sit"])
        self.assertEqual(self.captured, [False, False, False])

    async def test_hold_jitter_and_changed_pose_do_not_rearm_without_absence(self):
        await self.service.enable("puppy")
        await self.gesture()
        self.now += 4
        await self.gesture()
        await self.feed(O("uncertain", reason="low_confidence"))
        await self.gesture("Closed_Fist")
        await self.feed(O("absent"), O("uncertain"))
        await self.gesture()
        self.assertEqual(self.actions, ["wave"])
        await self.reset_hand()
        await self.gesture("Closed_Fist")
        self.assertEqual(self.actions, ["wave", "sit"])

    async def test_cooldown_consumes_pose_no_delayed_replay(self):
        await self.service.enable("puppy")
        await self.gesture()
        await self.reset_hand()
        await self.gesture("Closed_Fist")
        self.now += 4
        await self.gesture("Closed_Fist")
        self.assertEqual(self.actions, ["wave"])

    async def test_three_motion_limit_still_allows_confirmed_stop(self):
        await self.service.enable("puppy")
        for _ in range(4):
            await self.reset_hand()
            self.now += 4
            await self.gesture()
        self.assertEqual(self.actions, ["wave"] * 3)
        await self.gesture("Open_Palm")
        await self.gesture("Open_Palm")
        self.assertEqual(self.actions, ["wave"] * 3 + ["stop"])
        self.assertEqual(self.service.status("puppy")["movement_count"], 3)

    async def test_uncertain_cancels_only_own_running_action(self):
        self.hold_action = True
        self.release.clear()
        await self.service.enable("puppy")
        await self.gesture()
        await self.feed(O("uncertain", reason="not_one_hand"))
        self.assertEqual(self.cancelled, ["wave"])
        self.assertEqual(self.actions, ["wave"])  # No unconditional global stop.
        self.assertFalse(self.service.status("puppy")["last_result"]["physical_stop_confirmed"])

    async def test_no_hand_cancels_motion_and_confirmed_palm_ignores_cooldown(self):
        self.hold_action = True
        self.release.clear()
        await self.service.enable("puppy")
        await self.gesture()
        await self.feed(O("absent"))
        self.assertEqual(self.cancelled, ["wave"])
        await self.gesture("Open_Palm")
        self.assertEqual(self.actions, ["wave", "stop"])

    async def test_empty_capture_and_missing_model_never_reach_executor(self):
        self.service.recognize = None
        with self.assertRaises(RuntimeError):
            await self.service.enable("puppy")
        self.service.recognize = self.recognize
        await self.service.enable("puppy")
        self.frames.put_nowait(None)
        await self.until(lambda: not self.service.status("puppy")["enabled"])
        self.assertEqual(self.service.status("puppy")["reason"], "capture_failed")
        self.assertEqual(self.observed, 0)
        self.assertEqual(self.actions, [])

    async def test_disable_after_completed_action_does_not_stop_another_job(self):
        await self.service.enable("puppy")
        await self.gesture()
        await self.service.disable("puppy")
        self.assertEqual(self.actions, ["wave"])
        self.assertEqual(self.cancelled, [])

    async def test_capture_failure_closes_session_and_cancels_motion(self):
        self.hold_action = True
        self.release.clear()
        await self.service.enable("puppy")
        await self.gesture()
        self.frames.put_nowait(RuntimeError("camera failed"))
        await self.until(lambda: bool(self.cancelled))
        self.assertEqual(self.service.status("puppy")["reason"], "capture_failed")
        self.assertEqual(self.actions, ["wave"])

    async def test_disconnect_and_expiry_cancel_pending_action(self):
        for reason in ("disconnected", "expired"):
            with self.subTest(reason=reason):
                self.connected = True
                self.hold_action = True
                self.release.clear()
                await self.service.enable("puppy")
                await self.gesture()
                if reason == "disconnected":
                    self.connected = False
                else:
                    self.now += 60
                session = self.service._session
                await self.until(lambda: session.cleanup is not None)
                await session.cleanup
                self.assertEqual(self.service.status("puppy")["reason"], reason)
        self.assertEqual(self.cancelled, ["wave", "wave"])

    async def test_expired_inference_result_cannot_dispatch(self):
        await self.service.enable("puppy", 1)
        await self.feed(O("gesture", "Thumb_Up", .95))
        self.hold_recognition = True
        self.recognition_entered.clear()
        self.release.clear()
        self.frames.put_nowait(O("gesture", "Thumb_Up", .95))
        await self.recognition_entered.wait()
        self.now += 1
        self.release.set()
        await asyncio.sleep(.02)
        self.assertEqual(self.actions, [])
        self.assertFalse(self.service.status("puppy")["enabled"])

    async def test_recognition_timeout_closes_without_replay(self):
        self.hold_recognition = True
        self.release.clear()
        self.service.stage_timeout = .015
        await self.service.enable("puppy")
        self.frames.put_nowait(O("gesture", "Thumb_Up", .95))
        await self.until(lambda: not self.service.status("puppy")["enabled"])
        self.release.set()
        await self.service._session.cleanup
        self.assertEqual(self.service.status("puppy")["reason"], "recognition_failed")
        self.assertEqual(self.actions, [])

    async def test_parent_close_waits_for_old_worker_before_new_enable(self):
        cancelled = asyncio.Event()
        release_old = asyncio.Event()
        async def stubborn(jpeg):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release_old.wait()
                return O("gesture", "Thumb_Up", .99)
        self.service.recognize = stubborn
        await self.service.enable("puppy")
        self.frames.put_nowait(O("gesture", "Thumb_Up", .99))
        await self.until(lambda: bool(self.captured))
        await asyncio.sleep(.01)
        closing = asyncio.create_task(self.service.disable("puppy"))
        await cancelled.wait()
        with self.assertRaises(RuntimeError):
            await self.service.enable("puppy")
        release_old.set()
        await closing
        self.assertEqual(self.actions, [])

    async def test_expiry_cancels_action_before_stubborn_recognizer_finishes(self):
        self.hold_action = True
        self.release.clear()
        await self.service.enable("puppy")
        await self.gesture()
        entered, cancelled, release_old = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def stubborn(jpeg):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release_old.wait()
                return O("gesture", "Thumb_Up", .99)

        self.service.recognize = stubborn
        self.frames.put_nowait(O("gesture", "Thumb_Up", .99))
        await entered.wait()
        self.now += 60
        try:
            await cancelled.wait()
            await self.until(lambda: self.cancelled == ["wave"])
            self.assertFalse(release_old.is_set())
            self.assertFalse(self.service._session.cleanup.done())
            with self.assertRaises(RuntimeError):
                await self.service.enable("puppy")
        finally:
            release_old.set()
            await self.service._session.cleanup
        self.assertEqual(self.actions, ["wave"])

    async def test_send_guard_rejects_late_work_and_uses_loop_deadline(self):
        entered, release_send, cancel_seen = asyncio.Event(), asyncio.Event(), asyncio.Event()
        callbacks, deadlines, sent = [], [], []

        async def delayed(device, action, *, before_send, deadline):
            callbacks.append(before_send)
            deadlines.append(deadline)
            entered.set()
            try:
                await release_send.wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                await release_send.wait()  # Simulate a sender finishing its wait.
            if before_send() and asyncio.get_running_loop().time() < deadline:
                sent.append(action)
            return {"ok": bool(sent), "phase": "completed"}

        self.service.execute = delayed
        started = asyncio.get_running_loop().time()
        await self.service.enable("puppy", 2)
        await self.gesture()
        await entered.wait()
        self.assertGreaterEqual(deadlines[0], started + 1.9)
        self.assertLessEqual(deadlines[0], asyncio.get_running_loop().time() + 2)
        self.now += 2
        try:
            await cancel_seen.wait()
            self.assertFalse(callbacks[0]())
            with self.assertRaises(RuntimeError):
                await self.service.enable("puppy")
        finally:
            release_send.set()
            await self.service._session.cleanup
        self.assertEqual(sent, [])
        await self.service.enable("puppy")
        self.assertFalse(callbacks[0]())  # New consent cannot revive an old send.

    async def test_unknown_terminal_result_revokes_permission(self):
        async def nonterminal(device, action, *, before_send, deadline):
            return {"ok": True, "phase": "accepted"}
        self.service.execute = nonterminal
        await self.service.enable("puppy")
        await self.gesture()
        await self.until(lambda: not self.service.status("puppy")["enabled"])
        self.assertEqual(self.service.status("puppy")["last_result"]["state"], "unknown")

    async def test_api_requires_parent_and_new_consent_and_strict_fields(self):
        import httpx
        from fastapi import FastAPI, Header, HTTPException
        from .gesture_api import create_gesture_router
        async def parent(authorization: str | None = Header(default=None)):
            if authorization != "Bearer test":
                raise HTTPException(401, "parent required")
        app = FastAPI()
        app.include_router(create_gesture_router(self.service, parent))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            self.assertEqual((await client.get("/api/gestures/devices/puppy")).status_code, 401)
            client.headers["Authorization"] = "Bearer test"
            for body in ({}, {"consent": False}, {"consent": True, "seconds": 61},
                         {"consent": True, "seconds": True}, {"consent": True, "action": "hug"}):
                self.assertEqual((await client.post("/api/gestures/devices/puppy/enable", json=body)).status_code, 422)
            reply = await client.post("/api/gestures/devices/puppy/enable", json={"consent": True})
            self.assertTrue(reply.json()["enabled"])
            self.assertFalse((await client.post("/api/gestures/devices/puppy/disable")).json()["enabled"])


class WorkerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancel_during_process_reap_preserves_busy_until_exit(self):
        from unittest.mock import patch

        for already_exited in (False, True):
            with self.subTest(already_exited=already_exited), tempfile.TemporaryDirectory(prefix="gesture-reap-") as folder:
                model = Path(folder) / "model.task"
                model.write_bytes(b"fake")
                read_entered, reap_entered, release_reap = asyncio.Event(), asyncio.Event(), asyncio.Event()

                class Input:
                    def write(self, data):
                        pass
                    async def drain(self):
                        pass
                    def close(self):
                        pass

                class Output:
                    async def read(self, count):
                        read_entered.set()
                        await asyncio.Event().wait()

                class Process:
                    stdin, stdout = Input(), Output()
                    returncode = None
                    kills = 0

                    def kill(self):
                        self.kills += 1
                        if already_exited:
                            raise ProcessLookupError

                    async def wait(self):
                        reap_entered.set()
                        await release_reap.wait()
                        self.returncode = -9
                        return self.returncode

                process = Process()

                async def spawn(*args, **kwargs):
                    return process

                engine = SubprocessRecognizer(sys.executable, model, timeout=1)
                with patch("asyncio.create_subprocess_exec", spawn):
                    caller = asyncio.create_task(engine(b"fake"))
                    await read_entered.wait()
                    caller.cancel()
                    await reap_entered.wait()
                    caller.cancel()
                    await asyncio.sleep(0)
                    try:
                        self.assertTrue(engine.busy)
                        self.assertFalse(caller.done())
                        with self.assertRaises(RuntimeError):
                            await engine(b"fake")
                    finally:
                        release_reap.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await caller
                    self.assertFalse(engine.busy)
                    self.assertEqual(process.kills, 1)

    async def test_subprocess_output_is_bounded_and_timeout_worker_is_reaped(self):
        with tempfile.TemporaryDirectory(prefix="gesture-worker-") as folder:
            root = Path(folder)
            model = root / "model.task"
            model.write_bytes(b"fake")
            worker = root / "worker.py"
            worker.write_text("import sys\nsys.stdin.buffer.read()\nprint('x' * 20000)\n")
            engine = SubprocessRecognizer(sys.executable, model, timeout=1)
            engine.worker = str(worker)
            with self.assertRaises(ValueError):
                await engine(b"fake")
            self.assertFalse(engine.busy)
            pid = root / "pid"
            worker.write_text(f"import os,time,pathlib\npathlib.Path({str(pid)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n")
            engine.timeout = .2
            with self.assertRaises(TimeoutError):
                await engine(b"fake")
            self.assertFalse(engine.busy)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    async def test_split_json_result_and_concurrent_call_rejection(self):
        with tempfile.TemporaryDirectory(prefix="gesture-worker-") as folder:
            root = Path(folder)
            model = root / "model.task"
            model.write_bytes(b"fake")
            worker = root / "worker.py"
            worker.write_text("import sys,time\nsys.stdin.buffer.read()\nprint('{\"kind\":',end='',flush=True)\ntime.sleep(.05)\nprint('\"absent\"}',flush=True)\n")
            engine = SubprocessRecognizer(sys.executable, model, timeout=1)
            engine.worker = str(worker)
            first = asyncio.create_task(engine(b"fake"))
            await asyncio.sleep(.01)
            with self.assertRaises(RuntimeError):
                await engine(b"fake")
            self.assertEqual((await first).kind, "absent")

    async def test_low_or_nonfinite_confidence_cannot_be_executable_observation(self):
        for score in (.84, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                O("gesture", "Thumb_Up", score)
        with self.assertRaises(ValueError):
            O("gesture", "Victory", .99)


if __name__ == "__main__":
    unittest.main()
