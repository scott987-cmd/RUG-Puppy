"""Fake-clock budget tests. No model, speech, camera, synthesis or playback is used."""
import concurrent.futures
import importlib.util
import sqlite3
import stat
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from companion.learning import LearningStore


class LearningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "private" / "learning.sqlite3"
        self.now = datetime(2026, 9, 24, 10, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        self.store = self.reopen()

    def reopen(self):
        return LearningStore(self.path, clock=lambda: self.now)

    def tearDown(self):
        self.temp.cleanup()

    def begin(self, person="child-1", session="session-a", **kwargs):
        return self.store.before_turn("puppy", person, session_id=session, **kwargs)

    def finish(self, turn, person="child-1", seconds=1, **kwargs):
        self.now += seconds
        return self.store.after_turn("puppy", person, turn_id=turn["turn_id"], **kwargs)

    def test_defaults_member_override_and_unknown_isolation(self):
        default = self.store.get_settings("puppy")
        self.assertEqual(default["daily_minutes"], 5)
        self.assertEqual(default["theme"], "animals")
        self.store.set_settings("puppy", "child-1", patch={"theme": "objects", "daily_minutes": 8})
        self.store.set_settings("puppy", patch={"theme": "colors", "daily_minutes": 4})
        self.assertEqual(self.store.get_settings("puppy", "child-1")["theme"], "objects")
        self.assertEqual(self.store.get_settings("puppy", "child-2")["theme"], "colors")
        self.assertEqual(self.store.get_settings("puppy")["daily_minutes"], 4)
        self.finish(self.begin(), seconds=10)
        self.assertEqual(self.store.status("puppy")["used_seconds"], 0)
        self.assertEqual(self.store.status("other-puppy", "child-1")["used_seconds"], 0)
        self.assertEqual(self.store.status("puppy", "child-2")["used_seconds"], 0)

    def test_settings_and_private_permissions_survive_restart(self):
        self.store.set_settings("puppy", "child-1", patch={"daily_minutes": 7, "save_progress": False, "photo_learning": True})
        saved = self.reopen().get_settings("puppy", "child-1")
        self.assertEqual(saved["daily_minutes"], 7)
        self.assertFalse(saved["save_progress"])
        self.assertTrue(saved["photo_learning"])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)

    def test_settings_validation_is_strict(self):
        for patch in ({}, {"daily_minutes": 0}, {"daily_minutes": 21}, {"daily_minutes": True},
                      {"daily_minutes": 2.5}, {"theme": "anything"}, {"save_progress": "false"},
                      {"photo_learning": 1}, {"other": 1}):
            with self.assertRaises(ValueError):
                self.store.set_settings("puppy", patch=patch)

    def test_short_session_counts_elapsed_once_and_stop_is_immediate(self):
        turn = self.begin()
        finished = self.finish(turn, seconds=2)
        self.assertTrue(finished["allowed"])
        self.now += 10
        self.assertEqual(self.store.status("puppy", "child-1")["used_seconds"], 12)
        self.assertEqual(self.store.status("puppy", "child-1")["used_seconds"], 12)
        stopped = self.store.end_session("puppy", "child-1")
        self.assertFalse(stopped["active"])
        self.now += 100
        self.assertEqual(self.reopen().status("puppy", "child-1")["used_seconds"], 12)

    def test_inactivity_deadline_is_persisted_across_restart(self):
        turn = self.begin(); self.finish(turn, seconds=2)
        self.now += 8
        self.assertEqual(self.reopen().status("puppy", "child-1")["used_seconds"], 10)
        self.now += 90
        state = self.reopen().status("puppy", "child-1")
        self.assertEqual(state["used_seconds"], 47)  # 2-second turn + 45-second activity window.
        self.assertFalse(state["active"])

    def test_daily_limit_cannot_be_reset_by_new_session_or_process(self):
        self.store.set_settings("puppy", "child-1", patch={"daily_minutes": 1})
        turn = self.begin(); self.finish(turn)
        self.now += 39
        turn = self.begin(session="new-session")
        self.finish(turn, seconds=5)
        self.now += 30
        self.store = self.reopen()
        stop = self.begin(session="another-new-session")
        self.assertFalse(stop["allowed"])
        self.assertEqual(stop["reason"], "daily_limit")
        self.assertEqual(stop["used_seconds"], 60)
        self.assertEqual(stop["stop"]["kind"], "finish_learning")
        self.assertNotIn("reply", stop)

    def test_concurrent_callers_cannot_start_multiple_turns(self):
        def attempt(index):
            store = self.reopen()
            return store.before_turn("puppy", "child-1", session_id=f"parallel-{index}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            decisions = list(pool.map(attempt, range(8)))
        self.assertEqual(sum(item["allowed"] for item in decisions), 1)
        self.assertTrue(all(item["allowed"] or item["reason"] == "turn_in_progress" for item in decisions))

    def test_late_result_cannot_renew_or_end_newer_turn(self):
        old = self.begin()
        self.now += 31
        current = self.begin(session="new-session")
        self.assertTrue(current["allowed"])
        late = self.store.after_turn("puppy", "child-1", turn_id=old["turn_id"])
        self.assertEqual(late["reason"], "stale_turn")
        self.assertTrue(late["busy"])
        completed = self.finish(current)
        self.assertTrue(completed["allowed"])
        self.assertEqual(completed["used_seconds"], 31)

    def test_failed_turn_and_end_session_stop_further_charging(self):
        failed = self.finish(self.begin(), seconds=3, success=False)
        self.assertFalse(failed["active"])
        self.now += 50
        self.assertEqual(self.store.status("puppy", "child-1")["used_seconds"], 3)
        turn = self.begin(); self.now += 2
        self.store.end_session("puppy", "child-1")
        self.now += 10
        late = self.store.after_turn("puppy", "child-1", turn_id=turn["turn_id"])
        self.assertEqual(late["reason"], "stale_turn")
        self.assertEqual(late["used_seconds"], 5)

    def test_midnight_caps_old_session_and_preserves_both_day_ledgers(self):
        self.now = datetime(2026, 9, 24, 23, 59, 50, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        old = self.begin(); self.now += 15
        late = self.store.after_turn("puppy", "child-1", turn_id=old["turn_id"])
        self.assertEqual(late["reason"], "stale_turn")
        self.assertEqual(late["local_day"], "2026-09-25")
        self.assertEqual(late["used_seconds"], 0)
        self.finish(self.begin(), seconds=5)
        with self.store.connect() as db:
            rows = db.execute("SELECT local_day, used_seconds FROM learning_usage ORDER BY local_day").fetchall()
        self.assertEqual([(row["local_day"], row["used_seconds"]) for row in rows], [("2026-09-24", 10), ("2026-09-25", 5)])
        self.assertEqual(self.reopen().status("puppy", "child-1")["used_seconds"], 5)

    def test_clock_rollback_cannot_reopen_previous_budget_or_give_free_turns(self):
        self.finish(self.begin(), seconds=10)
        self.store.end_session("puppy", "child-1")
        self.now -= 100
        self.store = self.reopen()
        blocked = self.begin()
        self.assertEqual(blocked["reason"], "clock_changed")
        self.assertFalse(blocked["allowed"])
        self.assertEqual(blocked["used_seconds"], 10)
        self.now += 100
        self.assertTrue(self.begin()["allowed"])

    def test_progress_switch_does_not_disable_daily_accounting(self):
        self.store.set_settings("puppy", "child-1", patch={"save_progress": False})
        self.finish(self.begin(), seconds=5)
        state = self.store.status("puppy", "child-1")
        self.assertFalse(state["settings"]["save_progress"])
        self.assertEqual(state["used_seconds"], 5)

    def test_photo_switch_is_checked_before_starting_a_turn(self):
        denied = self.begin(use_photo=True)
        self.assertEqual(denied["reason"], "photo_learning_disabled")
        self.assertFalse(denied["active"])
        self.store.set_settings("puppy", "child-1", patch={"photo_learning": True})
        self.assertTrue(self.begin(use_photo=True)["allowed"])

    def test_changing_limit_preserves_used_time(self):
        for _ in range(3):
            turn = self.begin(); self.finish(turn, seconds=20)
            self.now += 10
        self.store.end_session("puppy", "child-1")
        lowered = self.store.set_settings("puppy", "child-1", patch={"daily_minutes": 1})
        self.assertEqual(lowered["used_seconds"], 90)
        self.assertEqual(lowered["remaining_seconds"], 0)
        raised = self.store.set_settings("puppy", "child-1", patch={"daily_minutes": 2})
        self.assertEqual(raised["used_seconds"], 90)
        self.assertEqual(raised["remaining_seconds"], 30)

    def test_fractional_turns_cannot_overrun_budget(self):
        self.store.set_settings("puppy", "child-1", patch={"daily_minutes": 1})
        for _ in range(100):
            decision = self.begin()
            if not decision["allowed"]:
                break
            self.finish(decision, seconds=.25)
            self.now += .5
        status = self.store.status("puppy", "child-1")
        self.assertAlmostEqual(status["used_seconds"], 60)
        self.assertEqual(status["remaining_seconds"], 0)

    def test_timezone_change_cannot_rebucket_existing_ledger(self):
        self.finish(self.begin(), seconds=5)
        with self.assertRaises(ValueError):
            LearningStore(self.path, timezone="UTC", clock=lambda: self.now)
        self.assertEqual(self.reopen().status("puppy", "child-1")["used_seconds"], 5)


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "optional API test dependencies missing")
class LearningAPITest(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.testclient import TestClient
        from companion.learning_api import create_learning_router
        self.temp = tempfile.TemporaryDirectory()
        self.now = datetime(2026, 9, 24, 10, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        self.store = LearningStore(Path(self.temp.name) / "learning.sqlite3", clock=lambda: self.now)

        def parent(authorization: str = Header(default="")):
            if authorization != "Bearer fake-parent":
                raise HTTPException(401, "parent required")
        app = FastAPI(); app.include_router(create_learning_router(self.store, parent))
        self.client = TestClient(app)
        self.auth = {"Authorization": "Bearer fake-parent"}

    def tearDown(self):
        self.client.close(); self.temp.cleanup()

    def test_every_endpoint_requires_parent(self):
        for path in ("/api/learning/puppy", "/api/learning/puppy/members/child-1"):
            for method in ("GET", "PATCH"):
                self.assertEqual(self.client.request(method, path).status_code, 401)
            self.assertEqual(self.client.post(path + "/stop").status_code, 401)

    def test_parent_patch_is_strict_and_member_isolated(self):
        path = "/api/learning/puppy/members/child-1"
        response = self.client.patch(path, json={"daily_minutes": 3, "theme": "greetings", "save_progress": False}, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["settings"]["theme"], "greetings")
        defaults = self.client.get("/api/learning/puppy", headers=self.auth).json()
        self.assertEqual(defaults["settings"]["daily_minutes"], 5)
        for value in ({"daily_minutes": True}, {"daily_minutes": 21}, {"save_progress": "false"},
                      {"theme": None}, {"extra": 1}, {}):
            result = self.client.patch(path, json=value, headers=self.auth)
            self.assertEqual(result.status_code, 422, (value, result.text))

    def test_parent_stop_preserves_usage_and_invokes_no_model(self):
        turn = self.store.before_turn("puppy", "child-1", session_id="test")
        self.now += 3
        self.store.after_turn("puppy", "child-1", turn_id=turn["turn_id"])
        self.now += 2
        result = self.client.post("/api/learning/puppy/members/child-1/stop", headers=self.auth)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["used_seconds"], 5)
        self.assertFalse(result.json()["active"])


if __name__ == "__main__":
    unittest.main()
