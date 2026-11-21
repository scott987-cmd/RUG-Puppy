"""Temporary drafts and fake library/model only; no device or audio imports."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import uuid

from .action_drafts import (ActionDraftService, ApprovedActionConflict, DraftError,
                            normalize_approved_steps, normalize_draft, parse_model_draft)
from .turn_commit import TurnCancelled, TurnCommit


class ActionDraftTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="action-drafts-test-")
        self.database = Path(self.temp.name) / "drafts.sqlite3"
        self.raw = '{"name":"开心舞","steps":[{"action":"wave","wait_ms":300},{"action":"sit"}]}'
        self.prompts, self.writes = [], []
        self.approved = {}
        self.ownership = {}
        self.library_lock = threading.Lock()
        self.model_entered, self.model_release = asyncio.Event(), asyncio.Event()
        self.hold_model = self.ignore_cancel = False
        self.save_race = self.delete_race = self.save_then_fail = False
        self.service = self.make_service()

    async def asyncTearDown(self):
        self.model_release.set()
        self.temp.cleanup()

    def make_service(self):
        return ActionDraftService(self.database, local_model_generate=self.generate,
                                  approved_actions_snapshot=self.catalog,
                                  save_approved=self.save, delete_approved=self.remove)

    async def generate(self, prompt):
        self.prompts.append(prompt)
        self.model_entered.set()
        if self.hold_model:
            try:
                await self.model_release.wait()
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise
        return self.raw

    def catalog(self):
        with self.library_lock:
            return deepcopy(self.approved)

    def save(self, name, steps, *, expected, draft_id, ownership_nonce):
        self.assertIsNone(expected)
        with self.library_lock:
            if self.save_race:
                self.approved[name] = [{"action": "hug"}]
            if name in self.approved:
                raise ApprovedActionConflict()
            self.approved[name] = deepcopy(steps)
            receipt = {"draft_id": draft_id, "ownership_nonce": ownership_nonce}
            self.ownership[name] = deepcopy(receipt)
            self.writes.append(("save", name))
            if self.save_then_fail:
                raise RuntimeError("simulated post-commit failure")
            return receipt

    def remove(self, name, expected_steps, *, ownership_receipt):
        with self.library_lock:
            if self.delete_race:
                self.approved[name] = [{"action": "stretch"}]
                self.ownership.pop(name, None)
            if (name not in self.approved or normalize_approved_steps(self.approved[name]) != normalize_approved_steps(expected_steps)
                    or self.ownership.get(name) != ownership_receipt):
                raise ApprovedActionConflict()
            del self.approved[name]
            del self.ownership[name]
            self.writes.append(("delete", name))
            return deepcopy(ownership_receipt)

    async def propose(self):
        return (await self.service.propose("test-puppy", "想教你开心舞，挥手再坐下", "fake-session", "fake-child"))["draft"]

    async def test_child_proposal_is_only_draft_with_source_and_no_library_write(self):
        draft = await self.propose()
        self.assertEqual(draft["state"], "draft")
        self.assertEqual(draft["version"], 1)
        self.assertEqual((draft["device_id"], draft["session_id"], draft["person_id"]), ("test-puppy", "fake-session", "fake-child"))
        self.assertFalse(draft["executed"])
        self.assertEqual(self.approved, {})
        self.assertEqual(self.writes, [])
        restored = await self.make_service().get(draft["draft_id"])
        self.assertEqual(restored, draft)
        self.assertIn('"description": "想教你开心舞，挥手再坐下"', self.prompts[0])

    async def test_parent_edit_then_approve_saves_only_returned_snapshot(self):
        draft = await self.propose()
        edited = await self.service.edit(draft["draft_id"], 1, "温柔舞", [{"action": "hug"}, {"wait_ms": 300}])
        self.assertEqual(edited["version"], 2)
        self.assertEqual(self.approved, {})
        result = await self.service.approve(draft["draft_id"], 2)
        self.assertEqual(result["state"], "approved")
        self.assertEqual(result["version"], 4)
        self.assertEqual(self.approved, {"温柔舞": edited["steps"]})
        self.assertEqual(self.writes, [("save", "温柔舞")])
        self.assertEqual(result["ownership_receipt"], self.ownership["温柔舞"])
        self.assertEqual((await self.make_service().get(draft["draft_id"]))["ownership_receipt"], result["ownership_receipt"])
        self.assertTrue(result["can_revoke"])

    async def test_stale_edit_approve_delete_reject_all_fail_cas(self):
        draft = await self.propose()
        key = draft["draft_id"]
        await self.service.edit(key, 1, "新名字", [{"action": "wave"}])
        for call in (self.service.approve(key, 1), self.service.reject(key, 1), self.service.delete(key, 1),
                     self.service.edit(key, 1, "旧名字", [{"action": "sit"}])):
            with self.assertRaises(DraftError):
                await call
        self.assertEqual(self.writes, [])

    async def test_existing_name_is_not_overwritten_even_same_content(self):
        draft = await self.propose()
        self.approved["开心舞"] = deepcopy(draft["steps"])
        with self.assertRaises(DraftError):
            await self.service.approve(draft["draft_id"], 1)
        self.assertEqual(self.writes, [])
        self.assertEqual((await self.service.get(draft["draft_id"]))["version"], 1)

    async def test_atomic_save_callback_catches_name_created_after_catalog_check(self):
        draft = await self.propose()
        self.save_race = True
        result = await self.service.approve(draft["draft_id"], 1)
        self.assertEqual(result["state"], "draft")
        self.assertIn("冲突", result["error"])
        self.assertEqual(self.approved["开心舞"], [{"action": "hug"}])
        self.assertEqual(self.writes, [])

    async def test_reject_and_delete_do_not_touch_approved_library(self):
        draft = await self.propose()
        rejected = await self.service.reject(draft["draft_id"], 1)
        self.assertEqual(rejected["state"], "rejected")
        with self.assertRaises(DraftError):
            await self.service.approve(draft["draft_id"], rejected["version"])
        result = await self.service.delete(draft["draft_id"], rejected["version"])
        self.assertFalse(result["approved_library_changed"])
        self.assertEqual((await self.service.list("test-puppy"))["drafts"], [])
        self.assertEqual(self.writes, [])

    async def test_deleting_approved_draft_keeps_action_and_explicit_revoke_deletes_match(self):
        draft = await self.propose()
        approved = await self.service.approve(draft["draft_id"], 1)
        revoked = await self.service.revoke(draft["draft_id"], approved["version"])
        self.assertEqual(revoked["state"], "revoked")
        self.assertEqual(self.approved, {})
        second = await self.propose()
        approved = await self.service.approve(second["draft_id"], 1)
        await self.service.delete(second["draft_id"], approved["version"])
        self.assertIn("开心舞", self.approved)

    async def test_revoke_refuses_later_edited_action(self):
        draft = await self.propose()
        approved = await self.service.approve(draft["draft_id"], 1)
        self.approved["开心舞"] = [{"action": "stretch"}]
        with self.assertRaises(DraftError):
            await self.service.revoke(draft["draft_id"], approved["version"])
        self.assertEqual(self.writes, [("save", "开心舞")])

    async def test_revoke_matches_default_fields_and_persisted_ownership(self):
        draft = await self.propose()
        draft = await self.service.edit(draft["draft_id"], 1, "开心舞", [{"action": "wave"}, {"wait_ms": 100}])
        approved = await self.service.approve(draft["draft_id"], draft["version"])
        self.approved["开心舞"] = [{"action": "wave", "wait_ms": 0}, {"action": None, "wait_ms": 100}]
        revoked = await self.make_service().revoke(draft["draft_id"], approved["version"])
        self.assertEqual(revoked["state"], "revoked")
        self.assertFalse(revoked["can_revoke"])
        self.assertNotIn("开心舞", self.approved)

    async def test_revoke_cannot_delete_identical_action_recreated_by_other_owner(self):
        for new_owner in (None, {"draft_id": str(uuid.uuid4()), "ownership_nonce": str(uuid.uuid4())}):
            self.approved.clear()
            self.ownership.clear()
            draft = await self.propose()
            approved = await self.service.approve(draft["draft_id"], 1)
            # Ordinary save clears ownership; another draft gets a fresh owner.
            self.approved["开心舞"] = deepcopy(draft["steps"])
            self.ownership.pop("开心舞")
            if new_owner:
                self.ownership["开心舞"] = new_owner
            result = await self.service.revoke(draft["draft_id"], approved["version"])
            self.assertEqual(result["state"], "approved")
            self.assertIn("冲突", result["error"])
            self.assertEqual(self.approved["开心舞"], draft["steps"])
        self.assertFalse(any(write[0] == "delete" for write in self.writes))

    async def test_atomic_revoke_catches_same_content_aba_after_snapshot(self):
        draft = await self.propose()
        approved = await self.service.approve(draft["draft_id"], 1)
        def concurrent_replace(name, steps, *, ownership_receipt):
            self.ownership[name] = {"draft_id": draft["draft_id"], "ownership_nonce": str(uuid.uuid4())}
            return self.remove(name, steps, ownership_receipt=ownership_receipt)
        self.service.delete_approved = concurrent_replace
        result = await self.service.revoke(draft["draft_id"], approved["version"])
        self.assertEqual(result["state"], "approved")
        self.assertEqual(self.approved["开心舞"], draft["steps"])
        self.assertEqual(self.writes, [("save", "开心舞")])

    async def test_unknown_states_and_legacy_approved_cannot_revoke(self):
        draft = await self.propose()
        self.save_then_fail = True
        unknown = await self.service.approve(draft["draft_id"], 1)
        self.assertFalse(unknown["can_revoke"])
        self.assertIsNone(unknown["ownership_receipt"])
        # Even an unrelated, identical replacement must remain untouched.
        self.ownership.clear()
        for state in ("approval_unknown", "revoke_unknown", "approved"):
            with sqlite3.connect(self.database) as db:
                db.execute("UPDATE action_drafts SET state=?,ownership_receipt=NULL", (state,))
            with self.assertRaises(DraftError):
                await self.make_service().revoke(draft["draft_id"], unknown["version"])
            self.assertEqual(self.approved["开心舞"], draft["steps"])
        self.assertEqual(self.writes, [("save", "开心舞")])

    async def test_invalid_save_receipt_never_approves_or_allows_revoke(self):
        for receipt in (None, {}, {"draft_id": "wrong", "ownership_nonce": str(uuid.uuid4())}):
            self.approved.clear()
            draft = await self.propose()
            def invalid_receipt(name, steps, **kwargs):
                self.save(name, steps, **kwargs)
                return receipt
            self.service.save_approved = invalid_receipt
            result = await self.service.approve(draft["draft_id"], 1)
            self.assertEqual(result["state"], "approval_unknown")
            self.assertFalse(result["can_revoke"])
            self.assertIsNone(result["ownership_receipt"])
            with self.assertRaises(DraftError):
                await self.service.revoke(draft["draft_id"], result["version"])

    async def test_retried_conflicted_approval_has_fresh_nonce(self):
        draft = await self.propose()
        nonces = []
        def record_attempt(name, steps, **kwargs):
            nonces.append(kwargs["ownership_nonce"])
            if len(nonces) == 1:
                raise ApprovedActionConflict()
            return self.save(name, steps, **kwargs)
        self.service.save_approved = record_attempt
        conflict = await self.service.approve(draft["draft_id"], 1)
        result = await self.service.approve(draft["draft_id"], conflict["version"])
        self.assertNotEqual(nonces[0], nonces[1])
        self.assertEqual(result["ownership_receipt"]["ownership_nonce"], nonces[1])

    async def test_old_database_migration_preserves_records_without_inventing_ownership(self):
        with sqlite3.connect(self.database) as db:
            db.execute("""CREATE TABLE action_drafts (
                draft_id TEXT PRIMARY KEY, version INTEGER NOT NULL, state TEXT NOT NULL,
                device_id TEXT NOT NULL, session_id TEXT, person_id TEXT, source_text TEXT NOT NULL,
                name TEXT NOT NULL, steps TEXT NOT NULL, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, error TEXT)""")
            db.execute("INSERT INTO action_drafts VALUES (?,3,'approved','test-puppy',NULL,NULL,'fake','旧舞',?,1,1,NULL)",
                       (str(uuid.UUID(int=1)), '[{"action":"sit"}]'))
        old = await self.service.get(str(uuid.UUID(int=1)))
        self.assertEqual(old["state"], "approved")
        self.assertFalse(old["can_revoke"])
        self.assertIsNone(old["ownership_receipt"])
        with self.assertRaises(DraftError):
            await self.service.revoke(old["draft_id"], 3)
        # Explicit column insertion remains compatible after ALTER TABLE.
        self.assertEqual((await self.propose())["state"], "draft")

    async def test_failed_revoke_receipt_is_unknown_and_cannot_be_replayed(self):
        draft = await self.propose()
        approved = await self.service.approve(draft["draft_id"], 1)
        def lost_receipt(name, steps, **kwargs):
            self.remove(name, steps, **kwargs)
            return None
        self.service.delete_approved = lost_receipt
        result = await self.service.revoke(draft["draft_id"], approved["version"])
        self.assertEqual(result["state"], "revoke_unknown")
        self.assertFalse(result["can_revoke"])
        self.approved["开心舞"] = deepcopy(draft["steps"])
        with self.assertRaises(DraftError):
            await self.make_service().revoke(draft["draft_id"], result["version"])
        self.assertIn("开心舞", self.approved)

    async def test_atomic_delete_conflict_cannot_remove_concurrent_edit(self):
        draft = await self.propose()
        approved = await self.service.approve(draft["draft_id"], 1)
        self.delete_race = True
        result = await self.service.revoke(draft["draft_id"], approved["version"])
        self.assertEqual(result["state"], "approved")
        self.assertEqual(self.approved["开心舞"], [{"action": "stretch"}])

    async def test_external_write_failure_is_unknown_never_replayed_after_restart(self):
        draft = await self.propose()
        self.save_then_fail = True
        result = await self.service.approve(draft["draft_id"], 1)
        self.assertEqual(result["state"], "approval_unknown")
        self.assertIn("开心舞", self.approved)
        restored = await self.make_service().get(draft["draft_id"])
        self.assertEqual(restored["state"], "approval_unknown")
        self.assertEqual(self.writes, [("save", "开心舞")])
        with self.assertRaises(DraftError):
            await self.service.approve(draft["draft_id"], result["version"])

    async def test_interrupted_approval_shown_unknown_without_auto_save(self):
        draft = await self.propose()
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE action_drafts SET state='approving', version=2")
        restored = await self.make_service().get(draft["draft_id"])
        self.assertEqual(restored["state"], "approval_unknown")
        self.assertEqual(self.writes, [])

    async def test_cancel_during_model_does_not_store_even_if_model_swallows_cancel(self):
        for swallowed in (False, True):
            self.hold_model, self.ignore_cancel = True, swallowed
            self.model_entered.clear()
            task = asyncio.create_task(self.propose())
            await self.model_entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual((await self.service.list("test-puppy"))["drafts"], [])
        self.assertEqual(self.writes, [])

    async def test_cancel_during_precommit_rolls_back_and_repeated_cancel_waits(self):
        entered, release = threading.Event(), threading.Event()
        class BlockedCommit(TurnCommit):
            def commit(self, operation):
                entered.set()
                release.wait(2)
                return super().commit(operation)
        task = asyncio.create_task(self.service.propose("test-puppy", "新把戏", commit_guard=BlockedCommit()))
        async with asyncio.timeout(.5):
            while not entered.is_set():
                await asyncio.sleep(.002)
        try:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(.005)
            self.assertFalse(task.done())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual((await self.service.list("test-puppy"))["drafts"], [])

    async def test_external_turn_guard_cancels_model_result_before_draft_commit(self):
        self.hold_model = True
        guard = TurnCommit()
        task = asyncio.create_task(self.service.propose("test-puppy", "开心舞", commit_guard=guard))
        await self.model_entered.wait()
        guard.cancel()
        self.model_release.set()
        with self.assertRaises(TurnCancelled):
            await task
        self.assertEqual((await self.service.list("test-puppy"))["drafts"], [])

    async def test_repeated_cancel_of_parent_approval_waits_for_library_commit(self):
        entered, release = threading.Event(), threading.Event()
        def slow_save(name, steps, **kwargs):
            entered.set()
            release.wait(2)
            return self.save(name, steps, **kwargs)
        self.service.save_approved = slow_save
        draft = await self.propose()
        approving = asyncio.create_task(self.service.approve(draft["draft_id"], 1))
        async with asyncio.timeout(.5):
            while not entered.is_set():
                await asyncio.sleep(.002)
        deleting = asyncio.create_task(self.service.delete(draft["draft_id"], 3))
        try:
            for _ in range(3):
                approving.cancel()
                await asyncio.sleep(.005)
            self.assertFalse(approving.done())
            self.assertFalse(deleting.done())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await approving
            self.assertTrue((await deleting)["deleted"])
        self.assertEqual(self.writes, [("save", "开心舞")])
        self.assertIn("开心舞", self.approved)  # Deleting source never revokes approval.

    async def test_invalid_model_output_does_not_create_any_draft(self):
        for raw in ('{"name":"bad","steps":[{"action":"laser"}]}', '```json\n{}\n```',
                    '{"name":"a","name":"b","steps":[{"action":"wave"}]}',
                    '{"name":"bad","steps":[{"action":"wave","angle":90}]}'):
            self.raw = raw
            with self.assertRaises(DraftError):
                await self.propose()
        self.assertEqual((await self.service.list("test-puppy"))["drafts"], [])
        self.assertEqual(self.writes, [])

    async def test_source_text_is_bounded_before_model_call(self):
        for text in (" " * 301, "舞" + " " * 300, "\ud800"):
            with self.assertRaises(DraftError):
                await self.service.propose("test-puppy", text)
        self.assertEqual(self.prompts, [])

    async def test_cross_instance_cas_only_one_edit_wins(self):
        draft = await self.propose()
        other = self.make_service()
        results = await asyncio.gather(self.service.edit(draft["draft_id"], 1, "新A", [{"action": "wave"}]),
                                       other.edit(draft["draft_id"], 1, "新B", [{"action": "sit"}]), return_exceptions=True)
        self.assertEqual(sum(isinstance(result, DraftError) for result in results), 1)
        self.assertEqual((await self.service.get(draft["draft_id"]))["version"], 2)

    async def test_parent_api_auth_strict_fields_and_cas(self):
        import httpx
        from fastapi import FastAPI, Header, HTTPException
        from .action_drafts_api import create_action_drafts_router
        async def parent(authorization: str | None = Header(default=None)):
            if authorization != "Bearer test":
                raise HTTPException(401)
        app = FastAPI()
        app.include_router(create_action_drafts_router(self.service, parent))
        draft = await self.propose()
        path = "/api/action-drafts/" + draft["draft_id"]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            self.assertEqual((await client.get(path)).status_code, 401)
            self.assertEqual((await client.post(path + "/approve", json={"version": 1})).status_code, 401)
            client.headers["Authorization"] = "Bearer test"
            for body in ({"version": True}, {"version": 1, "execute": True}):
                self.assertEqual((await client.post(path + "/approve", json=body)).status_code, 422)
            response = await client.patch(path, json={"version": 1, "name": "新舞", "steps": [{"action": "sit"}]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual((await client.post(path + "/approve", json={"version": 1})).status_code, 409)
            self.assertEqual((await client.post(path + "/approve", json={"version": 2})).status_code, 200)
        self.assertEqual(self.writes, [("save", "新舞")])


class SchemaTests(unittest.TestCase):
    def test_strict_json_parameters_name_and_six_step_bounds(self):
        for value in ({"name": "dance", "steps": []}, {"name": "wave", "steps": [{"action": "wave"}]},
                      {"name": "狗" * 22, "steps": [{"action": "wave"}]},
                      {"name": "dance", "steps": [{"action": "wave"}] * 7},
                      {"name": "dance", "steps": [{"action": None, "wait_ms": 100}]},
                      {"name": "dance", "steps": [{"wait_ms": 0}]},
                      {"name": "dance", "steps": [{"action": "hug", "wait_ms": True}]},
                      {"name": "dance", "steps": [{"action": "hug", "wait_ms": 3001}]},
                      {"name": "dance", "steps": [{"action": "hug", "loop": 3}]}):
            with self.assertRaises(DraftError):
                normalize_draft(value)
        for raw in ('{"name":"dance","steps":[{"wait_ms":NaN}]}', '\ud800', ' ' * 8193):
            with self.assertRaises(DraftError):
                parse_model_draft(raw)
        self.assertEqual(normalize_draft({"name": "舞", "steps": [{"wait_ms": 3000}] * 6})["steps"], [{"wait_ms": 3000}] * 6)


if __name__ == "__main__":
    unittest.main()
