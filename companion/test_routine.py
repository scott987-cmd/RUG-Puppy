"""Text-only profile/routine behavior, CAS, cancellation and parent authorization."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from .agent import PetAgent
from .pet_profile import PetProfile, RevisionConflict
from .routine import RoutineService, explicit_request
from .store import Store
from .turn_commit import TurnCancelled, TurnCommit


class RoutineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-routine-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.store = Store(self.path)
        self.routines = RoutineService(self.store)
        self.profile = PetProfile(self.store)
        self.initial = self.routines.put("puppy", "child", "toys", title="收玩具",
            steps=["把积木放进盒子", "把小汽车停进车库"], expected_revision=0)

    def transition(self, value, operation):
        return self.routines.transition("puppy", "child", value["routine_id"], operation=operation,
            expected_revision=value["revision"], expected_step_index=value["step_index"])

    @staticmethod
    def proposal(current, operation, quote):
        return {"reply": "我们一起把积木放进盒子。", "intent": "routine", "routine": {
            "routine_id": current["routine_id"], "operation": operation, "expected_revision": current["revision"],
            "expected_step_index": current["step_index"], "user_quote": quote}}

    @staticmethod
    def model(answer, *, explicit=True, prompts=None):
        def call(system, user):
            if prompts is not None:
                prompts.append((system, user))
            return json.dumps({"explicit": explicit} if system.startswith("你只核验") else answer, ensure_ascii=False)
        return call

    def test_profile_defaults_cas_and_restart(self):
        self.assertEqual(self.profile.get("puppy")["greeting_action"], "none")
        saved = self.profile.set("puppy", style="patient", greeting_action="wave", expected_revision=0)
        self.assertEqual(saved["revision"], 1)
        self.assertEqual(PetProfile(Store(self.path)).get("puppy"), saved)
        self.assertEqual(self.profile.get("other")["revision"], 0)
        with self.assertRaises(RevisionConflict):
            self.profile.set("puppy", style="lively", greeting_action="hug", expected_revision=0)
        for fields in ({"style": "angry", "greeting_action": "wave", "expected_revision": 1},
                       {"style": "gentle", "greeting_action": "walk", "expected_revision": 1},
                       {"style": "gentle", "greeting_action": "none", "expected_revision": True}):
            with self.assertRaises(ValueError):
                self.profile.set("puppy", **fields)
        self.assertEqual(self.routines.get("puppy", "child", "toys"), self.initial)

    def test_lifecycle_persists_and_changes_one_step_at_a_time(self):
        value = self.transition(self.initial, "start")
        self.assertEqual((value["state"], value["step_index"]), ("active", 0))
        value = self.transition(value, "advance")
        self.assertEqual(value["current_step"], "把小汽车停进车库")
        value = self.transition(value, "pause")
        restored = RoutineService(Store(self.path)).get("puppy", "child", "toys")
        self.assertEqual(restored, value)
        with self.assertRaises(RevisionConflict):
            self.transition(value, "advance")
        value = self.transition(value, "resume")
        value = self.transition(value, "advance")
        self.assertEqual((value["state"], value["step_index"], value["current_step"]), ("completed", 2, None))
        with self.assertRaises(RevisionConflict):
            self.transition(value, "start")
        value = self.transition(value, "reset")
        self.assertEqual((value["state"], value["step_index"]), ("idle", 0))

    def test_edit_resets_progress_and_invalidates_old_version(self):
        active = self.transition(self.initial, "start")
        edited = self.routines.put("puppy", "child", "toys", title="一起收好", steps=["只放好积木"], expected_revision=active["revision"])
        self.assertEqual((edited["state"], edited["step_index"]), ("idle", 0))
        with self.assertRaises(RevisionConflict):
            self.transition(active, "advance")

    def test_parallel_advance_is_exactly_once(self):
        active = self.transition(self.initial, "start")
        def advance(_):
            try:
                return self.transition(active, "advance")
            except RevisionConflict:
                return None
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(advance, range(6)))
        self.assertEqual(sum(value is not None for value in results), 1)
        self.assertEqual(self.routines.get("puppy", "child", "toys")["step_index"], 1)

    def test_only_one_active_routine_and_member_device_isolation(self):
        second = self.routines.put("puppy", "child", "teeth", title="刷牙", steps=["请家长准备牙刷"], expected_revision=0)
        active = self.transition(self.initial, "start")
        with self.assertRaises(RevisionConflict):
            self.transition(second, "start")
        self.transition(active, "pause")
        self.assertEqual(self.transition(second, "start")["state"], "active")
        self.assertEqual(self.routines.list("other", "child"), [])
        self.assertEqual(self.routines.list("puppy", "another-child"), [])
        with patch.object(self.store, "connect", side_effect=AssertionError("unknown must not query")):
            self.assertEqual(self.routines.list("puppy", None), [])

    def test_delete_recreate_and_erasing_member_cannot_reuse_revision(self):
        old = self.initial
        self.routines.delete("puppy", "child", "toys", expected_revision=old["revision"])
        recreated = self.routines.put("puppy", "child", "toys", title="新的任务", steps=["坐在旁边等家长"], expected_revision=0)
        self.assertGreater(recreated["revision"], old["revision"])
        with self.assertRaises(RevisionConflict):
            self.transition(old, "start")
        self.store.forget_person("puppy", "child")
        self.assertEqual(self.routines.list("puppy", "child"), [])
        newest = self.routines.put("puppy", "child", "toys", title="新的任务", steps=["坐在旁边等家长"], expected_revision=0)
        self.assertGreater(newest["revision"], recreated["revision"])

    def test_bounded_parent_written_steps_and_strict_versions(self):
        for steps in ([], ["一步"] * 9, [""], ["x" * 81], [5], ["换\n行"]):
            with self.assertRaises(ValueError):
                self.routines.put("puppy", "child", "bad", title="测试", steps=steps, expected_revision=0)
        for index in range(7):
            self.routines.put("puppy", "child", f"r{index}", title="任务", steps=["请家长一起"], expected_revision=0)
        with self.assertRaises(ValueError):
            self.routines.put("puppy", "child", "ninth", title="任务", steps=["请家长一起"], expected_revision=0)
        for version, index in ((True, 0), (1.0, 0), (self.initial["revision"], False), (self.initial["revision"], 9)):
            with self.assertRaises(ValueError):
                self.routines.transition("puppy", "child", "toys", operation="start", expected_revision=version, expected_step_index=index)

    def test_agent_explicit_routine_request_commits_then_restart_retains_it(self):
        prompts = []
        answer = self.proposal(self.initial, "start", "开始收玩具")
        result = PetAgent(self.store, self.model(answer, prompts=prompts)).reply("puppy", "开始收玩具", "child")
        self.assertEqual(result["routine"]["state"], "active")
        self.assertEqual(result["reply"], answer["reply"])
        self.assertEqual(RoutineService(Store(self.path)).get("puppy", "child", "toys"), result["routine"])
        self.assertEqual(len(prompts), 2)
        self.assertTrue(prompts[1][0].startswith("你只核验"))

    def test_current_request_semantic_rejection_does_not_advance(self):
        active = self.transition(self.initial, "start")
        answer = self.proposal(active, "advance", "这一步是什么意思")
        calls = []
        def model(system, user):
            calls.append(system)
            if system.startswith("你只核验"):
                return '{"explicit":false}'
            if len(calls) == 1:
                return json.dumps(answer, ensure_ascii=False)
            return '{"reply":"就是把散落的积木放进盒子里。","intent":"chat"}'
        result = PetAgent(self.store, model).reply("puppy", "这一步是什么意思", "child")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(self.routines.get("puppy", "child", "toys"), active)

    def test_negation_quotation_and_future_still_need_independent_semantic_approval(self):
        for text in ("不要开始收玩具", "妈妈说：开始收玩具", "明天再开始收玩具"):
            with self.subTest(text=text):
                main_calls, checks = [], []
                proposal = self.proposal(self.initial, "start", text)
                def model(system, user):
                    if system.startswith("你只核验"):
                        checks.append(json.loads(user))
                        return '{"explicit":false}'
                    main_calls.append(user)
                    return json.dumps(proposal if len(main_calls) == 1 else
                        {"reply": "我们先聊一聊。", "intent": "chat"}, ensure_ascii=False)
                answer = PetAgent(self.store, model).reply("puppy", text, "child")
                self.assertEqual(answer["intent"], "chat")
                self.assertEqual(self.routines.get("puppy", "child", "toys"), self.initial)
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0]["current_utterance"], text)
                self.assertEqual(checks[0]["operation"], "routine/start")

    def test_unknown_operation_cannot_reach_model_or_get_semantic_approval(self):
        def forbidden(*args):
            raise AssertionError("unknown operation must fail closed")
        self.assertFalse(explicit_request(forbidden, "开始", "routine/unknown", context={}))

    def test_malformed_stale_or_historical_proposal_never_changes_progress(self):
        valid = self.proposal(self.initial, "start", "开始收玩具")
        for change in ({"expected_revision": 999}, {"expected_step_index": True}, {"expected_step_index": 1},
                       {"state": "active"}, {"operation": "skip_all"}, {"user_quote": "上一轮说的话"},
                       {"routine_id": "someone-elses"}):
            with self.subTest(change):
                answer = {**valid, "routine": {**valid["routine"], **change}}
                with self.assertRaises(ValueError):
                    PetAgent(self.store, self.model(answer)).reply("puppy", "开始收玩具", "child")
                self.assertEqual(self.routines.get("puppy", "child", "toys"), self.initial)

    def test_unknown_agent_prompt_has_no_personal_routine_or_write(self):
        prompts = []
        answer = self.proposal(self.initial, "start", "开始收玩具")
        result = PetAgent(self.store, self.model(answer, prompts=prompts)).reply("puppy", "开始收玩具")
        self.assertEqual(result["business_status"], {"code": "personal_access_unavailable"})
        self.assertEqual(result["reply_source"], "business")
        self.assertEqual((result["intent"], result["action"]), ("chat", ""))
        self.assertNotIn("routine", result)
        for system, user in prompts:
            self.assertNotIn("把积木放进盒子", system)
            self.assertNotIn("把小汽车停进车库", system)
        self.assertEqual(self.routines.get("puppy", "child", "toys"), self.initial)

    def test_parent_edit_during_inference_rejects_stale_agent_and_history(self):
        answer = self.proposal(self.initial, "start", "开始收玩具")
        def model(system, user):
            if system.startswith("你只核验"):
                return '{"explicit":true}'
            self.routines.put("puppy", "child", "toys", title="改过的流程", steps=["先找家长"], expected_revision=self.initial["revision"])
            return json.dumps(answer, ensure_ascii=False)
        with self.assertRaises(RevisionConflict):
            PetAgent(self.store, model).reply("puppy", "开始收玩具", "child")
        self.assertEqual(self.routines.get("puppy", "child", "toys")["state"], "idle")
        self.assertEqual(self.store.conversation("puppy", "child"), [])

    def test_generation_cancellation_and_outer_scope_rollback_include_routine(self):
        answer = self.proposal(self.initial, "start", "开始收玩具")
        entered, release, guard = threading.Event(), threading.Event(), TurnCommit()
        def model(system, user):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test model timeout")
            return json.dumps(answer, ensure_ascii=False)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(PetAgent(self.store, model).reply, "puppy", "开始收玩具", "child", turn_commit=guard)
            try:
                self.assertTrue(entered.wait(2)); guard.cancel()
            finally:
                release.set()
            with self.assertRaises(TurnCancelled):
                future.result(timeout=3)
        guard = TurnCommit()
        with self.assertRaises(TurnCancelled):
            with self.store.commit_scope(guard):
                PetAgent(self.store, self.model(answer)).reply("puppy", "开始收玩具", "child", turn_commit=guard)
                guard.cancel()
        self.assertEqual(self.routines.get("puppy", "child", "toys"), self.initial)
        self.assertEqual(self.store.conversation("puppy", "child"), [])

    def test_style_and_explicit_greeting_only_return_saved_safe_proposal(self):
        saved = self.profile.set("puppy", style="patient", greeting_action="wave", expected_revision=0)
        prompts = []
        result = PetAgent(self.store, self.model({"reply": "很高兴和你打招呼。", "intent": "greet"}, prompts=prompts)).reply(
            "puppy", "请按约定打个招呼")
        self.assertEqual((result["intent"], result["action"], result["profile_revision"]), ("action", "wave", saved["revision"]))
        self.assertTrue(result["greeting"])
        self.assertIn("性格风格为耐心", prompts[0][0])
        quiet = PetAgent(self.store, self.model({"reply": "你好呀。", "intent": "chat"})).reply("puppy", "你好")
        self.assertEqual(quiet.get("action"), "")
        self.profile.set("puppy", style="gentle", greeting_action="none", expected_revision=saved["revision"])
        still = PetAgent(self.store, self.model({"reply": "你好呀。", "intent": "greet"})).reply("puppy", "按约定打招呼")
        self.assertEqual((still["intent"], still["action"]), ("chat", ""))

    def test_chat_never_returns_stray_single_or_sequence_action_fields(self):
        result = PetAgent(self.store, self.model({"reply": "你好呀。", "intent": "chat",
                         "action": "wave", "actions": ["wave", "sit"]})).reply("puppy", "你好")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(result["action"], "")
        self.assertNotIn("actions", result)
        self.assertEqual(self.store.actions(), {})

    def test_greeting_proposal_carries_old_revision_if_parent_changes_during_model(self):
        saved = self.profile.set("puppy", style="gentle", greeting_action="wave", expected_revision=0)
        def model(system, user):
            if system.startswith("你只核验"):
                return '{"explicit":true}'
            self.profile.set("puppy", style="gentle", greeting_action="none", expected_revision=saved["revision"])
            return '{"reply":"我来打招呼。","intent":"greet"}'
        result = PetAgent(self.store, model).reply("puppy", "按约定打招呼")
        self.assertEqual(result["profile_revision"], saved["revision"])
        self.assertNotEqual(result["profile_revision"], self.profile.get("puppy")["revision"])

    def test_new_named_action_is_a_draft_without_steps_or_execution(self):
        text = "教你一个叫星星舞的新把戏，以后记住它"
        result = PetAgent(self.store, self.model({"reply": "我可以请家长看看这个建议。", "intent": "action_draft"})).reply("puppy", text)
        self.assertEqual(set(result), {"reply", "intent", "draft_request", "reply_source"})
        self.assertEqual(result["reply_source"], "model")
        self.assertEqual(result["draft_request"], text)
        self.assertEqual(self.store.actions(), {})

    def test_draft_verifier_can_reject_a_false_claim_of_approval(self):
        calls = []
        def model(system, user):
            if system.startswith("你只核验"):
                data = json.loads(user)
                self.assertEqual(data["context"]["proposed_reply"], "已经批准并学会星星舞。")
                return '{"explicit":false}'
            calls.append(user)
            return ('{"reply":"已经批准并学会星星舞。","intent":"action_draft"}' if len(calls) == 1 else
                    '{"reply":"可以请家长一起设计这个新把戏。","intent":"chat"}')
        answer = PetAgent(self.store, model).reply("puppy", "教你一个叫星星舞的新把戏")
        self.assertEqual(answer["intent"], "chat")
        self.assertNotIn("draft_request", answer)
        self.assertEqual(self.store.actions(), {})

    def test_explicit_short_sequence_returns_ordered_existing_actions_only(self):
        answer = {"reply": "我来挥手，然后坐下。", "intent": "action_sequence", "actions": ["wave", "sit"]}
        result = PetAgent(self.store, self.model(answer)).reply("puppy", "挥手再坐下")
        self.assertEqual(result, {**answer, "reply_source": "model"})
        self.assertEqual(self.store.actions(), {})
        self.assertEqual(result["intent"], "action_sequence")

    def test_invalid_draft_and_sequence_parameters_never_become_executable(self):
        candidates = [{"reply": "建议", "intent": "action_draft", "steps": [{"action": "wave"}]},
                      {"reply": "建议", "intent": "action_draft", "actions": ["wave", "sit"]},
                      {"reply": "建议", "intent": "action_sequence", "action": ["wave", "sit"]},
                      {"reply": "建议", "intent": "greet", "action": "wave"},
                      {"reply": "建议", "intent": "action_sequence", "actions": ["wave"], "angle": 90}]
        candidates += [{"reply": "建议", "intent": "action_sequence", "actions": value}
                       for value in ([], ["wave"] * 5, ["stop"], ["开心舞"], ["walk"], [{"action": "wave"}])]
        for answer in candidates:
            with self.subTest(answer):
                with self.assertRaises(ValueError):
                    PetAgent(self.store, self.model(answer)).reply("puppy", "挥手再坐下", approved_actions=("开心舞",))
        with self.assertRaises(ValueError):
            PetAgent(self.store, self.model({"reply": "建议", "intent": "action_sequence", "actions": ["wave", "sit"]}, explicit=False)).reply(
                "puppy", "如果明天挥手再坐下会怎么样")


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is available in the MacBook test environment")
class RoutineApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.testclient import TestClient
        from .routine_api import create_routine_router
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-routine-api-")
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "memory.sqlite3")
        def parent(authorization: str = Header(default="")):
            if authorization != "Bearer test-parent":
                raise HTTPException(401, "需要家长授权")
        def member(person):
            if person != "child":
                raise KeyError(person)
        app = FastAPI(); app.include_router(create_routine_router(self.store, parent, require_member=member))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.auth = {"Authorization": "Bearer test-parent"}
        self.base = "/api/routines/puppy/members/child"
        self.body = {"title": "收玩具", "steps": ["放好积木", "放好小车"], "expected_revision": 0}

    def test_parent_required_for_every_profile_and_routine_endpoint(self):
        for method, url, body in (("GET", "/api/pet-profile/puppy", None),
                ("PUT", "/api/pet-profile/puppy", {"style": "gentle", "greeting_action": "none", "expected_revision": 0}),
                ("GET", self.base, None), ("GET", self.base + "/toys", None),
                ("PUT", self.base + "/toys", self.body), ("DELETE", self.base + "/toys", {"expected_revision": 0}),
                ("POST", self.base + "/toys/transition", {"operation": "start", "expected_revision": 1, "expected_step_index": 0})):
            with self.subTest(url, method=method):
                self.assertEqual(self.client.request(method, url, json=body).status_code, 401)
        self.assertEqual(RoutineService(self.store).list("puppy", "child"), [])

    def test_strict_fields_versions_member_check_and_crud(self):
        route = self.base + "/toys"
        for change in ({"expected_revision": True}, {"expected_revision": "0"}, {"steps": [1]},
                       {"steps": []}, {"steps": ["x"] * 9}, {"steps": [" " ]}, {"action": "wave"}):
            with self.subTest(change):
                self.assertEqual(self.client.put(route, headers=self.auth, json={**self.body, **change}).status_code, 422)
        self.assertEqual(self.client.get("/api/routines/puppy/members/unknown", headers=self.auth).status_code, 404)
        saved = self.client.put(route, headers=self.auth, json=self.body).json()
        self.assertEqual(self.client.put(route, headers=self.auth, json=self.body).status_code, 409)
        active = self.client.post(route + "/transition", headers=self.auth, json={"operation": "start",
            "expected_revision": saved["revision"], "expected_step_index": 0}).json()
        self.assertEqual(active["state"], "active")
        self.assertEqual(self.client.post(route + "/transition", headers=self.auth, json={"operation": "advance",
            "expected_revision": saved["revision"], "expected_step_index": 0}).status_code, 409)
        self.assertEqual(self.client.request("DELETE", route, headers=self.auth, json={"expected_revision": active["revision"]}).status_code, 200)
        self.assertEqual(self.client.get(self.base, headers=self.auth).json(), {"routines": []})

    def test_profile_update_uses_cas_and_never_invokes_devices(self):
        route = "/api/pet-profile/puppy"
        body = {"style": "lively", "greeting_action": "wave", "expected_revision": 0}
        self.assertEqual(self.client.put(route, headers=self.auth, json=body).status_code, 200)
        self.assertEqual(self.client.put(route, headers=self.auth, json=body).status_code, 409)
        self.assertEqual(self.client.put(route, headers=self.auth, json={**body, "execute": True}).status_code, 422)
        self.assertEqual(self.client.get(route, headers=self.auth).json()["greeting_action"], "wave")


if __name__ == "__main__":
    unittest.main()
