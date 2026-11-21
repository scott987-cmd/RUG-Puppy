"""Voice task routing, process leases, and failure recovery."""

import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from .hermes_voice import HermesVoiceTasks, write_record


class HermesCandidateTests(unittest.TestCase):
    def setUp(self):
        self.source = (
            'def reply(allowed):\n'
            '    if not allowed:\n'
            '        return {"reply": "要家长允许看图，我才能帮你看。"}\n'
            '    return {"reply": "我刚才没看清，请再试一次。"}\n'
        )

    def test_requested_reply_change_is_accepted(self):
        from scripts.run_hermes_voice_task import validate_reply_change
        candidate = self.source.replace("我刚才没看清，请再试一次。", "相机暂时不可用，请稍后再试。")
        self.assertEqual(validate_reply_change(self.source, candidate), "相机暂时不可用，请稍后再试。")

    def test_permission_reply_change_is_rejected(self):
        from scripts.run_hermes_voice_task import validate_reply_change
        candidate = self.source.replace("我刚才没看清，请再试一次。", "相机暂时不可用，请稍后再试。")
        candidate = candidate.replace("要家长允许看图，我才能帮你看。", "相机暂时不可用，请稍后再试。")
        with self.assertRaises(ValueError):
            validate_reply_change(self.source, candidate)

    def test_followup_task_uses_the_reply_from_the_deployed_baseline(self):
        from scripts.run_hermes_voice_task import validate_reply_change
        deployed_reply = "相机暂时不可用。请稍后再试"
        before = self.source.replace("我刚才没看清，请再试一次。", deployed_reply)
        after = before.replace(deployed_reply, "相机暂时看不了，请稍后再试。")
        self.assertEqual(validate_reply_change(before, after, deployed_reply),
                         "相机暂时看不了，请稍后再试。")

    def test_unrelated_executable_change_is_rejected(self):
        from scripts.run_hermes_voice_task import validate_reply_change
        candidate = self.source.replace("我刚才没看清，请再试一次。", "相机暂时不可用，请稍后再试。")
        candidate += "\nlaunch_production()\n"
        with self.assertRaises(ValueError):
            validate_reply_change(self.source, candidate)


class HermesVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.worker = self.root / "worker.py"
        self.worker.write_text(
            "import json, os, pathlib, sys, time\n"
            "p = pathlib.Path(sys.argv[1]) / 'task.json'\n"
            "time.sleep(.15)\n"
            "row = json.loads(p.read_text())\n"
            "row['state'] = 'review_ready'\n"
            "q = p.with_suffix('.tmp')\n"
            "q.write_text(json.dumps(row))\n"
            "os.replace(q, p)\n"
        )
        self.service = HermesVoiceTasks(self.root / "tasks", self.worker, enabled=True)

    async def finish(self, service=None):
        service = service or self.service
        await asyncio.gather(*tuple(service.watchers))

    async def test_normal_chat_and_quoted_instructions_do_not_create_tasks(self):
        for text in ("小猫有几条腿", "给我讲个改进任务的故事", "怎么把小狗改进任务接起来",
                     "我说小狗改进任务，你能听懂吗"):
            self.assertIsNone(await self.service.respond("device", text))
        self.assertFalse(self.service.root.exists())

    async def test_bare_prefix_requests_the_missing_instruction(self):
        answer = await self.service.respond("device", "小狗改进任务。")
        self.assertIn("想改什么", answer["reply"])
        self.assertFalse(self.service.root.exists())

    async def test_status_question_with_prefix_punctuation_and_asr_tail_does_not_submit(self):
        task = self.service.root / "ready"
        task.mkdir(parents=True)
        write_record(task / "task.json", {
            "id": task.name, "device_id": "device", "state": "review_ready", "created_at": 1,
        })
        for text in ("小狗改进任务，完成了吗？", "改进任务完成了吗?只要我",
                     "小狗改進任務。完成了嗎？", "改进任务，进度怎么样了"):
            with self.subTest(text=text):
                result = await self.service.respond("device", text)
                self.assertIn("补丁和测试结果已准备好", result["reply"])
                self.assertEqual(len(list(self.service.root.glob("*/task.json"))), 1)
        self.assertEqual(self.service.watchers, set())

    async def test_ignored_misrouted_task_does_not_replace_real_review_result(self):
        for name, created, ignored in (("ready", 1, False), ("misrouted", 2, True)):
            task = self.service.root / name
            task.mkdir(parents=True)
            write_record(task / "task.json", {
                "id": name, "device_id": "device", "state": "review_ready",
                "created_at": created, "ignored": ignored,
            })
        self.assertEqual(self.service.latest("device")["id"], "ready")

    async def test_published_task_reports_release_instead_of_pending_review(self):
        task = self.service.root / "published"
        task.mkdir(parents=True)
        write_record(task / "task.json", {
            "id": task.name, "device_id": "device", "state": "review_ready",
            "created_at": 1, "published_at": 2,
        })
        answer = await self.service.respond("device", "改进任务完成了吗？")
        self.assertEqual(answer["reply"], "这次改进已经上线，可以使用了。")

    async def test_disabled_entry_does_not_start_worker(self):
        self.service.enabled = False
        answer = await self.service.respond("device", "小狗改进任务，把提示说清楚")
        self.assertIn("没有开启", answer["reply"])
        self.assertFalse(self.service.root.exists())

    async def test_real_subprocess_receives_instruction_and_result_survives_reload(self):
        answer = await self.service.respond("device", "小狗改進任務，把看圖失敗提示说清楚。")
        self.assertIn("已开始", answer["reply"])
        await self.finish()
        reloaded = HermesVoiceTasks(self.service.root, self.worker, enabled=True)
        record = reloaded.latest("device")
        self.assertEqual(record["state"], "review_ready")
        self.assertEqual(record["instruction"], "把看图失敗提示说清楚")
        self.assertIsNone(reloaded.latest("another-device"))
        result = await reloaded.respond("device", "改进任务完成了吗？")
        self.assertIn("家长", result["reply"])
        self.assertIn("补丁", result["reply"])

    async def test_busy_lease_is_shared_across_service_instances_and_released(self):
        await self.service.respond("one", "改进任务，把提示说清楚")
        second = HermesVoiceTasks(self.service.root, self.worker, enabled=True)
        reply = await second.respond("two", "改进任务，把提示说清楚")
        self.assertIn("已有一个", reply["reply"])
        self.assertIsNone(second.latest("two"))
        await self.finish()
        answer = await second.respond("two", "改进任务，把提示说清楚")
        self.assertIn("已开始", answer["reply"])
        await self.finish(second)

    async def test_crashed_worker_is_reported_and_does_not_keep_lease(self):
        self.worker.write_text("raise SystemExit(3)\n")
        await self.service.respond("device", "改进任务，把提示说清楚")
        await self.finish()
        result = self.service.latest("device")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["exit_code"], 3)
        self.assertIn("没有通过", (await self.service.respond("device", "改进任务进度"))["reply"])
        self.worker.write_text("raise SystemExit(0)\n")
        self.assertIn("已开始", (await self.service.respond("device", "改进任务，把提示说清楚"))["reply"])
        await self.finish()

    async def test_atomic_record_replacement_leaves_one_complete_json(self):
        path = self.root / "record.json"
        write_record(path, {"state": "starting"})
        write_record(path, {"state": "review_ready"})
        self.assertEqual(json.loads(path.read_text()), {"state": "review_ready"})
        self.assertEqual(list(self.root.glob("record.json.*.tmp")), [])

    async def test_reloaded_service_reports_an_orphaned_task_as_failed(self):
        task = self.service.root / "orphan"
        task.mkdir(parents=True)
        write_record(task / "task.json", {
            "id": task.name, "device_id": "device", "state": "hermes_running", "created_at": 1,
        })
        self.assertEqual(self.service.latest("device")["state"], "failed")

    async def test_device_audio_dispatches_before_camera_and_regular_model(self):
        from . import server
        replies = []
        instruction = "小狗改进任务，把提示说清楚"
        class Socket:
            headers = {"authorization": "Bearer test-device", "device-id": "review-device"}
            accept = AsyncMock()
        class Session:
            def __init__(self, socket, **callbacks):
                self.callbacks = callbacks
            async def run(self):
                replies.append(await self.callbacks["chat"](bytes(100), instruction))
        response = {"reply": "测试已受理"}
        with ExitStack() as stack:
            stack.enter_context(patch.object(server, "DEVICE_TOKEN", "test-device"))
            stack.enter_context(patch.object(server, "AudioSession", Session))
            stack.enter_context(patch.object(server, "OpusCodec", object))
            finish_learning = stack.enter_context(patch.object(server.LEARNING, "end_session"))
            submit = stack.enter_context(patch.object(server.HERMES_TASKS, "respond",
                                                       AsyncMock(return_value=response)))
            camera = stack.enter_context(patch.object(server.DIALOGUE_CAMERA, "prepare", AsyncMock()))
            model = stack.enter_context(patch.object(server, "chat", AsyncMock()))
            await server.device_audio(Socket())
        submit.assert_awaited_once_with("review-device", instruction)
        finish_learning.assert_called_once_with("review-device", None)
        camera.assert_not_awaited()
        model.assert_not_awaited()
        self.assertEqual(replies, [response])
