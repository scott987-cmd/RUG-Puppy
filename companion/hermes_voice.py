"""Voice submission and review status for isolated Hermes development tasks."""

import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


PREFIX = re.compile(r"^(?:小狗)?改进任务[\s，,。．：:！!]*(.*)$", re.DOTALL)
FORMS = str.maketrans("進務圖機暫時無請後試嗎樣麼", "进务图机暂时无请后试吗样么")
STATUS_REQUESTS = {"改进任务进度", "改进任务怎么样了", "改进任务完成了吗"}
STATUS_SUFFIXES = ("进度", "怎么样了", "完成了吗", "完成了没有", "做好了吗", "好了吗", "做完了吗")
TERMINAL_STATES = {"review_ready", "failed", "no_change", "cancelled"}


def write_record(path, record):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class HermesVoiceTasks:
    def __init__(self, root, worker, *, enabled=False):
        self.root = Path(root)
        self.worker = Path(worker)
        self.enabled = enabled
        self.watchers = set()

    def latest(self, device_id):
        if not self.root.exists():
            return None
        records = []
        for path in self.root.glob("*/task.json"):
            try:
                record = json.loads(path.read_text())
                if record.get("device_id") == device_id and not record.get("ignored"):
                    records.append(record)
            except (OSError, ValueError):
                continue
        record = max(records, key=lambda row: row["created_at"], default=None)
        if record and record["state"] not in TERMINAL_STATES:
            # The inherited lease outlives a gateway restart, and is released
            # by the OS if the worker crashes before it can write a result.
            with (self.root / ".worker.lock").open("a+") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return record
                record.update(state="failed", error="worker_exited_before_results")
                write_record(self.root / record["id"] / "task.json", record)
        return record

    def status_reply(self, device_id):
        record = self.latest(device_id)
        if record is None:
            return {"reply": "还没有改进任务。"}
        if record.get("published_at"):
            return {"reply": "这次改进已经上线，可以使用了。"}
        messages = {
            "review_ready": "改进补丁和测试结果已准备好，等家长查看后确认上线。",
            "failed": "这次改进没有通过验证，结果已记录，请家长查看。",
            "no_change": "这次没有生成代码改动，请家长查看任务记录。",
            "cancelled": "这个改进任务已取消。",
        }
        return {"reply": messages.get(record["state"], "改进任务还在处理中，完成后会留下补丁和测试结果。")}

    async def respond(self, device_id, transcript):
        text = transcript.strip().translate(FORMS).rstrip("。？！?! ")
        if text in STATUS_REQUESTS:
            return self.status_reply(device_id)
        match = PREFIX.fullmatch(text)
        if match is None:
            return None
        instruction = match.group(1).strip()
        if instruction.startswith(STATUS_SUFFIXES):
            return self.status_reply(device_id)
        if not self.enabled:
            return {"reply": "改进任务入口还没有开启，请家长查看设置。"}
        if len(instruction) < 4:
            return {"reply": "请说改进任务，再告诉我想改什么。"}
        if len(instruction) > 800:
            return {"reply": "这次改进要求有点长，请先说一个要改的问题。"}
        try:
            record = self.submit(device_id, instruction)
        except BlockingIOError:
            return {"reply": "已有一个改进任务在处理，请等它完成。"}
        except OSError:
            return {"reply": "改进任务暂时没有启动成功，请稍后再试。"}
        watcher = asyncio.create_task(self.watch(*record))
        self.watchers.add(watcher)
        watcher.add_done_callback(self.watchers.discard)
        return {"reply": "收到改进任务，已开始在隔离副本里处理，完成后请家长查看补丁和测试结果。"}

    def submit(self, device_id, instruction):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lease = (self.root / ".worker.lock").open("a+")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            task = self.root / uuid.uuid4().hex
            task.mkdir(mode=0o700)
            record = {"id": task.name, "device_id": device_id, "instruction": instruction,
                      "state": "starting", "created_at": time.time()}
            write_record(task / "task.json", record)
            try:
                with (task / "worker.log").open("ab") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-B", str(self.worker), str(task),
                         "--lease-fd", str(lease.fileno())],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        pass_fds=(lease.fileno(),), start_new_session=True,
                    )
            except OSError:
                write_record(task / "task.json", {**record, "state": "failed",
                                                  "error": "worker_start_failed"})
                raise
            return process, task
        finally:
            lease.close()

    async def watch(self, process, task):
        code = await asyncio.to_thread(process.wait)
        path = task / "task.json"
        try:
            record = json.loads(path.read_text())
            if record["state"] not in TERMINAL_STATES:
                write_record(path, {**record, "state": "failed", "exit_code": code,
                                    "error": "worker_exited_before_results"})
        except (OSError, ValueError):
            return
