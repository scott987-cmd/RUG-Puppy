import asyncio
import threading
import unittest

from .inference import InferenceQueue


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_caller_cannot_overlap_models_or_delete_before_write_finishes(self):
        queue = InferenceQueue()
        started, release = threading.Event(), threading.Event()
        operations = []

        def first():
            started.set()
            release.wait(timeout=2)
            operations.append("write")

        task = asyncio.create_task(queue.run(first))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(queue.lock.locked())

        async def deletion():
            async with queue.lock:
                operations.append("delete")

        delete_task = asyncio.create_task(deletion())
        second = asyncio.create_task(queue.run(lambda: operations.append("second_model")))
        await asyncio.sleep(.02)
        self.assertEqual(operations, [])
        release.set()
        await asyncio.gather(delete_task, second)
        self.assertEqual(operations, ["write", "delete", "second_model"])
        self.assertFalse(queue.lock.locked())

    async def test_failed_worker_releases_actual_execution_lock(self):
        queue = InferenceQueue()
        def fail():
            raise ValueError("invalid model result")
        with self.assertRaises(ValueError):
            await queue.run(fail)
        self.assertEqual(await queue.run(lambda: "next"), "next")

    async def test_cancelled_shutdown_retains_worker_lock_and_rejects_new_work(self):
        queue = InferenceQueue()
        started, release = threading.Event(), threading.Event()
        def work():
            started.set()
            release.wait(timeout=2)
        task = asyncio.create_task(queue.run(work))
        await asyncio.to_thread(started.wait, 1)
        closing = asyncio.create_task(queue.close())
        await asyncio.sleep(0)
        closing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.assertTrue(queue.lock.locked())
        with self.assertRaises(RuntimeError):
            await queue.run(lambda: None)
        release.set()
        await task
        self.assertFalse(queue.lock.locked())


if __name__ == "__main__":
    unittest.main()
