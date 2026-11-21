"""Cancellation may discard an answer, but cannot stop an MLX worker thread."""

import asyncio
from functools import partial


class InferenceQueue:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.running = set()
        self.closing = False

    async def run(self, function, *args, **kwargs):
        if self.closing:
            raise RuntimeError("本地推理服务正在停止")
        await self.lock.acquire()
        try:
            if self.closing:
                raise RuntimeError("本地推理服务正在停止")
            task = asyncio.create_task(asyncio.to_thread(partial(function, *args, **kwargs)))
        except BaseException:
            self.lock.release()
            raise
        self.running.add(task)

        def finished(done):
            self.running.discard(done)
            self.lock.release()
            # Retrieve abandoned failures after a client cancels. Awaiters still
            # receive the same exception when they remain connected.
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def close(self):
        self.closing = True
        if self.running:
            await asyncio.shield(asyncio.gather(*self.running, return_exceptions=True))
