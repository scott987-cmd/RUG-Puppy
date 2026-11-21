"""Model startup priming without loading real weights or contacting a service."""
import asyncio
import os
import unittest
from unittest.mock import patch

from .inference import InferenceQueue
from . import server


class PreloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_status_starts_while_preload_is_still_running(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_preload():
            started.set()
            await release.wait()

        class FakeService:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            async def close(self):
                pass

            def status(self):
                return {"running": False}

        with patch.dict(os.environ, {"PUPPY_MODEL_PRELOAD": "1"}, clear=True), \
             patch.object(server, "INFERENCE", InferenceQueue()), \
             patch.object(server, "PatrolService", FakeService), \
             patch.object(server, "GestureService", FakeService), \
             patch.object(server, "preload_model", slow_preload), \
             patch.object(server, "REMOTE", None), \
             patch.object(server, "PATROL", None), \
             patch.object(server, "GESTURES", None), \
             patch.object(server, "RUNNING", {}), \
             patch.object(server, "STOPPING", {}):
            async with asyncio.timeout(1):
                async with server.lifespan(None):
                    await started.wait()
                    self.assertFalse(release.is_set())
                    self.assertIn("devices", await server.status())
                    release.set()

    async def test_preload_primes_once_and_reuses_the_model(self):
        queue = InferenceQueue()
        calls = []

        class FakeModel:
            def __init__(self, path):
                calls.append(("load", path))

            def requests_camera(self, text):
                calls.append(("prime", text))
                return False

        with patch.object(server, "INFERENCE", queue), \
             patch.object(server, "MODEL", None), \
             patch.object(server, "MODEL_PATH", "/test-only/local-model"), \
             patch.object(server, "LocalModel", FakeModel):
            await server.preload_model()
            await server.preload_model()
            self.assertEqual(calls, [("load", "/test-only/local-model"),
                                     ("prime", "你好")])
        await queue.close()

    async def test_preload_failure_does_not_escape_background_task(self):
        queue = InferenceQueue()

        def fail(_):
            raise RuntimeError("synthetic load failure")

        with patch.object(server, "INFERENCE", queue), \
             patch.object(server, "MODEL", None), \
             patch.object(server, "LocalModel", fail):
            await server.preload_model()
            self.assertIsNone(server.MODEL)
        await queue.close()


if __name__ == "__main__":
    unittest.main()
