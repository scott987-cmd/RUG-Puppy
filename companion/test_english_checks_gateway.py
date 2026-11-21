import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from .test_server import server
from .learning import LearningStore
from .store import Store


class EnglishChecksGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-checks-api-")
        root = Path(self.temp.name)
        self.store = Store(root / "pet.sqlite3")
        self.learning = LearningStore(root / "learning.sqlite3")
        self.patches = [patch.object(server, "STORE", self.store),
                        patch.object(server, "LEARNING", self.learning),
                        patch.object(server, "PARENT_TOKEN", "fictional-parent")]
        for item in self.patches:
            item.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
            base_url="http://isolated.test", headers={"Authorization": "Bearer fictional-parent"})
        self.path = "/api/memory/puppy/child/english-checks"

    async def asyncTearDown(self):
        await self.client.aclose()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def test_parent_check_requires_word_permission_and_delay(self):
        payload = {"word": "dog", "checkpoint": "baseline", "result": "not_yet", "modality": "pointed"}
        with patch("companion.store.time.time", return_value=1000000.0):
            response = await self.client.post(self.path, json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.store.record_english("puppy", "child", "dog", False)
        with patch("companion.store.time.time", return_value=1000001.0):
            early = await self.client.post(self.path, json={**payload, "checkpoint": "next_day"})
        self.assertEqual(early.status_code, 422, early.text)
        with patch("companion.store.time.time", return_value=1000000.0 + 24 * 3600):
            later = await self.client.post(self.path, json={**payload, "checkpoint": "next_day"})
        self.assertEqual(later.status_code, 200, later.text)
        self.assertEqual(later.json()["source"], "parent_report")
        page = await self.client.get("/api/memory/puppy/child")
        self.assertEqual(page.status_code, 200, page.text)
        self.assertEqual(len(page.json()["english_checks"]), 2)
        self.assertTrue(all(item["source"] == "parent_report" for item in page.json()["english_checks"]))
        self.assertTrue(page.json()["english_review_plan"])
        unknown = await self.client.post(self.path, json={**payload, "word": "secret"})
        self.assertEqual(unknown.status_code, 422)
        self.learning.set_settings("puppy", "child", patch={"save_progress": False})
        disabled = await self.client.post(self.path, json={**payload, "checkpoint": "week_later"})
        self.assertEqual(disabled.status_code, 409)


if __name__ == "__main__":
    unittest.main()
