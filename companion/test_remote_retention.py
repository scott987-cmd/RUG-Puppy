"""Media retention tests use only temporary local and fake-cloud directories."""

import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
import uuid

from unittest.mock import patch

from companion.remote import mirror_files, mirror_once
from companion.remote_retention import apply_media, plan_media


class RemoteRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-retention-")
        self.root = Path(self.temp.name)
        self.local = self.root / "local"
        self.cloud = self.root / "PuppyCompanion"
        (self.local / "outbox/media").mkdir(parents=True)
        (self.cloud / "media").mkdir(parents=True)
        self.now = 1900000000
        self.id = str(uuid.uuid4())
        self.jpeg = b"\xff\xd8retention-test\xff\xd9"
        self.name = self.id + ".jpg"
        self.db = sqlite3.connect(self.local / "remote.sqlite3")
        self.db.execute("CREATE TABLE commands(request_id TEXT, state TEXT, receipt TEXT)")
        self.original = self.root / "gateway-original.jpg"
        self.original.write_bytes(self.jpeg)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def insert(self, *, age_days=8, kind="photo"):
        receipt = {"request_id": self.id, "state": "succeeded", "kind": kind,
                   "updated_at": self.now - age_days * 86400,
                   "result": {"ok": True, "media": {"file": "media/" + self.name,
                       "sha256": hashlib.sha256(self.jpeg).hexdigest(), "bytes": len(self.jpeg)}}}
        self.db.execute("INSERT INTO commands VALUES(?,?,?)", (self.id, "succeeded", json.dumps(receipt)))
        self.db.commit()

    def copies(self):
        (self.local / "outbox/media" / self.name).write_bytes(self.jpeg)
        (self.cloud / "media" / self.name).write_bytes(self.jpeg)

    def test_preview_then_apply_only_expired_owned_media(self):
        self.insert()
        self.copies()
        candidates = plan_media(self.local, self.cloud, 7, now=self.now)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["ready"])
        self.assertTrue((self.cloud / "media" / self.name).exists())
        self.assertEqual(apply_media(self.local, self.cloud, 7, now=self.now),
                         {"eligible": 1, "blocked": 0, "local_deleted": 1, "cloud_deleted": 1})
        self.assertTrue(self.original.exists())
        self.assertEqual(apply_media(self.local, self.cloud, 7, now=self.now)["eligible"], 0)

    def test_recent_photo_is_ignored(self):
        self.insert(age_days=2)
        self.copies()
        self.assertEqual(plan_media(self.local, self.cloud, 7, now=self.now), [])
        self.assertEqual(apply_media(self.local, self.cloud, 7, now=self.now)["eligible"], 0)
        self.assertTrue((self.cloud / "media" / self.name).exists())

    def test_old_non_photo_receipt_is_ignored(self):
        self.insert(kind="status")
        self.copies()
        self.assertEqual(plan_media(self.local, self.cloud, 7, now=self.now), [])
        self.assertTrue((self.cloud / "media" / self.name).exists())

    def test_mismatched_cloud_blocks_both_deletions(self):
        self.insert()
        self.copies()
        (self.cloud / "media" / self.name).write_bytes(b"different")
        candidate = plan_media(self.local, self.cloud, 7, now=self.now)[0]
        self.assertEqual(candidate["cloud"], "mismatch")
        self.assertFalse(candidate["ready"])
        self.assertEqual(apply_media(self.local, self.cloud, 7, now=self.now)["blocked"], 1)
        self.assertTrue((self.local / "outbox/media" / self.name).exists())

    def test_missing_local_copy_can_finish_interrupted_cleanup(self):
        self.insert()
        (self.cloud / "media" / self.name).write_bytes(self.jpeg)
        self.assertEqual(apply_media(self.local, self.cloud, 7, now=self.now)["cloud_deleted"], 1)

    def test_running_consumer_refuses_apply(self):
        self.insert()
        self.copies()
        fd = os.open(self.local / "consumer.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "stop_gateway"):
                apply_media(self.local, self.cloud, 7, now=self.now)
        finally:
            os.close(fd)
        self.assertTrue((self.cloud / "media" / self.name).exists())

    def test_symlinked_media_is_not_removed(self):
        self.insert()
        (self.local / "outbox/media" / self.name).symlink_to(self.original)
        (self.cloud / "media" / self.name).write_bytes(self.jpeg)
        candidate = plan_media(self.local, self.cloud, 7, now=self.now)[0]
        self.assertEqual(candidate["local"], "symlink")
        self.assertFalse(candidate["ready"])
        self.assertEqual(apply_media(self.local, self.cloud, 7, now=self.now)["blocked"], 1)
        self.assertTrue(self.original.exists())

    def test_opt_in_mirror_prunes_before_media_can_be_reuploaded(self):
        self.insert()
        (self.local / "outbox/media" / self.name).write_bytes(self.jpeg)
        with patch("companion.remote_retention.time.time", return_value=self.now):
            self.assertEqual(mirror_files(self.local, self.cloud)["pushed"], 1)
            pruned = mirror_files(self.local, self.cloud, retention_days=7)
        self.assertEqual(pruned["retention"]["local_deleted"], 1)
        self.assertEqual(pruned["retention"]["cloud_deleted"], 1)
        self.assertEqual(pruned["pushed"], 0)
        self.assertEqual(mirror_files(self.local, self.cloud)["pushed"], 0)
        self.assertFalse((self.cloud / "media" / self.name).exists())

    def test_subprocess_mirror_passes_explicit_retention_option(self):
        self.now = int(time.time())
        self.insert()
        (self.local / "outbox/media" / self.name).write_bytes(self.jpeg)
        result = asyncio.run(mirror_once(self.local, self.cloud, retention_days=7))
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["retention"]["local_deleted"], 1)
        self.assertEqual(result["pushed"], 0)
        self.assertFalse((self.cloud / "media" / self.name).exists())

    def test_retention_failure_keeps_receipts_moving_but_skips_media(self):
        self.insert()
        (self.local / "outbox/media" / self.name).write_bytes(self.jpeg)
        receipts = self.local / "outbox/receipts"
        receipts.mkdir()
        (receipts / "public.json").write_bytes(b"public receipt")
        result = mirror_files(self.local, self.cloud, retention_days=0)
        self.assertIn("retention_unavailable", result["errors"])
        self.assertTrue((self.cloud / "receipts/public.json").exists())
        self.assertFalse((self.cloud / "media" / self.name).exists())


if __name__ == "__main__":
    unittest.main()
