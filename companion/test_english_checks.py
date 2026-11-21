import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .store import Store


class EnglishChecksTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="puppy-english-checks-")
        self.path = Path(self.directory.name) / "pet.sqlite3"
        self.store = Store(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_baseline_then_delayed_parent_observations(self):
        with patch("companion.store.time.time", return_value=1000000.0):
            self.store.record_english_check("puppy", "child", "dog", "baseline", "not_yet", "pointed")
            self.store.record_english("puppy", "child", "dog", False, activity="introduce")
        self.assertEqual(self.store.english_review_plan("puppy", None), [])
        with patch("companion.store.time.time", return_value=1000001.0):
            plan = self.store.english_review_plan("puppy", "child")
            self.assertEqual([item["status"] for item in plan], ["upcoming", "upcoming"])
            with self.assertRaisesRegex(ValueError, "尚未到达"):
                self.store.record_english_check("puppy", "child", "dog", "next_day", "independent", "spoken")
        with patch("companion.store.time.time", return_value=1000000.0 + 24 * 3600):
            self.store.record_english_check("puppy", "child", "dog", "next_day", "independent", "pointed")
            self.assertEqual(self.store.english_review_plan("puppy", "child")[0]["status"], "recorded")
        with patch("companion.store.time.time", return_value=1000000.0 + 7 * 24 * 3600):
            self.store.record_english_check("puppy", "child", "dog", "week_later", "prompted", "used_in_context")
        self.assertEqual(len(self.store.english_checks("puppy", "child")), 3)
        self.store.forget_person("puppy", "child")
        self.assertEqual(self.store.english_checks("puppy", "child"), [])

    def test_baseline_cannot_be_invented_after_learning(self):
        self.store.record_english("puppy", "child", "cat", False)
        with self.assertRaisesRegex(ValueError, "第一次练习前"):
            self.store.record_english_check("puppy", "child", "cat", "baseline", "not_yet", "spoken")

    def test_existing_progress_migrates_with_latest_contact_as_review_anchor(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE english_progress")
            db.execute("""CREATE TABLE english_progress (
                device_id TEXT NOT NULL, person_id TEXT NOT NULL, word TEXT NOT NULL,
                presentations INTEGER NOT NULL, text_matches INTEGER NOT NULL,
                updated_at REAL NOT NULL, PRIMARY KEY(device_id,person_id,word))""")
            db.execute("INSERT INTO english_progress VALUES (?,?,?,?,?,?)",
                       ("puppy", "child", "ball", 2, 1, 123.0))
        self.store = Store(self.path)
        self.assertEqual(self.store.english_progress("puppy", "child")[0]["review_anchor_at"], 123.0)


if __name__ == "__main__":
    unittest.main()
