"""Protocol tests use synthetic vectors, not a household accuracy benchmark.

Set PUPPY_IDENTITY_MODEL_DIR, PUPPY_IDENTITY_FACE_FIXTURE (public test image),
and PUPPY_IDENTITY_VOICE_FIXTURE (synthetic mono 16 kHz WAV) for real engines.
"""
import math
import hashlib
import os
import stat
import tempfile
import time
import unittest
import wave
from unittest.mock import patch
from pathlib import Path

from companion.identity import (IdentityPolicy, IdentityService, IdentityStore,
                                Observation, QualityError, normalized)


class FixedEngine:
    def __init__(self, modality, vector):
        self.modality, self.vector = modality, vector
        self.calls = 0

    def extract(self, data, *, captured_at):
        self.calls += 1
        if data == b"bad":
            raise QualityError("multiple_faces" if self.modality == "face" else "voice_too_quiet")
        return Observation(self.modality, "test-model", self.vector, {}, captured_at)


class IdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "private" / "identity.sqlite3"
        self.store = IdentityStore(self.path)
        self.face, self.voice = FixedEngine("face", [1, 0]), FixedEngine("voice", [1, 0])
        self.service = IdentityService(self.store, self.face, self.voice)

    def tearDown(self):
        self.temp.cleanup()

    def enroll(self, person="member-a", modality="face", axis=0):
        for sign in (-1, 1):
            vector = [1, sign * .05] if axis == 0 else [sign * .05, 1]
            self.store.enroll(person, Observation(modality, "test-model", vector), parent_confirmed=True)

    def recognize(self, **changes):
        values = dict(face_jpeg=b"face", pcm16=b"voice", face_captured_at=time.time(), voice_captured_at=time.time())
        values.update(changes)
        return self.service.recognize(**values)

    def test_registration_requires_parent_before_extracting(self):
        with self.assertRaises(PermissionError):
            self.service.enroll("member-a", "face", b"face")
        self.assertEqual(self.face.calls, 0)
        self.assertEqual(self.store.list_samples(), [])

    def test_registry_is_private_persistent_and_contains_no_raw_media(self):
        self.enroll()
        samples = IdentityStore(self.path).list_samples("member-a")
        self.assertEqual(len(samples), 2)
        self.assertNotIn("embedding", samples[0])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        with self.store.connect() as db:
            columns = [row["name"] for row in db.execute("PRAGMA table_info(identity_samples)")]
            self.assertFalse(any("image" in name or "audio" in name for name in columns))

    def test_two_modalities_and_two_samples_required(self):
        self.store.enroll("member-a", Observation("face", "test-model", [1, 0]), parent_confirmed=True)
        self.assertIsNone(self.recognize()["person_id"])
        self.enroll(modality="voice")
        self.assertIsNone(self.recognize()["person_id"])
        self.store.enroll("member-a", Observation("face", "test-model", [1, .08]), parent_confirmed=True)
        self.assertEqual(self.recognize()["person_id"], "member-a")
        self.assertIsNone(self.recognize(pcm16=None)["person_id"])
        self.assertIsNone(self.recognize(face_jpeg=None)["person_id"])

    def test_low_score_and_multiple_candidates_are_unknown(self):
        self.enroll()
        self.enroll(modality="voice")
        self.face.vector = [.5, math.sqrt(.75)]
        result = self.recognize()
        self.assertIsNone(result["person_id"])
        self.assertEqual(result["evidence"]["face"]["reason"], "below_threshold")
        self.face.vector = [1, 0]
        self.enroll("member-b")
        result = self.recognize()
        self.assertIsNone(result["person_id"])
        self.assertEqual(result["reason"], "ambiguous_candidates")

    def test_face_voice_conflict_does_not_select_any_member(self):
        self.enroll()
        self.enroll(modality="voice")
        self.enroll("member-b", "voice", axis=1)
        self.voice.vector = [0, 1]
        result = self.recognize()
        self.assertIsNone(result["person_id"])
        self.assertEqual(result["reason"], "modality_conflict")

    def test_quality_errors_missing_times_and_stale_frames_are_unknown(self):
        self.enroll()
        self.enroll(modality="voice")
        for changes in ({"face_jpeg": b"bad"}, {"pcm16": b"bad"}, {"face_captured_at": None},
                        {"face_captured_at": time.time() - 20}, {"voice_captured_at": float("nan")}):
            self.assertIsNone(self.recognize(**changes)["person_id"])
        mismatch = self.recognize(face_captured_at=time.time() - 10)
        self.assertEqual(mismatch["reason"], "capture_time_mismatch")

    def test_revocation_takes_effect_immediately_and_requires_parent(self):
        self.enroll()
        self.enroll(modality="voice")
        sample = self.store.list_samples()[0]["sample_id"]
        with self.assertRaises(PermissionError):
            self.store.revoke_sample(sample)
        self.assertEqual(self.recognize()["person_id"], "member-a")
        self.assertTrue(self.store.revoke_sample(sample, parent_confirmed=True))
        self.assertIsNone(self.recognize()["person_id"])
        self.assertEqual(self.store.revoke_person("member-a", parent_confirmed=True), 3)
        self.assertEqual(self.store.list_samples(), [])

    def test_duplicate_and_model_changed_samples_are_rejected(self):
        original = Observation("face", "test-model", [1, .1])
        self.store.enroll("member-a", original, parent_confirmed=True)
        with self.assertRaisesRegex(ValueError, "duplicate_sample"):
            self.store.enroll("member-a", original, parent_confirmed=True)
        with self.assertRaisesRegex(ValueError, "model_changed"):
            self.store.enroll("member-a", Observation("face", "other-model", [1, .2]), parent_confirmed=True)
        self.assertEqual(len(self.store.list_samples()), 1)

    def test_invalid_vectors_and_policy_are_rejected(self):
        for vector in ([0, 0], [1, float("nan")], [float("inf"), 1], [1]):
            with self.assertRaises(ValueError):
                normalized(vector)
        with self.assertRaises(ValueError):
            IdentityPolicy(face_threshold=-1)
        with self.assertRaises(ValueError):
            IdentityPolicy(min_samples=1)

    def test_inconsistent_enrollment_sample_is_rejected(self):
        self.service.enroll("member-a", "face", b"image", parent_confirmed=True)
        self.face.vector = [0, 1]
        with self.assertRaisesRegex(QualityError, "enrollment_inconsistent"):
            self.service.enroll("member-a", "face", b"other-image", parent_confirmed=True)
        self.assertEqual(len(self.store.list_samples()), 1)

    def test_model_checksum_detects_same_size_corruption(self):
        from companion.identity_models import MODELS, verified_model
        model = Path(self.temp.name) / "test.onnx"
        model.write_bytes(b"abc")
        spec = {"test.onnx": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}
        with patch.dict(MODELS, spec):
            self.assertEqual(verified_model(self.temp.name, "test.onnx"), model)
            model.write_bytes(b"xyz")
            with self.assertRaisesRegex(ValueError, "checksum_mismatch"):
                verified_model(self.temp.name, "test.onnx")


@unittest.skipUnless(os.environ.get("PUPPY_IDENTITY_MODEL_DIR"), "optional real identity models not configured")
class RealEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from companion.identity_models import FaceEngine, VoiceEngine
        cls.face = FaceEngine(os.environ["PUPPY_IDENTITY_MODEL_DIR"])
        cls.voice = VoiceEngine(os.environ["PUPPY_IDENTITY_MODEL_DIR"])

    def test_real_face_forward_and_multiple_faces_rejected(self):
        import cv2
        import numpy as np
        path = os.environ.get("PUPPY_IDENTITY_FACE_FIXTURE")
        if not path:
            self.skipTest("public face fixture not configured")
        data = Path(path).read_bytes()
        result = self.face.extract(data, captured_at=time.time())
        self.assertEqual(len(result.vector), 128)
        self.assertAlmostEqual(sum(x * x for x in result.vector), 1)
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        combined = cv2.hconcat([decoded, decoded])
        _, jpeg = cv2.imencode(".jpg", combined)
        with self.assertRaisesRegex(QualityError, "multiple_faces"):
            self.face.extract(jpeg.tobytes(), captured_at=time.time())

    def test_real_voice_forward_and_silence_rejected(self):
        path = os.environ.get("PUPPY_IDENTITY_VOICE_FIXTURE")
        if not path:
            self.skipTest("synthetic voice fixture not configured")
        with wave.open(path, "rb") as file:
            self.assertEqual((file.getnchannels(), file.getsampwidth(), file.getframerate()), (1, 2, 16000))
            pcm = file.readframes(file.getnframes())
        result = self.voice.extract(pcm, captured_at=time.time())
        self.assertEqual(len(result.vector), 256)
        self.assertGreaterEqual(result.quality["speech_seconds"], 2.5)
        with self.assertRaisesRegex(QualityError, "voice_too_quiet"):
            self.voice.extract(bytes(5 * 32000), captured_at=time.time())


if __name__ == "__main__":
    unittest.main()
