"""Isolated parent-registration tests. No camera, microphone or real identity samples."""
import asyncio
import io
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from companion.identity import IdentityService, IdentityStore, Observation
from companion.identity_api import (IdentityRegistration, create_identity_router,
                                    decode_wav, MAX_JPEG_BYTES)


def jpeg():
    output = io.BytesIO()
    Image.new("RGB", (128, 128), "green").save(output, format="JPEG")
    return output.getvalue()


def wav(rate=16000, channels=1, width=2, seconds=3):
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setnchannels(channels); file.setsampwidth(width); file.setframerate(rate)
        file.writeframes(b"\x11" * int(rate * seconds * channels * width))
    return output.getvalue()


class Engine:
    def __init__(self, modality):
        self.modality, self.calls = modality, 0
        self.started, self.release = None, None

    def extract(self, data, *, captured_at):
        self.calls += 1
        if self.started:
            self.started.set()
            if not self.release.wait(3):
                raise ValueError("test preparation not released")
        return Observation(self.modality, "api-test-model", [1, self.calls * .04], {}, captured_at)


def setup_service(directory):
    face, voice = Engine("face"), Engine("voice")
    return IdentityService(IdentityStore(Path(directory) / "identity.sqlite3"), face, voice), face, voice


class RegistrationHTTPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service, self.face, self.voice = setup_service(self.temp.name)
        self.captures = []

        async def capture(device_id):
            self.captures.append(device_id)
            return jpeg()

        self.registration = IdentityRegistration(self.service, capture_jpeg=capture)
        self.app = FastAPI()

        def parent(authorization: str = Header(default="")):
            if authorization != "Bearer test-parent":
                raise HTTPException(401, "parent required")

        self.app.include_router(create_identity_router(self.registration, parent))
        self.client = TestClient(self.app)
        self.auth = {"Authorization": "Bearer test-parent"}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def member(self, person="child-1", name="孩子"):
        response = self.client.put("/api/identity/members/" + person, json={"display_name": name}, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)

    def test_all_registration_endpoints_require_parent(self):
        routes = [("GET", "/members"), ("PUT", "/members/child-1"), ("GET", "/members/child-1/samples"),
                  ("DELETE", "/members/child-1"), ("DELETE", "/members/child-1/samples/sample"),
                  ("POST", "/members/child-1/face"), ("POST", "/members/child-1/voice"),
                  ("POST", "/members/child-1/capture-face"), ("POST", "/voice-tickets"),
                  ("GET", "/voice-tickets/ticket"), ("DELETE", "/voice-tickets/ticket")]
        for method, path in routes:
            result = self.client.request(method, "/api/identity" + path)
            self.assertEqual(result.status_code, 401, (method, path, result.text))
        self.assertEqual(self.face.calls + self.voice.calls, 0)
        self.assertEqual(self.captures, [])
        self.assertEqual(self.service.store.list_samples(), [])

    def test_member_name_and_samples_persist_without_raw_files(self):
        self.member(name="豆豆的朋友")
        result = self.client.post("/api/identity/members/child-1/face", files={"file": ("face.jpg", jpeg(), "image/jpeg")}, headers=self.auth)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.face.calls, 1)
        reopened = IdentityRegistration(IdentityService(IdentityStore(Path(self.temp.name) / "identity.sqlite3")))
        self.assertEqual(reopened.list_members()[0]["display_name"], "豆豆的朋友")
        self.assertEqual(reopened.list_members()[0]["counts"], {"face": 1, "voice": 0})
        self.assertEqual([path.name for path in Path(self.temp.name).iterdir()], ["identity.sqlite3"])

    def test_puppy_capture_injects_only_bytes(self):
        self.member()
        response = self.client.post("/api/identity/members/child-1/capture-face", json={"device_id": "puppy-1"}, headers=self.auth)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.captures, ["puppy-1"])
        self.assertEqual(self.face.calls, 1)

    def test_wav_upload_validates_actual_format_and_duration(self):
        self.member()
        good = self.client.post("/api/identity/members/child-1/voice", files={"file": ("sample.wav", wav(), "audio/wav")}, headers=self.auth)
        self.assertEqual(good.status_code, 200, good.text)
        invalid = [wav(rate=8000), wav(channels=2), wav(width=1), wav(seconds=2), wav()[:-8]]
        for content in invalid:
            response = self.client.post("/api/identity/members/child-1/voice", files={"file": ("sample.wav", content, "audio/wav")}, headers=self.auth)
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.voice.calls, 1)

    def test_bad_jpeg_oversize_and_multiple_files_never_reach_model(self):
        self.member()
        for content, expected in ((b"not a jpeg", 422), (b"x" * (MAX_JPEG_BYTES + 1), 413),
                                  (b"x" * (MAX_JPEG_BYTES + 20000), 413)):
            response = self.client.post("/api/identity/members/child-1/face", files={"file": ("face.jpg", content, "image/jpeg")}, headers=self.auth)
            self.assertEqual(response.status_code, expected, response.text)
        response = self.client.post("/api/identity/members/child-1/face", files=[("file", ("a.jpg", jpeg())), ("file", ("b.jpg", jpeg()))], headers=self.auth)
        self.assertEqual(response.status_code, 413, response.text)
        response = self.client.post("/api/identity/members/child-1/face", json={"file": "bad"}, headers=self.auth)
        self.assertEqual(response.status_code, 415)
        self.assertEqual(self.face.calls, 0)

    def test_delete_sample_checks_owner_and_delete_member_revokes_all(self):
        self.member(); self.member("child-2", "另一个孩子")
        result = self.client.post("/api/identity/members/child-1/face", files={"file": ("a.jpg", jpeg())}, headers=self.auth).json()
        sample_id = result["sample_id"]
        wrong = self.client.delete("/api/identity/members/child-2/samples/" + sample_id, headers=self.auth)
        self.assertEqual(wrong.status_code, 404)
        removed = self.client.delete("/api/identity/members/child-1", headers=self.auth)
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json()["revoked_samples"], 1)
        self.assertEqual(self.service.store.list_samples(), [])
        self.assertEqual(len(self.registration.list_members()), 1)

    def test_ticket_creation_is_bound_to_existing_member_and_device(self):
        missing = self.client.post("/api/identity/voice-tickets", json={"device_id": "puppy-1", "person_id": "child-1"}, headers=self.auth)
        self.assertEqual(missing.status_code, 404)
        self.member()
        result = self.client.post("/api/identity/voice-tickets", json={"device_id": "puppy-1", "person_id": "child-1"}, headers=self.auth)
        self.assertEqual(result.status_code, 200)
        ticket = result.json()
        self.assertEqual(ticket["state"], "waiting")
        duplicate = self.client.post("/api/identity/voice-tickets", json={"device_id": "puppy-1", "person_id": "child-1"}, headers=self.auth)
        self.assertEqual(duplicate.status_code, 409)
        cancelled = self.client.delete("/api/identity/voice-tickets/" + ticket["ticket_id"], headers=self.auth)
        self.assertEqual(cancelled.json()["state"], "cancelled")
        self.assertEqual(self.voice.calls, 0)

    def test_ticket_requires_ready_voice_model(self):
        self.member()
        self.service.voice_engine = None
        response = self.client.post("/api/identity/voice-tickets",
                                    json={"device_id": "puppy-1", "person_id": "child-1"},
                                    headers=self.auth)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.registration.tickets, {})


class RegistrationRaceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service, self.face, self.voice = setup_service(self.temp.name)
        self.now = 100.0
        self.registration = IdentityRegistration(self.service, clock=lambda: self.now)
        self.registration.save_member("child-1", "孩子")
        self.pcm = bytes(3 * 32000)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def blocked_consumption(self):
        self.voice.started, self.voice.release = threading.Event(), threading.Event()
        ticket = await self.registration.create_ticket("puppy-1", "child-1")
        task = asyncio.create_task(self.registration.consume_enrollment("puppy-1", self.pcm))
        self.assertTrue(await asyncio.to_thread(self.voice.started.wait, 2))
        return ticket, task

    async def test_no_ticket_other_device_expired_or_invalid_pcm_never_register(self):
        self.assertIsNone(await self.registration.consume_enrollment("puppy-1", self.pcm))
        await self.registration.create_ticket("puppy-1", "child-1")
        self.assertIsNone(await self.registration.consume_enrollment("other-device", self.pcm))
        self.now += 61
        result = await self.registration.consume_enrollment("puppy-1", self.pcm)
        self.assertEqual(result["state"], "expired")
        await self.registration.create_ticket("puppy-1", "child-1")
        result = await self.registration.consume_enrollment("puppy-1", b"too short")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(self.voice.calls, 0)
        self.assertEqual(self.service.store.list_samples(), [])

    async def test_concurrent_audio_can_consume_ticket_only_once(self):
        ticket, task = await self.blocked_consumption()
        self.assertIsNone(await self.registration.consume_enrollment("puppy-1", self.pcm))
        self.voice.release.set()
        result = await task
        self.assertEqual(result["state"], "completed")
        self.assertEqual(len(self.service.store.list_samples()), 1)
        self.assertIsNone(await self.registration.consume_enrollment("puppy-1", self.pcm))

    async def test_expiry_during_real_preparation_cannot_commit(self):
        ticket, task = await self.blocked_consumption()
        self.now += 61
        self.voice.release.set()
        self.assertEqual((await task)["state"], "expired")
        self.assertEqual(self.service.store.list_samples(), [])

    async def test_parent_cancel_during_preparation_cannot_commit(self):
        ticket, task = await self.blocked_consumption()
        await self.registration.cancel_ticket(ticket["ticket_id"])
        self.voice.release.set()
        self.assertEqual((await task)["state"], "cancelled")
        self.assertEqual(self.service.store.list_samples(), [])

    async def test_member_revoke_and_recreate_does_not_accept_late_sample(self):
        ticket, task = await self.blocked_consumption()
        await self.registration.revoke_person("child-1")
        self.registration.save_member("child-1", "重建的成员")
        self.voice.release.set()
        self.assertEqual((await task)["state"], "cancelled")
        self.assertEqual(self.service.store.list_samples(), [])

    async def test_audio_abort_during_preparation_cannot_commit(self):
        ticket, task = await self.blocked_consumption()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.voice.release.set()
        await asyncio.sleep(.03)
        self.assertEqual(self.registration.ticket_status(ticket["ticket_id"])["state"], "cancelled")
        self.assertEqual(self.service.store.list_samples(), [])

    async def test_prepare_does_not_write_and_commit_still_requires_parent(self):
        observation = self.service.prepare_enrollment("voice", self.pcm)
        self.assertEqual(self.service.store.list_samples(), [])
        with self.assertRaises(PermissionError):
            self.service.commit_enrollment("child-1", observation)
        self.assertEqual(self.service.store.list_samples(), [])

    async def test_upload_cannot_commit_to_recreated_member(self):
        generation = self.registration.member("child-1")["generation"]
        await self.registration.revoke_person("child-1")
        self.registration.save_member("child-1", "新的成员")
        with self.assertRaises(HTTPException) as error:
            await self.registration.enroll_data("child-1", "face", jpeg(), generation)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(self.face.calls, 0)


if __name__ == "__main__":
    unittest.main()
