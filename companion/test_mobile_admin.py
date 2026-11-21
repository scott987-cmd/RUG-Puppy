"""Mobile administration authentication checks; no real gateway or device."""
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from .mobile_admin import canonical, create_app, valid_target


class MobileAdminTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = Path(self.temp.name) / "remote.json"
        self.key = Ed25519PrivateKey.generate()
        self.write_keys({"iphone-test": base64.b64encode(self.key.public_key().public_bytes_raw()).decode()})
        self.clock = 1_800_000_000
        self.client = TestClient(create_app(self.config, "p" * 32, now=lambda: self.clock))

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def write_keys(self, keys):
        self.config.write_text(json.dumps({"parent_public_keys": keys}))

    def headers(self, method="GET", target="/api/identity/members", body=b"", *, nonce="a" * 32,
                timestamp=None, source="iphone-test"):
        when = str(timestamp if timestamp is not None else self.clock)
        signature = self.key.sign(canonical(method, target, body, when, nonce))
        return {"x-puppy-source": source, "x-puppy-timestamp": when, "x-puppy-nonce": nonce,
                "x-puppy-signature": base64.b64encode(signature).decode()}

    @patch("companion.mobile_admin.forward", return_value=(200, b'{"members":[]}', "application/json"))
    def test_valid_signed_request_forwards_only_to_loopback_with_token(self, forward):
        response = self.client.get("/api/identity/members", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"members": []})
        forward.assert_called_once_with("GET", "/api/identity/members", b"", "", "p" * 32, 8000)
        self.assertEqual(response.headers["cache-control"], "no-store")

    @patch("companion.mobile_admin.forward", return_value=(200, b'{"ok":true}', "application/json"))
    def test_signed_voice_upload_forwards_exact_multipart_bytes(self, forward):
        content_type = "multipart/form-data; boundary=puppy-test"
        body = b"--puppy-test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\n\r\nRIFFtest\r\n--puppy-test--\r\n"
        target = "/api/identity/members/family-test/voice"
        headers = {**self.headers("POST", target, body, nonce="f" * 32), "content-type": content_type}
        response = self.client.post(target, content=body, headers=headers)
        self.assertEqual(response.status_code, 200)
        forward.assert_called_once_with("POST", target, body, content_type, "p" * 32, 8000)

    @patch("companion.mobile_admin.forward")
    def test_replay_stale_tampered_and_revoked_requests_fail(self, forward):
        good = self.headers()
        forward.return_value = (200, b"{}", "application/json")
        self.assertEqual(self.client.get("/api/identity/members", headers=good).status_code, 200)
        self.assertEqual(self.client.get("/api/identity/members", headers=good).status_code, 409)
        self.assertEqual(self.client.post("/api/identity/members", headers=self.headers(nonce="b" * 32)).status_code, 401)
        self.assertEqual(self.client.get("/api/identity/members", headers=self.headers(nonce="c" * 32,
                                                timestamp=self.clock - 91)).status_code, 401)
        self.write_keys({})
        self.assertEqual(self.client.get("/api/identity/members", headers=self.headers(nonce="d" * 32)).status_code, 401)
        self.assertEqual(forward.call_count, 1)

    @patch("companion.mobile_admin.forward", return_value=(200, b"{}", "application/json"))
    def test_replay_survives_service_restart(self, forward):
        signed = self.headers(nonce="f" * 32)
        self.assertEqual(self.client.get("/api/identity/members", headers=signed).status_code, 200)
        restarted = TestClient(create_app(self.config, "p" * 32, now=lambda: self.clock))
        try:
            self.assertEqual(restarted.get("/api/identity/members", headers=signed).status_code, 409)
        finally:
            restarted.close()
        forward.assert_called_once()

    @patch("companion.mobile_admin.forward")
    def test_invalid_route_and_large_body_rejected_before_proxy(self, forward):
        for target in ("/api/../secret", "/api/%2e%2e/secret", "/api//members", "/api/"):
            self.assertFalse(valid_target(target))
        for target in ("/api/%2Fsecret", "/api/%252Fsecret", "/api/%0Asecret"):
            self.assertFalse(valid_target(target))
        self.assertTrue(valid_target("/api/actions/%E5%BC%80%E5%BF%83%E8%88%9E"))
        body = b"x" * (1_048_576 + 1)
        response = self.client.put("/api/identity/members/person", content=body,
            headers=self.headers("PUT", "/api/identity/members/person", body, nonce="e" * 32))
        self.assertEqual(response.status_code, 413)
        forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
