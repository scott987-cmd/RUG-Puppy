"""Isolated file-transport and crash tests; never use a real iCloud directory."""

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption

from companion.remote import (RemoteBridge, atomic_write, json_bytes, mirror_files,
                              mirror_once, parse_json, sign_payload, verify_payload,
                              macos_helper_path, request_macos_downloads)


def keys():
    key = Ed25519PrivateKey.generate()
    return {
        "private_key": base64.b64encode(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode(),
        "public_key": base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode(),
    }


class RemoteTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-remote-test-")
        self.root = Path(self.temp.name)
        self.local = self.root / "local"
        self.cloud = self.root / "PuppyCompanion"
        self.cloud.mkdir()
        self.parent_keys = keys()
        self.server_keys = keys()
        self.now = 1900000000
        self.calls = []

        async def execute(command):
            self.calls.append(command)
            return {"ok": True, "device_receipt": {"ok": True}}

        self.execute = execute
        self.bridge = self.new_bridge()

    def new_bridge(self, **kwargs):
        return RemoteBridge(self.local, "home-mac", {"puppy-1"},
                            {"parent-phone": self.parent_keys["public_key"]},
                            self.server_keys["private_key"], kwargs.pop("executor", self.execute),
                            clock=lambda: self.now, **kwargs)

    def tearDown(self):
        self.bridge.close()
        self.temp.cleanup()

    def command(self, **changes):
        result = {"version": 1, "request_id": str(uuid.uuid4()), "source_id": "parent-phone",
                  "server_id": "home-mac", "device_id": "puppy-1", "kind": "status", "params": {},
                  "created_at": self.now, "expires_at": self.now + 120}
        result.update(changes)
        return result

    def write_command(self, command, *, cloud=False, signer=None):
        data = json_bytes(sign_payload(command, signer or self.parent_keys["private_key"]))
        path = (self.cloud / "commands" if cloud else self.local / "inbox") / (command["request_id"] + ".json")
        atomic_write(path, data)
        return path

    def receipt(self, command):
        path = self.local / "outbox/receipts" / (command["request_id"] + ".json")
        return verify_payload(parse_json(path.read_bytes()), self.server_keys["public_key"])

    async def test_end_to_end_mirror_consumes_actual_executor_and_signed_receipt(self):
        command = self.command(kind="patrol_once")
        self.write_command(command, cloud=True)
        first = await mirror_once(self.local, self.cloud)
        self.assertEqual(first["pulled"], 1)
        self.assertEqual(first["errors"], [])
        result = await self.bridge.process_once()
        self.assertEqual(result[0]["state"], "succeeded")
        self.assertEqual(self.calls[0].kind, "patrol_once")
        self.assertEqual(self.calls[0].params, {})
        second = await mirror_once(self.local, self.cloud)
        self.assertGreaterEqual(second["pushed"], 1)
        remote = parse_json((self.cloud / "receipts" / (command["request_id"] + ".json")).read_bytes())
        self.assertEqual(verify_payload(remote, self.server_keys["public_key"])["result"]["device_receipt"]["ok"], True)
        self.assertTrue((self.cloud / "commands" / (command["request_id"] + ".json")).exists())
        self.assertEqual(await self.bridge.process_once(), [])

    async def test_invalid_commands_are_rejected_without_execution(self):
        cases = [
            ({"expires_at": self.now}, "expired"),
            ({"expires_at": self.now + 301}, "invalid_ttl"),
            ({"created_at": self.now + 60, "expires_at": self.now + 180}, "future_command"),
            ({"device_id": "other-device"}, "wrong_target"),
            ({"server_id": "another-mac"}, "wrong_target"),
            ({"kind": "set_volume", "params": {"volume": 60}}, "unsupported_command"),
            ({"kind": "action", "params": {"name": "laser"}}, "unapproved_action"),
            ({"kind": "patrol_once", "params": {"drive": True}}, "unexpected_parameters"),
            ({"created_at": True}, "invalid_timestamp"),
        ]
        for changes, error in cases:
            with self.subTest(error=error, changes=changes):
                command = self.command(**changes)
                # expired must still have a valid original lifetime.
                if error == "expired":
                    command["created_at"] = self.now - 120
                self.write_command(command)
                receipt = (await self.bridge.process_once())[0]
                self.assertEqual(receipt["state"], "rejected")
                self.assertEqual(receipt["error"], error)
        self.assertEqual(self.calls, [])

    async def test_signature_tampering_revocation_and_uuid_conflict(self):
        command = self.command()
        path = self.write_command(command)
        envelope = parse_json(path.read_bytes())
        changed = {**command, "kind": "action", "params": {"name": "wave"}}
        envelope["payload"] = base64.b64encode(json_bytes(changed)).decode()
        atomic_write(path, json_bytes(envelope))
        self.assertEqual((await self.bridge.process_once())[0]["error"], "invalid_signature")
        self.assertEqual(self.calls, [])
        self.write_command(command)
        self.assertEqual((await self.bridge.process_once())[0]["state"], "succeeded")
        self.write_command(changed)
        self.assertEqual((await self.bridge.process_once())[0]["error"], "request_id_conflict")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.receipt(command)["kind"], "status")
        self.bridge.keys.clear()
        self.write_command(self.command())
        self.assertEqual((await self.bridge.process_once())[0]["error"], "unpaired_source")

    async def test_dedup_survives_restart_and_expiry(self):
        command = self.command(kind="action", params={"name": "wave"})
        self.write_command(command)
        await self.bridge.process_once()
        first = self.receipt(command)
        self.bridge.close()
        self.now += 1000
        self.bridge = self.new_bridge()
        self.write_command(command)
        result = (await self.bridge.process_once())[0]
        self.assertEqual(result, first)
        self.assertEqual(len(self.calls), 1)

    async def test_crash_after_started_never_replays_action(self):
        command = self.command(kind="action", params={"name": "wave"})
        self.write_command(command)
        self.bridge.close()
        key_file = self.root / "crash-keys.json"
        atomic_write(key_file, json_bytes({"parent": self.parent_keys["public_key"], "server": self.server_keys["private_key"]}))
        script = '''
import asyncio,json,os,sys
from pathlib import Path
from companion.remote import RemoteBridge
root=Path(sys.argv[1]); key=json.loads(Path(sys.argv[2]).read_text())
async def execute(command):
    (root / 'effect-observed').write_text(command.request_id)
    os._exit(17)
bridge=RemoteBridge(root,'home-mac',{'puppy-1'},{'parent-phone':key['parent']},key['server'],execute,clock=lambda:1900000000)
asyncio.run(bridge.process_once())
'''
        completed = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, str(self.local), str(key_file)],
                                             capture_output=True, timeout=10)
        self.assertEqual(completed.returncode, 17, completed.stderr.decode())
        self.assertEqual((self.local / "effect-observed").read_text(), command["request_id"])
        self.bridge = self.new_bridge()
        result = (await self.bridge.process_once())[0]
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(self.calls, [])

    async def test_accepted_publish_failure_recovers_to_terminal_rejected(self):
        command = self.command()
        self.write_command(command)
        with patch.object(self.bridge, "_publish", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await self.bridge.process_once()
        self.assertEqual(self.calls, [])
        self.bridge.close()
        self.bridge = self.new_bridge()
        result = (await self.bridge.process_once())[0]
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.calls, [])

    async def test_completed_publish_failure_does_not_repeat_device_effect(self):
        command = self.command()
        self.write_command(command)
        publish = self.bridge._publish

        def fail_completion(receipt):
            if receipt["state"] == "succeeded":
                raise OSError("disk full while publishing")
            publish(receipt)

        with patch.object(self.bridge, "_publish", side_effect=fail_completion):
            with self.assertRaises(OSError):
                await self.bridge.process_once()
        self.assertEqual(len(self.calls), 1)
        self.bridge.close()
        self.bridge = self.new_bridge()
        self.assertEqual((await self.bridge.process_once())[0]["state"], "succeeded")
        self.assertEqual(len(self.calls), 1)

    async def test_timeout_and_missing_device_receipt_are_unknown_not_success(self):
        async def hang(command):
            self.calls.append(command)
            await asyncio.Event().wait()

        self.bridge.close()
        self.bridge = self.new_bridge(executor=hang, executor_timeout=0.01)
        command = self.command()
        self.write_command(command)
        self.assertEqual((await self.bridge.process_once())[0]["state"], "unknown")
        self.write_command(command)
        self.assertEqual((await self.bridge.process_once())[0]["state"], "unknown")
        self.assertEqual(len(self.calls), 1)

        async def missing(command):
            return {"submitted": True}

        self.bridge.executor = missing
        self.write_command(self.command())
        self.assertEqual((await self.bridge.process_once())[0]["state"], "unknown")

    async def test_single_consumer_and_concurrent_scans(self):
        with self.assertRaisesRegex(RuntimeError, "already_running"):
            self.new_bridge()
        self.write_command(self.command())
        await asyncio.gather(self.bridge.process_once(), self.bridge.process_once())
        self.assertEqual(len(self.calls), 1)

    async def test_batches_do_not_starve_the_33rd_or_257th_command(self):
        for _ in range(257):
            self.write_command(self.command(), cloud=True)
        one = mirror_files(self.local, self.cloud)
        two = mirror_files(self.local, self.cloud)
        self.assertEqual(one["pulled"], 256)
        self.assertEqual(two["pulled"], 1)
        for _ in range(9):
            await self.bridge.process_once()
        self.assertEqual(len(self.calls), 257)
        self.assertEqual(await self.bridge.process_once(), [])
        self.assertEqual(mirror_files(self.local, self.cloud)["pulled"], 0)

    async def test_outbound_cursor_reaches_513th_file_and_smaller_status_replaces_old(self):
        for index in range(513):
            atomic_write(self.local / "outbox/receipts" / f"{index:04d}.json", json_bytes({"number": index}))
        self.assertEqual(mirror_files(self.local, self.cloud)["pushed"], 512)
        self.assertEqual(mirror_files(self.local, self.cloud)["pushed"], 1)
        self.assertTrue((self.cloud / "receipts/0512.json").exists())
        self.bridge.publish_status("puppy-1", {"online": True, "volume": 0, "pet_name": "long-pet-name"})
        mirror_files(self.local, self.cloud)
        self.bridge.publish_status("puppy-1", {"online": False})
        self.assertEqual(mirror_files(self.local, self.cloud)["errors"], [])
        status = verify_payload(parse_json((self.cloud / "status/puppy-1.json").read_bytes()), self.server_keys["public_key"])
        self.assertEqual(status["status"], {"online": False})

    async def test_executor_is_cancelled_at_expiry_before_further_photo_retries(self):
        cancelled = asyncio.Event()

        async def delayed_photo(command):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.bridge.executor = delayed_photo
        self.bridge.clock = lambda: self.now + 119.98
        self.write_command(self.command(kind="photo"))
        result = await self.bridge.process_once()
        self.assertEqual(result[0]["state"], "unknown")
        self.assertTrue(cancelled.is_set())

    async def test_placeholder_read_failure_and_conflict_preserve_local_data(self):
        command = self.command()
        self.write_command(command, cloud=True)
        mirror_files(self.local, self.cloud)
        original = (self.local / "inbox" / (command["request_id"] + ".json")).read_bytes()
        (self.cloud / "commands" / ".pending.json.icloud").write_bytes(b"")
        (self.cloud / "commands" / (str(uuid.uuid4()) + ".json")).write_bytes(b'{"partial":')
        self.write_command({**command, "kind": "stop"}, cloud=True)
        result = mirror_files(self.local, self.cloud)
        self.assertEqual(result["placeholders"], 1)
        self.assertTrue(any(value.startswith("command_unavailable") for value in result["errors"]))
        self.assertTrue(any(value.startswith("immutable_command_conflict") for value in result["errors"]))
        self.assertEqual((self.local / "inbox" / (command["request_id"] + ".json")).read_bytes(), original)
        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            # No local or remote file is deleted even when listing fails.
            with self.assertRaises(PermissionError):
                mirror_files(self.local, self.cloud)
        self.assertEqual((self.local / "inbox" / (command["request_id"] + ".json")).read_bytes(), original)

    async def test_ordinary_outdated_json_requests_download_before_reading_or_execution(self):
        command = self.command()
        path = self.write_command(command, cloud=True)
        states = {str(path): {"state": "requested"}}
        with patch("companion.remote.request_macos_downloads", return_value=states) as request:
            result = mirror_files(self.local, self.cloud)
        request.assert_called_once_with(self.local, [path])
        self.assertEqual(result["placeholders"], 1)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["pulled"], 0)
        self.assertEqual(await self.bridge.process_once(), [])
        states[str(path)]["state"] = "current"
        with patch("companion.remote.request_macos_downloads", return_value=states):
            self.assertEqual(mirror_files(self.local, self.cloud)["pulled"], 1)
        self.assertEqual((await self.bridge.process_once())[0]["state"], "succeeded")

    async def test_old_style_placeholder_is_submitted_to_native_downloader(self):
        command = self.command()
        directory = self.cloud / "commands"
        directory.mkdir()
        placeholder = directory / ("." + command["request_id"] + ".json.icloud")
        placeholder.write_bytes(b"")
        with patch("companion.remote.request_macos_downloads", return_value={str(placeholder): {"state": "requested"}}) as request:
            result = mirror_files(self.local, self.cloud)
        request.assert_called_once_with(self.local, [placeholder])
        self.assertEqual(result["placeholders"], 1)
        self.assertEqual(result["pulled"], 0)
        self.assertEqual(self.calls, [])

    async def test_file_provider_error_retains_errno_without_command_content(self):
        import companion.remote as remote
        command = self.command()
        path = self.write_command(command, cloud=True)
        real_read = remote.read_regular

        def unavailable(source, limit):
            if source == path:
                raise BlockingIOError(35, "private command must not enter diagnostics")
            return real_read(source, limit)

        with patch("companion.remote.read_regular", side_effect=unavailable):
            result = mirror_files(self.local, self.cloud)
        self.assertEqual(result["file_errors"], [{"request_id": command["request_id"], "type": "BlockingIOError", "errno": 35}])
        self.assertNotIn("private command", json.dumps(result))
        self.assertFalse((self.local / "inbox" / path.name).exists())

    @unittest.skipUnless(sys.platform == "darwin", "native downloader is macOS only")
    async def test_native_downloader_hang_is_bounded(self):
        helper = macos_helper_path(self.local)
        helper.parent.mkdir(parents=True)
        helper.write_text("#!/bin/sh\nexec /bin/sleep 60\n")
        helper.chmod(0o700)
        source = self.cloud / "commands" / (str(uuid.uuid4()) + ".json")
        result = request_macos_downloads(self.local, [source], timeout=0.02)
        self.assertEqual(result[str(source)]["state"], "failed")

    async def test_symlink_and_fifo_are_never_read_as_commands(self):
        external = self.root / "external.json"
        external.write_text("do not read")
        request_id = str(uuid.uuid4())
        (self.local / "inbox" / (request_id + ".json")).symlink_to(external)
        result = await self.bridge.process_once()
        self.assertEqual(result[0]["state"], "rejected")
        self.assertEqual(external.read_text(), "do not read")
        fifo = self.local / "inbox" / (str(uuid.uuid4()) + ".json")
        os.mkfifo(fifo)
        self.assertEqual((await self.bridge.process_once())[0]["state"], "rejected")
        self.assertEqual(self.calls, [])

    async def test_mirror_timeout_preserves_last_success_and_kills_worker(self):
        atomic_write(self.local / "mirror-state.json", json_bytes({"last_success_at": 123}))
        real_create = asyncio.create_subprocess_exec
        processes = []

        async def stalled_worker(*args, **kwargs):
            process = await real_create(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
            processes.append(process)
            return process

        with patch("companion.remote.asyncio.create_subprocess_exec", side_effect=stalled_worker):
            result = await mirror_once(self.local, self.cloud, timeout=0.02)
        self.assertEqual(result["errors"], ["mirror_timeout"])
        self.assertEqual(result["last_success_at"], 123)
        self.assertIsNotNone(processes[0].returncode)

    async def test_worker_failure_records_local_stage_without_stderr_contents(self):
        real_create = asyncio.create_subprocess_exec

        async def denied_worker(*args, **kwargs):
            return await real_create(sys.executable, "-c",
                "import sys; print('PermissionError: private-command-and-key-data',file=sys.stderr); sys.exit(1)", **kwargs)

        with patch("companion.remote.asyncio.create_subprocess_exec", side_effect=denied_worker):
            result = await mirror_once(self.local, self.cloud)
        self.assertEqual(result["errors"], ["mirror_worker_failed"])
        diagnostic = (self.local / "mirror-worker-diagnostic.json").read_text()
        self.assertEqual(json.loads(diagnostic)["stderr_kind"], "PermissionError")
        self.assertNotIn("private-command", diagnostic)
        self.assertNotIn("stderr_kind", json.dumps(result))
        self.assertFalse((self.cloud / "mirror-worker-diagnostic.json").exists())

    async def test_worker_uses_package_directory_when_caller_cwd_is_different(self):
        original = Path.cwd()
        try:
            os.chdir(self.root)
            result = await mirror_once(self.local, self.cloud)
        finally:
            os.chdir(original)
        self.assertEqual(result["errors"], [])
        diagnostic = parse_json((self.local / "mirror-worker-diagnostic.json").read_bytes())
        self.assertEqual(diagnostic["stage"], "completed")
        self.assertIn("pid", diagnostic)

    async def test_only_explicit_photo_exports_media_and_status_filters_private_data(self):
        async def photo(command):
            exported = self.bridge.export_photo(command, b"\xff\xd8fixture\xff\xd9")
            return {"ok": True, "media": exported}

        self.bridge.executor = photo
        command = self.command(kind="photo")
        self.write_command(command)
        receipt = (await self.bridge.process_once())[0]
        self.assertEqual(receipt["state"], "succeeded")
        self.assertEqual(receipt["result"]["media"]["sha256"], hashlib.sha256(b"\xff\xd8fixture\xff\xd9").hexdigest())
        self.bridge.publish_status("puppy-1", {"volume": 0, "online": True, "child_voiceprint": "private"})
        status = verify_payload(parse_json((self.local / "outbox/status/puppy-1.json").read_bytes()),
                                self.server_keys["public_key"])
        self.assertEqual(status["status"], {"volume": 0, "online": True})
        self.write_command(self.command(kind="status"))
        self.assertEqual((await self.bridge.process_once())[0]["state"], "unknown")

    async def test_cli_key_permissions_submission_and_verified_receipt(self):
        private_file = self.root / "parent-key.json"
        result = await asyncio.to_thread(subprocess.run,
            [sys.executable, "-m", "companion.remote", "keygen", "--output", str(private_file)],
            capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(private_file.stat().st_mode & 0o777, 0o600)
        key = parse_json(private_file.read_bytes())
        self.bridge.keys["cli-parent"] = key["public_key"]
        self.now = int(__import__("time").time())
        submit = await asyncio.to_thread(subprocess.run,
            [sys.executable, "-m", "companion.remote", "submit", "--cloud-root", str(self.cloud),
             "--key-file", str(private_file), "--source-id", "cli-parent", "--server-id", "home-mac",
             "--device-id", "puppy-1", "--kind", "status"], capture_output=True, timeout=10)
        self.assertEqual(submit.returncode, 0, submit.stderr.decode())
        request_id = parse_json(submit.stdout)["request_id"]
        self.assertEqual(parse_json(submit.stdout)["state"], "submitted")
        mirror_files(self.local, self.cloud)
        await self.bridge.process_once()
        server_public = self.root / "server-public.json"
        server_public.write_text(json.dumps({"public_key": self.server_keys["public_key"]}))
        verify = await asyncio.to_thread(subprocess.run,
            [sys.executable, "-m", "companion.remote", "verify", "--public-key-file", str(server_public),
             "--file", str(self.local / "outbox/receipts" / (request_id + ".json"))], capture_output=True, timeout=10)
        self.assertEqual(verify.returncode, 0, verify.stderr.decode())
        self.assertEqual(parse_json(verify.stdout)["state"], "succeeded")


if __name__ == "__main__":
    unittest.main()
