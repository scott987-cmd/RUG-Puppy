"""Signed iCloud Drive file transport; no iCloud access occurs in RemoteBridge.

The phone writes ``commands/<request UUID>.json``. A separate, time-bounded
mirror process copies commands into local ``inbox`` and copies local ``outbox``
back to the cloud. Cloud files are never deleted or treated as a lock service.
Exactly one designated server consumes commands; its SQLite ledger is local.

Wire envelope: {"payload": base64(UTF-8 JSON), "signature": base64(Ed25519)}.
Sign the *decoded payload bytes*, not re-serialized JSON. Keys are raw 32-byte
Ed25519 keys encoded as base64; keep private keys and the ledger out of iCloud.
``cryptography`` is needed only for the consumer/client, not the mirror worker.
"""

import argparse
import asyncio
import base64
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable


MAX_COMMAND_BYTES = 16384
MAX_RESULT_BYTES = 65536
MAX_MEDIA_BYTES = 512000
MAX_TTL_SECONDS = 300
SAFE_ACTIONS = frozenset({"wave", "sit", "stretch", "hug", "stop"})
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
COMMAND_FIELDS = {"version", "request_id", "source_id", "server_id", "device_id",
                  "kind", "params", "created_at", "expires_at"}


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_json(data: bytes):
    return json.loads(data, object_pairs_hook=_unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid_number")))


def _base64(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("invalid_base64")
    return base64.b64decode(value, validate=True)


def sign_payload(payload: dict, private_key: str) -> dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    body = json_bytes(payload)
    signature = Ed25519PrivateKey.from_private_bytes(_base64(private_key)).sign(body)
    return {"payload": base64.b64encode(body).decode("ascii"),
            "signature": base64.b64encode(signature).decode("ascii")}


def verify_payload(envelope: dict, public_key: str) -> dict:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
        raise ValueError("invalid_envelope")
    body = _base64(envelope["payload"])
    try:
        Ed25519PublicKey.from_public_bytes(_base64(public_key)).verify(
            _base64(envelope["signature"]), body)
    except (InvalidSignature, TypeError) as exc:
        raise ValueError("invalid_signature") from exc
    value = parse_json(body)
    if not isinstance(value, dict):
        raise ValueError("invalid_payload")
    return value


def atomic_write(path: Path, data: bytes):
    """Same-directory atomic replacement, durable before publishing completion."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".puppy-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_regular(path: Path, limit: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("invalid_file")
        data = source.read(limit + 1)
        if len(data) > limit:
            raise ValueError("file_too_large")
        return data


def macos_helper_path(local_root: Path) -> Path:
    source = Path(__file__).with_name("icloud_download.swift")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return Path(local_root) / "helpers" / ("icloud-download-" + digest)


def prepare_macos_helper(local_root: Path) -> Path:
    """Explicit installation step, separate from the eight-second mirror loop."""
    if sys.platform != "darwin":
        raise ValueError("macos_required")
    output = macos_helper_path(local_root)
    if output.is_file():
        return output
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".compile-", dir=output.parent)
    os.close(descriptor)
    try:
        subprocess.run(["/usr/bin/swiftc", "-O", "-module-cache-path", str(output.parent / "ModuleCache"),
                        str(Path(__file__).with_name("icloud_download.swift")),
                        "-o", temporary], check=True, capture_output=True, timeout=60)
        os.chmod(temporary, 0o700)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def request_macos_downloads(local_root: Path, files: list[Path], timeout: float = 3) -> dict[str, dict] | None:
    """Inspect Foundation freshness and request hydration without reading payloads.

    `.icloud` placeholders and ordinary but outdated JSON both need hydration.
    The native helper is a bounded child process; File Provider cannot stall the
    gateway event loop. None means the helper is not installed on this host.
    """
    if sys.platform != "darwin" or not files:
        return None
    helper = macos_helper_path(local_root)
    if not helper.is_file():
        return None
    try:
        result = subprocess.run([str(helper)], input=json_bytes({"files": [str(path) for path in files]}),
                                capture_output=True, timeout=timeout, check=True)
        items = parse_json(result.stdout)
        if not isinstance(items, list) or len(items) != len(files):
            raise ValueError("invalid_helper_output")
        states = {}
        for index, item in enumerate(items):
            if (not isinstance(item, dict) or item.get("index") != index
                    or item.get("state") not in {"current", "local", "requested", "failed", "rejected"}):
                raise ValueError("invalid_helper_output")
            states[str(files[index])] = item
        return states
    except (OSError, ValueError, subprocess.SubprocessError):
        return {str(path): {"state": "failed", "errorDomain": "PuppyDownloadHelper"} for path in files}


@dataclass(frozen=True)
class Command:
    request_id: str
    source_id: str
    server_id: str
    device_id: str
    kind: str
    params: dict
    created_at: int
    expires_at: int


class RemoteBridge:
    """Local consumer. Executor must report actual device receipt as ``ok``.

    Supported kinds: status, photo, stop, action {name}, patrol_once {}.
    Patrol means one fixed-position photo; it never drives or changes volume.
    ``parent_public_keys`` is consulted per request, so removing an entry revokes
    that phone. Executor result is a bounded public dict, not private memories.
    On timeout, cancellation, exception, or crash, commands become ``unknown``
    and will never be automatically replayed, even if they may have succeeded.
    """

    def __init__(self, local_root: Path, server_id: str, device_ids: set[str],
                 parent_public_keys: dict[str, str], server_private_key: str,
                 executor: Callable[[Command], Awaitable[dict]], *,
                 clock: Callable[[], float] = time.time, executor_timeout: float = 100):
        if (not IDENTIFIER.fullmatch(server_id) or not device_ids
                or any(not IDENTIFIER.fullmatch(value) for value in device_ids)):
            raise ValueError("invalid_server_configuration")
        self.root = Path(local_root)
        if "Mobile Documents" in self.root.resolve().parts:
            raise ValueError("local_ledger_must_be_outside_icloud")
        self.server_id = server_id
        self.device_ids = set(device_ids)
        self.keys = parent_public_keys
        self.signing_key = server_private_key
        self.executor = executor
        self.clock = clock
        self.executor_timeout = executor_timeout
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in ("inbox", "processed", "outbox/receipts", "outbox/rejections", "outbox/status", "outbox/media"):
            (self.root / directory).mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock_fd = os.open(self.root / "consumer.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._lock_fd)
            raise RuntimeError("remote_consumer_already_running")
        try:
            # Check signer at startup, before a device command could execute.
            sign_payload({"version": 1}, self.signing_key)
            self.db = sqlite3.connect(self.root / "remote.sqlite3")
            os.chmod(self.root / "remote.sqlite3", 0o600)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS commands (
                request_id TEXT PRIMARY KEY, digest TEXT NOT NULL,
                command TEXT NOT NULL, state TEXT NOT NULL, receipt TEXT NOT NULL)""")
            self.db.commit()
            self._processing = asyncio.Lock()
            self._recover()
        except BaseException:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise

    def close(self):
        if getattr(self, "db", None) is not None:
            self.db.close()
            self.db = None
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _receipt(self, command: Command, state: str, **details):
        return {"version": 1, "request_id": command.request_id,
                "source_id": command.source_id, "server_id": self.server_id,
                "device_id": command.device_id, "kind": command.kind,
                "state": state, "updated_at": int(self.clock()), **details}

    def _publish(self, receipt: dict):
        atomic_write(self.root / "outbox/receipts" / (receipt["request_id"] + ".json"),
                     json_bytes(sign_payload(receipt, self.signing_key)))

    def _finish(self, command: Command, state: str, **details):
        receipt = self._receipt(command, state, **details)
        with self.db:
            self.db.execute("UPDATE commands SET state=?,receipt=? WHERE request_id=?",
                            (state, json_bytes(receipt).decode(), command.request_id))
        # DB commit precedes publication. A publication failure cannot cause replay.
        self._publish(receipt)
        return receipt

    def _recover(self):
        for row in self.db.execute("SELECT * FROM commands").fetchall():
            if row["state"] in {"accepted", "executing"}:
                fields = json.loads(row["command"])
                fields.pop("version")
                state = "unknown" if row["state"] == "executing" else "rejected"
                self._finish(Command(**fields), state, error="server_restarted_before_completion")
            else:
                self._publish(json.loads(row["receipt"]))

    def _decode(self, envelope: dict, request_id: str):
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
            raise ValueError("invalid_envelope")
        untrusted = parse_json(_base64(envelope["payload"]))
        if not isinstance(untrusted, dict) or not isinstance(untrusted.get("source_id"), str):
            raise ValueError("invalid_source")
        public_key = self.keys.get(untrusted["source_id"])
        if not public_key:
            raise ValueError("unpaired_source")
        fields = verify_payload(envelope, public_key)
        if set(fields) != COMMAND_FIELDS or type(fields.get("version")) is not int or fields["version"] != 1:
            raise ValueError("unsupported_schema")
        if fields["request_id"] != request_id:
            raise ValueError("filename_mismatch")
        for key in ("source_id", "server_id", "device_id"):
            if not isinstance(fields[key], str) or not IDENTIFIER.fullmatch(fields[key]):
                raise ValueError("invalid_identifier")
        if fields["server_id"] != self.server_id or fields["device_id"] not in self.device_ids:
            raise ValueError("wrong_target")
        fields.pop("version")
        command = Command(**fields)
        return command

    def _validate_execution(self, command: Command):
        if type(command.created_at) is not int or type(command.expires_at) is not int:
            raise ValueError("invalid_timestamp")
        if not 0 < command.expires_at - command.created_at <= MAX_TTL_SECONDS:
            raise ValueError("invalid_ttl")
        if command.created_at > self.clock() + 30:
            raise ValueError("future_command")
        if command.expires_at <= self.clock():
            raise ValueError("expired")
        if not isinstance(command.params, dict) or not isinstance(command.kind, str):
            raise ValueError("invalid_command")
        if command.kind == "action":
            if set(command.params) != {"name"} or command.params["name"] not in SAFE_ACTIONS:
                raise ValueError("unapproved_action")
        elif command.kind in {"status", "photo", "stop", "patrol_once"}:
            if command.params:
                raise ValueError("unexpected_parameters")
        else:
            raise ValueError("unsupported_command")

    def _reject(self, request_id: str, digest: str, error: str):
        result = {"version": 1, "request_id": request_id, "server_id": self.server_id,
                  "state": "rejected", "error": error, "updated_at": int(self.clock())}
        # An invalid/conflicting file must never overwrite a genuine completion.
        path = self.root / "outbox/rejections" / f"{request_id}-{digest[:16]}.json"
        atomic_write(path, json_bytes(sign_payload(result, self.signing_key)))
        return result

    async def process_once(self, limit: int = 32) -> list[dict]:
        results = []
        async with self._processing:
            for path in sorted((self.root / "inbox").glob("*.json")):
                if len(results) >= limit:
                    break
                if path.is_dir():
                    continue
                try:
                    if str(uuid.UUID(path.stem)) != path.stem:
                        continue
                except ValueError:
                    continue
                try:
                    results.append(await self._consume_file(path))
                except asyncio.CancelledError:
                    raise
                # Archiving is local only. Failure leaves the file for the ledger
                # to handle next time, never causing another device execution.
                try:
                    data = read_regular(path, MAX_COMMAND_BYTES)
                    digest = hashlib.sha256(data).hexdigest()
                except (OSError, ValueError):
                    digest = "unreadable"
                archive = self.root / "processed" / path.stem / (digest + ".json")
                archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.replace(path, archive)
        return results

    async def _consume_file(self, path: Path) -> dict:
        digest = "unreadable"
        try:
            data = read_regular(path, MAX_COMMAND_BYTES)
            digest = hashlib.sha256(data).hexdigest()
            command = self._decode(parse_json(data), path.stem)
            previous = self.db.execute("SELECT * FROM commands WHERE request_id=?",
                                       (command.request_id,)).fetchone()
            if previous:
                if previous["digest"] != digest:
                    raise ValueError("request_id_conflict")
                if previous["state"] in {"accepted", "executing"}:
                    state = "unknown" if previous["state"] == "executing" else "rejected"
                    return self._finish(command, state, error="previous_processing_interrupted")
                receipt = json.loads(previous["receipt"])
                self._publish(receipt)
                return receipt
            self._validate_execution(command)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return self._reject(path.stem, digest, str(exc)[:120])
        receipt = self._receipt(command, "accepted")
        with self.db:
            self.db.execute("INSERT INTO commands VALUES(?,?,?,?,?)",
                            (command.request_id, digest,
                             json_bytes({"version": 1, **command.__dict__}).decode(),
                             "accepted", json_bytes(receipt).decode()))
        self._publish(receipt)
        # TTL is checked again after durable acceptance, before execution.
        try:
            self._validate_execution(command)
        except ValueError as exc:
            return self._finish(command, "rejected", error=str(exc))
        self._finish(command, "executing")
        try:
            remaining = command.expires_at - self.clock()
            if remaining <= 0:
                return self._finish(command, "rejected", error="expired")
            result = await asyncio.wait_for(self.executor(command), timeout=min(self.executor_timeout, remaining))
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise ValueError("executor_missing_actual_receipt")
            if len(json_bytes(result)) > MAX_RESULT_BYTES:
                raise ValueError("executor_result_too_large")
        except asyncio.CancelledError:
            self._finish(command, "unknown", error="execution_cancelled")
            raise
        except Exception:
            return self._finish(command, "unknown", error="execution_not_confirmed")
        return self._finish(command, "succeeded" if result["ok"] else "failed", result=result)

    def publish_status(self, device_id: str, status: dict):
        """Caller provides only public device state; no conversation/person data."""
        if device_id not in self.device_ids:
            raise ValueError("wrong_target")
        allowed = {"online", "observed_at", "volume", "battery", "mode", "last_seen", "pet_name"}
        public = {key: value for key, value in status.items() if key in allowed}
        payload = {"version": 1, "server_id": self.server_id, "device_id": device_id,
                   "published_at": int(self.clock()), "status": public}
        atomic_write(self.root / "outbox/status" / (device_id + ".json"),
                     json_bytes(sign_payload(payload, self.signing_key)))

    def export_photo(self, command: Command, jpeg: bytes) -> dict:
        """Explicit photo/patrol request only. Caller must validate/decode JPEG."""
        if command.kind not in {"photo", "patrol_once"} or command.device_id not in self.device_ids:
            raise ValueError("photo_export_not_authorized")
        row = self.db.execute("SELECT state,command FROM commands WHERE request_id=?",
                              (command.request_id,)).fetchone()
        if not row or row["state"] != "executing" or json.loads(row["command"]) != {"version": 1, **command.__dict__}:
            raise ValueError("photo_request_not_executing")
        if len(jpeg) > MAX_MEDIA_BYTES or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("invalid_photo")
        name = command.request_id + ".jpg"
        atomic_write(self.root / "outbox/media" / name, jpeg)
        return {"file": "media/" + name, "sha256": hashlib.sha256(jpeg).hexdigest(), "bytes": len(jpeg)}


def mirror_files(local_root: Path, cloud_root: Path, progress: Callable[[str], None] | None = None,
                 retention_days: int | None = None) -> dict:
    """Worker only: copy commands, push outbound files, optionally prune owned media.

    Require a preselected existing PuppyCompanion directory. Do not auto-select
    the user's iCloud account or reuse another project's production queue.
    """
    local_root, cloud_root = Path(local_root), Path(cloud_root)
    progress = progress or (lambda stage: None)
    progress("check_cloud_root")
    if cloud_root.name != "PuppyCompanion" or not cloud_root.is_dir() or cloud_root.is_symlink():
        raise ValueError("select_existing_PuppyCompanion_directory")
    progress("check_spool_location")
    if local_root.resolve() == cloud_root.resolve() or cloud_root.resolve() in local_root.resolve().parents:
        raise ValueError("local_spool_must_be_outside_cloud")
    result = {"pulled": 0, "pushed": 0, "placeholders": 0, "errors": []}
    cursor_file = local_root / "mirror-cursors.json"
    try:
        cursors = parse_json(read_regular(cursor_file, MAX_RESULT_BYTES))
    except (OSError, ValueError):
        cursors = {}

    def batch(entries, key, limit):
        ordered = sorted(entries, key=lambda path: path.name)
        after = [path for path in ordered if path.name > cursors.get(key, "")]
        before = [path for path in ordered if path.name <= cursors.get(key, "")]
        for path in (after + before)[:limit]:
            # Advance before a potentially blocking provider read. A hung file
            # can be retried next cycle without starving every file after it.
            cursors[key] = path.name
            atomic_write(cursor_file, json_bytes(cursors))
            yield path

    commands = cloud_root / "commands"
    progress("list_cloud_commands")
    if commands.is_symlink():
        raise ValueError("commands_directory_symlink")
    try:
        entries = list(commands.iterdir())
    except FileNotFoundError:
        entries = []
    except OSError:
        entries = []
        result["errors"].append("commands_unreadable")
    candidates = list(batch(entries, "commands", 256))
    hydration_candidates = []
    for source in candidates:
        name = source.name[1:-7] if source.name.startswith(".") and source.name.endswith(".icloud") else source.name
        if Path(name).suffix == ".json":
            try:
                if str(uuid.UUID(Path(name).stem)) == Path(name).stem:
                    hydration_candidates.append(source)
            except ValueError:
                pass
    progress("request_cloud_downloads")
    hydration = request_macos_downloads(local_root, hydration_candidates)
    result["download_helper"] = ("unused" if not hydration_candidates else
                                 "ready" if hydration is not None else "unavailable")
    for source in candidates:
        freshness = hydration.get(str(source)) if hydration is not None else None
        if freshness and freshness["state"] == "requested":
            result["placeholders"] += 1
            continue
        if freshness and freshness["state"] in {"failed", "rejected"}:
            result["errors"].append("download_unavailable:" + source.name)
            result.setdefault("file_errors", []).append({"file": source.name,
                "type": freshness.get("errorDomain"), "errno": freshness.get("errorCode")})
            continue
        if source.name.endswith(".icloud"):
            result["placeholders"] += 1
            continue
        if source.suffix != ".json" or source.name.startswith("."):
            continue
        try:
            if str(uuid.UUID(source.stem)) != source.stem:
                continue
            progress("read_cloud_command")
            data = read_regular(source, MAX_COMMAND_BYTES)
            # Leave partial JSON in the cloud for a later completed download.
            parse_json(data)
            destination = local_root / "inbox" / source.name
            archived = local_root / "processed" / source.stem / (hashlib.sha256(data).hexdigest() + ".json")
            if archived.exists():
                continue
            if destination.exists():
                if read_regular(destination, MAX_COMMAND_BYTES) != data:
                    result["errors"].append("immutable_command_conflict:" + source.stem)
                continue
            progress("write_local_inbox")
            atomic_write(destination, data)
            result["pulled"] += 1
        except (OSError, ValueError) as exc:
            result["errors"].append("command_unavailable:" + source.stem)
            result.setdefault("file_errors", []).append({"request_id": source.stem,
                "type": type(exc).__name__, "errno": getattr(exc, "errno", None)})
    for directory in ("receipts", "rejections", "status", "observations", "media"):
        if directory == "media" and retention_days is not None:
            # Sync small receipts/status first; prune before scanning media so
            # a successful deletion cannot be immediately re-uploaded.
            try:
                from .remote_retention import prune_media_in_mirror
                progress("prune_expired_media")
                result["retention"] = prune_media_in_mirror(local_root, cloud_root, retention_days)
            except (OSError, ValueError, RuntimeError, sqlite3.Error):
                result["errors"].append("retention_unavailable")
                continue
        progress("list_outbound_" + directory)
        source_dir = local_root / "outbox" / directory
        if source_dir.is_symlink() or (cloud_root / directory).is_symlink():
            result["errors"].append("directory_symlink:" + directory)
            continue
        if not source_dir.exists():
            continue
        for source in batch(source_dir.iterdir(), directory, 512):
            if source.name.startswith(".") or source.suffix not in {".json", ".jpg"}:
                continue
            try:
                progress("read_local_" + directory)
                data = read_regular(source, MAX_MEDIA_BYTES if directory == "media" else MAX_RESULT_BYTES * 2)
                destination = cloud_root / directory / source.name
                progress("read_cloud_" + directory)
                if destination.exists() and read_regular(destination, MAX_MEDIA_BYTES if directory == "media" else MAX_RESULT_BYTES * 2) == data:
                    continue
                progress("write_cloud_" + directory)
                atomic_write(destination, data)
                result["pushed"] += 1
            except (OSError, ValueError):
                result["errors"].append("outbound_unavailable:" + directory + "/" + source.name)
    progress("completed")
    return result


async def mirror_once(local_root: Path, cloud_root: Path, timeout: float = 8,
                      retention_days: int | None = None) -> dict:
    """Isolate potentially hung File Provider calls in a killable subprocess."""
    diagnostic_path = Path(local_root) / "mirror-worker-diagnostic.json"
    worker_args = (["--retention-days", str(retention_days)] if retention_days is not None else [])
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "companion.remote", "mirror", "--local-root", str(local_root),
        "--cloud-root", str(cloud_root), "--diagnostics", str(diagnostic_path), *worker_args,
        cwd=Path(__file__).resolve().parent.parent,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, stderr = await asyncio.wait_for(process.communicate(), timeout)
        result = parse_json(output) if process.returncode == 0 else {"errors": ["mirror_worker_failed"]}
        if process.returncode != 0:
            try:
                diagnostic = parse_json(read_regular(diagnostic_path, MAX_RESULT_BYTES))
            except (ValueError, OSError):
                diagnostic = {}
            # Never retain a traceback, command body, or exception message. The
            # worker has no private keys; only known error identifiers survive.
            kinds = re.findall(rb"\b(?:ModuleNotFoundError|ImportError|PermissionError|FileNotFoundError|OSError|ValueError|RuntimeError)\b", stderr[-4096:])
            diagnostic.update({"exit_code": process.returncode, "stderr_bytes": len(stderr),
                               "stderr_kind": kinds[-1].decode() if kinds else "unclassified"})
            atomic_write(diagnostic_path, json_bytes(diagnostic))
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        process.kill()
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        result = {"errors": ["mirror_timeout"]}
    except (ValueError, OSError):
        result = {"errors": ["mirror_worker_invalid_output"]}
    state_path = Path(local_root) / "mirror-state.json"
    try:
        previous = parse_json(read_regular(state_path, MAX_RESULT_BYTES))
    except (OSError, ValueError):
        previous = {}
    now = int(time.time())
    state = {**result, "attempted_at": now,
             "last_success_at": now if not result.get("errors") and not result.get("placeholders")
             else previous.get("last_success_at")}
    atomic_write(state_path, json_bytes(state))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    mirror = commands.add_parser("mirror", help="one mirror cycle; normally launched by mirror_once")
    mirror.add_argument("--local-root", type=Path, required=True)
    mirror.add_argument("--cloud-root", type=Path, required=True)
    mirror.add_argument("--diagnostics", type=Path)
    mirror.add_argument("--retention-days", type=int)
    prepare = commands.add_parser("prepare-macos-helper", help="compile the native File Provider download helper locally")
    prepare.add_argument("--local-root", type=Path, required=True)
    download = commands.add_parser("download", help="request iCloud downloads; a request is not completed delivery")
    download.add_argument("--local-root", type=Path, required=True)
    download.add_argument("--file", type=Path, action="append", required=True)
    keygen = commands.add_parser("keygen", help="create a local private/public key JSON (never in iCloud)")
    keygen.add_argument("--output", type=Path, required=True)
    submit = commands.add_parser("submit", help="submit one signed parent command; submission is not completion")
    submit.add_argument("--cloud-root", type=Path, required=True)
    submit.add_argument("--key-file", type=Path, required=True)
    submit.add_argument("--source-id", required=True)
    submit.add_argument("--server-id", required=True)
    submit.add_argument("--device-id", required=True)
    submit.add_argument("--kind", choices=["status", "photo", "patrol_once", "stop", "action"], required=True)
    submit.add_argument("--action", choices=sorted(SAFE_ACTIONS))
    submit.add_argument("--ttl", type=int, default=120)
    verify = commands.add_parser("verify", help="verify receipt/status against a separately trusted server key")
    verify.add_argument("--file", type=Path, required=True)
    verify.add_argument("--public-key-file", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.operation == "mirror":
        def progress(stage):
            if arguments.diagnostics:
                atomic_write(arguments.diagnostics, json_bytes({"stage": stage, "entered_at": time.time(), "pid": os.getpid()}))

        try:
            progress("worker_started")
            result = mirror_files(arguments.local_root, arguments.cloud_root, progress,
                                  arguments.retention_days)
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "errno": getattr(exc, "errno", None)}
            if arguments.diagnostics:
                try:
                    diagnostic = parse_json(read_regular(arguments.diagnostics, MAX_RESULT_BYTES))
                except (ValueError, OSError):
                    diagnostic = {}
                atomic_write(arguments.diagnostics, json_bytes({**diagnostic, **error}))
            print(json.dumps(error), file=sys.stderr)
            raise SystemExit(1)
    elif arguments.operation == "prepare-macos-helper":
        result = {"helper": str(prepare_macos_helper(arguments.local_root.expanduser()))}
    elif arguments.operation == "download":
        result = {"files": request_macos_downloads(arguments.local_root.expanduser(), arguments.file)}
    elif arguments.operation == "keygen":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption

        output = arguments.output.expanduser().resolve()
        if "Mobile Documents" in output.parts or "PuppyCompanion" in output.parts:
            parser.error("private keys must be stored outside iCloud/PuppyCompanion")
        key = Ed25519PrivateKey.generate()
        value = {"private_key": base64.b64encode(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode(),
                 "public_key": base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        result = {"key_file": str(output), "public_key": value["public_key"]}
    elif arguments.operation == "submit":
        root = arguments.cloud_root
        if root.name != "PuppyCompanion" or not root.is_dir() or root.is_symlink() or (root / "commands").is_symlink():
            parser.error("select an existing PuppyCompanion directory")
        if any(not IDENTIFIER.fullmatch(value) for value in (arguments.source_id, arguments.server_id, arguments.device_id)):
            parser.error("invalid source, server, or device identifier")
        if not 1 <= arguments.ttl <= MAX_TTL_SECONDS:
            parser.error("TTL must be between 1 and 300 seconds")
        if (arguments.kind == "action") != (arguments.action is not None):
            parser.error("--action is required only with --kind action")
        key_file = arguments.key_file
        if key_file.stat().st_mode & 0o077:
            parser.error("private key file permissions must be 0600")
        key = parse_json(read_regular(key_file, MAX_COMMAND_BYTES))["private_key"]
        now = int(time.time())
        request_id = str(uuid.uuid4())
        fields = {"version": 1, "request_id": request_id, "source_id": arguments.source_id,
                  "server_id": arguments.server_id, "device_id": arguments.device_id,
                  "kind": arguments.kind, "params": {"name": arguments.action} if arguments.action else {},
                  "created_at": now, "expires_at": now + arguments.ttl}
        atomic_write(root / "commands" / (request_id + ".json"), json_bytes(sign_payload(fields, key)))
        result = {"request_id": request_id, "state": "submitted", "expires_at": fields["expires_at"]}
    else:
        key = parse_json(read_regular(arguments.public_key_file, MAX_COMMAND_BYTES))["public_key"]
        result = verify_payload(parse_json(read_regular(arguments.file, MAX_RESULT_BYTES * 2)), key)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
