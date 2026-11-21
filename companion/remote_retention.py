"""Preview or remove expired server-owned iCloud media.

Only signed-command results already committed in the private server ledger can
identify media. The original gateway photo library and immutable receipts are
left intact. Manual apply refuses a running RemoteBridge consumer; the opted-in
mirror worker prunes before its own outbound media scan.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
import uuid

from .remote import MAX_MEDIA_BYTES, read_regular


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _roots(local_root: Path, cloud_root: Path) -> tuple[Path, Path]:
    local_root, cloud_root = Path(local_root), Path(cloud_root)
    if (not local_root.is_dir() or local_root.is_symlink()
            or not cloud_root.is_dir() or cloud_root.is_symlink()
            or cloud_root.name != "PuppyCompanion"
            or local_root.resolve() == cloud_root.resolve()
            or cloud_root.resolve() in local_root.resolve().parents):
        raise ValueError("invalid_retention_roots")
    for root in (local_root / "outbox/media", cloud_root / "media"):
        if not root.is_dir() or root.is_symlink():
            raise ValueError("invalid_media_directory")
    return local_root, cloud_root


def _matching_file(path: Path, expected_size: int, expected_hash: str) -> tuple[bool, str]:
    if path.is_symlink():
        return False, "symlink"
    try:
        data = read_regular(path, MAX_MEDIA_BYTES)
    except FileNotFoundError:
        return False, "missing"
    except (OSError, ValueError):
        return False, "unreadable"
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_hash:
        return False, "mismatch"
    return True, "matched"


def plan_media(local_root: Path, cloud_root: Path, retention_days: int, *, now: float | None = None) -> list[dict]:
    """Read only. A plan is advisory; apply always revalidates bytes and age."""
    if type(retention_days) is not int or not 1 <= retention_days <= 30:
        raise ValueError("retention_days_must_be_1_to_30")
    local_root, cloud_root = _roots(local_root, cloud_root)
    cutoff = (time.time() if now is None else now) - retention_days * 86400
    db_path = local_root / "remote.sqlite3"
    if db_path.is_symlink() or not db_path.is_file():
        raise ValueError("invalid_remote_ledger")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT request_id,receipt FROM commands WHERE state='succeeded'").fetchall()
    finally:
        connection.close()
    planned = []
    for request_id, raw_receipt in rows:
        try:
            if str(uuid.UUID(request_id)) != request_id:
                continue
            receipt = json.loads(raw_receipt)
            media = receipt["result"]["media"]
            age = receipt["updated_at"]
            name = request_id + ".jpg"
            if (receipt["request_id"] != request_id or receipt["state"] != "succeeded"
                    or receipt["kind"] not in {"photo", "patrol_once"}
                    or receipt["result"]["ok"] is not True
                    or type(age) is not int or age > cutoff
                    or media["file"] != "media/" + name
                    or type(media["bytes"]) is not int or not 1 <= media["bytes"] <= MAX_MEDIA_BYTES
                    or not isinstance(media["sha256"], str) or not SHA256.fullmatch(media["sha256"])):
                continue
            local_ok, local_state = _matching_file(local_root / "outbox/media" / name,
                                                   media["bytes"], media["sha256"])
            cloud_ok, cloud_state = _matching_file(cloud_root / "media" / name,
                                                   media["bytes"], media["sha256"])
            ready = ((local_ok or local_state == "missing") and (cloud_ok or cloud_state == "missing")
                     and (local_ok or cloud_ok))
            planned.append({"request_id": request_id, "file": "media/" + name,
                            "updated_at": age, "bytes": media["bytes"], "sha256": media["sha256"],
                            "local": local_state, "cloud": cloud_state, "ready": ready})
        except (KeyError, TypeError, ValueError):
            continue
    return planned


def apply_media(local_root: Path, cloud_root: Path, retention_days: int, *, now: float | None = None) -> dict:
    """Delete only revalidated expired copies, local spool first, then iCloud."""
    local_root, cloud_root = _roots(local_root, cloud_root)
    lock_fd = os.open(local_root / "consumer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ValueError("invalid_consumer_lock")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("stop_gateway_before_media_retention") from exc
        return prune_media_in_mirror(local_root, cloud_root, retention_days, now=now)
    finally:
        os.close(lock_fd)


def prune_media_in_mirror(local_root: Path, cloud_root: Path, retention_days: int,
                          *, now: float | None = None) -> dict:
    """Mirror-worker entry point; call only before that worker scans outbound media."""
    local_root, cloud_root = _roots(local_root, cloud_root)
    candidates = plan_media(local_root, cloud_root, retention_days, now=now)
    result = {"eligible": sum(item["ready"] for item in candidates),
              "blocked": sum(not item["ready"] for item in candidates),
              "local_deleted": 0, "cloud_deleted": 0}
    for item in candidates:
        if not item["ready"]:
            continue
        name = item["request_id"] + ".jpg"
        # Re-check each file immediately before removal. Keeping the local
        # source removed first prevents the next mirror cycle re-uploading.
        for root, key in ((local_root / "outbox/media", "local_deleted"),
                          (cloud_root / "media", "cloud_deleted")):
            path = root / name
            matched, state = _matching_file(path, item["bytes"], item["sha256"])
            if matched:
                path.unlink()
                if key == "local_deleted":
                    directory_fd = os.open(root, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                result[key] += 1
            elif state != "missing":
                raise RuntimeError("media_changed_during_retention")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="delete verified media; requires stopped gateway")
    args = parser.parse_args()
    result = (apply_media(args.local_root, args.cloud_root, args.days) if args.apply
              else {"candidates": [{key: value for key, value in item.items() if key != "sha256"}
                                    for item in plan_media(args.local_root, args.cloud_root, args.days)]})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
