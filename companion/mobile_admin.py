"""TLS-only phone administration bridge for the existing local parent API.

The device keeps its plain WS gateway on port 8000. This separate process
accepts only signed, short-lived requests from registered parent phones and
forwards them over loopback with the existing parent token. Neither the token
nor family data enters iCloud.
"""
import argparse
import asyncio
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import unquote

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import uvicorn

from .remote import read_regular
from .tls import tls_options

MAX_BODY = 1_048_576
MAX_RESPONSE = 2_097_152
MAX_SKEW = 90
SOURCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
NONCE = re.compile(r"^[a-f0-9]{32}$")
METHODS = {"GET", "PUT", "POST", "PATCH", "DELETE"}


def canonical(method: str, target: str, body: bytes, timestamp: str, nonce: str) -> bytes:
    return f"PUPPY-ADMIN-V1\n{method}\n{target}\n{hashlib.sha256(body).hexdigest()}\n{timestamp}\n{nonce}".encode("ascii")


class PhoneAuthorizer:
    def __init__(self, config_file: Path, nonce_file: Path, *, now=time.time):
        self.config_file = config_file
        self.now = now
        nonce_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(nonce_file, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(fd)
        os.chmod(nonce_file, 0o600)
        self.nonce_file = nonce_file
        with sqlite3.connect(nonce_file) as db:
            db.execute("CREATE TABLE IF NOT EXISTS used (source TEXT NOT NULL, nonce TEXT NOT NULL, expires REAL NOT NULL, PRIMARY KEY(source,nonce))")
        self.lock = asyncio.Lock()

    async def verify(self, source: str, timestamp: str, nonce: str, signature: str,
                     method: str, target: str, body: bytes):
        if not SOURCE.fullmatch(source) or not NONCE.fullmatch(nonce) or method not in METHODS:
            raise HTTPException(401, "手机签名无效")
        try:
            instant = int(timestamp)
            if str(instant) != timestamp or abs(self.now() - instant) > MAX_SKEW:
                raise ValueError()
            raw_signature = base64.b64decode(signature, validate=True)
            if len(raw_signature) != 64:
                raise ValueError()
            config = json.loads(read_regular(self.config_file, 65536))
            key = config["parent_public_keys"][source]
            raw_key = base64.b64decode(key, validate=True)
            Ed25519PublicKey.from_public_bytes(raw_key).verify(
                raw_signature, canonical(method, target, body, timestamp, nonce))
        except (ValueError, KeyError, TypeError, OSError, InvalidSignature, json.JSONDecodeError):
            raise HTTPException(401, "手机签名无效") from None
        async with self.lock:
            current = self.now()
            with sqlite3.connect(self.nonce_file, timeout=5) as db:
                db.execute("DELETE FROM used WHERE expires<=?", (current,))
                if db.execute("SELECT COUNT(*) FROM used").fetchone()[0] >= 4096:
                    raise HTTPException(503, "请求繁忙，请稍后重试")
                try:
                    db.execute("INSERT INTO used VALUES (?,?,?)", (source, nonce, current + MAX_SKEW * 2))
                except sqlite3.IntegrityError:
                    raise HTTPException(409, "请求已使用") from None


def valid_target(target: str) -> bool:
    if not target.startswith("/api/") or len(target) > 2048 or any(ord(c) < 32 or ord(c) > 126 for c in target):
        return False
    path = target.split("?", 1)[0]
    if re.search(r"%(?![0-9A-Fa-f]{2})|%(?:2f|5c|00|2e|3f|23|25)", path, re.IGNORECASE):
        return False
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeDecodeError:
        return False
    return not ("//" in decoded or "\\" in decoded or any(piece in {".", ".."} for piece in decoded.split("/"))
                or any(ord(char) < 32 for char in decoded) or decoded == "/api/")


def forward(method: str, target: str, body: bytes, content_type: str, token: str, port: int) -> tuple[int, bytes, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=35)
    try:
        connection.request(method, target, body=body, headers={"Authorization": "Bearer " + token,
                           "Content-Type": content_type or "application/json", "Accept": "application/json"})
        result = connection.getresponse()
        data = result.read(MAX_RESPONSE + 1)
        if len(data) > MAX_RESPONSE:
            raise ValueError("response_too_large")
        return result.status, data, result.getheader("Content-Type", "application/json")
    finally:
        connection.close()


def create_app(config_file: Path, token: str, *, nonce_file: Path | None = None,
               gateway_port: int = 8000, now=time.time) -> FastAPI:
    if len(token) < 24:
        raise ValueError("parent token unavailable")
    authorizer = PhoneAuthorizer(config_file, nonce_file or config_file.parent / "mobile-admin-nonces.sqlite3", now=now)
    app = FastAPI(title="Puppy mobile administration", docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/api/{rest:path}", methods=sorted(METHODS))
    async def proxy(request: Request, rest: str):
        target = request.scope.get("raw_path", b"").decode("ascii", errors="replace")
        query = request.scope.get("query_string", b"")
        if query:
            target += "?" + query.decode("ascii", errors="replace")
        if not valid_target(target):
            raise HTTPException(400, "无效请求路径")
        length = request.headers.get("content-length")
        if length is not None and (not length.isdecimal() or int(length) > MAX_BODY):
            raise HTTPException(413, "请求过大")
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > MAX_BODY:
                raise HTTPException(413, "请求过大")
        await authorizer.verify(request.headers.get("x-puppy-source", ""),
                                request.headers.get("x-puppy-timestamp", ""),
                                request.headers.get("x-puppy-nonce", ""),
                                request.headers.get("x-puppy-signature", ""),
                                request.method, target, bytes(body))
        try:
            status, data, mime = await asyncio.to_thread(
                forward, request.method, target, bytes(body), request.headers.get("content-type", ""), token, gateway_port)
        except (OSError, TimeoutError, ValueError):
            raise HTTPException(502, "家庭网关暂不可用") from None
        return Response(content=data, status_code=status, media_type=mime.split(";", 1)[0],
                        headers={"Cache-Control": "no-store"})

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    config = json.loads(read_regular(root / "gateway-config.json", 65536))
    mobile_tls = tls_options({"PUPPY_TLS_CERT_PATH": config["PUPPY_MOBILE_TLS_CERT_PATH"],
                              "PUPPY_TLS_KEY_PATH": config["PUPPY_MOBILE_TLS_KEY_PATH"]}, root)
    remote_path = Path(config["PUPPY_REMOTE_CONFIG"]).expanduser()
    if not remote_path.is_absolute():
        remote_path = root / remote_path
    remote_path = remote_path.resolve()
    if not remote_path.is_file():
        raise SystemExit("Remote parent key registry missing")
    token_file = root / ".parent-token"
    if token_file.stat().st_mode & 0o077:
        raise SystemExit("Parent token permissions too broad")
    token = token_file.read_text().strip()
    port = int(config.get("PUPPY_MOBILE_PORT", 8443))
    gateway_port = int(config.get("PUPPY_PORT", 8000))
    if not 1024 <= port <= 65535 or not 1024 <= gateway_port <= 65535 or port == gateway_port:
        raise SystemExit("Invalid mobile or gateway port")
    uvicorn.run(create_app(remote_path, token, nonce_file=root / "data/mobile-admin-nonces.sqlite3",
                           gateway_port=gateway_port), host="0.0.0.0", port=port,
                access_log=False, **mobile_tls)


if __name__ == "__main__":
    main()
