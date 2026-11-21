"""Parent-authorized registration routes and one-use Puppy voice tickets.

Root integration:
    registrations = IdentityRegistration(identity_service, capture_jpeg=capture,
                                         settings_guard=INFERENCE.lock)
    app.include_router(create_identity_router(registrations, require_parent))
Before transcription of authenticated device audio:
    enrollment = await registrations.consume_enrollment(device_id, pcm16)
    if enrollment is not None: handle registration result; skip chat for this audio.

capture(device_id) is async and returns JPEG bytes. Its owner must clean up any
intermediate device photo. This module never captures from the Mac camera/mic.

Read identity_settings(device_id) before identity extraction. Carry its revision
with the result and check identity_allowed(device_id, revision) again inside the
same INFERENCE lock, before any personal-memory read/write. The settings guard
makes disable wait for existing commits; stale results remain invalid after a
disable/re-enable cycle. Registration and sample revocation never enable identity.
"""
import asyncio
import io
import time
import uuid
import wave
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from PIL import Image

from .identity import IdentityService

MAX_JPEG_BYTES = 512000
MAX_WAV_BYTES = 20 * 32000 + 4096
TICKET_SECONDS = 60
TERMINAL_RETENTION_SECONDS = 300


def validate_jpeg(data: bytes) -> bytes:
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_JPEG_BYTES:
        raise ValueError("JPEG 须小于 512 KB")
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("需要完整的 JPEG 照片")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or min(image.size) < 64 or max(image.size) > 1920:
                raise ValueError("JPEG 尺寸须在 64 至 1920 像素之间")
            image.verify()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("JPEG 无法解码") from exc
    return data


def validate_pcm(data: bytes) -> bytes:
    if not isinstance(data, bytes) or len(data) % 2 or not 3 * 32000 <= len(data) <= 20 * 32000:
        raise ValueError("语音须为 3 至 20 秒的 16 kHz 单声道 PCM16")
    return data


def decode_wav(data: bytes) -> bytes:
    if not isinstance(data, bytes) or not 44 <= len(data) <= MAX_WAV_BYTES:
        raise ValueError("WAV 文件过大或不完整")
    if (data[:4] != b"RIFF" or data[8:12] != b"WAVE"
            or int.from_bytes(data[4:8], "little") + 8 != len(data)):
        raise ValueError("需要标准完整的 RIFF WAV 文件")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("WAV 必须是 16 kHz、单声道、16 位未压缩 PCM")
            frames = audio.getnframes()
            if not 3 * 16000 <= frames <= 20 * 16000:
                raise ValueError("语音须为 3 至 20 秒")
            pcm = audio.readframes(frames + 1)
            if len(pcm) != frames * 2:
                raise ValueError("WAV 音频数据不完整")
    except (wave.Error, EOFError) as exc:
        raise ValueError("WAV 格式无效") from exc
    return validate_pcm(pcm)


async def read_upload(request: Request, limit: int) -> bytes:
    """Read after parent auth, cap the whole request, and keep bounded files in memory."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
        raise HTTPException(415, "请使用 multipart/form-data 上传一个 file 文件")
    body_limit = limit + 16384

    async def bounded_stream():
        count = 0
        async for chunk in request.stream():
            count += len(chunk)
            if count > body_limit:
                raise MultiPartException("上传超过大小限制")
            yield chunk

    parser = MultiPartParser(request.headers, bounded_stream(), max_files=1, max_fields=0,
                             max_part_size=limit)
    parser.spool_max_size = body_limit  # The request cap is below the spool threshold.
    form = None
    try:
        form = await parser.parse()
        file = form.get("file")
        if len(form.multi_items()) != 1 or not isinstance(file, UploadFile):
            raise HTTPException(422, "只接受一个名为 file 的文件")
        data = await file.read(limit + 1)
        if not 0 < len(data) <= limit:
            raise HTTPException(413, "文件为空或超过大小限制")
        return data
    except MultiPartException as exc:
        raise HTTPException(413, str(exc)) from exc
    finally:
        if form is not None:
            await form.close()


class MemberInput(BaseModel):
    display_name: str = Field(min_length=1, max_length=32)

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value):
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("请输入有效成员名字")
        return value


class DeviceInput(BaseModel):
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9:_.-]+$")


class TicketInput(DeviceInput):
    person_id: str = Field(min_length=1, max_length=64, pattern=r"^[\w-]+$")


class IdentitySettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


def valid_device(device_id: str):
    import re
    if not isinstance(device_id, str) or not re.fullmatch(r"[A-Za-z0-9:_.-]{1,64}", device_id):
        raise HTTPException(422, "设备 ID 仅使用字母、数字、冒号、点、下划线或横线")


def valid_person(person_id: str):
    import re
    if not re.fullmatch(r"[\w-]{1,64}", person_id):
        raise HTTPException(422, "成员 ID 仅使用字母、数字、下划线或横线")


@dataclass
class VoiceTicket:
    ticket_id: str
    device_id: str
    person_id: str
    member_generation: str
    deadline: float
    expires_at: float
    state: str = "waiting"
    result: dict | None = None
    error: str | None = None


class IdentityRegistration:
    def __init__(self, service: IdentityService, *,
                 capture_jpeg: Callable[[str], Awaitable[bytes]] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 settings_guard: asyncio.Lock | None = None):
        self.service, self.capture_jpeg, self.clock = service, capture_jpeg, clock
        self.lock = asyncio.Lock()
        # A supplied inference lock must remain held until its worker has really
        # finished, even if its HTTP/audio caller was cancelled.
        self.settings_guard = settings_guard if settings_guard is not None else asyncio.Lock()
        self.tickets: dict[str, VoiceTicket] = {}
        self.device_tickets: dict[str, str] = {}
        with service.store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS identity_members (
                person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
                generation TEXT NOT NULL, updated_at REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS identity_device_settings (
                device_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)),
                revision INTEGER NOT NULL CHECK(revision >= 0))""")

    def identity_settings(self, device_id: str) -> dict:
        valid_device(device_id)
        with self.service.store.connect() as db:
            row = db.execute("SELECT enabled, revision FROM identity_device_settings WHERE device_id=?",
                             (device_id,)).fetchone()
        return {"device_id": device_id, "enabled": bool(row["enabled"]) if row else False,
                "revision": row["revision"] if row else 0}

    def identity_allowed(self, device_id: str, revision: int | None = None) -> bool:
        if revision is not None and (type(revision) is not int or revision < 0):
            return False
        settings = self.identity_settings(device_id)
        return settings["enabled"] and (revision is None or revision == settings["revision"])

    async def set_identity_settings(self, device_id: str, enabled: bool) -> dict:
        valid_device(device_id)
        if type(enabled) is not bool:
            raise HTTPException(422, "enabled 必须是布尔值")
        # The parent response is sent only after all earlier personal-memory
        # commits sharing this guard finish and the new revision is durable.
        async with self.settings_guard:
            with self.service.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("""INSERT INTO identity_device_settings VALUES (?, ?, 1)
                    ON CONFLICT(device_id) DO UPDATE SET enabled=excluded.enabled,
                    revision=identity_device_settings.revision+1""", (device_id, int(enabled)))
                revision = db.execute("SELECT revision FROM identity_device_settings WHERE device_id=?",
                                      (device_id,)).fetchone()["revision"]
            return {"device_id": device_id, "enabled": enabled, "revision": revision}

    def member(self, person_id: str) -> dict:
        valid_person(person_id)
        with self.service.store.connect() as db:
            row = db.execute("SELECT * FROM identity_members WHERE person_id=?", (person_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "成员不存在，请先保存成员")
        return dict(row)

    def save_member(self, person_id: str, display_name: str) -> dict:
        valid_person(person_id)
        with self.service.store.connect() as db:
            db.execute("""INSERT INTO identity_members VALUES (?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET display_name=excluded.display_name,
                updated_at=excluded.updated_at""", (person_id, display_name, str(uuid.uuid4()), time.time()))
        return {"person_id": person_id, "display_name": display_name}

    def list_members(self) -> list[dict]:
        samples = self.service.store.list_samples()
        with self.service.store.connect() as db:
            rows = db.execute("SELECT person_id, display_name FROM identity_members ORDER BY display_name").fetchall()
        result = []
        current_models = {"face": getattr(self.service.face_engine, "model_id", None),
                          "voice": getattr(self.service.voice_engine, "model_id", None)}
        for row in rows:
            member = dict(row)
            member["counts"] = {kind: sum(sample["person_id"] == row["person_id"] and sample["modality"] == kind
                                            for sample in samples) for kind in ("face", "voice")}
            member["ready"] = all(current_models[kind] and sum(
                sample["person_id"] == row["person_id"] and sample["modality"] == kind
                and sample["model_id"] == current_models[kind] for sample in samples
            ) >= self.service.policy.min_samples for kind in ("face", "voice"))
            result.append(member)
        return result

    def _expire(self):
        now = self.clock()
        for ticket in self.tickets.values():
            if ticket.state in {"waiting", "processing"} and now >= ticket.deadline:
                ticket.state = "expired"
        for key, ticket in list(self.tickets.items()):
            if now > ticket.deadline + TERMINAL_RETENTION_SECONDS:
                del self.tickets[key]
                if self.device_tickets.get(ticket.device_id) == key:
                    del self.device_tickets[ticket.device_id]

    def ticket_status(self, ticket_id: str) -> dict:
        self._expire()
        ticket = self.tickets.get(ticket_id)
        if ticket is None:
            raise HTTPException(404, "注册窗口不存在")
        return {"ticket_id": ticket.ticket_id, "device_id": ticket.device_id,
                "person_id": ticket.person_id, "state": ticket.state,
                "expires_at": ticket.expires_at, "remaining_seconds": max(0, int(ticket.deadline - self.clock())),
                "result": ticket.result, "error": ticket.error}

    async def create_ticket(self, device_id: str, person_id: str) -> dict:
        async with self.lock:
            self._expire()
            member = self.member(person_id)
            if self.service.voice_engine is None:
                raise HTTPException(503, "声纹模型未就绪，请检查家中 Mac 的身份模型")
            prior = self.tickets.get(self.device_tickets.get(device_id, ""))
            if prior and prior.state in {"waiting", "processing"}:
                raise HTTPException(409, "该 Puppy 已有注册窗口，请先取消或等待结束")
            if len(self.tickets) >= 100:
                raise HTTPException(429, "注册请求过多，请稍后再试")
            ticket = VoiceTicket(str(uuid.uuid4()), device_id, person_id, member["generation"],
                                 self.clock() + TICKET_SECONDS, time.time() + TICKET_SECONDS)
            self.tickets[ticket.ticket_id] = ticket
            self.device_tickets[device_id] = ticket.ticket_id
            return self.ticket_status(ticket.ticket_id)

    async def cancel_ticket(self, ticket_id: str) -> dict:
        async with self.lock:
            status = self.ticket_status(ticket_id)
            if status["state"] in {"waiting", "processing"}:
                self.tickets[ticket_id].state = "cancelled"
            return self.ticket_status(ticket_id)

    def _member_unchanged(self, person_id: str, generation: str) -> bool:
        try:
            return self.member(person_id)["generation"] == generation
        except HTTPException:
            return False

    async def enroll_data(self, person_id: str, modality: str, data: bytes,
                          expected_generation: str | None = None) -> dict:
        member = self.member(person_id)
        if expected_generation is not None and member["generation"] != expected_generation:
            raise HTTPException(409, "上传期间成员已撤销，请重新选择成员")
        observation = await asyncio.to_thread(self.service.prepare_enrollment, modality, data)
        async with self.lock:
            if not self._member_unchanged(person_id, member["generation"]):
                raise HTTPException(409, "注册期间成员已撤销，请重新选择成员")
            return self.service.commit_enrollment(person_id, observation, parent_confirmed=True)

    async def capture_face(self, person_id: str, device_id: str) -> dict:
        # Validate the member before asking a device to capture anything.
        member = self.member(person_id)
        if self.capture_jpeg is None:
            raise HTTPException(503, "Puppy 拍照注册暂不可用")
        data = validate_jpeg(await self.capture_jpeg(device_id))
        if not self._member_unchanged(person_id, member["generation"]):
            raise HTTPException(409, "拍照期间成员已撤销")
        return await self.enroll_data(person_id, "face", data, member["generation"])

    async def consume_enrollment(self, device_id: str, pcm: bytes) -> dict | None:
        """Call only for authenticated Puppy audio, before ASR; None means normal chat."""
        async with self.lock:
            ticket = self.tickets.get(self.device_tickets.get(device_id, ""))
            if ticket is None or ticket.state != "waiting":
                return None
            if self.clock() >= ticket.deadline:
                ticket.state = "expired"
                return {"handled": True, **self.ticket_status(ticket.ticket_id)}
            ticket.state = "processing"
        try:
            validate_pcm(pcm)
            observation = await asyncio.wait_for(
                asyncio.to_thread(self.service.prepare_enrollment, "voice", pcm),
                timeout=max(.01, ticket.deadline - self.clock()))
            async with self.lock:
                if self.clock() >= ticket.deadline:
                    ticket.state = "expired"
                elif ticket.state == "processing" and self._member_unchanged(ticket.person_id, ticket.member_generation):
                    ticket.result = self.service.commit_enrollment(ticket.person_id, observation, parent_confirmed=True)
                    ticket.state = "completed"
                elif ticket.state == "processing":
                    ticket.state = "cancelled"
        except asyncio.CancelledError:
            ticket.state = "cancelled"
            raise
        except TimeoutError:
            ticket.state = "expired"
        except ValueError as exc:
            if ticket.state == "processing":
                ticket.state, ticket.error = "failed", str(exc)
        except Exception:
            if ticket.state == "processing":
                ticket.state, ticket.error = "failed", "身份模型暂不可用，请稍后重试"
        return {"handled": True, **self.ticket_status(ticket.ticket_id)}

    async def revoke_person(self, person_id: str) -> dict:
        valid_person(person_id)
        # Use the same ordering as recognition settings: earlier personal turns
        # finish before revocation succeeds. Never wait for this guard while
        # holding the enrollment lock.
        async with self.settings_guard:
            async with self.lock:
                with self.service.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    if db.execute("SELECT 1 FROM identity_members WHERE person_id=?",
                                  (person_id,)).fetchone() is None:
                        raise HTTPException(404, "成员不存在，请先保存成员")
                    count = db.execute("DELETE FROM identity_samples WHERE person_id=?",
                                       (person_id,)).rowcount
                    db.execute("DELETE FROM identity_members WHERE person_id=?", (person_id,))
                    # A queued recognition may still reference this member,
                    # including after the same ID is registered again.
                    db.execute("UPDATE identity_device_settings SET revision=revision+1")
                for ticket in self.tickets.values():
                    if ticket.person_id == person_id and ticket.state in {"waiting", "processing"}:
                        ticket.state = "cancelled"
                return {"person_id": person_id, "revoked_samples": count}

    async def revoke_sample(self, person_id: str, sample_id: str) -> dict:
        valid_person(person_id)
        # Removing even one reference invalidates earlier matching decisions.
        # Keep the member, but use the same ordering and revision fence as a
        # complete member revocation.
        async with self.settings_guard:
            async with self.lock:
                with self.service.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    if db.execute("SELECT 1 FROM identity_members WHERE person_id=?",
                                  (person_id,)).fetchone() is None:
                        raise HTTPException(404, "成员不存在，请先保存成员")
                    count = db.execute("DELETE FROM identity_samples WHERE person_id=? AND sample_id=?",
                                       (person_id, sample_id)).rowcount
                    if not count:
                        raise HTTPException(404, "该成员的样本不存在")
                    db.execute("UPDATE identity_device_settings SET revision=revision+1")
                return {"revoked": True}


def create_identity_router(registrations: IdentityRegistration, require_parent) -> APIRouter:
    router = APIRouter(prefix="/api/identity", dependencies=[Depends(require_parent)])

    @router.get("/devices/{device_id}/settings")
    async def identity_settings(device_id: str):
        return registrations.identity_settings(device_id)

    @router.put("/devices/{device_id}/settings")
    async def set_identity_settings(device_id: str, settings: IdentitySettingsInput):
        return await registrations.set_identity_settings(device_id, settings.enabled)

    @router.get("/members")
    async def members():
        return {"members": registrations.list_members(), "purpose": "conversation_identity",
                "min_samples_per_modality": registrations.service.policy.min_samples}

    @router.put("/members/{person_id}")
    async def save_member(person_id: str, member: MemberInput):
        async with registrations.lock:
            return registrations.save_member(person_id, member.display_name)

    @router.get("/members/{person_id}/samples")
    async def samples(person_id: str):
        registrations.member(person_id)
        return registrations.service.store.list_samples(person_id)

    @router.delete("/members/{person_id}")
    async def revoke_person(person_id: str):
        return await registrations.revoke_person(person_id)

    @router.delete("/members/{person_id}/samples/{sample_id}")
    async def revoke_sample(person_id: str, sample_id: str):
        return await registrations.revoke_sample(person_id, sample_id)

    async def uploaded(request, person_id, modality):
        member = registrations.member(person_id)
        data = await read_upload(request, MAX_JPEG_BYTES if modality == "face" else MAX_WAV_BYTES)
        try:
            data = validate_jpeg(data) if modality == "face" else decode_wav(data)
            return await registrations.enroll_data(person_id, modality, data, member["generation"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/members/{person_id}/face")
    async def face_upload(person_id: str, request: Request):
        return await uploaded(request, person_id, "face")

    @router.post("/members/{person_id}/voice")
    async def voice_upload(person_id: str, request: Request):
        return await uploaded(request, person_id, "voice")

    @router.post("/members/{person_id}/capture-face")
    async def capture_face(person_id: str, device: DeviceInput):
        try:
            return await registrations.capture_face(person_id, device.device_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/voice-tickets")
    async def create_ticket(ticket: TicketInput):
        return await registrations.create_ticket(ticket.device_id, ticket.person_id)

    @router.get("/voice-tickets/{ticket_id}")
    async def ticket_status(ticket_id: str):
        async with registrations.lock:
            return registrations.ticket_status(ticket_id)

    @router.delete("/voice-tickets/{ticket_id}")
    async def cancel_ticket(ticket_id: str):
        return await registrations.cancel_ticket(ticket_id)

    return router
