"""Local MacBook gateway. Start with `python -m companion.server`."""

import asyncio
import hmac
import io
import json
import logging
import os
import re
import time
import uuid
from copy import deepcopy
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, PrivateAttr
import uvicorn
from PIL import Image

from .agent import LocalModel, ModelResponseError, PetAgent
from .minimax_model import MiniMaxModel, voice_remote_only
from .audio import OpusCodec, SpeechServices, boost_quiet_speech, pcm_peak_frame_rms
from .audio_session import AudioSession
from .audio_identity import device_audio_conversation_id
from .asr_disagreement import (apple_transcribe_pcm, clarification_for_transcripts,
                               numeric_only_clarification, prefer_clear_arithmetic_question,
                               prefer_quiet_translation_question, prefer_quiet_story_request, public_animal_body_count,
                               public_animal_group_count,
                               resolve_body_count, prefer_clear_activity_control)
from .story_routing import pauses_story, replacement_character, requests_story, short_story_followup
from .child_activity import requests_english, stops_activity
from .store import Store
from .inference import InferenceQueue
from .identity import IdentityService, IdentityStore
from .identity_api import IdentityRegistration, create_identity_router
from .learning import LearningStore
from .learning_api import create_learning_router
from .patrol import PatrolService
from .patrol_api import create_patrol_router
from .camera_dialogue import CameraDialogue
from .turn_commit import TurnCommit
from .gesture import GestureService, SubprocessRecognizer
from .gesture_api import create_gesture_router
from .offline_library import OfflineLibraryService
from .offline_library_api import create_offline_library_router
from .gateway_config import GatewayConfigService
from .gateway_config_api import create_gateway_config_router
from .tls import tls_options
from .pet_profile import PetProfile, RevisionConflict
from .routine_api import create_routine_router
from .action_drafts import ActionDraftService, ApprovedActionConflict, DraftError
from .action_drafts_api import create_action_drafts_router
from .panel_assets import PanelAssets
from .hermes_voice import HermesVoiceTasks
from .personal_access import (PersonalAccess, PersonalAccessError, asks_about_sharing_with_strangers, classify_personal_request,
                              is_public_english_response, is_public_english_start, is_public_why_question, unavailable_reply)
from .voice_motion import (motion_command, reconcile_motion, declined_stop,
                           stop_negation_prefix, explicit_hold_still,
                           unclear_motion, motion_confirmation, motion_stop_command,
                           reconcile_confirmation)


DATA_DIR = Path(os.environ.get("PUPPY_DATA_DIR", Path.home() / "Library/Application Support/PuppyCompanion")).expanduser().resolve()
STORE = Store(DATA_DIR / "companion.sqlite3")
DEVICE_TOKEN = os.environ.get("PUPPY_DEVICE_TOKEN", "")
PARENT_TOKEN = os.environ.get("PUPPY_PARENT_TOKEN", "")
MODEL_PATH = os.environ.get("PUPPY_MODEL_PATH", str(Path.home() / "RUGPuppyGateway/models/Qwen3.5-4B-4bit"))
TEXT_PROVIDER = os.environ.get("PUPPY_TEXT_PROVIDER", "local")
MINIMAX_KEY_FILE = os.environ.get("PUPPY_MINIMAX_API_KEY_FILE", "")
MINIMAX_REGION = os.environ.get("PUPPY_MINIMAX_REGION", "cn")
WHISPER_PATH = os.environ.get("PUPPY_WHISPER_PATH", str(Path.home() / "RUGPuppyGateway/models/whisper-small-mlx"))
APPLE_ASR_BINARY = os.environ.get("PUPPY_APPLE_ASR_BINARY", "")
QUIET_ASR_CROSSCHECK_SECONDS = 1.5
QUIET_VOICE_PEAK_RMS = 650
SAFE_ACTIONS = {"wave", "sit", "stretch", "hug", "stop"}
STATIONARY_PLAY = os.environ.get("PUPPY_STATIONARY_PLAY") == "1"
STATIONARY_REPLY = "现在是安静陪玩时间。我们可以聊天、讲故事或学英语。"
HERMES_TASKS = HermesVoiceTasks(
    os.environ.get("PUPPY_HERMES_TASK_DIR", str(DATA_DIR / "hermes-tasks")),
    Path(__file__).resolve().parents[1] / "scripts/run_hermes_voice_task.py",
    enabled=not STATIONARY_PLAY and os.environ.get("PUPPY_HERMES_ENABLED") == "1",
)


@dataclass
class PendingCommand:
    device_id: str
    socket: WebSocket
    kind: str
    future: asyncio.Future
    retain_photo: bool = True
    photo_data: bytes | None = None
    photo_captured_at: float | None = None
    photo_requested_at: float | None = None


class DeviceHub:
    def __init__(self, command_timeout=8, photo_timeout=30, retry_delay=1, action_timeout=20):
        self.connections: dict[str, WebSocket] = {}
        self.status: dict[str, dict] = {}
        self.pending: dict[str, PendingCommand] = {}
        self.send_locks: dict[str, asyncio.Lock] = {}
        self.photo_locks: dict[str, asyncio.Lock] = {}
        self.command_timeout = command_timeout
        self.photo_timeout = photo_timeout
        self.retry_delay = retry_delay
        self.action_timeout = action_timeout

    async def command(self, device_id: str, kind: str, *, retain_photo=True, photo_attempts=3,
                      before_send=None, deadline=None, **fields) -> dict:
        if kind == "action" and self.status.get(device_id, {}).get("motion_protocol_version") != 2:
            raise HTTPException(409, "请先更新 Puppy 动作回执固件并等待设备状态")
        try:
            # A caller's total budget includes waiting for a previous photo,
            # send backpressure and retries, not just receipt time.
            async with asyncio.timeout_at(deadline):
                if kind == "photo":
                    async with self.photo_locks.setdefault(device_id, asyncio.Lock()):
                        return await self._command(device_id, kind, retain_photo=retain_photo,
                            photo_attempts=photo_attempts, before_send=before_send, deadline=deadline, **fields)
                return await self._command(device_id, kind, before_send=before_send, deadline=deadline, **fields)
        except TimeoutError as exc:
            raise HTTPException(504, "设备命令总等待时间已到") from exc

    async def _command(self, device_id: str, kind: str, *, retain_photo=True, photo_attempts=3,
                       before_send=None, deadline=None, **fields) -> dict:
        if photo_attempts not in {1, 2, 3}:
            raise ValueError("照片重试次数须为 1 至 3")
        attempts = photo_attempts if kind == "photo" else 1
        for attempt in range(attempts):
            socket = self.connections.get(device_id)
            if socket is None:
                if attempt + 1 == attempts:
                    raise HTTPException(503, "Puppy 当前不在线")
                await asyncio.sleep(self.retry_delay)
                continue
            request_id = str(uuid.uuid4())
            future = asyncio.get_running_loop().create_future()
            pending = PendingCommand(device_id, socket, kind, future, retain_photo)
            self.pending[request_id] = pending
            try:
                timeout = (self.photo_timeout if kind == "photo" else self.action_timeout if kind == "action"
                           else 35 if kind in {"set_offline_actions", "set_gateway_config"} else self.command_timeout)
                async with asyncio.timeout(timeout):
                    lock = self.send_locks[device_id]
                    async with lock:
                        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                            raise HTTPException(504, "本轮设备请求已过期")
                        if before_send is not None and before_send() is not True:
                            raise HTTPException(409, "家长已撤销本次设备请求的权限")
                        if kind == "photo":
                            pending.photo_requested_at = time.time()
                        await socket.send_json({"id": request_id, "type": kind, **fields})
                    receipt = await future
            except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError, KeyError, OSError) as exc:
                if kind == "photo" and (pending.photo_data is not None or STORE.photo_file(request_id)):
                    return self.photo_result(pending, {"type": "receipt", "id": request_id, "ok": True, "photo_id": request_id})
                if attempt + 1 == attempts:
                    raise HTTPException(504, "设备命令超时") from exc
                await asyncio.sleep(self.retry_delay)
                continue
            finally:
                self.pending.pop(request_id, None)
            if kind == "photo" and not receipt.get("ok") and attempt + 1 < attempts:
                await asyncio.sleep(self.retry_delay)
                continue
            return self.photo_result(pending, receipt) if kind == "photo" else receipt
        raise HTTPException(504, "设备命令超时")

    def owns_pending(self, device_id: str, socket: WebSocket, request_id: str,
                     kind: str | None = None) -> bool:
        pending = self.pending.get(request_id)
        return bool(pending and pending.device_id == device_id and pending.socket is socket
                    and (kind is None or pending.kind == kind) and not pending.future.done())

    async def control_for_confirmed_motion(self, device_id: str):
        """Prove bidirectional control before submitting a confirmed action."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 8
        probed = None
        while loop.time() < deadline:
            socket = self.connections.get(device_id)
            status = self.status.get(device_id, {})
            if socket is not None and socket is not probed and status.get("motion_protocol_version") == 2:
                probed = socket
                try:
                    receipt = await self.command(device_id, "status", deadline=min(deadline, loop.time() + 1.4))
                    if receipt.get("ok") and self.connections.get(device_id) is socket:
                        return socket
                except HTTPException:
                    pass
                if self.connections.get(device_id) is socket:
                    # An unresponsive control connection can coexist with a
                    # healthy audio connection. Replace it before sending any
                    # movement. A status probe cannot start the action.
                    try:
                        await asyncio.wait_for(socket.close(code=1012), timeout=.5)
                    except (TimeoutError, WebSocketDisconnect, OSError, RuntimeError):
                        pass
            await asyncio.sleep(.08)
        raise HTTPException(503, "设备控制连接未恢复")

    def receipt(self, device_id: str, socket: WebSocket, packet: dict):
        request_id = packet.get("id")
        if not isinstance(request_id, str) or not self.owns_pending(device_id, socket, request_id):
            return
        pending = self.pending[request_id]
        if pending.kind == "action":
            phase = packet.get("phase")
            if phase == "accepted" or (packet.get("ok") and phase not in {"completed", "cancelled", "failed"}):
                return  # Acknowledging receipt is not proof of physical completion.
            if phase in {"cancelled", "failed"}:
                packet = {**packet, "ok": False}
        if pending.kind == "photo" and packet.get("ok"):
            if packet.get("photo_id") != request_id or not (pending.photo_data is not None or STORE.photo_file(request_id)):
                packet = {**packet, "ok": False, "error": "照片未成功保存"}
        pending.future.set_result(packet)

    @staticmethod
    def photo_result(pending, receipt):
        if pending.photo_data is not None:
            return {**receipt, "_photo_bytes": pending.photo_data, "captured_at": pending.photo_captured_at,
                    "requested_at": pending.photo_requested_at, "time_source": "gateway_receive"}
        return receipt

    def receive_photo(self, device_id, socket, request_id, data):
        if not self.owns_pending(device_id, socket, request_id, "photo"):
            return
        pending = self.pending[request_id]
        if pending.retain_photo:
            save_photo(data, device_id, request_id)
        else:
            validate_photo(data)
            pending.photo_data = data
            pending.photo_captured_at = time.time()

    def disconnected(self, device_id: str, socket: WebSocket):
        for pending in self.pending.values():
            if pending.device_id == device_id and pending.socket is socket and not pending.future.done():
                pending.future.set_exception(WebSocketDisconnect())


HUB = DeviceHub()
MODEL = None
INFERENCE = InferenceQueue()
MODEL_LOCK = INFERENCE.lock


def new_model():
    if TEXT_PROVIDER == "minimax":
        return MiniMaxModel(MINIMAX_KEY_FILE, MODEL_PATH, region=MINIMAX_REGION)
    return LocalModel(MODEL_PATH)


RUNNING: dict[str, asyncio.Task] = {}
STOPPING: dict[str, asyncio.Task] = {}
STOP_REQUESTS: dict[str, dict] = {}
ACTION_JOBS: dict[str, dict] = {}
AUDIO_SESSIONS: dict[str, set[AudioSession]] = {}
REMOTE = None
PATROL = None
GESTURES = None
IDENTITY = IdentityService(IdentityStore(DATA_DIR / "identity.sqlite3"))
LEARNING = LearningStore(DATA_DIR / "learning.sqlite3")


async def preload_model():
    """Prime the local MLX model without delaying the gateway's HTTP startup."""
    def load_and_prime():
        global MODEL
        if MODEL is not None:
            return False
        MODEL = new_model()
        # A small synthetic request also initializes the first generation path.
        # It reads no family data and its answer is discarded.
        if TEXT_PROVIDER == "local":
            MODEL.requests_camera("你好")
        return True

    started = time.monotonic()
    try:
        primed = await INFERENCE.run(load_and_prime)
        if primed:
            LOGGER.info("puppy_model_preload_ready elapsed_ms=%d", int((time.monotonic() - started) * 1000))
        else:
            LOGGER.info("puppy_model_preload_skipped model_already_loaded=true")
    except asyncio.CancelledError:
        raise
    except Exception as error:
        LOGGER.warning("puppy_model_preload_failed error=%s", type(error).__name__)


@asynccontextmanager
async def lifespan(app):
    global REMOTE, PATROL, GESTURES
    preload_task = None
    identity_models = os.environ.get("PUPPY_IDENTITY_MODELS")
    if identity_models:
        from .identity_models import FaceEngine, VoiceEngine
        try:
            IDENTITY.face_engine = await asyncio.to_thread(FaceEngine, identity_models)
            IDENTITY.voice_engine = await asyncio.to_thread(VoiceEngine, identity_models)
        except (ImportError, ValueError, OSError) as exc:
            LOGGER.warning("Identity models unavailable: %s", type(exc).__name__)
            IDENTITY.face_engine = IDENTITY.voice_engine = None
    try:
        config_path = os.environ.get("PUPPY_REMOTE_CONFIG")
        if config_path:
            from .remote_runtime import RemoteRuntime
            REMOTE = RemoteRuntime(config_path, HUB, STORE, stop, execute_safe_action, describe_saved_photo)
            REMOTE.start()
        PATROL = PatrolService(DATA_DIR / "patrol", capture_handler=capture_patrol_jpeg,
                              describe_handler=describe_patrol_path)
        PATROL.start()
        gesture_python = os.environ.get("PUPPY_GESTURE_PYTHON")
        gesture_model = os.environ.get("PUPPY_GESTURE_MODEL")
        recognizer = SubprocessRecognizer(gesture_python, gesture_model) if gesture_python and gesture_model else None
        GESTURES = GestureService(capture_handler=capture_gesture_jpeg, recognize_handler=recognizer,
            execute_handler=execute_gesture_action, online_handler=lambda device: device in HUB.connections)
        if os.environ.get("PUPPY_MODEL_PRELOAD") == "1":
            preload_task = asyncio.create_task(preload_model(), name="puppy-model-preload")
        yield
    finally:
        if preload_task is not None:
            preload_task.cancel()
            await asyncio.gather(preload_task, return_exceptions=True)
        if GESTURES is not None:
            await GESTURES.close()
            GESTURES = None
        if PATROL is not None:
            await PATROL.close()
            PATROL = None
        if REMOTE is not None:
            await REMOTE.close()
            REMOTE = None
        tasks = [task for task in RUNNING.values() if not task.done()]
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*STOPPING.values(), return_exceptions=True)
        await INFERENCE.close()


app = FastAPI(title="Puppy 家庭本地控制", lifespan=lifespan)
LOGGER = logging.getLogger("uvicorn.error")
MODEL_RETRY_DETAIL = "本地模型暂未生成合适的回答，请再试一次"


@app.exception_handler(ModelResponseError)
@app.exception_handler(PersonalAccessError)
async def model_response_error(request: Request, error: Exception):
    return JSONResponse(status_code=503, content={"detail": MODEL_RETRY_DETAIL})


def require_parent(authorization: str = Header(default="")):
    expected = "Bearer " + PARENT_TOKEN
    if not PARENT_TOKEN or not hmac.compare_digest(authorization, expected):
        raise HTTPException(401, "家长令牌无效")


class ChatRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=300)
    person_id: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    mode: str = Field(default="chat", pattern="^(chat|english|story)$")
    photo_id: str | None = Field(default=None, min_length=1, max_length=64)
    # Device-side recognition and ephemeral observations cannot be submitted
    # as JSON by an API caller. The parent text route keeps its manual selection.
    _identity_revision: int | None = PrivateAttr(default=None)
    _from_device: bool = PrivateAttr(default=False)
    _vision_observation: dict | None = PrivateAttr(default=None)
    _vision_used: bool = PrivateAttr(default=False)
    _vision_times: tuple[float, float] | None = PrivateAttr(default=None)
    _previous_guest_turn: dict | None = PrivateAttr(default=None)
    _heard_alternatives: tuple[str, ...] = PrivateAttr(default=())


class ModeRequest(BaseModel):
    mode: str = Field(pattern="^(chat|english|story)$")


class EnglishCheckRequest(BaseModel):
    word: str = Field(min_length=2, max_length=24)
    checkpoint: str = Field(pattern="^(baseline|next_day|week_later)$")
    result: str = Field(pattern="^(independent|prompted|not_yet)$")
    modality: str = Field(pattern="^(pointed|spoken|used_in_context)$")


class ActionStep(BaseModel):
    action: str | None = None
    wait_ms: int = Field(default=0, ge=0, le=3000)


class SaveAction(BaseModel):
    name: str = Field(min_length=1, max_length=32, pattern=r"^[\w\-\u4e00-\u9fff]+$")
    steps: list[ActionStep] = Field(min_length=1, max_length=12)


class DeleteAction(BaseModel):
    expected_steps: list[ActionStep] = Field(min_length=1, max_length=12)


class VolumeRequest(BaseModel):
    volume: int = Field(strict=True, ge=0, le=60)


class SleepPolicyRequest(BaseModel):
    enabled: bool = Field(strict=True)


class VisionRequest(BaseModel):
    question: str = Field(default="请描述当前看得清的物体和环境。", min_length=1, max_length=200)


def validate_steps(steps: list[dict]):
    if sum(step.get("wait_ms", 0) for step in steps) > 20000:
        raise HTTPException(422, "动作总等待时间超过 20 秒")
    for step in steps:
        if step.get("action") not in SAFE_ACTIONS | {None}:
            raise HTTPException(422, "不支持的动作")
        if not step.get("action") and not step.get("wait_ms"):
            raise HTTPException(422, "空动作步骤")


@app.get("/")
def panel():
    return PanelAssets(Path(__file__).parent).page()


@app.get("/api/status", dependencies=[Depends(require_parent)])
async def status():
    return {"devices": {device_id: {**HUB.status.get(device_id, {}), "online": True}
                        for device_id in HUB.connections}, "actions": ACTION_JOBS,
            "stationary_play": STATIONARY_PLAY,
            "remote": REMOTE.status() if REMOTE else {"enabled": False},
            "patrol": PATROL.status() if PATROL else {"running": False},
            "identity_models_ready": bool(IDENTITY.face_engine and IDENTITY.voice_engine)}




@app.get("/api/pet/{device_id}", dependencies=[Depends(require_parent)])
async def pet(device_id: str):
    return {**STORE.pet(device_id), "conversation_mode": STORE.conversation_mode(device_id)}


@app.get("/api/devices/{device_id}/improvements/latest", dependencies=[Depends(require_parent)])
async def improvement_status(device_id: str):
    return {"task": HERMES_TASKS.latest(device_id)}


@app.put("/api/devices/{device_id}/mode", dependencies=[Depends(require_parent)])
async def set_mode(device_id: str, request: ModeRequest):
    STORE.set_conversation_mode(device_id, request.mode)
    STORE.child_sessions.clear_device(device_id)
    if request.mode != "english":
        LEARNING.end_device_sessions(device_id)
    return {"mode": request.mode}


@app.get("/api/memory/{device_id}/{person_id}", dependencies=[Depends(require_parent)])
async def memories(device_id: str, person_id: str):
    return {"memories": STORE.memory_entries(device_id, person_id),
            "english": STORE.english_progress(device_id, person_id),
            "english_words": [{"word": word, "meaning": meaning} for word, meaning in PetAgent.ENGLISH_WORDS],
            "english_checks": STORE.english_checks(device_id, person_id),
            "english_review_plan": STORE.english_review_plan(device_id, person_id),
            "story": STORE.story(device_id, person_id),
            "identity_source": "parent_selected"}


@app.post("/api/memory/{device_id}/{person_id}/english-checks", dependencies=[Depends(require_parent)])
async def record_english_check(device_id: str, person_id: str, request: EnglishCheckRequest):
    if request.word.lower() not in {word for word, _ in PetAgent.ENGLISH_WORDS}:
        raise HTTPException(422, "这个词不在当前英语词表中")
    with LEARNING.saving_permission(device_id, person_id) as allowed:
        if not allowed:
            raise HTTPException(409, "该成员已关闭英语进度保存")
        try:
            return STORE.record_english_check(device_id, person_id, request.word,
                                              request.checkpoint, request.result, request.modality)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.delete("/api/story/{device_id}/{person_id}", dependencies=[Depends(require_parent)])
async def forget_story(device_id: str, person_id: str):
    async with MODEL_LOCK:
        STORE.forget_story(device_id, person_id)
    return {"deleted_story": True}


@app.delete("/api/memory/{device_id}/{person_id}/{memory_key}", dependencies=[Depends(require_parent)])
async def forget_memory(device_id: str, person_id: str, memory_key: str):
    async with MODEL_LOCK:
        STORE.forget(device_id, person_id, memory_key)
    return {"deleted": memory_key}


@app.delete("/api/memory/{device_id}/{person_id}", dependencies=[Depends(require_parent)])
async def forget_person(device_id: str, person_id: str):
    async with MODEL_LOCK:
        STORE.forget_person(device_id, person_id)
    return {"deleted_person": person_id}


@app.get("/api/photos/{device_id}", dependencies=[Depends(require_parent)])
async def photos(device_id: str):
    return STORE.photos(device_id)


@app.get("/api/photo/{photo_id}", dependencies=[Depends(require_parent)])
async def photo_file(photo_id: str):
    file_path = STORE.photo_file(photo_id)
    if not file_path:
        raise HTTPException(404, "照片不存在")
    return FileResponse(file_path, media_type="image/jpeg")


async def describe_saved_photo(photo_id: str, question: str):
    path = STORE.photo_file(photo_id)
    if not path:
        raise HTTPException(404, "照片不存在")
    answer = await describe_local_path(path, question)
    return {"photo_id": photo_id, **answer, "scope": "single_photo"}


async def describe_local_path(path, question):
    def infer():
        global MODEL
        if MODEL is None:
            MODEL = new_model()
        return MODEL.describe(path, question)
    return await INFERENCE.run(infer)


@app.post("/api/photo/{photo_id}/describe", dependencies=[Depends(require_parent)])
async def describe_photo(photo_id: str, request: VisionRequest):
    return await describe_saved_photo(photo_id, request.question)


def validate_photo(data: bytes):
    if len(data) > 512000 or not data.startswith(b"\xff\xd8"):
        raise ValueError("照片不是有效的小尺寸 JPEG")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
        if image.format != "JPEG" or image.width > 640 or image.height > 640:
            raise ValueError("照片格式或尺寸不支持")


def save_photo(data: bytes, device_id: str, photo_id: str):
    validate_photo(data)
    directory = DATA_DIR / "photos"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = directory / f"{photo_id}.jpg"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
        STORE.add_photo(photo_id, device_id, str(destination))
    except Exception:
        destination.unlink(missing_ok=True)
        raise


@app.post("/v1/photo")
async def upload_photo(file: UploadFile = File(...), question: str = Form(default=""),
                       authorization: str = Header(default=""), device_id: str = Header(default="")):
    if not DEVICE_TOKEN or not hmac.compare_digest(authorization, "Bearer " + DEVICE_TOKEN):
        raise HTTPException(401, "设备令牌无效")
    if not device_id or len(device_id) > 64:
        raise HTTPException(422, "设备 ID 无效")
    data = await file.read(512001)
    photo_id = str(uuid.uuid4())
    try:
        save_photo(data, device_id, photo_id)
    except Exception as exc:
        raise HTTPException(422, "照片解码失败") from exc
    return {"photo_id": photo_id, "question": question[:100]}


@app.post("/api/devices/{device_id}/photo", dependencies=[Depends(require_parent)])
async def take_photo(device_id: str):
    return await HUB.command(device_id, "photo")


async def capture_identity_photo(device_id: str, *, before_send=None, deadline=None):
    result = await HUB.command(device_id, "photo", retain_photo=False, photo_attempts=1,
                               before_send=before_send, deadline=deadline)
    if not result.get("ok") or not result.get("_photo_bytes"):
        raise HTTPException(503, "身份照片暂不可用")
    return result


async def capture_identity_jpeg(device_id: str):
    return (await capture_identity_photo(device_id))["_photo_bytes"]


async def capture_patrol_jpeg(device_id: str):
    result = await HUB.command(device_id, "photo", retain_photo=False, photo_attempts=1)
    if not result.get("ok") or not result.get("_photo_bytes"):
        raise RuntimeError("Puppy 照片暂不可用")
    return result["_photo_bytes"]


async def capture_gesture_jpeg(device_id, *, retain_photo=False):
    if retain_photo is not False:
        raise ValueError("手势照片不能保存")
    result = await HUB.command(device_id, "photo", retain_photo=False, photo_attempts=1,
        before_send=lambda: GESTURES is not None and GESTURES.status(device_id)["enabled"],
        deadline=asyncio.get_running_loop().time() + 8)
    if result.get("ok") is not True or not result.get("_photo_bytes"):
        raise RuntimeError("手势照片暂不可用")
    return result["_photo_bytes"]


async def describe_patrol_path(path):
    return await describe_local_path(str(path), "请简短描述这张照片里看得清的物品和环境，看不清就说明。")


class PatrolProxy:
    def __getattr__(self, name):
        if PATROL is None:
            raise HTTPException(503, "巡看服务未启动")
        return getattr(PATROL, name)


class GestureProxy:
    def __getattr__(self, name):
        if GESTURES is None:
            raise HTTPException(503, "手势服务未启动")
        return getattr(GESTURES, name)


REGISTRATIONS = IdentityRegistration(IDENTITY, capture_jpeg=capture_identity_jpeg,
                                    settings_guard=INFERENCE.lock)


async def send_offline_library(device_id, kind, **fields):
    return await HUB.command(device_id, kind, **fields)


def canonical_action_steps(steps):
    result = [ActionStep.model_validate(step).model_dump() for step in steps]
    validate_steps(result)
    return result


def save_approved_draft(name, steps, *, expected=None, draft_id, ownership_nonce):
    if expected is not None or name in SAFE_ACTIONS:
        raise ApprovedActionConflict("不能覆盖已有动作")
    SaveAction(name=name, steps=steps)
    steps = canonical_action_steps(steps)
    with STORE.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM actions WHERE name=?", (name,)).fetchone():
            raise ApprovedActionConflict("已有同名动作，请先修改草稿名称")
        db.execute("INSERT INTO actions VALUES (?,?,?)", (name, json.dumps(steps), time.time()))
        db.execute("INSERT INTO action_ownership VALUES (?,?,?)", (name, draft_id, ownership_nonce))
    return {"draft_id": draft_id, "ownership_nonce": ownership_nonce}


def delete_approved_draft(name, expected_steps, *, ownership_receipt):
    expected = canonical_action_steps(expected_steps)
    with STORE.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT steps FROM actions WHERE name=?", (name,)).fetchone()
        owner = db.execute("SELECT draft_id, ownership_nonce FROM action_ownership WHERE name=?", (name,)).fetchone()
        if not owner or dict(owner) != ownership_receipt:
            raise ApprovedActionConflict("动作归属已变化，请在动作管理中核对")
        if not row or canonical_action_steps(json.loads(row[0])) != expected:
            raise ApprovedActionConflict("动作已修改或不存在，请刷新后再操作")
        db.execute("DELETE FROM actions WHERE name=?", (name,))
    return dict(owner)


def delete_saved_action(name, expected_steps):
    expected = canonical_action_steps(expected_steps)
    with STORE.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT steps FROM actions WHERE name=?", (name,)).fetchone()
        if not row or canonical_action_steps(json.loads(row[0])) != expected:
            raise ApprovedActionConflict("动作已修改或不存在，请刷新后再操作")
        db.execute("DELETE FROM actions WHERE name=?", (name,))


async def generate_action_draft(prompt):
    def infer():
        global MODEL
        if MODEL is None:
            MODEL = new_model()
        return MODEL(prompt, "请输出这次新把戏的草稿 JSON。")
    return await INFERENCE.run(infer)


OFFLINE_LIBRARY = OfflineLibraryService(DATA_DIR / "offline-library.sqlite3", command=send_offline_library,
    get_status=lambda device: dict(HUB.status.get(device, {})) if device in HUB.connections else None,
    approved_actions_snapshot=lambda: STORE.actions())
GATEWAY_CONFIGURATION = GatewayConfigService(DATA_DIR / "gateway-configurations.sqlite3", command=send_offline_library,
    get_status=lambda device: dict(HUB.status.get(device, {})) if device in HUB.connections else None,
    current_token=lambda: DEVICE_TOKEN)
ACTION_DRAFTS = ActionDraftService(DATA_DIR / "action-drafts.sqlite3", local_model_generate=generate_action_draft,
    approved_actions_snapshot=lambda: STORE.actions(), save_approved=save_approved_draft,
    delete_approved=delete_approved_draft)
app.include_router(create_identity_router(REGISTRATIONS, require_parent))
app.include_router(create_learning_router(LEARNING, require_parent))
app.include_router(create_patrol_router(PatrolProxy(), require_parent))
app.include_router(create_gesture_router(GestureProxy(), require_parent))
app.include_router(create_offline_library_router(OFFLINE_LIBRARY, require_parent))
app.include_router(create_gateway_config_router(GATEWAY_CONFIGURATION, require_parent))
app.include_router(create_routine_router(STORE, require_parent, require_member=REGISTRATIONS.member))
app.include_router(create_action_drafts_router(ACTION_DRAFTS, require_parent))


@app.get("/identity_panel.js")
def identity_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("identity_panel.js", request.headers.get("if-none-match"))


@app.get("/learning_panel.js")
def learning_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("learning_panel.js", request.headers.get("if-none-match"))


@app.get("/patrol_panel.js")
def patrol_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("patrol_panel.js", request.headers.get("if-none-match"))


@app.get("/gesture_panel.js")
def gesture_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("gesture_panel.js", request.headers.get("if-none-match"))


@app.get("/offline_library_panel.js")
def offline_library_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("offline_library_panel.js", request.headers.get("if-none-match"))


@app.get("/gateway_config_panel.js")
def gateway_config_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("gateway_config_panel.js", request.headers.get("if-none-match"))


@app.get("/routine_panel.js")
def routine_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("routine_panel.js", request.headers.get("if-none-match"))


@app.get("/action_drafts_panel.js")
def action_drafts_panel_script(request: Request):
    return PanelAssets(Path(__file__).parent).script("action_drafts_panel.js", request.headers.get("if-none-match"))


async def record_draft_history(request, answer, member_generation, initially_save_learning):
    if not request.person_id or member_generation is None:
        return
    # No model work or awaited worker runs under this short commit lock. A
    # cancelled waiter cannot append later, and identity disable shares the lock.
    async with MODEL_LOCK:
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        if request._identity_revision is not None and not REGISTRATIONS.identity_allowed(
                request.device_id, request._identity_revision):
            return
        try:
            if REGISTRATIONS.member(request.person_id).get("generation") != member_generation:
                return
        except HTTPException:
            return
        def commit():
            STORE.append_conversation(request.device_id, request.person_id, request.text,
                                      answer["reply"], request.mode, request.session_id)
        if request.mode == "english":
            if initially_save_learning:
                LEARNING.commit_if_saving(request.device_id, request.person_id, commit)
        else:
            commit()


@app.post("/api/chat", dependencies=[Depends(require_parent)])
async def chat(request: ChatRequest):
    previous_story = (request._previous_guest_turn["assistant"]
                      if request._previous_guest_turn and request._previous_guest_turn["mode"] == "story" else "")
    if request.mode == "story" and pauses_story(request.text):
        # The saved device mode stays story, so the next explicit continuation
        # can resume the chapter. This turn and its history are ordinary chat.
        request.mode = "chat"
    elif ((request.mode == "chat" and requests_story(request.text))
          or replacement_character(request.text, previous_story)):
        # A spoken story request gets narrative rules for this turn only. The
        # next ordinary question remains chat unless the parent set story mode.
        request.mode = "story"
    elif request.mode == "chat" and requests_english(request.text):
        request.mode = "english"
    turn_guard = TurnCommit()
    class LearningFinished(Exception):
        def __init__(self, state, access):
            self.state, self.access = state, access
    name_before = STORE.pet(request.device_id)["name"]
    stationary_voice = STATIONARY_PLAY and request._from_device
    action_snapshot = {} if stationary_voice else approved_chat_actions()
    draft_member_generation = None
    initially_save_learning = False
    photo_path = None
    if request.photo_id:
        if not any(p["photo_id"] == request.photo_id for p in STORE.photos(request.device_id)):
            raise HTTPException(404, "此设备的最近照片中没有这一张")
        photo_path = STORE.photo_file(request.photo_id)
    def infer_turn(settings_gates):
        global MODEL
        nonlocal draft_member_generation, initially_save_learning
        # This runs under the same lock as the parent's recognition setting.
        # An old result cannot survive disable/re-enable or commit afterward.
        device_origin = request._from_device or request._identity_revision is not None
        access_reason = None
        if device_origin:
            if (request._identity_revision is None
                    or not REGISTRATIONS.identity_allowed(request.device_id, request._identity_revision)):
                access_reason = "stale_identity" if request.person_id else "identity_unconfirmed"
                request.person_id = None
            elif request.person_id:
                try:
                    REGISTRATIONS.member(request.person_id)
                except HTTPException:
                    access_reason = "stale_identity"
                    request.person_id = None
            # Keep the private revision even after anonymizing. A stale device
            # identity must never become a parent's manual member selection.
        session_lesson = (STORE.child_sessions.lesson(request.device_id, request.session_id,
                                                     request.person_id)
                          if request.mode == "english" and request.session_id else None)
        # A frame can age out while another inference owns the queue. Never
        # feed its description to the next model as current visual evidence.
        if request._vision_times is not None and not DIALOGUE_CAMERA.fresh_times(request._vision_times):
            request._vision_observation = CameraDialogue.unavailable("照片已过期，当前无法看清")["observation"]
            request._vision_used = False
        access = PersonalAccess(request.person_id, "device_audio" if device_origin else "parent_http",
                                access_reason)
        plain_voice_chat = (request._from_device and request.mode in {"chat", "english"} and not photo_path
                            and request._vision_observation is None and not request._vision_used)
        anatomy = (public_animal_body_count(request.text) or public_animal_group_count(request.text)
                   if plain_voice_chat and request.mode == "chat" else None)
        vocabulary = PetAgent.english_word_question(request.text) if plain_voice_chat else None
        repeated = False
        if plain_voice_chat and not vocabulary:
            vocabulary = PetAgent.english_followup(request.text, request._previous_guest_turn)
            if not vocabulary and request.mode == "english" and session_lesson:
                # A confirmed member also has a current session word, without
                # needing the anonymous public-history buffer.
                current_word = session_lesson.get("word")
                if current_word in dict(PetAgent.ENGLISH_WORDS) and re.fullmatch(r"\W*"+re.escape(current_word)+r"\W*", request.text, re.I):
                    vocabulary = (current_word, dict(PetAgent.ENGLISH_WORDS)[current_word])
            repeated = bool(vocabulary and re.fullmatch(r"\W*"+re.escape(vocabulary[0])+r"\W*", request.text, re.I))
        local_repeat = (request.mode == "english" and repeated and session_lesson
                        and session_lesson.get("word") == vocabulary[0] and session_lesson.get("stage") == "repeat")
        if request.mode == "english" and repeated and not local_repeat:
            vocabulary = None  # Later practice stages still run their games/response rules.
        local_answer = None
        if vocabulary:
            word, meaning = vocabulary
            local_answer = {"reply": f"{meaning}的英文是 {word}。", "reply_source": "vocabulary",
                            "vocabulary": vocabulary}
            if local_repeat:
                local_answer["lesson"] = {"target_word": word, "meaning": meaning,
                                          "activity": "repeat", "text_match": True}
        elif anatomy:
            fact = PetAgent.animal_count_reply(*anatomy)
            if fact:
                local_answer = {"reply": fact, "reply_source": "public_fact"}
        if local_answer and request.mode == "chat":
            LEARNING.end_session(request.device_id, request.person_id)
            if request.person_id:
                STORE.begin_commit()
                STORE.append_conversation(request.device_id, request.person_id, request.text,
                                          local_answer["reply"], request.mode, request.session_id)
            return {**local_answer, "intent": "chat", "action": "", "value": "",
                    "personal_access": access.public_status()}
        if MODEL is None:
            MODEL = new_model()
        # Never send either the visual observation or its surrounding turn to
        # an online text model. This also covers saved-photo chat requests.
        turn_model = (MODEL.private_context_model()
                      if (photo_path or request._vision_observation is not None or request._vision_used)
                      and isinstance(MODEL, MiniMaxModel) else MODEL)
        needs_personal_context = None
        if not request.person_id:
            try:
                public_english_turn = request.mode == "english" and (
                    is_public_english_start(request.text)
                    or is_public_english_response(
                        request.text, session_lesson,
                        dict(PetAgent.ENGLISH_WORDS).get(session_lesson.get("word")) if session_lesson else None))
                public_story_turn = request.mode == "story" and (
                    replacement_character(request.text, previous_story)
                    or (previous_story and short_story_followup(request.text)))
                public_why = (request._from_device and request.mode in {"chat", "story"} and not photo_path
                              and request._vision_observation is None
                              and is_public_why_question(request.text, request._previous_guest_turn))
                needs_personal_context = (False if public_english_turn or anatomy or local_answer or public_story_turn or public_why
                                          else classify_personal_request(turn_model, request.text, turn_guard,
                                                                         public_turn=request._previous_guest_turn))
            except BaseException:
                # Classification precedes the next learning permit. A failed
                # request must still settle a guest's previous activity window.
                LEARNING.end_session(request.device_id, None)
                raise
            if needs_personal_context:
                LEARNING.end_session(request.device_id, None)
                return unavailable_reply(access)
        agent = PetAgent(STORE, turn_model)
        permit = None
        if request.mode == "english":
            permit = LEARNING.before_turn(request.device_id, request.person_id,
                                          session_id=request.session_id or str(uuid.uuid4()),
                                          use_photo=bool(photo_path) or request._vision_used)
            initially_save_learning = permit["settings"]["save_progress"]
            if not permit["allowed"]:
                if permit["reason"] == "daily_limit":
                    result = agent.finish_learning(permit)
                    result["personal_access"] = access.public_status()
                    return result
                if permit["reason"] == "photo_learning_disabled":
                    raise HTTPException(409, "请先在家长英语设置中允许照片学习")
                raise HTTPException(409, "此学习回合已暂停或有请求正在处理")
        else:
            LEARNING.end_session(request.device_id, request.person_id)
        person_name = None
        if request.person_id:
            try:
                member = REGISTRATIONS.member(request.person_id)
                person_name = member["display_name"]
                draft_member_generation = member.get("generation")
            except HTTPException:
                pass
        try:
            observation = MODEL.describe(photo_path, "简短描述这张照片里看得清的物品与颜色。") if photo_path else request._vision_observation
            result = ({**local_answer, "intent": "chat", "action": "", "value": "",
                       "english_kind": "practice" if local_answer.get("lesson") else "question"}
                      if local_answer else agent.reply(request.device_id, request.text, request.person_id, request.mode,
                                 session_id=request.session_id, person_name=person_name,
                                 learning_settings=permit["settings"] if permit else None,
                                 record_learning=False, vision_context=observation,
                                 approved_actions=tuple(action_snapshot), turn_commit=turn_guard,
                                 personal_request=needs_personal_context,
                                 session_lesson=session_lesson if permit else None,
                                 previous_guest_turn=request._previous_guest_turn,
                                 heard_alternatives=request._heard_alternatives,
                                 review_reply=request._from_device))
            result["personal_access"] = access.public_status()
        except BaseException:
            if permit:
                LEARNING.after_turn(request.device_id, request.person_id, turn_id=permit["turn_id"], success=False)
            raise
        if result.get("business_status", {}).get("code") == "personal_access_unavailable":
            # A missed classification may still be caught by the agent's
            # anonymous operation guard. Close the permit without opening a new
            # learning activity window or recording a lesson/history.
            if permit:
                LEARNING.after_turn(request.device_id, request.person_id,
                                    turn_id=permit["turn_id"], success=False)
            LEARNING.end_session(request.device_id, request.person_id)
            return unavailable_reply(access)
        if permit:
            state = LEARNING.after_turn(request.device_id, request.person_id, turn_id=permit["turn_id"])
            if not state["allowed"]:
                if state["reason"] == "daily_limit":
                    # Exit the memory transaction before generating a farewell;
                    # the over-budget answer's writes must roll back together.
                    raise LearningFinished(state, access)
                raise HTTPException(409, "学习回合已取消，请重新开始")
            if (photo_path or request._vision_used) and not state["settings"]["photo_learning"]:
                raise HTTPException(409, "照片学习已关闭")
            result["learning"] = state
            if (initially_save_learning and request.person_id
                    and result.get("intent") not in {"forget_memory", "action_draft"}):
                # Hold this short gate through the outer memory transaction's
                # actual commit/rollback. No model runs after acquiring it.
                allowed = settings_gates.enter_context(
                    LEARNING.saving_permission(request.device_id, request.person_id))
                if allowed:
                    if result.get("lesson"):
                        lesson = result["lesson"]
                        STORE.record_english(request.device_id, request.person_id, lesson["target_word"],
                                             lesson["text_match"], activity=lesson["activity"],
                                             advance=lesson.get("advance", True),
                                             review_checkpoint=lesson.get("review_checkpoint") if lesson.get("advance", True) else None)
                    STORE.append_conversation(request.device_id, request.person_id, request.text, result["reply"],
                                              "english", request.session_id)
        return result
    def infer():
        try:
            with ExitStack() as settings_gates:
                if request._from_device:
                    settings_gates.enter_context(voice_remote_only())
                with STORE.commit_scope(turn_guard):
                    return infer_turn(settings_gates)
        except LearningFinished as finished:
            turn_guard.check()
            result = PetAgent(STORE, MODEL).finish_learning(finished.state)
            result["personal_access"] = finished.access.public_status()
            return result
    try:
        answer = await INFERENCE.run(infer)
    except RevisionConflict as exc:
        raise HTTPException(409, "资料或生活流程已更新，请重新发出指令") from exc
    except (ModelResponseError, PersonalAccessError) as exc:
        raise HTTPException(503, MODEL_RETRY_DETAIL) from exc
    except asyncio.CancelledError:
        turn_guard.cancel()
        if request.mode == "english" or not request.person_id:
            LEARNING.end_session(request.device_id, request.person_id)
        raise
    if request.mode == "english" and request.session_id and answer.get("lesson"):
        lesson = answer["lesson"]
        STORE.child_sessions.record_lesson(request.device_id, request.session_id, request.person_id,
                                     lesson["target_word"], lesson["activity"],
                                     advance=lesson.get("advance", True))
    elif (request.mode == "english" and request.session_id and answer.get("vocabulary")
          and answer.get("learning", {}).get("allowed")):
        # An explicitly chosen word becomes the temporary topic without
        # claiming a practice result or writing a member's saved progress.
        current = STORE.child_sessions.lesson(request.device_id, request.session_id, request.person_id)
        if not current or current["word"] != answer["vocabulary"][0]:
            STORE.child_sessions.record_lesson(request.device_id, request.session_id, request.person_id,
                                               answer["vocabulary"][0], "introduce")
    if request.session_id and answer.get("learning_closed"):
        STORE.child_sessions.finish(request.device_id, request.session_id)
    name_after = STORE.pet(request.device_id)["name"]
    if name_after != name_before and request.device_id in HUB.connections:
        try:
            answer["name_sync"] = await HUB.command(request.device_id, "set_name", name=name_after)
        except HTTPException as exc:
            answer["name_sync"] = {"ok": False, "error": exc.detail}
    if stationary_voice and answer.get("intent") in {"action", "action_sequence", "action_draft"}:
        return {"reply": STATIONARY_REPLY, "intent": "chat", "action": "", "value": "",
                "reply_source": "stationary_play"}
    if answer.get("intent") == "action":
        try:
            before_send = None
            if answer.get("greeting"):
                def before_send():
                    profile = PetProfile(STORE).get(request.device_id)
                    return (profile["revision"] == answer.get("profile_revision")
                            and profile["greeting_action"] == answer.get("action"))
                if not before_send():
                    raise HTTPException(409, "家长已修改招呼动作，请重新打招呼")
            answer["device_receipt"] = await execute_chat_action(request.device_id, answer.get("action"), action_snapshot,
                                                                before_send=before_send)
        except HTTPException as exc:
            answer["device_receipt"] = {"ok": False, "error": exc.detail}
    elif answer.get("intent") == "action_sequence":
        try:
            sequence = answer.get("actions")
            if (not isinstance(sequence, list) or not 1 <= len(sequence) <= 4
                    or any(not isinstance(item, str) or item not in {"wave", "sit", "stretch", "hug"} for item in sequence)):
                raise HTTPException(422, "无效的短动作组合")
            job = start_action_job(request.device_id, "本轮动作组合", [{"action": action} for action in sequence])
            answer["device_receipt"] = await await_action_job(job)
        except HTTPException as exc:
            answer["device_receipt"] = {"ok": False, "error": exc.detail}
    elif answer.get("intent") == "action_draft":
        try:
            created = await ACTION_DRAFTS.propose(request.device_id, request.text, request.session_id, request.person_id)
            # The draft DB may already have committed if cancellation wins while
            # waiting here. Keep that unapproved draft; do not add an unseen reply
            # or attempt a cross-database rollback.
            await record_draft_history(request, answer, draft_member_generation, initially_save_learning)
        except DraftError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        except asyncio.CancelledError:
            if request.mode == "english":
                LEARNING.end_session(request.device_id, request.person_id)
            raise
        answer["draft"] = created["draft"]
        answer["executed"] = False
    return answer


async def plan_camera_request(text):
    def infer():
        global MODEL
        if MODEL is None:
            MODEL = new_model()
        return MODEL.requests_camera(text)
    return await INFERENCE.run(infer)


async def capture_dialogue_photo(device_id, *, before_send=None, deadline=None):
    try:
        result = await HUB.command(device_id, "photo", retain_photo=False, photo_attempts=1,
                                   before_send=before_send, deadline=deadline)
        if not result.get("ok") or not result.get("_photo_bytes"):
            raise RuntimeError("本轮拍照失败")
        return result
    except HTTPException as exc:
        raise RuntimeError("本轮摄像头不可用") from exc


async def describe_dialogue_photo(data, question):
    def infer():
        global MODEL
        if MODEL is None:
            MODEL = new_model()
        return MODEL.describe_bytes(data, question)
    return await INFERENCE.run(infer)


DIALOGUE_CAMERA = CameraDialogue(plan_camera_request, capture_dialogue_photo, describe_dialogue_photo)


def approved_chat_actions():
    """Bound the prompt and reject stale/corrupt stored definitions defensively."""
    approved = {}
    for name, steps in STORE.actions().items():
        if name in SAFE_ACTIONS:
            continue  # Saved names cannot shadow built-in movement or stop.
        try:
            SaveAction(name=name, steps=steps)
            if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
                continue
            if any(not isinstance(step, dict) or set(step) - {"action", "wait_ms"}
                   or (step.get("action") is not None and not isinstance(step["action"], str))
                   or type(step.get("wait_ms", 0)) is not int or not 0 <= step.get("wait_ms", 0) <= 3000
                   for step in steps):
                continue
            validate_steps(steps)
        except (ValueError, TypeError, HTTPException):
            continue
        approved[name] = deepcopy(steps)
        if len(approved) >= PetAgent.CUSTOM_ACTION_LIMIT:
            break
    return approved


async def execute_chat_action(device_id: str, name: str, approved: dict[str, list[dict]], *, before_send=None):
    if not isinstance(name, str):
        raise HTTPException(422, "不支持的动作")
    if name in SAFE_ACTIONS:
        return await execute_safe_action(device_id, name, before_send=before_send)
    if name not in approved:
        raise HTTPException(422, "这个动作尚未由家长保存")
    # No await between comparison and job creation. Edits during inference
    # invalidate the proposal; later edits cannot alter this private snapshot.
    if STORE.actions().get(name) != approved[name]:
        raise HTTPException(409, "动作定义已更新，请重新发出指令")
    job = start_action_job(device_id, name, deepcopy(approved[name]), before_send=before_send)
    return await await_action_job(job)


@app.get("/api/actions", dependencies=[Depends(require_parent)])
async def actions():
    return STORE.actions()


@app.put("/api/actions", dependencies=[Depends(require_parent)])
async def save_action(action: SaveAction):
    steps = [step.model_dump() for step in action.steps]
    validate_steps(steps)
    STORE.save_action(action.name, steps)
    return {"saved": action.name}


@app.delete("/api/actions/{name}", dependencies=[Depends(require_parent)])
async def delete_action(name: str, request: DeleteAction):
    try:
        await asyncio.to_thread(delete_saved_action, name, [step.model_dump() for step in request.expected_steps])
    except ApprovedActionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"deleted": name}


def start_action_job(device_id: str, name: str, steps: list[dict], *, before_send=None, deadline=None,
                     require_control_probe=False):
    if (STATIONARY_PLAY or HUB.status.get(device_id, {}).get("stationary_play") is True) and \
            any(step.get("action") not in {None, "stop"} for step in steps):
        raise HTTPException(409, "安静陪玩期间暂停动作")
    if device_id in STOPPING and not STOPPING[device_id].done():
        raise HTTPException(409, "正在停止，请等待设备回执")
    if device_id in RUNNING and not RUNNING[device_id].done():
        raise HTTPException(409, "已有动作正在执行")
    if not require_control_probe and device_id not in HUB.connections:
        raise HTTPException(503, "Puppy 当前不在线")
    validate_steps(steps)
    if not require_control_probe and any(step.get("action") not in {None, "stop"} for step in steps) and \
            HUB.status.get(device_id, {}).get("motion_protocol_version") != 2:
        raise HTTPException(409, "请先更新 Puppy 动作回执固件并等待设备状态")

    async def execute():
        needs_stop = not require_control_probe
        try:
            send_check = before_send
            if require_control_probe:
                socket = await HUB.control_for_confirmed_motion(device_id)
                send_check = lambda: (HUB.connections.get(device_id) is socket
                                      and (before_send is None or before_send() is True))
            for index, step in enumerate(steps):
                ACTION_JOBS[device_id].update(step=index + 1, total=len(steps))
                if step.get("action"):
                    kind = "stop" if step["action"] == "stop" else "action"
                    needs_stop = True
                    receipt = await HUB.command(device_id, kind, before_send=send_check, deadline=deadline,
                                                **({"name": step["action"]} if kind == "action" else {}))
                    if not receipt.get("ok"):
                        if receipt.get("phase") == "cancelled":
                            detail = receipt.get("error", "动作已停止")
                            ACTION_JOBS[device_id].update(state="cancelled", error=detail)
                            return {"ok": False, "phase": "cancelled", "error": detail}
                        raise RuntimeError(receipt.get("error", "设备拒绝动作"))
                if step.get("wait_ms"):
                    await asyncio.sleep(step["wait_ms"] / 1000)
            ACTION_JOBS[device_id].update(state="completed")
            return {"ok": True, "phase": "completed", "name": name}
        except asyncio.CancelledError:
            ACTION_JOBS[device_id].update(state="cancelled")
            raise
        except Exception as exc:
            detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            ACTION_JOBS[device_id].update(state="failed", error=detail)
            return {"ok": False, "phase": "failed", "error": detail}
        finally:
            if needs_stop:
                try:
                    await HUB.command(device_id, "stop")
                except HTTPException:
                    pass

    ACTION_JOBS[device_id] = {"name": name, "state": "running", "step": 0, "total": len(steps)}
    task = asyncio.create_task(execute())
    RUNNING[device_id] = task
    return task


@app.post("/api/devices/{device_id}/action/{name}", dependencies=[Depends(require_parent)])
async def run_action(device_id: str, name: str):
    if name == "stop":
        return await stop(device_id)
    steps = [{"action": name, "wait_ms": 0}] if name in SAFE_ACTIONS else STORE.actions().get(name)
    if steps is None:
        raise HTTPException(404, "没有这个动作")
    start_action_job(device_id, name, steps)
    return {"accepted": True, "device_id": device_id, "name": name}


@app.post("/api/devices/{device_id}/stop", dependencies=[Depends(require_parent)])
async def stop(device_id: str):
    return await stop_device(device_id)


async def stop_device(device_id: str, *, before_send=None, deadline=None, drain_on_cancel=False):
    if before_send is not None and before_send() is not True:
        raise HTTPException(409, "本次停止请求已过期")
    current = STOPPING.get(device_id)
    if current is None or current.done():
        owned = RUNNING.get(device_id)
        gate = {"parent": before_send is None}
        STOP_REQUESTS[device_id] = gate

        def permitted():
            return gate["parent"] or ((deadline is None or asyncio.get_running_loop().time() < deadline)
                                     and before_send() is True)

        async def finish():
            error = None
            try:
                if not permitted():
                    raise HTTPException(409, "本次停止请求已过期")
                if owned and not owned.done() and not owned.cancelling():
                    owned.cancel()
                # Keep the normal bounded command timeout. Its send-time gate
                # can be upgraded by a concurrent explicit parent stop.
                receipt = await HUB.command(device_id, "stop", before_send=permitted)
            except HTTPException as exc:
                error = exc
            finally:
                if owned and not owned.done() and owned.cancelling():
                    await asyncio.gather(owned, return_exceptions=True)
            if error is not None:
                if gate["parent"]:
                    return await HUB.command(device_id, "stop")
                raise error
            return receipt

        current = asyncio.create_task(finish())
        STOPPING[device_id] = current
        def finished(task):
            if STOPPING.get(device_id) is task:
                STOPPING.pop(device_id, None)
                STOP_REQUESTS.pop(device_id, None)
            if not task.cancelled():
                task.exception()
        current.add_done_callback(finished)
    elif before_send is None:
        STOP_REQUESTS[device_id]["parent"] = True
    # A caller going away cannot admit a replacement while its stop is in flight.
    try:
        return await asyncio.shield(current)
    except asyncio.CancelledError:
        if drain_on_cancel:
            await asyncio.shield(asyncio.gather(current, return_exceptions=True))
        raise


async def execute_safe_action(device_id: str, name: str, *, before_send=None, confirmed_by_voice=False):
    if name == "stop":
        return await stop_device(device_id, before_send=before_send)
    if name not in SAFE_ACTIONS:
        raise HTTPException(422, "不支持的动作")
    job = start_action_job(device_id, name, [{"action": name, "wait_ms": 0}], before_send=before_send,
                           require_control_probe=confirmed_by_voice)
    return await await_action_job(job)


async def await_action_job(job):
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        # The job retains ownership while its bounded stop cleanup completes.
        # A cancelled conversation can release its receive loop immediately.
        if not job.cancelling():
            job.cancel()
        raise


async def execute_gesture_action(device_id, name, *, before_send, deadline):
    if name == "stop":
        return await stop_device(device_id, before_send=before_send, deadline=deadline, drain_on_cancel=True)
    if name not in {"wave", "sit"}:
        raise HTTPException(422, "不支持的手势动作")
    job = start_action_job(device_id, name, [{"action": name}], before_send=before_send, deadline=deadline)
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        if not job.cancelling():
            job.cancel()
        # Session cleanup must wait for this exact job, never a newer owner's.
        await asyncio.shield(asyncio.gather(job, return_exceptions=True))
        raise


@app.post("/api/devices/{device_id}/volume", dependencies=[Depends(require_parent)])
async def set_volume(device_id: str, request: VolumeRequest):
    return await HUB.command(device_id, "set_volume", value=request.volume)


@app.post("/api/devices/{device_id}/sleep-policy", dependencies=[Depends(require_parent)])
async def set_sleep_policy(device_id: str, request: SleepPolicyRequest):
    return await HUB.command(device_id, "set_sleep_policy", enabled=request.enabled)


@app.post("/api/devices/{device_id}/lamp-test", dependencies=[Depends(require_parent)])
async def test_back_lamp(device_id: str, duration_ms: int = 1000):
    if not 1000 <= duration_ms <= 10000:
        raise HTTPException(422, "背灯持续时间须为 1 至 10 秒")
    return await HUB.command(device_id, "lamp_test", duration_ms=duration_ms)


@app.post("/api/devices/{device_id}/audio/stop", dependencies=[Depends(require_parent)])
async def stop_device_audio(device_id: str):
    sessions = tuple(AUDIO_SESSIONS.get(device_id, ()))
    if not sessions:
        return {"ok": True, "phase": "idle"}
    results = await asyncio.gather(*(session.cancel_and_close() for session in sessions))
    return {"ok": True, "phase": "completed" if any(results) else "idle"}


@app.websocket("/v1/control")
async def device_control(socket: WebSocket):
    token = socket.headers.get("authorization", "")
    if not DEVICE_TOKEN or not hmac.compare_digest(token, "Bearer " + DEVICE_TOKEN):
        await socket.close(code=1008)
        return
    device_id = socket.headers.get("device-id", "")
    if not device_id or len(device_id) > 64:
        await socket.close(code=1008)
        return
    await socket.accept()
    previous = HUB.connections.get(device_id)
    connected_at = time.time()
    LOGGER.info("puppy_control_connected epoch_ms=%d device=%r replaced=%s",
                int(connected_at * 1000), device_id, previous is not None)
    if previous is not None:
        HUB.disconnected(device_id, previous)
        try:
            await previous.close(code=1012)
        except (RuntimeError, WebSocketDisconnect):
            pass
    HUB.connections[device_id] = socket
    HUB.status.pop(device_id, None)  # Capabilities belong to this connection only.
    HUB.send_locks[device_id] = asyncio.Lock()
    async def sync_clock():
        while HUB.connections.get(device_id) is socket:
            try:
                receipt = await HUB.command(device_id, "set_clock", epoch_ms=int(time.time() * 1000))
                if not receipt.get("ok"):
                    LOGGER.warning("puppy_clock_sync_rejected device=%r error=%r", device_id, receipt.get("error"))
            except (HTTPException, RuntimeError, WebSocketDisconnect) as exc:
                LOGGER.warning("puppy_clock_sync_failed device=%r error=%s", device_id, exc)
            await asyncio.sleep(6 * 3600)
    clock_sync_task = asyncio.create_task(sync_clock())
    async def sync_name():
        try:
            await HUB.command(device_id, "set_name", name=STORE.pet(device_id)["name"])
        except HTTPException:
            pass
    asyncio.create_task(sync_name())
    incoming_photo: dict | None = None
    disconnect_code: int | None = None
    try:
        while True:
            event = await socket.receive()
            if event["type"] == "websocket.disconnect":
                disconnect_code = event.get("code")
                break
            if HUB.connections.get(device_id) is not socket:
                break
            binary = event.get("bytes")
            if binary is not None:
                if incoming_photo is not None:
                    incoming_photo["data"].extend(binary)
                    LOGGER.info("photo chunk %s bytes=%d/%d", incoming_photo["id"],
                                len(incoming_photo["data"]), incoming_photo["length"])
                    if len(incoming_photo["data"]) > incoming_photo["length"]:
                        incoming_photo = None
                    else:
                        async with HUB.send_locks[device_id]:
                            await socket.send_json({"type": "photo_ack", "id": incoming_photo["id"],
                                                    "received": len(incoming_photo["data"])})
                continue
            if not event.get("text"):
                continue
            try:
                packet = json.loads(event["text"])
            except json.JSONDecodeError:
                continue
            if not isinstance(packet, dict):
                continue
            if packet.get("type") == "status":
                if isinstance(packet.get("status"), dict):
                    HUB.status[device_id] = {**packet["status"], "observed_at": time.time()}
            elif packet.get("type") == "photo_begin":
                photo_id = packet.get("id", "")
                length = packet.get("length")
                if isinstance(photo_id, str) and HUB.owns_pending(device_id, socket, photo_id, "photo") and \
                        type(length) is int and 0 < length <= 512000:
                    incoming_photo = {"id": photo_id, "length": length, "data": bytearray()}
                    LOGGER.info("photo begin %s bytes=%d", photo_id, length)
            elif packet.get("type") == "photo_end":
                if incoming_photo and HUB.owns_pending(device_id, socket, incoming_photo["id"], "photo") and \
                        packet.get("id") == incoming_photo["id"] and \
                        len(incoming_photo["data"]) == incoming_photo["length"]:
                    try:
                        HUB.receive_photo(device_id, socket, incoming_photo["id"], bytes(incoming_photo["data"]))
                        LOGGER.info("photo received %s", incoming_photo["id"])
                    except Exception as exc:
                        LOGGER.warning("photo save failed: %s", exc)
                incoming_photo = None
            elif packet.get("type") == "receipt":
                HUB.receipt(device_id, socket, packet)
    except WebSocketDisconnect as exc:
        disconnect_code = exc.code
    finally:
        clock_sync_task.cancel()
        current = HUB.connections.get(device_id) is socket
        last_report = HUB.status.get(device_id, {}).get("observed_at") if current else None
        closed_at = time.time()
        report_age_ms = (int((closed_at - last_report) * 1000)
                         if type(last_report) in (int, float) else None)
        LOGGER.info("puppy_control_closed epoch_ms=%d device=%r current=%s code=%r "
                    "session_age_ms=%d report_age_ms=%r",
                    int(closed_at * 1000), device_id, current, disconnect_code,
                    int((closed_at - connected_at) * 1000), report_age_ms)
        HUB.disconnected(device_id, socket)
        if current:
            HUB.connections.pop(device_id, None)
            HUB.send_locks.pop(device_id, None)
            HUB.status.pop(device_id, None)


@app.websocket("/v1/audio")
async def device_audio(socket: WebSocket):
    token = socket.headers.get("authorization", "")
    if not DEVICE_TOKEN or not hmac.compare_digest(token, "Bearer " + DEVICE_TOKEN):
        await socket.close(code=1008)
        return
    device_id = socket.headers.get("device-id", "")
    if not device_id or len(device_id) > 64:
        await socket.close(code=1008)
        return
    await socket.accept()
    audio_session = str(uuid.uuid4())
    conversation_session = device_audio_conversation_id(device_id)
    guest_turn = None
    next_guest_turn = None
    pending_motion = None
    next_motion = None
    motion_suggestion = None
    motion_waiting = False
    def record_audio_event(name, details):
        nonlocal guest_turn, next_guest_turn, vocabulary, pending_motion, next_motion
        if name in {"motion_voice_stop", "abort_received", "parent_audio_stopped", "closed"}:
            pending_motion = None
            next_motion = None
        if name == "motion_voice_stop":
            # The interrupted answer was not spoken. Neither its English
            # voice override nor its pending public history belongs to the ack.
            next_guest_turn = None
            vocabulary = None
        if name == "tts_sent" and next_guest_turn is not None:
            guest_turn = next_guest_turn, time.monotonic()
        if name == "tts_sent" and next_motion is not None:
            pending_motion = next_motion, time.monotonic()
            next_motion = None
        # Structural diagnostics only. Never log PCM, transcripts or replies.
        LOGGER.info("puppy_audio_event session=%s device=%r stage=%s details=%s",
                    audio_session, device_id, name, details)

    speech = SpeechServices(WHISPER_PATH)
    turn_deadline = None
    learning_people = set()
    asr_clarification = None
    vocabulary = None
    heard_alternatives = ()

    async def prepare(recorded):
        nonlocal turn_deadline, asr_clarification, next_guest_turn, guest_turn
        nonlocal vocabulary, heard_alternatives
        turn_deadline = asyncio.get_running_loop().time() + 45
        next_guest_turn = None
        asr_clarification = None
        vocabulary = None
        heard_alternatives = ()
        async with asyncio.timeout_at(turn_deadline):
            enrollment = await REGISTRATIONS.consume_enrollment(device_id, recorded)
        if enrollment is not None:
            guest_turn = None
            # This is an operational status, never a model-generated conversation.
            return {"reply": "声音样本已登记。" if enrollment["state"] == "completed" else "这次未完成登记，请查看家长面板。"}
        return None

    async def respond(recorded, transcript):
        nonlocal motion_waiting
        motion_waiting = False
        try:
            # Leave time to speak even when a provider is slow.
            async with asyncio.timeout(20):
                return await respond_turn(recorded, transcript)
        except HTTPException as exc:
            if exc.status_code != 503 or exc.detail != MODEL_RETRY_DETAIL:
                raise
        except asyncio.TimeoutError:
            pass
        record_audio_event("reply_unavailable", {})
        if motion_waiting:
            return {"reply": "没有收到动作完成确认，请用机身按键停止。", "reply_source": "voice_motion_timeout"}
        return {"reply": "刚才没有答出来，请再问我一次好吗？"}

    async def respond_turn(recorded, transcript):
        nonlocal guest_turn, next_guest_turn, vocabulary, pending_motion, next_motion, motion_waiting
        if asr_clarification:
            pending_motion = None
            if motion_suggestion is not None and motion_suggestion[0] is recorded:
                next_motion = motion_suggestion[1]
                guest_turn = next_guest_turn = vocabulary = None
                LEARNING.end_session(device_id, None)
            return ({"reply": asr_clarification, "reply_source": "voice_motion_confirmation"}
                    if next_motion else {"reply": asr_clarification})
        if asks_about_sharing_with_strangers(transcript):
            pending_motion = None
            guest_turn = None
            LEARNING.end_session(device_id, None)
            return {"reply": "先别告诉陌生人。去找爸爸妈妈，或者你信任的大人帮忙。",
                    "reply_source": "child_safety"}
        action = motion_command(transcript)
        confirmed_by_voice = False
        if pending_motion:
            suggested, when = pending_motion
            pending_motion = None
            confirmed = motion_confirmation(transcript)
            if time.monotonic() - when <= 25 and confirmed is not None:
                if not confirmed:
                    return {"reply": "好，不做这个动作。", "reply_source": "voice_motion_confirmation"}
                action = suggested
                confirmed_by_voice = True
        if action:
            if STATIONARY_PLAY and action != "stop":
                return {"reply": STATIONARY_REPLY, "reply_source": "stationary_play"}
            guest_turn = next_guest_turn = vocabulary = None
            LEARNING.end_session(device_id, None)
            try:
                motion_waiting = True
                if confirmed_by_voice:
                    receipt = await execute_safe_action(device_id, action, confirmed_by_voice=True)
                else:
                    receipt = await execute_safe_action(device_id, action)
            except HTTPException as exc:
                receipt = {"ok": False, "error": exc.detail}
            motion_waiting = False
            complete = receipt.get("ok") is True and receipt.get("phase") == "completed"
            record_audio_event("voice_motion_finished", {"action": action, "completed": complete})
            replies = {"wave": "挥手完成啦。", "sit": "坐下完成啦。", "stretch": "伸个懒腰完成啦。",
                       "hug": "抱抱完成啦。", "stop": "好，停下了。"}
            return {"reply": replies[action] if complete else "没有收到完成确认，请用机身按键停止。",
                    "intent": "action", "action": action, "value": "", "reply_source": "voice_motion",
                    "device_receipt": receipt}
        improvement = await HERMES_TASKS.respond(device_id, transcript)
        if improvement is not None:
            guest_turn = None
            LEARNING.end_session(device_id, None)
            record_audio_event("improvement_handled", {})
            return improvement
        person_id = None
        mode = STORE.child_sessions.route(device_id, conversation_session, transcript,
                                    STORE.conversation_mode(device_id))
        if stops_activity(transcript) and re.search(r"故事|英[语語]|休息|不(?:学|學|玩|想玩|想听|想聽)", transcript):
            guest_turn = None
            LEARNING.end_session(device_id, person_id)
            return {"reply": "好，先休息一下，想聊别的也可以。", "intent": "chat",
                    "reply_source": "activity_stop", "action": "", "value": ""}

        def allow_visual():
            return mode != "english" or LEARNING.get_settings(device_id, None)["photo_learning"]

        visual_started = time.monotonic()
        record_audio_event("visual_gate_started", {})
        try:
            text_question = (PetAgent.english_word_question(transcript) or public_animal_body_count(transcript)
                             or public_animal_group_count(transcript)
                             or PetAgent.english_followup(transcript, guest_turn[0] if guest_turn else None))
            visual = (None if text_question else await DIALOGUE_CAMERA.prepare(
                device_id, transcript, allowed=allow_visual, deadline=turn_deadline))
        finally:
            record_audio_event("visual_gate_finished", {
                "elapsed_ms": int((time.monotonic() - visual_started) * 1000)})
        if visual and visual["observation"].get("available") is False:
            guest_turn = None
            LEARNING.end_session(device_id, person_id)
            record_audio_event("visual_unavailable", {})
            if not allow_visual():
                return {"reply": "要家长允许看图，我才能帮你看。"}
            return {"reply": "相机暂时不可用。请稍后再试"}

        request = ChatRequest(device_id=device_id, text=transcript, session_id=conversation_session,
                              person_id=person_id, mode=mode)
        request._from_device = True
        request._heard_alternatives = heard_alternatives
        if visual:
            request._vision_observation = visual["observation"]
            request._vision_used = visual["used_photo"]
            request._vision_times = visual.get("times")
        elif not person_id and guest_turn is not None and time.monotonic() - guest_turn[1] < 40:
            request._previous_guest_turn = guest_turn[0]
        guest_turn = None
        try:
            reply_started = time.monotonic()
            record_audio_event("reply_generation_started", {})
            try:
                answer = await chat(request)
                if answer.get("business_status", {}).get("code") == "personal_access_unavailable":
                    answer["reply"] = "个人记录可以请爸爸妈妈帮你查看。我们可以接着聊天。"
                if answer.get("reply_source") == "vocabulary":
                    vocabulary = answer["vocabulary"]
                    record_audio_event("reply_vocabulary", {})
                elif answer.get("reply_source") == "public_fact":
                    record_audio_event("reply_public_fact", {})
                if (not person_id and not visual and request.mode in {"chat", "story", "english"}
                        and answer.get("intent") == "chat"
                        and answer.get("reply_source") in {"model", "vocabulary", "public_fact"}
                        and not answer.get("business_status")):
                    previous = request._previous_guest_turn
                    same_mode = previous and previous["mode"] == request.mode
                    history = (previous.get("history") or [
                        {key: previous[key] for key in ("user", "assistant")}]) if same_mode else []
                    next_guest_turn = {"user": transcript, "assistant": answer["reply"], "mode": request.mode,
                                       "history": (history + [{"user": transcript, "assistant": answer["reply"]}])[-4:]}
                    if request.mode == "story":
                        next_guest_turn["story_request"] = (previous.get("story_request", previous["user"])
                                                            if same_mode and not requests_story(transcript) else transcript)
                        character = replacement_character(transcript, previous["assistant"] if same_mode else "")
                        if character or (same_mode and previous.get("story_character")):
                            next_guest_turn["story_character"] = character or previous["story_character"]
                    taught_word = answer.get("vocabulary") or (
                        (answer["lesson"]["target_word"], answer["lesson"]["meaning"]) if answer.get("lesson") else None)
                    if not taught_word and same_mode and request.mode == "english":
                        previous_word = previous.get("vocabulary")
                        if previous_word and any(PetAgent._contains_word(answer["reply"], form)
                                                 for form in (previous_word[0], previous_word[0]+"s", previous_word[0]+"es")):
                            taught_word = previous_word
                    if taught_word:
                        next_guest_turn["vocabulary"] = taught_word
                return answer
            finally:
                record_audio_event("reply_generation_finished", {
                    "elapsed_ms": int((time.monotonic() - reply_started) * 1000)})
        finally:
            learning_people.add(request.person_id)

    async def primary_transcript(pcm):
        try:
            async with asyncio.timeout(10):
                if pending_motion and time.monotonic() - pending_motion[1] <= 25:
                    return await speech.transcribe_async(pcm, language="zh")
                return await speech.transcribe_async(pcm)
        except TimeoutError:
            record_audio_event("primary_asr_timeout", {})
        except (OSError, RuntimeError, ValueError):
            record_audio_event("primary_asr_unavailable", {})
        return ""

    async def secondary_transcript(pcm):
        if not APPLE_ASR_BINARY:
            return ""
        try:
            return await apple_transcribe_pcm(pcm, Path(APPLE_ASR_BINARY), timeout=4)
        except (OSError, ValueError, RuntimeError, TimeoutError):
            record_audio_event("secondary_asr_unavailable", {})
            return ""

    async def transcribe(recorded):
        nonlocal asr_clarification, heard_alternatives, motion_suggestion
        motion_suggestion = None
        quiet = bool(APPLE_ASR_BINARY) and pcm_peak_frame_rms(recorded) < QUIET_VOICE_PEAK_RMS
        pcm = boost_quiet_speech(recorded)
        secondary_word = None
        if quiet:
            async with asyncio.TaskGroup() as recognizers:
                primary_task = recognizers.create_task(primary_transcript(pcm))
                secondary = await secondary_transcript(pcm)
                secondary_word = PetAgent.english_word_question(secondary)
                secondary_motion = motion_command(secondary)
                anatomy = public_animal_body_count(secondary)
                if secondary_word or secondary_motion or (anatomy and PetAgent.animal_count_reply(*anatomy)):
                    done, _ = await asyncio.wait({primary_task}, timeout=QUIET_ASR_CROSSCHECK_SECONDS)
                    if not done:
                        primary_task.cancel()
                        record_audio_event("asr_vocabulary_ready" if secondary_word else
                                           "asr_motion_ready" if secondary_motion else "asr_public_fact_ready", {})
                        return secondary
                primary = await primary_task
        else:
            primary = await primary_transcript(pcm)
            secondary = await secondary_transcript(pcm)

        heard_alternatives = tuple(dict.fromkeys(text[:300] for text in (primary, secondary) if text))
        if pending_motion and time.monotonic() - pending_motion[1] <= 25:
            primary, unclear = reconcile_confirmation(primary, secondary)
            if unclear:
                motion_suggestion = (recorded, pending_motion[0])
                label = {"stretch": "伸懒腰", "hug": "抱抱"}[pending_motion[0]]
                asr_clarification = f"确认没听清。要{label}请说可以，不做请说不要。"
                return primary or secondary
        selected_motion, motion_clarification = reconcile_motion(primary, secondary, short_audio=len(recorded) <= 4 * 32000)
        if motion_clarification:
            asr_clarification = motion_clarification
            record_audio_event("asr_motion_clarification", {})
            return primary
        if selected_motion != primary:
            primary = selected_motion
            record_audio_event("asr_motion_recovered", {})
        elif not STATIONARY_PLAY and (suggested := unclear_motion(primary, secondary, short_audio=len(recorded) <= 4 * 32000)):
            # Only the matching turn may offer this suggestion; actual
            # confirmation is armed by tts_sent after the question is spoken.
            motion_suggestion = (recorded, suggested)
            label = {"stretch": "伸个懒腰", "hug": "抱抱"}[suggested]
            asr_clarification = f"动作没听清，你是想让我{label}吗？"
            record_audio_event("asr_motion_confirmation", {})
            return primary or secondary
        if not primary and secondary:
            primary = secondary
            record_audio_event("secondary_asr_recovered", {})
        elif primary and secondary:
            asr_clarification = clarification_for_transcripts(primary, secondary)
            primary_word = PetAgent.english_word_question(primary) if secondary_word else None
            if primary_word and primary_word != secondary_word:
                asr_clarification = f"你问的是{primary_word[1]}，还是{secondary_word[1]}的英文呀？"
            if asr_clarification:
                record_audio_event("asr_clarification", {})
            else:
                preferred = prefer_clear_arithmetic_question(primary, secondary)
                if preferred != primary:
                    primary = preferred
                    record_audio_event("asr_math_repaired", {})
                else:
                    previous_question = (guest_turn[0]["user"] if guest_turn and guest_turn[0]["mode"] == "chat"
                                         and time.monotonic() - guest_turn[1] < 40 else "")
                    preferred, asr_clarification = resolve_body_count(primary, secondary, previous_question)
                    if asr_clarification:
                        record_audio_event("asr_clarification", {})
                    elif preferred != primary:
                        primary = preferred
                        record_audio_event("asr_count_question_recovered", {})
                    elif quiet:
                        preferred = prefer_quiet_translation_question(primary, secondary)
                        event = "asr_quiet_translation_recovered"
                        if preferred == primary:
                            preferred = prefer_quiet_story_request(primary, secondary)
                            event = "asr_quiet_story_recovered"
                        if preferred == primary:
                            preferred = prefer_clear_activity_control(primary, secondary)
                            event = "asr_activity_control_recovered"
                        if preferred != primary:
                            primary = preferred
                            record_audio_event(event, {})
        previous = guest_turn[0] if guest_turn and time.monotonic() - guest_turn[1] < 40 else None
        if not previous and primary.strip().rstrip("。.!！?？") in {"为什么", "為什麼", "怎么", "怎麼", "是什么", "是什麼"}:
            asr_clarification = "问题后半句我没听清，可以完整说一次吗？"
        if not asr_clarification and previous and previous["mode"] == "english" and previous.get("vocabulary"):
            spoken_word = next(((word, meaning) for word, meaning in PetAgent.ENGLISH_WORDS
                                if re.fullmatch(r"\W*" + word + r"\W*", primary, re.I)), None)
            if spoken_word and spoken_word[0] != previous["vocabulary"][0]:
                asr_clarification = (f"刚才这个词是{spoken_word[1]}的英文。"
                                     f"我们正在练{previous['vocabulary'][1]}这个词，可以再试一次吗？")
        if (not asr_clarification and previous and previous["mode"] == "english"
                and previous.get("vocabulary") and len(recorded) <= 16000 * 2 * 3
                and not motion_command(primary)
                and (0 < len(re.sub(r"[\W_]+", "", primary)) <= 4
                     or re.fullmatch(r"\W*[A-Za-z]{1,16}(?:\s+(?:ah|uh|er)){0,2}\W*", primary, re.I))
                and not any(re.fullmatch(r"\W*" + word + r"\W*", primary, re.I)
                            for word, _ in PetAgent.ENGLISH_WORDS)
                and not stops_activity(primary)
                and primary.strip().rstrip("。.!！?？") not in {"好", "好的", "你好", "谢谢", "謝謝", "嗯"}):
            # Only a short follow-up to a word just taught gets an English
            # retry. A recognizer must actually hear that word; do not assume
            # the child repeated it from the lesson state alone.
            english = ""
            if APPLE_ASR_BINARY:
                try:
                    english = await apple_transcribe_pcm(pcm, Path(APPLE_ASR_BINARY), timeout=1.5, locale="en_US")
                    if not re.search(r"[a-zA-Z]", english):
                        english = ""
                except (OSError, RuntimeError, ValueError, TimeoutError):
                    record_audio_event("english_native_asr_unavailable", {})
            try:
                if not english:
                    async with asyncio.timeout(2.5):
                        english = await speech.transcribe_async(pcm, language="en")
            except (OSError, RuntimeError, ValueError, TimeoutError):
                record_audio_event("english_asr_unavailable", {})
            if re.fullmatch(r"\W*(?:(?:a|an|the)\s+)?" + re.escape(previous["vocabulary"][0]) + r"\W*", english, re.I):
                primary = previous["vocabulary"][0]
                record_audio_event("asr_english_repeat_recovered", {})
            else:
                asr_clarification = "这个英文词我没听清，可以再说一次吗？"
        if not asr_clarification:
            asr_clarification = numeric_only_clarification(primary)
            if asr_clarification:
                record_audio_event("asr_numeric_clarification", {})
        return primary

    async def synthesize(spoken):
        if vocabulary is not None:
            word, meaning = vocabulary
            return await speech.synthesize_vocabulary_async(f"{meaning}的英文是", word)
        return await speech.synthesize_async(spoken)

    async def finish_turn():
        nonlocal next_guest_turn, next_motion, motion_suggestion
        next_guest_turn = None
        next_motion = None
        motion_suggestion = None

    def current_motion():
        owner = RUNNING.get(device_id)
        return owner if owner is not None and not owner.done() else None

    async def stop_moving(recorded, owner, *, resolving_negation=False):
        if current_motion() is not owner:
            return None
        pcm = boost_quiet_speech(recorded)
        try:
            if APPLE_ASR_BINARY:
                text = await apple_transcribe_pcm(pcm, Path(APPLE_ASR_BINARY), timeout=1)
            else:
                async with asyncio.timeout(1.5):
                    text = await speech.transcribe_async(pcm, language="zh")
        except (OSError, RuntimeError, ValueError, TimeoutError):
            record_audio_event("motion_stop_asr_unavailable", {})
            return None
        if declined_stop(text):
            return {"ignore": True}
        if resolving_negation and not explicit_hold_still(text):
            return None
        if stop_negation_prefix(text):
            return {"negation_prefix": True}
        if not motion_stop_command(text) or current_motion() is not owner:
            return None
        record_audio_event("motion_stop_recognized", {})
        try:
            receipt = await stop_device(device_id, before_send=lambda: RUNNING.get(device_id) is owner)
        except HTTPException as exc:
            receipt = {"ok": False, "error": exc.detail}
        confirmed = receipt.get("ok") is True and receipt.get("phase") == "completed"
        record_audio_event("motion_stop_receipt", {"confirmed": confirmed})
        return {"confirmed": confirmed,
                "reply": "好，停下了。" if confirmed else "没有收到停止确认，请用机身按键停止。"}

    session = AudioSession(socket, codec=OpusCodec(), transcribe=transcribe,
                           chat=respond, synthesize=synthesize,
                           prepare_reply=prepare, finish_turn=finish_turn,
                           motion_owner=current_motion, stop_motion=stop_moving,
                           short_answer_expected=lambda: bool(pending_motion and
                               time.monotonic() - pending_motion[1] <= 25),
                           on_event=record_audio_event, session_id=audio_session)
    AUDIO_SESSIONS.setdefault(device_id, set()).add(session)
    try:
        await session.run()
    finally:
        sessions = AUDIO_SESSIONS.get(device_id)
        if sessions is not None:
            sessions.discard(session)
            if not sessions:
                AUDIO_SESSIONS.pop(device_id, None)
        await finish_turn()
        for person_id in learning_people:
            LEARNING.end_session(device_id, person_id)


if __name__ == "__main__":
    if not DEVICE_TOKEN or not PARENT_TOKEN or DEVICE_TOKEN == PARENT_TOKEN:
        raise SystemExit("Set distinct PUPPY_DEVICE_TOKEN and PUPPY_PARENT_TOKEN")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PUPPY_PORT", "8000")),
                timeout_graceful_shutdown=10, **tls_options(os.environ, Path.cwd()))
