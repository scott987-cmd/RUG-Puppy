"""MiniMax M3.1 text adapter; camera images remain on the home MacBook."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent import LocalModel, ModelResponseError


LOGGER = logging.getLogger("uvicorn.error")
_TEXT_MODEL = "MiniMax-M3.1-Flash-Preview"
_VOICE_REMOTE_ONLY = ContextVar("puppy_voice_remote_only", default=False)


@contextmanager
def voice_remote_only():
    """Keep a slow local fallback from consuming a live voice turn."""
    token = _VOICE_REMOTE_ONLY.set(True)
    try:
        yield
    finally:
        _VOICE_REMOTE_ONLY.reset(token)


_ENDPOINTS = {
    "cn": "https://api.minimax.cn/v1/chat/completions",
    "global": "https://api.minimax.io/v1/chat/completions",
}
_CAMERA_SYSTEM = (
    "判断当前这句话是否明确要求机器狗查看眼前的实物、颜色或场景。"
    "例如‘看看我手里是什么’‘这是什么颜色’‘你看到了什么’‘what do you see’为true。"
    "普通知识问题、讲故事、改名、动作指令、问候、回忆过去或明确说不要拍照为false。"
    "不确定则false。不要因为话语提到相机或照片就自动拍照。"
    '只输出JSON {"use_camera":true} 或 {"use_camera":false}，不回答用户问题。'
)


class MiniMaxModel:
    """Use M3.1 for text and lazy local MLX for vision or network fallback."""

    def __init__(self, key_file: str | Path, local_model_path: str, *, region: str = "cn",
                 timeout: float = 8):
        if region not in _ENDPOINTS:
            raise ValueError("Unsupported MiniMax region")
        self.key_file = Path(key_file)
        self.local_model_path = local_model_path
        self.endpoint = _ENDPOINTS[region]
        self.timeout = timeout
        self._local = None

    def _local_model(self):
        if self._local is None:
            self._local = LocalModel(self.local_model_path)
        return self._local

    def _key(self):
        try:
            if self.key_file.is_symlink() or self.key_file.stat().st_mode & 0o077:
                raise ModelResponseError("MiniMax 凭据文件权限不安全")
            key = self.key_file.read_text().strip()
        except OSError as exc:
            raise ModelResponseError("MiniMax 凭据不可用") from exc
        if len(key) < 24:
            raise ModelResponseError("MiniMax 凭据不可用")
        return key

    def _remote(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": _TEXT_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "reasoning_effort": "low",
            "max_completion_tokens": 1024,
            "temperature": 0.2,
        }, ensure_ascii=False).encode()
        request = Request(self.endpoint, data=payload, method="POST", headers={
            "Authorization": "Bearer " + self._key(),
            "Content-Type": "application/json",
        })
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(256001)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # HTTP bodies and prompts may contain private family data.
            raise ModelResponseError("MiniMax M3.1 当前不可用") from exc
        if len(raw) > 256000:
            raise ModelResponseError("MiniMax M3.1 回答过长")
        try:
            data = json.loads(raw)
            if data.get("base_resp", {}).get("status_code", 0) != 0:
                raise ValueError("provider status")
            choice = data["choices"][0]
            if choice["finish_reason"] == "length":
                raise ValueError("length")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip() or "<think>" in content:
                raise ValueError("invalid content")
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelResponseError("MiniMax M3.1 未返回有效回答") from exc

    def __call__(self, system: str, user: str) -> str:
        try:
            answer = self._remote(system, user)
            LOGGER.info("puppy_model_call provider=minimax model=%s", _TEXT_MODEL)
            return answer
        except ModelResponseError as error:
            if _VOICE_REMOTE_ONLY.get():
                LOGGER.warning("puppy_model_call provider=minimax result=unavailable voice=true")
                raise
            # Existing MLX model remains a private, slower offline fallback.
            LOGGER.warning("puppy_model_call provider=local_fallback reason=%s", type(error).__name__)
            return self._local_model()(system, user)

    def requests_camera(self, text: str) -> bool:
        try:
            # Visual requests may contain details about the child's surroundings.
            # Keep the camera decision and the subsequent visual conversation local.
            raw = self._local_model()(_CAMERA_SYSTEM, text)
            start = raw.index("{")
            value, _ = json.JSONDecoder().raw_decode(raw[start:])
            return isinstance(value, dict) and set(value) == {"use_camera"} and value["use_camera"] is True
        except (ValueError, TypeError):
            return False

    def describe(self, image_path: str, question: str) -> dict:
        return self._local_model().describe(image_path, question)

    def describe_bytes(self, data: bytes, question: str) -> dict:
        return self._local_model().describe_bytes(data, question)

    def private_context_model(self):
        """Text model for turns containing a local camera observation."""
        return self._local_model()
