"""Real offline identity engines; model downloads are explicit, never at inference.

Sources:
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet (MIT)
https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface (Apache-2.0)
https://k2-fsa.github.io/sherpa/onnx/speaker-identification/index.html
https://github.com/wenet-e2e/wespeaker (Apache-2.0)
https://github.com/snakers4/silero-vad (MIT)
Validated with opencv-python 5.0.0.93 and sherpa-onnx 1.13.8 on Apple Silicon.
"""
import argparse
import hashlib
import io
import json
import threading
import urllib.request
import uuid
from pathlib import Path

from .identity import Observation, QualityError, normalized

_OPENCV = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
MODELS = {
    "face_detection_yunet_2026may.onnx": {
        "url": _OPENCV + "face_detection_yunet/face_detection_yunet_2026may.onnx",
        "bytes": 229738, "sha256": "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"},
    "face_recognition_sface_2021dec.onnx": {
        "url": _OPENCV + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "bytes": 38696353, "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"},
    "wespeaker_zh_cnceleb_resnet34.onnx": {
        "url": _SHERPA + "speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx",
        "bytes": 26534363, "sha256": "f86cd6c509f331f0e20b07bd48d1b2eb7de54202643401c4e84695ac861a0e5a"},
    "silero_vad.onnx": {
        "url": _SHERPA + "asr-models/silero_vad.onnx",
        "bytes": 643854, "sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"},
}


def verified_model(directory: str | Path, name: str) -> Path:
    path = Path(directory) / name
    specification = MODELS[name]
    if not path.is_file() or path.stat().st_size != specification["bytes"]:
        raise FileNotFoundError(f"missing_or_incomplete_model:{name}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != specification["sha256"]:
        raise ValueError(f"model_checksum_mismatch:{name}")
    return path


def download_models(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    result = []
    for name, specification in MODELS.items():
        path = directory / name
        if not path.exists():
            partial = directory / (name + "." + uuid.uuid4().hex + ".partial")
            try:
                with urllib.request.urlopen(specification["url"], timeout=60) as source, partial.open("xb") as target:
                    written = 0
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > specification["bytes"]:
                            raise ValueError("model_download_size_exceeded")
                        target.write(chunk)
                if partial.stat().st_size != specification["bytes"]:
                    raise ValueError("model_download_incomplete")
                if hashlib.sha256(partial.read_bytes()).hexdigest() != specification["sha256"]:
                    raise ValueError("model_download_checksum_mismatch")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        verified_model(directory, name)
        result.append({"name": name, "bytes": specification["bytes"], "sha256": specification["sha256"]})
    return result


class FaceEngine:
    """YuNet face/landmark detection and aligned SFace embeddings on the CPU."""

    def __init__(self, model_directory: str | Path, *, min_face_pixels: int = 48,
                 min_sharpness: float = 40.0, min_detection_score: float = 0.90):
        import cv2
        self.cv = cv2
        self.min_face_pixels, self.min_sharpness = min_face_pixels, min_sharpness
        detector = verified_model(model_directory, "face_detection_yunet_2026may.onnx")
        recognizer = verified_model(model_directory, "face_recognition_sface_2021dec.onnx")
        self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320),
                                                score_threshold=min_detection_score)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.model_id = "sface-2021dec:" + MODELS[recognizer.name]["sha256"]
        self.lock = threading.Lock()

    def extract(self, data: bytes, *, captured_at: float) -> Observation:
        import numpy as np
        from PIL import Image
        if not isinstance(data, bytes) or not 0 < len(data) <= 2 * 1024 * 1024:
            raise QualityError("invalid_face_image_size")
        try:
            with Image.open(io.BytesIO(data)) as header:
                if (header.format not in {"JPEG", "PNG"} or min(header.size) < 64
                        or max(header.size) > 1920):
                    raise QualityError("unsupported_face_image")
            image = self.cv.imdecode(np.frombuffer(data, np.uint8), self.cv.IMREAD_COLOR)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise QualityError("invalid_face_image") from exc
        if image is None:
            raise QualityError("invalid_face_image")
        height, width = image.shape[:2]
        with self.lock:
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(image)
            if faces is None or len(faces) == 0:
                raise QualityError("no_face")
            if len(faces) != 1:
                raise QualityError("multiple_faces")
            face = faces[0]
            x, y, w, h = map(float, face[:4])
            if min(w, h) < self.min_face_pixels:
                raise QualityError("face_too_small")
            if x < 0 or y < 0 or x + w > width or y + h > height:
                raise QualityError("face_clipped")
            crop = image[int(y):int(y + h), int(x):int(x + w)]
            gray = self.cv.cvtColor(crop, self.cv.COLOR_BGR2GRAY)
            sharpness = float(self.cv.Laplacian(gray, self.cv.CV_64F).var())
            if sharpness < self.min_sharpness:
                raise QualityError("face_blurred")
            brightness = float(gray.mean())
            if not 30 <= brightness <= 225:
                raise QualityError("face_lighting")
            eye_distance = float(np.linalg.norm(face[4:6] - face[6:8]))
            if eye_distance < 0.20 * w:
                raise QualityError("face_pose")
            aligned = self.recognizer.alignCrop(image, face)
            vector = normalized(self.recognizer.feature(aligned).reshape(-1).tolist())
        return Observation("face", self.model_id, vector,
                           {"detection_score": float(face[-1]), "face_pixels": min(w, h),
                            "sharpness": sharpness, "brightness": brightness}, captured_at)


class VoiceEngine:
    """Silero VAD followed by WeSpeaker CN-Celeb embeddings; mono 16 kHz PCM16."""

    def __init__(self, model_directory: str | Path, *, min_speech_seconds: float = 2.5):
        import sherpa_onnx
        self.sherpa = sherpa_onnx
        model = verified_model(model_directory, "wespeaker_zh_cnceleb_resnet34.onnx")
        self.vad_model = verified_model(model_directory, "silero_vad.onnx")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2, provider="cpu")
        if not config.validate():
            raise ValueError("speaker_model_config_invalid")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.model_id = "wespeaker-cnceleb-resnet34:" + MODELS[model.name]["sha256"]
        self.min_speech_seconds = min_speech_seconds
        self.lock = threading.Lock()

    def extract(self, data: bytes, *, captured_at: float) -> Observation:
        import numpy as np
        if not isinstance(data, bytes) or len(data) % 2 or not 3 * 32000 <= len(data) <= 20 * 32000:
            raise QualityError("voice_requires_3_to_20_seconds_pcm16")
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        clipping = float(np.mean(np.abs(samples) >= 0.98))
        if rms < 0.008:
            raise QualityError("voice_too_quiet")
        if clipping > 0.02:
            raise QualityError("voice_clipped")
        with self.lock:
            config = self.sherpa.VadModelConfig()
            config.silero_vad.model = str(self.vad_model)
            config.silero_vad.min_speech_duration = 0.3
            config.silero_vad.min_silence_duration = 0.25
            config.sample_rate, config.num_threads, config.provider = 16000, 1, "cpu"
            vad = self.sherpa.VoiceActivityDetector(config, buffer_size_in_seconds=30)
            window = config.silero_vad.window_size
            for offset in range(0, len(samples), window):
                block = samples[offset:offset + window]
                if len(block) < window:
                    block = np.pad(block, (0, window - len(block)))
                vad.accept_waveform(block)
            vad.flush()
            segments = []
            while not vad.empty():
                segments.append(np.asarray(vad.front.samples, dtype=np.float32))
                vad.pop()
            speech = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
            speech_seconds = len(speech) / 16000
            if speech_seconds < self.min_speech_seconds:
                raise QualityError("insufficient_speech")
            stream = self.extractor.create_stream()
            stream.accept_waveform(sample_rate=16000, waveform=speech)
            stream.input_finished()
            if not self.extractor.is_ready(stream):
                raise QualityError("speaker_embedding_not_ready")
            vector = normalized(self.extractor.compute(stream))
        return Observation("voice", self.model_id, vector,
                           {"speech_seconds": speech_seconds, "rms": rms,
                            "clipped_fraction": clipping}, captured_at)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify optional offline identity models")
    parser.add_argument("--download", type=Path, required=True, metavar="DIRECTORY")
    options = parser.parse_args()
    print(json.dumps(download_models(options.download), ensure_ascii=False, indent=2))
