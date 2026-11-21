"""16 kHz Opus transport and offline macOS speech services.

Use the async speech methods in cancellable sessions. Cancelling ASR discards
its result but cannot interrupt an MLX kernel; the actual worker retains the
shared gate until completion. TTS subprocesses are terminated on cancellation.
"""

import asyncio
from array import array
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import ctypes
import ctypes.util
import math
import os
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from weakref import WeakKeyDictionary


SAMPLE_RATE = 16000
FRAME_SAMPLES = 960  # 60 ms
_ASR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="puppy-asr")
_ASR_THREAD_LOCK = threading.Lock()
_ASR_GATES = WeakKeyDictionary()


class OpusCodec:
    def __init__(self, library: str | None = None):
        library = library or os.environ.get("PUPPY_OPUS_LIBRARY") or ctypes.util.find_library("opus")
        if not library:
            raise RuntimeError("libopus is unavailable; set PUPPY_OPUS_LIBRARY")
        self.lib = ctypes.CDLL(library)
        self.decoder = self.encoder = None
        self.lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                                       ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.opus_decode.restype = ctypes.c_int
        self.lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_encoder_create.restype = ctypes.c_void_p
        self.lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int32]
        self.lib.opus_encode.restype = ctypes.c_int
        self.lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_decoder_destroy.restype = None
        self.lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_encoder_destroy.restype = None
        self.lib.opus_encoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.opus_encoder_ctl.restype = ctypes.c_int
        error = ctypes.c_int()
        try:
            self.decoder = self.lib.opus_decoder_create(SAMPLE_RATE, 1, ctypes.byref(error))
            if not self.decoder or error.value:
                raise RuntimeError(f"Opus decoder error {error.value}")
            self.encoder = self.lib.opus_encoder_create(SAMPLE_RATE, 1, 2048, ctypes.byref(error))
            if not self.encoder or error.value:
                raise RuntimeError(f"Opus encoder error {error.value}")
        except Exception:
            self.close()
            raise

    def close(self):
        if self.decoder:
            self.lib.opus_decoder_destroy(self.decoder)
            self.decoder = None
        if self.encoder:
            self.lib.opus_encoder_destroy(self.encoder)
            self.encoder = None

    def reset_encoder(self):
        if not self.encoder:
            raise ValueError("Opus codec is closed")
        result = self.lib.opus_encoder_ctl(self.encoder, 4028)  # OPUS_RESET_STATE
        if result:
            raise ValueError(f"Opus reset error {result}")

    def decode(self, packet: bytes) -> bytes:
        if not self.decoder:
            raise ValueError("Opus codec is closed")
        if not packet or len(packet) > 4000:
            raise ValueError("Invalid Opus packet")
        data = ctypes.create_string_buffer(packet)
        samples = (ctypes.c_int16 * 5760)()
        count = self.lib.opus_decode(self.decoder, data, len(packet), samples, 5760, 0)
        if count <= 0:
            raise ValueError(f"Opus decode error {count}")
        return ctypes.string_at(samples, count * 2)

    def encode(self, pcm: bytes) -> bytes:
        if not self.encoder:
            raise ValueError("Opus codec is closed")
        if len(pcm) != FRAME_SAMPLES * 2:
            raise ValueError("Opus input must be a 60 ms 16 kHz mono frame")
        source = ctypes.create_string_buffer(pcm)
        output = ctypes.create_string_buffer(4000)
        count = self.lib.opus_encode(self.encoder, source, FRAME_SAMPLES, output, len(output))
        if count <= 0:
            raise ValueError(f"Opus encode error {count}")
        return output.raw[:count]


def pcm_rms(pcm: bytes) -> float:
    if len(pcm) % 2:
        raise ValueError("PCM must contain complete 16-bit samples")
    if not pcm:
        return 0.0
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def pcm_peak_frame_rms(pcm: bytes) -> float:
    frame_bytes = FRAME_SAMPLES * 2
    return max((pcm_rms(pcm[offset:offset + frame_bytes])
                for offset in range(0, len(pcm), frame_bytes)), default=0.0)


def boost_quiet_speech(pcm: bytes) -> bytes:
    """Give quiet recordings more level for ASR without clipping peaks."""
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max(map(abs, samples), default=0)
    if not 0 < peak < 6000:
        return pcm
    gain = min(8.0, 6000 / peak)
    boosted = array("h", (round(sample * gain) for sample in samples))
    if sys.byteorder != "little":
        boosted.byteswap()
    return boosted.tobytes()


class SpeechServices:
    def __init__(self, whisper_model: str, voice: str = "Tingting"):
        self.whisper_model = whisper_model
        self.voice = voice

    def _transcribe(self, pcm: bytes, *, language: str | None = None) -> str:
        import mlx_whisper
        import numpy as np

        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        result = mlx_whisper.transcribe(waveform, path_or_hf_repo=self.whisper_model,
                                        verbose=False, condition_on_previous_text=False,
                                        **({"language": language} if language else {}))
        return result["text"].strip()

    def transcribe(self, pcm: bytes, *, language: str | None = None) -> str:
        if len(pcm) % 2 or len(pcm) > SAMPLE_RATE * 2 * 30:
            raise ValueError("ASR requires at most 30 seconds of complete 16 kHz PCM")
        if len(pcm) < SAMPLE_RATE * 2 // 3:
            return ""
        # Also protects legacy synchronous callers and callers on another loop.
        with _ASR_THREAD_LOCK:
            return self._transcribe(pcm, language=language) if language else self._transcribe(pcm)

    async def transcribe_async(self, pcm: bytes, *, language: str | None = None) -> str:
        loop = asyncio.get_running_loop()
        gate = _ASR_GATES.setdefault(loop, asyncio.Lock())
        await gate.acquire()
        try:
            work = loop.run_in_executor(_ASR_EXECUTOR, partial(self.transcribe, pcm, language=language)
                                        if language else partial(self.transcribe, pcm))
        except BaseException:
            gate.release()
            raise

        def completed(future):
            # Consume errors even when the caller was cancelled. The executor
            # future is shielded, so release corresponds to actual completion.
            if not future.cancelled():
                future.exception()
            gate.release()

        work.add_done_callback(completed)
        return await asyncio.shield(work)

    @staticmethod
    def _validate_text(text: str):
        if not isinstance(text, str) or len(text) > 1000:
            raise ValueError("TTS text must be a string of at most 1000 characters")

    @staticmethod
    def _read_pcm(path: Path) -> bytes:
        with wave.open(str(path), "rb") as reader:
            if reader.getnchannels() != 1 or reader.getframerate() != SAMPLE_RATE or reader.getsampwidth() != 2:
                raise ValueError("TTS output is not 16 kHz mono PCM")
            frames = reader.getnframes()
            if not 0 < frames <= SAMPLE_RATE * 60:
                raise ValueError("TTS output must contain 0–60 seconds of audio")
            pcm = reader.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("TTS output is truncated")
            return pcm

    def _commands(self, text: str, directory: str):
        aiff = Path(directory) / "speech.aiff"
        wav = Path(directory) / "speech.wav"
        # -o is mandatory: say always writes a file, never the system speaker.
        return (["/usr/bin/say", "-v", self.voice, "-r", "170", "-o", str(aiff), text],
                ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                 str(aiff), str(wav)], wav)

    def synthesize(self, text: str) -> bytes:
        self._validate_text(text)
        if not text.strip():
            return b""
        with tempfile.TemporaryDirectory(prefix="puppy-tts-") as directory:
            say, convert, wav = self._commands(text, directory)
            subprocess.run(say, check=True, timeout=20, stdout=subprocess.DEVNULL)
            subprocess.run(convert, check=True, timeout=10, stdout=subprocess.DEVNULL)
            return self._read_pcm(wav)

    async def synthesize_async(self, text: str) -> bytes:
        self._validate_text(text)
        if not text.strip():
            return b""
        with tempfile.TemporaryDirectory(prefix="puppy-tts-") as directory:
            say, convert, wav = self._commands(text, directory)
            await _run_process(say, timeout=20)
            await _run_process(convert, timeout=10)
            return self._read_pcm(wav)

    async def synthesize_vocabulary_async(self, prefix: str, word: str) -> bytes:
        """Use the installed English voice for the word in a Mandarin reply."""
        english = SpeechServices(self.whisper_model, voice="Samantha")
        english_text = "Cat. C, A, T." if word == "cat" else f"{word}."
        intro_pcm, word_pcm = await asyncio.gather(
            self.synthesize_async(prefix), english.synthesize_async(english_text),
            return_exceptions=True)
        if isinstance(intro_pcm, bytes) and intro_pcm and isinstance(word_pcm, bytes) and word_pcm:
            # Quiet trailing frames give the English final consonant time to play.
            tail = bytes(FRAME_SAMPLES * 2 * 2) if word == "cat" else b""
            return intro_pcm + bytes(SAMPLE_RATE // 8 * 2) + word_pcm + tail
        # A Mac without the English voice can still say the complete answer.
        return await self.synthesize_async(f"{prefix} {word}。")


async def _run_process(command: list[str], timeout: float):
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        if process.returncode:
            raise RuntimeError(f"Speech subprocess failed ({process.returncode}): {stderr.decode(errors='replace')[:200]}")
    except BaseException:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=.25)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        raise


def opus_frames(codec: OpusCodec, pcm: bytes):
    if len(pcm) % 2:
        raise ValueError("PCM must contain complete 16-bit samples")
    frame_bytes = FRAME_SAMPLES * 2
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset:offset + frame_bytes]
        if len(frame) < frame_bytes:
            frame += b"\0" * (frame_bytes - len(frame))
        yield codec.encode(frame)
