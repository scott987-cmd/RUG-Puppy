"""Cancellable, bounded Puppy v1 audio sessions (no FastAPI dependency).

After authentication and socket.accept(), run AudioSession(...).run(). Inject
async transcribe(pcm), chat(recorded_pcm, transcript), and synthesize(text)
callables. chat may return a reply string or a mapping with a 'reply' string;
its PCM argument is the optional identity-recognition hook. Each socket owns
its codec. The session closes it after stopping the active turn.

Cancellation prevents further packets from the old turn. Already transmitted
packets and already started model/database side effects cannot be recalled.
Synchronous speech/model work must be wrapped with cancellation-safe execution
ownership (SpeechServices.transcribe_async provides this for local Whisper).
"""

import asyncio
from dataclasses import dataclass
import json
import logging
from typing import Awaitable, Callable, Mapping, Protocol
import uuid

from .audio import FRAME_SAMPLES, SAMPLE_RATE, opus_frames, pcm_rms


LOGGER = logging.getLogger(__name__)


class MotionStopCheck(Protocol):
    async def __call__(self, pcm: bytes, owner: object, *,
                       resolving_negation: bool = False) -> Mapping | None: ...


@dataclass(frozen=True)
class AudioSessionConfig:
    max_record_seconds: float = 12
    max_listen_seconds: float = 20
    pre_roll_seconds: float = 1.2
    silence_seconds: float = .84
    post_tts_echo_seconds: float = .7
    max_tts_seconds: float = 30
    frame_interval: float = FRAME_SAMPLES / SAMPLE_RATE
    hello_timeout: float = 10
    send_timeout: float = 3
    max_decode_errors: int = 3
    max_transcript_chars: int = 300
    max_reply_chars: int = 400

    def __post_init__(self):
        positive = (self.max_record_seconds, self.max_listen_seconds, self.silence_seconds,
                    self.max_tts_seconds, self.hello_timeout, self.send_timeout,
                    self.max_decode_errors, self.max_transcript_chars, self.max_reply_chars)
        if (any(value <= 0 for value in positive) or self.frame_interval < 0
                or self.pre_roll_seconds < 0 or self.post_tts_echo_seconds < 0):
            raise ValueError("Audio session limits must be positive")


class AudioSession:
    def __init__(self, socket, *, codec,
                 transcribe: Callable[[bytes], Awaitable[str]],
                 chat: Callable[[bytes, str], Awaitable[str | Mapping]],
                 synthesize: Callable[[str], Awaitable[bytes]],
                 prepare_reply: Callable[[bytes], Awaitable[str | Mapping | None]] | None = None,
                 finish_turn: Callable[[], Awaitable[None]] | None = None,
                 motion_owner: Callable[[], object | None] | None = None,
                 stop_motion: MotionStopCheck | None = None,
                 short_answer_expected: Callable[[], bool] | None = None,
                 on_event: Callable[[str, Mapping[str, int | bool | str]], None] | None = None,
                 session_id: str | None = None,
                 config: AudioSessionConfig | None = None):
        self.socket = socket
        self.codec = codec
        self.transcribe = transcribe
        self.chat = chat
        self.synthesize = synthesize
        self.prepare_reply = prepare_reply
        self.finish_turn = finish_turn
        self.motion_owner = motion_owner
        self.stop_motion = stop_motion
        self.on_event = on_event
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or AudioSessionConfig()
        self.short_answer_expected = short_answer_expected
        self.listening = False
        self.closed = False
        self.mode = "auto"
        self.last_error: str | None = None
        self._recording = bytearray()
        self._speech_seen = False
        self._voiced_bytes = 0
        self._soft_voice_bytes = 0
        self._quiet_bytes = 0
        self._noise_floor = 120.0
        self._decode_errors = 0
        self._listen_deadline = 0.0
        self._listen_started_at = 0.0
        self._first_audio_at = 0.0
        self._decoded_frames = 0
        self._peak_rms = 0
        self._echo_suppressed_frames = 0
        self._ignore_audio_until = 0.0
        self._awaiting_post_tts_listen = False
        self._generation = 0
        self._turn: asyncio.Task | None = None
        self._tts_generation: int | None = None
        self._empty_transcripts = 0
        self._send_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._motion_negation_prefix = b""
        self._motion_check: asyncio.Task | None = None
        self._motion_token = None
        self._motion_recording = bytearray()
        self._motion_seen = False
        self._motion_since_check = 0

    def _event(self, name: str, **details: int | bool | str):
        if self.on_event is not None:
            try:
                self.on_event(name, details)
            except Exception:
                LOGGER.warning("Puppy audio event observer failed")

    @property
    def recorded_bytes(self) -> int:
        return len(self._recording)

    def _current(self, generation: int):
        if self.closed or generation != self._generation:
            raise asyncio.CancelledError

    async def _send(self, message, *, generation: int | None = None):
        async with self._send_lock:
            if generation is not None:
                self._current(generation)
            elif self.closed:
                return
            operation = self.socket.send_bytes(message) if isinstance(message, bytes) else self.socket.send_json(message)
            await asyncio.wait_for(operation, timeout=self.config.send_timeout)

    async def _terminate(self, code: int):
        self.closed = True
        self.listening = False
        try:
            await asyncio.wait_for(self.socket.close(code=code), timeout=self.config.send_timeout)
        except Exception:
            pass

    async def cancel_and_close(self) -> bool:
        """Parent-initiated stop of this audio session, without a motion command."""
        async with self._stop_lock:
            if self.closed:
                return False
            await self._interrupt(notify=True)
            await self._terminate(1000)
            self._event("parent_audio_stopped")
            return True

    def _clear_recording(self):
        self._recording.clear()
        self._speech_seen = False
        self._voiced_bytes = 0
        self._soft_voice_bytes = 0
        self._quiet_bytes = 0
        # Keep the measured room floor between utterances. Resetting it to a
        # loud default here makes the next quiet follow-up disappear into the
        # fresh calibration period. New connections initialize it in __init__.
        self._decode_errors = 0
        self._first_audio_at = 0.0
        self._decoded_frames = 0
        self._peak_rms = 0
        self._echo_suppressed_frames = 0

    def _start_recording(self):
        self._clear_recording()
        if self._awaiting_post_tts_listen:
            # listen/start acknowledges that the device drained its playback
            # queue. The network stop may have preceded this by more than the
            # echo window; guard the actual microphone restart as well.
            if self.mode == "auto":
                self._ignore_audio_until = (asyncio.get_running_loop().time()
                                            + self.config.post_tts_echo_seconds)
            self._awaiting_post_tts_listen = False
        self.listening = True
        self._listen_started_at = asyncio.get_running_loop().time()
        self._listen_deadline = self._listen_started_at + self.config.max_listen_seconds
        self._event("listen_started", mode=self.mode)

    def _input_diagnostics(self):
        first_audio_ms = (int((self._first_audio_at - self._listen_started_at) * 1000)
                          if self._first_audio_at else -1)
        return {"frames": self._decoded_frames, "peak_rms": self._peak_rms,
                "noise_floor_rms": round(self._noise_floor),
                "first_audio_ms": first_audio_ms,
                "echo_suppressed_frames": self._echo_suppressed_frames}

    async def _interrupt(self, *, notify: bool):
        self._generation += 1  # Fence even a dependency that returns after cancellation.
        self.listening = False
        self._clear_recording()
        self._motion_recording.clear()
        self._motion_negation_prefix = b""
        self._motion_seen, self._motion_token = False, None
        self._motion_since_check = 0
        checking = self._motion_check
        if checking is not None and checking is not asyncio.current_task():
            self._motion_check = None
            checking.cancel()
            await asyncio.gather(checking, return_exceptions=True)
        turn, self._turn = self._turn, None
        if turn is not None:
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
        if self._tts_generation is not None:
            self._tts_generation = None
            if notify and not self.closed:
                await self._send({"type": "tts", "state": "stop"})

    async def run(self):
        try:
            event = await asyncio.wait_for(self.socket.receive(), timeout=self.config.hello_timeout)
            hello = self._json_message(event)
            if not self._valid_hello(hello):
                await self._terminate(1002)
                return
            self._event("connected")
            await self._send({"type": "hello", "transport": "websocket", "session_id": self.session_id,
                              "audio_params": {"format": "opus", "channels": 1,
                                               "sample_rate": SAMPLE_RATE, "frame_duration": 60}})
            while not self.closed:
                timeout = 1.0  # Also notices a background send failure on a stalled peer.
                if self.listening:
                    remaining = self._listen_deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        if self._speech_seen and self._recording:
                            self._finish_recording()
                        else:
                            self._event("listen_timeout", **self._input_diagnostics())
                            await self._terminate(1000)  # Return a silent, abandoned microphone to idle.
                        continue
                    timeout = min(timeout, remaining)
                try:
                    event = await asyncio.wait_for(self.socket.receive(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                if event.get("type") == "websocket.disconnect":
                    break
                if event.get("text") is not None:
                    message = self._json_message(event)
                    if not isinstance(message, dict):
                        continue
                    if message.get("session_id", self.session_id) != self.session_id:
                        continue
                    if message.get("type") == "abort":
                        phase = ("tts" if self._tts_generation is not None else
                                 "processing" if self._turn is not None else
                                 "listening" if self.listening else "idle")
                        reason = ("wake_word_detected" if message.get("reason") == "wake_word_detected"
                                  else "other")
                        self._event("abort_received", phase=phase, reason=reason)
                        await self._interrupt(notify=True)
                        if self.mode == "realtime":
                            self._start_recording()
                    elif message.get("type") == "listen":
                        if message.get("state") == "start":
                            await self._interrupt(notify=True)
                            self.mode = message.get("mode", "auto")
                            if self.mode not in {"auto", "manual", "realtime"}:
                                await self._terminate(1002)
                                return
                            self._start_recording()
                        elif message.get("state") == "stop" and self.listening:
                            self._finish_recording()
                        # listen/detect is the wake-word label, not a second utterance.
                elif event.get("bytes") is not None and self.listening:
                    await self._receive_audio(event["bytes"])
                elif (event.get("bytes") is not None and self.motion_owner is not None
                      and self.stop_motion is not None and self._turn is not None
                      and self._tts_generation is None):
                    await self._receive_motion_audio(event["bytes"])
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.last_error = type(error).__name__
            self._event("session_failed", error=self.last_error)
            LOGGER.warning("Puppy audio connection ended: %s", self.last_error)
            await self._terminate(1001)
        finally:
            self.closed = True
            await self._interrupt(notify=False)
            self.codec.close()
            self._event("closed")

    @staticmethod
    def _json_message(event):
        text = event.get("text")
        if not isinstance(text, str) or len(text) > 4096:
            return None
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _valid_hello(hello) -> bool:
        if not isinstance(hello, dict) or hello.get("type") != "hello" or hello.get("transport") != "websocket":
            return False
        if hello.get("version", 1) != 1:
            return False
        parameters = hello.get("audio_params", {})
        return (isinstance(parameters, dict) and parameters.get("format", "opus") == "opus"
                and parameters.get("sample_rate", SAMPLE_RATE) == SAMPLE_RATE
                and parameters.get("channels", 1) == 1
                and parameters.get("frame_duration", 60) == 60)

    async def _decode_audio(self, packet: bytes):
        try:
            if not isinstance(packet, bytes) or not 0 < len(packet) <= 4000:
                raise ValueError("Invalid Opus packet")
            pcm = self.codec.decode(packet)
            if not isinstance(pcm, bytes) or not 0 < len(pcm) <= SAMPLE_RATE * 2 * .12 or len(pcm) % 2:
                raise ValueError("Invalid decoded PCM frame")
            level = pcm_rms(pcm)
        except (ValueError, RuntimeError, OSError):
            self._decode_errors += 1
            if self._decode_errors >= self.config.max_decode_errors:
                self.last_error = "invalid_audio"
                await self._terminate(1003)
            return
        self._decode_errors = 0
        return pcm, level

    async def _receive_motion_audio(self, packet: bytes):
        decoded = await self._decode_audio(packet)
        if decoded is None:
            return
        pcm, level = decoded
        owner = self.motion_owner()
        if owner is not self._motion_token:
            self._motion_recording.clear()
            self._motion_negation_prefix = b""
            self._motion_seen, self._motion_token = False, owner
            self._motion_since_check = 0
        if owner is None:
            return
        self._motion_recording.extend(pcm)
        self._motion_since_check += len(pcm)
        if level > max(25, self._noise_floor * 1.5):
            self._motion_seen = True
        limit = SAMPLE_RATE * 2 * 3
        if not self._motion_seen:
            del self._motion_recording[:-int(SAMPLE_RATE * 2 * .36)]
            return
        del self._motion_recording[:max(0, len(self._motion_recording) - limit)]
        if self._motion_check is None or self._motion_check.done():
            window = self._next_motion_window()
            if window:
                self._motion_check = asyncio.create_task(
                    self._check_motion_stop(window, owner, self._generation), name="puppy-motion-stop")

    def _next_motion_window(self):
        minimum = int(SAMPLE_RATE * 2 * .6)
        if len(self._motion_recording) < minimum or self._motion_since_check < SAMPLE_RATE * 2 * .48:
            return None
        if self._motion_negation_prefix:
            # Resolve the prefix together with one short continuation. Never feed
            # that continuation alone to the ordinary stop recognizer.
            recorded = self._motion_negation_prefix + self._motion_recording[:minimum]
            self._motion_negation_prefix = b""
            del self._motion_recording[:minimum]
            self._motion_since_check = len(self._motion_recording)
            resolving = True
        else:
            recorded = bytes(self._motion_recording)
            self._motion_since_check = 0
            resolving = False
        self._event("motion_stop_recording", bytes=len(recorded))
        return recorded, resolving

    async def _check_motion_stop(self, window, owner, generation):
        try:
            while window:
                recorded, resolving = window
                if resolving:
                    result = await self.stop_motion(recorded, owner, resolving_negation=True)
                else:
                    result = await self.stop_motion(recorded, owner)
                self._current(generation)
                if self.motion_owner() is not owner and not (result and result.get("reply")):
                    return
                if not resolving and result and (result.get("ignore") or result.get("negation_prefix")):
                    # New packets may already be queued while ASR examines the
                    # snapshot. Consume only the audio that was examined.
                    keep = min(len(self._motion_recording), self._motion_since_check)
                    del self._motion_recording[:len(self._motion_recording) - keep]
                    self._motion_negation_prefix = recorded if result.get("negation_prefix") else b""
                    self._motion_seen = bool(keep or self._motion_negation_prefix)
                reply = result.get("reply") if result else None
                if isinstance(reply, str) and reply:
                    ack_generation = generation + 1
                    await self._interrupt(notify=True)
                    self._current(ack_generation)
                    self._event("motion_voice_stop", confirmed=result.get("confirmed") is True)
                    await self._speak(reply, ack_generation)
                    return
                if self.motion_owner() is not owner:
                    return
                window = self._next_motion_window()
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            self._event("motion_stop_unavailable", error=type(error).__name__)
        finally:
            if self._motion_check is asyncio.current_task():
                self._motion_check = None

    async def _receive_audio(self, packet: bytes):
        decoded = await self._decode_audio(packet)
        if decoded is None:
            return
        pcm, level = decoded
        if asyncio.get_running_loop().time() < self._ignore_audio_until:
            self._echo_suppressed_frames += 1
            return
        if not self._first_audio_at:
            self._first_audio_at = asyncio.get_running_loop().time()
        self._decoded_frames += 1
        self._peak_rms = max(self._peak_rms, round(level))
        if self._speech_seen:
            # A short onset transient can inflate the initial estimate. Let
            # genuinely quieter frames lower it so the following soft words
            # keep the same utterance open; do not lengthen the silence timer.
            if level < self._noise_floor:
                self._noise_floor = .85 * self._noise_floor + .15 * level
            voiced = level > max(20, self._noise_floor * 1.15)
        else:
            # New connections must settle to a genuinely quiet room before a
            # short child utterance ends. Raise the estimate slowly; lower it
            # faster, while keeping the sustained-onset check below.
            weight = .30 if level < self._noise_floor else .05
            updated_floor = (1 - weight) * self._noise_floor + weight * min(level, 900)
            voiced = level > max(200, updated_floor * 1.75)
            # A low-level onset needs 180 ms above a quiet room baseline.
            # Hold the baseline while checking so it cannot absorb the onset.
            if max(20, self._noise_floor * 2.5) < level <= 200:
                self._soft_voice_bytes += len(pcm)
                # A prompted yes/no can be a single soft syllable. Its
                # server-side consent check still requires two recognizers.
                onset = .06 if self.short_answer_expected and self.short_answer_expected() else .18
                voiced = self._soft_voice_bytes >= int(SAMPLE_RATE * onset) * 2
            else:
                self._soft_voice_bytes = 0
                self._noise_floor = updated_floor
        if voiced:
            self._speech_seen = True
            self._voiced_bytes += len(pcm)
            self._quiet_bytes = 0
        elif self._speech_seen:
            self._quiet_bytes += len(pcm)
        maximum = int(self.config.max_record_seconds * SAMPLE_RATE) * 2
        self._recording.extend(pcm[:max(0, maximum - len(self._recording))])
        if not self._speech_seen:
            retained = min(maximum, int(self.config.pre_roll_seconds * SAMPLE_RATE) * 2)
            del self._recording[:max(0, len(self._recording) - retained)]
        if self._speech_seen and (len(self._recording) >= maximum or
                (self.mode != "manual" and self._quiet_bytes >= self.config.silence_seconds * SAMPLE_RATE * 2)):
            self._finish_recording()

    def _finish_recording(self):
        self.listening = False
        recorded = bytes(self._recording)
        # A single captured tail/impulse must not start a spoken retry when
        # ASR finds no words. Keep real short recognized words, and retain the
        # explicit manual-recording retry behavior.
        speech_seen = self._speech_seen and (
            self.mode == "manual" or self._voiced_bytes >= int(SAMPLE_RATE * .12) * 2
            or bool(self.short_answer_expected and self.short_answer_expected()))
        diagnostics = self._input_diagnostics()
        self._clear_recording()
        self._event("recording_finished", bytes=len(recorded), speech_detected=speech_seen,
                    **diagnostics)
        if recorded:
            generation = self._generation
            self._turn = asyncio.create_task(self._answer(recorded, generation, speech_seen), name="puppy-audio-turn")

    async def _speak(self, spoken: str, generation: int):
        pcm = await self.synthesize(spoken)
        self._current(generation)
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2 or len(pcm) > self.config.max_tts_seconds * SAMPLE_RATE * 2:
            raise ValueError("TTS returned invalid or oversized PCM")
        self._event("tts_ready", bytes=len(pcm))
        self.codec.reset_encoder()
        self._tts_generation = generation
        await self._send({"type": "tts", "state": "start"}, generation=generation)
        await self._send({"type": "tts", "state": "sentence_start", "text": spoken}, generation=generation)
        await asyncio.sleep(self.config.frame_interval)
        for packet in opus_frames(self.codec, pcm):
            self._current(generation)
            await self._send(packet, generation=generation)
            await asyncio.sleep(self.config.frame_interval)
        await self._send({"type": "tts", "state": "stop"}, generation=generation)
        self._tts_generation = None
        self._event("tts_sent")
        # The device has no microphone echo reference. Its speaker can keep
        # playing a buffered tail after the last network packet is sent.
        if self.mode != "realtime":
            self._ignore_audio_until = (asyncio.get_running_loop().time()
                                        + self.config.post_tts_echo_seconds)
            self._awaiting_post_tts_listen = self.mode == "auto"

    async def _answer(self, recorded: bytes, generation: int, speech_seen: bool):
        try:
            # Parent enrollment consumes the recording without interpreting it as
            # a child request or storing a conversation. Cancellation remains live.
            result = await self.prepare_reply(recorded) if self.prepare_reply else None
            self._current(generation)
            if result is None:
                transcript = await self.transcribe(recorded)
                self._current(generation)
                if not isinstance(transcript, str):
                    raise ValueError("ASR did not return text")
                transcript = transcript.strip()[:self.config.max_transcript_chars]
                self._event("transcribed", chars=len(transcript))
                if not transcript:
                    if speech_seen:
                        self._empty_transcripts += 1
                        if self._empty_transcripts == 1:
                            await self._speak("我没听清，再说一次好吗？", generation)
                            if self.mode != "manual":
                                self._start_recording()
                        else:
                            await self._speak("我还是没听清，我们等会儿再试吧。", generation)
                            await self._terminate(1000)
                    elif self.mode != "manual":
                        self._start_recording()
                    return
                self._empty_transcripts = 0
                await self._send({"type": "stt", "text": transcript}, generation=generation)
                result = await self.chat(recorded, transcript)
                self._current(generation)
                self._event("reply_ready")
            spoken = result.get("reply") if isinstance(result, Mapping) else result
            if not isinstance(spoken, str) or not spoken.strip():
                raise ValueError("Chat did not return a reply")
            spoken = spoken.strip()[:self.config.max_reply_chars]
            await self._speak(spoken, generation)
            # Realtime firmware keeps its input processor running while it
            # speaks, so it does not send another listen/start after TTS.
            if self.mode == "realtime":
                self._start_recording()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.last_error = type(error).__name__
            self._event("turn_failed", error=self.last_error)
            LOGGER.warning("Puppy audio turn failed: %s", self.last_error)
            if not self.closed and generation == self._generation:
                try:
                    if self._tts_generation is not None:
                        await self._send({"type": "tts", "state": "stop"}, generation=generation)
                        self._tts_generation = None
                    # Device local-companion alerts are visual and do not bark.
                    await self._send({"type": "alert", "status": "语音暂不可用",
                                      "message": "请再试一次", "emotion": "neutral"}, generation=generation)
                    if self.mode != "manual":
                        self._start_recording()
                except Exception:
                    await self._terminate(1011)
        finally:
            if self.finish_turn is not None:
                await self.finish_turn()
            if self._turn is asyncio.current_task():
                self._turn = None
