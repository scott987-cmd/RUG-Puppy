"""Audio protocol regressions; no test plays through a speaker.

Run: python3 -m unittest companion.test_audio -v
Opt-in macOS file synthesis: PUPPY_TEST_SYSTEM_TTS=1 python3 -m unittest companion.test_audio -v
"""

import asyncio
import ctypes.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
import wave
from unittest.mock import patch

from companion.audio import (FRAME_SAMPLES, SAMPLE_RATE, OpusCodec, SpeechServices,
                             _run_process, boost_quiet_speech, opus_frames, pcm_rms)
from companion.audio_session import AudioSession, AudioSessionConfig


VOICE = struct.pack("<h", 1600) * FRAME_SAMPLES
SOFT_VOICE = struct.pack("<h", 300) * FRAME_SAMPLES
ROOM_NOISE = struct.pack("<h", 180) * FRAME_SAMPLES
LOW_VOICE = struct.pack("<h", 166) * FRAME_SAMPLES
QUIET_ROOM = struct.pack("<h", 40) * FRAME_SAMPLES
QUIET = b"\0" * len(VOICE)


async def until(predicate, timeout=1):
    async def wait():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), timeout)


class FakeSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.close_code = None
        self.block_binary = None
        self.binary_started = asyncio.Event()
        self.binary_cancelled = asyncio.Event()

    def message(self, message):
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(message)})

    def packet(self, data):
        self.incoming.put_nowait({"type": "websocket.receive", "bytes": data})

    def disconnect(self):
        self.incoming.put_nowait({"type": "websocket.disconnect"})

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, message):
        self.sent.append(message)

    async def send_bytes(self, data):
        self.binary_started.set()
        if self.block_binary is not None:
            try:
                await self.block_binary.wait()
            except asyncio.CancelledError:
                self.binary_cancelled.set()
                raise
        self.sent.append(data)

    async def close(self, code):
        self.close_code = code
        self.disconnect()

    def count(self, kind, state=None):
        return sum(isinstance(item, dict) and item.get("type") == kind and
                   (state is None or item.get("state") == state) for item in self.sent)

    @property
    def packets(self):
        return [item for item in self.sent if isinstance(item, bytes)]


class FakeCodec:
    def __init__(self):
        self.closed = False
        self.reset_count = 0

    def decode(self, packet):
        assert not self.closed
        if packet == b"voice":
            return VOICE
        if packet == b"soft":
            return SOFT_VOICE
        if packet == b"room":
            return ROOM_NOISE
        if packet == b"low-voice":
            return LOW_VOICE
        if packet == b"very-low-voice":
            return struct.pack("<h", 115) * FRAME_SAMPLES
        if packet == b"post-motion-voice":
            return struct.pack("<h", 70) * FRAME_SAMPLES
        if packet == b"distant-short-voice":
            return struct.pack("<h", 35) * FRAME_SAMPLES
        if packet == b"faint-short-voice":
            return struct.pack("<h", 22) * FRAME_SAMPLES
        if packet == b"very-still-room":
            return struct.pack("<h", 6) * FRAME_SAMPLES
        if packet == b"quiet-room":
            return QUIET_ROOM
        if packet == b"still-room":
            return struct.pack("<h", 12) * FRAME_SAMPLES
        if packet == b"quiet":
            return QUIET
        if packet == b"odd":
            return b"x"
        if packet == b"large":
            return VOICE * 10
        raise ValueError("malformed packet")

    def encode(self, pcm):
        assert not self.closed
        assert len(pcm) == len(VOICE)
        return b"encoded"

    def reset_encoder(self):
        assert not self.closed
        self.reset_count += 1

    def close(self):
        self.closed = True


class AudioSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tasks = []
        self.sessions = []

    async def asyncTearDown(self):
        for task, session in zip(self.tasks, self.sessions, strict=True):
            if not task.done():
                # Deliver the normal disconnect event. Cancelling a receive
                # with queued packets can race completion of asyncio.wait_for.
                session.socket.disconnect()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def start(self, *, transcribe=None, chat=None, synthesize=None, prepare_reply=None,
                    on_event=None, config=None, hello=None):
        socket, codec = FakeSocket(), FakeCodec()

        async def default_transcribe(pcm):
            return "你好"

        async def default_chat(pcm, transcript):
            return {"reply": "你好呀"}

        async def default_synthesize(text):
            return VOICE * 2

        session = AudioSession(socket, codec=codec, transcribe=transcribe or default_transcribe,
                               chat=chat or default_chat, synthesize=synthesize or default_synthesize,
                               prepare_reply=prepare_reply, on_event=on_event,
                               config=config or AudioSessionConfig(frame_interval=.001))
        socket.message(hello or {"type": "hello", "transport": "websocket"})
        task = asyncio.create_task(session.run())
        self.tasks.append(task)
        self.sessions.append(session)
        await until(lambda: socket.count("hello") or task.done())
        return socket, codec, session, task

    @staticmethod
    def utterance(socket, mode="manual"):
        socket.message({"type": "listen", "state": "start", "mode": mode})
        socket.packet(b"voice")
        socket.message({"type": "listen", "state": "stop"})

    async def test_roundtrip_preserves_pcm_identity_hook_and_manual_boundary(self):
        seen = []

        async def chat(pcm, transcript):
            seen.append((pcm, transcript))
            return {"reply": "hello"}

        socket, codec, _, task = await self.start(chat=chat)
        socket.message({"type": "listen", "state": "start", "mode": "manual"})
        socket.packet(b"voice")
        for _ in range(16):
            socket.packet(b"quiet")
        await until(socket.incoming.empty)
        self.assertFalse(seen)
        socket.message({"type": "listen", "state": "stop"})
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertEqual(seen, [(VOICE + QUIET * 16, "你好")])
        self.assertEqual(len(socket.packets), 2)
        self.assertEqual(codec.reset_count, 1)
        self.assertEqual([m["state"] for m in socket.sent if isinstance(m, dict) and m.get("type") == "tts"],
                         ["start", "sentence_start", "stop"])
        socket.disconnect()
        await task
        self.assertTrue(codec.closed)

    async def test_diagnostics_show_stages_without_private_content(self):
        events = []
        transcript = "孩子的私人问题"
        reply = "只给孩子的回答"

        async def transcribe(_pcm):
            return transcript

        async def chat(_pcm, _transcript):
            return {"reply": reply}

        socket, _, _, task = await self.start(
            transcribe=transcribe, chat=chat,
            on_event=lambda name, details: events.append((name, details)))
        self.utterance(socket)
        await until(lambda: socket.count("tts", "stop") == 1)
        socket.disconnect()
        await task
        names = [name for name, _ in events]
        for stage in ("connected", "listen_started", "recording_finished", "transcribed",
                      "reply_ready", "tts_ready", "tts_sent", "closed"):
            self.assertIn(stage, names)
        self.assertEqual(next(details for name, details in events if name == "transcribed"),
                         {"chars": len(transcript)})
        self.assertNotIn(transcript, repr(events))
        self.assertNotIn(reply, repr(events))

    async def test_realtime_listens_again_without_another_start_message(self):
        socket, _, session, _ = await self.start()
        self.utterance(socket, mode="realtime")
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertTrue(session.listening)
        socket.packet(b"voice")
        socket.message({"type": "listen", "state": "stop"})
        await until(lambda: socket.count("tts", "stop") == 2)
        self.assertTrue(session.listening)

    async def test_unrecognized_child_speech_gets_one_retry_then_rest(self):
        spoken = []
        async def transcribe(_pcm):
            return ""
        async def synthesize(text):
            spoken.append(text)
            return VOICE * 2
        socket, _, session, task = await self.start(
            transcribe=transcribe, synthesize=synthesize,
            config=AudioSessionConfig(frame_interval=.001, post_tts_echo_seconds=.05))
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        socket.packet(b"voice")
        socket.packet(b"voice")
        socket.message({"type": "listen", "state": "stop"})
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertTrue(session.listening)
        self.assertEqual(spoken, ["我没听清，再说一次好吗？"])
        await asyncio.sleep(.06)
        socket.packet(b"voice")
        socket.packet(b"voice")
        socket.message({"type": "listen", "state": "stop"})
        await asyncio.wait_for(task, 1)
        self.assertEqual(spoken[-1], "我还是没听清，我们等会儿再试吧。")
        self.assertEqual(socket.close_code, 1000)

    async def test_empty_single_frame_transient_reopens_without_spoken_retry(self):
        recordings, events = [], []
        async def transcribe(pcm):
            recordings.append(pcm)
            return ""
        socket, _, session, _ = await self.start(
            transcribe=transcribe,
            on_event=lambda name, details: events.append((name, details)))
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        socket.packet(b"voice")  # Captured tail/impulse: 60 ms, then silence.
        for _ in range(15):
            socket.packet(b"quiet")
        await until(lambda: len([n for n, _ in events if n == "listen_started"]) == 2)
        self.assertEqual(len(recordings), 1)
        self.assertTrue(session.listening)
        self.assertEqual(socket.count("tts"), 0)
        self.assertEqual(session._empty_transcripts, 0)

    async def test_speaker_tail_after_reply_is_not_treated_as_child_speech(self):
        socket, _, session, _ = await self.start(
            config=AudioSessionConfig(frame_interval=.001, post_tts_echo_seconds=.05))
        self.utterance(socket, mode="auto")
        await until(lambda: socket.count("tts", "stop") == 1)
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        socket.packet(b"voice")  # Echo from the device speaker.
        await until(socket.incoming.empty)
        self.assertFalse(session._speech_seen)
        self.assertEqual(session.recorded_bytes, 0)
        await asyncio.sleep(.06)
        socket.packet(b"voice")  # A new child utterance after the speaker tail.
        socket.message({"type": "listen", "state": "stop"})
        await until(lambda: socket.count("tts", "stop") == 2)

    async def test_late_device_playback_completion_gets_its_own_echo_window(self):
        socket, _, session, _ = await self.start(
            config=AudioSessionConfig(frame_interval=.001, post_tts_echo_seconds=.05))
        self.utterance(socket, mode="auto")
        await until(lambda: socket.count("tts", "stop") == 1)
        # Device output may drain after the gateway's original guard expired.
        await asyncio.sleep(.06)
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        socket.packet(b"voice")
        await until(socket.incoming.empty)
        self.assertFalse(session._speech_seen)
        self.assertEqual(session.recorded_bytes, 0)
        await asyncio.sleep(.06)
        socket.packet(b"voice")
        socket.message({"type": "listen", "state": "stop"})
        await until(lambda: socket.count("tts", "stop") == 2)

    async def test_enrollment_recording_skips_asr_and_conversation(self):
        seen = []
        async def prepare(pcm):
            seen.append(pcm)
            return {"reply": "登记好了"}
        async def unexpected(*args):
            self.fail("Enrollment audio must not enter ASR or chat")
        socket, _, _, _ = await self.start(prepare_reply=prepare, transcribe=unexpected, chat=unexpected)
        self.utterance(socket)
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertEqual(seen, [VOICE])
        self.assertEqual(socket.count("stt"), 0)

    async def test_abort_during_asr_discards_even_a_late_return_and_accepts_new_listen(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        seen = []
        calls = 0

        async def transcribe(pcm):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    return "过期的转写"
            return "新的转写"

        async def chat(pcm, transcript):
            seen.append(transcript)
            return "新的回答"

        socket, _, _, _ = await self.start(transcribe=transcribe, chat=chat)
        self.utterance(socket)
        await started.wait()
        socket.message({"type": "abort"})
        await asyncio.wait_for(cancelled.wait(), .2)
        self.utterance(socket)
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertEqual(seen, ["新的转写"])
        self.assertNotIn("过期的转写", str(socket.sent))

    async def test_new_listen_cancels_chat_before_synthesis(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        spoken = []

        async def chat(pcm, transcript):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def synthesize(text):
            spoken.append(text)
            return VOICE

        socket, _, session, _ = await self.start(chat=chat, synthesize=synthesize)
        self.utterance(socket)
        await entered.wait()
        socket.message({"type": "listen", "state": "start"})
        await asyncio.wait_for(cancelled.wait(), .2)
        await until(lambda: session.listening)
        self.assertFalse(spoken)
        self.assertEqual(socket.count("tts", "start"), 0)

    async def test_abort_stops_tts_packets(self):
        async def synthesize(text):
            return VOICE * 30

        events = []
        socket, _, _, _ = await self.start(synthesize=synthesize,
                                           config=AudioSessionConfig(frame_interval=.01),
                                           on_event=lambda name, details: events.append((name, details)))
        self.utterance(socket)
        await until(lambda: len(socket.packets) >= 1)
        socket.message({"type": "abort", "reason": "wake_word_detected"})
        await until(lambda: socket.count("tts", "stop") == 1)
        sent = len(socket.packets)
        await asyncio.sleep(.04)
        self.assertEqual(len(socket.packets), sent)
        self.assertLess(sent, 30)
        self.assertIn(("abort_received", {"phase": "tts", "reason": "wake_word_detected"}), events)

    async def test_parent_stop_closes_active_tts_without_more_audio(self):
        async def synthesize(_text):
            return VOICE * 30

        events = []
        socket, codec, session, task = await self.start(
            synthesize=synthesize, config=AudioSessionConfig(frame_interval=.01),
            on_event=lambda name, details: events.append((name, details)))
        self.utterance(socket)
        await until(lambda: len(socket.packets) >= 1)
        self.assertTrue(await session.cancel_and_close())
        await asyncio.wait_for(task, 1)
        sent = len(socket.packets)
        await asyncio.sleep(.04)
        self.assertEqual(len(socket.packets), sent)
        self.assertLess(sent, 30)
        self.assertEqual(socket.count("tts", "stop"), 1)
        self.assertEqual(socket.close_code, 1000)
        self.assertTrue(codec.closed)
        self.assertFalse(await session.cancel_and_close())
        self.assertIn(("parent_audio_stopped", {}), events)

    async def test_abort_cancels_a_backpressured_send_and_releases_send_lock(self):
        socket, _, _, _ = await self.start()
        socket.block_binary = asyncio.Event()
        self.utterance(socket)
        await socket.binary_started.wait()
        socket.message({"type": "abort"})
        await asyncio.wait_for(socket.binary_cancelled.wait(), .2)
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertFalse(socket.packets)

    async def test_disconnect_cancels_synthesis_and_closes_codec(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def synthesize(text):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        socket, codec, _, task = await self.start(synthesize=synthesize)
        self.utterance(socket)
        await entered.wait()
        socket.disconnect()
        await asyncio.wait_for(task, .2)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(codec.closed)
        self.assertEqual(socket.count("tts", "start"), 0)

    async def test_voice_and_silent_recordings_are_bounded(self):
        recordings = []

        async def transcribe(pcm):
            recordings.append(pcm)
            return ""

        config = AudioSessionConfig(max_record_seconds=.12, pre_roll_seconds=.06, frame_interval=0)
        socket, _, session, _ = await self.start(transcribe=transcribe, config=config)
        socket.message({"type": "listen", "state": "start"})
        for _ in range(100):
            socket.packet(b"quiet")
        await until(socket.incoming.empty)
        self.assertLessEqual(session.recorded_bytes, len(QUIET))
        self.assertFalse(recordings)
        socket.packet(b"voice")
        await until(lambda: bool(recordings))
        self.assertLessEqual(len(recordings[0]), len(VOICE) * 2)

    async def test_soft_voice_above_room_noise_starts_and_finishes_recording(self):
        recordings = []

        async def transcribe(pcm):
            recordings.append(pcm)
            return ""

        socket, _, session, _ = await self.start(transcribe=transcribe)
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        for _ in range(20):
            socket.packet(b"room")
        await until(socket.incoming.empty)
        self.assertFalse(session._speech_seen)
        self.assertFalse(recordings)
        socket.packet(b"soft")
        for _ in range(16):
            socket.packet(b"quiet")
        await until(lambda: bool(recordings))
        self.assertTrue(recordings[0].startswith(ROOM_NOISE))
        self.assertIn(SOFT_VOICE, recordings[0])

    async def test_low_voice_after_quiet_room_reaches_transcription_and_reply(self):
        recordings = []

        async def transcribe(pcm):
            recordings.append(pcm)
            return "三加二等于几"

        socket, _, _, _ = await self.start(transcribe=transcribe)
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        for _ in range(40):
            socket.packet(b"quiet-room")
        for _ in range(6):
            socket.packet(b"low-voice")
        for _ in range(16):
            socket.packet(b"quiet-room")

        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertEqual(len(recordings), 1)
        self.assertIn(LOW_VOICE * 6, recordings[0])
        self.assertIn({"type": "stt", "text": "三加二等于几"}, socket.sent)

    async def test_separate_low_volume_noises_do_not_start_a_turn(self):
        socket, _, _, task = await self.start(config=AudioSessionConfig(max_listen_seconds=.1))
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        for _ in range(40):
            socket.packet(b"quiet-room")
        for _ in range(3):
            socket.packet(b"low-voice")
            socket.packet(b"low-voice")
            socket.packet(b"quiet-room")

        await asyncio.wait_for(task, .5)
        self.assertEqual(socket.close_code, 1000)
        self.assertEqual(socket.count("stt"), 0)
        self.assertEqual(socket.count("tts"), 0)

    async def test_very_quiet_followup_reaches_asr_without_a_longer_silence_window(self):
        recordings = []
        async def transcribe(pcm):
            recordings.append(pcm)
            return "这个词是什么意思呀"
        socket, _, _, _ = await self.start(transcribe=transcribe)
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        for _ in range(80):
            socket.packet(b"quiet-room")
        for _ in range(8):
            socket.packet(b"very-low-voice")
        for _ in range(15):
            socket.packet(b"quiet-room")
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertEqual(len(recordings), 1)
        self.assertIn(struct.pack("<h", 115) * FRAME_SAMPLES * 8, recordings[0])

    async def test_onset_noise_does_not_cut_off_the_following_quiet_words(self):
        recordings = []
        async def transcribe(pcm):
            recordings.append(pcm)
            return "为什么月亮会发光呀"
        socket, _, _, _ = await self.start(transcribe=transcribe)
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        socket.packet(b"soft")
        for _ in range(8):
            socket.packet(b"quiet-room")
        for _ in range(8):
            socket.packet(b"very-low-voice")
        for _ in range(15):
            socket.packet(b"quiet-room")
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertEqual(len(recordings), 1)
        self.assertIn(struct.pack("<h", 115) * FRAME_SAMPLES * 8, recordings[0])

    async def test_quiet_followup_reuses_room_calibration_instead_of_timing_out(self):
        recordings = []
        async def transcribe(pcm):
            recordings.append(pcm)
            return "这个词是什么意思"
        socket, _, session, _ = await self.start(
            transcribe=transcribe, config=AudioSessionConfig(frame_interval=.001, post_tts_echo_seconds=0))
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        socket.packet(b"voice")
        for _ in range(15):
            socket.packet(b"still-room")
        await until(lambda: socket.count("tts", "stop") == 1)
        socket.message({"type": "listen", "state": "start", "mode": "auto"})
        await until(lambda: session.listening)
        for _ in range(8):
            socket.packet(b"very-low-voice")
        for _ in range(15):
            socket.packet(b"still-room")
        await until(lambda: socket.count("tts", "stop") == 2)
        self.assertEqual(len(recordings), 2)
        self.assertIn(struct.pack("<h", 115) * FRAME_SAMPLES * 8, recordings[1])

    async def test_motion_stop_hears_pcm_while_action_waits_and_fences_old_reply(self):
        entered, cancelled, heard = asyncio.Event(), asyncio.Event(), asyncio.Event()
        owner, spoken = [None], []
        calls = 0
        async def chat(pcm, text):
            nonlocal calls
            calls += 1
            if calls > 1:
                return "继续聊天"
            owner[0] = object()
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                return "旧动作已经完成"  # Must be fenced even if cancellation is swallowed.
        async def stop(pcm, token):
            self.assertIs(token, owner[0])
            self.assertIn(VOICE, pcm)
            owner[0] = None
            heard.set()
            return {"reply": "好，停下了。", "confirmed": True}
        async def synthesize(text):
            spoken.append(text)
            return VOICE * 2
        socket, _, session, _ = await self.start(
            chat=chat, synthesize=synthesize,
            config=AudioSessionConfig(frame_interval=.001, post_tts_echo_seconds=0))
        session.motion_owner, session.stop_motion = lambda: owner[0], stop
        self.utterance(socket, mode="auto")
        await until(entered.is_set)
        socket.packet(b"voice")
        for _ in range(12):
            socket.packet(b"quiet")
        await until(heard.is_set)
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(spoken, ["好，停下了。"])
        self.utterance(socket, mode="auto")
        await until(lambda: socket.count("tts", "stop") == 2)
        self.assertEqual(spoken, ["好，停下了。", "继续聊天"])

    async def test_post_motion_quiet_followup_below_old_floor_is_not_lost(self):
        recordings = []
        async def transcribe(pcm):
            recordings.append(pcm)
            return "小猫有几条腿"
        socket, _, session, _ = await self.start(
            transcribe=transcribe,
            config=AudioSessionConfig(frame_interval=.001, post_tts_echo_seconds=0))
        socket.message({"type":"listen","state":"start","mode":"auto"})
        socket.packet(b"voice")
        for _ in range(20):socket.packet(b"still-room")
        await until(lambda: socket.count("tts","stop")==1)
        socket.message({"type":"listen","state":"start","mode":"auto"})
        for _ in range(8):socket.packet(b"post-motion-voice")
        for _ in range(15):socket.packet(b"still-room")
        await until(lambda: socket.count("tts","stop")==2)
        self.assertEqual(len(recordings),2)
        self.assertIn(struct.pack("<h",70)*FRAME_SAMPLES*8,recordings[-1])

    async def test_new_connection_calibrates_before_the_first_quiet_short_command(self):
        socket, _, _, _ = await self.start()
        socket.message({"type":"listen","state":"start","mode":"auto"})
        for _ in range(15):socket.packet(b"still-room")
        for _ in range(8):socket.packet(b"distant-short-voice")
        for _ in range(15):socket.packet(b"still-room")
        await until(lambda: socket.count("tts","stop")==1)

    async def test_faint_short_voice_in_measured_six_rms_room_is_not_discarded(self):
        socket, _, _, _ = await self.start()
        socket.message({"type":"listen","state":"start","mode":"auto"})
        for _ in range(20):socket.packet(b"very-still-room")
        for _ in range(7):socket.packet(b"faint-short-voice")
        for _ in range(15):socket.packet(b"very-still-room")
        await until(lambda: socket.count("tts","stop")==1)

    async def test_prompted_confirmation_captures_a_single_soft_syllable(self):
        socket, _, session, _ = await self.start()
        session.short_answer_expected = lambda: True
        socket.message({"type":"listen","state":"start","mode":"auto"})
        for _ in range(20):socket.packet(b"very-still-room")
        socket.packet(b"faint-short-voice")
        for _ in range(15):socket.packet(b"very-still-room")
        await until(lambda: socket.count("tts","stop")==1)

    async def test_nonstop_motor_audio_does_not_cancel_action_or_speak(self):
        entered, checked = asyncio.Event(), asyncio.Event()
        owner = object()
        async def chat(pcm, text):
            entered.set()
            await asyncio.Event().wait()
        async def stop(pcm, token):
            checked.set()
            return None
        socket, _, session, _ = await self.start(chat=chat)
        session.motion_owner, session.stop_motion = lambda: owner, stop
        self.utterance(socket, mode="auto")
        await until(entered.is_set)
        socket.packet(b"voice")
        for _ in range(12):
            socket.packet(b"quiet")
        await until(checked.is_set)
        self.assertFalse(session._turn.done())
        self.assertEqual(socket.count("tts"), 0)

    async def test_motion_stop_recognizer_is_not_open_for_an_idle_connection(self):
        seen = []
        async def stop(pcm, token):
            seen.append(pcm)
        socket, _, session, _ = await self.start()
        session.motion_owner, session.stop_motion = lambda: None, stop
        socket.packet(b"voice")
        for _ in range(12):
            socket.packet(b"quiet")
        await until(socket.incoming.empty)
        self.assertFalse(seen)

    async def test_motion_stop_retains_the_whole_phrase_across_soft_internal_pauses(self):
        entered, heard = asyncio.Event(), asyncio.Event()
        owner=[None]
        phrase=VOICE*3+QUIET*6+VOICE*3+QUIET*6+VOICE*3
        async def chat(pcm,text):
            owner[0]=object();entered.set()
            await asyncio.Event().wait()
        async def stop(pcm,token):
            if phrase not in pcm:return None
            owner[0]=None;heard.set()
            return {"reply":"停下了","confirmed":True}
        socket,_,session,_=await self.start(chat=chat,
            config=AudioSessionConfig(frame_interval=.001,post_tts_echo_seconds=0))
        session.motion_owner,session.stop_motion=lambda:owner[0],stop
        self.utterance(socket,mode="auto");await until(entered.is_set)
        for packets in ([b"voice"]*3+[b"quiet"]*6+[b"voice"]*3+
                        [b"quiet"]*6+[b"voice"]*3+[b"quiet"]*12):socket.packet(packets)
        await until(heard.is_set)
        await until(lambda: socket.count("tts","stop")==1)

    async def test_partial_negation_consumes_its_tail_and_preserves_new_stop_during_asr(self):
        # Simulate two ASR calls with additional packets arriving during each call.
        owner = object()
        prefix, suffix, new_word = VOICE * 10, SOFT_VOICE * 10, LOW_VOICE * 10
        calls = []
        async def recognize(pcm, token, *, resolving_negation=False):
            calls.append((pcm, resolving_negation))
            if len(calls) == 1:
                for _ in range(10):
                    await session._receive_motion_audio(b"soft")
                return {"negation_prefix": True}
            if resolving_negation:
                for _ in range(10):
                    await session._receive_motion_audio(b"low-voice")
            return None

        _, _, session, _ = await self.start()
        session.motion_owner, session.stop_motion = lambda: owner, recognize
        for _ in range(10):
            await session._receive_motion_audio(b"voice")
        await until(lambda: len(calls) == 3)
        self.assertEqual(calls, [(prefix, False), (prefix + suffix, True), (new_word, False)])

    async def test_new_listen_during_stop_cleanup_cannot_receive_an_old_stop_ack(self):
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        owner, spoken = [None], []
        calls=0
        async def chat(pcm,text):
            nonlocal calls
            calls+=1
            if calls>1:return "新问题的答复"
            owner[0]=object();entered.set()
            try:await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                return "旧动作答复"
        async def stop(pcm,token):
            owner[0]=None
            return {"reply":"旧停止确认","confirmed":True}
        async def synthesize(text):
            spoken.append(text)
            return VOICE*2
        socket,_,session,_=await self.start(chat=chat,synthesize=synthesize,
            config=AudioSessionConfig(frame_interval=.001,post_tts_echo_seconds=0))
        session.motion_owner,session.stop_motion=lambda:owner[0],stop
        self.utterance(socket,mode="auto");await until(entered.is_set)
        socket.packet(b"voice")
        for _ in range(12):socket.packet(b"quiet")
        await until(cleaning.is_set)
        socket.message({"type":"listen","state":"start","mode":"auto"})
        await until(lambda: session.listening)
        release.set()
        await asyncio.sleep(.02)
        socket.packet(b"voice");socket.message({"type":"listen","state":"stop"})
        await until(lambda: "新问题的答复" in spoken)
        self.assertEqual(spoken,["新问题的答复"])

    async def test_persistent_decode_errors_close_without_transcription(self):
        socket, codec, session, task = await self.start()
        socket.message({"type": "listen", "state": "start"})
        for packet in (b"bad", b"odd", b"large"):
            socket.packet(packet)
        await asyncio.wait_for(task, 1)
        self.assertEqual(socket.close_code, 1003)
        self.assertEqual(session.last_error, "invalid_audio")
        self.assertTrue(codec.closed)
        self.assertEqual(socket.count("stt"), 0)

    async def test_isolated_decode_error_does_not_break_valid_audio(self):
        socket, _, _, _ = await self.start()
        socket.message({"type": "listen", "state": "start", "mode": "manual"})
        socket.packet(b"bad")
        socket.packet(b"voice")
        socket.message({"type": "listen", "state": "stop"})
        await until(lambda: socket.count("tts", "stop") == 1)
        self.assertIsNone(socket.close_code)

    async def test_invalid_negotiation_closes_and_silent_listening_expires(self):
        socket, codec, _, task = await self.start(hello={"type": "hello", "transport": "websocket",
                                                       "audio_params": {"sample_rate": 24000}})
        await task
        self.assertEqual(socket.close_code, 1002)
        self.assertTrue(codec.closed)
        socket, codec, _, task = await self.start(config=AudioSessionConfig(max_listen_seconds=.02))
        socket.message({"type": "listen", "state": "start"})
        await asyncio.wait_for(task, .5)
        self.assertEqual(socket.close_code, 1000)
        self.assertTrue(codec.closed)

    async def test_synthesis_failure_recovers_visually_without_audio(self):
        async def synthesize(text):
            raise RuntimeError("fake TTS failure")

        socket, _, session, _ = await self.start(synthesize=synthesize)
        self.utterance(socket, mode="auto")
        await until(lambda: socket.count("alert") == 1)
        self.assertTrue(session.listening)
        self.assertFalse(socket.packets)


class SpeechServicesTest(unittest.IsolatedAsyncioTestCase):
    async def test_vocabulary_speaks_the_word_in_english_and_keeps_a_single_voice_fallback(self):
        calls = []
        english_available = True

        async def render(service, text):
            calls.append((service.voice, text))
            if service.voice == "Samantha":
                if not english_available:
                    raise RuntimeError("Voice unavailable")
                return b"\x02\x00"
            return b"\x01\x00"

        speech = SpeechServices("unused")
        with patch.object(SpeechServices, "synthesize_async", render):
            pcm = await speech.synthesize_vocabulary_async("小猫的英文是", "cat")
            self.assertEqual(pcm, b"\x01\x00" + bytes(SAMPLE_RATE // 8 * 2) + b"\x02\x00" + bytes(FRAME_SAMPLES * 4))
            self.assertCountEqual(calls, [("Tingting", "小猫的英文是"), ("Samantha", "Cat. C, A, T.")])
            english_available = False
            pcm = await speech.synthesize_vocabulary_async("小猫的英文是", "cat")
        self.assertEqual(pcm, b"\x01\x00")
        self.assertEqual(calls[-1], ("Tingting", "小猫的英文是 cat。"))

    def test_quiet_speech_gain_is_bounded_and_leaves_loud_audio_alone(self):
        quiet = struct.pack("<hhh", -292, 0, 160)
        self.assertEqual(struct.unpack("<hhh", boost_quiet_speech(quiet)), (-2336, 0, 1280))
        louder = struct.pack("<hh", -32768, 6000)
        self.assertEqual(boost_quiet_speech(louder), louder)
        self.assertEqual(boost_quiet_speech(b"\0\0"), b"\0\0")

    def test_empty_system_audio_file_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="puppy-empty-audio-") as directory:
            path = Path(directory) / "empty.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(SAMPLE_RATE)
            with self.assertRaises(ValueError):
                SpeechServices._read_pcm(path)

    async def test_cancelled_asr_keeps_real_worker_serial_and_queued_cancellation_skips_work(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        active = 0
        maximum = 0
        accounting = threading.Lock()

        class FakeSpeech(SpeechServices):
            def _transcribe(self, pcm):
                nonlocal active, maximum
                with accounting:
                    active += 1
                    maximum = max(maximum, active)
                    calls.append(self.whisper_model)
                try:
                    if self.whisper_model == "first":
                        entered.set()
                        release.wait(timeout=2)
                    return self.whisper_model
                finally:
                    with accounting:
                        active -= 1

        pcm = VOICE * 6
        first = asyncio.create_task(FakeSpeech("first").transcribe_async(pcm))
        second = third = None
        try:
            await until(entered.is_set)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(first, .2)
            second = asyncio.create_task(FakeSpeech("second").transcribe_async(pcm))
            third = asyncio.create_task(FakeSpeech("third").transcribe_async(pcm))
            await asyncio.sleep(.02)
            third.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await third
            self.assertEqual(calls, ["first"])
            release.set()
            self.assertEqual(await asyncio.wait_for(second, 1), "second")
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(maximum, 1)
        finally:
            release.set()
            await asyncio.gather(*(task for task in (first, second, third) if task), return_exceptions=True)

    async def test_cancelled_subprocess_is_terminated_and_reaped(self):
        with tempfile.TemporaryDirectory(prefix="puppy-process-test-") as directory:
            marker = Path(directory) / "pid"
            command = [sys.executable, "-c",
                       f"import os,time;from pathlib import Path;Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(60)"]
            task = asyncio.create_task(_run_process(command, timeout=5))
            try:
                await until(marker.exists)
                pid = int(marker.read_text())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, .5)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    @unittest.skipUnless(sys.platform == "darwin" and os.environ.get("PUPPY_TEST_SYSTEM_TTS") == "1",
                         "Set PUPPY_TEST_SYSTEM_TTS=1 for macOS say-to-file validation")
    async def test_system_say_writes_valid_pcm_file_without_playback(self):
        speech = SpeechServices("unused", voice=os.environ.get("PUPPY_TEST_VOICE", "Tingting"))
        pcm = await speech.synthesize_async("你好")
        self.assertTrue(pcm)
        self.assertEqual(len(pcm) % 2, 0)
        self.assertGreater(pcm_rms(pcm), 0)


class OpusTest(unittest.TestCase):
    def test_rms_rejects_partial_sample(self):
        self.assertAlmostEqual(pcm_rms(VOICE), 1600)
        with self.assertRaises(ValueError):
            pcm_rms(b"x")

    @unittest.skipUnless(ctypes.util.find_library("opus") or os.environ.get("PUPPY_OPUS_LIBRARY"),
                         "libopus is unavailable")
    def test_native_opus_roundtrip_padding_and_close(self):
        codec = OpusCodec()
        try:
            pcm = b"".join(struct.pack("<h", int(1000 * math.sin(2 * math.pi * 440 * n / SAMPLE_RATE)))
                           for n in range(FRAME_SAMPLES + 37))
            codec.reset_encoder()
            packets = list(opus_frames(codec, pcm))
            self.assertEqual(len(packets), 2)
            decoded = b"".join(codec.decode(packet) for packet in packets)
            self.assertEqual(len(decoded), FRAME_SAMPLES * 2 * 2)
            self.assertGreater(pcm_rms(decoded), 0)
            with self.assertRaises(ValueError):
                codec.decode(b"")
        finally:
            codec.close()
        codec.close()
        with self.assertRaises(ValueError):
            codec.encode(VOICE)
        with self.assertRaises(ValueError):
            codec.decode(b"invalid")


if __name__ == "__main__":
    unittest.main()
