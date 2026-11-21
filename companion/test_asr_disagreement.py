import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from companion.asr_disagreement import (
    CLARIFY_HOMOPHONE, apple_transcribe_pcm, clarification_for_transcripts,
    numeric_only_clarification, prefer_clear_arithmetic_question,
    prefer_quiet_translation_question, public_animal_body_count, public_animal_group_count, resolve_body_count,
)
from companion.audio_identity import device_audio_conversation_id


class DisagreementTests(unittest.TestCase):
    def test_count_repair_requires_the_same_question_and_an_unclear_body_part(self):
        legs = "小猫有几条腿呀？"
        feet = "小猫有几只脚丫 ？"
        self.assertEqual(resolve_body_count("小貓有幾條呀。", legs), (legs, None))
        self.assertEqual(resolve_body_count("小猫有几只狡雅。", feet), (feet, None))
        self.assertEqual(resolve_body_count("小猫有几只脚鸭", feet), (feet, None))
        for primary, secondary in (
            ("小狗有几条呀", legs),
            ("小猫有几只狡雅。", "小猫有几只鸭子？"),
            ("小猫有几条呀，你还记得我吗？", legs),
        ):
            with self.subTest(primary=primary, secondary=secondary):
                self.assertEqual(resolve_body_count(primary, secondary), (primary, None))

    def test_invalid_body_part_unit_uses_agreeing_generic_secondary(self):
        legs = "小猫有几条腿呀？"
        self.assertEqual(resolve_body_count("小猫有几条嘴呀", legs), (legs, None))
        self.assertEqual(resolve_body_count("小猫有几条眼睛呀", legs), (legs, None))
        self.assertEqual(resolve_body_count("小狗有几条嘴呀", legs), ("小狗有几条嘴呀", None))
        self.assertEqual(resolve_body_count("我家小猫有几条嘴呀", legs),
                         ("我家小猫有几条嘴呀", None))

    def test_conflicting_valid_body_parts_ask_which_question_was_spoken(self):
        legs = "小猫有几条腿呀？"
        for primary in ("小猫有几张嘴呀？", "小猫有几条尾巴？"):
            with self.subTest(primary=primary):
                selected, clarification = resolve_body_count(primary, legs)
                self.assertEqual(selected, primary)
                self.assertIn("还是有几条腿", clarification)
        self.assertEqual(resolve_body_count("小猫有几只脚呀", legs),
                         ("小猫有几只脚呀", None))

    def test_natural_pauses_preserve_exact_public_count_questions(self):
        self.assertEqual(public_animal_body_count("小猫，有几条，腿呀？"), ("小猫", "腿"))
        self.assertIsNone(public_animal_body_count("小猫有几条腿，还有几个耳朵？"))
        self.assertIsNone(public_animal_body_count("我家的小猫，有几条腿？"))

    def test_generic_group_count_preserves_explicit_quantity(self):
        self.assertEqual(public_animal_group_count("那兩隻小貓，一共有幾條腿呀？"), ("小猫", "腿", 2))
        self.assertEqual(public_animal_group_count("大兩隻小貓一共有幾條腿呀!"), ("小猫", "腿", 2))
        self.assertEqual(public_animal_group_count("12只小鸭加起来有几只脚？"), ("小鸭", "脚", 12))
        self.assertIsNone(public_animal_group_count("我家两只小猫一共有几条腿？"))
        self.assertIsNone(public_animal_group_count("两只小猫和一只小狗一共有几条腿？"))
        two = "两只小猫一共有几条腿？"
        self.assertEqual(resolve_body_count(two, "六只小猫一共有几条腿？"),
                         (two, "这次是2只还是6只小猫呀？"))

    def test_secondary_recovers_a_same_animal_leg_fragment_without_a_parsed_count(self):
        legs = "小猫有几条腿？"
        for primary in ("小猫有鸡条腿", "小貓有短短腿", "小猫有湿润脚"):
            with self.subTest(primary=primary):
                self.assertEqual(resolve_body_count(primary, legs), (legs, None))
        for primary in ("小狗有短短腿", "我家小猫有短短腿", "小猫有三条腿",
                        "小猫有几双腿", "小猫的腿受伤了", "小猫有几条尾巴"):
            with self.subTest(primary=primary):
                selected, clarification = resolve_body_count(primary, legs)
                self.assertEqual(selected, primary)

    def test_body_part_disagreement_uses_the_same_animals_previous_count_or_clarifies(self):
        primary, secondary = "小猫有几只小鸭？", "小猫有几只脚丫？"
        self.assertEqual(resolve_body_count(primary, secondary, "小猫有几条腿呀？"), (secondary, None))
        self.assertEqual(resolve_body_count(primary, secondary), (
            primary, "你问的是小猫有几只脚，还是有几只小鸭呀？"))
        self.assertEqual(resolve_body_count(primary, secondary, "小狗有几条腿呀？")[0], primary)

    def test_public_count_question_excludes_owned_animals_and_multiple_requests(self):
        self.assertEqual(public_animal_body_count("小貓有幾隻腳丫 ？"), ("小猫", "脚丫"))
        self.assertEqual(public_animal_body_count("小狗一般有几条腿呀？"), ("小狗", "腿"))
        self.assertIsNone(public_animal_body_count("我家的小猫有几只脚？"))
        self.assertIsNone(public_animal_body_count("小猫有几只脚？你还记得我家小猫叫什么吗？"))
        self.assertIsNone(public_animal_body_count("小猫有几只小鸭？"))

    def test_quiet_mandarin_translation_preserves_the_question_shape(self):
        self.assertEqual(prefer_quiet_translation_question(
            "聽說過英語怎麼說呀？", "苹果用英语怎么说呀？"), "苹果用英语怎么说呀？")
        self.assertEqual(prefer_quiet_translation_question(
            "汪詠英語怎麼說呀?", "猫用英语怎么说呀 ？"), "猫用英语怎么说呀 ？")
        self.assertEqual(prefer_quiet_translation_question(
            "蘋果的英文是什麼？", "苹果"), "蘋果的英文是什麼？")
        self.assertEqual(prefer_quiet_translation_question("apple", "苹果"), "apple")

    def test_secondary_repairs_only_matching_malformed_arithmetic_question(self):
        self.assertEqual(prefer_clear_arithmetic_question(
            "2加2等于紧", "2+2等于几？"), "2+2等于几？")
        self.assertEqual(prefer_clear_arithmetic_question(
            "2加2等于几", "2+2等于几？"), "2加2等于几")
        self.assertEqual(prefer_clear_arithmetic_question(
            "3加2等于紧", "2+2等于几？"), "3加2等于紧")
        self.assertEqual(prefer_clear_arithmetic_question(
            "2加2的雨季", "2+2等于几？"), "2+2等于几？")
        self.assertEqual(prefer_clear_arithmetic_question(
            "3加2的雨季", "2+2等于几？"), "3加2的雨季")
        self.assertEqual(prefer_clear_arithmetic_question(
            "2加2的故事", "2+2等于几？"), "2加2的故事")
        self.assertEqual(prefer_clear_arithmetic_question(
            "小狗叫什么", "小狗叫几？"), "小狗叫什么")
        self.assertEqual(prefer_clear_arithmetic_question(
            "3200", "3+2等于几？"), "3+2等于几？")
        self.assertEqual(prefer_clear_arithmetic_question(
            "3200", "三加二等于几？"), "三加二等于几？")
        self.assertEqual(prefer_clear_arithmetic_question(
            "3200", "4+2等于几？"), "3200")

    def test_bare_long_number_asks_child_to_repeat(self):
        self.assertIsNotNone(numeric_only_clarification("3200"))
        self.assertIsNone(numeric_only_clarification("32"))
        self.assertIsNone(numeric_only_clarification("3+2等于几"))

    def test_short_banana_conflict_asks(self):
        self.assertEqual(clarification_for_transcripts(
            "你好，相交的英文怎么说？", "你好，香蕉的英文怎么说？"), CLARIFY_HOMOPHONE)

    def test_geometry_context_conflict_asks_even_when_both_mishear(self):
        self.assertEqual(clarification_for_transcripts(
            "两个元香蕉的英文怎么说？", "两个圆香蕉的英文怎么说？"), CLARIFY_HOMOPHONE)

    def test_unrelated_and_consistent_questions_pass(self):
        self.assertIsNone(clarification_for_transcripts(
            "苹果的英文怎么说？", "苹果的英文怎么说？"))
        self.assertIsNone(clarification_for_transcripts(
            "水果香蕉用英语怎么说？", "水果香蕉用英语怎么说？"))
        self.assertIsNone(clarification_for_transcripts(
            "两个圆相交的英文怎么说？", "两个圆相交的英文怎么说？"))


class SecondaryProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_receives_private_wav_and_removes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "fake-asr"
            binary.write_text("#!/usr/bin/env python3\nimport sys,wave\n"
                              "with wave.open(sys.argv[1], 'rb') as w:\n"
                              " assert (w.getnchannels(),w.getsampwidth(),w.getframerate()) == (1,2,16000)\n"
                              "print('香蕉')\n")
            binary.chmod(0o700)
            self.assertEqual(await apple_transcribe_pcm(b"\0\0" * 8000, binary), "香蕉")

    async def test_timeout_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "slow-asr"
            binary.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n")
            binary.chmod(0o700)
            with self.assertRaises(asyncio.TimeoutError):
                await apple_transcribe_pcm(b"\0\0" * 8000, binary, timeout=0.05)

    async def test_cancel_terminates_secondary_process(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "slow-asr"
            binary.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n")
            binary.chmod(0o700)
            task = asyncio.create_task(apple_transcribe_pcm(b"\0\0" * 8000, binary))
            await asyncio.sleep(0.1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)


class AudioGatewayIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_count_disagreement_is_handled_at_both_input_levels_before_chat(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PUPPY_DATA_DIR": directory}):
            try:
                from companion import server
            except ModuleNotFoundError as error:
                if error.name == "fastapi":
                    self.skipTest("Gateway dependencies are not installed locally")
                raise
            primary_text = ""
            recorded_level = 0
            answers = []
            class Socket:
                headers = {"authorization": "Bearer test-device", "device-id": "synthetic"}
                accept = AsyncMock()
            class Speech:
                def __init__(self, _model):
                    pass
                async def transcribe_async(self, _pcm):
                    return primary_text
            class Session:
                def __init__(self, _socket, **callbacks):
                    self.callbacks = callbacks
                async def run(self):
                    pcm = recorded_level.to_bytes(2, "little", signed=True) * 8000
                    await self.callbacks["prepare_reply"](pcm)
                    text = await self.callbacks["transcribe"](pcm)
                    answers.append(await self.callbacks["chat"](pcm, text))
            with ExitStack() as stack:
                for name, value in {
                    "DEVICE_TOKEN": "test-device",
                    "APPLE_ASR_BINARY": "/isolated/probe",
                    "SpeechServices": Speech,
                    "AudioSession": Session,
                    "OpusCodec": object,
                }.items():
                    stack.enter_context(patch.object(server, name, value))
                stack.enter_context(patch.object(server, "apple_transcribe_pcm",
                    AsyncMock(return_value="小猫有几条腿呀？")))
                stack.enter_context(patch.object(server.REGISTRATIONS, "consume_enrollment",
                    AsyncMock(return_value=None)))
                stack.enter_context(patch.object(server.REGISTRATIONS, "identity_settings",
                    return_value={"enabled": False}))
                stack.enter_context(patch.object(server.STORE, "conversation_mode", return_value="chat"))
                stack.enter_context(patch.object(server.LEARNING, "end_session", return_value=None))
                camera = stack.enter_context(patch.object(server.DIALOGUE_CAMERA, "prepare",
                    AsyncMock(return_value=None)))
                model = stack.enter_context(patch.object(server, "chat",
                    AsyncMock(return_value={"reply": "小猫通常有四条腿。"})))
                for recorded_level in (128, 900):
                    for primary_text in ("小猫有几条嘴呀", "小猫有鸡条腿", "小猫有几张嘴呀？"):
                        with self.subTest(level=recorded_level, primary=primary_text):
                            model.reset_mock()
                            camera.reset_mock()
                            await server.device_audio(Socket())
                            if "几张嘴" not in primary_text:
                                model.assert_awaited_once()
                                self.assertEqual(model.await_args.args[0].text, "小猫有几条腿呀？")
                            else:
                                model.assert_not_awaited()
                                camera.assert_not_awaited()
                                self.assertIn("几张嘴", answers[-1]["reply"])
                                self.assertIn("几条腿", answers[-1]["reply"])

    async def test_quiet_known_question_can_finish_without_a_stalled_primary(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PUPPY_DATA_DIR": directory}):
            try:
                from companion import server
            except ModuleNotFoundError as error:
                if error.name == "fastapi":
                    self.skipTest("Gateway dependencies are not installed locally")
                raise

        started, cancelled = asyncio.Event(), asyncio.Event()
        socket = AsyncMock()
        socket.headers = {"authorization": "Bearer test-device", "device-id": "synthetic"}
        heard = []
        recorded_level = 128
        transcript = ""

        async def stalled_primary(_pcm):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def secondary(_pcm, _path, *, timeout):
            await asyncio.wait_for(started.wait(), 1)
            return transcript

        class Session:
            def __init__(self, _socket, **callbacks):
                self.callbacks = callbacks

            async def run(self):
                pcm = recorded_level.to_bytes(2, "little", signed=True) * 8000
                heard.append(await self.callbacks["transcribe"](pcm))

        speech = AsyncMock()
        speech.transcribe_async = stalled_primary
        with ExitStack() as stack:
            for name, value in {
                "DEVICE_TOKEN": "test-device",
                "APPLE_ASR_BINARY": "/isolated/probe",
                "QUIET_ASR_CROSSCHECK_SECONDS": 0.01,
                "SpeechServices": lambda _model: speech,
                "AudioSession": Session,
                "OpusCodec": object,
                "apple_transcribe_pcm": secondary,
            }.items():
                stack.enter_context(patch.object(server, name, value))
            samples = [(128, "苹果用英语怎么说呀？"), (600, "苹果用英语怎么说呀？"),
                       (600, "小猫有几条腿呀？")]
            for recorded_level, transcript in samples:
                with self.subTest(recorded_level=recorded_level, transcript=transcript):
                    started.clear()
                    cancelled.clear()
                    await asyncio.wait_for(server.device_audio(socket), 2)
                    self.assertTrue(cancelled.is_set())
        self.assertEqual(heard, [question for _, question in samples])

    async def test_malformed_math_uses_agreeing_secondary_before_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PUPPY_DATA_DIR": directory}):
                try:
                    from companion import server
                except ModuleNotFoundError as error:
                    if error.name == "fastapi":
                        self.skipTest("Gateway dependencies are not installed locally")
                    raise

            class Socket:
                headers = {"authorization": "Bearer test-device", "device-id": "synthetic"}

                async def accept(self):
                    return None

            class Speech:
                timed_out = False

                def __init__(self, _model):
                    pass

                async def transcribe_async(self, _pcm):
                    if self.timed_out:
                        raise TimeoutError
                    return "2加2等于紧"

                async def synthesize_async(self, _text):
                    raise AssertionError("Fake session must not play audio")

            class Session:
                def __init__(self, _socket, **kwargs):
                    self.callbacks = kwargs

                async def run(self):
                    recorded = b"\0\0" * 8000
                    await self.callbacks["prepare_reply"](recorded)
                    transcript = await self.callbacks["transcribe"](recorded)
                    await self.callbacks["chat"](recorded, transcript)

            with patch.object(server, "DEVICE_TOKEN", "test-device"), \
                 patch.object(server, "APPLE_ASR_BINARY", "/isolated/probe"), \
                 patch.object(server, "SpeechServices", Speech), \
                 patch.object(server, "OpusCodec", object), \
                 patch.object(server, "AudioSession", Session), \
                 patch.object(server, "apple_transcribe_pcm", new_callable=AsyncMock,
                              return_value="2+2等于几？"), \
                 patch.object(server.REGISTRATIONS, "consume_enrollment",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server.REGISTRATIONS, "identity_settings",
                              return_value={"enabled": False}), \
                 patch.object(server.STORE, "conversation_mode", return_value="chat"), \
                 patch.object(server.LEARNING, "end_session", return_value=None), \
                 patch.object(server.DIALOGUE_CAMERA, "prepare",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server, "chat", new_callable=AsyncMock,
                              return_value={"reply": "2加2等于4"}) as model:
                await server.device_audio(Socket())
                self.assertEqual(model.await_args.args[0].text, "2+2等于几？")
                Speech.timed_out = True
                model.reset_mock()
                await server.device_audio(Socket())
                model.assert_awaited_once()
                self.assertEqual(model.await_args.args[0].text, "2+2等于几？")

    async def test_numeric_only_audio_asks_again_without_calling_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PUPPY_DATA_DIR": directory}):
                try:
                    from companion import server
                except ModuleNotFoundError as error:
                    if error.name == "fastapi":
                        self.skipTest("Gateway dependencies are not installed locally")
                    raise

            class Socket:
                headers = {"authorization": "Bearer test-device", "device-id": "synthetic"}

                async def accept(self):
                    return None

            class Speech:
                def __init__(self, _model):
                    pass

                async def transcribe_async(self, _pcm):
                    return "3200"

                async def synthesize_async(self, _text):
                    raise AssertionError("Fake session must not play audio")

            class Session:
                answer = None

                def __init__(self, _socket, **kwargs):
                    self.callbacks = kwargs

                async def run(self):
                    recorded = b"\0\0" * 8000
                    await self.callbacks["prepare_reply"](recorded)
                    transcript = await self.callbacks["transcribe"](recorded)
                    self.__class__.answer = await self.callbacks["chat"](recorded, transcript)

            with patch.object(server, "DEVICE_TOKEN", "test-device"), \
                 patch.object(server, "APPLE_ASR_BINARY", "/isolated/probe"), \
                 patch.object(server, "SpeechServices", Speech), \
                 patch.object(server, "OpusCodec", object), \
                 patch.object(server, "AudioSession", Session), \
                 patch.object(server, "apple_transcribe_pcm", new_callable=AsyncMock,
                              return_value="3200"), \
                 patch.object(server.REGISTRATIONS, "consume_enrollment",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server.REGISTRATIONS, "identity_settings",
                              return_value={"enabled": False}), \
                 patch.object(server.DIALOGUE_CAMERA, "prepare",
                              new_callable=AsyncMock) as camera, \
                 patch.object(server, "chat", new_callable=AsyncMock) as model:
                await server.device_audio(Socket())
                self.assertIn("请再说一次", Session.answer["reply"])
                camera.assert_not_awaited()
                model.assert_not_awaited()

    async def test_voice_model_failure_gets_spoken_retry_instead_of_stalling(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PUPPY_DATA_DIR": directory}):
                try:
                    from companion import server
                except ModuleNotFoundError as error:
                    if error.name == "fastapi":
                        self.skipTest("Gateway dependencies are not installed locally")
                    raise

            class Socket:
                headers = {"authorization": "Bearer test-device", "device-id": "synthetic"}

                async def accept(self):
                    return None

            class Speech:
                def __init__(self, _model):
                    pass

                async def transcribe_async(self, _pcm):
                    return "继续讲故事"

                async def synthesize_async(self, _text):
                    raise AssertionError("Fake session must not play audio")

            class Session:
                answer = None

                def __init__(self, _socket, **kwargs):
                    self.callbacks = kwargs

                async def run(self):
                    recorded = b"\0\0" * 8000
                    await self.callbacks["prepare_reply"](recorded)
                    transcript = await self.callbacks["transcribe"](recorded)
                    self.__class__.answer = await self.callbacks["chat"](recorded, transcript)

            with patch.object(server, "DEVICE_TOKEN", "test-device"), \
                 patch.object(server, "APPLE_ASR_BINARY", ""), \
                 patch.object(server, "SpeechServices", Speech), \
                 patch.object(server, "OpusCodec", object), \
                 patch.object(server, "AudioSession", Session), \
                 patch.object(server.REGISTRATIONS, "consume_enrollment",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server.REGISTRATIONS, "identity_settings",
                              return_value={"enabled": False}), \
                 patch.object(server.DIALOGUE_CAMERA, "prepare",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server.STORE, "conversation_mode", return_value="chat"), \
                 patch.object(server.LEARNING, "end_session", return_value=None), \
                 patch.object(server, "chat", new_callable=AsyncMock,
                              side_effect=server.HTTPException(503, server.MODEL_RETRY_DETAIL)) as model:
                await server.device_audio(Socket())
                model.assert_awaited_once()
                self.assertIn("请再问我一次", Session.answer["reply"])

    async def test_disagreement_asks_child_without_running_model_or_camera(self):
        # Import-time gateway storage must be outside the family's database.
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PUPPY_DATA_DIR": directory}):
                try:
                    from companion import server
                except ModuleNotFoundError as error:
                    if error.name == "fastapi":
                        self.skipTest("Gateway dependencies are not installed locally")
                    raise

            class Socket:
                headers = {"authorization": "Bearer test-device", "device-id": "synthetic"}

                async def accept(self):
                    return None

            class Speech:
                def __init__(self, _model):
                    pass

                async def transcribe_async(self, _pcm):
                    return "你好，相交的英文怎么说？"

                async def synthesize_async(self, _text):
                    raise AssertionError("Fake session must not play audio")

            class Session:
                answer = None

                def __init__(self, _socket, **kwargs):
                    self.callbacks = kwargs

                async def run(self):
                    recorded = b"\0\0" * 8000
                    await self.callbacks["prepare_reply"](recorded)
                    transcript = await self.callbacks["transcribe"](recorded)
                    self.__class__.answer = await self.callbacks["chat"](recorded, transcript)

            with patch.object(server, "DEVICE_TOKEN", "test-device"), \
                 patch.object(server, "APPLE_ASR_BINARY", "/isolated/probe"), \
                 patch.object(server, "SpeechServices", Speech), \
                 patch.object(server, "OpusCodec", object), \
                 patch.object(server, "AudioSession", Session), \
                 patch.object(server, "apple_transcribe_pcm", new_callable=AsyncMock,
                              return_value="你好，香蕉的英文怎么说？"), \
                 patch.object(server.REGISTRATIONS, "consume_enrollment",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server.REGISTRATIONS, "identity_settings",
                              return_value={"enabled": False}), \
                 patch.object(server.DIALOGUE_CAMERA, "prepare",
                              new_callable=AsyncMock) as camera, \
                 patch.object(server, "chat", new_callable=AsyncMock) as model:
                await server.device_audio(Socket())
                self.assertEqual(Session.answer, {"reply": CLARIFY_HOMOPHONE})
                camera.assert_not_awaited()
                model.assert_not_awaited()

            Session.answer = None
            with patch.object(server, "DEVICE_TOKEN", "test-device"), \
                 patch.object(server, "APPLE_ASR_BINARY", "/isolated/probe"), \
                 patch.object(server, "SpeechServices", Speech), \
                 patch.object(server, "OpusCodec", object), \
                 patch.object(server, "AudioSession", Session), \
                 patch.object(server, "apple_transcribe_pcm", new_callable=AsyncMock,
                              side_effect=RuntimeError("secondary unavailable")), \
                 patch.object(server.REGISTRATIONS, "consume_enrollment",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server.REGISTRATIONS, "identity_settings",
                              return_value={"enabled": False}), \
                 patch.object(server.STORE, "conversation_mode", return_value="chat"), \
                 patch.object(server.LEARNING, "end_session", return_value=None), \
                 patch.object(server.DIALOGUE_CAMERA, "prepare",
                              new_callable=AsyncMock, return_value=None), \
                 patch.object(server, "chat", new_callable=AsyncMock,
                              return_value={"reply": "大模型回答"}) as model:
                await server.device_audio(Socket())
                self.assertEqual(Session.answer, {"reply": "大模型回答"})
                model.assert_awaited_once()
                self.assertEqual(model.await_args.args[0].session_id,
                                 device_audio_conversation_id("synthetic"))


if __name__ == "__main__":
    unittest.main()
