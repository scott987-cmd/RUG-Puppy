import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock, patch
from .test_child_activity_gateway import ChildActivityGatewayTests, server


class MotionGatewayTests(ChildActivityGatewayTests):
    async def setup_audio(self, primary="Come here.", secondary="请坐下。", *, run_turn=True,
                          followup_event=None, speak=True, expire_followup=False, execute_error=None,
                          identity_enabled=False):
        captured, answers = {}, []
        primary = primary if isinstance(primary,list) else [primary]
        secondary = secondary if isinstance(secondary,list) else [secondary]
        index = 0
        class Socket:
            headers = {"authorization": "Bearer test", "device-id": "puppy"}
            accept = AsyncMock()
        class Speech:
            def __init__(self, _):pass
            async def transcribe_async(self, pcm, **kwargs):
                captured.setdefault("asr_languages",[]).append(kwargs.get("language"))
                return primary[index]
            async def synthesize_async(self,text):
                captured.setdefault("spoken",[]).append(text)
                return b"\0\0"*16000
            async def synthesize_vocabulary_async(self,prefix,word):
                captured.setdefault("spoken",[]).append("vocabulary:"+word)
                return b"\0\0"*16000
        class Session:
            def __init__(self, socket, **kwargs):captured.update(kwargs)
            async def run(self):
                nonlocal index
                if run_turn:
                    for index in range(len(primary)):
                        if index and followup_event:captured["on_event"](followup_event,{})
                        captured.setdefault("short_states",[]).append(captured["short_answer_expected"]())
                        pcm=b"\x90\0"*16000
                        await captured["prepare_reply"](pcm)
                        text=await captured["transcribe"](pcm)
                        clock = (patch.object(server.time,"monotonic",return_value=server.time.monotonic()+26)
                                 if index and expire_followup else nullcontext())
                        with clock:answers.append(await captured["chat"](pcm,text))
                        captured.setdefault("action_counts",[]).append(execute.await_count)
                        if speak:captured["on_event"]("tts_sent",{})
        model=Mock(side_effect=AssertionError("Exact body motion should not need a language model"))
        execute=AsyncMock(return_value={"ok":True,"phase":"completed"},side_effect=execute_error)
        with patch.object(server,"DEVICE_TOKEN","test"), patch.object(server,"MODEL",model), \
             patch.object(server,"SpeechServices",Speech), patch.object(server,"OpusCodec",object), \
             patch.object(server,"AudioSession",Session), patch.object(server,"AUDIO_SESSIONS",{}), \
             patch.object(server,"APPLE_ASR_BINARY","/local/probe"), \
             patch.object(server,"apple_transcribe_pcm",AsyncMock(side_effect=secondary)), \
             patch.object(server,"execute_safe_action",execute), \
             patch.object(server.REGISTRATIONS,"consume_enrollment",AsyncMock(return_value=None)), \
             patch.object(server.REGISTRATIONS, "identity_settings",
                          return_value={"enabled": identity_enabled, "revision": 1}), \
             patch.object(server.HERMES_TASKS,"respond",AsyncMock(side_effect=lambda _d,t: {"reply":"ordinary fallback"} if t=="是的" else None)), \
             patch.object(server.DIALOGUE_CAMERA,"prepare",AsyncMock(return_value=None)):
            await server.device_audio(Socket())
        return captured,answers,model,execute

    async def test_ambiguous_motion_needs_the_childs_next_confirmation_before_moving(self):
        callbacks,answers,model,execute=await self.setup_audio(["新乃妖","是的"],["鲜奶腰","是的"])
        self.assertEqual(answers[0]["reply_source"],"voice_motion_confirmation")
        self.assertEqual(callbacks["action_counts"],[0,1])
        self.assertEqual(callbacks["short_states"],[False,True])
        self.assertEqual(callbacks["asr_languages"],[None,"zh"])
        execute.assert_awaited_once_with("puppy","stretch",confirmed_by_voice=True)
        model.assert_not_called()

    async def test_stationary_play_blocks_exact_motion_but_keeps_stop(self):
        with patch.object(server, "STATIONARY_PLAY", True):
            _, answers, _, execute = await self.setup_audio(
                ["挥手", "坐下", "伸懒腰", "抱抱", "停"],
                ["挥手", "坐下", "伸懒腰", "抱抱", "停"])
        self.assertEqual([row["reply_source"] for row in answers],
                         ["stationary_play"] * 4 + ["voice_motion"])
        execute.assert_awaited_once_with("puppy", "stop")

    async def test_stationary_play_does_not_arm_voice_motion_confirmation(self):
        with (patch.object(server, "STATIONARY_PLAY", True),
              patch.object(server, "chat", AsyncMock(return_value={"reply": "聊聊吧。"}))):
            callbacks, _, _, execute = await self.setup_audio(["新乃妖", "是的"], ["鲜奶腰", "是的"])
        self.assertEqual(callbacks["short_states"], [False, False])
        execute.assert_not_awaited()

    async def test_regular_device_audio_skips_automatic_identification(self):
        with (patch.object(server, "capture_identity_photo", AsyncMock()) as photo,
              patch.object(server.IDENTITY, "recognize") as recognize,
              patch.object(server.REGISTRATIONS, "list_members", return_value=[{"ready": True}]),
              patch.object(server, "chat", AsyncMock(return_value={"reply": "你好。"})) as chat):
            await self.setup_audio("你好", "你好", identity_enabled=True)
        photo.assert_not_awaited()
        recognize.assert_not_called()
        self.assertIsNone(chat.call_args.args[0].person_id)

    async def test_child_can_ask_whether_to_share_their_address_without_personal_access(self):
        text = "陌生人問我家住哪裡？我該告訴他嗎？"
        _, answers, model, execute = await self.setup_audio(text, text)
        self.assertEqual(answers[0]["reply_source"], "child_safety")
        self.assertIn("先别告诉", answers[0]["reply"])
        self.assertIn("大人帮忙", answers[0]["reply"])
        model.assert_not_called()
        execute.assert_not_awaited()

    async def test_stationary_play_rejects_gateway_job_before_creating_motion(self):
        with patch.object(server, "STATIONARY_PLAY", True):
            with self.assertRaises(server.HTTPException) as error:
                server.start_action_job("puppy", "组合", [{"action": "wave"}])
        self.assertEqual(error.exception.status_code, 409)

    async def test_stationary_play_blocks_motion_proposed_by_chat(self):
        request = server.ChatRequest(device_id="puppy", text="先挥手，再坐下。")
        request._from_device = True
        for intent in ("action", "action_sequence", "action_draft"):
            with (self.subTest(intent=intent),
                  patch.object(server, "STATIONARY_PLAY", True),
                  patch.object(server.PetAgent, "reply", return_value={"intent": intent, "reply": "好，开始了。"}),
                  patch.object(server, "execute_chat_action", AsyncMock()) as execute,
                  patch.object(server, "start_action_job") as start,
                  patch.object(server.ACTION_DRAFTS, "propose", AsyncMock()) as propose):
                result = await server.chat(request)
                self.assertEqual(result["reply_source"], "stationary_play")
                execute.assert_not_awaited()
                start.assert_not_called()
                propose.assert_not_awaited()

    async def test_real_soft_confirmation_disagreement_is_recovered_but_denial_conflicts_ask_again(self):
        _,_,model,execute=await self.setup_audio(["新乃妖","是吧"],["鲜奶腰","去吧"])
        execute.assert_awaited_once_with("puppy","stretch",confirmed_by_voice=True)
        model.assert_not_called()
        for primary,secondary in [("是的","不是"),("不是","是的"),("是的","先别做"),
                                  ("是吧","是吧"),("谢谢","是的"),("是的",""),("","是的")]:
            with self.subTest(primary=primary,secondary=secondary):
                _,answers,model,execute=await self.setup_audio(["新乃妖",primary],["鲜奶腰",secondary])
                self.assertEqual(answers[-1]["reply_source"],"voice_motion_confirmation")
                execute.assert_not_awaited()
                model.assert_not_called()

    async def test_unknown_confirmation_reasks_in_chinese_without_moving_or_chatting(self):
        _,answers,model,execute=await self.setup_audio(["新乃妖","Безусловно."],["鲜奶腰","这个"])
        self.assertEqual(answers[-1]["reply_source"],"voice_motion_confirmation")
        self.assertIn("可以", answers[-1]["reply"])
        execute.assert_not_awaited()
        model.assert_not_called()

    async def test_real_hug_misrecognition_asks_first_and_requires_explicit_consent(self):
        for reply in ["是的", "不是"]:
            callbacks,answers,model,execute=await self.setup_audio(["又次滑掉了",reply],["报道。",reply])
            self.assertEqual(callbacks["action_counts"][0],0)
            self.assertIn("抱抱吗", answers[0]["reply"])
            if reply=="是的":execute.assert_awaited_once_with("puppy","hug",confirmed_by_voice=True)
            else:execute.assert_not_awaited()
            model.assert_not_called()

    async def test_real_clear_secondary_denial_cancels_instead_of_entering_chat(self):
        _,answers,model,execute=await self.setup_audio(["信我们","哦嘻"],["伸斩腰。","不是。"])
        self.assertEqual(answers[-1]["reply"],"好，不做这个动作。")
        execute.assert_not_awaited()
        model.assert_not_called()

    async def test_missing_action_completion_reports_a_motion_failure(self):
        _,answers,model,_=await self.setup_audio("坐下","坐下",execute_error=TimeoutError)
        self.assertEqual(answers[0]["reply_source"],"voice_motion_timeout")
        self.assertIn("没有收到动作完成确认",answers[0]["reply"])
        model.assert_not_called()

    async def test_declining_or_interrupting_a_confirmation_never_moves(self):
        for reply,event in [("不是",None),("是的","abort_received"),("是的","parent_audio_stopped"),
                            ("是的","motion_voice_stop")]:
            with self.subTest(reply=reply,event=event):
                _,answers,model,execute=await self.setup_audio(["新乃妖",reply],["鲜奶腰",reply],followup_event=event)
                execute.assert_not_awaited()
                model.assert_not_called()

    async def test_an_unspoken_suggestion_cannot_be_confirmed(self):
        callbacks,_,model,execute=await self.setup_audio(["新乃妖","是的"],["鲜奶腰","是的"],speak=False)
        self.assertEqual(callbacks["short_states"],[False,False])
        execute.assert_not_awaited()
        model.assert_not_called()

    async def test_expired_or_changed_topic_suggestions_cannot_be_confirmed(self):
        for primary,expire in [(["新乃妖","是的"],True),
                               (["新乃妖","小猫有几条腿？","是的"],False)]:
            secondary=["鲜奶腰"]+primary[1:]
            with self.subTest(expire=expire):
                _,_,model,execute=await self.setup_audio(primary,secondary,expire_followup=expire)
                execute.assert_not_awaited()
                model.assert_not_called()

    async def test_real_secondary_short_motion_bypasses_camera_identity_and_model(self):
        for primary,secondary,action in [("Come here.","请坐下。","sit"),
                                         ("请报报我。","请抱抱我。","hug"),
                                         ("挥手","挥手","wave")]:
            callbacks,answers,model,execute=await self.setup_audio(primary,secondary)
            self.assertEqual(answers[0]["intent"],"action")
            self.assertEqual(answers[0]["action"],action)
            self.assertEqual(answers[0]["device_receipt"]["phase"],"completed")
            execute.assert_awaited_once_with("puppy",action)
            model.assert_not_called()

    async def test_stop_confirmation_never_lies_about_failed_device_receipt(self):
        callbacks,_,_,_=await self.setup_audio(run_turn=False)
        owner=asyncio.create_task(asyncio.Event().wait())
        try:
            with patch.object(server,"RUNNING",{"puppy":owner}), \
                 patch.object(server,"APPLE_ASR_BINARY","/local/probe"), \
                 patch.object(server,"apple_transcribe_pcm",AsyncMock(return_value="停止动作")), \
                 patch.object(server,"stop_device",AsyncMock(return_value={"ok":False,"phase":"failed"})):
                result=await callbacks["stop_motion"](b"\x90\0"*16000,owner)
            self.assertFalse(result["confirmed"])
            self.assertIn("没有收到停止确认",result["reply"])
        finally:
            owner.cancel();await asyncio.gather(owner,return_exceptions=True)

    async def test_recognized_listen_request_pauses_only_the_owned_movement(self):
        callbacks,_,_,_=await self.setup_audio(run_turn=False)
        owner=asyncio.create_task(asyncio.Event().wait())
        stop=AsyncMock(return_value={"ok":True,"phase":"completed"})
        try:
            with patch.object(server,"RUNNING",{"puppy":owner}),patch.object(server,"stop_device",stop), \
                 patch.object(server,"APPLE_ASR_BINARY","/local/probe"):
                for text in ["听一下","请听一下","停止动","体动作。","身体动作","动作"]:
                    stop.reset_mock()
                    with patch.object(server,"apple_transcribe_pcm",AsyncMock(return_value=text)):
                        result=await callbacks["stop_motion"](b"\x90\0"*16000,owner)
                    self.assertTrue(result["confirmed"])
                    stop.assert_awaited_once()
        finally:
            owner.cancel();await asyncio.gather(owner,return_exceptions=True)

    async def test_late_stop_asr_cannot_stop_a_replacement_action(self):
        callbacks,_,_,_=await self.setup_audio(run_turn=False)
        first=asyncio.create_task(asyncio.Event().wait())
        second=asyncio.create_task(asyncio.Event().wait())
        running={"puppy":first}
        async def recognized(*args,**kwargs):
            running["puppy"]=second
            return "停止动作"
        stop=AsyncMock()
        try:
            with patch.object(server,"RUNNING",running), \
                 patch.object(server,"APPLE_ASR_BINARY","/local/probe"), \
                 patch.object(server,"apple_transcribe_pcm",recognized),patch.object(server,"stop_device",stop):
                self.assertIsNone(await callbacks["stop_motion"](b"\x90\0"*16000,first))
            stop.assert_not_awaited()
        finally:
            first.cancel();second.cancel()
            await asyncio.gather(first,second,return_exceptions=True)

    async def test_split_negation_cannot_dispatch_its_tail_or_disable_hold_still(self):
        callbacks, _, _, _ = await self.setup_audio(run_turn=False)
        owner = asyncio.create_task(asyncio.Event().wait())
        stop = AsyncMock(return_value={"ok": True, "phase": "completed"})
        asr = AsyncMock(side_effect=["不要。", "停", "别动。", "停"])
        try:
            with patch.object(server, "RUNNING", {"puppy": owner}), patch.object(server, "stop_device", stop), \
                 patch.object(server, "APPLE_ASR_BINARY", "/local/probe"), patch.object(server, "apple_transcribe_pcm", asr):
                check = callbacks["stop_motion"]
                pcm = b"\x90\0" * 16000
                self.assertEqual(await check(pcm, owner), {"negation_prefix": True})
                self.assertIsNone(await check(pcm, owner, resolving_negation=True))
                stop.assert_not_awaited()
                self.assertTrue((await check(pcm, owner, resolving_negation=True))["confirmed"])
                self.assertTrue((await check(pcm, owner))["confirmed"])
                self.assertEqual(asr.await_count, 4)
        finally:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    async def test_negative_stop_and_story_text_never_dispatch_stop(self):
        callbacks,_,_,_=await self.setup_audio(run_turn=False)
        owner=asyncio.create_task(asyncio.Event().wait())
        stop=AsyncMock()
        try:
            with patch.object(server,"RUNNING",{"puppy":owner}),patch.object(server,"stop_device",stop), \
                 patch.object(server,"APPLE_ASR_BINARY","/local/probe"):
                for text in ["不要停","别停止动作","不要听一下","小兔子说停止动作","停止动作是什么意思"]:
                    with patch.object(server,"apple_transcribe_pcm",AsyncMock(return_value=text)):
                        result=await callbacks["stop_motion"](b"\x90\0"*16000,owner)
                    self.assertTrue(result is None or result.get("ignore"))
            stop.assert_not_awaited()
        finally:
            owner.cancel();await asyncio.gather(owner,return_exceptions=True)

    async def test_secondary_negative_cannot_turn_into_an_exact_action(self):
        callbacks,answers,model,execute=await self.setup_audio("坐下", "不要坐下")
        self.assertIn("请再说一次",answers[0]["reply"])
        execute.assert_not_awaited()
        model.assert_not_called()

    async def test_stop_ack_cannot_replay_a_pending_english_vocabulary_reply(self):
        callbacks,answers,model,execute=await self.setup_audio("苹果用英语怎么说呀？", "苹果用英语怎么说呀？")
        self.assertIn("apple",answers[0]["reply"])
        callbacks["on_event"]("motion_voice_stop",{"confirmed":True})
        await callbacks["synthesize"]("好，停下了。")
        self.assertEqual(callbacks["spoken"],["好，停下了。"])

