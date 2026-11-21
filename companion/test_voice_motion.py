"""Real short-command failures and non-command boundaries, without hardware."""
import unittest
from .voice_motion import (motion_command, reconcile_motion, unclear_stretch, unclear_motion,
                           motion_confirmation, motion_stop_command, reconcile_confirmation)


class MotionWordsTests(unittest.TestCase):
    def test_short_commands_and_homophone_recovery(self):
        for text, action in [("坐下", "sit"), ("请坐下。", "sit"), ("請你坐下吧", "sit"),
                             ("挥手", "wave"), ("伸个懒腰", "stretch"), ("抱抱我", "hug"),
                             ("请做一个抱抱的动作", "hug"), ("停止动作", "stop"),
                             ("停", "stop"), ("别动", "stop"), ("不要动", "stop")]:
            with self.subTest(text=text):self.assertEqual(motion_command(text), action)
        self.assertEqual(reconcile_motion("Come here.", "请坐下。"), ("请坐下。", None))
        self.assertEqual(reconcile_motion("请报报我。", "请抱抱我。"), ("请抱抱我。", None))

    def test_real_two_second_hug_can_recover_a_long_english_hallucination(self):
        text = "I can do the chit-chit! I can be a shi-i-i-c!"
        self.assertEqual(reconcile_motion(text, "抱抱。", short_audio=True), ("抱抱。", None))
        self.assertEqual(reconcile_motion(text, "抱抱。"), (text, None))
        for text in ["no hug", "hug then wave", "抱抱再坐下", "挥手然后坐下", "without a hug"]:
            self.assertEqual(reconcile_motion(text, "抱抱。", short_audio=True), (text, None))

    def test_negations_stories_questions_and_extra_operations_are_not_commands(self):
        for text in ("不要坐下", "不要停", "别停止", "坐下是什么意思", "停止动作是什么意思",
                     "讲个小兔子挥手的故事", "请坐下，记住我喜欢猫", "如果我说停你会怎么样",
                     "坐下吗", "Don't stop", "请不要抱抱我", "挥手再坐下", "记住坐下"):
            with self.subTest(text=text):self.assertIsNone(motion_command(text))
        self.assertEqual(reconcile_motion("不要坐下", "坐下"), ("不要坐下", None))
        self.assertEqual(reconcile_motion("记住我喜欢抱抱", "抱抱"), ("记住我喜欢抱抱", None))
        self.assertIsNotNone(reconcile_motion("坐下", "抱抱")[1])
        self.assertIsNotNone(reconcile_motion("坐下", "不要坐下")[1])
        self.assertIsNotNone(reconcile_motion("坐下", "不坐下")[1])
        self.assertEqual(reconcile_motion("Don’t sit down", "坐下"), ("Don’t sit down", None))
        for text in ["怎么坐下","挥手的英语","改进挥手","修改伸懒腰代码","improve wave","how to sit down"]:
            with self.subTest(text=text):self.assertEqual(reconcile_motion(text,"坐下"),(text,None))

    def test_unclear_waist_words_only_offer_a_confirmation(self):
        for first,second in [("新乃妖","鲜奶腰"),("新增染腰","深格懒腰"),("","身子懒腰"),
                             ("新买药","鲜奶腰")]:
            self.assertTrue(unclear_stretch(first,second))
            self.assertIsNone(motion_command(first))
            self.assertIsNone(motion_command(second))
        for first,second in [("伸懒腰","伸懒腰"),("我的腰","我腰"),("腰痛","鲜奶腰"),
                             ("买药怎么吃","鲜奶腰"),
                             ("我叫新乃妖","鲜奶腰"),("不要伸懒腰","深格懒腰"),
                             ("小兔子弯腰的故事","鲜奶腰"),("请坐下","深格懒腰"),
                             ("鲜奶腰","鲜奶腰"),("猫为什么弯腰","鲜奶腰")]:
            with self.subTest(first=first,second=second):self.assertFalse(unclear_stretch(first,second))
        self.assertIs(motion_confirmation("是的。"),True)
        self.assertIs(motion_confirmation("不要"),False)
        self.assertIsNone(motion_confirmation("好，先讲故事"))
        self.assertEqual(motion_command("停止动"),"stop")
        self.assertIsNone(motion_command("不要停止动"))

    def test_ambiguous_hug_words_can_only_offer_a_question_after_short_disagreement(self):
        self.assertEqual(unclear_motion("又次滑掉了", "报道。", short_audio=True), "hug")
        self.assertEqual(unclear_motion("", "棒棒。", short_audio=True), "hug")
        self.assertIsNone(motion_command("报道。"))
        for primary,secondary,short in [("报道","报道",True),("不要抱抱","报道",True),
                                         ("报道是什么","报道",True),("坐下","报道",True),
                                         ("讲个宝宝的故事","宝宝",True),("我叫宝宝","宝宝",True),
                                         ("又次滑掉了","报道",False)]:
            self.assertIsNone(unclear_motion(primary,secondary,short_audio=short))

    def test_motion_attention_request_pauses_without_broad_substring_matching(self):
        for text in ["停一下","停止动","听一下","请你听一下吧","体动作。","身体动作","动作"]:
            self.assertTrue(motion_stop_command(text))
        self.assertIsNone(motion_command("听一下"))  # An idle request is not an action.
        for text in ["不要停一下","不要听一下","听一下我的故事","小兔子说听一下", "停一下是什么意思",
                     "不要停止动作","身体动作是什么意思","讲个动作的故事"]:
            self.assertFalse(motion_stop_command(text))

    def test_confirmation_recovery_cannot_replace_a_topic_or_override_a_denial(self):
        self.assertEqual(reconcile_confirmation("是吧","去吧"),("去吧",False))
        self.assertEqual(reconcile_confirmation("是的","是吧"),("是的",False))
        self.assertEqual(reconcile_confirmation("哦嘻","不是。"),("不是。",False))
        self.assertEqual(reconcile_confirmation("姐姐","是的"),("姐姐",True))
        self.assertEqual(reconcile_confirmation("Безусловно.","这个"),("Безусловно.",True))
        self.assertEqual(reconcile_confirmation("小猫有几条腿？","小猫有几条腿？"),("小猫有几条腿？",False))
        self.assertEqual(reconcile_confirmation("不是","是的"),("不是",True))
        self.assertEqual(reconcile_confirmation("是的","先别做"),("是的",True))
        self.assertEqual(reconcile_confirmation("是的","不要做动作"),("是的",True))

