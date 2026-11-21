"""Exact, bounded body-voice movement commands; never infer a new action."""
import re

_FORMS = str.maketrans("請揮懶個動機器來別", "请挥懒个动机器来别")
_PREFIX = r"(?:(?:小狗|机器人)[，,\s]*)?(?:请你|请|你|给我)?"
_SUFFIX = r"(?:吧|呀|啦)?"
_COMMANDS = {
    "wave": r"(?:挥挥手|挥手)(?:一下)?",
    "sit": r"(?:坐下(?:来)?|坐好|坐一下)",
    "stretch": r"伸(?:个|一个)?懒腰",
    "hug": r"(?:抱抱(?:我|一下)?|做(?:个|一个)抱抱(?:的)?动作)",
    "stop": r"(?:停(?:止(?:动(?:作)?)?|下(?:来)?|一下|住)?|别动|不要动)",
}
_ENGLISH = {"wave": "wave", "sit down": "sit", "stretch": "stretch", "hug": "hug", "stop": "stop"}
_NOT_COMMAND = re.compile(
    r"不|没|沒|无|無|勿|別|别|故事|如果|假如|是否|什么|什麼|为什么|嗎|吗|"
    r"怎么|怎麼|怎样|怎樣|如何|多少|几|幾|谁|誰|英文|英语|英語|"
    r"改进|改進|修改|代码|代碼|补丁|補丁|上线|上線|升级|升級|更新|"
    r"记住|記住|保存|删除|刪除|名字|昨天|我家|喜欢|喜歡|教你|"
    r"\b(?:no|not|without|don['’]t|never|why|what|how|story|remember|rename|fix|improve|code)\b", re.I)


def motion_command(text: str) -> str | None:
    text = text.translate(_FORMS).strip().rstrip("。.!！?？").strip()
    if text.lower() in _ENGLISH:
        return _ENGLISH[text.lower()]
    for action, phrase in _COMMANDS.items():
        if re.fullmatch(_PREFIX + phrase + _SUFFIX, text):
            return action
    return None


def reconcile_motion(primary: str, secondary: str, *, short_audio: bool = False) -> tuple[str, str | None]:
    """Use a real second recognition, preserving negations and conflicts."""
    first, second = motion_command(primary), motion_command(secondary)
    if first and second and first != second:
        return primary, "两次识别的动作不一样，请再说一次要做的动作好吗？"
    if first and not second and _NOT_COMMAND.search(secondary):
        return primary, "你是想做这个动作，还是先不做？请再说一次好吗？"
    # Whisper can hallucinate a long English sentence from a two-second Chinese
    # command. Only relax its text length for short audio and a bounded ASCII
    # result; a real second recognition must still be an exact command.
    bounded = len(primary) <= 32 or (short_audio and primary.isascii() and len(primary) <= 96)
    compound = re.search(r"然后|再|接着|之后|\b(?:and|then)\b", primary, re.I)
    actions = {a for a, phrase in _COMMANDS.items() if re.search(phrase, primary)}
    if second and not first and bounded and not compound and len(actions) < 2 and not _NOT_COMMAND.search(primary):
        return secondary, None
    return primary, None


def stop_negation_prefix(text: str) -> bool:
    value = text.translate(_FORMS).strip().rstrip("。.!！?？").strip()
    return value in {"不要", "别", "不许", "不能"}


def explicit_hold_still(text: str) -> bool:
    value = text.translate(_FORMS).strip().rstrip("。.!！?？").strip()
    return bool(re.fullmatch(_PREFIX + r"(?:不要动|别动)" + _SUFFIX, value))


def declined_stop(text: str) -> bool:
    text = text.translate(_FORMS).replace("’", "'").strip().rstrip("。.!！?？").strip()
    return bool(re.fullmatch(r"(?:不要|别|不许|不能)(?:停(?:止)?(?:下|动作)?|听一下)(?:呀|啊|吧)?", text)
                or re.fullmatch(r"(?:don't|do not|never) stop", text, re.I))


def motion_stop_command(text: str) -> bool:
    """Pause for exact attention requests and clipped stop phrases in motion.

    停一下 is often transcribed as its near-homophone 听一下. Only this
    complete attention request is accepted, not stories or longer sentences.
    Motor noise can remove 停止 from 停止动作, leaving 体动作/身体动作/动作.
    These ambiguous fragments only pause an owned movement; they never start
    an action or acquire a motion meaning while idle.
    """
    value = text.translate(_FORMS).strip().rstrip("。.!！?？").strip()
    return (motion_command(value) == "stop" or value in {"体动作", "身体动作", "动作"}
            or bool(re.fullmatch(_PREFIX + r"听一下" + _SUFFIX, value)))


def unclear_stretch(primary: str, secondary: str) -> bool:
    """Suggest, never execute, a waist-motion sound the recognizers disagree on."""
    values = [re.sub(r"[\s，,。.!！?？]", "", s.translate(_FORMS)) for s in (primary, secondary)]
    if values[0] == values[1] or any(motion_command(s) or _NOT_COMMAND.search(s) for s in values):
        return False
    if any(re.search(r"我的|你的|他的|她的|我是|我叫|疼|痛|受伤", s) for s in values):
        return False
    return any(re.fullmatch(r"[\u4e00-\u9fff]{2,6}腰", s) for s in values)


def unclear_motion(primary: str, secondary: str, *, short_audio: bool) -> str | None:
    if unclear_stretch(primary, secondary):
        return "stretch"
    # These are ambiguous words, not commands. Offer an audible question; only
    # the next explicit confirmation can authorize a hug.
    values = [s.translate(_FORMS).strip().rstrip("。.!！?？") for s in (primary, secondary)]
    if (short_audio and values[0] != values[1]
            and any(s in {"报道", "報道", "报报", "報報", "宝宝", "寶寶", "包包", "棒棒"} for s in values)
            and not any(motion_command(s) or _NOT_COMMAND.search(s) or len(s) > 32
                        or re.search(r"我叫|我是|我的|你的|你叫|他叫|她叫", s) for s in values)):
        return "hug"
    return None


def motion_confirmation(text: str) -> bool | None:
    value = text.translate(_FORMS).strip().rstrip("。.!！?？").strip().lower()
    if value in {"是", "是的", "对", "对的", "好", "好的", "嗯", "可以", "没错", "去吧", "yes", "yes please", "go ahead"}:
        return True
    if value in {"不", "不是", "不是的", "不要", "不做", "不用", "不对", "不可以", "先别做", "别做了", "算了", "no", "no thanks"}:
        return False
    return None


def reconcile_confirmation(primary: str, secondary: str) -> tuple[str, bool]:
    """Only resolve consent-like fragments within an already offered action."""
    first, second = motion_confirmation(primary), motion_confirmation(secondary)
    fragment = lambda s: bool(re.fullmatch(r"(?:是吧|对吧|多的|好的吧)[。.!！?？]?", s.strip()))
    if first is not None and second is not None:
        return primary, first != second
    if second is False and first is None:
        return secondary, False  # A clear denial can only cancel, never authorize.
    if first is False:
        return primary, False
    if first is True:
        return primary, not fragment(secondary)
    if second is True and not primary:
        return secondary, True
    if second is not None and (not primary or fragment(primary)):
        return secondary, False
    acknowledgment = {"谢谢", "謝謝", "谢谢你", "謝謝你", "好吧"}
    unclear = fragment(primary) or fragment(secondary) or any(
        s.strip().rstrip("。.!！?？") in acknowledgment for s in (primary, secondary))
    # An unknown fragment during a spoken confirmation must not silently start
    # another-language conversation or authorize movement. Explicit new topics
    # and other exact actions can still leave the confirmation normally.
    new_topic = any(motion_command(s) or _NOT_COMMAND.search(s) for s in (primary, secondary))
    return primary, unclear or not new_topic

