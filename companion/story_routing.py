"""Recognize explicit story requests and revisions in spoken Mandarin."""

import re


_EXPLICIT_PAUSE = re.compile(
    r"^\s*(?:先|现在)?(?:(?:停一下|暂停|不续|不讲|别讲)故事(?:了|啦)?"
    r"|(?:停一下|暂停)(?:了|啦)?)(?=[，,。.!！?？\s]|$)"
)
_EXPLICIT_START = re.compile(
    r"^\s*(?:(?:小狗|你)[，,\s]*)?(?:请|给我|给我们|来|开始)?"
    r"(?:讲|说)(?:个|一个|一段)?[^。！？!?，,]{0,30}故事(?:吧|呀|啦)?"
    r"(?:[，,\s]*(?:讲)?(?:一|两|三|几)句(?:就好|就行|就可以))?[。！？!?\s]*$"
)
_EXPLICIT_CONTINUE = re.compile(
    r"^\s*(?:继续|接着|再)(?:给我)?讲(?:刚才的|那个|这个)?故事(?:吧|呀|啦)?[。！？!?\s]*$"
)
_SHORT_FOLLOWUP = re.compile(r"\s*(?:[然最][后後]|後來|后来|接下[来來])(?:呢|呀|啊)[。.!！?？\s]*")
_CHARACTER = r"[\u4e00-\u9fff]{1,10}?"
_REPLACE_CHARACTER = re.compile(
    rf"(?:把)?(?:故事(?:里|裡)?的?)?(?:主角|主人公)(?:的名字)?(?:改|换|換)成(?P<new>{_CHARACTER})(?:吧|呀|啦|好了)?"
)
_CORRECT_CHARACTER = re.compile(
    rf"(?:不要|别用|別用)(?P<old>{_CHARACTER})(?:了)?[，,。.!！\s]*(?:把主角)?(?:改|换|換)成"
    rf"(?P<new>{_CHARACTER})(?:吧|呀|啦|好了)?"
)
_CORRECTION_LEAD = re.compile(r"^(?:(?:哎|诶|唉)[，,\s]*)?(?:我[说說][错錯](?:了|啦)[，,。\s]*)?")
_SPOKEN_FORMS = str.maketrans("給們請講說個兩幾續剛這來開", "给们请讲说个两几续刚这来开")


def pauses_story(text: str) -> bool:
    """Only an explicit leading request pauses narrative for this turn."""
    return bool(_EXPLICIT_PAUSE.match(text.translate(_SPOKEN_FORMS)))


def requests_story(text: str) -> bool:
    """Use one story turn for an unmistakable request, leaving device mode alone."""
    text = text.translate(_SPOKEN_FORMS)
    return bool(_EXPLICIT_START.match(text) or _EXPLICIT_CONTINUE.match(text))


def short_story_followup(text: str) -> bool:
    return _SHORT_FOLLOWUP.fullmatch(text) is not None


def replacement_character(text: str, prior_story: str = "") -> str | None:
    """A bare swap needs an earlier story that actually mentioned the character."""
    utterance = _CORRECTION_LEAD.sub("", text.strip(), count=1).rstrip("。.!！?？ ")
    match = _REPLACE_CHARACTER.fullmatch(utterance)
    if match:
        return match["new"]
    match = _CORRECT_CHARACTER.fullmatch(utterance)
    if match and match["old"] in prior_story:
        return match["new"]
    return None
