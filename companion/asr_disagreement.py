"""Local Mandarin ASR cross-checks and a short-lived secondary recognizer."""

import asyncio
from pathlib import Path
import re
import tempfile
import wave

from .story_routing import requests_story
from .child_activity import stops_activity


CLARIFY_HOMOPHONE = "你说的是水果香蕉，还是两个东西相交？"
_CHILD_NUMBER = r"[0-9零一二三四五六七八九十两]{1,4}"
_ARITHMETIC = re.compile(
    rf"^(?P<left>{_CHILD_NUMBER})(?P<operator>加|减|乘|除|[+\-×÷])"
    rf"(?P<right>{_CHILD_NUMBER})(?:等于|是)(?P<ending>.+)$")
_HOMOPHONE_ARITHMETIC = re.compile(
    rf"^(?P<left>{_CHILD_NUMBER})(?P<operator>加|减|乘|除|[+\-×÷])"
    rf"(?P<right>{_CHILD_NUMBER})(?P<ending>[的得]雨季)$")
_QUESTION_ENDINGS = {"几", "多少", "什么"}
_NUMERIC_ONLY = re.compile(r"[0-9]{3,6}")
_SINGLE_DIGITS = dict(zip("零一二三四五六七八九", "0123456789"))
_SINGLE_DIGITS["两"] = "2"
_TRANSLATION_QUESTION = re.compile(
    r"英[语語文][\s，,]*(?:怎[么麼样樣][说說]|是什[么麼])"
)
_COUNT_QUESTION = re.compile(
    r"(?P<subject>[\u4e00-\u9fff]{1,12}?)(?:通常|一般)?(?:有|长着|长了)"
    r"(?:几|多少)(?P<unit>[条只个对根片张])(?P<part>[\u4e00-\u9fff]{0,12})"
)
_COUNT_FORMS = str.maketrans("貓鳥魚鴨雞馬幾條隻個對長腳觸鬍鬚張兩總來", "猫鸟鱼鸭鸡马几条只个对长脚触胡须张两总来")
_BODY_PARTS = {
    "条": {"腿", "尾巴", "触手"},
    "只": {"腿", "脚", "脚丫", "脚爪", "眼睛", "耳朵", "翅膀"},
    "个": {"头", "鼻子", "嘴", "嘴巴", "脚趾", "眼睛"},
    "对": {"耳朵", "翅膀", "眼睛", "触角"},
    "根": {"腿", "胡须"},
    "片": {"翅膀"},
    "张": {"嘴", "嘴巴"},
}
_KNOWN_BODY_PARTS = set().union(*_BODY_PARTS.values())
_ANIMALS = {"猫", "狗", "兔", "兔子", "鸭", "鸭子", "鸡", "鸟", "鱼", "羊",
            "马", "牛", "蜘蛛", "螃蟹", "蝴蝶", "蜜蜂"}


def _body_count(text: str) -> re.Match | None:
    clean = re.sub(r"[\s，,]", "", text.translate(_COUNT_FORMS)).rstrip("。.!！?？")
    return _COUNT_QUESTION.fullmatch(clean)


def public_animal_body_count(text: str) -> tuple[str, str] | None:
    """Return the animal and part for an exact, generic anatomy question."""
    match = _body_count(text)
    if match and match["subject"].removeprefix("小") in _ANIMALS:
        part = match["part"].rstrip("呀啊啦呢嘛")
        if part in _BODY_PARTS[match["unit"]]:
            return match["subject"], part
    return None


def public_animal_group_count(text: str) -> tuple[str, str, int] | None:
    """Parse an explicit total for a stated number of the same animal."""
    clean = re.sub(r"[\s，,]", "", text.translate(_COUNT_FORMS)).rstrip("。.!！?？")
    # In quiet recordings "那两只" can be rendered "大兩隻". Accept this
    # lead-in only when the explicit count, animal and whole question match.
    match = re.fullmatch(r"(?:那|大(?=两只))?(?P<count>[1-9][0-9]?|[一二三四五六七八九十两])"
                         r"[只头匹](?P<animal>[\u4e00-\u9fff]{1,4}?)(?:一共|总共|加起来)有"
                         r"(?P<question>.+)", clean)
    if match:
        body = public_animal_body_count(f"{match['animal']}有{match['question']}")
        if body:
            count = match["count"]
            number = 10 if count == "十" else int(_SINGLE_DIGITS.get(count, count))
            return body[0], body[1], number
    return None


def resolve_body_count(primary: str, secondary: str, previous_question: str = "") -> tuple[str, str | None]:
    """Repair malformed body parts and clarify conflicting generic anatomy questions."""
    group_a, group_b = public_animal_group_count(primary), public_animal_group_count(secondary)
    if group_a and group_b and group_a[:2] == group_b[:2] and group_a[2] != group_b[2]:
        return primary, f"这次是{group_a[2]}只还是{group_b[2]}只{group_a[0]}呀？"
    first, second = _body_count(primary), _body_count(secondary)
    if not second:
        return primary, None
    part = second["part"].rstrip("呀啊啦呢嘛")
    if part not in _BODY_PARTS[second["unit"]]:
        return primary, None
    second_public = public_animal_body_count(secondary)
    if not first:
        if second_public and part in {"腿", "脚", "脚丫", "脚爪"}:
            clean = re.sub(r"[\s，,]", "", primary.translate(_COUNT_FORMS)).rstrip("。.!！?？")
            animal = re.escape(second_public[0].removeprefix("小"))
            fragment = re.fullmatch(
                rf"小?{animal}有(?P<body>[\u4e00-\u9fff]{{0,8}}(?:腿|脚|脚丫|脚爪)(?:呀|啊|呢|啦|嘛)?)",
                clean,
            )
            if fragment and not re.search(r"[0-9零一二三四五六七八九十两双对]", fragment["body"]):
                return secondary, None
        return primary, None
    primary_part = first["part"].rstrip("呀啊啦呢嘛")
    first_public = public_animal_body_count(primary)
    if (first_public and second_public
            and first_public[0].removeprefix("小") == second_public[0].removeprefix("小")
            and primary_part != part
            and not {primary_part, part} <= {"腿", "脚", "脚丫", "脚爪"}):
        return primary, (f"你问的是{first['subject']}有几{first['unit']}{primary_part}，"
                         f"还是有几{second['unit']}{part}呀？")
    if first[0][:first.start("part")] != second[0][:second.start("part")]:
        return primary, None
    if not primary_part:
        return secondary, None
    if (second_public and primary_part in _KNOWN_BODY_PARTS
            and primary_part not in _BODY_PARTS[first["unit"]]):
        return secondary, None
    if part not in {"脚", "脚丫"}:
        return primary, None
    if first["part"] in {"狡雅", "狡呀", "脚鸭"}:
        return secondary, None
    if primary_part == "小鸭" and public_animal_body_count(secondary):
        earlier = public_animal_body_count(previous_question)
        subject = second["subject"]
        if (earlier and earlier[0].removeprefix("小") == subject.removeprefix("小")
                and earlier[1] in {"腿", "脚", "脚丫", "脚爪"}):
            return secondary, None
        return primary, f"你问的是{subject}有几只脚，还是有几只小鸭呀？"
    return primary, None


def prefer_quiet_translation_question(primary: str, secondary: str) -> str:
    """Prefer the Mandarin recognizer when both heard a translation question."""
    if _TRANSLATION_QUESTION.search(primary) and _TRANSLATION_QUESTION.search(secondary):
        return secondary
    return primary


def prefer_quiet_story_request(primary: str, secondary: str) -> str:
    """Cross-check a quiet story request only when the activity and hero agree."""
    if not (requests_story(primary) and requests_story(secondary)):
        return primary
    forms = str.maketrans("貓鳥給講個這", "猫鸟给讲个这")
    first, second = primary.translate(forms), secondary.translate(forms)
    for marker in ("不要", "不讲", "别", "我的", "我家", "昨天", "上次", "记得", "记住", "保存"):
        if (marker in first) != (marker in second):
            return primary
    hero = re.search(r"小[熊兔猫狗鸟羊牛马]", first)
    if hero and hero[0] in second:
        return secondary
    return primary


def prefer_clear_activity_control(primary: str, secondary: str) -> str:
    """Recover a malformed short stop command, never reverse a clear start."""
    def compact(text):
        return re.sub(r"[\s，,。.!！?？]", "", text).translate(str.maketrans("講學語", "讲学语"))
    first, second = compact(primary), compact(secondary)
    if stops_activity(primary) or not stops_activity(secondary) or len(first) != len(second):
        return primary
    if requests_story(primary) or re.search(r"继续|接着|开始|给我|先[去要想来來在会會就能再只]", first):
        return primary
    differences = [(a, b) for a, b in zip(first, second) if a != b]
    if (len(differences) == 1 and differences[0][1] == "不"
            and any(activity in first and activity in second for activity in ("讲故事", "学英语", "玩英语"))):
        return secondary
    return primary


def numeric_only_clarification(transcript: str) -> str | None:
    """Do not answer a long bare number as if it were a complete question."""
    compact = re.sub(r"[\s，。？！?!]", "", transcript)
    if _NUMERIC_ONLY.fullmatch(compact):
        return "我只听到一串数字，没听清问题，请再说一次好吗？"
    return None


def prefer_clear_arithmetic_question(primary: str, secondary: str) -> str:
    """Use a second local recognizer only to repair a malformed math question.

    Both recognizers must agree on the numbers and operation. This does not
    guess a child's intended numbers or replace otherwise valid questions.
    """
    def parts(text: str):
        compact = re.sub(r"[\s，。？！?!]", "", text.replace("於", "于"))
        match = _ARITHMETIC.fullmatch(compact) or _HOMOPHONE_ARITHMETIC.fullmatch(compact)
        if not match:
            return None
        operator = {"+": "加", "-": "减", "×": "乘", "÷": "除"}.get(
            match["operator"], match["operator"])
        return match["left"], operator, match["right"], match["ending"]

    first, second = parts(primary), parts(secondary)
    if not first and second and second[3] in _QUESTION_ENDINGS:
        compact_primary = re.sub(r"[\s，。？！?!]", "", primary)
        digits = "".join(_SINGLE_DIGITS.get(part, part) for part in (second[0], second[2]))
        if _NUMERIC_ONLY.fullmatch(compact_primary) and len(digits) == 2 and digits.isdigit() \
                and compact_primary.startswith(digits):
            return secondary
    if (first and second and first[:3] == second[:3]
            and first[3] not in _QUESTION_ENDINGS and second[3] in _QUESTION_ENDINGS):
        return secondary
    return primary


def clarification_for_transcripts(primary: str, secondary: str) -> str | None:
    """Ask rather than confidently answer when 香蕉/相交 meaning is uncertain."""
    compact_primary = "".join(primary.split())
    compact_secondary = "".join(secondary.split())
    if not compact_primary or not compact_secondary:
        return None
    # One recognizer heard the fruit while the other heard the geometry term.
    primary_terms = ("香蕉" in compact_primary, "相交" in compact_primary)
    secondary_terms = ("香蕉" in compact_secondary, "相交" in compact_secondary)
    if primary_terms != secondary_terms and (
        primary_terms in ((True, False), (False, True))
        and secondary_terms in ((True, False), (False, True))
    ):
        return CLARIFY_HOMOPHONE
    # Both recognizers can make the same substitution. An impossible pairing
    # of a fruit with circles, or geometry with explicit fruit, is also unsafe.
    for text in (compact_primary, compact_secondary):
        if ("香蕉" in text and any(word in text for word in ("两个圆", "两圆", "圆形"))
                or "相交" in text and "水果" in text):
            return CLARIFY_HOMOPHONE
    return None


async def apple_transcribe_pcm(pcm: bytes, binary: Path, timeout: float = 6.0,
                               *, locale: str | None = None) -> str:
    """Transcribe complete 16 kHz mono PCM with an installed local probe."""
    if not pcm or len(pcm) % 2 or len(pcm) > 16000 * 2 * 30:
        raise ValueError("Expected at most 30 seconds of complete 16 kHz PCM")
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if locale not in {None, "en_US"}:
        raise ValueError("Unsupported speech locale")
    with tempfile.TemporaryDirectory(prefix="puppy-apple-asr-") as directory:
        path = Path(directory) / "audio.wav"
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm)
        process = await asyncio.create_subprocess_exec(
            str(binary), str(path), *([locale] if locale else []), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if process.returncode != 0:
            raise RuntimeError("Local secondary ASR failed")
        return stdout.decode("utf-8").strip()[:300]
