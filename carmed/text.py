"""Language detection, symptom routing, and the grounding check.

All deterministic. No model, no dependency.

The symptom table is trilingual on purpose. It narrows every retrieval and it
is what an optional safety floor keys on, so an English-only version would
silently fail on Armenian input -- the input this product most needs to get
right. Latin transliterations are included because that is what people
actually type on a phone.
"""

from __future__ import annotations

import re
import unicodedata

from carmed.models import Lang, SystemArea

# Unicode blocks, not word lists. Armenian, Russian and English use three
# different alphabets, so counting characters settles it -- no model, no
# dependency, and it works on a two-word fragment.
_ARMENIAN = re.compile(r"[԰-֏]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> Lang:
    """Whichever alphabet appears most. UNKNOWN if none do.

    Known gap: transliterated Armenian typed on a Latin keyboard ("anvaheci
    dzayn") reads as English. Handled by keeping transliterations in the
    symptom table below rather than by guessing here.
    """
    if not text.strip():
        return Lang.UNKNOWN
    counts = {
        Lang.HY: len(_ARMENIAN.findall(text)),
        Lang.RU: len(_CYRILLIC.findall(text)),
        Lang.EN: len(_LATIN.findall(text)),
    }
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else Lang.UNKNOWN


def normalize(text: str) -> str:
    """Casefold, drop accents and punctuation. Keeps Armenian intact.

    NFKD splits accented characters into base + combining mark, so dropping
    the marks makes "ё" match "е" without touching Armenian, which has no
    combining marks to lose.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", stripped)).strip()


SYMPTOM_KEYWORDS: dict[SystemArea, tuple[str, ...]] = {
    SystemArea.BRAKES: (
        "արգելակ", "կալոդկա", "тормоз", "колодк", "суппорт", "абс",
        # "braking" is listed separately because "brake" is not a substring of
        # it. Without this the safety floor silently fails to fire on the most
        # common English phrasing of a brake fault ("noise when braking"), which
        # is the one miss that actually hurts someone. The Russian and Armenian
        # entries are already stems and need no equivalent.
        "brake", "braking", "pad", "caliper", "rotor", "abs", "argelak", "tormoz",
    ),
    SystemArea.STEERING: (
        "ղեկ", "руль", "рулев", "рейк", "гур",
        "steering", "steer", "rack", "tie rod", "ghek",
    ),
    SystemArea.SUSPENSION: (
        "ամորտիզատոր", "անվահեծ", "կախոց", "առանցքակալ",
        "амортизатор", "подвеск", "рычаг", "стойк", "шаровая", "подшипник", "ступиц",
        "suspension", "shock", "strut", "bushing", "ball joint", "wheel bearing",
        "bearing", "control arm", "amortizator", "anvahec",
    ),
    SystemArea.AIRBAG: (
        "բարձիկ", "подушк", "airbag", "air bag", "srs", "podushka",
    ),
    SystemArea.ENGINE: (
        "շարժիչ", "մոտոր", "մոմ", "двигател", "мотор", "масл", "свеч", "троит",
        "engine", "motor", "misfire", "spark plug", "stall", "dvigatel",
    ),
    SystemArea.TRANSMISSION: (
        "փոխանցման", "կցորդիչ", "коробк", "кпп", "акпп", "сцеплен", "переключ",
        "transmission", "gearbox", "clutch", "shift", "cvt", "korobka",
    ),
    SystemArea.ELECTRICAL: (
        "մարտկոց", "գեներատոր", "аккумулятор", "генератор", "стартер", "предохранител",
        "electrical", "battery", "alternator", "fuse", "starter", "wiring",
    ),
    SystemArea.COOLING: (
        "ռադիատոր", "անտիֆրիզ", "радиатор", "охлажден", "антифриз", "перегрев",
        "cooling", "radiator", "coolant", "thermostat", "overheat",
    ),
    SystemArea.EXHAUST: (
        "կատալիզատոր", "выхлоп", "глушител", "катализатор", "лямбда",
        "exhaust", "muffler", "catalytic", "lambda", "o2 sensor",
    ),
    SystemArea.HVAC: (
        "կոնդիցիոներ", "кондиционер", "печк", "отопител",
        "air conditioning", "heater", "hvac", "blower",
    ),
    SystemArea.BODY: (
        "դուռ", "ապակի", "լուսարձակ", "кузов", "дверь", "стекл", "фар", "бампер",
        "door", "window", "headlight", "bumper", "fender",
    ),
}  # fmt: skip

#: Longest keywords first, so specific beats generic.
_FLAT = tuple(
    sorted(
        ((normalize(k), area) for area, kws in SYMPTOM_KEYWORDS.items() for k in kws),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)


def classify_area(text: str) -> SystemArea:
    """Single best area, used to narrow retrieval and match shops.

    First match wins, and ``_FLAT`` is sorted longest-first so a specific word
    beats a generic one. Known limitation: a sentence naming two systems
    ("brakes squeal and the engine misfires") yields only one, and it may not
    be the safety-critical one -- which is exactly why the safety floor uses
    :func:`all_areas` instead of this.
    """
    hay = normalize(text)
    for needle, area in _FLAT:
        if needle and needle in hay:
            return area
    return SystemArea.UNKNOWN


def all_areas(text: str) -> set[SystemArea]:
    """Every area mentioned.

    A safety floor must see all of them: missing "brakes" because "misfire"
    matched first would be the one failure that actually hurts someone.
    """
    hay = normalize(text)
    return {area for needle, area in _FLAT if needle and needle in hay}


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

#: Money, phone and URL shapes. Agents are told never to write these; this is
#: the check that they didn't. It is tested against known-bad strings, because
#: a counter that has never fired looks exactly like a broken one.
_LEAK_PATTERNS = (
    re.compile(
        r"(?:[$€₽֏]\s?\d[\d\s,.]*)"
        r"|(?:\d[\d\s,.]*\s?(?:amd|usd|rub|eur|դր|դրամ|руб|dollars?))",
        re.IGNORECASE,
    ),
    re.compile(r"\+374[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}|\b0\d{2}[\s\-]?\d{6}\b"),
    re.compile(r"https?://\S+|\blist\.am\S*", re.IGNORECASE),
)


def find_leaks(text: str) -> list[str]:
    """Commerce-shaped substrings an agent should not have written itself."""
    if not text:
        return []
    return [m.group(0).strip() for p in _LEAK_PATTERNS for m in p.finditer(text)]
