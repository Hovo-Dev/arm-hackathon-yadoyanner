"""Streaming chat completions against the OpenRouter gateway (see
/setup/README.md) for the diagnostic conversation.

Loaded lazily and cached, same as apps.cases.embeddings: the client is only
constructed the first time something actually needs to talk to the LLM.
"""
import logging
import re
from functools import lru_cache

from django.conf import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.OPENROUTER_API_KEY, base_url=settings.OPENROUTER_BASE_URL)


# There is deliberately no free-text `stream_reply` here any more. The
# assistant's side of the conversation comes from the agentic run alone, so
# that every reply carries an urgency verdict and traceable evidence. A plain
# chat completion over the transcript could not, and answered confidently from
# nothing -- given the input "miedrmi" it invented a cylinder, a fault code and
# a manufacturer, while the graph correctly asked for clarification.


def complete(messages, *, reasoning: bool = False) -> str:
    """One-shot (non-streaming) chat completion. Shared by every LLM call in
    this app that wants a single finished string back instead of a stream:
    symptom summarization and the end-of-conversation final summary.

    Reasoning is off by default. deepseek-v4-pro is a reasoning model: it emits
    chain-of-thought before answering, and the caller waits for all of it.
    Measured on the summarizer, that was 18 seconds to produce one sentence --
    roughly 2,400 characters of deliberation about "say what the car is doing".
    The same call without reasoning returns in about 5 and says the same thing.

    Pass reasoning=True for a call whose quality genuinely depends on the model
    working through something. Summarizing a transcript is not one.
    """
    completion = _get_client().chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=messages,
        stream=False,
        # Every call through here is a transformation of text the caller
        # already has -- translate this, summarize that -- so sampling buys
        # nothing and costs correctness. Left unset it inherited the provider
        # default (~1.0), which is how a request to translate "is the squealing
        # constant while braking" came back as a fluent Armenian sentence about
        # valves, scale and a battery. carmed's own agents have always set this;
        # this path was simply missed.
        temperature=0,
        extra_body={} if reasoning else {"reasoning": {"enabled": False}},
    )
    return (completion.choices[0].message.content or "").strip()


# Latin-keyboard Armenian and Russian terms, glossed into the English the Case
# KB is written in. This exists because the model does not reliably know these
# spellings: "pervi vaxt" (first gear) came back as "for the first time", and
# "matory ercnuma" (the engine is boiling) as "car jerks and hesitates" -- a
# cooling fault rewritten as a driveability one, which the graph then diagnosed
# confidently as a crankshaft sensor.
#
# Deliberately single words and short noun phrases, never example sentences.
# The previous prompt ended with a full example symptom ("Clicking noise when
# turning left at low speed") that is, character for character, the opening of
# seeded case C1 in adapters/dummy.py; on input it could not parse the model
# echoed it back as the summary. That is where the "reproduced seeded case
# text" failure came from -- retrieval never leaked, the prompt quoted itself.
# A glossary entry that gets echoed costs one wrong word; an example sentence
# that gets echoed costs an entire fabricated complaint.
#
# carmed.text.SYMPTOM_KEYWORDS carries the same transliterations for a
# different purpose (keyword -> SystemArea) and is the place to look when
# adding here, but the two cannot share a table: that one needs only to know a
# word is about brakes, this one needs to know what it means.
_TRANSLITERATION_GLOSSARY = (
    "matory, motory, sharzhich, dvigatel = engine; "
    "ercnum, ercnuma, eruma, peregrev = boiling over / overheating; "
    "antifriz, tosol = antifreeze / coolant; "
    "radiator, termostat = radiator, thermostat; "
    "argelak, argelaknery, tormoz, tormaza = brakes; "
    "kalodka, kolodka = brake pads; "
    "crrum, crrua, skripit = squealing / grinding noise; "
    "karobka, karobki, korobka, kpp = gearbox / transmission; "
    "mexanika = manual gearbox; avtomat, akpp = automatic gearbox; "
    "pervi, pervaya = first (as in first gear); vtoroy = second; "
    "vaxt, peredacha = gear / speed; sceplenie, kcordich = clutch; "
    "dzayn, dzena, dzayner, zvuk, shum = noise / sound; "
    "anvahec, podshipnik = wheel bearing; anvadog, shina, rezin = tyre; "
    "amortizator, stoyka = shock absorber / strut; "
    "akumlyator, akumulyator = battery; generator = alternator; "
    "starter, svecha, mom = starter, spark plug; "
    "takic, snizu = from underneath; katacel, techet = leaking / dripping; "
    "chi varvum, ne zavoditsya = will not start; "
    "trcum, dergaetsya = jerking; aravot, utrom = in the morning"
)

# Faithfulness rules first, style last, because the expensive failure here is
# not an ugly sentence -- it is a fluent one about a car nobody described.
# symptom_text is what gets embedded for pgvector case matching, so it has to
# be canonical English (the KB is written in English, Russian and Armenian
# script; measured, Latin-script Armenian embeds to nothing useful -- it ranked
# the WRONG case top in 3 of 4 probes and never cleared the 0.75 gate). That is
# why this call cannot simply be skipped for short inputs: the summary is the
# normalization step the case gate depends on.
_SYMPTOM_SUMMARY_SYSTEM_PROMPT = (
    "You translate a car owner's complaint into one canonical English symptom "
    "description, for matching against a database of past cases. The owner "
    "writes in English, Armenian, Russian, or Armenian typed on a Latin "
    "keyboard.\n"
    "\n"
    "Rules, in order of importance:\n"
    "1. Report ONLY what the owner actually said. Never add a component, "
    "noise, behaviour, driving condition or timing that is not in their words, "
    "and never make one more specific than they made it. If they said the "
    "engine is overheating, do not decide why.\n"
    "2. If you are unsure what a word means, keep it as the owner wrote it. An "
    "untranslated word is a small problem; a confidently wrong translation is "
    "the failure this prompt exists to prevent.\n"
    "3. Do not name a system the owner did not name. A complaint about the "
    "gearbox must not come back as one about the engine.\n"
    "4. No diagnosis, no cause, no parts, no urgency, no advice, no questions, "
    "no apologies, no commentary -- only what the car is doing.\n"
    "5. 1-3 sentences of plain English, present tense. Output the description "
    "alone.\n"
    "\n"
    "If you cannot follow these rules, output a plain English restatement of "
    "the owner's words and nothing more. Saying less than the owner did is "
    "always acceptable; saying something they did not is never.\n"
    "\n"
    "Glossary of terms owners here type on a Latin keyboard:\n"
    f"{_TRANSLITERATION_GLOSSARY}"
)


def summarize_symptom(transcript: str) -> str:
    """Collapses the whole conversation so far into a symptom_text-shaped
    string, so it stays comparable to CaseRecord.symptom_text in embedding
    space.

    Runs on every input, short ones included. Skipping it for terse complaints
    is tempting -- five words have no redundancy to compress, and compression
    is where the invention happened -- but measured against the KB, that trades
    a wrong summary for a guaranteed missed match: raw Latin-script Armenian
    ranked the wrong case first in 3 of 4 probes and scored 0.10-0.42 where the
    gate needs 0.75. Translation, not compression, is what this call is for.

    An empty return leaves the previous symptom_text alone (see
    services.refresh_symptom_and_matches). That is safe now that the graph
    reads the owner's raw words from Query.raw_text rather than from here --
    a failed summary costs a missed case match, not a lost complaint.
    """
    return complete(
        [
            {"role": "system", "content": _SYMPTOM_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
    )


# The same glossary as above, used in the other direction. Written once and
# read twice on purpose: the words the owner types are exactly the words the
# reply should come back in, and two lists would drift.
#
# Armenian script rather than Latin transliteration, from an A/B on the same
# eight replies. Latin came back readable but with invented words the glossary
# had not covered ("chherqakan", "votchntchacrel", "Poqi komplekt"). Script
# came back in spoken Armenian -- the loanwords drivers use in their own
# alphabet, the colloquial copula, and the Armenian question mark placed
# correctly inside the word. Script also gets the failure mode we can actually
# detect: a reply in the wrong alphabet is a codepoint count, while a Latin
# word that does not exist is not checkable at all.
_REPLY_SYSTEM_PROMPT = (
    "Rewrite the assistant's reply in Armenian, written in the ARMENIAN "
    "ALPHABET (Հայերեն տառերով). Never answer in Latin letters.\n\n"
    "Use everyday spoken Armenian, the way a driver in Yerevan talks -- not "
    "formal literary Armenian.\n\n"
    "Rules, in order:\n"
    "1. Keep every number exactly as written: prices, years, part numbers, "
    "measurements, OBD codes. Never convert or round them.\n"
    "2. Keep the meaning. Do not add advice, do not drop a caveat, do not "
    "make a hedged statement sound certain.\n"
    "3. Use the everyday words drivers here actually use -- mostly Russian "
    "loanwords, spelled in Armenian letters (տոսոլ, կալոդկա, տորմոզ, "
    "կարոբկա, մատոր) -- not invented literary equivalents:\n"
    f"   {_TRANSLITERATION_GLOSSARY}\n"
    "4. If a term has no everyday Armenian form, leave the English word.\n"
    "5. Reply with the rewritten text only. No quotes, no notes, no original."
)

_ARMENIAN_CHARS = re.compile(r"[\u0530-\u058F\uFB13-\uFB17]")
_LATIN_CHARS = re.compile(r"[A-Za-z]")

#: Below this the model ignored the alphabet instruction. Not 1.0: part
#: numbers, "AMD" and untranslatable English terms are Latin on purpose.
_MIN_ARMENIAN = 0.6


def _armenian_share(text: str) -> float:
    armenian = len(_ARMENIAN_CHARS.findall(text))
    latin = len(_LATIN_CHARS.findall(text))
    total = armenian + latin
    return armenian / total if total else 0.0


#: Latin-script Armenian markers. Every one of these is either a word that
#: does not exist in English, or a Russian loanword no English car complaint
#: would use. Words the two languages share are deliberately absent -- an
#: English owner writing "the radiator is leaking" must not be answered in
#: Armenian, so "radiator", "generator", "starter" and "termostat" are all
#: excluded even though they are in the glossary above.
#:
#: One hit is enough. Someone typing Latin-script Armenian reaches for these
#: constantly; someone typing English never reaches for them at all.
_LATIN_ARMENIAN_MARKERS = frozenset("""
matory motory sharzhich dvigatel ercnum ercnuma eruma peregrev antifriz tosol
argelak argelaknery tormoz tormoza tormozner kalodka kolodka kalodkanery
crrum crrua skripit karobka karobki korobka kpp mexanika avtomat akpp
pervi pervaya vtoroy vaxt peredacha sceplenie kcordich
dzayn dzena dzayner anvahec podshipnik anvadog shina rezin amortizator stoyka
akumlyator akumulyator svecha takic snizu katacel techet trcum dergaetsya
aravot utrom chi che chka varvum zavoditsya inch incha anem anum exa exav
vonc vor bayc erb miayn noric enq eiq petqa petq karam uzum lyuft suloc
poshi meqenan mashinan yuxi yuxa yughi makardak stugel poxel gnum galis linum
litr kilometry pizdec varum
""".split())

#: The complementary signal. English car complaints are full of these; Latin
#: -script Armenian contains none of them, because they are English grammar
#: rather than English vocabulary. Single letters are excluded on purpose --
#: "a" is the Armenian copula ("dzayn a talis") and would fire on every line.
#:
#: This exists so the marker list above does not have to grow forever. It
#: catches what the markers miss: "Yuxa varum pizdec 1000 kilometry 1 litr"
#: hits no marker but contains no English either.
_ENGLISH_STOPWORDS = frozenset("""
the and or but is are was were be been being has have had does do did
when while after before if then than that this these those there here
my your his her its our their me you it him them
on at in to of for from with without into over under
only not no very more most some any all both each
car engine noise when make model year problem issue sound
""".split())


_WORD = re.compile(r"[a-z]+")


def looks_armenian(text: str) -> bool:
    """Should the reply come back in Armenian?

    Three inputs have to be told apart and only two of them are visible to an
    alphabet counter. Armenian script is obvious. English is obvious. Latin
    -script Armenian -- "karobki pervi vaxt dzena galis" -- is Latin letters
    and reads as English to `carmed.text.detect_lang`, which is documented
    there as a known gap.

    So this looks at vocabulary rather than alphabet, against words English
    does not have. Deliberately not a model call: this decision runs on every
    answer, and it is a lookup.
    """
    body = (text or "").strip()
    if not body:
        return False
    if _ARMENIAN_CHARS.search(body):
        return True

    words = _WORD.findall(body.lower())
    if _LATIN_ARMENIAN_MARKERS & set(words):
        return True

    # Nothing Armenian recognised, but nothing English either. Short inputs
    # are excluded: two words is not enough absence to conclude anything.
    return len(words) >= 3 and not (_ENGLISH_STOPWORDS & set(words))


def to_armenian(text: str) -> str:
    """English reply -> the Armenian the owner reads.

    Display only, and deliberately the last thing that happens. Everything the
    system reasons with, stores and matches on stays English: `summary` feeds
    the case store (`Case.fix` is built from `causes[0].title`), and a KB half
    in Armenian would stop matching the English symptoms it is queried with.

    Separating translation from generation is the point. Asked to *produce*
    Armenian, the diagnostician does the reasoning and the wording in one pass,
    so a bad word choice and a bad diagnosis become the same failure. Asked to
    *translate* a finished English answer, only the wording can go wrong -- and
    measured, the facts survive: prices and part numbers came through verbatim.

    Retried once when the answer comes back in the wrong alphabet, which
    measured at 3 in 10 even with the instruction written in Armenian script.
    The retry is worth having precisely because this failure is countable --
    the whole reason for preferring script over transliteration.

    Never falls back to English on a bad alphabet: a reply in transliterated
    Armenian still reads to the owner, and English does not. English comes
    back only when the call itself failed.
    """
    body = (text or "").strip()
    if not body:
        return text

    best = ""
    for attempt in range(2):
        messages = [
            {"role": "system", "content": _REPLY_SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]
        if attempt:
            messages.insert(1, {
                "role": "system",
                "content": ("The previous attempt used Latin letters. Answer "
                            "ONLY in the Armenian alphabet this time."),
            })
        try:
            out = complete(messages).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reply translation failed (%s) -- sending English.", exc)
            return best or text
        if not out:
            continue
        if _armenian_share(out) >= _MIN_ARMENIAN:
            return out
        # Keep the fuller attempt rather than the last one.
        if _armenian_share(out) > _armenian_share(best):
            best = out
        logger.info("Reply came back in Latin script (attempt %d) -- retrying.", attempt + 1)

    return best or text
