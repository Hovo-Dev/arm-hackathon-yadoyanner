"""Streaming chat completions against the OpenRouter gateway (see
/setup/README.md) for the diagnostic conversation.

Loaded lazily and cached, same as apps.cases.embeddings: the client is only
constructed the first time something actually needs to talk to the LLM.
"""
from functools import lru_cache

from django.conf import settings


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
