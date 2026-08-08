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


def stream_reply(messages):
    """messages: list of {"role", "content"} dicts, oldest first.
    Yields text deltas as they arrive from the model."""
    stream = _get_client().chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def complete(messages) -> str:
    """One-shot (non-streaming) chat completion. Shared by every LLM call in
    this app that wants a single finished string back instead of a stream:
    symptom summarization and the end-of-conversation final summary."""
    completion = _get_client().chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=messages,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


_SYMPTOM_SUMMARY_SYSTEM_PROMPT = (
    "You maintain a short, current summary of the car problem being diagnosed, "
    "based on a conversation between a car owner and an assistant. Output ONLY "
    "the symptom description in 1-3 concise sentences, in the style of a forum "
    "post (e.g. 'Clicking noise when turning left at low speed'). Do not include "
    "a diagnosis, parts, urgency, or any commentary -- just what the car is doing."
)


def summarize_symptom(transcript: str) -> str:
    """Collapses the whole conversation so far into a symptom_text-shaped
    string, so it stays comparable to CaseRecord.symptom_text in embedding
    space."""
    return complete(
        [
            {"role": "system", "content": _SYMPTOM_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
    )
