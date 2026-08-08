import logging

from django.conf import settings

from apps.cases.case_store import DjangoCaseStore
from apps.cases.embeddings import embed_text

from . import agentic
from .enums import DiagnosticStatus
from .llm import summarize_symptom
from .models import DiagnosticCaseMatch

logger = logging.getLogger(__name__)

def _conversation_transcript(diagnostic_request):
    messages = diagnostic_request.messages.order_by("created_at")
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def refresh_symptom_and_matches(diagnostic_request):
    """Called after every user turn: re-summarizes the whole conversation so
    far into a symptom_text-shaped string, re-embeds it, and re-runs the
    Case KB match against the fresh embedding. Without this, case_matches
    would freeze at whatever the very first message said (spec section 3 --
    Agent 4 is meant to be a live gate, not a one-time lookup)."""
    transcript = _conversation_transcript(diagnostic_request)
    if not transcript:
        return

    try:
        symptom_text = summarize_symptom(transcript)
    except Exception:
        # A flaky summarization call shouldn't block the conversation --
        # matches just stay as they were until the next successful turn.
        return
    if not symptom_text:
        return

    diagnostic_request.symptom_text = symptom_text
    diagnostic_request.embedding = embed_text(symptom_text)
    diagnostic_request.save(update_fields=["symptom_text", "embedding", "updated_at"])

    # Same store the agentic layer's case gate uses, so the matches shown during
    # the conversation and the ones the graph reasons over at finalize can never
    # disagree. carmed.Case.id is the CaseRecord pk as a string.
    found = DjangoCaseStore().find_similar(
        text=symptom_text,
        vehicle=agentic.build_query(diagnostic_request).vehicle,
        limit=settings.CARMED_CASE_LIMIT,
    )

    diagnostic_request.case_matches.all().delete()
    matches = [
        DiagnosticCaseMatch(
            request=diagnostic_request, case_id=int(match.case.id), confidence=match.score
        )
        for match in found
        if match.score >= settings.CASE_MATCH_CONFIDENCE_THRESHOLD
    ]
    if matches:
        DiagnosticCaseMatch.objects.bulk_create(matches)
        diagnostic_request.status = DiagnosticStatus.CASE_MATCHED
    elif diagnostic_request.status == DiagnosticStatus.CASE_MATCHED:
        diagnostic_request.status = DiagnosticStatus.PENDING
    diagnostic_request.save(update_fields=["status"])

    # The retrieval half of the flow, logged in the same place as the agentic
    # trace -- otherwise the only visible sign a turn re-ran the KB search is
    # the case_matches list quietly changing in the API response.
    logger.info(
        "kb refresh #%s | symptom=%r -> %d candidate(s) %s, %d kept above %.2f",
        diagnostic_request.pk,
        symptom_text,
        len(found),
        [f"case:{m.case.id}@{m.score:.2f}/{m.tier}" for m in found],
        len(matches),
        settings.CASE_MATCH_CONFIDENCE_THRESHOLD,
    )


def build_final_summary(diagnostic_request):
    """Run the agentic layer over the conversation and return the Answer as JSON.

    This used to be a single prompt asking an LLM to *imitate* this shape. It is
    now the real thing: carmed routes the intent, checks the Case KB gate, runs
    the diagnostician and parts explorer as ReAct agents, applies the safety
    floor, and strips any reference that doesn't resolve. The grounding rules
    the old prompt merely requested are enforced in code -- an agent that writes
    a price has nowhere to put it, and unresolvable ids are dropped and counted
    in `dropped_refs`.

    Stored verbatim on DiagnosticRequest.summary; see carmed.models.Answer for
    the field list, and `trace` for what actually ran and what it cost.
    """
    return agentic.run_for_request(diagnostic_request).model_dump(mode="json")
