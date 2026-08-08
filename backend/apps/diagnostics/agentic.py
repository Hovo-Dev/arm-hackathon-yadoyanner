"""The bridge between Django and the agentic layer (/carmed).

This is the only module that calls `carmed.run`. Everything it needs comes from
two ports we implement: `apps.cases.case_store.DjangoCaseStore` (pgvector over
the Case KB) and `apps.diagnostics.research.get_research_tool` (blank for now).

carmed itself is untouched -- it stays byte-identical to feature/agentic-loop so
that branch keeps working and the incoming ResearchTool branch merges cleanly.
"""
import logging
from functools import lru_cache

from django.conf import settings

from apps.cases.case_store import DjangoCaseStore

from carmed import Answer, Query, Vehicle, run

from .research import get_research_tool

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chat_model():
    """The LangChain chat model carmed drives its agents with.

    Returns None when no key is configured. carmed treats that as offline mode:
    the case gate, VIN checks and shop lookup still run, and the agents abstain
    instead of raising. That degradation is why this returns None rather than
    letting ChatOpenAI fail later, mid-graph.
    """
    if not settings.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is unset -- running the agentic loop offline.")
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.OPENROUTER_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=0,
    )


def build_query(diagnostic_request) -> Query:
    """DiagnosticRequest -> carmed Query.

    `symptom_text` is the right thing to send: it is the rolling summary of the
    whole conversation, refreshed after every user turn (see
    services.refresh_symptom_and_matches), so it already reflects the clarified
    problem rather than the user's first vague sentence.
    """
    return Query(
        text=diagnostic_request.symptom_text or "",
        vehicle=Vehicle(
            make=diagnostic_request.car_make or None,
            model=diagnostic_request.car_model or None,
            year=diagnostic_request.car_year,
            vin=diagnostic_request.vin or None,
        ),
        city=settings.CARMED_DEFAULT_CITY,
    )


def run_query(query: Query) -> Answer:
    """One pass through the graph. Synchronous and does network I/O -- it blocks
    the request thread, which is acceptable at this scale (put it behind Celery
    if that changes)."""
    answer = run(
        query,
        research=get_research_tool(),
        case_store=DjangoCaseStore(),
        model=get_chat_model(),
        safety_floor=settings.CARMED_SAFETY_FLOOR,
    )
    log_trace(query, answer)
    return answer


def run_for_request(diagnostic_request) -> Answer:
    return run_query(build_query(diagnostic_request))


def log_trace(query: Query, answer: Answer) -> None:
    """Emit the trace to the log as well as storing it on the request.

    Same shape as `python -m carmed.cli --trace`, so `docker compose logs web`
    shows which steps ran, what the request cost, and what it fetched -- without
    having to fish the JSON out of the database.
    """
    trace = answer.trace
    logger.info(
        "carmed %s | %s\n  steps      %s\n  llm calls  %s (cost: %d)\n"
        "  lookups    %s\n  notes      %s",
        answer.status.value,
        query.vehicle.describe(),
        " -> ".join(trace.steps) or "-",
        ", ".join(trace.llm_calls) or "none",
        trace.cost,
        "; ".join(trace.lookups) or "-",
        " | ".join(trace.notes) or "-",
    )
    if answer.dropped_refs:
        # Non-zero means an agent cited an id that resolved to nothing, which
        # points at unstable ids in a port implementation. Worth noticing.
        logger.warning("carmed dropped %d unresolvable reference(s)", answer.dropped_refs)
