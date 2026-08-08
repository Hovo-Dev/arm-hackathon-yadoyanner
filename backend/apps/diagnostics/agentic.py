"""The bridge between Django and the agentic layer (/carmed).

This is the only module that calls `carmed.run`. Everything it needs comes from
two ports we implement: `apps.cases.case_store.DjangoCaseStore` (pgvector over
the Case KB) and `apps.diagnostics.research.get_research_tool` (blank for now).

carmed itself is untouched -- it stays byte-identical to feature/agentic-loop so
that branch keeps working and the incoming ResearchTool branch merges cleanly.
"""
import logging
import queue
import threading
from functools import lru_cache

from django.conf import settings
from django.db import connection

from apps.cases.case_store import DjangoCaseStore

from carmed import Answer, Query, Vehicle, build, run
from carmed.graph import State
from carmed.models import Diagnosis, Trace

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


# ---------------------------------------------------------------------------
# Streaming the same run, so a UI can show the pipeline working
# ---------------------------------------------------------------------------


def _tap(log, events: queue.Queue) -> None:
    """Mirror everything the graph records into a queue as it happens.

    carmed's RunLog is written to synchronously by each step, but only readable
    once `run()` returns -- fine for logging, useless for showing progress. The
    four record methods are wrapped on the instance (the steps look the
    attribute up at call time, so this takes effect for a graph already built)
    and keep their original behaviour, so `log` still ends up complete and the
    final Trace is unaffected.
    """
    original = {name: getattr(log, name) for name in ("step", "llm", "lookup", "note")}

    def mirror(name, to_event):
        def wrapped(*args):
            original[name](*args)
            events.put(to_event(*args))

        return wrapped

    log.step = mirror("step", lambda name: {"type": "step", "name": name})
    log.llm = mirror("llm", lambda agent: {"type": "llm", "agent": agent})
    log.lookup = mirror("lookup", lambda name, count: {"type": "lookup", "name": name, "count": count})
    log.note = mirror("note", lambda text: {"type": "note", "text": text})


def _assemble_answer(final_state: State, log) -> Answer:
    """State + RunLog -> Answer.

    Mirrors the tail of carmed.graph.run, which we can't reuse directly because
    it owns its RunLog and only hands it back after the run has finished.
    `test_streaming_matches_the_plain_run` pins the two together so this can't
    drift silently.
    """
    diagnosis = final_state.diagnosis or Diagnosis()
    return Answer(
        status=final_state.status,
        vehicle=final_state.vehicle,
        intent=final_state.intent,
        causes=diagnosis.causes,
        urgency=diagnosis.urgency,
        urgency_reason=diagnosis.urgency_reason,
        repair_steps=diagnosis.repair_steps,
        parts=final_state.parts,
        shops=final_state.shops,
        listings=final_state.listings,
        message=final_state.message,
        from_cache=final_state.cache_hit,
        dropped_refs=final_state.dropped_refs,
        trace=Trace(
            steps=list(log.steps),
            llm_calls=list(log.llm_calls),
            lookups=list(log.lookups),
            notes=list(log.notes),
        ),
    )


def stream_run(query: Query):
    """Run the graph and yield events as each step happens, ending with
    `{"type": "done", "answer": ...}`.

    The graph runs on a worker thread while this generator drains the event
    queue, because `graph.invoke` is a single blocking call -- there is no way
    to observe it from the inside. The worker closes its own DB connection: it
    gets a fresh one from Django's thread-local pool and nothing else will.
    """
    graph, log = build(
        research=get_research_tool(),
        case_store=DjangoCaseStore(),
        model=get_chat_model(),
        safety_floor=settings.CARMED_SAFETY_FLOOR,
    )
    events: queue.Queue = queue.Queue()
    _tap(log, events)

    outcome = {}

    def work():
        try:
            outcome["state"] = graph.invoke(State(query=query))
        except Exception as exc:  # surfaced to the caller after the drain
            outcome["error"] = exc
        finally:
            connection.close()
            events.put(None)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()

    while True:
        event = events.get()
        if event is None:
            break
        yield event
    worker.join()

    if "error" in outcome:
        raise outcome["error"]

    answer = _assemble_answer(State.model_validate(outcome["state"]), log)
    log_trace(query, answer)
    yield {"type": "done", "answer": answer.model_dump(mode="json")}


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
