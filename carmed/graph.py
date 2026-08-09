"""The graph: state, steps, routing, and the public ``run`` entry point.

Why a graph and not a straight pipeline: not every question needs every agent.
"Is it safe to drive?" needs a diagnosis and nothing else. "Where do I buy
front pads?" needs no diagnosis at all. A VIN that fails its checksum exits
before anything is spent.

Cost ceiling: 3 model calls (router, diagnostician, parts). A case-gate hit
removes the diagnostician; intent routing removes more. Everything else -- VIN
maths, language detection, symptom routing, case matching, shop lookup,
fitment checks, grounding -- is deterministic and free.

Every step records what it did into a ``RunLog``, which comes back on
``Answer.trace``. That is how you verify routing and cost without reading
terminal output.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from carmed import agents as ag
from carmed import text as txt
from carmed import vehicle as veh
from carmed.models import (
    SAFETY_CRITICAL,
    Answer,
    AnswerStatus,
    CaseMatch,
    Cause,
    Diagnosis,
    Intent,
    IntentDecision,
    MatchTier,
    PartListing,
    PartsResult,
    Query,
    Shop,
    SystemArea,
    Trace,
    Urgency,
    Vehicle,
)
from carmed.ports import CaseStore, ResearchTool

#: Cosine floor for reusing a stored answer instead of diagnosing afresh.
#: Cosine has no absolute meaning across embedding models, so recalibrate this
#: whenever the Django layer changes its embedder -- on held-out examples, not
#: on your evaluation set.
CACHE_HIT_THRESHOLD = 0.80

#: Only these may serve a cached answer. A make-only match can inform a
#: diagnosis but must never be returned as one.
CACHEABLE_TIERS = (MatchTier.EXACT, MatchTier.NEAR)

#: Which steps each intent actually needs. This table is the routing.
NEEDS: dict[Intent, frozenset[str]] = {
    Intent.DIAGNOSE: frozenset({"diagnose", "parts", "shops"}),
    Intent.PART_LOOKUP: frozenset({"parts"}),
    Intent.SHOP_LOOKUP: frozenset({"shops"}),
    Intent.SAFETY_CHECK: frozenset({"diagnose"}),
    # Nothing to research, so nothing runs. The empty set is the whole
    # mechanism: every edge predicate already asks what this intent needs.
    Intent.SMALL_TALK: frozenset(),
}


class State(BaseModel):
    query: Query
    vehicle: Vehicle = Field(default_factory=Vehicle)
    lang: str = "xx"
    system_area: SystemArea = SystemArea.UNKNOWN
    intent: Intent = Intent.DIAGNOSE
    named_parts: list[str] = Field(default_factory=list)

    matches: list[CaseMatch] = Field(default_factory=list)
    cache_hit: bool = False
    diagnosis: Diagnosis | None = None
    parts: PartsResult = Field(default_factory=PartsResult)
    shops: list[Shop] = Field(default_factory=list)
    listings: dict[str, PartListing] = Field(default_factory=dict)

    status: AnswerStatus = AnswerStatus.ANSWERED
    message: str = ""
    dropped_refs: int = 0


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _make_vehicle(log: ag.RunLog):
    def vehicle_step(state: State) -> dict:
        log.step("vehicle")
        car, problem = veh.resolve(state.query.vehicle)
        if problem:
            # Refuse before spending anything: a failed checksum means a
            # misread character, and a wrong VIN buys the wrong part.
            log.note(f"VIN rejected: {problem}")
            return {
                "vehicle": car,
                "status": AnswerStatus.REFUSED_BAD_VIN,
                "message": problem,
            }
        # The owner's own words first -- the symptom table is transliteration
        # aware, so it usually reads them directly. The normalized text is the
        # fallback rather than the primary because it is a translation and can
        # be wrong; but when a transliteration is simply missing from the table
        # (they are added as we meet them), plain English is what saves the
        # routing from UNKNOWN.
        area = txt.classify_area(state.query.asked)
        if area is SystemArea.UNKNOWN:
            area = txt.classify_area(state.query.text)
        lang = txt.detect_lang(state.query.asked).value
        log.note(f"car={car.describe()} region={car.region.value} lang={lang} area={area}")
        return {"vehicle": car, "lang": lang, "system_area": area}

    return vehicle_step


#: Matched against the whole message, in the four scripts people actually type
#: here. Anything longer than a bare greeting goes to the router, which can
#: read it; this only exists so "hello" costs nothing at all.
_GREETINGS = frozenset({
    "hi", "hello", "hey", "yo", "hi there", "hello there",
    "barev", "barev dzez", "barev dzez!", "voghjuyn", "ողջույն", "բարև",
    "բարև ձեզ", "privet", "привет", "здравствуйте", "salam",
    "ok", "okay", "k", "lav", "լավ", "хорошо",
    "thanks", "thank you", "thx", "merci", "մերսի", "shnorhakalutyun",
    "շնորհակալություն", "spasibo", "спасибо",
    "bye", "goodbye", "ցտեսություն", "пока",
})

_HEURISTICS = (
    (Intent.SAFETY_CHECK, ("safe to drive", "can i drive", "безопасно", "վտանգավոր")),
    (Intent.SHOP_LOOKUP, ("who fixes", "mechanic", "сервис", "мастер", "վարպետ")),
    (Intent.PART_LOOKUP, ("where to buy", "where can i buy", "купить", "գնել")),
)


def _make_route(model: Any, log: ag.RunLog):
    router = ag.build_router(model) if model is not None else None

    def route(state: State) -> dict:
        log.step("route")
        lowered = state.query.asked.casefold()

        # Whole-string, unlike the heuristics below, which match substrings:
        # "hi" occurs inside "this" and "which", and a greeting rule that fired
        # on those would silently swallow real questions. A message that is
        # nothing but a greeting is the only thing this may catch.
        if lowered.strip(" .,!?…\n\t") in _GREETINGS:
            log.note("intent=small_talk (greeting, no model call)")
            return {"intent": Intent.SMALL_TALK}

        for intent, needles in _HEURISTICS:
            if any(n in lowered for n in needles):
                log.note(f"intent={intent} (keyword, no model call)")
                return {"intent": intent}

        if router is None:
            log.note("intent=diagnose (no model configured)")
            return {"intent": Intent.DIAGNOSE}

        try:
            decision, _ = ag.invoke(
                router,
                prompt=state.query.asked,
                vehicle=state.vehicle,
                city=state.query.city,
                system_area=state.system_area,
                schema=IntentDecision,
            )
            log.llm("router")
        except Exception as exc:
            log.note(f"router failed, defaulting to diagnose: {exc}")
            return {"intent": Intent.DIAGNOSE}

        decision = decision or IntentDecision()
        log.note(f"intent={decision.intent} (router)")
        return {"intent": decision.intent, "named_parts": decision.named_parts}

    return route


def _make_gate(case_store: CaseStore, log: ag.RunLog):
    """The case gate -- deliberately not an agent.

    Embedding a sentence and ranking rows by distance is a database query.
    A model in front of it would add cost and latency without adding
    judgement, so this step just asks the store and applies a threshold.
    """

    def gate(state: State) -> dict:
        log.step("gate")
        try:
            # The one step that wants `text` and not `asked`: this is a vector
            # lookup against a case base written in English, Russian and
            # Armenian script, so it needs the normalized form. Measured, the
            # owner's raw Latin-script Armenian embeds to nothing useful -- it
            # ranked the wrong case first in 3 of 4 probes -- which is exactly
            # why the normalization exists, and exactly why nothing that
            # reasons should depend on it having come out right.
            matches = case_store.find_similar(
                text=state.query.text, vehicle=state.vehicle, limit=5
            )
        except Exception as exc:
            log.note(f"case store unavailable: {exc}")
            return {}

        log.lookup("case_store.find_similar", len(matches))
        matches = sorted(matches, key=lambda m: m.score, reverse=True)
        top = matches[0] if matches else None
        hit = bool(
            top
            and top.case.verified
            and top.tier in CACHEABLE_TIERS
            and top.score >= CACHE_HIT_THRESHOLD
        )
        detail = f"top={top.score:.2f} tier={top.tier}" if top else "no matches"
        log.note(f"cache {'HIT' if hit else 'miss'} ({detail})")
        return {"matches": matches, "cache_hit": hit}

    return gate


def _from_cases(matches: list[CaseMatch]) -> Diagnosis:
    """Build a diagnosis from stored fixes. No model call.

    Only cases close to the top match get through: the metadata filter alone
    would happily return every case for this car, and listing a brake fix as
    an alternative cause for a steering noise is nonsense wearing the costume
    of evidence.
    """
    floor = max(CACHE_HIT_THRESHOLD, matches[0].score - 0.08)
    causes, seen = [], set()
    for match in matches[:3]:
        case = match.case
        key = case.fix.casefold()
        if not case.fix or key in seen or (causes and match.score < floor):
            continue
        seen.add(key)
        causes.append(
            Cause(
                title=case.fix,
                explanation=(
                    f"Confirmed fix on a {case.vehicle.describe()} reporting a "
                    f"similar symptom ({match.tier} match, similarity "
                    f"{match.score:.2f})."
                ),
                system_area=case.system_area,
                likely_parts=case.part_names,
                evidence=[f"case:{case.id}"],
                confidence="strong" if match.score > 0.9 else "moderate",
            )
        )
    top = matches[0].case
    return Diagnosis(
        causes=causes,
        urgency=top.urgency,
        urgency_reason=f"Taken from confirmed case {top.id}.",
    )


def _case_lines(matches: list[CaseMatch]) -> str:
    return "\n".join(
        f"- case:{m.case.id} [{m.tier}] {m.case.vehicle.describe()} — "
        f"{m.case.symptom[:110]} → fix: {m.case.fix[:110]}"
        for m in matches[:5]
    )


def _make_diagnose(model: Any, tools: dict, log: ag.RunLog):
    agent = ag.build_diagnostician(model, tools["knowledge"]) if model else None

    def diagnose(state: State) -> dict:
        log.step("diagnose")

        if state.cache_hit and state.matches:
            log.note("reused stored cases -- no model call")
            return {"diagnosis": _from_cases(state.matches)}

        if agent is None:
            log.note("no model configured -- abstaining")
            return {"diagnosis": Diagnosis(abstain=True)}

        lines = [f"Car: {state.vehicle.describe()}"]
        if state.vehicle.region is not veh.SpecRegion.UNKNOWN:
            lines.append(f"Market: {state.vehicle.region.value}")
        if state.query.mileage_km:
            lines.append(f"Mileage: {state.query.mileage_km:,} km")
        lines.append(f"Problem ({state.lang}): {state.query.asked}")
        if state.matches:
            lines.append("\nSimilar past cases you may cite as evidence:")
            lines.append(_case_lines(state.matches))

        try:
            diagnosis, result = ag.invoke(
                agent,
                prompt="\n".join(lines),
                vehicle=state.vehicle,
                city=state.query.city,
                system_area=state.system_area,
                schema=Diagnosis,
                # The diagnostician is the longest call in the run. Each cause
                # goes out as it is written so the wait is spent reading.
                on_partial=lambda cause: log.partial("cause", cause),
                on_reset=lambda: log.partial("reset", {"of": "cause"}),
                partial_key="causes",
            )
            log.llm("diagnostician")
        except Exception as exc:
            log.note(f"diagnostician failed: {exc}")
            return {"diagnosis": Diagnosis(abstain=True)}

        if not ag.called_tool(result, "search_repair_knowledge"):
            log.note("WARNING: diagnostician skipped the knowledge search")
        if diagnosis is None:
            log.note("diagnostician returned unparseable output -- abstaining")
        return {"diagnosis": diagnosis or Diagnosis(abstain=True)}

    return diagnose


def _make_parts(model: Any, tools: dict, log: ag.RunLog):
    agent = ag.build_parts_explorer(model, tools["listings"]) if model else None

    def parts(state: State) -> dict:
        log.step("parts")
        causes = state.diagnosis.causes if state.diagnosis else []

        if agent is None:
            log.note("no model configured -- skipping parts")
            return {}
        if not (state.named_parts or causes):
            log.note("nothing to look up")
            return {}

        lines = [f"Car: {state.vehicle.describe()}"]
        if causes:
            lines.append("Diagnosis:")
            lines += [
                f"{i}. {c.title} (parts: {', '.join(c.likely_parts) or 'unspecified'})"
                for i, c in enumerate(causes, 1)
            ]
        if state.named_parts:
            lines.append(f"User asked for: {', '.join(state.named_parts)}")
        lines.append("\nFind what to buy, then group duplicate listings.")

        try:
            result_parts, result = ag.invoke(
                agent,
                prompt="\n".join(lines),
                vehicle=state.vehicle,
                city=state.query.city,
                system_area=state.system_area,
                schema=PartsResult,
                on_partial=lambda option: log.partial("part", option),
                on_reset=lambda: log.partial("reset", {"of": "part"}),
                partial_key="options",
            )
            log.llm("parts_explorer")
        except Exception as exc:
            log.note(f"parts explorer failed: {exc}")
            return {}

        result_parts = result_parts or PartsResult()
        listings = dict(log.listings)
        result_parts.fitment_warnings += veh.fitment_warnings(
            state.vehicle, list(listings.values())
        )
        if not ag.called_tool(result, "search_parts_listings"):
            log.note("WARNING: parts explorer skipped the listing search")
        if result_parts.fitment_warnings:
            log.note(f"{len(result_parts.fitment_warnings)} fitment warning(s)")
        return {"parts": result_parts, "listings": listings}

    return parts


def _make_shops(research: ResearchTool, log: ag.RunLog):
    """No model here on purpose.

    Matching a shop to (make, area, city) is a filter. Running it through a
    model would only create an opportunity to invent a phone number.
    """

    def shops(state: State) -> dict:
        log.step("shops")
        try:
            found = research.search_shops(
                make=state.vehicle.make,
                system_area=state.system_area,
                city=state.query.city,
                limit=5,
            )
        except Exception as exc:
            log.note(f"shop search failed: {exc}")
            return {}
        log.lookup("research.search_shops", len(found))
        return {"shops": found}

    return shops


def _make_finalize(log: ag.RunLog, safety_floor: bool):
    def finalize(state: State) -> dict:
        log.step("finalize")
        if state.status is AnswerStatus.REFUSED_BAD_VIN:
            return {}

        diagnosis = state.diagnosis
        update: dict = {}
        dropped = 0

        # Drop references that resolve to nothing, and count prose containing
        # a price/phone/URL the agent was told never to write.
        known = {f"case:{m.case.id}" for m in state.matches} | {
            f"doc:{d}" for d in log.knowledge
        }
        if diagnosis:
            clean = []
            for cause in diagnosis.causes:
                kept = [e for e in cause.evidence if e in known]
                dropped += len(cause.evidence) - len(kept)
                dropped += len(txt.find_leaks(f"{cause.title} {cause.explanation}"))
                patch: dict = {"evidence": kept}
                # The label is derived, not taken on trust. A cause with no
                # surviving reference is a standard diagnosis whatever the
                # model called it -- otherwise "basis" would be one more field
                # it could fill in optimistically, and the confidence cap that
                # hangs off it would never fire.
                if not kept:
                    patch["basis"] = "standard_diagnosis"
                    if cause.confidence == "strong":
                        patch["confidence"] = "moderate"
                clean.append(cause.model_copy(update=patch))
            diagnosis = diagnosis.model_copy(update={"causes": clean})
            update["diagnosis"] = diagnosis

        options = []
        for option in state.parts.options:
            good = [i for i in option.listing_ids if i in state.listings]
            dropped += len(option.listing_ids) - len(good)
            if good:
                options.append(
                    option.model_copy(
                        update={
                            "listing_ids": good,
                            "suspicious_listing_ids": [
                                i for i in option.suspicious_listing_ids
                                if i in state.listings
                            ],
                        }
                    )
                )
        update["parts"] = state.parts.model_copy(update={"options": options})
        update["dropped_refs"] = dropped

        if diagnosis and safety_floor:
            # Both forms, deliberately. A safety floor must never see *fewer*
            # areas than before: the raw text catches transliterated brake
            # words the English summary lost, the summary catches the ones the
            # table does not know yet, and missing "brakes" is the one failure
            # here that hurts someone.
            areas = (
                txt.all_areas(state.query.asked)
                | txt.all_areas(state.query.text)
                | {state.system_area}
            )
            critical = areas & SAFETY_CRITICAL
            if critical and diagnosis.urgency < Urgency.DO_NOT_DRIVE:
                names = ", ".join(sorted(a.value for a in critical))
                log.note(f"safety floor escalated urgency ({names})")
                update["diagnosis"] = diagnosis.model_copy(
                    update={
                        "urgency": Urgency.DO_NOT_DRIVE,
                        "urgency_reason": (
                            f"Escalated by rule: the description mentions {names}, "
                            f"which is safety-critical."
                        ),
                    }
                )

        if state.intent is Intent.SMALL_TALK:
            # Answered without a model call, because a greeting does not need
            # one and the round trip is the entire latency the user feels.
            update["status"] = AnswerStatus.ANSWERED
            update["message"] = (
                "Hi. Tell me what the car is doing -- the noise, when it "
                "happens, and what you were doing at the time."
            )
        elif diagnosis and diagnosis.clarifying_question:
            update["status"] = AnswerStatus.NEEDS_CLARIFICATION
            update["message"] = diagnosis.clarifying_question
        elif diagnosis and (diagnosis.abstain or not diagnosis.causes):
            if state.intent in (Intent.DIAGNOSE, Intent.SAFETY_CHECK):
                update["status"] = AnswerStatus.ABSTAINED
                update["message"] = (
                    "Not enough information to name a likely cause. A mechanic "
                    "should look at this."
                )

        if dropped:
            log.note(f"dropped {dropped} unresolvable reference(s)")
        log.note(f"status={update.get('status', state.status)}")
        return update

    return finalize


# ---------------------------------------------------------------------------
# Routing -- each reads a value the previous step computed
# ---------------------------------------------------------------------------


def _after_vehicle(state: State) -> str:
    return "finalize" if state.status is AnswerStatus.REFUSED_BAD_VIN else "route"


def _after_route(state: State) -> str:
    needs = NEEDS[state.intent]
    if not needs:
        return "finalize"
    if "diagnose" in needs:
        return "gate"
    return "parts" if "parts" in needs else "shops"


def _after_diagnose(state: State) -> str:
    d = state.diagnosis
    if d is None or d.abstain or d.clarifying_question:
        return "finalize"
    needs = NEEDS[state.intent]
    if "parts" in needs:
        return "parts"
    return "shops" if "shops" in needs else "finalize"


def _after_parts(state: State) -> str:
    return "shops" if "shops" in NEEDS[state.intent] else "finalize"


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------


def build(
    *,
    research: ResearchTool,
    case_store: CaseStore,
    model: Any = None,
    safety_floor: bool = False,
):
    """Compile the graph. Returns ``(compiled_graph, run_log)``.

    Built per request -- compiling costs microseconds, and a fresh log means
    two concurrent requests can never see each other's listings or traces.
    """
    log = ag.RunLog()
    tools = ag.build_tools(research, log)

    g: StateGraph = StateGraph(State)
    g.add_node("vehicle", _make_vehicle(log))
    g.add_node("route", _make_route(model, log))
    g.add_node("gate", _make_gate(case_store, log))
    g.add_node("diagnose", _make_diagnose(model, tools, log))
    g.add_node("parts", _make_parts(model, tools, log))
    g.add_node("shops", _make_shops(research, log))
    g.add_node("finalize", _make_finalize(log, safety_floor))

    g.add_edge(START, "vehicle")
    g.add_conditional_edges("vehicle", _after_vehicle, ["route", "finalize"])
    g.add_conditional_edges("route", _after_route, ["gate", "parts", "shops", "finalize"])
    g.add_edge("gate", "diagnose")
    g.add_conditional_edges("diagnose", _after_diagnose, ["parts", "shops", "finalize"])
    g.add_conditional_edges("parts", _after_parts, ["shops", "finalize"])
    g.add_edge("shops", "finalize")
    g.add_edge("finalize", END)

    return g.compile(), log


def run(
    query: Query,
    *,
    research: ResearchTool,
    case_store: CaseStore,
    model: Any = None,
    safety_floor: bool = False,
) -> Answer:
    """Run one request end to end.

    ``model`` is any LangChain chat model. Pass None to run with no model at
    all: the gate, shop lookup and VIN checks still work, and the agents
    degrade to abstaining.

    ``safety_floor=True`` forces "do not drive" whenever the text mentions
    brakes, steering, suspension or airbags, regardless of what the model
    said. Off by default -- the model decides urgency.

    The returned ``Answer.trace`` records which steps ran, which agents made a
    model call, and every lookup against your ports.
    """
    graph, log = build(
        research=research,
        case_store=case_store,
        model=model,
        safety_floor=safety_floor,
    )
    final = State.model_validate(graph.invoke(State(query=query)))

    diagnosis = final.diagnosis or Diagnosis()
    return Answer(
        status=final.status,
        vehicle=final.vehicle,
        intent=final.intent,
        causes=diagnosis.causes,
        urgency=diagnosis.urgency,
        urgency_reason=diagnosis.urgency_reason,
        repair_steps=diagnosis.repair_steps,
        parts=final.parts,
        shops=final.shops,
        listings=final.listings,
        message=final.message,
        from_cache=final.cache_hit,
        dropped_refs=final.dropped_refs,
        trace=Trace(
            steps=list(log.steps),
            llm_calls=list(log.llm_calls),
            lookups=list(log.lookups),
            notes=list(log.notes),
        ),
    )
