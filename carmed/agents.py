"""The LLM agents, and the research tools they call.

Two agents use a model: the diagnostician and the parts explorer. Each is a
LangGraph ReAct subgraph with exactly one tool, so it decides for itself
whether to search, with what wording, and whether to search again. A third
tiny agent classifies intent and has no tools.

Structured output note
----------------------
``create_react_agent(..., response_format=Schema)`` is implemented with
OpenAI's strict ``json_schema`` mode, which this gateway's DeepSeek deployment
rejects:

    This response_format type is unavailable now

Probing the provider directly showed what it does support: tool calling,
multi-turn tool round-trips, and ``{"type": "json_object"}`` -- including
tools and JSON mode in the same request. So the agents run in JSON mode with
the schema written into the prompt, and the final message is parsed here. Same
number of model calls, no strict-schema dependency.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypeVar

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState as ReactState
from pydantic import BaseModel, ValidationError

from carmed.models import (
    Diagnosis,
    IntentDecision,
    KnowledgeSnippet,
    PartListing,
    PartsResult,
    SystemArea,
    Vehicle,
)
from carmed.ports import ResearchTool

T = TypeVar("T", bound=BaseModel)

#: Ceiling on the ReAct loop. Prompts ask for at most two searches; this is
#: the enforcement, so a confused model cannot burn the budget. Prompts are
#: advisory -- models have been observed doing five searches against a stated
#: maximum of two.
RECURSION_LIMIT = 8

#: Ceiling on how many results are rendered into a prompt, whatever the
#: research tool returns. The ``limit`` argument is a request, not a promise:
#: an implementation that ignores it and returns five hundred listings would
#: otherwise put all five hundred into the model's context, on every loop.
MAX_RENDERED = 25


class AgentState(ReactState):
    """ReAct state plus read-only context for the tools.

    The vehicle rides on the state rather than in the tool signature, so the
    model supplies only the search terms and cannot misremember the year --
    and a wrong year means a wrong part.
    """

    vehicle: Vehicle
    city: str
    system_area: SystemArea


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """\
Classify what the user wants.

- diagnose: describes a symptom, cause unknown ("clicking when I turn left")
- part_lookup: already knows the part, wants to buy it ("front pads for E60")
- shop_lookup: wants a mechanic ("who fixes BMW gearboxes in Yerevan")
- safety_check: asks whether the car is drivable ("can I drive with this?")
- small_talk: not about the car at all -- a greeting, a thank-you, an aside

Symptom plus "is it safe" -> safety_check. Symptom plus "what do I buy" ->
diagnose. When the message describes the car in any way, choose diagnose; it
is the widest path. Reserve small_talk for messages that say nothing about a
car -- and never infer a symptom from an earlier message to avoid it.

Put any part the user named explicitly into named_parts, verbatim.
"""

DIAGNOSTICIAN_PROMPT = """\
You are a diagnostic mechanic for cars in Armenia. Most cars here are rebuilt
US-market vehicles, so assume US specification unless told otherwise.

Call `search_repair_knowledge` before answering. Search using the symptom in
your own words -- the car is already known, so do not put make, model or year
in the query. If the first search comes back thin, search once more with
different wording. Never search more than twice.

Then give at most three causes, most likely first.

Rules:
- Cite evidence as "case:<id>" or "doc:<id>", using only ids you were shown.
- Never write a price, a phone number, or a URL. You have not seen any.
- confidence is "strong", "moderate", "weak" or "unclear", based on how much
  retrieved evidence actually supports the cause. Do not invent percentages.
- urgency is an integer: 0 drive it, 1 fix this week, 2 fix now, 3 do not
  drive. Judge it from the symptom and what you retrieved. Brakes, steering,
  suspension and airbags carry real risk -- weigh that honestly.
- Set basis to "evidence" when retrieved records support the cause, and to
  "standard_diagnosis" when it is simply what this symptom usually means on
  this kind of car. Both are useful; mislabelling one as the other is not.
- The searches often return nothing, because the sources do not cover every
  market. That is not a reason to withhold the differential a mechanic would
  give from the symptom alone. Name those causes with basis
  "standard_diagnosis", no evidence ids, and confidence no higher than
  "moderate".
- Only abstain when the symptom itself is too vague to narrow at all -- not
  merely because the searches came back empty.
- If one missing fact would change the diagnosis, put it in
  clarifying_question. You may still list the causes you can already name.
"""

PARTS_PROMPT = """\
You match a diagnosis to the actual part to buy on the Armenian market.

Call `search_parts_listings`. Sellers write the same part in Armenian, Russian
and English with inconsistent spelling, so pass several forms in one call --
e.g. ["wheel bearing", "подшипник ступицы", "անվահեծի առանցքակալ"]. Search
again with different wording if the first pass is thin. Two searches maximum.

Your real job is grouping: one physical part appears under many titles.
Collapse them into a single option with the variants listed in aliases.

Rules:
- Reference listings only by the ids you were shown, in listing_ids.
- NEVER write a price, phone number or URL. Cite the id; the price is filled
  in from the record afterwards. A price you type is a fabrication.
- Put listings priced far below the others for the same part into
  suspicious_listing_ids -- on this market that usually means a copy.
- name_en is the plain English name of the part.
- If nothing matches, return an empty list. Do not invent a listing.
"""


# ---------------------------------------------------------------------------
# Research tools
# ---------------------------------------------------------------------------


class RunLog:
    """Everything that happened during one request.

    Two jobs. It holds the records the research tool returned, which are the
    rendering source of truth -- an id not in here was hallucinated, and the
    final answer drops it. And it records what ran, so a caller can verify
    behaviour and cost without reading the terminal output.

    One per request, so nothing leaks between concurrent callers.
    """

    def __init__(self) -> None:
        self.listings: dict[str, PartListing] = {}
        self.knowledge: dict[str, KnowledgeSnippet] = {}
        self.steps: list[str] = []
        self.llm_calls: list[str] = []
        self.lookups: list[str] = []
        self.notes: list[str] = []

    def step(self, name: str) -> None:
        self.steps.append(name)

    def llm(self, agent: str) -> None:
        """Record a model call. The length of this list is the bill."""
        self.llm_calls.append(agent)

    def lookup(self, name: str, count: int) -> None:
        self.lookups.append(f"{name}->{count}")

    def note(self, message: str) -> None:
        self.notes.append(message)

    def partial(self, kind: str, payload: dict) -> None:
        """A piece of the answer, emitted while the model is still writing.

        Not recorded on the log: this is a view concern. The finished object
        arrives in the normal way and is the source of truth -- a partial that
        never lands (a cut connection, a malformed tail) must not be able to
        change what the run produced. Callers that are not streaming pass no
        `on_partial` and nothing here is ever called.
        """


def _ctx(state: Any) -> tuple[Vehicle, SystemArea]:
    get = state.get if isinstance(state, dict) else lambda k, d=None: getattr(state, k, d)
    return get("vehicle") or Vehicle(), get("system_area") or SystemArea.UNKNOWN


def build_tools(research: ResearchTool, log: RunLog) -> dict[str, BaseTool]:
    """Wrap the research port as model-callable tools.

    Results are rendered to the model as compact lines tagged with ids, while
    the full records land in the registry.

    The rendered line does include the price -- the parts agent needs it to
    spot a listing priced far below the rest, which on this market usually
    means a copy. The safety is not that the model cannot see prices; it is
    that ``PartOption`` has no price field, so a generated one has nowhere to
    go. Displayed prices and URLs always come from the stored record.
    """

    @tool("search_repair_knowledge", parse_docstring=False)
    def search_repair_knowledge(
        query: str, state: Annotated[dict, InjectedState]
    ) -> str:
        """Search service manuals, catalogs and mechanic forum threads.

        Pass a focused symptom description, e.g. "clicking noise when turning
        left at low speed". The car is already known -- leave it out.
        """
        vehicle, _ = _ctx(state)
        try:
            found = research.search_knowledge(query=query, vehicle=vehicle, limit=6)
        except Exception as exc:
            # Reaches the model as a plain sentence, not a stack trace, so it
            # can try different wording instead of reasoning about our bug.
            log.note(f"    knowledge search FAILED: {exc}")
            return "The knowledge search is unavailable right now."

        log.lookup("research.search_knowledge", len(found))
        log.note(f"    searched knowledge for {query!r} -> {len(found)}")
        for snippet in found:
            log.knowledge[snippet.id] = snippet
            # Retrieved, not yet cited. The UI labels it that way: what the
            # answer ends up standing on is decided later, in finalize.
            log.partial("source", {
                "id": snippet.id,
                "title": snippet.title or snippet.source_url,
                "kind": snippet.kind,
                "url": snippet.source_url,
            })
        if not found:
            return "No sources found for that symptom. Try different wording."
        return "\n\n".join(
            f"[doc:{s.id}] ({s.kind}) {s.title}\n{s.text[:600]}"
            for s in found[:MAX_RENDERED]
        )

    @tool("search_parts_listings", parse_docstring=False)
    def search_parts_listings(
        part_terms: list[str], state: Annotated[dict, InjectedState]
    ) -> str:
        """Search parts for sale. Pass the part name in several languages at
        once; each spelling finds different listings. The car is already known.
        """
        vehicle, _ = _ctx(state)
        try:
            found = research.search_listings(
                part_terms=part_terms, vehicle=vehicle, limit=20
            )
        except Exception as exc:
            log.note(f"    listing search FAILED: {exc}")
            return "The parts search is unavailable right now."

        log.lookup("research.search_listings", len(found))
        log.note(f"    searched listings for {part_terms} -> {len(found)}")
        # Every returned record is kept, even beyond MAX_RENDERED: an id the
        # model never saw cannot be cited, but one it did must always resolve.
        for listing in found:
            log.listings[listing.id] = listing
        if not found:
            return "No listings found for those terms. Try other spellings."
        return "\n".join(
            f"[listing:{x.id}] {x.title} | {x.price or 'no price'} | "
            f"{x.condition or 'unknown condition'}"
            for x in found[:MAX_RENDERED]
        )

    return {"knowledge": search_repair_knowledge, "listings": search_parts_listings}


# ---------------------------------------------------------------------------
# Construction and invocation
# ---------------------------------------------------------------------------


#: A worked example beats a JSON schema for keeping field names right.
#: Nested models are the failure mode -- given only a schema, the model
#: reliably invents its own key for a nested object.
EXAMPLES: dict[str, str] = {
    "router": json.dumps(
        {"intent": "diagnose", "reason": "describes a symptom", "named_parts": []},
        ensure_ascii=False,
    ),
    "diagnostician": json.dumps(
        {
            "causes": [
                {
                    "title": "Brake pad wear indicator touching the rotor",
                    "explanation": "A high squeal on first braking matches the "
                    "wear indicator design described in the manual.",
                    "system_area": "brakes",
                    "likely_parts": ["brake pads"],
                    "evidence": ["doc:D3"],
                    "confidence": "moderate",
                }
            ],
            "urgency": 2,
            "urgency_reason": "Braking is safety critical and the pads are at "
            "the indicator.",
            "repair_steps": ["Measure pad thickness", "Replace pads below 3 mm"],
            "abstain": False,
            "clarifying_question": None,
        },
        ensure_ascii=False,
    ),
    "parts_explorer": json.dumps(
        {
            "options": [
                {
                    "name_en": "CV joint",
                    "aliases": ["ШРУС", "կիսասռնու հոդակապ"],
                    "part_numbers": ["43430-06340"],
                    "listing_ids": ["L1", "L2"],
                    "suspicious_listing_ids": ["L4"],
                }
            ],
            "fitment_warnings": [],
        },
        ensure_ascii=False,
    ),
}


def _schema_hint(schema: type[BaseModel], example: str) -> str:
    """JSON mode guarantees valid JSON, not JSON of the right shape.

    ``$defs`` must stay: nested models are referenced by ``$ref``, and
    dropping the definitions leaves the model guessing what a ``Cause``
    contains -- which it then does, with its own field names.
    """
    return (
        "\n\nReply with exactly one JSON object -- no prose, no code fence.\n"
        "Use these exact field names. Example of a valid reply:\n"
        + example
        + "\n\nFull schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def _make(model: Any, tools: list[BaseTool], prompt: str, schema: type[BaseModel],
          name: str, reasoning: bool = True):
    """Wire one agent.

    `reasoning=False` turns off the model's chain-of-thought for agents whose
    job does not need it. deepseek-v4-pro emits its deliberation before
    answering and the caller waits for all of it: measured, that is ~16s for
    the router to choose one of four intents, against ~2s without. The
    diagnostician keeps it, because weighing which cause the evidence supports
    is exactly the work reasoning is for.
    """
    bind: dict[str, Any] = {"response_format": {"type": "json_object"}}
    if not reasoning:
        bind["extra_body"] = {"reasoning": {"enabled": False}}
    try:
        model = model.bind(**bind)
    except Exception:
        pass
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt + _schema_hint(schema, EXAMPLES[name]),
        state_schema=AgentState,
        name=name,
    )


def build_router(model: Any):
    # Picking one of four labels from a sentence. No deliberation required.
    return _make(model, [], ROUTER_PROMPT, IntentDecision, "router", reasoning=False)


def build_diagnostician(model: Any, knowledge_tool: BaseTool):
    return _make(model, [knowledge_tool], DIAGNOSTICIAN_PROMPT, Diagnosis, "diagnostician")


def build_parts_explorer(model: Any, listings_tool: BaseTool):
    return _make(model, [listings_tool], PARTS_PROMPT, PartsResult, "parts_explorer")


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class _ArrayScanner:
    """Yields complete JSON objects out of a named array as text arrives.

    The model writes one object top to bottom, so waiting for the closing
    brace means waiting for the entire answer -- but each element of
    ``causes`` is finished long before that. This walks the buffer once,
    tracking string state and brace depth, and hands back every element the
    moment it closes.

    Malformed input simply produces nothing. The parsed final message stays
    the source of truth, so a scanner that loses its place costs the user a
    progressive render, never a wrong answer.
    """

    def __init__(self, key: str) -> None:
        self._key = key
        self._buf = ""
        self._pos = 0
        self._start: int | None = None
        self._depth = 0
        self._in_str = False
        self._escaped = False
        self._found = False
        self._done = False

    def feed(self, chunk: str) -> list[dict]:
        self._buf += chunk
        if self._done:
            return []

        if not self._found:
            marker = f'"{self._key}"'
            at = self._buf.find(marker)
            if at == -1:
                return []
            bracket = self._buf.find("[", at + len(marker))
            if bracket == -1:
                return []
            self._found, self._pos = True, bracket + 1

        out: list[dict] = []
        while self._pos < len(self._buf):
            char = self._buf[self._pos]
            self._pos += 1
            if self._in_str:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_str = False
            elif char == '"':
                self._in_str = True
            elif char == "{":
                if self._depth == 0:
                    self._start = self._pos - 1
                self._depth += 1
            elif char == "}":
                self._depth -= 1
                if self._depth == 0 and self._start is not None:
                    raw, self._start = self._buf[self._start : self._pos], None
                    try:
                        out.append(json.loads(raw))
                    except ValueError:
                        pass
            elif char == "]" and self._depth == 0:
                self._done = True
                break
        return out


def _stream(agent, payload: dict, config: dict, on_partial, on_reset, key: str) -> dict:
    """Run the agent in streaming mode, emitting array elements as they close.

    ``values`` carries the graph state after each step and ``messages`` the
    token deltas; asking for both means the final state is still returned
    exactly as ``invoke`` would have returned it, and the streaming is purely
    additive.
    """
    scanner = _ArrayScanner(key)
    result: dict = {}
    for mode, chunk in agent.stream(
        payload, config=config, stream_mode=["values", "messages"]
    ):
        if mode == "values":
            result = chunk
            continue
        message = chunk[0] if isinstance(chunk, tuple) else chunk
        if isinstance(message, ToolMessage):
            # A tool result sends the model back to writing from the top, so
            # anything half-scanned belongs to an abandoned attempt -- and so
            # does anything already emitted from it. Dropping the scanner is
            # not enough on its own: the caller has shown those elements, and
            # the model does abandon a full set of causes and come back with a
            # clarifying question instead. Telling it to discard is what keeps
            # the display honest.
            scanner = _ArrayScanner(key)
            if on_reset is not None:
                try:
                    on_reset()
                except Exception:  # noqa: BLE001 - a display sink must not fail a run
                    pass
            continue
        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text:
            continue
        for element in scanner.feed(text):
            try:
                on_partial(element)
            except Exception:  # noqa: BLE001 - a display sink must not fail a run
                pass
    return result


def invoke(agent, *, prompt: str, vehicle: Vehicle, city: str,
           system_area: SystemArea, schema: type[T],
           on_partial=None, on_reset=None,
           partial_key: str | None = None) -> tuple[T | None, dict]:
    """Run an agent and parse its final message.

    Returns ``(None, result)`` when the model produced nothing usable. Callers
    degrade rather than raise -- a malformed response should abstain, not take
    down the request.

    Pass ``on_partial`` with the name of an array field to have each element
    handed over as the model finishes writing it. The parsed return value is
    unchanged either way: partials are a preview, not a second result path.
    """
    payload = {
        "messages": [HumanMessage(content=prompt)],
        "vehicle": vehicle,
        "city": city,
        "system_area": system_area,
    }
    config = {"recursion_limit": RECURSION_LIMIT}
    if on_partial is not None and partial_key:
        result = _stream(agent, payload, config, on_partial, on_reset, partial_key)
    else:
        result = agent.invoke(payload, config=config)
    for message in reversed(result.get("messages") or []):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            try:
                return schema.model_validate_json(_extract_json(content)), result
            except (ValidationError, ValueError):
                return None, result
    return None, result


def called_tool(result: dict, name: str) -> bool:
    return any(
        isinstance(m, ToolMessage) and m.name == name
        for m in result.get("messages", [])
    )
