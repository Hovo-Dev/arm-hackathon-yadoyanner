"""Tool definitions the reasoning model may call, plus the dispatcher.

The allowlist lives here rather than in the system prompt on purpose: a prompt
instruction is a request, a dispatcher is a boundary. The model cannot reach a
source that has no entry in TOOL_IMPLS.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from .schemas import Evidence, VehicleContext
from .sources import nhtsa, obd, turnam, web
from .sources.base import ProviderUnavailable

log = logging.getLogger(__name__)

# Populated when a provider refuses for account reasons (credits, auth).
PROVIDER_FAILURES: set[str] = set()


# --- OpenAI-style tool schemas -------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_obd_codes",
            "description": (
                "Resolve OBD-II diagnostic trouble codes to their standard "
                "definitions and common causes. Offline and authoritative for "
                "generic P0/B0/C0/U0 codes. Call this first whenever the user "
                "supplied any code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Codes such as ['P0420', 'P0171'].",
                    }
                },
                "required": ["codes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_owner_complaints",
            "description": (
                "Search real owner-reported failures for this exact vehicle from "
                "the US regulator's database. Each result pairs a described "
                "symptom with what a dealer actually diagnosed and replaced. "
                "This is the strongest available proxy for mechanic experience — "
                "use it on nearly every diagnosis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short symptom phrases in English.",
                    },
                    "obd_codes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symptoms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_failure_priors",
            "description": (
                "Return which components this model-year is most complained "
                "about, ranked by volume. Use to break ties between competing "
                "causes: a component with many complaints is a better prior than "
                "one with few."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_recalls",
            "description": (
                "Check open safety recalls for this vehicle. Any match is "
                "material and must appear in the report — recall work is "
                "typically free at a dealer."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the live web, restricted to an allowlist of Armenian, "
                "Russian and European parts and automotive sites. Use for parts "
                "availability, current pricing and recent failure reports. "
                "Returns result titles and URLs — call scrape_page on a "
                "promising URL to read the actual listing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "deep": {
                        "type": "boolean",
                        "description": (
                            "Autonomous multi-step crawl. Slow and costly — only "
                            "for a genuinely novel problem."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_services",
            "description": (
                "Find workshops in Armenia that can carry out a repair, with "
                "name, address, phone number and customer rating. Use it once "
                "you know what needs doing, to answer 'who do I take it to'. "
                "Reads Armenian service directories directly, so it works "
                "regardless of query wording or language. "
                "Call it for jobs a workshop must do, not for checks the owner "
                "can perform themselves."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job": {
                        "type": "string",
                        "description": "The repair, e.g. 'radiator replacement'.",
                    },
                    "city": {
                        "type": "string",
                        "description": "Default Yerevan.",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Trade category to search. Pick the closest match "
                            "from the enum — you understand the symptom better "
                            "than keyword matching does. Omit it only if none "
                            "fit."
                        ),
                    },
                },
                "required": ["job"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": (
                "Read the full contents of one page as text — a parts listing, a "
                "forum thread, a catalogue entry. Use it to turn a URL from "
                "search_web into concrete detail: price, condition, seller, "
                "compatibility, part number. "
                "Only URLs on the allowlisted domains can be fetched; anything "
                "else is refused. Pass a URL you got from a previous tool "
                "result, never one you composed yourself, because a guessed URL "
                "is how fabricated listings get into a report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL, taken from an earlier search result.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


def available_tool_schemas(vehicle: VehicleContext | None = None,
                           obd_codes: list[str] | None = None,
                           ) -> list[dict[str, Any]]:
    """Only advertise tools that can actually do something right now.

    Offering `search_web` with no search key configured is worse than useless:
    the model burns rounds retrying it with reworded queries, paying for the
    context each time and learning nothing. If a capability is off, the tool
    simply is not on the menu.
    """
    from . import config  # local import keeps module import order simple

    caps = config.Capabilities.detect()
    disabled: set[str] = set()

    # Either provider can back search_web; the router in sources.web picks one.
    if not (caps.perplexity_fast or caps.firecrawl):
        disabled.update({"search_web", "find_services"})
    if not caps.firecrawl:
        disabled.add("scrape_page")

    # Most drivers have never plugged in a scanner, so a code is the exception
    # rather than the rule. Offering the lookup when none were supplied invites
    # the model to spend a round discovering there is nothing to look up — the
    # same waste as offering a search tool with no key.
    if not obd_codes:
        disabled.add("lookup_obd_codes")


    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] not in disabled]

    # Fill the find_services category enum from the live directory. The model
    # choosing from real options beats our keyword matching: `job` is the only
    # argument it writes freely, so the wording is unpredictable, and "braking
    # noise when slowing down" defeats a fixed keyword list while being obvious
    # to a reader. Copied per call so the shared TOOL_SCHEMAS stays clean.
    for schema in schemas:
        if schema["function"]["name"] != "find_services":
            continue
        try:
            live = turnam.vehicle_categories()
        except Exception as exc:  # noqa: BLE001 - never block on the directory
            log.warning("Could not load turn.am categories: %s", exc)
            break
        if live:
            fn = json.loads(json.dumps(schema["function"]))   # deep copy
            fn["parameters"]["properties"]["category"]["enum"] = live
            schemas[schemas.index(schema)] = {"type": "function", "function": fn}
        break

    return schemas


# --- dispatch -------------------------------------------------------------

def build_dispatcher(
    vehicle: VehicleContext,
) -> dict[str, Callable[..., list[Evidence]]]:
    """Bind tools to this run's vehicle so the model never passes it around."""

    def _lookup_obd_codes(codes: list[str] | None = None, **_: Any) -> list[Evidence]:
        return obd.lookup(codes or [])

    def _search_owner_complaints(
        symptoms: list[str] | None = None,
        obd_codes: list[str] | None = None,
        **_: Any,
    ) -> list[Evidence]:
        return nhtsa.search_complaints(vehicle, symptoms or [], obd_codes or [])

    def _get_failure_priors(**_: Any) -> list[Evidence]:
        hist = nhtsa.component_histogram(vehicle)
        if not hist:
            return []
        body = "; ".join(f"{comp}: {n} complaints" for comp, n in hist)
        from .schemas import SourceKind  # local import avoids a cycle at module load

        return [
            Evidence(
                source_key="nhtsa",
                source_kind=SourceKind.OFFICIAL,
                title=f"Most-complained components — {vehicle.label()}",
                snippet=body,
                url=None,
                relevance=0.75,
                raw={"histogram": hist},
            )
        ]

    def _check_recalls(**_: Any) -> list[Evidence]:
        return nhtsa.search_recalls(vehicle)

    def _search_web(query: str = "", deep: bool = False, **_: Any) -> list[Evidence]:
        scoped = f"{vehicle.label()} {query}".strip()
        return web.search_parts(scoped)

    def _find_services(job: str = "", city: str = "Yerevan",
                       category: str = "", **_: Any) -> list[Evidence]:
        if not job:
            return []
        # Directory listings first: they carry a phone number and a rating, and
        # they keep working when the search provider is out of credits.
        found = turnam.find_workshops(job, category=category or None)
        if len(found) < 3:
            found += web.find_services(vehicle.label(), job, city or "Yerevan")
        return found[:8]

    def _scrape_page(url: str = "", **_: Any) -> list[Evidence]:
        return web.firecrawl_scrape(url) if url else []

    return {
        "lookup_obd_codes": _lookup_obd_codes,
        "search_owner_complaints": _search_owner_complaints,
        "get_failure_priors": _get_failure_priors,
        "check_recalls": _check_recalls,
        "search_web": _search_web,
        "find_services": _find_services,
        "scrape_page": _scrape_page,
    }


def execute_tool_call(
    name: str,
    arguments: str | dict[str, Any],
    dispatcher: dict[str, Callable[..., list[Evidence]]],
) -> list[Evidence]:
    """Run one model-requested tool call. Never raises into the agent loop."""
    fn = dispatcher.get(name)
    if fn is None:
        log.warning("Model requested unknown tool %r — refusing", name)
        return []

    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            log.warning("Bad tool arguments for %s: %r", name, arguments[:200])
            args = {}
    else:
        args = arguments or {}

    try:
        return fn(**args)
    except ProviderUnavailable as exc:
        # Record on the module so the agent can warn the user. A dead provider
        # must not read as "this source had nothing to say".
        PROVIDER_FAILURES.add(str(exc))
        log.error("PROVIDER UNAVAILABLE during %s: %s", name, exc)
        return []
    except TypeError as exc:
        log.warning("Bad arguments for %s (%s) — %s", name, args, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("Tool %s failed: %s", name, exc)
        return []


def summarise_for_model(evidence: list[Evidence], offset: int) -> str:
    """Render tool output back to the model with stable citation indices.

    The indices are the contract: the model cites [3], and [3] resolves to a
    real retrieved item in the final report. That is what makes grounding
    checkable rather than aspirational.
    """
    if not evidence:
        return "No results. Do not infer anything from this absence."
    lines = []
    for i, ev in enumerate(evidence, start=offset):
        loc = f" <{ev.url}>" if ev.url else ""
        lines.append(f"[{i}] ({ev.source_key}) {ev.title}{loc}\n    {ev.snippet}")
    return "\n".join(lines)
