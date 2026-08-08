"""Tool-facing surface for the research agent.

This module exists so the research component can be registered as a tool inside
a larger agentic system. `research()` in agent.py is the right API for calling
Python directly; it is the wrong API for a tool, for three reasons:

  1. It takes a `VehicleContext` object. A tool schema can only describe flat
     JSON, so the caller must be able to pass make/model/year as plain fields.
  2. It returns a `ResearchReport` object of ~7,000 tokens. A parent agent gets
     tool results injected straight into its context; two calls would consume
     more context than the whole conversation.
  3. It raises `LLMError`. An exception crossing a tool boundary kills the
     parent's loop. A tool reports failure in its return value.

So: flat parameters in, compact JSON out, never raises.

    from research.tool_api import RESEARCH_TOOL_SCHEMA, diagnose_vehicle
    ...
    result = diagnose_vehicle(**json.loads(tool_call.function.arguments))
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from .agent import research
from .schemas import ResearchReport, VehicleContext
from .sources import vpic

log = logging.getLogger(__name__)


# Drop this straight into any OpenAI-compatible agent's `tools` list.
RESEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "diagnose_vehicle",
        "description": (
            "Diagnose a car problem for a driver in Armenia. Researches the "
            "fault across regulator complaint data, OBD-II code references, "
            "owner's manuals and Armenian/regional parts listings, then returns "
            "ranked likely causes, parts to price, and next steps — every claim "
            "backed by a cited source.\n\n"
            "Takes 60-120 seconds and costs roughly $0.01 per call, so ask once "
            "with the full problem description rather than repeatedly with "
            "fragments. Vehicle make and model are required; supplying the year "
            "materially improves accuracy because complaint data, recalls and "
            "part fitment are all year-specific."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The problem in the driver's own words. Armenian, "
                        "Russian or English. Include everything observed: when "
                        "it happens, what it sounds or smells like, what makes "
                        "it worse. Do not put the vehicle here — use the "
                        "dedicated fields."
                    ),
                },
                "make": {"type": "string", "description": "e.g. 'Nissan'"},
                "model": {"type": "string", "description": "e.g. 'Altima'"},
                "year": {"type": "integer", "description": "Model year, e.g. 2004"},
                "engine": {"type": "string", "description": "e.g. '2.5L QR25DE'"},
                "mileage_km": {"type": "integer"},
                "vin": {
                    "type": "string",
                    "description": (
                        "Vehicle Identification Number, if the driver can read "
                        "it off the windscreen or door frame. Strongly "
                        "preferred: it resolves the exact engine, trim and "
                        "drivetrain from the official vehicle catalogue instead "
                        "of inferring them from the model name, and a wrong "
                        "engine means wrong parts."
                    ),
                },
                "obd_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Scan-tool codes, e.g. ['P0420'].",
                },
                "photo_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Descriptions of what user photos show, produced by a "
                        "vision model. This agent is text-only and cannot read "
                        "images itself."
                    ),
                },
                "detail": {
                    "type": "string",
                    "enum": ["compact", "full"],
                    "description": (
                        "'compact' (default) returns the answer and cited "
                        "sources, roughly 1,000 tokens. 'full' adds every piece "
                        "of evidence gathered and is around 7,000 tokens — only "
                        "request it when you need to audit the reasoning."
                    ),
                },
            },
            "required": ["query", "make", "model"],
        },
    },
}


def _compact(report: ResearchReport) -> dict[str, Any]:
    """Answer-shaped result: what a parent agent needs to act or reply.

    Deliberately omits the evidence array. The parent gets citation indices and
    source labels, which is enough to attribute a claim or ask for `detail:
    full`; it does not need 30 complaint bodies to decide what to do next.
    """
    fitment_issues = [
        issue for part in report.parts for issue in part.claim_issues
    ]
    return {
        "ok": True,
        "vehicle": report.problem.vehicle.label(),
        "understood_symptoms": report.problem.symptoms,
        "obd_codes": report.problem.obd_codes,
        "summary": report.summary,
        "causes": [
            {
                "rank": i,
                "title": c.title,
                "confidence": round(c.confidence, 2),
                "severity": c.severity,
                "basis": c.basis,
                "why": c.explanation,
                "sources": c.evidence_ids,
            }
            for i, c in enumerate(report.causes, 1)
        ],
        "parts": [
            {
                "name": p.name,
                "part_number": p.part_number,
                "type": p.oem_or_aftermarket,
                "fitment": p.fitment_status,
                "notes": p.notes,
                "sources": p.evidence_ids,
            }
            for p in report.parts
        ],
        "next_steps": [
            {"step": s.step, "why": s.why,
             "who": "workshop" if s.requires_shop else "owner"}
            for s in report.next_steps
        ],
        "sources": [
            {"n": c.index, "label": c.label[:120], "from": c.source, "url": c.url}
            for c in report.citations()
        ],
        "quality": {
            "grounded": report.grounded(),
            "fitment_issues": fitment_issues,
            "warnings": report.warnings,
        },
        "cost": {
            "rounds": report.rounds_used,
            "tool_calls": len(report.tool_calls),
            "duration_s": report.duration_s,
            "usd": report.est_cost_usd,
        },
        # Downstream routing hints — consumed by the service-finder and
        # parts-availability agents rather than shown to the user.
        "handoff": report.to_handoff(),
    }


def diagnose_vehicle(
    query: str,
    make: str,
    model: str,
    year: int | None = None,
    engine: str | None = None,
    mileage_km: int | None = None,
    obd_codes: list[str] | None = None,
    photo_findings: list[str] | None = None,
    vin: str | None = None,
    detail: Literal["compact", "full"] = "compact",
    max_rounds: int | None = None,
) -> dict[str, Any]:
    """Tool entry point. Flat JSON in, JSON-serializable out, never raises."""
    if not query or not query.strip():
        return {"ok": False, "error": "query is empty",
                "hint": "Describe the symptom in the driver\'s own words."}

    # A decoded VIN is factory record, not inference — it wins over typed
    # fields, including a make/model the caller got wrong.
    decoded = None
    if vin:
        decoded, _ = vpic.decode_vin(vin)
        if decoded:
            make, model = decoded.make, decoded.model

    if not make or not model:
        return {"ok": False, "error": "make and model are required (or a valid VIN)",
                "hint": "Ask the user for the vehicle before calling this tool."}

    # Codes supplied as a parameter are appended to the text so the intake step
    # sees them even when the driver never wrote them out.
    text = query.strip()
    if obd_codes:
        text += "\nScan-tool codes reported: " + ", ".join(obd_codes)

    vehicle = VehicleContext(
        make=make,
        model=model,
        year=(decoded.year if decoded and decoded.year else year),
        engine=(decoded.engine if decoded and decoded.engine else engine),
        transmission=(decoded.transmission if decoded else None),
        mileage_km=mileage_km,
        vin=vin,
    )

    try:
        report = research(
            text,
            vehicle_hint=vehicle,
            max_rounds=max_rounds,
            image_findings=photo_findings or None,
        )
    except Exception as exc:  # noqa: BLE001 - must not escape into the parent loop
        log.exception("diagnose_vehicle failed")
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "The research run failed. Retrying once is reasonable; "
                    "if it fails again, answer from general knowledge and say "
                    "the diagnosis is unverified.",
        }

    result = _compact(report)
    if decoded:
        result["vin_decoded"] = {
            "source": "NHTSA vPIC (official)",
            "resolved": decoded.label(),
            "note": "Engine and trim taken from the factory record, not inferred.",
        }
    if detail == "full":
        result["evidence"] = [
            {"n": i, "source": e.source_key, "title": e.title,
             "url": e.url, "text": e.snippet}
            for i, e in enumerate(report.evidence)
        ]
    return result


# Convenience for frameworks that want name -> callable.
TOOL_REGISTRY = {"diagnose_vehicle": diagnose_vehicle}

__all__ = ["RESEARCH_TOOL_SCHEMA", "diagnose_vehicle", "TOOL_REGISTRY"]
