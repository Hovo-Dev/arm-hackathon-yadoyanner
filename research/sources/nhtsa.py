"""NHTSA adapter — free, official, no API key.

Why this matters more than it looks: the complaints endpoint returns real
owner-reported symptoms alongside what the dealer actually diagnosed and
replaced. That is the "experienced mechanic who has seen this before" signal,
already written down a few hundred thousand times, and it is the honest
substitute for scraping paid OEM service manuals.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from ..schemas import Evidence, SourceKind, VehicleContext
from .base import get_json, truncate

log = logging.getLogger(__name__)

BASE = "https://api.nhtsa.gov"

# NHTSA indexes US-market names; Armenian roads run plenty of JDM/EU variants.
MODEL_ALIASES: dict[str, str] = {
    "x-trail": "rogue",
    "xtrail": "rogue",
    "note e-power": "versa",
    "tiida": "versa",
    "sylphy": "sentra",
    "teana": "altima",
    "fuga": "m37",
    "vezel": "hr-v",
    "fit": "fit",
    "odyssey": "odyssey",
}


def _normalise(make: str, model: str) -> tuple[str, str]:
    m = (model or "").strip().lower()
    return (make or "").strip().lower(), MODEL_ALIASES.get(m, m)


def _score(summary: str, symptoms: list[str], codes: list[str]) -> float:
    """Cheap lexical overlap. Good enough to rank before the LLM re-reads them."""
    text = (summary or "").lower()
    if not text:
        return 0.0
    hits = 0
    terms = [t.lower() for s in symptoms for t in re.findall(r"\w{4,}", s)]
    for term in set(terms):
        if term in text:
            hits += 1
    for code in codes:
        if code.lower() in text:
            hits += 3
    denom = max(len(set(terms)) + 3 * len(codes), 1)
    return min(hits / denom, 1.0)


def search_complaints(
    vehicle: VehicleContext,
    symptoms: list[str] | None = None,
    obd_codes: list[str] | None = None,
    limit: int = 6,
) -> list[Evidence]:
    """Owner complaints with dealer diagnoses for this vehicle."""
    if not vehicle.year:
        return []
    make, model = _normalise(vehicle.make, vehicle.model)
    data = get_json(
        f"{BASE}/complaints/complaintsByVehicle",
        params={"make": make, "model": model, "modelYear": vehicle.year},
    )
    if not data or not data.get("results"):
        return []

    symptoms = symptoms or []
    obd_codes = obd_codes or []
    scored = []
    for item in data["results"]:
        summary = item.get("summary") or ""
        scored.append((_score(summary, symptoms, obd_codes), item))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[Evidence] = []
    for score, item in scored[:limit]:
        comp = item.get("components") or "unspecified"
        out.append(
            Evidence(
                source_key="nhtsa",
                source_kind=SourceKind.OFFICIAL,
                title=f"NHTSA complaint {item.get('odiNumber')} — {comp}",
                snippet=truncate(item.get("summary") or ""),
                url=f"https://www.nhtsa.gov/vehicle/{vehicle.year}/{make}/{model}",
                relevance=round(score, 3),
                raw={
                    "components": comp,
                    "crash": item.get("crash"),
                    "fire": item.get("fire"),
                    "dateOfIncident": item.get("dateOfIncident"),
                },
            )
        )
    return out


def component_histogram(vehicle: VehicleContext, top: int = 8) -> list[tuple[str, int]]:
    """Which components this model actually fails on, by complaint volume.

    A strong prior: if a symptom is ambiguous, the component with 200 complaints
    beats the one with 2.
    """
    if not vehicle.year:
        return []
    make, model = _normalise(vehicle.make, vehicle.model)
    data = get_json(
        f"{BASE}/complaints/complaintsByVehicle",
        params={"make": make, "model": model, "modelYear": vehicle.year},
    )
    if not data or not data.get("results"):
        return []
    counter: Counter[str] = Counter()
    for item in data["results"]:
        for comp in (item.get("components") or "").split(","):
            comp = comp.strip()
            if comp:
                counter[comp] += 1
    return counter.most_common(top)


def search_recalls(vehicle: VehicleContext, limit: int = 5) -> list[Evidence]:
    """Open recalls — safety-critical, so these outrank everything else."""
    if not vehicle.year:
        return []
    make, model = _normalise(vehicle.make, vehicle.model)
    data = get_json(
        f"{BASE}/recalls/recallsByVehicle",
        params={"make": make, "model": model, "modelYear": vehicle.year},
    )
    if not data or not data.get("results"):
        return []
    out: list[Evidence] = []
    for item in data["results"][:limit]:
        out.append(
            Evidence(
                source_key="nhtsa",
                source_kind=SourceKind.OFFICIAL,
                title=f"RECALL {item.get('NHTSACampaignNumber')} — {item.get('Component')}",
                snippet=truncate(
                    f"{item.get('Summary', '')} REMEDY: {item.get('Remedy', '')}"
                ),
                url="https://www.nhtsa.gov/recalls",
                relevance=1.0,  # a matching recall is always material
                raw={"campaign": item.get("NHTSACampaignNumber")},
            )
        )
    return out
