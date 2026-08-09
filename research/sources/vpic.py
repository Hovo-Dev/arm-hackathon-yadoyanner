"""vPIC — NHTSA's vehicle product information catalogue. Free, official, no key.

This is the strongest anti-hallucination source available, because it replaces
inference with lookup. Everything here is a fact the government holds about a
manufactured vehicle, not a guess the model makes from a name:

  decode_vin("1N4AL3AP7FC...")  -> 2015 Nissan Altima, 2.5L, 4-cyl, FWD, sedan
  canonical_models("nissan", 2004) -> the exact model names that year

Two failure modes it removes:

  1. Engine guessing. A 2004 Altima shipped with a 2.5L QR25DE or a 3.5L VQ35DE.
     Those take different parts. Without the VIN the model either asks, guesses,
     or silently assumes — and a guessed engine produces confidently wrong part
     recommendations.

  2. Model-name drift. Users write "x-trail" for a car NHTSA indexes as "Rogue",
     or mistype outright. Checking against the real model list for that year
     catches it before the whole run is spent querying a car that does not exist.

Unlike the OBD bulk list this data is first-party from the regulator that
assigns it, which is why it is trusted directly rather than quarantined.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .. import cache
from ..schemas import Evidence, SourceKind, VehicleContext
from .base import get_json

log = logging.getLogger(__name__)

VPIC = "https://vpic.nhtsa.dot.gov/api/vehicles"

#: A VIN decode is a fact about a car that was already built.
_FOREVER = 100 * 365 * 24 * 3600

# Fields worth carrying into a diagnosis. vPIC returns ~140 per VIN, most of
# them empty or irrelevant to repair work.
_USEFUL = [
    ("Make", "make"),
    ("Model", "model"),
    ("ModelYear", "year"),
    ("Trim", "trim"),
    ("EngineModel", "engine_model"),
    ("DisplacementL", "displacement_l"),
    ("EngineCylinders", "cylinders"),
    ("EngineConfiguration", "engine_config"),
    ("FuelTypePrimary", "fuel"),
    ("TransmissionStyle", "transmission"),
    ("TransmissionSpeeds", "gears"),
    ("DriveType", "drive"),
    ("BodyClass", "body"),
    ("PlantCountry", "built_in"),
    ("Series", "series"),
]


def spec(vin: str) -> dict[str, str]:
    """VIN -> the useful factory fields, or ``{}`` when it does not decode.

    Cached on disk without expiry: what a factory built into a given VIN is
    not going to change. This matters because the decode sits in the request
    path -- the first lookup for a car costs a round trip, every later one
    costs a file read.

    Returns ``{}`` rather than raising, for the two ordinary cases: a car
    never sold in the US (vPIC is NHTSA's, so Opel, Skoda, Lada, Peugeot and
    Renault decode to nothing) and a VIN typed wrong.
    """
    vin = (vin or "").strip().upper()
    if len(vin) < 11:  # a partial VIN still decodes; shorter is not a VIN
        return {}

    cached = cache.get("vpic", {"vin": vin}, ttl_s=_FOREVER)
    if cached is not None:
        return cached

    data = get_json(f"{VPIC}/DecodeVinValues/{vin}", params={"format": "json"})
    if not data or not data.get("Results"):
        return {}  # a transient failure must not be cached as "no such car"

    row = data["Results"][0]
    fields: dict[str, str] = {}
    for src, dest in _USEFUL:
        value = (row.get(src) or "").strip()
        if value and value.lower() not in {"not applicable", "0"}:
            fields[dest] = value

    if not fields.get("make") or not fields.get("model"):
        code = row.get("ErrorText") or "VIN did not decode"
        log.info("VIN decode failed: %s", str(code)[:120])
        fields = {}

    # Cached either way: "this VIN does not decode" is a real answer from a
    # reachable service, and re-asking it on every request is pure waste.
    cache.put("vpic", {"vin": vin}, fields)
    return fields


def decode_vin(vin: str) -> tuple[VehicleContext | None, list[Evidence]]:
    """VIN -> exact factory specification.

    Returns (vehicle, evidence). The vehicle is authoritative and should
    override anything the user typed or the model inferred.
    """
    vin = (vin or "").strip().upper()
    fields = spec(vin)
    if not fields:
        return None, []

    year = None
    try:
        year = int(fields["year"]) if fields.get("year") else None
    except ValueError:
        pass

    engine_bits = [
        fields.get("displacement_l") and f"{fields['displacement_l']}L",
        fields.get("engine_model"),
        fields.get("cylinders") and f"{fields['cylinders']}-cyl",
    ]
    engine = " ".join(b for b in engine_bits if b) or None

    vehicle = VehicleContext(
        make=fields["make"].title(),
        model=fields["model"],
        year=year,
        engine=engine,
        transmission=fields.get("transmission"),
    )

    summary = ", ".join(f"{k}: {v}" for k, v in fields.items())
    evidence = [
        Evidence(
            source_key="vpic",
            source_kind=SourceKind.OFFICIAL,
            title=f"VIN {vin[:11]}… decoded — {vehicle.label()}",
            snippet=(
                f"Factory specification from the NHTSA vehicle catalogue: {summary}. "
                "These are recorded manufacturing facts; prefer them over any "
                "engine or trim inferred from the model name."
            ),
            url=f"https://vpic.nhtsa.dot.gov/decoder/Decoder?VIN={vin}",
            relevance=1.0,
            raw={"fields": fields},
        )
    ]
    return vehicle, evidence


@lru_cache(maxsize=64)
def canonical_models(make: str, year: int) -> tuple[str, ...]:
    """Real model names for a make in a given year, deduplicated."""
    data = get_json(
        f"{VPIC}/GetModelsForMakeYear/make/{make.strip().lower()}/modelyear/{year}",
        params={"format": "json"},
    )
    if not data or not data.get("Results"):
        return ()
    seen: dict[str, None] = {}
    for row in data["Results"]:
        name = (row.get("Model_Name") or "").strip()
        if name:
            seen.setdefault(name, None)
    return tuple(seen)


def check_vehicle(vehicle: VehicleContext) -> list[str]:
    """Warn when make/model/year do not match the official catalogue.

    Advisory, never blocking: vPIC indexes the US market, and plenty of cars on
    Armenian roads were never sold there. A miss means "could not confirm",
    not "does not exist".
    """
    if not vehicle.year or not vehicle.make:
        return []

    models = canonical_models(vehicle.make, vehicle.year)
    if not models:
        return []

    wanted = (vehicle.model or "").strip().lower()
    if any(wanted == m.lower() for m in models):
        return []

    near = [m for m in models if wanted and (wanted in m.lower() or m.lower() in wanted)]
    if near:
        return [
            f"'{vehicle.model}' is not an exact {vehicle.year} {vehicle.make} "
            f"model name; closest official match: {', '.join(near[:3])}"
        ]
    return [
        f"'{vehicle.model}' is not in the official {vehicle.year} "
        f"{vehicle.make} model list (it may be a non-US-market model)"
    ]
