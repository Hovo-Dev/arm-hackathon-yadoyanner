"""OBD-II diagnostic trouble code reference. Offline, no key, no rate limit.

PRECEDENCE — verified table first, ingested data second. This is not the
obvious way round, and there is a reason.

The widely-copied public DTC list (mytrile/obd-trouble-codes, and the verbatim
copy inside aws-solutions/aws-connected-vehicle-solution) is misaligned across
several ranges. Measured against SAE J2012, 11 of 21 spot-checked common codes
were wrong:

    P0420  ->  "Secondary Air Injection System Relay B"   (is: catalyst efficiency)
    P0300  ->  "Cylinder 12 Contribution/Range Fault"     (is: random misfire)
    P0700  ->  "Fuel Level Output Circuit Malfunction"    (is: transmission control)
    P0335  ->  "Knock Sensor 2 Circuit Intermittent"      (is: crankshaft position)

Those are among the most frequently seen codes in the world. A tool that tells
someone their misfire is a fuel-level circuit fault sends them to buy the wrong
part — the most expensive mistake this system could make. So:

  VERIFIED   — hand-checked against SAE J2012 / ISO 15031-6. Authoritative.
               Covers the codes that actually come up on high-mileage imports.
  INGESTED   — data/obd_codes.json, ~3,000 entries for breadth. Used ONLY for
               codes absent from VERIFIED, flagged unverified, lower relevance,
               and the snippet says so out loud.

scripts/validate_obd.py enforces this: any future dataset must pass the
known-good check before it can be trusted.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from ..config import ROOT
from ..schemas import Evidence, SourceKind

log = logging.getLogger(__name__)

CODE_RE = re.compile(r"\b([PBCU][0-3][0-9A-F]{3})\b", re.I)
DATA_PATH = ROOT / "data" / "obd_codes.json"


# --- Layer 1: verified definitions + field experience ----------------------
# (definition, common causes). Definition checked against SAE J2012.
VERIFIED: dict[str, tuple[str, str]] = {
    # Fuel & air metering
    "P0100": ("Mass or volume air flow circuit malfunction",
              "Contaminated MAF element, intake leak, wiring"),
    "P0101": ("Mass or volume air flow circuit range/performance",
              "Dirty MAF, post-filter intake leak, restricted air filter"),
    "P0102": ("Mass or volume air flow circuit low input",
              "Failed MAF, open circuit, heavily clogged filter"),
    "P0113": ("Intake air temperature sensor 1 circuit high",
              "Failed IAT sensor or open circuit"),
    "P0117": ("Engine coolant temperature circuit low",
              "Failed ECT sensor, shorted wiring"),
    "P0128": ("Coolant temperature below thermostat regulating temperature",
              "Thermostat stuck open, wrong-temperature thermostat fitted, failing coolant sensor"),
    "P0171": ("System too lean (Bank 1)",
              "Vacuum/intake leak, dirty MAF, weak fuel pump, leaking injector seals"),
    "P0172": ("System too rich (Bank 1)",
              "Leaking injector, failed MAF/MAP, bad coolant temp sensor, clogged air filter"),
    "P0174": ("System too lean (Bank 2)",
              "Vacuum/intake leak, dirty MAF, weak fuel pump (bank 2)"),
    "P0175": ("System too rich (Bank 2)", "As P0172, bank 2"),

    # Injectors
    "P0201": ("Injector circuit/open — cylinder 1", "Failed injector, connector, or driver circuit"),
    "P0202": ("Injector circuit/open — cylinder 2", "Failed injector, connector, or driver circuit"),
    "P0203": ("Injector circuit/open — cylinder 3", "Failed injector, connector, or driver circuit"),
    "P0204": ("Injector circuit/open — cylinder 4", "Failed injector, connector, or driver circuit"),

    # Ignition / misfire
    "P0300": ("Random/multiple cylinder misfire detected",
              "Worn plugs/coils, vacuum leak, low fuel pressure, carbon build-up"),
    "P0301": ("Cylinder 1 misfire detected",
              "Coil pack, spark plug, injector, or compression loss on that cylinder"),
    "P0302": ("Cylinder 2 misfire detected", "Coil, plug, injector, or compression loss"),
    "P0303": ("Cylinder 3 misfire detected", "Coil, plug, injector, or compression loss"),
    "P0304": ("Cylinder 4 misfire detected", "Coil, plug, injector, or compression loss"),
    "P0305": ("Cylinder 5 misfire detected", "Coil, plug, injector, or compression loss"),
    "P0306": ("Cylinder 6 misfire detected", "Coil, plug, injector, or compression loss"),
    "P0325": ("Knock sensor 1 circuit (Bank 1)", "Failed knock sensor, wiring, or connector corrosion"),

    # Timing / position sensors
    "P0011": ("Camshaft position — timing over-advanced (Bank 1)",
              "Low or dirty oil, failed VVT solenoid, stretched timing chain"),
    "P0016": ("Crankshaft/camshaft position correlation (Bank 1 Sensor A)",
              "Stretched timing chain, worn guides, failed VVT actuator — do not ignore"),
    "P0335": ("Crankshaft position sensor A circuit",
              "Failed crankshaft position sensor, damaged reluctor ring, wiring"),
    "P0340": ("Camshaft position sensor A circuit (Bank 1)",
              "Failed camshaft position sensor, wiring, or timing chain wear"),

    # Emissions
    "P0135": ("O2 sensor heater circuit (Bank 1, Sensor 1)",
              "Failed sensor heater element, blown fuse, corroded wiring/connector"),
    "P0136": ("O2 sensor circuit (Bank 1, Sensor 2)", "Aged downstream sensor, exhaust leak, wiring"),
    "P0401": ("Exhaust gas recirculation flow insufficient",
              "Carboned EGR valve or passages, failed DPFE/EGR position sensor"),
    "P0402": ("Exhaust gas recirculation flow excessive",
              "Stuck-open EGR valve, faulty EGR solenoid"),
    "P0420": ("Catalyst system efficiency below threshold (Bank 1)",
              "Aged catalytic converter — but very often a lazy downstream O2 sensor or an "
              "exhaust leak upstream of it. Confirm before replacing the converter"),
    "P0430": ("Catalyst system efficiency below threshold (Bank 2)",
              "As P0420, bank 2. Same warning about replacing the converter first"),
    "P0440": ("Evaporative emission control system malfunction",
              "Fuel cap seal, EVAP hoses, purge or vent valve"),
    "P0442": ("Evaporative emission system leak detected (small leak)",
              "Loose or perished fuel-cap seal, cracked EVAP hose, purge valve"),
    "P0446": ("Evaporative emission system vent control circuit",
              "Blocked or failed vent valve, wiring"),
    "P0455": ("Evaporative emission system leak detected (gross leak)",
              "Missing or failed fuel cap, disconnected EVAP hose"),

    # Cooling / speed / idle
    "P0217": ("Engine over-temperature condition",
              "Failed water pump, blocked radiator, or head gasket — stop driving"),
    "P0500": ("Vehicle speed sensor malfunction", "Failed VSS, wiring, or ABS ring damage"),
    "P0505": ("Idle air control system malfunction",
              "Carboned throttle body or IAC valve, vacuum leak"),

    # Transmission — CVTs are everywhere on the Armenian used fleet
    "P0700": ("Transmission control system malfunction (MIL request)",
              "Placeholder only — read TCM-specific codes. Common on worn CVTs"),
    "P0710": ("Transmission fluid temperature sensor circuit",
              "Failed sensor, degraded fluid, wiring"),
    "P0740": ("Torque converter clutch circuit malfunction",
              "Low or burnt ATF, failed torque-converter clutch solenoid, converter wear"),
    "P0741": ("Torque converter clutch stuck off / performance",
              "Degraded fluid, failed solenoid, valve body wear"),
    "P0750": ("Shift solenoid A malfunction", "Failed solenoid, valve body, low fluid"),
    "P0776": ("Pressure control solenoid B performance/stuck off",
              "Valve body wear, contaminated fluid — frequent on high-mileage CVTs"),

    # Communication
    "U0100": ("Lost communication with ECM/PCM A",
              "CAN bus wiring, corroded connector, failed module, low battery voltage"),
    "U0101": ("Lost communication with TCM", "CAN wiring, failed TCM, connector corrosion"),
}


@lru_cache(maxsize=1)
def _ingested() -> dict[str, str]:
    """Breadth layer. Unverified — see module docstring."""
    if not DATA_PATH.exists():
        log.info("%s not present; verified table only (%d codes).",
                 DATA_PATH, len(VERIFIED))
        return {}
    try:
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        return {k.upper(): v for k, v in (payload.get("codes") or {}).items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s (%s).", DATA_PATH, exc)
        return {}


def definition_count() -> tuple[int, int]:
    """(verified, ingested) — surfaced by `run_research.py --status`."""
    return len(VERIFIED), len(_ingested())


def extract_codes(text: str) -> list[str]:
    """Pull OBD codes out of free text — users paste them mid-sentence."""
    return sorted({m.upper() for m in CODE_RE.findall(text or "")})


def lookup(codes: list[str]) -> list[Evidence]:
    ingested = _ingested()
    out: list[Evidence] = []

    for raw in codes:
        code = raw.upper().strip()

        if code in VERIFIED:
            desc, causes = VERIFIED[code]
            out.append(
                Evidence(
                    source_key="obd",
                    source_kind=SourceKind.REFERENCE,
                    title=f"{code} — {desc}",
                    snippet=f"Standard definition: {desc}. Commonly caused by: {causes}.",
                    relevance=1.0,
                    raw={"code": code, "verified": True},
                )
            )
            continue

        if re.match(r"^[PBCU][1-3]", code):
            out.append(
                Evidence(
                    source_key="obd",
                    source_kind=SourceKind.REFERENCE,
                    title=f"{code} — manufacturer-specific code",
                    snippet=(
                        f"{code} falls in the manufacturer-specific range and is not part "
                        "of the standard set. Its meaning differs per marque and must be "
                        "confirmed with marque documentation or a scan tool, not assumed."
                    ),
                    relevance=0.6,
                    raw={"code": code, "verified": False},
                )
            )
            continue

        loose = ingested.get(code)
        if loose:
            out.append(
                Evidence(
                    source_key="obd",
                    source_kind=SourceKind.REFERENCE,
                    title=f"{code} — {loose} (UNVERIFIED)",
                    snippet=(
                        f"Bulk-list definition: {loose}. This code is not in our verified "
                        "table and comes from a public dataset known to contain misaligned "
                        "entries. Treat as a hint only and confirm with a scan tool before "
                        "buying any part."
                    ),
                    relevance=0.35,
                    raw={"code": code, "verified": False, "source": "bulk"},
                )
            )
        else:
            out.append(
                Evidence(
                    source_key="obd",
                    source_kind=SourceKind.REFERENCE,
                    title=f"{code} — not in reference",
                    snippet=f"No definition found for {code}. Treat as unverified; it may be mistyped.",
                    relevance=0.2,
                    raw={"code": code, "verified": False},
                )
            )
    return out
