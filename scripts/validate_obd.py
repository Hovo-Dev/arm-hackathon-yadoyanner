#!/usr/bin/env python3
"""Guard against corrupt OBD code data.

This exists because the most widely-copied public DTC dataset is wrong. It is
mirrored verbatim in aws-solutions/aws-connected-vehicle-solution, so "lots of
projects use it" is not evidence of correctness — it is evidence that nobody
checked. 11 of 21 spot-checked common codes were misaligned.

Run this against any dataset before trusting it:

    python scripts/validate_obd.py                  # check what is wired now
    python scripts/validate_obd.py --file X.json    # vet a candidate dataset

Exit code 0 = passed, 1 = failed. Wire it into CI so a future "let's just pull
a bigger code list" cannot silently poison diagnoses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.sources.obd import VERIFIED, lookup  # noqa: E402

# Ground truth: substrings that MUST appear in a correct definition.
# Sourced from SAE J2012 / ISO 15031-6 and cross-checked by hand.
GROUND_TRUTH: dict[str, list[str]] = {
    "P0100": ["air flow"],
    "P0128": ["thermostat"],
    "P0135": ["heater"],
    "P0171": ["lean"],
    "P0172": ["rich"],
    "P0174": ["lean"],
    "P0300": ["misfire"],
    "P0301": ["misfire"],
    "P0325": ["knock"],
    "P0335": ["crankshaft"],
    "P0340": ["camshaft"],
    "P0401": ["recirculation", "egr"],
    "P0420": ["catalyst"],
    "P0430": ["catalyst"],
    "P0442": ["evaporative", "evap"],
    "P0455": ["evaporative", "evap"],
    "P0500": ["speed"],
    "P0700": ["transmission"],
    "P0740": ["torque converter"],
    "U0100": ["communication"],
}


def check_mapping(get_desc, label: str) -> int:
    """Returns the number of failures."""
    failures: list[tuple[str, str, str | None]] = []
    missing = 0

    for code, accepted in GROUND_TRUTH.items():
        desc = get_desc(code)
        if desc is None:
            missing += 1
            continue
        if not any(token in desc.lower() for token in accepted):
            failures.append((code, " / ".join(accepted), desc))

    checked = len(GROUND_TRUTH) - missing
    print(f"\n{label}")
    print(f"  checked {checked}/{len(GROUND_TRUTH)} codes"
          + (f" ({missing} absent)" if missing else ""))

    if failures:
        print(f"  FAILED {len(failures)}:")
        for code, expected, got in failures:
            print(f"    {code}: expected to mention '{expected}'")
            print(f"           got: {got!r}")
    else:
        print("  all present codes correct")

    return len(failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path,
                    help="Vet a candidate dataset JSON ({'codes': {...}}) instead")
    args = ap.parse_args()

    if args.file:
        payload = json.loads(args.file.read_text(encoding="utf-8"))
        codes = {k.upper(): v for k, v in (payload.get("codes") or payload).items()}
        failures = check_mapping(lambda c: codes.get(c), f"CANDIDATE: {args.file}")
        verdict = "TRUSTED" if failures == 0 else "REJECTED"
        print(f"\n  verdict: {verdict}")
        return 0 if failures == 0 else 1

    # 1. The verified table is the thing diagnoses actually rest on.
    failures = check_mapping(
        lambda c: VERIFIED[c][0] if c in VERIFIED else None,
        "VERIFIED table (authoritative)",
    )

    # 2. End-to-end: what the agent would really be handed.
    def via_lookup(code: str) -> str | None:
        ev = lookup([code])
        return ev[0].title if ev else None

    failures += check_mapping(via_lookup, "lookup() end-to-end")

    print()
    if failures:
        print(f"FAIL — {failures} incorrect definition(s). Do not ship this.")
        return 1
    print("PASS — every ground-truth code resolves correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
