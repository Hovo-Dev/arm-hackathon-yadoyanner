#!/usr/bin/env python3
"""Ingest the public OBD-II trouble-code list into data/obd_codes.json.

Source: https://github.com/mytrile/obd-trouble-codes (MIT, © 2014 Dimitar Kostov)
~3,000 standard DTC definitions — replaces the small hand-written table that
was only ever meant as a bootstrap.

    python scripts/ingest_obd.py            # refresh data/obd_codes.json
    python scripts/ingest_obd.py --check    # report status, write nothing

The curated "common causes" overlay in research/sources/obd.py is layered on
top of this at lookup time. The two are deliberately separate: definitions are
public reference data, the cause hints are our own judgement about what
actually fails on high-mileage imports.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "obd_codes.json"
URL = (
    "https://raw.githubusercontent.com/mytrile/obd-trouble-codes/"
    "master/obd-trouble-codes.csv"
)
ATTRIBUTION = {
    "source": "https://github.com/mytrile/obd-trouble-codes",
    "license": "MIT",
    "copyright": "© 2014 Dimitar Kostov",
}


def fetch() -> dict[str, str]:
    resp = httpx.get(URL, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    codes: dict[str, str] = {}
    for row in csv.reader(io.StringIO(resp.text)):
        if len(row) < 2:
            continue
        code, desc = row[0].strip().upper(), row[1].strip()
        if code and desc:
            codes[code] = desc
    return codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        if not OUT.exists():
            print(f"missing: {OUT.relative_to(ROOT)} — run this script to create it")
            return 1
        payload = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"{len(payload.get('codes', {}))} codes in {OUT.relative_to(ROOT)}")
        return 0

    print(f"Fetching {URL} ...")
    try:
        codes = fetch()
    except Exception as exc:  # noqa: BLE001
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1

    if len(codes) < 500:
        print(f"Only {len(codes)} codes parsed — refusing to overwrite. "
              "Upstream format may have changed.", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"attribution": ATTRIBUTION, "codes": codes},
                   indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(codes)} codes -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
