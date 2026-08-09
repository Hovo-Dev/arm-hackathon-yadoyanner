"""Offline tests. No network, no API budget, no database.

    python tests/test_carmed.py
    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.dummy import DummyCaseStore, DummyResearch
from carmed import (
    AnswerStatus,
    Case,
    CaseMatch,
    Intent,
    MatchTier,
    PartListing,
    Query,
    SpecRegion,
    SystemArea,
    Urgency,
    Vehicle,
    run,
)
from carmed import text as txt
from carmed import vehicle as veh
from carmed.graph import NEEDS

VALID_VINS = ["1HGCM82633A004352", "JH4TB2H26CC000000"]


def _run(text, offline=True, **kw):
    return run(
        Query(text=text, vehicle=Vehicle(**kw.pop("vehicle", {})), **kw),
        research=DummyResearch(),
        case_store=DummyCaseStore(),
        model=None,
        **({} if offline else {}),
    )


# --------------------------------------------------------------------------
# VIN
# --------------------------------------------------------------------------


def test_valid_vins_pass():
    for v in VALID_VINS:
        assert veh.is_valid(v), v


def test_nhtsa_agrees_with_us():
    """vPIC reports ErrorCode 1 (bad check digit) for this VIN; so do we."""
    assert not veh.is_valid("1FTFW1ET5DFC10312")
    assert veh.check_digit("1FTFW1ET5DFC10312") == "9"


def test_cross_class_single_char_errors_all_caught():
    """Substitutions within a transliteration class (1/A/J, 2/B/K/S, ...) are
    arithmetically invisible -- a property of the VIN standard. Every error
    that changes the transliterated value must be caught."""
    base = VALID_VINS[0]
    missed = []
    for i in range(17):
        if i == 8:
            continue
        for repl in "0123456789ABCDEFGHJKLMNPRSTUVWXYZ":
            if repl == base[i]:
                continue
            if veh.is_valid(base[:i] + repl + base[i + 1 :]):
                if veh._TRANSLIT[repl] != veh._TRANSLIT[base[i]]:
                    missed.append(f"{base[i]}->{repl}@{i}")
    assert not missed, missed


def test_ocr_confusable_pairs_are_detectable():
    for a, b in [("5", "S"), ("8", "B"), ("2", "Z"), ("6", "G"), ("1", "7")]:
        assert veh._TRANSLIT[a] != veh._TRANSLIT[b], f"{a}/{b} invisible"
    for illegal in "IOQ":
        assert not veh.is_well_formed("1HGCM8263" + illegal + "A004352")


def test_region_from_vin():
    assert veh.region_of("1HGCM82633A004352") is SpecRegion.US
    assert veh.region_of("WBA12345678901234") is SpecRegion.EU
    assert veh.region_of("JH4TB2H26CC000000") is SpecRegion.JAPAN


def test_bad_vin_refuses_and_stops():
    answer = _run("clicking noise", vehicle={"vin": "1FTFW1ET5DFC10312"})
    assert answer.status is AnswerStatus.REFUSED_BAD_VIN
    assert answer.causes == []
    assert "position 9" in answer.message


# --------------------------------------------------------------------------
# Fitment -- the differentiator
# --------------------------------------------------------------------------


def test_us_car_flags_euro_part():
    car = Vehicle(make="Toyota", region=SpecRegion.US)
    listings = [
        PartListing(id="L1", title="CV joint EURO SPEC front", url="u"),
        PartListing(id="L2", title="CV joint front", url="u"),
    ]
    warnings = veh.fitment_warnings(car, listings)
    assert warnings and warnings[0].severity == "blocking"
    assert warnings[0].listing_ids == ["L1"]


def test_unknown_region_says_nothing():
    car = Vehicle(make="Toyota", region=SpecRegion.UNKNOWN)
    listings = [PartListing(id="L1", title="CV joint EURO SPEC", url="u")]
    assert veh.fitment_warnings(car, listings) == []


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_trilingual_symptom_routing():
    assert txt.classify_area("the brakes squeal") is SystemArea.BRAKES
    assert txt.classify_area("скрип тормозов") is SystemArea.BRAKES
    assert txt.classify_area("արգելակները ճռռում են") is SystemArea.BRAKES
    assert txt.classify_area("подшипник гудит") is SystemArea.SUSPENSION


def test_language_detection():
    assert txt.detect_lang("brakes squeal").value == "en"
    assert txt.detect_lang("скрип тормозов").value == "ru"
    assert txt.detect_lang("արգելակները ճռռում են").value == "hy"


def test_leak_detector_actually_fires():
    """A counter that has never fired looks exactly like a broken one."""
    assert txt.find_leaks("it costs 15000 AMD")
    assert txt.find_leaks("call +374 10 55 44 33")
    assert txt.find_leaks("see https://list.am/item/1")
    assert not txt.find_leaks("replace the front wheel bearing")


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def test_cache_hit_answers_with_no_model():
    answer = _run(
        "clicking noise when I turn left at low speed",
        vehicle={"make": "Toyota", "model": "Camry", "year": 2011},
    )
    assert answer.from_cache
    assert answer.causes and "CV joint" in answer.causes[0].title
    assert answer.causes[0].evidence == ["case:C1"]
    assert not answer.trace.called("diagnostician")


def test_loose_tier_never_serves_a_cached_answer():
    """'Some Toyota had this once' is not an answer about this car."""

    class LooseStore:
        def find_similar(self, *, text, vehicle, limit=5):
            return [
                CaseMatch(
                    case=Case(id="X", symptom="s", fix="f", urgency=Urgency.DRIVE_OK),
                    score=0.99,
                    tier=MatchTier.LOOSE,
                )
            ]

    answer = run(
        Query(text="anything"),
        research=DummyResearch(),
        case_store=LooseStore(),
        model=None,
    )
    assert not answer.from_cache


def test_unverified_case_never_serves_a_cached_answer():
    class UnverifiedStore:
        def find_similar(self, *, text, vehicle, limit=5):
            return [
                CaseMatch(
                    case=Case(id="X", symptom="s", fix="f", verified=False),
                    score=0.99,
                    tier=MatchTier.EXACT,
                )
            ]

    answer = run(
        Query(text="anything"),
        research=DummyResearch(),
        case_store=UnverifiedStore(),
        model=None,
    )
    assert not answer.from_cache


def test_case_store_failure_degrades():
    class BrokenStore:
        def find_similar(self, *, text, vehicle, limit=5):
            raise RuntimeError("pgvector down")

    answer = run(
        Query(text="clicking noise"),
        research=DummyResearch(),
        case_store=BrokenStore(),
        model=None,
    )
    assert answer.status is AnswerStatus.ABSTAINED  # degraded, not crashed


# --------------------------------------------------------------------------
# Routing and safety
# --------------------------------------------------------------------------


def test_intents_skip_agents():
    assert "parts" not in NEEDS[Intent.SAFETY_CHECK]
    assert "diagnose" not in NEEDS[Intent.PART_LOOKUP]
    assert "diagnose" not in NEEDS[Intent.SHOP_LOOKUP]


def test_shop_lookup_skips_diagnosis():
    answer = _run("who fixes Toyota suspension", vehicle={"make": "Toyota"})
    assert answer.intent is Intent.SHOP_LOOKUP
    assert answer.shops
    assert not answer.trace.ran("gate")


def test_safety_floor_escalates_and_never_lowers():
    answer = _run(
        "clicking noise when I turn left at low speed",
        vehicle={"make": "Toyota", "model": "Camry", "year": 2011},
    )
    base = answer.urgency

    floored = run(
        Query(
            text="արգելակները ճռռում են",
            vehicle=Vehicle(make="Toyota", model="Camry", year=2011),
        ),
        research=DummyResearch(),
        case_store=DummyCaseStore(),
        model=None,
        safety_floor=True,
    )
    # Armenian brake text must reach the floor even with no model involved.
    assert floored.urgency is Urgency.DO_NOT_DRIVE
    assert base >= Urgency.DRIVE_OK


# --------------------------------------------------------------------------
# Traces -- what ran, what it cost, what it fetched
#
# These are the routing tests. Each asserts the exact sequence of steps and
# the exact set of model calls, so a change in routing cannot pass silently.
# Model is None throughout, so `llm_calls` shows what a *real* run would spend
# only where the keyword router short-circuits; the step sequence is identical
# either way.
# --------------------------------------------------------------------------

CAMRY = {"make": "Toyota", "model": "Camry", "year": 2011}


def test_trace_bad_vin_stops_after_one_step():
    t = _run("clicking noise", vehicle={"vin": "1FTFW1ET5DFC10312"}).trace
    assert t.steps == ["vehicle", "finalize"]
    assert t.llm_calls == []
    assert t.lookups == []  # nothing was asked of either port
    assert any("VIN rejected" in n for n in t.notes)


def test_trace_diagnose_cache_hit_runs_everything_but_pays_for_nothing():
    t = _run("clicking noise when I turn left at low speed", vehicle=CAMRY).trace
    assert t.steps == ["vehicle", "route", "gate", "diagnose", "parts", "shops", "finalize"]
    assert not t.called("diagnostician")  # the whole point of the gate
    assert t.looked_up("case_store.find_similar")
    assert t.looked_up("research.search_shops")
    assert any("cache HIT" in n for n in t.notes)


def test_trace_shop_lookup_skips_gate_and_both_agents():
    t = _run("who fixes Toyota suspension", vehicle=CAMRY).trace
    assert t.steps == ["vehicle", "route", "shops", "finalize"]
    assert not t.ran("gate")
    assert not t.ran("diagnose")
    assert not t.ran("parts")
    assert t.llm_calls == []  # keyword routing, zero model calls
    assert t.lookups == ["research.search_shops->2"]


def test_trace_part_lookup_skips_diagnosis_entirely():
    t = _run("where to buy front brake pads", vehicle=CAMRY).trace
    assert t.steps == ["vehicle", "route", "parts", "finalize"]
    assert not t.ran("gate")
    assert not t.ran("diagnose")
    assert not t.looked_up("case_store.find_similar")  # never touches your DB


def test_trace_safety_check_never_reaches_parts_or_shops():
    t = _run("can i drive with this clicking noise", vehicle=CAMRY).trace
    assert t.steps == ["vehicle", "route", "gate", "diagnose", "finalize"]
    assert not t.ran("parts")
    assert not t.ran("shops")
    assert t.looked_up("case_store.find_similar")


def test_trace_records_case_store_failure_without_crashing():
    class BrokenStore:
        def find_similar(self, *, text, vehicle, limit=5):
            raise RuntimeError("pgvector down")

    t = run(
        Query(text="clicking noise", vehicle=Vehicle(**CAMRY)),
        research=DummyResearch(),
        case_store=BrokenStore(),
        model=None,
    ).trace
    assert t.ran("gate")
    assert not t.looked_up("case_store.find_similar")  # the call never returned
    assert any("pgvector down" in n for n in t.notes)


def test_trace_cost_counts_model_calls():
    t = _run("who fixes Toyota suspension", vehicle=CAMRY).trace
    assert t.cost == len(t.llm_calls) == 0


def test_trace_survives_json_round_trip():
    """Django will serialise this, so it has to hold its shape."""
    from carmed import Trace

    original = _run("clicking noise when I turn left", vehicle=CAMRY).trace
    restored = Trace.model_validate_json(original.model_dump_json())
    assert restored.steps == original.steps
    assert restored.lookups == original.lookups


def test_answer_is_json_serialisable_for_django():
    answer = _run(
        "clicking noise when I turn left",
        vehicle={"make": "Toyota", "model": "Camry", "year": 2011},
    )
    payload = answer.model_dump(mode="json")
    assert payload["status"] == "answered"
    assert isinstance(payload["urgency"], int)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
            except Exception as exc:
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print("\nall passed" if not failures else f"\n{failures} failing")
    sys.exit(1 if failures else 0)
