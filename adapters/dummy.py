"""Stand-in implementations of both ports, so the CLI runs today.

Replace both when the real pieces land:

- ``DummyCaseStore`` -> a Django service doing embedding + pgvector search.
  The matching here is naive token overlap. It is NOT an embedding model and
  is not meant to become one.
- ``DummyResearch`` -> your deep research tool.

Read these to see the shapes the real implementations must return.
"""

from __future__ import annotations

from carmed.models import (
    Case,
    CaseMatch,
    KnowledgeSnippet,
    MatchTier,
    Money,
    PartListing,
    Shop,
    SpecRegion,
    SystemArea,
    Urgency,
    Vehicle,
)
from carmed.text import normalize

# ---------------------------------------------------------------------------
# Fake corpus
# ---------------------------------------------------------------------------

CASES = [
    Case(
        id="C1",
        symptom="clicking noise when turning left at low speed, louder on sharper turns",
        fix="Front left CV joint worn, boot torn -- joint replaced",
        vehicle=Vehicle(make="Toyota", model="Camry", year=2011, engine_l=2.5, region=SpecRegion.US),
        system_area=SystemArea.SUSPENSION,
        part_names=["CV joint", "ШРУС", "կիսասռնու հոդակապ"],
        urgency=Urgency.FIX_NOW,
        source_url="https://example.invalid/thread/1",
    ),
    Case(
        id="C2",
        symptom="скрип тормозов спереди при торможении, особенно по утрам",
        fix="Front brake pads at the wear indicator -- pads and rotors replaced",
        vehicle=Vehicle(make="Toyota", model="Camry", year=2012, engine_l=2.5, region=SpecRegion.US),
        system_area=SystemArea.BRAKES,
        part_names=["brake pads", "колодки"],
        urgency=Urgency.DO_NOT_DRIVE,
        source_url="https://example.invalid/thread/2",
    ),
    Case(
        id="C3",
        symptom="անվահեծի աղմուկ արագացնելիս, դղրդյուն 60 կմ/ժ-ից բարձր",
        fix="Front wheel bearing failed -- hub assembly replaced",
        vehicle=Vehicle(make="Bmw", model="5 Series", year=2008, engine_l=3.0, region=SpecRegion.EU),
        system_area=SystemArea.SUSPENSION,
        part_names=["wheel bearing", "подшипник ступицы"],
        urgency=Urgency.FIX_NOW,
        source_url="https://example.invalid/thread/3",
    ),
]

LISTINGS = [
    PartListing(id="L1", title="CV joint ШРУС Toyota Camry 2007-2011 передний новый",
                url="https://list.am/item/1", price=Money(amount=24000), condition="new"),
    PartListing(id="L2", title="Կիսասռնու հոդակապ CV joint Camry 2.5 օգտագործված",
                url="https://list.am/item/2", price=Money(amount=12000), condition="used"),
    PartListing(id="L3", title="CV joint Toyota Camry EURO SPEC euro version front",
                url="https://list.am/item/3", price=Money(amount=19000), condition="new"),
    PartListing(id="L4", title="ШРУС граната Camry копия дешево",
                url="https://list.am/item/4", price=Money(amount=4500), condition="new"),
    PartListing(id="L5", title="Brake pads колодки тормозные Toyota Camry передние",
                url="https://list.am/item/5", price=Money(amount=18000), condition="new"),
    PartListing(id="L6", title="Wheel bearing подшипник ступицы BMW 5 E60 передний",
                url="https://list.am/item/6", price=Money(amount=32000), condition="new"),
]  # fmt: skip

SHOPS = [
    Shop(id="S1", name="Avto Servis Nairi", phone="+374 10 55 44 33", area="Shengavit",
         specialties=[SystemArea.SUSPENSION, SystemArea.BRAKES, SystemArea.ENGINE]),
    Shop(id="S2", name="Bavaria Motors", phone="+374 91 22 11 00", area="Arabkir",
         specialties=[SystemArea.SUSPENSION, SystemArea.TRANSMISSION]),
    Shop(id="S3", name="Master Brake", phone="+374 77 90 80 70", area="Kentron",
         specialties=[SystemArea.BRAKES, SystemArea.STEERING]),
]  # fmt: skip

KNOWLEDGE = [
    KnowledgeSnippet(
        id="D1", kind="manual", title="CV joint diagnosis",
        source_url="https://example.invalid/manual/cv",
        text="A clicking or popping noise that appears when turning and grows with "
             "steering angle is characteristic of a worn outer constant velocity "
             "joint. It shows up first at low speed on tight turns. Inspect the boot "
             "for splits: a torn boot lets grease out and grit in, which destroys "
             "the joint.",
    ),
    KnowledgeSnippet(
        id="D2", kind="forum", title="Camry clicking when turning",
        source_url="https://example.invalid/forum/551",
        text="Same clicking turning left on my 2011 Camry. Replaced the front left "
             "CV axle and it is completely gone. The boot was torn and had been "
             "throwing grease into the wheel well for months.",
    ),
    KnowledgeSnippet(
        id="D3", kind="manual", title="Brake pad wear indicators",
        source_url="https://example.invalid/manual/brakes",
        text="A high pitched squeal while braking usually means the pad wear "
             "indicator is touching the rotor. Measure pad thickness and replace "
             "below 3 mm. Braking is safety critical; a car with pads down to the "
             "indicator should not be driven far.",
    ),
    KnowledgeSnippet(
        id="D4", kind="forum", title="Гул подшипника ступицы",
        source_url="https://example.invalid/forum/902",
        text="Гул на скорости выше 60 км/ч, меняется при перестроении - классический "
             "признак износа подшипника ступицы. Проверяется вывешиванием колеса.",
    ),
]


def _overlap(haystack: str, needles: list[str]) -> int:
    """Count how many search terms appear in a piece of text.

    A stand-in for relevance scoring. Real retrieval embeds both sides and
    compares vectors; this counts substring hits, which is enough to make the
    CLI behave sensibly against a corpus of four documents.
    """
    hay = normalize(haystack)
    return sum(1 for n in needles if n and normalize(n) in hay)


class DummyCaseStore:
    """Token-overlap stand-in for pgvector similarity search.

    The real one embeds ``text``, filters by vehicle metadata first, then
    ranks by cosine distance -- and sets ``tier`` to say how tightly it
    filtered. See ``carmed.ports.CaseStore``.
    """

    def __init__(self, cases: list[Case] | None = None) -> None:
        self.cases = cases if cases is not None else CASES

    def find_similar(self, *, text: str, vehicle: Vehicle, limit: int = 5) -> list[CaseMatch]:
        # Words shorter than 4 characters are mostly "the", "and", "at" and
        # carry no signal. A real embedding model needs no such crutch.
        words = [w for w in normalize(text).split() if len(w) > 3]
        matches: list[CaseMatch] = []

        for case in self.cases:
            if vehicle.make and case.vehicle.make:
                if vehicle.make.casefold() != case.vehicle.make.casefold():
                    continue

            # Fraction of query words present, scaled so a good match lands
            # near 0.9 and clears CACHE_HIT_THRESHOLD. The 1.6 is pure
            # calibration to make the demo behave -- it means nothing.
            hits = _overlap(case.symptom, words)
            score = min(0.99, hits / max(len(words), 1) * 1.6) if hits else 0.0
            if score <= 0:
                continue

            same_model = (
                vehicle.model and case.vehicle.model
                and vehicle.model.casefold() == case.vehicle.model.casefold()
            )
            close_year = (
                vehicle.year and case.vehicle.year
                and abs(vehicle.year - case.vehicle.year) <= 1
            )
            # Tier says how tightly we filtered, and it gates whether this
            # match may be served as a cached answer at all. Your Django
            # implementation sets it from its actual WHERE clause.
            if same_model and close_year:
                tier = MatchTier.EXACT
            elif same_model:
                tier = MatchTier.NEAR
            else:
                tier = MatchTier.LOOSE

            matches.append(CaseMatch(case=case, score=score, tier=tier))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]


class DummyResearch:
    """Keyword-matching stand-in for the deep research tool."""

    def search_knowledge(self, *, query: str, vehicle: Vehicle, limit: int = 6) -> list[KnowledgeSnippet]:
        words = [w for w in normalize(query).split() if len(w) > 3]
        scored = [
            (_overlap(f"{k.title} {k.text}", words), k) for k in KNOWLEDGE
        ]
        return [k for score, k in sorted(scored, key=lambda p: -p[0]) if score][:limit]

    def search_listings(self, *, part_terms: list[str], vehicle: Vehicle, limit: int = 20) -> list[PartListing]:
        scored = [(_overlap(x.title, part_terms), x) for x in LISTINGS]
        return [x for score, x in sorted(scored, key=lambda p: -p[0]) if score][:limit]

    def search_shops(self, *, make: str | None, system_area: SystemArea,
                     city: str, limit: int = 5) -> list[Shop]:
        out = [
            s for s in SHOPS
            if system_area is SystemArea.UNKNOWN
            or not s.specialties
            or system_area in s.specialties
        ]
        return out[:limit]
