"""Real `ResearchTool`, backed by the sources in the `research` package.

Drop-in replacement for `DummyResearch`. Nothing in `carmed` changes: this
implements the same Protocol and returns the same models.

Where the data comes from:

    search_knowledge   open web search (Firecrawl or Perplexity), plus NHTSA
                       owner complaints for the exact vehicle
    search_listings    list.am and Russian suppliers, with the top few listing
                       pages opened to recover price and fitment
    search_shops       turn.am's service directory, read from its schema.org
                       markup — no API key, no credits

On the Protocol's two hard requirements:

  1. Ids are a SHA1 of the record's URL, so the same listing keeps the same id
     across calls and processes. A counter would not survive a second search,
     and the graph cites ids in the final answer.
  2. Nothing here invents a URL or a phone number. Records missing either are
     dropped — the Protocol says returning fewer results is always acceptable.
"""

from __future__ import annotations

import hashlib
import logging
import re

from carmed.models import (
    KnowledgeSnippet,
    Money,
    PartListing,
    Shop,
    SpecRegion,
    SystemArea,
    Vehicle,
)
from research.schemas import VehicleContext
from research.sources import nhtsa, turnam, web

log = logging.getLogger(__name__)


def _rid(prefix: str, *parts: object) -> str:
    payload = "|".join(str(p) for p in parts if p)
    return f"{prefix}_{hashlib.sha1(payload.encode()).hexdigest()[:12]}"


def _ctx(v: Vehicle) -> VehicleContext:
    """carmed.Vehicle -> the VehicleContext the research sources expect."""
    return VehicleContext(
        make=v.make or "unknown",
        model=v.model or "unknown",
        year=v.year,
        engine=f"{v.engine_l}L" if v.engine_l else None,
        vin=v.vin,
    )


# turn.am groups workshops by trade; SystemArea is the graph's vocabulary.
_AREA_JOBS: dict[SystemArea, str] = {
    SystemArea.BRAKES: "brake pads discs repair",
    SystemArea.STEERING: "steering rack repair",
    SystemArea.SUSPENSION: "suspension shock absorber repair",
    SystemArea.AIRBAG: "airbag SRS electrical repair",
    SystemArea.ENGINE: "engine repair misfire",
    SystemArea.TRANSMISSION: "gearbox transmission repair",
    SystemArea.ELECTRICAL: "auto electrician alternator battery",
    SystemArea.COOLING: "radiator cooling system repair",
    SystemArea.EXHAUST: "exhaust muffler catalytic repair",
    SystemArea.HVAC: "air conditioning climate repair",
    SystemArea.BODY: "body panel dent paint repair",
    SystemArea.UNKNOWN: "general car repair",
}

# Rough mapping from a turn.am category slug back to SystemArea, so Shop
# records carry specialties the graph can filter on.
_SLUG_AREAS: list[tuple[str, SystemArea]] = [
    ("brake", SystemArea.BRAKES),
    ("engine", SystemArea.ENGINE),
    ("elektrik", SystemArea.ELECTRICAL),
    ("elektronik", SystemArea.ELECTRICAL),
    ("tire", SystemArea.SUSPENSION),
    ("wheel", SystemArea.SUSPENSION),
    ("anvadox", SystemArea.SUSPENSION),
    ("body", SystemArea.BODY),
    ("hghkvoum", SystemArea.BODY),
    ("tapq", SystemArea.BODY),
    ("glass", SystemArea.BODY),
    ("gas", SystemArea.ENGINE),
]

_VEHICLE_AD_WORDS = ("gas", "petrol", "diesel", "mileage", "км", "sedan",
                     "hatchback", "мотор", "մեքենա")

# Marques the search may return for a query about a different make. list.am
# titles name their marque, so a title naming someone else's is not this car's
# part — an "Opel Combo ignition coil" came back for a Honda Accord radiator
# cap query and was offered as a buyable option.
_KNOWN_MAKES = (
    "toyota", "honda", "nissan", "mazda", "mitsubishi", "subaru", "suzuki",
    "lexus", "infiniti", "acura", "hyundai", "kia", "ssangyong", "daewoo",
    "bmw", "mercedes", "audi", "volkswagen", "vw", "opel", "skoda", "seat",
    "renault", "peugeot", "citroen", "fiat", "volvo", "land rover", "jaguar",
    "mini", "porsche", "ford", "chevrolet", "chevy", "dodge", "jeep",
    "chrysler", "cadillac", "gmc", "lada", "vaz", "uaz", "gaz", "niva",
)


def _wrong_make(title: str, make: str | None) -> bool:
    """True when the title names a marque that is not this car's.

    Only fires when another make is named and ours is absent — a title with no
    marque at all ("Ռադիատորի կափարիչ Masuma MOX-202") is a universal part and
    stays, which is what the fitment check is there to qualify.
    """
    if not make:
        return False
    low = f" {(title or '').lower()} "
    ours = make.strip().lower()
    if ours in low:
        return False
    return any(f" {m} " in low or f" {m}-" in low for m in _KNOWN_MAKES
               if m != ours)
_YEAR_ENGINE = re.compile(r"\b(19|20)\d{2}\b.*\b\d\.\dL?\b")

# Where a listing's market can be read off the seller's own wording.
_REGION_WORDS: list[tuple[tuple[str, ...], SpecRegion]] = [
    (("usa", "american", "США", "американск"), SpecRegion.US),
    (("euro", "european", "европейск", "eu spec"), SpecRegion.EU),
    (("japan", "jdm", "японск"), SpecRegion.JAPAN),
    (("korea", "корейск"), SpecRegion.KOREA),
]


def _region_of(text: str) -> SpecRegion:
    low = (text or "").lower()
    for words, region in _REGION_WORDS:
        if any(w.lower() in low for w in words):
            return region
    return SpecRegion.UNKNOWN


def _part_likelihood(title: str, terms: list[str]) -> float:
    """Rank component ads above whole-car ads, and on-topic above off-topic.

    Two jobs. First, a parts search returns cars — they are wordier and repeat
    the make, model and year the query contains, so they outrank the parts, and
    opening one produced a PartListing for a 3,000,000 AMD car when the user
    wanted a radiator. Second, it decides whether a listing is about this part
    at all.

    Scored per WORD, not per phrase. Terms arrive as phrases
    ("ռադիատորի կափարիչ") and sellers write their own wording
    ("ռադիատորի վերևի ռեզին"), so requiring the whole phrase scored every real
    listing at zero and returned nothing. A shared word is weak evidence; every
    word of a term is strong.
    """
    low = (title or "").lower()
    score = 0.0
    for term in terms:
        words = [w for w in re.split(r"\W+", term.lower()) if len(w) > 2]
        if not words:
            continue
        hits = sum(1 for w in words if w in low)
        if hits:
            # Full phrase present scores 1.0; a single shared word scores less.
            score += hits / len(words)
    if any(w in low for w in _VEHICLE_AD_WORDS):
        score -= 1.5
    if _YEAR_ENGINE.search(low):
        score -= 1.0
    return score


class ArmenianResearch:
    """`ResearchTool` over Armenian and regional sources."""

    #: Firecrawl calls allowed for the whole request, across every call the
    #: graph makes. The parts explorer's prompt says "two searches maximum"
    #: and it made fifteen on one run -- clutch master cylinder, slave
    #: cylinder, pilot bearing, synchro rings, each in four languages, several
    #: repeated with different wording. At 5-10s a call that was 158 seconds
    #: of the 194-second run, and it exhausted the Firecrawl quota mid-answer.
    #: A per-instance budget bounds the cost of an agent that does not respect
    #: its own instruction, without needing to change its prompt.
    REQUEST_CALL_BUDGET = 8

    def __init__(self, *, enrich_budget: int = 3) -> None:
        self._calls_used = 0
        #: Listing pages to open per call. Each is a scrape, and prices only
        #: exist on the page — search results carry titles alone.
        #:
        #: This is now also the result cap: only opened listings are returned,
        #: because an unopened URL cannot be shown to be live. Kept low because
        #: the graph calls search_listings once per part option — at 4 the
        #: burst reached 12 scrapes in seconds and Firecrawl 429'd most of
        #: them. Three per call is a few real listings rather than many
        #: unchecked ones.
        self.enrich_budget = enrich_budget

    # -- knowledge ---------------------------------------------------------
    def search_knowledge(
        self, *, query: str, vehicle: Vehicle, limit: int = 6
    ) -> list[KnowledgeSnippet]:
        out: list[KnowledgeSnippet] = []
        ctx = _ctx(vehicle)

        # Regulator complaints first: they are the strongest signal available
        # for a specific vehicle, free, and need no key.
        if vehicle.year and vehicle.make:
            try:
                for ev in nhtsa.search_complaints(ctx, [query], limit=2):
                    out.append(KnowledgeSnippet(
                        id=_rid("kn", ev.title),
                        text=ev.snippet,
                        source_url=ev.url or "https://www.nhtsa.gov/recalls",
                        title=ev.title,
                        kind="catalog",
                    ))
            except Exception as exc:  # noqa: BLE001
                log.warning("NHTSA lookup failed: %s", exc)

        try:
            found = web.search(f"{vehicle.describe()} {query}".strip(), domains=[])
        except Exception as exc:  # noqa: BLE001
            log.warning("knowledge search failed: %s", exc)
            found = []

        for ev in found:
            if len(out) >= limit:
                break
            if not ev.url:          # Protocol: source_url must be real
                continue
            url = ev.url.lower()
            kind = ("manual" if any(k in url for k in ("manual", "techinfo", "owners"))
                    else "forum" if any(k in url for k in ("forum", "reddit", "club"))
                    else "catalog")
            out.append(KnowledgeSnippet(
                id=_rid("kn", ev.url),
                text=ev.snippet,
                source_url=ev.url,
                title=ev.title or "",
                kind=kind,
            ))
        return out[:limit]

    # -- listings ----------------------------------------------------------
    def search_listings(
        self, *, part_terms: list[str], vehicle: Vehicle, limit: int = 20
    ) -> list[PartListing]:
        ctx = _ctx(vehicle)
        seen: set[str] = set()
        candidates: list[tuple[float, object]] = []

        # One search per term. The terms arrive multilingual by design, and a
        # single concatenated query matches nothing — sellers write "radiator",
        # "ռադիատոր" and "радиатор" on otherwise identical ads.
        # Two terms, not four. Sellers list the same part in Armenian and
        # Russian, so two languages already reach most ads, and each extra term
        # is another search against a rate-limited API.
        for term in part_terms[:2]:
            query = f"{ctx.make} {ctx.model} {vehicle.year or ''} {term}".strip()
            self._calls_used += 1
            try:
                found = web.search_parts(query, allow_open=not candidates)
            except Exception as exc:  # noqa: BLE001
                log.warning("listing search %r failed: %s", term, exc)
                continue
            for ev in found:
                if not ev.url or ev.url in seen:
                    continue
                if not web.is_single_listing(ev.url):
                    continue     # category pages carry no price or fitment
                if _wrong_make(ev.title, vehicle.make):
                    log.info("Dropping other-make listing: %s", str(ev.title)[:60])
                    continue

                # The title has to name the part in at least one of the
                # languages asked for. Search returns whatever is loosely
                # related on the same marque, so a radiator-cap query came back
                # with a gearbox mount, a door and an ignition coil — all real,
                # all Honda, none of them the part. `part_terms` arrives
                # multilingual precisely so this test can be made in the
                # language the seller wrote in.
                score = _part_likelihood(ev.title, part_terms)
                if score <= 0:
                    log.info("Dropping off-topic listing (%.1f): %s",
                             score, str(ev.title)[:60])
                    continue

                seen.add(ev.url)
                candidates.append((score, ev))

        candidates.sort(key=lambda c: c[0], reverse=True)

        out: list[PartListing] = []
        budget = self.enrich_budget
        for _, ev in candidates:
            if len(out) >= limit:
                break
            fields = (ev.raw or {}).get("fields") or {}
            if not fields and budget > 0:
                budget -= 1
                detail = web.firecrawl_extract(ev.url)
                if detail:
                    fields = (detail[0].raw or {}).get("fields") or {}
                    ev.title = detail[0].title or ev.title

                    # Re-check the make now that the real title is known. Search
                    # returns a truncated headline — "2024) ռադիատորի վերևի
                    # ռեզին" carries no marque — and the full title turned out
                    # to be a Dodge Charger part, offered for a Honda. The
                    # earlier check could not have caught it: the evidence
                    # arrives only when the page is opened.
                    if _wrong_make(ev.title, vehicle.make):
                        log.info("Dropping other-make listing after open: %s",
                                 str(ev.title)[:60])
                        continue
                    fits = fields.get("compatible_models") or []
                    if fits and vehicle.make and not any(
                        vehicle.make.lower() in str(f).lower() for f in fits
                    ):
                        log.info("Dropping listing whose fitment excludes %s: %s",
                                 vehicle.make, str(ev.title)[:60])
                        continue

            kind = str(fields.get("part_or_vehicle") or "").lower()
            if "vehicle" in kind or kind.strip() == "car":
                log.info("Dropping vehicle ad from parts: %s", str(ev.title)[:60])
                continue

            # Only listings we actually opened are returned.
            #
            # list.am ads expire and the search index keeps serving the title
            # long after the page is gone. A radiator cap shown to a user came
            # back HTTP 404 with 3KB of navigation while its title still read
            # like a live ad, and no cheap check separates the two: list.am
            # answers 403 to every non-browser request, dead or alive, so only
            # a rendered fetch knows.
            #
            # The Protocol says returning fewer results is always acceptable
            # and fabricating one is not. A dead link presented as something to
            # buy is a fabrication in effect, so the unopened ones are dropped
            # rather than shown with a caveat nobody reads.
            if not fields:
                log.info("Dropping unverified listing (never opened): %s",
                         str(ev.title)[:60])
                continue

            amount = fields.get("price_amd")
            price = (Money(amount=float(amount), currency="AMD")
                     if isinstance(amount, (int, float)) and amount > 0 else None)

            out.append(PartListing(
                id=_rid("pl", ev.url),
                title=ev.title,
                url=ev.url,
                price=price,
                condition=(fields.get("condition") or None),
                source=(ev.url.split("/")[2] if "//" in ev.url else "list.am"),
                region=_region_of(f"{ev.title} {ev.snippet}"),
            ))
        return out

    # -- shops -------------------------------------------------------------
    def search_shops(
        self, *, make: str | None, system_area: SystemArea, city: str,
        limit: int = 5,
    ) -> list[Shop]:
        job = _AREA_JOBS.get(system_area, _AREA_JOBS[SystemArea.UNKNOWN])
        if make:
            job = f"{make} {job}"

        try:
            found = turnam.find_workshops(job, limit=limit * 2)
        except Exception as exc:  # noqa: BLE001
            log.warning("shop lookup failed: %s", exc)
            return []

        out: list[Shop] = []
        for ev in found:
            info = (ev.raw or {}).get("business") or {}
            phone = str(info.get("phone") or "").strip()
            if not phone:
                # Protocol: omit any record whose phone cannot be verified.
                log.info("Dropping shop without phone: %s", info.get("name"))
                continue

            slug = str((ev.raw or {}).get("category") or "")
            specialties = [area for token, area in _SLUG_AREAS if token in slug]

            area_bits = [b for b in (info.get("street"), info.get("city")) if b]
            out.append(Shop(
                id=_rid("sh", ev.url or "", info.get("name") or ""),
                name=str(info.get("name") or "unnamed"),
                phone=phone,
                area=", ".join(area_bits) or None,
                specialties=specialties or ([system_area] if system_area
                                            != SystemArea.UNKNOWN else []),
            ))
            if len(out) >= limit:
                break
        return out
