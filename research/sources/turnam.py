"""turn.am — Armenian services marketplace. Direct adapter, no search API needed.

Answers the step after a diagnosis: who actually fixes this, and how do I reach
them. turn.am publishes schema.org LocalBusiness JSON-LD on every workshop page,
so name, phone, street address and customer ratings come out structured rather
than scraped from prose.

Two properties make this the most reliable source in the project:

  - No API key and no credits. Plain HTTPS, and robots.txt permits it
    (only /*watchlist*, /get-address-map-box, */redirect-external-reserve/*,
    /get-armenian-exchange-rates and */ad-widget* are disallowed; category and
    business pages are not). It kept working when Firecrawl ran out of credits.
  - Category browsing instead of search. The site groups workshops by trade, so
    an engine problem goes to /category/engine-repair directly. No query
    wording to get wrong, no search engine deciding what is relevant, and no
    language barrier — the categories are fixed URLs.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any

import httpx

from .. import cache, config
from ..schemas import Evidence, SourceKind
from .base import _throttle

log = logging.getLogger(__name__)

BASE = "https://turn.am"
CATEGORY_URL = BASE + "/category/{slug}"
YEREVAN = {"latitude": "40.1772616", "longitude": "44.5128372", "distance": "200"}

SITEMAP_CATEGORIES = BASE + "/sitemap_categories"

# Slugs that indicate a vehicle trade. Matched against the LIVE sitemap rather
# than being a list of categories themselves — the site publishes 378 and adds
# more, and a hand-written subset silently misroutes.
#
# The first version hardcoded seven slugs read off one page. It missed
# `brake-system-maintenance-repair`, `tire-services`, `wheel-alignment`,
# `avtoelektrik` and `car-gas-installation`, so a brake job was sent to the
# generic category and the specialists were never shown.
_VEHICLE_MARKERS = (
    "car", "auto", "avto", "vehicle", "brake", "tire", "wheel", "engine",
    "ev-", "electric-vehicle", "anvadox", "body-p", "body-pol",
)
# Categories matching a vehicle marker but belonging to another trade.
_NOT_VEHICLE = (
    # "carayutyun" is ծառայություն — Armenian for "services" — transliterated.
    # It contains the substring "car", so the naive marker matched pool
    # cleaning, funeral services, publishing and business consulting: 21 of 61
    # discovered categories were not vehicle-related at all.
    "carayutyun", "carayutyunner",
    "carpet", "career", "cargo", "carbon-and-kevlar", "car-insurance",
    "car-rental", "corporate-car-rental", "automatic-gates", "automation",
    "engineering", "autopilot", "biznes", "chatbot", "healthcare",
)

# Words that point at a specific trade. Additive hints on top of the live
# category list — never the list itself.
# Each trade lists its terms in four written forms, because all four turn up in
# real messages:
#   English          "brakes squeal"
#   Armenian script  "արգելակները ճռռում են"
#   Russian          "тормоза скрипят"
#   LATIN-SCRIPT ARMENIAN — "axpers argelaknery crrum en"
#
# The fourth is the one that is easy to forget and the most common in chat.
# Armenians routinely type Armenian in Latin letters, and there is no single
# standard: brakes appear as argelak / argelag / arkelak. turn.am's own slugs
# are transliterated the same way (avtoelektrik, anvadoxeri, hghkvoum), so
# matching Latin-Armenian against them is natural rather than a special case.
_ROUTING_HINTS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("brake", "արգելակ", "тормоз", "pad", "rotor", "caliper",
      "argelak", "argelag", "arkelak", "tormoz", "kalodka", "kolodka"),
     ("brake-system",)),
    (("tire", "tyre", "puncture", "անվադող", "шина", "резин",
      "anvadog", "anvadox", "shina", "rezin", "balon"),
     ("tire-services", "anvadox")),
    (("align", "camber", "steering pull", "развал",
      "razval", "shozhdenie"), ("wheel-alignment",)),
    (("electric", "wiring", "alternator", "starter", "battery", "sensor",
      "էլեկտր", "электрик",
      "elektrik", "elektrakan", "generator", "akumlyator", "akumulator",
      "starter", "datchik"), ("avtoelektrik", "elektronikayi")),
    (("engine", "motor", "misfire", "timing", "piston", "head gasket",
      "compression", "շարժիչ", "двигател",
      "sharzhich", "sharjich", "motor", "dvigatel", "porshen", "klapan"),
     ("engine-repair",)),
    (("glass", "windscreen", "windshield", "ապակի", "стекл",
      "apaki", "steklo", "lobovoy"), ("auto-glass",)),
    (("paint", "rust", "թափք", "покрас", "кузов",
      "nerkel", "nerk", "pokras", "kuzov", "zhang"),
     ("body-pol", "hghkvoum")),
    (("dent", "panel", "collision", "вмятин",
      "tapq", "tapk", "poso", "vmyatin", "udar"),
     ("body-plastic", "tapqi")),
    (("gas", "lpg", "cng", "գազ", "gaz", "metan", "propan"),
     ("car-gas-installation",)),
    (("interior", "upholster", "salon", "salon", "shapik"),
     ("car-interior",)),
    (("key", "lock", "immobiliser", "բանալի", "banali", "klyuch", "zamok"),
     ("car-key",)),
    (("clean", "detail", "wash", "polish", "մաքր", "мойк",
      "maqrel", "maqur", "moyka", "polirovka"),
     ("car-detailing", "carwash")),
    (("inspection", "tech check", "техосмотр", "texzhtum", "texosmotr"),
     ("vehicle-technical-inspection",)),
]

# Used when nothing more specific matches, in preference order.
_GENERAL_FALLBACKS = ("comprehensive-auto-repair-services",
                      "specialized-auto-services", "car-services")

_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


def vehicle_categories() -> list[str]:
    """Live list of vehicle-related category slugs, from turn.am's sitemap.

    Cached for a week. Falls back to the general categories if the sitemap is
    unreachable, so a network problem degrades routing rather than breaking it.
    """
    cached = cache.get("turnam_categories", {"v": 2}, ttl_s=7 * 24 * 3600)
    if cached:
        return cached

    xml = _fetch(SITEMAP_CATEGORIES)
    slugs: list[str] = []
    for loc in _LOC_RE.findall(xml or ""):
        if "/category/" not in loc:
            continue
        slug = loc.rsplit("/category/", 1)[-1].strip().lower()
        if not slug or any(bad in slug for bad in _NOT_VEHICLE):
            continue
        if any(marker in slug for marker in _VEHICLE_MARKERS):
            slugs.append(slug)

    if not slugs:
        log.warning("turn.am sitemap gave no vehicle categories — using fallbacks.")
        return list(_GENERAL_FALLBACKS)

    slugs = sorted(set(slugs))
    cache.put("turnam_categories", {"v": 2}, slugs)
    log.info("turn.am: %d vehicle categories discovered", len(slugs))
    return slugs


# Slug words that carry no routing signal — every category has them.
_STOP_SLUG_WORDS = {"services", "service", "repair", "maintenance", "car",
                    "auto", "vehicle", "and", "of", "the", "carayutyunner"}


def _slug_tokens(slug: str) -> set[str]:
    return {w for w in re.split(r"[-_]", slug.lower())
            if len(w) > 2 and w not in _STOP_SLUG_WORDS}


def _similarity(job_words: set[str], slug: str) -> float:
    """How well does this category match the request?

    Exact substring matching alone is brittle: `job` is the one argument the
    model writes freely — every other field comes from the onboarding form —
    so the wording is unpredictable. "brake pads worn", "braking noise",
    "argelaknery mashvel en" and "needs new pads" all mean the same category
    and only the first would match a fixed keyword.

    Scored on two signals: shared words, plus fuzzy closeness for spelling
    variants and transliteration (argelak / argelag / arkelak).
    """
    slug_words = _slug_tokens(slug)
    if not slug_words or not job_words:
        return 0.0

    overlap = len(job_words & slug_words) / len(slug_words)

    fuzzy = 0.0
    for jw in job_words:
        for sw in slug_words:
            ratio = difflib.SequenceMatcher(None, jw, sw).ratio()
            if ratio > fuzzy:
                fuzzy = ratio

    return overlap * 0.7 + (fuzzy if fuzzy >= 0.8 else 0.0) * 0.3


def route_job(job: str, category: str | None = None) -> str:
    """Map a repair description to the best live turn.am category.

    `category` short-circuits everything: when the caller already knows which
    category it wants — the model can be shown the real list and choose — that
    beats any inference we do here.
    """
    available = vehicle_categories()
    if category:
        for slug in available:
            if slug == category or category in slug:
                return slug
        log.info("Requested category %r not live; inferring instead.", category)

    low = (job or "").lower()

    # 1. Explicit multilingual hints. High precision, so tried first.
    for keywords, slug_hints in _ROUTING_HINTS:
        if not any(k in low for k in keywords):
            continue
        for hint in slug_hints:
            for slug in available:
                if hint in slug:
                    return slug

    # 2. Similarity against the live slugs, for wording the hints do not cover.
    job_words = {w for w in re.split(r"\W+", low) if len(w) > 2}
    scored = [(_similarity(job_words, slug), slug) for slug in available]
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.35:
        log.info("Routed %r -> %s by similarity (%.2f)",
                 job[:40], scored[0][1], scored[0][0])
        return scored[0][1]

    for fallback in _GENERAL_FALLBACKS:
        if fallback in available:
            return fallback
    return available[0] if available else "car-services"


def category_label(slug: str) -> str:
    """Human-readable name for a slug, derived rather than looked up."""
    return slug.replace("-", " ").replace("_", " ").strip()


def _fetch(url: str, params: dict[str, str] | None = None) -> str:
    _throttle("turn.am")
    try:
        resp = httpx.get(
            url, params=params,
            headers={"User-Agent": config.USER_AGENT,
                     "Accept-Language": "hy,ru;q=0.8,en;q=0.6"},
            timeout=30.0, follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001 - a dead source degrades, never aborts
        log.warning("turn.am fetch failed for %s: %s", url, exc)
        return ""


_BUSINESS_RE = re.compile(r'href="(https://turn\.am/business/[^"/]+/detail)"')
_LDJSON_RE = re.compile(
    r'<script type="application/ld\+json">(.*?)</script>', re.S)


def list_workshops(slug: str, limit: int = 6) -> list[str]:
    """Workshop detail URLs from one category page, Yerevan-centred."""
    html = _fetch(CATEGORY_URL.format(slug=slug), params=YEREVAN)
    if not html:
        return []
    seen: list[str] = []
    for url in _BUSINESS_RE.findall(html):
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


def _parse_business(html: str) -> dict[str, Any] | None:
    """Pull the LocalBusiness record out of a workshop page."""
    for block in _LDJSON_RE.findall(html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("@type") not in ("LocalBusiness", "AutoRepair", "Organization"):
                continue

            addr = entry.get("address") or {}
            reviews = entry.get("reviews") or []
            ratings = []
            for r in reviews:
                try:
                    ratings.append(float((r.get("reviewRating") or {}).get("ratingValue")))
                except (TypeError, ValueError):
                    pass

            return {
                "name": entry.get("name"),
                "phone": entry.get("telephone"),
                "street": addr.get("streetAddress"),
                "city": addr.get("addressLocality"),
                "url": entry.get("url") or entry.get("@id"),
                "rating": round(sum(ratings) / len(ratings), 1) if ratings else None,
                "review_count": len(reviews),
            }
    return None


def find_workshops(job: str, limit: int = 5,
                   category: str | None = None) -> list[Evidence]:
    """Workshops that handle this repair, with contact details.

    Cached per category: the same category serves many different questions, and
    a workshop list does not change hour to hour.
    """
    slug = route_job(job, category)
    ckey = {"slug": slug, "limit": limit}

    cached = cache.get("turnam_workshops", ckey, ttl_s=7 * 24 * 3600)
    if cached is not None:
        # `raw` is declared exclude=True on Evidence, so model_dump() drops it.
        # Caching dumps therefore lost the business record — name, phone,
        # address — and every cached lookup returned shops with no contact
        # details, which the caller then discarded. The first call worked and
        # every one after it silently returned nothing.
        return [
            Evidence.model_validate(e).model_copy(update={"raw": e.get("raw", {})})
            for e in cached
        ]

    urls = list_workshops(slug, limit=limit)
    if not urls:
        return []

    out: list[Evidence] = []
    for url in urls:
        info = _parse_business(_fetch(url))
        if not info or not info.get("name"):
            continue

        bits = [f"{info['name']}"]
        if info.get("rating"):
            bits.append(f"rated {info['rating']}/5 from {info['review_count']} review(s)")
        if info.get("street"):
            bits.append(f"{info['street']}, {info.get('city') or 'Yerevan'}")
        if info.get("phone"):
            bits.append(f"tel {info['phone']}")
        bits.append(f"category: {category_label(slug)}")

        out.append(
            Evidence(
                source_key="workshop",
                source_kind=SourceKind.MARKETPLACE,
                title=f"{info['name']} — {category_label(slug)}",
                snippet=" | ".join(bits),
                url=info.get("url") or url,
                relevance=0.85 if info.get("rating") else 0.7,
                raw={"business": info, "category": slug},
            )
        )

    if out:
        cache.put("turnam_workshops", ckey, [
            {**e.model_dump(mode="json"), "raw": e.raw} for e in out
        ])
    return out
