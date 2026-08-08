"""Live-web adapters: Perplexity (search) and Firecrawl (scrape).

Both are optional. Without their keys the functions return an empty list and
record a warning, so the research loop still completes on NHTSA, the OBD table
and turn.am — none of which need a key. That property is deliberate: the demo
must not be one expired key away from failing on stage, and it held when
Firecrawl ran out of credits mid-session.

Perplexity speaks the OpenAI chat-completions protocol, so it goes through the
same SDK as the reasoning core — one client type, one retry path, one place
where auth is handled. Its non-standard parameters (`search_domain_filter`,
`search_recency_filter`) ride along in `extra_body`.

Firecrawl has no OpenAI-shaped API and its own SDK would be an extra
dependency for a single POST, so that one stays on plain httpx.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI

from .. import config
from ..config import ROOT
from ..prompts import WEB_RETRIEVER
from ..schemas import Evidence, SourceKind
from .base import TransientError, post_json, truncate

log = logging.getLogger(__name__)

DOMAINS_PATH = ROOT / "data" / "search_domains.json"

# Used only if the data file is missing or unreadable.
FALLBACK_DOMAINS: list[str] = ["list.am", "auto.am", "nhtsa.gov", "exist.ru"]

MAX_DOMAINS = 20  # provider cap on search_domain_filter


@lru_cache(maxsize=1)
def allowed_domains() -> list[str]:
    """Load the search allowlist from data/search_domains.json.

    Kept as data so a teammate can add a parts site mid-hackathon without
    touching Python. The agent cannot search outside this list.
    """
    if not DOMAINS_PATH.exists():
        log.warning("%s missing — using %d fallback domains.",
                    DOMAINS_PATH, len(FALLBACK_DOMAINS))
        return list(FALLBACK_DOMAINS)
    try:
        payload = json.loads(DOMAINS_PATH.read_text(encoding="utf-8"))
        groups = payload.get("groups") or {}
        enabled = payload.get("enabled_groups") or list(groups)
        out: list[str] = []
        for name in enabled:
            for domain in (groups.get(name) or {}).get("domains") or []:
                if domain not in out:
                    out.append(domain)
        if not out:
            raise ValueError("no domains enabled")
        if len(out) > MAX_DOMAINS:
            log.warning("Allowlist has %d domains; provider caps at %d. Truncating.",
                        len(out), MAX_DOMAINS)
        return out[:MAX_DOMAINS]
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s (%s) — using fallback.", DOMAINS_PATH, exc)
        return list(FALLBACK_DOMAINS)


@lru_cache(maxsize=1)
def _client() -> OpenAI | None:
    if not config.PERPLEXITY_API_KEY:
        return None
    return OpenAI(
        api_key=config.PERPLEXITY_API_KEY,
        base_url=config.PERPLEXITY_BASE_URL,
        timeout=300.0,
    )


def _citations_from(response: Any) -> list[Any]:
    """Pull citations out wherever this API version happens to put them."""
    for attr in ("citations", "search_results"):
        found = getattr(response, attr, None)
        if found:
            return list(found)
    raw = getattr(response, "model_extra", None) or {}
    for key in ("citations", "search_results"):
        if raw.get(key):
            return list(raw[key])
    return []


def perplexity_search(
    query: str,
    domains: list[str] | None = None,
    deep: bool = False,
    recency: str | None = None,
) -> list[Evidence]:
    """Search via Perplexity, restricted to the allowlist.

    `deep=True` selects sonar-deep-research: minutes, not seconds, and dollars,
    not cents. Reserve it for a genuine cache miss.
    """
    client = _client()
    if client is None:
        log.info("PERPLEXITY_API_KEY absent — skipping web search for %r", query)
        return []

    domains = (domains or allowed_domains())[:MAX_DOMAINS]
    model = config.PERPLEXITY_DEEP_MODEL if deep else config.PERPLEXITY_FAST_MODEL

    extra: dict[str, Any] = {"search_domain_filter": domains}
    if recency:
        extra["search_recency_filter"] = recency

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": WEB_RETRIEVER},
                {"role": "user", "content": query},
            ],
            extra_body=extra,
        )
    except Exception as exc:  # noqa: BLE001 - a dead source degrades, never aborts
        log.warning("Perplexity search failed (%s): %s", model, exc)
        return []

    content = (response.choices[0].message.content or "").strip()
    if not content:
        return []

    citations = _citations_from(response)
    out = [
        Evidence(
            source_key="perplexity_deep" if deep else "perplexity",
            source_kind=SourceKind.WEB,
            title=f"Web research: {truncate(query, 80)}",
            snippet=truncate(content, 1500),
            url=None,
            relevance=0.8,
            raw={"model": model, "citation_count": len(citations)},
        )
    ]

    for cite in citations[:8]:
        if isinstance(cite, str):
            url, title, snippet = cite, cite, ""
        elif isinstance(cite, dict):
            url = cite.get("url")
            title = cite.get("title") or url
            snippet = cite.get("snippet") or ""
        else:
            continue
        if url:
            out.append(
                Evidence(
                    source_key="perplexity_citation",
                    source_kind=SourceKind.WEB,
                    title=title or url,
                    snippet=truncate(snippet or url, 400),
                    url=url,
                    relevance=0.6,
                )
            )
    return out


def url_allowed(url: str) -> bool:
    """Is this URL inside the allowlist?

    Enforced on every scrape. The model reads web content during a run, and web
    content is untrusted input — a page can contain text telling the model to
    "fetch http://attacker/". Without this check, a scrape tool turns the agent
    into an open redirector for whatever it just read. The allowlist means the
    worst case is a wasted call.
    """
    try:
        host = (httpx.URL(url).host or "").lower()
    except Exception:  # noqa: BLE001
        return False
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in allowed_domains())


def firecrawl_search(
    query: str,
    domains: list[str] | None = None,
    limit: int = 5,
) -> list[Evidence]:
    """Web search via Firecrawl. Stands in for Perplexity when that key is absent.

    Firecrawl has no domain-filter parameter, so scoping is done with `site:`
    operators and then re-checked against the allowlist on the way out — the
    operator is a hint to the engine, `url_allowed` is the actual boundary.
    """
    if not config.FIRECRAWL_API_KEY:
        return []

    domains = allowed_domains() if domains is None else domains
    scoped = query
    if domains:
        # A few sites per query; too many `site:` terms returns nothing at all.
        sites = " OR ".join(f"site:{d}" for d in domains[:4])
        scoped = f"{query} ({sites})"

    data = post_json(
        "https://api.firecrawl.dev/v2/search",
        {"query": scoped, "limit": limit},
        headers={"Authorization": f"Bearer {config.FIRECRAWL_API_KEY}"},
        timeout=120.0,
    )
    if not data or not data.get("success"):
        return []

    results = (data.get("data") or {}).get("web") or []
    out: list[Evidence] = []
    for item in results[:limit]:
        url = item.get("url") or ""
        if domains and not url_allowed(url):
            log.info("Dropping off-allowlist search result: %s", url)
            continue

        # Marketplace search returns a mix of individual listings and category
        # pages. Only the former carries a price, a condition and a fitment
        # line. Saying which is which in the snippet is what actually gets the
        # model to open the useful ones — it will not infer it from the URL.
        #
        # This annotation is about SHOPPING and must not reach other result
        # types: it was labelling a Honda tech-info manual PDF "category/search
        # page — no price here; not worth scraping", which is both meaningless
        # and actively discourages opening the best source available.
        single = is_single_listing(url)
        if _is_marketplace(url):
            marker = (
                "[SINGLE LISTING — call scrape_page on this URL for price, "
                "condition and fitment] "
                if single
                else "[category/search page — no price here; not worth scraping] "
            )
        elif url.lower().endswith(".pdf"):
            marker = "[PDF document — cite it as the source; do not scrape it] "
        else:
            marker = ""
        out.append(
            Evidence(
                source_key="firecrawl_search",
                source_kind=SourceKind.WEB,
                title=item.get("title") or url,
                snippet=marker + truncate(
                    item.get("description") or item.get("title") or url, 400
                ),
                url=url,
                relevance=(0.7 if single else 0.4) if _is_marketplace(url) else 0.6,
                raw={"provider": "firecrawl", "single_listing": single},
            )
        )
    # Individual listings first, so they survive any downstream truncation.
    out.sort(key=lambda e: e.relevance, reverse=True)
    return out


# Path patterns that identify one listing rather than a category or search page.
_ITEM_PATTERNS = ("/item/", "/offer/", "/products/", "/tovar/")


def _is_marketplace(url: str) -> bool:
    """True for classifieds/parts shops, where 'is this one listing' matters."""
    try:
        host = (httpx.URL(url).host or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return any(host == h or host.endswith("." + h) for h in _LISTING_HOSTS)


def is_single_listing(url: str) -> bool:
    try:
        path = (httpx.URL(url).path or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return any(p in path for p in _ITEM_PATTERNS)



# The first version kept a hand-written table of manufacturer portals. It
# covered Honda, Toyota and Nissan, so every other make fell back to three
# generic spec sites — and because the search was pinned to those, a query for
# a 2010 BMW 320i came back with pages for a 2018 320i xDrive and a 2002 525i.
# That is not reduced coverage, it is the wrong car's specifications, which is
# the same failure the local corpus produced.
#
# A fixed list cannot span Mercedes, BMW, Opel, Kia, Lada and everything else
# on Armenian roads. So the search runs open and authority is judged from the
# result instead: a hostname containing the marque name is the manufacturer's
# own site, whoever the marque happens to be.
#
# Safety note: this widens SEARCH only. scrape_page stays on the allowlist,
# because fetching a page is where untrusted content could act on us.

# Words that appear in manufacturer hostnames but are not the marque itself.
# Aftermarket references that are useful but not authoritative.

# Hosts that surface for manual queries but cannot be cited as a specification.
# Social and document-dump sites republish manuals without provenance or
# version, and a spec quoted from a Facebook post is not checkable. They are
# demoted rather than blocked: an unrestricted search is what makes any vehicle
# reachable, so quality is handled by ranking, not by another fixed list of
# who is allowed to exist.
_LOW_TRUST_HOSTS = (
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "pinterest.com", "scribd.com", "yumpu.com", "slideshare.net",
    "issuu.com", "quora.com", "answers.com",
)

# Forums are not authoritative but are often correct and specific, so they sit
# between OEM sources and social noise rather than being lumped in with it.
_FORUM_MARKERS = ("forum", "reddit.com", "post.com", "addicts", "fest", "club")


def _trust_adjust(url: str) -> float:
    """Relevance adjustment based on how citable the host is."""
    if not url:
        return 0.0
    try:
        host = (httpx.URL(url).host or "").lower()
    except Exception:  # noqa: BLE001
        return 0.0
    if any(host == h or host.endswith("." + h) for h in _LOW_TRUST_HOSTS):
        return -0.35
    if any(m in host for m in _FORUM_MARKERS):
        return -0.10
    return 0.0


# Where an Armenian owner can realistically buy a part, best first. Used to
# RANK results, never to exclude them.
#
# The previous design pinned parts search to an allowlist, which failed twice
# over: a supplier not on the list was invisible, and because Firecrawl only
# tolerates about four `site:` operators the code sent `domains[:4]` — so 16 of
# the 20 allowlisted domains, including every Russian supplier, were never
# searched at all. The list described an intention the code did not implement.
#
# Ranking states the same preference without either failure: local sellers win
# when they exist, and a supplier nobody listed can still surface.
_LOCALITY_BOOST: list[tuple[tuple[str, ...], float]] = [
    (("list.am", "auto.am", "am.all.biz",
      "turn.am", "spyur.am"), 0.30),                     # Armenia
    (("am", "ge"), 0.20),                                # any .am / .ge domain
    (("exist.ru", "emex.ru", "avito.ru", "drom.ru",
      "autopiter.ru", "avtoto.ru"), 0.15),               # Russia — usual import route
    (("autodoc.de", "partsouq.com", "rockauto.com"), 0.05),  # catalogues, cross-ref
]


def _host_matches(host: str, needle: str) -> bool:
    """Match on host boundaries, never as a bare substring.

    `"turn.am" in url` is true for turn.am.evil.com, which would hand a spoofed
    domain the local-seller boost and float it to the top of a parts list. The
    scrape allowlist already used host matching; the ranking did not.
    """
    needle = needle.strip("/.").lower()
    if needle.startswith("."):        # TLD-style hint such as ".ge"
        return host.endswith(needle)
    return host == needle or host.endswith("." + needle)


def _locality_score(url: str) -> float:
    """How reachable is this seller for a buyer in Yerevan?"""
    try:
        host = (httpx.URL(url or "").host or "").lower()
    except Exception:  # noqa: BLE001
        return 0.0
    if not host:
        return 0.0
    for hosts, boost in _LOCALITY_BOOST:
        for needle in hosts:
            clean = needle.strip("/.").replace("//", "")
            if not clean:
                continue
            if clean.startswith("am") or clean.startswith("ge"):
                # country hints like ".am/" or "//am." — match the TLD
                if host.endswith("." + clean) or host == clean:
                    return boost
                continue
            if _host_matches(host, clean):
                return boost
    return 0.0



# Markets an Armenian buyer can actually use, in the order they would try them.
# Grouped in fours because Firecrawl returns nothing when handed many more
# `site:` operators than that.
# The markets an Armenian buyer can reach, as ONE `site:` filter. Firecrawl
# tolerates about four operators before returning nothing, which is why this is
# four and not the whole allowlist.
_PARTS_SITES = ["list.am", "auto.am", "exist.ru", "emex.ru"]


def search_parts(query: str, allow_open: bool = False) -> list[Evidence]:
    """Parts search: one scoped query, ranked by how reachable the seller is.

    Deliberately a single request. The previous version ran two local passes
    plus an open pass, and the caller repeated it per part term — 12 searches
    for one part, and the graph asks for several parts per run. That reached
    30-45 Firecrawl calls and spent most of them collecting 429s, so a run that
    looked like it worked came back with no prices and no listings.

    `allow_open` adds one unscoped query, for callers that would rather have a
    distant seller than nothing. Off by default: parts are geography-bound, and
    an AutoZone link does not help someone in Yerevan.
    """
    try:
        merged = search(query, domains=_PARTS_SITES)
    except TransientError as exc:
        log.warning("Parts search failed transiently: %s", exc)
        merged = []

    if allow_open and not merged:
        try:
            merged = search(query, domains=[])
        except TransientError as exc:
            log.warning("Open parts search failed transiently: %s", exc)
            merged = []

    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in merged:
        if not ev.url or ev.url in seen:
            continue
        seen.add(ev.url)
        ev.relevance = max(0.05, min(
            ev.relevance + _locality_score(ev.url) + _trust_adjust(ev.url), 1.0))
        out.append(ev)

    out.sort(key=lambda e: e.relevance, reverse=True)
    return out


def search(query: str, deep: bool = False,
           domains: list[str] | None = None) -> list[Evidence]:
    """Provider-routing entry point for web search.

    Perplexity when its key is present (better synthesis, native domain filter),
    Firecrawl otherwise, nothing if neither. Callers never branch on provider.
    """
    if config.PERPLEXITY_API_KEY:
        return perplexity_search(query, domains=domains, deep=deep)
    if config.FIRECRAWL_API_KEY:
        if deep:
            log.info("deep=True requested but only Firecrawl available — "
                     "running a normal search instead.")
        return firecrawl_search(query, domains=domains)
    log.info("No search provider configured — skipping %r", query)
    return []


# Pages whose "content" is really site chrome. See the status-code guard below.
_BOILERPLATE = ("set up your shop", "your cart", "categories", "login", "sign in")

# Marketplaces where a page is one listing, so structured extraction beats
# dumping markdown.
_LISTING_HOSTS = ("list.am", "auto.am", "avito.ru", "drom.ru", "exist.ru", "emex.ru")

LISTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "price_amd": {"type": "number", "description": "Price in AMD if shown"},
        "price_other": {"type": "string", "description": "Price with currency if not AMD"},
        "condition": {"type": "string", "description": "New, used, refurbished"},
        "part_or_vehicle": {"type": "string", "description": "'Car Part' or 'Vehicle'"},
        "compatible_models": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact models and year ranges this part fits",
        },
        "part_number": {"type": "string"},
        "location": {"type": "string"},
        "seller_type": {"type": "string"},
        "description": {"type": "string"},
    },
}


def _is_listing(url: str) -> bool:
    host = (httpx.URL(url).host or "").lower()
    return any(host == h or host.endswith("." + h) for h in _LISTING_HOSTS)


def _render_listing(fields: dict[str, Any]) -> str:
    """Flatten extracted fields into a compact line for the model.

    Fitment goes first and is called out explicitly. The whole reason to prefer
    structured extraction here is that a marketplace page buries "fits L33
    2013-2018" inside kilobytes of navigation, and a part that does not fit the
    car is the single most expensive thing this tool can recommend.
    """
    parts: list[str] = []
    fits = fields.get("compatible_models")
    if fits:
        parts.append("FITS: " + "; ".join(str(f) for f in fits))
    if fields.get("part_number"):
        parts.append(f"part number: {fields['part_number']}")
    price = fields.get("price_amd")
    if price:
        parts.append(f"price: {int(price):,} AMD")
    elif fields.get("price_other"):
        parts.append(f"price: {fields['price_other']}")
    for key, label in (
        ("condition", "condition"), ("part_or_vehicle", "type"),
        ("location", "location"), ("seller_type", "seller"),
    ):
        if fields.get(key):
            parts.append(f"{label}: {fields[key]}")
    if fields.get("description"):
        parts.append(f"— {truncate(str(fields['description']), 300)}")
    return " | ".join(parts) if parts else ""


def firecrawl_extract(url: str) -> list[Evidence]:
    """Pull a marketplace listing into structured fields rather than markdown."""
    if not config.FIRECRAWL_API_KEY or not url_allowed(url):
        return []

    data = post_json(
        "https://api.firecrawl.dev/v2/scrape",
        {
            "url": url,
            "formats": [{"type": "json", "schema": LISTING_SCHEMA}],
            "onlyMainContent": True,
        },
        headers={"Authorization": f"Bearer {config.FIRECRAWL_API_KEY}"},
        timeout=180.0,
    )
    if not data or not data.get("success"):
        return []

    payload = data.get("data") or {}
    fields = payload.get("json") or {}
    body = _render_listing(fields)
    if not body:
        return []

    meta = payload.get("metadata") or {}
    status = meta.get("statusCode")
    if isinstance(status, int) and not (200 <= status < 300):
        return []

    return [
        Evidence(
            source_key="listing",
            source_kind=SourceKind.MARKETPLACE,
            title=str(fields.get("title") or meta.get("title") or url)[:160],
            snippet=body,
            url=url,
            relevance=0.85,  # a concrete priced listing outranks prose about one
            raw={"fields": fields, "status": status},
        )
    ]


def firecrawl_scrape(url: str, max_chars: int = 4000) -> list[Evidence]:
    """Read one page.

    Marketplace listings go through structured extraction; everything else
    (forum threads, catalogue pages, articles) comes back as markdown, where
    prose is the point and there are no fields to pull.
    """
    if not config.FIRECRAWL_API_KEY:
        log.info("FIRECRAWL_API_KEY absent — skipping scrape of %s", url)
        return []

    if not url_allowed(url):
        log.warning("Refusing to scrape off-allowlist URL: %s", url)
        return []

    if _is_listing(url):
        structured = firecrawl_extract(url)
        if structured:
            return structured
        log.info("Structured extraction empty for %s — falling back to markdown.", url)

    data = post_json(
        "https://api.firecrawl.dev/v2/scrape",
        {"url": url, "formats": ["markdown"], "onlyMainContent": True},
        headers={"Authorization": f"Bearer {config.FIRECRAWL_API_KEY}"},
        timeout=120.0,
    )
    if not data:
        return []

    payload = data.get("data") or {}
    markdown = (payload.get("markdown") or "").strip()
    meta = payload.get("metadata") or {}
    status = meta.get("statusCode")

    # Firecrawl reports success=true for a 404 and hands back the site's
    # navigation chrome. An expired list.am listing came back as 3KB of menus
    # and cart links, which would otherwise be filed as evidence about a part.
    if isinstance(status, int) and not (200 <= status < 300):
        log.info("Scrape of %s returned HTTP %s — discarding (likely expired listing).",
                 url, status)
        return []

    if not markdown:
        return []

    head = markdown[:400].lower()
    if sum(1 for token in _BOILERPLATE if token in head) >= 3:
        log.info("Scrape of %s looks like site chrome rather than content — discarding.",
                 url)
        return []

    return [
        Evidence(
            source_key="firecrawl",
            source_kind=SourceKind.WEB,
            title=meta.get("title") or url,
            snippet=truncate(markdown, max_chars),
            url=url,
            relevance=0.7,
            raw={"status": status},
        )
    ]
