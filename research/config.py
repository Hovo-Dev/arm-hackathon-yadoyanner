"""Central configuration for the deep-research component.

Everything that might change when a new API key arrives lives here, so the
agent code never has to know which provider is actually backing a capability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader so we don't add python-dotenv as a dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


# --- LLM (reasoning core) -------------------------------------------------
# The hackathon key is provider-locked to DeepSeek; every non-DeepSeek model
# returns 404 with requested_providers=["deepseek"]. Verified 2026-08-08.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REASONING_MODEL = os.environ.get("REASONING_MODEL", "deepseek/deepseek-v4-pro")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Optional providers. Absent key == capability degrades, never crashes.
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_FAST_MODEL = "sonar"                    # delta / freshness checks
PERPLEXITY_DEEP_MODEL = "sonar-deep-research"      # autonomous crawl, expensive


# --- Cost guardrails ------------------------------------------------------
# $120 one-time budget, expires 2026-08-13. A runaway tool loop can burn that
# in an afternoon, so every run is bounded.
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "6"))
MAX_EVIDENCE_ITEMS = int(os.environ.get("MAX_EVIDENCE_ITEMS", "40"))
REQUEST_TIMEOUT = 45.0

# Rough OpenRouter list price for deepseek-v4-pro, USD per 1M tokens.
# Only used for the local burn-rate estimate printed after each run.
PRICE_IN_PER_M = float(os.environ.get("PRICE_IN_PER_M", "0.28"))
PRICE_OUT_PER_M = float(os.environ.get("PRICE_OUT_PER_M", "0.42"))


# --- Source registry ------------------------------------------------------
# The research agent is deliberately NOT allowed to roam the open web. It may
# only pull from sources declared here. This is the "validated data, not random
# research" requirement enforced in code rather than in a prompt.


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    kind: str          # marketplace | official | forum | reference
    base_url: str
    lang: tuple[str, ...]
    notes: str = ""
    enabled: bool = True


SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            key="nhtsa",
            name="NHTSA (US regulator)",
            kind="official",
            base_url="https://api.nhtsa.gov",
            lang=("en",),
            notes=(
                "Free, official, no API key. Complaints carry real "
                "symptom -> dealer-diagnosis pairs; also recalls. This is the "
                "closest thing to 'a mechanic who has seen this before'."
            ),
        ),
        Source(
            key="auto_am",
            name="Auto.am",
            kind="marketplace",
            base_url="https://auto.am",
            lang=("hy", "ru", "en"),
            notes=(
                "Publishes /llms-full.txt with a documented search endpoint and "
                "explicit 'never invent listings or prices' rules. Vehicles for "
                "sale, NOT parts."
            ),
        ),
        Source(
            key="list_am",
            name="List.am",
            kind="marketplace",
            base_url="https://www.list.am",
            lang=("hy", "ru", "en"),
            notes=(
                "robots.txt: User-agent:* -> Allow:/ with "
                "Content-Signal: search=yes, ai-train=no, use=reference. "
                "So: cite and link back, never train on it. Primary parts source."
            ),
        ),
        Source(
            key="obd",
            name="OBD-II code reference",
            kind="reference",
            base_url="local",
            lang=("en",),
            notes="Offline table, no network call.",
        ),
    ]
}

# Explicitly excluded, with the reason, so nobody re-adds them by accident.
BLOCKED_SOURCES: dict[str, str] = {
    "drive2.ru": (
        "robots.txt: 'User-Agent: * / Disallow: /' plus named blocks on GPTBot, "
        "PerplexityBot and ClaudeBot. Do not crawl."
    ),
    "factory-manuals.com": (
        "Commercial vendor (Triple M FZCO) selling licensed OEM service manuals. "
        "Caching their PDFs into a KB is redistribution of a paid product. "
        "Use NHTSA + OBD references instead."
    ),
}


@dataclass
class Capabilities:
    """What this process can actually do right now, given the keys present."""

    reasoning: bool = field(default=False)
    perplexity_fast: bool = field(default=False)
    perplexity_deep: bool = field(default=False)
    firecrawl: bool = field(default=False)

    @classmethod
    def detect(cls) -> "Capabilities":
        return cls(
            reasoning=bool(OPENROUTER_API_KEY),
            perplexity_fast=bool(PERPLEXITY_API_KEY),
            perplexity_deep=bool(PERPLEXITY_API_KEY),
            firecrawl=bool(FIRECRAWL_API_KEY),
        )

    def summary(self) -> str:
        rows = [
            ("reasoning (DeepSeek)", self.reasoning, "OPENROUTER_API_KEY"),
            ("web search (Perplexity sonar)", self.perplexity_fast, "PERPLEXITY_API_KEY"),
            ("deep crawl (sonar-deep-research)", self.perplexity_deep, "PERPLEXITY_API_KEY"),
            ("page scrape (Firecrawl)", self.firecrawl, "FIRECRAWL_API_KEY"),
        ]
        lines = []
        for label, on, env in rows:
            mark = "on " if on else "off"
            suffix = "" if on else f"  (set {env})"
            lines.append(f"  [{mark}] {label}{suffix}")
        return "\n".join(lines)


USER_AGENT = (
    "YadoyannerResearchBot/0.1 (Hack Armenia 2026 prototype; "
    "respects robots.txt; contact: hovhannes.baghdasaryan.03@gmail.com)"
)
