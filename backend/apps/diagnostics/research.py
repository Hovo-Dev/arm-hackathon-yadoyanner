"""The `carmed.ports.ResearchTool` seam -- intentionally empty.

The real implementation (list.am listings, forum/manual retrieval, shop
directory) is being built on a separate branch and has not landed yet. This
placeholder exists so the agentic loop can run end to end today, and so that
branch has exactly one function to replace: `get_research_tool`.

**Returning nothing is a supported state, not a broken one.** Every consumer in
carmed degrades on its own:

- the diagnostician is told "No sources found for that symptom. Try different
  wording." and works from matched KB cases alone (or abstains, which is a
  correct answer)
- the parts explorer produces no options; `PartOption` has no price field, so
  there is nowhere for an invented one to land anyway
- the shops step returns an empty list
- the trace records `research.search_knowledge->0`, so the emptiness is visible
  rather than silent

Returning a fabricated row to look busy is the one unrecoverable mistake here:
a made-up phone number or price is worse than no answer.
"""
from carmed.models import KnowledgeSnippet, PartListing, Shop, SystemArea, Vehicle


class NullResearchTool:
    """No sources wired up yet. Every method honestly returns nothing."""

    def search_knowledge(
        self, *, query: str, vehicle: Vehicle, limit: int = 6
    ) -> list[KnowledgeSnippet]:
        return []

    def search_listings(
        self, *, part_terms: list[str], vehicle: Vehicle, limit: int = 20
    ) -> list[PartListing]:
        return []

    def search_shops(
        self, *, make: str | None, system_area: SystemArea, city: str, limit: int = 5
    ) -> list[Shop]:
        return []


def get_research_tool():
    """The single swap point. Point this at the real tool when it lands --
    nothing else in the Django layer needs to change.

    `ArmenianResearch` is that tool: NHTSA complaints and open web search for
    knowledge, list.am plus Russian suppliers for parts, turn.am's directory
    for workshops. It falls back to `NullResearchTool` when the package is not
    importable, because an unavailable source is a supported state here and a
    500 on every diagnosis is not.

    Only turn.am needs no key. Parts and knowledge search need
    FIRECRAWL_API_KEY (or PERPLEXITY_API_KEY); without one those two methods
    return nothing, exactly as the Null tool does, and the trace records the
    zero rather than hiding it.
    """
    try:
        from adapters.armenian import ArmenianResearch
    except Exception as exc:  # noqa: BLE001 - never break the request path
        import logging

        logging.getLogger(__name__).warning(
            "ArmenianResearch unavailable (%s) — falling back to NullResearchTool", exc
        )
        return NullResearchTool()
    return ArmenianResearch()
