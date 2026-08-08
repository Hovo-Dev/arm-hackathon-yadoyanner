"""The two things the Django layer must provide.

This layer owns no database, no embedding model and no HTTP client. It asks
these two Protocols for everything it cannot compute itself.

Implement them anywhere (Django services, a scraper package, a test double)
and pass instances to ``carmed.run``. Duck typing is enough -- no base class
to inherit, no registration.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from carmed.models import (
    CaseMatch,
    KnowledgeSnippet,
    PartListing,
    Shop,
    SystemArea,
    Vehicle,
)


@runtime_checkable
class CaseStore(Protocol):
    """Similarity search over past solved cases. Backed by pgvector in Django.

    This is the "case gate" -- deliberately not an agent. Embedding a sentence
    and ranking rows by cosine distance is a database query, and putting a
    model in front of it would add latency and cost without adding judgement.

    The Django implementation owns the embedding call and the SQL. It should:

    - embed ``text``
    - filter by vehicle metadata before ranking (make/model/year/engine), so a
      Camry question cannot match a BMW case
    - set ``tier`` to say how tightly it filtered, because only EXACT and NEAR
      matches are eligible to be served as a cached answer
    - never return unverified or held-out rows
    """

    def find_similar(
        self, *, text: str, vehicle: Vehicle, limit: int = 5
    ) -> list[CaseMatch]:
        """Best matches first. Returning an empty list is normal and fine."""
        ...


@runtime_checkable
class ResearchTool(Protocol):
    """Retrieval over list.am, forums, manuals and catalogs.

    The agents call these themselves as LangGraph tools -- the graph does not
    fetch on their behalf and then hand results over. That is what lets the
    diagnostician decide to search twice with different wording, or not at all.

    Two hard requirements, because grounding depends on them:

    1. Every record needs a stable, unique ``id``. Agents cite ids and the
       final answer renders prices and links from the record, so an id that
       changes between calls silently breaks the citation.
    2. ``url`` / ``source_url`` must be real. Returning fewer results, or
       none, is always acceptable. Fabricating one is not.
    """

    def search_knowledge(
        self, *, query: str, vehicle: Vehicle, limit: int = 6
    ) -> list[KnowledgeSnippet]:
        """Manual sections, catalog pages and forum threads, best first."""
        ...

    def search_listings(
        self, *, part_terms: list[str], vehicle: Vehicle, limit: int = 20
    ) -> list[PartListing]:
        """Parts for sale. ``part_terms`` arrives multilingual by design --
        sellers write the same part in Armenian, Russian and English."""
        ...

    def search_shops(
        self,
        *,
        make: str | None,
        system_area: SystemArea,
        city: str,
        limit: int = 5,
    ) -> list[Shop]:
        """Repair shops. Omit any record whose phone number you cannot verify."""
        ...
