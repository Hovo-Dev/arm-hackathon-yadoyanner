"""Car maintenance assistant -- agentic layer.

Input a :class:`Query`, get an :class:`Answer`. Supply two dependencies:
a :class:`CaseStore` (pgvector, in Django) and a :class:`ResearchTool`.

    from carmed import Query, Vehicle, run

    answer = run(
        Query(text="clicking when I turn left",
              vehicle=Vehicle(make="Toyota", model="Camry", year=2011)),
        research=my_research_tool,
        case_store=my_case_store,
        model=chat_model,
    )
    print(answer.urgency.label, [c.title for c in answer.causes])

See README.md for the Django integration.
"""

from carmed.graph import build, run
from carmed.models import (
    Answer,
    AnswerStatus,
    Case,
    CaseMatch,
    Cause,
    Diagnosis,
    FitmentWarning,
    Intent,
    KnowledgeSnippet,
    Lang,
    MatchTier,
    Money,
    PartListing,
    PartOption,
    PartsResult,
    Query,
    Shop,
    SpecRegion,
    SystemArea,
    Trace,
    Urgency,
    Vehicle,
)
from carmed.ports import CaseStore, ResearchTool

__all__ = [
    "run", "build",
    "Query", "Vehicle", "Answer", "AnswerStatus",
    "Case", "CaseMatch", "MatchTier",
    "Cause", "Diagnosis", "Intent",
    "PartListing", "PartOption", "PartsResult", "FitmentWarning",
    "KnowledgeSnippet", "Shop", "Money",
    "Lang", "SystemArea", "Urgency", "SpecRegion", "Trace",
    "CaseStore", "ResearchTool",
]  # fmt: skip
