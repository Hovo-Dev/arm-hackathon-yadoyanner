"""Typed contracts between the research agent and everything downstream.

The service-finder, parts-availability and past-orders agents consume
`ResearchReport` — so this file is the integration surface. Keep it stable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class VehicleContext(BaseModel):
    """What we know about the car. Everything optional except make/model —
    real users rarely know their engine code."""

    make: str
    model: str
    year: int | None = None
    engine: str | None = None
    transmission: str | None = None
    mileage_km: int | None = None
    vin: str | None = None
    market: str = Field(
        default="AM",
        description="Where the car lives. Drives parts-sourcing region.",
    )

    def label(self) -> str:
        bits = [str(self.year) if self.year else "", self.make, self.model]
        if self.engine:
            bits.append(f"({self.engine})")
        return " ".join(b for b in bits if b)


class CarProblem(BaseModel):
    """The user's complaint, normalised."""

    raw_text: str
    vehicle: VehicleContext
    symptoms: list[str] = Field(default_factory=list)
    obd_codes: list[str] = Field(default_factory=list)
    image_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Text descriptions extracted from user photos by a vision model. "
            "DeepSeek is text-only, so images MUST be captioned upstream."
        ),
    )
    language: str = Field(default="hy", description="hy | ru | en")


class SourceKind(str, Enum):
    OFFICIAL = "official"
    MARKETPLACE = "marketplace"
    FORUM = "forum"
    REFERENCE = "reference"
    WEB = "web"


class Evidence(BaseModel):
    """One retrieved fact with full provenance.

    Nothing reaches the final report without one of these behind it — that is
    what makes the output auditable and what the eval harness scores against.
    """

    source_key: str
    source_kind: SourceKind
    title: str
    snippet: str
    url: str | None = None
    retrieved_at: datetime = Field(default_factory=_now)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def cite(self) -> str:
        return f"[{self.source_key}] {self.title}" + (f" — {self.url}" if self.url else "")


class Cause(BaseModel):
    """A candidate root cause, ranked."""

    title: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[int] = Field(
        default_factory=list,
        description="Indices into ResearchReport.evidence backing this cause.",
    )
    typical_symptoms: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high", "safety-critical"] = "medium"

    basis: Literal["evidence", "standard_diagnosis"] = "evidence"
    """Where this cause comes from.

    'evidence'  — retrieved data about this specific vehicle backs it.
    'standard_diagnosis' — established practice for this symptom on any car
        (a boiling engine means checking thermostat, water pump, radiator and
        fan) with nothing vehicle-specific found.

    Both belong in a report; conflating them does not. Requiring a citation for
    every cause looked rigorous but made the agent drop the textbook
    differential for an overheating Accord and reach for an unrelated recall
    just to have something to cite. A cause the sources cannot speak to is not
    the same as a cause that is wrong.
    """


class PartSuggestion(BaseModel):
    name: str
    part_number: str | None = None
    oem_or_aftermarket: Literal["oem", "aftermarket", "unknown"] = "unknown"
    notes: str | None = None
    evidence_ids: list[int] = Field(default_factory=list)
    # Filled in later by the parts-availability agent, not by research.
    local_availability: str | None = None

    # Set by agent._verify_part_claims. A model can cite a real listing and
    # still describe it wrongly, so citations resolving is not the same as
    # claims being true.
    fitment_status: Literal["verified", "mismatch", "unstated"] = "unstated"
    claim_issues: list[str] = Field(default_factory=list)


class DiagnosticStep(BaseModel):
    step: str
    why: str
    requires_shop: bool = False


class ResearchEvent(BaseModel):
    """One progress update from a streaming run.

    A full run takes 60-120s. Blocking a chat UI for that long reads as broken,
    so the agent emits these as it goes: the user sees which source is being
    consulted and why, and the reasoning becomes legible instead of being a
    spinner followed by a wall of text.
    """

    kind: Literal[
        "intake",        # problem understood
        "round",         # a new tool-calling round began
        "tool_start",    # a tool was invoked
        "tool_result",   # that tool returned N evidence items
        "synthesis",     # writing the report
        "report",        # final report ready
        "error",
    ]
    message: str = Field(description="Human-readable, safe to show in chat")
    data: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=_now)


class Citation(BaseModel):
    """A footnote the UI can render next to a claim."""

    index: int
    label: str
    source: str
    url: str | None = None

    def markdown(self) -> str:
        return f"[{self.index}] {self.label}" + (f" ({self.url})" if self.url else "")


class ResearchReport(BaseModel):
    """The deliverable. Downstream agents key off `parts` and `causes`."""

    problem: CarProblem
    summary: str
    causes: list[Cause] = Field(default_factory=list)
    parts: list[PartSuggestion] = Field(default_factory=list)
    next_steps: list[DiagnosticStep] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    # Observability — the hackathon budget is finite and judged work needs numbers.
    rounds_used: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost_usd: float = 0.0
    duration_s: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    cache_path: Literal["deep_research", "cache_hit", "unknown"] = "deep_research"

    def grounded(self) -> bool:
        """True when every evidence-based cause cites evidence that resolves.

        Causes marked `standard_diagnosis` are exempt by design — they claim no
        vehicle-specific backing, and say so. What this catches is a cause that
        claims evidence and cites an index that is not there.

        Note this is a check on citation integrity, not on truthfulness: a claim
        can cite a real listing and still misdescribe it. `_verify_part_claims`
        covers that separately.
        """
        evidence_based = [c for c in self.causes if c.basis == "evidence"]
        if not self.causes:
            return False
        n = len(self.evidence)
        return all(
            c.evidence_ids and all(0 <= i < n for i in c.evidence_ids)
            for c in evidence_based
        )

    def citations(self) -> list[Citation]:
        """Only the evidence actually cited by a cause or part.

        A run gathers 30+ items; a report typically leans on 5. Showing all of
        them buries the ones that carried the conclusion.
        """
        used: set[int] = set()
        for cause in self.causes:
            used.update(cause.evidence_ids)
        for part in self.parts:
            used.update(part.evidence_ids)
        return [
            Citation(
                index=i,
                label=self.evidence[i].title,
                source=self.evidence[i].source_key,
                url=self.evidence[i].url,
            )
            for i in sorted(used)
            if 0 <= i < len(self.evidence)
        ]

    def to_handoff(self) -> dict[str, Any]:
        """Compact payload for the service-finder / parts agents."""
        return {
            "vehicle": self.problem.vehicle.model_dump(),
            "top_causes": [
                {"title": c.title, "confidence": c.confidence, "severity": c.severity}
                for c in self.causes[:3]
            ],
            "parts": [
                {"name": p.name, "part_number": p.part_number, "type": p.oem_or_aftermarket}
                for p in self.parts
            ],
            "requires_shop": any(s.requires_shop for s in self.next_steps),
            "grounded": self.grounded(),
        }
