"""All data contracts for the agentic layer.

Everything crossing a boundary -- into the layer, out of it, or between an
agent and a tool -- is defined here. They are pydantic models, so the Django
layer can go straight to ``.model_dump()`` in a serializer and
``Model(**payload)`` on the way back.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Lang(StrEnum):
    HY = "hy"
    RU = "ru"
    EN = "en"
    UNKNOWN = "xx"


class SystemArea(StrEnum):
    BRAKES = "brakes"
    STEERING = "steering"
    SUSPENSION = "suspension"
    AIRBAG = "airbag"
    ENGINE = "engine"
    TRANSMISSION = "transmission"
    ELECTRICAL = "electrical"
    COOLING = "cooling"
    EXHAUST = "exhaust"
    HVAC = "hvac"
    BODY = "body"
    UNKNOWN = "unknown"


SAFETY_CRITICAL = {
    SystemArea.BRAKES,
    SystemArea.STEERING,
    SystemArea.SUSPENSION,
    SystemArea.AIRBAG,
}


class Urgency(IntEnum):
    """Ordered, so escalating is ``max()``. Do not reorder."""

    DRIVE_OK = 0
    FIX_THIS_WEEK = 1
    FIX_NOW = 2
    DO_NOT_DRIVE = 3

    @property
    def label(self) -> str:
        return {0: "drive it", 1: "fix this week", 2: "fix now", 3: "do not drive"}[
            int(self)
        ]


class SpecRegion(StrEnum):
    US = "US"
    CANADA = "CA"
    MEXICO = "MX"
    EU = "EU"
    JAPAN = "JP"
    KOREA = "KR"
    UNKNOWN = "??"


class MatchTier(StrEnum):
    """How tightly a stored case matched the car.

    The case store sets this. It matters because a loose match may inform a
    diagnosis but must never be served as a cached answer -- "some Toyota had
    this once" is not an answer about this car.
    """

    EXACT = "exact"  # make + model + year (+/-1) + engine
    NEAR = "near"  # make + model, nearby years
    LOOSE = "loose"  # make only, or symptom only


class Intent(StrEnum):
    DIAGNOSE = "diagnose"
    PART_LOOKUP = "part_lookup"
    SHOP_LOOKUP = "shop_lookup"
    SAFETY_CHECK = "safety_check"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    NEEDS_CLARIFICATION = "needs_clarification"
    REFUSED_BAD_VIN = "refused_bad_vin"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class Vehicle(BaseModel):
    make: str | None = None
    model: str | None = None
    year: int | None = None
    engine_l: float | None = None
    vin: str | None = None
    #: Set by this layer from the VIN checksum. None when no VIN was given.
    vin_valid: bool | None = None
    #: Where the car was built for. Drives the US-vs-Euro parts check.
    region: SpecRegion = SpecRegion.UNKNOWN

    def describe(self) -> str:
        bits = [str(x) for x in (self.year, self.make, self.model) if x]
        if self.engine_l:
            bits.append(f"{self.engine_l}L")
        return " ".join(bits) or "unknown vehicle"


class Query(BaseModel):
    """The input to this component."""

    #: Canonical, normalized English. This is the *retrieval* key: it is what
    #: gets embedded and compared against Case.symptom, so it has to be in the
    #: language the case base is written in rather than the one the owner used.
    text: str = ""
    #: What the owner actually typed, in whatever language and script they
    #: typed it. Optional; when blank, `asked` falls back to `text` and this
    #: component behaves exactly as it did before the field existed.
    #:
    #: The two are separate because `text` is lossy on purpose -- producing it
    #: means translating terse Latin-script Armenian ("matory ercnuma"), which
    #: is a step that can and does get the complaint wrong. Everything that
    #: *reasons* should read `asked`, so a bad normalization costs a missed
    #: case match instead of a confident diagnosis of a problem nobody has.
    raw_text: str = ""
    vehicle: Vehicle = Field(default_factory=Vehicle)
    city: str = "Yerevan"
    mileage_km: int | None = None

    @property
    def asked(self) -> str:
        """The owner's own words where we have them, the normalized text
        otherwise. Use this for routing, language detection, the symptom
        keyword tables and anything shown to an agent -- all of them are
        better on the original, and carmed.text is transliteration-aware.
        Retrieval is the one caller that wants `text` instead."""
        return self.raw_text or self.text


# ---------------------------------------------------------------------------
# What the ports return
# ---------------------------------------------------------------------------


class Money(BaseModel):
    amount: float
    currency: str = "AMD"

    def __str__(self) -> str:
        return f"{self.amount:,.0f} {self.currency}"


class Case(BaseModel):
    """A past problem with a known outcome."""

    id: str
    symptom: str
    fix: str
    vehicle: Vehicle = Field(default_factory=Vehicle)
    system_area: SystemArea = SystemArea.UNKNOWN
    part_names: list[str] = Field(default_factory=list)
    urgency: Urgency = Urgency.FIX_THIS_WEEK
    source_url: str | None = None
    #: Only human-confirmed outcomes may be served as a cached answer.
    verified: bool = True


class CaseMatch(BaseModel):
    case: Case
    score: float  # cosine similarity, 0..1
    tier: MatchTier = MatchTier.LOOSE


class KnowledgeSnippet(BaseModel):
    id: str
    text: str
    source_url: str
    title: str = ""
    kind: str = "forum"  # forum | manual | catalog


class PartListing(BaseModel):
    id: str
    title: str
    url: str
    price: Money | None = None
    condition: str | None = None  # new | used | dismantler
    source: str = "list.am"
    #: Which market the part is for, when the source can tell.
    region: SpecRegion = SpecRegion.UNKNOWN


class Shop(BaseModel):
    id: str
    name: str
    phone: str
    area: str | None = None
    specialties: list[SystemArea] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# What the agents produce
# ---------------------------------------------------------------------------


class Cause(BaseModel):
    title: str
    explanation: str = ""
    system_area: SystemArea = SystemArea.UNKNOWN
    likely_parts: list[str] = Field(default_factory=list)
    #: Ids of what supports this, as "case:C1" / "doc:D2". Never a URL.
    evidence: list[str] = Field(default_factory=list)
    confidence: str = "unclear"  # strong | moderate | weak | unclear


class Diagnosis(BaseModel):
    """Agent 1 output (also built without a model on a case-gate hit)."""

    causes: list[Cause] = Field(default_factory=list, max_length=3)
    urgency: Urgency = Urgency.FIX_THIS_WEEK
    urgency_reason: str = ""
    repair_steps: list[str] = Field(default_factory=list)
    abstain: bool = False
    clarifying_question: str | None = None


class FitmentWarning(BaseModel):
    """US-spec car, Euro-market part. The failure this product exists for."""

    message: str
    severity: str = "check"  # blocking | check
    listing_ids: list[str] = Field(default_factory=list)


class PartOption(BaseModel):
    name_en: str
    #: The same part as written by different sellers, across languages.
    aliases: list[str] = Field(default_factory=list)
    part_numbers: list[str] = Field(default_factory=list)
    #: Ids only -- deliberately no price field here. The agent sees prices (it
    #: needs them to spot copies) but has nowhere to write one, so every
    #: displayed price comes from the stored listing record.
    listing_ids: list[str] = Field(default_factory=list)
    suspicious_listing_ids: list[str] = Field(default_factory=list)


class PartsResult(BaseModel):
    """Agent 2 output."""

    options: list[PartOption] = Field(default_factory=list)
    fitment_warnings: list[FitmentWarning] = Field(default_factory=list)


class IntentDecision(BaseModel):
    """Router output."""

    intent: Intent = Intent.DIAGNOSE
    reason: str = ""
    named_parts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This is not a mechanic's diagnosis. Confirm with a professional before "
    "buying parts or driving a car you believe is unsafe."
)


class Trace(BaseModel):
    """A record of what actually ran, for verifying behaviour and cost.

    Separated into three lists because they answer three different questions:
    which steps executed, what was paid for, and what was fetched.
    """

    #: Steps executed, in order. e.g. ["vehicle", "route", "gate", ...]
    steps: list[str] = Field(default_factory=list)
    #: Agents that made a model call. Length is the bill for this request.
    llm_calls: list[str] = Field(default_factory=list)
    #: Every call out to your ports, as "name->result_count".
    lookups: list[str] = Field(default_factory=list)
    #: Human-readable notes: decisions, warnings, failures.
    notes: list[str] = Field(default_factory=list)

    @property
    def cost(self) -> int:
        """Number of model calls this request made."""
        return len(self.llm_calls)

    def ran(self, step: str) -> bool:
        return step in self.steps

    def called(self, agent: str) -> bool:
        return agent in self.llm_calls

    def looked_up(self, name: str) -> bool:
        return any(x.split("->")[0] == name for x in self.lookups)


class Answer(BaseModel):
    """The output of this component."""

    status: AnswerStatus = AnswerStatus.ANSWERED
    vehicle: Vehicle = Field(default_factory=Vehicle)
    intent: Intent = Intent.DIAGNOSE

    causes: list[Cause] = Field(default_factory=list)
    urgency: Urgency = Urgency.FIX_THIS_WEEK
    urgency_reason: str = ""
    repair_steps: list[str] = Field(default_factory=list)

    parts: PartsResult = Field(default_factory=PartsResult)
    shops: list[Shop] = Field(default_factory=list)
    #: Rendering source of truth: ``PartOption.listing_ids`` resolve here.
    listings: dict[str, PartListing] = Field(default_factory=dict)

    #: Set when status is NEEDS_CLARIFICATION or REFUSED_BAD_VIN.
    message: str = ""

    from_cache: bool = False
    #: References an agent produced that could not be resolved, and were
    #: dropped. Should be zero; non-zero means investigate.
    dropped_refs: int = 0
    #: What ran, what it cost, what it fetched. See :class:`Trace`.
    trace: Trace = Field(default_factory=Trace)
    disclaimer: str = DISCLAIMER
