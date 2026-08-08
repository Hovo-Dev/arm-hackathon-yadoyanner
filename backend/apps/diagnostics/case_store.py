"""The `carmed.ports.CaseStore` seam, backed by this system's own history.

There is no curated Case KB any more. What the diagnostician gets shown is the
set of *past diagnoses this system actually finished* for the same car:

    same make + same model + same year  ->  status COMPLETE  ->  similarity at
    least CASE_MATCH_MIN_SIMILARITY  ->  nearest N

Deliberately not an agent: filtering rows and ranking them by cosine distance
is a database query, and a model in front of it would add cost and latency
without adding judgement.

Two properties this file exists to guarantee:

- **Close enough to be about the same problem.** The vehicle filter cannot tell
  a brake question from a radio one, so without a similarity floor the store
  returned the nearest five whatever their score -- five brake repairs at 0.21
  offered to the diagnostician as "similar past cases you may cite as
  evidence". Below the floor the honest answer is no cases at all.
- **Same car only.** A confirmed outcome is evidence about the car it happened
  on. A Toyota Century question must never be answered out of Camry history --
  same badge, different car, and a fix that fits one is how a wrong part gets
  bought. There is no widening fallback: a car we cannot fully identify gets
  nothing rather than something adjacent.
- **Two tiers, and the gap between them matters.** At or above
  CASE_SERVE_MIN_SIMILARITY a case is marked `verified`, which lets carmed's
  gate answer from it directly and skip the diagnostician, parts and shops
  agents entirely -- re-deriving an answer we already hold costs three model
  calls and half a minute to land in the same place. Below that, down to
  CASE_MATCH_MIN_SIMILARITY, it is only ever context for a fresh diagnosis.

  The bar for serving is set high on purpose. These are our own previous
  answers, not human-confirmed repairs, so anything served is an unreviewed
  model output being handed to the next person verbatim -- and because the new
  request is then stored COMPLETE too, a wrong answer copies itself forward.
  Only a near-identical description of the same problem on the same car earns
  that; everything else goes back through the diagnostician.
"""
from django.conf import settings

from carmed.models import Case, CaseMatch, MatchTier, Vehicle
from carmed.text import classify_area

from .embeddings import embed_text
from .enums import DiagnosticStatus


def _fix_from_summary(summary) -> str:
    """The past answer, rendered as the one-line "what it turned out to be"
    that `Case.fix` is. Built from the stored Answer rather than free text, so
    it can only ever say what that run concluded."""
    summary = summary or {}
    causes = summary.get("causes") or []
    if causes:
        # The top cause alone. Appending the repair steps made this the title of
        # a served answer's cause, which then read as a paragraph.
        return causes[0].get("title") or ""
    return "; ".join(summary.get("repair_steps") or [])


def _parts_from_summary(summary) -> list[str]:
    """Part names the past answer settled on, so a served case still says what
    to buy instead of arriving with an empty parts list."""
    options = ((summary or {}).get("parts") or {}).get("options") or []
    return [name for option in options if (name := option.get("name_en"))]


def to_case(request, *, servable: bool = False) -> Case:
    """DiagnosticRequest -> the shape carmed reasons about.

    `servable` becomes carmed's `verified`, which is the flag its gate checks
    before answering from a case instead of diagnosing afresh. It is set per
    row from the similarity score rather than being a property of the record:
    the same past diagnosis is quotable for a question that matches it almost
    exactly, and merely suggestive for one that does not.

    `urgency` keeps carmed's neutral default rather than being copied from the
    past answer: urgency is a judgement about the car in front of us now, and
    the safety floor re-derives it from the symptom on every run.
    """
    return Case(
        id=str(request.pk),
        symptom=request.symptom_text,
        fix=_fix_from_summary(request.summary),
        vehicle=Vehicle(
            make=request.car_make or None,
            model=request.car_model or None,
            year=request.car_year,
        ),
        # Free and deterministic -- reuses the agentic layer's trilingual
        # symptom table rather than storing a column that would drift from it.
        system_area=classify_area(request.symptom_text),
        part_names=_parts_from_summary(request.summary),
        source_url=None,
        verified=servable,
    )


class PastCaseStore:
    """pgvector similarity search over completed DiagnosticRequests.

    `exclude_request_id` keeps the request being diagnosed out of its own
    evidence. Without it a re-run of a finished request would retrieve itself
    at similarity 1.00 and cite its own previous answer as support for
    repeating it.
    """

    def __init__(self, exclude_request_id=None):
        self.exclude_request_id = exclude_request_id

    def find_similar(self, *, text: str, vehicle: Vehicle, limit: int = 5) -> list[CaseMatch]:
        """Best matches first, all of them completed diagnoses of this exact
        make/model/year. An empty list is normal and fine."""
        from .models import DiagnosticRequest

        if not text:
            return []

        candidates = DiagnosticRequest.objects.completed_for_car(
            make=vehicle.make or "",
            model_name=vehicle.model or "",
            year=vehicle.year,
            exclude_id=self.exclude_request_id,
        )
        # An unidentified car yields `.none()`, which answers this without
        # touching the database -- and returning here skips embedding the text,
        # which is the expensive half of the lookup.
        if not candidates.exists():
            return []

        rows = candidates.similar_to(
            embed_text(text),
            limit=limit,
            min_similarity=settings.CASE_MATCH_MIN_SIMILARITY,
        )

        matches = []
        for row in rows:
            # Embeddings are normalized, so similarity is 1 - cosine distance.
            score = max(0.0, min(1.0, 1 - float(row.distance)))
            matches.append(
                CaseMatch(
                    case=to_case(row, servable=score >= settings.CASE_SERVE_MIN_SIMILARITY),
                    score=score,
                    # Every candidate cleared the same make/model/year filter,
                    # so there is only one tier left to report.
                    tier=MatchTier.EXACT,
                )
            )
        return matches


__all__ = ["PastCaseStore", "to_case", "DiagnosticStatus"]
