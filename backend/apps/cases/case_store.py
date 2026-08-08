"""The Case KB behind `carmed.ports.CaseStore` -- the agentic layer's case gate.

Deliberately not an agent: embedding a sentence and ranking rows by cosine
distance is a database query, and a model in front of it would add cost and
latency without adding judgement.

Duck-typed against the Protocol, not inherited from it -- carmed asks only that
the method signature matches.
"""
from carmed.models import Case, CaseMatch, MatchTier, Vehicle
from carmed.text import classify_area

from .embeddings import embed_text
from .models import CaseRecord


def to_case(record: CaseRecord) -> Case:
    """CaseRecord -> the shape carmed reasons about.

    Two fields the KB has no column for, and what we do instead of inventing one:

    - `urgency` keeps carmed's neutral default. Guessing "do not drive" from a
      fix description would be a fabrication; escalation is handled honestly by
      the safety floor (settings.CARMED_SAFETY_FLOOR), which keys on the
      symptom text rather than on us.
    - `verified` is True because every row in this table today is curated seed
      data. When a provenance/split column lands, filter it *here* (or better,
      in the queryset) -- never at the call site, and never by remembering to.
    """
    return Case(
        id=str(record.pk),
        symptom=record.symptom_text,
        fix=record.confirmed_fix,
        vehicle=Vehicle(
            make=record.car_make or None,
            model=record.car_model or None,
            year=record.car_year_start,
        ),
        # Free and deterministic -- reuses the agentic layer's trilingual
        # symptom table rather than adding a column that would drift from it.
        system_area=classify_area(record.symptom_text),
        part_names=list(record.parts_named or []),
        source_url=None,
        verified=True,
    )


class DjangoCaseStore:
    """pgvector similarity search over CaseRecord."""

    def find_similar(self, *, text: str, vehicle: Vehicle, limit: int = 5) -> list[CaseMatch]:
        """Best matches first. An empty list is normal and fine.

        Filters by vehicle *before* ranking, deliberately. With a global HNSW
        index Postgres applies WHERE after the ANN traversal, so ranking first
        and filtering second would quietly destroy recall -- and would let a
        Camry question match a BMW case.
        """
        if not text:
            return []

        candidates, tier = CaseRecord.objects.tiered_for_car(
            make=vehicle.make or "",
            model_name=vehicle.model or "",
            year=vehicle.year,
        )
        rows = candidates.similar_to(embed_text(text), limit=limit)

        return [
            CaseMatch(
                case=to_case(row),
                # Embeddings are normalized, so similarity is 1 - cosine distance.
                score=max(0.0, min(1.0, 1 - float(row.distance))),
                tier=MatchTier(tier),
            )
            for row in rows
        ]
