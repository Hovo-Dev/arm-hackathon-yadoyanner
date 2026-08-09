from django.db import models
from pgvector.django import CosineDistance

from .enums import DiagnosticStatus


class DiagnosticMessageQuerySet(models.QuerySet):
    def recent(self, limit=10):
        """Last `limit` turns, newest first. Reverse before handing to the LLM
        so the conversation reads oldest-to-newest in the prompt."""
        return self.order_by("-created_at")[:limit]


class DiagnosticRequestQuerySet(models.QuerySet):
    def completed_for_car(self, *, make="", model_name="", year=None, exclude_id=None):
        """Finished diagnoses of exactly this car, ready to be ranked.

        Three filters, all required:

        - **same make, model and year.** No widening fallback. Dropping to
          make-only when the model has no history is how a Toyota Century
          question ends up being answered out of Camry cases, and a filter that
          spans every generation of a model is barely narrower. A car we cannot
          fully identify returns nothing rather than something adjacent.
        - **COMPLETE only.** An unfinished request is a conversation still
          being clarified; its symptom summary describes a problem nobody has
          got to the bottom of yet, so it is not evidence about anything.
        - **an embedding.** Rows without one cannot be ranked, and pgvector
          would order them arbitrarily rather than excluding them.

        `exclude_id` keeps a request out of its own evidence on a re-run.
        """
        if not (make and model_name and year):
            return self.none()

        qs = self.filter(
            car_make__iexact=make,
            car_model__iexact=model_name,
            car_year=year,
            status=DiagnosticStatus.COMPLETE,
            embedding__isnull=False,
        )
        return qs.exclude(pk=exclude_id) if exclude_id else qs

    def similar_to(self, embedding, limit=5, min_similarity=None):
        """Closest first by cosine distance, no further than `min_similarity`.

        Embeddings are normalized, so similarity is 1 - distance; the floor is
        expressed as a distance ceiling to let pgvector's index do the work
        rather than fetching rows only to discard them.
        """
        qs = self.filter(embedding__isnull=False).annotate(
            distance=CosineDistance("embedding", embedding)
        )
        if min_similarity is not None:
            qs = qs.filter(distance__lte=1 - min_similarity)
        return qs.order_by("distance")[:limit]
