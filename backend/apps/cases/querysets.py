from django.db import models
from pgvector.django import CosineDistance


class CaseRecordQuerySet(models.QuerySet):
    def matching_car(self, *, make="", model_name="", year=None):
        qs = self
        if make:
            qs = qs.filter(car_make__iexact=make)
        if model_name:
            qs = qs.filter(car_model__iexact=model_name)
        if year:
            qs = qs.filter(
                models.Q(car_year_start__isnull=True) | models.Q(car_year_start__lte=year)
            ).filter(models.Q(car_year_end__isnull=True) | models.Q(car_year_end__gte=year))
        return qs

    def tiered_for_car(self, *, make="", model_name="", year=None):
        """Narrow to the tightest vehicle filter that still has rows, and say
        how tightly we ended up filtering.

        Returns `(queryset, tier)` where tier is "exact" | "near" | "loose" --
        the string values of carmed's MatchTier, so this app never has to
        import the agentic layer. The tier is what decides whether a match may
        be *served* as an answer or only used as context: "some Toyota had this
        once" is not an answer about this car.

        Falls back tighter -> looser, because an over-tight filter that returns
        nothing is worse than a loose match honestly labelled as loose.
        """
        if not make:
            return self, "loose"

        by_make = self.filter(car_make__iexact=make)
        if not model_name:
            return by_make, "loose"

        by_model = by_make.filter(car_model__iexact=model_name)
        if not by_model.exists():
            return by_make, "loose"
        if not year:
            return by_model, "near"

        by_year = by_model.matching_car(year=year)
        return (by_year, "exact") if by_year.exists() else (by_model, "near")

    def similar_to(self, embedding, limit=5):
        return (
            self.filter(embedding__isnull=False)
            .annotate(distance=CosineDistance("embedding", embedding))
            .order_by("distance")[:limit]
        )

    def confident_matches(self, embedding, threshold, limit=20):
        """Cases whose cosine similarity to `embedding` clears `threshold`.
        Embeddings are normalized, so similarity == 1 - cosine distance;
        expressed as a distance filter to let pgvector's index do the work."""
        max_distance = 1 - threshold
        return (
            self.filter(embedding__isnull=False)
            .annotate(distance=CosineDistance("embedding", embedding))
            .filter(distance__lte=max_distance)
            .order_by("distance")[:limit]
        )
