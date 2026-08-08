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

    def for_car(self, *, make="", model_name="", year=None):
        """Cases for exactly this car: same make, same model, and a year range
        covering this one.

        There is deliberately no widening fallback. Matching a Toyota Century
        against Camry cases because both are Toyotas puts another car's
        confirmed fix in front of the diagnostician as evidence, and a fix
        confirmed on a different model is not information about this one --
        labelling it "loose" did not stop it being read as a lead.

        A car we cannot fully identify is the same problem: without the model
        the filter is make-wide, and without the year it spans every generation
        of that model. Both return nothing rather than something adjacent.

        A row with open year bounds still matches -- that is the KB saying the
        case applies to the model regardless of year, not a widened filter.
        """
        if not (make and model_name and year):
            return self.none()

        return self.filter(car_make__iexact=make, car_model__iexact=model_name).matching_car(
            year=year
        )

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
