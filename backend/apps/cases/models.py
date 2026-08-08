from django.conf import settings
from django.db import models
from pgvector.django import HnswIndex, VectorField

from .querysets import CaseRecordQuerySet


class CaseRecord(models.Model):
    """A single solved forum thread: confirmed symptom -> confirmed fix.
    This is the Case Knowledge Base that Agent 4 checks before anything
    else runs (cache-first gate, spec section 3)."""

    car_make = models.CharField(max_length=100, blank=True)
    car_model = models.CharField(max_length=100, blank=True)
    car_year_start = models.PositiveSmallIntegerField(null=True, blank=True)
    car_year_end = models.PositiveSmallIntegerField(null=True, blank=True)

    symptom_text = models.TextField()
    confirmed_fix = models.TextField()
    parts_named = models.JSONField(default=list, blank=True)

    embedding = VectorField(dimensions=settings.EMBEDDING_DIM, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = CaseRecordQuerySet.as_manager()

    class Meta:
        indexes = [
            HnswIndex(
                name="case_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return f"Case #{self.pk} ({self.car_make} {self.car_model})"
