from django.conf import settings
from django.db import models
from pgvector.django import VectorField

from apps.cases.models import CaseRecord

from .enums import DiagnosticImageType, DiagnosticStatus, MessageRole
from .querysets import DiagnosticMessageQuerySet


class DiagnosticRequest(models.Model):
    """One user session: text/voice description plus any photos, working
    towards the four-block output in spec section 4. This is the "actual
    case development" side, separate from the read-only Case KB."""

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    car_make = models.CharField(max_length=100, blank=True)
    car_model = models.CharField(max_length=100, blank=True)
    car_year = models.PositiveSmallIntegerField(null=True, blank=True)
    vin = models.CharField(max_length=17, blank=True)

    symptom_text = models.TextField(
        blank=True,
        help_text=(
            "Rolling summary of the whole conversation so far, refreshed after every user turn. "
            "Embedded and matched against CaseRecord.symptom_text -- keeping both fields the same "
            "shape keeps the two embedding spaces comparable. There is no separate 'original "
            "text' field: the first turn's content is just this field's first version."
        ),
    )
    embedding = VectorField(dimensions=settings.EMBEDDING_DIM, null=True, blank=True)

    status = models.CharField(max_length=20, choices=DiagnosticStatus.choices, default=DiagnosticStatus.PENDING)
    summary = models.JSONField(null=True, blank=True)

    trace = models.JSONField(
        default=list,
        blank=True,
        help_text=(
            "The pipeline events from the last run -- the `step`/`llm`/`lookup`/`note` stream "
            "the UI replays to show which agents ran and what they found. Persisted because "
            "the SSE feed is live-only: without this the pipeline view would be blank for "
            "every request except the one currently being streamed."
        ),
    )

    def __str__(self):
        return f"Request #{self.pk} ({self.car_make} {self.car_model})".strip()


class DiagnosticRun(models.Model):
    """One completed pass of the agentic graph, kept as a row so the answers
    accumulate down the conversation instead of overwriting each other.

    DiagnosticRequest.summary still holds the latest answer -- it is what the
    status logic and the `finalize` endpoint read. This is the history behind
    it: every turn re-runs the graph, and the answer to "does it only happen
    when cold?" is a different answer, not a correction of the previous one."""

    request = models.ForeignKey(DiagnosticRequest, on_delete=models.CASCADE, related_name="runs")
    answer = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"run for request #{self.request_id} at {self.created_at:%Y-%m-%d %H:%M}"


class DiagnosticCaseMatch(models.Model):
    """A CaseRecord whose similarity to a DiagnosticRequest's embedding
    cleared CASE_MATCH_CONFIDENCE_THRESHOLD. Unlike a single best match,
    several KB cases can independently score above the bar, so this is a
    row per (request, case) pair rather than a field on DiagnosticRequest."""

    request = models.ForeignKey(DiagnosticRequest, on_delete=models.CASCADE, related_name="case_matches")
    case = models.ForeignKey(CaseRecord, on_delete=models.CASCADE, related_name="diagnostic_matches")
    confidence = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-confidence"]
        constraints = [
            models.UniqueConstraint(fields=["request", "case"], name="unique_diagnostic_case_match"),
        ]

    def __str__(self):
        return f"case #{self.case_id} matched to request #{self.request_id} ({self.confidence:.2f})"


class DiagnosticMessage(models.Model):
    """One turn in the conversation for a DiagnosticRequest -- lets the user
    ask follow-up questions in the same session instead of a single one-shot
    description. No per-message embedding: retrieval happens once per turn
    at the request level, against DiagnosticRequest.symptom_text/embedding
    (see services.refresh_symptom_and_matches)."""

    request = models.ForeignKey(DiagnosticRequest, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=MessageRole.choices)
    content = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    objects = DiagnosticMessageQuerySet.as_manager()

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role} message for request #{self.request_id}"


def diagnostic_image_upload_to(instance, filename):
    return f"diagnostics/{instance.request_id}/{instance.image_type}/{filename}"


class DiagnosticImage(models.Model):
    request = models.ForeignKey(DiagnosticRequest, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to=diagnostic_image_upload_to)
    image_type = models.CharField(max_length=20, choices=DiagnosticImageType.choices)
    decoded_data = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.image_type} for request #{self.request_id}"
