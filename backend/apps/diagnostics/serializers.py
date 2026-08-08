from rest_framework import serializers

from apps.cases.serializers import CaseRecordSerializer

from .models import (
    DiagnosticCaseMatch,
    DiagnosticImage,
    DiagnosticMessage,
    DiagnosticRequest,
    DiagnosticRun,
)


class DiagnosticImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiagnosticImage
        fields = ["id", "request", "image", "image_type", "decoded_data", "created_at"]
        read_only_fields = ["id", "decoded_data", "created_at"]


class DiagnosticMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiagnosticMessage
        fields = ["id", "request", "role", "content", "created_at"]
        read_only_fields = ["id", "created_at"]


class DiagnosticRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiagnosticRun
        fields = ["id", "answer", "created_at"]
        read_only_fields = fields


class DiagnosticCaseMatchSerializer(serializers.ModelSerializer):
    case = CaseRecordSerializer(read_only=True)

    class Meta:
        model = DiagnosticCaseMatch
        fields = ["id", "case", "confidence", "created_at"]
        read_only_fields = fields


class DiagnoseInputSerializer(serializers.Serializer):
    """Input to the stateless POST /api/diagnose/ -- one pass through the
    agentic layer with nothing persisted. Deliberately not a ModelSerializer:
    there is no row behind it."""

    text = serializers.CharField()
    car_make = serializers.CharField(required=False, allow_blank=True, default="")
    car_model = serializers.CharField(required=False, allow_blank=True, default="")
    car_year = serializers.IntegerField(required=False, allow_null=True, default=None)
    vin = serializers.CharField(required=False, allow_blank=True, default="")
    city = serializers.CharField(required=False, allow_blank=True, default="")
    mileage_km = serializers.IntegerField(required=False, allow_null=True, default=None)


class DiagnosticRequestSerializer(serializers.ModelSerializer):
    # Write-only and not a model field: it seeds the first chat message on
    # create (see DiagnosticRequestViewSet._record_initial_turn). The
    # request's actual persisted state of "what's wrong" is symptom_text,
    # which starts as this text and gets rewritten every turn after.
    raw_text = serializers.CharField(write_only=True, required=False, allow_blank=True, default="")

    images = DiagnosticImageSerializer(many=True, read_only=True)
    messages = DiagnosticMessageSerializer(many=True, read_only=True)
    case_matches = DiagnosticCaseMatchSerializer(many=True, read_only=True)
    runs = DiagnosticRunSerializer(many=True, read_only=True)

    class Meta:
        model = DiagnosticRequest
        fields = [
            "id",
            "created_at",
            "updated_at",
            "car_make",
            "car_model",
            "car_year",
            "vin",
            "raw_text",
            "symptom_text",
            "status",
            "summary",
            "trace",
            "runs",
            "images",
            "messages",
            "case_matches",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "symptom_text",
            "status",
            "summary",
            "trace",
            "runs",
            "images",
            "messages",
            "case_matches",
        ]
