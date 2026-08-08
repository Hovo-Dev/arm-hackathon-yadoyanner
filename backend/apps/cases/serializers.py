from rest_framework import serializers

from .models import CaseRecord


class CaseRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = CaseRecord
        fields = [
            "id",
            "car_make",
            "car_model",
            "car_year_start",
            "car_year_end",
            "symptom_text",
            "confirmed_fix",
            "parts_named",
            "created_at",
        ]
        read_only_fields = fields


class CaseSearchResultSerializer(CaseRecordSerializer):
    distance = serializers.FloatField(read_only=True)

    class Meta(CaseRecordSerializer.Meta):
        fields = CaseRecordSerializer.Meta.fields + ["distance"]
        read_only_fields = fields


class CaseSearchRequestSerializer(serializers.Serializer):
    query_text = serializers.CharField()
    car_make = serializers.CharField(required=False, allow_blank=True, default="")
    car_model = serializers.CharField(required=False, allow_blank=True, default="")
    car_year = serializers.IntegerField(required=False, allow_null=True, default=None)
    limit = serializers.IntegerField(required=False, default=5, min_value=1, max_value=20)
