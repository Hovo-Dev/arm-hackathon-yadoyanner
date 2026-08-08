from rest_framework import response, status, views, viewsets

from .embeddings import embed_text
from .models import CaseRecord
from .serializers import (
    CaseRecordSerializer,
    CaseSearchRequestSerializer,
    CaseSearchResultSerializer,
)


class CaseRecordViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only browsing of the Case KB."""

    serializer_class = CaseRecordSerializer

    def get_queryset(self):
        return CaseRecord.objects.order_by("-created_at")


class CaseSearchView(views.APIView):
    """Agent 4 -- Case Gate. Metadata-filters the KB by car, then ranks by
    embedding similarity to the user's symptom text. A confident top match
    here means A1-A3 and the Deep Research Engine never have to run."""

    def post(self, request):
        params = CaseSearchRequestSerializer(data=request.data)
        params.is_valid(raise_exception=True)
        data = params.validated_data

        candidates = CaseRecord.objects.matching_car(
            make=data["car_make"], model_name=data["car_model"], year=data["car_year"]
        )
        query_embedding = embed_text(data["query_text"])
        results = candidates.similar_to(query_embedding, limit=data["limit"])

        return response.Response(
            CaseSearchResultSerializer(results, many=True).data, status=status.HTTP_200_OK
        )
