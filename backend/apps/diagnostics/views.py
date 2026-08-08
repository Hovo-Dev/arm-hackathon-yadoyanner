import json

from django.conf import settings
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import parsers, response, status, views, viewsets
from rest_framework.decorators import action

from carmed import AnswerStatus, Query, Vehicle

from .agentic import run_query
from .enums import DiagnosticStatus, MessageRole
from .llm import stream_reply
from .models import DiagnosticImage, DiagnosticMessage, DiagnosticRequest
from .serializers import (
    DiagnoseInputSerializer,
    DiagnosticImageSerializer,
    DiagnosticMessageSerializer,
    DiagnosticRequestSerializer,
)
from .services import build_context, build_final_summary, refresh_symptom_and_matches

# A clarifying question is the one outcome that isn't terminal -- the user still
# owes us an answer, so the request stays open. Everything else is done with.
_TERMINAL_STATUSES = {
    AnswerStatus.ANSWERED,
    AnswerStatus.ABSTAINED,
    AnswerStatus.REFUSED_BAD_VIN,
}


class DiagnosticRequestViewSet(viewsets.ModelViewSet):
    """Intake for a user's case: text/voice transcript, car metadata, and
    (via DiagnosticImageViewSet) any photos."""

    queryset = (
        DiagnosticRequest.objects.all()
        .prefetch_related("images", "messages", "case_matches__case")
        .order_by("-created_at")
    )
    serializer_class = DiagnosticRequestSerializer

    def perform_create(self, serializer):
        # raw_text is write-only and not a model field (see the serializer) --
        # pull it out before .save() so ModelSerializer.create() doesn't try
        # to pass it to DiagnosticRequest.objects.create().
        raw_text = serializer.validated_data.pop("raw_text", "").strip()
        instance = serializer.save()
        if raw_text:
            self._create_initial_reply(instance, raw_text)

    def _create_initial_reply(self, instance, raw_text):
        """The description submitted on "Start diagnosis" becomes the first
        chat turn. refresh_symptom_and_matches summarizes it into
        symptom_text, embeds that, and runs the Agent 4 case-gate match
        before the assistant's first reply is generated, so the reply (and
        every later turn) is grounded in whatever the KB currently matches."""
        DiagnosticMessage.objects.create(request=instance, role=MessageRole.USER, content=raw_text)
        refresh_symptom_and_matches(instance)

        try:
            reply_text = "".join(stream_reply(build_context(instance)))
        except Exception:
            # A flaky LLM call shouldn't fail case creation -- the user still
            # gets their request, case matches, and can retry via the chat box.
            return

        if reply_text:
            DiagnosticMessage.objects.create(request=instance, role=MessageRole.ASSISTANT, content=reply_text)

    @action(detail=True, methods=["post"])
    def finalize(self, request, pk=None):
        """Run the agentic layer over the conversation so far and store the
        resulting Answer (causes, urgency, parts, shops, and the trace of what
        ran) on `summary`."""
        instance = self.get_object()
        summary = build_final_summary(instance)
        instance.summary = summary

        if AnswerStatus(summary["status"]) in _TERMINAL_STATUSES:
            instance.status = DiagnosticStatus.COMPLETE
        instance.save(update_fields=["summary", "status", "updated_at"])
        return response.Response(self.get_serializer(instance).data)


class DiagnoseView(views.APIView):
    """Stateless one-shot run of the agentic layer: text plus car in, a full
    Answer out, nothing written to the database.

    Sits alongside the conversational flow rather than replacing it -- it is the
    fastest way to exercise routing, the case gate and the trace end to end, and
    it is what a caller wants when there is no conversation to have."""

    def post(self, request):
        params = DiagnoseInputSerializer(data=request.data)
        params.is_valid(raise_exception=True)
        data = params.validated_data

        query = Query(
            text=data["text"],
            vehicle=Vehicle(
                make=data["car_make"] or None,
                model=data["car_model"] or None,
                year=data["car_year"],
                vin=data["vin"] or None,
            ),
            mileage_km=data["mileage_km"],
            city=data["city"] or settings.CARMED_DEFAULT_CITY,
        )
        return response.Response(run_query(query).model_dump(mode="json"))


class DiagnosticImageViewSet(viewsets.ModelViewSet):
    """Upload endpoint for the four image use cases in spec section 2:
    VIN plate, old part, dashboard light, damage/leak."""

    queryset = DiagnosticImage.objects.all().order_by("-created_at")
    serializer_class = DiagnosticImageSerializer
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]


class DiagnosticMessageViewSet(viewsets.ModelViewSet):
    """One turn per row, so a DiagnosticRequest can hold a back-and-forth
    conversation."""

    queryset = DiagnosticMessage.objects.all().order_by("created_at")
    serializer_class = DiagnosticMessageSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        if instance.role == MessageRole.USER:
            refresh_symptom_and_matches(instance.request)


class DiagnosticMessageStreamView(views.APIView):
    """The actual chat interaction: post a user message, get the assistant's
    reply streamed back over SSE as the LLM generates it. Persists the user
    turn immediately and the assistant turn once the stream completes, so a
    dropped connection can't leave the conversation half-written."""

    def post(self, request, request_id):
        diagnostic_request = get_object_or_404(DiagnosticRequest, pk=request_id)
        content = (request.data.get("content") or "").strip()
        if not content:
            return response.Response(
                {"content": ["This field is required."]}, status=status.HTTP_400_BAD_REQUEST
            )

        DiagnosticMessage.objects.create(request=diagnostic_request, role=MessageRole.USER, content=content)

        # Re-summarize the conversation (now including this turn) and re-run
        # the case-gate match before replying, so every reply reflects the
        # latest understanding of the symptom -- not just what was said first.
        refresh_symptom_and_matches(diagnostic_request)

        llm_messages = build_context(diagnostic_request)

        return StreamingHttpResponse(
            self._sse_stream(diagnostic_request, llm_messages),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _sse_stream(self, diagnostic_request, llm_messages):
        chunks = []
        for delta in stream_reply(llm_messages):
            chunks.append(delta)
            yield f"data: {json.dumps({'delta': delta})}\n\n"

        full_text = "".join(chunks)
        if full_text:
            assistant_message = DiagnosticMessage.objects.create(
                request=diagnostic_request, role=MessageRole.ASSISTANT, content=full_text
            )
            yield f"data: {json.dumps({'done': True, 'message_id': assistant_message.id})}\n\n"
