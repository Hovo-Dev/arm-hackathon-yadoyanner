import json

from django.conf import settings
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import parsers, response, status, views, viewsets
from rest_framework.decorators import action

from carmed import AnswerStatus, Query, Vehicle

from .agentic import _raw_user_text, build_query, run_query, stream_run
from .enums import DiagnosticStatus, MessageRole
from .models import DiagnosticImage, DiagnosticMessage, DiagnosticRequest, DiagnosticRun
from .serializers import (
    DiagnoseInputSerializer,
    DiagnosticImageSerializer,
    DiagnosticMessageSerializer,
    DiagnosticRequestSerializer,
)
from .llm import looks_armenian, to_armenian
from .services import build_final_summary, refresh_symptom_and_matches

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
        .prefetch_related("images", "messages", "runs", "case_matches__matched")
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
            self._record_initial_turn(instance, raw_text)

    def _record_initial_turn(self, instance, raw_text):
        """The description submitted on "Start diagnosis" becomes the first
        chat turn. refresh_symptom_and_matches summarizes it into
        symptom_text, embeds that, and runs the Agent 4 case-gate match, so
        the run that follows is grounded in whatever the KB currently matches.

        No assistant reply is generated here. A free-text reply from the chat
        model and the agentic Answer are two answers to the same question, and
        the chat one carries no urgency verdict, no evidence refs and no
        source-checked claims -- it only ever contradicted the real one."""
        DiagnosticMessage.objects.create(request=instance, role=MessageRole.USER, content=raw_text)
        # Deliberately NOT summarizing here. It is an LLM call plus an
        # embedding plus a vector query -- about eight seconds during which
        # POST /api/diagnostics/ has not returned, so the browser shows nothing
        # at all and the user cannot tell the app from a hang. The run stream
        # does it as its first step instead, where the work is narrated.

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


class DiagnosticRunStreamView(views.APIView):
    """The same agentic run as `finalize`, streamed step by step over SSE so the
    UI can show the pipeline working instead of spinning for half a minute.

    Emits one event per thing the graph records -- `step`, `llm`, `lookup`,
    `note` -- then a final `done` carrying the Answer. The Answer is persisted
    to `summary` exactly as `finalize` does, so the two endpoints leave the
    request in the same state; this one just narrates the journey."""

    def post(self, request, request_id):
        diagnostic_request = get_object_or_404(DiagnosticRequest, pk=request_id)

        # Nothing to diagnose means every agent downstream is guessing. carmed
        # would dutifully run the whole graph and spend two model calls asking
        # what the car is -- and the diagnostician's prompt would carry an empty
        # "Problem (xx):" line, xx being the unknown-language tag.
        # An empty symptom is only an error when there is nothing to make one
        # from. With a user turn present the stream summarizes it first --
        # creation no longer does, so that the POST returns instantly.
        if not diagnostic_request.symptom_text.strip() and not (
            diagnostic_request.messages.filter(role=MessageRole.USER).exists()
        ):
            return response.Response(
                {"detail": "Describe the problem first -- there is nothing to diagnose yet."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return StreamingHttpResponse(
            self._sse_stream(diagnostic_request),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _sse_stream(self, diagnostic_request):
        summary = None
        # Kept so the run can be replayed later. The `done` event is dropped:
        # its payload is the Answer, which `summary` already holds, and storing
        # it twice means the two can drift.
        trace = []
        try:
            # Understanding the problem is the first visible step rather than a
            # silent wait on the create call. Emitting it here means the user
            # sees "reading what you wrote" immediately and watches the case KB
            # being checked, instead of staring at a frozen form.
            if not diagnostic_request.symptom_text.strip():
                opening = {"type": "step", "name": "understand"}
                trace.append(opening)
                yield f"data: {json.dumps(opening)}\n\n"

                refresh_symptom_and_matches(diagnostic_request)
                diagnostic_request.refresh_from_db()

                noted = {
                    "type": "note",
                    "text": f"understood as: {diagnostic_request.symptom_text}",
                }
                trace.append(noted)
                yield f"data: {json.dumps(noted)}\n\n"

                matched = {
                    "type": "lookup",
                    "name": "case_store.find_similar",
                    "count": diagnostic_request.case_matches.count(),
                }
                trace.append(matched)
                yield f"data: {json.dumps(matched)}\n\n"

            # exclude_request_id is Davit's: the KB is now this system's own
            # finished diagnoses, so without it a request would match itself
            # and be served its own unanswered question as a past case.
            for event in stream_run(
                build_query(diagnostic_request), exclude_request_id=diagnostic_request.pk
            ):
                if event["type"] == "done":
                    summary = event["answer"]
                else:
                    trace.append(event)
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            # The connection is already open with a 200, so an error has to be
            # delivered as an event rather than a status code.
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        if summary is None:
            return

        # The answer's prose -- a clarifying question when the graph needs more
        # from the user, a summary of the diagnosis otherwise -- is the
        # assistant's turn in the conversation, so it is persisted as a message
        # like any other. Without this the transcript is user turns only, and a
        # question the assistant asked ("what is the engine doing?") is lost the
        # moment the next run replaces the answer that carried it.
        #
        # Created before the run row so it sorts ahead of it in the transcript.
        answer_text = (summary.get("message") or "").strip()
        if answer_text:
            # Translated here and nowhere else. This row is the chat bubble --
            # the only prose the owner reads -- while `summary` below keeps the
            # English the rest of the system depends on: PastCaseStore builds
            # Case.fix from summary["causes"][0]["title"], and a knowledge base
            # written half in Armenian would stop matching the English symptoms
            # it gets queried with.
            #
            # Answer in the language the question came in. English in, English
            # out -- and no model call at all, which is the common case for a
            # tester and costs nothing to get right.
            #
            # Narrated when it does run, because it is a model call the user is
            # waiting on and an unexplained pause after "Done" reads as a hang.
            if looks_armenian(_raw_user_text(diagnostic_request)):
                replying = {"type": "step", "name": "reply"}
                trace.append(replying)
                yield f"data: {json.dumps(replying)}\n\n"
                answer_text = to_armenian(answer_text)
            DiagnosticMessage.objects.create(
                request=diagnostic_request, role=MessageRole.ASSISTANT, content=answer_text
            )

        # The structured side of the same answer is appended to the history...
        DiagnosticRun.objects.create(request=diagnostic_request, answer=summary)

        # ...while `summary` and `trace` track only the latest pass. The status
        # logic and the pipeline view both want "where does this stand now",
        # which is the last run, not the sequence of them.
        diagnostic_request.summary = summary
        diagnostic_request.trace = trace
        if AnswerStatus(summary["status"]) in _TERMINAL_STATUSES:
            diagnostic_request.status = DiagnosticStatus.COMPLETE
        diagnostic_request.save(update_fields=["summary", "trace", "status", "updated_at"])
        yield f"data: {json.dumps({'type': 'saved', 'status': diagnostic_request.status})}\n\n"


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


class DiagnosticTurnView(views.APIView):
    """Record a user turn and re-derive what the conversation is about.

    Replaces the old message-stream endpoint, which also streamed back a chat
    model's free-text reply. That reply was a second, competing answer to the
    same question -- one with no urgency verdict and no traceable evidence --
    so the assistant's side of the conversation is now the agentic run alone
    (POST .../run/stream/), which the client calls straight after this."""

    def post(self, request, request_id):
        diagnostic_request = get_object_or_404(DiagnosticRequest, pk=request_id)
        content = (request.data.get("content") or "").strip()
        if not content:
            return response.Response(
                {"content": ["This field is required."]}, status=status.HTTP_400_BAD_REQUEST
            )

        message = DiagnosticMessage.objects.create(
            request=diagnostic_request, role=MessageRole.USER, content=content
        )

        # Re-summarize the conversation (now including this turn) and re-run the
        # case-gate match, so the run that follows reflects the latest
        # understanding of the symptom -- not just what was said first.
        refresh_symptom_and_matches(diagnostic_request)

        return response.Response(
            DiagnosticMessageSerializer(message).data, status=status.HTTP_201_CREATED
        )
