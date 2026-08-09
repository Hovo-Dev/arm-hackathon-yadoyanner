"""The agentic layer driven by the real Django ports, offline.

Every test here runs with `model=None`, which is carmed's no-spend mode: the
case gate, VIN maths, shop lookup and grounding all still run, and the two LLM
agents abstain instead of being called. That covers the routing and the parts of
the pipeline that must never depend on a model -- and it costs nothing to run in
CI.
"""
import json

from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse

from carmed import AnswerStatus, Query, Urgency, Vehicle, run

from apps.diagnostics import agentic
from apps.diagnostics.case_store import PastCaseStore
from apps.diagnostics.embeddings import embed_text
from apps.diagnostics.enums import DiagnosticStatus
from apps.diagnostics.models import DiagnosticRequest
from apps.diagnostics.research import NullResearchTool

BRAKES = "Grinding noise when braking at low speed, worse in the morning"

# Fails its position-9 checksum -- vPIC reports the same. See carmed/vehicle.py.
BAD_VIN = "1FTFW1ET5DFC10312"


FIX = "Replace worn brake pads and resurface the rotors"


def seed_camry_brake_case(**overrides):
    """A finished diagnosis of a 2014 Camry -- the only kind of row that is
    evidence for a later question about the same car."""
    data = dict(
        car_make="Toyota",
        car_model="Camry",
        car_year=2014,
        symptom_text=BRAKES,
        status=DiagnosticStatus.COMPLETE,
        summary={
            "status": "answered",
            "causes": [{"title": FIX, "confidence": "strong", "evidence": []}],
            "repair_steps": ["Measure pad thickness", "Replace pads below 3 mm"],
        },
    )
    data.update(overrides)
    record = DiagnosticRequest.objects.create(**data)
    record.embedding = embed_text(record.symptom_text)
    record.save(update_fields=["embedding"])
    return record


def offline_run(text, **vehicle):
    return run(
        Query(text=text, vehicle=Vehicle(**vehicle)),
        research=NullResearchTool(),
        case_store=PastCaseStore(),
        model=None,
        safety_floor=True,
    )


class PastCaseRetrievalTests(TestCase):
    """What the store is allowed to hand the diagnostician, and what it is not.

    Retrieved cases are this system's own earlier answers, so they are evidence
    to reason over and never a verdict to copy -- `verified=False` makes
    carmed's cache-hit condition unsatisfiable by construction.
    """

    @classmethod
    def setUpTestData(cls):
        cls.record = seed_camry_brake_case()

    def test_a_past_diagnosis_of_the_same_car_is_retrieved(self):
        trace = offline_run(BRAKES, make="Toyota", model="Camry", year=2014).trace
        self.assertTrue(trace.ran("gate"))
        self.assertIn("case_store.find_similar->1", trace.lookups)

    def test_a_near_identical_symptom_is_served_without_a_model_call(self):
        """The point of keeping past cases: re-deriving an answer we already
        hold costs three model calls to land in the same place."""
        answer = offline_run(BRAKES, make="Toyota", model="Camry", year=2014)

        self.assertTrue(answer.from_cache)
        self.assertEqual(answer.trace.llm_calls, [], "a served answer must cost nothing")
        self.assertEqual(answer.causes[0].title, FIX)
        self.assertEqual(answer.causes[0].evidence, [f"case:{self.record.pk}"])

    def test_a_merely_similar_symptom_is_not_served(self):
        """Above the evidence floor but below the serving bar: it informs a
        fresh diagnosis rather than being replayed as one."""
        answer = offline_run(
            "there is a grinding sound from the wheels when I slow down",
            make="Toyota",
            model="Camry",
            year=2014,
        )
        self.assertIn("case_store.find_similar->1", answer.trace.lookups)
        self.assertFalse(answer.from_cache)

    @override_settings(CASE_SERVE_MIN_SIMILARITY=1.01)
    def test_the_serving_bar_is_configurable(self):
        """Unreachable bar -- nothing may be served, however close."""
        answer = offline_run(BRAKES, make="Toyota", model="Camry", year=2014)
        self.assertFalse(answer.from_cache)

    def test_an_unfinished_request_is_not_evidence(self):
        """Its symptom summary describes a problem nobody has got to the bottom
        of yet."""
        DiagnosticRequest.objects.all().delete()
        seed_camry_brake_case(status=DiagnosticStatus.PENDING)
        trace = offline_run(BRAKES, make="Toyota", model="Camry", year=2014).trace
        self.assertIn("case_store.find_similar->0", trace.lookups)

    def test_another_model_from_the_same_make_is_not_evidence(self):
        """A Century question must not be answered out of Camry history."""
        trace = offline_run(BRAKES, make="Toyota", model="Century", year=2014).trace
        self.assertIn("case_store.find_similar->0", trace.lookups)

    def test_another_year_of_the_same_model_is_not_evidence(self):
        trace = offline_run(BRAKES, make="Toyota", model="Camry", year=2019).trace
        self.assertIn("case_store.find_similar->0", trace.lookups)

    def test_a_different_make_is_not_evidence(self):
        trace = offline_run(BRAKES, make="BMW", model="5 Series", year=2008).trace
        self.assertIn("case_store.find_similar->0", trace.lookups)

    def test_a_request_is_not_evidence_about_itself(self):
        """A re-run would otherwise retrieve itself at similarity 1.00 and cite
        its own previous answer as support for repeating it."""
        store = PastCaseStore(exclude_request_id=self.record.pk)
        found = store.find_similar(
            text=BRAKES, vehicle=Vehicle(make="Toyota", model="Camry", year=2014)
        )
        self.assertEqual(found, [])

    def test_the_retrieved_case_carries_the_past_answer(self):
        found = PastCaseStore().find_similar(
            text=BRAKES, vehicle=Vehicle(make="Toyota", model="Camry", year=2014)
        )
        self.assertEqual([m.case.id for m in found], [str(self.record.pk)])
        self.assertEqual(found[0].case.fix, FIX)
        self.assertTrue(found[0].case.verified, "an all-but-identical match may be served")

    def test_only_the_configured_number_of_cases_is_returned(self):
        for i in range(7):
            seed_camry_brake_case(symptom_text=f"{BRAKES} variant {i}")
        found = PastCaseStore().find_similar(
            text=BRAKES, vehicle=Vehicle(make="Toyota", model="Camry", year=2014), limit=5
        )
        self.assertEqual(len(found), 5)

    def test_an_unrelated_symptom_on_the_same_car_is_not_evidence(self):
        """The vehicle filter cannot tell a brake question from a radio one.
        Without the similarity floor these came back at ~0.3 and were offered
        to the diagnostician as cases it may cite."""
        trace = offline_run(
            "the radio cuts out and the speakers crackle over bumps",
            make="Toyota",
            model="Camry",
            year=2014,
        ).trace
        self.assertIn("case_store.find_similar->0", trace.lookups)

    @override_settings(CASE_MATCH_MIN_SIMILARITY=0.99)
    def test_the_floor_is_configurable(self):
        found = PastCaseStore().find_similar(
            text="grinding from the wheels when slowing down",
            vehicle=Vehicle(make="Toyota", model="Camry", year=2014),
        )
        self.assertEqual(found, [])

    def test_safety_floor_escalates_a_brake_problem(self):
        """The floor keys on the symptom, not on the diagnosis, so it holds even
        when there is no model to produce one."""
        answer = offline_run(BRAKES, make="Toyota", model="Camry", year=2014)
        self.assertEqual(answer.urgency, Urgency.DO_NOT_DRIVE)


class VinTests(TestCase):
    def test_a_failed_checksum_refuses_before_anything_is_spent(self):
        answer = offline_run("brakes squeal", make="Toyota", vin=BAD_VIN)

        self.assertEqual(answer.status, AnswerStatus.REFUSED_BAD_VIN)
        self.assertEqual(answer.trace.steps, ["vehicle", "finalize"])
        self.assertEqual(answer.trace.cost, 0)
        self.assertFalse(answer.trace.looked_up("case_store.find_similar"))
        self.assertIn("Checksum failed", answer.message)


class NullResearchToolTests(TestCase):
    """The research port is deliberately unimplemented until another branch
    lands. Empty results must degrade, not break."""

    @classmethod
    def setUpTestData(cls):
        seed_camry_brake_case()

    def test_the_run_completes_and_records_the_empty_lookups(self):
        answer = offline_run(BRAKES, make="Toyota", model="Camry", year=2014)

        self.assertEqual(answer.status, AnswerStatus.ANSWERED)
        self.assertIn("research.search_shops->0", answer.trace.lookups)
        self.assertEqual(answer.shops, [])
        self.assertEqual(answer.parts.options, [])
        self.assertEqual(answer.listings, {})

    def test_nothing_is_cited_that_cannot_be_resolved(self):
        answer = offline_run(BRAKES, make="Toyota", model="Camry", year=2014)
        self.assertEqual(answer.dropped_refs, 0)

    def test_all_three_methods_return_empty(self):
        tool = NullResearchTool()
        vehicle = Vehicle(make="Toyota")
        self.assertEqual(tool.search_knowledge(query="x", vehicle=vehicle), [])
        self.assertEqual(tool.search_listings(part_terms=["x"], vehicle=vehicle), [])
        self.assertEqual(
            tool.search_shops(make="Toyota", system_area="brakes", city="Yerevan"), []
        )


@override_settings(OPENROUTER_API_KEY="")
class SummaryPersistenceTests(TestCase):
    """The Answer is stored verbatim on DiagnosticRequest.summary, so it has to
    survive the JSONField round trip unchanged."""

    def setUp(self):
        seed_camry_brake_case()
        # get_chat_model is lru_cached; clear it so the blank key above takes
        # effect and these tests stay offline.
        agentic.get_chat_model.cache_clear()
        self.addCleanup(agentic.get_chat_model.cache_clear)

        self.request = DiagnosticRequest.objects.create(
            car_make="Toyota", car_model="Camry", car_year=2014, symptom_text=BRAKES
        )

    def test_finalize_stores_the_answer_and_closes_the_request(self):
        url = reverse("diagnostic-finalize", args=[self.request.pk])
        stored = self.client.post(url).json()["summary"]

        self.assertEqual(stored["status"], AnswerStatus.ANSWERED.value)
        self.assertTrue(stored["from_cache"])
        self.assertEqual(stored["urgency"], int(Urgency.DO_NOT_DRIVE))
        self.assertTrue(stored["disclaimer"])

        self.request.refresh_from_db()
        self.assertEqual(self.request.status, DiagnosticStatus.COMPLETE)

    def test_the_trace_survives_the_json_round_trip(self):
        answer = agentic.run_for_request(self.request)
        self.request.summary = answer.model_dump(mode="json")
        self.request.save(update_fields=["summary"])

        self.request.refresh_from_db()
        trace = self.request.summary["trace"]
        self.assertEqual(trace["steps"], answer.trace.steps)
        self.assertEqual(trace["lookups"], answer.trace.lookups)
        self.assertEqual(trace["notes"], answer.trace.notes)
        self.assertEqual(trace["llm_calls"], answer.trace.llm_calls)


@override_settings(OPENROUTER_API_KEY="")
class StreamingTests(TransactionTestCase):
    """The streamed run has to be the *same* run, not a second implementation.

    TransactionTestCase rather than TestCase because stream_run does the graph
    work on a worker thread, which gets its own DB connection and so cannot see
    data held open in another connection's uncommitted transaction.
    """

    def setUp(self):
        seed_camry_brake_case()
        agentic.get_chat_model.cache_clear()
        self.addCleanup(agentic.get_chat_model.cache_clear)
        self.query = Query(
            text=BRAKES, vehicle=Vehicle(make="Toyota", model="Camry", year=2014)
        )

    def collect(self, query=None):
        return list(agentic.stream_run(query or self.query))

    def test_streaming_matches_the_plain_run(self):
        """_assemble_answer duplicates the tail of carmed.graph.run. This is what
        catches it drifting."""
        streamed = self.collect()[-1]["answer"]
        direct = agentic.run_query(self.query).model_dump(mode="json")

        # The trace notes carry similarity figures that are stable, but compare
        # the decision-shaped fields explicitly so a mismatch names itself.
        for field in ("status", "causes", "urgency", "from_cache", "dropped_refs", "intent"):
            self.assertEqual(streamed[field], direct[field], field)
        self.assertEqual(streamed["trace"]["steps"], direct["trace"]["steps"])
        self.assertEqual(streamed["trace"]["llm_calls"], direct["trace"]["llm_calls"])

    def test_events_arrive_in_pipeline_order(self):
        events = self.collect()
        steps = [e["name"] for e in events if e["type"] == "step"]
        self.assertEqual(steps, ["vehicle", "route", "gate", "diagnose", "parts", "shops", "finalize"])
        self.assertEqual(events[-1]["type"], "done")

    def test_the_gate_result_is_visible_as_an_event(self):
        """The UI keys its gate badge off this."""
        notes = [e["text"] for e in self.collect() if e["type"] == "note"]
        self.assertTrue(any(n.startswith("cache HIT") for n in notes), notes)

    def test_lookups_report_their_result_counts(self):
        lookups = {
            e["name"]: e["count"] for e in self.collect() if e["type"] == "lookup"
        }
        self.assertEqual(lookups["case_store.find_similar"], 1)
        self.assertEqual(lookups["research.search_shops"], 0)

    def test_the_endpoint_streams_and_persists(self):
        url = reverse("diagnostic-run-stream", args=[self.request_pk()])
        response = self.client.post(url)
        self.assertEqual(response["Content-Type"], "text/event-stream")

        events = [
            json.loads(chunk.decode().removeprefix("data: ").strip())
            for chunk in response.streaming_content
            if chunk.strip()
        ]
        kinds = [e["type"] for e in events]
        self.assertIn("step", kinds)
        self.assertIn("done", kinds)
        self.assertEqual(kinds[-1], "saved")

        request = DiagnosticRequest.objects.get(pk=self.request_pk())
        self.assertEqual(request.status, DiagnosticStatus.COMPLETE)
        self.assertEqual(request.summary["status"], AnswerStatus.ANSWERED.value)
        self.assertTrue(request.summary["causes"])

    def test_an_empty_symptom_is_refused_before_anything_is_spent(self):
        """Running the graph on an empty description costs two model calls to
        ask what the car is, and hands the diagnostician a prompt reading
        "Problem (xx):" -- xx being carmed's unknown-language tag."""
        blank = DiagnosticRequest.objects.create(car_make="Toyota", symptom_text="   ")
        response = self.client.post(reverse("diagnostic-run-stream", args=[blank.pk]))

        self.assertEqual(response.status_code, 400)
        self.assertIn("Describe the problem", response.json()["detail"])

    def request_pk(self):
        if not hasattr(self, "_pk"):
            self._pk = DiagnosticRequest.objects.create(
                car_make="Toyota", car_model="Camry", car_year=2014, symptom_text=BRAKES
            ).pk
        return self._pk


@override_settings(OPENROUTER_API_KEY="")
class TraceLoggingTests(TestCase):
    """The trace is the only way to see what actually ran, so it has to reach
    the terminal -- not just the database. Django's default logging config
    attaches no handler to apps.*, which silently swallowed this once already."""

    def setUp(self):
        seed_camry_brake_case()
        agentic.get_chat_model.cache_clear()
        self.addCleanup(agentic.get_chat_model.cache_clear)

    def test_the_trace_is_logged_at_info(self):
        with self.assertLogs("apps.diagnostics.agentic", level="INFO") as captured:
            agentic.run_query(
                Query(text=BRAKES, vehicle=Vehicle(make="Toyota", model="Camry", year=2014))
            )

        line = "\n".join(captured.output)
        self.assertIn("2014 Toyota Camry", line)
        self.assertIn("gate -> diagnose", line)
        self.assertIn("case_store.find_similar->1", line)
        self.assertIn("cache HIT", line)

    def test_apps_loggers_have_a_console_handler_configured(self):
        import logging

        logger = logging.getLogger("apps.diagnostics.agentic")
        self.assertTrue(logger.isEnabledFor(logging.INFO))
        self.assertTrue(
            any(h for h in logging.getLogger("apps").handlers),
            "apps.* has no handler -- INFO logs would be dropped",
        )


@override_settings(OPENROUTER_API_KEY="")
class DiagnoseEndpointTests(TestCase):
    def setUp(self):
        seed_camry_brake_case()
        agentic.get_chat_model.cache_clear()
        self.addCleanup(agentic.get_chat_model.cache_clear)

    def post(self, **payload):
        return self.client.post(
            reverse("diagnose"), data=payload, content_type="application/json"
        )

    def test_a_known_symptom_reaches_the_gate_and_retrieves_the_past_case(self):
        body = self.post(
            text=BRAKES, car_make="Toyota", car_model="Camry", car_year=2014
        ).json()

        self.assertEqual(body["status"], AnswerStatus.ANSWERED.value)
        self.assertTrue(body["from_cache"])
        self.assertEqual(body["dropped_refs"], 0)
        self.assertIn("gate", body["trace"]["steps"])
        self.assertIn("case_store.find_similar->1", body["trace"]["lookups"])

    def test_text_is_required(self):
        self.assertEqual(self.post(car_make="Toyota").status_code, 400)

    def test_nothing_is_persisted(self):
        """The fixture case is itself a DiagnosticRequest now, so this counts
        the delta rather than the total."""
        before = DiagnosticRequest.objects.count()
        self.post(text=BRAKES, car_make="Toyota")
        self.assertEqual(DiagnosticRequest.objects.count(), before)

    def test_a_bad_vin_is_refused(self):
        body = self.post(text="brakes squeal", vin=BAD_VIN).json()
        self.assertEqual(body["status"], AnswerStatus.REFUSED_BAD_VIN.value)
        self.assertEqual(body["trace"]["steps"], ["vehicle", "finalize"])
