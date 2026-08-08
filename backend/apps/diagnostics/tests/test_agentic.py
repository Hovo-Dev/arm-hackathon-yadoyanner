"""The agentic layer driven by the real Django ports, offline.

Every test here runs with `model=None`, which is carmed's no-spend mode: the
case gate, VIN maths, shop lookup and grounding all still run, and the two LLM
agents abstain instead of being called. That covers the routing and the parts of
the pipeline that must never depend on a model -- and it costs nothing to run in
CI.
"""
from django.test import TestCase, override_settings
from django.urls import reverse

from carmed import AnswerStatus, Query, Urgency, Vehicle, run

from apps.cases.case_store import DjangoCaseStore
from apps.cases.embeddings import embed_text
from apps.cases.models import CaseRecord
from apps.diagnostics import agentic
from apps.diagnostics.enums import DiagnosticStatus
from apps.diagnostics.models import DiagnosticRequest
from apps.diagnostics.research import NullResearchTool

BRAKES = "Grinding noise when braking at low speed, worse in the morning"

# Fails its position-9 checksum -- vPIC reports the same. See carmed/vehicle.py.
BAD_VIN = "1FTFW1ET5DFC10312"


def seed_camry_brake_case():
    record = CaseRecord.objects.create(
        car_make="Toyota",
        car_model="Camry",
        car_year_start=2010,
        car_year_end=2017,
        symptom_text=BRAKES,
        confirmed_fix="Replace worn brake pads and resurface the rotors",
        parts_named=["brake pads", "rotors"],
    )
    record.embedding = embed_text(record.symptom_text)
    record.save(update_fields=["embedding", "updated_at"])
    return record


def offline_run(text, **vehicle):
    return run(
        Query(text=text, vehicle=Vehicle(**vehicle)),
        research=NullResearchTool(),
        case_store=DjangoCaseStore(),
        model=None,
        safety_floor=True,
    )


class CaseGateTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.record = seed_camry_brake_case()

    def test_a_known_symptom_is_answered_from_the_kb_for_free(self):
        answer = offline_run(BRAKES, make="Toyota", model="Camry", year=2014)

        self.assertTrue(answer.from_cache)
        self.assertEqual(answer.trace.llm_calls, [], "a cached answer must cost nothing")
        self.assertTrue(answer.causes)
        self.assertEqual(answer.causes[0].title, self.record.confirmed_fix)
        self.assertEqual(answer.causes[0].evidence, [f"case:{self.record.pk}"])
        self.assertEqual(answer.causes[0].likely_parts, ["brake pads", "rotors"])

    def test_the_store_is_actually_reached(self):
        trace = offline_run(BRAKES, make="Toyota", model="Camry", year=2014).trace
        self.assertTrue(trace.ran("gate"))
        self.assertIn("case_store.find_similar->1", trace.lookups)

    def test_an_unrelated_symptom_does_not_hit_the_cache(self):
        answer = offline_run(
            "the radio cuts out and the speakers crackle over bumps",
            make="Toyota",
            model="Camry",
            year=2014,
        )
        self.assertFalse(answer.from_cache)
        # No model configured, so with nothing cached there is nothing to say.
        self.assertEqual(answer.status, AnswerStatus.ABSTAINED)

    def test_a_different_make_cannot_reuse_the_answer(self):
        answer = offline_run(BRAKES, make="BMW", model="5 Series", year=2008)
        self.assertFalse(answer.from_cache)
        self.assertIn("case_store.find_similar->0", answer.trace.lookups)

    def test_safety_floor_escalates_a_brake_problem(self):
        """CaseRecord has no urgency column, so a cached answer inherits carmed's
        neutral default. The floor is what stops a brake fault being reported as
        'fix this week'."""
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
        self.assertTrue(stored["causes"])
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

    def test_a_known_symptom_comes_back_from_the_kb(self):
        body = self.post(
            text=BRAKES, car_make="Toyota", car_model="Camry", car_year=2014
        ).json()

        self.assertEqual(body["status"], AnswerStatus.ANSWERED.value)
        self.assertTrue(body["from_cache"])
        self.assertEqual(body["dropped_refs"], 0)
        self.assertIn("gate", body["trace"]["steps"])

    def test_text_is_required(self):
        self.assertEqual(self.post(car_make="Toyota").status_code, 400)

    def test_nothing_is_persisted(self):
        self.post(text=BRAKES, car_make="Toyota")
        self.assertEqual(DiagnosticRequest.objects.count(), 0)

    def test_a_bad_vin_is_refused(self):
        body = self.post(text="brakes squeal", vin=BAD_VIN).json()
        self.assertEqual(body["status"], AnswerStatus.REFUSED_BAD_VIN.value)
        self.assertEqual(body["trace"]["steps"], ["vehicle", "finalize"])
