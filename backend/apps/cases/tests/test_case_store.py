"""The Django implementation of carmed.ports.CaseStore, against real pgvector.

No network beyond the local embedding model. The properties tested here are the
ones the agentic layer's correctness rests on: a match is labelled with how
tightly we filtered, and a filter that is too tight degrades to a loose match
rather than to silence.
"""
from django.test import TestCase

from carmed.models import MatchTier, SystemArea, Vehicle

from apps.cases.case_store import DjangoCaseStore
from apps.cases.models import CaseRecord

BRAKES = "Grinding noise when braking at low speed, worse in the morning"
COOLING = "Temperature gauge climbs into the red in traffic and coolant smells sweet"


def make_case(**overrides):
    data = dict(
        car_make="Toyota",
        car_model="Camry",
        car_year_start=2010,
        car_year_end=2017,
        symptom_text=BRAKES,
        confirmed_fix="Replace worn brake pads and resurface the rotors",
        parts_named=["brake pads", "rotors"],
    )
    data.update(overrides)
    record = CaseRecord.objects.create(**data)
    record.refresh_from_db()
    from apps.cases.embeddings import embed_text

    record.embedding = embed_text(record.symptom_text)
    record.save(update_fields=["embedding", "updated_at"])
    return record


class TieringTests(TestCase):
    """`tier` says how tightly we filtered, and it is what decides whether a
    match may be *served* as an answer or only used as context."""

    @classmethod
    def setUpTestData(cls):
        cls.camry = make_case()

    def find(self, **vehicle):
        return DjangoCaseStore().find_similar(text=BRAKES, vehicle=Vehicle(**vehicle))

    def test_exact_when_make_model_and_year_all_match(self):
        matches = self.find(make="Toyota", model="Camry", year=2014)
        self.assertEqual([m.tier for m in matches], [MatchTier.EXACT])

    def test_near_when_year_falls_outside_the_covered_range(self):
        """The case covers 2010-2017. A 2020 Camry is the same model, so this is
        still worth showing -- but it is not an exact match and must not claim
        to be."""
        matches = self.find(make="Toyota", model="Camry", year=2020)
        self.assertEqual([m.tier for m in matches], [MatchTier.NEAR])

    def test_near_when_year_is_unknown(self):
        matches = self.find(make="Toyota", model="Camry")
        self.assertEqual([m.tier for m in matches], [MatchTier.NEAR])

    def test_loose_when_only_the_make_is_known(self):
        matches = self.find(make="Toyota")
        self.assertEqual([m.tier for m in matches], [MatchTier.LOOSE])

    def test_loose_when_the_model_has_no_cases_at_all(self):
        """Degrade to a loose match rather than returning nothing: 'another
        Toyota had this' is useful context, as long as it is labelled as such."""
        matches = self.find(make="Toyota", model="Corolla", year=2014)
        self.assertEqual([m.tier for m in matches], [MatchTier.LOOSE])

    def test_no_vehicle_at_all_still_searches(self):
        self.assertEqual([m.tier for m in self.find()], [MatchTier.LOOSE])


class FilteringTests(TestCase):
    def test_a_camry_case_never_reaches_a_bmw_question(self):
        """The whole reason the vehicle filter runs before the ranking."""
        make_case()
        matches = DjangoCaseStore().find_similar(
            text=BRAKES, vehicle=Vehicle(make="BMW", model="5 Series", year=2008)
        )
        self.assertEqual(matches, [])

    def test_empty_kb_returns_empty_rather_than_raising(self):
        matches = DjangoCaseStore().find_similar(
            text=BRAKES, vehicle=Vehicle(make="Toyota", model="Camry", year=2014)
        )
        self.assertEqual(matches, [])

    def test_empty_query_returns_empty_without_embedding_anything(self):
        make_case()
        self.assertEqual(
            DjangoCaseStore().find_similar(text="", vehicle=Vehicle(make="Toyota")), []
        )

    def test_rows_without_an_embedding_are_skipped(self):
        CaseRecord.objects.create(
            car_make="Toyota", car_model="Camry", symptom_text=BRAKES, confirmed_fix="x"
        )
        self.assertEqual(
            DjangoCaseStore().find_similar(text=BRAKES, vehicle=Vehicle(make="Toyota")), []
        )


class ScoringTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.brakes = make_case()
        cls.cooling = make_case(symptom_text=COOLING, confirmed_fix="Replace the head gasket")

    def test_scores_are_similarities_in_range_and_best_first(self):
        matches = DjangoCaseStore().find_similar(
            text="loud grinding from the front wheels when I press the brake",
            vehicle=Vehicle(make="Toyota", model="Camry", year=2014),
        )
        scores = [m.score for m in matches]
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores), scores)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(matches[0].case.id, str(self.brakes.pk))

    def test_limit_is_honoured(self):
        matches = DjangoCaseStore().find_similar(
            text=BRAKES, vehicle=Vehicle(make="Toyota"), limit=1
        )
        self.assertEqual(len(matches), 1)


class MappingTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.record = make_case()

    def test_case_carries_the_pk_as_its_id(self):
        """Agents cite "case:<id>" and finalize resolves it back. An id that
        isn't stable means the citation is silently dropped."""
        match = DjangoCaseStore().find_similar(text=BRAKES, vehicle=Vehicle(make="Toyota"))[0]
        self.assertEqual(match.case.id, str(self.record.pk))

    def test_symptom_fix_and_parts_come_across(self):
        case = DjangoCaseStore().find_similar(text=BRAKES, vehicle=Vehicle(make="Toyota"))[0].case
        self.assertEqual(case.symptom, self.record.symptom_text)
        self.assertEqual(case.fix, self.record.confirmed_fix)
        self.assertEqual(case.part_names, ["brake pads", "rotors"])

    def test_system_area_is_classified_from_the_symptom(self):
        """Derived from carmed's trilingual symptom table rather than stored, so
        it can never drift out of step with the routing that uses it."""
        case = DjangoCaseStore().find_similar(text=BRAKES, vehicle=Vehicle(make="Toyota"))[0].case
        self.assertEqual(case.system_area, SystemArea.BRAKES)

    def test_no_source_url_is_invented(self):
        case = DjangoCaseStore().find_similar(text=BRAKES, vehicle=Vehicle(make="Toyota"))[0].case
        self.assertIsNone(case.source_url)
