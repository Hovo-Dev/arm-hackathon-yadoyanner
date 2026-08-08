"""The Django implementation of carmed.ports.CaseStore, against real pgvector.

No network beyond the local embedding model. The property the agentic layer's
correctness rests on: a case is only ever returned for the exact car it was
confirmed on -- same make, same model, a year range covering this one. Anything
less identified returns nothing rather than something adjacent.
"""
from django.test import TestCase

from carmed.models import MatchTier, SystemArea, Vehicle

from apps.cases.case_store import DjangoCaseStore
from apps.cases.models import CaseRecord

BRAKES = "Grinding noise when braking at low speed, worse in the morning"
COOLING = "Temperature gauge climbs into the red in traffic and coolant smells sweet"

# The car every fixture case is confirmed on. Lookups have to name it in full:
# make, model and year are all required now, so a partial vehicle finds nothing.
CAMRY = Vehicle(make="Toyota", model="Camry", year=2014)


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


class VehicleScopeTests(TestCase):
    """The filter is the whole safety property: a confirmed fix is evidence
    about the car it was confirmed on and about no other."""

    @classmethod
    def setUpTestData(cls):
        cls.camry = make_case()

    def find(self, **vehicle):
        return DjangoCaseStore().find_similar(text=BRAKES, vehicle=Vehicle(**vehicle))

    def test_exact_when_make_model_and_year_all_match(self):
        matches = self.find(make="Toyota", model="Camry", year=2014)
        self.assertEqual([m.tier for m in matches], [MatchTier.EXACT])

    def test_another_model_from_the_same_make_is_never_returned(self):
        """A Toyota Century question must not be answered out of Camry cases.
        Same badge, different car -- its confirmed fix says nothing about this
        one, and offering it as a lead is how a wrong part gets bought."""
        self.assertEqual(self.find(make="Toyota", model="Century", year=2014), [])

    def test_year_outside_the_covered_range_is_not_returned(self):
        """The case covers 2010-2017. A 2020 Camry is a different generation."""
        self.assertEqual(self.find(make="Toyota", model="Camry", year=2020), [])

    def test_unknown_year_returns_nothing(self):
        """Without a year the filter spans every generation of the model."""
        self.assertEqual(self.find(make="Toyota", model="Camry"), [])

    def test_make_alone_returns_nothing(self):
        self.assertEqual(self.find(make="Toyota"), [])

    def test_no_vehicle_at_all_returns_nothing(self):
        self.assertEqual(self.find(), [])

    def test_open_year_bounds_still_match(self):
        """Null bounds are the KB saying the case holds for the model whatever
        the year -- a statement about the data, not a widened filter."""
        make_case(car_year_start=None, car_year_end=None, symptom_text=COOLING)
        matches = self.find(make="Toyota", model="Camry", year=1994)
        self.assertEqual([m.case.symptom for m in matches], [COOLING])


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
        self.assertEqual(DjangoCaseStore().find_similar(text="", vehicle=CAMRY), [])

    def test_rows_without_an_embedding_are_skipped(self):
        CaseRecord.objects.create(
            car_make="Toyota", car_model="Camry", symptom_text=BRAKES, confirmed_fix="x"
        )
        self.assertEqual(DjangoCaseStore().find_similar(text=BRAKES, vehicle=CAMRY), [])


class ScoringTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.brakes = make_case()
        cls.cooling = make_case(symptom_text=COOLING, confirmed_fix="Replace the head gasket")

    def test_scores_are_similarities_in_range_and_best_first(self):
        matches = DjangoCaseStore().find_similar(
            text="loud grinding from the front wheels when I press the brake",
            vehicle=CAMRY,
        )
        scores = [m.score for m in matches]
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores), scores)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(matches[0].case.id, str(self.brakes.pk))

    def test_limit_is_honoured(self):
        matches = DjangoCaseStore().find_similar(text=BRAKES, vehicle=CAMRY, limit=1)
        self.assertEqual(len(matches), 1)


class MappingTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.record = make_case()

    def test_case_carries_the_pk_as_its_id(self):
        """Agents cite "case:<id>" and finalize resolves it back. An id that
        isn't stable means the citation is silently dropped."""
        match = DjangoCaseStore().find_similar(text=BRAKES, vehicle=CAMRY)[0]
        self.assertEqual(match.case.id, str(self.record.pk))

    def test_symptom_fix_and_parts_come_across(self):
        case = DjangoCaseStore().find_similar(text=BRAKES, vehicle=CAMRY)[0].case
        self.assertEqual(case.symptom, self.record.symptom_text)
        self.assertEqual(case.fix, self.record.confirmed_fix)
        self.assertEqual(case.part_names, ["brake pads", "rotors"])

    def test_system_area_is_classified_from_the_symptom(self):
        """Derived from carmed's trilingual symptom table rather than stored, so
        it can never drift out of step with the routing that uses it."""
        case = DjangoCaseStore().find_similar(text=BRAKES, vehicle=CAMRY)[0].case
        self.assertEqual(case.system_area, SystemArea.BRAKES)

    def test_no_source_url_is_invented(self):
        case = DjangoCaseStore().find_similar(text=BRAKES, vehicle=CAMRY)[0].case
        self.assertIsNone(case.source_url)
