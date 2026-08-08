from django.core.management.base import BaseCommand

from apps.cases.embeddings import embed_text
from apps.cases.models import CaseRecord

DEMO_CASES = [
    dict(
        car_make="Toyota",
        car_model="Camry",
        car_year_start=2010,
        car_year_end=2017,
        symptom_text="Grinding noise when braking at low speed, worse in the morning",
        confirmed_fix="Replace worn brake pads and resurface the rotors",
        parts_named=["brake pads", "rotors"],
    ),
    dict(
        car_make="Honda",
        car_model="Civic",
        car_year_start=2012,
        car_year_end=2020,
        symptom_text="Check engine light on with a rough idle and occasional stalling at stops",
        confirmed_fix="Replace the spark plugs and clean the throttle body",
        parts_named=["spark plugs"],
    ),
    dict(
        car_make="Toyota",
        car_model="Corolla",
        car_year_start=2015,
        car_year_end=2022,
        symptom_text="AC blows warm air, compressor clutch not engaging",
        confirmed_fix="Recharge refrigerant and replace the AC compressor clutch relay",
        parts_named=["AC compressor clutch relay", "refrigerant"],
    ),
]


class Command(BaseCommand):
    help = "Seed a few demo CaseRecord rows (with embeddings) for local testing of the case-match pipeline."

    def handle(self, *args, **options):
        for data in DEMO_CASES:
            case, created = CaseRecord.objects.get_or_create(
                car_make=data["car_make"], symptom_text=data["symptom_text"], defaults=data
            )
            case.embedding = embed_text(case.symptom_text)
            case.save(update_fields=["embedding", "updated_at"])
            label = "created" if created else "already existed, re-embedded"
            self.stdout.write(f"{case.car_make} {case.car_model}: {label} (id={case.id})")
