"""Drop the curated Case KB.

Replaced by this system's own completed diagnoses -- see
apps.diagnostics.case_store.PastCaseStore. The seed data was three rows of
demo content; nothing references CaseRecord once diagnostics 0010 has moved
DiagnosticCaseMatch off it, which is why this depends on that migration.

The app itself stays registered: diagnostics' migration history depends on
cases 0001, so unregistering it would break `migrate` on a fresh database.
"""
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("cases", "0002_remove_caserecord_language_and_more"),
        # The FK onto CaseRecord has to be gone before the table can be.
        ("diagnostics", "0010_case_matches_point_at_past_requests"),
    ]

    operations = [migrations.DeleteModel(name="CaseRecord")]
