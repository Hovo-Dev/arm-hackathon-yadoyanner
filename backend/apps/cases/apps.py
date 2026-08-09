"""Migrations-only remnant of the Case KB.

CaseRecord is gone -- what this system knows about a car is now the set of
diagnoses it has completed for it (see apps.diagnostics.case_store). The app
stays registered because `diagnostics`' own migration history depends on this
one, and unregistering it would break `migrate` on a fresh database.

Nothing should be added here. When the migration graph is next squashed, this
package can go with it.
"""
from django.apps import AppConfig


class CasesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.cases"
    label = "cases"
