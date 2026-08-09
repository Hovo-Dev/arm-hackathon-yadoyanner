"""Repoint DiagnosticCaseMatch from the Case KB onto past diagnoses.

The curated CaseRecord table is being removed: what this system knows about a
car is now the set of requests it has carried to COMPLETE for that car. A match
is therefore between two DiagnosticRequests.

Existing rows are deleted rather than migrated. They reference CaseRecord ids
that will not exist in a moment, and there is nothing to translate them into --
`refresh_symptom_and_matches` rebuilds the whole set for a request on every user
turn anyway, so the only cost is that a conversation mid-flight shows no similar
cases until its next message.
"""
import django.db.models.deletion
from django.db import migrations, models


def drop_stale_matches(apps, schema_editor):
    apps.get_model("diagnostics", "DiagnosticCaseMatch").objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [("diagnostics", "0009_backfill_agent_assistant_messages")]

    operations = [
        migrations.RunPython(drop_stale_matches, migrations.RunPython.noop),
        # Constraint first: it is defined over the column being dropped.
        migrations.RemoveConstraint(
            model_name="diagnosticcasematch",
            name="unique_diagnostic_case_match",
        ),
        migrations.RemoveField(model_name="diagnosticcasematch", name="case"),
        migrations.AddField(
            model_name="diagnosticcasematch",
            name="matched",
            field=models.ForeignKey(
                default=None,
                help_text="The earlier, completed request this one resembles.",
                on_delete=django.db.models.deletion.CASCADE,
                related_name="matched_by",
                to="diagnostics.diagnosticrequest",
            ),
            # The table was emptied above, so a non-null column needs no
            # usable default -- nothing is left to backfill.
            preserve_default=False,
        ),
        migrations.AddConstraint(
            model_name="diagnosticcasematch",
            constraint=models.UniqueConstraint(
                fields=("request", "matched"), name="unique_diagnostic_case_match"
            ),
        ),
        migrations.AddConstraint(
            model_name="diagnosticcasematch",
            constraint=models.CheckConstraint(
                check=models.Q(("request", models.F("matched")), _negated=True),
                name="diagnostic_match_is_not_self",
            ),
        ),
        migrations.AlterField(
            model_name="diagnosticrequest",
            name="symptom_text",
            field=models.TextField(
                blank=True,
                help_text=(
                    "Rolling summary of the whole conversation so far, refreshed after every "
                    "user turn. Embedded, and matched against the same field on past completed "
                    "requests -- one field on both sides of the comparison keeps the embedding "
                    "space coherent. There is no separate 'original text' field: the first "
                    "turn's content is just this field's first version."
                ),
            ),
        ),
    ]
