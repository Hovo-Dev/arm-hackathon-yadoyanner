"""Give every stored run's prose an assistant message.

Runs recorded before the run endpoint started persisting `Answer.message` have
their clarifying question only inside the answer JSON. The transcript renders
one response per run -- the prose when there is any, the structured card
otherwise -- so without this those turns would show nothing at all once their
card is correctly suppressed.

Additive: it only inserts messages that are missing, and matches on text so
re-running cannot duplicate them.
"""
from django.db import migrations


def backfill(apps, schema_editor):
    DiagnosticMessage = apps.get_model("diagnostics", "DiagnosticMessage")
    DiagnosticRun = apps.get_model("diagnostics", "DiagnosticRun")

    existing = {
        (message.request_id, message.content.strip())
        for message in DiagnosticMessage.objects.filter(role="assistant").iterator()
    }

    missing = []
    for run in DiagnosticRun.objects.all().order_by("created_at").iterator():
        text = ((run.answer or {}).get("message") or "").strip()
        if not text or (run.request_id, text) in existing:
            continue
        existing.add((run.request_id, text))
        missing.append(
            DiagnosticMessage(
                request_id=run.request_id,
                role="assistant",
                content=text,
                # auto_now_add is bypassed by bulk_create, which is what we
                # want: the message belongs at its run's moment in the
                # transcript, not at migration time.
                created_at=run.created_at,
            )
        )

    DiagnosticMessage.objects.bulk_create(missing)


def noop(apps, schema_editor):
    """Leaves the backfilled rows in place; they are valid agent output."""


class Migration(migrations.Migration):
    dependencies = [("diagnostics", "0008_purge_non_agent_assistant_messages")]

    operations = [migrations.RunPython(backfill, noop)]
