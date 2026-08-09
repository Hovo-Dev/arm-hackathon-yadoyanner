"""Delete assistant messages that no agentic run produced.

Before the chat layer was removed, every turn generated a free-text reply from
a plain chat completion over the transcript (the old `llm.stream_reply`). Those
replies had no urgency verdict, no evidence refs and nothing checking them
against retrieved sources, and they sat in the transcript next to the agentic
answer saying something different -- two responses to one question.

An assistant message is agent-produced exactly when its text is the `message`
of one of its request's stored runs. Anything else predates the removal and is
deleted here; nothing writes such rows any more (DiagnosticRunStreamView is the
only writer of assistant messages).
"""
from django.db import migrations


def purge(apps, schema_editor):
    DiagnosticMessage = apps.get_model("diagnostics", "DiagnosticMessage")
    DiagnosticRun = apps.get_model("diagnostics", "DiagnosticRun")

    # request_id -> every prose string the graph has actually emitted for it.
    from_agent = {}
    for run in DiagnosticRun.objects.all().iterator():
        text = ((run.answer or {}).get("message") or "").strip()
        if text:
            from_agent.setdefault(run.request_id, set()).add(text)

    doomed = [
        message.pk
        for message in DiagnosticMessage.objects.filter(role="assistant").iterator()
        if message.content.strip() not in from_agent.get(message.request_id, ())
    ]
    DiagnosticMessage.objects.filter(pk__in=doomed).delete()


def noop(apps, schema_editor):
    """Irreversible: the deleted text is not recorded anywhere else. Reversing
    the migration leaves the rows gone rather than failing the whole rollback."""


class Migration(migrations.Migration):
    dependencies = [("diagnostics", "0007_diagnosticrun")]

    operations = [migrations.RunPython(purge, noop)]
