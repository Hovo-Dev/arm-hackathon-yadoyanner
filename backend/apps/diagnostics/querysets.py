from django.db import models


class DiagnosticMessageQuerySet(models.QuerySet):
    def recent(self, limit=10):
        """Last `limit` turns, newest first. Reverse before handing to the LLM
        so the conversation reads oldest-to-newest in the prompt."""
        return self.order_by("-created_at")[:limit]
