from django.db import models


class DiagnosticStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    CASE_MATCHED = "case_matched", "Matched to a prior verified case"
    RESEARCHING = "researching", "Deep research in progress"
    COMPLETE = "complete", "Complete"


class MessageRole(models.TextChoices):
    USER = "user", "User"
    ASSISTANT = "assistant", "Assistant"


class DiagnosticImageType(models.TextChoices):
    """Ranked by how well they work per spec section 2 -- VIN plate is the
    highest-value shot, damage/leak photos are supporting signal only."""

    VIN_PLATE = "vin_plate", "VIN plate"
    PART = "part", "Old part"
    DASHBOARD_LIGHT = "dashboard_light", "Dashboard warning light"
    DAMAGE = "damage", "Damage / leak"
