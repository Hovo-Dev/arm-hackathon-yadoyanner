from django.apps import AppConfig
from django.conf import settings


class DiagnosticsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.diagnostics"
    label = "diagnostics"

    def ready(self):
        # carmed ships CACHE_HIT_THRESHOLD = 0.80, calibrated against its own
        # embedder. Cosine similarity has no absolute meaning across models, so
        # the knob belongs next to EMBEDDING_MODEL_NAME in Django settings --
        # and setting it here keeps /carmed byte-identical to its branch.
        from carmed import graph

        graph.CACHE_HIT_THRESHOLD = settings.CARMED_CACHE_HIT_THRESHOLD
