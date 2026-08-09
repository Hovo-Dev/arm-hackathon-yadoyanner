from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse
from django.urls import include, path


def healthz(request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("healthz/", healthz, name="healthz"),
    path("api/", include("apps.diagnostics.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
