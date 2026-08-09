from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    DiagnoseView,
    DiagnosticImageViewSet,
    DiagnosticMessageViewSet,
    DiagnosticRequestViewSet,
    DiagnosticResolveView,
    DiagnosticRunStreamView,
    DiagnosticTurnView,
)

router = DefaultRouter()
router.register("diagnostics", DiagnosticRequestViewSet, basename="diagnostic")
router.register("diagnostic-images", DiagnosticImageViewSet, basename="diagnostic-image")
router.register("diagnostic-messages", DiagnosticMessageViewSet, basename="diagnostic-message")

urlpatterns = [
    path("diagnose/", DiagnoseView.as_view(), name="diagnose"),
    path(
        "diagnostics/<int:request_id>/run/stream/",
        DiagnosticRunStreamView.as_view(),
        name="diagnostic-run-stream",
    ),
    path(
        "diagnostics/<int:request_id>/resolve/",
        DiagnosticResolveView.as_view(),
        name="diagnostic-resolve",
    ),
    path(
        "diagnostics/<int:request_id>/turns/",
        DiagnosticTurnView.as_view(),
        name="diagnostic-turn",
    ),
    path("", include(router.urls)),
]
