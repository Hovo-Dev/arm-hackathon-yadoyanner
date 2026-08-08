from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    DiagnoseView,
    DiagnosticImageViewSet,
    DiagnosticMessageStreamView,
    DiagnosticMessageViewSet,
    DiagnosticRequestViewSet,
)

router = DefaultRouter()
router.register("diagnostics", DiagnosticRequestViewSet, basename="diagnostic")
router.register("diagnostic-images", DiagnosticImageViewSet, basename="diagnostic-image")
router.register("diagnostic-messages", DiagnosticMessageViewSet, basename="diagnostic-message")

urlpatterns = [
    path("diagnose/", DiagnoseView.as_view(), name="diagnose"),
    path(
        "diagnostics/<int:request_id>/messages/stream/",
        DiagnosticMessageStreamView.as_view(),
        name="diagnostic-message-stream",
    ),
    path("", include(router.urls)),
]
