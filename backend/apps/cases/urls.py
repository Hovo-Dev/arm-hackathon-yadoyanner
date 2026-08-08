from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import CaseRecordViewSet, CaseSearchView

router = DefaultRouter()
router.register("cases", CaseRecordViewSet, basename="case")

urlpatterns = [
    path("cases/search/", CaseSearchView.as_view(), name="case-search"),
    path("", include(router.urls)),
]
