from django.urls import path
from . import views

urlpatterns = [
    path("", views.performance_logs_view, name="performance_logs"),
]
