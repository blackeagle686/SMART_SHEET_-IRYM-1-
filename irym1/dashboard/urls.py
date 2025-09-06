from django.urls import path 
from . import views 

urlpatterns = [
    path("dashboard/<int:analysis_id>/", views.dashboard, name="dashboard"),
    path("history/", views.dashboard_history, name="history"),
    path("code_gen/", views.generate_code_view, name="code_gen"),
    path("export/pdf/", views.export_pdf, name="export_pdf"),
    path("export/pptx/", views.export_pptx, name="export_pptx"),
    path("export/csv/", views.download_csv, name="download_csv"),
]
