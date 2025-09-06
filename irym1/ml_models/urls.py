from django.urls import path 
from . import views 

urlpatterns = [
    # Main pages
    # path('', views.project_list, name='project_list'),
    path('project/<int:analysis_id>/', views.ml_model_result, name='ml_model_result'),
    path('create/<int:analysis_id>/', views.create_project, name='create_project'),
    
    # API endpoints
    path('predict/<int:project_id>/', views.make_prediction, name='make_prediction'),
    path('evaluate/<int:project_id>/', views.start_evaluation, name='start_evaluation'),
    
    # Webhook endpoints
    path('webhook/training/<int:project_id>/', views.training_status_webhook, name='training_status_webhook'),
    path('webhook/evaluation/<int:project_id>/', views.evaluation_status_webhook, name='evaluation_status_webhook'),
    
    # Configuration endpoints
    path('config/generate/<int:project_id>/', views.generate_config, name='generate_config'),
    path('config/status/', views.config_service_status, name='config_service_status'),
]