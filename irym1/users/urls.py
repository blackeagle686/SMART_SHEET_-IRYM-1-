from django.urls import path 
from . import views 

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("profile/", views.user_profile, name="user_profile"),
    path("settings/", view=views.settings_view, name="settings"),
    path("logout/", views.logout_view, name="logout"), 
    path("delete_account/", views.delete_account, name="delete_account"), 
    
]
