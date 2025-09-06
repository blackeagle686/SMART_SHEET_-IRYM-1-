from django.urls import path 
from . import views 

urlpatterns = [
    path("", views.prompt_room, name="prompt-room")
]
