from django.urls import path

from .views import home, sentimento

urlpatterns = [
    path('', home, name='home'),
    path('sentimento/', sentimento, name='sentimento'),
]
