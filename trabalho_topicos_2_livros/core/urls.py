from django.urls import path

from .views import dashboard, hub, sentimento

urlpatterns = [
    path('', hub, name='home'),
    path('graficos/', dashboard, name='graficos'),
    path('sentimento/', sentimento, name='sentimento'),
]
