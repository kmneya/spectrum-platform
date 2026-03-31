from django.urls import path
from .views import run_simulation,simulation_history,dashboard

urlpatterns = [
    path('api/run/', run_simulation, name='run_simulation'),
    path('api/history/', simulation_history),
    path('', dashboard),
]