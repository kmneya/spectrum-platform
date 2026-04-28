# simulation/views.py
import json
import logging
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .engine import run_real_simulation
from .models import SimulationRun

logger = logging.getLogger(__name__)


def dashboard(request):
    return render(request, 'simulation/dashboard.html')


def simulation_history(request):
    runs = SimulationRun.objects.all().order_by('-created_at')[:50]
    data = list(runs.values())
    return JsonResponse(data, safe=False)


@csrf_exempt
def run_simulation(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    lte_agents = int(data.get('lte_agents', 3))
    wifi_aps   = int(data.get('wifi_aps', 5))
    traffic    = int(data.get('traffic', 50))
    algorithm  = data.get('algorithm', 'madrl')

    try:
        results = run_real_simulation(lte_agents, wifi_aps, traffic, algorithm)
    except Exception as e:
        logger.exception("Simulation engine error")
        return JsonResponse({'error': f'Simulation failed: {str(e)}'}, status=500)

    try:
        lte_pl_avg = np.mean(results['packet_loss']['lte']) if results['packet_loss']['lte'] else 0.0
        SimulationRun.objects.create(
            lte_agents=lte_agents,
            wifi_aps=wifi_aps,
            traffic=traffic,
            algorithm=algorithm,
            category=results.get('category', ''),
            throughput_lte=sum(results['throughput']['lte']),
            throughput_wifi=results['throughput']['wifi'],
            throughput_total=results['throughput']['total'],
            fairness=results['fairness'],
            packet_loss_lte=round(float(lte_pl_avg), 2),
            packet_loss_wifi=results['packet_loss']['wifi'],
            latency_lte=results['latency']['lte'],
            latency_wifi=results['latency']['wifi'],
            duty_cycle=np.mean(results['rl']['duty_cycles']) if results['rl']['duty_cycles'] else 0.0,
            reward=results['rl']['reward'],
        )
    except Exception as e:
        logger.warning("Failed to save simulation run: %s", e)

    return JsonResponse(results, safe=False)
