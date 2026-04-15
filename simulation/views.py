import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .engine import run_real_simulation
from .models import SimulationRun
from django.core.serializers import serialize
from django.shortcuts import render


def simulation_history(request):
    """API endpoint to get simulation history"""
    runs = SimulationRun.objects.all().order_by('-created_at')[:50]
    data = list(runs.values())
    return JsonResponse(data, safe=False)
# later we import your simulation here

"""@csrf_exempt
def run_simulation(request):
    #API endpoint to run simulation
    
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    # Extract parameters
    lte_agents = data.get('lte_agents', 0)
    wifi_aps = data.get('wifi_aps', 0)
    traffic = data.get('traffic', 3)
    algorithm = data.get('algorithm', 'madrl')
    
    # Run simulation
    results = run_real_simulation(lte_agents, wifi_aps, traffic, algorithm)
    
    # Save to database
    simulation_run = SimulationRun.objects.create(
        lte_agents=lte_agents,
        wifi_aps=wifi_aps,
        traffic=traffic,
        algorithm=algorithm,
        throughput_lte=results['throughput'][0],
        throughput_wifi=results['throughput'][1],
        throughput_total=results['throughput'][2],
        fairness=results['fairness'][0],
        packet_loss_lte=results['packet_loss'][0],
        packet_loss_wifi=results['packet_loss'][1],
        duty_cycle=results['rl']['duty_cycle'] if results.get('rl') else None,
        reward=results['rl']['reward'] if results.get('rl') else None,
    )
    
    # Add history to response for RL
    if results.get('rl') and results['rl'].get('history'):
        results['rl']['history'] = results['rl']['history']
    
    return JsonResponse(results)
"""


@csrf_exempt
def run_simulation(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            lte_agents = data.get('lte_agents')
            wifi_aps = data.get('wifi_aps')
            traffic = data.get('traffic')
            algorithm = data.get('algorithm')

            # 🔥 THIS is where real simulation will go
            results = run_real_simulation(
                lte_agents,
                wifi_aps,
                traffic,
                algorithm
            )

            return JsonResponse(results, safe=False)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def dashboard(request):
    return render(request, 'simulation/dashboard.html')