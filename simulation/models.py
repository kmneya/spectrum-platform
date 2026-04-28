from django.db import models

class SimulationRun(models.Model):
    # Input parameters
    lte_agents = models.IntegerField()
    wifi_aps = models.IntegerField()
    traffic = models.IntegerField()
    algorithm = models.CharField(max_length=50)
    category = models.CharField(max_length=20, default='')

    # Output results
    throughput_lte = models.FloatField()
    throughput_wifi = models.FloatField()
    throughput_total = models.FloatField()
    fairness = models.FloatField()
    packet_loss_lte = models.FloatField()
    packet_loss_wifi = models.FloatField()
    latency_lte = models.FloatField(null=True, blank=True)
    latency_wifi = models.FloatField(null=True, blank=True)

    # RL data
    duty_cycle = models.FloatField(null=True, blank=True)
    reward = models.FloatField(null=True, blank=True)

    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Run {self.id} - {self.algorithm} [{self.category}] - {self.created_at}"
