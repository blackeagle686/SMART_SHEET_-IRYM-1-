from django.db import models
from django.contrib.auth.models import User


class PerformanceLog(models.Model):
    function_name = models.CharField(max_length=255)
    time_taken_sec = models.FloatField()
    memory_current_kb = models.FloatField()
    memory_peak_kb = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    # لو عايز تربطه بمستخدم منفذ العملية (اختياري)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.function_name} @ {self.created_at}"
