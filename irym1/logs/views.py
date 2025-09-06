from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render
from .models import PerformanceLog
from django.conf import settings

paths = settings.T_PATH

@staff_member_required  # ✅ يضمن إن بس ال staff/admin يقدر يشوف
def performance_logs_view(request):
    logs = PerformanceLog.objects.all().order_by("-created_at")[:100]  # آخر 100 فقط
    return render(request, paths['logs'], {"logs": logs})
