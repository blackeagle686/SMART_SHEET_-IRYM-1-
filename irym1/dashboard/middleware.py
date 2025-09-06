from django.utils.deprecation import MiddlewareMixin
from core_performance import performance_logger,PERFORMANCE_REPORT, print_report

class PerformanceReportMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        if PERFORMANCE_REPORT:  # لو فيه داتا
            print_report()      # اطبعها في الـ console
        return response
