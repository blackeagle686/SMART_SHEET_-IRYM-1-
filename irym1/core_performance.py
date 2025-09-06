import time
import tracemalloc
import json
from logs.models import PerformanceLog

PERFORMANCE_REPORT = {}



def save_report_json(filename="performance_report.json"):
    with open(filename, "w") as f:
        json.dump(PERFORMANCE_REPORT, f, indent=4)

def print_report():
    print("=" * 40)
    print("📊 PERFORMANCE REPORT")
    print("=" * 40)
    for func, metrics in PERFORMANCE_REPORT.items():
        print(f"Function: {func}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("-" * 40)


def performance_logger(func):
    def wrapper(*args, **kwargs):
        # Start memory tracking
        tracemalloc.start()

        # Start time
        start_time = time.time()

        # Run function
        result = func(*args, **kwargs)

        # End time
        end_time = time.time()

        # Get memory info
        current, peak = tracemalloc.get_traced_memory()

        # Stop memory tracking
        tracemalloc.stop()

        # Save performance info into DB
        PerformanceLog.objects.create(
            function_name=func.__name__,
            time_taken_sec=round(end_time - start_time, 6),
            memory_current_kb=round(current / 1024, 3),
            memory_peak_kb=round(peak / 1024, 3),
            # لو عندك user في kwargs ضيفه هنا 👇
            user=kwargs.get("request").user if "request" in kwargs else None
        )

        return result
    return wrapper