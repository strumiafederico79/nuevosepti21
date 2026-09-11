"""Monitor de sistema para dashboard en tiempo real: CPU, RAM, cola de jobs, estado."""
import time
from typing import Optional
import psutil

def get_system_stats(jobs: dict) -> dict:
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()

    queued = sum(1 for j in jobs.values() if j.get("status") == "queued")
    processing = sum(1 for j in jobs.values() if j.get("status") == "processing")
    done = sum(1 for j in jobs.values() if j.get("status") == "done")
    error = sum(1 for j in jobs.values() if j.get("status") == "error")

    active_job = None
    for job_id, job in jobs.items():
        if job.get("status") == "processing":
            # BUGFIX: si no hay started_at, no se puede calcular elapsed correctamente
            # El default time.time() hace que elapsed sea 0, engañando al ETA
            started_at = job.get("started_at")
            if started_at is None:
                continue  # Skip jobs sin tiempo de inicio válido
            elapsed = time.time() - started_at
            eta = estimate_remaining(job, elapsed)
            active_job = {"job_id": job_id, "filename": job.get("filename"), "elapsed_sec": round(elapsed, 1), "eta_sec": eta}
            break

    return {
        "cpu_percent": round(cpu_percent, 1),
        "ram_percent": round(mem.percent, 1),
        "ram_used_mb": round(mem.used / 1024 / 1024, 1),
        "ram_total_mb": round(mem.total / 1024 / 1024, 1),
        "queue": {"queued": queued, "processing": processing, "done": done, "error": error, "total": len(jobs)},
        "active_job": active_job,
        "timestamp": time.time(),
    }

def estimate_remaining(job: dict, elapsed: float) -> Optional[float]:
    avg_ratio = job.get("_avg_proc_ratio", 1.2)
    duration_sec = job.get("params", {}).get("_input_duration_sec")
    if not duration_sec:
        return None
    expected_total = duration_sec * avg_ratio
    remaining = max(0.0, expected_total - elapsed)
    return round(remaining, 1)