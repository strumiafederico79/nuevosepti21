from __future__ import annotations

import time
from typing import Any, Optional

try:
    from .job_store import JobStore
except ImportError:  # pragma: no cover - fallback for direct script execution
    from job_store import JobStore


class JobService:
    """Servicio de jobs con workflow profesional para mastering y exports."""

    VALID_STAGES = {
        "queued",
        "analyzing",
        "preparing_assets",
        "rendering",
        "generating_preview",
        "preview_ready",
        "failed",
        "archived",
        "canceled",
    }

    def __init__(self, storage_dir: Optional[str] = None):
        self._store = JobStore(storage_dir=storage_dir)
        self._queue = []
        self._projects: dict[str, dict] = {}

    def create_project(self, project_id: str, payload: Optional[dict] = None) -> dict:
        project = dict(payload or {})
        project.setdefault("project_id", str(project_id))
        project.setdefault("status", "active")
        project.setdefault("created_at", time.time())
        project.setdefault("updated_at", project["created_at"])
        project.setdefault("versions", [])
        self._projects[str(project_id)] = project
        return self.get_project(str(project_id))

    def get_project(self, project_id: str) -> dict:
        project = self._projects[str(project_id)]
        return dict(project)

    def list_projects(self) -> dict:
        return {project_id: dict(project) for project_id, project in self._projects.items()}

    def _ensure_project_version(self, project_id: str, version_name: str) -> dict:
        project = self._projects.setdefault(str(project_id), {"project_id": str(project_id), "versions": []})
        project.setdefault("status", "active")
        project.setdefault("created_at", time.time())
        project.setdefault("updated_at", project["created_at"])
        project.setdefault("versions", [])
        for version in project["versions"]:
            if version.get("version_name") == str(version_name):
                return version
        version = {
            "version_name": str(version_name),
            "created_at": time.time(),
            "exports": [],
        }
        project["versions"].append(version)
        project["updated_at"] = time.time()
        return version

    def enqueue_job(self, job_id: str, *, priority: int = 5, metadata: Optional[dict] = None) -> dict:
        job = self._store.setdefault(job_id, {})
        job["priority"] = int(priority)
        job["status"] = "queued"
        job["stage"] = "queued"
        if metadata:
            job.setdefault("queue_metadata", {}).update(metadata)
        # BUGFIX: evitar llamar get_job() dos veces por cada job en la llave
        # Cachear los datos necesarios para sorting
        def _sort_key(jid):
            j = self.get_job(jid)
            return (-j.get("priority", 0), j.get("created_at", 0))
        self._queue = sorted(
            [*self._queue, job_id],
            key=_sort_key,
        )
        self._queue = list(dict.fromkeys(self._queue))
        return self.get_job(job_id)

    def dequeue_next_job(self) -> Optional[str]:
        if not self._queue:
            return None
        next_job = self._queue.pop(0)
        if self.exists(next_job):
            job = self.get_job(next_job)
            if job.get("status") == "queued":
                self.update_job(next_job, status="processing", stage="rendering")
        return next_job

    def get_queue(self) -> list[dict]:
        jobs = []
        for job_id in self._queue:
            if self.exists(job_id):
                job = self.get_job(job_id)
                jobs.append({"job_id": job_id, "status": job.get("status"), "priority": job.get("priority", 0), "stage": job.get("stage"), "filename": job.get("filename")})
        return jobs

    def create_job(self, job_id: str, payload: Optional[dict] = None) -> dict:
        job = dict(payload or {})
        job.setdefault("status", "queued")
        job.setdefault("stage", "queued")
        job.setdefault("progress", 0)
        job.setdefault("created_at", time.time())
        job.setdefault("logs", [])
        self._store[job_id] = job
        priority = int(job.get("priority", 5))
        self.enqueue_job(job_id, priority=priority, metadata={"source": job.get("type", "mastering")})
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict:
        return dict(self._store[job_id])

    def get_all(self) -> dict:
        return {job_id: dict(job) for job_id, job in self._store.items()}

    def update_job(self, job_id: str, **updates: Any) -> dict:
        job = self._store.setdefault(job_id, {})
        job.update(updates)
        if "status" in updates and updates["status"] == "done":
            job.setdefault("finished_at", time.time())
        if "status" in updates and updates["status"] == "archived":
            self._queue = [jid for jid in self._queue if jid != job_id]
        return self.get_job(job_id)

    def set_stage(self, job_id: str, stage: str, *, progress: Optional[int] = None, status: Optional[str] = None, eta_seconds: Optional[float] = None) -> dict:
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Estado de job inválido: {stage}")
        job = self._store.setdefault(job_id, {})
        job["stage"] = stage
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = int(progress)
        if eta_seconds is not None:
            job["eta_seconds"] = float(eta_seconds)
        if job.get("status") in {"queued", "processing", "done"} and stage in {"analyzing", "preparing_assets", "rendering", "generating_preview"}:
            job["status"] = "processing"
        return self.get_job(job_id)

    def append_log(self, job_id: str, message: str, *, stage: Optional[str] = None, level: str = "info", **extra: Any) -> dict:
        job = self._store.setdefault(job_id, {})
        job.setdefault("logs", [])
        entry = {
            "timestamp": time.time(),
            "message": str(message),
            "level": level,
            "stage": stage or job.get("stage", "queued"),
        }
        entry.update(extra)
        job["logs"].append(entry)
        job["last_log"] = entry
        return self.get_job(job_id)

    def snapshot_params(self, job_id: str, params: Optional[dict]) -> dict:
        job = self._store.setdefault(job_id, {})
        if params is None:
            params = {}
        job["preset_snapshot"] = dict(params)
        job["params_snapshot_at"] = time.time()
        job.setdefault("params", {})
        return self.get_job(job_id)

    def set_eta(self, job_id: str, eta_seconds: float) -> dict:
        job = self._store.setdefault(job_id, {})
        job["eta_seconds"] = float(eta_seconds)
        return self.get_job(job_id)

    def mark_preview_ready(self, job_id: str, preview_path: Optional[str] = None, *, preview_seconds: Optional[float] = None) -> dict:
        job = self._store.setdefault(job_id, {})
        job["status"] = "preview_ready"
        job["stage"] = "preview_ready"
        job["progress"] = 90
        if preview_path:
            job["preview_path"] = preview_path
        if preview_seconds is not None:
            job["preview_seconds"] = float(preview_seconds)
        return self.get_job(job_id)

    def mark_failed(self, job_id: str, error: str, *, stage: str = "failed", progress: int = 0) -> dict:
        job = self._store.setdefault(job_id, {})
        job["status"] = "failed"
        job["stage"] = stage
        job["progress"] = int(progress)
        job["error"] = str(error)
        job["failed_at"] = time.time()
        return self.get_job(job_id)

    def archive(self, job_id: str, *, reason: Optional[str] = None) -> dict:
        job = self._store.setdefault(job_id, {})
        job["status"] = "archived"
        job["stage"] = "archived"
        job["progress"] = 100
        job["archived_at"] = time.time()
        if reason:
            job["archive_reason"] = reason
        return self.get_job(job_id)

    def add_version(self, entity_id: str, version_name: str, *, export_path: Optional[str] = None, format: Optional[str] = None, **extra: Any) -> dict:
        if entity_id in self._projects:
            project = self._projects.setdefault(str(entity_id), {"project_id": str(entity_id), "versions": []})
            version = self._ensure_project_version(str(entity_id), str(version_name))
            version.setdefault("exports", [])
            version["created_at"] = version.get("created_at", time.time())
            version["export_path"] = export_path
            version["format"] = format
            version.update(extra)
            project["updated_at"] = time.time()
            return dict(version)

        job = self._store.setdefault(entity_id, {})
        job.setdefault("versions", [])
        version = {
            "version_name": str(version_name),
            "created_at": time.time(),
            "export_path": export_path,
            "format": format,
        }
        version.update(extra)
        job["versions"].append(version)
        return self.get_job(entity_id)

    def add_export(self, entity_id: str, *args: Any, **kwargs: Any) -> dict:
        if entity_id in self._projects:
            if len(args) == 3:
                version_name, export_id, export_path = args
            elif len(args) == 2:
                version_name = kwargs.pop("version_name", None)
                export_id, export_path = args
            else:
                raise TypeError("add_export(project_id, version_name, export_id, export_path, ...) requiere 3 argumentos posicionales")

            project = self._projects.setdefault(str(entity_id), {"project_id": str(entity_id), "versions": []})
            version = self._ensure_project_version(str(entity_id), str(version_name or "default"))
            version.setdefault("exports", [])
            export = {
                "export_id": str(export_id),
                "path": str(export_path),
                "format": kwargs.get("format"),
                "created_at": time.time(),
                "version_name": version_name,
            }
            export.update(kwargs)
            version["exports"].append(export)
            version["updated_at"] = time.time()
            project["updated_at"] = time.time()
            return dict(version)

        version_name = kwargs.pop("version_name", None)
        if len(args) == 2:
            export_id, export_path = args
        elif len(args) == 1:
            export_id = args[0]
            export_path = kwargs.pop("export_path")
        else:
            raise TypeError("add_export(job_id, export_id, export_path, ...) requiere 2 argumentos posicionales")

        job = self._store.setdefault(entity_id, {})
        job.setdefault("exports", [])
        export = {
            "export_id": str(export_id),
            "path": str(export_path),
            "format": kwargs.get("format"),
            "created_at": time.time(),
            "version_name": version_name,
        }
        export.update(kwargs)
        job["exports"].append(export)
        if version_name:
            version_extra = dict(kwargs)
            version_extra.pop("format", None)
            self.add_version(entity_id, version_name, export_path=export_path, format=kwargs.get("format"), **version_extra)
        return self.get_job(entity_id)

    def set_preview(self, job_id: str, preview_path: str, *, duration_seconds: Optional[float] = None, format: str = "wav") -> dict:
        job = self._store.setdefault(job_id, {})
        job["preview_path"] = preview_path
        job["preview_format"] = format
        if duration_seconds is not None:
            job["preview_seconds"] = float(duration_seconds)
        return self.mark_preview_ready(job_id, preview_path, preview_seconds=duration_seconds)

    def exists(self, job_id: str) -> bool:
        return job_id in self._store

    def delete(self, job_id: str) -> None:
        self._store.pop(job_id, None)
