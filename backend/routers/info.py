from __future__ import annotations

from fastapi import APIRouter, HTTPException


def create_info_router(*, app, jobs, upload_dir: str, processed_dir: str, stems_dir: str,
                       max_file_size: int, mastering_presets: dict, get_preset,
                       platform_loudness_targets: dict, frontend_origin: str | None = None,
                       reference_library_module=None) -> APIRouter:
    router = APIRouter()

    @router.get("/", tags=["Info"])
    def root():
        return {
            "service": "Audio Mastering API",
            "version": app.version,
            "max_file_mb": max_file_size // 1024 // 1024,
            "endpoints": [
                "/master", "/master/sync", "/master/reference", "/master/reference/sync",
                "/preview", "/analyze", "/spectrum",
                "/mix-advice", "/job/{id}", "/download/{id}", "/report/{id}", "/report/{id}/visual",
                "/stems/separate", "/stems/download/{id}/{stem}",
                "/presets", "/preset/{name}", "/platform-targets",
                "/dashboard", "/ws/dashboard", "/ws/master-stream", "/ws/mix-stream",
            ],
        }

    @router.get("/health", tags=["Info"])
    def health():
        payload = {
            "status": "ok",
            "service": "Audio Mastering API",
            "version": app.version,
            "jobs": len(jobs.get_all()),
            "max_file_size_mb": max_file_size // 1024 // 1024,
            "frontend_origin": frontend_origin,
        }
        if reference_library_module is not None:
            payload["reference_library"] = reference_library_module.diagnostics()
        return payload

    @router.get("/presets", tags=["Presets"])
    def list_presets():
        return {name: preset for name, preset in mastering_presets.items()}

    @router.get("/preset/{name}", tags=["Presets"])
    def get_preset_endpoint(name: str):
        try:
            return get_preset(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/platform-targets", tags=["Mastering"])
    def platform_targets():
        return platform_loudness_targets

    return router
