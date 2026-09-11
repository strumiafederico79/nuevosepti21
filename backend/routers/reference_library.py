from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool


def create_reference_library_router(*, reference_library_module, reference_library_dir: str) -> APIRouter:
    router = APIRouter()

    @router.get("/reference-library", tags=["Reference Library"])
    async def get_reference_library():
        """Lista todas las referencias en REFERENCE_LIBRARY_DIR."""
        return {"entries": reference_library_module.list_entries(), "dir": reference_library_dir}

    @router.post("/reference-library/rescan", tags=["Reference Library"])
    async def rescan_reference_library():
        """Fuerza re-escaneo completo."""
        n = await run_in_threadpool(reference_library_module.scan, True)
        return {"status": "ok", "entries": n, "dir": reference_library_dir}

    return router
