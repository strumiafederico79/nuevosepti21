from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse


def create_library_router(*, library_module, library_dir: str, read_and_validate,
                          validate_audio_file) -> APIRouter:
    router = APIRouter()

    @router.post("/library/upload", tags=["Librería"])
    async def library_upload(file: UploadFile = File(...)):
        """Guarda el archivo de forma permanente en el servidor y lo agrega a la librería."""
        validate_audio_file(file.filename)
        data = await read_and_validate(file)
        meta = await run_in_threadpool(library_module.add_file, library_dir, file.filename, data)
        return meta

    @router.get("/library", tags=["Librería"])
    def library_list(offset: int = 0, limit: int = 100):
        """Lista archivos guardados con paginación, más reciente primero."""
        if offset < 0 or limit < 1 or limit > 500:
            raise HTTPException(400, "offset>=0 y 1<=limit<=500.")
        files, total = library_module.list_files(library_dir, offset=offset, limit=limit)
        return {"files": files, "total": total, "offset": offset, "limit": limit}

    @router.get("/library/{file_id}/download", tags=["Librería"])
    def library_download(file_id: str):
        """Devuelve el archivo tal cual se guardó."""
        path = library_module.get_path(library_dir, file_id)
        if path is None:
            raise HTTPException(404, "Archivo no encontrado en la librería (¿se borró?).")
        meta = library_module.get_meta(library_dir, file_id)
        filename = meta["original_filename"] if meta else os.path.basename(path)
        return FileResponse(path, media_type="application/octet-stream", filename=filename)

    @router.delete("/library/{file_id}", tags=["Librería"])
    def library_delete(file_id: str):
        ok = library_module.delete_file(library_dir, file_id)
        if not ok:
            raise HTTPException(404, "Archivo no encontrado en la librería.")
        return {"deleted": file_id}

    return router
