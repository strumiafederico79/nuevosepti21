"""API de proyectos: CRUD, versiones, exportes, descarga."""
from __future__ import annotations

import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

try:
    from ..auth import get_current_user
except ImportError:
    from auth import get_current_user


def create_projects_router(*, jobs, sanitize_track_name, processed_dir: Optional[str] = None) -> APIRouter:
    """Router de gestión de proyectos con versiones y exportes.

    Endpoints:
      POST   /projects               → crear proyecto
      GET    /projects               → listar proyectos
      GET    /projects/{id}          → detalle de proyecto
      PUT    /projects/{id}          → actualizar metadatos
      POST   /projects/{id}/versions → crear versión (asocia job)
      GET    /projects/{id}/versions/{name} → detalle de versión
      POST   /projects/{id}/versions/{name}/exports → registrar exportes
      GET    /projects/{id}/versions/{name}/download/{export_id} → descargar
      DELETE /projects/{id}          → archivar proyecto

    SEGURIDAD (bugfix): antes NINGÚN endpoint de acá exigía sesión, y
    list_projects() devolvía los proyectos de TODOS los usuarios sin
    filtrar — cualquiera con la URL pública de la API podía listar, leer,
    editar y borrar los proyectos de cualquier otra persona sin loguearse.
    Ahora cada ruta exige `get_current_user` y se valida que el proyecto
    pertenezca al usuario que hace el pedido (`owner_id`), devolviendo 404
    — no 403 — cuando no es el dueño, para no filtrar ni siquiera que el
    proyecto existe.

    Además, `register_export` aceptaba `export_path` tal cual mandado por
    el cliente y solo chequeaba que el archivo existiera en disco —
    `download_export` lo servía después con FileResponse sin validar nada
    más. Eso es lectura arbitraria de archivos del servidor (con la ruta
    correcta se podía pedir el .env, users_db.json, etc.). Ahora se exige
    que el path resuelto quede confinado adentro de `processed_dir`.
    """
    router = APIRouter()
    processed_dir = processed_dir or os.path.join(os.getcwd(), "processed")

    def _ensure_export_dir() -> str:
        os.makedirs(processed_dir, exist_ok=True)
        return processed_dir

    def _get_owned_project(project_id: str, user_id: str) -> dict:
        """Busca el proyecto y confirma que pertenece a user_id.
        404 (no 403) en ambos casos de fallo — no hay que revelar si un
        project_id ajeno existe."""
        try:
            project = jobs.get_project(project_id)
        except KeyError:
            raise HTTPException(404, f"Proyecto {project_id} no encontrado")
        if project.get("owner_id") != user_id:
            raise HTTPException(404, f"Proyecto {project_id} no encontrado")
        return project

    def _safe_export_path(export_path: str) -> str:
        """Confirma que export_path resuelve a una ruta adentro de
        processed_dir. Lanza 400 si intenta escapar (../, ruta absoluta
        fuera de la carpeta, symlink hacia afuera, etc.)."""
        base = os.path.realpath(_ensure_export_dir())
        resolved = os.path.realpath(export_path)
        if os.path.commonpath([base, resolved]) != base:
            raise HTTPException(
                400,
                "export_path debe estar dentro de la carpeta de procesados del servidor.",
            )
        return resolved

    # ─────────────────────────────────────────────────────────────────
    # Proyectos: CRUD
    # ─────────────────────────────────────────────────────────────────

    @router.post("/projects", tags=["Projects"], response_model=dict)
    def create_project(
        title: str,
        artist: Optional[str] = None,
        metadata: Optional[dict] = None,
        current_user: dict = Depends(get_current_user),
    ):
        """Crear un proyecto nuevo (contenedor de versiones/exportes)."""
        project_id = str(uuid.uuid4())[:8]
        payload = {
            "title": str(title),
            "artist": str(artist or "Unknown"),
            "status": "active",
            "owner_id": current_user["id"],
        }
        if metadata:
            payload.update(metadata)
        return jobs.create_project(project_id, payload)

    @router.get("/projects", tags=["Projects"], response_model=dict)
    def list_projects(current_user: dict = Depends(get_current_user)):
        """Listar los proyectos del usuario logueado."""
        return {
            "projects": [
                v for v in jobs.list_projects().values()
                if v.get("owner_id") == current_user["id"]
            ]
        }

    @router.get("/projects/{project_id}", tags=["Projects"], response_model=dict)
    def get_project(project_id: str, current_user: dict = Depends(get_current_user)):
        """Obtener detalles de un proyecto."""
        return _get_owned_project(project_id, current_user["id"])

    @router.put("/projects/{project_id}", tags=["Projects"], response_model=dict)
    def update_project(
        project_id: str,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        status: Optional[str] = None,
        current_user: dict = Depends(get_current_user),
    ):
        """Actualizar metadatos del proyecto."""
        project = _get_owned_project(project_id, current_user["id"])

        if title:
            project["title"] = str(title)
        if artist:
            project["artist"] = str(artist)
        if status:
            project["status"] = str(status)

        project["updated_at"] = time.time()
        return project

    @router.delete("/projects/{project_id}", tags=["Projects"], response_model=dict)
    def archive_project(
        project_id: str,
        reason: Optional[str] = None,
        current_user: dict = Depends(get_current_user),
    ):
        """Archivar un proyecto (no borrar, mantener histórico)."""
        project = _get_owned_project(project_id, current_user["id"])

        project["status"] = "archived"
        project["archived_at"] = time.time()
        if reason:
            project["archive_reason"] = str(reason)
        return project

    # ─────────────────────────────────────────────────────────────────
    # Versiones: CRUD
    # ─────────────────────────────────────────────────────────────────

    @router.get("/projects/{project_id}/versions/{version_name}", tags=["Projects"])
    def get_version(
        project_id: str,
        version_name: str,
        current_user: dict = Depends(get_current_user),
    ):
        """Obtener detalles de una versión específica."""
        project = _get_owned_project(project_id, current_user["id"])

        for version in project.get("versions", []):
            if version.get("version_name") == version_name:
                return version

        raise HTTPException(404, f"Versión {version_name} no encontrada en {project_id}")

    @router.post("/projects/{project_id}/versions", tags=["Projects"])
    def create_version(
        project_id: str,
        version_name: str,
        job_id: Optional[str] = None,
        preset_snapshot: Optional[dict] = None,
        description: Optional[str] = None,
        current_user: dict = Depends(get_current_user),
    ):
        """Crear una versión nueva asociada con un job (resultado de mastering)."""
        _get_owned_project(project_id, current_user["id"])

        payload = {
            "version_name": str(version_name),
            "job_id": str(job_id) if job_id else None,
            "preset_snapshot": dict(preset_snapshot or {}),
            "description": str(description) if description else None,
        }

        jobs.add_version(project_id, version_name, **payload)
        return jobs.get_project(project_id)

    # ─────────────────────────────────────────────────────────────────
    # Exportes: registro y descarga
    # ─────────────────────────────────────────────────────────────────

    @router.post("/projects/{project_id}/versions/{version_name}/exports", tags=["Projects"])
    def register_export(
        project_id: str,
        version_name: str,
        export_id: str,
        export_path: str,
        format: Optional[str] = None,
        bit_depth: Optional[int] = None,
        bitrate: Optional[str] = None,
        platform_target: Optional[str] = None,
        current_user: dict = Depends(get_current_user),
    ):
        """Registrar un archivo de exportación en una versión."""
        _get_owned_project(project_id, current_user["id"])

        safe_path = _safe_export_path(export_path)
        if not os.path.exists(safe_path):
            raise HTTPException(410, f"Archivo {export_path} no existe")

        payload = {
            "format": str(format) if format else None,
            "bit_depth": int(bit_depth) if bit_depth else None,
            "bitrate": str(bitrate) if bitrate else None,
            "platform_target": str(platform_target) if platform_target else None,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        jobs.add_export(project_id, version_name, export_id, safe_path, **payload)
        return jobs.get_project(project_id)

    @router.get("/projects/{project_id}/versions/{version_name}/download/{export_id}", tags=["Projects"])
    def download_export(
        project_id: str,
        version_name: str,
        export_id: str,
        name: Optional[str] = Query(None, description="Nombre del archivo a descargar"),
        current_user: dict = Depends(get_current_user),
    ):
        """Descargar un archivo de exportación específico de una versión."""
        project = _get_owned_project(project_id, current_user["id"])

        version = None
        for v in project.get("versions", []):
            if v.get("version_name") == version_name:
                version = v
                break

        if not version:
            raise HTTPException(404, f"Versión {version_name} no encontrada")

        export = None
        for exp in version.get("exports", []):
            if exp.get("export_id") == export_id:
                export = exp
                break

        if not export:
            raise HTTPException(404, f"Exportación {export_id} no encontrada")

        export_path = export.get("path")
        if not export_path or not os.path.exists(export_path):
            raise HTTPException(410, "Archivo de exportación expirado o no disponible")
        # Blindaje extra: aunque ya se validó al registrar, se revalida acá
        # por si el JSON de jobs fue editado a mano o vino de una versión
        # vieja sin el chequeo.
        export_path = _safe_export_path(export_path)

        fmt = export.get("format", "wav")
        media_type_map = {
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "aiff": "audio/aiff",
            "aac": "audio/aac",
            "wav": "audio/wav",
        }
        media_type = media_type_map.get(fmt, "audio/wav")

        # Sanear nombre del proyecto para usarlo en el archivo
        project_name = sanitize_track_name(name or project.get("title", "export"))
        filename = f"{project_name}_{version_name}_{export_id}.{fmt}"

        return FileResponse(export_path, media_type=media_type, filename=filename)

    @router.get("/projects/{project_id}/versions/{version_name}/exports", tags=["Projects"])
    def list_exports(
        project_id: str,
        version_name: str,
        current_user: dict = Depends(get_current_user),
    ):
        """Listar todos los exportes de una versión."""
        project = _get_owned_project(project_id, current_user["id"])

        for version in project.get("versions", []):
            if version.get("version_name") == version_name:
                return {
                    "version_name": version_name,
                    "exports": version.get("exports", []),
                }

        raise HTTPException(404, f"Versión {version_name} no encontrada")

    @router.get("/projects/{project_id}/all-exports", tags=["Projects"])
    def list_all_project_exports(
        project_id: str,
        current_user: dict = Depends(get_current_user),
    ):
        """Listar todos los exportes de todas las versiones de un proyecto."""
        project = _get_owned_project(project_id, current_user["id"])

        all_exports = []
        for version in project.get("versions", []):
            for export in version.get("exports", []):
                export_copy = dict(export)
                export_copy["version_name"] = version.get("version_name")
                all_exports.append(export_copy)

        return {"project_id": project_id, "project_title": project.get("title"), "exports": all_exports}

    return router
