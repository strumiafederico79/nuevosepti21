from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import os
import time
import uuid

try:
    from ..auth import get_current_user
except ImportError:
    from auth import get_current_user

router = APIRouter()


def create_router(**dependencies):
    global jobs, read_and_validate, run_stems_job, validate_audio_file
    global _get_input_duration, UPLOAD_DIR

    # Attach dependencies as router attributes for backwards compatibility
    for key, val in dependencies.items():
        setattr(router, key, val)

    return router


@router.post("/stems/separate", tags=["Stems"])
async def stems_separate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("demucs_4stem"),
    current_user: dict = Depends(get_current_user),
):
    """Separa el track en stems con Demucs (mode="demucs_4stem", default:
    vocals/drums/bass/other) o con BS-RoFormer/Mel-RoFormer (mode="vocals_hq":
    solo vocals/instrumental, mejor aislamiento de voz), analiza cada uno
    individualmente y detecta colisiones espectrales entre ellos (ej. kick
    tapando al bajo — solo aplica en modo demucs_4stem). Encola el job igual
    que /master — se pollea con el mismo /job/{job_id} de siempre."""
    if mode not in ("demucs_4stem", "vocals_hq"):
        raise HTTPException(400, f"mode inválido: \'{mode}\'. Válidos: demucs_4stem, vocals_hq")
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(data)

    duration = _get_input_duration(input_path)
    job_params = {"mode": mode}
    if duration is not None:
        job_params["_input_duration_sec"] = duration

    jobs.create_job(job_id, {
        "status": "queued", "type": "stems", "filename": file.filename,
        "created_at": time.time(), "params": job_params, "progress": 0, "stage": "En cola",
    })
    background_tasks.add_task(run_stems_job, job_id, input_path, mode)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/job/{job_id}"}


@router.get("/stems/download/{job_id}/{stem_name}", tags=["Stems"], dependencies=[Depends(get_current_user)])
def stems_download(job_id: str, stem_name: str):
    if not jobs.exists(job_id):
        raise HTTPException(404, "Job no encontrado")
    job = jobs.get_job(job_id)
    if job.get("type") != "stems" or job["status"] != "done":
        raise HTTPException(400, f"Job no listo: {job.get('status')}")
    stem_path = job.get("stem_paths", {}).get(stem_name)
    if not stem_path or not os.path.exists(stem_path):
        raise HTTPException(410, "Stem no encontrado o expirado. Volvé a separar el track.")
    return FileResponse(stem_path, media_type="audio/wav", filename=f"{stem_name}.wav")