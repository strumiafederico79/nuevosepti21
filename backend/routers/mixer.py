from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import glob as _glob
import json as _json
import os

import librosa as _lr
import numpy as np

try:
    from ..auth import get_current_user
except ImportError:
    from auth import get_current_user

router = APIRouter()


def create_router(**dependencies):
    global ai_assistant, analyze_audio, library, read_and_validate
    global validate_audio_file, UPLOAD_DIR, STEM_LIBRARY_DIR

    # Resolve dependencies injected from app.py
    get_current_user = dependencies["get_current_user"]
    read_and_validate = dependencies["read_and_validate"]
    validate_audio_file = dependencies["validate_audio_file"]
    ai_assistant = dependencies["ai_assistant"]
    library = dependencies["library"]
    analyze_audio = dependencies["analyze_audio"]
    UPLOAD_DIR = dependencies["UPLOAD_DIR"]
    STEM_LIBRARY_DIR = dependencies["STEM_LIBRARY_DIR"]

    # Attach for any direct access
    router.UPLOAD_DIR = UPLOAD_DIR
    router.STEM_LIBRARY_DIR = STEM_LIBRARY_DIR
    router.ai_assistant = ai_assistant
    router.library = library
    router.analyze_audio = analyze_audio
    router.get_current_user = get_current_user
    router.read_and_validate = read_and_validate
    router.validate_audio_file = validate_audio_file

    return router


@router.post("/mix/upload-stem", tags=["Mixer"])
async def mix_upload_stem(
    file: UploadFile = File(...),
    session_id: str = Form(..., description="ID de sesión del mixer para agrupar stems"),
    stem_name: str = Form("", description="Nombre del stem (opcional, usa el nombre del archivo si se omite)"),
    save_to_library: bool = Form(False, description="Guarda una copia reutilizable en la librería de stems del mixer"),
    current_user: dict = Depends(get_current_user),
):
    """Sube un stem individual para una sesión de mixer.
    El frontend puede subir stems de a uno y luego llamar a /mix/submit con el session_id.
    """
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    name = stem_name.strip() or os.path.splitext(file.filename)[0]
    path = os.path.join(UPLOAD_DIR, f"mix_{session_id}_{name}{os.path.splitext(file.filename)[1]}")
    with open(path, "wb") as fh:
        fh.write(data)

    # Analizar duración para el frontend
    try:
        duration = _lr.get_duration(path=path)
    except Exception:
        duration = None

    library_meta = None
    if save_to_library:
        library_meta = library.add_file(STEM_LIBRARY_DIR, file.filename, data)

    return {
        "session_id": session_id,
        "stem_name": name,
        "filename": file.filename,
        "path": path,
        "duration_sec": round(duration, 2) if duration else None,
        "library_item": library_meta,
    }


@router.get("/mix/stem-library", tags=["Mixer"], dependencies=[Depends(get_current_user)])
def mix_stem_library_list():
    """Lista stems guardados para reutilizar en futuras sesiones del mixer."""
    return {"files": library.list_files(STEM_LIBRARY_DIR)}


@router.post("/mix/stem-library/upload", tags=["Mixer"], dependencies=[Depends(get_current_user)])
async def mix_stem_library_upload(file: UploadFile = File(...)):
    """Guarda un stem directamente en la librería reutilizable del mixer."""
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    return library.add_file(STEM_LIBRARY_DIR, file.filename, data)


@router.get("/mix/stem-library/{file_id}/download", tags=["Mixer"], dependencies=[Depends(get_current_user)])
def mix_stem_library_download(file_id: str):
    path = library.get_path(STEM_LIBRARY_DIR, file_id)
    if path is None:
        raise HTTPException(404, "Stem no encontrado en la librería del mixer.")
    meta = library.get_meta(STEM_LIBRARY_DIR, file_id)
    filename = meta["original_filename"] if meta else os.path.basename(path)
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


@router.delete("/mix/stem-library/{file_id}", tags=["Mixer"], dependencies=[Depends(get_current_user)])
def mix_stem_library_delete(file_id: str):
    ok = library.delete_file(STEM_LIBRARY_DIR, file_id)
    if not ok:
        raise HTTPException(404, "Stem no encontrado en la librería del mixer.")
    return {"deleted": file_id}


@router.post("/mix/ai-suggest", tags=["Mixer"])
async def mix_ai_suggest(
    session_id: str = Form(...),
    stem_names: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Analiza los stems ya subidos y devuelve sugerencias de parámetros de mezcla
    generadas por IA. No procesa audio — solo sugiere. El usuario decide si aplica."""
    names = _json.loads(stem_names)
    if not names:
        raise HTTPException(status_code=400, detail="No hay stems para analizar")

    stems_analysis = {}

    for name in names:
        # Buscar el archivo del stem — el upload lo guarda como mix_{session_id}_{name}.{ext}
        # BUGFIX: mismo problema que _mix_session_stem_path — escapar el nombre
        # para que corchetes/paréntesis/asteriscos en el nombre del stem no se
        # interpreten como wildcards de glob.
        pattern = os.path.join(UPLOAD_DIR, _glob.escape(f"mix_{session_id}_{name}") + ".*")

        matches = _glob.glob(pattern)
        if not matches:
            raise HTTPException(status_code=404, detail=f"Stem \'{name}\' no encontrado. Subilo primero.")
        stem_path = matches[0]

        try:
            audio, sr = _lr.load(stem_path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error cargando stem \'{name}\': {e}")

        analysis = analyze_audio(audio, sr)

        # Detectar stem_type del nombre
        n_low = name.lower()
        if any(x in n_low for x in ["kick", "bd", "bombo"]):         stem_type = "kick"
        elif any(x in n_low for x in ["snare", "caja", "rim"]):      stem_type = "snare"
        elif any(x in n_low for x in ["bass", "bajo", "808"]):       stem_type = "bass"
        elif any(x in n_low for x in ["voc", "voice", "vocal"]):     stem_type = "vocals"
        elif any(x in n_low for x in ["guitar", "guit"]):            stem_type = "guitar"
        elif any(x in n_low for x in ["synth", "pad", "keys"]):      stem_type = "synth"
        elif any(x in n_low for x in ["drum", "perc", "hat"]):       stem_type = "drums"
        elif any(x in n_low for x in ["fx", "effect", "atm"]):       stem_type = "fx"
        else:                                                       stem_type = "other"

        analysis["stem_type"] = stem_type
        stems_analysis[name] = analysis

    suggestions = await ai_assistant.decide_mix(stems_analysis)
    return {"suggestions": suggestions, "stems_analyzed": list(stems_analysis.keys())}
