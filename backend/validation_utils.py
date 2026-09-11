import os
from typing import Optional
import numpy as np

from fastapi import HTTPException

# Importar MAX_FILE_SIZE desde config.py (fuente única de verdad)
try:
    from .config import MAX_FILE_SIZE
except ImportError:
    from config import MAX_FILE_SIZE

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}

_BOOL_QUERY_KEYS = {
    "use_lufs_normalize", "comp_stereo_link", "comp_bypass", "stereo_bypass", "limiter_bypass", "mb_bypass", "mb_stereo_bypass",
    "use_stereo_enhancer", "glue_bypass",
    # BUGFIX: faltaban estas — sin estar en el set, coerce_ws_chain_params()
    # dejaba pasar el string literal "false" (truthy en Python) en vez de
    # convertirlo a bool, así que dynamic_eq_band() siempre veía bypass=True
    # sin importar el checkbox. Esto rompía los meters de GR de de-esser y
    # resonancias dinámicas (y potencialmente lp/ms_eq/clipper/nr) solo en
    # el preview en vivo por WebSocket — el render final vía /master no se
    # veía afectado porque ahí FastAPI parsea los Query(bool) correctamente.
    "dyneq_bypass", "reso_bypass", "lp_bypass", "ms_eq_bypass", "ms_comp_bypass",
    "clipper_bypass", "nr_bypass",
    "parallel_bypass",   # BUGFIX: faltaba — sin esto, el string "false" llegaba truthy por WS
    # Toggles de PDR (Program-Dependent Release) agregados en el compresor
    # de banda ancha/paralela, glue, multibanda y M/S — mismo bug potencial
    # que los bypass de arriba si no se declaran acá.
    "comp_pdr", "glue_pdr", "mb_pdr", "ms_comp_pdr",
}


def validate_audio_file(filename: str) -> None:
    if not filename or not isinstance(filename, str):
        raise HTTPException(400, "Nombre de archivo inválido o faltante.")
    # El nombre se usa posteriormente para construir rutas temporales.
    # Rechazar separadores evita path traversal en archivos multipart maliciosos.
    if "/" in filename or "\\" in filename:
        raise HTTPException(400, "Nombre de archivo inválido.")
    if filename in {".", ".."} or "\x00" in filename:
        raise HTTPException(400, "Nombre de archivo inválido.")
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Formato '{ext}' no soportado. Válidos: {sorted(ALLOWED_EXTENSIONS)}")


def coerce_ws_chain_params(params: dict) -> dict:
    """Convierte params recibidos por WebSocket desde URLSearchParams/JSON."""
    out = {}
    for key, value in params.items():
        if key in _BOOL_QUERY_KEYS:
            if isinstance(value, str):
                out[key] = value.strip().lower() in {"1", "true", "yes", "on", "sí", "si"}
            else:
                out[key] = bool(value)
            continue
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                continue
            try:
                fval = float(value)
                # BUGFIX: rechazar NaN e Inf que podrían romper el procesamiento
                if not np.isfinite(fval):
                    raise ValueError(f"Valor inválido para {key}: {value} (NaN o Inf)")
                out[key] = fval
                continue
            except ValueError:
                pass
        out[key] = value
    return out


def read_and_validate_upload(file, max_file_size: int = MAX_FILE_SIZE) -> bytes:
    data = file.read()
    if len(data) > max_file_size:
        raise HTTPException(413, f"Archivo demasiado grande. Máximo: {max_file_size // 1024 // 1024} MB")
    return data
