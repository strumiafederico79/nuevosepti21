from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional, List, Dict
import os, uuid, logging, time, asyncio, math, json, warnings

# Python 3.14 puede emitir SyntaxWarning al compilar pydub.utils antiguo.
# El warning pertenece a la dependencia, no a LGMDM; se silencia únicamente
# para ese modulo para mantener el arranque limpio sin ocultar warnings propios.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pydub\.utils")
import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field
try:
    from .job_service import JobService
    from .audio_service import AudioService
    from .validation_utils import MAX_FILE_SIZE, coerce_ws_chain_params, validate_audio_file
    from .audio_cache import get as audio_cache_get, put as audio_cache_put
    from . import library
    from .job_runners import create_job_runners
    from .routers import (
        create_ai_router,
        create_analysis_router,
        create_auth_router,
        create_dashboard_router,
        create_info_router,
        create_jobs_router,
        create_library_router,
        create_projects_router,
        create_reference_library_router,
        create_audio_router,
        create_mastering_router, create_mixer_router, create_stems_router, create_streaming_router,
        create_preview_router,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from job_service import JobService
    from audio_service import AudioService
    from validation_utils import MAX_FILE_SIZE, coerce_ws_chain_params, validate_audio_file
    from audio_cache import get as audio_cache_get, put as audio_cache_put
    import library
    from job_runners import create_job_runners
    from routers import (
        create_ai_router,
        create_analysis_router,
        create_auth_router,
        create_dashboard_router,
        create_info_router,
        create_jobs_router,
        create_library_router,
        create_projects_router,
        create_reference_library_router,
        create_audio_router,
        create_mastering_router, create_mixer_router, create_stems_router, create_streaming_router,
        create_preview_router,
    )
try:
    from .mastering import (
        process_audio, analyze_audio, spectrum_analysis_fft, mix_advice, apply_mastering_chain,
        MASTERING_PRESETS, get_preset, PLATFORM_LOUDNESS_TARGETS, get_platform_target,
        process_audio_with_reference, _crop_preview, measure_lufs_integrated,
        compute_ms_eq_curves, apply_ms_matching_fir,
        compute_lufs_corrected_gain,
        spectral_energy_at_bands, compute_reference_eq_curve, compute_reference_eq_curve_ddsp,
        build_matching_fir, apply_matching_fir, eq_high_pass, eq_parametric_band,
        spectral_energy_at_bands_multires,
        derive_mb_chain_params_from_reference,
        normalize_by_lufs,
    )
    from .streaming_engine import master_stream_to_pcm16, iter_mastering_chunks
    from .mixer import mix_and_master, StemParams, MixParams, process_stem, apply_sidechain, _ensure_stereo, _match_length
    from .stem_separation import separate_stems, separate_vocals_hq
    from .stem_analysis import analyze_stems_full
    from .system_monitor import get_system_stats
    from .pitch_correction import PitchCorrectionProcessor
    from . import ai_assistant
    from .preview_service import PreviewRenderer
    from .config import UPLOAD_DIR, PROCESSED_DIR, STEMS_DIR, PROCESSED_TTL, MAX_FILE_SIZE, REFERENCE_LIBRARY_DIR, STEM_LIBRARY_DIR
    from . import reference_library as ref_lib
except ImportError:
    from mastering import (
        process_audio, analyze_audio, spectrum_analysis_fft, mix_advice, apply_mastering_chain,
        MASTERING_PRESETS, get_preset, PLATFORM_LOUDNESS_TARGETS, get_platform_target,
        process_audio_with_reference, _crop_preview, measure_lufs_integrated,
        compute_ms_eq_curves, apply_ms_matching_fir,
        compute_lufs_corrected_gain,
        spectral_energy_at_bands, compute_reference_eq_curve, compute_reference_eq_curve_ddsp,
        build_matching_fir, apply_matching_fir, eq_high_pass, eq_parametric_band,
        spectral_energy_at_bands_multires,
        derive_mb_chain_params_from_reference,
        normalize_by_lufs,
    )
    from streaming_engine import master_stream_to_pcm16, iter_mastering_chunks
    from mixer import mix_and_master, StemParams, MixParams, process_stem, apply_sidechain, _ensure_stereo, _match_length
    from stem_separation import separate_stems, separate_vocals_hq
    from stem_analysis import analyze_stems_full
    from system_monitor import get_system_stats
    from pitch_correction import PitchCorrectionProcessor
    import ai_assistant
    from preview_service import PreviewRenderer
    from config import UPLOAD_DIR, PROCESSED_DIR, STEMS_DIR, PROCESSED_TTL, MAX_FILE_SIZE, REFERENCE_LIBRARY_DIR, STEM_LIBRARY_DIR
    import reference_library as ref_lib
except ImportError:
    from streaming_engine import master_stream_to_pcm16, iter_mastering_chunks
    from mixer import mix_and_master, StemParams, MixParams, process_stem, apply_sidechain, _ensure_stereo, _match_length
    from stem_separation import separate_stems, separate_vocals_hq
    from stem_analysis import analyze_stems_full
    from system_monitor import get_system_stats
    from pitch_correction import PitchCorrectionProcessor
    import ai_assistant
    from config import UPLOAD_DIR, PROCESSED_DIR, STEMS_DIR, PROCESSED_TTL, MAX_FILE_SIZE, REFERENCE_LIBRARY_DIR, STEM_LIBRARY_DIR
    import reference_library as ref_lib

try:
    from .auth import (
        bootstrap_admin, get_current_user, get_admin_user, _verify_jwt, _get_user_by_id,
        handle_register, handle_login, handle_me,
        handle_list_users, handle_approve_user, handle_reject_user,
        handle_delete_user, handle_change_password,
    )
except ImportError:
    from auth import (
        bootstrap_admin, get_current_user, get_admin_user, _verify_jwt, _get_user_by_id,
        handle_register, handle_login, handle_me,
        handle_list_users, handle_approve_user, handle_reject_user,
        handle_delete_user, handle_change_password,
    )


def verify_ws_token(token: Optional[str]) -> bool:
    """Auth para WebSockets vía ?token=, reusable por cualquier router.

    Los WebSocket nativos del browser no pueden mandar headers custom en el
    handshake, así que Depends(get_current_user) (que lee el header
    Authorization) nunca funciona ahí — de ahí este helper aparte, con la
    misma validación (JWT válido + usuario approved) que ya usaba
    /ws/mix-stream de forma ad-hoc.
    """
    if not token:
        return False
    try:
        payload = _verify_jwt(token)
        # WebSocket frontends receive a short-lived dedicated ticket instead of
        # placing the long-lived session JWT in the query string. Direct JWTs
        # remain accepted for backward compatibility during rollout.
        if payload.get("aud") not in (None, "websocket"):
            return False
        user = _get_user_by_id(payload["sub"])
        return bool(user and user.get("status") == "approved")
    except Exception:
        return False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Mastering API", version="7.0.1")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://masteringstudio.duckdns.org").rstrip("/")
CORS_ORIGINS = [origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", FRONTEND_ORIGIN).split(",") if origin.strip()]

limiter = Limiter(key_func=get_remote_address)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Detected-Key", "X-Confidence", "X-Mode", "X-Output-LUFS", "X-Reference-Match"],
    max_age=86400,
)

@app.middleware("http")
async def enforce_frontend_cors(request, call_next):
    """Defensa de última milla para evitar que el proxy deje una respuesta sin ACAO."""
    response = await call_next(request)
    origin = request.headers.get("origin")
    if origin in CORS_ORIGINS:
        response.headers.setdefault("Access-Control-Allow-Origin", origin)
        response.headers.setdefault("Access-Control-Allow-Credentials", "true")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
        response.headers.setdefault("Access-Control-Allow-Headers", "Authorization,Content-Type,X-Requested-With")
        response.headers.setdefault("Vary", "Origin")
    return response

# Crear admin si no existe
bootstrap_admin()

UPLOAD_DIR    = UPLOAD_DIR
PROCESSED_DIR = PROCESSED_DIR
STEMS_DIR     = STEMS_DIR   # subcarpeta por job_id con los 4 WAV de stems
PROCESSED_TTL  = PROCESSED_TTL

# Librería persistente de archivos originales (a diferencia de UPLOAD_DIR, NO
# tiene TTL ni se toca en cleanup_old() — vive hasta que el usuario borra un
# archivo explícitamente desde la web). Se calcula como hermano de UPLOAD_DIR
# para no requerir tocar config.py; si preferís definirlo ahí como LIBRARY_DIR
# y pisar esta línea con el import, funciona igual.
LIBRARY_DIR = os.path.join(os.path.dirname(os.path.normpath(UPLOAD_DIR)), "library")

# Snapshots inmutables de 25s + renders cancelables del Preview en vivo
# (ver preview_service.PreviewRenderer). Mismo patrón que LIBRARY_DIR:
# hermano de UPLOAD_DIR para no tocar config.py.
PREVIEW_DIR = os.path.join(os.path.dirname(os.path.normpath(UPLOAD_DIR)), "preview")

os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
ref_lib.init(REFERENCE_LIBRARY_DIR)  # carga índice + arranca watcher
os.makedirs(STEMS_DIR,     exist_ok=True)
os.makedirs(LIBRARY_DIR,   exist_ok=True)
os.makedirs(PREVIEW_DIR,   exist_ok=True)

preview_renderer = PreviewRenderer(directory=PREVIEW_DIR)
os.makedirs(STEM_LIBRARY_DIR, exist_ok=True)

jobs = JobService()
audio_service = AudioService(upload_dir=UPLOAD_DIR)

@app.on_event("startup")
def _check_production_secrets():
    """Arranque: falla si JWT_SECRET fue generado automáticamente (no configurado)."""
    try:
        from .auth import JWT_SECRET_WAS_GENERATED
    except ImportError:
        from auth import JWT_SECRET_WAS_GENERATED
    if JWT_SECRET_WAS_GENERATED:
        import logging as _logging
        _logging.getLogger(__name__).critical(
            "JWT_SECRET not configured. Set JWT_SECRET environment variable before running in production."
        )
        raise RuntimeError(
            "JWT_SECRET environment variable must be set before running in production. "
            "Generated secrets are not persistent and compromise security."
        )


@app.on_event("shutdown")
def _shutdown_runtime_resources():
    """Cierra recursos persistentes antes de terminar el proceso."""
    try:
        from . import streaming_engine as _streaming_engine
    except ImportError:
        try:
            import streaming_engine as _streaming_engine
        except Exception:
            _streaming_engine = None
    if _streaming_engine is not None:
        try:
            _streaming_engine.shutdown_chain_pool()
        except Exception:
            logger.exception("No se pudo cerrar recursos de streaming correctamente")

    try:
        ref_lib.shutdown()
    except Exception:
        logger.exception("No se pudo detener reference_library watcher correctamente")

def sanitize_track_name(name: Optional[str], fallback: str = "mastered") -> str:
    """Limpia un nombre de tema provisto por el usuario para usarlo como filename seguro."""
    if not name:
        return fallback
    name = name.strip()
    if not name:
        return fallback
    name = name.replace("/", "-").replace("\\", "-")
    name = "".join(c for c in name if c.isprintable())
    safe = "".join(c for c in name if c.isalnum() or c in " ._-()[]áéíóúÁÉÍÓÚñÑüÜ")
    safe = safe.strip(" .")
    safe = safe[:120]
    return safe or fallback

# Magic bytes por formato — RIFF/WAVE, FORM/AIFF y fLaC/OggS tienen firma
# fija y confiable. MP3 no (ID3v2 opcional + frame sync variable sin un
# patrón simple y confiable), así que se deja sin chequear, igual que
# siempre — no vale la pena arriesgar falsos rechazos de MP3s legítimos.
_AUDIO_MAGIC_CHECKS = {
    ".wav":  lambda d: len(d) >= 12 and d[:4] == b"RIFF" and d[8:12] == b"WAVE",
    ".aiff": lambda d: len(d) >= 12 and d[:4] == b"FORM" and d[8:12] == b"AIFF",
    ".aif":  lambda d: len(d) >= 12 and d[:4] == b"FORM" and d[8:12] == b"AIFF",
    ".flac": lambda d: len(d) >= 4 and d[:4] == b"fLaC",
    ".ogg":  lambda d: len(d) >= 4 and d[:4] == b"OggS",
}


def _check_audio_magic_bytes(filename: str, data: bytes) -> None:
    ext = os.path.splitext(filename)[-1].lower()
    check = _AUDIO_MAGIC_CHECKS.get(ext)
    if check and not check(data):
        raise HTTPException(400, f"El archivo no parece ser un {ext} válido (cabecera incorrecta) — ¿le cambiaron la extensión?")


async def read_and_validate(file: UploadFile) -> bytes:
    """Lee y valida un archivo subido: extensión permitida + tamaño máximo +
    magic bytes (evita que un archivo con extensión falseada, ej. un .exe
    renombrado a .wav, pase la validación solo por el nombre)."""
    # Validar extensión PRIMERO antes de leer (evita procesar datos innecesarios)
    validate_audio_file(file.filename)
    # Luego validar tamaño
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE // 1024 // 1024} MB")
    _check_audio_magic_bytes(file.filename, data)
    logger.info(f"✓ Upload validado: {file.filename} ({len(data) / 1024 / 1024:.1f} MB)")
    return data

async def resolve_input_source(file: Optional[UploadFile], library_id: Optional[str]) -> tuple:
    """Resuelve el audio de entrada de un endpoint que acepta O un archivo
    subido O un library_id (archivo ya guardado en LIBRARY_DIR). Devuelve
    (data: bytes, filename: str). Reemplaza el
    `validate_audio_file(file.filename); data = await read_and_validate(file)`
    que se repetía en cada endpoint de mastering — agregar soporte de
    librería a un endpoint nuevo es agregar `library_id` a la firma y
    reemplazar esas dos líneas por una llamada acá."""
    if library_id:
        path = library.get_path(LIBRARY_DIR, library_id)
        if path is None:
            raise HTTPException(404, "Archivo de la librería no encontrado (¿se borró?).")
        meta = library.get_meta(LIBRARY_DIR, library_id)
        filename = meta["original_filename"] if meta else os.path.basename(path)
        with open(path, "rb") as f:
            data = f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE // 1024 // 1024} MB")
        return data, filename
    if file is None:
        raise HTTPException(400, "Falta el archivo: mandá 'file' o 'library_id'.")
    validate_audio_file(file.filename)
    data = await read_and_validate(file)
    return data, file.filename


# BUGFIX (/ai/auto-master): ai_assistant.decide_mastering() devuelve estas 4
# claves en escala LINEAL (0-1) con el nombre interno viejo del motor, pero
# process_audio()/apply_mastering_chain() esperan la versión "_db" (en dB) —
# no existe ningún "comp_threshold" ni "mb_low_threshold" etc. en la firma de
# process_audio(), que tampoco tiene **kwargs. Sin este fix, process_audio(
# **params) explota con "unexpected keyword argument" y el job de auto-master
# termina siempre en status=error. Se convierte lineal->dB (20*log10) y se
# renombra a la clave real que el motor espera.
_AI_LINEAR_TO_DB_PARAMS = {
    "comp_threshold": "comp_threshold_db",
    "mb_low_threshold": "mb_low_threshold_db",
    "mb_mid_threshold": "mb_mid_threshold_db",
    "mb_high_threshold": "mb_high_threshold_db",
}


def _fix_ai_decision_params(decision: dict) -> dict:
    """Normaliza el dict que devuelve decide_mastering() a los nombres/escala
    reales que acepta process_audio() (ver _AI_LINEAR_TO_DB_PARAMS)."""
    fixed = dict(decision)
    for linear_key, db_key in _AI_LINEAR_TO_DB_PARAMS.items():
        if linear_key in fixed:
            linear_val = fixed.pop(linear_key)
            try:
                fixed[db_key] = round(20.0 * math.log10(max(float(linear_val), 1e-6)), 2)
            except (TypeError, ValueError):
                pass
    return fixed


def _get_input_duration(input_path: str) -> Optional[float]:
    """Calcula la duración del archivo para que /dashboard pueda estimar el ETA del job."""
    duration = audio_service.get_duration(input_path)
    if duration is None:
        logger.warning(f"No se pudo calcular la duración de '{input_path}'")
    return duration


def cleanup_old() -> None:
    """Elimina archivos viejos (con TTL) en PROCESSED_DIR y STEMS_DIR.
    Evita llenar disco y reduce clutter. Se llama antes de cada job importante."""
    now = time.time()
    deleted_files = 0
    deleted_dirs = 0
    
    # Limpiar archivos individuales en PROCESSED_DIR
    try:
        for fname in os.listdir(PROCESSED_DIR):
            fpath = os.path.join(PROCESSED_DIR, fname)
            try:
                if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > PROCESSED_TTL:
                    os.remove(fpath)
                    deleted_files += 1
            except OSError as e:
                logger.warning(f"No se pudo borrar {fpath}: {e}")
    except OSError as e:
        logger.warning(f"Error accediendo PROCESSED_DIR: {e}")
    
    # Limpiar directorios viejos en STEMS_DIR
    try:
        import shutil
        for dirname in os.listdir(STEMS_DIR):
            dpath = os.path.join(STEMS_DIR, dirname)
            try:
                if os.path.isdir(dpath) and (now - os.path.getmtime(dpath)) > PROCESSED_TTL:
                    shutil.rmtree(dpath, ignore_errors=True)
                    deleted_dirs += 1
            except OSError as e:
                logger.warning(f"No se pudo borrar directorio {dpath}: {e}")
    except OSError as e:
        logger.warning(f"Error accediendo STEMS_DIR: {e}")
    
    if deleted_files > 0 or deleted_dirs > 0:
        logger.info(f"🧹 Cleanup: {deleted_files} archivos + {deleted_dirs} directorios borrados (TTL: {PROCESSED_TTL}s)")

# Workers de background: la lógica de ejecución vive en job_runners.py;
# app.py solo ensambla sus dependencias y expone la API HTTP/WebSocket.
run_mastering_job, run_reference_job, run_normalize_job, run_stems_job = create_job_runners(
    jobs=jobs,
    cleanup_old=cleanup_old,
    process_audio=process_audio,
    process_audio_with_reference=process_audio_with_reference,
    normalize_by_lufs=normalize_by_lufs,
    separate_stems=separate_stems,
    separate_vocals_hq=separate_vocals_hq,
    analyze_stems_full=analyze_stems_full,
    measure_lufs_integrated=measure_lufs_integrated,
    stems_dir=STEMS_DIR,
)


app.include_router(create_info_router(
    app=app,
    jobs=jobs,
    upload_dir=UPLOAD_DIR,
    processed_dir=PROCESSED_DIR,
    stems_dir=STEMS_DIR,
    max_file_size=MAX_FILE_SIZE,
    mastering_presets=MASTERING_PRESETS,
    get_preset=get_preset,
    platform_loudness_targets=PLATFORM_LOUDNESS_TARGETS,
    frontend_origin=FRONTEND_ORIGIN,
    reference_library_module=ref_lib,
))
# BUGFIX (seguridad): estos routers manejan jobs, archivos y datos de usuario
# pero no tenían NINGUNA dependencia de auth — con el JWT armado y todo,
# quedaban accesibles sin login (cualquiera podía listar/bajar masters
# ajenos, tocar la librería, archivar jobs, etc). `dependencies=` a nivel
# include_router aplica get_current_user a todos los endpoints HTTP del
# router de una sola vez, sin tocar cada handler. El WS de /ws/dashboard es
# la excepción — ver nota en create_dashboard_router, ahí se autentica
# aparte porque los WebSocket del browser no pueden mandar header
# Authorization.
app.include_router(create_library_router(
    library_module=library,
    library_dir=LIBRARY_DIR,
    read_and_validate=read_and_validate,
    validate_audio_file=validate_audio_file,
), dependencies=[Depends(get_current_user)])
app.include_router(create_reference_library_router(
    reference_library_module=ref_lib,
    reference_library_dir=REFERENCE_LIBRARY_DIR,
), dependencies=[Depends(get_current_user)])
# dashboard mezcla un GET normal con un WebSocket — el WS no puede llevar
# el dependencies=[...] de acá porque el browser no puede mandarle un header
# Authorization al handshake (ver /ws/mix-stream más abajo, mismo problema).
# create_dashboard_router() aplica el auth manualmente adentro para poder
# usar ?token= en el WS y Depends(get_current_user) en el GET.
app.include_router(create_dashboard_router(
    jobs=jobs,
    get_system_stats=get_system_stats,
    logger=logger,
    get_current_user=get_current_user,
    verify_ws_token=verify_ws_token,
))
app.include_router(create_jobs_router(
    jobs=jobs,
    sanitize_track_name=sanitize_track_name,
    processed_dir=PROCESSED_DIR,
), dependencies=[Depends(get_current_user)])
app.include_router(create_projects_router(
    jobs=jobs,
    sanitize_track_name=sanitize_track_name,
    processed_dir=PROCESSED_DIR,
), dependencies=[Depends(get_current_user)])
app.include_router(create_auth_router(logger=logger))
app.include_router(create_analysis_router(
    upload_dir=UPLOAD_DIR,
    read_and_validate=read_and_validate,
    logger=logger,
    current_user_dependency=get_current_user,
    audio_service=audio_service,
))
app.include_router(create_ai_router(
    upload_dir=UPLOAD_DIR,
    read_and_validate=read_and_validate,
    resolve_input_source=resolve_input_source,
    validate_audio_file=validate_audio_file,
    jobs=jobs,
    logger=logger,
    run_mastering_job=run_mastering_job,
    current_user_dependency=get_current_user,
    limiter=limiter,
))
app.include_router(create_preview_router(
    preview_renderer=preview_renderer,
    library_dir=LIBRARY_DIR,
    read_and_validate=read_and_validate,
    validate_audio_file=validate_audio_file,
    current_user_dependency=get_current_user,
))

# ── Endpoints movidos a routers/ ───────────────────────────────────────────────
# - auth
# - analysis
# - ai
# Se mantienen aquí solo utilidades de negocio y endpoints de mastering/mix.

# Audio/mastering/mix endpoints are isolated from the app entrypoint.
_audio_router_dependencies = dict(
    globals()
)
app.include_router(create_stems_router(**_audio_router_dependencies))
app.include_router(create_mastering_router(**_audio_router_dependencies))
app.include_router(create_mixer_router(**_audio_router_dependencies))
app.include_router(create_streaming_router(**_audio_router_dependencies))
