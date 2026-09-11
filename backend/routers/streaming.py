from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, WebSocket
from fastapi.responses import FileResponse
import json
import os
import time
import uuid

import librosa
import numpy as np
import soundfile as sf

try:
    from ..auth import get_current_user
except ImportError:
    from auth import get_current_user

router = APIRouter()


def create_router(**dependencies):
    global MAX_FILE_SIZE, UPLOAD_DIR, PROCESSED_DIR, LIBRARY_DIR
    global audio_cache_get, audio_cache_put, cleanup_old
    global coerce_ws_chain_params, derive_mb_chain_params_from_reference
    global get_platform_target, get_preset, jobs, library
    global master_stream_to_pcm16, process_audio_with_reference
    global read_and_validate, ref_lib, resolve_input_source, run_in_threadpool
    global validate_audio_file, verify_ws_token, analyze_audio
    global apply_mastering_chain, apply_sidechain, build_matching_fir
    global compute_ms_eq_curves, compute_reference_eq_curve
    global compute_reference_eq_curve_ddsp, eq_high_pass, eq_parametric_band
    global apply_matching_fir, apply_ms_matching_fir
    global spectral_energy_at_bands, spectral_energy_at_bands_multires
    global StemParams, MixParams, process_stem, mix_and_master
    global normalize_by_lufs, run_normalize_job
    global _crop_preview, _ensure_stereo, _get_input_duration
    global _match_length

    # Resolve dependencies injected from app.py
    MAX_FILE_SIZE = dependencies["MAX_FILE_SIZE"]
    UPLOAD_DIR = dependencies["UPLOAD_DIR"]
    PROCESSED_DIR = dependencies["PROCESSED_DIR"]
    LIBRARY_DIR = dependencies["LIBRARY_DIR"]
    audio_cache_get = dependencies["audio_cache_get"]
    audio_cache_put = dependencies["audio_cache_put"]
    cleanup_old = dependencies["cleanup_old"]
    coerce_ws_chain_params = dependencies["coerce_ws_chain_params"]
    derive_mb_chain_params_from_reference = dependencies["derive_mb_chain_params_from_reference"]
    get_platform_target = dependencies["get_platform_target"]
    get_preset = dependencies["get_preset"]
    jobs = dependencies["jobs"]
    library = dependencies["library"]
    master_stream_to_pcm16 = dependencies["master_stream_to_pcm16"]
    process_audio_with_reference = dependencies["process_audio_with_reference"]
    read_and_validate = dependencies["read_and_validate"]
    ref_lib = dependencies["ref_lib"]
    resolve_input_source = dependencies["resolve_input_source"]
    run_in_threadpool = dependencies["run_in_threadpool"]
    validate_audio_file = dependencies["validate_audio_file"]
    verify_ws_token = dependencies["verify_ws_token"]
    analyze_audio = dependencies["analyze_audio"]
    apply_mastering_chain = dependencies["apply_mastering_chain"]
    apply_sidechain = dependencies["apply_sidechain"]
    build_matching_fir = dependencies["build_matching_fir"]
    compute_ms_eq_curves = dependencies["compute_ms_eq_curves"]
    compute_reference_eq_curve = dependencies["compute_reference_eq_curve"]
    compute_reference_eq_curve_ddsp = dependencies["compute_reference_eq_curve_ddsp"]
    eq_high_pass = dependencies["eq_high_pass"]
    eq_parametric_band = dependencies["eq_parametric_band"]
    apply_matching_fir = dependencies["apply_matching_fir"]
    apply_ms_matching_fir = dependencies["apply_ms_matching_fir"]
    spectral_energy_at_bands = dependencies["spectral_energy_at_bands"]
    spectral_energy_at_bands_multires = dependencies["spectral_energy_at_bands_multires"]
    StemParams = dependencies["StemParams"]
    MixParams = dependencies["MixParams"]
    process_stem = dependencies["process_stem"]
    mix_and_master = dependencies["mix_and_master"]
    normalize_by_lufs = dependencies["normalize_by_lufs"]
    run_normalize_job = dependencies["run_normalize_job"]
    _crop_preview = dependencies["_crop_preview"]
    _ensure_stereo = dependencies["_ensure_stereo"]
    _get_input_duration = dependencies["_get_input_duration"]
    _match_length = dependencies["_match_length"]

    # Attach for any direct access
    for key, val in dependencies.items():
        setattr(router, key, val)

    return router


@router.websocket("/ws/master-stream")
async def ws_master_stream(websocket: WebSocket, token: str = Query(None)):
    if not verify_ws_token(token):
        await websocket.close(code=4001)
        return
    await websocket.accept()
    tmp_path = None
    try:
        config_msg = await websocket.receive_json()
        chunk_seconds = float(config_msg.get("chunk_seconds", 1.0))
        preset_name = config_msg.get("preset")
        platform = config_msg.get("platform_target")
        preview_seconds_stream = config_msg.get("preview_seconds")
        stream_pcm_format = str(config_msg.get("stream_pcm_format", "int16")).lower()
        if stream_pcm_format not in ("int16", "pcm24", "float32"):
            stream_pcm_format = "int16"
        # session_id identifica el archivo actual en el caché del servidor.
        # El cliente lo genera al cargar un archivo (crypto.randomUUID()) y lo
        # envía en cada preview del mismo archivo para evitar re-subir los bytes.
        session_id = config_msg.get("session_id")
        # library_id: el archivo ya vive en LIBRARY_DIR (subido antes desde el
        # panel de librería). Si viene y no hay cache aún para este session_id,
        # se lee directo del disco del servidor — el cliente no manda bytes.
        library_id = config_msg.get("library_id")

        chain_params = {k: v for k, v in config_msg.items() if k not in (
            "chunk_seconds", "preset", "platform_target", "preview_seconds", "type",
            "session_id", "library_id", "stream_pcm_format",
        )}
        if preset_name:
            chain_params = {**get_preset(preset_name), **chain_params}
            chain_params.pop("label", None)
        if platform:
            platform_target_data = get_platform_target(platform)
            if not platform_target_data or "lufs" not in platform_target_data:
                raise ValueError(f"Plataforma '{platform}' no tiene configuración de LUFS válida")
            chain_params["use_lufs_normalize"] = True
            chain_params["target_lufs"] = platform_target_data["lufs"]
        chain_params = coerce_ws_chain_params(chain_params)

        # ── Audio: intentar reusar del caché antes de pedir el upload ─────────
        audio = sr = None

        if session_id:
            cached = audio_cache_get(session_id)
            if cached is not None:
                audio, sr = cached
                # Avisamos al cliente: puede saltarse el upload de bytes.
                # El cliente responde {"event":"params_only"} y no envía binarios.
                await websocket.send_json({"event": "use_cache"})

        if audio is None and library_id:
            lib_path = library.get_path(LIBRARY_DIR, library_id)
            if lib_path is None:
                await websocket.send_json({
                    "event": "error",
                    "message": "Archivo de la librería no encontrado (¿se borró?).",
                })
                return
            # librosa.load es CPU-bound → threadpool para no bloquear el event loop.
            audio, sr = await run_in_threadpool(librosa.load, lib_path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            preview_window = float(preview_seconds_stream) if preview_seconds_stream else 10.0
            audio = _crop_preview(audio, sr, preview_window)
            if session_id:
                audio_cache_put(session_id, audio, sr)
            # Igual que con el caché: el cliente no necesita mandar bytes.
            await websocket.send_json({"event": "use_cache"})

        if audio is None:
            # No hay caché ni library_id utilizable → hace falta el archivo.
            # BUGFIX: antes el cliente empezaba a mandar los bytes del
            # archivo en cuanto abría el WebSocket, SIN esperar ninguna
            # confirmación del servidor — por eso "use_cache" nunca ahorraba
            # banda de verdad: el cliente igual mandaba todo en paralelo.
            # Este evento explícito es la señal que el cliente ahora espera
            # antes de leer/enviar el archivo (ver index.html, ws.onmessage).
            await websocket.send_json({"event": "need_upload"})
            # Recibir el archivo en trozos (igual que antes).
            audio_chunks = []
            total_size = 0
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    chunk = message["bytes"]
                    total_size += len(chunk)
                    if total_size > MAX_FILE_SIZE:
                        await websocket.send_json({
                            "event": "error",
                            "message": f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE // 1024 // 1024} MB",
                        })
                        return
                    audio_chunks.append(chunk)
                elif message.get("text") is not None:
                    try:
                        ctrl = json.loads(message["text"])
                    except Exception:
                        ctrl = {}
                    # "upload_complete" = flujo viejo; "params_only" = flujo nuevo con caché
                    if ctrl.get("event") in ("upload_complete", "params_only"):
                        break
                elif message.get("type") == "websocket.disconnect":
                    return
                else:
                    break

            audio_bytes = b"".join(audio_chunks)
            if not audio_bytes:
                await websocket.send_json({"event": "error", "message": "No se recibió audio."})
                return

            tmp_path = os.path.join(UPLOAD_DIR, f"stream_{uuid.uuid4().hex}")
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)

            # librosa.load es CPU-bound → threadpool para no bloquear el event loop.
            audio, sr = await run_in_threadpool(librosa.load, tmp_path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            # Recortar el preview ANTES de cachear: así todos los previews
            # siguientes (mismo session_id, distintos parámetros) usan exactamente
            # el mismo extracto sin volver a recortar.
            preview_window = float(preview_seconds_stream) if preview_seconds_stream else 10.0
            audio = _crop_preview(audio, sr, preview_window)

            # Guardar en caché para los próximos previews de esta sesión.
            if session_id:
                audio_cache_put(session_id, audio, sr)

        chain_params.pop("output_format", None)
        chain_params.pop("preview_seconds", None)

        # PERF: bypasear stages costosos que no son perceptibles en preview de 10s.
        # Reducen el tiempo de procesamiento por chunk ~40%.
        # El usuario puede activarlos explícitamente si los necesita.
        for _bypass_key in ("nr_bypass", "dyneq_bypass", "reso_bypass", "tonal_balance_bypass"):
            chain_params.setdefault(_bypass_key, True)

        # BUGFIX: apply_mastering_chain (lo que corre por chunk) ignora
        # use_lufs_normalize/target_peak/target_lufs — esos campos no hacen
        # nada dentro de la cadena en sí. Antes esto significaba que
        # "Normalizar por LUFS" no tenía ningún efecto en el preview en vivo,
        # aunque sí funcionara en el archivo final (/master, /master/sync,
        # /preview pasan por process_audio, que sí corre el safety check).
        # Acá se corre el mismo safety check UNA vez, en batch, sobre el
        # audio ya recortado al preview — no por chunk, porque sería carísimo
        # y generaría saltos de gain audibles en tiempo real — y el
        # input_gain_db corregido resultante es el que se usa para generar
        # todos los chunks del stream.
        # PERF: compute_lufs_corrected_gain analiza el audio completo — antes
        # bloqueaba el inicio del stream. Ahora el stream arranca inmediato y
        # el gain LUFS se aplica a partir del segundo chunk si ya está listo.
        _lufs_gain_ready = False
        _lufs_gain_db = 0.0
        if chain_params.get("use_lufs_normalize"):
            target_lufs_val = float(chain_params.get("target_lufs", -14.0))
            import asyncio
            _lufs_fut = asyncio.ensure_future(run_in_threadpool(
                compute_lufs_corrected_gain, audio, sr, dict(chain_params), target_lufs_val
            ))
        else:
            _lufs_fut = None

        chunk_gen = master_stream_to_pcm16(audio, sr, chunk_seconds=chunk_seconds,
                                          pcm_format=stream_pcm_format, **chain_params)
        _SENTINEL = object()

        def _next_ws_chunk():
            try:
                return next(chunk_gen)
            except StopIteration:
                return _SENTINEL

        while True:
            # Aplicar gain LUFS en cuanto esté listo (sin bloquear el stream)
            # BUGFIX: antes se recreaba chunk_gen, lo que causaba repetición/pérdida de chunks.
            # Ahora solo almacenamos el gain y lo aplicamos al PCM ya generado.
            if _lufs_fut is not None and _lufs_fut.done() and not _lufs_gain_ready:
                try:
                    corrected_gain, lufs_notes = _lufs_fut.result()
                    _lufs_gain_db = corrected_gain
                    _lufs_gain_ready = True
                    await websocket.send_json({
                        "event": "lufs_safety",
                        "target_lufs": round(target_lufs_val, 2),
                        "corrected_input_gain_db": round(corrected_gain, 2),
                        "notes": lufs_notes,
                    })
                except Exception:
                    _lufs_fut = None
            item = await run_in_threadpool(_next_ws_chunk)
            if item is _SENTINEL:
                break
            pcm_bytes, metrics = item
            
            # Aplicar gain LUFS si está listo (solo después de que se calcule)
            if _lufs_gain_ready and abs(_lufs_gain_db) > 0.01:
                # Convertir PCM int16 → float → aplicar gain → float → PCM int16
                pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                gain_linear = 10.0 ** (_lufs_gain_db / 20.0)
                pcm_data = np.clip(pcm_data * gain_linear, -32768, 32767)
                pcm_bytes = pcm_data.astype(np.int16).tobytes()
            
            await websocket.send_json({"event": "chunk", "metrics": metrics, "sample_rate": sr, "channels": int(audio.shape[0])})
            await websocket.send_bytes(pcm_bytes)

        await websocket.send_json({"event": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"ws_master_stream error: {e}", exc_info=True)
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── WebSocket: preview en tiempo real con referencia ─────────────────────────
# Flujo:
#   1. Cliente envía JSON de config (params de matching + band_gains_db)
#   2. Servidor responde need_upload / use_cache para el archivo PROPIO
#   3. Cliente responde need_upload_ref / use_cache_ref para la REFERENCIA
#   4. Servidor calcula EQ de matching FIR UNA VEZ contra la referencia
#   5. Aplica ganancias manuales por banda (band_gains_db)
#   6. Hace streaming del audio procesado chunk a chunk (PCM16 + métricas)
#
# Preview: siempre 10 segundos a partir del segundo 40 (o desde el inicio
# si la pista es más corta). El mismo fragmento se reutiliza mientras no
# cambie el session_id, asegurando comparación consistente entre distintas
# configuraciones de sliders.

PREVIEW_START_SEC = 40.0
PREVIEW_DURATION_SEC = 10.0

def _crop_ref_preview(audio: np.ndarray, sr: int,
                      start_sec: float = PREVIEW_START_SEC,
                      duration_sec: float = PREVIEW_DURATION_SEC) -> np.ndarray:
    """Recorta `duration_sec` segundos a partir de `start_sec`.
    Si la pista es más corta que start_sec, arranca desde 0.
    Siempre devuelve exactamente duration_sec segundos (o el total si es más corto)."""
    total = audio.shape[-1]
    start_sample = int(min(start_sec, max(0.0, total / sr - duration_sec)) * sr)
    end_sample = min(start_sample + int(duration_sec * sr), total)
    return audio[:, start_sample:end_sample]

@router.websocket("/ws/ref-stream")
async def ws_ref_stream(websocket: WebSocket, token: str = Query(None)):
    """Preview en tiempo real del match por referencia con sliders de banda."""
    if not verify_ws_token(token):
        await websocket.close(code=4001)
        return
    await websocket.accept()
    tmp_src = tmp_ref = None
    try:
        # ── 1. Configuración ───────────────────────────────────────────────────
        cfg = await websocket.receive_json()

        session_id     = cfg.get("session_id")      # caché del archivo propio
        ref_session_id = cfg.get("ref_session_id")  # caché de la referencia
        library_id     = cfg.get("library_id")
        ref_library_id = cfg.get("ref_library_id")
        chunk_seconds  = float(cfg.get("chunk_seconds", 2.0))

        # Parámetros de matching
        eq_bands        = int(cfg.get("eq_bands", 28))
        eq_max_boost    = float(cfg.get("eq_max_boost_db", 6.0))
        eq_max_cut      = float(cfg.get("eq_max_cut_db", -9.0))
        eq_q            = float(cfg.get("eq_q", 1.3))
        eq_blend        = float(cfg.get("eq_match_blend", 0.75))
        eq_fit_method   = str(cfg.get("eq_fit_method", "heuristic"))
        ms_eq_matching  = bool(cfg.get("ms_eq_matching", True))
        hp_cutoff       = float(cfg.get("hp_cutoff", 30.0))
        band_gains_db   = cfg.get("band_gains_array") or cfg.get("band_gains_db") or []

        # ── 2. Cargar archivo PROPIO ───────────────────────────────────────────
        audio = sr = None

        if session_id:
            cached = audio_cache_get(session_id)
            if cached is not None:
                audio, sr = cached
                await websocket.send_json({"event": "use_cache"})

        if audio is None and library_id:
            lib_path = library.get_path(LIBRARY_DIR, library_id)
            if lib_path is None:
                await websocket.send_json({"event": "error", "message": "Archivo propio no encontrado en librería."})
                return
            audio, sr = await run_in_threadpool(librosa.load, lib_path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            audio = _crop_ref_preview(audio, sr)
            if session_id:
                audio_cache_put(session_id, audio, sr)
            await websocket.send_json({"event": "use_cache"})

        if audio is None:
            await websocket.send_json({"event": "need_upload"})
            chunks, total_size = [], 0
            while True:
                msg = await websocket.receive()
                if msg.get("bytes"):
                    total_size += len(msg["bytes"])
                    if total_size > MAX_FILE_SIZE:
                        await websocket.send_json({"event": "error", "message": "Archivo demasiado grande."})
                        return
                    chunks.append(msg["bytes"])
                elif msg.get("text"):
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("event") in ("upload_complete", "params_only"):
                        break
                elif msg.get("type") == "websocket.disconnect":
                    return
            audio_bytes = b"".join(chunks)
            if not audio_bytes:
                await websocket.send_json({"event": "error", "message": "No se recibió audio."})
                return
            tmp_src = os.path.join(UPLOAD_DIR, f"refws_src_{uuid.uuid4().hex}")
            with open(tmp_src, "wb") as f:
                f.write(audio_bytes)
            audio, sr = await run_in_threadpool(librosa.load, tmp_src, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            audio = _crop_ref_preview(audio, sr)
            if session_id:
                audio_cache_put(session_id, audio, sr)

        # ── 3. Cargar REFERENCIA ───────────────────────────────────────────────
        ref_audio = ref_sr = None

        if ref_session_id:
            cached_ref = audio_cache_get(ref_session_id)
            if cached_ref is not None:
                ref_audio, ref_sr = cached_ref
                await websocket.send_json({"event": "use_cache_ref"})

        if ref_audio is None and ref_library_id:
            lib_ref_path = library.get_path(LIBRARY_DIR, ref_library_id)
            if lib_ref_path is None:
                await websocket.send_json({"event": "error", "message": "Referencia no encontrada en librería."})
                return
            ref_audio, ref_sr = await run_in_threadpool(librosa.load, lib_ref_path, sr=None, mono=False)
            if ref_audio.ndim == 1:
                ref_audio = ref_audio[np.newaxis, :]
            if ref_session_id:
                audio_cache_put(ref_session_id, ref_audio, ref_sr)
            await websocket.send_json({"event": "use_cache_ref"})

        if ref_audio is None:
            await websocket.send_json({"event": "need_upload_ref"})
            ref_chunks, ref_total = [], 0
            while True:
                msg = await websocket.receive()
                if msg.get("bytes"):
                    ref_total += len(msg["bytes"])
                    if ref_total > MAX_FILE_SIZE:
                        await websocket.send_json({"event": "error", "message": "Referencia demasiado grande."})
                        return
                    ref_chunks.append(msg["bytes"])
                elif msg.get("text"):
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("event") in ("upload_complete", "params_only"):
                        break
                elif msg.get("type") == "websocket.disconnect":
                    return
            ref_bytes = b"".join(ref_chunks)
            if not ref_bytes:
                await websocket.send_json({"event": "error", "message": "No se recibió referencia."})
                return
            tmp_ref = os.path.join(UPLOAD_DIR, f"refws_ref_{uuid.uuid4().hex}")
            with open(tmp_ref, "wb") as f:
                f.write(ref_bytes)
            ref_audio, ref_sr = await run_in_threadpool(librosa.load, tmp_ref, sr=None, mono=False)
            if ref_audio.ndim == 1:
                ref_audio = ref_audio[np.newaxis, :]
            if ref_session_id:
                audio_cache_put(ref_session_id, ref_audio, ref_sr)

        # ── 4. Calcular EQ de matching FIR contra la referencia ───────────────
        await websocket.send_json({"event": "analyzing", "message": "Calculando EQ de matching..."})

        def _compute_matching(audio, sr, ref_audio, ref_sr):
            nyquist  = min(sr, ref_sr) / 2.0
            max_freq = float(np.clip(min(20000.0, nyquist - 100.0), 200.0, nyquist - 1.0))
            edges    = np.logspace(np.log10(20.0), np.log10(max_freq), eq_bands + 1)
            band_edges = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
            centers  = [float(np.sqrt(lo * hi)) for lo, hi in band_edges]
            src_bands_db = spectral_energy_at_bands(audio, sr, band_edges)
            ref_bands_db = spectral_energy_at_bands(ref_audio, ref_sr, band_edges)
            if eq_fit_method == "ddsp":
                src_mr = spectral_energy_at_bands_multires(audio, sr, band_edges)
                ref_mr = spectral_energy_at_bands_multires(ref_audio, ref_sr, band_edges)
                curve  = compute_reference_eq_curve_ddsp(src_mr, ref_mr, centers,
                                                          max_boost_db=eq_max_boost,
                                                          max_cut_db=eq_max_cut)
            else:
                curve = compute_reference_eq_curve(src_bands_db, ref_bands_db, centers,
                                                   max_boost_db=eq_max_boost,
                                                   max_cut_db=eq_max_cut,
                                                   blend=eq_blend)
            processed = eq_high_pass(audio, sr, cutoff_hz=hp_cutoff)
            if ms_eq_matching and processed.ndim == 2 and processed.shape[0] == 2:
                src_mr_ms = src_mr if eq_fit_method == "ddsp" else None
                ref_mr_ms = ref_mr if eq_fit_method == "ddsp" else None
                curve_mid, curve_side = compute_ms_eq_curves(
                    processed, sr, ref_audio, ref_sr,
                    band_edges=band_edges, centers=centers,
                    max_boost_db=eq_max_boost, max_cut_db=eq_max_cut,
                    blend=eq_blend, eq_fit_method=eq_fit_method,
                    src_bands_multires=src_mr_ms, ref_bands_multires=ref_mr_ms,
                )
                processed = apply_ms_matching_fir(processed, sr, curve_mid, curve_side, eq_q=eq_q)
                curve = curve_mid  # para el evento matching_ready / reporte (Mid es la más representativa)
            else:
                fir_taps = build_matching_fir(curve, sr, precision=eq_q)
                processed = apply_matching_fir(processed, sr, fir_taps)
            # Ganancias manuales por banda — array dinámico [{freq_hz, gain_db}]
            n = len(band_gains_db)
            auto_q = float(max(0.7, min(2.0, 0.5 + (n / 28.0) * 1.5))) if n else 1.0
            for entry in band_gains_db:
                freq_hz = float(entry.get("freq_hz", 0))
                gain    = float(entry.get("gain_db", 0.0))
                if freq_hz >= 10 and abs(gain) >= 0.1:
                    processed = eq_parametric_band(processed, sr, freq=freq_hz, gain_db=gain, q=auto_q)
            return processed, curve

        audio_matched, eq_curve = await run_in_threadpool(_compute_matching, audio, sr, ref_audio, ref_sr)

        await websocket.send_json({
            "event": "matching_ready",
            "eq_curve": [{"freq_hz": round(f, 1), "gain_db": round(g, 2)} for f, g in eq_curve],
        })

        # ── 4b. Dinámica multibanda calibrada contra la referencia ────────────
        # BUGFIX (preview de match mastering sin GR real): antes acá no se
        # pasaba ningún chain_param a master_stream_to_pcm16, así que el
        # multibanda de apply_mastering_chain corría con thresholds genéricos
        # fijos (-18dB, ratio 2.0) sin ninguna relación con la referencia — el
        # GR que se veía (si se veía) no tenía nada que ver con el ajuste real
        # que después aplica process_audio_with_reference en el render final.
        # Ahora se calcula una sola vez, con la misma fórmula banda-por-banda
        # que match_dynamics_bands, y se fuerza mb_bypass=False para que el
        # preview muestre el mismo GR calibrado que va a tener el archivo final.
        mb_chain_params = await run_in_threadpool(
            derive_mb_chain_params_from_reference, audio_matched, sr, ref_audio, ref_sr
        )

        # ── 5. Streaming chunk a chunk ─────────────────────────────────────────
        # Usamos master_stream_to_pcm16 que ya convierte a float32 bytes interleaved
        chunk_gen = master_stream_to_pcm16(audio_matched, sr, chunk_seconds=chunk_seconds,
                                           detect_dynamic_eq=False, **mb_chain_params)
        _SENTINEL = object()

        def _next_ref_chunk():
            try:
                return next(chunk_gen)
            except StopIteration:
                return _SENTINEL

        while True:
            item = await run_in_threadpool(_next_ref_chunk)
            if item is _SENTINEL:
                break
            pcm_bytes, metrics = item
            await websocket.send_json({
                "event": "chunk",
                "metrics": metrics,
                "sample_rate": sr,
                "channels": int(audio_matched.shape[0]),
            })
            await websocket.send_bytes(pcm_bytes)

        await websocket.send_json({"event": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"ws_ref_stream error: {e}", exc_info=True)
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        for p in (tmp_src, tmp_ref):
            if p and os.path.exists(p):
                os.remove(p)

# ─── Endpoint con preset (parámetros multibanda ahora opcionales) ──────────────────

@router.post("/mix/submit", tags=["Mixer"])
async def mix_submit(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    current_user: dict = Depends(get_current_user),
    stem_names: str = Form(..., description="JSON list de nombres de stems subidos"),
    stem_params: str = Form("{}", description="JSON: {nombre: StemParams}"),
    mix_params: str = Form("{}", description="JSON: MixParams"),
    stem_library_ids: str = Form("{}", description="JSON opcional: {nombre: library_id} para stems reutilizables"),
    sr: int = Form(44100),
):
    """Inicia el job de mezcla con los stems ya subidos via /mix/upload-stem."""
    import json as _json

    try:
        names = _json.loads(stem_names)
        s_params_dict = _json.loads(stem_params)
        m_params_dict = _json.loads(mix_params)
        library_ids = _json.loads(stem_library_ids or "{}")
    except Exception as e:
        raise HTTPException(400, f"JSON inválido: {e}")

    # Reconstruir paths desde librería persistente o desde uploads temporales de sesión.
    stem_paths = {}
    cleanup_paths = set()
    for name in names:
        library_path = _mix_library_stem_path((library_ids or {}).get(name))
        session_path = _mix_session_stem_path(session_id, name)
        path = library_path or session_path
        if not path:
            raise HTTPException(404, f"Stem '{name}' no encontrado para session_id '{session_id}' ni en librería.")
        stem_paths[name] = path
        if path == session_path:
            cleanup_paths.add(path)

    if not stem_paths:
        raise HTTPException(400, "No se encontraron stems para esta sesión.")

    job_id = uuid.uuid4().hex
    jobs.create_job(job_id, {
        "status": "queued",
        "type": "mix",
        "session_id": session_id,
        "stem_names": list(stem_paths.keys()),
        "created_at": time.time(),
        "progress": 0,
        "stage": "En cola",
    })

    background_tasks.add_task(
        run_mix_job, job_id, stem_paths, sr, s_params_dict, m_params_dict, cleanup_paths
    )

    return {"job_id": job_id, "status": "queued",
            "stem_names": list(stem_paths.keys()),
            "poll_url": f"/job/{job_id}"}

@router.websocket("/ws/mix-stream")
async def ws_mix_stream(websocket: WebSocket, token: str = Query(None)):
    # Auth via query param (WS no soporta headers custom). Reusa
    # verify_ws_token() — misma validación (JWT + usuario approved) que
    # ahora también usa /ws/dashboard.
    if not verify_ws_token(token):
        await websocket.close(code=4001)
        return
    """Preview en vivo del mixdown multistem, streameado como PCM16.

    Reusa el mismo protocolo de eventos que /ws/master-stream (chunk/done/error)
    para que el frontend pueda compartir la lógica de reproducción. A diferencia
    del preview de mastering, acá los stems YA están en disco (subidos antes
    via /mix/upload-stem) — el cliente solo manda session_id + params, nunca
    bytes de audio.

    Flujo: cargar (o reusar del caché) cada stem recortado a `preview_seconds`
    → process_stem individual + sidechain + suma + gain/normalize del mix bus
    → esa mezcla se pasa por master_stream_to_pcm16 igual que el preview normal
    (con el mismo bypass de stages costosos), chunkeada y streameada.
    """
    await websocket.accept()
    try:
        config_msg = await websocket.receive_json()
        session_id = config_msg.get("session_id")
        stem_names = config_msg.get("stem_names") or []
        stem_library_ids = config_msg.get("stem_library_ids") or {}
        stem_params_dict = config_msg.get("stem_params") or {}
        mix_params_dict = config_msg.get("mix_params") or {}
        chunk_seconds = float(config_msg.get("chunk_seconds", 1.0))
        preview_seconds = float(config_msg.get("preview_seconds", 12.0))
        sr = int(config_msg.get("sr", 44100))

        if not session_id or not stem_names:
            await websocket.send_json({"event": "error", "message": "Falta session_id o stem_names."})
            return

        # ── Cargar (o reusar del caché) cada stem, ya recortado al preview ────
        stems: dict = {}
        for name in stem_names:
            library_id = stem_library_ids.get(name)
            cache_key = f"mixlib_{library_id}" if library_id else f"mix_{session_id}_{name}"
            cached = audio_cache_get(cache_key)
            if cached is not None:
                audio, file_sr = cached
            else:
                path = _resolve_mix_stem_path(session_id, name, stem_library_ids)
                if not path:
                    await websocket.send_json({"event": "error", "message": f"Stem '{name}' no encontrado (¿se subió o existe en librería?)."})
                    return
                # librosa.load es CPU-bound → threadpool para no bloquear el event loop.
                audio, file_sr = await run_in_threadpool(librosa.load, path, sr=None, mono=False)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]
                audio = _crop_preview(audio, file_sr, preview_seconds)
                audio_cache_put(cache_key, audio, file_sr)
            if file_sr != sr:
                audio = await run_in_threadpool(librosa.resample, audio, orig_sr=file_sr, target_sr=sr)
            stems[name] = audio.astype(np.float32)

        # ── Reconstruir StemParams / MixParams desde el JSON del cliente ──────
        s_params = {}
        for name in stem_names:
            sp = StemParams(name=name)
            for k, v in (stem_params_dict.get(name) or {}).items():
                if hasattr(sp, k):
                    setattr(sp, k, v)
            s_params[name] = sp

        mp = MixParams()
        for k, v in mix_params_dict.items():
            if hasattr(mp, k):
                setattr(mp, k, v)

        # ── Procesar stems + sidechain + suma (batch, sobre el crop corto) ────
        def _build_mix():
            processed = {}
            solo_active = any(p.solo for p in s_params.values())
            for name, audio in stems.items():
                p = s_params[name]
                if solo_active and not p.solo:
                    processed[name] = np.zeros_like(_ensure_stereo(audio))
                    continue
                proc, _m = process_stem(_ensure_stereo(audio), sr, p)
                processed[name] = proc
            for name, p in s_params.items():
                if p.sidechain_trigger_name and p.sidechain_trigger_name in processed:
                    ducked, _sc = apply_sidechain(
                        processed[name], processed[p.sidechain_trigger_name], sr,
                        threshold=p.sidechain_threshold, ratio=p.sidechain_ratio,
                        attack_ms=p.sidechain_attack_ms, release_ms=p.sidechain_release_ms,
                    )
                    processed[name] = ducked
            arrays = _match_length(list(processed.values()))
            mix = np.sum(arrays, axis=0).astype(np.float32)
            if abs(mp.master_gain_db) > 0.01:
                mix = mix * 10.0 ** (mp.master_gain_db / 20.0)
            if mp.normalize_before_master:
                peak = np.max(np.abs(mix))
                if peak > 0.9:
                    mix = mix * (0.9 / peak)
            return mix

        mix = await run_in_threadpool(_build_mix)

        # ── Streaming del mix bus por la cadena de mastering ──────────────────
        # Mismo protocolo de eventos y mismo bypass de stages costosos que
        # /ws/master-stream — ver ese endpoint para el detalle del porqué.
        chain_params = coerce_ws_chain_params(dict(mp.chain_params))
        for _bypass_key in ("nr_bypass", "dyneq_bypass", "reso_bypass", "tonal_balance_bypass"):
            chain_params.setdefault(_bypass_key, True)

        chunk_gen = master_stream_to_pcm16(mix, sr, chunk_seconds=chunk_seconds,
                                          pcm_format="int16", **chain_params)
        _SENTINEL = object()

        def _next_chunk():
            try:
                return next(chunk_gen)
            except StopIteration:
                return _SENTINEL

        while True:
            item = await run_in_threadpool(_next_chunk)
            if item is _SENTINEL:
                break
            pcm_bytes, metrics = item
            await websocket.send_json({"event": "chunk", "metrics": metrics, "sample_rate": sr, "channels": int(mix.shape[0])})
            await websocket.send_bytes(pcm_bytes)

        await websocket.send_json({"event": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"ws_mix_stream error: {e}", exc_info=True)
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass

