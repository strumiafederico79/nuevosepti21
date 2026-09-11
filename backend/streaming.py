import os
import json
import uuid
import time
import asyncio
import numpy as np
import librosa
from fastapi import APIRouter, WebSocket, Query, WebSocketDisconnect, BackgroundTasks, Form, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

# ==============================================================================
# IMPORTANTE: Descomenta e importa las dependencias internas de tu proyecto aquí
# ==============================================================================
# from dependencias.auth import verify_ws_token, get_current_user
# from dependencias.config import LIBRARY_DIR, UPLOAD_DIR, MAX_FILE_SIZE
# from dependencias.cache import audio_cache_get, audio_cache_put
# from dependencias.utils import logger, library
# from dependencias.procesamiento import (
#     get_preset, get_platform_target, coerce_ws_chain_params, _crop_preview,
#     compute_lufs_corrected_gain, master_stream_to_pcm16, spectral_energy_at_bands,
#     spectral_energy_at_bands_multires, compute_reference_eq_curve_ddsp,
#     compute_reference_eq_curve, eq_high_pass, compute_ms_eq_curves,
#     apply_ms_matching_fir, build_matching_fir, apply_matching_fir,
#     eq_parametric_band, derive_mb_chain_params_from_reference,
#     _mix_library_stem_path, _mix_session_stem_path, _resolve_mix_stem_path,
#     StemParams, MixParams, process_stem, apply_sidechain, _match_length, _ensure_stereo
# )
# from dependencias.jobs import jobs, run_mix_job


router = APIRouter()

@router.websocket("/ws/master-stream")
async def ws_master_stream(websocket: WebSocket, token: str = Query(None)):
    if not verify_ws_token(token):
        await websocket.close(code=4001)
        return
    await websocket.accept()
    tmp_path = None
    _lufs_fut = None  # Inicializamos aquí para poder cancelarlo en el finally
    try:
        config_msg = await websocket.receive_json()
        chunk_seconds = float(config_msg.get("chunk_seconds", 1.0))
        preset_name = config_msg.get("preset")
        platform = config_msg.get("platform_target")
        preview_seconds_stream = config_msg.get("preview_seconds")
        stream_pcm_format = str(config_msg.get("stream_pcm_format", "int16")).lower()
        if stream_pcm_format not in ("int16", "pcm24", "float32"):
            stream_pcm_format = "int16"
        
        session_id = config_msg.get("session_id")
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
                await websocket.send_json({"event": "use_cache"})

        if audio is None and library_id:
            lib_path = library.get_path(LIBRARY_DIR, library_id)
            if lib_path is None:
                await websocket.send_json({
                    "event": "error",
                    "message": "Archivo de la librería no encontrado (¿se borró?).",
                })
                return
            audio, sr = await run_in_threadpool(librosa.load, lib_path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            preview_window = float(preview_seconds_stream) if preview_seconds_stream else 10.0
            audio = _crop_preview(audio, sr, preview_window)
            if session_id:
                audio_cache_put(session_id, audio, sr)
            await websocket.send_json({"event": "use_cache"})

        if audio is None:
            await websocket.send_json({"event": "need_upload"})
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

            audio, sr = await run_in_threadpool(librosa.load, tmp_path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            preview_window = float(preview_seconds_stream) if preview_seconds_stream else 10.0
            audio = _crop_preview(audio, sr, preview_window)

            if session_id:
                audio_cache_put(session_id, audio, sr)

        chain_params.pop("output_format", None)
        chain_params.pop("preview_seconds", None)

        for _bypass_key in ("nr_bypass", "dyneq_bypass", "reso_bypass", "tonal_balance_bypass"):
            chain_params.setdefault(_bypass_key, True)

        _lufs_gain_ready = False
        _lufs_gain_db = 0.0
        if chain_params.get("use_lufs_normalize"):
            target_lufs_val = float(chain_params.get("target_lufs", -14.0))
            _lufs_fut = asyncio.ensure_future(run_in_threadpool(
                compute_lufs_corrected_gain, audio, sr, dict(chain_params), target_lufs_val
            ))

        chunk_gen = master_stream_to_pcm16(audio, sr, chunk_seconds=chunk_seconds,
                                          pcm_format=stream_pcm_format, **chain_params)
        _SENTINEL = object()

        def _next_ws_chunk():
            try:
                return next(chunk_gen)
            except StopIteration:
                return _SENTINEL

        while True:
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
            
            if _lufs_gain_ready and abs(_lufs_gain_db) > 0.01:
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
        # CORRECCIÓN: Cancelar la tarea en background si el socket se cierra antes de terminar
        if _lufs_fut is not None and not _lufs_fut.done():
            _lufs_fut.cancel()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── WebSocket: preview en tiempo real con referencia ─────────────────────────

PREVIEW_START_SEC = 40.0
PREVIEW_DURATION_SEC = 10.0

def _crop_ref_preview(audio: np.ndarray, sr: int,
                      start_sec: float = PREVIEW_START_SEC,
                      duration_sec: float = PREVIEW_DURATION_SEC) -> np.ndarray:
    total = audio.shape[-1]
    start_sample = int(min(start_sec, max(0.0, total / sr - duration_sec)) * sr)
    end_sample = min(start_sample + int(duration_sec * sr), total)
    return audio[:, start_sample:end_sample]


@router.websocket("/ws/ref-stream")
async def ws_ref_stream(websocket: WebSocket, token: str = Query(None)):
    if not verify_ws_token(token):
        await websocket.close(code=4001)
        return
    await websocket.accept()
    tmp_src = tmp_ref = None
    try:
        cfg = await websocket.receive_json()

        session_id     = cfg.get("session_id")
        ref_session_id = cfg.get("ref_session_id")
        library_id     = cfg.get("library_id")
        ref_library_id = cfg.get("ref_library_id")
        chunk_seconds  = float(cfg.get("chunk_seconds", 2.0))

        eq_bands        = int(cfg.get("eq_bands", 28))
        eq_max_boost    = float(cfg.get("eq_max_boost_db", 6.0))
        eq_max_cut      = float(cfg.get("eq_max_cut_db", -9.0))
        eq_q            = float(cfg.get("eq_q", 1.3))
        eq_blend        = float(cfg.get("eq_match_blend", 0.75))
        eq_fit_method   = str(cfg.get("eq_fit_method", "heuristic"))
        ms_eq_matching  = bool(cfg.get("ms_eq_matching", True))
        hp_cutoff       = float(cfg.get("hp_cutoff", 30.0))
        band_gains_db   = cfg.get("band_gains_array") or cfg.get("band_gains_db") or []

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
                curve = curve_mid
            else:
                fir_taps = build_matching_fir(curve, sr, precision=eq_q)
                processed = apply_matching_fir(processed, sr, fir_taps)
                
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

        mb_chain_params = await run_in_threadpool(
            derive_mb_chain_params_from_reference, audio_matched, sr, ref_audio, ref_sr
        )

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
    try:
        names = json.loads(stem_names)
        s_params_dict = json.loads(stem_params)
        m_params_dict = json.loads(mix_params)
        library_ids = json.loads(stem_library_ids or "{}")
    except Exception as e:
        raise HTTPException(400, f"JSON inválido: {e}")

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
    if not verify_ws_token(token):
        await websocket.close(code=4001)
        return
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
                audio, file_sr = await run_in_threadpool(librosa.load, path, sr=None, mono=False)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]
                audio = _crop_preview(audio, file_sr, preview_seconds)
                audio_cache_put(cache_key, audio, file_sr)
            if file_sr != sr:
                audio = await run_in_threadpool(librosa.resample, audio, orig_sr=file_sr, target_sr=sr)
            stems[name] = audio.astype(np.float32)

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

        def _build_mix():
            processed = {}
            solo_active = any(p.solo for p in s_params.values())
            for name, audio_data in stems.items():
                p = s_params[name]
                if solo_active and not p.solo:
                    processed[name] = np.zeros_like(_ensure_stereo(audio_data))
                    continue
                proc, _m = process_stem(_ensure_stereo(audio_data), sr, p)
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
            mix_audio = np.sum(arrays, axis=0).astype(np.float32)
            if abs(mp.master_gain_db) > 0.01:
                mix_audio = mix_audio * 10.0 ** (mp.master_gain_db / 20.0)
            if mp.normalize_before_master:
                peak = np.max(np.abs(mix_audio))
                if peak > 0.9:
                    mix_audio = mix_audio * (0.9 / peak)
            return mix_audio

        mix = await run_in_threadpool(_build_mix)

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