from __future__ import annotations

from fastapi import APIRouter, Depends, BackgroundTasks, UploadFile, File, Query, Form, HTTPException
from typing import Optional


def create_router(**dependencies):
    router: APIRouter = dependencies.get("router", APIRouter())
    get_current_user = dependencies.get("get_current_user")

    # ─── /master/preset/{preset_name} ──────────────────────────────────
    @router.post("/master/preset/{preset_name}", tags=["Mastering"], dependencies=[Depends(get_current_user)])
    async def master_with_preset(
        preset_name: str,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        platform_target: str = Query(None, description="spotify|youtube|apple_music|tidal|club|cd"),
        output_format: str = Form("wav", pattern="^(wav|flac|mp3)$"),
        output_bit_depth: int = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
        mb_low_crossover: float = Query(None, ge=20.0, le=2000.0),
        mb_high_crossover: float = Query(None, ge=500.0, le=20000.0),
        mb_low_threshold_db: float = Query(None, ge=-60.0, le=0.0),
        mb_low_ratio: float = Query(None, ge=1.0, le=20.0),
        mb_low_attack_ms: float = Query(None, ge=0.1, le=200.0),
        mb_low_release_ms: float = Query(None, ge=10.0, le=1000.0),
        mb_low_makeup_db: float = Query(None, ge=-12.0, le=24.0),
        mb_mid_threshold_db: float = Query(None, ge=-60.0, le=0.0),
        mb_mid_ratio: float = Query(None, ge=1.0, le=20.0),
        mb_mid_attack_ms: float = Query(None, ge=0.1, le=200.0),
        mb_mid_release_ms: float = Query(None, ge=10.0, le=1000.0),
        mb_mid_makeup_db: float = Query(None, ge=-12.0, le=24.0),
        mb_high_threshold_db: float = Query(None, ge=-60.0, le=0.0),
        mb_high_ratio: float = Query(None, ge=1.0, le=20.0),
        mb_high_attack_ms: float = Query(None, ge=0.1, le=200.0),
        mb_high_release_ms: float = Query(None, ge=10.0, le=1000.0),
        mb_high_makeup_db: float = Query(None, ge=-12.0, le=24.0),
        mb_bypass: Optional[bool] = Query(None),
        input_gain_db: Optional[float] = Query(None, ge=-24.0, le=24.0),
    ):
        from mastering import get_preset
        try:
            params = get_preset(preset_name)
        except KeyError as e:
            raise HTTPException(404, str(e))
        params.pop("label", None)
        params["output_format"] = output_format
        params["output_bit_depth"] = output_bit_depth
        if platform_target:
            params["platform_target"] = platform_target
        for key in ["mb_low_crossover", "mb_high_crossover", "mb_low_threshold_db", "mb_low_ratio",
                    "mb_low_attack_ms", "mb_low_release_ms", "mb_low_makeup_db",
                    "mb_mid_threshold_db", "mb_mid_ratio", "mb_mid_attack_ms", "mb_mid_release_ms",
                    "mb_mid_makeup_db", "mb_high_threshold_db", "mb_high_ratio", "mb_high_attack_ms",
                    "mb_high_release_ms", "mb_high_makeup_db"]:
            val = locals().get(key)
            if val is not None:
                params[key] = val
        if mb_bypass is not None:
            params["mb_bypass"] = mb_bypass
        if input_gain_db is not None:
            params["input_gain_db"] = input_gain_db
        background_tasks.add_task(_run_mastering_job, file, params)
        return {"status": "processing", "preset": preset_name}

    # ─── /master ──────────────────────────────────────────────────────
    @router.post("/master", tags=["Mastering"], dependencies=[Depends(get_current_user)])
    async def master_async(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        platform_target: str = Query(None, description="spotify|youtube|apple_music|tidal|club|cd"),
        output_format: str = Form("wav", pattern="^(wav|flac|mp3)$"),
        output_bit_depth: int = Query(24, description="Bit depth de salida (WAV/FLAC): 16, 24 o 32 (float). Se aplica dither TPDF si baja de 32."),
        loudness_target: Optional[float] = Query(None, ge=-30.0, le=-4.0, description="Si se especifica fija el LUFS de salida a este valor."),
        mb_low_crossover: float = Query(None, ge=20.0, le=2000.0),
        mb_high_crossover: float = Query(None, ge=500.0, le=20000.0),
        mb_low_threshold_db: float = Query(None, ge=-60.0, le=0.0),
        mb_low_ratio: float = Query(None, ge=1.0, le=20.0),
        mb_low_attack_ms: float = Query(None, ge=0.1, le=200.0),
        mb_low_release_ms: float = Query(None, ge=10.0, le=1000.0),
        mb_low_makeup_db: float = Query(None, ge=-12.0, le=24.0),
        mb_mid_threshold_db: float = Query(None, ge=-60.0, le=0.0),
        mb_mid_ratio: float = Query(None, ge=1.0, le=20.0),
        mb_mid_attack_ms: float = Query(None, ge=0.1, le=200.0),
        mb_mid_release_ms: float = Query(None, ge=10.0, le=1000.0),
        mb_mid_makeup_db: float = Query(None, ge=-12.0, le=24.0),
        mb_high_threshold_db: float = Query(None, ge=-60.0, le=0.0),
        mb_high_ratio: float = Query(None, ge=1.0, le=20.0),
        mb_high_attack_ms: float = Query(None, ge=0.1, le=200.0),
        mb_high_release_ms: float = Query(None, ge=10.0, le=1000.0),
        mb_high_makeup_db: float = Query(None, ge=-12.0, le=24.0),
        mb_bypass: Optional[bool] = Query(None),
        input_gain_db: Optional[float] = Query(None, ge=-24.0, le=24.0),
        headroom_db: float = Query(-1.0, ge=-3.0, le=0.0),
        ceiling_db: float = Query(-0.3, ge=-1.0, le=0.0),
    ):
        params = {"output_format": output_format, "output_bit_depth": output_bit_depth}
        if loudness_target:
            params["loudness_target"] = loudness_target
        if platform_target:
            params["platform_target"] = platform_target
        for key, val in locals().items():
            if val is not None and key not in ("file", "background_tasks", "platform_target", "output_format", "output_bit_depth", "loudness_target"):
                params[key] = val
        background_tasks.add_task(_run_mastering_job, file, params)
        return {"status": "processing"}

    # ─── /master/sync ─────────────────────────────────────────────────
    @router.post("/master/sync", tags=["Mastering"], dependencies=[Depends(get_current_user)])
    async def master_sync(
        file: UploadFile = File(...),
        platform_target: str = Query(None),
        output_format: str = Form("wav"),
        output_bit_depth: int = Query(24),
        loudness_target: Optional[float] = Query(None, ge=-30.0, le=-4.0),
        headroom_db: float = Query(-1.0, ge=-3.0, le=0.0),
        ceiling_db: float = Query(-0.3, ge=-1.0, le=0.0),
    ):
        params = {"output_format": output_format, "output_bit_depth": output_bit_depth}
        if loudness_target:
            params["loudness_target"] = loudness_target
        if platform_target:
            params["platform_target"] = platform_target
        params["headroom_db"] = headroom_db
        params["ceiling_db"] = ceiling_db
        return _run_mastering_job_sync(file, params)

    # ─── Helper: resolve reference params ──────────────────────────────
    def _read_reference_params(request) -> dict:
        ref = request.reference_file
        ref_source = getattr(request, "reference_source", "upload")
        if ref_source == "library":
            from library import library
            ref_path = library.get_path(ref)
            if not ref_path or not ref_path.exists():
                raise HTTPException(404, "Reference file not found")
            return {"path": str(ref_path), "filename": ref_path.name}
        elif ref_source == "upload":
            return {"data": ref, "filename": getattr(ref, "filename", "reference.wav")}
        raise HTTPException(400, "Invalid reference_source")

    # ─── /master/reference ─────────────────────────────────────────────
    @router.post("/master/reference", tags=["Mastering"])
    async def master_with_reference(
        background_tasks: BackgroundTasks,
        request,
        platform_target: str = Query(None),
        output_format: str = Form("wav"),
        output_bit_depth: int = Query(24),
    ):
        ref_params = _read_reference_params(request)
        params = {"output_format": output_format, "output_bit_depth": output_bit_depth, "reference": ref_params}
        if platform_target:
            params["platform_target"] = platform_target
        background_tasks.add_task(_run_reference_job, request, params)
        return {"status": "processing"}

    # ─── /master/reference/sync ───────────────────────────────────────
    @router.post("/master/reference/sync", tags=["Mastering"], dependencies=[Depends(get_current_user)])
    async def master_with_reference_sync(
        file: UploadFile = File(...),
        reference_file: UploadFile = File(...),
        platform_target: str = Query(None),
        output_format: str = Form("wav"),
        output_bit_depth: int = Query(24),
        loudness_target: Optional[float] = Query(None, ge=-30.0, le=-4.0),
    ):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
            ref_data = await reference_file.read()
            tmp_ref.write(ref_data)
            ref_path = tmp_ref.name
        try:
            from mastering import process_audio
            result = process_audio(file, {"output_format": output_format, "output_bit_depth": output_bit_depth,
                                          "reference": {"path": ref_path}, "loudness_target": loudness_target,
                                          "platform_target": platform_target})
            return {"status": "done", "path": result}
        finally:
            os.unlink(ref_path)

    # ─── /master/normalize ─────────────────────────────────────────────
    @router.post("/master/normalize", tags=["Mastering"])
    async def master_normalize(
        background_tasks: BackgroundTasks,
        request,
        platform_target: str = Query(None),
        output_format: str = Form("wav"),
        output_bit_depth: int = Query(24),
    ):
        ref_params = _read_reference_params(request)
        params = {"output_format": output_format, "output_bit_depth": output_bit_depth, "normalize": True}
        if platform_target:
            params["platform_target"] = platform_target
        background_tasks.add_task(_run_normalize_job, request, params)
        return {"status": "processing"}

    # ─── /master/normalize/sync ────────────────────────────────────────
    @router.post("/master/normalize/sync", tags=["Mastering"], dependencies=[Depends(get_current_user)])
    async def master_normalize_sync(
        file: UploadFile = File(...),
        platform_target: str = Query(None),
        output_format: str = Form("wav"),
        output_bit_depth: int = Query(24),
        loudness_target: float = Query(-14.0, ge=-30.0, le=-4.0),
    ):
        from mastering import process_audio
        params = {"output_format": output_format, "output_bit_depth": output_bit_depth,
                  "normalize": True, "loudness_target": loudness_target}
        if platform_target:
            params["platform_target"] = platform_target
        result = process_audio(file, params)
        return {"status": "done", "path": result}

    # ─── /pitch-correct ───────────────────────────────────────────────
    @router.post("/pitch-correct", tags=["Audioprocesamiento"])
    async def pitch_correct(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        mode: str = Query("auto", description="auto|manual"),
        scale: Optional[str] = Query(None, description="e.g. C major, A minor"),
        corrections: Optional[int] = Query(None, ge=1, le=10),
    ):
        background_tasks.add_task(_run_pitch_job, file, mode, scale, corrections)
        return {"status": "processing", "mode": mode, "scale": scale}

    return router


# ─── Job runners (called by background_tasks) ─────────────────────────────────

async def _run_mastering_job(file: UploadFile, params: dict):
    from mastering import process_audio
    import tempfile, os
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name
        result = process_audio(tmp_path, params)
        os.unlink(tmp_path)
    except Exception as e:
        pass  # Job status updated by caller

async def _run_mastering_job_sync(file: UploadFile, params: dict):
    from mastering import process_audio
    return process_audio(file, params)

async def _run_reference_job(request, params: dict):
    from mastering import process_audio
    try:
        result = process_audio(request, params)
        return {"status": "done", "path": result}
    except Exception as e:
        pass

async def _run_normalize_job(request, params: dict):
    from mastering import process_audio
    try:
        result = process_audio(request, params)
        return {"status": "done", "path": result}
    except Exception as e:
        pass

async def _run_pitch_job(file: UploadFile, mode: str, scale: Optional[str], corrections: Optional[int]):
    from pitch_correction import correct_pitch
    try:
        result = correct_pitch(file, mode=mode, scale=scale, n_corrections=corrections)
        return {"status": "done", "path": result}
    except Exception as e:
        pass
