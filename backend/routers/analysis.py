from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

try:
    from ..audio_service import AudioService
    from ..mastering import mix_advice, spectrum_analysis_fft
    from ..validation_utils import validate_audio_file
except ImportError:  # pragma: no cover
    from audio_service import AudioService
    from mastering import mix_advice, spectrum_analysis_fft
    from validation_utils import validate_audio_file


def _spectrum_from_file(file_path: str, n_fft: int, n_bins: int):
    service = AudioService()
    return service.spectrum_file(file_path, n_fft=n_fft, n_bins=n_bins)


def _analyze_from_file(file_path: str) -> dict:
    try:
        return AudioService().analyze_file(file_path)
    except Exception as e:
        raise RuntimeError(f"Error analizando archivo: {e}") from e


def create_analysis_router(*, upload_dir: str, read_and_validate, logger, current_user_dependency, audio_service: AudioService | None = None) -> APIRouter:
    router = APIRouter(tags=["Análisis"])

    @router.post("/analysis")
    async def analyze_complete(
        file: UploadFile = File(...),
        n_fft: int = Query(4096, ge=256, le=16384),
        n_bins: int = Query(64, ge=8, le=256),
        current_user: dict = Depends(current_user_dependency),
    ):
        logger.info("🔍 %s análisis server-side: %s", current_user["email"], file.filename)
        validate_audio_file(file.filename)
        data = await read_and_validate(file)
        tmp = os.path.join(upload_dir, f"analysis_{uuid.uuid4().hex}")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            if audio_service:
                result = await run_in_threadpool(audio_service.analyze_with_spectrum, tmp, n_fft, n_bins)
            else:
                result = await run_in_threadpool(AudioService().analyze_with_spectrum, tmp, n_fft, n_bins)
            logger.info("✓ Análisis completo server-side: %s", file.filename)
            return result
        except Exception as exc:
            logger.error("❌ Error análisis server-side %s: %s", file.filename, exc, exc_info=True)
            raise HTTPException(500, str(exc)) from exc
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @router.post("/analyze")
    async def analyze_legacy(file: UploadFile = File(...), current_user: dict = Depends(current_user_dependency)):
        validate_audio_file(file.filename)
        data = await read_and_validate(file)
        tmp = os.path.join(upload_dir, f"analyze_{uuid.uuid4().hex}")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            return await run_in_threadpool(audio_service.analyze_file if audio_service else _analyze_from_file, tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @router.post("/mix-advice")
    async def get_mix_advice(file: UploadFile = File(...), current_user: dict = Depends(current_user_dependency)):
        validate_audio_file(file.filename)
        data = await read_and_validate(file)
        tmp = os.path.join(upload_dir, f"advice_{uuid.uuid4().hex}")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            analysis = await run_in_threadpool(audio_service.analyze_file if audio_service else _analyze_from_file, tmp)
            return {"analysis": analysis, **mix_advice(analysis)}
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @router.post("/spectrum")
    async def spectrum(
        file: UploadFile = File(...),
        n_fft: int = Query(4096, ge=256, le=16384),
        n_bins: int = Query(64, ge=8, le=256),
        current_user: dict = Depends(current_user_dependency),
    ):
        validate_audio_file(file.filename)
        data = await read_and_validate(file)
        tmp = os.path.join(upload_dir, f"spectrum_{uuid.uuid4().hex}")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            if audio_service:
                return await run_in_threadpool(audio_service.spectrum_file, tmp, n_fft, n_bins)
            return await run_in_threadpool(_spectrum_from_file, tmp, n_fft, n_bins)
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    return router
