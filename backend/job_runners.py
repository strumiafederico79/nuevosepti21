"""Workers de background para procesamiento de audio.

Este módulo contiene la lógica de ejecución de jobs que no debería vivir en
el punto de entrada HTTP. Las dependencias se inyectan mediante
``create_job_runners`` para mantenerlo testeable y evitar imports circulares.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import librosa
import soundfile as sf


logger = logging.getLogger(__name__)


def create_job_runners(
    *,
    jobs,
    cleanup_old: Callable[[], None],
    process_audio,
    process_audio_with_reference,
    normalize_by_lufs,
    separate_stems,
    separate_vocals_hq,
    analyze_stems_full,
    measure_lufs_integrated,
    stems_dir: str,
):
    """Construye los workers con las dependencias concretas de la aplicación."""

    def make_progress_cb(job_id: str):
        def _cb(pct: int, stage: str):
            if not jobs.exists(job_id):
                return
            jobs.update_job(job_id, progress=pct, stage=stage)
        return _cb

    def run_mastering_job(job_id: str, input_path: str, params: dict):
        jobs.update_job(job_id, status="processing", started_at=time.time(), progress=0, stage="Iniciando procesamiento")
        jobs.set_stage(job_id, "analyzing", progress=10)
        jobs.append_log(job_id, "Preparando análisis del archivo", stage="analyzing")
        jobs.snapshot_params(job_id, params)
        try:
            cleanup_old()
            jobs.append_log(job_id, "Procesando mastering con la cadena de audio", stage="rendering")
            jobs.set_stage(job_id, "rendering", progress=25)
            result = process_audio(input_path, progress_cb=make_progress_cb(job_id), **params)
            jobs.set_stage(job_id, "generating_preview", progress=85)
            jobs.append_log(job_id, "Render finalizado; preparando exportes y preview", stage="generating_preview")
            output_path = result.get("output_path")
            output_format = str(params.get("output_format", "wav")).lower()
            if output_path and os.path.exists(output_path):
                jobs.add_export(
                    job_id,
                    "master_final",
                    output_path,
                    version_name="master_final",
                    format=output_format,
                    bit_depth=params.get("output_bit_depth", 24),
                    target_lufs=params.get("target_lufs"),
                    platform_target=params.get("platform_target"),
                )
            jobs.update_job(job_id, status="done", result=result, finished_at=time.time(), progress=100, stage="Completado")
            jobs.append_log(job_id, f"Master finalizado correctamente: {output_path or 'sin ruta'}", stage="preview_ready")
            logger.info("Job %s done: %s", job_id, result.get("output_path"))
        except (ValueError, RuntimeError) as e:
            jobs.mark_failed(job_id, str(e), stage="failed", progress=0)
            jobs.append_log(job_id, f"Error en el job: {e}", stage="failed", level="error")
            logger.error("Job %s failed (value/runtime error): %s", job_id, e, exc_info=True)
        except OSError as e:
            jobs.mark_failed(job_id, str(e), stage="failed", progress=0)
            jobs.append_log(job_id, f"Error de sistema: {e}", stage="failed", level="error")
            logger.error("Job %s failed (OS error): %s", job_id, e, exc_info=True)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def run_reference_job(job_id: str, input_path: str, reference_path: str, params: dict):
        jobs.update_job(job_id, status="processing", started_at=time.time(), progress=0, stage="Iniciando procesamiento")
        jobs.set_stage(job_id, "analyzing", progress=10)
        jobs.snapshot_params(job_id, params)
        try:
            cleanup_old()
            jobs.append_log(job_id, "Procesando mastering con referencia", stage="rendering")
            jobs.set_stage(job_id, "rendering", progress=25)
            result = process_audio_with_reference(input_path, reference_path, progress_cb=make_progress_cb(job_id), **params)
            output_path = result.get("output_path")
            if output_path and os.path.exists(output_path):
                jobs.add_export(
                    job_id,
                    "reference_master",
                    output_path,
                    version_name="reference_master",
                    format=str(params.get("output_format", "wav")).lower(),
                )
            jobs.update_job(job_id, status="done", result=result, finished_at=time.time(), progress=100, stage="Completado")
            logger.info("Job %s (reference match) done: %s", job_id, result.get("output_path"))
        except (ValueError, RuntimeError) as e:
            jobs.mark_failed(job_id, str(e), stage="failed", progress=0)
            logger.error("Job %s (reference match) failed: %s", job_id, e, exc_info=True)
        except OSError as e:
            jobs.mark_failed(job_id, str(e), stage="failed", progress=0)
            logger.error("Job %s (reference match) failed (OS error): %s", job_id, e, exc_info=True)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(reference_path):
                os.remove(reference_path)

    def run_normalize_job(job_id: str, input_path: str, params: dict):
        jobs.update_job(job_id, status="processing", started_at=time.time(), progress=0, stage="Iniciando normalización")
        jobs.set_stage(job_id, "analyzing", progress=10)
        jobs.snapshot_params(job_id, params)
        try:
            cleanup_old()
            jobs.append_log(job_id, "Normalizando por LUFS", stage="rendering")
            jobs.set_stage(job_id, "rendering", progress=25)
            result = normalize_by_lufs(input_path, progress_cb=make_progress_cb(job_id), **params)
            output_path = result.get("output_path")
            if output_path and os.path.exists(output_path):
                jobs.add_export(
                    job_id,
                    "lufs_normalized",
                    output_path,
                    version_name="lufs_normalized",
                    format=str(params.get("output_format", "wav")).lower(),
                )
            jobs.update_job(job_id, status="done", result=result, finished_at=time.time(), progress=100, stage="Completado")
            logger.info("Job %s (lufs normalize) done: %s", job_id, result.get("output_path"))
        except (ValueError, RuntimeError) as e:
            jobs.mark_failed(job_id, str(e), stage="failed", progress=0)
            logger.error("Job %s (lufs normalize) failed: %s", job_id, e, exc_info=True)
        except OSError as e:
            jobs.mark_failed(job_id, str(e), stage="failed", progress=0)
            logger.error("Job %s (lufs normalize) failed (OS error): %s", job_id, e, exc_info=True)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def run_stems_job(job_id: str, input_path: str, mode: str = "demucs_4stem"):
        jobs.update_job(job_id, status="processing", started_at=time.time(), progress=0, stage="Iniciando separación")
        try:
            cleanup_old()
            audio, sr = librosa.load(input_path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[None, :]

            if mode == "vocals_hq":
                stems = separate_vocals_hq(audio, sr, progress_cb=make_progress_cb(job_id))
            else:
                stems = separate_stems(audio, sr, progress_cb=make_progress_cb(job_id))

            jobs.update_job(job_id, stage="Analizando stems", progress=96)
            # SERIE: análisis de stems sin subproceso. Antes corría dentro de un
            # ThreadPoolExecutor(max_workers=1) solo para tener timeout; como
            # ahora todo el procesamiento de audio es estrictamente secuencial,
            # se llama directo. Si analyze_stems_full se cuelga, el caller
            # (job_store) verá el job sin progreso y podrá cancelarlo.
            analysis = analyze_stems_full(stems, sr, measure_lufs_integrated)

            stem_dir = os.path.join(stems_dir, job_id)
            os.makedirs(stem_dir, exist_ok=True)
            stem_paths = {}
            for name, stem_audio in stems.items():
                out_path = os.path.join(stem_dir, f"{name}.wav")
                data_to_write = stem_audio.T if stem_audio.ndim == 2 else stem_audio
                sf.write(out_path, data_to_write, sr, subtype="PCM_24")
                stem_paths[name] = out_path

            jobs.update_job(
                job_id,
                status="done",
                finished_at=time.time(),
                progress=100,
                stage="Completado",
                stem_analysis=analysis,
                stem_paths=stem_paths,
                available_stems=list(stem_paths.keys()),
            )
            logger.info("Job %s (stems) done: %s", job_id, list(stem_paths.keys()))
        except (ValueError, RuntimeError) as e:
            jobs.update_job(job_id, status="error", error=str(e))
            logger.error("Job %s (stems) failed: %s", job_id, e, exc_info=True)
        except OSError as e:
            jobs.update_job(job_id, status="error", error=str(e))
            logger.error("Job %s (stems) failed (OS error): %s", job_id, e, exc_info=True)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    return run_mastering_job, run_reference_job, run_normalize_job, run_stems_job
