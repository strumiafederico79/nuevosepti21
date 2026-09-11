from __future__ import annotations

import os

# Configure native numerical runtimes before importing numpy/librosa. This is
# required because preview renders run in isolated multiprocessing children.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hashlib
import json
import multiprocessing as mp
import re
import shutil
import signal
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
import librosa
# Preview workers must not initialize Numba's native JIT on Python versions
# where its compiled extension may be incompatible with the host runtime.
if mp.current_process().name != "MainProcess":
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("LGMDM_DISABLE_NUMBA", "1")


SOURCE_ID_PATTERN = re.compile(r"^[a-f0-9]{16,128}$")


def _safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:  # noqa: BLE001
        return f"absent({type(exc).__name__})"
    return getattr(module, "__version__", "?")


def _is_valid_source_id(source_id: str) -> bool:
    return bool(SOURCE_ID_PATTERN.fullmatch(source_id or ""))


def _crop_preview(audio: np.ndarray, sr: int, preview_seconds: float) -> np.ndarray:
    max_samples = min(int(preview_seconds * sr), audio.shape[1])
    if max_samples <= 0:
        return audio
    start = min(int(15 * sr), max(0, audio.shape[1] - max_samples))
    return audio[:, start:start + max_samples]


class PreviewSnapshotError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    """Convierte tipos numpy (float32/float64/int64/ndarray, etc.) a tipos
    nativos de Python antes de json.dumps. chain_meters ya viene mayormente
    redondeado con round(float(...), 2), pero esto es una red de seguridad
    barata por si algún sub-dict se cuela sin convertir."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class PreviewRenderer:
    """Owns immutable 25-second preview sources and cancellable renders."""

    def __init__(self, directory: str, duration_sec: int = 25, ttl_sec: int = 3600):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.directory / "snapshots"
        self.renders_dir = self.directory / "renders"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.duration_sec = duration_sec
        self.ttl_sec = ttl_sec
        # Progreso del render en curso, por source_id. Vive en un Manager
        # (proceso propio de multiprocessing) para que sea picklable y
        # compartible con el proceso hijo `spawn` que hace el render real
        # — un callback Python normal no se puede pasar a otro proceso.
        self._progress_manager = mp.Manager()
        self._progress = self._progress_manager.dict()

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl_sec
        for path in list(self.snapshots_dir.glob("*")) + list(self.renders_dir.glob("*")):
            try:
                if path.stat().st_mtime < cutoff:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
            except OSError:
                continue

    def _source_paths(self, source_id: str) -> tuple[Path, Path]:
        return (
            self.snapshots_dir / f"{source_id}.wav",
            self.snapshots_dir / f"{source_id}.json",
        )

    def _meters_path(self, source_id: str) -> Path:
        return self.snapshots_dir / f"{source_id}.meters.json"

    def create_snapshot(self, input_path: str, owner_id: str) -> dict[str, Any]:
        if not os.path.exists(input_path):
            raise PreviewSnapshotError("El archivo original no existe")

        source_id = uuid.uuid4().hex
        audio, sr = librosa.load(input_path, sr=None, mono=False, duration=25, offset=0)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if audio.shape[1] == 0 or sr <= 0:
            raise PreviewSnapshotError("El archivo no contiene audio utilizable")

        # Validar que no haya NaN/Inf (causan SIGSEGV en scipy.fft)
        if not np.isfinite(audio).all():
            raise PreviewSnapshotError("El audio contiene valores NaN o Inf y no puede procesarse")

        cropped = _crop_preview(audio, sr, self.duration_sec).astype(np.float32, copy=False)
        source_path, meta_path = self._source_paths(source_id)
        tmp_audio = source_path.with_suffix(".tmp.wav")
        sf.write(tmp_audio, cropped.T, sr, subtype="PCM_24", format="WAV")
        os.replace(tmp_audio, source_path)

        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        duration = float(cropped.shape[1] / sr)
        meta = {
            "source_id": source_id,
            "owner_id": owner_id,
            "duration_sec": duration,
            "sample_rate": int(sr),
            "channels": int(cropped.shape[0]),
            "source_sha256": digest,
            "created_at": time.time(),
        }
        tmp_meta = meta_path.with_suffix(".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
        return meta

    def get_source(self, source_id: str, owner_id: str) -> tuple[str, dict[str, Any]]:
        if not _is_valid_source_id(source_id):
            raise PreviewSnapshotError("Identificador de snapshot inválido")
        source_path, meta_path = self._source_paths(source_id)
        if not source_path.exists() or not meta_path.exists():
            raise PreviewSnapshotError("Snapshot de Preview inexistente o expirado")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("owner_id") != owner_id:
            raise PreviewSnapshotError("El snapshot no pertenece al usuario actual")
        return str(source_path), meta

    @staticmethod
    def _render_worker(
        source_path: str,
        output_path: str,
        meters_path: str,
        params: dict[str, Any],
        duration_sec: int,
        progress_dict,
        source_id: str,
    ) -> None:
        """Render real-time del preview con la cadena DSP completa.

        El render corre dentro de un subproceso ``spawn`` para que cualquier
        segfault de las extensiones nativas (SciPy, Numba, Torch) quede
        contenido y el servidor principal siga respondiendo. Si el proceso
        muere por una señal, ``render_cancellable`` la decodifica y la
        expone como ``RuntimeError`` con el nombre real (``SIGSEGV``,
        ``SIGKILL``, etc.). No hay fallback: el preview siempre es render
        real, no copia del snapshot."""
        import logging
        import faulthandler
        import sys
        import traceback

        log_path = os.environ.get(
            "LGMDM_PREVIEW_LOG",
            "/tmp/lgmdm-preview.log",
        )
        logger = logging.getLogger("lgmdm.preview")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            try:
                handler = logging.FileHandler(log_path, encoding="utf-8")
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(process)d] %(levelname)s %(message)s"
                    )
                )
                logger.addHandler(handler)
            except OSError:
                pass

        faulthandler.enable()
        logger.info(
            "render worker boot pid=%d python=%s numpy=%s scipy=%s numba=%s torch=%s",
            os.getpid(),
            sys.version.split()[0],
            _safe_version("numpy"),
            _safe_version("scipy"),
            _safe_version("numba"),
            _safe_version("torch"),
        )

        try:
            from backend.mastering import process_audio
        except ImportError:
            from mastering import process_audio

        # Validación rápida del snapshot. Si el WAV está corrupto o tiene
        # NaN/Inf, fallamos antes de entrar a la cadena DSP para que el
        # log muestre la causa real.
        audio_check, sr_check = sf.read(source_path, dtype="float32", always_2d=True)
        if not np.isfinite(audio_check).all():
            raise RuntimeError("Audio de snapshot contiene NaN/Inf")

        def _progress_cb(pct, stage):
            try:
                progress_dict[source_id] = {
                    "percent": int(pct),
                    "stage": str(stage),
                    "done": False,
                }
            except Exception:
                # El dict es un proxy de multiprocessing.Manager — si el
                # proceso padre ya lo cerró (render cancelado/servidor
                # bajando), escribir puede fallar. No debe tirar abajo el
                # render por esto, igual que _report() en mastering.py.
                pass

        clean = dict(params)
        clean.pop("progress_cb", None)
        clean["progress_cb"] = _progress_cb
        clean["input_path"] = source_path
        clean["preview_seconds"] = duration_sec
        clean["output_format"] = "wav"
        clean["output_bit_depth"] = 24

        logger.info(
            "process_audio input=%s sr=%d shape=%s params=%d",
            source_path,
            sr_check,
            audio_check.shape,
            len(clean),
        )
        try:
            result = process_audio(**clean)
        except Exception as exc:
            logger.error("process_audio raised: %s", exc)
            logger.error("traceback:\n%s", traceback.format_exc())
            raise
        produced = result.get("output_path")
        if not produced or not os.path.exists(produced):
            raise RuntimeError("El motor de mastering no generó el Preview")
        if os.path.abspath(produced) != os.path.abspath(output_path):
            shutil.move(produced, output_path)
        else:
            os.replace(produced, output_path)

        chain_meters = result.get("chain_meters") or {}
        if not isinstance(chain_meters, dict):
            chain_meters = {"value": chain_meters}
        chain_meters.setdefault("preview_mode", "full-dsp")
        tmp_meters = meters_path + ".tmp"
        with open(tmp_meters, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(chain_meters), handle, ensure_ascii=False)
        os.replace(tmp_meters, meters_path)
        try:
            progress_dict[source_id] = {"percent": 100, "stage": "Completado", "done": True}
        except Exception:
            pass
        logger.info("render worker done output=%s", output_path)

    def render_cancellable(
        self,
        source_path: str,
        params: dict[str, Any],
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Corre el render en un subproceso aislado (spawn) para que un
        segfault del stack nativo afecte solo a este render, no a todo el
        servidor. Recuperar la cancelación real es un beneficio secundario
        del aislamiento."""
        source_id = Path(source_path).stem
        output_path = str(self.renders_dir / f"render-{uuid.uuid4().hex}.wav")
        meters_path = str(self._meters_path(source_id))
        ctx = mp.get_context("spawn")
        process = ctx.Process(
            target=self._render_worker,
            args=(source_path, output_path, meters_path, params, self.duration_sec,
                  self._progress, source_id),
            daemon=True,
        )
        process.start()
        try:
            while process.is_alive():
                if cancel_check is not None and cancel_check():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=2)
                    raise InterruptedError("Render de Preview cancelado")
                time.sleep(0.20)
            process.join(timeout=1)
            if process.exitcode != 0:
                if process.exitcode < 0:
                    try:
                        signal_name = signal.Signals(-process.exitcode).name
                    except ValueError:
                        signal_name = f"SIG{-process.exitcode}"
                    raise RuntimeError(
                        f"Render de Preview terminó por señal {signal_name} "
                        f"(exitcode {process.exitcode})"
                    )
                raise RuntimeError(f"Render de Preview finalizar con código {process.exitcode}")
            if not os.path.exists(output_path):
                raise RuntimeError("Render de Preview finalizar sin archivo de salida")
            return output_path
        except InterruptedError:
            self._progress[source_id] = {"percent": 0, "stage": "Cancelado", "done": True}
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
        except BaseException:
            self._progress[source_id] = {"percent": 0, "stage": "Error", "done": True}
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            if os.path.exists(output_path):
                os.remove(output_path)
            raise

    def get_meters(self, source_id: str) -> dict[str, Any]:
        if not _is_valid_source_id(source_id):
            raise PreviewSnapshotError("Identificador de snapshot inválido")
        path = self._meters_path(source_id)
        if not path.exists():
            raise PreviewSnapshotError("Todavía no hay telemetría de GR para este snapshot")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_progress(self, source_id: str) -> dict[str, Any]:
        """Estado más reciente reportado por el proceso de render para este
        source_id. Devuelve un estado neutro si todavía no arrancó a
        reportar (por ejemplo, justo al lanzar el proceso hijo) o si ya
        no hay nada corriendo para ese id."""
        if not _is_valid_source_id(source_id):
            raise PreviewSnapshotError("Identificador de snapshot inválido")
        entry = self._progress.get(source_id)
        if not entry:
            return {"percent": 0, "stage": "Iniciando render...", "done": False}
        return dict(entry)

    @staticmethod
    def remove_render(path: Optional[str]) -> None:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
