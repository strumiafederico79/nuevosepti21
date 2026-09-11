"""Persistent reference-library index and lightweight watcher."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
INDEX_FILENAME = "reference_library_index.json"

_index: Dict[str, dict] = {}
_index_lock = threading.Lock()
_library_dir: str = ""
_index_path: str = ""
_watcher_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _file_id(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]


def _analyze_file(path: str) -> Optional[dict]:
    try:
        info = sf.info(path)
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        max_frames = min(info.frames, int(info.samplerate * 180))
        audio, sr = sf.read(path, frames=max_frames, dtype="float32", always_2d=True)
        audio = audio.T
        peak_db = float(20 * np.log10(np.max(np.abs(audio)) + 1e-9))
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            mono = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
            lufs = float(meter.integrated_loudness(mono))
            if not np.isfinite(lufs):
                lufs = float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-9)) - 0.691
        except Exception:
            mono = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
            lufs = float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-9)) - 0.691
        return {
            "filename": os.path.basename(path),
            "path": os.path.abspath(path),
            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
            "duration_sec": round(duration, 2),
            "lufs": round(lufs, 1),
            "peak_db": round(peak_db, 1),
            "sr": info.samplerate,
            "channels": info.channels,
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        logger.warning("reference_library: error analizando %s: %s", path, exc)
        return None


def _save_index() -> None:
    if not _index_path:
        return
    try:
        tmp = _index_path + ".tmp"
        with _index_lock:
            payload = dict(_index)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, _index_path)
    except Exception as exc:
        logger.error("reference_library: error guardando índice: %s", exc)


def _load_index() -> None:
    global _index
    if not _index_path or not os.path.exists(_index_path):
        return
    try:
        with open(_index_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            logger.warning("reference_library: índice inválido; se reconstruye")
            return
        with _index_lock:
            _index = data
    except Exception as exc:
        logger.warning("reference_library: índice corrupto; se reconstruye: %s", exc)


def scan(force: bool = False) -> int:
    """Escanea y actualiza el índice; devuelve el número de entradas."""
    if not _library_dir or not os.path.isdir(_library_dir):
        with _index_lock:
            return len(_index)

    found_ids = set()
    changed = False
    for fname in sorted(os.listdir(_library_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED:
            continue
        fpath = os.path.join(_library_dir, fname)
        try:
            mtime = os.path.getmtime(fpath)
        except OSError:
            continue
        fid = _file_id(fname)
        found_ids.add(fid)
        with _index_lock:
            existing = _index.get(fid)
            needs_analysis = (
                force
                or existing is None
                or existing.get("path") != os.path.abspath(fpath)
                or abs(existing.get("_mtime", 0) - mtime) > 1.0
            )
        if needs_analysis:
            entry = _analyze_file(fpath)
            if entry:
                entry["id"] = fid
                entry["_mtime"] = mtime
                with _index_lock:
                    _index[fid] = entry
                changed = True
                logger.info("reference_library: indexado %s (LUFS %s)", fname, entry["lufs"])

    with _index_lock:
        stale = [fid for fid in _index if fid not in found_ids]
        for fid in stale:
            del _index[fid]
            changed = True
        count = len(_index)
    if changed:
        _save_index()
    return count


def _snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    if not _library_dir or not os.path.isdir(_library_dir):
        return snapshot
    for fname in os.listdir(_library_dir):
        if os.path.splitext(fname)[1].lower() not in ALLOWED:
            continue
        try:
            snapshot[fname] = os.path.getmtime(os.path.join(_library_dir, fname))
        except OSError:
            pass
    return snapshot


def _watch_loop(interval: float = 3.0) -> None:
    last_snapshot = _snapshot()
    while not _stop_event.wait(interval):
        try:
            current = _snapshot()
            if current != last_snapshot:
                logger.info("reference_library: cambio detectado, re-escaneando")
                scan()
                last_snapshot = current
        except Exception as exc:
            logger.warning("reference_library watcher error: %s", exc)


def init(library_dir: str) -> None:
    """Inicializa índice y watcher. Es idempotente."""
    global _library_dir, _index_path, _watcher_thread
    _library_dir = os.path.abspath(library_dir)
    _index_path = os.path.join(_library_dir, INDEX_FILENAME)
    os.makedirs(_library_dir, exist_ok=True)
    _stop_event.clear()
    _load_index()
    n = scan()
    logger.info("reference_library: %d referencias en '%s'", n, _library_dir)
    if _watcher_thread is None or not _watcher_thread.is_alive():
        _watcher_thread = threading.Thread(
            target=_watch_loop, daemon=True, name="ref-lib-watcher"
        )
        _watcher_thread.start()


def shutdown() -> None:
    """Detiene el watcher sin dejar hilos persistentes al apagar el proceso."""
    global _watcher_thread
    _stop_event.set()
    thread = _watcher_thread
    if thread and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.5)
    _watcher_thread = None


def list_entries() -> List[dict]:
    with _index_lock:
        entries = list(_index.values())
    return sorted(
        [{k: v for k, v in entry.items() if not k.startswith("_")} for entry in entries],
        key=lambda entry: entry.get("filename", "").lower(),
    )


def get_path(ref_id: str) -> Optional[str]:
    with _index_lock:
        entry = _index.get(ref_id)
    if not entry:
        return None
    path = entry.get("path")
    if path and os.path.exists(path):
        return path
    with _index_lock:
        _index.pop(ref_id, None)
    _save_index()
    return None


def diagnostics() -> dict:
    """Estado operativo para distinguir carpeta vacía de ruta incorrecta."""
    exists = bool(_library_dir and os.path.isdir(_library_dir))
    files = []
    if exists:
        files = [
            name for name in os.listdir(_library_dir)
            if os.path.splitext(name)[1].lower() in ALLOWED
        ]
    with _index_lock:
        indexed = len(_index)
    return {
        "directory": _library_dir,
        "exists": exists,
        "audio_files_on_disk": len(files),
        "indexed_entries": indexed,
        "watcher_alive": bool(_watcher_thread and _watcher_thread.is_alive()),
        "allowed_extensions": sorted(ALLOWED),
    }
