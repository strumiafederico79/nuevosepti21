from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

try:
    from .mastering import analyze_audio, mix_advice, spectrum_analysis_fft
except ImportError:  # pragma: no cover
    from mastering import analyze_audio, mix_advice, spectrum_analysis_fft

logger = logging.getLogger(__name__)


class AudioService:
    """Servicio de lectura y análisis server-side de audio."""

    def __init__(self, upload_dir: str = "uploads"):
        import os
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        analysis = analyze_audio(audio, sr)
        analysis["mix_advice"] = mix_advice(analysis)
        return analysis

    def analyze_with_spectrum(self, file_path: str, n_fft: int = 4096, n_bins: int = 64) -> Dict[str, Any]:
        """Carga el audio una vez y devuelve análisis + espectro calculados en servidor."""
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        analysis = analyze_audio(audio, sr)
        analysis["mix_advice"] = mix_advice(analysis)
        analysis["fft_spectrum"] = spectrum_analysis_fft(audio, sr, n_fft=n_fft, n_bins=n_bins)
        analysis["analysis_source"] = {
            "sample_rate": int(sr),
            "channels": int(audio.shape[0]),
            "samples": int(audio.shape[1]),
            "duration_sec": round(float(audio.shape[1] / sr), 3),
            "computed_server_side": True,
        }
        return analysis

    def spectrum_file(self, file_path: str, n_fft: int = 4096, n_bins: int = 64) -> Dict[str, Any]:
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        return spectrum_analysis_fft(audio, sr, n_fft=n_fft, n_bins=n_bins)

    def read_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        return audio, sr

    def get_duration(self, file_path: str) -> Optional[float]:
        try:
            info = sf.info(file_path)
            if info.samplerate:
                return round(info.frames / info.samplerate, 3)
        except Exception as e:
            logger.warning(f"⚠️ Error leyendo duración de {file_path}: {e}")
        return None
