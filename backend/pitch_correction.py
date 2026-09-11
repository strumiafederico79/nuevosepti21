"""
pitch_correction.py — Auto Pitch Correction para vocales/instrumentos.

Notas:
- Usa PYIN para detectar F0 y cromagramas para estimar tonalidad.
- Cuantiza el pitch a una escala musical.
- Aplica una curva de corrección variable en el tiempo, en lugar de usar
  un único desplazamiento medio para todo el audio.
- Mantiene los canales de audio y evita contaminar la corrección con frames
  no sonoros/no detectados.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Pitch classes (C=0 ... B=11)
SCALES = {
    "C_major":  [0, 2, 4, 5, 7, 9, 11],
    "G_major":  [7, 9, 11, 0, 2, 4, 6],
    "D_major":  [2, 4, 6, 7, 9, 11, 1],
    "A_major":  [9, 11, 1, 2, 4, 6, 8],
    "E_major":  [4, 6, 8, 9, 11, 1, 3],
    "B_major":  [11, 1, 3, 4, 6, 8, 10],
    "F#_major": [6, 8, 10, 11, 1, 3, 5],
    "Db_major": [1, 3, 5, 6, 8, 10, 0],
    "Ab_major": [8, 10, 0, 1, 3, 5, 7],
    "Eb_major": [3, 5, 7, 8, 10, 0, 2],
    "Bb_major": [10, 0, 2, 3, 5, 7, 9],
    "F_major":  [5, 7, 9, 10, 0, 2, 4],
    "A_minor":  [9, 11, 0, 2, 4, 5, 7],
    "E_minor":  [4, 6, 7, 9, 11, 0, 2],
    "B_minor":  [11, 1, 2, 4, 6, 7, 9],
    "F#_minor": [6, 8, 9, 11, 1, 2, 4],
    "C#_minor": [1, 3, 4, 6, 8, 9, 11],
    "G#_minor": [8, 10, 11, 1, 3, 4, 6],
    "D#_minor": [3, 5, 6, 8, 10, 11, 1],
    "A#_minor": [10, 0, 1, 3, 5, 6, 8],
    "F_minor":  [5, 7, 8, 10, 0, 1, 3],
    "C_minor":  [0, 2, 3, 5, 7, 8, 10],
    "G_minor":  [7, 9, 10, 0, 2, 3, 5],
    "D_minor":  [2, 4, 5, 7, 9, 10, 0],
}


def _as_audio_array(audio: np.ndarray) -> np.ndarray:
    """Valida y normaliza el dtype del audio sin cambiar su forma."""
    audio = np.asarray(audio)
    if audio.ndim not in (1, 2):
        raise ValueError("audio debe ser mono (n,) o multicanal (n_channels, n_samples).")
    if not np.issubdtype(audio.dtype, np.number):
        raise TypeError("audio debe contener valores numéricos.")
    return audio.astype(np.float32, copy=False)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convierte (n,) o (channels, n) a mono."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0)


def detect_pitch_contour(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 256,
    fmin: float = 60.0,
    fmax: float = 1200.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detecta el contorno F0 con PYIN.

    Devuelve:
        pitches_hz: F0 por frame; NaN cuando no hay pitch fiable.
        confidences: probabilidad de frame sonoro (0..1).
        times: tiempo de cada frame en segundos.
    """
    if librosa is None:
        raise RuntimeError("librosa no está instalada; es necesaria para pitch correction.")

    if sr <= 0:
        raise ValueError("sr debe ser mayor que 0.")
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Debe cumplirse 0 < fmin < fmax.")

    audio = _as_audio_array(audio)
    mono = _to_mono(audio)

    if mono.size == 0:
        return np.array([]), np.array([]), np.array([])

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            mono,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )
    except (AttributeError, TypeError):
        # Compatibilidad con versiones de librosa donde PYIN no está disponible.
        logger.warning("librosa.pyin no disponible; usando piptrack como fallback.")
        try:
            spectrum = np.abs(
                librosa.stft(mono, n_fft=2048, hop_length=hop_length)
            )
            pitches, magnitudes = librosa.piptrack(
                S=spectrum,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                threshold=0.1,
            )

            indexes = np.argmax(magnitudes, axis=0)
            frame_idx = np.arange(magnitudes.shape[1])
            pitches_hz = pitches[indexes, frame_idx].astype(float)
            confidence = magnitudes[indexes, frame_idx].astype(float)

            max_conf = np.max(confidence) if confidence.size else 0.0
            if max_conf > 0:
                confidence /= max_conf
            pitches_hz[pitches_hz <= 0] = np.nan

            times = librosa.frames_to_time(
                np.arange(len(pitches_hz)), sr=sr, hop_length=hop_length
            )
            return pitches_hz, confidence, times
        except Exception as exc:
            logger.exception("Error en fallback de pitch detection: %s", exc)
            return np.array([]), np.array([]), np.array([])
    except Exception as exc:
        logger.exception("Error en pitch detection: %s", exc)
        return np.array([]), np.array([]), np.array([])

    # PYIN devuelve NaN en frames sin F0; voiced_probs aporta confianza.
    pitches_hz = np.asarray(f0, dtype=float)
    confidences = np.asarray(voiced_probs, dtype=float)
    confidences[~np.isfinite(confidences)] = 0.0
    pitches_hz[~np.isfinite(pitches_hz)] = np.nan

    # No usamos voiced_flag para reemplazar la F0: PYIN ya representa la
    # incertidumbre correctamente con NaN + voiced probability.
    del voiced_flag

    times = librosa.frames_to_time(
        np.arange(len(pitches_hz)), sr=sr, hop_length=hop_length
    )
    return pitches_hz, confidences, times


def hz_to_midi(hz: float) -> float:
    """Convierte Hz a número de nota MIDI."""
    if not np.isfinite(hz) or hz <= 0:
        return np.nan
    return 12.0 * np.log2(hz / 440.0) + 69.0


def midi_to_hz(midi: float) -> float:
    """Convierte número MIDI a Hz."""
    if not np.isfinite(midi):
        return np.nan
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def pitch_to_cents_error(pitch_hz: float, reference_hz: float) -> float:
    """Devuelve cuánto hay que desplazar `pitch_hz` para llegar a `reference_hz`.

    Negativo = bajar. Positivo = subir.
    """
    if pitch_hz <= 0 or reference_hz <= 0:
        return 0.0
    midi_pitch = hz_to_midi(pitch_hz)
    midi_ref = hz_to_midi(reference_hz)
    if not np.isfinite(midi_pitch) or not np.isfinite(midi_ref):
        return 0.0
    return (midi_ref - midi_pitch) * 100.0


def detect_key(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[str, float]:
    """Estima la tonalidad como (scale_name, confidence)."""
    if librosa is None:
        return "C_major", 0.0

    audio = _as_audio_array(audio)
    mono = _to_mono(audio)

    if mono.size == 0:
        return "C_major", 0.0

    try:
        chroma = librosa.feature.chroma_cqt(
            y=mono,
            sr=sr,
            hop_length=hop_length,
        )
        chroma_mean = np.mean(chroma, axis=1)
        total = float(np.sum(chroma_mean))

        if total <= 0:
            return "C_major", 0.0

        # Puntúa la energía relativa dentro de la escala, no un valor
        # absoluto dependiente de duración/ganancia.
        normalized = chroma_mean / total
        best_scale = max(
            SCALES,
            key=lambda name: float(np.sum(normalized[SCALES[name]])),
        )
        best_score = float(np.sum(normalized[SCALES[best_scale]]))

        # Confianza: qué tan concentrado está el score en la escala detectada.
        # Si best_score = 7/12 (distribución uniforme), conf = 0.
        # Si best_score ≈ 1.0 (todo en 7 notas), conf = 1.
        # Escala ajustada: 7/12 (baseline) -> 1.0 (pico).
        confidence = float(np.clip((best_score - 7.0 / 12.0) / (1.0 - 7.0 / 12.0), 0.0, 1.0))
        return best_scale, confidence
    except Exception as exc:
        logger.exception("Error en detección de tonalidad: %s", exc)
        return "C_major", 0.0


def quantize_to_scale(
    pitch_hz: float,
    scale_name: str,
    cents_tolerance: float = 50.0,
) -> float:
    """Cuantiza un pitch a la nota más cercana de una escala."""
    if (
        not np.isfinite(pitch_hz)
        or pitch_hz <= 0
        or scale_name not in SCALES
        or cents_tolerance < 0
    ):
        return pitch_hz

    midi_pitch = hz_to_midi(pitch_hz)
    if not np.isfinite(midi_pitch):
        return pitch_hz

    # Buscar en varios registros evita problemas alrededor de los límites
    # entre octavas.
    base_octave = int(np.floor(midi_pitch / 12.0))
    candidates = [
        (octave * 12 + pitch_class)
        for octave in range(base_octave - 1, base_octave + 2)
        for pitch_class in SCALES[scale_name]
    ]
    best_midi = min(candidates, key=lambda value: abs(midi_pitch - value))
    error_cents = abs(best_midi - midi_pitch) * 100.0

    if error_cents <= cents_tolerance:
        corrected_hz = midi_to_hz(float(best_midi))
        logger.debug(
            "Pitch correction: %.1f Hz -> %.1f Hz (%.1f cents)",
            pitch_hz,
            corrected_hz,
            (best_midi - midi_pitch) * 100.0,
        )
        return corrected_hz

    return pitch_hz


def _smooth_valid_curve(
    values: np.ndarray,
    valid: np.ndarray,
    size: int,
) -> np.ndarray:
    """Suaviza solo muestras válidas sin dejar que NaN contamine la curva."""
    if size <= 1:
        return values.copy()

    from scipy.ndimage import uniform_filter1d

    values_filled = np.where(valid, values, 0.0)
    weights = valid.astype(np.float32)

    numerator = uniform_filter1d(values_filled, size=size, mode="nearest")
    denominator = uniform_filter1d(weights, size=size, mode="nearest")

    smoothed = np.zeros_like(values, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=smoothed,
        where=denominator > 1e-8,
    )
    return np.where(valid, smoothed, 0.0)


def _resample_shifts(
    shifts_cents: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpola la curva de cents al centro de cada bloque de audio."""
    valid = (
        np.isfinite(shifts_cents)
        & np.isfinite(times)
        & (times >= 0)
        & (times <= times[-1] if len(times) else False)
    )
    if np.count_nonzero(valid) == 0:
        return np.zeros_like(target_times, dtype=float)

    valid_times = times[valid]
    valid_shifts = shifts_cents[valid]

    if len(valid_times) == 1:
        return np.full_like(target_times, valid_shifts[0], dtype=float)

    return np.interp(
        target_times,
        valid_times,
        valid_shifts,
        left=valid_shifts[0],
        right=valid_shifts[-1],
    )


def apply_pitch_shift(
    audio: np.ndarray,
    sr: int,
    cents: float,
    mode: str = "librosa",
    n_fft: int = 2048,
) -> np.ndarray:
    """Aplica un pitch shift global manteniendo la duración."""
    audio = _as_audio_array(audio)

    if abs(float(cents)) < 1.0:
        return audio.copy()

    if librosa is None or mode != "librosa":
        logger.warning(
            "No hay backend de pitch shifting disponible; se devuelve audio sin modificar."
        )
        return audio.copy()

    try:
        return librosa.effects.pitch_shift(
            audio,
            sr=sr,
            n_steps=float(cents) / 100.0,
            n_fft=n_fft,
        ).astype(audio.dtype, copy=False)
    except Exception as exc:
        logger.exception("Error aplicando pitch shift global: %s", exc)
        return audio.copy()


def apply_time_varying_pitch_shift(
    audio: np.ndarray,
    sr: int,
    shifts_cents: np.ndarray,
    times: np.ndarray,
    hop_length: int = 512,
    window_length: int = 4096,
) -> np.ndarray:
    """Aplica una curva de pitch variable en el tiempo con overlap-add.

    Cada bloque recibe el shift correspondiente a su centro temporal y se
    mezcla con ventanas Hann para reducir clicks entre bloques.
    """
    if librosa is None:
        return _as_audio_array(audio).copy()

    audio = _as_audio_array(audio)
    if audio.size == 0 or len(shifts_cents) == 0:
        return audio.copy()

    n_samples = audio.shape[-1]
    window_length = int(min(max(window_length, 1024), n_samples))
    if window_length < 1024:
        # Audio muy corto: no hay suficientes samples para overlap-add.
        # Aplicar shift global como fallback (promedio de la curva).
        median_shift = float(np.nanmedian(shifts_cents)) if np.any(np.isfinite(shifts_cents)) else 0.0
        logger.debug(f"Audio muy corto ({n_samples} samples); aplicando shift global de {median_shift:.1f} cents")
        return apply_pitch_shift(audio, sr, median_shift)

    if window_length % 2:
        window_length -= 1
    hop = max(256, min(int(hop_length * 2), window_length // 2))

    centers = np.arange(0, n_samples, hop, dtype=int)
    center_times = centers / float(sr)
    block_shifts = _resample_shifts(
        np.asarray(shifts_cents, dtype=float),
        np.asarray(times, dtype=float),
        center_times,
    )

    # Procesar una primera señal seca para conservar transitorios en los
    # bloques donde el shift es prácticamente 0.
    if np.nanmax(np.abs(block_shifts)) < 1.0:
        return audio.copy()

    if audio.ndim == 1:
        out = np.zeros(n_samples, dtype=np.float32)
    else:
        out = np.zeros_like(audio, dtype=np.float32)

    weight = np.zeros(n_samples, dtype=np.float32)
    window = np.hanning(window_length).astype(np.float32)

    for center, shift in zip(centers, block_shifts):
        start = max(0, center - window_length // 2)
        end = min(n_samples, start + window_length)
        start = max(0, end - window_length)
        block_len = end - start

        if block_len < 256:
            continue

        block = audio[..., start:end]
        if abs(float(shift)) < 1.0:
            shifted = block
        else:
            shifted = apply_pitch_shift(
                block,
                sr,
                float(shift),
                n_fft=min(2048, block_len),
            )

        local_window = window[:block_len]
        if audio.ndim == 1:
            out[start:end] += shifted.astype(np.float32) * local_window
        else:
            out[..., start:end] += shifted.astype(np.float32) * local_window

        weight[start:end] += local_window

    # Evita división por cero en los extremos y conserva audio original donde
    # un bloque no pudo procesarse.
    # BUGFIX: para estéreo, weight es 1D (n_samples,); sin newaxis,
    # NumPy hace broadcast correcto con (n_channels, n_samples).
    weight_mask = weight > 1e-6
    if audio.ndim == 1:
        processed = np.divide(
            out,
            np.maximum(weight, 1e-6),
            out=np.zeros_like(out),
            where=weight_mask,
        )
        processed = np.where(weight_mask, processed, audio)
    else:
        # Para multicanal, weight_mask se expande automáticamente
        processed = np.divide(
            out,
            np.maximum(weight, 1e-6),
            out=np.zeros_like(out),
            where=weight_mask,
        )
        processed = np.where(weight_mask, processed, audio)

    return processed.astype(audio.dtype, copy=False)


class PitchCorrectionProcessor:
    """Corrector de pitch automático para vocales/instrumentos."""

    MODES = {
        "OFF": 0.0,
        "LIGHT": 20.0,
        "MEDIUM": 50.0,
        "STRONG": 100.0,
    }

    def __init__(self, sr: int):
        if sr <= 0:
            raise ValueError("sr debe ser mayor que 0.")
        self.sr = int(sr)
        self._detected_key: Optional[str] = None
        self._detected_key_confidence: float = 0.0

    def process(
        self,
        audio: np.ndarray,
        mode: str = "MEDIUM",
        scale: Optional[str] = None,
        glide_time_ms: float = 50.0,
        hop_length: int = 256,
        fmin: float = 60.0,
        fmax: float = 1200.0,
        confidence_threshold: float = 0.20,
    ) -> np.ndarray:
        """Aplica corrección de pitch variable en el tiempo."""
        audio = _as_audio_array(audio)

        mode = mode.upper()
        if mode == "OFF":
            return audio.copy()
        if mode not in self.MODES:
            raise ValueError(
                f"mode inválido: {mode!r}. Usa uno de {tuple(self.MODES)}."
            )
        # BUGFIX: el frontend puede mandar scale="" (string vacío) o None
        # para indicar "detección automática". Aquí normalizamos ambos a None.
        if scale == "":
            scale = None
        if scale is not None and scale not in SCALES:
            raise ValueError(
                f"scale inválida: {scale!r}. Usa una de {tuple(SCALES)}."
            )
        if glide_time_ms < 0:
            raise ValueError("glide_time_ms no puede ser negativo.")

        pitches, confidences, times = detect_pitch_contour(
            audio,
            self.sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )
        if len(pitches) == 0:
            return audio.copy()

        if scale is None:
            scale, conf = detect_key(audio, self.sr, hop_length=hop_length)
            self._detected_key = scale
            self._detected_key_confidence = conf
        else:
            self._detected_key = scale
            self._detected_key_confidence = 1.0

        logger.info(
            "Pitch correction: mode=%s scale=%s confidence=%.2f",
            mode,
            self._detected_key,
            self._detected_key_confidence,
        )

        # Solo corrige frames con pitch y confianza suficientes.
        valid = (
            np.isfinite(pitches)
            & np.isfinite(confidences)
            & (confidences >= confidence_threshold)
        )
        shifts_cents = np.zeros_like(pitches, dtype=float)

        for idx in np.flatnonzero(valid):
            corrected = quantize_to_scale(
                float(pitches[idx]),
                scale,
                self.MODES[mode],
            )
            shifts_cents[idx] = pitch_to_cents_error(
                float(pitches[idx]),
                float(corrected),
            )

        # Glide expresado en número de frames.
        if glide_time_ms > 0 and len(shifts_cents) > 1:
            frame_duration_ms = 1000.0 * hop_length / self.sr
            glide_frames = max(1, int(round(glide_time_ms / frame_duration_ms)))
            shifts_cents = _smooth_valid_curve(
                shifts_cents,
                valid,
                glide_frames,
            )

        if not np.any(valid):
            logger.warning("No se encontraron frames con pitch suficientemente fiable.")
            return audio.copy()

        return apply_time_varying_pitch_shift(
            audio,
            self.sr,
            shifts_cents=shifts_cents,
            times=times,
            hop_length=hop_length,
            window_length=max(2048, hop_length * 8),
        )

    def get_detected_key(self) -> Tuple[str, float]:
        """Retorna tonalidad detectada y confianza."""
        return (
            self._detected_key or "C_major",
            self._detected_key_confidence,
        )
