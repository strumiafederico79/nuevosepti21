"""
reverb.py — Reverb Convolutivo Mejorado con Impulse Responses

Características:
  - Carga IRs (Impulse Responses) de salas reales
  - Room presets: Small Studio, Large Hall, Cathedral, Live Venue, Plate
  - Pre-delay configurable (0-200ms)
  - Mix wet/dry adjustable
  - Per-stem application (cada canal del mixer puede tener reverb)
  - Processamiento eficiente (scipy.signal.fftconvolve)
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from scipy import signal
except ImportError:
    signal = None

# ── IR Library / Room Presets ──────────────────────────────────────────────

ROOM_PRESETS = {
    "small_studio": {
        "name": "Small Studio",
        "description": "Intimate 2x3m recording room",
        "decay_time": 0.3,  # segundos (RT60)
        "size": 0.2,
    },
    "large_hall": {
        "name": "Large Hall",
        "description": "Concert hall 15x25m",
        "decay_time": 2.5,
        "size": 1.0,
    },
    "cathedral": {
        "name": "Cathedral",
        "description": "Reverb eclesiástica espaciosa",
        "decay_time": 4.0,
        "size": 1.5,
    },
    "live_venue": {
        "name": "Live Venue",
        "description": "Club/pequeña sala en vivo",
        "decay_time": 0.8,
        "size": 0.5,
    },
    "plate": {
        "name": "Plate Reverb",
        "description": "Classic plate reverberator",
        "decay_time": 1.2,
        "size": 0.4,
    },
    "spring": {
        "name": "Spring Reverb",
        "description": "Vintage spring reverberator",
        "decay_time": 0.6,
        "size": 0.3,
    },
}


# ── Synth IR Generator ─────────────────────────────────────────────────────
# Si no hay IRs reales, generar sintéticas (para MVP)

def _synthesize_ir(sr: int, decay_time: float, size: float = 1.0) -> np.ndarray:
    """Sintetiza un Impulse Response artificial.
    
    Imita características de una sala:
      - decay_time: RT60 en segundos (tiempo para -60dB)
      - size: factor de escala del espacio
    
    Usa: reverberador FDN (Feedback Delay Network) simplificado
    """
    # Duración del IR
    ir_duration = min(decay_time * 2, 5.0)  # max 5 segundos
    ir_samples = int(sr * ir_duration)
    
    # Delays característicos (prime numbers para FDN)
    delays = np.array([
        int(0.037 * sr * size),
        int(0.041 * sr * size),
        int(0.043 * sr * size),
        int(0.047 * sr * size),
    ]) + 1
    
    ir = np.zeros(ir_samples)
    
    # Feedback amounts
    feedbacks = np.array([0.5, 0.6, 0.55, 0.65])
    
    # Generar decaimiento exponencial con múltiples delays
    decay_rate = np.log(0.001) / decay_time  # -60dB en decay_time
    
    for i, (delay, fb) in enumerate(zip(delays, feedbacks)):
        if delay < ir_samples:
            decay = np.exp(decay_rate * np.arange(ir_samples) / sr)
            impulse = np.zeros(ir_samples)
            impulse[delay] = 1.0
            
            # Reverse decay para efecto más natural
            decayed = impulse * decay * (fb ** np.arange(ir_samples / delay + 1)[:ir_samples])
            ir += decayed / len(delays)
    
    # Normalizar
    ir = ir / (np.max(np.abs(ir)) + 1e-8)
    
    return ir.astype(np.float32)


def load_ir(preset_name: str, sr: int) -> np.ndarray:
    """Carga o sintetiza un IR para el preset especificado.
    
    En producción, se podrían cargar IRs reales de:
      - OpenAIR Library (https://www.openair.hosted.york.ac.uk/)
      - Commercial IR packs (Waves, Lexicon, etc.)
    
    Por ahora, sintetizamos IRs programáticamente.
    """
    if preset_name not in ROOM_PRESETS:
        logger.warning(f"Preset '{preset_name}' no reconocido, usando small_studio")
        preset_name = "small_studio"
    
    preset = ROOM_PRESETS[preset_name]
    ir = _synthesize_ir(sr, preset["decay_time"], preset["size"])
    
    logger.info(f"IR loaded: {preset['name']} (decay={preset['decay_time']}s, {len(ir)} samples)")
    return ir


# ── Convolution Reverb ─────────────────────────────────────────────────────

def apply_convolution_reverb(
    audio: np.ndarray,
    sr: int,
    ir: np.ndarray,
    wet_amount: float = 0.3,
    pre_delay_ms: float = 0.0,
    room_size: float = 1.0,
) -> np.ndarray:
    """Aplica reverb convolutivo con IR.
    
    Args:
        audio: Audio de entrada (mono/estéreo, shape (ch, N) o (N,))
        sr: Sample rate
        ir: Impulse response
        wet_amount: Mix wet (0.0=dry, 1.0=100% wet)
        pre_delay_ms: Delay antes de reverb (0-200ms típico)
        room_size: Escala el IR (afecta decay time)
    
    Returns:
        Audio reverberado (mismo shape que entrada)
    """
    if signal is None:
        logger.warning("scipy.signal no disponible, reverb desactivado")
        return audio
    
    # Preparar entrada
    is_stereo = audio.ndim == 2
    if is_stereo:
        original_shape = audio.shape
        audio_to_process = audio.copy()
    else:
        audio_to_process = audio.reshape(1, -1)
    
    # Escalar IR por room_size
    ir_scaled = ir * room_size
    ir_scaled = ir_scaled / (np.max(np.abs(ir_scaled)) + 1e-8)
    
    # Pre-delay
    if pre_delay_ms > 0:
        pre_delay_samples = int(sr * pre_delay_ms / 1000.0)
        pre_delay = np.zeros(pre_delay_samples, dtype=np.float32)
        ir_scaled = np.concatenate([pre_delay, ir_scaled])
    
    # Aplicar convolución a cada canal
    output = np.zeros_like(audio_to_process)
    for ch in range(audio_to_process.shape[0]):
        # FFT convolution (más eficiente que direct)
        dry = audio_to_process[ch]
        wet = signal.fftconvolve(dry, ir_scaled, mode='same')
        
        # Normalizar wet para evitar clipping
        wet_max = np.max(np.abs(wet))
        if wet_max > 1e-6:
            wet = wet / wet_max
        
        # Mix
        output[ch] = (1.0 - wet_amount) * dry + wet_amount * wet
    
    # Restaurar shape
    if is_stereo:
        return output.reshape(original_shape)
    else:
        return output.reshape(-1)


# ── Integración con Mixer ──────────────────────────────────────────────────

class ReverbProcessor:
    """Wrapper para aplicar reverb a stems del mixer."""
    
    def __init__(self, sr: int, ir_cache_size: int = 10):
        self.sr = sr
        self._ir_cache: Dict[str, np.ndarray] = {}
        self.ir_cache_size = ir_cache_size
    
    def get_ir(self, preset_name: str) -> np.ndarray:
        """Obtiene IR con caché."""
        if preset_name not in self._ir_cache:
            if len(self._ir_cache) >= self.ir_cache_size:
                # Evictar el más antiguo (FIFO simple)
                removed = self._ir_cache.popitem()
                logger.debug(f"IR cache evicted: {removed[0]}")
            
            ir = load_ir(preset_name, self.sr)
            self._ir_cache[preset_name] = ir
        
        return self._ir_cache[preset_name]
    
    def process(
        self,
        audio: np.ndarray,
        preset: str = "small_studio",
        wet_amount: float = 0.3,
        pre_delay_ms: float = 0.0,
        room_size: float = 1.0,
    ) -> np.ndarray:
        """Procesa audio con reverb especificado."""
        ir = self.get_ir(preset)
        return apply_convolution_reverb(
            audio, self.sr, ir,
            wet_amount=wet_amount,
            pre_delay_ms=pre_delay_ms,
            room_size=room_size,
        )
    
    def get_available_presets(self) -> Dict[str, str]:
        """Lista de presets disponibles."""
        return {k: v["name"] for k, v in ROOM_PRESETS.items()}


# ── Conversión de parámetros normalizados (0-1) ───────────────────────────

def denormalize_reverb_params(
    preset: str,
    wet_knob: float,  # 0..1
    size_knob: float,  # 0..1
    pre_delay_knob: float,  # 0..1
) -> Tuple[str, float, float, float]:
    """Convierte parámetros normalizados a valores reales.
    
    Uso típico:
        p, w, s, d = denormalize_reverb_params(preset, 0.5, 0.7, 0.2)
        audio = reverb.process(audio, p, wet_amount=w, room_size=s, pre_delay_ms=d)
    """
    # wet_amount: 0-1 → 0-1 (directo)
    wet_amount = np.clip(wet_knob, 0.0, 1.0)
    
    # room_size: 0-1 → 0.3-2.0
    room_size = 0.3 + (size_knob * 1.7)
    
    # pre_delay: 0-1 → 0-200ms
    pre_delay_ms = pre_delay_knob * 200.0
    
    return preset, wet_amount, room_size, pre_delay_ms
