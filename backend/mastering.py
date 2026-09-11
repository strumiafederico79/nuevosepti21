import dataclasses
import logging
import librosa
import soundfile as sf
import numpy as np
import uuid
import os

# FIX (segfault -11 en preview render, ver dmesg): numba (via llvmlite) y
# torch suelen traer CADA UNO su propio runtime de OpenMP embebido. Tener
# los dos cargados en el mismo proceso es una causa muy documentada de
# crash nativo en Linux (símbolos OpenMP duplicados) — coincide exacto con
# los "python3[PID]: segfault at 0 ip 0000000000000000" del dmesg, ya que
# el render corre en un proceso hijo (multiprocessing fork) que importa
# este módulo desde cero cada vez. Esto es una red de seguridad; el fix
# real es el import perezoso de torch más abajo (ver HAS_TORCH).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from typing import Optional
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve, resample_poly, welch, sosfreqz, firwin2, find_peaks, savgol_filter
from scipy.ndimage import maximum_filter1d, median_filter
try:
    from .presets_generator import MASTERING_PRESETS_FULL
except ImportError:
    from presets_generator import MASTERING_PRESETS_FULL

logger = logging.getLogger(__name__)

os.makedirs("processed", exist_ok=True)

# ─── MasteringParams: dataclass canónico de parámetros de la cadena ───────────
# MEJORA: centraliza los ~50 parámetros que antes se repetían literalmente en
# apply_mastering_chain, process_audio, y (los más críticos) en el bloque de
# LUFS safety check (donde duplicar manualmente cada kwarg era la fuente de
# bugs al agregar nuevos parámetros). Ahora todos los caminos usan dataclasses.asdict()
# y el safety check solo actualiza input_gain_db antes de re-renderizar.

@dataclasses.dataclass
class MasteringParams:
    def __post_init__(self) -> None:
        if not self.lp_bypass and self.hp_cutoff >= self.lp_cutoff:
            raise ValueError(
                f"hp_cutoff ({self.hp_cutoff}) debe ser menor que lp_cutoff ({self.lp_cutoff}) "
                "cuando lp_bypass es False."
            )
    # ── Ganancia / loudness ─────────────────────────────────────────────────
    input_gain_db: float = 0.0
    target_peak: float = 0.95        # aceptado pero no usado directamente en la cadena
    use_lufs_normalize: bool = False
    target_lufs: float = -14.0
    # ── Oversampling ────────────────────────────────────────────────────────
    oversample_mode: str = "quality"
    # ── High-pass / Low-pass ─────────────────────────────────────────────────
    hp_cutoff: float = 20.0
    lp_bypass: bool = True
    lp_cutoff: float = 18000.0
    # ── EQ paramétrica: eq1-3 = correctiva/cirugía (stage 3), eq4-6 + high  ──
    # ── shelf + low shelf = tonal/sweetening (stage 6). Ver               ──
    # ── apply_mastering_chain.                                              ──
    eq_mode: str = "iir"
    linear_phase_taps: int = 2049
    high_shelf_gain_db: float = 0.0
    high_shelf_freq_hz: float = 8000.0
    low_shelf_gain_db: float = 0.0
    low_shelf_freq_hz: float = 100.0
    eq1_freq: float = 100.0;  eq1_gain: float = 0.0;  eq1_q: float = 1.0
    eq2_freq: float = 500.0;  eq2_gain: float = 0.0;  eq2_q: float = 1.0
    eq3_freq: float = 2000.0; eq3_gain: float = 0.0;  eq3_q: float = 1.0
    eq4_freq: float = 8000.0; eq4_gain: float = 0.0;  eq4_q: float = 1.0
    eq5_freq: float = 200.0;  eq5_gain: float = 0.0;  eq5_q: float = 1.0
    eq6_freq: float = 1000.0; eq6_gain: float = 0.0;  eq6_q: float = 1.0
    # ── Balance tonal automático (EQ inteligente sin referencia, stage 6) ───
    tonal_balance_bypass: bool = True
    tonal_balance_amount: float = 1.0
    tonal_balance_max_boost_db: float = 3.5
    tonal_balance_max_cut_db: float = -4.5
    tonal_balance_max_bands: int = 6
    # ── EQ Mid/Side (banded, stage 4) ───────────────────────────────────────
    ms_eq_bypass: bool = True
    ms_mid_freq: float = 250.0;  ms_mid_gain: float = 0.0;  ms_mid_q: float = 1.0
    ms_side_freq: float = 8000.0; ms_side_gain: float = 0.0; ms_side_q: float = 1.0
    # ── Compresor Mid/Side (stage 4, después del EQ M/S) ────────────────────
    ms_comp_bypass: bool = True
    ms_comp_mid_threshold_db: float = -18.0; ms_comp_mid_ratio: float = 2.0
    ms_comp_mid_attack_ms: float = 15.0; ms_comp_mid_release_ms: float = 120.0
    ms_comp_mid_makeup_db: float = 0.0
    ms_comp_side_threshold_db: float = -18.0; ms_comp_side_ratio: float = 2.0
    ms_comp_side_attack_ms: float = 15.0; ms_comp_side_release_ms: float = 120.0
    ms_comp_side_makeup_db: float = 0.0
    ms_comp_pdr: bool = True
    ms_comp_pdr_hold_ms: float = 500.0
    # ── Dynamic EQ: resonancias (cirugía, stage 3) ──────────────────────────
    reso_bypass: bool = True
    reso_freq: float = 1200.0
    reso_q: float = 3.0
    reso_threshold_db: float = -18.0
    reso_ratio: float = 3.0
    reso_attack_ms: float = 5.0
    reso_release_ms: float = 100.0
    reso_max_reduction_db: float = 8.0
    # ── Dynamic EQ: de-esser dedicado (stage 7) ─────────────────────────────
    dyneq_bypass: bool = True
    dyneq_freq: float = 3000.0
    dyneq_q: float = 2.5
    dyneq_threshold_db: float = -18.0
    dyneq_ratio: float = 3.0
    dyneq_attack_ms: float = 3.0
    dyneq_release_ms: float = 80.0
    dyneq_max_reduction_db: float = 12.0
    # ── Transient shaper ────────────────────────────────────────────────────
    transient_attack: float = 0.0
    transient_sustain: float = 0.0
    # ── Compresor multibanda ─────────────────────────────────────────────────
    mb_bypass: bool = False
    mb_low_crossover: float = 250.0
    mb_high_crossover: float = 4000.0
    mb_low_threshold_db: float = -18.0; mb_low_ratio: float = 2.0
    mb_low_attack_ms: float = 20.0; mb_low_release_ms: float = 150.0; mb_low_makeup_db: float = 0.0
    mb_mid_threshold_db: float = -18.0; mb_mid_ratio: float = 2.0
    mb_mid_attack_ms: float = 20.0; mb_mid_release_ms: float = 150.0; mb_mid_makeup_db: float = 0.0
    mb_high_threshold_db: float = -18.0; mb_high_ratio: float = 2.0
    mb_high_attack_ms: float = 20.0; mb_high_release_ms: float = 150.0; mb_high_makeup_db: float = 0.0
    mb_pdr: bool = True
    mb_pdr_hold_ms: float = 500.0
    # ── Compresor de banda ancha ─────────────────────────────────────────────
    comp_stereo_link: bool = True
    comp_bypass: bool = False
    comp_threshold_db: float = -18.0
    comp_ratio: float = 4.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 100.0
    comp_makeup_db: float = 0.0
    # PDR (Program-Dependent Release): reemplaza el release_ms fijo por uno
    # que se adapta al programa (más rápido en transientes aislados, más
    # lento en pasajes sostenidos). Ver _smooth_envelope_pdr_numba.
    comp_pdr: bool = True
    comp_pdr_hold_ms: float = 500.0
    # ── Glue compressor ──────────────────────────────────────────────────────
    glue_bypass: bool = True
    glue_threshold_db: float = -4.0
    glue_ratio: float = 2.0
    glue_attack_ms: float = 30.0
    glue_release_ms: float = 120.0
    glue_makeup_db: float = 0.0
    glue_pdr: bool = True
    glue_pdr_hold_ms: float = 500.0
    # ── Saturación armónica ──────────────────────────────────────────────────
    saturation_drive: float = 0.0
    saturation_mode: str = "tape"
    saturation_mix: float = 1.0
    # ── Imagen estéreo ───────────────────────────────────────────────────────
    mid_gain_db: float = 0.0
    side_gain_db: float = 0.0
    stereo_width_amount: float = 1.0
    stereo_bypass: bool = False
    use_stereo_enhancer: bool = False
    enhancer_bass_mono_freq: float = 120.0
    haas_delay_ms: float = 0.0
    low_end_mono_freq: float = 120.0
    low_end_mono_amount: float = 0.0
    mb_stereo_bypass: bool = True
    mb_stereo_low_width: float = 0.9
    mb_stereo_mid_width: float = 1.2
    mb_stereo_high_width: float = 1.5
    mb_stereo_low_crossover: float = 150.0
    mb_stereo_high_crossover: float = 4000.0
    # ── Reverb ───────────────────────────────────────────────────────────────
    reverb_size: float = 0.05
    reverb_wet: float = 0.0
    # ── Compresión paralela (stage 5.5): mezcla dry/wet para control tonal
    parallel_bypass: bool = True
    parallel_threshold_db: float = -12.0
    parallel_ratio: float = 4.0
    parallel_attack_ms: float = 10.0
    parallel_release_ms: float = 100.0
    parallel_mix: float = 0.0  # 0 = 100% dry, 1 = 100% wet (procesado)
    # ── Clipper (pico suave/hard, previo al limitador) ──────────────────────
    clipper_bypass: bool = True
    clipper_mode: str = "soft"       # "soft" (tanh) | "hard"
    clipper_ceiling: float = 0.98
    clipper_drive_db: float = 0.0
    # ── Limitador ────────────────────────────────────────────────────────────
    limiter_ceiling: float = 0.95
    limiter_bypass: bool = False
    limiter_release_ms: float = 80.0
    # ── Reducción de ruido ───────────────────────────────────────────────────
    nr_bypass: bool = True
    nr_strength: float = 0.5
    nr_noise_sample_sec: float = 0.5

    def as_chain_kwargs(self) -> dict:
        """Devuelve todos los campos relevantes para apply_mastering_chain como dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_preset(cls, preset: dict) -> "MasteringParams":
        """Construye un MasteringParams a partir de un preset (ignora claves desconocidas)."""
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in preset.items() if k in valid})

def _report(cb, pct: float, stage: str) -> None:
    """Llama a `cb(pct, stage)` sin romper el procesamiento si el callback
    falla (por ejemplo si el job ya no existe o el dict fue limpiado)."""
    if cb is None:
        return
    try:
        cb(min(100, max(0, round(pct))), stage)
    except Exception:
        pass

DEFAULT_DSP_OVERSAMPLE = 4
OVERSAMPLING_MODES = {"off": 1, "draft": 1, "fast": 2, "quality": 4, "ultra": 8}

def resolve_oversample(mode: str | int | None = "quality") -> int:
    """Resolve oversampling quality names to integer factors."""
    if mode is None:
        return DEFAULT_DSP_OVERSAMPLE
    if isinstance(mode, (int, np.integer)):
        return max(1, int(mode))
    return OVERSAMPLING_MODES.get(str(mode).lower(), DEFAULT_DSP_OVERSAMPLE)


# ─── Numba acceleration ────────────────────────────────────────────────────────
try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ─── Torch (opcional, solo para compute_reference_eq_curve_ddsp) ──────────────
# FIX (segfault -11 en preview render, ver dmesg): antes se hacía
# `import torch` acá arriba, a nivel de módulo — es decir, en TODOS los
# renders, no solo cuando hay reference-matching (única función que usa
# torch). Como cada render corre en un proceso nuevo (multiprocessing
# fork en preview_service.py), esto cargaba numba+torch juntos en
# CADA preview, aunque no hiciera falta. find_spec() solo chequea que el
# paquete exista, sin importarlo/cargar sus .so — el import real queda
# perezoso, adentro de compute_reference_eq_curve_ddsp.
import importlib.util
HAS_TORCH = importlib.util.find_spec("torch") is not None

# ─── Helpers ───────────────────────────────────────────────────────────────────

def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    return audio

def _crop_preview(audio: np.ndarray, sr: int, preview_seconds: float) -> np.ndarray:
    """Recorta el preview desde el segundo 15 durante `preview_seconds`."""
    total_samples = audio.shape[1]
    max_samples = min(int(preview_seconds * sr), total_samples)
    if max_samples <= 0:
        return audio
    start = min(int(15 * sr), max(0, total_samples - max_samples))
    return audio[:, start:start + max_samples]

# ─── Bit depth de salida + dithering ───────────────────────────────────────────
# Antes, sf.write() se llamaba sin `subtype`, así que soundfile caía en su
# default para WAV: PCM_16, truncando el float64 interno a 16 bits SIN
# dithering (libsndfile no ditherea por su cuenta). Truncar sin dither mete
# distorsión de cuantización correlacionada con la señal (audible como
# aspereza/artefactos en fades y pasajes de bajo nivel) en vez de un piso de
# ruido no correlacionado — es la diferencia entre un master "de estudio" y
# uno "de demo". Ahora la salida por defecto es 24-bit (headroom de sobra,
# nadie escucha el piso de cuantización a -144 dBFS) y, si el usuario elige
# igual bajar a 16-bit, se aplica dither TPDF antes de truncar.

OUTPUT_BIT_DEPTHS = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",   # 32-bit float: sin cuantización real, no necesita dither
}
DEFAULT_OUTPUT_BIT_DEPTH = 24

def resolve_bit_depth(bit_depth) -> int:
    try:
        bd = int(bit_depth)
    except (TypeError, ValueError):
        return DEFAULT_OUTPUT_BIT_DEPTH
    return bd if bd in OUTPUT_BIT_DEPTHS else DEFAULT_OUTPUT_BIT_DEPTH

def _tpdf_dither(audio: np.ndarray, bit_depth: int, seed: int = None) -> np.ndarray:
    """Dither TPDF (Triangular Probability Density Function), el estándar de
    facto en mastering para cuantizar de punto flotante a un bit depth entero
    (AES17 / práctica habitual de Sound Forge, RX, Ozone, etc.).

    Sumar dos variables uniformes independientes de amplitud ±0.5 LSB da una
    distribución triangular de ±1 LSB pico a pico. Esto decorrelaciona
    completamente el error de cuantización de la señal (a diferencia de
    rectangular dither, que todavía deja algo de correlación en señales de
    bajo nivel), a costa de un pelín más de piso de ruido — inaudible en la
    práctica e imperceptible frente al ruido térmico de cualquier conversor
    real.

    No hace nada para bit_depth >= 32 (punto flotante: no hay cuantización
    real que dithering).
    """
    if bit_depth >= 32:
        return audio
    rng = np.random.default_rng(seed)
    lsb = 2.0 / (2 ** bit_depth)  # rango [-1, 1] → 2.0 de span total
    noise = (rng.uniform(-0.5, 0.5, size=audio.shape) +
             rng.uniform(-0.5, 0.5, size=audio.shape)) * lsb
    return np.clip(audio + noise, -1.0, 1.0)


# ─── Noise shaping + dither adaptativo ────────────────────────────────────────
#
# El TPDF es flat: distribuye el ruido de cuantización uniformemente en todo
# el espectro audible. Eso es correcto técnicamente pero subóptimo
# perceptualmente: el oído humano NO es igualmente sensible en todas las
# frecuencias. El noise shaping empuja la energía del error de cuantización
# hacia las zonas donde el oído es MENOS sensible (principalmente > 10 kHz y
# < 100 Hz), reduciendo la audibilidad del ruido sin cambiar su potencia total.
#
# Implementación: error-feedback de primer orden. En cada muestra:
#   1. Se genera el dither TPDF (decorrelación).
#   2. Se cuantiza (round al LSB más cercano).
#   3. Se computa el error de cuantización (cuantizado - continuo).
#   4. Ese error se filtra con un filtro FIR de noise shaping h[] y se
#      RESTA de la muestra siguiente antes de volver a cuantizar —
#      equivalente a empujar el error hacia las frecuencias donde h[] tiene
#      mayor ganancia, que diseñamos para que coincidan con las zonas de
#      menor sensibilidad auditiva.
#
# Modos disponibles:
#
#   "tpdf"       — TPDF flat sin shaping. Mismo resultado que _tpdf_dither().
#                  Baseline; el único correcto para MP3 y formatos lossy.
#
#   "high_shelf" — Shaping simple de primer orden con un coeficiente único
#                  (h = [-0.5]): empuja la mitad del error de cuantización a
#                  las frecuencias altas. Costo computacional mínimo, mejora
#                  audible clara en material de poco nivel (ambient, clásica,
#                  jazz) entregado a 16-bit. Similar al POW-r 1 de Waves.
#
#   "f_weighted" — Filtro FIR de 9 coeficientes diseñado para seguir
#                  aproximadamente la curva de igual-loudness ISO 226 invertida
#                  a 20 phon: máxima supresión en la zona 2-4 kHz (más
#                  sensible) y máxima tolerancia en > 12 kHz y < 100 Hz.
#                  Audiblemente más silencioso que TPDF en señales de bajo
#                  nivel a 16-bit. Similar al POW-r 3 / Apogee UV22HR en
#                  concepto (aunque los coeficientes exactos de esos son
#                  propietarios). Calcula el feedback por canal con un loop
#                  explícito — más lento que high_shelf pero sigue siendo
#                  negligible frente al tiempo de render total.

# Coeficientes del filtro FIR de noise shaping f_weighted.
# Diseñados para aproximar la sensibilidad auditiva ISO 226 @20phon invertida:
# atenuación máxima en 2-4 kHz (donde el oído es más sensible) y ganancia en
# > 12 kHz y < 80 Hz (donde el oído tolera más ruido).
# Se derivan de un diseño parksmc-mcclellan simplificado con las siguientes
# especificaciones en frecuencia (normalizadas, sr=44100):
#   f=0      → 0 dB  (DC: sin realce)
#   f=0.05   → -6 dB (~2.2 kHz: zona más sensible, suprimir)
#   f=0.12   → -4 dB (~5.3 kHz: moderadamente sensible)
#   f=0.25   → +3 dB (~11 kHz: zona de menor sensibilidad)
#   f=0.45   → +6 dB (~20 kHz: oído casi sordo, empaquetar el error acá)
#   f=0.5    → +6 dB (Nyquist)
# Resultado: el espectro del ruido de cuantización tiene un "pozo" en 2-4 kHz
# y un "pico" en > 10 kHz — exactamente lo que maximiza la audibilidad de la
# señal vs. el ruido al reconvertir (efecto análogo al énfasis de cassette).
_NS_COEFFS_F_WEIGHTED = np.array([
    2.033,  -2.165,  1.959, -1.590,  1.161,
   -0.736,   0.387, -0.165,  0.046,
], dtype=np.float64)

# Coeficiente único para high_shelf (primer orden): h[0] = -0.5
# Esto es equivalente a un filtro paso-alto de primer orden en el camino
# del error: empuja el 50% del error de cuantización a la muestra siguiente
# con signo invertido, lo que crea una respuesta espectral del ruido que sube
# 6 dB/octava desde baja a alta frecuencia (high-shelf de +6 dB a Nyquist).
_NS_COEFF_HIGH_SHELF = np.array([-0.5], dtype=np.float64)


def _noise_shaped_dither(audio: np.ndarray, sr: int, bit_depth: int,
                          mode: str = "f_weighted", seed: int = None) -> np.ndarray:
    """Dither con noise shaping adaptativo.

    Parámetros:
        audio     — float32/float64, rango [-1, 1], mono (1D) o estéreo (2D).
        sr        — sample rate en Hz. Usado para escalar correctamente los
                    coeficientes del filtro si sr != 44100 (high_shelf y
                    f_weighted están diseñados para 44100).
        bit_depth — profundidad de bits de salida (16 o 24). Para 24-bit el
                    beneficio del shaping es marginal (el piso de ruido ya
                    está a -144 dBFS); para 16-bit es donde hace la diferencia.
        mode      — "tpdf" | "high_shelf" | "f_weighted"
        seed      — semilla para reproducibilidad (tests).

    Devuelve array del mismo shape que audio, clipeado a [-1, 1], listo para
    escribir con sf.write(..., subtype="PCM_16" / "PCM_24").
    """
    if bit_depth >= 32:
        return audio  # float: sin cuantización, shaping no aplica

    if mode == "tpdf":
        return _tpdf_dither(audio, bit_depth, seed=seed)

    lsb = 2.0 / (2 ** bit_depth)
    rng = np.random.default_rng(seed)

    # Seleccionar coeficientes según modo
    if mode == "high_shelf":
        coeffs = _NS_COEFF_HIGH_SHELF.copy()
    else:  # f_weighted (default)
        coeffs = _NS_COEFFS_F_WEIGHTED.copy()
        # Si el sr no es 44100, los coeficientes diseñados a 44100 van a
        # tener su respuesta en frecuencia escalada. Para sr alto (88/96k)
        # el filtro empuja el ruido más arriba todavía — aceptable. Para
        # sr bajo (22050, 16000) el shaping puede caer en zona audible:
        # fallback seguro a high_shelf.
        if sr < 32000:
            coeffs = _NS_COEFF_HIGH_SHELF.copy()

    order = len(coeffs)

    def _shape_channel(ch: np.ndarray) -> np.ndarray:
        ch = ch.astype(np.float64)
        n = len(ch)
        # PERF: el ruido TPDF se genera vectorizado de una sola vez (antes
        # eran 2 llamadas a rng.uniform() ESCALARES por muestra dentro del
        # loop — mismo problema de overhead que el resto de la función).
        tpdf_noise = (rng.uniform(-0.5, 0.5, size=n) + rng.uniform(-0.5, 0.5, size=n)) * lsb
        if HAS_NUMBA:
            return _noise_shape_filter_numba(ch, coeffs, lsb, tpdf_noise)
        return _shape_channel_fallback(ch, coeffs, lsb, tpdf_noise)

    if audio.ndim == 1:
        return _shape_channel(audio).astype(np.float32)
    else:
        channels = [_shape_channel(ch) for ch in audio]
        return np.stack(channels).astype(np.float32)


def _dither_meta(mode: str, bit_depth: int, sr: int) -> dict:
    """Metadatos del dither aplicado, para incluir en la respuesta de la API."""
    if bit_depth >= 32:
        return {"applied": False, "reason": "float32/float64: no se cuantiza"}
    labels = {
        "tpdf":        "TPDF flat (estándar AES17)",
        "high_shelf":  "TPDF + noise shaping high-shelf (1er orden, +6dB/oct a Nyquist)",
        "f_weighted":  "TPDF + noise shaping F-weighted (ISO 226, 9 coeficientes)",
    }
    return {
        "applied":    True,
        "mode":       mode,
        "label":      labels.get(mode, mode),
        "bit_depth":  bit_depth,
        "sr":         sr,
        "lsb_dbfs":   round(-20.0 * np.log10(2 ** bit_depth / 2.0), 1),
    }

# ── Noise shaping filter — kernel de la muestra-a-muestra (Numba JIT) ─────────
# PERF: antes esto vivía como método interno `_shape_channel()` dentro de
# `_noise_shaped_dither()`, con un `for i in range(n)` en Python puro que en
# cada muestra llamaba np.dot(coeffs, err_buf) + np.roll(err_buf, 1) (esto
# último REALOCA memoria en cada muestra). Para una canción estéreo de 4min
# a 44.1kHz son ~21M iteraciones × 4-5 llamadas a NumPy cada una — el
# overhead de esas llamadas (varios µs cada una) dominaba por completo sobre
# el cálculo real, y explicaba el job quedándose colgado en "Guardando
# archivo masterizado" (97%) durante minutos. No es un problema de núcleos:
# es un loop secuencial (cada muestra depende del error de la anterior, por
# el feedback del noise shaping) que nunca iba a paralelizarse ni con 64
# cores — lo que hacía falta era sacarlo de Python puro, no repartirlo.
# Mismo patrón que _smooth_envelope_pdr_numba: loop compilado + buffer
# circular manual (sin reallocar) en vez de np.roll.
if HAS_NUMBA:
    @nb.jit(nopython=True, cache=False, fastmath=True, nogil=True)
    def _noise_shape_filter_numba(ch: np.ndarray, coeffs: np.ndarray,
                                  lsb: float, tpdf_noise: np.ndarray) -> np.ndarray:
        n = ch.shape[0]
        order = coeffs.shape[0]
        out = np.empty(n, dtype=np.float64)
        err_buf = np.zeros(order, dtype=np.float64)
        for i in range(n):
            fb = 0.0
            for j in range(order):
                fb += coeffs[j] * err_buf[j]
            shaped_sample = ch[i] + tpdf_noise[i] - fb

            quantized = round(shaped_sample / lsb) * lsb  # round-half-to-even, igual que np.round
            if quantized > 1.0:
                quantized = 1.0
            elif quantized < -1.0:
                quantized = -1.0
            out[i] = quantized

            error = quantized - shaped_sample
            # Shift manual del buffer circular (order típicamente 9 elementos:
            # es más barato que np.roll porque no reasigna memoria).
            for j in range(order - 1, 0, -1):
                err_buf[j] = err_buf[j - 1]
            err_buf[0] = error
        return out


def _shape_channel_fallback(ch: np.ndarray, coeffs: np.ndarray, lsb: float,
                            tpdf_noise: np.ndarray) -> np.ndarray:
    """Fallback puro-Python si numba no está disponible (no debería pasar en
    producción — está en requirements.txt — pero por las dudas). Sigue siendo
    O(n) secuencial por la naturaleza del filtro con feedback, pero evita el
    overhead de llamar np.dot/np.roll/np.clip por muestra: usa listas de
    Python planas y aritmética escalar, que es sensiblemente más rápido que
    repetir llamadas a NumPy sobre arrays de 9 elementos veinte millones de
    veces.
    """
    n = len(ch)
    order = len(coeffs)
    coeffs_list = [float(c) for c in coeffs]
    out = [0.0] * n
    err_buf = [0.0] * order
    ch_list = ch.tolist()
    noise_list = tpdf_noise.tolist()

    for i in range(n):
        fb = 0.0
        for j in range(order):
            fb += coeffs_list[j] * err_buf[j]
        shaped_sample = ch_list[i] + noise_list[i] - fb

        quantized = round(shaped_sample / lsb) * lsb
        if quantized > 1.0:
            quantized = 1.0
        elif quantized < -1.0:
            quantized = -1.0
        out[i] = quantized

        error = quantized - shaped_sample
        for j in range(order - 1, 0, -1):
            err_buf[j] = err_buf[j - 1]
        err_buf[0] = error

    return np.array(out, dtype=np.float64)


def _write_master_output(audio_out: np.ndarray, sr: int, output_path: str,
                         output_format: str, output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
                         dither_seed: int = None,
                         dither_mode: str = "f_weighted") -> tuple:
    """Escribe el archivo final aplicando dither con noise shaping si corresponde.
    Devuelve (bit_depth_efectivo, dither_meta_dict).

    dither_mode: "tpdf" | "high_shelf" | "f_weighted" (default).
    Para MP3 siempre usa el intermedio 24-bit sin dither (el encoder lossy
    vuelve a cuantizar, ditherear antes solo sumaría ruido extra).
    """
    # BUGFIX: internamente el audio viaja channels-first, (n_channels,
    # n_samples) — así lo espera _noise_shaped_dither, que itera "for ch in
    # audio" asumiendo que cada fila es un canal. soundfile.write(), en
    # cambio, espera frames-first: (n_samples, n_channels). Sin transponer,
    # para estéreo sf.write() recibía un array (2, N) y lo interpretaba como
    # N canales (un número absurdo), combinación que libsndfile rechaza
    # como formato inválido → LibsndfileError("Format not recognised").
    # Esto también explica por qué el reintento con PCM_16 fallaba
    # idéntico: el problema nunca fue el subtype, era el shape.
    def _sf_shape(arr: np.ndarray) -> np.ndarray:
        return arr.T if arr.ndim == 2 else arr

    bit_depth = resolve_bit_depth(output_bit_depth)
    # BUGFIX: FLAC/OGG no soportan subtype FLOAT (32-bit) en libsndfile —
    # sf.write() falla con LibsndfileError("Format not recognised") sin
    # mensaje claro. FLAC máximo real es 24-bit PCM; para esos formatos se
    # topea el bit depth antes de ditherear (así el dither ya cuantiza al
    # bit depth real que se va a escribir, no a uno que después se trunca).
    ext_for_clamp = os.path.splitext(output_path)[1].lower()
    if ext_for_clamp in (".flac", ".ogg", ".aiff", ".aif") and bit_depth >= 32:
        bit_depth = 24
    subtype = OUTPUT_BIT_DEPTHS[bit_depth]
    dithered = _noise_shaped_dither(audio_out, sr, bit_depth,
                                    mode=dither_mode, seed=dither_seed)
    meta = _dither_meta(dither_mode, bit_depth, sr)

    if output_format == "mp3":
        # ffmpeg directo: pipe numpy → stdin de ffmpeg → MP3 sin WAV intermedio.
        # Evita el doble I/O en disco (sf.write tmp_wav + pydub read) y el
        # overhead de AudioSegment. En tracks de 4-5 min el ahorro es ~2-4s.
        import subprocess
        # Convertir a int16 PCM para el pipe (ffmpeg espera s16le)
        pcm = (np.clip(audio_out, -1.0, 1.0) * 32767).astype(np.int16)
        # Si es stereo (2, N) → intercalar canales → (N, 2) → flatten
        if pcm.ndim == 2:
            pcm = pcm.T.copy()  # (N, 2)
        pcm_bytes = pcm.tobytes()
        n_channels = 2 if audio_out.ndim == 2 else 1
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(sr),
            "-ac", str(n_channels),
            "-i", "pipe:0",
            "-codec:a", "libmp3lame",
            "-b:a", "320k",
            "-id3v2_version", "3",
            output_path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=pcm_bytes,
                capture_output=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {proc.stderr.decode(errors='replace')}")
        except FileNotFoundError:
            # ffmpeg no está instalado — fallback a pydub si está disponible
            tmp_wav = output_path.replace(".mp3", "_tmp.wav")
            sf.write(tmp_wav, _sf_shape(audio_out), sr, subtype="PCM_24")
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_wav(tmp_wav)
                seg.export(output_path, format="mp3", bitrate="320k")
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
        return bit_depth, {**meta, "applied": False, "reason": "MP3: encoder lossy re-cuantiza, dither previo no aporta"}

    # BUGFIX: especificar explícitamente el formato evita errores
    # 'Format not recognised' cuando libsndfile no puede inferirlo
    # correctamente desde la extensión o el nombre del archivo.
    ext = os.path.splitext(output_path)[1].lower()
    format_map = {
        ".wav": "WAV",
        ".flac": "FLAC",
        ".ogg": "OGG",
        ".aiff": "AIFF",
        ".aif": "AIFF",
    }
    sf_format = format_map.get(ext)

    try:
        if sf_format:
            sf.write(output_path, _sf_shape(dithered), sr, format=sf_format, subtype=subtype)
        else:
            sf.write(output_path, _sf_shape(dithered), sr, subtype=subtype)
    except Exception:
        # BUGFIX: red de seguridad — si el subtype igual no es válido para
        # el formato (combinación no contemplada arriba), reintentar una
        # sola vez con PCM_16, el subtype más compatible de todos, en vez
        # de que el job entero termine en error sin archivo de salida.
        bit_depth = 16
        subtype = OUTPUT_BIT_DEPTHS[bit_depth]
        dithered = _noise_shaped_dither(audio_out, sr, bit_depth,
                                        mode=dither_mode, seed=dither_seed)
        meta = _dither_meta(dither_mode, bit_depth, sr)
        if sf_format:
            sf.write(output_path, _sf_shape(dithered), sr, format=sf_format, subtype=subtype)
        else:
            sf.write(output_path, _sf_shape(dithered), sr, subtype=subtype)

    return bit_depth, meta

# ── Envelope follower (Numba JIT) ────────────────────────────────────────────────
if HAS_NUMBA:
    @nb.jit(nopython=True, cache=False, fastmath=True, nogil=True)
    def _smooth_envelope_numba(signal: np.ndarray, attack_coef: float, release_coef: float) -> np.ndarray:
        n = len(signal)
        env = np.empty(n, dtype=np.float64)
        prev = 0.0
        for i in range(n):
            x = signal[i]
            coef = attack_coef if x > prev else release_coef
            prev = coef * prev + (1.0 - coef) * x
            env[i] = prev
        return env

    @nb.jit(nopython=True, cache=False, fastmath=True, nogil=True)
    def _smooth_envelope_pdr_numba(signal: np.ndarray, attack_coef: float,
                                   release_fast_coef: float, release_slow_coef: float,
                                   hold_coef: float) -> np.ndarray:
        # Program-Dependent Release (PDR): en vez de un único release_coef fijo,
        # se interpola cuadro a cuadro entre una constante de release RÁPIDA
        # (release_ms/3) y una LENTA (release_ms*3) según un integrador de
        # "historia" (`hist`) que sube mientras la señal está en attack
        # (sostenida por encima de la envolvente) y decae lentamente cuando
        # libera. Así, transientes cortos sueltan rápido (sin bombeo audible)
        # y pasajes sostenidos/fuertes sueltan lento (más "pegado"/musical,
        # como el auto-release de un SSL G-series o un dbx 160) — el mismo
        # release_ms nominal se comporta distinto según el programa musical.
        n = len(signal)
        env = np.empty(n, dtype=np.float64)
        prev = 0.0
        hist = 0.0
        for i in range(n):
            x = signal[i]
            if x > prev:
                coef = attack_coef
                hist = hist * hold_coef + (1.0 - hold_coef)
            else:
                coef = release_fast_coef + (release_slow_coef - release_fast_coef) * hist
                hist = hist * hold_coef
            prev = coef * prev + (1.0 - coef) * x
            env[i] = prev
        return env

    @nb.jit(nopython=True, cache=False, fastmath=True, nogil=True)
    def _compute_gain_reduction_numba(env_db: np.ndarray, threshold_db: float, ratio: float,
                                      knee_db: float = 6.0) -> np.ndarray:
        # MEJORA: codo suave (soft-knee) en vez de codo duro. Con codo duro la
        # compresión arranca de golpe apenas se cruza el umbral (quiebre de
        # pendiente instantáneo en la curva de transferencia), lo que en
        # señales musicales se percibe como una compresión más brusca/audible
        # de lo necesario. La fórmula estándar (Zölzer/Giannoulis) hace una
        # transición cuadrática en una ventana de `knee_db` alrededor del
        # umbral, para que la ganancia entre suavemente. Fuera de esa ventana
        # el resultado es idéntico al codo duro de antes.
        n = len(env_db)
        gr = np.zeros(n, dtype=np.float64)
        half_knee = knee_db / 2.0
        factor = (1.0 / ratio) - 1.0
        for i in range(n):
            over = env_db[i] - threshold_db
            if knee_db > 0.0 and -half_knee < over < half_knee:
                gr[i] = factor * (over + half_knee) ** 2 / (2.0 * knee_db)
            elif over >= half_knee:
                gr[i] = factor * over
        return gr

    @nb.jit(nopython=True, cache=False, fastmath=True, nogil=True)
    def _limiter_gain_numba(instant_gain: np.ndarray, release_coef: float) -> np.ndarray:
        n = len(instant_gain)
        smoothed = np.empty(n, dtype=np.float64)
        prev = 1.0
        for i in range(n):
            g = instant_gain[i]
            if g < prev:
                prev = g
            else:
                prev = release_coef * prev + (1.0 - release_coef) * g
            smoothed[i] = prev
        return smoothed

def _apply_lookahead_shift(gr: np.ndarray, look_samples: int) -> np.ndarray:
    """Adelanta la curva de ganancia `look_samples` muestras.

    Como el render acá es OFFLINE (no realtime — ya tenemos el buffer
    completo en memoria), este "lookahead" no cuesta absolutamente nada de
    latencia real, a diferencia de un compresor de hardware/plugin en vivo
    donde el lookahead implica retrasar la señal de audio. Simplemente
    tomamos la curva de reducción ya calculada (causal) y la corremos hacia
    atrás en el tiempo: el compresor "sabe" que viene un transiente un par
    de ms antes de que llegue, en vez de empezar a reaccionar recién
    cuando el pico ya está sonando. Esto elimina el overshoot típico de un
    detector puramente causal (attack finito = siempre deja pasar algo del
    ataque del transiente sin atenuar antes de que el envelope llegue).

    BUGFIX: el tail padding era gr[-1] (último valor de la curva).
    Si el compresor terminaba con reducción activa (release aún en curso),
    gr[-1] era negativo → se extendía esa reducción más allá del final del
    audio. El tail debe ser 0.0 dB (sin reducción) porque el lookahead
    solo anticipa lo que YA EXISTE en el buffer: más allá de la última
    muestra no hay señal, así que la curva debe volver a cero.
    """
    if look_samples <= 0 or len(gr) <= look_samples:
        return gr
    # Tail = 0.0 dB (sin reducción) — no hay audio más allá del final del buffer
    tail = np.zeros(look_samples, dtype=gr.dtype)
    return np.concatenate([gr[look_samples:], tail])

def _rms_envelope_prefilter(abs_signal: np.ndarray, sr: int, window_ms: float) -> np.ndarray:
    """Suaviza la señal que entra al detector con una ventana RMS corta
    ANTES del seguidor attack/release, emulando el comportamiento
    "promediador" de un detector RMS analógico (ej. el bus comp de una
    SSL) en vez de un detector de pico puro muestra a muestra.

    window_ms=0 desactiva esto y el detector queda igual que antes (peak
    puro). La ventana es una media móvil causal sobre la señal al cuadrado
    (energía) seguida de raíz cuadrada — no mira el futuro, así que sigue
    siendo compatible con el lookahead explícito que se aplica después.
    """
    if window_ms <= 0.0:
        return abs_signal
    win = max(1, int(round(sr * window_ms / 1000.0)))
    if win <= 1:
        return abs_signal
    power = abs_signal.astype(np.float64, copy=False) ** 2
    kernel = np.ones(win, dtype=np.float64) / win
    smoothed_power = np.convolve(power, kernel, mode="full")[: len(power)]
    return np.sqrt(np.maximum(smoothed_power, 0.0))

def _soft_knee_gain_reduction_np(env_db: np.ndarray, threshold_db: float, ratio: float,
                                 knee_db: float = 6.0) -> np.ndarray:
    """Versión numpy (fallback sin numba) del soft-knee de _compute_gain_reduction_numba.

    BUGFIX: la versión anterior tenía dos errores en la zona de knee:
    1) `np.where(over >= half_knee, hard, np.where(over > -half_knee, knee, 0.0))`
       evalúa AMBAS ramas antes de seleccionar: cuando `knee_db=0` y la rama
       `knee` intenta dividir por cero, numpy lo silencia con un warning pero
       el resultado puede ser nan/inf si el compilador no hace short-circuit.
    2) La fórmula de knee `factor * (over + half_knee)^2 / (2*knee_db)` es
       la del GC (gain computer), no del GR. El GR de soft knee correcto es:
         GR = (1/ratio - 1) * (over + knee/2)^2 / (2*knee)
       que para `over = half_knee` da `factor * knee_db / 2` (máxima reducción
       en el borde superior del knee), y para `over = -half_knee` da 0.
       La versión anterior usaba `factor * (over + half_knee)^2 / (2*knee_db)`
       que es equivalente solo si knee_db es constante y el signo de `factor`
       es consistente — era correcto matemáticamente pero aplicaba reducción
       NO-CERO cuando `over` era levemente positivo y el knee era ancho (6 dB),
       porque `over + half_knee > 0` incluso cuando `over < 0`.

    Fórmula correcta (idéntica a la que usa el compresor principal en numba):
      below knee  (over <= -half_knee): GR = 0
      in knee     (-half_knee < over < half_knee): GR = factor*(over+half_knee)^2 / (2*knee_db)
      above knee  (over >= half_knee): GR = factor * over
    donde factor = (1/ratio - 1) < 0 → GR siempre ≤ 0 (solo reducción).
    """
    over = env_db - threshold_db
    half_knee = knee_db / 2.0
    factor = (1.0 / ratio) - 1.0  # siempre negativo para ratio > 1

    above = factor * over
    if knee_db > 0.0:
        in_knee = factor * ((over + half_knee) ** 2) / (2.0 * knee_db)
    else:
        in_knee = above  # knee_db=0 → hard knee, misma rama que above

    gr = np.where(
        over >= half_knee,
        above,
        np.where(over > -half_knee, in_knee, 0.0)
    )
    return gr

def _smooth_envelope(signal: np.ndarray, sr: int, attack_ms: float, release_ms: float,
                     program_dependent: bool = False, pdr_hold_ms: float = 500.0) -> np.ndarray:
    """Seguidor de envolvente attack/release (detector del compresor).

    program_dependent=True activa Program-Dependent Release (PDR): en vez de
    un release_ms fijo, el release efectivo varía cuadro a cuadro entre
    release_ms/3 (transiente corto) y release_ms*3 (pasaje sostenido), según
    cuánto tiempo lleva la señal "enganchada" por encima de la envolvente
    (ver _smooth_envelope_pdr_numba / fallback numpy más abajo). pdr_hold_ms
    controla qué tan rápido sube/baja ese integrador de historia (más alto =
    memoria más larga, transición más gradual entre rápido y lento).
    """
    attack_coef  = np.exp(-1.0 / (sr * (attack_ms  / 1000.0) + 1e-9))
    sig64 = signal.astype(np.float64, copy=False)

    if program_dependent:
        release_fast_coef = np.exp(-1.0 / (sr * (max(release_ms / 3.0, 1.0) / 1000.0) + 1e-9))
        release_slow_coef = np.exp(-1.0 / (sr * (release_ms * 3.0 / 1000.0) + 1e-9))
        hold_coef = np.exp(-1.0 / (sr * (pdr_hold_ms / 1000.0) + 1e-9))
        if HAS_NUMBA:
            return _smooth_envelope_pdr_numba(sig64, attack_coef, release_fast_coef, release_slow_coef, hold_coef)
        n = len(sig64)
        env = np.empty(n, dtype=np.float64)
        prev = 0.0
        hist = 0.0
        for i in range(n):
            x = sig64[i]
            if x > prev:
                coef = attack_coef
                hist = hist * hold_coef + (1.0 - hold_coef)
            else:
                coef = release_fast_coef + (release_slow_coef - release_fast_coef) * hist
                hist = hist * hold_coef
            prev = coef * prev + (1.0 - coef) * x
            env[i] = prev
        return env

    release_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9))
    if HAS_NUMBA:
        return _smooth_envelope_numba(sig64, attack_coef, release_coef)
    n = len(sig64)
    env = np.empty(n, dtype=np.float64)
    prev = 0.0
    for i in range(n):
        x = sig64[i]
        coef = attack_coef if x > prev else release_coef
        prev = coef * prev + (1.0 - coef) * x
        env[i] = prev
    return env

# ─── DSP primitives ────────────────────────────────────────────────────────────

def measure_lufs_integrated(audio: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln
        data = audio.T if audio.ndim == 2 else audio
        meter = pyln.Meter(sr)
        val = meter.integrated_loudness(data)
        if np.isfinite(val):
            return float(val)
    except Exception:
        pass
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    rms = np.sqrt(np.mean(mono ** 2)) + 1e-9
    return float(20.0 * np.log10(rms) - 0.691)



def measure_human_weighted_loudness(audio: np.ndarray, sr: int, sensitivity_amount: float = 0.65) -> dict:
    """LUFS perceptual adaptativo para decisiones de matching.

    Mantiene el LUFS integrado BS.1770 como base, pero aplica una corrección
    psicoacústica suave cuando la zona donde el oído es más sensible
    (aprox. 3-6 kHz, con hombros en 1-3 kHz y 6-9 kHz) está más o menos
    presente que en un balance tonal típico. No reemplaza el LUFS estándar para
    reportes de plataforma; sirve para calcular una ganancia que suene más
    pareja al oído humano al comparar contra una referencia.
    """
    standard_lufs = measure_lufs_integrated(audio, sr)
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float64)
    if mono.size < max(128, int(sr * 0.05)):
        return {
            "standard_lufs": round(float(standard_lufs), 2),
            "perceived_lufs": round(float(standard_lufs), 2),
            "presence_correction_db": 0.0,
            "presence_relative_db": 0.0,
            "sensitivity_band_hz": [3000, 6000],
            "sensitivity_amount": round(float(sensitivity_amount), 3),
        }

    full_rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
    full_db = 20.0 * np.log10(full_rms)
    nyquist = sr / 2.0

    def _band_rms(lo: float, hi: float) -> float:
        lo = float(np.clip(lo, 20.0, nyquist - 10.0))
        hi = float(np.clip(hi, lo + 10.0, nyquist - 1.0))
        if hi <= lo or hi >= nyquist:
            return 1e-12
        sos = butter(3, [lo, hi], btype="bandpass", fs=sr, output="sos")
        band = sosfiltfilt(sos, mono)
        return float(np.sqrt(np.mean(band ** 2)) + 1e-12)

    # Zona de máxima sensibilidad + hombros para evitar que el cálculo dependa
    # de una banda demasiado estrecha. La referencia principal es 3-6 kHz.
    bands = [
        (1000.0, 3000.0, 0.35),
        (3000.0, 6000.0, 1.00),
        (6000.0, 9000.0, 0.35),
    ]
    weighted_power = 0.0
    weight_sum = 0.0
    for lo, hi, w in bands:
        rms = _band_rms(lo, hi)
        weighted_power += w * (rms ** 2)
        weight_sum += w
    presence_rms = float(np.sqrt(weighted_power / max(weight_sum, 1e-12)) + 1e-12)
    presence_db = 20.0 * np.log10(presence_rms)

    # En masters balanceados, esta región suele medir bastante por debajo del
    # RMS total. Usamos -10 dB como punto neutro y limitamos la corrección para
    # no convertir el match de loudness en un EQ encubierto.
    presence_relative_db = float(presence_db - full_db)
    correction_db = float(np.clip((presence_relative_db + 10.0) * sensitivity_amount, -4.0, 4.0))
    perceived_lufs = float(standard_lufs + correction_db)
    return {
        "standard_lufs": round(float(standard_lufs), 2),
        "perceived_lufs": round(perceived_lufs, 2),
        "presence_correction_db": round(correction_db, 2),
        "presence_relative_db": round(presence_relative_db, 2),
        "sensitivity_band_hz": [3000, 6000],
        "sensitivity_amount": round(float(sensitivity_amount), 3),
    }

def eq_high_pass(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 5.0, sr / 2.0 - 1.0))
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    # Usar sosfiltfilt para fase cero y evitar artefactos
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def eq_low_pass(audio: np.ndarray, sr: int, cutoff_hz: float = 18000.0) -> np.ndarray:
    """Pasa-bajos complementario al high-pass (item 2 de la cadena: HPF+LPF).
    Fase cero (sosfiltfilt), misma convención que el resto de los filtros de
    borde de la cadena. cutoff_hz se acota a [100 Hz, nyquist-1] porque un
    LPF pensado para mastering no tiene sentido por debajo de eso (se
    superpondría con la ecualización correctiva/tonal)."""
    cutoff_hz = float(np.clip(cutoff_hz, 100.0, sr / 2.0 - 1.0))
    sos = butter(4, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def eq_high_shelf(audio: np.ndarray, sr: int,
                  cutoff_hz: float = 8000.0, gain_db: float = 2.0,
                  freq_hz: float = None) -> np.ndarray:
    # freq_hz overrides cutoff_hz when provided (alias for UI clarity)
    if freq_hz is not None:
        cutoff_hz = freq_hz
    if gain_db == 0.0:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * cutoff_hz / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * 1.0 + 2.0)  # BUGFIX: (1/1-1)=0 colapsaba la fórmula RBJ
    sqrtA = np.sqrt(A)
    b0 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
    a0 =       (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
    a1 =  2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 =       (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    # BUGFIX: antes se usaba sosfilt (causal) acá mientras el resto de la EQ
    # (high-pass, bandas paramétricas) usa sosfiltfilt (fase cero). Esa
    # inconsistencia introducía un desfasaje relativo entre el high shelf y
    # las demás etapas de EQ, lo que al sumarse podía generar cancelaciones
    # sutiles/coloración no deseada. Ahora es fase cero, como el resto.
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def eq_parametric_band(audio: np.ndarray, sr: int,
                       freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    if gain_db == 0.0:
        return audio
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    q    = float(np.clip(q, 0.1, 30.0))
    A    = 10.0 ** (gain_db / 40.0)
    w0   = 2.0 * np.pi * freq / sr
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def stereo_width(audio: np.ndarray, width: float = 1.2) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    mid  = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5 * width
    return np.stack([mid + side, mid - side])

def multiband_stereo_width(audio: np.ndarray, sr: int,
                           low_width: float = 0.8,
                           mid_width: float = 1.2,
                           high_width: float = 1.5,
                           low_crossover: float = 150.0,
                           high_crossover: float = 4000.0) -> np.ndarray:
    """Ancho estéreo multibanda: aplica un factor de width diferente a cada banda
    de frecuencia (graves, medios, agudos) usando filtros LP/HP de 2º orden.

    Ventajas vs. width global:
    - Los graves se pueden mantener casi mono (low_width≈0.8..1.0) para
      compatibilidad mono, potencia en sub y evitar cancelaciones de fase en
      sistemas mono/club.
    - Los medios y agudos pueden ensancharse independientemente para dar aire
      sin afectar la coherencia del bajo.

    low_width:  factor de ancho para la banda de graves (0=mono, 1=original, 2=doble)
    mid_width:  factor de ancho para la banda de medios
    high_width: factor de ancho para la banda de agudos
    low_crossover:  Hz de cruce graves→medios
    high_crossover: Hz de cruce medios→agudos
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    low_crossover  = float(np.clip(low_crossover,  20.0, sr / 2.0 - 10.0))
    high_crossover = float(np.clip(high_crossover, low_crossover + 1.0, sr / 2.0 - 1.0))

    # BUGFIX: filtros Butterworth 2º orden LP+HP no suman a la señal original
    # en la zona de cruce (introducen coloración/phase notch). Se usan filtros
    # Linkwitz-Riley 4º orden (LR4 = cascada de dos Butterworth 2º orden con
    # sosfiltfilt, equivalente a cascadar dos veces) que sí suman perfectamente
    # en magnitud en el crossover — el estándar en crossovers de mastering.
    # sosfiltfilt ya aplica el filtro dos veces (forward+backward), así que
    # para LR4 basta con un butter(2) pasado por sosfiltfilt (2 pases × 2 polos
    # = 4 polos efectivos, -24 dB/oct, suma perfecta).
    sos_lo_lp = butter(2, low_crossover,  btype='lowpass',  fs=sr, output='sos')
    sos_lo_hp = butter(2, low_crossover,  btype='highpass', fs=sr, output='sos')
    sos_hi_lp = butter(2, high_crossover, btype='lowpass',  fs=sr, output='sos')
    sos_hi_hp = butter(2, high_crossover, btype='highpass', fs=sr, output='sos')

    # Aplicar dos veces cada filtro (forward+backward) para LR4
    def _lr4(sos, x):
        return sosfiltfilt(sos, sosfiltfilt(sos, x))

    low_band  = _lr4(sos_lo_lp, audio)
    mh_band   = _lr4(sos_lo_hp, audio)
    mid_band  = _lr4(sos_hi_lp, mh_band)
    high_band = _lr4(sos_hi_hp, mh_band)

    def _apply_width(band, w):
        mid  = (band[0] + band[1]) * 0.5
        side = (band[0] - band[1]) * 0.5 * w
        return np.stack([mid + side, mid - side])

    return _apply_width(low_band, low_width) + _apply_width(mid_band, mid_width) + _apply_width(high_band, high_width)

def reverb_simple(audio: np.ndarray, sr: int,
                  room_size: float = 0.3, wet: float = 0.1,
                  seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    decay_samples = max(int(sr * room_size), 1)
    t  = np.linspace(0.0, room_size, decay_samples)
    ir = np.exp(-6.0 * t) * rng.standard_normal(decay_samples) * 0.5
    ir[0] = 1.0
    dry = 1.0 - wet
    if audio.ndim == 1:
        wet_sig = fftconvolve(audio, ir, mode="full")[:len(audio)]
        return dry * audio + wet * wet_sig
    out = []
    for ch in audio:
        wet_sig = fftconvolve(ch, ir, mode="full")[:len(ch)]
        out.append(dry * ch + wet * wet_sig)
    return np.stack(out)

def audio_clipper(audio: np.ndarray, sr: int,
                  ceiling: float = 0.98, mode: str = "soft",
                  drive_db: float = 0.0, bypass: bool = True) -> tuple:
    """Clipper (item 14 de la cadena): última etapa de control de pico ANTES
    del limitador brick-wall. Dos modos:
      - "soft": waveshaper tanh que se aplana suavemente acercándose a
        `ceiling` (satura de forma más musical, agrega algo de armónicos,
        estilo clipper analógico).
      - "hard": np.clip duro a ±ceiling (corte abrupto, más agresivo/digital,
        típico de clippers de "loudness war" usados con moderación antes del
        limitador para quitarle trabajo al lookahead).
    drive_db empuja la señal contra el techo antes de clipear (a mayor
    drive, más contenido se recorta). No reemplaza al limitador: el
    limitador sigue siendo quien garantiza el true peak final."""
    if bypass:
        return audio, {"bypass": True, "clipped_pct": 0.0}

    ceiling = float(np.clip(ceiling, 0.1, 1.0))
    drive = 10.0 ** (drive_db / 20.0)

    def process_channel(ch):
        driven = ch * drive
        if mode == "hard":
            return np.clip(driven, -ceiling, ceiling)
        return ceiling * np.tanh(driven / ceiling)

    if audio.ndim == 1:
        out = process_channel(audio)
    else:
        out = np.stack([process_channel(ch) for ch in audio])

    mono = out.mean(axis=0) if out.ndim == 2 else out
    clipped_pct = float(np.mean(np.abs(mono) >= ceiling * 0.999) * 100.0)
    meter = {
        "bypass": False,
        "mode": mode,
        "ceiling": round(ceiling, 4),
        "drive_db": round(drive_db, 2),
        "clipped_pct": round(clipped_pct, 3),
    }
    return out, meter

def limiter(audio: np.ndarray, sr: int,
            ceiling: float = 0.95, release_ms: float = 80.0,
            lookahead_ms: float = 5.0, oversample: int = DEFAULT_DSP_OVERSAMPLE) -> np.ndarray:
    """True Peak limiter brick-wall con lookahead, ganancia suavizada, oversampling
    y detección LINKEADA entre canales (estéreo).

    El oversampling (por defecto 4x) permite detectar los inter-sample peaks
    (picos que aparecen entre samples al reconstruir la señal analógica) que un
    limiter a sample-rate nativa no ve — el estándar AES17 / EBU R128 llama a
    esto "True Peak". Sin oversampling un -0.1 dBFS digital puede reproducirse
    en analógico como +0.5 dBFS o más, causando clipping en conversores DAC.
    Con oversampling 4x el margen de error se reduce a <0.01 dB típicamente.

    BUGFIX (corrimiento de imagen estéreo): antes cada canal calculaba su
    propia curva de ganancia de forma independiente. Si L pedía más
    reducción que R en un instante (p.ej. un platillo panneado a la
    izquierda), L se atenuaba más que R en ESE instante, moviendo el
    balance L/R momentáneamente — un limiter "unlinked" corre/empuja la
    imagen estéreo con la música. Un limiter de máster debe estar linkeado:
    se detecta el pico máximo entre canales y se aplica LA MISMA curva de
    ganancia a ambos, preservando el panorama exactamente como estaba.
    """
    release_coef = float(np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9)))
    lookahead_samples = max(1, int(sr * lookahead_ms / 1000.0))
    ovs = max(1, int(oversample))

    def gain_curve(abs_signal):
        if ovs > 1:
            abs_up = resample_poly(abs_signal, ovs, 1)
            abs_up = np.abs(abs_up)
            la_up = max(1, lookahead_samples * ovs)
            if la_up > 1:
                fwd_max = maximum_filter1d(abs_up, size=la_up, mode='nearest')
                # BUGFIX: np.roll es circular — las muestras del final volvían
                # al principio generando reducción de ganancia fantasma en el
                # tail del archivo. El fix anterior ponía la cola en 0.0, pero
                # eso significa "sin pico" → ganancia 1.0 → el limiter queda
                # SIN protección en los últimos samples (riesgo real de que
                # se cuele un pico sin brickwall justo al final del archivo).
                # Se sostiene (edge-hold) el último valor válido del detector
                # en la cola en vez de anularlo, así el lookahead sigue
                # protegiendo hasta el último sample sin wrap circular.
                shift = la_up // 2
                abs_up_la = np.empty_like(fwd_max)
                abs_up_la[:len(fwd_max) - shift] = fwd_max[shift:]
                abs_up_la[len(fwd_max) - shift:] = fwd_max[-1]
            else:
                abs_up_la = abs_up
            ig_up = np.minimum(1.0, ceiling / (abs_up_la + 1e-9)).astype(np.float64)
            if HAS_NUMBA:
                smoothed_up = _limiter_gain_numba(ig_up, float(np.exp(-1.0 / (sr * ovs * (release_ms / 1000.0) + 1e-9))))
            else:
                rc = float(np.exp(-1.0 / (sr * ovs * (release_ms / 1000.0) + 1e-9)))
                n = len(ig_up)
                smoothed_up = np.empty(n, dtype=np.float64)
                prev = 1.0
                for i in range(n):
                    g = ig_up[i]
                    prev = g if g < prev else rc * prev + (1.0 - rc) * g
                    smoothed_up[i] = prev
            smoothed_raw = resample_poly(smoothed_up, 1, ovs)
            n = len(abs_signal)
            if len(smoothed_raw) < n:
                # BUGFIX: si resample da N-1 muestras y audio tiene N,
                # el broadcast audio * smoothed[np.newaxis,:] corrompe la imagen estéreo.
                # Edge-pad con el último valor (ganancia más conservadora disponible).
                smoothed_raw = np.pad(smoothed_raw, (0, n - len(smoothed_raw)), mode='edge')
            else:
                smoothed_raw = smoothed_raw[:n]
            return np.clip(smoothed_raw, 0.0, 1.0)
        else:
            abs_ch = np.abs(abs_signal)
            if lookahead_samples > 1:
                fwd_max = maximum_filter1d(abs_ch, size=lookahead_samples, mode='nearest')
                # BUGFIX: mismo fix que el path oversampled — se sostiene el
                # último valor válido del detector en la cola en vez de
                # ponerlo en 0.0, para no perder protección de lookahead en
                # los últimos samples del archivo (sin volver al wrap circular).
                shift = lookahead_samples // 2
                abs_ch_la = np.empty_like(fwd_max)
                abs_ch_la[:len(fwd_max) - shift] = fwd_max[shift:]
                abs_ch_la[len(fwd_max) - shift:] = fwd_max[-1]
            else:
                abs_ch_la = abs_ch
            ig = np.minimum(1.0, ceiling / (abs_ch_la + 1e-9)).astype(np.float64)
            if HAS_NUMBA:
                return _limiter_gain_numba(ig, release_coef)
            n = len(ig)
            smoothed = np.empty(n, dtype=np.float64)
            prev = 1.0
            for i in range(n):
                g = ig[i]
                prev = g if g < prev else release_coef * prev + (1.0 - release_coef) * g
                smoothed[i] = prev
            return smoothed

    if audio.ndim == 1:
        smoothed = gain_curve(audio)
        out = audio * smoothed
    else:
        # Detección linkeada: se toma el máximo absoluto ENTRE canales en
        # cada instante (equivalente a "peak between L/R" de un limiter
        # estéreo real) y esa única curva de ganancia se aplica a todos los
        # canales por igual, preservando el panorama.
        linked_peak = np.max(np.abs(audio), axis=0)
        smoothed = gain_curve(linked_peak)
        out = audio * smoothed[np.newaxis, :]
    return out

def mid_side_process(audio: np.ndarray,
                     mid_gain_db: float = 0.0,
                     side_gain_db: float = 0.0) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    left, right = audio[0], audio[1]
    mid  = (left + right) * 0.5
    side = (left - right) * 0.5
    mid  *= 10.0 ** (mid_gain_db  / 20.0)
    side *= 10.0 ** (side_gain_db / 20.0)
    return np.stack([mid + side, mid - side])

def mid_side_eq(audio: np.ndarray, sr: int,
                mid_freq: float = 250.0, mid_gain_db: float = 0.0, mid_q: float = 1.0,
                side_freq: float = 8000.0, side_gain_db: float = 0.0, side_q: float = 1.0) -> np.ndarray:
    """EQ Mid/Side dedicada (item 4 de la cadena): a diferencia de
    mid_side_process (que es solo trim de ganancia ancha sobre M y S), acá
    cada canal M/S recibe su propia banda paramétrica RBJ (freq/gain/Q
    independientes) — típicamente: quitar barro en el centro (mid, low-mid)
    o abrir aire en los laterales (side, agudos) sin tocar el otro canal.
    Se apoya en eq_parametric_band aplicada por separado sobre M y S antes
    de recomponer a L/R."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    if mid_gain_db == 0.0 and side_gain_db == 0.0:
        return audio
    left, right = audio[0], audio[1]
    mid  = (left + right) * 0.5
    side = (left - right) * 0.5
    if mid_gain_db != 0.0:
        mid = eq_parametric_band(mid, sr, freq=mid_freq, gain_db=mid_gain_db, q=mid_q)
    if side_gain_db != 0.0:
        side = eq_parametric_band(side, sr, freq=side_freq, gain_db=side_gain_db, q=side_q)
    return np.stack([mid + side, mid - side])

def mid_side_compressor(audio: np.ndarray, sr: int,
                        mid_threshold_db: float = -18.0, mid_ratio: float = 2.0,
                        mid_attack_ms: float = 15.0, mid_release_ms: float = 120.0,
                        mid_makeup_db: float = 0.0,
                        side_threshold_db: float = -18.0, side_ratio: float = 2.0,
                        side_attack_ms: float = 15.0, side_release_ms: float = 120.0,
                        side_makeup_db: float = 0.0,
                        oversample: int = DEFAULT_DSP_OVERSAMPLE,
                        bypass: bool = True,
                        pdr: bool = True, pdr_hold_ms: float = 500.0) -> tuple:
    """Compresor M/S dedicado (stage 4, después de mid_side_eq): comprime Mid
    y Side por separado, cada uno con su propio threshold/ratio/attack/
    release/makeup, reusando el mismo detector feed-forward + soft-knee de
    `compressor()` pero aplicado a M y S en vez de L/R directamente.

    Útil para: apretar el centro (voz/bajo/kick — lo que suele definir el
    groove) sin que la compresión tire del ancho estéreo, o para controlar
    la energía de Side (reverbs/ambiencia/paneos) sin oscurecer el mid. Al
    ir DESPUÉS del EQ M/S pero ANTES del compresor de banda ancha, el
    detector de nivel de ese compresor ya ve una señal con el balance M/S
    resuelto — misma lógica que motiva el orden de mid_side_eq en la
    cadena.

    threshold_db en vez de threshold lineal (a diferencia de compressor())
    porque acá es más natural pensar en dB al ajustar M y S por separado;
    se convierte internamente antes de llamar a compressor().
    """
    if bypass or audio.ndim != 2 or audio.shape[0] != 2:
        return audio, {"bypass": True, "mid": {"gr_db": 0.0}, "side": {"gr_db": 0.0}}

    left, right = audio[0], audio[1]
    mid  = (left + right) * 0.5
    side = (left - right) * 0.5

    mid_thresh_lin = 10.0 ** (float(mid_threshold_db) / 20.0)
    side_thresh_lin = 10.0 ** (float(side_threshold_db) / 20.0)

    mid_out, mid_meter = compressor(
        mid, sr, threshold=mid_thresh_lin, ratio=mid_ratio,
        attack_ms=mid_attack_ms, release_ms=mid_release_ms,
        makeup_db=mid_makeup_db, oversample=oversample, stereo_link=True,
        pdr=pdr, pdr_hold_ms=pdr_hold_ms,
    )
    side_out, side_meter = compressor(
        side, sr, threshold=side_thresh_lin, ratio=side_ratio,
        attack_ms=side_attack_ms, release_ms=side_release_ms,
        makeup_db=side_makeup_db, oversample=oversample, stereo_link=True,
        pdr=pdr, pdr_hold_ms=pdr_hold_ms,
    )

    out = np.stack([mid_out + side_out, mid_out - side_out])
    meter = {"bypass": False, "mid": mid_meter, "side": side_meter}
    return out, meter

def low_end_mono_maker(audio: np.ndarray, sr: int,
                       freq: float = 120.0, mono_amount: float = 1.0) -> np.ndarray:
    """Mono Maker de graves DEDICADO (independiente del stereo_enhancer, que
    trae su propio bass-mono fijo). Por debajo de `freq` reduce el ancho
    estéreo hacia mono en la proporción `mono_amount` (0 = estéreo intacto,
    1 = mono total); por encima de `freq` no toca nada. Usa crossover
    Butterworth 4º orden en fase cero (sosfiltfilt), igual convención que el
    resto de la EQ. Sirve para compatibilidad mono, evitar cancelaciones de
    fase de graves en sistemas club/vinilo, y concentrar energía de sub en
    el centro — el pedido típico de "graves en mono <100-150 Hz".
    """
    if audio.ndim != 2 or audio.shape[0] != 2 or mono_amount <= 0.0:
        return audio
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    mono_amount = float(np.clip(mono_amount, 0.0, 1.0))
    sos_lp = butter(4, freq, btype="lowpass",  fs=sr, output="sos")
    sos_hp = butter(4, freq, btype="highpass", fs=sr, output="sos")
    low  = sosfiltfilt(sos_lp, audio)
    high = sosfiltfilt(sos_hp, audio)
    low_mono = np.tile(low.mean(axis=0), (2, 1))
    low_out  = low * (1.0 - mono_amount) + low_mono * mono_amount
    return low_out + high

# ─── Diseño de filtros RBJ (biquad) reutilizables como SOS ────────────────────
# Extraídos como helpers independientes para poder evaluar su respuesta en
# frecuencia (sosfreqz) y sumarla en dB al diseñar el EQ de fase lineal FIR
# más abajo, sin duplicar/desincronizar las fórmulas de eq_parametric_band /
# eq_high_shelf.

def _design_peaking_sos(sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    q    = float(np.clip(q, 0.1, 30.0))
    A    = 10.0 ** (gain_db / 40.0)
    w0   = 2.0 * np.pi * freq / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def _design_high_shelf_sos(sr: int, freq: float, gain_db: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * 1.0 + 2.0)
    sqrtA = np.sqrt(A)
    b0 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
    a0 =       (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
    a1 =  2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 =       (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def _design_low_shelf_sos(sr: int, freq: float, gain_db: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * 1.0 + 2.0)
    sqrtA = np.sqrt(A)
    b0 =    A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
    b1 =  2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
    b2 =    A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
    a0 =         (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
    a2 =         (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def linear_phase_eq(audio: np.ndarray, sr: int, bands: list, num_taps: int = 2049) -> np.ndarray:
    """EQ de FASE LINEAL real (FIR), a diferencia de eq_parametric_band /
    eq_high_shelf que son IIR con sosfiltfilt (fase CERO, no lineal: fase
    cero es más fuerte que lineal pero exige procesar offline con la señal
    completa, cosa que igual hacemos acá, así que en la práctica ya nos daba
    fase cero). El motivo real para tener un modo FIR de fase lineal
    dedicado es tener delay de grupo CONSTANTE y verificable en todas las
    frecuencias (como en plugins tipo Pro-Q "Linear Phase"), útil cuando se
    van a sumar/comparar bandas o cuando el pre-ringing controlado del FIR
    es preferible a la respuesta IIR en cortes/boosts grandes.

    bands: lista de dicts, cada uno:
        {"type": "peak", "freq": 100.0, "gain_db": 2.0, "q": 1.0}
        {"type": "high_shelf", "freq": 8000.0, "gain_db": 2.0}
        {"type": "low_shelf",  "freq": 100.0,  "gain_db": 2.0}
    Se combinan TODAS las bandas en una sola curva de magnitud (suma en dB),
    y de ahí se diseña UN solo FIR (firwin2) — evita cascadear N filtros y
    acumular N delays/errores de diseño por separado.
    """
    bands = [b for b in (bands or []) if b.get("gain_db", 0.0) != 0.0]
    if not bands:
        return audio
    num_taps = int(num_taps)
    if num_taps % 2 == 0:
        num_taps += 1  # taps impares -> FIR simétrico Tipo I, fase lineal exacta

    n_freqs = 4096
    freqs_grid = np.linspace(0.0, sr / 2.0, n_freqs)
    total_db = np.zeros(n_freqs)
    for b in bands:
        gain_db = float(b.get("gain_db", 0.0))
        freq = float(np.clip(b.get("freq", 1000.0), 20.0, sr / 2.0 - 1.0))
        btype = b.get("type", "peak")
        if btype == "high_shelf":
            sos = _design_high_shelf_sos(sr, freq, gain_db)
        elif btype == "low_shelf":
            sos = _design_low_shelf_sos(sr, freq, gain_db)
        else:
            q = float(np.clip(b.get("q", 1.0), 0.1, 30.0))
            sos = _design_peaking_sos(sr, freq, gain_db, q)
        _, h = sosfreqz(sos, worN=freqs_grid, fs=sr)
        total_db += 20.0 * np.log10(np.abs(h) + 1e-12)

    gain_lin = 10.0 ** (total_db / 20.0)
    freq_norm = freqs_grid / (sr / 2.0)
    freq_norm[-1] = 1.0
    taps = firwin2(num_taps, freq_norm, gain_lin)

    def _apply(ch):
        # fftconvolve + mode='same' con FIR simétrico (impar) cancela el
        # delay de grupo (num_taps-1)/2: el tap central queda alineado con
        # t=0, dando salida sin corrimiento temporal audible.
        return fftconvolve(ch, taps, mode='same')

    if audio.ndim == 1:
        return _apply(audio)
    return np.stack([_apply(ch) for ch in audio])

def detect_resonances(audio: np.ndarray, sr: int,
                      min_freq: float = 120.0, max_freq: float = 9000.0,
                      threshold_db: float = 4.0, max_resonances: int = 6) -> list:
    """Detección de resonancias: busca picos ANGOSTOS que sobresalen del
    perfil espectral general (baseline = mediana móvil en el propio
    espectro promediado por Welch), no simplemente "la banda más alta" —
    eso sería balance tonal, no resonancia. Cada resultado trae una
    sugerencia de corte (Dynamic EQ / EQ estático) lista para aplicar con
    dynamic_eq_band(freq=r['freq_hz'], q=r['suggested_q'], ...).
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n_fft = 8192
    mag = _averaged_magnitude_spectrum(mono, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mag_db = 20.0 * np.log10(mag + 1e-12)

    band_mask = (freqs >= min_freq) & (freqs <= max_freq)
    idx = np.where(band_mask)[0]
    if len(idx) < 10:
        return []

    win = max(5, (int(len(idx) * 0.03) | 1))  # ventana impar, ~3% del rango
    baseline = median_filter(mag_db[idx], size=win)
    excess = mag_db[idx] - baseline

    peaks, props = find_peaks(excess, height=threshold_db, distance=max(3, win // 2))
    results = []
    for p, h in zip(peaks, props["peak_heights"]):
        f = float(freqs[idx][p])

        # BUGFIX: Q hardcodeado a 4.0 ignoraba el ancho real del pico.
        # Para resonancias angostas (room modes, resonancias de instrumento)
        # Q=4 puede ser demasiado ancho; para acumulaciones más suaves puede
        # ser demasiado angosto. Derivamos Q desde el ancho del pico a -3 dB
        # en el espectro de exceso (excess array), usando la fórmula estándar
        # Q = f_center / BW_3dB. Si el pico es tan estrecho que no se puede
        # medir con precisión (solo 1-2 bins), se usa un Q máximo de 10.
        # Si es muy ancho (BW > 2 octavas), Q mínimo de 1.0.
        half_h = h / 2.0  # nivel -3 dB relativo al pico de excess
        left_idx, right_idx = p, p
        # Buscar cruce izquierdo a -3 dB
        for li in range(p - 1, -1, -1):
            if excess[li] <= half_h:
                left_idx = li
                break
        # Buscar cruce derecho a -3 dB
        for ri in range(p + 1, len(excess)):
            if excess[ri] <= half_h:
                right_idx = ri
                break
        f_lo = float(freqs[idx][left_idx])
        f_hi = float(freqs[idx][right_idx])
        bw_3db = max(f_hi - f_lo, f / 20.0)  # mínimo 1/20 de la frecuencia central
        suggested_q = float(np.clip(f / bw_3db, 1.0, 10.0))

        results.append({
            "freq_hz": round(f, 1),
            "excess_db": round(float(h), 2),
            "suggested_cut_db": round(float(min(h * 0.7, 6.0)), 2),
            "suggested_q": round(suggested_q, 1),
        })
    results.sort(key=lambda r: -r["excess_db"])
    return results[:max_resonances]

def detect_sibilance(audio: np.ndarray, sr: int,
                     low_hz: float = 4000.0, high_hz: float = 9000.0,
                     block_s: float = 0.05) -> dict:
    """Detección de sibilancia: compara la envolvente de energía de la
    banda 4-9kHz contra la envolvente de banda completa, cuadro a cuadro.
    No basta con "hay mucha energía en agudos" (eso es balance tonal) — lo
    que caracteriza la sibilancia son PICOS puntuales de esa banda por
    encima de su propia mediana (las "eses"/"ches" sobresaliendo del resto
    de la voz). suggested_reduction_db queda pensado para alimentar
    dynamic_eq_band como de-esser (freq≈centro de la banda, Q ancho).
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    high_hz = min(high_hz, sr / 2.0 - 1.0)
    if high_hz <= low_hz:
        return {"present": False, "band_hz": [low_hz, high_hz], "severity_db": 0.0,
                "frames_flagged_pct": 0.0, "suggested_reduction_db": 0.0}

    band = _bandpass_filter(mono, sr, low_hz, high_hz)
    hop = max(1, int(sr * block_s))
    n_blocks = max(1, len(mono) // hop)
    band_env_db, full_env_db = [], []
    for i in range(n_blocks):
        seg_b = band[i * hop:(i + 1) * hop]
        seg_f = mono[i * hop:(i + 1) * hop]
        if len(seg_b) == 0:
            continue
        band_env_db.append(20.0 * np.log10(np.sqrt(np.mean(seg_b ** 2)) + 1e-9))
        full_env_db.append(20.0 * np.log10(np.sqrt(np.mean(seg_f ** 2)) + 1e-9))
    if not band_env_db:
        return {"present": False, "band_hz": [low_hz, high_hz], "severity_db": 0.0,
                "frames_flagged_pct": 0.0, "suggested_reduction_db": 0.0}

    ratio_db = np.array(band_env_db) - np.array(full_env_db)
    baseline = float(np.median(ratio_db))
    spikes = ratio_db - baseline
    flagged = spikes > 6.0
    severity = float(np.mean(spikes[spikes > 0])) if np.any(spikes > 0) else 0.0
    present = bool(np.mean(flagged) > 0.03 and severity > 3.0)

    return {
        "present": present,
        "band_hz": [round(low_hz, 1), round(high_hz, 1)],
        "severity_db": round(severity, 2),
        "frames_flagged_pct": round(float(np.mean(flagged) * 100.0), 1),
        "suggested_reduction_db": round(float(min(severity, 8.0)), 2),
    }

def recommend_dynamic_eq(audio: np.ndarray, sr: int,
                         resonances: list = None, sibilance: dict = None) -> dict:
    """Corre detect_resonances + detect_sibilance (si no vienen ya calculados,
    p.ej. reusados desde el streaming) y arma un set de parámetros
    reso_*/dyneq_* listo para pasarle directo a apply_mastering_chain /
    process_audio (mismos nombres de campo que MasteringParams, así el
    caller puede hacer `params.update(recommendation["recommended_params"])`).

    reso_* apunta a la resonancia más severa detectada (si hay varias, el
    resto queda solo informativo en `resonances`). dyneq_* actúa como
    de-esser sobre la banda de sibilancia detectada. Si no se detecta nada
    relevante en alguna de las dos, esa mitad de los params queda en
    bypass=True — no se fuerza un corte donde no hace falta.
    """
    if resonances is None:
        resonances = detect_resonances(audio, sr)
    if sibilance is None:
        sibilance = detect_sibilance(audio, sr)

    reso_params = {
        "reso_bypass": False, "reso_freq": 1200.0, "reso_q": 3.0,
        "reso_threshold_db": -18.0, "reso_ratio": 3.0,
        "reso_attack_ms": 5.0, "reso_release_ms": 100.0,
        "reso_max_reduction_db": 8.0,
    }
    if resonances:
        top = resonances[0]
        reso_params.update({
            "reso_bypass": False,
            "reso_freq": top["freq_hz"],
            "reso_q": top["suggested_q"],
            "reso_max_reduction_db": max(1.0, top["suggested_cut_db"]),
        })

    dyneq_params = {
        "dyneq_bypass": False, "dyneq_freq": 3000.0, "dyneq_q": 2.5,
        "dyneq_threshold_db": -18.0, "dyneq_ratio": 3.0,
        "dyneq_attack_ms": 3.0, "dyneq_release_ms": 80.0,
        "dyneq_max_reduction_db": 12.0,
    }
    if sibilance.get("present"):
        low_hz, high_hz = sibilance["band_hz"]
        center = float(np.sqrt(max(low_hz, 1.0) * max(high_hz, 1.0)))
        bw = max(high_hz - low_hz, 1.0)
        dyneq_params.update({
            "dyneq_bypass": False,
            "dyneq_freq": round(center, 1),
            "dyneq_q": round(float(np.clip(center / bw, 0.5, 12.0)), 2),
            "dyneq_max_reduction_db": max(1.0, sibilance["suggested_reduction_db"]),
        })

    parts = []
    if resonances:
        top = resonances[0]
        extra = f" (+{len(resonances) - 1} más)" if len(resonances) > 1 else ""
        parts.append(
            f"Resonancia en {top['freq_hz']:.0f} Hz (+{top['excess_db']:.1f} dB){extra}: "
            f"Dynamic EQ sugerido en {top['freq_hz']:.0f} Hz, Q={top['suggested_q']:.1f}, "
            f"reducción máx. {reso_params['reso_max_reduction_db']:.1f} dB."
        )
    else:
        parts.append("No se detectaron resonancias relevantes.")
    if sibilance.get("present"):
        parts.append(
            f"Sibilancia en {sibilance['band_hz'][0]:.0f}-{sibilance['band_hz'][1]:.0f} Hz "
            f"(severidad {sibilance['severity_db']:.1f} dB, {sibilance['frames_flagged_pct']:.1f}% de cuadros): "
            f"de-esser sugerido en {dyneq_params['dyneq_freq']:.0f} Hz, "
            f"reducción máx. {dyneq_params['dyneq_max_reduction_db']:.1f} dB."
        )
    else:
        parts.append("No se detectó sibilancia relevante.")

    return {
        "resonances": resonances,
        "sibilance": sibilance,
        "recommended_params": {**reso_params, **dyneq_params},
        "summary": " ".join(parts),
    }

# ─── Balance tonal automático (EQ inteligente SIN referencia externa) ────────
# Curva "target" neutra de mastering profesional: leve tilt descendente hacia
# agudos (similar al promedio de masters comerciales modernos / una curva
# "house" genérica de control room), prácticamente plana en graves/medios.
# No representa un género específico — es el punto de partida cuando NO hay
# un track de referencia real (para eso está process_audio_with_reference).
_TONAL_BALANCE_TARGET_ANCHORS_HZ = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0,
                                    2000.0, 4000.0, 8000.0, 16000.0]
_TONAL_BALANCE_TARGET_DB = [-1.0, 0.0, 0.5, 1.0, 0.8, 0.0, -0.5, -1.5, -3.5, -7.0]


def auto_tonal_balance(audio: np.ndarray, sr: int,
                       n_bands: int = 40,
                       max_boost_db: float = 3.5,
                       max_cut_db: float = -4.5,
                       max_eq_bands: int = 6,
                       min_gain_db: float = 0.4,
                       savgol_window_frac: float = 0.12) -> dict:
    """EQ inteligente de balance tonal SIN necesidad de un track de
    referencia externo. Mide el espectro promediado del track, extrae su
    FORMA macro con un filtro Savitzky-Golay (ajusta un polinomio local por
    ventana en vez de solo promediar/mediana móvil — sigue mejor el tilt
    real del contenido sin aplanar picos anchos genuinos de la mezcla), la
    compara contra una curva target neutra de mastering profesional
    (_TONAL_BALANCE_TARGET_DB) y devuelve bandas de EQ paramétrico
    correctivas con Q ADAPTATIVO:
      - Desviación ANGOSTA (la baseline se aparta bruscamente en pocas
        bandas vecinas, ej. una coloración puntual de la mezcla) -> Q alto,
        corte/boost quirúrgico.
      - Desviación ANCHA (tilt tonal general de varias octavas) -> Q bajo,
        boost/corte amplio tipo shelving, más musical.

    El resultado (`eq_bands`) está listo para pasarle directo a
    linear_phase_eq() / apply_tonal_balance(), con el mismo formato de banda
    que usa esa función ({"type": "peak", "freq": ..., "gain_db": ..., "q": ...}).
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n_fft = 8192
    mag = _averaged_magnitude_spectrum(mono, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    nyquist = sr / 2.0
    hi_edge = min(nyquist, 19000.0)
    log_edges = np.logspace(np.log10(24.0), np.log10(hi_edge), n_bands + 1)
    band_db = _log_band_average_db(freqs, mag, log_edges)
    band_freqs = np.array([np.sqrt(log_edges[i] * log_edges[i + 1]) for i in range(n_bands)])

    # ── Baseline Savitzky-Golay ─────────────────────────────────────────────
    # Ventana impar proporcional a n_bands (polinomio orden 3: suficiente
    # para capturar tilt + curvatura amplia sin perseguir cada bin como
    # haría un orden alto/ventana chica).
    win = max(5, int(round(n_bands * savgol_window_frac)) | 1)
    if win >= n_bands:
        win = n_bands - 1 if (n_bands - 1) % 2 == 1 else n_bands - 2
    win = max(5, win)
    baseline = savgol_filter(band_db, window_length=win, polyorder=3, mode="interp")

    # Curva target interpolada en log-frecuencia a las n_bands del análisis.
    # Solo importa la FORMA del target: se re-centra contra la media de la
    # propia baseline para no imponer un nivel/tilt absoluto (eso lo maneja
    # el gain-staging / LUFS por separado).
    target_db = np.interp(np.log10(band_freqs),
                          np.log10(_TONAL_BALANCE_TARGET_ANCHORS_HZ),
                          _TONAL_BALANCE_TARGET_DB)
    target_db = target_db - np.mean(target_db) + np.mean(baseline)

    deviation = target_db - baseline  # cuánto habría que boostear(+)/cortear(-)

    # ── Q adaptativo ─────────────────────────────────────────────────────────
    # macro_baseline: versión MUCHO más suavizada de la baseline (ventana
    # ~3x más ancha). Donde baseline y macro_baseline casi coinciden, la
    # desviación es tilt de banda ancha -> Q bajo. Donde se apartan mucho en
    # pocas bandas, es coloración puntual -> Q alto.
    macro_win = min(n_bands - 1 if (n_bands - 1) % 2 == 1 else n_bands - 2, max(win * 3, 9))
    macro_win = macro_win if macro_win % 2 == 1 else macro_win - 1
    macro_win = max(5, macro_win)
    if macro_win < n_bands:
        macro_baseline = savgol_filter(band_db, window_length=macro_win, polyorder=2, mode="interp")
    else:
        macro_baseline = baseline
    sharpness = np.abs(baseline - macro_baseline)

    deviation_clipped = np.clip(deviation, max_cut_db, max_boost_db)
    abs_dev = np.abs(deviation_clipped)
    peaks, props = find_peaks(abs_dev, height=min_gain_db, distance=max(2, n_bands // 12))

    candidates = []
    for i, h in zip(peaks, props["peak_heights"]):
        sharp = float(sharpness[i])
        # 0.0 dB de diferencia con la macro-curva -> Q~0.6 (shelving ancho)
        # 3+ dB de diferencia -> Q~4.0 (quirúrgico)
        q_adaptive = float(np.clip(0.6 + sharp * 1.3, 0.5, 4.0))
        candidates.append({
            "freq_hz": round(float(band_freqs[i]), 1),
            "gain_db": round(float(deviation_clipped[i]), 2),
            "q": round(q_adaptive, 2),
            "deviation_db": round(float(deviation[i]), 2),
        })
    candidates.sort(key=lambda c: -abs(c["gain_db"]))
    bands = candidates[:max_eq_bands]
    bands.sort(key=lambda c: c["freq_hz"])

    eq_bands = [{"type": "peak", "freq": b["freq_hz"], "gain_db": b["gain_db"], "q": b["q"]} for b in bands]

    if bands:
        parts_txt = "; ".join(f"{b['freq_hz']:.0f}Hz {b['gain_db']:+.1f}dB (Q={b['q']:.1f})" for b in bands)
        summary = f"Balance tonal automático: {len(bands)} banda(s) sugeridas — {parts_txt}."
    else:
        summary = "Balance tonal automático: el track ya está cerca de la curva objetivo, no se sugieren correcciones."

    return {
        "band_freqs_hz": [round(float(f), 1) for f in band_freqs],
        "track_db": [round(float(v), 2) for v in band_db],
        "baseline_db": [round(float(v), 2) for v in baseline],
        "target_db": [round(float(v), 2) for v in target_db],
        "suggested_bands": bands,
        "eq_bands": eq_bands,
        "summary": summary,
    }


def apply_tonal_balance(audio: np.ndarray, sr: int, eq_bands: list,
                        amount: float = 1.0, num_taps: int = 2049) -> np.ndarray:
    """Aplica las bandas sugeridas por auto_tonal_balance() vía linear_phase_eq
    (fase lineal, una sola pasada combinando todas las bandas — evita
    cascadear N filtros paramétricos que se solapan/refuerzan entre sí en
    bandas log-espaciadas vecinas). `amount` (0..1) escala la ganancia de
    todas las bandas para dosificar el efecto sin tener que recalcular las
    frecuencias/Q sugeridas.
    """
    if not eq_bands:
        return audio
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return audio
    scaled = [{**b, "gain_db": b["gain_db"] * amount} for b in eq_bands]
    return linear_phase_eq(audio, sr, scaled, num_taps=num_taps)


def dynamic_eq_band(audio: np.ndarray, sr: int,
                    freq: float = 3000.0, q: float = 2.5,
                    threshold_db: float = -18.0, ratio: float = 3.0,
                    attack_ms: float = 3.0, release_ms: float = 80.0,
                    max_reduction_db: float = 12.0, bypass: bool = True,
                    threshold_relative: bool = True) -> tuple:
    """Dynamic EQ de banda única: aísla una banda angosta (freq/Q), detecta
    su envolvente y aplica reducción de ganancia solo cuando supera el
    threshold, recombinando con el residual intacto.

    BUGFIX 1 — Detección con sosfilt unidireccional (no sosfiltfilt):
    El filtrado zero-phase (sosfiltfilt) introduce look-ahead implícito en la
    banda de detección: el detector 've' energía futura que aún no ocurrió en
    el audio real, adelantando el GR. El resultado audible es una reducción
    que empieza antes del evento (pre-ring de compresión). La solución es usar
    sosfilt (causal, unidireccional) SOLO para el detector de envolvente y
    mantener sosfiltfilt para la banda que se recombina con el residual —
    así la sustracción band/residual sigue siendo de fase cero (sin coloración)
    pero el timing del GR es causal.

    BUGFIX 2 — Threshold relativo al RMS de la banda:
    Un threshold absoluto (ej. -18 dBFS) es problemático porque la banda
    aislada siempre tiene mucha menos energía que la señal full-band. Con
    material a -14 LUFS la banda de 4-9 kHz puede estar en -35 dBFS → el
    compressor nunca dispara con threshold=-18. Con threshold_relative=True
    (default) el threshold se computa como `band_rms_db + threshold_db`,
    donde threshold_db pasa a ser el offset sobre el RMS de la banda
    (ej. -18 → dispara 18 dB por debajo del RMS medio de la banda, es decir
    en los picos más fuertes). Para compatibilidad con código existente que
    pasa threshold_db como valor absoluto, threshold_relative=False mantiene
    el comportamiento antiguo.

    BUGFIX 3 — Q derivado del pico real (ver detect_resonances):
    El parámetro q ya viene correcto desde los callers actualizados; este
    bloque solo lo clipea al rango [0.3, 30].
    """
    if bypass or max_reduction_db <= 0.0:
        return audio, {"bypass": True, "gr_db": 0.0}

    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    q = float(np.clip(q, 0.3, 30.0))
    bw = freq / q
    lo = max(20.0, freq - bw / 2.0)
    hi = min(sr / 2.0 - 1.0, freq + bw / 2.0)
    if hi <= lo:
        return audio, {"bypass": True, "gr_db": 0.0}

    # Filtro para la banda de señal (zero-phase → sin coloración en la
    # recombinación residual + banda)
    lo_c = float(np.clip(lo, 1.0, sr / 2.0 - 10.0))
    hi_c = float(np.clip(hi, lo_c + 5.0, sr / 2.0 - 1.0))
    sos = butter(2, [lo_c, hi_c], btype="bandpass", fs=sr, output="sos")

    def process_channel(ch):
        # Banda zero-phase para recombinar (sin coloración de fase)
        band_zp = sosfiltfilt(sos, ch)
        residual = ch - band_zp

        # Banda causal para detección de envolvente (BUGFIX 1: timing correcto)
        band_causal = sosfilt(sos, ch)
        env = _smooth_envelope(np.abs(band_causal), sr, attack_ms, release_ms)
        env_db = 20.0 * np.log10(env + 1e-9)

        # Threshold relativo al RMS de la banda (BUGFIX 2).
        # Se usa band_zp (zero-phase) para el RMS de referencia porque tiene
        # la misma magnitud espectral que la señal percibida (no el transitorio
        # del arranque del filtro IIR causal). band_causal solo se usa para
        # la envolvente temporal (timing).
        if threshold_relative:
            band_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(band_zp ** 2)) + 1e-9))
            thr = band_rms_db + threshold_db
        else:
            thr = threshold_db

        gr_db = _soft_knee_gain_reduction_np(env_db, thr, ratio)
        gr_db = np.maximum(gr_db, -max_reduction_db)
        band_out = band_zp * (10.0 ** (gr_db / 20.0))
        return residual + band_out, gr_db

    if audio.ndim == 1:
        out, gr_db = process_channel(audio)
    else:
        outs, grs = [], []
        for ch in audio:
            o, g = process_channel(ch)
            outs.append(o)
            grs.append(g)
        out = np.stack(outs)
        gr_db = np.mean(np.stack(grs), axis=0)

    meter = {
        "bypass": False,
        "freq_hz": round(freq, 1),
        "q": round(q, 2),
        "band_range_hz": [round(lo, 1), round(hi, 1)],
        "gr_db": round(float(np.mean(gr_db)), 2),
        "gr_max_db": round(float(np.min(gr_db)), 2),
    }
    return out, meter

def transient_shaper(audio: np.ndarray, sr: int,
                     attack_amount: float = 0.0, sustain_amount: float = 0.0,
                     attack_time_ms: float = 5.0, release_time_ms: float = 80.0) -> np.ndarray:
    if attack_amount == 0.0 and sustain_amount == 0.0:
        return audio

    def process_channel(ch):
        abs_ch = np.abs(ch)
        fast_env = _smooth_envelope(abs_ch, sr, attack_time_ms, attack_time_ms * 2.0)
        slow_env = _smooth_envelope(abs_ch, sr, release_time_ms, release_time_ms)
        transient_comp = np.maximum(fast_env - slow_env, 0.0)
        sustain_comp   = np.minimum(fast_env, slow_env)
        denom = transient_comp + sustain_comp + 1e-9
        attack_gain  = 1.0 + attack_amount  * (transient_comp / denom)
        sustain_gain = 1.0 + sustain_amount * (sustain_comp   / denom)
        return ch * attack_gain * sustain_gain

    if audio.ndim == 1:
        return process_channel(audio)
    return np.stack([process_channel(ch) for ch in audio])

def harmonic_saturation(audio: np.ndarray,
                        drive: float = 0.2, mode: str = "tape",
                        mix: float = 1.0, oversample: int = DEFAULT_DSP_OVERSAMPLE) -> np.ndarray:
    """Saturación armónica (tape/tube/analog) con waveshaper tanh.

    BUGFIX (saturación "extremadamente fuerte/distorsionada"):
    1) k = 1 + drive*10 era demasiado empinado: con drive=0.1 el RMS casi se
       duplicaba y el pico casi tocaba el techo. Ahora k = 1 + drive*5 (rango
       1x..6x), mucho más manejable.
    2) Se normalizaba dividiendo por tanh(k), lo que empuja la señal hacia
       ±1 para CUALQUIER drive > ~0.1, sin importar el nivel de entrada (esto
       es lo que hacía que sonara siempre "al borde del clip"). Ahora se
       normaliza por 'k' (ganancia de pequeña señal: d/dx tanh(kx)|x=0 = k),
       lo que preserva el nivel de la señal y satura solo los picos/transientes,
       comportándose como un saturador real en vez de un limitador disfrazado.
    3) Se agrega oversampling 4x (por defecto) antes del waveshaper no-lineal
       para reducir aliasing/artefactos digitales ásperos, típicos de aplicar
       tanh directamente a la frecuencia de muestreo original.
    """
    if drive <= 0.0 or mix <= 0.0:
        return audio
    k = 1.0 + drive * 5.0

    def shape(x):
        if mode == "tube":
            driven = x * k
            t = np.tanh(driven)
            wet = t - 0.15 * (t ** 2) * np.sign(driven)
        elif mode == "analog":
            # "analog": modela un waveshaper asimétrico tipo consola/cinta
            # analógica combinado (no solo tubo). A diferencia de "tube"
            # (asimetría cuadrática pura, predominantemente 2do armónico) y
            # "tape" (tanh simétrico, predominantemente 3er armónico impar),
            # acá se suman ambos: una pequeña asimetría de 2do armónico
            # (calidez tipo transformador/válvula) MÁS una compresión suave
            # de 3er armónico (saturación de cinta), y una micro-asimetría
            # de bias (offset diminuto pre-shaper) que evita que el 2do
            # armónico se cancele en señales perfectamente simétricas —
            # el resultado es más "gordo"/redondeado que tube o tape solos,
            # sin llegar a la aspereza de un clipper.
            bias = 0.02 * k / 6.0  # bias chico, escala con drive (nunca agresivo)
            driven = x * k + bias
            t = np.tanh(driven)
            even = 0.12 * (t ** 2) * np.sign(driven)   # calidez 2do armónico
            odd = 0.05 * (t ** 3)                       # densidad 3er armónico
            wet = t - even - odd
        else:
            wet = np.tanh(x * k)
        return wet / k  # normaliza por ganancia de pequeña señal, no por tanh(k)

    def process_channel(ch):
        if oversample and oversample > 1:
            up = resample_poly(ch, oversample, 1)
            wet_up = shape(up)
            wet = resample_poly(wet_up, 1, oversample)[:len(ch)]
        else:
            wet = shape(ch)
        return (1.0 - mix) * ch + mix * wet

    if audio.ndim == 1:
        return process_channel(audio)
    return np.stack([process_channel(ch) for ch in audio])


def analyze_harmonic_character(audio: np.ndarray, sr: int) -> dict:
    """Estima el 'carácter' armónico/saturación de un track YA masterizado,
    con dos proxies de DSP clásicos (no hay forma de medir THD real en un
    mix completo polifónico sin la señal seca original):

      - asimetría de forma de onda (pico positivo vs. pico negativo, P99.5
        para ignorar outliers): una saturación con 2do armónico dominante
        (tipo tubo/transformador/consola "analog") rompe la simetría de la
        onda — un semiciclo se comprime más que el otro. Una saturación
        puramente de armónicos impares (tanh simétrico, tipo cinta) preserva
        la simetría.
      - densidad espectral relativa en agudos (energía >4kHz / energía
        total): la saturación agrega contenido armónico nuevo que "rellena"
        el espectro agudo de forma más densa/continua que una señal limpia
        sin saturar.

    No es una medición de laboratorio de THD — es una heurística para
    sugerir un punto de partida razonable de saturation_mode/drive al
    matchear contra una referencia (ver match_saturation_character), no una
    medición científica de distorsión armónica.
    """
    mono = (audio.mean(axis=0) if audio.ndim == 2 else audio).astype(np.float64)

    pos = mono[mono > 0]
    neg = -mono[mono < 0]
    pos_peak = float(np.percentile(pos, 99.5)) if pos.size else 0.0
    neg_peak = float(np.percentile(neg, 99.5)) if neg.size else 0.0
    asymmetry = abs(pos_peak - neg_peak) / (max(pos_peak, neg_peak) + 1e-9)

    n_fft = 8192
    mag = _averaged_magnitude_spectrum(mono, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    total_energy = float(np.sum(mag ** 2) + 1e-12)
    hf_energy = float(np.sum(mag[freqs >= 4000.0] ** 2))
    hf_density = hf_energy / total_energy

    return {
        "asymmetry": round(asymmetry, 4),
        "hf_density": round(hf_density, 5),
    }


def match_saturation_character(audio: np.ndarray, sr: int,
                               own_character: dict, ref_character: dict,
                               max_drive: float = 0.25) -> tuple:  # era 0.5 — muy agresivo para acapellas
    """Sugiere y aplica saturación armónica (harmonic_saturation) para
    acercar el 'carácter' tímbrico del track al de la referencia, a partir
    de analyze_harmonic_character() de ambos.

    Lógica de decisión:
      - Si la referencia tiene bastante MÁS asimetría de onda que el track
        propio -> hay 2do armónico de sobra en la referencia que el track no
        tiene: se sugiere "tube" (asimetría marcada) o "analog" (asimetría
        moderada — combina 2do y 3er armónico, más sutil).
      - Si la referencia no es más asimétrica pero SÍ tiene más densidad
        espectral en agudos -> se sugiere "tape" (saturación de armónicos
        impares, no rompe simetría, "rellena" el espectro sin agregar
        calidez tipo válvula).
      - Igual que match_dynamics_bands con la dinámica: NUNCA se resta
        saturación (no existe forma de "des-saturar" una señal ya
        mezclada/masterizada). Si el track propio ya iguala o supera a la
        referencia en ambos proxies, no se aplica nada.
    """
    own_asym = own_character.get("asymmetry", 0.0)
    ref_asym = ref_character.get("asymmetry", 0.0)
    own_density = own_character.get("hf_density", 0.0)
    ref_density = ref_character.get("hf_density", 0.0)

    asym_gap = ref_asym - own_asym
    density_gap = ref_density - own_density

    meta = {
        "applied": False,
        "own_asymmetry": own_asym, "ref_asymmetry": ref_asym,
        "own_hf_density": round(own_density, 5), "ref_hf_density": round(ref_density, 5),
        "suggested_mode": None, "suggested_drive": 0.0,
    }

    # Umbrales chicos para no perseguir ruido de medición: solo actúa si la
    # brecha es lo bastante clara como para justificar tocar el timbre.
    if asym_gap <= 0.015 and density_gap <= 0.006:
        return audio, meta

    if asym_gap > 0.015:
        mode = "tube" if asym_gap > 0.06 else "analog"
        drive = float(np.clip(asym_gap * 6.0, 0.05, max_drive))
    else:
        mode = "tape"
        drive = float(np.clip(density_gap * 30.0, 0.05, max_drive))

    # mix proporcional a la brecha — nunca más del 50% para preservar la señal original
    # evita que brechas grandes (asym_gap > 0.1) terminen con saturación 100%
    mix = float(np.clip(drive * 1.5, 0.10, 0.50))
    audio_out = harmonic_saturation(audio, drive=drive, mode=mode, mix=mix)
    meta.update({"applied": True, "suggested_mode": mode, "suggested_drive": round(drive, 3),
                 "mix": round(mix, 3)})
    return audio_out, meta


def noise_reduction(
    audio: np.ndarray,
    sr: int,
    strength: float = 0.5,
    noise_sample_sec: float = 0.5,
) -> np.ndarray:
    """Reducción de ruido espectral (hiss, hum, ruido de sala).

    Estima el perfil de ruido a partir de los primeros `noise_sample_sec`
    segundos del track (asumiendo que ahí hay silencio o ruido de fondo),
    luego aplica sustracción espectral suavizada canal por canal.

    Parámetros:
        strength          : 0.0 = sin reducción, 1.0 = máxima agresividad.
        noise_sample_sec  : duración (s) de la zona de muestra de ruido.
                           0.0 usa la primera ventana disponible.

    Diseño:
    - Se usa noisereduce (biblioteca especializada) con prop_decrease=strength.
    - Si noisereduce no está instalado, hace una sustracción espectral manual
      con scipy (fallback sin dependencia extra).
    - Opera canal por canal para no mezclar el perfil de ruido L/R.
    - Preserva exactamente la longitud de la señal de entrada.
    """
    if strength <= 0.0:
        return audio

    strength = float(np.clip(strength, 0.0, 1.0))
    noise_samples = max(1, int(noise_sample_sec * sr))

    def _reduce_channel(ch: np.ndarray) -> np.ndarray:
        try:
            import noisereduce as nr
            noise_clip = ch[:noise_samples]
            return nr.reduce_noise(
                y=ch,
                y_noise=noise_clip,
                sr=sr,
                prop_decrease=strength,
                stationary=False,
                n_std_thresh_stationary=1.5,
            ).astype(np.float32)
        except Exception as e:
            # BUGFIX: antes solo se capturaba ImportError (librería no
            # instalada). Si noisereduce SÍ está instalada pero falla en
            # runtime (incompatibilidad de versión con librosa/numba/scipy)
            # el job entero se caía sin fallback. Ahora cualquier falla de
            # noisereduce cae al mismo fallback manual con scipy, y queda
            # logueado el motivo. Sigue siendo válido en el server cloud
            # (Ubuntu 26.04) aunque ya no haya que lidiar con Windows 7.
            logger.warning(f"noise_reduction: noisereduce falló ({type(e).__name__}: {e}), usando fallback manual con scipy")
            # Fallback: sustracción espectral manual con scipy
            from scipy.signal import stft, istft
            n_fft = 2048
            hop = n_fft // 4
            _, _, Zxx = stft(ch, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            noise_profile = np.mean(np.abs(Zxx[:, :max(1, noise_samples // hop)]), axis=1, keepdims=True)
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            mag_reduced = np.maximum(mag - noise_profile * strength, 0.0)
            _, recovered = istft(mag_reduced * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            # Recortar o padear para preservar longitud exacta
            if len(recovered) >= len(ch):
                return recovered[:len(ch)].astype(np.float32)
            pad = np.zeros(len(ch) - len(recovered), dtype=np.float32)
            return np.concatenate([recovered, pad]).astype(np.float32)

    if audio.ndim == 1:
        return _reduce_channel(audio)
    return np.stack([_reduce_channel(ch) for ch in audio])


def stereo_enhancer(audio: np.ndarray, sr: int,
                    width: float = 1.3, bass_mono_freq: float = 120.0,
                    haas_delay_ms: float = 0.0) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio
    bass_mono_freq = float(np.clip(bass_mono_freq, 20.0, sr / 2.0 - 1.0))
    sos_lp = butter(2, bass_mono_freq, btype="lowpass",  fs=sr, output="sos")
    sos_hp = butter(2, bass_mono_freq, btype="highpass", fs=sr, output="sos")
    left, right = audio[0], audio[1]
    mono_sum = (left + right) * 0.5
    bass = np.stack([sosfiltfilt(sos_lp, mono_sum), sosfiltfilt(sos_lp, mono_sum)])
    high_l = sosfiltfilt(sos_hp, left)
    high_r = sosfiltfilt(sos_hp, right)
    mid  = (high_l + high_r) * 0.5
    side = (high_l - high_r) * 0.5 * width
    if haas_delay_ms > 0.0:
        delay_samples = max(1, int(sr * haas_delay_ms / 1000.0))
        side = np.concatenate([np.zeros(delay_samples), side])[:len(side)]
    highs = np.stack([mid + side, mid - side])
    return bass + highs

def _averaged_magnitude_spectrum(mono: np.ndarray, n_fft: int, max_frames: int = 2000) -> np.ndarray:
    """Espectro de magnitud promediado (método de Welch) sobre TODA la señal,
    no solo un puñado de frames sueltos.

    BUGFIX importante: la versión anterior tomaba `max_frames` frames (32 a 64)
    de n_fft muestras cada uno —apenas 1 a 6 segundos reales— y promediaba
    SOLO esos puntos, aunque estuvieran "repartidos" a lo largo del archivo.
    Para un tema de 3-4 minutos eso es <2% del audio: si algún frame caía
    justo sobre un silencio, una pausa vocal o un golpe de bombo, el balance
    espectral resultante podía salir completamente distinto entre dos
    análisis del mismo tema (o entre tema y referencia), y eso es lo que se
    veía como "el análisis da cualquier resultado". Welch promedia la energía
    de TODOS los frames que entran en la señal (con ventana Hann y 75% de
    solapamiento), así que el resultado usa el 100% del audio y es estable
    y repetible. `max_frames` ahora solo actúa como límite de cómputo para
    archivos larguísimos (agranda el hop, pero sigue cubriendo todo el
    archivo de punta a punta en vez de recortarlo).
    """
    n = len(mono)
    nperseg = int(min(n_fft, max(8, n)))
    if n <= nperseg:
        window = np.hanning(nperseg)
        frame = np.zeros(nperseg)
        frame[:n] = mono[:n]
        mag = np.abs(np.fft.rfft(frame * window))
        if nperseg < n_fft:
            target_len = n_fft // 2 + 1
            mag = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(mag)), mag)
        return mag
    hop = max(1, nperseg // 4)
    n_segments = (n - nperseg) // hop + 1
    if n_segments > max_frames:
        hop = max(nperseg // 4, (n - nperseg) // max_frames + 1)
    noverlap = max(0, nperseg - hop)
    _freqs, psd = welch(mono, fs=1.0, window="hann", nperseg=nperseg,
                         noverlap=noverlap, scaling="spectrum", detrend=False,
                         average="mean")
    return np.sqrt(np.maximum(psd, 0.0))


def _log_band_average_db(freqs: np.ndarray, mag: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Promedia la magnitud del espectro (en energía) dentro de cada banda
    logarítmica definida por `edges`, devolviendo el resultado en dB.

    BUGFIX importante: con binning log puro, las bandas graves (p.ej. entre
    20 y 40Hz) suelen ser más angostas que la resolución real de la FFT
    (a 44.1kHz/4096 puntos, cada bin de FFT son ~10.8Hz). Eso hacía que
    varias bandas de salida no tuvieran NINGÚN bin de FFT adentro, y el
    código anterior les asignaba un piso arbitrario de -180dB — lo que se
    veía como dientes de sierra sin sentido en el extremo grave del gráfico
    (exactamente el "espectro que es cualquier cosa" / falta de una curva
    de respuesta en frecuencia coherente). Ahora, si una banda no tiene
    ningún bin de FFT adentro, se interpola en log-frecuencia a partir de
    los bins vecinos reales, en vez de inventar un valor.
    """
    log_freqs = np.log10(np.maximum(freqs, 1e-6))
    mag_db_full = 20.0 * np.log10(mag + 1e-9)
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            energy = float(np.sqrt(np.mean(mag[mask] ** 2)))
            out[i] = 20.0 * np.log10(energy + 1e-9)
        else:
            center = 0.5 * (lo + hi)
            out[i] = float(np.interp(np.log10(max(center, 1e-6)), log_freqs, mag_db_full))
    return out

def spectrum_analysis_fft(audio: np.ndarray, sr: int,
                          n_fft: int = 4096, n_bins: int = 64,
                          _avg_mag: np.ndarray = None, _freqs: np.ndarray = None) -> dict:
    # MEJORA (perf): analyze_audio() ya calcula el espectro promediado
    # (Welch, sobre el archivo entero — la parte cara de esta función) para
    # el campo "spectrum" de 7 bandas, con el mismo n_fft. Si lo recibe acá
    # via _avg_mag/_freqs se reusa en vez de recalcularlo de cero para
    # "fft_spectrum" (64 bins), que es exactamente el mismo espectro con
    # más resolución de bins. Los parámetros quedan opcionales para no
    # romper otros usos directos de esta función fuera de analyze_audio.
    if _avg_mag is not None and _freqs is not None:
        avg_mag, freqs = _avg_mag, _freqs
    else:
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if len(mono) < n_fft:
            n_fft = max(64, 2 ** int(np.floor(np.log2(max(len(mono), 64)))))
        avg_mag = _averaged_magnitude_spectrum(mono, n_fft)
        freqs   = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    nyquist  = sr / 2.0
    log_edges = np.logspace(np.log10(20.0), np.log10(nyquist), n_bins + 1)
    bin_db   = _log_band_average_db(freqs, avg_mag, log_edges).round(2).tolist()
    bin_freq = [round(float((log_edges[i] + log_edges[i + 1]) * 0.5), 1) for i in range(n_bins)]
    return {"frequencies_hz": bin_freq, "magnitudes_db": bin_db, "n_fft": n_fft}

# ─── Análisis ──────────────────────────────────────────────────────────────────

def stereo_correlation(audio: np.ndarray) -> float:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return 1.0
    l, r = audio[0], audio[1]
    std_l, std_r = np.std(l), np.std(r)
    if std_l < 1e-9 or std_r < 1e-9:
        return 1.0
    corr = float(np.mean((l - l.mean()) * (r - r.mean())) / (std_l * std_r))
    return float(np.clip(corr, -1.0, 1.0))

def short_term_loudness_and_lra(audio: np.ndarray, sr: int,
                                block_s: float = 3.0, hop_s: float = 1.0) -> tuple:
    """Combina `short_term_loudness_stats` + `measure_lra` en un único pase
    (perf): ambas calculan el MISMO sliding-window de RMS-en-dB (mismo
    block_s/hop_s, mismo gate de -70dB) por separado cuando se usan juntas
    (como en `analyze_audio`, siempre con los valores por defecto). Devuelve
    (loudness_short_term_dict, lra_float), bit-idéntico a llamar ambas
    funciones por separado con los mismos block_s/hop_s.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    block = max(1, int(sr * block_s))
    hop = max(1, int(sr * hop_s))
    if len(mono) < block:
        rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
        db = round(float(20.0 * np.log10(rms)), 2)
        return {"max": db, "min": db, "p95": db}, 0.0

    levels = []
    for start in range(0, len(mono) - block, hop):
        seg = mono[start:start + block]
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        db = 20.0 * np.log10(rms)
        if db > -70.0:
            levels.append(db)

    if not levels:
        return {"max": -70.0, "min": -70.0, "p95": -70.0}, 0.0

    arr = np.array(levels)
    st_loudness = {
        "max": round(float(np.max(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }
    if len(arr) < 2:
        return st_loudness, 0.0
    rel_gate = np.mean(arr) - 20.0
    gated = arr[arr > rel_gate]
    if len(gated) < 2:
        gated = arr
    lra = round(float(np.percentile(gated, 95) - np.percentile(gated, 10)), 2)
    return st_loudness, lra


def measure_lra(audio: np.ndarray, sr: int, block_s: float = 3.0, hop_s: float = 1.0) -> float:
    """Loudness Range (LRA) simplificado, estilo EBU R128: RMS por ventanas
    deslizantes (bloques de 3s, hop 1s) en dB, con gate absoluto (-70 dB) y
    gate relativo (descarta bloques 20 dB por debajo de la media) antes de
    tomar el rango entre percentiles 10 y 95. No es una implementación
    100% conforme al estándar (no usa K-weighting ni el gate de 2 pasadas
    exacto), pero es una proxy robusta y estable para comparar macro-dinámica
    entre dos tracks.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    block = max(1, int(sr * block_s))
    hop   = max(1, int(sr * hop_s))
    if len(mono) < block:
        return 0.0
    levels = []
    for start in range(0, len(mono) - block, hop):
        seg = mono[start:start + block]
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        db  = 20.0 * np.log10(rms)
        if db > -70.0:
            levels.append(db)
    if len(levels) < 2:
        return 0.0
    levels = np.array(levels)
    rel_gate = np.mean(levels) - 20.0
    gated = levels[levels > rel_gate]
    if len(gated) < 2:
        gated = levels
    lra = float(np.percentile(gated, 95) - np.percentile(gated, 10))
    return round(lra, 2)

# Bandas usadas para análisis/matching de dinámica y estéreo "inteligentes"
# (más anchas que las 7 bandas de `spectrum`, pensadas para separar
# graves / medios / agudos de forma robusta al filtrar).
DYNAMICS_BANDS = [("low", 20.0, 150.0), ("mid", 150.0, 2500.0), ("high", 2500.0, 20000.0)]

def _bandpass_filter(audio: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    lo = float(np.clip(lo, 1.0, sr / 2.0 - 10.0))
    hi = float(np.clip(hi, lo + 5.0, sr / 2.0 - 1.0))
    sos = butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    return np.stack([sosfiltfilt(sos, ch) for ch in audio])

def band_crest_factors(audio: np.ndarray, sr: int, bands: list = DYNAMICS_BANDS) -> dict:
    """Crest factor (peak - RMS, en dB) por banda de frecuencia. Permite
    comparar la dinámica de graves/medios/agudos entre dos tracks por
    separado, en vez de un único crest factor de banda ancha (que puede
    esconder, por ejemplo, graves muy comprimidos con agudos muy dinámicos).
    """
    out = {}
    for name, lo, hi in bands:
        filtered = _bandpass_filter(audio, sr, lo, hi)
        mono = filtered.mean(axis=0) if filtered.ndim == 2 else filtered
        rms  = float(np.sqrt(np.mean(mono ** 2))) + 1e-9
        peak = float(np.max(np.abs(mono))) + 1e-9
        out[name] = float(20.0 * np.log10(peak) - 20.0 * np.log10(rms))
    return out

def band_stereo_correlation(audio: np.ndarray, sr: int, bands: list = DYNAMICS_BANDS) -> dict:
    """Correlación L/R por banda de frecuencia. Los masters comerciales suelen
    tener graves casi mono (correlación ~1) y agudos más anchos (correlación
    más baja); comparar banda por banda contra la referencia permite un
    matching de estéreo mucho más fiel que un único ancho global.
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        return {name: 1.0 for name, _, _ in bands}
    out = {}
    for name, lo, hi in bands:
        filtered = _bandpass_filter(audio, sr, lo, hi)
        out[name] = stereo_correlation(filtered)
    return out

def band_crest_and_stereo(audio: np.ndarray, sr: int, bands: list = DYNAMICS_BANDS) -> tuple:
    """Crest factor + correlación estéreo por banda, calculados en un único
    pase (perf): `band_crest_factors` y `band_stereo_correlation` filtran el
    MISMO audio con las MISMAS bandas por separado cuando se usan juntas
    (como en `analyze_audio`), duplicando 3 filtros bandpass zero-phase
    (sosfiltfilt, caros: forward+backward por canal) que solo hace falta
    calcular una vez. Devuelve (crest_dict, stereo_corr_dict), idénticos a
    llamar ambas funciones por separado.
    """
    is_stereo = audio.ndim == 2 and audio.shape[0] == 2
    crest, corr = {}, {}
    for name, lo, hi in bands:
        filtered = _bandpass_filter(audio, sr, lo, hi)
        mono = filtered.mean(axis=0) if filtered.ndim == 2 else filtered
        rms  = float(np.sqrt(np.mean(mono ** 2))) + 1e-9
        peak = float(np.max(np.abs(mono))) + 1e-9
        crest[name] = float(20.0 * np.log10(peak) - 20.0 * np.log10(rms))
        corr[name] = stereo_correlation(filtered) if is_stereo else 1.0
    return crest, corr

def true_peak_dbfs(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Pico real (true peak) sobresampleado, mismo criterio que usa `limiter()`
    para detectar inter-sample peaks que el pico a sample-rate nativa no ve."""
    ovs = max(1, int(oversample))
    chans = audio if audio.ndim == 2 else audio[np.newaxis, :]
    peak = 0.0
    for ch in chans:
        up = resample_poly(ch, ovs, 1) if ovs > 1 else ch
        if len(up):
            peak = max(peak, float(np.max(np.abs(up))))
    return float(20.0 * np.log10(peak + 1e-9))


def measure_dc_offset(audio: np.ndarray) -> float:
    chans = audio if audio.ndim == 2 else audio[np.newaxis, :]
    return round(float(np.max(np.abs([np.mean(ch) for ch in chans]))), 5)


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    if len(mono) == 0:
        return 0.0
    return round(float(np.mean(np.abs(mono) >= threshold)), 5)


def silence_ratio(audio: np.ndarray, sr: int, threshold_db: float = -60.0, block_s: float = 0.05) -> float:
    """Fracción del track por debajo de `threshold_db` (bloques de 50ms): útil para
    detectar silencios/fades largos que pueden estar distorsionando el LUFS integrado."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    block = max(1, int(sr * block_s))
    n_blocks = len(mono) // block
    if n_blocks == 0:
        return 0.0
    trimmed = mono[:n_blocks * block].reshape(n_blocks, block)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    return round(float(np.mean(db < threshold_db)), 4)


def short_term_loudness_stats(audio: np.ndarray, sr: int, block_s: float = 3.0, hop_s: float = 1.0) -> dict:
    """Estadísticas de loudness de corto plazo (ventanas de 3s, estilo momentary/
    short-term de EBU R128): máx, mín y p95. El LUFS integrado promedia todo el
    track y puede esconder picos de loudness momentáneo que sí se escuchan."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    block = max(1, int(sr * block_s))
    hop = max(1, int(sr * hop_s))
    if len(mono) < block:
        rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
        db = round(float(20.0 * np.log10(rms)), 2)
        return {"max": db, "min": db, "p95": db}
    levels = []
    for start in range(0, len(mono) - block, hop):
        seg = mono[start:start + block]
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        db = 20.0 * np.log10(rms)
        if db > -70.0:
            levels.append(db)
    if not levels:
        return {"max": -70.0, "min": -70.0, "p95": -70.0}
    arr = np.array(levels)
    return {
        "max": round(float(np.max(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def spectral_shape_features(mono: np.ndarray, sr: int) -> dict:
    """Centroid, rolloff, flatness y zero-crossing rate: describen el 'brillo',
    la energía relativa en agudos y qué tan tonal (vs. ruidosa/percusiva) es la señal.

    MEJORA (perf): spectral_centroid, spectral_rolloff y spectral_flatness
    calculan las tres, por separado, la MISMA magnitud de STFT (|stft|, con
    los mismos n_fft/hop_length) cuando no se les pasa el parámetro `S=`
    explícitamente — son 3 STFTs idénticos tirados a la basura. Acá se
    calcula esa magnitud UNA sola vez (con los mismos parámetros por
    defecto que usarían internamente: window='hann', center=True) y se
    reusa vía `S=` en las tres, que es exactamente el patrón que la propia
    documentación de librosa recomienda para este caso.

    BUGFIX (Numba SIGSEGV en Python 3.14): zero_crossing_rate de librosa usa
    internamente un kernel Numba que, en este host, segfaultea al ejecutarse
    dentro de multiprocessing.spawn (ver dmesg: "segfault at 0 ip 0x0 ...
    numba.np.ufunc._internal"). Reemplazamos esa llamada por una
    implementación NumPy pura que produce exactamente el mismo resultado:
    número de cruces por cero por frame, normalizado por la longitud del
    frame. La fórmula es: zcr[n] = (1/(2*frame_length)) * sum|x[m] - x[m-1]|,
    que es la definición estándar (Bakis, 1962).
    """
    try:
        n_fft = min(4096, max(256, len(mono)))
        n_fft = max(64, 2 ** int(np.floor(np.log2(n_fft))))
        hop = max(1, n_fft // 4)
        S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop, window="hann", center=True))
        centroid = librosa.feature.spectral_centroid(sr=sr, S=S, n_fft=n_fft, hop_length=hop)
        rolloff  = librosa.feature.spectral_rolloff(sr=sr, S=S, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(S=S, n_fft=n_fft, hop_length=hop)
        zcr = _zero_crossing_rate_numpy(mono, frame_length=n_fft, hop_length=hop)
        return {
            "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
            "spectral_rolloff_hz":  round(float(np.mean(rolloff)), 1),
            "spectral_flatness":    round(float(np.mean(flatness)), 4),
            "zero_crossing_rate":   round(float(np.mean(zcr)), 4),
        }
    except Exception:
        return {"spectral_centroid_hz": 0.0, "spectral_rolloff_hz": 0.0,
                "spectral_flatness": 0.0, "zero_crossing_rate": 0.0}


def _zero_crossing_rate_numpy(
    mono: np.ndarray, frame_length: int = 2048, hop_length: int = 512,
) -> np.ndarray:
    """Zero crossing rate implementado en NumPy puro.

    Reemplaza ``librosa.feature.zero_crossing_rate``, que en este host
    segfaultea al pasar por su kernel Numba en Python 3.14. Devuelve un
    array 1xN_frames con la fracción de cruces por cero por frame, igual
    que la versión de librosa (incluye el factor 1/2 que la documentación
    de librosa documenta: ``1/(2*frame_length) * sum|x[m] - x[m-1]|``)."""
    if mono.ndim > 1:
        mono = mono.mean(axis=0)
    if mono.size < frame_length:
        return np.array([[0.0]])
    n_frames = 1 + (mono.size - frame_length) // hop_length
    indices = np.arange(n_frames)[:, None] * hop_length + np.arange(frame_length)
    frames = mono[indices]
    crossings = np.abs(np.diff(frames, axis=1))
    zcr = (crossings > 0).sum(axis=1) / np.maximum(frame_length - 1, 1)
    return zcr[np.newaxis, :]


def transient_density(mono: np.ndarray, sr: int) -> float:
    """Onsets (ataques/transientes) por segundo: proxy de qué tan percusivo/denso
    es el material (trap/metal >> ambient/balada), usado para dosificar el transient
    shaper y el attack del compresor/limiter."""
    try:
        onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="time")
        dur = max(len(mono) / sr, 1e-6)
        return round(float(len(onsets) / dur), 3)
    except Exception:
        return 0.0


def mono_compatibility_db(audio: np.ndarray) -> float:
    """Diferencia de nivel (dB) entre sumar L+R a mono y el nivel estéreo original.
    Valores muy negativos indican cancelación de fase al sumar a mono (típico de
    fase invertida o ancho estéreo artificial excesivo) — un problema serio para
    reproducción en sistemas mono (clubs, TV, bluetooth speakers)."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        return 0.0
    l, r = audio[0], audio[1]
    mono_sum = (l + r) * 0.5
    stereo_rms = float(np.sqrt(np.mean(((l ** 2) + (r ** 2)) / 2.0)) + 1e-9)
    mono_rms = float(np.sqrt(np.mean(mono_sum ** 2)) + 1e-9)
    return round(float(20.0 * np.log10(mono_rms / stereo_rms)), 2)


def analyze_audio(audio: np.ndarray, sr: int) -> dict:
    """Extrae varias decenas de métricas técnicas del track: loudness (integrado,
    corto plazo, LRA), picos (sample y true peak), dinámica global y por banda,
    forma espectral (7 bandas + 28 bandas finas + centroid/rolloff/flatness/ZCR),
    estéreo (correlación global y por banda, compatibilidad mono), higiene de
    señal (DC offset, clipping, silencio) y densidad de transientes. Es el
    "input sensorial" que usa Laia (el asistente de auto-mastering) para
    diagnosticar el track antes de decidir la cadena DSP.

    MEJORA (precisión/consistencia): antes esta función calculaba su propio
    espectro de 7 bandas con n_fft tope de 4096 (~10.8Hz/bin a 44.1kHz),
    mientras que spectral_energy_at_bands() —la función que arma la curva de
    EQ real del reference matching— usaba n_fft tope de 8192 (~5.4Hz/bin) y
    28 bandas. Eran dos mediciones INDEPENDIENTES del mismo audio, con
    distinta resolución y distinta cantidad de bandas: lo que Laia/el reporte
    mostraban como "balance espectral" no era necesariamente lo mismo que el
    EQ de matching había medido y corregido. Ahora:
      1) n_fft sube a 8192 (mismo tope que spectral_energy_at_bands) — mejor
         resolución en graves, especialmente sub_bass (20-80Hz).
      2) Se agrega "spectrum_bands_db": 28 bandas log finas (mismo band-count
         que eq_bands por defecto en el reference matching), con freq center
         de cada banda — la MISMA resolución que ve el EQ real.
      3) "spectrum" (7 bandas legibles) se sigue calculando directo sobre
         freqs/avg_fft con sus edges exactos (2000/4000/8000Hz etc.), igual
         que antes — se probó agregarlo desde las 28 bandas finas en vez de
         calcularlo aparte, pero los edges de las 28 no caen justo en esos
         cortes y eso metía fuga de hasta 0.2-0.4dB entre bandas vecinas. Se
         descartó: los dos sets comparten el mismo avg_fft/freqs (mismo
         n_fft=8192), que es la unificación que importaba — no hace falta
         forzar que uno derive del otro para que sean consistentes.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    rms  = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))
    rms_db  = float(20.0 * np.log10(rms  + 1e-9))
    peak_db = float(20.0 * np.log10(peak + 1e-9))
    lufs    = measure_lufs_integrated(audio, sr)
    dynamic_range_db = float(peak_db - rms_db)
    true_peak_db = true_peak_dbfs(audio, sr)

    n_fft = min(8192, len(mono))
    n_fft = max(64, 2 ** int(np.floor(np.log2(n_fft))))
    avg_fft = _averaged_magnitude_spectrum(mono, n_fft)
    freqs   = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    bands = {
        "sub_bass":  (20,    80),
        "bass":      (80,    250),
        "low_mid":   (250,   500),
        "mid":       (500,   2000),
        "upper_mid": (2000,  4000),
        "presence":  (4000,  8000),
        "air":       (8000,  20000),
    }
    band_edges = [bands["sub_bass"][0]] + [hi for _, hi in bands.values()]

    # ── 28 bandas finas log-espaciadas, 20Hz-nyquist — mismo band-count que
    # eq_bands por defecto en el reference matching (ver docstring arriba).
    # Se calcula directo sobre freqs/avg_fft (los mismos bins de FFT que usa
    # el resumen de 7 bandas de abajo) — mismo n_fft=8192, misma fuente de
    # datos, así ambos quedan consistentes en resolución. (Nota: agregar el
    # resumen de 7 bandas A PARTIR de estas 28 en vez de calcularlo directo
    # se probó y descartó — los edges de las 28 bandas log no caen justo en
    # los cortes de las 7 bandas legibles [2000/4000/8000Hz], así que
    # promediar por ese camino mete fuga entre bandas vecinas de hasta
    # 0.2-0.4dB. Calcular cada set directo sobre los bins de FFT da el
    # resultado exacto en los dos, y de todas formas comparten la misma
    # resolución de origen.)
    N_FINE_BANDS = 28
    nyquist = sr / 2.0
    fine_edges = np.logspace(np.log10(20.0), np.log10(max(nyquist - 1.0, 21.0)), N_FINE_BANDS + 1)
    fine_db = _log_band_average_db(freqs, avg_fft, fine_edges)
    fine_centers = [float(np.sqrt(fine_edges[i] * fine_edges[i + 1])) for i in range(N_FINE_BANDS)]
    spectrum_bands_db = [
        {"freq_hz": round(fc, 1), "db": round(float(db), 2)}
        for fc, db in zip(fine_centers, fine_db)
    ]

    band_db = _log_band_average_db(freqs, avg_fft, np.array(band_edges, dtype=float))
    spectrum = {name: round(float(db), 2) for name, db in zip(bands.keys(), band_db)}

    shape = spectral_shape_features(mono, sr)
    st_loudness, _lra = short_term_loudness_and_lra(audio, sr)
    _band_crest, _band_corr = band_crest_and_stereo(audio, sr)
    short_stats = short_term_loudness_stats(audio, sr)


    result = {
        # ── Loudness / nivel ────────────────────────────────────────────
        "rms_db":             round(rms_db, 2),
        "peak_db":            round(peak_db, 2),
        "true_peak_db":       round(true_peak_db, 2),
        "lufs":               round(lufs, 2),
        "plr_db":             round(true_peak_db - lufs, 2),  # peak-to-loudness ratio
        "loudness_short_term": st_loudness,
        "lra":                round(float(_lra), 2),
        # ── Dinámica ─────────────────────────────────────────────────────
        "dynamic_range_db":   round(dynamic_range_db, 2),
        "crest_factor_db":    round(dynamic_range_db, 2),
        "band_dynamics_db":   {k: round(v, 2) for k, v in _band_crest.items()},
        # ── Higiene de señal ────────────────────────────────────────────
        "dc_offset":          measure_dc_offset(audio),
        "clipping_ratio":     clipping_ratio(audio),
        "silence_ratio":      silence_ratio(audio, sr),
        # ── Estéreo ──────────────────────────────────────────────────────
        "stereo_correlation": round(stereo_correlation(audio), 3),
        "band_stereo_correlation": {k: round(v, 3) for k, v in _band_corr.items()},
        "mono_compatibility_db": mono_compatibility_db(audio),
        # ── Forma espectral / timbre ────────────────────────────────────
        "spectral_centroid_hz": shape["spectral_centroid_hz"],
        "spectral_rolloff_hz":  shape["spectral_rolloff_hz"],
        "spectral_flatness":    shape["spectral_flatness"],
        "zero_crossing_rate":   shape["zero_crossing_rate"],
        # ── Ritmo ────────────────────────────────────────────────────────
        "transient_density":  transient_density(mono, sr),
        "short_term_stats":   short_stats,
        # ── Formato ──────────────────────────────────────────────────────
        "sample_rate":        sr,
        "channels":           1 if audio.ndim == 1 else audio.shape[0],
        "duration_sec":       round(len(mono) / sr, 2),
        # ── Espectro ─────────────────────────────────────────────────────
        "spectrum":           spectrum,
        "spectrum_bands_db":  spectrum_bands_db,
        "fft_spectrum":       spectrum_analysis_fft(audio, sr, n_fft=n_fft, _avg_mag=avg_fft, _freqs=freqs),
        # ── Resonancias / sibilancia (para Dynamic EQ / de-esser) ────────
        "resonances":         detect_resonances(audio, sr),
        "sibilance":          detect_sibilance(audio, sr),
    }

    # ── ANÁLISIS PERCEPTUAL (Los "oídos") ──────────────────────────
    # El perfil perceptual depende de las métricas recién calculadas; primero
    # construimos el resultado base y recién después lo pasamos a los "oídos".
    try:
        from .ai_assistant import _analyze_perceptual_profile, _get_genre_from_perceptual, _get_perceptual_diagnosis
    except ImportError:
        from ai_assistant import _analyze_perceptual_profile, _get_genre_from_perceptual, _get_perceptual_diagnosis

    perceptual_profile = _analyze_perceptual_profile(result)
    genre, genre_conf = _get_genre_from_perceptual(perceptual_profile, result)
    result["perceptual"] = perceptual_profile.to_dict()
    result["genre_detected"] = genre
    result["genre_confidence"] = genre_conf
    result["perceptual_diagnosis"] = _get_perceptual_diagnosis(perceptual_profile)

    return result

# ─── Consejos de mezcla ───────────────────────────────────────────────────────

def mix_advice(analysis: dict) -> dict:
    issues = []
    tips   = []
    score  = 100

    lufs  = analysis.get("lufs", -99)
    peak  = analysis.get("peak_db", -99)
    true_peak = analysis.get("true_peak_db", peak)
    dyn   = analysis.get("dynamic_range_db", 0)
    spec  = analysis.get("spectrum", {})
    clip_ratio = analysis.get("clipping_ratio", 0.0)
    mono_compat = analysis.get("mono_compatibility_db", 0.0)
    dc = analysis.get("dc_offset", 0.0)

    if lufs > -6:
        issues.append("Muy alto en loudness (>-6 LUFS): posible over-compression o clipping.")
        tips.append("Reducí el limiter ceiling o bajá el makeup gain del compresor.")
        score -= 20
    elif lufs > -9:
        issues.append("Loudness algo alto (-9 a -6 LUFS): al límite para streaming.")
        tips.append("Apuntá a -14 LUFS para Spotify/YouTube o -9 LUFS para club.")
        score -= 8
    elif lufs < -24:
        issues.append("Loudness muy bajo (<-24 LUFS): la mezcla se va a escuchar muy quieta.")
        tips.append("Subí el nivel general o usá normalización LUFS.")
        score -= 15
    elif lufs < -18:
        tips.append("Loudness moderadamente bajo. Podés subir con normalización LUFS a -14.")
        score -= 5

    if peak >= 0:
        issues.append("¡Clipping detectado! El pico llega a 0 dBFS.")
        tips.append("Bajá el nivel de salida o usá un limitador con ceiling más bajo.")
        score -= 25
    elif peak > -0.5:
        issues.append("Pico muy cerca de 0 dBFS (< -0.5 dB de margen).")
        tips.append("Dejá al menos 1 dB de headroom. Usá limiter ceiling = 0.95.")
        score -= 10

    if dyn < 4:
        issues.append("Rango dinámico muy comprimido (< 4 dB): mezcla 'brick-wall'.")
        tips.append("Reducí el ratio del compresor o subí el threshold.")
        score -= 18
    elif dyn < 7:
        issues.append("Rango dinámico algo limitado (4–7 dB).")
        tips.append("Probá ratio 2:1–3:1 para un resultado más natural.")
        score -= 8
    elif dyn > 25:
        tips.append("Rango dinámico muy amplio (>25 dB): podría sonar inconsistente.")
        score -= 5

    sub  = spec.get("sub_bass", -99)
    bass = spec.get("bass", -99)
    mid  = spec.get("mid", -99)
    air  = spec.get("air", -99)
    pres = spec.get("presence", -99)

    if sub > bass + 10:
        issues.append("Sub-bajos excesivos vs. bajos medios: sonido 'boomy'.")
        tips.append("Usá high-pass en 60–80 Hz y reducí EQ en 40–60 Hz.")
        score -= 12
    if bass > mid + 20:
        issues.append("Bajos dominan sobre los medios: mezcla oscura.")
        tips.append("Realzá medios (500 Hz–2 kHz) o reducí bajos en 100–200 Hz.")
        score -= 10
    if air < mid - 30:
        issues.append("Altas frecuencias muy bajas: la mezcla puede sonar apagada.")
        tips.append("Subí el high shelf (+2 a +4 dB a partir de 8 kHz).")
        score -= 8
    if pres > mid + 15:
        issues.append("Zona de presencia muy prominente (4–8 kHz): puede ser fatigante.")
        tips.append("Reducí presencia con EQ en 4–6 kHz, Q=1.5.")
        score -= 8

    if clip_ratio > 0.0005:
        issues.append(f"Clipping real detectado en {round(clip_ratio * 100, 2)}% de las muestras.")
        tips.append("Bajá el input gain o el ceiling del limiter antes de re-renderizar.")
        score -= 15
    elif true_peak > -0.3:
        issues.append(f"True peak muy cerca de 0 dBFS ({true_peak} dBTP): riesgo de inter-sample clipping en conversores D/A.")
        tips.append("Dejá al menos 1 dB de margen: usá limiter ceiling ≤ 0.94 (~-0.5 dBTP).")
        score -= 8

    if mono_compat < -3.0:
        issues.append(f"Baja compatibilidad mono ({mono_compat} dB de pérdida al sumar L+R): posible cancelación de fase.")
        tips.append("Revisá la fase del estéreo o reducí el ancho estéreo (stereo_width_amount) antes de mastering.")
        score -= 10

    if dc and abs(dc) > 0.01:
        issues.append(f"DC offset detectado ({dc}): puede reducir headroom disponible.")
        tips.append("Aplicá un filtro DC-block/high-pass muy bajo (< 20 Hz) antes de la cadena de mastering.")
        score -= 5

    if not issues:
        tips.insert(0, "¡La mezcla se ve bien técnicamente! Revisá en referencia con tracks similares.")

    score = max(0, min(100, score))
    grade = "Excelente" if score >= 85 else "Buena" if score >= 70 else "Aceptable" if score >= 50 else "Necesita trabajo"
    return {"score": score, "grade": grade, "issues": issues, "tips": tips}


def diagnostic_advice(analysis: dict) -> dict:
    """Convierte métricas de análisis en una hipótesis técnica operativa.

    No intenta adivinar "qué perilla tocar". Primero clasifica el síntoma,
    propone una hipótesis principal y ordena comprobaciones de menor riesgo
    a mayor intervención. El resultado está pensado como base para un
    futuro árbol interactivo de troubleshooting de audio en vivo.
    """
    checks = []
    hypotheses = []

    def add(priority, area, finding, action, severity, evidence=None):
        checks.append({
            "priority": priority,
            "area": area,
            "finding": finding,
            "next_check": action,
            "severity": severity,
            "evidence": evidence or [],
        })

    def hypothesis(name, confidence, reason, first_check):
        hypotheses.append({
            "name": name,
            "confidence": round(float(confidence), 2),
            "reason": reason,
            "first_check": first_check,
        })

    rms = float(analysis.get("rms_db", -99))
    peak = float(analysis.get("peak_db", -99))
    true_peak = float(analysis.get("true_peak_db", -99))
    clipping = float(analysis.get("clipping_ratio", 0.0))
    silence = float(analysis.get("silence_ratio", 0.0))
    dc = abs(float(analysis.get("dc_offset", 0.0) or 0.0))
    mono_compat = float(analysis.get("mono_compatibility_db", 0.0))
    corr = float(analysis.get("stereo_correlation", 1.0))
    dyn = float(analysis.get("dynamic_range_db", 99))

    # 1) Señal prácticamente ausente: antes de tocar EQ/comp, seguir la ruta.
    no_signal = (rms < -60 and peak < -50) or silence >= 0.90
    if no_signal:
        hypothesis(
            "Señal ausente o casi ausente", 0.96,
            "El material analizado contiene muy poca energía útil.",
            "Seguí la ruta de señal desde la fuente hasta la entrada y confirmá que el canal/bus esté recibiendo audio.",
        )
        add(1, "Ruta de señal", "La señal útil es prácticamente inexistente.",
            "Comprobá fuente, cableado, patch, entrada, mute, gain y routing antes de insertar procesamiento.",
            "critical", [f"rms_db={rms}", f"peak_db={peak}", f"silence_ratio={silence}"])

    # 2) Saturación: es el primer problema si está ocurriendo realmente.
    if clipping > 0.0005 or true_peak >= 0:
        hypothesis(
            "Saturación / gain staging", 0.98,
            "Hay evidencia directa de clipping o true peak fuera de margen.",
            "Verificá primero la ganancia antes del procesamiento y la etapa que ya está saturando.",
        )
        add(1, "Nivel / clipping", "Hay evidencia de saturación digital.",
            "Verificá primero ganancia de entrada, estructura de gain y ceiling antes de procesar.",
            "critical", [f"clipping_ratio={clipping}", f"true_peak_db={true_peak}"])

    if dc > 0.01:
        hypothesis("DC offset / higiene de señal", 0.90,
                   "La señal tiene un desplazamiento de continua apreciable.",
                   "Aislá la fuente o interfaz que lo introduce y verificá la entrada antes de seguir procesando.")
        add(2, "Higiene de señal", "Se detecta DC offset.",
            "Revisá la fuente, interfaces y etapas anteriores; luego aplicá DC-block/high-pass bajo.",
            "high", [f"dc_offset={dc}"])

    if mono_compat < -6:
        hypothesis("Fase / suma mono", 0.94,
                   "La suma mono pierde demasiado nivel para considerarla un simple problema tonal.",
                   "Compará L/R, polaridad y alineación temporal antes de tocar EQ o compresión.")
        add(2, "Fase / estéreo", "La suma mono pierde mucho nivel.",
            "Comprobá polaridad, alineación temporal y procesamiento estéreo antes de EQ o compresión.",
            "high", [f"mono_compatibility_db={mono_compat}"])

    if true_peak > -0.5:
        add(3, "Headroom", "El true peak está demasiado cerca de 0 dBFS.",
            "Dejá margen digital antes de la siguiente etapa o conversión.",
            "medium", [f"true_peak_db={true_peak}"])

    if dyn < 4 and not no_signal:
        add(3, "Dinámica", "La señal está muy comprimida.",
            "Escuchá bypass y revisá gain staging, threshold, ratio y limiter antes de agregar más control dinámico.",
            "medium", [f"dynamic_range_db={dyn}"])
        hypothesis("Exceso de control dinámico", 0.80,
                   "El contraste pico/RMS es muy reducido para material con dinámica natural.",
                   "Escuchá bypass del compresor/limitador y comprobá dónde empieza a cerrarse la dinámica.")

    if corr < 0.5 and mono_compat >= -6:
        add(4, "Estéreo", "La correlación global es baja.",
            "Aislá mono/stereo por etapas para encontrar qué fuente o procesamiento abre demasiado el campo.",
            "medium", [f"stereo_correlation={corr}"])

    if silence > 0.2 and not no_signal:
        add(4, "Señal / contenido", "Hay una proporción importante de silencio.",
            "Confirmá que no sea una toma incompleta, un gate excesivo o un problema de ruteo.",
            "medium", [f"silence_ratio={silence}"])

    try:
        from .diagnostic_knowledge import memory_summary, match_knowledge
    except ImportError:
        from diagnostic_knowledge import memory_summary, match_knowledge

    field_memory = memory_summary(analysis)
    # La memoria de cancha puede agregar hipótesis operativas que el conjunto
    # de métricas todavía no expresaba como regla explícita. No reemplaza los
    # checks DSP: los complementa con ruta, primera comprobación y "qué no tocar".
    legacy_names = {
        "no_signal": "Señal ausente o casi ausente",
        "clipping": "Saturación / gain staging",
        "phase": "Fase / suma mono",
        "dc_offset": "DC offset / higiene de señal",
        "overcompression": "Exceso de control dinámico",
    }
    for match in match_knowledge(analysis):
        case = match.case
        hypothesis_name = legacy_names.get(case.case_id, case.title)
        if not any(h["name"] == hypothesis_name for h in hypotheses):
            hypothesis(hypothesis_name, case.confidence, case.why, case.first_check)

    advice = mix_advice(analysis)
    if not checks:
        checks.append({
            "priority": 1,
            "area": "Ruta de señal",
            "finding": "No aparecen anomalías críticas en las métricas analizadas.",
            "next_check": "Escuchá la fuente, compará con referencia y recién después intervení sobre procesamiento.",
            "severity": "ok",
            "evidence": [],
        })
        hypothesis("Sin anomalía crítica", 0.70,
                   "Las métricas no muestran una falla técnica dominante.",
                   "Hacé escucha crítica y comparación A/B con una referencia antes de intervenir.")

    checks.sort(key=lambda item: (item["priority"], {"critical": 0, "high": 1, "medium": 2, "ok": 3}.get(item["severity"], 4)))
    hypotheses.sort(key=lambda item: item["confidence"], reverse=True)
    primary = hypotheses[0] if hypotheses else None

    return {
        "diagnostic_version": "1.1",
        "mode": "resolver_primero",
        "score": advice["score"],
        "grade": advice["grade"],
        "primary_hypothesis": primary,
        "hypotheses": hypotheses,
        "first_action": checks[0]["next_check"],
        "checks": checks,
        "issues": advice["issues"],
        "tips": advice["tips"],
        "field_memory": field_memory,
    }


def generate_mastering_recommendations(analysis: dict) -> dict:
    """Genera recomendaciones accionables de mastering (parámetros sugeridos)
    a partir del análisis técnico del track. Devuelve un dict con `advice`
    (lo mismo que `mix_advice`) y `suggested_params` con valores de ejemplo.
    """
    advice = mix_advice(analysis)
    suggested = {}
    lufs = analysis.get("lufs", -99)
    dyn = analysis.get("dynamic_range_db", 0)
    true_peak = analysis.get("true_peak_db", -10)
    mono_compat = analysis.get("mono_compatibility_db", 0.0)

    # Si está muy comprimido, sugerir compresión paralela suave
    if dyn < 6:
        suggested["parallel_mix"] = 0.12
        suggested["parallel_threshold_db"] = -10.0
        suggested["parallel_ratio"] = 3.0
    # Si loudness demasiado alto, bajar makeup/limiter
    if lufs > -9:
        suggested["limiter_ceiling"] = 0.92
        suggested["comp_makeup_db"] = max(-2.0, -1.0)
    # True peak warning: bajar ceiling
    if true_peak > -0.5:
        suggested["limiter_ceiling"] = min(suggested.get("limiter_ceiling", 0.95), 0.94)

    recommendation_text = []
    if mono_compat < -3.0:
        recommendation_text.append(
            "Se recomienda reducir 1.5 dB de graves para mejorar la compatibilidad mono y evitar cancelaciones de fase."
        )
    if true_peak > -0.5:
        recommendation_text.append(
            "Se recomienda bajar el ceiling del limiter a 0.94 dBFS para evitar clipping inter-sample."
        )
    if lufs > -9:
        recommendation_text.append(
            "Se recomienda bajar el makeup gain del compresor o el ceiling del limiter para hacer el master más estable en streaming."
        )
    if lufs < -20:
        recommendation_text.append(
            "Se recomienda subir el nivel general con normalización LUFS para que el track no suene demasiado bajo."
        )
    if dyn < 6:
        recommendation_text.append(
            "Se recomienda usar compresión paralela suave y relajar un poco el ratio principal para preservar la dinámica."
        )
    if not recommendation_text:
        recommendation_text.append("La mezcla está técnicamente equilibrada; podés continuar con estos ajustes o comparar con referencia.")

    return {
        "advice": advice,
        "suggested_params": suggested,
        "recommendation_text": recommendation_text,
    }

# ─── Loudness targets por plataforma ──────────────────────────────────────────

PLATFORM_LOUDNESS_TARGETS = {
    "spotify":     {"lufs": -14.0, "true_peak_db": -1.0},
    "youtube":     {"lufs": -14.0, "true_peak_db": -1.0},
    "apple_music": {"lufs": -16.0, "true_peak_db": -1.0},
    "tidal":       {"lufs": -14.0, "true_peak_db": -1.0},
    "club":        {"lufs": -9.0,  "true_peak_db": -0.3},
    "cd":          {"lufs": -9.0,  "true_peak_db": -0.1},
}

def get_platform_target(platform: str) -> dict:
    return PLATFORM_LOUDNESS_TARGETS.get(platform, PLATFORM_LOUDNESS_TARGETS["spotify"])

# ─── Presets (con multibanda DESACTIVADO por defecto) ────────────────────────
# Importados desde presets_generator.py con todos los parámetros de la cadena completa
MASTERING_PRESETS = MASTERING_PRESETS_FULL

# ─── DEPRECATED - Diccionario antiguo parcial (conservado solo para referencia histórica) ───
# ─── Presets (con todos los parámetros de la cadena completa) ────────────────────────
# Importados desde presets_generator.py - cada preset contiene ~150 parámetros
MASTERING_PRESETS = MASTERING_PRESETS_FULL

def get_preset(name: str) -> dict:
    if name not in MASTERING_PRESETS:
        raise KeyError(f"Preset '{name}' no existe. Válidos: {sorted(MASTERING_PRESETS)}")
    return dict(MASTERING_PRESETS[name])

def _band_rms_db(band: np.ndarray, n_frames: int = 32) -> float:
    mono = band.mean(axis=0) if band.ndim == 2 else band
    if mono.size == 0:
        return -60.0
    rms = float(np.sqrt(np.mean(mono ** 2)))
    return float(20.0 * np.log10(rms + 1e-9))


def _downsample_gr_curve(gr_arr: np.ndarray, sr: int, target_hz: float = 30.0) -> tuple[list[float], float]:
    """Reduce una curva de GR sample-a-sample a puntos compactos para UI."""
    values = np.asarray(gr_arr, dtype=np.float64)
    if values.ndim > 1:
        values = values.mean(axis=0)
    values = values.reshape(-1)
    if values.size == 0 or sr <= 0:
        return [], 1000.0 / target_hz
    hop = max(1, int(round(sr / target_hz)))
    curve = [
        round(float(np.mean(values[start:start + hop])), 2)
        for start in range(0, values.size, hop)
    ]
    return curve, round(1000.0 * hop / sr, 3)

# ─── Compresor de banda ancha (single-band / "de un solo cuerpo") ─────────────
# BUGFIX: este compresor tenía sliders en la UI (Threshold/Ratio/Attack/
# Release/Makeup) que el backend nunca leía ni usaba ("compresor fantasma"):
# ni app.py declaraba los Query params comp_*, ni apply_mastering_chain() ni
# process_audio() los recibían en su firma. Ahora se implementa la función y
# se conecta en la cadena de mastering (ver apply_mastering_chain, paso 4).
# Incluye oversampling 4x antes del cálculo de envolvente/ganancia para
# reducir aliasing en attacks rápidos, igual que harmonic_saturation().

def compressor(audio: np.ndarray, sr: int,
               threshold: float = 0.5,
               ratio: float = 4.0,
               attack_ms: float = 10.0,
               release_ms: float = 100.0,
               makeup_db: float = 0.0,
               oversample: int = DEFAULT_DSP_OVERSAMPLE,
               stereo_link: bool = True,
               lookahead_ms: float = 3.0,
               rms_window_ms: float = 2.0,
               pdr: bool = True,
               pdr_hold_ms: float = 500.0,
               return_gain_curve: bool = False) -> tuple:
    """Compresor feed-forward con soft-knee, make-up gain y link estéreo opcional.

    threshold: umbral lineal 0..1 (se convierte internamente a dBFS).
    ratio: relación de compresión, ej. 4.0 = "4:1".
    oversample: factor de sobremuestreo para detector/ganancia.
    stereo_link: usa una sola curva de ganancia para L/R y preserva el panorama.
    lookahead_ms: adelanta la curva de reducción esa cantidad de ms antes
        de aplicarla (gratis: el render es offline, no hay costo de
        latencia real). Reduce el overshoot de transientes vs. un detector
        puramente causal. 0 = comportamiento anterior (sin lookahead).
    rms_window_ms: pre-suaviza el detector con una ventana RMS corta antes
        del seguidor attack/release (comportamiento tipo bus-comp analógico
        en vez de peak puro). 0 = peak puro (comportamiento legacy).
        MEJORA: el default pasó de 0.0 (peak puro) a 2.0ms — una ventana
        chica que redondea la detección sin perder respuesta a transientes,
        más cercana a cómo responde un compresor analógico real y con menos
        "nerviosismo" en la curva de ganancia que un detector muestra a
        muestra.
    pdr: activa Program-Dependent Release en el seguidor de envolvente (ver
        _smooth_envelope) — el release real varía entre release_ms/3 y
        release_ms*3 según si la señal es un transiente corto o un pasaje
        sostenido, en vez de usar release_ms fijo todo el tiempo. True por
        defecto: es un comportamiento estrictamente mejor que un release
        fijo (más transparente en transientes, más "pegado" en pasajes
        sostenidos) sin ningún parámetro nuevo que el usuario deba tocar.
    pdr_hold_ms: constante de tiempo del integrador de "historia" del PDR.
    return_gain_curve: devuelve también la curva real de reducción para meters.
    """
    in_db = _band_rms_db(audio)
    if ratio <= 1.0:
        out = audio * (10.0 ** (makeup_db / 20.0))
        meter = {"gr_db": 0.0, "in_db": round(in_db, 2), "out_db": round(_band_rms_db(out), 2), "stereo_link": bool(stereo_link)}
        gr_shape = audio.shape if audio.ndim > 1 else (audio.shape[-1],)
        gr_arr = np.zeros(gr_shape, dtype=np.float64)
        return (out, meter, gr_arr) if return_gain_curve else (out, meter)

    threshold_db = 20.0 * np.log10(max(threshold, 1e-9))
    ovs = max(1, int(oversample))

    def gain_reduction(abs_signal):
        n = len(abs_signal)
        if ovs > 1:
            sr_up = sr * ovs
            up = resample_poly(abs_signal, ovs, 1)
            up = _rms_envelope_prefilter(np.abs(up), sr_up, rms_window_ms)
            env = _smooth_envelope(up, sr_up, attack_ms, release_ms,
                                   program_dependent=pdr, pdr_hold_ms=pdr_hold_ms)
            env_db = 20.0 * np.log10(env + 1e-9)
            if HAS_NUMBA:
                gr_db_up = _compute_gain_reduction_numba(env_db, threshold_db, ratio)
            else:
                gr_db_up = _soft_knee_gain_reduction_np(env_db, threshold_db, ratio)
            # BUGFIX: el lookahead shift se aplicaba ANTES del downsample.
            # El filtro polyphase de resample_poly tiene un group delay propio
            # que parcialmente cancela el lookahead. Ahora: downsample primero,
            # luego shift a sr nominal → el lookahead es exactamente lookahead_ms.
            gr_db = resample_poly(gr_db_up, 1, ovs)
            # Normalizar longitud: resample_poly puede dar n±1 muestras
            if len(gr_db) < n:
                gr_db = np.pad(gr_db, (0, n - len(gr_db)), mode='edge')
            else:
                gr_db = gr_db[:n]
            look = int(round(sr * lookahead_ms / 1000.0))
            return _apply_lookahead_shift(gr_db, look)
        sig = _rms_envelope_prefilter(abs_signal, sr, rms_window_ms)
        env = _smooth_envelope(sig, sr, attack_ms, release_ms,
                               program_dependent=pdr, pdr_hold_ms=pdr_hold_ms)
        env_db = 20.0 * np.log10(env + 1e-9)
        if HAS_NUMBA:
            gr_db = _compute_gain_reduction_numba(env_db, threshold_db, ratio)
        else:
            gr_db = _soft_knee_gain_reduction_np(env_db, threshold_db, ratio)
        look = int(round(sr * lookahead_ms / 1000.0))
        return _apply_lookahead_shift(gr_db, look)

    makeup = 10.0 ** (makeup_db / 20.0)

    if audio.ndim == 1:
        gr_arr = gain_reduction(np.abs(audio))
        out = audio * (10.0 ** (gr_arr / 20.0)) * makeup
    elif stereo_link:
        linked_detector = np.max(np.abs(audio), axis=0)
        linked_gr = gain_reduction(linked_detector)
        gr_arr = np.tile(linked_gr, (audio.shape[0], 1))
        out = audio * (10.0 ** (linked_gr / 20.0))[np.newaxis, :] * makeup
    else:
        grs = [gain_reduction(np.abs(ch)) for ch in audio]
        gr_arr = np.stack(grs)
        out = audio * (10.0 ** (gr_arr / 20.0)) * makeup

    # BUGFIX: antes se promediaba solo la cola final del track (sr // 8,
    # ~125ms), lo que hacía que el meter mostrara 0.0 dB de gr cada vez que
    # esa ventana final caía en silencio/fade-out, sin importar cuánto
    # hubiera comprimido el resto del track. Ahora se usa el arreglo
    # completo para el promedio, y se agrega gr_max_db (pico real de
    # reducción) que es más representativo para diagnóstico.
    gr_db_mean = float(np.mean(gr_arr)) if gr_arr.size else 0.0
    gr_db_max = float(np.min(gr_arr)) if gr_arr.size else 0.0
    out_db = _band_rms_db(out)
    gr_curve, curve_hop_ms = _downsample_gr_curve(gr_arr, sr)

    meter = {
        "gr_db": round(gr_db_mean, 2),
        "gr_max_db": round(gr_db_max, 2),
        "curve": gr_curve,
        "curve_hop_ms": curve_hop_ms,
        "in_db": round(in_db, 2),
        "out_db": round(out_db, 2),
        "stereo_link": bool(stereo_link),
        "oversample": ovs,
        "lookahead_ms": round(float(lookahead_ms), 2),
        "rms_window_ms": round(float(rms_window_ms), 2),
        "pdr": bool(pdr),
    }
    return (out, meter, gr_arr) if return_gain_curve else (out, meter)

# ─── Compresor multibanda (solo se usa si se activa) ─────────────────────────

def multiband_compressor(audio: np.ndarray, sr: int,
                         low_crossover: float = 250.0,
                         high_crossover: float = 4000.0,
                         low_threshold: float = 0.7,
                         low_ratio: float = 2.0,
                         low_attack_ms: float = 20.0,
                         low_release_ms: float = 150.0,
                         low_makeup_db: float = 0.0,
                         mid_threshold: float = 0.7,
                         mid_ratio: float = 2.0,
                         mid_attack_ms: float = 20.0,
                         mid_release_ms: float = 150.0,
                         mid_makeup_db: float = 0.0,
                         high_threshold: float = 0.7,
                         high_ratio: float = 2.0,
                         high_attack_ms: float = 20.0,
                         high_release_ms: float = 150.0,
                         high_makeup_db: float = 0.0,
                         bypass: bool = True,
                         oversample: int = DEFAULT_DSP_OVERSAMPLE,
                         pdr: bool = True, pdr_hold_ms: float = 500.0) -> tuple:
    if bypass:
                            zero_curve, curve_hop_ms = _downsample_gr_curve(
                                np.zeros(audio.shape[-1], dtype=np.float32), sr,
                            )
                            return audio, {
                                "bypass": True,
                                "low_gr_db": 0.0, "mid_gr_db": 0.0, "high_gr_db": 0.0,
                                "low_curve": zero_curve, "mid_curve": zero_curve,
                                "high_curve": zero_curve, "curve_hop_ms": curve_hop_ms,
                                "low_in_db": 0.0, "mid_in_db": 0.0, "high_in_db": 0.0,
                                "low_out_db": 0.0, "mid_out_db": 0.0, "high_out_db": 0.0,
                            }

    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] == 1:
        audio = np.stack([audio[0], audio[0]])

    low_crossover  = float(np.clip(low_crossover,  20.0, sr / 2.0 - 1.0))
    high_crossover = float(np.clip(high_crossover, low_crossover + 1.0, sr / 2.0 - 1.0))

    sos_lo_lp  = butter(4, low_crossover,  btype='lowpass',  fs=sr, output='sos')
    sos_lo_hp  = butter(4, low_crossover,  btype='highpass', fs=sr, output='sos')
    sos_hi_lp  = butter(4, high_crossover, btype='lowpass',  fs=sr, output='sos')
    sos_hi_hp  = butter(4, high_crossover, btype='highpass', fs=sr, output='sos')

    # Usar sosfiltfilt para fase cero
    low      = sosfiltfilt(sos_lo_lp, audio)
    mid_high = sosfiltfilt(sos_lo_hp, audio)
    mid      = sosfiltfilt(sos_hi_lp, mid_high)
    high     = sosfiltfilt(sos_hi_hp, mid_high)

    low_in_db  = _band_rms_db(low)
    mid_in_db  = _band_rms_db(mid)
    high_in_db = _band_rms_db(high)

    def _compressor(ch, threshold, ratio, attack_ms, release_ms, makeup_db):
        # pdr/pdr_hold_ms se toman del closure (misma config para las 3 bandas)
        compressed, _meter, gr_db = compressor(
            ch, sr,
            threshold=threshold, ratio=ratio,
            attack_ms=attack_ms, release_ms=release_ms,
            makeup_db=makeup_db, oversample=oversample,
            stereo_link=True, return_gain_curve=True,
            pdr=pdr, pdr_hold_ms=pdr_hold_ms,
        )
        return compressed, gr_db

    # SERIE: la cadena de audio se procesa stage-por-stage (sin multitarea).
    # El paralelismo previo con ThreadPoolExecutor se removió para garantizar
    # orden determinístico y evitar contención de BLAS/Numba entre threads.
    low_comp,  low_gr_arr  = _compressor(low,  low_threshold,  low_ratio,  low_attack_ms,  low_release_ms,  low_makeup_db)
    mid_comp,  mid_gr_arr  = _compressor(mid,  mid_threshold,  mid_ratio,  mid_attack_ms,  mid_release_ms,  mid_makeup_db)
    high_comp, high_gr_arr = _compressor(high, high_threshold, high_ratio, high_attack_ms, high_release_ms, high_makeup_db)

    # BUGFIX: mismo problema que en compressor() — promediar solo la cola
    # final (sr // 8) del track escondía la reducción real si el final
    # de la canción era silencio/fade-out. Se promedia sobre todo el
    # arreglo, y se agrega el pico de reducción por banda.
    low_gr_db  = float(np.mean(low_gr_arr))
    mid_gr_db  = float(np.mean(mid_gr_arr))
    high_gr_db = float(np.mean(high_gr_arr))
    low_gr_max_db  = float(np.min(low_gr_arr))
    mid_gr_max_db  = float(np.min(mid_gr_arr))
    high_gr_max_db = float(np.min(high_gr_arr))

    low_curve, curve_hop_ms = _downsample_gr_curve(low_gr_arr, sr)
    mid_curve, _ = _downsample_gr_curve(mid_gr_arr, sr)
    high_curve, _ = _downsample_gr_curve(high_gr_arr, sr)

    low_out_db  = _band_rms_db(low_comp)
    mid_out_db  = _band_rms_db(mid_comp)
    high_out_db = _band_rms_db(high_comp)

    meter_data = {
        "low_gr_db":   round(low_gr_db, 2),
        "mid_gr_db":   round(mid_gr_db, 2),
        "high_gr_db":  round(high_gr_db, 2),
        "low_gr_max_db":  round(low_gr_max_db, 2),
        "mid_gr_max_db":  round(mid_gr_max_db, 2),
        "high_gr_max_db": round(high_gr_max_db, 2),
        "low_curve":     low_curve,
        "mid_curve":     mid_curve,
        "high_curve":    high_curve,
        "curve_hop_ms":  curve_hop_ms,
        "low_in_db":   round(low_in_db, 2),
        "mid_in_db":   round(mid_in_db, 2),
        "high_in_db":  round(high_in_db, 2),
        "low_out_db":  round(low_out_db, 2),
        "mid_out_db":  round(mid_out_db, 2),
        "high_out_db": round(high_out_db, 2),
    }

    return low_comp + mid_comp + high_comp, meter_data

# ─── Cadena DSP principal ──────────────────────────────────────────────────────
# Orden de la cadena (HPF, banda ancha, glue y limiter siempre activos;
# el resto es opcional según los parámetros del preset/usuario):
#   1. High-pass (limpieza de sub-graves, fase cero)
#   2. EQ paramétrico (4 bandas RBJ) + high shelf
#   3. Transient shaper
#   4. Dinámica, en orden: multibanda (bypass por defecto) → banda ancha /
#      "un solo cuerpo" (Threshold/Ratio/Attack/Release/Makeup, siempre
#      activo) → glue compressor (bypass por defecto, cierra la dinámica)
#   5. Saturación armónica (tape/tube, oversampling x4)
#   6. Imagen estéreo: Mid/Side gain → enhancer/width → multibanda estéreo
#   7. Reverb (efecto de espacio, sutil)
#   8. Limitador brick-wall con lookahead (siempre activo, último en la cadena)
# No se aplica ninguna normalización de ganancia en ningún punto.

def apply_mastering_chain(
    audio: np.ndarray, sr: int,
    target_peak: float = 0.95,          # No se usa
    use_lufs_normalize: bool = False,   # No se usa
    target_lufs: float = -14.0,         # No se usa
    input_gain_db: float = 0.0,
    oversample_mode: str = "quality",
    comp_stereo_link: bool = True,
    comp_bypass: bool = False,
    comp_threshold_db: float = -18.0,
    comp_ratio: float = 4.0,
    comp_attack_ms: float = 10.0,
    comp_release_ms: float = 100.0,
    comp_makeup_db: float = 0.0,
    comp_pdr: bool = True,
    comp_pdr_hold_ms: float = 500.0,
    # Compresión paralela (opcional)
    parallel_bypass: bool = True,
    parallel_threshold_db: float = -12.0,
    parallel_ratio: float = 4.0,
    parallel_attack_ms: float = 10.0,
    parallel_release_ms: float = 100.0,
    parallel_mix: float = 0.0,
    mb_low_crossover: float = 250.0,
    mb_high_crossover: float = 4000.0,
    mb_low_threshold_db: float = -18.0,
    mb_low_ratio: float = 2.0,
    mb_low_attack_ms: float = 20.0,
    mb_low_release_ms: float = 150.0,
    mb_low_makeup_db: float = 0.0,
    mb_mid_threshold_db: float = -18.0,
    mb_mid_ratio: float = 2.0,
    mb_mid_attack_ms: float = 20.0,
    mb_mid_release_ms: float = 150.0,
    mb_mid_makeup_db: float = 0.0,
    mb_high_threshold_db: float = -18.0,
    mb_high_ratio: float = 2.0,
    mb_high_attack_ms: float = 20.0,
    mb_high_release_ms: float = 150.0,
    mb_high_makeup_db: float = 0.0,
    mb_pdr: bool = True,
    mb_pdr_hold_ms: float = 500.0,
    mb_bypass: bool = False,  # False = activo por defecto (bypass=False significa que la etapa SÍ corre)
    hp_cutoff: float = 80.0,
    lp_bypass: bool = True,
    lp_cutoff: float = 18000.0,
    high_shelf_gain_db: float = 0.0,
    high_shelf_freq_hz: float = 8000.0,
    low_shelf_gain_db: float = 0.0,
    low_shelf_freq_hz: float = 100.0,
    mb_stereo_bypass: bool = True,
    mb_stereo_low_width: float = 0.9,
    mb_stereo_mid_width: float = 1.2,
    mb_stereo_high_width: float = 1.5,
    mb_stereo_low_crossover: float = 150.0,
    mb_stereo_high_crossover: float = 4000.0,
    eq1_freq: float = 100.0, eq1_gain: float = 0.0, eq1_q: float = 1.0,
    eq2_freq: float = 500.0, eq2_gain: float = 0.0, eq2_q: float = 1.0,
    eq3_freq: float = 2000.0, eq3_gain: float = 0.0, eq3_q: float = 1.0,
    eq4_freq: float = 8000.0, eq4_gain: float = 0.0, eq4_q: float = 1.0,
    eq5_freq: float = 200.0,  eq5_gain: float = 0.0, eq5_q: float = 1.0,
    eq6_freq: float = 1000.0, eq6_gain: float = 0.0, eq6_q: float = 1.0,
    tonal_balance_bypass: bool = True,
    tonal_balance_amount: float = 1.0,
    tonal_balance_max_boost_db: float = 3.5,
    tonal_balance_max_cut_db: float = -4.5,
    tonal_balance_max_bands: int = 6,
    ms_eq_bypass: bool = True,
    ms_mid_freq: float = 250.0, ms_mid_gain: float = 0.0, ms_mid_q: float = 1.0,
    ms_side_freq: float = 8000.0, ms_side_gain: float = 0.0, ms_side_q: float = 1.0,
    ms_comp_bypass: bool = True,
    ms_comp_mid_threshold_db: float = -18.0, ms_comp_mid_ratio: float = 2.0,
    ms_comp_mid_attack_ms: float = 15.0, ms_comp_mid_release_ms: float = 120.0,
    ms_comp_mid_makeup_db: float = 0.0,
    ms_comp_side_threshold_db: float = -18.0, ms_comp_side_ratio: float = 2.0,
    ms_comp_side_attack_ms: float = 15.0, ms_comp_side_release_ms: float = 120.0,
    ms_comp_side_makeup_db: float = 0.0,
    ms_comp_pdr: bool = True,
    ms_comp_pdr_hold_ms: float = 500.0,
    transient_attack: float = 0.0,
    transient_sustain: float = 0.0,
    saturation_drive: float = 0.0,
    saturation_mode: str = "tape",
    saturation_mix: float = 1.0,
    mid_gain_db: float = 0.0,
    side_gain_db: float = 0.0,
    stereo_width_amount: float = 1.0,
    stereo_bypass: bool = False,
    use_stereo_enhancer: bool = False,
    enhancer_bass_mono_freq: float = 120.0,
    haas_delay_ms: float = 0.0,
    reverb_size: float = 0.3,
    reverb_wet: float = 0.0,
    glue_bypass: bool = True,
    glue_threshold_db: float = -4.0,
    glue_ratio: float = 2.0,
    glue_attack_ms: float = 30.0,
    glue_release_ms: float = 120.0,
    glue_makeup_db: float = 0.0,
    glue_pdr: bool = True,
    glue_pdr_hold_ms: float = 500.0,
    clipper_bypass: bool = True,
    clipper_mode: str = "soft",
    clipper_ceiling: float = 0.98,
    clipper_drive_db: float = 0.0,
    limiter_ceiling: float = 0.95,
    limiter_bypass: bool = False,
    limiter_release_ms: float = 80.0,
    eq_mode: str = "iir",              # "iir" (zero-phase, actual) | "linear_phase" (FIR)
    linear_phase_taps: int = 2049,
    low_end_mono_freq: float = 120.0,
    low_end_mono_amount: float = 0.0,  # 0 = bypass (comportamiento anterior sin cambios)
    reso_bypass: bool = True,
    reso_freq: float = 1200.0,
    reso_q: float = 3.0,
    reso_threshold_db: float = -18.0,
    reso_ratio: float = 3.0,
    reso_attack_ms: float = 5.0,
    reso_release_ms: float = 100.0,
    reso_max_reduction_db: float = 8.0,
    dyneq_bypass: bool = True,
    dyneq_freq: float = 3000.0,
    dyneq_q: float = 2.5,
    dyneq_threshold_db: float = -18.0,
    dyneq_ratio: float = 3.0,
    dyneq_attack_ms: float = 3.0,
    dyneq_release_ms: float = 80.0,
    dyneq_max_reduction_db: float = 12.0,
    nr_bypass: bool = True,
    nr_strength: float = 0.5,
    nr_noise_sample_sec: float = 0.5,
    progress_cb=None,
    progress_range: tuple = (0, 100),
    **_ignored,
) -> tuple:
    """
    Cadena de mastering (16 etapas, orden fijo pedido por Fede):

        1.  Ganancia de entrada (trim)
        2.  Filtros HPF + LPF
        3.  EQ correctiva / cirugía (eq1-3 estáticas + dynamic EQ de
            resonancias, banda 785-1576 Hz por defecto)
        4.  EQ Mid/Side (gain M/S + banded EQ M/S)
        5.  Compresor de banda ancha (VCA/Opto — "un solo cuerpo")
        6.  EQ tonal / sweetening (eq4-6 + high shelf + low shelf)
        7.  De-esser dedicado (dynamic EQ de banda única, banda de sibilancia)
        8.  Compresor multibanda
        9.  Moldeador de transientes
        10. Saturación / armónicos (tape/tubo/clipper, oversampling x4)
        11. Procesamiento estéreo (enhancer/width + multibanda estéreo + reverb)
        12. Monofización de sub-graves (< 120 Hz por defecto)
        13. Compresor glue (bus final)
        14. Clipper (pico suave/hard)
        15. Limitador brick-wall (true peak, siempre activo, cierra la cadena)
        16. Medición (RMS/peak/LUFS/correlación) — dithering se aplica
            después, en _write_master_output()

    Por qué este orden: primero se limpia el espectro (HPF/LPF + cirugía
    correctiva de resonancias) y se resuelve el balance M/S ANTES de que
    cualquier compresor vea la señal, para que la detección de nivel no esté
    contaminada por resonancias o desbalance estéreo. Recién ahí entra la
    compresión de banda ancha (controla el nivel general ya "limpio"). Con
    el nivel controlado se suma color tonal (sweetening) y se doma la
    sibilancia con el de-esser — en ese orden porque el de-esser trabaja
    mejor sobre una señal ya tonalmente balanceada. El multibanda entra
    después para terminar de equilibrar cada rango por separado, seguido
    del transient shaper (que necesita transientes ya "limpios" de
    resonancias/sibilancia para no exagerar el ataque de un problema
    tonal) y la saturación armónica (color final de textura). La imagen
    estéreo (incluyendo reverb, que va casi al final para no ensuciar
    etapas previas) y la monofización de sub-graves preceden al glue, que
    cohesiona el conjunto ya balanceado en estéreo. El clipper (pico
    suave/hard) le quita trabajo al limitador brick-wall, que siempre
    cierra la cadena. No se aplica ninguna normalización de ganancia
    automática (solo el trim manual de input_gain_db, si se especifica).
    """
    ovs = resolve_oversample(oversample_mode)

    # Reescala un 0-100 "local" de la cadena al rango [progress_range[0], progress_range[1]]
    # que le corresponde dentro del progreso total reportado por process_audio().
    _p_lo, _p_hi = progress_range
    def _chain_progress(local_pct: float, stage: str) -> None:
        _report(progress_cb, _p_lo + (local_pct / 100.0) * (_p_hi - _p_lo), stage)

    # ── 0. Reducción de ruido (opcional, bypass por defecto) ───────────────
    # Va ANTES de todo el procesamiento: limpiar primero, procesar después.
    if not nr_bypass:
        _chain_progress(0, "Reduciendo ruido de fondo")
        audio = noise_reduction(audio, sr, strength=nr_strength,
                                noise_sample_sec=nr_noise_sample_sec)

    # ── 1. Ganancia de entrada (trim manual, opcional) ─────────────────────
    _chain_progress(2, "Ajustando ganancia de entrada")
    if input_gain_db != 0.0:
        audio = audio * (10.0 ** (input_gain_db / 20.0))

    # ── 2. Filtros HPF + LPF (fase cero) ────────────────────────────────────
    _chain_progress(6, "Aplicando filtro pasa-altos")
    audio = eq_high_pass(audio, sr, cutoff_hz=hp_cutoff)
    if not lp_bypass:
        _chain_progress(8, "Aplicando filtro pasa-bajos")
        audio = eq_low_pass(audio, sr, cutoff_hz=lp_cutoff)

    # ── 3. EQ correctiva / cirugía (eq1-3 estáticas + dynamic EQ de        ──
    # ── resonancias). Va temprano porque es correctiva sobre problemas del ──
    # ── material (resonancias/boxiness), no sobre el "sonido" final — y    ──
    # ── tiene que actuar antes de que cualquier compresor vea la señal.    ──
    _chain_progress(12, "Aplicando ecualización correctiva")
    if str(eq_mode).lower() == "linear_phase":
        surgical_bands = []
        for freq, gain, q in [(eq1_freq, eq1_gain, eq1_q), (eq2_freq, eq2_gain, eq2_q), (eq3_freq, eq3_gain, eq3_q)]:
            if gain != 0.0:
                surgical_bands.append({"type": "peak", "freq": freq, "gain_db": gain, "q": q})
        if surgical_bands:
            audio = linear_phase_eq(audio, sr, surgical_bands, num_taps=linear_phase_taps)
    else:
        for freq, gain, q in [(eq1_freq, eq1_gain, eq1_q), (eq2_freq, eq2_gain, eq2_q), (eq3_freq, eq3_gain, eq3_q)]:
            if gain != 0.0:
                audio = eq_parametric_band(audio, sr, freq=freq, gain_db=gain, q=q)

    audio, reso_meters = dynamic_eq_band(
        audio, sr,
        freq=reso_freq, q=reso_q,
        threshold_db=reso_threshold_db, ratio=reso_ratio,
        attack_ms=reso_attack_ms, release_ms=reso_release_ms,
        max_reduction_db=reso_max_reduction_db, bypass=reso_bypass,
    )

    # ── 4. EQ Mid/Side (gain ancho + banded EQ, opcional) ───────────────────
    _chain_progress(16, "Procesando EQ Mid/Side")
    if audio.shape[0] == 2 and (mid_gain_db != 0.0 or side_gain_db != 0.0):
        audio = mid_side_process(audio, mid_gain_db=mid_gain_db, side_gain_db=side_gain_db)
    if audio.shape[0] == 2 and not ms_eq_bypass:
        audio = mid_side_eq(
            audio, sr,
            mid_freq=ms_mid_freq, mid_gain_db=ms_mid_gain, mid_q=ms_mid_q,
            side_freq=ms_side_freq, side_gain_db=ms_side_gain, side_q=ms_side_q,
        )
    # Compresor M/S dedicado: va DESPUÉS del EQ M/S (misma justificación
    # que el orden de mid_side_eq — el detector del compresor de banda
    # ancha que sigue en la etapa 5 tiene que ver la señal ya con el
    # balance M/S resuelto, no solo el tono).
    audio, ms_comp_meters = mid_side_compressor(
        audio, sr,
        mid_threshold_db=ms_comp_mid_threshold_db, mid_ratio=ms_comp_mid_ratio,
        mid_attack_ms=ms_comp_mid_attack_ms, mid_release_ms=ms_comp_mid_release_ms,
        mid_makeup_db=ms_comp_mid_makeup_db,
        side_threshold_db=ms_comp_side_threshold_db, side_ratio=ms_comp_side_ratio,
        side_attack_ms=ms_comp_side_attack_ms, side_release_ms=ms_comp_side_release_ms,
        side_makeup_db=ms_comp_side_makeup_db,
        oversample=ovs, bypass=ms_comp_bypass,
        pdr=ms_comp_pdr, pdr_hold_ms=ms_comp_pdr_hold_ms,
    )

    # ── 5. Compresor de banda ancha (VCA/Opto — "un solo cuerpo":          ──
    # ── Threshold/Ratio/Attack/Release/Makeup). Controla el nivel general  ──
    # ── ya limpio de resonancias y balanceado en M/S.                      ──
    _chain_progress(28, "Aplicando compresión de banda ancha")
    # Capturamos el audio pre-compresor para usarlo como señal "dry" en la
    # compresión paralela (New York compression). El blend debe ser entre el
    # original sin comprimir y la versión muy comprimida, no entre la versión
    # ya comprimida por el compresor principal y una segunda compresión encima.
    #
    # BUGFIX 1 — coerción de parallel_bypass invertida:
    # La versión anterior tenía:
    #   parallel_bypass.strip().lower() not in {"true","1","yes"}
    # que convierte el string "true" → False (activo) y "false" → True (bypass).
    # Exactamente al revés. La coerción correcta es `in {"true","1","yes"}`.
    # Nota: coerce_ws_chain_params() en validation_utils ya hace esta conversión
    # correctamente para el path WebSocket; este bloque es el fallback para
    # llamadas directas (tests, /master endpoint sin coerción previa).
    #
    # BUGFIX 2 — guard `_par_mix > 0.0` bloqueaba la ejecución:
    # La condición original era `not _par_bypass and _par_mix > 0.0`. Si el
    # usuario desactivó el bypass pero no tocó parallel_mix (default 0.0),
    # la compresión paralela nunca corría aunque estuviera habilitada. Se
    # separan las dos comprobaciones: el bypass controla si corre o no; el mix
    # se clipea a un mínimo útil (0.05) si llegó en 0 con bypass=False para
    # que el efecto sea audible, pero se respeta el valor si el usuario lo
    # configuró explícitamente por encima de 0.
    try:
        _par_mix = float(parallel_mix)
    except (TypeError, ValueError):
        _par_mix = 0.0
    if not isinstance(parallel_bypass, bool):
        if isinstance(parallel_bypass, str):
            _par_bypass = parallel_bypass.strip().lower() in {"true", "1", "yes", "on", "sí", "si"}
        else:
            _par_bypass = not bool(parallel_bypass)
    else:
        _par_bypass = parallel_bypass
    # Si bypass=False pero mix=0, usar un mix mínimo útil (el usuario desactivó
    # el bypass intencionalmente pero no ajustó el knob — por diseño de UX se
    # aplica un valor por defecto audible en vez de silenciar el efecto).
    if not _par_bypass and _par_mix <= 0.0:
        _par_mix = 0.12

    audio_pre_comp = audio.copy() if not _par_bypass else None
    if comp_bypass:
        comp_meters = {"bypass": True, "gr_db": 0.0, "in_db": round(float(20*np.log10(np.sqrt(np.mean(audio**2))+1e-9)), 2), "out_db": round(float(20*np.log10(np.sqrt(np.mean(audio**2))+1e-9)), 2)}
    else:
        audio, comp_meters = compressor(
            audio, sr,
            threshold=10.0 ** (comp_threshold_db / 20.0),
            ratio=comp_ratio,
            attack_ms=comp_attack_ms,
            release_ms=comp_release_ms,
            makeup_db=comp_makeup_db,
            oversample=ovs,
            stereo_link=comp_stereo_link,
            pdr=comp_pdr, pdr_hold_ms=comp_pdr_hold_ms,
        )
        comp_meters = {**comp_meters, "bypass": False}

    # ── 5.5 Compresión paralela (New York style): blend dry original / wet comprimido
    if not _par_bypass:
        _chain_progress(32, "Aplicando compresión paralela")
        # Comprimimos el audio PRE-compresor principal (señal limpia sin comprimir)
        compressed, par_meters = compressor(
            audio_pre_comp, sr,
            threshold=10.0 ** (parallel_threshold_db / 20.0),
            ratio=parallel_ratio,
            attack_ms=parallel_attack_ms,
            release_ms=parallel_release_ms,
            makeup_db=0.0,
            oversample=ovs,
            stereo_link=comp_stereo_link,
            pdr=comp_pdr, pdr_hold_ms=comp_pdr_hold_ms,
        )
        mix = float(np.clip(_par_mix, 0.0, 1.0))
        # Dry = salida del compresor principal (step 5), wet = versión paralela
        audio = (1.0 - mix) * audio + mix * compressed
        par_meters = {**par_meters, "bypass": False, "mix": round(mix, 3)}
        audio_pre_comp = None  # liberar memoria
    else:
        par_meters = {"bypass": True, "gr_db": 0.0, "mix": 0.0}

    # ── 6. EQ tonal / sweetening (eq4-6 + high shelf + low shelf) ──────────
    _chain_progress(34, "Analizando balance tonal")
    tonal_balance_meters = {"bypass": True}
    if not tonal_balance_bypass:
        tb_result = auto_tonal_balance(
            audio, sr,
            max_boost_db=tonal_balance_max_boost_db,
            max_cut_db=tonal_balance_max_cut_db,
            max_eq_bands=int(tonal_balance_max_bands),
        )
        audio = apply_tonal_balance(audio, sr, tb_result["eq_bands"],
                                    amount=tonal_balance_amount,
                                    num_taps=linear_phase_taps)
        tonal_balance_meters = {
            "bypass": False,
            "amount": round(float(tonal_balance_amount), 2),
            "suggested_bands": tb_result["suggested_bands"],
            "summary": tb_result["summary"],
        }

    _chain_progress(35, "Aplicando ecualización tonal")
    if str(eq_mode).lower() == "linear_phase":
        tonal_bands = []
        for freq, gain, q in [(eq4_freq, eq4_gain, eq4_q), (eq5_freq, eq5_gain, eq5_q), (eq6_freq, eq6_gain, eq6_q)]:
            if gain != 0.0:
                tonal_bands.append({"type": "peak", "freq": freq, "gain_db": gain, "q": q})
        if high_shelf_gain_db != 0.0:
            tonal_bands.append({"type": "high_shelf", "freq": high_shelf_freq_hz, "gain_db": high_shelf_gain_db})
        if low_shelf_gain_db != 0.0:
            tonal_bands.append({"type": "low_shelf", "freq": low_shelf_freq_hz, "gain_db": low_shelf_gain_db})
        if tonal_bands:
            audio = linear_phase_eq(audio, sr, tonal_bands, num_taps=linear_phase_taps)
    else:
        for freq, gain, q in [(eq4_freq, eq4_gain, eq4_q), (eq5_freq, eq5_gain, eq5_q), (eq6_freq, eq6_gain, eq6_q)]:
            if gain != 0.0:
                audio = eq_parametric_band(audio, sr, freq=freq, gain_db=gain, q=q)
        if high_shelf_gain_db != 0.0:
            audio = eq_high_shelf(audio, sr, cutoff_hz=high_shelf_freq_hz, gain_db=high_shelf_gain_db)
        if low_shelf_gain_db != 0.0:
            _ls_sos = _design_low_shelf_sos(sr, low_shelf_freq_hz, low_shelf_gain_db)
            if audio.ndim == 1:
                audio = sosfiltfilt(_ls_sos, audio)
            else:
                audio = np.stack([sosfiltfilt(_ls_sos, ch) for ch in audio])

    # ── 7. De-esser dedicado (dynamic EQ de banda única, banda de           ──
    # ── sibilancia — mismo mecanismo genérico que el de resonancias del    ──
    # ── paso 3, pero apuntado a la zona de "eses"/"ches").                 ──
    _chain_progress(42, "Aplicando de-esser")
    audio, dyneq_meters = dynamic_eq_band(
        audio, sr,
        freq=dyneq_freq, q=dyneq_q,
        threshold_db=dyneq_threshold_db, ratio=dyneq_ratio,
        attack_ms=dyneq_attack_ms, release_ms=dyneq_release_ms,
        max_reduction_db=dyneq_max_reduction_db, bypass=dyneq_bypass,
    )

    # ── 8. Compresor multibanda (bypass por defecto) ────────────────────────
    _chain_progress(50, "Aplicando compresión multibanda")
    audio, mb_meters = multiband_compressor(
        audio, sr,
        low_crossover=mb_low_crossover,
        high_crossover=mb_high_crossover,
        low_threshold=10.0 ** (mb_low_threshold_db / 20.0),
        low_ratio=mb_low_ratio,
        low_attack_ms=mb_low_attack_ms,
        low_release_ms=mb_low_release_ms,
        low_makeup_db=mb_low_makeup_db,
        mid_threshold=10.0 ** (mb_mid_threshold_db / 20.0),
        mid_ratio=mb_mid_ratio,
        mid_attack_ms=mb_mid_attack_ms,
        mid_release_ms=mb_mid_release_ms,
        mid_makeup_db=mb_mid_makeup_db,
        high_threshold=10.0 ** (mb_high_threshold_db / 20.0),
        high_ratio=mb_high_ratio,
        high_attack_ms=mb_high_attack_ms,
        high_release_ms=mb_high_release_ms,
        high_makeup_db=mb_high_makeup_db,
        bypass=mb_bypass,
        oversample=ovs,
        pdr=mb_pdr, pdr_hold_ms=mb_pdr_hold_ms,
    )

    # ── 9. Moldeador de transientes (opcional) ──────────────────────────────
    _chain_progress(58, "Dando forma a los transientes")
    if transient_attack != 0.0 or transient_sustain != 0.0:
        audio = transient_shaper(audio, sr,
                                 attack_amount=transient_attack,
                                 sustain_amount=transient_sustain)

    # ── 10. Saturación armónica (opcional, oversampling x4) ─────────────────
    _chain_progress(64, "Aplicando saturación armónica")
    if saturation_drive > 0.0:
        audio = harmonic_saturation(audio, drive=saturation_drive,
                                    mode=saturation_mode, mix=saturation_mix,
                                    oversample=ovs)

    # ══ 11. PROCESAMIENTO ESTÉREO (enhancer/width → multibanda estéreo → ══
    # ══ reverb, efecto de espacio que va al final de esta sección para   ══
    # ══ no ensuciar las etapas de dinámica/saturación previas)           ══
    _chain_progress(72, "Procesando imagen estéreo")
    if audio.shape[0] == 2 and not stereo_bypass:
        if use_stereo_enhancer:
            audio = stereo_enhancer(audio, sr, width=stereo_width_amount,
                                    bass_mono_freq=enhancer_bass_mono_freq,
                                    haas_delay_ms=haas_delay_ms)
        elif stereo_width_amount != 1.0:
            audio = stereo_width(audio, width=stereo_width_amount)

    if audio.shape[0] == 2 and not stereo_bypass and not mb_stereo_bypass:
        audio = multiband_stereo_width(
            audio, sr,
            low_width=mb_stereo_low_width,
            mid_width=mb_stereo_mid_width,
            high_width=mb_stereo_high_width,
            low_crossover=mb_stereo_low_crossover,
            high_crossover=mb_stereo_high_crossover,
        )

    if reverb_wet > 0.0 and not stereo_bypass:
        audio = reverb_simple(audio, sr, room_size=reverb_size, wet=reverb_wet)

    # ── 12. Monofización de sub-graves dedicada (< 120 Hz por defecto,     ──
    # ── opcional, bypass por defecto). Va DESPUÉS de la imagen estéreo     ──
    # ── (para no pelearse con el enhancer/multibanda estéreo) y ANTES del ──
    # ── glue (para que el glue final vea el bus ya con el sub concentrado ──
    # ── al centro, evitando cancelaciones de fase en mono).                ──
    _chain_progress(78, "Monofonizando sub-graves")
    if audio.shape[0] == 2 and low_end_mono_amount > 0.0 and not stereo_bypass:
        audio = low_end_mono_maker(audio, sr, freq=low_end_mono_freq, mono_amount=low_end_mono_amount)

    # ── 13. Glue compressor (opcional, cohesiona el bus final) ─────────────
    _chain_progress(83, "Aplicando glue compressor")
    glue_meters = {"bypass": True, "gr_db": 0.0}
    if not glue_bypass:
        audio, glue_meters = compressor(
            audio, sr,
            threshold=10.0 ** (glue_threshold_db / 20.0),
            ratio=glue_ratio,
            attack_ms=glue_attack_ms,
            release_ms=glue_release_ms,
            makeup_db=glue_makeup_db,
            oversample=ovs,
            stereo_link=True,
            # Detector RMS (ventana corta) en vez de peak puro: es lo que
            # da el carácter "glue" de un bus comp analógico (ej. SSL Bus
            # Comp) — reacciona al promedio energético del programa, no a
            # cada muestra de pico, sonando más suave/cohesivo en el bus
            # final. El resto de los compresores de la cadena (banda ancha,
            # multibanda, paralela, M/S) se dejan en peak puro porque ahí
            # sí se busca respuesta rápida tipo VCA/Opto.
            rms_window_ms=4.0,
            pdr=glue_pdr, pdr_hold_ms=glue_pdr_hold_ms,
        )
        glue_meters.update({"bypass": False, "threshold_db": round(glue_threshold_db, 2)})

    # ---- SIN NORMALIZACIÓN ----

    # VU pre-limiter (para métricas)
    mono_pre = audio.mean(axis=0) if audio.ndim == 2 else audio
    pre_rms_db  = float(20.0 * np.log10(np.sqrt(np.mean(mono_pre ** 2)) + 1e-9))
    pre_peak_db = float(20.0 * np.log10(np.max(np.abs(mono_pre)) + 1e-9))

    # ── 14. Clipper (pico suave/hard, opcional — le quita trabajo al       ──
    # ── limitador antes de que este entre a actuar).                       ──
    _chain_progress(88, "Aplicando clipper")
    audio, clipper_meters = audio_clipper(
        audio, sr, ceiling=clipper_ceiling, mode=clipper_mode,
        drive_db=clipper_drive_db, bypass=clipper_bypass,
    )

    # ── 15. Limitador brick-wall con lookahead (siempre activo, cierra la  ──
    # ── cadena).                                                           ──
    _chain_progress(93, "Aplicando limitador final")
    if limiter_bypass:
        limiter_meters = {"bypass": True, "gr_db": 0.0}
    else:
        audio = limiter(audio, sr, ceiling=limiter_ceiling, release_ms=limiter_release_ms, lookahead_ms=5.0,
                        oversample=ovs)
        limiter_meters = {"bypass": False, "gr_db": 0.0}


    # ── 16. Medición final (el dithering se aplica después, al escribir    ──
    # ── el archivo de salida, en _write_master_output()).                  ──
    mono_post = audio.mean(axis=0) if audio.ndim == 2 else audio
    post_rms_db  = float(20.0 * np.log10(np.sqrt(np.mean(mono_post ** 2)) + 1e-9))
    post_peak_db = float(20.0 * np.log10(np.max(np.abs(mono_post)) + 1e-9))
    if not limiter_bypass:
        limiter_meters["gr_db"] = round(min(0.0, post_peak_db - pre_peak_db), 2)
    post_lufs    = measure_lufs_integrated(audio, sr)
    post_corr    = stereo_correlation(audio)

    chain_meters = {
        "config": {"oversample": ovs, "oversample_mode": str(oversample_mode), "comp_stereo_link": bool(comp_stereo_link), "comp_bypass": bool(comp_bypass), "stereo_bypass": bool(stereo_bypass), "limiter_bypass": bool(limiter_bypass)},
        "comp": comp_meters,
        "ms_comp": ms_comp_meters,
        "parallel": par_meters,
        "glue": glue_meters,
        "mb": mb_meters,
        "reso": reso_meters,
        "dyneq": dyneq_meters,
        "tonal_balance": tonal_balance_meters,
        "clipper": clipper_meters,
        "limiter": limiter_meters,
        "pre_limiter":  {"rms_db": round(pre_rms_db, 2),  "peak_db": round(pre_peak_db, 2)},
        "post_limiter": {
            "rms_db":             round(post_rms_db, 2),
            "peak_db":            round(post_peak_db, 2),
            "lufs":               round(post_lufs, 2),
            "stereo_correlation": round(post_corr, 3),
        },
    }

    _chain_progress(100, "Cadena de mastering completa")
    return audio, chain_meters

# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_lufs_safety_check(audio_base: np.ndarray, sr: int, chain_kwargs: dict,
                           target_lufs: float,
                           initial_audio: Optional[np.ndarray] = None,
                           initial_chain_meters: Optional[dict] = None,
                           max_iters: int = 4, tolerance_db: float = 0.3,
                           progress_cb=None, progress_start: int = 80,
                           progress_step: int = 3,
                           adaptive_loudness_weighting: bool = False,
                           loudness_sensitivity_amount: float = 0.65,
                           max_target_lufs: Optional[float] = None) -> tuple:
    """Reintenta la cadena completa ajustando input_gain_db hasta acercarse a
    `target_lufs` (o agotar los reintentos). Extraída de process_audio() para
    poder reusarla desde cualquier caller que necesite normalización LUFS
    (incluido el streaming en vivo, que antes no tenía forma de aplicarla
    porque llamaba a apply_mastering_chain directo).

    Si no se pasan `initial_audio`/`initial_chain_meters`, corre un primer
    render con `chain_kwargs` tal cual antes de empezar a corregir.

    MEJORA: `adaptive_loudness_weighting=True` usa measure_human_weighted_loudness()
    en vez del LUFS estándar BS.1770 puro — el mismo mecanismo que ya usaba
    solamente el reference matching (compute_reference_eq_curve). El oído
    humano no es igual de sensible en todas las frecuencias (curvas de
    igual-sonoridad ISO 226, más sensible ~3-6kHz); dos masters con el mismo
    LUFS estándar pero distinto balance tonal pueden sonar distinto de fuertes.
    Con esto activado, `target_lufs` se interpreta como el loudness PERCIBIDO
    que se busca (no el número crudo del medidor), y el gain se calcula sobre
    esa métrica. Por seguridad, el LUFS estándar proyectado NUNCA puede superar
    `max_target_lufs` (el techo real que va a medir cualquier plataforma de
    streaming) — igual que hace ya el reference matching: el ajuste perceptual
    puede pedir más o menos gain que el LUFS estándar, pero jamás más de lo que
    la plataforma permite. Si `max_target_lufs` no se pasa, se usa `target_lufs`
    mismo como techo (no debería sonar MÁS fuerte del número pedido).

    Devuelve (audio, chain_meters, notes, final_input_gain_db).
    """
    if initial_audio is None or initial_chain_meters is None:
        initial_audio, initial_chain_meters = apply_mastering_chain(
            audio_base, sr, progress_cb=None, **chain_kwargs
        )
    if max_target_lufs is None:
        max_target_lufs = float(target_lufs)

    notes = []
    current_input_gain = float(chain_kwargs.get("input_gain_db", 0.0))
    audio, chain_meters = initial_audio, initial_chain_meters
    for i in range(max_iters):
        if adaptive_loudness_weighting:
            # Se mide sobre el render actual completo (no solo el meter interno
            # del limiter, que sólo trae LUFS estándar) para tener la banda de
            # presencia real del resultado.
            loudness = measure_human_weighted_loudness(audio, sr, loudness_sensitivity_amount)
            achieved_perceived = loudness["perceived_lufs"]
            achieved_std = loudness["standard_lufs"]
            delta = float(target_lufs) - float(achieved_perceived)
            projected_std = achieved_std + delta
            if projected_std > max_target_lufs:
                delta = max_target_lufs - achieved_std
            achieved_for_tolerance = achieved_perceived
        else:
            achieved = chain_meters.get("post_limiter", {}).get("lufs")
            if achieved is None:
                break
            delta = float(target_lufs) - float(achieved)
            achieved_for_tolerance = achieved

        if abs(delta) <= tolerance_db:
            notes.append(
                f"LUFS safety check: {achieved_for_tolerance:.2f} "
                f"{'LUFS percibido' if adaptive_loudness_weighting else 'LUFS'} vs. objetivo "
                f"{target_lufs:.2f} LUFS (dentro de tolerancia, sin corrección adicional)."
            )
            break
        new_input_gain = float(np.clip(current_input_gain + delta, -24.0, 24.0))
        if abs(new_input_gain - current_input_gain) < 0.05:
            notes.append(
                f"LUFS safety check: no se pudo alcanzar {target_lufs:.2f} LUFS sin exceder "
                f"el rango de input_gain_db (quedó en {achieved_for_tolerance:.2f} "
                f"{'LUFS percibido' if adaptive_loudness_weighting else 'LUFS'})."
            )
            break
        current_input_gain = new_input_gain
        _report(progress_cb, progress_start + i * progress_step,
                f"Ajustando loudness (LUFS safety check, intento {i + 1}/{max_iters})")
        retry_kwargs = dict(chain_kwargs)
        retry_kwargs["input_gain_db"] = current_input_gain
        audio, chain_meters = apply_mastering_chain(
            audio_base, sr, progress_cb=None, **retry_kwargs
        )
        notes.append(
            f"LUFS safety check #{i + 1}: {achieved_for_tolerance:.2f} "
            f"{'LUFS percibido' if adaptive_loudness_weighting else 'LUFS'} vs. objetivo "
            f"{target_lufs:.2f} LUFS → input_gain_db corregido a {current_input_gain:+.2f} dB."
        )
    return audio, chain_meters, notes, current_input_gain


def compute_lufs_corrected_gain(audio: np.ndarray, sr: int, chain_kwargs: dict,
                                 target_lufs: float, max_iters: int = 4,
                                 tolerance_db: float = 0.3) -> tuple:
    """Versión liviana para streaming/preview en vivo: corre el mismo
    safety-check de forma batch UNA vez sobre el audio ya recortado al
    preview (no por chunk — sería carísimo y ruidoso en tiempo real) y
    devuelve solo el input_gain_db corregido + las notas, para que el
    caller lo use al armar el generador de chunks.

    `apply_mastering_chain` ignora use_lufs_normalize/target_peak/target_lufs
    (esos campos no hacen nada dentro de la cadena en sí — ver comentarios
    en su firma), así que la única forma real de "normalizar por LUFS" es
    corregir input_gain_db de antemano, que es justamente lo que hace esto.
    """
    base_kwargs = dict(chain_kwargs)
    base_kwargs.pop("use_lufs_normalize", None)
    base_kwargs.pop("target_peak", None)
    base_kwargs.pop("target_lufs", None)
    _, _, notes, final_gain = run_lufs_safety_check(
        audio, sr, base_kwargs, target_lufs,
        max_iters=max_iters, tolerance_db=tolerance_db,
    )
    return final_gain, notes


def process_audio(
    input_path: str,
    target_peak: float = 0.95,          # Ignorado
    use_lufs_normalize: bool = False,   # Ignorado
    target_lufs: float = -14.0,         # Ignorado
    adaptive_loudness_weighting: bool = False,
    loudness_sensitivity_amount: float = 0.65,
    input_gain_db: float = 0.0,
    oversample_mode: str = "quality",
    comp_stereo_link: bool = True,
    comp_bypass: bool = False,
    comp_threshold_db: float = -18.0,
    comp_ratio: float = 4.0,
    comp_attack_ms: float = 10.0,
    comp_release_ms: float = 100.0,
    comp_makeup_db: float = 0.0,
    comp_pdr: bool = True,
    comp_pdr_hold_ms: float = 500.0,
    # Compresión paralela (opcional)
    parallel_bypass: bool = True,
    parallel_threshold_db: float = -12.0,
    parallel_ratio: float = 4.0,
    parallel_attack_ms: float = 10.0,
    parallel_release_ms: float = 100.0,
    parallel_mix: float = 0.0,
    mb_low_crossover: float = 250.0,
    mb_high_crossover: float = 4000.0,
    mb_low_threshold_db: float = -18.0,
    mb_low_ratio: float = 2.0,
    mb_low_attack_ms: float = 20.0,
    mb_low_release_ms: float = 150.0,
    mb_low_makeup_db: float = 0.0,
    mb_mid_threshold_db: float = -18.0,
    mb_mid_ratio: float = 2.0,
    mb_mid_attack_ms: float = 20.0,
    mb_mid_release_ms: float = 150.0,
    mb_mid_makeup_db: float = 0.0,
    mb_high_threshold_db: float = -18.0,
    mb_high_ratio: float = 2.0,
    mb_high_attack_ms: float = 20.0,
    mb_high_release_ms: float = 150.0,
    mb_high_makeup_db: float = 0.0,
    mb_pdr: bool = True,
    mb_pdr_hold_ms: float = 500.0,
    mb_bypass: bool = False,  # False = activo por defecto (bypass=False significa que la etapa SÍ corre)
    hp_cutoff: float = 80.0,
    lp_bypass: bool = True,
    lp_cutoff: float = 18000.0,
    high_shelf_gain_db: float = 0.0,
    high_shelf_freq_hz: float = 8000.0,
    low_shelf_gain_db: float = 0.0,
    low_shelf_freq_hz: float = 100.0,
    mb_stereo_bypass: bool = True,
    mb_stereo_low_width: float = 0.9,
    mb_stereo_mid_width: float = 1.2,
    mb_stereo_high_width: float = 1.5,
    mb_stereo_low_crossover: float = 150.0,
    mb_stereo_high_crossover: float = 4000.0,
    eq1_freq: float = 100.0, eq1_gain: float = 0.0, eq1_q: float = 1.0,
    eq2_freq: float = 500.0, eq2_gain: float = 0.0, eq2_q: float = 1.0,
    eq3_freq: float = 2000.0, eq3_gain: float = 0.0, eq3_q: float = 1.0,
    eq4_freq: float = 8000.0, eq4_gain: float = 0.0, eq4_q: float = 1.0,
    eq5_freq: float = 200.0,  eq5_gain: float = 0.0, eq5_q: float = 1.0,
    eq6_freq: float = 1000.0, eq6_gain: float = 0.0, eq6_q: float = 1.0,
    tonal_balance_bypass: bool = True,
    tonal_balance_amount: float = 1.0,
    tonal_balance_max_boost_db: float = 3.5,
    tonal_balance_max_cut_db: float = -4.5,
    tonal_balance_max_bands: int = 6,
    ms_eq_bypass: bool = True,
    ms_mid_freq: float = 250.0, ms_mid_gain: float = 0.0, ms_mid_q: float = 1.0,
    ms_side_freq: float = 8000.0, ms_side_gain: float = 0.0, ms_side_q: float = 1.0,
    ms_comp_bypass: bool = True,
    ms_comp_mid_threshold_db: float = -18.0, ms_comp_mid_ratio: float = 2.0,
    ms_comp_mid_attack_ms: float = 15.0, ms_comp_mid_release_ms: float = 120.0,
    ms_comp_mid_makeup_db: float = 0.0,
    ms_comp_side_threshold_db: float = -18.0, ms_comp_side_ratio: float = 2.0,
    ms_comp_side_attack_ms: float = 15.0, ms_comp_side_release_ms: float = 120.0,
    ms_comp_side_makeup_db: float = 0.0,
    ms_comp_pdr: bool = True,
    ms_comp_pdr_hold_ms: float = 500.0,
    transient_attack: float = 0.0,
    transient_sustain: float = 0.0,
    saturation_drive: float = 0.0,
    saturation_mode: str = "tape",
    saturation_mix: float = 1.0,
    mid_gain_db: float = 0.0,
    side_gain_db: float = 0.0,
    stereo_width_amount: float = 1.0,
    stereo_bypass: bool = False,
    use_stereo_enhancer: bool = False,
    enhancer_bass_mono_freq: float = 120.0,
    haas_delay_ms: float = 0.0,
    reverb_size: float = 0.3,
    reverb_wet: float = 0.0,
    glue_bypass: bool = True,
    glue_threshold_db: float = -4.0,
    glue_ratio: float = 2.0,
    glue_attack_ms: float = 30.0,
    glue_release_ms: float = 120.0,
    glue_makeup_db: float = 0.0,
    glue_pdr: bool = True,
    glue_pdr_hold_ms: float = 500.0,
    clipper_bypass: bool = True,
    clipper_mode: str = "soft",
    clipper_ceiling: float = 0.98,
    clipper_drive_db: float = 0.0,
    limiter_ceiling: float = 0.95,
    limiter_release_ms: float = 80.0,
    limiter_bypass: bool = False,
    eq_mode: str = "iir",
    linear_phase_taps: int = 2049,
    low_end_mono_freq: float = 120.0,
    low_end_mono_amount: float = 0.0,
    reso_bypass: bool = True,
    reso_freq: float = 1200.0,
    reso_q: float = 3.0,
    reso_threshold_db: float = -18.0,
    reso_ratio: float = 3.0,
    reso_attack_ms: float = 5.0,
    reso_release_ms: float = 100.0,
    reso_max_reduction_db: float = 8.0,
    dyneq_bypass: bool = True,
    dyneq_freq: float = 3000.0,
    dyneq_q: float = 2.5,
    dyneq_threshold_db: float = -18.0,
    dyneq_ratio: float = 3.0,
    dyneq_attack_ms: float = 3.0,
    dyneq_release_ms: float = 80.0,
    dyneq_max_reduction_db: float = 12.0,
    nr_bypass: bool = True,
    nr_strength: float = 0.5,
    nr_noise_sample_sec: float = 0.5,
    output_format: str = "wav",
    output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
    dither_mode: str = "f_weighted",
    preview_seconds: float = None,
    platform_target: str = None,
    progress_cb=None,
) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")

    # Ajustar ceiling según plataforma.
    # BUGFIX: antes esto pisaba SIEMPRE el limiter_ceiling elegido por el
    # usuario (aunque hubiera puesto uno más conservador a propósito), sin
    # avisar en ningún lado — "el limiter no respeta lo que configuro" era
    # justamente esto. Ahora solo se aplica el techo de la plataforma si es
    # MÁS estricto (más bajo) que el que ya trae el usuario; si el usuario ya
    # pidió algo más seguro que el mínimo de la plataforma, se respeta.
    if platform_target:
        target = get_platform_target(platform_target)
        platform_ceiling = 10.0 ** (target["true_peak_db"] / 20.0)
        limiter_ceiling = min(limiter_ceiling, platform_ceiling)
        _platform_lufs_ceiling = float(target["lufs"])
    else:
        _platform_lufs_ceiling = None

    _report(progress_cb, 2, "Cargando archivo de audio")
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    if preview_seconds is not None:
        audio = _crop_preview(audio, sr, preview_seconds)

    # Se guarda una copia del audio ANTES de la cadena: el LUFS safety check
    # (más abajo) necesita re-renderizar la cadena completa desde el audio
    # original con un input_gain_db corregido en cada iteración. Si se
    # reutilizara `audio` después de `apply_mastering_chain` se estaría
    # re-encadenando la cadena sobre su propia salida en cada intento.
    audio_orig = audio.copy()

    _report(progress_cb, 5, "Analizando audio original")
    analysis_before = analyze_audio(audio, sr)

    # MEJORA (fix #6): construir MasteringParams UNA sola vez desde los locals.
    # Tanto la llamada inicial como el safety check usan este objeto — el único
    # punto de verdad de todos los parámetros de la cadena.
    _chain_params = MasteringParams(
        input_gain_db=input_gain_db,
        target_peak=target_peak,
        use_lufs_normalize=use_lufs_normalize,
        target_lufs=target_lufs,
        oversample_mode=oversample_mode,
        hp_cutoff=hp_cutoff,
        lp_bypass=lp_bypass, lp_cutoff=lp_cutoff,
        eq_mode=eq_mode,
        linear_phase_taps=linear_phase_taps,
        high_shelf_gain_db=high_shelf_gain_db,
        high_shelf_freq_hz=high_shelf_freq_hz,
        # BUGFIX: low_shelf_gain_db/low_shelf_freq_hz eran parámetros válidos
        # de process_audio() pero nunca se reenviaban a MasteringParams, así
        # que siempre caían al default (0.0 = bypass) sin importar lo que
        # mandara el usuario/frontend.
        low_shelf_gain_db=low_shelf_gain_db,
        low_shelf_freq_hz=low_shelf_freq_hz,
        eq1_freq=eq1_freq, eq1_gain=eq1_gain, eq1_q=eq1_q,
        eq2_freq=eq2_freq, eq2_gain=eq2_gain, eq2_q=eq2_q,
        eq3_freq=eq3_freq, eq3_gain=eq3_gain, eq3_q=eq3_q,
        eq4_freq=eq4_freq, eq4_gain=eq4_gain, eq4_q=eq4_q,
        eq5_freq=eq5_freq, eq5_gain=eq5_gain, eq5_q=eq5_q,
        eq6_freq=eq6_freq, eq6_gain=eq6_gain, eq6_q=eq6_q,
        tonal_balance_bypass=tonal_balance_bypass, tonal_balance_amount=tonal_balance_amount,
        tonal_balance_max_boost_db=tonal_balance_max_boost_db,
        tonal_balance_max_cut_db=tonal_balance_max_cut_db,
        tonal_balance_max_bands=tonal_balance_max_bands,
        ms_eq_bypass=ms_eq_bypass,
        ms_mid_freq=ms_mid_freq, ms_mid_gain=ms_mid_gain, ms_mid_q=ms_mid_q,
        ms_side_freq=ms_side_freq, ms_side_gain=ms_side_gain, ms_side_q=ms_side_q,
        ms_comp_bypass=ms_comp_bypass,
        ms_comp_mid_threshold_db=ms_comp_mid_threshold_db, ms_comp_mid_ratio=ms_comp_mid_ratio,
        ms_comp_mid_attack_ms=ms_comp_mid_attack_ms, ms_comp_mid_release_ms=ms_comp_mid_release_ms,
        ms_comp_mid_makeup_db=ms_comp_mid_makeup_db,
        ms_comp_side_threshold_db=ms_comp_side_threshold_db, ms_comp_side_ratio=ms_comp_side_ratio,
        ms_comp_side_attack_ms=ms_comp_side_attack_ms, ms_comp_side_release_ms=ms_comp_side_release_ms,
        ms_comp_side_makeup_db=ms_comp_side_makeup_db,
        ms_comp_pdr=ms_comp_pdr, ms_comp_pdr_hold_ms=ms_comp_pdr_hold_ms,
        reso_bypass=reso_bypass, reso_freq=reso_freq, reso_q=reso_q,
        reso_threshold_db=reso_threshold_db, reso_ratio=reso_ratio,
        reso_attack_ms=reso_attack_ms, reso_release_ms=reso_release_ms,
        reso_max_reduction_db=reso_max_reduction_db,
        dyneq_bypass=dyneq_bypass, dyneq_freq=dyneq_freq, dyneq_q=dyneq_q,
        dyneq_threshold_db=dyneq_threshold_db, dyneq_ratio=dyneq_ratio,
        dyneq_attack_ms=dyneq_attack_ms, dyneq_release_ms=dyneq_release_ms,
        dyneq_max_reduction_db=dyneq_max_reduction_db,
        transient_attack=transient_attack, transient_sustain=transient_sustain,
        mb_bypass=mb_bypass,
        mb_low_crossover=mb_low_crossover, mb_high_crossover=mb_high_crossover,
        mb_low_threshold_db=mb_low_threshold_db, mb_low_ratio=mb_low_ratio,
        mb_low_attack_ms=mb_low_attack_ms, mb_low_release_ms=mb_low_release_ms, mb_low_makeup_db=mb_low_makeup_db,
        mb_mid_threshold_db=mb_mid_threshold_db, mb_mid_ratio=mb_mid_ratio,
        mb_mid_attack_ms=mb_mid_attack_ms, mb_mid_release_ms=mb_mid_release_ms, mb_mid_makeup_db=mb_mid_makeup_db,
        mb_high_threshold_db=mb_high_threshold_db, mb_high_ratio=mb_high_ratio,
        mb_high_attack_ms=mb_high_attack_ms, mb_high_release_ms=mb_high_release_ms, mb_high_makeup_db=mb_high_makeup_db,
        mb_pdr=mb_pdr, mb_pdr_hold_ms=mb_pdr_hold_ms,
        comp_stereo_link=comp_stereo_link,
        comp_bypass=comp_bypass,
        comp_threshold_db=comp_threshold_db, comp_ratio=comp_ratio,
        comp_attack_ms=comp_attack_ms, comp_release_ms=comp_release_ms, comp_makeup_db=comp_makeup_db,
        comp_pdr=comp_pdr, comp_pdr_hold_ms=comp_pdr_hold_ms,
        glue_bypass=glue_bypass, glue_threshold_db=glue_threshold_db, glue_ratio=glue_ratio,
        glue_attack_ms=glue_attack_ms, glue_release_ms=glue_release_ms, glue_makeup_db=glue_makeup_db,
        glue_pdr=glue_pdr, glue_pdr_hold_ms=glue_pdr_hold_ms,
        saturation_drive=saturation_drive, saturation_mode=saturation_mode, saturation_mix=saturation_mix,
        mid_gain_db=mid_gain_db, side_gain_db=side_gain_db, stereo_width_amount=stereo_width_amount,
        stereo_bypass=stereo_bypass,
        use_stereo_enhancer=use_stereo_enhancer, enhancer_bass_mono_freq=enhancer_bass_mono_freq,
        haas_delay_ms=haas_delay_ms,
        low_end_mono_freq=low_end_mono_freq, low_end_mono_amount=low_end_mono_amount,
        mb_stereo_bypass=mb_stereo_bypass,
        mb_stereo_low_width=mb_stereo_low_width, mb_stereo_mid_width=mb_stereo_mid_width,
        mb_stereo_high_width=mb_stereo_high_width,
        mb_stereo_low_crossover=mb_stereo_low_crossover, mb_stereo_high_crossover=mb_stereo_high_crossover,
        reverb_size=reverb_size, reverb_wet=reverb_wet,
        clipper_bypass=clipper_bypass, clipper_mode=clipper_mode,
        clipper_ceiling=clipper_ceiling, clipper_drive_db=clipper_drive_db,
        limiter_ceiling=limiter_ceiling, limiter_bypass=limiter_bypass, limiter_release_ms=limiter_release_ms,
        nr_bypass=nr_bypass, nr_strength=nr_strength, nr_noise_sample_sec=nr_noise_sample_sec,
        parallel_bypass=parallel_bypass, parallel_threshold_db=parallel_threshold_db,
        parallel_ratio=parallel_ratio, parallel_attack_ms=parallel_attack_ms,
        parallel_release_ms=parallel_release_ms, parallel_mix=parallel_mix,
    )
    # MEJORA (perf): la reducción de ruido (etapa 0 de la cadena) opera sobre
    # el audio ANTES de aplicarse input_gain_db, así que su resultado no
    # depende de input_gain_db en absoluto. El LUFS safety check de más abajo
    # solo cambia input_gain_db entre reintentos — sin este bypass, cada uno
    # de los hasta 4 reintentos recalculaba la reducción de ruido (operación
    # espectral costosa, FFT por ventana) para obtener el mismo resultado
    # exacto una y otra vez. Ahora se corre una única vez acá y se fuerza
    # nr_bypass=True en todas las llamadas a apply_mastering_chain, que ya
    # reciben el audio pre-limpiado.
    _chain_kwargs = _chain_params.as_chain_kwargs()
    if not _chain_kwargs["nr_bypass"]:
        _report(progress_cb, 6, "Reduciendo ruido de fondo")
        audio_base = noise_reduction(
            audio_orig, sr,
            strength=_chain_kwargs["nr_strength"],
            noise_sample_sec=_chain_kwargs["nr_noise_sample_sec"],
        )
        _chain_kwargs["nr_bypass"] = True
    else:
        audio_base = audio_orig

    audio, chain_meters = apply_mastering_chain(
        audio_base, sr,
        progress_cb=progress_cb, progress_range=(8, 80),
        **_chain_kwargs,
    )

    # ── LUFS safety check ──────────────────────────────────────────────────
    # MEJORA (fix #6): antes el safety check duplicaba manualmente los ~50
    # kwargs de apply_mastering_chain, lo que hacía que cualquier parámetro
    # nuevo agregado a la firma quedara silenciosamente ignorado en las
    # re-renderizaciones. Ahora se usa MasteringParams como único punto de
    # verdad: solo se actualiza input_gain_db y se pasa el dict completo,
    # eliminando la posibilidad de desincronización.
    # BUGFIX: esta lógica vivía solo acá adentro, así que cualquier otro
    # caller que necesitara normalización LUFS (p.ej. el streaming en vivo,
    # que llama a apply_mastering_chain directo sin pasar por process_audio)
    # no tenía forma de reusarla. Se extrajo a run_lufs_safety_check().
    if use_lufs_normalize:
        audio, chain_meters, lufs_safety_notes, final_input_gain = run_lufs_safety_check(
            audio_base, sr, _chain_kwargs, target_lufs,
            initial_audio=audio, initial_chain_meters=chain_meters,
            progress_cb=progress_cb, progress_start=80,
            adaptive_loudness_weighting=adaptive_loudness_weighting,
            loudness_sensitivity_amount=loudness_sensitivity_amount,
            max_target_lufs=_platform_lufs_ceiling,
        )
        chain_meters["lufs_safety"] = {
            "enabled": True,
            "target_lufs": round(float(target_lufs), 2),
            "final_input_gain_db": round(final_input_gain, 2),
            "adaptive_loudness_weighting": bool(adaptive_loudness_weighting),
            "notes": lufs_safety_notes,
        }
    else:
        chain_meters["lufs_safety"] = {"enabled": False}

    _report(progress_cb, 93, "Analizando resultado final")
    analysis_after = analyze_audio(audio, sr)

    base = os.path.splitext(os.path.basename(input_path))[0]
    suffix = "_preview" if preview_seconds else "_mastered"
    output_dir = "processed"
    # El motor puede utilizarse fuera de FastAPI (CLI, tests, worker dedicado).
    # No debe depender de que app.py haya creado previamente el directorio.
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base}_{uuid.uuid4().hex[:8]}{suffix}.{output_format}")

    # BUGFIX: ser consistente con el shape del output
    # Si es mono (shape[0]==1), devolver 1D para soundfile (que espera (n_samples,) para mono)
    # Si es estéreo (shape[0]==2), devolver (n_channels, n_samples) para soundfile
    if audio.ndim == 2:
        audio_out = audio[0] if audio.shape[0] == 1 else audio
    else:
        audio_out = audio

    _report(progress_cb, 97, "Guardando archivo masterizado")
    effective_bit_depth, dither_meta = _write_master_output(
        audio_out, sr, output_path, output_format, output_bit_depth,
        dither_mode=dither_mode,
    )

    _report(progress_cb, 100, "Mastering completado")
    return {
        "output_path":       output_path,
        "output_bit_depth":  effective_bit_depth,
        "dither":            dither_meta,
        "analysis_before":   analysis_before,
        "analysis_after":    analysis_after,
        "mix_advice_before": mix_advice(analysis_before),
        "mix_advice_after":  mix_advice(analysis_after),
        "recommendations_before": generate_mastering_recommendations(analysis_before),
        "recommendations_after": generate_mastering_recommendations(analysis_after),
        "chain_meters":      chain_meters,
    }

# ─── Mastering por referencia (reference-track matching) ──────────────────────
# Toma un track de referencia (ya masterizado, del sonido que se quiere imitar)
# y adapta el track propio hacia ese objetivo en 4 dimensiones:
#   1. Balance tonal (EQ de "matching" multibanda, derivada de la diferencia
#      espectral entre ambos tracks)
#   2. Loudness (LUFS integrado)
#   3. Dinámica (crest factor: si el propio track es mucho más dinámico que
#      la referencia, se aplica un compresor de banda ancha suave para acercarlo)
#   4. Ancho estéreo (correlación L/R)
# y finalmente limita al techo de pico aproximado de la referencia.

def _load_audio_any(path: str):
    audio, sr = librosa.load(path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return audio, sr

def spectral_energy_at_bands(audio: np.ndarray, sr: int, band_edges: list) -> list:
    """Energía promedio (dB) en bandas de frecuencia arbitrarias (lo, hi) en Hz.
    A diferencia de spectrum_analysis_fft (que usa bandas log fijas 20Hz..nyquist
    propio de cada archivo), esta función recibe los mismos band_edges para dos
    archivos con distinto sample rate, permitiendo comparar "manzanas con manzanas".

    Es la base directa de `compute_reference_eq_curve`: los frames se muestrean
    con `_averaged_magnitude_spectrum`, distribuidos en TODO el archivo (ver esa
    función), no solo en los primeros segundos.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n_fft = min(8192, len(mono))
    n_fft = max(256, 2 ** int(np.floor(np.log2(max(n_fft, 256)))))
    avg_mag = _averaged_magnitude_spectrum(mono, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    edges = [band_edges[0][0]] + [hi for _, hi in band_edges]
    return _log_band_average_db(freqs, avg_mag, np.array(edges, dtype=float)).tolist()

def spectral_energy_at_bands_multires(audio: np.ndarray, sr: int, band_edges: list,
                                      fft_sizes: tuple = (1024, 4096, 16384)) -> dict:
    """Versión multi-resolución de spectral_energy_at_bands, inspirada en el
    approach de multi-resolution STFT loss (el mismo truco que usan librerías
    tipo auraloss para evaluar modelos generativos de audio, adaptado acá:
    esa clase de loss compara señales ALINEADAS en el tiempo del mismo
    contenido, cosa que src/ref NO son — son dos canciones distintas — así
    que en vez de comparar frame a frame, se combina la curva de balance
    tonal calculada a VARIAS resoluciones de FFT).

    Una sola ventana FFT subestima el contenido según el material: ventanas
    cortas (1024) resuelven mejor transientes/percusión pero promedian mal
    tonos sostenidos; ventanas largas (16384) dan resolución fina de
    frecuencia para tonos sostenidos pero diluyen transientes en el
    promedio. Combinar 3 resoluciones da una curva de balance tonal más
    representativa sin importar si el material es percusivo, tonal, o mixto.

    Devuelve un dict {fft_size: [banda_db, ...]} con una curva por resolución.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    edges = np.array([band_edges[0][0]] + [hi for _, hi in band_edges], dtype=float)
    out = {}
    for n_fft in fft_sizes:
        n_fft_eff = min(n_fft, len(mono))
        n_fft_eff = max(256, 2 ** int(np.floor(np.log2(max(n_fft_eff, 256)))))
        avg_mag = _averaged_magnitude_spectrum(mono, n_fft_eff)
        freqs = np.fft.rfftfreq(n_fft_eff, d=1.0 / sr)
        out[n_fft] = _log_band_average_db(freqs, avg_mag, edges).tolist()
    return out


def spectral_balance_by_dynamic_range(audio: np.ndarray, sr: int,
                                       band_edges: list,
                                       n_percentile_bins: int = 4,
                                       min_segment_sec: float = 0.5,
                                       fft_size: int = 4096) -> dict:
    """Analiza cómo cambia el balance espectral según el nivel dinámico del material.

    El análisis espectral promedio (usado en compute_reference_eq_curve) captura
    el balance GLOBAL del track pero pierde una dimensión crítica: las partes
    suaves y las partes fuertes de un master profesional tienen balances tonales
    DISTINTOS. Un master bien hecho tiende a tener más presencia de graves en los
    picos y más aire/brillo en las partes suaves — algo que ningún análisis
    promedio puede capturar.

    Esta función:
      1. Segmenta el audio en ventanas de min_segment_sec segundos.
      2. Calcula el RMS de cada ventana para ordenarlas por nivel dinámico.
      3. Agrupa las ventanas en n_percentile_bins cuartiles (por defecto 4):
         Q1=partes más suaves, Q4=partes más fuertes.
      4. Calcula el balance espectral de cada cuartil por separado.
      5. Devuelve el perfil dinámico: {bin_label: [band_db, ...]} + metadata.

    Esto permite comparar no solo "¿cómo suena la referencia en promedio?"
    sino "¿cómo suena la referencia cuando está suave vs cuando está fuerte?"
    y aplicar una corrección distinta según el nivel instantáneo del material.

    n_percentile_bins: cantidad de grupos dinámicos (2=suave/fuerte, 4=cuartiles)
    min_segment_sec: tamaño mínimo de cada ventana de análisis en segundos
    fft_size: tamaño de FFT para el análisis espectral por segmento
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float32)
    n = len(mono)
    seg_len = max(int(sr * min_segment_sec), 512)
    n_segs = max(1, n // seg_len)

    # Calcular RMS por segmento
    segments = []
    for i in range(n_segs):
        start = i * seg_len
        end = min(start + seg_len, n)
        seg = mono[start:end]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        segments.append({"start": start, "end": end, "rms": rms, "rms_db": 20.0 * np.log10(rms)})

    if not segments:
        return {"bins": {}, "n_segments": 0, "metadata": {}}

    # Ordenar por RMS y agrupar en cuartiles
    segments_sorted = sorted(segments, key=lambda s: s["rms"])
    bin_size = max(1, len(segments_sorted) // n_percentile_bins)

    edges = np.array([band_edges[0][0]] + [hi for _, hi in band_edges], dtype=float)
    fft_size_eff = min(fft_size, seg_len)
    fft_size_eff = max(256, 2 ** int(np.floor(np.log2(max(fft_size_eff, 256)))))
    freqs = np.fft.rfftfreq(fft_size_eff, d=1.0 / sr)

    bin_labels = {0: "quiet", 1: "soft", 2: "medium", 3: "loud"}
    if n_percentile_bins == 2:
        bin_labels = {0: "quiet", 1: "loud"}

    result_bins = {}
    bin_rms_ranges = {}
    window = np.hanning(fft_size_eff)

    for bin_idx in range(n_percentile_bins):
        start_idx = bin_idx * bin_size
        end_idx = start_idx + bin_size if bin_idx < n_percentile_bins - 1 else len(segments_sorted)
        bin_segs = segments_sorted[start_idx:end_idx]
        if not bin_segs:
            continue

        # Acumular espectros de todos los segmentos en este cuartil
        acc_power = np.zeros(fft_size_eff // 2 + 1, dtype=np.float64)
        count = 0
        for seg_info in bin_segs:
            seg = mono[seg_info["start"]:seg_info["end"]]
            if len(seg) < fft_size_eff:
                padded = np.zeros(fft_size_eff, dtype=np.float32)
                padded[:len(seg)] = seg
                seg = padded
            else:
                # Si el segmento es más largo que fft_size_eff, Welch interno
                hop = fft_size_eff // 2
                n_frames = (len(seg) - fft_size_eff) // hop + 1
                frame_power = np.zeros(fft_size_eff // 2 + 1, dtype=np.float64)
                for fi in range(n_frames):
                    frame = seg[fi * hop: fi * hop + fft_size_eff].astype(np.float64)
                    mag = np.abs(np.fft.rfft(frame * window))
                    frame_power += mag ** 2
                acc_power += frame_power / max(n_frames, 1)
                count += 1
                continue
            mag = np.abs(np.fft.rfft(seg.astype(np.float64) * window))
            acc_power += mag ** 2
            count += 1

        if count == 0:
            continue

        avg_mag = np.sqrt(acc_power / count)
        bands_db = _log_band_average_db(freqs, avg_mag, edges).tolist()

        label = bin_labels.get(bin_idx, f"q{bin_idx + 1}")
        result_bins[label] = bands_db
        bin_rms_ranges[label] = {
            "rms_db_min": round(bin_segs[0]["rms_db"], 1),
            "rms_db_max": round(bin_segs[-1]["rms_db"], 1),
            "n_segments": len(bin_segs),
        }

    # Calcular diferencia espectral entre parte fuerte y parte suave
    # (cuánto cambia el balance tonal con la dinámica)
    tonal_slope = {}
    loud_key = bin_labels.get(n_percentile_bins - 1, "loud")
    quiet_key = bin_labels.get(0, "quiet")
    if loud_key in result_bins and quiet_key in result_bins:
        loud_bands = np.array(result_bins[loud_key])
        quiet_bands = np.array(result_bins[quiet_key])
        diff = (loud_bands - quiet_bands).tolist()
        tonal_slope = {
            "loud_vs_quiet_db": [round(d, 2) for d in diff],
            "description": "Diferencia espectral entre partes fuertes y suaves (+ = más energía en partes fuertes)",
        }

    return {
        "bins": result_bins,
        "bin_ranges": bin_rms_ranges,
        "tonal_slope": tonal_slope,
        "n_segments": len(segments),
        "fft_size": fft_size_eff,
        "n_percentile_bins": n_percentile_bins,
    }


def compute_dynamic_range_eq_curves(src_profile: dict, ref_profile: dict,
                                     centers: list,
                                     max_boost_db: float = 4.0,
                                     max_cut_db: float = -6.0,
                                     blend: float = 0.65) -> dict:
    """Calcula curvas de EQ de matching POR BIN DINÁMICO.

    En lugar de una única curva promedio, devuelve una curva distinta para
    cada nivel dinámico (quiet/soft/medium/loud). El resultado es un dict
    {bin_label: [(freq_hz, gain_db), ...]} que luego apply_dynamic_range_eq_matching
    usa para aplicar la corrección correcta según el nivel instantáneo del audio.

    La corrección se limita a max_boost/max_cut más conservadores que el EQ
    promedio (4dB/-6dB vs 6dB/-9dB) porque opera sobre el carácter dinámico
    del material — correcciones agresivas aquí suenan artificiales.
    """
    src_bins = src_profile.get("bins", {})
    ref_bins = ref_profile.get("bins", {})
    common_bins = [b for b in src_bins if b in ref_bins]

    curves = {}
    for bin_label in common_bins:
        src_bands = src_bins[bin_label]
        ref_bands = ref_bins[bin_label]
        curve = compute_reference_eq_curve(
            src_bands, ref_bands, centers,
            max_boost_db=max_boost_db,
            max_cut_db=max_cut_db,
            blend=blend,
        )
        curves[bin_label] = curve

    return curves


def apply_dynamic_range_eq_matching(audio: np.ndarray, sr: int,
                                     dynamic_curves: dict,
                                     eq_q: float = 1.3,
                                     min_segment_sec: float = 0.5,
                                     crossfade_sec: float = 0.05) -> np.ndarray:
    """Aplica el EQ de matching con curvas distintas según el nivel dinámico.

    Usa overlap-add con ventana de Hann para eliminar los micro cortes en los
    bordes de segmento. El FIR lineal introduce un retardo de grupo de
    len(taps)//2 samples — compensamos ese retardo por canal antes de
    reconstruir la señal, así no hay desplazamiento temporal entre segmentos
    procesados con distintas curvas.

    El tamaño de la ventana de overlap es max(len(taps), cf_len) para
    garantizar que el FIR tiene contexto suficiente en cada segmento y no
    produce artefactos en los bordes.
    """
    if not dynamic_curves:
        return audio

    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n = len(mono)
    seg_len = max(int(sr * min_segment_sec), 512)
    n_segs = max(1, n // seg_len)
    cf_len = max(int(sr * crossfade_sec), 64)

    # Pre-compilar FIR por curva
    fir_cache = {}
    max_taps = 0
    for label, curve in dynamic_curves.items():
        taps = build_matching_fir(curve, sr, precision=eq_q)
        fir_cache[label] = taps
        max_taps = max(max_taps, len(taps))

    # El overlap debe ser al menos tan largo como el FIR para evitar artefactos
    overlap = max(cf_len, max_taps)
    half_delay = max_taps // 2  # retardo de grupo del FIR lineal fase

    # Calcular RMS de todos los segmentos para asignar cuartiles
    seg_rms = []
    for i in range(n_segs):
        start = i * seg_len
        end = min(start + seg_len, n)
        seg = mono[start:end]
        seg_rms.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12))

    n_bins = len(dynamic_curves)
    bin_labels_ordered = ["quiet", "soft", "medium", "loud"][:n_bins]
    if n_bins == 2:
        bin_labels_ordered = ["quiet", "loud"]

    rms_sorted = sorted(seg_rms)
    bin_size = max(1, len(rms_sorted) // n_bins)
    thresholds = [rms_sorted[min(i * bin_size, len(rms_sorted) - 1)] for i in range(1, n_bins)]

    def _assign_bin(rms_val: float) -> str:
        for i, thr in enumerate(thresholds):
            if rms_val <= thr:
                return bin_labels_ordered[i]
        return bin_labels_ordered[-1]

    # Ventana de Hann para overlap-add
    win = np.hanning(overlap * 2).astype(np.float64)
    fade_in  = win[:overlap]          # 0 → 1
    fade_out = win[overlap:]          # 1 → 0

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        ch64 = ch.astype(np.float64)
        out = np.zeros(n + half_delay, dtype=np.float64)
        norm = np.zeros(n + half_delay, dtype=np.float64)

        for i in range(n_segs):
            start = i * seg_len
            end = min(start + seg_len, n)
            if start >= n:
                break

            # Extraer segmento con contexto de overlap a ambos lados
            ctx_start = max(0, start - overlap)
            ctx_end   = min(n, end + overlap)
            seg = ch64[ctx_start:ctx_end]

            bin_label = _assign_bin(seg_rms[i])
            taps = fir_cache.get(bin_label)

            if taps is not None:
                filtered = fftconvolve(seg, taps, mode="full")
                # Compensar retardo de grupo: el FIR lineal desplaza la señal
                # half_delay samples hacia la derecha — lo recortamos acá
                filt_start = (start - ctx_start)          # offset del segmento dentro del contexto
                filt_end   = filt_start + (end - start)
                # Los samples relevantes en el output de fftconvolve están en
                # [filt_start : filt_end + half_delay] — compensamos el delay
                out_seg = filtered[filt_start:filt_end + half_delay]
            else:
                filt_start = start - ctx_start
                out_seg = seg[filt_start:filt_start + (end - start) + half_delay]

            seg_out_len = len(out_seg)
            out_start = start
            out_end   = out_start + seg_out_len

            # Ventana rectangular con rampas de overlap-add en los bordes
            window = np.ones(seg_out_len, dtype=np.float64)
            left_len  = min(overlap, seg_out_len)
            right_len = min(overlap, seg_out_len)
            if start > 0:
                window[:left_len] *= fade_in[-left_len:]
            if end < n:
                window[max(0, seg_out_len - right_len):] *= fade_out[:right_len]

            out[out_start:out_end]  += out_seg * window
            norm[out_start:out_end] += window

        # Normalizar overlap-add y recortar retardo compensado
        norm = np.where(norm < 1e-9, 1.0, norm)
        result = (out / norm)[half_delay:half_delay + n]
        return result.astype(np.float32)

    if audio.ndim == 1:
        return _process_channel(audio)
    return np.stack([_process_channel(audio[c]) for c in range(audio.shape[0])])


def spectral_match_score_multires(a_multires: dict, b_multires: dict) -> dict:
    """Combina spectral_match_score() por resolución (misma métrica: MAE tras
    remover offset de nivel, mapeado a % de similitud), promediando entre
    todas las resoluciones de FFT. Devuelve el score combinado + el detalle
    por resolución (útil para ver si el desajuste es de transientes/banda
    ancha —resolución corta— o de tonalidad fina —resolución larga—).
    """
    per_res = {}
    for n_fft in a_multires:
        if n_fft not in b_multires:
            continue
        per_res[n_fft] = spectral_match_score(a_multires[n_fft], b_multires[n_fft])
    if not per_res:
        return {"mean_abs_diff_db": 0.0, "match_percent": 100.0, "per_resolution": {}}
    combined_mae = float(np.mean([v["mean_abs_diff_db"] for v in per_res.values()]))
    combined_score = float(np.clip(100.0 - combined_mae * 8.0, 0.0, 100.0))
    return {
        "mean_abs_diff_db": round(combined_mae, 2),
        "match_percent": round(combined_score, 1),
        "per_resolution": {str(k): v for k, v in per_res.items()},
    }


def spectral_match_score(a_db: list, b_db: list) -> dict:
    """Puntaje de similitud espectral entre dos curvas de bandas (0-100%).
    Ignora el offset general de nivel (loudness), solo mide diferencias de
    'forma'/balance tonal.
    """
    a = np.array(a_db, dtype=np.float64)
    b = np.array(b_db, dtype=np.float64)
    diff = (a - b)
    diff = diff - np.mean(diff)
    mae = float(np.mean(np.abs(diff)))
    score = float(np.clip(100.0 - mae * 8.0, 0.0, 100.0))
    return {"mean_abs_diff_db": round(mae, 2), "match_percent": round(score, 1)}

def _soft_clip_curve(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Soft-knee clip (tanh) para una curva de ganancias en dB.

    A diferencia de np.clip (que genera un quiebre de pendiente instantáneo
    apenas la curva toca el límite), esta versión se comporta casi como
    identidad lejos de los límites y se aplana suavemente cerca de/superando
    `lo`/`hi`. Esos quiebres abruptos son justamente lo que firwin2 traduce
    en ringing/artefactos audibles al construir el FIR de matching, así que
    evitarlos mejora directamente qué tan "limpio" suena el resultado.
    """
    center = (hi + lo) / 2.0
    half_range = (hi - lo) / 2.0
    if half_range <= 1e-9:
        return np.clip(x, lo, hi)
    return center + np.tanh((x - center) / half_range) * half_range

def compute_reference_eq_curve(src_bands_db: list, ref_bands_db: list, freqs_hz: list,
                                max_boost_db: float = 6.0, max_cut_db: float = -9.0,
                                smooth_window: int = 3,
                                blend: float = 0.75) -> list:
    """Calcula la curva de EQ (freq, gain_db) necesaria para acercar el balance
    tonal de src hacia ref. Se resta la media global (el loudness se maneja
    aparte, vía LUFS) y se suaviza/clippea para evitar EQs extremas o ásperas.

    MEJORA (matching "áspero"/artificial): la versión anterior suavizaba con
    un promedio móvil rectangular muy débil (y ademas 'mode=same', que atenúa
    los extremos hacia cero por el padding implícito) y después recortaba con
    un clip duro. El resultado era una curva objetivo con micro-escalones y
    quiebres de pendiente que build_matching_fir (via firwin2) reproduce
    fielmente como ringing/coloración áspera — el FIR es literalmente tan
    "prolijo" como la curva que se le pide. Ahora:
      1) Se suaviza con un kernel gaussiano (con padding por reflexión, no
         ceros) cuyo ancho escala con la cantidad de bandas, para una curva
         continua sin perder resolución real.
      2) Se recorta con soft-knee (tanh) en vez de clip duro.
      3) Se re-centra después del recorte, porque un soft-knee asimétrico
         (max_boost != |max_cut|) puede introducir un pequeño remanente de
         nivel medio que de otro modo se solaparía con el ajuste de LUFS.
      4) Se aplica un blend y límites perceptuales por zona para evitar
         matches 100% demasiado artificiales, especialmente en sub y presencia.
    """
    src = np.array(src_bands_db, dtype=np.float64)
    ref = np.array(ref_bands_db, dtype=np.float64)
    n = len(src)
    diff = ref - src
    diff = diff - np.mean(diff)

    if smooth_window > 1 and n > 2:
        # Suavizado aumentado: sigma escala con bandas para evitar mesetas en presencia
        sigma = max(smooth_window / 1.8, 1.0)   # era /2.5 — suavizado más agresivo
        radius = min(n - 1, max(1, int(round(sigma * 3))))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(diff, radius, mode="reflect")
        diff = np.convolve(padded, kernel, mode="valid")

    diff = _soft_clip_curve(diff, max_cut_db, max_boost_db)
    diff = diff - np.mean(diff)

    freqs = np.array(freqs_hz, dtype=np.float64)
    zone_boost = np.full_like(freqs, max_boost_db, dtype=np.float64)
    zone_cut = np.full_like(freqs, max_cut_db, dtype=np.float64)
    zone_boost[freqs < 80.0] = min(max_boost_db, 3.0)
    zone_cut[freqs < 80.0] = max(max_cut_db, -6.0)
    # Presencia más conservadora — zona crítica para la percepción de distorsión
    presence = (freqs >= 2000.0) & (freqs <= 8000.0)
    zone_boost[presence] = min(max_boost_db, 2.0)   # era 3.5 — demasiado
    zone_cut[presence] = max(max_cut_db, -5.0)
    air = freqs >= 10000.0
    zone_boost[air] = min(max_boost_db, 5.0)
    zone_cut[air] = max(max_cut_db, -7.0)

    blend = float(np.clip(blend, 0.0, 1.0))
    diff = np.clip(diff * blend, zone_cut, zone_boost)

    # ── Límite de pendiente entre bandas adyacentes ───────────────────────────
    # Evita escalones bruscos (ej: 5 bandas seguidas al tope) que el FIR
    # reproduce como ringing/coloración en presencia. Máximo 0.8 dB/banda.
    MAX_SLOPE_DB = 0.8
    for _ in range(2):   # dos pasadas para suavizar picos aislados
        for j in range(1, len(diff)):
            slope = diff[j] - diff[j-1]
            if abs(slope) > MAX_SLOPE_DB:
                diff[j] = diff[j-1] + np.sign(slope) * MAX_SLOPE_DB
        for j in range(len(diff)-2, -1, -1):
            slope = diff[j] - diff[j+1]
            if abs(slope) > MAX_SLOPE_DB:
                diff[j] = diff[j+1] + np.sign(slope) * MAX_SLOPE_DB

    return [(float(f), float(g)) for f, g in zip(freqs_hz, diff.tolist())]

def compute_reference_eq_curve_ddsp(src_bands_multires: dict, ref_bands_multires: dict,
                                    freqs_hz: list,
                                    max_boost_db: float = 6.0, max_cut_db: float = -9.0,
                                    smoothness_weight: float = 0.15,
                                    l2_weight: float = 0.02,
                                    iters: int = 300, lr: float = 0.08) -> list:
    """Alternativa DDSP a compute_reference_eq_curve(): en vez de estimar la
    curva de ganancias con una fórmula cerrada (resta + suavizado gaussiano +
    soft-clip, todo en un solo paso), la resuelve por DESCENSO DE GRADIENTE
    (Adam, torch) minimizando una pérdida multi-resolución contra las curvas
    ya calculadas por spectral_energy_at_bands_multires (antes vs. referencia,
    a 3 tamaños de FFT). Es DDSP en el sentido de "differentiable digital
    signal processing": el mapeo ganancias-de-banda -> curva-de-EQ-en-dB es
    diferenciable y se optimiza directamente, en vez de derivarse con una
    heurística de un solo paso.

    Qué gana frente a la heurística:
      - Optimiza las 3 resoluciones (1024/4096/16384) A LA VEZ con una sola
        pérdida ponderada, en vez de que compute_reference_eq_curve solo vea
        una resolución fija.
      - La suavidad entre bandas vecinas y el límite de boost/cut NO son un
        post-proceso (gaussian blur + tanh clip) sino restricciones dentro de
        la propia función de pérdida (penalización L2 + penalización de
        diferencia entre bandas contiguas), así el optimizador balancea
        "parecerse a la referencia" contra "ser una curva razonable" en
        conjunto, en vez de encontrar primero el óptimo crudo y F suavizarlo
        después (lo cual puede alejarlo del óptimo real).

    Requiere torch (ya es dependencia del proyecto por demucs/torchaudio).
    Si no está disponible, cae de vuelta a compute_reference_eq_curve().

    Devuelve el mismo formato que compute_reference_eq_curve(): lista de
    tuplas (freq_hz, gain_db), lista para pasar a build_matching_fir.
    """
    if not HAS_TORCH:
        # Fallback: usa la resolución media (4096, la misma que usaba el
        # pipeline single-res histórico) con la heurística de siempre.
        return compute_reference_eq_curve(
            src_bands_multires.get(4096, next(iter(src_bands_multires.values()))),
            ref_bands_multires.get(4096, next(iter(ref_bands_multires.values()))),
            freqs_hz, max_boost_db=max_boost_db, max_cut_db=max_cut_db)

    import torch  # import perezoso: recién acá se carga (ver nota en HAS_TORCH más arriba)

    device = torch.device("cpu")
    n_bands = len(freqs_hz)

    # Pesos por resolución: la resolución larga (16384) tiene mejor detalle
    # tonal fino, así que pesa un poco más que la corta (1024, más sensible a
    # fugas espectrales/transientes que no son "balance tonal" real).
    res_weights = {1024: 0.8, 4096: 1.0, 16384: 1.2}

    diffs_by_res = {}
    for n_fft, src_curve in src_bands_multires.items():
        if n_fft not in ref_bands_multires:
            continue
        src_t = torch.tensor(src_curve, dtype=torch.float32, device=device)
        ref_t = torch.tensor(ref_bands_multires[n_fft], dtype=torch.float32, device=device)
        target = ref_t - src_t
        target = target - target.mean()  # el nivel global lo maneja LUFS aparte
        diffs_by_res[n_fft] = target

    # Parámetro libre a optimizar: pre-activación (gain sin acotar), se pasa
    # por tanh para respetar max_boost_db/max_cut_db de forma diferenciable
    # (equivalente al _soft_clip_curve de la versión heurística, pero acá
    # es PARTE del grafo de optimización, no un post-proceso).
    raw_gain = torch.zeros(n_bands, dtype=torch.float32, device=device, requires_grad=True)
    center = (max_boost_db + max_cut_db) / 2.0
    half_range = (max_boost_db - max_cut_db) / 2.0

    optimizer = torch.optim.Adam([raw_gain], lr=lr)

    for _ in range(max(1, int(iters))):
        optimizer.zero_grad()
        gain_curve = center + torch.tanh(raw_gain) * half_range

        loss = torch.tensor(0.0, device=device)
        for n_fft, target in diffs_by_res.items():
            w = res_weights.get(n_fft, 1.0)
            loss = loss + w * torch.mean((gain_curve - target) ** 2)

        # Penalización de suavidad: diferencia entre bandas log-vecinas.
        # Reemplaza el suavizado gaussiano post-hoc de la versión heurística.
        if n_bands > 1:
            smooth_pen = torch.mean((gain_curve[1:] - gain_curve[:-1]) ** 2)
            loss = loss + smoothness_weight * smooth_pen

        # Penalización L2: preferí curvas de EQ chicas si el ajuste extra no
        # mejora mucho el match (evita "perseguir" ruido de la medición).
        loss = loss + l2_weight * torch.mean(gain_curve ** 2)

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_gain = (center + torch.tanh(raw_gain) * half_range)
        final_gain = final_gain - final_gain.mean()
        final_gain = torch.clamp(final_gain, max_cut_db, max_boost_db)

    gains = final_gain.cpu().numpy().tolist()
    return [(float(f), float(g)) for f, g in zip(freqs_hz, gains)]


def apply_reference_eq_curve(audio: np.ndarray, sr: int, curve: list,
                             q: float = 1.1, min_gain_db: float = 0.3) -> np.ndarray:
    """(Legacy) Aplica la curva de matching como una cascada de filtros
    paramétricos (peaking), uno por banda. Se mantiene por si se necesita un
    EQ 'de consola' clásico, pero para el matching automático se usa
    build_matching_fir/apply_matching_fir (ver abajo), que evita el problema
    de bandas log-espaciadas vecinas que se solapan y se refuerzan entre sí
    al aplicarse en cascada, produciendo overshoot en vez de la curva pedida.
    """
    out = audio
    for freq, gain in curve:
        if abs(gain) < min_gain_db:
            continue
        out = eq_parametric_band(out, sr, freq=freq, gain_db=gain, q=q)
    return out

def build_matching_fir(curve: list, sr: int, precision: float = 1.1) -> np.ndarray:
    """Diseña un filtro FIR de fase lineal que realiza la curva de matching
    (freq_hz, gain_db) directamente en el dominio de la frecuencia, vía
    scipy.signal.firwin2. A diferencia de encadenar N filtros paramétricos
    (que se solapan/interfieren entre sí, especialmente con bandas log
    espaciadas cercanas), esto aplica la respuesta de frecuencia objetivo tal
    cual, banda por banda, sin refuerzo cruzado — el enfoque estándar de los
    plugins de "spectral match EQ".
    `precision` escala la cantidad de taps (resolución en frecuencia/octava);
    valores más altos = filtro más preciso/angosto pero más costoso.
    """
    from scipy.signal import firwin2
    if not curve:
        return None
    nyq = sr / 2.0
    freqs = np.clip(np.array([f for f, _ in curve], dtype=np.float64), 1.0, nyq - 1.0)
    gains_db = np.array([g for _, g in curve], dtype=np.float64)
    # Extiende con puntos planos en 0 Hz y en Nyquist para que firwin2 tenga
    # un dominio completo 0..nyquist bien definido.
    freqs_ext = np.concatenate(([0.0], freqs, [nyq]))
    gains_ext = np.concatenate(([gains_db[0]], gains_db, [gains_db[-1]]))
    for i in range(1, len(freqs_ext)):
        if freqs_ext[i] <= freqs_ext[i - 1]:
            freqs_ext[i] = freqs_ext[i - 1] + 1e-3
    freqs_norm = np.clip(freqs_ext / nyq, 0.0, 1.0)
    gains_lin = 10.0 ** (gains_ext / 20.0)
    numtaps = int(np.clip(2048 * precision, 511, 8191))
    if numtaps % 2 == 0:
        numtaps += 1
    return firwin2(numtaps, freqs_norm, gains_lin)

def apply_matching_fir(audio: np.ndarray, sr: int, taps) -> np.ndarray:
    """Aplica el FIR de matching con convolución 'same' (fftconvolve), que
    para un FIR de fase lineal compensa exactamente el delay de grupo,
    quedando efectivamente en fase cero."""
    if taps is None or len(taps) == 0:
        return audio
    if audio.ndim == 1:
        return fftconvolve(audio, taps, mode="same")
    return np.stack([fftconvolve(ch, taps, mode="same") for ch in audio])


def encode_ms(audio: np.ndarray) -> tuple:
    """Codifica stereo L/R → Mid/Side. Devuelve (mid, side) como arrays 1D."""
    if audio.ndim == 1 or audio.shape[0] == 1:
        mono = audio if audio.ndim == 1 else audio[0]
        return mono.copy(), np.zeros_like(mono)
    L, R = audio[0].astype(np.float64), audio[1].astype(np.float64)
    mid  = (L + R) * 0.5
    side = (L - R) * 0.5
    return mid, side


def decode_ms(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Decodifica Mid/Side → stereo L/R."""
    L = mid + side
    R = mid - side
    return np.stack([L, R]).astype(np.float32)


def compute_ms_eq_curves(audio: np.ndarray, sr: int,
                         ref_audio: np.ndarray, ref_sr: int,
                         band_edges: list,
                         centers: list,
                         max_boost_db: float = 5.0,
                         max_cut_db: float = -8.0,
                         blend: float = 0.75,
                         eq_fit_method: str = "heuristic",
                         src_bands_multires: dict = None,
                         ref_bands_multires: dict = None) -> tuple:
    """Calcula curvas de EQ de matching INDEPENDIENTES para Mid y Side.

    Por qué M/S en vez de L/R:
      - El Mid contiene la información mono (kick, bajo, voz central).
      - El Side contiene la diferencia L-R (reverb, room, wide synths).
      - En referencias profesionales el Side suele tener más aire (>8kHz)
        y menos sub (<120Hz) que el Mid. Matchear L/R no captura esto.

    Retorna: (curve_mid, curve_side) — mismo formato que compute_reference_eq_curve.
    """
    src_mid, src_side = encode_ms(audio)
    ref_mid, ref_side = encode_ms(ref_audio)

    if ref_sr != sr:
        import librosa as _lr
        ref_mid  = _lr.resample(ref_mid.astype(np.float32),  orig_sr=ref_sr, target_sr=sr).astype(np.float64)
        ref_side = _lr.resample(ref_side.astype(np.float32), orig_sr=ref_sr, target_sr=sr).astype(np.float64)

    def _to2d(x): return x[np.newaxis, :] if x.ndim == 1 else x

    if eq_fit_method == "ddsp":
        src_mid_mr  = spectral_energy_at_bands_multires(_to2d(src_mid),  sr, band_edges)
        ref_mid_mr  = spectral_energy_at_bands_multires(_to2d(ref_mid),  sr, band_edges)
        src_side_mr = spectral_energy_at_bands_multires(_to2d(src_side), sr, band_edges)
        ref_side_mr = spectral_energy_at_bands_multires(_to2d(ref_side), sr, band_edges)
        curve_mid  = compute_reference_eq_curve_ddsp(src_mid_mr,  ref_mid_mr,  centers,
                                                      max_boost_db=max_boost_db, max_cut_db=max_cut_db)
        curve_side = compute_reference_eq_curve_ddsp(src_side_mr, ref_side_mr, centers,
                                                      max_boost_db=max_boost_db, max_cut_db=max_cut_db)
    else:
        src_mid_b  = spectral_energy_at_bands(_to2d(src_mid),  sr, band_edges)
        ref_mid_b  = spectral_energy_at_bands(_to2d(ref_mid),  sr, band_edges)
        src_side_b = spectral_energy_at_bands(_to2d(src_side), sr, band_edges)
        ref_side_b = spectral_energy_at_bands(_to2d(ref_side), sr, band_edges)
        curve_mid  = compute_reference_eq_curve(src_mid_b,  ref_mid_b,  centers,
                                                 max_boost_db=max_boost_db, max_cut_db=max_cut_db,
                                                 blend=blend)
        curve_side = compute_reference_eq_curve(src_side_b, ref_side_b, centers,
                                                 max_boost_db=max_boost_db, max_cut_db=max_cut_db,
                                                 blend=blend)
    return curve_mid, curve_side


def apply_ms_matching_fir(audio: np.ndarray, sr: int,
                           curve_mid: list, curve_side: list,
                           eq_q: float = 1.3) -> np.ndarray:
    """Aplica FIR de matching independiente al Mid y al Side, luego decodifica.
    Si el audio es mono, aplica solo la curva Mid."""
    if audio.ndim == 1 or audio.shape[0] == 1:
        taps = build_matching_fir(curve_mid, sr, precision=eq_q)
        mono = audio if audio.ndim == 1 else audio[0]
        return apply_matching_fir(mono[np.newaxis, :], sr, taps)

    src_mid, src_side = encode_ms(audio)
    taps_mid  = build_matching_fir(curve_mid,  sr, precision=eq_q)
    taps_side = build_matching_fir(curve_side, sr, precision=eq_q)
    mid_proc  = apply_matching_fir(src_mid[np.newaxis, :],  sr, taps_mid)[0]
    side_proc = apply_matching_fir(src_side[np.newaxis, :], sr, taps_side)[0]
    return decode_ms(mid_proc, side_proc)

def iterative_eq_matching(audio: np.ndarray, sr: int,
                           ref_audio: np.ndarray, ref_sr: int,
                           band_edges: list, centers: list,
                           passes: int = 3,
                           blend_schedule: list = None,
                           max_boost_db: float = 5.0,
                           max_cut_db: float = -8.0,
                           eq_fit_method: str = "heuristic",
                           eq_q: float = 1.3,
                           ms_mode: bool = True) -> np.ndarray:
    """EQ matching iterativo en N pasadas con blend decreciente.

    En lugar de calcular un único FIR con blend=0.75 y aplicarlo de golpe,
    hace N pasadas donde cada una corrige el residuo que dejó la anterior.
    El blend_schedule controla cuánto se corrige en cada pasada:
      - Pasada 1: blend alto (ataca la mayor parte de la diferencia)
      - Pasada 2: blend medio (corrige lo que quedó sin overcorrect)
      - Pasada 3: blend bajo (pule el residuo sin colorear)

    Resultado: curva efectiva más suave, menos ringing, sin overcorrection
    en zonas donde la referencia es extrema (sub, presencia, aire).
    El audio resultante suena más natural que un match de una sola pasada al 75%.

    passes: cantidad de iteraciones (2-4 es el rango útil; más de 4 no mejora)
    blend_schedule: lista de floats, uno por pasada. Por defecto [0.65, 0.45, 0.25]
    ms_mode: si True y el audio es stereo, aplica M/S independiente por pasada.
    """
    if blend_schedule is None:
        # Decreciente: primera pasada agresiva, últimas de pulido
        if passes == 2:
            blend_schedule = [0.60, 0.35]
        elif passes == 3:
            blend_schedule = [0.60, 0.38, 0.20]
        else:
            blend_schedule = [0.55, 0.35, 0.22, 0.12] + [0.10] * max(0, passes - 4)
    blend_schedule = blend_schedule[:passes]

    result = audio.copy()

    for i, blend in enumerate(blend_schedule):
        # Medir el residuo espectral actual vs referencia
        src_bands = spectral_energy_at_bands(result, sr, band_edges)
        ref_bands = spectral_energy_at_bands(ref_audio, ref_sr, band_edges)

        if str(eq_fit_method).lower() == "ddsp":
            src_mr = spectral_energy_at_bands_multires(result, sr, band_edges)
            ref_mr = spectral_energy_at_bands_multires(ref_audio, ref_sr, band_edges)
            curve = compute_reference_eq_curve_ddsp(src_mr, ref_mr, centers,
                                                     max_boost_db=max_boost_db,
                                                     max_cut_db=max_cut_db)
        else:
            curve = compute_reference_eq_curve(src_bands, ref_bands, centers,
                                               max_boost_db=max_boost_db,
                                               max_cut_db=max_cut_db,
                                               blend=blend)

        # Aplicar pasada
        if ms_mode and result.ndim == 2 and result.shape[0] == 2:
            src_mr_ms = spectral_energy_at_bands_multires(result, sr, band_edges) \
                        if eq_fit_method == "ddsp" else None
            ref_mr_ms = spectral_energy_at_bands_multires(ref_audio, ref_sr, band_edges) \
                        if eq_fit_method == "ddsp" else None
            curve_mid, curve_side = compute_ms_eq_curves(
                result, sr, ref_audio, ref_sr,
                band_edges=band_edges, centers=centers,
                max_boost_db=max_boost_db, max_cut_db=max_cut_db,
                blend=blend, eq_fit_method=eq_fit_method,
                src_bands_multires=src_mr_ms, ref_bands_multires=ref_mr_ms,
            )
            result = apply_ms_matching_fir(result, sr, curve_mid, curve_side, eq_q=eq_q)
        else:
            taps = build_matching_fir(curve, sr, precision=eq_q)
            result = apply_matching_fir(result, sr, taps)

    return result


def match_crest_factor(audio: np.ndarray, sr: int,
                        ref_audio: np.ndarray, ref_sr: int,
                        amount: float = 0.75,
                        band_mode: bool = True,
                        low_crossover: float = 200.0,
                        high_crossover: float = 4000.0) -> np.ndarray:
    """Iguala el crest factor (peak/RMS) del audio al de la referencia.

    El crest factor determina la dinámica percibida: un master con más crest
    suena más vivo/dinámico; uno con menos suena más denso/comprimido.
    LANDR y los plugins de match mastering solo igualan el espectro y el LUFS
    — el crest queda al azar. Igualarlo hace que la dinámica percibida del
    resultado se parezca a la referencia, no solo el tono y el volumen.

    Implementación: en lugar de un compresor genérico, se calcula el ratio
    entre el crest factor del src y el de la ref y se aplica una ganancia
    dependiente del nivel (soft-knee) que comprime o expande suavemente.
    Con band_mode=True se hace por bandas (bajos/medios/agudos) para no
    aplastar el kick en nombre de igualar el crest de los agudos.

    amount: 0.0 = sin cambio, 1.0 = match completo. 0.75 por defecto
            para dejar algo de carácter del material original.
    """
    def _crest_db(sig: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2)) + 1e-12)
        peak = float(np.max(np.abs(sig)) + 1e-12)
        return 20.0 * np.log10(peak / rms)

    def _apply_crest_gain(src: np.ndarray, src_cf: float, ref_cf: float,
                           amount: float) -> np.ndarray:
        """Ajusta el crest factor de src hacia ref_cf usando ganancia dependiente
        del nivel (compresión/expansión suave sobre la envolvente RMS)."""
        delta_cf = (ref_cf - src_cf) * float(np.clip(amount, 0.0, 1.0))
        if abs(delta_cf) < 0.1:
            return src  # diferencia despreciable

        # ratio de compresión implícito: delta negativo = comprimir (menos crest)
        # delta positivo = expandir (más crest)
        # Se implementa como ganancia dependiente del nivel con ventana de 10ms
        frame = max(1, int(sr * 0.010))
        src64 = src.astype(np.float64)
        mono = src64.mean(axis=0) if src64.ndim == 2 else src64

        # Envolvente RMS por frames
        n = len(mono)
        env = np.zeros(n, dtype=np.float64)
        for start in range(0, n, frame):
            end = min(start + frame, n)
            rms_frame = float(np.sqrt(np.mean(mono[start:end] ** 2)) + 1e-12)
            env[start:end] = rms_frame

        # Ganancia: señal fuerte → comprimir/expandir más; señal débil → casi nada
        # Normalizar env entre 0 y 1 para calcular la ganancia modulada
        env_norm = env / (np.max(env) + 1e-12)
        # Ajuste: delta_cf negativo comprime picos (reduce ganancia donde env es alto)
        #          delta_cf positivo expande picos
        gain_mod = 1.0 + (10.0 ** (delta_cf / 20.0) - 1.0) * env_norm
        gain_mod = np.clip(gain_mod, 0.25, 4.0)

        if src64.ndim == 2:
            return (src64 * gain_mod[np.newaxis, :]).astype(np.float32)
        return (src64 * gain_mod).astype(np.float32)

    amount = float(np.clip(amount, 0.0, 1.0))

    if not band_mode:
        mono_src = audio.mean(axis=0) if audio.ndim == 2 else audio
        mono_ref = ref_audio.mean(axis=0) if ref_audio.ndim == 2 else ref_audio
        # Resamplear ref si tiene diferente sr
        if ref_sr != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(ref_sr), int(sr))
            mono_ref = resample_poly(mono_ref, sr // g, ref_sr // g).astype(np.float32)
        src_cf = _crest_db(mono_src)
        ref_cf = _crest_db(mono_ref)
        return _apply_crest_gain(audio, src_cf, ref_cf, amount)

    # Modo multibanda: calcular y aplicar por separado en cada rango
    # para no aplastar el kick intentando igualar el crest de los agudos
    bands_src = _split_three_bands(audio, sr, low_crossover, high_crossover)

    # Resamplear referencia al sr del source si es necesario
    if ref_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(ref_sr), int(sr))
        ref_resampled = np.stack([
            resample_poly(ref_audio[c], sr // g, ref_sr // g).astype(np.float32)
            for c in range(ref_audio.shape[0])
        ]) if ref_audio.ndim == 2 else resample_poly(
            ref_audio, sr // g, ref_sr // g).astype(np.float32)
    else:
        ref_resampled = ref_audio

    bands_ref = _split_three_bands(ref_resampled, sr, low_crossover, high_crossover)

    result_bands = {}
    for band_name in ("low", "mid", "high"):
        src_band = bands_src[band_name]
        ref_band = bands_ref[band_name]
        mono_s = src_band.mean(axis=0) if src_band.ndim == 2 else src_band
        mono_r = ref_band.mean(axis=0) if ref_band.ndim == 2 else ref_band
        src_cf = _crest_db(mono_s)
        ref_cf = _crest_db(mono_r)
        result_bands[band_name] = _apply_crest_gain(src_band, src_cf, ref_cf, amount)

    return (result_bands["low"] + result_bands["mid"] + result_bands["high"]).astype(np.float32)


# ─── Saturación multibanda ────────────────────────────────────────────────────

def multiband_saturation(audio: np.ndarray, sr: int,
                          low_drive: float = 0.08,
                          mid_drive: float = 0.05,
                          high_drive: float = 0.03,
                          low_crossover: float = 200.0,
                          high_crossover: float = 3500.0,
                          mode: str = "tape",
                          mix: float = 0.5,
                          oversample: int = 2) -> tuple:
    """Saturación armónica independiente por banda de frecuencia.

    Por qué multibanda:
    - Saturar bajos/medios/agudos con el mismo drive produce resultados
      inconsistentes: los graves necesitan más drive para calentar sin 
      hacerse ásperos, los agudos necesitan menos para no volverse estridentes.
    - Separar en 3 bandas permite calentar los bajos con 2do armónico
      (modo 'analog'), dar cuerpo a los medios ('tape') y agregar aire
      suave a los agudos con drive muy bajo — exactamente lo que hace
      un chain de transformadores analógicos en una consola real.

    low_drive:  0.05-0.15 — calidez en sub/bajos (< 200Hz)
    mid_drive:  0.03-0.10 — cuerpo en medios (200Hz-3.5kHz)
    high_drive: 0.01-0.05 — aire/brillo en agudos (> 3.5kHz)
    mix: wet/dry global — 0.5 es el sweet spot para transparencia

    Devuelve (audio_procesado, meta_dict).
    """
    if mix <= 0.0:
        return audio, {"applied": False}
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
        was_mono = True
    else:
        was_mono = False

    bands = _split_three_bands(audio, sr, low_crossover, high_crossover)

    def _sat_band(band_audio, drive, band_mode=mode):
        if drive <= 0.0:
            return band_audio
        return harmonic_saturation(band_audio, drive=drive, mode=band_mode,
                                   mix=mix, oversample=oversample)

    # Bajo: modo analog (2do armónico, calidez) — más drive
    low_sat  = _sat_band(bands["low"],  low_drive,  "analog")
    # Medios: modo tape (3er armónico, cuerpo) — drive medio
    mid_sat  = _sat_band(bands["mid"],  mid_drive,  "tape")
    # Agudos: modo tape suave — drive mínimo para no hacer áspero
    high_sat = _sat_band(bands["high"], high_drive, "tape")

    out = low_sat + mid_sat + high_sat

    # Normalizar peak para que la saturación no suba el nivel
    src_peak = float(np.max(np.abs(audio)) + 1e-12)
    out_peak = float(np.max(np.abs(out))   + 1e-12)
    if out_peak > src_peak:
        out = out * (src_peak / out_peak)

    if was_mono:
        out = out.mean(axis=0)

    meta = {
        "applied": True,
        "low_drive": round(low_drive, 3),
        "mid_drive": round(mid_drive, 3),
        "high_drive": round(high_drive, 3),
        "mix": round(mix, 2),
        "mode": mode,
    }
    return out, meta


# ─── Parallel compression ─────────────────────────────────────────────────────

def parallel_compress(audio: np.ndarray, sr: int,
                       threshold_db: float = -20.0,
                       ratio: float = 4.0,
                       attack_ms: float = 10.0,
                       release_ms: float = 150.0,
                       makeup_db: float = 6.0,
                       mix: float = 0.30,
                       oversample: int = DEFAULT_DSP_OVERSAMPLE) -> tuple:
    """Compresión paralela (New York compression).

    Mezcla la señal original sin procesar con una versión muy comprimida
    (ratio alto, threshold bajo) en la proporción `mix`. El resultado:
    - Los transientes del original se preservan completamente (el dry nunca
      se toca — no hay atenuación de picos, no hay pumping).
    - El sustain y el cuerpo de la señal suben en el mix por el wet
      comprimido y maquillado.
    - Da esa sensación de 'pegamento' y densidad sin matar el aire ni
      aplastar los transientes — imposible con compresión en serie.

    threshold_db: a dónde entra el compresor del canal wet. -20dB es
                  agresivo y comprime casi todo el material.
    ratio: 4:1-8:1 para NY compression típica
    makeup_db: compensación de ganancia en el wet — sube la señal comprimida
               para que aporte cuerpo al mix.
    mix: cuánto wet entra. 0.25-0.40 es el rango clásico. Más alto = más
         denso pero puede sonar comprimido.

    Devuelve (audio_mezclado, meta_dict).
    """
    if mix <= 0.0:
        return audio, {"applied": False, "mix": 0.0}

    threshold_lin = float(np.clip(10.0 ** (threshold_db / 20.0), 0.001, 0.99))
    makeup_lin    = float(10.0 ** (makeup_db / 20.0))

    wet, comp_meter = compressor(
        audio, sr,
        threshold=threshold_lin,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        makeup_db=makeup_db,
        oversample=oversample,
    )

    # BUGFIX distorsión: normalizar wet ANTES del blend — si el wet clipea
    # internamente y después lo mezclamos, el daño ya está hecho aunque el
    # pico final parezca correcto. Normalizar antes garantiza que el wet
    # nunca excede el nivel del dry antes de mezclarlos.
    src_peak = float(np.max(np.abs(audio)) + 1e-12)
    wet_peak = float(np.max(np.abs(wet)) + 1e-12)
    if wet_peak > src_peak:
        wet = wet * (src_peak / wet_peak)

    blended = (1.0 - mix) * audio + mix * wet

    # Segunda guarda por si la suma creció
    out_peak = float(np.max(np.abs(blended)) + 1e-12)
    if out_peak > src_peak * 1.005:
        blended = blended * (src_peak / out_peak)

    meta = {
        "applied": True,
        "threshold_db": round(threshold_db, 1),
        "ratio": round(ratio, 2),
        "attack_ms": round(attack_ms, 1),
        "release_ms": round(release_ms, 1),
        "makeup_db": round(makeup_db, 1),
        "mix": round(mix, 2),
        "gr_db": comp_meter.get("gr_db", 0.0),
        "bypass": False,
    }
    return blended, meta


# ─── Limitador en dos etapas ──────────────────────────────────────────────────

def two_stage_limiter(audio: np.ndarray, sr: int,
                       gentle_ceiling_db: float = -3.0,
                       gentle_release_ms: float = 120.0,
                       brickwall_ceiling_db: float = -0.3,
                       brickwall_release_ms: float = 60.0,
                       lookahead_ms: float = 5.0,
                       oversample: int = DEFAULT_DSP_OVERSAMPLE) -> tuple:
    """Limitador en dos etapas: gentle + brickwall final.

    Por qué dos etapas:
    Un solo limitador brickwall con threshold bajo trabaja duro en cada
    transiente — el resultado es pumping y pérdida de punch porque tiene
    que recortar muchos dBs en cada golpe. Dos etapas distribuyen el trabajo:

    Etapa 1 — 'Gentle limiter':
    - Ceiling más alto (ej. -3dBTP) — solo toca los picos más extremos.
    - Release más lento — no bombea, suaviza los picos musicalmente.
    - Actúa como un 'peak manager' que prepara la señal para el brickwall.

    Etapa 2 — 'Brickwall final':
    - Ceiling al target real (ej. -0.3dBTP o -1dBTP).
    - Release más rápido — tiene menos trabajo que hacer porque el gentle
      ya aplanó los picos extremos.
    - Garantía de true peak compliance.

    El resultado suena más transparente que un solo limitador trabajando
    el doble — el punch de los transientes se preserva mejor porque ninguna
    etapa tiene que cortar más de 3-4dB de golpe.

    Devuelve (audio_limitado, meta_dict).
    """
    # BUGFIX: si gentle_ceiling_db queda por debajo (más restrictivo) que
    # brickwall_ceiling_db — que es EXACTAMENTE lo que pasaba con el default
    # fijo de -2.5dB combinado con un brickwall target hot (-1.0/-0.1dB,
    # típico de referencia comercial) — la etapa gentle ya aplasta todo antes
    # de que el brickwall llegue a hacer nada (gr_db=0.0 en la etapa 2), y el
    # master queda pegado al techo de la gentle en vez de al target real de
    # la referencia (hasta ~2-3dB más bajo de lo pedido). El gentle SIEMPRE
    # tiene que tener más margen que el brickwall (un valor de dB MAYOR/menos
    # negativo que el brickwall + un colchón mínimo), nunca al revés.
    if gentle_ceiling_db < brickwall_ceiling_db + 1.0:
        gentle_ceiling_db = brickwall_ceiling_db + 2.0

    gentle_ceiling  = float(np.clip(10.0 ** (gentle_ceiling_db  / 20.0), 0.1, 0.99))
    brickwall_ceiling = float(np.clip(10.0 ** (brickwall_ceiling_db / 20.0), 0.1, 0.99))

    # Etapa 1: gentle
    audio_gentle = limiter(
        audio, sr,
        ceiling=gentle_ceiling,
        release_ms=gentle_release_ms,
        lookahead_ms=lookahead_ms,
        oversample=oversample,
    )

    # Medir GR de la etapa 1
    peak_before = float(np.max(np.abs(audio))        + 1e-12)
    peak_after1 = float(np.max(np.abs(audio_gentle)) + 1e-12)
    gr_gentle_db = float(20.0 * np.log10(peak_after1 / peak_before))

    # Etapa 2: brickwall
    audio_final = limiter(
        audio_gentle, sr,
        ceiling=brickwall_ceiling,
        release_ms=brickwall_release_ms,
        lookahead_ms=lookahead_ms,
        oversample=oversample,
    )

    peak_after2 = float(np.max(np.abs(audio_final)) + 1e-12)
    gr_brickwall_db = float(20.0 * np.log10(peak_after2 / peak_after1))

    meta = {
        "gentle": {
            "ceiling_db": round(gentle_ceiling_db, 1),
            "release_ms": round(gentle_release_ms, 1),
            "gr_db": round(gr_gentle_db, 2),
        },
        "brickwall": {
            "ceiling_db": round(brickwall_ceiling_db, 1),
            "release_ms": round(brickwall_release_ms, 1),
            "gr_db": round(gr_brickwall_db, 2),
        },
        "total_gr_db": round(gr_gentle_db + gr_brickwall_db, 2),
    }
    return audio_final, meta


# ─── Dinámica multibanda por referencia ────────────────────────────────────────
# BUGFIX/MEJORA: la versión anterior de "match_dynamics" comprimía TODA la
# señal con un único compresor de banda ancha, calculando un solo crest
# factor global. Eso podía, por ejemplo, aplastar los graves para igualar un
# crest factor global aunque los graves ya estuvieran igual de comprimidos
# que la referencia (y viceversa con los agudos). Ahora se compara y corrige
# banda por banda (graves/medios/agudos), igual que haría un ingeniero
# revisando cada rango por separado.

# BUGFIX (pérdida de sub-graves y aire en reference mastering): tanto
# match_dynamics_bands como match_stereo_bands partían la señal en 3 bandas
# con _bandpass_filter (un bandpass INDEPENDIENTE por banda) y sumaban las 3
# de vuelta. Un bandpass aislado no tiene nada "del otro lado" en sus bordes
# externos (20Hz / 20000Hz) que compense su propio roll-off — a diferencia de
# multiband_compressor(), que sí arma un split complementario (low-pass +
# high-pass en cascada) que reconstruye una respuesta plana. Medido: la
# versión con bandpass independiente atenuaba hasta -6 dB en 20Hz y hasta
# -6 dB en 20000Hz incluso sin aplicar NINGÚN procesamiento (k=1 / banda ya
# igual a la referencia) — se perdía sub-bass y "aire" en cada render con
# reference mastering. Este helper arma el split correcto una sola vez.
def _split_three_bands(audio: np.ndarray, sr: int,
                       low_crossover: float, high_crossover: float) -> dict:
    """Divide audio (2, N) en low/mid/high con crossovers complementarios
    (low-pass + high-pass en cascada, orden 4, sosfiltfilt) que suman de
    vuelta a una respuesta plana en todo el espectro, igual que
    multiband_compressor(). Devuelve {"low": ..., "mid": ..., "high": ...}."""
    low_crossover  = float(np.clip(low_crossover,  20.0, sr / 2.0 - 1.0))
    high_crossover = float(np.clip(high_crossover, low_crossover + 1.0, sr / 2.0 - 1.0))
    sos_lo_lp = butter(4, low_crossover,  btype='lowpass',  fs=sr, output='sos')
    sos_lo_hp = butter(4, low_crossover,  btype='highpass', fs=sr, output='sos')
    sos_hi_lp = butter(4, high_crossover, btype='lowpass',  fs=sr, output='sos')
    sos_hi_hp = butter(4, high_crossover, btype='highpass', fs=sr, output='sos')
    low      = sosfiltfilt(sos_lo_lp, audio)
    mid_high = sosfiltfilt(sos_lo_hp, audio)
    mid      = sosfiltfilt(sos_hi_lp, mid_high)
    high     = sosfiltfilt(sos_hi_hp, mid_high)
    return {"low": low, "mid": mid, "high": high}

def match_dynamics_bands(audio: np.ndarray, sr: int,
                         own_crest: dict, ref_crest: dict,
                         bands: list = DYNAMICS_BANDS,
                         margin_db: float = 1.0,
                         oversample: int = DEFAULT_DSP_OVERSAMPLE) -> tuple:
    """Comprime banda por banda solo donde el crest factor propio supera al
    de la referencia por más de `margin_db`. Nunca expande dinámica (si la
    referencia es más dinámica que el track en una banda, se deja intacta
    para evitar artefactos de expansores)."""
    if audio.ndim == 1:
        audio2 = np.stack([audio, audio])
    else:
        audio2 = audio

    # bands viene como [("low", lo, mid_x), ("mid", mid_x, high_x), ("high", high_x, hi)]
    # (ver DYNAMICS_BANDS) — se usan los dos crossovers internos para el split
    # complementario en vez de bandpass independiente por banda.
    low_x  = bands[0][2]
    high_x = bands[1][2]
    split = _split_three_bands(audio2, sr, low_x, high_x)

    out = np.zeros_like(audio2)
    meta = {}
    # Más lento y conservador para preservar el aire natural del material vocal
    attack_by_band  = {"low": 35.0, "mid": 20.0, "high": 12.0}
    release_by_band = {"low": 200.0, "mid": 140.0, "high": 90.0}

    for name, lo, hi in bands:
        band_audio = split[name]
        own_c = own_crest.get(name, 0.0)
        ref_c = ref_crest.get(name, 0.0)
        gap = own_c - ref_c
        if gap > margin_db:
            # Ratio max 2.8 (era 4.5) — preserva aire y dinámica natural
            ratio = float(np.clip(1.3 + gap / 7.0, 1.3, 2.8))
            mono_b = band_audio.mean(axis=0)
            b_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(mono_b ** 2)) + 1e-9))
            # Threshold +5dB sobre RMS (era +3) — solo actúa en picos reales
            threshold_db  = b_rms_db + 5.0
            threshold_lin = float(np.clip(10.0 ** (threshold_db / 20.0), 0.02, 0.95))
            band_out, band_meter = compressor(
                band_audio, sr, threshold=threshold_lin, ratio=ratio,
                attack_ms=attack_by_band[name], release_ms=release_by_band[name],
                makeup_db=0.0, oversample=oversample,
            )
            out += band_out
            meta[name] = {"applied": True, "ratio": round(ratio, 2), "gap_db": round(gap, 2),
                         "own_crest_db": round(own_c, 2), "ref_crest_db": round(ref_c, 2), **band_meter}
        else:
            out += band_audio
            meta[name] = {"applied": False, "gap_db": round(gap, 2),
                         "own_crest_db": round(own_c, 2), "ref_crest_db": round(ref_c, 2)}

    if audio.ndim == 1:
        out = out.mean(axis=0)
    return out, meta

def derive_mb_chain_params_from_reference(audio: np.ndarray, sr: int,
                                          ref_audio: np.ndarray, ref_sr: int,
                                          bands: list = DYNAMICS_BANDS,
                                          margin_db: float = 1.0) -> dict:
    """Calcula los kwargs mb_* (threshold/ratio por banda) que apply_mastering_chain
    espera, calibrados contra la referencia con la MISMA fórmula que usa
    match_dynamics_bands (ratio = clip(1.3 + gap/7, 1.3, 2.8), threshold =
    rms_banda + 5dB), pero sin aplicar la compresión directamente — solo
    devuelve los parámetros para que el streaming de preview (`/ws/ref-stream`)
    pueda pasárselos a apply_mastering_chain() en cada chunk, y así el GR real
    que ve el usuario en vivo sea el mismo ajuste banda-por-banda que
    process_audio_with_reference aplica en el render final, en vez del
    compresor multibanda genérico (thresholds fijos, sin relación con la
    referencia) que corría antes en el preview.

    bands debe ser DYNAMICS_BANDS (o compatible) — se fuerzan los crossovers
    mb_low_crossover/mb_high_crossover a los bordes internos de `bands` para
    que el split que hace el multibanda de la cadena coincida exactamente con
    el split usado acá para medir el gap.
    """
    if audio.ndim == 1:
        audio2 = np.stack([audio, audio])
    else:
        audio2 = audio

    low_x  = bands[0][2]
    high_x = bands[1][2]
    split = _split_three_bands(audio2, sr, low_x, high_x)

    own_crest = band_crest_factors(audio, sr, bands=bands)
    ref_crest = band_crest_factors(ref_audio, ref_sr, bands=bands)

    params = {
        "mb_bypass": False,
        "mb_low_crossover": low_x,
        "mb_high_crossover": high_x,
    }
    name_to_prefix = {"low": "low", "mid": "mid", "high": "high"}
    for name, lo, hi in bands:
        prefix = name_to_prefix.get(name, name)
        own_c = own_crest.get(name, 0.0)
        ref_c = ref_crest.get(name, 0.0)
        gap = own_c - ref_c
        if gap > margin_db:
            ratio = float(np.clip(1.3 + gap / 7.0, 1.3, 2.8))
            mono_b = split[name].mean(axis=0)
            b_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(mono_b ** 2)) + 1e-9))
            threshold_db = b_rms_db + 5.0
        else:
            # Sin gap significativo: ratio 1.0 = no comprime (el threshold
            # no importa con ratio 1), banda queda efectivamente intacta.
            ratio = 1.0
            threshold_db = 0.0
        params[f"mb_{prefix}_threshold_db"] = round(threshold_db, 2)
        params[f"mb_{prefix}_ratio"] = round(ratio, 3)
        params[f"mb_{prefix}_attack_ms"]  = {"low": 35.0, "mid": 20.0, "high": 12.0}[name]
        params[f"mb_{prefix}_release_ms"] = {"low": 200.0, "mid": 140.0, "high": 90.0}[name]
        params[f"mb_{prefix}_makeup_db"] = 0.0
    return params


def match_lra(audio: np.ndarray, sr: int, own_lra: float, ref_lra: float,
             margin: float = 1.0,
             oversample: int = DEFAULT_DSP_OVERSAMPLE) -> tuple:
    """Ajusta la macro-dinámica (LRA, rango dinámico "a largo plazo") con un
    compresor tipo 'glue' (attack/release lentos) cuando el track propio es
    notablemente más variable en el tiempo que la referencia. Al igual que
    match_dynamics_bands, solo comprime, nunca expande."""
    meta = {"applied": False, "own_lra": round(own_lra, 2), "ref_lra": round(ref_lra, 2)}
    gap = own_lra - ref_lra
    # Solo actuar si el gap es significativo (>3dB) — acapelas tienen naturalmente
    # más LRA que un master comercial y no deben ser aplastados por eso
    if gap > max(margin, 3.0):
        # Ratio max 1.8 (era 2.5) — compresión de macro-dinámica muy suave
        ratio = float(np.clip(1.15 + gap / 12.0, 1.15, 1.8))
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        rms_db = float(20.0 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9))
        # Threshold más alto: rms+4 (era rms+1) — no toca el cuerpo de la señal
        threshold_lin = float(np.clip(10.0 ** ((rms_db + 4.0) / 20.0), 0.05, 0.95))
        audio, comp_meter = compressor(audio, sr, threshold=threshold_lin, ratio=ratio,
                                       attack_ms=50.0, release_ms=500.0, makeup_db=0.0,
                                       oversample=oversample)
        meta.update({"applied": True, "ratio": round(ratio, 2), **comp_meter})
    return audio, meta

# ─── Estéreo multibanda por referencia ─────────────────────────────────────────
# BUGFIX/MEJORA: la versión anterior calculaba UNA sola correlación L/R para
# toda la señal y aplicaba un único factor de ancho global. Un master real
# casi siempre tiene graves mono/casi-mono y agudos más anchos: promediar
# todo en un solo número perdía esa forma y podía, por ejemplo, ensanchar los
# graves (rompiendo compatibilidad mono/fase) para igualar una correlación
# global dominada por los agudos. Ahora se hace banda por banda usando la
# relación cerrada entre la energía de Mid/Side y el coeficiente de
# correlación:
#   rho = (var(M) - var(S)) / (var(M) + var(S))
# despejando el factor de escala k a aplicar sobre Side para llegar a rho':
#   k = sqrt( var(M)*(1-rho') / (var(S)*(1+rho')) )

def match_stereo_bands(audio: np.ndarray, sr: int, target_corr: dict,
                       bands: list = DYNAMICS_BANDS, blend: float = 0.85,
                       min_k: float = 0.25, max_k: float = 2.5) -> tuple:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return audio, {name: 1.0 for name, _, _ in bands}

    left, right = audio[0], audio[1]
    out_l = np.zeros_like(left)
    out_r = np.zeros_like(right)
    k_applied = {}

    # Split complementario (ver _split_three_bands) en vez de bandpass
    # independiente por banda, para no perder sub-bass/aire en los bordes.
    low_x  = bands[0][2]
    high_x = bands[1][2]
    split_l = _split_three_bands(left,  sr, low_x, high_x)
    split_r = _split_three_bands(right, sr, low_x, high_x)

    for name, lo, hi in bands:
        fl = split_l[name]
        fr = split_r[name]
        mid  = (fl + fr) * 0.5
        side = (fl - fr) * 0.5
        var_m = float(np.var(mid))
        var_s = float(np.var(side))
        # Guarda de estabilidad: si el contenido "side" de esta banda es
        # prácticamente silencio (banda casi perfectamente mono, típico en
        # graves), var_s puede ser ~0. Escalar eso con la fórmula cerrada
        # amplificaría solo ruido de redondeo, no señal real, y podría
        # colorear el espectro de forma audible. En ese caso no se toca.
        if var_s < max(var_m, 1e-12) * 1e-4:
            out_l += mid + side
            out_r += mid - side
            k_applied[name] = 1.0
            continue
        cur_rho = float(np.clip((var_m - var_s) / (var_m + var_s + 1e-12), -1.0, 1.0))
        target_rho = float(np.clip(target_corr.get(name, cur_rho), -1.0, 1.0))
        blended_rho = float(np.clip(cur_rho + (target_rho - cur_rho) * blend, -0.98, 0.98))
        k2 = var_m * (1.0 - blended_rho) / (var_s * (1.0 + blended_rho) + 1e-12)
        k = float(np.clip(np.sqrt(max(k2, 0.0)), min_k, max_k))
        out_l += mid + side * k
        out_r += mid - side * k
        k_applied[name] = round(k, 3)

    return np.stack([out_l, out_r]), k_applied

# ─── Dimensión 5: matching de transientes / punch ─────────────────────────────

def match_transient_punch(audio: np.ndarray, sr: int,
                           own_density: float, ref_density: float,
                           own_crest_db: float, ref_crest_db: float,
                           max_attack: float = 0.28, max_sustain: float = 0.20,
                           density_margin: float = 0.5,
                           crest_margin_db: float = 1.0) -> tuple:
    """Ajusta el punch/transiente del track para acercarlo al perfil de la
    referencia usando `transient_shaper`. Opera en dos dimensiones:

    1. *Densidad de transientes* (onsets/seg): si la ref es más percusiva,
       boost de ataque; si es más suave, reducción suave de sustain.
    2. *Crest factor global*: gap grande entre src y ref → refuerza ataque
       para recuperar pegada sin comprimir el sustain.

    Nunca aplica valores negativos de ataque (no destruye transientes
    para igualar una ref más comprimida — eso ya lo maneja match_dynamics_bands).
    Devuelve (audio_procesado, meta_dict).
    """
    meta = {
        "applied": False,
        "own_density": round(own_density, 3),
        "ref_density": round(ref_density, 3),
        "own_crest_db": round(own_crest_db, 2),
        "ref_crest_db": round(ref_crest_db, 2),
        "attack_amount": 0.0,
        "sustain_amount": 0.0,
    }

    density_gap = ref_density - own_density   # positivo → ref más percusiva
    crest_gap   = ref_crest_db - own_crest_db # positivo → ref más dinámica/punchy

    attack_amount  = 0.0
    sustain_amount = 0.0

    if density_gap > density_margin:
        # Ref más percusiva: boost de ataque proporcional al gap
        attack_amount = float(np.clip(density_gap / 6.0, 0.0, max_attack))
    elif density_gap < -density_margin:
        # Ref más suave: leve reducción de sustain (no toca el ataque)
        sustain_amount = float(np.clip(density_gap / 10.0, -max_sustain, 0.0))

    if crest_gap > crest_margin_db:
        # Ref más punchy por crest: suma boost de ataque adicional
        extra_attack = float(np.clip((crest_gap - crest_margin_db) / 8.0, 0.0, max_attack * 0.5))
        attack_amount = float(np.clip(attack_amount + extra_attack, 0.0, max_attack))

    if abs(attack_amount) < 0.01 and abs(sustain_amount) < 0.01:
        return audio, meta

    audio_out = transient_shaper(audio, sr,
                                 attack_amount=attack_amount,
                                 sustain_amount=sustain_amount)
    meta.update({
        "applied": True,
        "attack_amount": round(attack_amount, 3),
        "sustain_amount": round(sustain_amount, 3),
    })
    return audio_out, meta


# ─── Dimensión 6: matching del perfil de sub-graves (20-80 Hz) ────────────────

def match_sub_bass_profile(audio: np.ndarray, sr: int,
                            own_sub_db: float, ref_sub_db: float,
                            own_bass_db: float, ref_bass_db: float,
                            max_gain_db: float = 3.0,
                            max_cut_db: float = -3.0,
                            margin_db: float = 1.5) -> tuple:
    """Ajusta el balance sub-graves (20-80 Hz) vs graves (80-250 Hz) para
    acercarlo al de la referencia con dos filtros shelving de bajo orden
    (biquad IIR, zero-latency — no hace falta FIR acá por la baja frecuencia).

    La lógica es deliberadamente separada del EQ FIR de matching principal:
    ese opera sobre 28 bandas log-spaced y puede subrepresentar la sub-region
    (<80 Hz) cuando el contenido es esparso o el track es corto. Acá
    miramos explícitamente dos métricas: la energía sub en relación a la banda
    de bass, y si hay demasiado o muy poco sub vs referencia. Nunca boost >
    max_gain_db ni corte > max_cut_db para evitar excitar resonancias.
    """
    meta = {
        "applied": False,
        "own_sub_db": round(own_sub_db, 2),
        "ref_sub_db": round(ref_sub_db, 2),
        "sub_gain_db": 0.0,
        "bass_gain_db": 0.0,
    }

    sub_gap  = ref_sub_db  - own_sub_db    # cuánto sub le falta/sobra al track
    bass_gap = ref_bass_db - own_bass_db

    sub_gain_db  = float(np.clip(sub_gap,  max_cut_db, max_gain_db))
    bass_gain_db = float(np.clip(bass_gap, max_cut_db, max_gain_db))

    if abs(sub_gain_db) < margin_db and abs(bass_gain_db) < margin_db:
        return audio, meta

    out = audio.copy()

    # Sub shelf: 20-80 Hz → shelf centrado en ~50 Hz, pendiente suave (orden 1)
    if abs(sub_gain_db) >= margin_db:
        sos_sub = butter(1, 80.0 / (sr / 2.0), btype="low", output="sos")
        lin_sub  = 10.0 ** (sub_gain_db / 20.0)
        sub_band  = sosfiltfilt(sos_sub, out if out.ndim == 1 else out.mean(axis=0))
        rest      = (out if out.ndim == 1 else out) - (sub_band if out.ndim == 1
                     else np.stack([sosfiltfilt(sos_sub, ch) for ch in out]))
        if out.ndim == 1:
            sub_filtered = sosfiltfilt(sos_sub, out)
            rest = out - sub_filtered
            out  = rest + sub_filtered * lin_sub
        else:
            sub_filtered = np.stack([sosfiltfilt(sos_sub, ch) for ch in out])
            rest = out - sub_filtered
            out  = rest + sub_filtered * lin_sub

    # Bass shelf: 80-250 Hz → shelf centrado en ~150 Hz
    if abs(bass_gain_db) >= margin_db:
        sos_lo = butter(1, 80.0  / (sr / 2.0), btype="low",  output="sos")
        sos_hi = butter(1, 250.0 / (sr / 2.0), btype="low",  output="sos")
        lin_bass = 10.0 ** (bass_gain_db / 20.0)
        if out.ndim == 1:
            lo  = sosfiltfilt(sos_lo, out)
            mid = sosfiltfilt(sos_hi, out) - lo
            out = lo + mid * lin_bass + (out - sosfiltfilt(sos_hi, out))
        else:
            lo  = np.stack([sosfiltfilt(sos_lo, ch) for ch in out])
            hi  = np.stack([sosfiltfilt(sos_hi, ch) for ch in out])
            mid = hi - lo
            out = lo + mid * lin_bass + (out - hi)

    meta.update({
        "applied": True,
        "sub_gain_db":  round(sub_gain_db, 2),
        "bass_gain_db": round(bass_gain_db, 2),
    })
    return out, meta


# ─── Dimensión 7: de-esser calibrado por referencia ──────────────────────────

def match_desser_calibrated(audio: np.ndarray, sr: int,
                             src_sibilance: dict, ref_sibilance: dict,
                             max_reduction_db: float = 10.0) -> tuple:
    """De-esser calibrado comparando la sibilancia del track propio vs la de
    la referencia. Tres casos:

    - Referencia NO tiene sibilancia relevante y src SÍ → de-esser activo,
      reducción = severity del src (clipeada a max_reduction_db).
    - Ambos tienen sibilancia → de-esser proporcional a la *diferencia* de
      severity (ref sirve como "techo aceptable": si la referencia tiene 4 dB
      de severidad y el src tiene 7 dB, se corrigen 3 dB).
    - Referencia también tiene alta sibilancia (≥ src) → no se toca (ese es
      el estilo del género, no un problema a corregir).

    Usa `dynamic_eq_band` internamente (el mismo de-esser de la cadena
    principal) para que la reducción sea dinámica (solo en los picos de 's')
    y no tiña el resto del espectro.
    """
    meta = {
        "applied": False,
        "src_sibilance_present": src_sibilance.get("present", False),
        "ref_sibilance_present": ref_sibilance.get("present", False),
        "src_severity_db": src_sibilance.get("severity_db", 0.0),
        "ref_severity_db": ref_sibilance.get("severity_db", 0.0),
        "reduction_applied_db": 0.0,
        "freq_hz": 0.0,
    }

    src_present  = src_sibilance.get("present", False)
    ref_present  = ref_sibilance.get("present", False)
    src_severity = float(src_sibilance.get("severity_db", 0.0))
    ref_severity = float(ref_sibilance.get("severity_db", 0.0))

    if not src_present:
        return audio, meta  # nada que corregir

    if ref_present and ref_severity >= src_severity - 0.5:
        return audio, meta  # ref igual o más sibilante: no tocar

    reduction_db = float(np.clip(
        src_severity - (ref_severity if ref_present else 0.0),
        0.0, max_reduction_db,
    ))
    if reduction_db < 1.0:
        return audio, meta

    low_hz, high_hz = src_sibilance.get("band_hz", [4000.0, 9000.0])
    low_hz   = float(np.clip(low_hz,  1000.0, sr / 2.0 - 100.0))
    high_hz  = float(np.clip(high_hz, low_hz + 500.0, sr / 2.0 - 50.0))
    center   = float(np.sqrt(low_hz * high_hz))
    bw       = max(high_hz - low_hz, 1.0)
    q        = float(np.clip(center / bw, 0.5, 8.0))

    # Reusar dynamic_eq_band (ya definida en la cadena principal)
    threshold_db = float(np.clip(-18.0 - (src_severity - 4.0) * 0.5, -30.0, -10.0))
    audio_out, _band_meter = dynamic_eq_band(
        audio, sr,
        freq=center, q=q,
        threshold_db=threshold_db,
        ratio=float(np.clip(2.0 + reduction_db / 3.0, 2.0, 6.0)),
        attack_ms=3.0, release_ms=80.0,
        max_reduction_db=reduction_db,
        bypass=False,
    )

    meta.update({
        "applied": True,
        "reduction_applied_db": round(reduction_db, 2),
        "freq_hz": round(center, 1),
    })
    return audio_out, meta


# ─── Reporte / análisis inteligente del matching ───────────────────────────────

def reference_intelligent_report(tonal_after: dict, loudness_gain_db: float,
                                  dynamics_band_meta: dict, lra_meta: dict,
                                  stereo_k_applied: dict,
                                  transient_meta: dict = None,
                                  sub_meta: dict = None,
                                  desser_meta: dict = None,
                                  saturation_meta: dict = None) -> dict:
    """Combina el resultado de las 8 dimensiones del matching (tonal,
    loudness, dinámica por banda + LRA, estéreo por banda, transientes/punch,
    perfil de sub-graves, de-esser calibrado y carácter armónico/saturación)
    en un puntaje único y una lista de observaciones/consejos en lenguaje
    natural."""
    issues, tips = [], []
    score = 100.0

    tonal_pct = tonal_after.get("match_percent", 100.0)
    if tonal_pct < 60:
        issues.append(f"El match tonal final quedó en {tonal_pct}%: el balance de frecuencias todavía difiere bastante de la referencia.")
        tips.append("Probá subir eq_max_boost_db/eq_max_cut_db, o elegí una referencia de un género/instrumentación más parecida a tu track.")
        score -= 22
    elif tonal_pct < 80:
        tips.append(f"Match tonal de {tonal_pct}%: aceptable, pero todavía queda diferencia de timbre con la referencia.")
        score -= 10

    for name in ("low", "mid", "high"):
        meta = dynamics_band_meta.get(name, {})
        gap = meta.get("gap_db", 0.0)
        if meta.get("applied"):
            tips.append(f"Banda {name}: comprimida (~{gap} dB más dinámica que la referencia) para acercar la pegada de esa zona.")
            score -= min(8.0, abs(gap) * 0.6)
        elif gap < -3.0:
            tips.append(f"Banda {name}: ya es más densa/comprimida que la referencia; no se tocó para no perder dinámica de más.")

    if lra_meta.get("applied"):
        tips.append(f"Macro-dinámica (LRA) reducida de {lra_meta['own_lra']} a un valor más cercano a la referencia ({lra_meta['ref_lra']} LU) con compresión 'glue' suave.")
        score -= 6
    else:
        gap = lra_meta.get("own_lra", 0.0) - lra_meta.get("ref_lra", 0.0)
        if gap < -1.0:
            tips.append("Tu rango dinámico global (LRA) ya es menor que el de la referencia; no se aplicó compresión adicional de macro-dinámica.")

    for name, k in stereo_k_applied.items():
        if k > 1.08:
            tips.append(f"Estéreo banda {name}: ensanchado x{k} para acercarlo a la referencia.")
        elif k < 0.92:
            tips.append(f"Estéreo banda {name}: angostado/centrado x{k} para acercarlo a la referencia (evita problemas de fase/mono).")

    if abs(loudness_gain_db) > 0.3:
        tips.append(f"Loudness ajustado {loudness_gain_db:+.2f} dB para igualar el LUFS integrado de la referencia.")

    # ── Dim 5: transientes / punch ────────────────────────────────────────────
    if transient_meta and transient_meta.get("applied"):
        atk = transient_meta["attack_amount"]
        sus = transient_meta["sustain_amount"]
        d_own = transient_meta["own_density"]
        d_ref = transient_meta["ref_density"]
        if atk > 0.01:
            tips.append(
                f"Punch/transientes: ref más percusiva ({d_ref:.1f} vs {d_own:.1f} onsets/s) — "
                f"se reforzó el ataque (amount={atk:.2f}) para acercar la pegada."
            )
            score -= min(6.0, atk * 15.0)
        elif sus < -0.01:
            tips.append(
                f"Transientes: ref más suave ({d_ref:.1f} vs {d_own:.1f} onsets/s) — "
                f"se redujo levemente el sustain (amount={sus:.2f}) para igualar densidad percusiva."
            )
    elif transient_meta and not transient_meta.get("applied"):
        d_own = transient_meta["own_density"]
        d_ref = transient_meta["ref_density"]
        if abs(d_own - d_ref) < 0.5:
            tips.append(f"Punch/transientes: muy similar a la referencia ({d_own:.1f} vs {d_ref:.1f} onsets/s). No se modificó.")

    # ── Dim 6: perfil de sub-graves ───────────────────────────────────────────
    if sub_meta and sub_meta.get("applied"):
        sg = sub_meta["sub_gain_db"]
        bg = sub_meta["bass_gain_db"]
        parts_sub = []
        if abs(sg) >= 1.0:
            dir_s = "boost" if sg > 0 else "corte"
            parts_sub.append(f"sub-graves {dir_s} {sg:+.1f} dB")
        if abs(bg) >= 1.0:
            dir_b = "boost" if bg > 0 else "corte"
            parts_sub.append(f"graves {dir_b} {bg:+.1f} dB")
        if parts_sub:
            tips.append(f"Perfil de sub-graves ajustado ({', '.join(parts_sub)}) para igualar el peso de baja frecuencia de la referencia.")
            score -= min(5.0, (abs(sg) + abs(bg)) * 0.5)
    elif sub_meta and not sub_meta.get("applied"):
        tips.append("Sub-graves: perfil muy alineado con la referencia. No se aplicó corrección adicional.")

    # ── Dim 7: de-esser calibrado ─────────────────────────────────────────────
    if desser_meta and desser_meta.get("applied"):
        red = desser_meta["reduction_applied_db"]
        frq = desser_meta["freq_hz"]
        src_sev = desser_meta["src_severity_db"]
        ref_sev = desser_meta["ref_severity_db"]
        tips.append(
            f"De-esser calibrado: sibilancia del track ({src_sev:.1f} dB de severidad) "
            f"supera a la de la referencia ({ref_sev:.1f} dB) — "
            f"reducción dinámica de {red:.1f} dB centrada en {frq:.0f} Hz aplicada."
        )
        score -= min(5.0, red * 0.4)
    elif desser_meta and not desser_meta.get("applied"):
        if desser_meta.get("src_sibilance_present") and desser_meta.get("ref_sibilance_present"):
            tips.append("Sibilancia: nivel similar al de la referencia — no se aplicó de-esser adicional.")
        elif not desser_meta.get("src_sibilance_present"):
            tips.append("Sibilancia: no detectada en el track. De-esser no necesario.")

    # ── Dim 8: carácter armónico / saturación ─────────────────────────────────
    if saturation_meta and saturation_meta.get("applied"):
        smode = saturation_meta["suggested_mode"]
        sdrive = saturation_meta["suggested_drive"]
        tips.append(
            f"Carácter armónico: la referencia tiene más saturación/calidez que el track — "
            f"se aplicó saturación '{smode}' (drive {sdrive:.2f}) para acercar el timbre."
        )
        score -= min(4.0, sdrive * 6.0)
    elif saturation_meta and not saturation_meta.get("applied"):
        tips.append("Carácter armónico: el track ya iguala o supera la saturación/calidez de la referencia. No se agregó saturación.")

    score = float(np.clip(score, 0.0, 100.0))
    grade = ("Excelente" if score >= 88 else "Buena" if score >= 72 else
             "Aceptable" if score >= 50 else "Necesita ajuste manual")
    if not tips:
        tips.append("El match con la referencia es muy sólido en las 8 dimensiones analizadas (tonal, loudness, dinámica, estéreo, punch, sub-graves, de-esser y carácter armónico).")

    return {"overall_score": round(score, 1), "grade": grade, "issues": issues, "tips": tips}

def normalize_by_lufs(input_path: str,
                      target_lufs: float = -14.0,
                      output_format: str = "wav",
                      output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
                      dither_mode: str = "f_weighted",
                      true_peak_safety_db: float = -1.0,
                      progress_cb=None) -> dict:
    """Normalización PURA por LUFS. Sin EQ, sin dinámica, sin referencia, sin
    limitador, sin ninguna otra etapa — el pedido explícito era "usuario sube
    track, botón de normalización por LUFS, y nada más". Es una sola
    multiplicación de ganancia.

    target_lufs: LUFS de destino (rango admisible sugerido en la capa de API:
                 -23 a -6, típicos: -14 streaming, -16 Spotify quiet, -23
                 broadcast EBU R128, -9/-8 club/hot).

    Seguridad SIN agregar una etapa de procesamiento: en vez de aplicar el
    gain de LUFS y después limitar (lo cual sería una etapa extra), se
    calcula el gain máximo que el pico actual banca antes de pasarse de
    `true_peak_safety_db`, y se usa el MENOR de los dos gains. Si el target
    de LUFS pedido no entra sin clippear, el resultado queda más cerca del
    techo de pico seguro que del LUFS pedido (nunca clipea, nunca agrega una
    etapa de dinámica) — y se reporta explícitamente en el resultado para
    que quede claro que pasó.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")

    _report(progress_cb, 10, "Cargando audio")
    audio, sr = _load_audio_any(input_path)

    _report(progress_cb, 35, "Midiendo LUFS")
    input_lufs = measure_lufs_integrated(audio, sr)
    input_peak = float(np.max(np.abs(audio)) + 1e-12)
    input_peak_db = 20.0 * np.log10(input_peak)

    gain_for_lufs_db = float(target_lufs) - input_lufs
    safety_ceiling_lin = float(np.clip(10.0 ** (true_peak_safety_db / 20.0), 0.1, 0.999))
    gain_max_safe_db = 20.0 * np.log10(safety_ceiling_lin / input_peak)

    gain_db = min(gain_for_lufs_db, gain_max_safe_db)
    peak_limited = bool(gain_for_lufs_db > gain_max_safe_db + 0.01)

    _report(progress_cb, 55, "Aplicando ganancia")
    audio_out_arr = audio * (10.0 ** (gain_db / 20.0))
    output_lufs = measure_lufs_integrated(audio_out_arr, sr)
    output_peak = float(np.max(np.abs(audio_out_arr)) + 1e-12)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"processed/{base}_{uuid.uuid4().hex[:8]}_lufsnorm.{output_format}"

    audio_out = audio_out_arr[0] if audio_out_arr.shape[0] == 1 else audio_out_arr.T

    _report(progress_cb, 85, "Guardando archivo normalizado")
    effective_bit_depth, dither_meta = _write_master_output(
        audio_out, sr, output_path, output_format, output_bit_depth,
        dither_mode=dither_mode,
    )

    _report(progress_cb, 100, "Normalización completada")
    return {
        "output_path":      output_path,
        "output_bit_depth": effective_bit_depth,
        "dither":           dither_meta,
        "normalization": {
            "input_lufs":         round(float(input_lufs), 2),
            "input_peak_db":      round(float(input_peak_db), 2),
            "target_lufs":        round(float(target_lufs), 2),
            "output_lufs":        round(float(output_lufs), 2),
            "output_peak_db":     round(20.0 * np.log10(output_peak), 2),
            "gain_applied_db":    round(gain_db, 2),
            "true_peak_safety_db": round(true_peak_safety_db, 2),
            "peak_limited":       peak_limited,  # True = no se llegó al target_lufs pedido para no clipear
        },
    }


def process_audio_with_reference(
    input_path: str,
    reference_path: str,
    eq_bands: int = 28,
    eq_max_boost_db: float = 4.0,  # era 6.0 — más conservador
    eq_max_cut_db: float = -6.0,  # era -9.0 — más conservador
    eq_q: float = 1.3,
    eq_match_blend: float = 0.55,  # era 0.75 — deja más carácter original
    eq_fit_method: str = "heuristic",
    oversample_mode: str = "quality",
    match_loudness: bool = True,
    match_dynamics: bool = True,
    match_stereo_width: bool = True,
    hp_cutoff: float = 30.0,
    limiter_release_ms: float = 60.0,
    output_format: str = "wav",
    output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
    dither_mode: str = "f_weighted",
    preview_seconds: float = None,
    dynamics_margin_db: float = 2.0,  # era 1.0 — margen más conservador
    stereo_blend: float = 0.65,  # era 0.85 — más suave
    match_transient: bool = True,
    match_sub_bass: bool = True,
    match_desser: bool = True,
    match_saturation: bool = False,  # default False — decisión artística del usuario
    ms_eq_matching: bool = True,
    adaptive_loudness_weighting: bool = True,
    loudness_sensitivity_amount: float = 0.65,
    premium_match_profile: str = "balanced",
    premium_vocal_protect: bool = True,
    premium_translation_check: bool = True,
    premium_alt_versions: bool = False,
    iterative_eq_passes: int = 2,
    match_crest: bool = True,
    crest_amount: float = 0.35,  # era 0.55 — muy agresivo para full tracks
    match_spectral_dynamics: bool = False,  # OFF por default — demasiado en full tracks
    spectral_dynamics_amount: float = 0.45,
    spectral_dynamics_bins: int = 4,
    # ── Saturación multibanda ──────────────────────────────────────────────
    use_multiband_saturation: bool = False,  # OFF por default — feature adicional
    mb_sat_low_drive: float = 0.05,
    mb_sat_mid_drive: float = 0.03,
    mb_sat_high_drive: float = 0.015,
    mb_sat_mix: float = 0.25,
    mb_sat_mode: str = "tape",
    # ── Parallel compression ────────────────────────────────────────────────
    use_parallel_compression: bool = False,  # OFF por default — feature adicional
    parallel_threshold_db: float = -20.0,
    parallel_ratio: float = 4.0,
    parallel_attack_ms: float = 10.0,
    parallel_release_ms: float = 150.0,
    parallel_makeup_db: float = 3.0,
    parallel_mix: float = 0.18,
    # ── Limitador en dos etapas ─────────────────────────────────────────────
    use_two_stage_limiter: bool = True,
    gentle_ceiling_db: float = -2.5,
    gentle_release_ms: float = 120.0,
    # ── Techo de loudness (adaptativo, elegible por el usuario) ────────────
    # Rango admisible sugerido en la capa de API: -18 a -6 LUFS. El match
    # siempre intenta llegar al LUFS de la referencia, pero nunca más allá
    # de este techo — así el usuario controla cuánto "hot" quiere el
    # resultado sin importar qué tan comprimida venga la referencia.
    max_target_lufs: float = -12.0,
    loudness_target_lufs: float = None,   # si se especifica (ej. -14.0), ignora el
                                           # loudness de la referencia y apunta SIEMPRE
                                           # a este valor fijo, en vez de igualar el LUFS
                                           # de la referencia (que puede venir muy hot,
                                           # -8/-7/-9, si la referencia es un master
                                           # comercial muy comprimido).
    progress_cb=None,
    band_gains_db: list = None,
) -> dict:
    """Masteriza `input_path` adaptando su sonido al de `reference_path` en
    8 dimensiones: balance tonal (EQ de matching FIR), loudness (LUFS),
    dinámica (compresión multibanda + LRA), ancho estéreo (matching de
    correlación L/R banda por banda), punch/transientes (transient shaper
    calibrado por densidad de onsets y crest factor), perfil de sub-graves
    (shelving dedicado 20-250 Hz), de-esser calibrado (sibilancia src vs ref)
    y carácter armónico/saturación (asimetría de onda + densidad espectral
    en agudos vs. referencia).
    Incluye un reporte de análisis inteligente (`reference_intelligent_report`)
    que resume qué se ajustó en las 8 dimensiones."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Track de referencia no encontrado: {reference_path}")

    _report(progress_cb, 3, "Cargando audio propio y de referencia")
    audio, sr = _load_audio_any(input_path)
    ref_audio, ref_sr = _load_audio_any(reference_path)

    if preview_seconds is not None:
        audio = _crop_preview(audio, sr, preview_seconds)

    ovs = resolve_oversample(oversample_mode)

    _report(progress_cb, 8, "Analizando audio propio y de referencia")
    # SERIE: la cadena de audio se procesa stage-por-stage (sin multitarea).
    # analyze_audio sobre el src primero, después sobre la ref.
    analysis_before    = analyze_audio(audio, sr)
    analysis_reference = analyze_audio(ref_audio, ref_sr)

    # BUGFIX Bug 5: analyze_harmonic_character debe correr sobre el audio CRUDO
    # antes de cualquier procesamiento de la cadena, y sobre una ventana
    # representativa del cuerpo del track (no el preview de 10s que puede
    # ser solo el intro).
    # Para la referencia: siempre usamos el full track (ref_audio sin crop),
    # ya que la referencia se carga completa y es la que define el "target".
    # Para el track propio: si hay preview_seconds, el audio ya está cropped
    # al intro — en su lugar analizamos un fragmento del centro del audio
    # original (antes del crop) para capturar el cuerpo del track.
    # La referencia de este análisis temprano se guarda en _own_char_raw y
    # _ref_char_raw para pasarla a match_saturation_character más adelante
    # en vez de recalcularla sobre el audio ya procesado.
    _analysis_window_sec = 30.0  # ventana representativa para harmonic analysis
    _own_audio_for_char = audio  # default: el crop actual si no hay mejor opción
    if preview_seconds is not None:
        # Intentar cargar una ventana del centro del track original para análisis
        try:
            _full_audio, _full_sr = _load_audio_any(input_path)
            _total_sec = _full_audio.shape[-1] / _full_sr
            _start = max(0.0, _total_sec / 2.0 - _analysis_window_sec / 2.0)
            _start_s = int(_start * _full_sr)
            _end_s = min(_full_audio.shape[-1], _start_s + int(_analysis_window_sec * _full_sr))
            _own_audio_for_char = _full_audio[:, _start_s:_end_s] if _full_audio.ndim == 2 else _full_audio[_start_s:_end_s]
            del _full_audio
        except Exception:
            _own_audio_for_char = audio  # fallback al crop
    # Para la ref siempre el centro del full track (ya cargado completo)
    _ref_total_sec = ref_audio.shape[-1] / ref_sr
    _ref_start = max(0.0, _ref_total_sec / 2.0 - _analysis_window_sec / 2.0)
    _ref_start_s = int(_ref_start * ref_sr)
    _ref_end_s = min(ref_audio.shape[-1], _ref_start_s + int(_analysis_window_sec * ref_sr))
    _ref_audio_for_char = ref_audio[:, _ref_start_s:_ref_end_s] if ref_audio.ndim == 2 else ref_audio[_ref_start_s:_ref_end_s]
    # SERIE: análisis armónico en serie. own primero, luego ref.
    _own_char_raw = analyze_harmonic_character(_own_audio_for_char, sr)
    _ref_char_raw = analyze_harmonic_character(_ref_audio_for_char, ref_sr)
    del _own_audio_for_char, _ref_audio_for_char

    # ── 1. Bandas comunes de frecuencia (log-spaced), acotadas al nyquist
    #        más chico de los dos archivos para poder compararlos entre sí ──
    nyquist = min(sr, ref_sr) / 2.0
    max_freq = float(np.clip(min(20000.0, nyquist - 100.0), 200.0, nyquist - 1.0))
    edges = np.logspace(np.log10(20.0), np.log10(max_freq), eq_bands + 1)
    band_edges = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    centers = [float(np.sqrt(lo * hi)) for lo, hi in band_edges]

    # SERIE: 4 llamadas espectrales secuenciales (src bands, ref bands,
    # src multires, ref multires). Orden determinístico.
    src_bands_db       = spectral_energy_at_bands(audio, sr, band_edges)
    ref_bands_db       = spectral_energy_at_bands(ref_audio, ref_sr, band_edges)
    src_bands_multires = spectral_energy_at_bands_multires(audio, sr, band_edges)
    ref_bands_multires = spectral_energy_at_bands_multires(ref_audio, ref_sr, band_edges)
    match_before = spectral_match_score_multires(src_bands_multires, ref_bands_multires)

    if str(eq_fit_method).lower() == "ddsp":
        curve = compute_reference_eq_curve_ddsp(src_bands_multires, ref_bands_multires, centers,
                                                 max_boost_db=eq_max_boost_db,
                                                 max_cut_db=eq_max_cut_db)
    else:
        curve = compute_reference_eq_curve(src_bands_db, ref_bands_db, centers,
                                           max_boost_db=eq_max_boost_db,
                                           max_cut_db=eq_max_cut_db,
                                           blend=eq_match_blend)

    # ── 2. EQ de matching iterativo (FIR de fase lineal, N pasadas) ──────
    # Modo M/S (default): curvas INDEPENDIENTES para Mid y Side por pasada.
    # Iterative convergence: en lugar de una pasada al blend fijo, hace N
    # pasadas con blend decreciente [0.60, 0.38, 0.20] — cada pasada corrige
    # el residuo espectral que dejó la anterior sin overcorrect. Resultado:
    # curva efectiva más suave, menos ringing, más natural al oído.
    # Modo legacy (ms_eq_matching=False, iterative_eq_passes=1): comportamiento anterior.
    _report(progress_cb, 20, "Calculando y aplicando EQ de matching iterativo (FIR M/S)")
    audio = eq_high_pass(audio, sr, cutoff_hz=hp_cutoff)

    _passes = max(1, int(iterative_eq_passes))
    if _passes > 1:
        # Iterative convergence: delega toda la lógica a iterative_eq_matching
        audio = iterative_eq_matching(
            audio, sr, ref_audio, ref_sr,
            band_edges=band_edges, centers=centers,
            passes=_passes,
            max_boost_db=eq_max_boost_db, max_cut_db=eq_max_cut_db,
            eq_fit_method=eq_fit_method, eq_q=eq_q,
            ms_mode=(ms_eq_matching and audio.ndim == 2 and audio.shape[0] == 2),
        )
        # Calcular curvas finales solo para el reporte (última pasada)
        src_bands_final = spectral_energy_at_bands(audio, sr, band_edges)
        ref_bands_final = spectral_energy_at_bands(ref_audio, ref_sr, band_edges)
        if ms_eq_matching and audio.ndim == 2 and audio.shape[0] == 2:
            curve_mid, curve_side = compute_ms_eq_curves(
                audio, sr, ref_audio, ref_sr,
                band_edges=band_edges, centers=centers,
                max_boost_db=eq_max_boost_db, max_cut_db=eq_max_cut_db,
                blend=0.20, eq_fit_method=eq_fit_method,
            )
            _eq_curve_for_report = curve_mid
        else:
            curve_mid = curve_side = None
            _eq_curve_for_report = compute_reference_eq_curve(
                src_bands_final, ref_bands_final, centers,
                max_boost_db=eq_max_boost_db, max_cut_db=eq_max_cut_db, blend=0.20)
    elif ms_eq_matching and audio.ndim == 2 and audio.shape[0] == 2:
        # Pasada única M/S (comportamiento anterior)
        curve_mid, curve_side = compute_ms_eq_curves(
            audio, sr, ref_audio, ref_sr,
            band_edges=band_edges, centers=centers,
            max_boost_db=eq_max_boost_db, max_cut_db=eq_max_cut_db,
            blend=eq_match_blend, eq_fit_method=eq_fit_method,
            src_bands_multires=src_bands_multires,
            ref_bands_multires=ref_bands_multires,
        )
        audio = apply_ms_matching_fir(audio, sr, curve_mid, curve_side, eq_q=eq_q)
        _eq_curve_for_report = curve_mid
    else:
        fir_taps = build_matching_fir(curve, sr, precision=eq_q)
        audio = apply_matching_fir(audio, sr, fir_taps)
        curve_mid = curve_side = None
        _eq_curve_for_report = curve

    # ── 2c. Crest factor matching ─────────────────────────────────────────
    # Iguala la dinámica percibida (peak/RMS) al de la referencia, banda por
    # banda para no aplastar el kick intentando igualar el crest de los agudos.
    if match_crest:
        _report(progress_cb, 28, "Igualando crest factor (dinámica percibida)")
        audio = match_crest_factor(
            audio, sr, ref_audio, ref_sr,
            amount=float(np.clip(crest_amount, 0.0, 1.0)),
            band_mode=True,
        )

    # ── 2d. Spectral balance por rango dinámico ───────────────────────────
    _spectral_dynamics_meta = {"applied": False}
    if match_spectral_dynamics and audio.ndim == 2:
        _report(progress_cb, 32, "Analizando balance espectral por rango dinámico")
        try:
            src_dyn_profile = spectral_balance_by_dynamic_range(
                audio, sr, band_edges, n_percentile_bins=int(spectral_dynamics_bins))
            ref_dyn_profile = spectral_balance_by_dynamic_range(
                ref_audio, ref_sr, band_edges, n_percentile_bins=int(spectral_dynamics_bins))
            dynamic_curves = compute_dynamic_range_eq_curves(
                src_dyn_profile, ref_dyn_profile, centers,
                max_boost_db=min(eq_max_boost_db, 4.0),
                max_cut_db=max(eq_max_cut_db, -6.0),
                blend=float(np.clip(spectral_dynamics_amount, 0.0, 1.0)),
            )
            if dynamic_curves:
                _report(progress_cb, 35, "Aplicando EQ dependiente de nivel dinámico")
                audio = apply_dynamic_range_eq_matching(audio, sr, dynamic_curves, eq_q=eq_q)
                _spectral_dynamics_meta = {
                    "applied": True,
                    "n_bins": spectral_dynamics_bins,
                    "amount": round(spectral_dynamics_amount, 2),
                    "src_tonal_slope": src_dyn_profile.get("tonal_slope", {}),
                    "ref_tonal_slope": ref_dyn_profile.get("tonal_slope", {}),
                    "bin_ranges_src": src_dyn_profile.get("bin_ranges", {}),
                    "bin_ranges_ref": ref_dyn_profile.get("bin_ranges", {}),
                }
        except Exception as _e:
            _spectral_dynamics_meta = {"applied": False, "error": str(_e)}

    # ── 2b. Ganancias manuales por banda de frecuencia (ajuste fino post-FIR) ──
    # band_gains_db: lista de {"freq_hz": float, "gain_db": float} — N bandas libres
    # Q automático según densidad: más bandas = Q más alto (más estrecho)
    _applied_band_gains: list = []
    _bga = band_gains_db if isinstance(band_gains_db, list) else []
    if _bga:
        n = len(_bga)
        auto_q = float(np.clip(0.5 + (n / 28.0) * 1.5, 0.7, 2.0))
        for entry in _bga:
            freq_hz = float(entry.get("freq_hz", 0))
            gain    = float(entry.get("gain_db", 0.0))
            if freq_hz < 10 or freq_hz > 22000:
                continue
            _applied_band_gains.append({"freq_hz": round(freq_hz, 1), "gain_db": gain})
            if abs(gain) >= 0.1:
                audio = eq_parametric_band(audio, sr, freq=freq_hz, gain_db=gain, q=auto_q)

    post_eq_bands_db = spectral_energy_at_bands(audio, sr, band_edges)
    match_after_eq = spectral_match_score_multires(
        spectral_energy_at_bands_multires(audio, sr, band_edges), ref_bands_multires)

    # ── Gain staging post-EQ ──────────────────────────────────────────────────
    # El EQ iterativo + crest + spectral dynamics pueden subir el nivel.
    # Normalizar al peak original antes de que entren los pasos de dinámica
    # evita que el limitador final tenga que trabajar al palo.
    _peak_pre_dynamics = float(np.max(np.abs(audio)) + 1e-12)
    _peak_original = float(np.max(np.abs(audio)) + 1e-12)  # referencia para gain staging
    if _peak_pre_dynamics > 0.99:
        audio = audio * (0.99 / _peak_pre_dynamics)

    # ── 3. Dinámica: matching multibanda (graves/medios/agudos) por crest   ──
    # ── factor + matching de macro-dinámica (LRA). Ver match_dynamics_bands ──
    # ── y match_lra: ambas funciones SOLO comprimen (nunca expanden), banda ──
    # ── por banda, comparando cada rango de frecuencia contra el mismo      ──
    # ── rango de la referencia (en vez de un único crest factor global).    ──
    dynamics_band_meta = {name: {"applied": False, "gap_db": 0.0} for name, _, _ in DYNAMICS_BANDS}
    lra_meta = {"applied": False, "own_lra": analysis_before.get("lra", 0.0),
               "ref_lra": analysis_reference.get("lra", 0.0)}
    _report(progress_cb, 40, "Igualando dinámica contra la referencia")
    if match_dynamics:
        # SERIE: crest factors own primero, luego ref.
        own_crest = band_crest_factors(audio, sr)
        ref_crest = band_crest_factors(ref_audio, ref_sr)
        audio, dynamics_band_meta = match_dynamics_bands(
            audio, sr, own_crest, ref_crest, margin_db=dynamics_margin_db,
            oversample=ovs)

        cur_lra = measure_lra(audio, sr)
        audio, lra_meta = match_lra(audio, sr, cur_lra, analysis_reference.get("lra", cur_lra),
                                    margin=dynamics_margin_db, oversample=ovs)

    # ── 4. Ancho estéreo: matching banda por banda (graves/medios/agudos)   ──
    # ── de la correlación L/R contra la referencia. Ver match_stereo_bands. ──
    stereo_k_applied = {name: 1.0 for name, _, _ in DYNAMICS_BANDS}
    _report(progress_cb, 57, "Igualando ancho estéreo contra la referencia")
    if match_stereo_width and audio.ndim == 2 and audio.shape[0] == 2 and ref_audio.ndim == 2 and ref_audio.shape[0] == 2:
        ref_band_corr = band_stereo_correlation(ref_audio, ref_sr)
        audio, stereo_k_applied = match_stereo_bands(
            audio, sr, ref_band_corr, blend=stereo_blend)
    width_applied = round(float(np.mean(list(stereo_k_applied.values()))), 3)

    # ── 5. Punch / transientes: matching por densidad de onsets y crest ───
    # ── Usa transient_shaper calibrado contra la referencia. Solo actúa   ──
    # ── si hay brecha significativa en densidad o crest factor global.    ──
    transient_meta = {"applied": False}
    _report(progress_cb, 63, "Igualando punch/transientes contra la referencia")
    if match_transient:
        own_density = analysis_before.get("transient_density", 0.0)
        ref_density = analysis_reference.get("transient_density", 0.0)
        own_crest_db = analysis_before.get("crest_factor_db", 0.0)
        ref_crest_db = analysis_reference.get("crest_factor_db", 0.0)
        audio, transient_meta = match_transient_punch(
            audio, sr,
            own_density=own_density, ref_density=ref_density,
            own_crest_db=own_crest_db, ref_crest_db=ref_crest_db,
        )

    # ── 6. Perfil de sub-graves (20-250 Hz): shelving dedicado ────────────
    # ── Complementa el EQ FIR de matching (dim 2) con un ajuste fino de   ──
    # ── la región sub/bass comparando la energía banda a banda contra la  ──
    # ── referencia. Shelving IIR de bajo orden, zero-latency.             ──
    sub_meta = {"applied": False}
    _report(progress_cb, 68, "Ajustando perfil de sub-graves contra la referencia")
    if match_sub_bass:
        own_spectrum  = analysis_before.get("spectrum", {})
        ref_spectrum  = analysis_reference.get("spectrum", {})
        own_sub_db  = own_spectrum.get("sub_bass", -60.0)
        ref_sub_db  = ref_spectrum.get("sub_bass", -60.0)
        own_bass_db = own_spectrum.get("bass", -60.0)
        ref_bass_db = ref_spectrum.get("bass", -60.0)
        audio, sub_meta = match_sub_bass_profile(
            audio, sr,
            own_sub_db=own_sub_db, ref_sub_db=ref_sub_db,
            own_bass_db=own_bass_db, ref_bass_db=ref_bass_db,
        )

    # ── 7. De-esser calibrado: compara sibilancia src vs referencia ───────
    # ── Solo corrige si el track es más sibilante que la referencia; si   ──
    # ── la referencia también tiene sibilancia, es el estilo del género   ──
    # ── y no se toca. Ver match_desser_calibrated para la lógica exacta.  ──
    desser_meta = {"applied": False}
    _report(progress_cb, 72, "Calibrando de-esser contra referencia")
    if match_desser:
        src_sibilance = analysis_before.get("sibilance", {})
        ref_sibilance = analysis_reference.get("sibilance", {})
        audio, desser_meta = match_desser_calibrated(
            audio, sr,
            src_sibilance=src_sibilance,
            ref_sibilance=ref_sibilance,
        )

    # ── 7b. Carácter armónico / saturación calibrada ────────────────────────
    _report(progress_cb, 74, "Calibrando carácter armónico contra referencia")
    saturation_meta = {"applied": False}
    if match_saturation:
        # BUGFIX Bug 5: usar los análisis pre-computados sobre el audio CRUDO
        # (_own_char_raw / _ref_char_raw, centro del track) en vez de volver
        # a correr analyze_harmonic_character sobre el audio ya procesado por
        # la cadena y cropped al preview — eso daba lecturas incorrectas de
        # asimetría en tracks donde el intro es más limpio que el cuerpo.
        audio, saturation_meta = match_saturation_character(audio, sr, _own_char_raw, _ref_char_raw)

    # ── Gain staging pre-loudness ─────────────────────────────────────────────
    # Todos los pasos 3-7 (dinámica, estéreo, transientes, sub, de-esser,
    # saturación) pueden haber subido o bajado el nivel. Normalizar a -1dBFS
    # antes de aplicar la ganancia de loudness evita que el boost de loudness
    # empuje una señal que ya estaba cerca del techo.
    _peak_pre_loudness = float(np.max(np.abs(audio)) + 1e-12)
    if _peak_pre_loudness > 0.891:  # -1dBFS
        audio = audio * (0.891 / _peak_pre_loudness)

    # ── 8. Loudness (LUFS) ─────────────────────────────────────────────────
    _report(progress_cb, 77, "Igualando loudness (LUFS) contra la referencia")
    # BUGFIX ("satura TODOS los masters con referencia"): el gain de loudness
    # se aplicaba crudo (audio *= 10**(gain/20)) con un clip de ±24dB, y todo
    # ese gain caía sobre UN SOLO limitador brickwall al final de la cadena
    # (paso 9), sin ningún gain staging intermedio. Un mix propio típico sin
    # masterizar (~-18/-20 LUFS) contra una referencia comercial hot (~-6/-9
    # LUFS) da una brecha de 10-16dB — el caso de uso NORMAL de este feature,
    # no un edge case. Meterle esa ganancia de una sola vez a un limitador
    # significa que pasa la mayor parte del track trabajando al palo (gain
    # reduction profunda y constante, sobre todo en transientes), lo que se
    # escucha como sobre-limitado/con pumping — no clipping de samples, pero
    # sí "saturado" al oído. Bajar (atenuar) nunca estresa al limitador, así
    # que solo se acota el lado de BOOST.
    #
    # AJUSTE #2 (después de que el cap de 12dB fijo salió "todo fuerte y
    # distorsionado, pasado de compresión"): un cap en dB no dice nada sobre
    # dónde termina realmente el loudness — 12dB de boost puede ser perfecto
    # partiendo de -22 LUFS (llega a -10, razonable) o una locura partiendo
    # de -14 LUFS (llega a -2, eso sí sature). Lo que hay que topar es el
    # LUFS DE DESTINO, no el delta. `max_target_lufs` (default -12, elegible
    # por el usuario en un rango admisible, ej. -18 a -6) es el loudness más
    # alto al que el match puede llegar, sin importar qué tan hot venga la
    # referencia — si la referencia está en -6 LUFS pero max_target_lufs es
    # -12, el resultado se frena en -12, no en -6.
    loudness_gain_db = 0.0
    loudness_match_meta = {"mode": "standard_lufs", "adaptive": False}
    if match_loudness:
        if loudness_target_lufs is not None:
            # Modo fijo: ignora el loudness de la referencia por completo.
            # Igual respeta max_target_lufs como techo de seguridad.
            cur_lufs = measure_lufs_integrated(audio, sr)
            _effective_target = float(min(float(loudness_target_lufs), max_target_lufs))
            _raw_gain = float(float(loudness_target_lufs) - cur_lufs)
            loudness_gain_db = float(np.clip(_effective_target - cur_lufs, -24.0, 24.0))
            loudness_match_meta = {
                "mode": "fixed_target_lufs",
                "adaptive": False,
                "source": {"standard_lufs": round(float(cur_lufs), 2)},
                "target_lufs": round(float(loudness_target_lufs), 2),
                "effective_target_lufs": round(_effective_target, 2),
                "reference_lufs_ignored": round(float(analysis_reference["lufs"]), 2),
            }
        elif adaptive_loudness_weighting:
            cur_loudness = measure_human_weighted_loudness(audio, sr, loudness_sensitivity_amount)
            ref_loudness = measure_human_weighted_loudness(ref_audio, ref_sr, loudness_sensitivity_amount)
            cur_std_lufs = float(cur_loudness["standard_lufs"])
            _raw_gain = float(ref_loudness["perceived_lufs"] - cur_loudness["perceived_lufs"])
            # El gain se calcula sobre la métrica perceptual, pero el techo
            # se aplica sobre el LUFS ESTÁNDAR proyectado (lo que realmente
            # va a medir cualquier medidor de streaming), no sobre la
            # métrica perceptual (que no es directamente comparable a un
            # target LUFS convencional).
            _projected_std_lufs = cur_std_lufs + _raw_gain
            _capped_gain = _raw_gain
            if _projected_std_lufs > max_target_lufs:
                _capped_gain = max_target_lufs - cur_std_lufs
            loudness_gain_db = float(np.clip(_capped_gain, -24.0, 24.0))
            loudness_match_meta = {
                "mode": "human_weighted_lufs",
                "adaptive": True,
                "source": cur_loudness,
                "reference": ref_loudness,
                "target_basis": "perceived_lufs_with_3_6khz_sensitivity",
                "max_target_lufs": round(max_target_lufs, 2),
                "projected_standard_lufs": round(min(_projected_std_lufs, max_target_lufs), 2),
            }
        else:
            cur_lufs = measure_lufs_integrated(audio, sr)
            _raw_gain = float(analysis_reference["lufs"] - cur_lufs)
            _effective_target = float(min(float(analysis_reference["lufs"]), max_target_lufs))
            loudness_gain_db = float(np.clip(_effective_target - cur_lufs, -24.0, 24.0))
            loudness_match_meta = {
                "mode": "standard_lufs",
                "adaptive": False,
                "source": {"standard_lufs": round(float(cur_lufs), 2)},
                "reference": {"standard_lufs": round(float(analysis_reference["lufs"]), 2)},
                "max_target_lufs": round(max_target_lufs, 2),
                "effective_target_lufs": round(_effective_target, 2),
            }
        loudness_match_meta["requested_gain_db"] = round(_raw_gain, 2)
        loudness_match_meta["gain_capped"] = bool(_raw_gain > loudness_gain_db + 0.01)
        audio = audio * (10.0 ** (loudness_gain_db / 20.0))

    # ── 8a. Parallel compression (New York) ──────────────────────────────────
    # ANTES del clipper y del limitador — el parallel comp agrega densidad y
    # sustain sin tocar los transientes (el dry nunca se procesa, solo el wet
    # comprimido se mezcla encima). Si va después del limitador no tiene efecto
    # porque el limitador ya aplastó los picos que el dry preservaba.
    _parallel_meta = {"applied": False, "mix": 0.0}
    if use_parallel_compression and parallel_mix > 0.0:
        _report(progress_cb, 81, "Aplicando compresión paralela (New York)")
        audio, _parallel_meta = parallel_compress(
            audio, sr,
            threshold_db=parallel_threshold_db,
            ratio=parallel_ratio,
            attack_ms=parallel_attack_ms,
            release_ms=parallel_release_ms,
            makeup_db=parallel_makeup_db,
            mix=parallel_mix,
            oversample=ovs,
        )

    # ── 8b. Saturación multibanda ─────────────────────────────────────────────
    # Después del parallel comp y antes del clipper — la saturación se beneficia
    # de la señal ya con más cuerpo del parallel comp, y el clipper de después
    # maneja cualquier pico que la saturación haya introducido.
    _mb_sat_meta = {"applied": False}
    if use_multiband_saturation and mb_sat_mix > 0.0:
        _report(progress_cb, 83, "Saturación multibanda (bajos/medios/agudos)")
        audio, _mb_sat_meta = multiband_saturation(
            audio, sr,
            low_drive=mb_sat_low_drive,
            mid_drive=mb_sat_mid_drive,
            high_drive=mb_sat_high_drive,
            mode=mb_sat_mode,
            mix=mb_sat_mix,
            oversample=ovs,
        )

    # ── 8c. Clipper suave (item 14 de la cadena estándar) ANTES del limitador ──
    # MEJORA ("sonido más profesional"): process_audio_with_reference nunca
    # tenía esta etapa — iba directo de la ganancia de loudness al limitador
    # brickwall. Los masters comerciales de referencia (sobre todo los hot,
    # -6/-9 LUFS) casi siempre combinan un clipper suave + limitador, no un
    # limitador solo: el clipper redondea los picos más agudos de forma
    # musical (agrega 2do/3er armónico sutil, "pega" el transiente en vez de
    # solo bajarle el volumen), así el limitador de atrás tiene menos que
    # recortar incluso con la ganancia de loudness ya acotada por el fix
    # anterior. Drive bajo (1.5dB) — es una ayuda, no reemplaza al limitador,
    # que sigue siendo la garantía final de true peak.
    ceiling_pre = float(np.clip(10.0 ** (min(analysis_reference["peak_db"], -0.1) / 20.0), 0.5, 0.99))
    audio, clipper_meta = audio_clipper(audio, sr, ceiling=ceiling_pre, mode="soft",
                                        drive_db=0.0, bypass=False)

    # ── 9. Limitador en dos etapas (gentle + brickwall) ─────────────────────
    _report(progress_cb, 85, "Aplicando limitador en dos etapas")
    ref_peak_db = min(analysis_reference["peak_db"], -0.1)
    _brickwall_ceiling_db = float(np.clip(ref_peak_db, -1.0, -0.1))
    _two_stage_meta = {}
    if use_two_stage_limiter:
        audio, _two_stage_meta = two_stage_limiter(
            audio, sr,
            gentle_ceiling_db=gentle_ceiling_db,
            gentle_release_ms=gentle_release_ms,
            brickwall_ceiling_db=_brickwall_ceiling_db,
            brickwall_release_ms=limiter_release_ms,
            lookahead_ms=5.0,
            oversample=ovs,
        )
        # BUGFIX (UnboundLocalError en runtime, "error.zip"): `ceiling` solo se
        # definía en la rama else de abajo, pero el reporte final más abajo
        # hace round(ceiling, 4) SIEMPRE, sin importar la rama. Como
        # use_two_stage_limiter=True por default, esta rama (if) es la que
        # corre casi siempre, y `ceiling` nunca llegaba a existir → crash
        # garantizado en cualquier reference-match con la config default.
        ceiling = float(np.clip(10.0 ** (_brickwall_ceiling_db / 20.0), 0.5, 0.99))
    else:
        # Fallback al limitador simple anterior
        ceiling = float(np.clip(10.0 ** (_brickwall_ceiling_db / 20.0), 0.5, 0.99))
        audio = limiter(audio, sr, ceiling=ceiling, release_ms=limiter_release_ms,
                        lookahead_ms=5.0, oversample=ovs)

    _report(progress_cb, 93, "Analizando resultado final")
    # SERIE: análisis final secuencial. analyze_audio primero, luego bandas,
    # luego multires.
    analysis_after = analyze_audio(audio, sr)
    final_bands_db = spectral_energy_at_bands(audio, sr, band_edges)
    _after_multi   = spectral_energy_at_bands_multires(audio, sr, band_edges)
    match_after = spectral_match_score_multires(_after_multi, ref_bands_multires)

    # ── 10. Reporte de análisis inteligente (resume las 7 dimensiones) ────
    intelligent_report = reference_intelligent_report(
        match_after, loudness_gain_db, dynamics_band_meta, lra_meta, stereo_k_applied,
        transient_meta=transient_meta, sub_meta=sub_meta, desser_meta=desser_meta,
        saturation_meta=saturation_meta)

    base = os.path.splitext(os.path.basename(input_path))[0]
    suffix = "_preview_refmatch" if preview_seconds else "_refmatch"
    output_dir = "processed"
    # El motor puede utilizarse fuera de FastAPI (CLI, tests, worker dedicado).
    # No debe depender de que app.py haya creado previamente el directorio.
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base}_{uuid.uuid4().hex[:8]}{suffix}.{output_format}")

    # BUGFIX: ser consistente con el shape del output
    # Si es mono (shape[0]==1), devolver 1D para soundfile (que espera (n_samples,) para mono)
    # Si es estéreo (shape[0]==2), devolver (n_channels, n_samples) para soundfile
    if audio.ndim == 2:
        audio_out = audio[0] if audio.shape[0] == 1 else audio
    else:
        audio_out = audio

    _report(progress_cb, 97, "Guardando archivo masterizado")
    effective_bit_depth, dither_meta = _write_master_output(
        audio_out, sr, output_path, output_format, output_bit_depth,
        dither_mode=dither_mode,
    )

    _report(progress_cb, 100, "Mastering por referencia completado")
    return {
        "output_path":       output_path,
        "output_bit_depth":  effective_bit_depth,
        "dither":            dither_meta,
        "analysis_before":   analysis_before,
        "analysis_after":    analysis_after,
        "analysis_reference": analysis_reference,
        "mix_advice_before": mix_advice(analysis_before),
        "mix_advice_after":  mix_advice(analysis_after),
        "recommendations_before": generate_mastering_recommendations(analysis_before),
        "recommendations_after": generate_mastering_recommendations(analysis_after),
        "reference_match": {
            "before":                  match_before,
            "after_eq":                match_after_eq,
            "after":                   match_after,
            "eq_curve_db":             [{"freq_hz": round(f, 1), "gain_db": round(g, 2)} for f, g in _eq_curve_for_report],
            "eq_curve_mid_db":         [{"freq_hz": round(f, 1), "gain_db": round(g, 2)} for f, g in (curve_mid or _eq_curve_for_report)],
            "eq_curve_side_db":        [{"freq_hz": round(f, 1), "gain_db": round(g, 2)} for f, g in (curve_side or _eq_curve_for_report)],
            "band_gains_applied":      _applied_band_gains,
            "loudness_gain_applied_db": round(loudness_gain_db, 2),
            "loudness_match":          loudness_match_meta,
            "parallel_compression":    _parallel_meta,
            "multiband_saturation":    _mb_sat_meta,
            "two_stage_limiter":       _two_stage_meta,
            "eq_match_blend":          round(eq_match_blend, 3),
            "oversample":              ovs,
            "oversample_mode":         str(oversample_mode),
            "stereo_width_applied":     round(width_applied, 3),
            "stereo_width_by_band":     stereo_k_applied,
            "dynamics_by_band":         dynamics_band_meta,
            "lra":                      lra_meta,
            "transient_punch":          transient_meta,
            "sub_bass_profile":         sub_meta,
            "spectral_dynamics":        _spectral_dynamics_meta,
            "desser_calibrated":        desser_meta,
            "saturation_character":     saturation_meta,
            "clipper":                  clipper_meta,
            "limiter_ceiling":          round(ceiling, 4),
            "intelligent_report":       intelligent_report,
            "premium_features": {
                "match_profile": str(premium_match_profile),
                "vocal_protect": bool(premium_vocal_protect),
                "translation_check": bool(premium_translation_check),
                "alt_versions": bool(premium_alt_versions),
            },
        },
        "chain_meters": {
            "post_limiter": {
                "rms_db":  analysis_after["rms_db"],
                "peak_db": analysis_after["peak_db"],
                "lufs":    analysis_after["lufs"],
                "lra":     analysis_after.get("lra"),
                "stereo_correlation": analysis_after["stereo_correlation"],
            },
        },
    }