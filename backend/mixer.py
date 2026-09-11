"""
mixer.py — Motor de mezcla multistem.

Permite subir N stems (kick, bajo, voz, synth, etc.), procesar cada uno
independientemente (ganancia, pan, HP/LP, EQ 4 bandas, compresión, transient,
sidechain) y mezclarlos antes de pasar por la cadena de mastering.

Reutiliza las funciones DSP ya existentes en mastering.py:
  - eq_parametric_band, eq_high_pass, eq_low_pass
  - compressor
  - transient_shaper
  - stereo_width
  - apply_mastering_chain
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from .mastering import (
        eq_parametric_band, eq_high_pass, eq_low_pass,
        compressor, transient_shaper, stereo_width,
        apply_mastering_chain, measure_lufs_integrated,
        limiter,
    )
    from .config import PROCESSED_DIR
    from .reverb import ReverbProcessor
    from .pitch_correction import PitchCorrectionProcessor
except ImportError:
    from mastering import (
        eq_parametric_band, eq_high_pass, eq_low_pass,
        compressor, transient_shaper, stereo_width,
        apply_mastering_chain, measure_lufs_integrated,
        limiter,
    )
    from config import PROCESSED_DIR
    from reverb import ReverbProcessor
    from pitch_correction import PitchCorrectionProcessor


# ── Parámetros por stem ───────────────────────────────────────────────────────

@dataclass
class StemParams:
    """Todos los parámetros de procesamiento de un stem individual."""

    name: str = "stem"
    stem_type: str = "other"   # kick/snare/bass/guitar/vocals/synth/fx/other

    # Ganancia y panorama
    gain_db: float = 0.0          # -24 .. +12
    pan: float = 0.0              # -1.0 (izq) .. 1.0 (der)
    mute: bool = False
    solo: bool = False

    # Filtros de corte
    hp_cutoff_hz: float = 20.0    # high-pass — sacar rumble
    lp_cutoff_hz: float = 20000.0 # low-pass

    # EQ 4 bandas paramétricas
    eq_low_freq: float   = 100.0;  eq_low_gain_db: float   = 0.0;  eq_low_q: float   = 0.8
    eq_lomid_freq: float = 500.0;  eq_lomid_gain_db: float = 0.0;  eq_lomid_q: float = 1.0
    eq_himid_freq: float = 3000.0; eq_himid_gain_db: float = 0.0;  eq_himid_q: float = 1.0
    eq_high_freq: float  = 10000.0;eq_high_gain_db: float  = 0.0;  eq_high_q: float  = 0.8

    # Compresor
    comp_enabled: bool = False
    comp_threshold: float = 0.5     # lineal 0..1
    comp_ratio: float = 4.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 100.0
    comp_makeup_db: float = 0.0
    comp_stereo_link: bool = True
    comp_pdr: bool = True

    # Transient shaper
    transient_attack: float = 0.0   # -1..+1
    transient_sustain: float = 0.0  # -1..+1

    # Ancho estéreo por stem
    stereo_width_amount: float = 1.0   # 0=mono, 1=original, 2=extra wide

    # Sidechain — índice del stem que hace de trigger (None = desactivado)
    sidechain_trigger_name: Optional[str] = None
    sidechain_threshold: float = 0.3   # lineal 0..1
    sidechain_ratio: float = 6.0
    sidechain_attack_ms: float = 5.0
    sidechain_release_ms: float = 80.0

    # Reverb Convolutivo per-canal
    reverb_enabled: bool = False
    reverb_preset: str = "small_studio"  # small_studio, large_hall, cathedral, live_venue, plate, spring
    reverb_wet_amount: float = 0.3    # 0.0=dry, 1.0=100% wet
    reverb_pre_delay_ms: float = 0.0  # 0-200ms
    reverb_room_size: float = 1.0     # 0.3-2.0 (escala el decay time)

    # Pitch Correction per-canal
    pitch_correction_enabled: bool = False
    pitch_correction_mode: str = "MEDIUM"  # OFF, LIGHT, MEDIUM, STRONG
    pitch_correction_scale: Optional[str] = None  # C_major, A_minor, etc.
    pitch_correction_glide_ms: float = 50.0  # 0..200ms


@dataclass
class MixParams:
    """Parámetros globales del mix."""
    master_gain_db: float = 0.0
    master_limiter_ceiling: float = 0.95
    normalize_before_master: bool = True
    target_lufs: float = -14.0
    # Parámetros de la cadena de mastering (pasan directo a apply_mastering_chain)
    chain_params: dict = field(default_factory=dict)


# ── DSP por stem ──────────────────────────────────────────────────────────────

def process_stem(audio: np.ndarray, sr: int, p: StemParams) -> tuple:
    """Aplica toda la cadena de procesamiento a un stem individual.

    Orden de procesamiento:
      1. Ganancia de entrada
      2. HP / LP
      3. EQ 4 bandas
      4. Compresor
      5. Transient shaper
      6. Ancho estéreo
      7. Paneo

    Devuelve (audio_procesado, meters_dict).
    """
    if p.mute:
        return np.zeros_like(audio), {"muted": True}

    meters: dict = {}

    # 1. Ganancia
    gain_lin = 10.0 ** (p.gain_db / 20.0)
    audio = audio * gain_lin

    # 2. Filtros de corte
    if p.hp_cutoff_hz > 20.0:
        audio = eq_high_pass(audio, sr, cutoff_hz=p.hp_cutoff_hz)
    if p.lp_cutoff_hz < 20000.0:
        audio = eq_low_pass(audio, sr, cutoff_hz=p.lp_cutoff_hz)

    # 3. EQ 4 bandas
    eq_bands = [
        (p.eq_low_freq,   p.eq_low_gain_db,   p.eq_low_q),
        (p.eq_lomid_freq, p.eq_lomid_gain_db, p.eq_lomid_q),
        (p.eq_himid_freq, p.eq_himid_gain_db, p.eq_himid_q),
        (p.eq_high_freq,  p.eq_high_gain_db,  p.eq_high_q),
    ]
    for freq, gain, q in eq_bands:
        if abs(gain) >= 0.1:
            audio = eq_parametric_band(audio, sr, freq=freq, gain_db=gain, q=q)

    # 4. Compresor
    if p.comp_enabled and p.comp_ratio > 1.0:
        audio, comp_meters = compressor(
            audio, sr,
            threshold=p.comp_threshold,
            ratio=p.comp_ratio,
            attack_ms=p.comp_attack_ms,
            release_ms=p.comp_release_ms,
            makeup_db=p.comp_makeup_db,
            stereo_link=p.comp_stereo_link,
            pdr=p.comp_pdr,
        )
        meters["comp"] = comp_meters

    # 5. Transient shaper
    if abs(p.transient_attack) > 0.01 or abs(p.transient_sustain) > 0.01:
        audio = transient_shaper(
            audio, sr,
            attack_amount=p.transient_attack,
            sustain_amount=p.transient_sustain,
        )

    # 6. Ancho estéreo
    if p.stereo_width_amount != 1.0 and audio.ndim == 2 and audio.shape[0] == 2:
        audio = stereo_width(audio, width=p.stereo_width_amount)

    # 7. Paneo (ley de paneo de potencia constante)
    if abs(p.pan) > 0.01:
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        if audio.shape[0] == 2:
            angle = p.pan * np.pi / 4.0   # -45° .. +45°
            gain_l = np.cos(angle + np.pi / 4.0) * np.sqrt(2.0)
            gain_r = np.sin(angle + np.pi / 4.0) * np.sqrt(2.0)
            audio = np.stack([audio[0] * gain_l, audio[1] * gain_r])

    # 8. Reverb Convolutivo (si está habilitado)
    if p.reverb_enabled and p.reverb_wet_amount > 0.01:
        try:
            # Crear un ReverbProcessor lazily (per-stem)
            reverb_proc = ReverbProcessor(sr)
            audio = reverb_proc.process(
                audio,
                preset=p.reverb_preset,
                wet_amount=p.reverb_wet_amount,
                pre_delay_ms=p.reverb_pre_delay_ms,
                room_size=p.reverb_room_size,
            )
            meters["reverb"] = {
                "preset": p.reverb_preset,
                "wet_amount": p.reverb_wet_amount,
            }
        except Exception as e:
            import logging
            logging.warning(f"Error applying reverb to stem '{p.name}': {e}")

    # 9. Pitch Correction (si está habilitado)
    if p.pitch_correction_enabled and p.pitch_correction_mode != "OFF":
        try:
            processor = PitchCorrectionProcessor(sr)
            if audio.ndim == 2 and audio.shape[0] == 2:
                corrected = np.stack([
                    processor.process(audio[0], mode=p.pitch_correction_mode, scale=p.pitch_correction_scale, glide_time_ms=p.pitch_correction_glide_ms),
                    processor.process(audio[1], mode=p.pitch_correction_mode, scale=p.pitch_correction_scale, glide_time_ms=p.pitch_correction_glide_ms),
                ])
            else:
                corrected = processor.process(audio, mode=p.pitch_correction_mode, scale=p.pitch_correction_scale, glide_time_ms=p.pitch_correction_glide_ms)
            audio = corrected
            meters["pitch_correction"] = {
                "mode": p.pitch_correction_mode,
                "scale": p.pitch_correction_scale,
                "glide_ms": p.pitch_correction_glide_ms,
            }
        except Exception as e:
            import logging
            logging.warning(f"Error applying pitch correction to stem '{p.name}': {e}")

    meters["peak_db"]  = float(20.0 * np.log10(np.max(np.abs(audio)) + 1e-9))
    return audio.astype(np.float32), meters


def apply_sidechain(
    target: np.ndarray,
    trigger: np.ndarray,
    sr: int,
    threshold: float = 0.3,
    ratio: float = 6.0,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
) -> tuple:
    """Sidechain ducking: comprime `target` usando la envolvente de `trigger`.

    Típico uso: comprimir el bajo cuando el kick pega.
    Devuelve (audio_duckeado, meter).
    """
    # Asegurar misma longitud
    min_len = min(target.shape[-1], trigger.shape[-1])
    target  = target[..., :min_len]
    trigger = trigger[..., :min_len]

    # Extraer envolvente del trigger (mono)
    trigger_mono = trigger.mean(axis=0) if trigger.ndim == 2 else trigger
    trigger_mono = np.abs(trigger_mono).astype(np.float64)

    # Detector sidechain usando el mismo compresor pero con señal de control externa
    # Creamos un audio "ficticio" mono con la envolvente del trigger para que
    # el detector del compresor calcule la GR, y la aplicamos al target.
    try:
        from mastering import _smooth_envelope, _soft_knee_gain_reduction_np
    except ImportError:
        from mastering import _smooth_envelope, _soft_knee_gain_reduction_np

    threshold_db = 20.0 * np.log10(max(threshold, 1e-9))
    env = _smooth_envelope(trigger_mono, sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(env + 1e-9)
    gr_db  = _soft_knee_gain_reduction_np(env_db, threshold_db, ratio, knee_db=3.0)
    gr_lin = 10.0 ** (gr_db / 20.0)

    # Aplicar GR al target
    if target.ndim == 2:
        ducked = target * gr_lin[np.newaxis, :]
    else:
        ducked = target * gr_lin

    avg_gr = float(np.mean(gr_db))
    return ducked.astype(np.float32), {"avg_gr_db": round(avg_gr, 2)}


# ── Mix ───────────────────────────────────────────────────────────────────────

def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Garantiza shape (2, N)."""
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    return audio


def _match_length(arrays: list) -> list:
    """Rellena con ceros los arrays más cortos para que todos tengan la misma longitud."""
    max_len = max(a.shape[-1] for a in arrays)
    result = []
    for a in arrays:
        if a.shape[-1] < max_len:
            pad = np.zeros((*a.shape[:-1], max_len - a.shape[-1]), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=-1)
        result.append(a)
    return result


def mix_and_master(
    stems: dict[str, np.ndarray],
    sr: int,
    stem_params: dict[str, StemParams],
    mix_params: MixParams,
    progress_cb=None,
) -> dict:
    """Pipeline completo: stems → proceso individual → sidechain → mix → mastering.

    Args:
        stems: {nombre: audio_array (2D float32)}
        sr: sample rate (todos los stems deben tener el mismo)
        stem_params: {nombre: StemParams}
        mix_params: MixParams globales
        progress_cb: callback(pct, stage)

    Returns:
        dict con output_path, analysis, meters, etc.
    """
    def _report(pct, stage):
        if progress_cb:
            progress_cb(pct, stage)

    # BUGFIX: validar que todos los stems en stem_params existen en stems
    invalid_stems = set(stem_params.keys()) - set(stems.keys())
    if invalid_stems:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"stem_params contiene stems que no existen: {invalid_stems}. Se ignorarán.")

    _report(5, "Procesando stems individuales")

    # ── 1. Procesar cada stem ─────────────────────────────────────────────────
    processed: dict[str, np.ndarray] = {}
    stem_meters: dict[str, dict] = {}
    solo_active = any(p.solo for p in stem_params.values() if p is not None)

    for name, audio in stems.items():
        p = stem_params.get(name, StemParams(name=name))
        # Si hay solo activo, mutear todos los que no tienen solo
        if solo_active and not p.solo:
            processed[name] = np.zeros_like(audio)
            stem_meters[name] = {"muted": True, "soloed_out": True}
            continue
        audio_2d = _ensure_stereo(audio)
        proc, meters = process_stem(audio_2d, sr, p)
        processed[name] = proc
        stem_meters[name] = meters

    _report(30, "Aplicando sidechain")

    # ── 2. Sidechain ─────────────────────────────────────────────────────────
    sidechain_meters: dict[str, dict] = {}
    for name, p in stem_params.items():
        if p.sidechain_trigger_name and p.sidechain_trigger_name in processed:
            trigger = processed[p.sidechain_trigger_name]
            target  = processed[name]
            ducked, sc_meters = apply_sidechain(
                target, trigger, sr,
                threshold=p.sidechain_threshold,
                ratio=p.sidechain_ratio,
                attack_ms=p.sidechain_attack_ms,
                release_ms=p.sidechain_release_ms,
            )
            processed[name] = ducked
            sidechain_meters[name] = sc_meters

    _report(45, "Mezclando stems")

    # ── 3. Mix ────────────────────────────────────────────────────────────────
    arrays = _match_length(list(processed.values()))
    mix = np.sum(arrays, axis=0).astype(np.float32)

    # Ganancia master
    if abs(mix_params.master_gain_db) > 0.01:
        mix = mix * 10.0 ** (mix_params.master_gain_db / 20.0)

    # Normalización pre-master (evita saturar la cadena)
    if mix_params.normalize_before_master:
        peak = np.max(np.abs(mix))
        if peak > 0.9:
            mix = mix * (0.9 / peak)

    _report(55, "Aplicando cadena de mastering")

    # ── 4. Mastering chain ────────────────────────────────────────────────────
    import tempfile, os, soundfile as sf

    # Guardar mix temporal para process_audio si se pasan chain_params completos
    # Alternativamente, aplicar apply_mastering_chain directo
    chain_params = dict(mix_params.chain_params)
    mastered, chain_meters = apply_mastering_chain(mix, sr, **chain_params)
    mastered = mastered.astype(np.float32)

    _report(80, "Limitador final")

    # Limiter de seguridad final
    ceiling = float(mix_params.master_limiter_ceiling)
    mastered = limiter(mastered, sr, ceiling=ceiling, release_ms=60.0, lookahead_ms=5.0)

    _report(90, "Guardando")

    # ── 5. Guardar output ─────────────────────────────────────────────────────
    # Antes esto guardaba en os.environ.get("PROCESSED_DIR", "/tmp") — esa es
    # una variable de entorno del sistema, no la config de la app, así que
    # casi siempre caía en "/tmp" en vez de la carpeta "processed/" real que
    # usan todos los demás jobs (y que la app efectivamente crea y cuida).
    import uuid as _uuid
    output_path = os.path.join(PROCESSED_DIR, f"mix_{_uuid.uuid4().hex}.wav")
    data_to_write = mastered.T if mastered.ndim == 2 else mastered
    sf.write(output_path, data_to_write, sr, subtype="PCM_24")

    # ── 6. Análisis ───────────────────────────────────────────────────────────
    try:
        lufs = float(measure_lufs_integrated(mastered, sr))
    except Exception:
        lufs = float(20.0 * np.log10(np.sqrt(np.mean(mastered**2)) + 1e-9))

    peak_db = float(20.0 * np.log10(np.max(np.abs(mastered)) + 1e-9))

    _report(100, "Completado")

    return {
        "output_path":    output_path,
        "lufs":           round(lufs, 2),
        "peak_db":        round(peak_db, 2),
        "stem_meters":    stem_meters,
        "sidechain_meters": sidechain_meters,
        "chain_meters":   chain_meters,
        "n_stems":        len(stems),
        "stem_names":     list(stems.keys()),
    }
