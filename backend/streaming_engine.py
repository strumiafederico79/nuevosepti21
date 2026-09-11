"""
Motor de mastering en tiempo real / streaming.
Procesa el audio en bloques (chunks) entregando PCM float32 y métricas completas
(LUFS, peak, RMS, correlación estéreo, GR multibanda, GR banda ancha y
glue) para visualización en vivo sin bloquear la interfaz.

Optimizaciones de CPU:
- Chunks de 4s por defecto (antes 2s) — mitad de llamadas a apply_mastering_chain
- LUFS integrado solo cada 4 chunks — el cálculo es costoso y no cambia tanto chunk a chunk
- FFT con N_FFT=1024 (antes 2048) — suficiente resolución para 32 bandas visuales
- recommend_dynamic_eq cada 8s (antes 6s)
- true_peak y mono_compat solo cada 8 chunks
"""
import numpy as np # type: ignore
import os
import atexit

try:
    from .mastering import apply_mastering_chain, measure_lufs_integrated, stereo_correlation, recommend_dynamic_eq, true_peak_dbfs, mono_compatibility_db
except ImportError:  # pragma: no cover - fallback for direct script execution
    from mastering import apply_mastering_chain, measure_lufs_integrated, stereo_correlation, recommend_dynamic_eq, true_peak_dbfs, mono_compatibility_db

# El ProcessPoolExecutor anterior se creaba al importar el módulo aunque no se usaba.
# En Python 3.14 eso puede dejar semáforos IPC pendientes al terminar el proceso.
# El streaming actual procesa por bloques sin usar ese pool global, así que se elimina
# la creación eager y se conserva una API de shutdown idempotente para el lifecycle.

def shutdown_chain_pool() -> None:
    return None


CHUNK_SECONDS_DEFAULT = 4.0        # era 2.0 — mitad de chunks = mitad de DSP calls
# Overlap-Add (OLA): el overlap da contexto a los filtros IIR pero en vez de
# descartar esas muestras se hace crossfade (OLA) entre la cola procesada del
# chunk anterior y la cabeza del chunk actual. Así no se pierde audio real —
# crítico para acapellas y voz continua donde el overlap previo cortaba el sonido.
DEFAULT_OVERLAP_SECONDS = 0.05     # 50ms — suficiente para preview (era 100ms)
OLA_CROSSFADE_SECONDS   = 0.05     # crossfade 50ms (era 100ms)
DYNAMIC_EQ_DETECT_SECONDS_DEFAULT = 8.0  # era 6.0
_LUFS_EVERY_N_CHUNKS = 4           # LUFS integrado cada 4 chunks
_HEAVY_METRICS_EVERY_N = 8         # true_peak y mono_compat cada 8 chunks
_FFT_SIZE = 1024                   # era 2048 — suficiente para 32 bandas visuales


def iter_mastering_chunks(audio: np.ndarray, sr: int,
                          chunk_seconds: float = CHUNK_SECONDS_DEFAULT,
                          overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
                          detect_dynamic_eq: bool = True,
                          dynamic_eq_detect_seconds: float = DYNAMIC_EQ_DETECT_SECONDS_DEFAULT,
                          **chain_params):
    """
    Yields (processed_block, metrics_dict) for each chunk.

    Para reducir artefactos entre bloques, el motor procesa cada chunk con un
    pequeño contexto del chunk anterior (solapamiento temporal). Esto mejora la
    continuidad del procesamiento sin cambiar la API pública.

    Si `detect_dynamic_eq=True` (default), cada `dynamic_eq_detect_seconds`
    se corre `recommend_dynamic_eq` sobre el bloque crudo (antes de la cadena).
    """
    if audio is None:
        return

    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
        input_is_mono = True
    elif audio.ndim == 2:
        audio_2d = audio
        input_is_mono = False
    else:
        raise ValueError("audio debe ser mono o estéreo (1D/2D)")

    total_samples = int(audio_2d.shape[-1])
    chunk_samples = max(1, int(chunk_seconds * sr))
    overlap_samples = max(0, int(overlap_seconds * sr))
    overlap_samples = min(overlap_samples, max(1, chunk_samples // 2))
    n_chunks = int(np.ceil(total_samples / chunk_samples))

    # OLA: buffer con la cola procesada del chunk anterior para el crossfade
    ola_tail = None   # np.ndarray (channels, ola_samples) o None en el primer chunk
    ola_samples = max(0, int(OLA_CROSSFADE_SECONDS * sr))
    ola_samples = min(ola_samples, chunk_samples // 2)

    # Ventana de crossfade coseno (igual potencia — suma al cuadrado = 1.0)
    # Evita el dip de amplitud en el centro que da la rampa lineal.
    if ola_samples > 0:
        t = np.linspace(0.0, np.pi / 2.0, ola_samples, dtype=np.float32)
        ola_fade_out = np.cos(t)      # 1 → 0  (cola del chunk anterior)
        ola_fade_in  = np.sin(t)      # 0 → 1  (cabeza del chunk actual)

    context = np.zeros((audio_2d.shape[0], overlap_samples), dtype=np.float32)

    dynamic_eq_recommendation = None
    last_dynamic_eq_detect_time = -float(dynamic_eq_detect_seconds)

    _last_lufs = -70.0
    _last_true_peak = -70.0
    _last_mono_compat = 0.0
    _last_corr = 0.0
    _fft_window = np.hanning(_FFT_SIZE).astype(np.float32)

    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, total_samples)
        block = audio_2d[:, start:end]
        if block.shape[-1] == 0:
            continue

        block = np.asarray(block, dtype=np.float32)
        if overlap_samples > 0 and context.shape[-1] > 0:
            combined = np.concatenate([context, block], axis=1)
        else:
            combined = block

        processed, chain_meters = apply_mastering_chain(combined, sr, **chain_params)
        processed = np.asarray(processed, dtype=np.float32)

        # Extraer el bloque útil (sin el overlap de contexto)
        if overlap_samples > 0 and processed.shape[-1] > overlap_samples:
            out_block = processed[:, overlap_samples:overlap_samples + block.shape[-1]]
        else:
            out_block = processed[:, :block.shape[-1]]

        if out_block.shape[-1] < block.shape[-1]:
            pad = np.zeros((out_block.shape[0], block.shape[-1] - out_block.shape[-1]), dtype=np.float32)
            out_block = np.concatenate([out_block, pad], axis=1)

        # ── Overlap-Add (OLA) en el borde entre chunks ───────────────────────
        # En vez de descartar el overlap o hacer un fade a cero (que cortaba
        # la voz), mezclamos la COLA del chunk anterior con la CABEZA del actual
        # usando una ventana coseno de igual potencia. No se pierde ninguna
        # muestra — el crossfade solo suaviza la transición entre los dos
        # resultados procesados independientemente.
        if ola_samples > 0 and ola_tail is not None and out_block.shape[-1] >= ola_samples:
            cf = min(ola_samples, ola_tail.shape[-1])
            # Mezclar: tail_anterior * fade_out + head_actual * fade_in
            out_block[:, :cf] = (ola_tail[:, -cf:] * ola_fade_out[-cf:][np.newaxis, :]
                                 + out_block[:, :cf] * ola_fade_in[-cf:][np.newaxis, :])

        # Guardar la cola de este chunk para el próximo OLA
        if ola_samples > 0 and out_block.shape[-1] >= ola_samples:
            ola_tail = out_block[:, -ola_samples:].copy()
        else:
            ola_tail = None

        if input_is_mono:
            out_block = out_block[0]

        mono = out_block.mean(axis=0) if out_block.ndim == 2 else out_block
        mono = np.asarray(mono, dtype=np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
        peak_db = float(20.0 * np.log10(peak + 1e-9))
        rms_db  = float(20.0 * np.log10(rms  + 1e-9))

        # LUFS — costoso, solo cada N chunks
        if i % _LUFS_EVERY_N_CHUNKS == 0:
            try:
                v = measure_lufs_integrated(out_block, sr)
                _last_lufs = v if np.isfinite(v) else rms_db - 0.691
            except Exception:
                _last_lufs = rms_db - 0.691
        lufs_chunk = _last_lufs

        # Correlación estéreo — liviana, cada chunk
        try:
            _last_corr = stereo_correlation(out_block)
        except Exception:
            pass
        corr = _last_corr

        # True peak y mono compat — costosos, solo cada N chunks
        if i % _HEAVY_METRICS_EVERY_N == 0:
            try:
                _last_true_peak = true_peak_dbfs(out_block, sr)
            except Exception:
                _last_true_peak = peak_db
            try:
                _last_mono_compat = mono_compatibility_db(out_block)
            except Exception:
                _last_mono_compat = 0.0
        true_peak = _last_true_peak
        mono_compat = _last_mono_compat

        # Dynamic EQ detection
        if detect_dynamic_eq:
            current_time_sec = start / sr
            if current_time_sec - last_dynamic_eq_detect_time >= dynamic_eq_detect_seconds:
                try:
                    dynamic_eq_recommendation = recommend_dynamic_eq(block, sr)
                except Exception:
                    pass
                last_dynamic_eq_detect_time = current_time_sec

        # FFT compacta — 32 bandas log, N_FFT=1024
        try:
            mono_fft = (out_block.mean(axis=0) if out_block.ndim == 2 else out_block).astype(np.float32)
            if len(mono_fft) >= _FFT_SIZE:
                frame = mono_fft[-_FFT_SIZE:]
            else:
                frame = np.pad(mono_fft, (_FFT_SIZE - len(mono_fft), 0))
            spectrum = np.abs(np.fft.rfft(frame * _fft_window))
            freqs = np.fft.rfftfreq(_FFT_SIZE, 1.0 / sr)
            N_BANDS = 32
            edges = np.logspace(np.log10(20.0), np.log10(20000.0), N_BANDS + 1)
            bands_db = []
            for b in range(N_BANDS):
                mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
                val = float(np.mean(spectrum[mask])) if mask.any() else 0.0
                bands_db.append(round(float(20.0 * np.log10(val + 1e-9)), 1))
            spectrum_data = {"bands_db": bands_db, "freq_edges": [round(e, 1) for e in edges.tolist()]}
        except Exception:
            spectrum_data = {}

        metrics = {
            "chunk_index": i,
            "n_chunks": n_chunks,
            "progress_pct": round(((i + 1) / n_chunks) * 100.0, 1),
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2),
            "lufs_momentary": round(lufs_chunk, 2),
            "true_peak_db": round(true_peak, 2),
            "mono_compatibility_db": round(mono_compat, 2),
            "stereo_correlation": round(corr, 3),
            "time_sec": round(start / sr, 2),
            "chain_meters":   chain_meters,
            "comp_gr_db": round(float(chain_meters.get("comp", {}).get("gr_db", 0.0)), 2),
            "limiter_gr_db": round(float(chain_meters.get("limiter", {}).get("gr_db", 0.0)), 2),
            "glue_gr_db": round(float(chain_meters.get("glue", {}).get("gr_db", 0.0)), 2),
            "limiter_meters": chain_meters.get("limiter", {}),
            "mb_meters":      chain_meters.get("mb", {}),
            "comp_meters":    chain_meters.get("comp", {}),
            "parallel_meters": chain_meters.get("parallel", {}),
            "glue_meters":    chain_meters.get("glue", {}),
            "dyneq_meters":   chain_meters.get("dyneq", {}),
            "reso_meters":    chain_meters.get("reso", {}),
            "ms_comp_meters": chain_meters.get("ms_comp", {}),
            "tonal_balance_meters": chain_meters.get("tonal_balance", {}),
            "pre_limiter":    chain_meters.get("pre_limiter", {}),
            "post_limiter":   chain_meters.get("post_limiter", {}),
            "spectrum": spectrum_data,
            "dynamic_eq_recommendation": dynamic_eq_recommendation,
        }

        context = block[:, -overlap_samples:] if overlap_samples > 0 else np.zeros((block.shape[0], 0), dtype=np.float32)
        yield out_block, metrics


def master_stream_to_pcm(audio: np.ndarray, sr: int,
                           chunk_seconds: float = CHUNK_SECONDS_DEFAULT,
                           pcm_format: str = "float32",
                           prefetch: int = 3,
                           **chain_params):
    """
    Yields (pcm_bytes, metrics_dict) para cada chunk procesado.

    `prefetch` mantiene una pequeña cola de chunks ya procesados para amortiguar
    jitter del consumidor. El procesamiento de DSP sigue siendo secuencial y
    ordenado; no se anuncia paralelismo que el generador no realiza.

    pcm_format="int16": preview estándar (/ws/master-stream)
    pcm_format="float32": ref-stream
    pcm_format="pcm24": signed 24-bit PCM little-endian (3 bytes/sample)
    """
    chain_params.setdefault("detect_dynamic_eq", False)

    # Preparar chunks como lista de bloques de audio pre-cortados
    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio
    total = audio_2d.shape[-1]
    chunk_samples = max(1, int(chunk_seconds * sr))
    n_chunks = int(np.ceil(total / chunk_samples))

    def _to_pcm(processed: np.ndarray) -> bytes:
        block = processed.T if processed.ndim == 2 else processed
        clipped = np.clip(block, -1.0, 1.0)
        if pcm_format == "int16":
            return (clipped * 32767.0).astype(np.int16).tobytes()
        if pcm_format in ("pcm24", "int24"):
            # PCM lineal entero firmado de 24 bits, little-endian, intercalado.
            # El rango útil es [-8388608, 8388607].
            q = np.rint(clipped * 8388607.0).astype(np.int32)
            u = q.astype(np.uint32)
            packed = np.stack((
                u & 0xFF,
                (u >> 8) & 0xFF,
                (u >> 16) & 0xFF,
            ), axis=-1).astype(np.uint8)
            return packed.reshape(-1).tobytes()
        return clipped.astype(np.float32).tobytes()

    # Generador base (sin prefetch) — fallback y para ref-stream
    def _serial_gen():
        for processed, metrics in iter_mastering_chunks(audio, sr,
                                                        chunk_seconds=chunk_seconds,
                                                        **chain_params):
            yield _to_pcm(processed), metrics

    # Cola de prefetch secuencial: mantiene N chunks preparados para amortiguar
    # jitter del consumidor sin introducir otro scheduler de DSP.
    try:
        from collections import deque
        queue = deque()
        gen = _serial_gen()

        # Llenar la cola inicial con prefetch chunks
        for _ in range(prefetch):
            try:
                item = next(gen)
                queue.append(item)
            except StopIteration:
                break

        while queue:
            # Yield el chunk más antiguo (ya procesado)
            yield queue.popleft()
            # Intentar agregar otro chunk al final de la cola
            try:
                queue.append(next(gen))
            except StopIteration:
                pass

    except Exception:
        # Fallback serial si algo falla
        yield from _serial_gen()


# Compatibilidad con integraciones existentes. Nuevo nombre canónico: master_stream_to_pcm.
master_stream_to_pcm16 = master_stream_to_pcm
