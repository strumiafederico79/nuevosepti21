"""
Stem separation (#13) usando Demucs (htdemucs_ft, 4 stems: vocals/drums/bass/other).

htdemucs_ft es la variante "fine-tuned": un ensamble (bag) de 4 submodelos,
cada uno afinado para separar mejor un stem específico. Mejor calidad que
htdemucs plano, a costa de ~4x más tiempo de cómputo (corre las 4 pasadas
internamente). demucs_infer.pretrained.get_model() maneja el bag-of-models
de forma transparente, así que apply_model() no necesita cambios.

Requisitos (agregar a requirements.txt):
    demucs-infer>=4.1.2   (fork mantenido de Demucs, compatible con torch 2.x
                           y Python 3.10+; el "demucs" original de Meta ya no
                           se mantiene y no instala en Python 3.10+)
    torch>=2.6, torchaudio>=2.6   (si hay GPU: instalar build con CUDA)

Notas de hardware:
- Con GPU (CUDA), un track de ~4 min separa en uno o pocos minutos.
- En CPU puro puede tardar bastante más por track (htdemucs_ft es ~4x más
  pesado que htdemucs plano, al ser un ensamble de 4 modelos). Como el VPS
  no tiene límite de recursos, dejamos device='auto' por defecto (usa CUDA
  si torch.cuda.is_available(), si no cae a CPU). El job corre en background
  sin timeout propio, así que tarda lo que tarde sin cortarse.
- El modelo se descarga una única vez (~80-300MB por submodelo, x4 en el
  caso de htdemucs_ft) y se cachea en ~/.cache/torch/hub/checkpoints — la
  primera separación va a ser más lenta por la descarga.

API pública:
    separate_stems(audio, sr, progress_cb=None, model_name="htdemucs_ft", device=None)
        -> dict[str, np.ndarray]   # {"vocals":..,"drums":..,"bass":..,"other":..}
        Cada array tiene shape (channels, samples) en el sr original de entrada.
"""
import functools
import numpy as np

STEM_NAMES = ["drums", "bass", "other", "vocals"]  # orden nativo de htdemucs

@functools.lru_cache(maxsize=4)
def _get_model(model_name="htdemucs_ft"):
    from demucs_infer.pretrained import get_model
    model = get_model(model_name)
    model.eval()
    return model


def _get_device(device):
    import torch
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resample(audio_2d: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio_2d
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr_in), int(sr_out))
    up, down = sr_out // g, sr_in // g
    return resample_poly(audio_2d, up, down, axis=-1).astype(np.float32)


def separate_stems(audio: np.ndarray, sr: int, progress_cb=None,
                    model_name: str = "htdemucs_ft", device: str = None) -> dict:
    """
    audio: np.ndarray mono (samples,) o estéreo (2, samples) o (samples, 2).
    progress_cb: callable(pct: float, stage: str) -> None, pct en [0,100].
    Devuelve dict {stem_name: np.ndarray (channels, samples)} en sr original.
    """
    import torch
    from demucs_infer.apply import apply_model

    def _report(pct, stage):
        if progress_cb:
            try:
                progress_cb(min(99.0, max(0.0, pct)), stage)
            except Exception:
                pass

    _report(1, "Cargando modelo Demucs…")
    model = _get_model(model_name)
    dev = _get_device(device)
    model.to(dev)
    model_sr = model.samplerate  # típicamente 44100

    # Normalizar shape a (channels, samples), forzar estéreo (demucs espera 2ch)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio_2d = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2:
        # BUGFIX: validar forma antes de transpose
        # Si shape[0] es muy grande y shape[1] es chico, probablemente sea
        # (samples, channels) que necesita transpose.
        # Si ambos son razonables, asumir (channels, samples).
        if audio.shape[0] > 2 and audio.shape[1] <= 2:
            # Probablemente (samples, channels) — transpose a (channels, samples)
            audio_2d = audio.T
        else:
            # Asumir (channels, samples); si >2 canales, tomar solo los primeros 2
            if audio.shape[0] > 2:
                _report(2, f"Audio de {audio.shape[0]} canales; tomando solo los 2 primeros…")
                audio_2d = audio[:2, :]
            else:
                audio_2d = audio
        
        # Garantizar que sea estéreo (Demucs espera exactamente 2 canales)
        if audio_2d.shape[0] == 1:
            audio_2d = np.concatenate([audio_2d, audio_2d], axis=0)
    else:
        raise ValueError("audio debe ser mono o estéreo")

    _report(3, "Remuestreando…")
    audio_model_sr = _resample(audio_2d, sr, model_sr)

    wav = torch.from_numpy(audio_model_sr)
    ref_mean = wav.mean()
    ref_std = wav.std() + 1e-8
    wav_norm = (wav - ref_mean) / ref_std

    _report(5, "Separando stems (htdemucs_ft: puede tardar varios minutos)…")
    # NOTA: demucs==4.0.1 (última versión publicada en PyPI) NO soporta un
    # parámetro `callback` en apply_model — esa firma es de una versión más
    # nueva que todavía no está en PyPI. Por eso acá no hay progreso granular
    # durante la separación en sí: pasamos de 5% a 90% de una sola vez. Si en
    # el futuro actualizan demucs a una versión con callback, se puede volver
    # a conectar progress_cb en cada segmento.
    with torch.no_grad():
        out = apply_model(
            model, wav_norm[None], device=dev, progress=False,
            split=True, overlap=0.25, shifts=1,
        )[0]
    
    # BUGFIX: validar forma del output para evitar artefactos silenciosos
    if out.ndim != 3:
        raise ValueError(f"Forma inesperada de Demucs: {out.ndim}D (esperado 3D)")
    if out.shape[0] != len(model.sources):
        raise ValueError(f"Stems inesperados: {out.shape[0]} (esperado {len(model.sources)})")
    
    _report(90, "Separación completa, reconstruyendo stems…")

    out = out * ref_std + ref_mean
    out_np = out.cpu().numpy().astype(np.float32)  # (n_stems, channels, samples)

    # Evita deadlocks OpenMP entre los threads internos de torch (usados por
    # Demucs) y los de scipy (usados después en stem_analysis.py, en el mismo
    # proceso). Es un problema conocido cuando ambas libs corren en CPU y
    # comparten runtime OpenMP — bajar torch a 1 thread ni bien terminamos de
    # usarlo evita que se cuelgue el paso de análisis que viene justo después.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    _report(97, "Remuestreando stems al sample rate original…")
    stems = {}
    for i, name in enumerate(model.sources):
        stem_audio = _resample(out_np[i], model_sr, sr)
        stems[name] = stem_audio

    # Reordenar a orden canónico si el modelo trae otro orden de sources
    stems = {name: stems[name] for name in STEM_NAMES if name in stems}

    _report(100, "Separación completa")
    return stems


# ── Modo alternativo: solo voz/instrumental con Mel-Band / BS-RoFormer ──────
#
# Vía el paquete `audio-separator` (pip install "audio-separator[cpu]" o
# "[gpu]" si hay CUDA), que es el mismo wrapper que usa Ultimate Vocal
# Remover (UVR) por debajo. NO reemplaza a Demucs (que sigue siendo el modo
# default de 4 stems) — es un modo extra para cuando lo que importa es la
# MEJOR calidad posible de separación voz/instrumental, sacrificando los
# stems de batería/bajo por separado.
#
# OJO — riesgo de confiabilidad conocido: los checkpoints entrenados de
# BS-RoFormer/Mel-RoFormer se distribuyen desde cuentas personales de
# HuggingFace de la comunidad (no un registro oficial estable). Ya pasó que
# una de esas cuentas (jarredou) se borró y tiró abajo el checkpoint default
# de varios wrappers. Por eso ACÁ ADENTRO se atrapa el error de carga/descarga
# explícitamente y se relanza con un mensaje claro, en vez de dejar que
# revient con un traceback críptico de audio_separator/onnxruntime.
ROFORMER_MODEL_DEFAULT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # BS-Roformer Viperx — mejor SDR vocals/instrumental del registro de audio-separator

_ROFORMER_SEPARATOR_CACHE = {}


def _get_roformer_separator(model_filename: str = ROFORMER_MODEL_DEFAULT):
    if model_filename in _ROFORMER_SEPARATOR_CACHE:
        return _ROFORMER_SEPARATOR_CACHE[model_filename]
    import os
    from audio_separator.separator import Separator

    model_dir = os.path.join(os.path.expanduser("~"), ".cache", "audio-separator-models")
    separator = Separator(
        model_file_dir=model_dir,
        output_format="WAV",
        normalization_threshold=0.95,
    )
    separator.load_model(model_filename=model_filename)
    _ROFORMER_SEPARATOR_CACHE[model_filename] = separator
    return separator


def separate_vocals_hq(audio: np.ndarray, sr: int, progress_cb=None,
                        model_filename: str = ROFORMER_MODEL_DEFAULT) -> dict:
    """
    Separación de alta calidad SOLO voz vs. instrumental, usando un modelo
    Mel-Band/BS-RoFormer (arquitectura transformer, estado del arte actual
    para aislar voz de una mezcla ya masterizada — mejor que Demucs en este
    caso puntual, aunque Demucs le gana separando batería/bajo por separado).

    audio: np.ndarray mono (samples,) o estéreo (2, samples) o (samples, 2).
    progress_cb: callable(pct: float, stage: str) -> None, pct en [0,100].
    Devuelve dict {"vocals": ndarray, "instrumental": ndarray} (channels, samples)
    en el sr original.

    Lanza RuntimeError con mensaje claro (en vez de dejar propagar el
    traceback interno) si el modelo no se puede cargar o descargar — ver nota
    de confiabilidad más arriba.
    """
    import os as _os
    import tempfile
    import soundfile as sf

    def _report(pct, stage):
        if progress_cb:
            try:
                progress_cb(min(99.0, max(0.0, pct)), stage)
            except Exception:
                pass

    _report(1, "Cargando modelo Roformer (vocals/instrumental)…")
    try:
        separator = _get_roformer_separator(model_filename)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo Roformer '{model_filename}'. Los "
            f"checkpoints de este ecosistema se alojan en cuentas personales "
            f"de HuggingFace/GitHub de la comunidad y a veces desaparecen sin "
            f"aviso — no es necesariamente un problema de este servidor. "
            f"Probá de nuevo más tarde o usá el modo Demucs (4 stems) mientras "
            f"tanto. Detalle técnico: {e}"
        ) from e

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio_2d = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2:
        audio_2d = audio if audio.shape[0] <= 2 else audio.T
        if audio_2d.shape[0] == 1:
            audio_2d = np.concatenate([audio_2d, audio_2d], axis=0)
    else:
        raise ValueError("audio debe ser mono o estéreo")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = _os.path.join(tmpdir, "input.wav")
        _report(5, "Escribiendo audio temporal…")
        sf.write(in_path, audio_2d.T, sr, subtype="PCM_24")

        separator.output_dir = tmpdir
        _report(10, "Separando voz/instrumental (Roformer, puede tardar)…")
        try:
            output_files = separator.separate(
                in_path, {"Vocals": "vocals", "Instrumental": "instrumental"}
            )
        except Exception as e:
            raise RuntimeError(
                f"Falló la separación con el modelo Roformer '{model_filename}' "
                f"(checkpoint corrupto o descarga incompleta). Probá de nuevo o "
                f"usá el modo Demucs mientras tanto. Detalle técnico: {e}"
            ) from e

        _report(90, "Leyendo stems separados…")
        stems = {}
        for path in output_files:
            fname = _os.path.basename(path).lower()
            if "vocals" in fname:
                name = "vocals"
            elif "instrumental" in fname:
                name = "instrumental"
            else:
                continue
            data, file_sr = sf.read(path, dtype="float32", always_2d=True)
            data = data.T  # (channels, samples)
            if file_sr != sr:
                data = _resample(data, file_sr, sr)
            stems[name] = data

    # Mismo motivo que en separate_stems(): evitar deadlock OpenMP entre
    # torch (usado por audio_separator/onnxruntime internamente) y los
    # threads de scipy que se usan justo después en stem_analysis.py.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    _report(100, "Separación completa")
    return stems
