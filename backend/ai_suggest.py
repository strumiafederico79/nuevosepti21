# ai_suggest.py

import os
import json
import re
import librosa
import numpy as np
import pyloudnorm as pyln
import google.generativeai as genai
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

# ─── Configuración de Gemini ──────────────────────────────────────────────
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("GEMINI_API_KEY no configurada en variables de entorno. Gemini AI suggestions desactivadas.")
    model = None
else:
    genai.configure(api_key=_GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-pro")  # o "gemini-1.5-pro"

router = APIRouter(prefix="/ai", tags=["ai"])

# ─── Modelos de datos ──────────────────────────────────────────────────────
class StemSuggestionItem(BaseModel):
    name: str
    library_id: str

class StemSuggestionRequest(BaseModel):
    stems: List[StemSuggestionItem]

class StemSuggestionResponse(BaseModel):
    suggestions: Dict[str, Dict[str, Any]]

# ─── Función de análisis (espectro, LUFS, etc.) ──────────────────────────
def analyze_stem(file_path: str) -> dict:
    """Extrae características numéricas de un archivo de audio."""
    try:
        # BUGFIX: librosa.load puede tardar muchísimo si el archivo es enorme.
        # Limitar a 10 minutos de audio para que el análisis sea rápido.
        y, sr = librosa.load(file_path, sr=44100, mono=True, duration=600.0)  # 10 min max
        
        if len(y) == 0:
            raise ValueError("Audio vacío después de cargar")
        
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(y)
        
        rms_db = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12)
        peak_db = 20 * np.log10(np.max(np.abs(y)) + 1e-12)
        crest = peak_db - rms_db
        
        # Espectro de 32 bandas logarítmicas (20Hz – 20kHz)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        bands = np.logspace(np.log10(20), np.log10(20000), num=33)
        spec = []
        for i in range(len(bands)-1):
            idx = (freqs >= bands[i]) & (freqs < bands[i+1])
            if np.any(idx):
                power = np.mean(S[idx, :]**2)
                spec.append(10 * np.log10(power + 1e-12))
            else:
                spec.append(-80)
        
        return {
            "lufs": round(lufs, 2),
            "rms_db": round(rms_db, 2),
            "peak_db": round(peak_db, 2),
            "crest_factor_db": round(crest, 2),
            "spectrum_32": [round(v, 2) for v in spec],
            "duration_sec": librosa.get_duration(y=y, sr=sr)
        }
    except Exception as e:
        raise ValueError(f"Error analizando stem: {str(e)}")

# ─── Función para obtener la ruta del archivo (¡ADAPTAR A TU BD!) ──────
def get_stem_path(library_id: str) -> str | None:
    """
    Consulta tu base de datos y devuelve la ruta absoluta del archivo.
    EJEMPLO con SQLAlchemy (descomenta y adapta):
    
    from your_database import SessionLocal, StemLibrary
    db = SessionLocal()
    stem = db.query(StemLibrary).filter(StemLibrary.id == library_id).first()
    db.close()
    return stem.file_path if stem else None
    """
    # Usar la misma lógica que el resto del proyecto
    try:
        from library import get_path as _get_path
        from config import STEM_LIBRARY_DIR as _STEM_DIR
        return _get_path(_STEM_DIR, library_id)
    except Exception:
        return None

# ─── ENDPOINT ──────────────────────────────────────────────────────────────
@router.post("/suggest-stems", response_model=StemSuggestionResponse)
async def suggest_stems(request: StemSuggestionRequest):
    if model is None:
        raise HTTPException(503, "Gemini AI no está configurado. Configura GEMINI_API_KEY en variables de entorno.")
    
    # 1. Validar y analizar cada stem
    stems_data = []
    for item in request.stems:
        file_path = get_stem_path(item.library_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(404, f"Stem '{item.name}' (ID {item.library_id}) no encontrado.")
        
        analysis = analyze_stem(file_path)
        stems_data.append({
            "name": item.name,
            **analysis
        })
    
    # 2. Construir el prompt para Gemini
    prompt = f"""
Eres un ingeniero de mezcla experto. Recibirás datos de varios stems (pistas) de una canción.
Cada stem tiene: LUFS, RMS, pico, crest factor y un espectro de 32 bandas (de 20 Hz a 20 kHz en dB).

Tu tarea:
- Comparar TODOS los stems entre sí (NO uses los nombres como regla fija, solo como identificadores).
- Sugerir ajustes de mezcla PARA CADA STEM para que suenen balanceados en conjunto.
- No masterices el bus final. Solo ajustá los parámetros de los stems individuales.
- La decisión debe basarse ÚNICAMENTE en los datos numéricos y en cómo se relacionan entre sí.

Debes devolver ÚNICAMENTE un objeto JSON con la clave "suggestions".
Dentro de "suggestions", un objeto por cada nombre de stem.

El JSON DEBE tener esta estructura EXACTA (sin texto adicional, sin markdown, solo el JSON puro):
{{
  "suggestions": {{
    "nombre_del_stem_1": {{
      "gain_db": float (-18 a 18),
      "hp_cutoff_hz": float (20 a 500),
      "lp_cutoff_hz": float (2000 a 20000),
      "eq_low_gain_db": float (-12 a 12),
      "eq_low_freq": float (40 a 400),
      "eq_low_q": float (0.3 a 4),
      "eq_lomid_gain_db": float (-12 a 12),
      "eq_lomid_freq": float (200 a 1500),
      "eq_lomid_q": float (0.3 a 4),
      "eq_himid_gain_db": float (-12 a 12),
      "eq_himid_freq": float (800 a 8000),
      "eq_himid_q": float (0.3 a 4),
      "eq_high_gain_db": float (-12 a 12),
      "eq_high_freq": float (4000 a 18000),
      "eq_high_q": float (0.3 a 4),
      "comp_enabled": boolean,
      "comp_threshold": float (0.01 a 1),
      "comp_ratio": float (1 a 20),
      "comp_attack_ms": float (0.1 a 100),
      "comp_release_ms": float (10 a 500),
      "comp_makeup_db": float (-6 a 24),
      "pan": float (-1 a 1)
    }}
  }}
}}

Aquí están los datos de los stems (espectro: 32 bandas logarítmicas en dB):
"""
    for s in stems_data:
        prompt += f"\n--- Stem: {s['name']} ---\n"
        prompt += f"LUFS: {s['lufs']}\n"
        prompt += f"RMS: {s['rms_db']} dB\n"
        prompt += f"Pico: {s['peak_db']} dBFS\n"
        prompt += f"Crest Factor: {s['crest_factor_db']} dB\n"
        prompt += f"Espectro (32 bandas): {s['spectrum_32']}\n"

    prompt += """
¡IMPORTANTE! NO uses heurísticas predefinidas (ej. "si es bajo, cortar en 80 Hz").
Cada decisión debe basarse en los datos numéricos de TODOS los stems y en sus diferencias relativas.
Si un stem tiene mucha energía en una zona y otro también, sugiere cortes o atenuaciones para que no se pisen.
Si un stem es más silencioso que el resto, subí su ganancia.
Sé creativo pero realista. Devuelve SOLO el JSON.
"""

    # 3. Llamar a Gemini
    try:
        # Opcional: forzar respuesta JSON (si tu versión lo soporta)
        # response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        raw = response.text
        
        # Limpiar posibles markdown
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        data = json.loads(raw)
        suggestions = data.get("suggestions", {})
        
        if not suggestions and isinstance(data, dict):
            # Si el JSON es directamente el objeto de sugerencias
            if all(isinstance(v, dict) for v in data.values()):
                suggestions = data
            else:
                raise ValueError("El JSON no tiene la estructura esperada.")
    
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Gemini devolvió JSON inválido: {raw[:200]}...")
    except Exception as e:
        raise HTTPException(500, f"Error llamando a Gemini: {str(e)}")

    # 4. Sanitizar / Clampear valores
    for name, params in suggestions.items():
        params["gain_db"] = max(-18, min(18, params.get("gain_db", 0.0)))
        params["hp_cutoff_hz"] = max(20, min(500, params.get("hp_cutoff_hz", 20.0)))
        params["lp_cutoff_hz"] = max(2000, min(20000, params.get("lp_cutoff_hz", 20000.0)))
        params["eq_low_gain_db"] = max(-12, min(12, params.get("eq_low_gain_db", 0.0)))
        params["eq_low_freq"] = max(40, min(400, params.get("eq_low_freq", 100.0)))
        params["eq_low_q"] = max(0.3, min(4, params.get("eq_low_q", 0.8)))
        params["eq_lomid_gain_db"] = max(-12, min(12, params.get("eq_lomid_gain_db", 0.0)))
        params["eq_lomid_freq"] = max(200, min(1500, params.get("eq_lomid_freq", 500.0)))
        params["eq_lomid_q"] = max(0.3, min(4, params.get("eq_lomid_q", 1.0)))
        params["eq_himid_gain_db"] = max(-12, min(12, params.get("eq_himid_gain_db", 0.0)))
        params["eq_himid_freq"] = max(800, min(8000, params.get("eq_himid_freq", 3000.0)))
        params["eq_himid_q"] = max(0.3, min(4, params.get("eq_himid_q", 1.0)))
        params["eq_high_gain_db"] = max(-12, min(12, params.get("eq_high_gain_db", 0.0)))
        params["eq_high_freq"] = max(4000, min(18000, params.get("eq_high_freq", 10000.0)))
        params["eq_high_q"] = max(0.3, min(4, params.get("eq_high_q", 0.8)))
        params["comp_threshold"] = max(0.01, min(1, params.get("comp_threshold", 0.5)))
        params["comp_ratio"] = max(1, min(20, params.get("comp_ratio", 4.0)))
        params["comp_attack_ms"] = max(0.1, min(100, params.get("comp_attack_ms", 10.0)))
        params["comp_release_ms"] = max(10, min(500, params.get("comp_release_ms", 100.0)))
        params["comp_makeup_db"] = max(-6, min(24, params.get("comp_makeup_db", 0.0)))
        params["pan"] = max(-1, min(1, params.get("pan", 0.0)))
        params["comp_enabled"] = bool(params.get("comp_enabled", False))

    return StemSuggestionResponse(suggestions=suggestions)