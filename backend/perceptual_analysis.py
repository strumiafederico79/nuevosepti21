# ============================================================
# perceptual_analysis.py — Sistema de "oídos" para Laia
# Convierte métricas técnicas en descripción perceptual real
# ============================================================

from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Magic numbers extraídos como constantes nombradas ───────────────────────
# Thresholds de claridad
CLARITY_DARK_CENTROID_HZ = 2000
CLARITY_HARSH_FLATNESS = 0.3
CLARITY_HARSH_CENTROID_HZ = 6000
AIR_ENERGY_VERY_LOW_DB = -30

# Thresholds de dinámica
LOOSE_PLR_DB = 14
LOOSE_LRA_LU = 7
TIGHT_PLR_DB = 6
TIGHT_LRA_LU = 3

# Thresholds tonales
TONAL_DARK_CENTROID_HZ = 2000
TONAL_BRIGHT_CENTROID_HZ = 5000

# Thresholds estéreo
STEREO_MONO_CORRELATION = 0.9
STEREO_SEPARATED_CORRELATION = 0.5
PHASE_ISSUE_DB = -6

# Thresholds de definición instrumental
CLEAR_FLATNESS = 0.7
PRESENCE_CLEAR_DB = 5
BLENDED_FLATNESS = 0.3
OVERLY_DEFINED_PRESENCE_DB = 8

# Thresholds de presencia
IN_YOUR_FACE_CENTROID_HZ = 5500
IN_YOUR_FACE_PRESENCE_DB = 6
DISTANT_CENTROID_HZ = 2500
DISTANT_LUFS = -18
DISTANT_AIR_DB = -25

# Thresholds de fatiga
FATIGUE_CENTROID_HZ = 6000
FATIGUE_PLR_DB = 4
FATIGUE_PRESENCE_DB = 10
FATIGUE_LUFS = -6

# Thresholds de cohesión
OVER_COMPRESSED_PLR_DB = 3
DISCONNECTED_CORRELATION = 0.6

# Thresholds de balance de frecuencias
BASS_HEAVY_SUBS_DB = 5
BASS_HEAVY_MIDS_DB = -5
TREBLE_HEAVY_AIR_DB = 5
TREBLE_HEAVY_SUBS_DB = -10

# Thresholds de headroom
CRAMPED_TRUE_PEAK_DB = -0.5
CRAMPED_CLIPPING_RATIO = 0.1
EMPTY_TRUE_PEAK_DB = -6
EMPTY_LUFS = -18

# Thresholds de clasificación de género (get_genre_from_perceptual)
TRAP_LUFS = -7
TRAP_PLR = 8
BALADA_LUFS = -13
BALADA_PLR = 12
BALADA_LRA = 6
ROCK_LUFS_MIN = -10
ROCK_LUFS_MAX = -8
ROCK_PLR_MIN = 8
ROCK_PLR_MAX = 14
ROCK_TRANSIENT_DENSITY = 0.6
PODCAST_PLR = 16
PODCAST_CENTROID_HZ = 3000
AMBIENT_PLR = 12
POP_LUFS_MIN = -10
POP_LUFS_MAX = -7
POP_PLR_MIN = 6
POP_PLR_MAX = 12
EDM_LUFS = -6
EDM_PLR = 6


class PerceptualProfile:
    """Describe cómo suena la mezcla en términos humanos, no técnicos"""
    
    def __init__(self):
        self.clarity = "unknown"  # "muddy" | "balanced" | "harsh"
        self.dynamic_feel = "unknown"  # "loose" | "balanced" | "tight"
        self.tonal_balance = "unknown"  # "dark" | "balanced" | "bright"
        self.stereo_coherence = "unknown"  # "mono" | "coherent" | "separated" | "phase_issues"
        self.instrumental_definition = "unknown"  # "blended" | "clear" | "overly_defined"
        self.presence_feel = "unknown"  # "distant" | "present" | "in_your_face"
        self.fatigue_risk = 0.0  # 0-1, riesgo de listening fatigue
        self.mix_cohesion = "unknown"  # "disconnected" | "glued" | "over_compressed"
        self.frequency_balance = "unknown"  # "bass_heavy" | "balanced" | "treble_heavy"
        self.headroom_feel = "unknown"  # "cramped" | "comfortable" | "empty"
        
    def to_dict(self) -> Dict:
        return {
            "clarity": self.clarity,
            "dynamic_feel": self.dynamic_feel,
            "tonal_balance": self.tonal_balance,
            "stereo_coherence": self.stereo_coherence,
            "instrumental_definition": self.instrumental_definition,
            "presence_feel": self.presence_feel,
            "fatigue_risk": round(self.fatigue_risk, 2),
            "mix_cohesion": self.mix_cohesion,
            "frequency_balance": self.frequency_balance,
            "headroom_feel": self.headroom_feel,
        }
    
    def to_prompt_text(self) -> str:
        """Devuelve descripción legible para el prompt del asistente"""
        return f"""
PERCEPCIÓN AUDITIVA (cómo suena la mezcla):
- Claridad: {self.clarity} (si está "fango" o "cristalino")
- Dinámica: {self.dynamic_feel} (si suena viva o aplastada)
- Balance tonal: {self.tonal_balance} (si es oscura, balanceada o brillante)
- Coherencia estéreo: {self.stereo_coherence} (si los elementos están pegados o flotando)
- Definición instrumental: {self.instrumental_definition} (si se escuchan detalles o todo mezclado)
- Presencia: {self.presence_feel} (si está cercana o lejana)
- Riesgo de fatiga: {int(self.fatigue_risk * 100)}% (agresividad percibida)
- Cohesión de la mezcla: {self.mix_cohesion} (si suena "pegada" o desconectada)
- Balance de frecuencias: {self.frequency_balance} (sub-peso de graves/medios/agudos)
- Headroom: {self.headroom_feel} (si hay espacio o está todo comprimido)
"""


def analyze_perceptual_profile(analysis: Dict) -> PerceptualProfile:
    """
    Convierte métricas técnicas reales en descripción perceptual.
    
    Lee: LUFS, PLR, LRA, spectral_centroid, spectral_flatness, 
         correlation, dynamic_range, clipping_ratio, etc.
    
    Retorna: PerceptualProfile con descripción humana de cómo suena.
    """
    
    profile = PerceptualProfile()
    
    # ── CLARIDAD (muddy vs balanced vs harsh) ────────────────────────────────
    # Muddy: bajos/medios bajos dominan, spectral_flatness baja, poca energía en air
    # Harsh: demasiados agudos, spectral_centroid muy alto, fatigue_risk sube
    #
    # NOTA claves: este dict es el mismo que arma analyze_audio() en
    # mastering.py — usa SUS nombres reales, no los "ideales" (loudness_
    # integrated, spectral_centroid, correlation_global, etc. no existen).
    
    centroid = analysis.get("spectral_centroid_hz", 3000)
    flatness = analysis.get("spectral_flatness", 0.5)
    rolloff = analysis.get("spectral_rolloff_hz", 8000)
    band_energies = analysis.get("spectrum", {})  # {sub_bass|bass|low_mid|mid|upper_mid|presence|air: dB}
    
    # Energía en "aire" (8k-20k)
    air_energy = band_energies.get("air", -20)
    low_mids_energy = band_energies.get("low_mid", -20)
    
    if air_energy < AIR_ENERGY_VERY_LOW_DB and centroid < CLARITY_DARK_CENTROID_HZ:
        # Muy oscuro, bajos dominan
        profile.clarity = "muddy"
    elif flatness < CLARITY_HARSH_FLATNESS or centroid > CLARITY_HARSH_CENTROID_HZ:
        # Picos muy angostos o demasiado aire
        profile.clarity = "harsh"
    else:
        # Balanceado
        profile.clarity = "balanced"
    
    # ── DINÁMICA (loose | balanced | tight) ────────────────────────────────
    # Loose: PLR alto, LRA alto, mucha variación
    # Tight: PLR bajo, LRA bajo, muy comprimido
    
    plr = analysis.get("plr_db", 8)
    lra = analysis.get("lra", 4)
    
    if plr > LOOSE_PLR_DB and lra > LOOSE_LRA_LU:
        profile.dynamic_feel = "loose"
    elif plr < TIGHT_PLR_DB and lra < TIGHT_LRA_LU:
        profile.dynamic_feel = "tight"
    else:
        profile.dynamic_feel = "balanced"
    
    # ── BALANCE TONAL (dark | balanced | bright) ────────────────────────────
    if centroid < TONAL_DARK_CENTROID_HZ:
        profile.tonal_balance = "dark"
    elif centroid > TONAL_BRIGHT_CENTROID_HZ:
        profile.tonal_balance = "bright"
    else:
        profile.tonal_balance = "balanced"
    
    # ── COHERENCIA ESTÉREO (mono | coherent | separated | phase_issues) ─────
    correlation_global = analysis.get("stereo_correlation", 0.8)
    correlation_by_band = analysis.get("band_stereo_correlation", {})
    mono_compatibility = analysis.get("mono_compatibility_db", -3)
    
    # Si correlación es baja pero mono_compatibility es OK → separated (normal)
    # Si mono_compatibility es muy negativa → phase issues
    if correlation_global > STEREO_MONO_CORRELATION:
        profile.stereo_coherence = "mono"  # O muy poco estéreo
    elif mono_compatibility < PHASE_ISSUE_DB:
        profile.stereo_coherence = "phase_issues"
    elif correlation_global < STEREO_SEPARATED_CORRELATION:
        profile.stereo_coherence = "separated"
    else:
        profile.stereo_coherence = "coherent"
    
    # ── DEFINICIÓN INSTRUMENTAL (blended | clear | overly_defined) ──────────
    # Clear: spectral_flatness alta, energía distribuida, cada instrumento visible
    # Blended: flatness baja, poca definición, todo emborronado
    # Overly defined: mucha presencia en 2-4k, sibilancia, fatiga
    
    presence_band = band_energies.get("presence", -20)  # 2k-4k
    
    if flatness > CLEAR_FLATNESS and presence_band < PRESENCE_CLEAR_DB:
        profile.instrumental_definition = "clear"
    elif flatness < BLENDED_FLATNESS:
        profile.instrumental_definition = "blended"
    elif presence_band > OVERLY_DEFINED_PRESENCE_DB:
        profile.instrumental_definition = "overly_defined"
    else:
        profile.instrumental_definition = "clear"
    
    # ── PRESENCIA (distant | present | in_your_face) ────────────────────────
    # Presente: centroid 3-5k, energía en medios, LUFS > -12
    # Distant: centroid bajo, energía en subs, mucho reverb perceived
    # In your face: centroid muy alto, sibilancia, presencia agresiva
    
    lufs = analysis.get("lufs", -14)
    
    if centroid > IN_YOUR_FACE_CENTROID_HZ and presence_band > IN_YOUR_FACE_PRESENCE_DB:
        profile.presence_feel = "in_your_face"
    elif centroid < DISTANT_CENTROID_HZ or (lufs < DISTANT_LUFS and air_energy < DISTANT_AIR_DB):
        profile.presence_feel = "distant"
    else:
        profile.presence_feel = "present"
    
    # ── RIESGO DE FATIGA (0-1) ─────────────────────────────────────────────
    # Sube con: spectral_centroid muy alto, dinámica muy comprimida, sibilancia
    # Baja con: espacio, aire, dinámica natural
    
    fatigue = 0.0
    if centroid > FATIGUE_CENTROID_HZ:
        fatigue += 0.3
    if plr < FATIGUE_PLR_DB:
        fatigue += 0.2
    if presence_band > FATIGUE_PRESENCE_DB:
        fatigue += 0.25
    if lufs > FATIGUE_LUFS:
        fatigue += 0.15
    
    profile.fatigue_risk = min(1.0, fatigue)
    
    # ── COHESIÓN DE MEZCLA (disconnected | glued | over_compressed) ─────────
    # Glued: correlation alta, comp uniforme, LUFS constante
    # Over-compressed: PLR muy bajo, nada de dinámica, suena "plano"
    # Disconnected: elementos suenan por separado, falta glue
    
    if plr < OVER_COMPRESSED_PLR_DB:
        profile.mix_cohesion = "over_compressed"
    elif correlation_global < DISCONNECTED_CORRELATION:
        profile.mix_cohesion = "disconnected"
    else:
        profile.mix_cohesion = "glued"
    
    # ── BALANCE DE FRECUENCIAS ─────────────────────────────────────────────
    subs_energy = band_energies.get("sub_bass", -20)
    mids_energy = band_energies.get("mid", -20)
    air_energy_val = band_energies.get("air", -20)
    
    if subs_energy > BASS_HEAVY_SUBS_DB and mids_energy < BASS_HEAVY_MIDS_DB:
        profile.frequency_balance = "bass_heavy"
    elif air_energy_val > TREBLE_HEAVY_AIR_DB and subs_energy < TREBLE_HEAVY_SUBS_DB:
        profile.frequency_balance = "treble_heavy"
    else:
        profile.frequency_balance = "balanced"
    
    # ── HEADROOM (cramped | comfortable | empty) ───────────────────────────
    true_peak = analysis.get("true_peak_db", -3)
    clipping_ratio = analysis.get("clipping_ratio", 0.0)
    
    if true_peak > CRAMPED_TRUE_PEAK_DB or clipping_ratio > CRAMPED_CLIPPING_RATIO:
        profile.headroom_feel = "cramped"
    elif true_peak < EMPTY_TRUE_PEAK_DB or lufs < EMPTY_LUFS:
        profile.headroom_feel = "empty"
    else:
        profile.headroom_feel = "comfortable"
    
    return profile


def get_genre_from_perceptual(profile: PerceptualProfile, 
                               analysis: Dict) -> Tuple[str, float]:
    """
    Detecta probable género a partir de perfil perceptual + análisis.
    
    Retorna: (genre, confidence: 0-1)
    """
    
    confidence = 0.0
    genre_scores = {}
    
    lufs = analysis.get("lufs", -14)
    plr = analysis.get("plr_db", 8)
    lra = analysis.get("lra", 4)
    transient_density = analysis.get("transient_density", 0.5)
    centroid = analysis.get("spectral_centroid_hz", 3000)
    
    # TRAP: loud, bass-heavy, tight dynamic, defined (no reverb)
    if lufs > TRAP_LUFS and profile.frequency_balance == "bass_heavy" and plr < TRAP_PLR:
        genre_scores["trap"] = 0.85
    
    # BALADA: quiet, loose dynamic, presente emotivamente, reverb
    if lufs < BALADA_LUFS and plr > BALADA_PLR and lra > BALADA_LRA and profile.dynamic_feel == "loose":
        genre_scores["balada"] = 0.80
    
    # ROCK: comp media, air presente, transientes claros, dinámico
    if ROCK_LUFS_MIN < lufs < ROCK_LUFS_MAX and ROCK_PLR_MIN < plr < ROCK_PLR_MAX and transient_density > ROCK_TRANSIENT_DENSITY:
        genre_scores["rock"] = 0.75
    
    # PODCAST/VOZ: extremo PLR, medios claros, mínimo reverb
    if plr > PODCAST_PLR and centroid < PODCAST_CENTROID_HZ and profile.clarity == "clear":
        genre_scores["podcast"] = 0.80
    
    # AMBIENT: dinámico, reverb, oscuro, presencia baja
    if plr > AMBIENT_PLR and profile.presence_feel == "distant" and profile.tonal_balance == "dark":
        genre_scores["ambient"] = 0.75
    
    # POP: balanced todo, presente, comp suave, aire moderado
    if POP_LUFS_MIN < lufs < POP_LUFS_MAX and POP_PLR_MIN < plr < POP_PLR_MAX and profile.presence_feel == "present":
        genre_scores["pop"] = 0.70
    
    # EDM/DANCE: very loud, tight, bass, presencia agresiva
    if lufs > EDM_LUFS and plr < EDM_PLR and profile.frequency_balance == "bass_heavy":
        genre_scores["edm"] = 0.75
    
    if not genre_scores:
        return "mixed", 0.3
    
    best_genre = max(genre_scores.items(), key=lambda x: x[1])
    return best_genre[0], best_genre[1]


def get_perceptual_diagnosis(profile: PerceptualProfile, 
                              analysis: Dict) -> str:
    """
    Genera un "diagnóstico" en lenguaje natural de qué se escucha.
    
    Ejemplo: "Mezcla natural pero con bajos algo comprimidos; 
             aire presente pero medios pegados. Presencia correcta, 
             buen headroom."
    """
    
    parts = []
    
    # Diagnóstico general
    if profile.mix_cohesion == "glued":
        parts.append("Mezcla bien pegada")
    elif profile.mix_cohesion == "over_compressed":
        parts.append("Mezcla muy comprimida (falta groove)")
    else:
        parts.append("Mezcla con elementos desconectados")
    
    # Dinámica
    if profile.dynamic_feel == "tight":
        parts.append(", dinámica limitada")
    elif profile.dynamic_feel == "loose":
        parts.append(", dinámica muy abierta")
    
    # Tono
    if profile.tonal_balance == "dark":
        parts.append(", tonalidad oscura")
    elif profile.tonal_balance == "bright":
        parts.append(", tonalidad brillante")
    
    # Estéreo
    if profile.stereo_coherence == "separated":
        parts.append(", estéreo muy abierto")
    elif profile.stereo_coherence == "phase_issues":
        parts.append(", issues de fase en estéreo")
    
    # Presencia
    if profile.presence_feel == "distant":
        parts.append(", suena lejana")
    elif profile.presence_feel == "in_your_face":
        parts.append(", presencia agresiva")
    
    # Fatiga
    if profile.fatigue_risk > 0.7:
        parts.append(f". RIESGO DE FATIGA: {int(profile.fatigue_risk*100)}%")
    elif profile.fatigue_risk > 0.4:
        parts.append(f" — algo agresiva ({int(profile.fatigue_risk*100)}% fatiga)")
    
    # Headroom
    if profile.headroom_feel == "cramped":
        parts.append(". Headroom limitado")
    elif profile.headroom_feel == "empty":
        parts.append(". Mucho headroom (puede subir)")
    
    diagnosis = "".join(parts)
    if not diagnosis:
        diagnosis = "Mezcla con características promedio"
    
    return diagnosis.rstrip(",")


# ── INTEGRACIÓN CON AI_ASSISTANT ────────────────────────────────────────────

def enhance_system_prompt_with_perceptual(system_prompt: str,
                                          analysis: Dict) -> str:
    """
    Agrega sección de análisis perceptual al system prompt de Laia.
    """
    
    profile = analyze_perceptual_profile(analysis)
    genre, genre_confidence = get_genre_from_perceptual(profile, analysis)
    diagnosis = get_perceptual_diagnosis(profile, analysis)
    
    perceptual_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANÁLISIS PERCEPTUAL (LO QUE LAIA "ESCUCHA"):

Género probable: {genre.upper()} (confianza: {int(genre_confidence*100)}%)
Diagnóstico auditivo: {diagnosis}

{profile.to_prompt_text()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCCIONES BASADAS EN LA PERCEPCIÓN:

Tu decisión de FASE 2 (estrategia) debe estar fundamentada no solo en números,
sino también en CÓMO SUENA esta mezcla. El diagnóstico arriba describe el 
"carácter" del track tal como lo escucharía un ingeniero de mastering.

Por ejemplo:
- Si suena "muddy": correctiva en bajos/bajos-medios ANTES que color.
- Si suena "tight": relaja la compresión, no la aprietes más.
- Si tiene "fatiga_risk" alto: EQ correctiva conservadora, evita sibilancia.
- Si es "{genre}": aplica el criterio específico del género (ver abajo).

CRITERIO ESPECÍFICO POR GÉNERO:
"""
    
    # Agregar guía específica del género detectado
    genre_guidance = {
        "trap": """
TRAP: prioriza graves/medios bajos (energía del sub), comp muy agresiva en banda 
ancha (3:1+), mantén aire definido. No temas sibilancia moderada — es parte del 
carácter. Usa transient shaper para definir kicks/snares.
""",
        "balada": """
BALADA: preserva dinámica natural, comp suave (1.5:1), EQ correctiva mínima si 
es necesaria. El track debe sonar "respirando". Focus en claridad de voz si existe, 
balance estéreo natural (no demasiado ancho).
""",
        "rock": """
ROCK: comp media (2.5:1), air presente, EQ de color para carácter (subida en 3k 
u 8k según carácter). Transientes claros. Usa limiter conservador para picos.
""",
        "podcast": """
PODCAST: NO comprimas la dinámica — es parte del contenido. EQ correctiva en 
1-3k si suena "feo". Normalize LUFS, no dinámicamente. Mínimo reverb/saturación.
""",
        "ambient": """
AMBIENT: preserva dinámica y espacialidad. Comp muy suave o nada. EQ correctiva 
si es necesaria, color minimal. El track debe sentirse "flotante".
""",
        "pop": """
POP: balance entre dinámica y loudness. Comp uniforme (2:1), EQ de color en 
presencia (2-5k) para llevar adelante voz/lead. Estéreo moderado, headroom claro.
""",
        "edm": """
EDM: comp muy agresiva (4:1+), maximize loudness. Prioriza subs/medios. 
Glue compressor importante. Air puede estar comprimido — es estilo.
""",
    }
    
    if genre in genre_guidance:
        perceptual_section += genre_guidance[genre]
    else:
        perceptual_section += "\nGÉNERO MIXTO: aplica criterio híbrido. Prioriza claridad sobre loudness.\n"
    
    return system_prompt + perceptual_section

