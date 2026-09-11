"""Memoria de cancha para el motor de diagnóstico de SorpresaGPT.

La biblioteca no reemplaza las mediciones DSP: agrega contexto operativo,
orden de comprobación y lenguaje de troubleshooting de audio en vivo.
Cada caso representa un patrón reusable que puede ampliarse sin tocar el
núcleo de análisis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class FieldCase:
    case_id: str
    title: str
    symptom: str
    confidence: float
    why: str
    route: tuple[str, ...]
    first_check: str
    do_not_touch_first: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeMatch:
    case: FieldCase
    evidence: tuple[str, ...]
    priority: int



def _f(name: str, default: float = 0.0):
    return lambda a: float(a.get(name, default) or default)


_CASES = (
    FieldCase(
        case_id="no_signal",
        title="No hay señal: seguir la ruta",
        symptom="La señal útil está prácticamente ausente.",
        confidence=0.97,
        why="Cuando no hay energía útil, el problema suele estar antes del procesamiento.",
        route=("fuente", "cableado", "patch", "entrada", "preamp/gain", "mute", "routing", "bus/salida"),
        first_check="Confirmá señal en cada punto de la cadena, empezando por la fuente y avanzando hacia la salida.",
        do_not_touch_first=("EQ", "compresor", "limiter", "master bus"),
    ),
    FieldCase(
        case_id="clipping",
        title="Saturación: localizar la etapa que satura",
        symptom="Hay clipping real o true peak fuera de margen.",
        confidence=0.98,
        why="Corregir la última perilla no garantiza que una etapa anterior no siga saturando.",
        route=("fuente", "ganancia de entrada", "preamp", "procesamiento", "bus", "salida/conversión"),
        first_check="Encontrá la primera etapa de la cadena donde aparece el clip y corregí la ganancia allí.",
        do_not_touch_first=("EQ para bajar picos", "estéreo", "normalización"),
    ),
    FieldCase(
        case_id="phase",
        title="Fase: comparar antes de ecualizar",
        symptom="La suma mono pierde demasiado nivel y/o la correlación es baja.",
        confidence=0.95,
        why="Una cancelación de fase puede parecer un problema tonal y empeorar si se intenta arreglar con EQ.",
        route=("L/R", "polaridad", "alineación temporal", "microfonía", "procesamiento estéreo"),
        first_check="Escuchá en mono y compará L/R; luego verificá polaridad y posibles retrasos.",
        do_not_touch_first=("EQ para compensar la pérdida", "compresión", "más ancho estéreo"),
    ),
    FieldCase(
        case_id="dc_offset",
        title="DC offset: higiene antes del procesamiento",
        symptom="Hay desplazamiento de continua apreciable.",
        confidence=0.90,
        why="El DC offset consume headroom y puede confundir el diagnóstico de nivel.",
        route=("fuente", "interfaz", "conversión", "entrada", "archivo"),
        first_check="Aislá qué etapa introduce el desplazamiento y verificá la señal antes de continuar.",
        do_not_touch_first=("limiter", "loudness maximizer", "EQ quirúrgico"),
    ),
    FieldCase(
        case_id="overcompression",
        title="Dinámica cerrada: buscar dónde se comprimió",
        symptom="El contraste pico/RMS es muy bajo.",
        confidence=0.82,
        why="Agregar más compresión o limiting puede esconder el síntoma y agravar la pérdida de dinámica.",
        route=("fuente", "preamp", "compresión por canal", "bus", "master", "limiter"),
        first_check="Hacé bypass por etapas y escuchá dónde empieza a cerrarse la envolvente.",
        do_not_touch_first=("más compresión", "más limiting", "normalización de loudness"),
    ),
    FieldCase(
        case_id="stereo_instability",
        title="Estéreo inestable: aislar la fuente",
        symptom="La correlación global es baja sin una caída mono extrema.",
        confidence=0.76,
        why="Una sola fuente o efecto puede estar abriendo demasiado el campo y volverlo inestable.",
        route=("fuentes estéreo", "send/return", "delays", "chorus", "wideners", "bus"),
        first_check="Aislá elementos estéreo uno por uno y compará con mono.",
        do_not_touch_first=("más widening", "EQ global", "limiter"),
    ),
)


def match_knowledge(analysis: dict) -> list[KnowledgeMatch]:
    """Devuelve coincidencias ordenadas por prioridad operativa."""
    rms = _f("rms_db", -99)(analysis)
    peak = _f("peak_db", -99)(analysis)
    true_peak = _f("true_peak_db", -99)(analysis)
    clipping = _f("clipping_ratio")(analysis)
    silence = _f("silence_ratio")(analysis)
    dc = abs(_f("dc_offset")(analysis))
    mono = _f("mono_compatibility_db")(analysis)
    corr = _f("stereo_correlation", 1.0)(analysis)
    dyn = _f("dynamic_range_db", 99)(analysis)

    matches: list[KnowledgeMatch] = []

    if (rms < -60 and peak < -50) or silence >= 0.90:
        matches.append(KnowledgeMatch(_CASES[0], (
            f"rms_db={rms}", f"peak_db={peak}", f"silence_ratio={silence}"), 1))
    if clipping > 0.0005 or true_peak >= 0:
        matches.append(KnowledgeMatch(_CASES[1], (
            f"clipping_ratio={clipping}", f"true_peak_db={true_peak}"), 1))
    if mono < -6:
        matches.append(KnowledgeMatch(_CASES[2], (f"mono_compatibility_db={mono}",), 2))
    if dc > 0.01:
        matches.append(KnowledgeMatch(_CASES[3], (f"dc_offset={dc}",), 2))
    if dyn < 4 and not ((rms < -60 and peak < -50) or silence >= 0.90):
        matches.append(KnowledgeMatch(_CASES[4], (f"dynamic_range_db={dyn}",), 3))
    if corr < 0.5 and mono >= -6:
        matches.append(KnowledgeMatch(_CASES[5], (f"stereo_correlation={corr}",), 4))

    return sorted(matches, key=lambda m: (m.priority, -m.case.confidence))


def case_to_dict(match: KnowledgeMatch) -> dict:
    case = match.case
    return {
        "case_id": case.case_id,
        "title": case.title,
        "symptom": case.symptom,
        "confidence": round(case.confidence, 2),
        "why": case.why,
        "route": list(case.route),
        "first_check": case.first_check,
        "do_not_touch_first": list(case.do_not_touch_first),
        "evidence": list(match.evidence),
        "priority": match.priority,
    }


def memory_summary(analysis: dict) -> dict:
    """Resumen público de la memoria de cancha para una respuesta API."""
    matches = match_knowledge(analysis)
    primary = case_to_dict(matches[0]) if matches else None
    return {
        "memory_version": "0.1",
        "source": "memoria_de_cancha",
        "matched_cases": [case_to_dict(m) for m in matches],
        "primary_case": primary,
        "principle": "diagnosticar la cadena antes de tocar procesamiento",
    }
