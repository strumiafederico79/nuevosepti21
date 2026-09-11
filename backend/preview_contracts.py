from __future__ import annotations

import inspect
import math
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator


SOURCE_ID_PATTERN = re.compile(r"^[a-f0-9]{16,128}$")


def _is_valid_source_id(value: str) -> bool:
    return bool(SOURCE_ID_PATTERN.fullmatch(value or ""))


def _annotation_from_default(default: Any) -> Any:
    if default is None:
        return Optional[str]
    if isinstance(default, bool):
        return bool
    if isinstance(default, int) and not isinstance(default, bool):
        return int
    if isinstance(default, float):
        return float
    if isinstance(default, str):
        return str
    return Any


def _build_preview_params_model() -> type[BaseModel]:
    """Infer PreviewParams from the process_audio signature.

    The DSP stack is imported lazily here so that simply importing
    ``preview_contracts`` (e.g. from the preview router) does not pull
    SciPy, Numba or Torch into the parent process. They are loaded the
    first time the model is built, which happens only inside the render
    subprocess, never in the FastAPI main loop."""
    try:
        from backend.mastering import process_audio
    except ImportError:  # pragma: no cover - direct uvicorn app:app
        from mastering import process_audio

    fields: dict[str, tuple[Any, Any]] = {}
    sig = inspect.signature(process_audio)
    excluded = {"input_path", "progress_cb", "preview_seconds"}
    for name, parameter in sig.parameters.items():
        if name in excluded:
            continue
        default = parameter.default
        if default is inspect.Parameter.empty:
            annotation = parameter.annotation if parameter.annotation is not inspect.Parameter.empty else Any
            fields[name] = (annotation, ...)
        else:
            fields[name] = (_annotation_from_default(default), default)

    class _Base(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)

        @field_validator("*", mode="after")
        @classmethod
        def validate_values(cls, value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Los parámetros numéricos deben ser finitos")
            return value

    return create_model(
        "PreviewParams",
        __base__=_Base,
        __module__=__name__,
        **fields,
    )


# Construimos el modelo en import-time. El import perezoso dentro de la
# factoría mantiene la carga del stack DSP fuera de FastAPI.
PreviewParams = _build_preview_params_model()


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_source_id: str = Field(min_length=16, max_length=128)
    preview_duration_sec: int = Field(default=25, ge=25, le=25)
    params: PreviewParams

    @field_validator("preview_source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not _is_valid_source_id(value):
            raise ValueError("preview_source_id inválido")
        return value


class PreviewSourceResponse(BaseModel):
    source_id: str
    duration_sec: float = Field(gt=0, le=25)
    source_sha256: str = Field(min_length=64, max_length=64)
