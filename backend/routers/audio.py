"""Compatibility facade for the audio routers.

The audio domain is split into mastering, mixer, stems and streaming modules.
`create_audio_router` keeps the original public import for callers that still
expect a single combined router.
"""
from __future__ import annotations
from fastapi import APIRouter
from .mastering import create_router as create_mastering_router
from .mixer import create_router as create_mixer_router
from .stems import create_router as create_stems_router
from .streaming import create_router as create_streaming_router

def create_audio_router(**dependencies):
    router = APIRouter()
    router.include_router(create_stems_router(**dependencies))
    router.include_router(create_mastering_router(**dependencies))
    router.include_router(create_mixer_router(**dependencies))
    router.include_router(create_streaming_router(**dependencies))
    return router
