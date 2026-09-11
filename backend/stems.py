"""Compatibility facade for the stems router.

The active implementation lives in ``backend.routers.stems``. Keeping this
module as a facade avoids a second, stale endpoint implementation with its own
dependency graph.
"""
from __future__ import annotations

from .routers.stems import create_router

__all__ = ["create_router"]
