"""FastAPI routers for the Audio Mastering API."""

from .ai import create_ai_router
from .analysis import create_analysis_router
from .auth import create_auth_router
from .dashboard import create_dashboard_router
from .info import create_info_router
from .jobs import create_jobs_router
from .library import create_library_router
from .projects import create_projects_router
from .reference_library import create_reference_library_router
from .audio import create_audio_router

__all__ = [
    "create_ai_router",
    "create_analysis_router",
    "create_auth_router",
    "create_dashboard_router",
    "create_info_router",
    "create_jobs_router",
    "create_library_router",
    "create_projects_router",
    "create_reference_library_router",
    "create_audio_router",
]

from .mastering import create_router as create_mastering_router
from .mixer import create_router as create_mixer_router
from .stems import create_router as create_stems_router
from .streaming import create_router as create_streaming_router
from .preview import create_preview_router
