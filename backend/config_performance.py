"""
Configuración de performance para 8 CPUs / 32GB RAM.
Importar en app.py si se quiere override de los defaults.
"""
import os

# Workers del process pool para apply_mastering_chain
# 6 = 8 CPUs - 2 (reservar para event loops del servidor)
CHAIN_POOL_WORKERS = int(os.getenv("CHAIN_POOL_WORKERS", "6"))

# Cache de audio en RAM — con 32GB podemos cachear muchos tracks
# 512MB de caché = ~50 tracks de 3min a 44100 stereo float32
AUDIO_CACHE_MAX_MB = int(os.getenv("AUDIO_CACHE_MAX_MB", "512"))

# Chunks del preview WS — 1s = respuesta más rápida
WS_PREVIEW_CHUNK_SEC = float(os.getenv("WS_PREVIEW_CHUNK_SEC", "1.0"))

# Prefetch de chunks paralelos
WS_PREFETCH_CHUNKS = int(os.getenv("WS_PREFETCH_CHUNKS", "3"))
