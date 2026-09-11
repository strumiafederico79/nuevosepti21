import os

UPLOAD_DIR     = os.getenv("UPLOAD_DIR",     "uploads")
PROCESSED_DIR  = os.getenv("PROCESSED_DIR",  "processed")
STEMS_DIR      = os.getenv("STEMS_DIR",      "processed_stems")

# BUGFIX: parsear variables de entorno con try/except para no explotar
# si llegan valores inválidos (ej: PROCESSED_TTL="invalid")
try:
    PROCESSED_TTL  = int(os.getenv("PROCESSED_TTL", "3600"))
except ValueError:
    PROCESSED_TTL = 3600

try:
    MAX_FILE_SIZE  = int(os.getenv("MAX_FILE_SIZE", str(300 * 1024 * 1024)))
except ValueError:
    MAX_FILE_SIZE = 300 * 1024 * 1024

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}

# Carpeta de referencias permanentes — el usuario pone sus tracks de referencia
# acá y el servidor los indexa automáticamente. Se puede cambiar por env var.
REFERENCE_LIBRARY_DIR = os.getenv("REFERENCE_LIBRARY_DIR", "reference_library")

# Librería persistente de stems del mixer — permite reutilizar pistas individuales
# sin volver a subirlas en cada sesión.
STEM_LIBRARY_DIR = os.getenv("STEM_LIBRARY_DIR", "stem_library")

PREVIEW_DIR = os.getenv("PREVIEW_DIR", "preview_sources")
PREVIEW_TTL = int(os.getenv("PREVIEW_TTL", "3600"))
