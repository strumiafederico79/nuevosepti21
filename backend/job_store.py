import atexit
import json
import os
import threading
from typing import Any, Callable, Optional


class PersistableJobState(dict):
    """Dict que persiste su estado cuando se modifica."""

    def __init__(self, data: Optional[dict] = None, persist_cb: Optional[Callable[[], None]] = None):
        super().__init__()
        self._persist_cb = persist_cb
        # Each job state is accessed from request/background threads.
        # The original implementation referenced `_state_lock` in __getitem__
        # and related methods but never initialized it, causing:
        # AttributeError: 'PersistableJobState' object has no attribute '_state_lock'
        self._state_lock = threading.RLock()
        if data:
            with self._state_lock:
                super().update(data)

    def _persist(self) -> None:
        if self._persist_cb is not None:
            self._persist_cb()

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._persist()


    def __getitem__(self, key: str) -> Any:
        with self._state_lock:
            return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        with self._state_lock:
            return super().__contains__(key)

    def items(self):
        with self._state_lock:
            return list(super().items())

    def update(self, *args, **kwargs) -> None:
        """Actualiza el dict de forma thread-safe y persiste cambios."""
        # En PersistableJobState no tenemos _state_lock, pero es creado
        # con un persist callback del JobStore que ya tiene sus propios locks.
        # Sin embargo, para máxima seguridad en actualizaciones complejas,
        # es preferible llamar a __setitem__ individualmente que a super().update(),
        # lo cual es mejor para detectar cambios. Pero aquí mantenemos super().update()
        # porque el JobStore que lo llama ya maneja sincronización a nivel superior.
        super().update(*args, **kwargs)
        self._persist()

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        super().__setitem__(key, default)
        self._persist()
        return default

    def pop(self, key: str, default: Any = None) -> Any:
        value = super().pop(key, default)
        self._persist()
        return value

    def popitem(self):
        value = super().popitem()
        self._persist()
        return value

    def clear(self) -> None:
        super().clear()
        self._persist()


class JobStore(dict):
    """Almacén de jobs en memoria con persistencia JSON diferida (debounced).

    La escritura a disco ocurre como máximo una vez cada DEBOUNCE_SEC segundos,
    aunque haya muchos updates de progreso intermedios (antes se escribía en CADA
    update, llegando a 20 escrituras/seg durante un job en curso).
    """

    DEBOUNCE_SEC = 0.5   # segundos de inactividad antes de escribir a disco

    def __init__(self, storage_dir: Optional[str] = None, filename: str = "jobs.json"):
        super().__init__()
        self.storage_dir = storage_dir or os.getenv("JOB_STORE_DIR") or os.path.join(
            os.path.dirname(__file__), "..", "data", "jobs"
        )
        self.filename = filename
        os.makedirs(self.storage_dir, exist_ok=True)
        self._path = os.path.join(self.storage_dir, filename)

        # Debounce: un solo timer activo por instancia
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.RLock()
        self._state_lock = threading.RLock()

        self._load()
        # Garantizar flush final si el proceso se cierra sin que el timer haya disparado
        atexit.register(self._flush)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return

        with self._state_lock:
            super().clear()
            for key, value in raw.items():
                if isinstance(value, dict):
                    super().__setitem__(key, PersistableJobState(value, self._persist))
                else:
                    super().__setitem__(key, value)

    def _to_serializable(self) -> dict:
        with self._state_lock:
            # BUGFIX: usar super().items() en lugar de self.items() para evitar
            # deadlock — self.items() ya adquiere _state_lock (línea 181)
            return {key: dict(value) if isinstance(value, PersistableJobState) else value for key, value in super().items()}

    # ── Persistencia diferida ─────────────────────────────────────────────────

    def _persist(self) -> None:
        """Programa una escritura diferida; cancela y reemplaza el timer anterior."""
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.DEBOUNCE_SEC, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Escribe a disco de forma atómica (tmp + rename). Sin debounce."""
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        tmp = self._path + ".tmp"
        try:
            data = self._to_serializable()
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)   # atómico en POSIX
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def __del__(self) -> None:
        try:
            self._flush()
        except Exception:
            pass

    # ── Overrides del dict ────────────────────────────────────────────────────

    def __setitem__(self, key: str, value: Any) -> None:
        with self._state_lock:
            if isinstance(value, dict) and not isinstance(value, PersistableJobState):
                value = PersistableJobState(value, self._persist)
            super().__setitem__(key, value)
        # Creación de job nuevo → flush inmediato para no perder si cae el proceso
        self._flush()


    def __getitem__(self, key: str) -> Any:
        with self._state_lock:
            return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        with self._state_lock:
            return super().__contains__(key)

    def items(self):
        with self._state_lock:
            return list(super().items())

    def update(self, *args, **kwargs) -> None:
        for mapping in args:
            if hasattr(mapping, "items"):
                for key, value in mapping.items():
                    self[key] = value
            else:
                raise TypeError("update expected a mapping")
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        with self._state_lock:
            if key in self:
                return self[key]
        self[key] = default
        with self._state_lock:
            return self[key]

    def pop(self, key: str, default: Any = None) -> Any:
        with self._state_lock:
            value = super().pop(key, default)
        self._flush()   # baja frecuencia → flush inmediato
        return value

    def popitem(self):
        with self._state_lock:
            value = super().popitem()
        self._flush()
        return value

    def clear(self) -> None:
        with self._state_lock:
            super().clear()
        self._flush()
