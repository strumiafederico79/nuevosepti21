"""
auth.py — Sistema de autenticación con registro + aprobación manual.

Flujo:
  1. Usuario se registra (POST /auth/register) → queda en estado pending
  2. Admin aprueba (POST /auth/admin/approve/{user_id}) → usuario puede loguearse
  3. Login (POST /auth/login) → devuelve JWT
  4. Endpoints protegidos requieren header: Authorization: Bearer <token>

Almacenamiento: JSON en disco (users_db.json) — simple, sin dependencias extra.
Para producción con muchos usuarios migrar a SQLite o PostgreSQL.

Admin: se crea automáticamente al arrancar si no existe. Credenciales en .env:
  ADMIN_EMAIL=admin@tudominio.com
  ADMIN_PASSWORD=tu_password_seguro
  JWT_SECRET=string_aleatorio_largo
"""

import os, json, uuid, hashlib, hmac, time, base64, threading, secrets, string
from typing import Optional
from fastapi import HTTPException, Depends, Header
import logging

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

USERS_DB_PATH  = os.getenv("USERS_DB_PATH", "users_db.json")

# JWT_SECRET: CRÍTICO para seguridad. Si no está configurado en producción,
# se genera un valor aleatorio (único por ejecución) en dev, pero se loguea
# una advertencia. En producción, DEBE configurarse en .env o secrets.
def _get_or_generate_jwt_secret() -> tuple[str, bool]:
    """Returns (secret, was_generated)."""
    secret = os.getenv("JWT_SECRET", "").strip()
    if secret:
        return secret, False
    generated = secrets.token_urlsafe(32)
    return generated, True

_jwt_secret, _JWT_SECRET_WAS_GENERATED = _get_or_generate_jwt_secret()
JWT_SECRET = _jwt_secret

# Flag экспортируемый para que app.py pueda проверять при старте
JWT_SECRET_WAS_GENERATED = _JWT_SECRET_WAS_GENERATED
JWT_EXPIRY_SEC = int(os.getenv("JWT_EXPIRY_HOURS", "168")) * 3600  # 7 días por defecto
ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", "admin@master.local")

# ADMIN_PASSWORD: genera una contraseña fuerte aleatoria si no está configurada.
# El admin debe cambiarla en la primera sesión o configurarla en .env.
def _get_or_generate_admin_password() -> str:
    password = os.getenv("ADMIN_PASSWORD", "").strip()
    if password:
        if len(password) < 8:
            logger.error("❌ ADMIN_PASSWORD es muy corta (mín. 8 caracteres)")
            raise ValueError("ADMIN_PASSWORD configurado pero es demasiado débil")
        return password
    # Generar una contraseña segura aleatoria: 16 caracteres alfanuméricos + símbolos
    generated = "".join(
        secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(16)
    )
    logger.debug(
        "ADMIN_PASSWORD no configurado. Contraseña generada disponible en logs de debug."
        " CÁMBIALA en la primera sesión. En producción, define ADMIN_PASSWORD en .env."
    )
    return generated

ADMIN_PASSWORD = _get_or_generate_admin_password()

_db_lock = threading.RLock()

# ── Helpers de hash y JWT ─────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return base64.b64encode(salt + key).decode()

def _verify_password(password: str, hashed: str) -> bool:
    try:
        raw = base64.b64decode(hashed.encode())
        salt, key = raw[:16], raw[16:]
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        return hmac.compare_digest(key, candidate)
    except Exception:
        return False

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _create_jwt(user_id: str, email: str, role: str) -> str:
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_SEC,
    }).encode())
    sig = _b64url(hmac.new(
        JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256
    ).digest())
    return f"{header}.{payload}.{sig}"

def _verify_jwt(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("formato inválido")
        header, payload, sig = parts
        expected_sig = _b64url(hmac.new(
            JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256
        ).digest())
        if not hmac.compare_digest(sig, expected_sig):
            raise ValueError("firma inválida")
        padding = 4 - len(payload) % 4
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * padding))
        if data.get("exp", 0) < time.time():
            raise ValueError("token expirado")
        return data
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token inválido: {e}")

# ── DB de usuarios (JSON en disco) ────────────────────────────────────────────

def _load_db() -> dict:
    with _db_lock:
        if not os.path.exists(USERS_DB_PATH):
            return {"users": {}}
        try:
            with open(USERS_DB_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {"users": {}}

def _save_db(db: dict) -> None:
    with _db_lock:
        with open(USERS_DB_PATH, "w") as f:
            json.dump(db, f, indent=2)

def _get_user_by_email(email: str) -> Optional[dict]:
    db = _load_db()
    for u in db["users"].values():
        if u["email"].lower() == email.lower():
            return u
    return None

def _get_user_by_id(user_id: str) -> Optional[dict]:
    db = _load_db()
    return db["users"].get(user_id)

# ── Bootstrap: crear admin si no existe ──────────────────────────────────────

def bootstrap_admin() -> None:
    existing = _get_user_by_email(ADMIN_EMAIL)
    if existing:
        return
    db = _load_db()
    admin_id = str(uuid.uuid4())
    db["users"][admin_id] = {
        "id": admin_id,
        "email": ADMIN_EMAIL,
        "password_hash": _hash_password(ADMIN_PASSWORD),
        "role": "admin",
        "status": "approved",
        "name": "Admin",
        "created_at": time.time(),
        "approved_at": time.time(),
    }
    _save_db(db)

# ── FastAPI dependency: usuario actual ────────────────────────────────────────

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Se requiere autenticación")
    token = authorization.split(" ", 1)[1]
    payload = _verify_jwt(token)
    user = _get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    if user["status"] != "approved":
        raise HTTPException(status_code=403, detail="Tu cuenta está pendiente de aprobación")
    return user

def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Se requiere rol de administrador")
    return current_user

# ── Handlers de endpoints (se registran en app.py) ───────────────────────────

def handle_register(email: str, password: str, name: str) -> dict:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Email inválido")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 8 caracteres")
    if _get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Ya existe una cuenta con ese email")

    db = _load_db()
    user_id = str(uuid.uuid4())
    db["users"][user_id] = {
        "id": user_id,
        "email": email,
        "password_hash": _hash_password(password),
        "role": "user",
        "status": "pending",
        "name": name.strip() or email.split("@")[0],
        "created_at": time.time(),
        "approved_at": None,
    }
    _save_db(db)
    return {
        "message": "Registro exitoso. Tu cuenta está pendiente de aprobación.",
        "user_id": user_id,
        "status": "pending",
    }

def handle_login(email: str, password: str) -> dict:
    user = _get_user_by_email(email.strip().lower())
    if not user or not _verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")
    if user["status"] == "pending":
        raise HTTPException(status_code=403, detail="Tu cuenta está pendiente de aprobación")
    if user["status"] == "rejected":
        raise HTTPException(status_code=403, detail="Tu cuenta ha sido rechazada")

    token = _create_jwt(user["id"], user["email"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRY_SEC,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        },
    }



WS_TICKET_EXPIRY_SEC = int(os.getenv("WS_TICKET_EXPIRY_SEC", "60"))

def handle_ws_ticket(current_user: dict = Depends(get_current_user)) -> dict:
    """Issue a short-lived signed token dedicated to a WebSocket handshake."""
    now = int(time.time())
    payload = {
        "sub": current_user["id"],
        "email": current_user["email"],
        "role": current_user["role"],
        "iat": now,
        "exp": now + WS_TICKET_EXPIRY_SEC,
        "aud": "websocket",
        "typ": "ws-ticket",
    }
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(payload).encode())
    sig = _b64url(hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest())
    return {"token": f"{header}.{body}.{sig}", "expires_in": WS_TICKET_EXPIRY_SEC}

def handle_me(current_user: dict = Depends(get_current_user)) -> dict:
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"],
        "role": current_user["role"],
        "status": current_user["status"],
    }

def handle_list_users(admin: dict = Depends(get_admin_user)) -> list:
    db = _load_db()
    return [
        {
            "id": u["id"],
            "email": u["email"],
            "name": u["name"],
            "role": u["role"],
            "status": u["status"],
            "created_at": u["created_at"],
            "approved_at": u.get("approved_at"),
        }
        for u in db["users"].values()
        if u["id"] != admin["id"]
    ]

def handle_approve_user(user_id: str, admin: dict = Depends(get_admin_user)) -> dict:
    db = _load_db()
    user = db["users"].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    user["status"] = "approved"
    user["approved_at"] = time.time()
    _save_db(db)
    return {"message": f"Usuario {user['email']} aprobado", "user_id": user_id}

def handle_reject_user(user_id: str, admin: dict = Depends(get_admin_user)) -> dict:
    db = _load_db()
    user = db["users"].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if user.get("role") == "admin":
        raise HTTPException(status_code=400, detail="No podés rechazar al admin")
    user["status"] = "rejected"
    _save_db(db)
    return {"message": f"Usuario {user['email']} rechazado", "user_id": user_id}

def handle_delete_user(user_id: str, admin: dict = Depends(get_admin_user)) -> dict:
    db = _load_db()
    user = db["users"].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if user.get("role") == "admin":
        raise HTTPException(status_code=400, detail="No podés eliminar al admin")
    del db["users"][user_id]
    _save_db(db)
    return {"message": f"Usuario eliminado", "user_id": user_id}

def handle_change_password(
    current_password: str,
    new_password: str,
    current_user: dict = Depends(get_current_user)
) -> dict:
    if not _verify_password(current_password, current_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Contraseña actual incorrecta")
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="La nueva contraseña debe tener al menos 8 caracteres")
    db = _load_db()
    db["users"][current_user["id"]]["password_hash"] = _hash_password(new_password)
    _save_db(db)
    return {"message": "Contraseña actualizada"}
