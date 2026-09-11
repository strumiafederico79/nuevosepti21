from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

try:
    from ..auth import (
        get_admin_user,
        get_current_user,
        handle_approve_user,
        handle_change_password,
        handle_delete_user,
        handle_list_users,
        handle_login,
        handle_me,
        handle_ws_ticket,
        handle_register,
        handle_reject_user,
    )
except ImportError:
    from auth import (
        get_admin_user,
        get_current_user,
        handle_approve_user,
        handle_change_password,
        handle_delete_user,
        handle_list_users,
        handle_login,
        handle_me,
        handle_ws_ticket,
        handle_register,
        handle_reject_user,
    )


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def create_auth_router(*, logger) -> APIRouter:
    router = APIRouter()

    @router.post("/auth/register", tags=["Auth"])
    def register(req: RegisterRequest):
        logger.info(f"📝 Registro: {req.email}")
        try:
            result = handle_register(req.email, req.password, req.name)
            logger.info(f"✓ Registro exitoso: {req.email}")
            return result
        except HTTPException as exc:
            logger.warning(f"⚠️  Registro rechazado: {req.email} - {exc.detail}")
            raise

    @router.post("/auth/login", tags=["Auth"])
    def login(req: LoginRequest):
        try:
            result = handle_login(req.email, req.password)
            logger.info(f"🔓 Login exitoso: {req.email}")
            return result
        except HTTPException as exc:
            logger.warning(f"❌ Login fallido: {req.email} - {exc.detail}")
            raise

    @router.get("/auth/ws-ticket", tags=["Auth"])
    def ws_ticket(current_user: dict = Depends(get_current_user)):
        return handle_ws_ticket(current_user)

    @router.get("/auth/me", tags=["Auth"])
    def me(current_user: dict = Depends(get_current_user)):
        return handle_me(current_user)

    @router.post("/auth/change-password", tags=["Auth"])
    def change_password(req: ChangePasswordRequest, current_user: dict = Depends(get_current_user)):
        logger.info(f"🔐 Cambio de contraseña: {current_user['email']}")
        try:
            result = handle_change_password(req.current_password, req.new_password, current_user)
            logger.info(f"✓ Contraseña cambiada: {current_user['email']}")
            return result
        except HTTPException:
            logger.warning(f"⚠️  Cambio de contraseña fallido: {current_user['email']}")
            raise

    @router.get("/auth/admin/users", tags=["Auth"])
    def list_users(admin: dict = Depends(get_admin_user)):
        logger.info(f"📋 Admin {admin['email']} listó usuarios")
        return handle_list_users(admin)

    @router.post("/auth/admin/approve/{user_id}", tags=["Auth"])
    def approve_user(user_id: str, admin: dict = Depends(get_admin_user)):
        logger.info(f"✅ Admin {admin['email']} aprobó usuario {user_id}")
        try:
            result = handle_approve_user(user_id, admin)
            logger.info(f"✓ Usuario {user_id} aprobado")
            return result
        except Exception as exc:
            logger.error(f"❌ Error aprobando usuario {user_id}: {exc}")
            raise

    @router.post("/auth/admin/reject/{user_id}", tags=["Auth"])
    def reject_user(user_id: str, admin: dict = Depends(get_admin_user)):
        logger.info(f"🚫 Admin {admin['email']} rechazó usuario {user_id}")
        try:
            result = handle_reject_user(user_id, admin)
            logger.info(f"✓ Usuario {user_id} rechazado")
            return result
        except Exception as exc:
            logger.error(f"❌ Error rechazando usuario {user_id}: {exc}")
            raise

    @router.delete("/auth/admin/users/{user_id}", tags=["Auth"])
    def delete_user(user_id: str, admin: dict = Depends(get_admin_user)):
        logger.warning(f"🗑️  Admin {admin['email']} eliminó usuario {user_id}")
        try:
            result = handle_delete_user(user_id, admin)
            logger.info(f"✓ Usuario {user_id} eliminado")
            return result
        except Exception as exc:
            logger.error(f"❌ Error eliminando usuario {user_id}: {exc}")
            raise

    return router
