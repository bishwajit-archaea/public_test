from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks, Query, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import httpx

from app.core.database import get_db, AsyncSessionLocal
from app.core.config import settings
from app.models.app_user import AppUser
from app.middleware.user import get_current_user
from app.schemas.auth import LoginRequest, LoginResponse
from app.services.user_sync import sync_user_token_internal
from typing import Optional, Any

from app.core.logger import logger

router = APIRouter()

@router.post("/login", response_model=LoginResponse)
async def login(
    background_tasks: BackgroundTasks,
    login_data: LoginRequest = Body(...),
):
    """
    Login endpoint:
    1. Proxies login to auth-service.
    2. Returns token and user info.
    """
    try:
        async with httpx.AsyncClient() as client:
            # Prepare payload for auth-service
            payload = {
                "username": login_data.username,
                "password": login_data.password
            }
            if login_data.providerId:
                payload["providerId"] = login_data.providerId

            auth_res = await client.post(
                f"{settings.AUTH_SERVICE_URL}/auth/login",
                json=payload,
                headers={"X-Internal-Token": settings.AUTH_SERVICE_TOKEN},
                timeout=10.0
            )
            
            if auth_res.status_code != 201: # NestJS returns 201 for POST
                detail = "Invalid credentials"
                try:
                    detail = auth_res.json().get("message", detail)
                except:
                    pass
                raise HTTPException(status_code=auth_res.status_code, detail=detail)
                
            auth_data = auth_res.json()
            access_token = auth_data.get("access_token")

            # Get user info from profile
            profile_res = await client.get(
                f"{settings.AUTH_SERVICE_URL}/auth/profile",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-Internal-Token": settings.AUTH_SERVICE_TOKEN
                },
                timeout=10.0
            )

            if profile_res.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch user profile")

            profile_data = profile_res.json()
            user_info = profile_data.get("user")
            user_id =user_info.get("sub")
            # 2. Sync Background Token (Background Task)
            background_tasks.add_task(sync_user_token_internal, user_id)

            return {
                "status": "success",
                "user": user_info,
                "token": access_token,
                "refresh_token": auth_data.get("refresh_token")
            }

    except httpx.RequestError as e:
        logger.error(f"Auth service error: {e}")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")


@router.get("/providers")
async def get_providers():
    """
    Fetch active providers from auth-service.
    """
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{settings.AUTH_SERVICE_URL}/auth/providers",
                headers={"X-Internal-Token": settings.AUTH_SERVICE_TOKEN},
                timeout=10.0
            )
            
            if res.status_code != 200:
                raise HTTPException(status_code=res.status_code, detail="Failed to fetch providers")
                
            return res.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

@router.get("/profile")
async def get_profile(
    user_id: str = Depends(get_current_user),
    authorization: str = Header(None, alias="Authorization")
):
    """
    Fetch user profile from auth-service by proxying the request.
    """
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required")

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{settings.AUTH_SERVICE_URL}/auth/profile",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Internal-Token": settings.AUTH_SERVICE_TOKEN
                },
                timeout=10.0
            )
            
            if res.status_code != 200:
                detail = "Failed to fetch profile"
                try:
                    detail = res.json().get("message", detail)
                except:
                    pass
                raise HTTPException(status_code=res.status_code, detail=detail)
                
            return res.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")


@router.post("/refresh")
async def refresh_token(payload: Any = Body(...)):
    """
    Silently refresh access token using a refresh_token.
    No auth middleware — the refresh token IS the credential.
    Forwards to auth-service POST /refresh-token/refresh.
    """
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{settings.AUTH_SERVICE_URL}/refresh-token/refresh",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Internal-Token": settings.AUTH_SERVICE_TOKEN
                },
                timeout=10.0
            )
            if res.status_code != 200 and res.status_code != 201:
                detail = "Token refresh failed"
                try: detail = res.json().get("message", detail)
                except Exception: pass
                raise HTTPException(status_code=res.status_code, detail=detail)
            return res.json()
    except HTTPException: raise
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token refresh error: {str(e)}")

