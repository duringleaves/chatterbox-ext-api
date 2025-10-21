"""Simple API key authentication dependency."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from .config import settings

API_KEY_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing API key",
    headers={"WWW-Authenticate": "Bearer"},
)


def require_api_key(request: Request, authorization: str = Header(default="")) -> str:
    token = ""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value:
            token = value.strip()
    if not token:
        token = request.query_params.get("api_key", "").strip()
    if not token or token != settings.api_key:
        raise API_KEY_ERROR
    return token


def get_authorized_token(token: str = Depends(require_api_key)) -> str:
    return token
