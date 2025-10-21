"""Simple API key authentication dependency."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from .config import settings

API_KEY_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing API key",
    headers={"WWW-Authenticate": "Bearer"},
)


def require_api_key(authorization: str = Header(default="")) -> str:
    if not authorization:
        raise API_KEY_ERROR
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise API_KEY_ERROR
    if token.strip() != settings.api_key:
        raise API_KEY_ERROR
    return token.strip()


def get_authorized_token(token: str = Depends(require_api_key)) -> str:
    return token
