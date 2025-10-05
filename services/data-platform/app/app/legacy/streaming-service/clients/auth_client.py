"""
HTTP client for Auth Service.
"""
import logging
import os
from typing import Optional, Dict, Any

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8010")


async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token via Auth Service.
    
    Args:
        token: JWT token string
    
    Returns:
        dict: User claims
        
    Raises:
        HTTPException: If token is invalid or auth service unavailable
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/auth/verify",
                json={"token": token}
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            data = response.json()
            
            if not data.get("valid"):
                raise HTTPException(
                    status_code=401,
                    detail=data.get("error", "Token validation failed")
                )
            
            return data.get("user_claims", {})
            
    except httpx.TimeoutException:
        logger.error("Auth service timeout")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    except httpx.HTTPError as e:
        logger.error(f"Auth service error: {e}")
        raise HTTPException(status_code=503, detail="Authentication service error")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in token verification: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")


async def verify_ws_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token for WebSocket connections.
    
    Args:
        token: JWT token string
    
    Returns:
        dict: User claims
        
    Raises:
        ValueError: If token is invalid
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/auth/verify-ws",
                json={"token": token}
            )
            
            if response.status_code != 200:
                raise ValueError("Invalid token")
            
            data = response.json()
            
            if not data.get("valid"):
                raise ValueError(data.get("error", "Token validation failed"))
            
            return data.get("user_claims", {})
            
    except Exception as e:
        logger.error(f"WebSocket token verification failed: {e}")
        raise ValueError(f"Token verification failed: {str(e)}")

