"""
Settings and API key management routes.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models.schemas import ApiKeyRequest, ApiKeyResponse
from app.services.llm_service import initialize_groq_client, save_api_key, get_available_models

router = APIRouter(prefix="/settings", tags=["settings"])

@router.post("/api-key", response_model=ApiKeyResponse)
async def set_api_key(request: ApiKeyRequest):
    """
    Save the Groq API key for future use
    """
    try:
        # Try to initialize with the new key
        client = initialize_groq_client(request.api_key)
        
        if client:
            # Save the key for future use
            save_api_key(request.api_key)
            return ApiKeyResponse(
                success=True,
                message="API key validated and saved successfully!"
            )
        else:
            return ApiKeyResponse(
                success=False,
                message="Failed to validate API key with Groq. Please check the key and try again."
            )
    except Exception as e:
        return ApiKeyResponse(
            success=False,
            message=f"Error setting API key: {str(e)}"
        )

@router.get("/api-key/check")
async def check_api_key():
    """
    Check if a Groq API key is available (not validating it)
    """
    from app.services.llm_service import get_stored_api_key
    
    api_key = get_stored_api_key()
    if api_key:
        masked_key = api_key[:4] + "..." + api_key[-4:]
        return {
            "available": True,
            "masked_key": masked_key
        }
    else:
        return {
            "available": False
        }

@router.get("/models")
async def get_groq_models():
    """
    Return available Groq models organized by category
    """
    try:
        # Initialize the client if needed
        from app.services.llm_service import groq_client, initialize_groq_client
        if not groq_client:
            initialize_groq_client()
        
        # If client still not available, return just the static lists
        if not groq_client:
            return {
                "status": "warning",
                "message": "No API key available, showing static model lists only",
                "models": get_available_models(),
                "default_model": None,
                "can_use_api": False
            }
        
        # Otherwise include status info from the client
        return {
            "status": "success",
            "message": "Successfully retrieved model information",
            "models": get_available_models(),
            "default_model": "llama3-70b-8192",
            "can_use_api": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting model information: {str(e)}",
            "models": get_available_models(),
            "default_model": None,
            "can_use_api": False
        }
