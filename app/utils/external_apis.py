"""
Utility functions for interacting with external APIs like BDAP.
"""
import requests
from typing import Dict, Any, Optional
from app.core.config import BDAP_ROOT, BDAP_ALT_ROOT

def bdap_api(endpoint: str, payload=None):
    """
    Make a request to the BDAP CKAN API
    """
    try:
        url = f"{BDAP_ROOT}/{endpoint}"
        response = requests.get(url, params=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"BDAP API Error: {str(e)}")
        return {"error": str(e)}

def bdap_alt_api(endpoint: str):
    """
    Make a request to the alternative BDAP REST API
    """
    try:
        url = f"{BDAP_ALT_ROOT}/{endpoint}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"BDAP Alt API Error: {str(e)}")
        return {"error": str(e)}
