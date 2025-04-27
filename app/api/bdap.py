"""
BDAP (Italian Ministry of Economy and Finance) API routes.
"""
from fastapi import APIRouter, HTTPException
import io
import requests
import pandas as pd
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from app.utils.external_apis import bdap_api, bdap_alt_api

router = APIRouter(prefix="/bdap", tags=["bdap"])

@router.get("/datasets")
async def get_bdap_datasets(q: str = ""):
    """
    Get BDAP datasets using the primary CKAN API
    """
    try:
        payload = {"q": q} if q else None
        response = bdap_api("package_search", payload)
        
        if "result" in response and "results" in response["result"]:
            datasets = response["result"]["results"]
            return {"datasets": datasets}
        else:
            return {"datasets": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching datasets: {str(e)}")

@router.get("/alternative/datasets")
async def get_bdap_alt_datasets():
    """
    Get BDAP datasets using the alternative REST API
    """
    try:
        response = bdap_alt_api("dataset")
        return {"datasets": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching datasets: {str(e)}")

@router.get("/alternative/dataset/{dataset_id}")
async def get_bdap_alt_dataset_details(dataset_id: str):
    """
    Get details for a specific BDAP dataset using the alternative REST API
    """
    try:
        response = bdap_alt_api(f"dataset/{dataset_id}")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dataset details: {str(e)}")

@router.get("/resource/{resource_id}")
async def get_bdap_resource(resource_id: str):
    """
    Get a specific BDAP resource and return as CSV
    """
    try:
        url = f"https://bdap-opendata.rgs.mef.gov.it/api/3/action/datastore_search?resource_id={resource_id}&limit=1000"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame and then to CSV
        if "result" in data and "records" in data["result"]:
            records = data["result"]["records"]
            df = pd.DataFrame(records)
            
            # Convert to CSV
            csv_data = io.StringIO()
            df.to_csv(csv_data, index=False)
            
            # Return as streaming response
            csv_data.seek(0)
            return StreamingResponse(
                io.BytesIO(csv_data.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={resource_id}.csv"}
            )
        else:
            raise HTTPException(status_code=404, detail="Resource data not found")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching resource: {str(e)}")
