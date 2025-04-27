"""
Visualization API routes.
"""
from fastapi import APIRouter, Query
from typing import List, Optional
from app.models.schemas import VectorVisualizationResponse
from app.services.visualization import get_vector_visualization

router = APIRouter(prefix="/visualization", tags=["visualization"])

@router.get("", response_model=VectorVisualizationResponse)
async def vector_visualization(selected_sources: Optional[List[str]] = Query(None)):
    """
    Get vector visualization data (PCA and t-SNE)
    """
    result = get_vector_visualization(selected_sources)
    
    return VectorVisualizationResponse(
        pca_data=result["pca_data"],
        tsne_data=result["tsne_data"],
        explained_variance=result["explained_variance"],
        document_counts=result["document_counts"]
    )
