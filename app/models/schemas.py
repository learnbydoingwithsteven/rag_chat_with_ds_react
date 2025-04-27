"""
Pydantic models for API request and response schemas.
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

# Chat models
class ChatRequest(BaseModel):
    query: str
    language: str = "en"  # Default to English
    similarity_top_k: int = 5  # Default to 5 similar paragraphs
    temperature: float = 0.2  # Default to low temperature for more factual responses
    model: str = ""  # Model ID to use, if empty will use default
    provider: str = "groq"  # Default to 'groq', can be 'ollama' for local models

class OllamaChatRequest(BaseModel):
    query: str
    similarity_top_k: int = 3
    temperature: float = 0.7
    model: str  # Required Ollama model
    max_tokens: int = 1024

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    model: Optional[str] = None

# API key models
class ApiKeyRequest(BaseModel):
    api_key: str
    
class ApiKeyResponse(BaseModel):
    success: bool
    message: str

# Vector visualization models
class DocumentEmbedding(BaseModel):
    x: float
    y: float
    source: str

class VectorVisualizationResponse(BaseModel):
    pca_data: List[DocumentEmbedding]
    tsne_data: List[DocumentEmbedding]
    explained_variance: float
    document_counts: Dict[str, int]

# BDAP models
class BDAPDatasetResponse(BaseModel):
    datasets: List[Dict[str, Any]]

class BDAPResourceResponse(BaseModel):
    resources: List[Dict[str, Any]]

# PDF processing models
class ProcessPdfRequest(BaseModel):
    files: Optional[List[str]] = None
