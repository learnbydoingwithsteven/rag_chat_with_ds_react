"""
Chat-related API routes.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import groq_client, initialize_groq_client, get_stored_api_key, generate_response
from app.db.database import get_similar_paragraphs
from app.core.config import GROQ_MODEL

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the LLM using RAG
    """
    # Move all global declarations to the top of the function
    global groq_client
    # Get available API key
    api_key = get_stored_api_key()
    print(f"Chat endpoint: API key found={api_key is not None}")
    
    # We'll initialize the client on demand in the generate_response function
    # This approach doesn't rely on the global groq_client variable
    
    # Check if API key is available at all
    if not api_key:
        error_message = "The LLM service is not available."
        
        if not api_key:
            detail = "No API key has been set. Please configure your Groq API key in the API Key Settings."
        else:
            detail = "Your API key might be invalid or there might be connectivity issues with the Groq API. Please check your API key in the API Key Settings."
        
        print(f"Chat error: {error_message} {detail}")
        
        return ChatResponse(
            answer=f"{error_message} {detail}",
            sources=[],
            model=None
        )
    
    # Get context from vector database
    # Use default values if not specified
    similarity_top_k = request.similarity_top_k or 5
    temperature = request.temperature or 0.2
    model_to_use = request.model or GROQ_MODEL
    
    print(f"Chat request: query='{request.query[:50]}...', model={model_to_use}, temp={temperature}, top_k={similarity_top_k}")
    
    try:
        # Get similar paragraphs from the database
        similar_results = get_similar_paragraphs(request.query, request.similarity_top_k)
        print(f"Retrieved {len(similar_results)} similar paragraphs for context")
    except Exception as e:
        error_msg = f"Error retrieving context: {str(e)}"
        print(f"Database error: {error_msg}")
        return ChatResponse(
            answer=error_msg,
            sources=[],
            model=None
        )
        
    # Determine if we should use Ollama
    provider = getattr(request, 'provider', 'groq')
    use_ollama = provider == 'ollama'
    print(f"Provider: {provider}, using Ollama: {use_ollama}")
    
    # The API key and client are now handled in the generate_response function
    # We no longer need to manually check or initialize the client here
        
    # Generate response from LLM
    try:
        result = generate_response(
            query=request.query, 
            context=similar_results,  # Pass the full results list
            model_id=model_to_use,
            temperature=temperature,
            use_ollama=use_ollama  # Pass the provider information
        )
        
        # Return response
        answer = result.get("answer", "")
        model_used = result.get("model")
        print(f"Successfully generated response using model {model_used}")
        
        # Format sources for the response with proper frontend compatibility
        sources = [{
            "text": result["text"], 
            "file": result["source"],
            "page": result.get("page", "1"),  # Default to page 1 if not available
            "similarity": result.get("similarity", 0.5)  # Include similarity score
        } for result in similar_results]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            model=model_used
        )
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(f"LLM error: {error_msg}")
        return ChatResponse(
            answer=error_msg,
            sources=[],
            model=None
        )
