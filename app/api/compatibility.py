"""
Compatibility routes for backward compatibility with the old monolithic API.
These routes redirect to the new modular API endpoints.
"""
import os
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional

from app.models.schemas import ProcessPdfRequest, ChatRequest, OllamaChatRequest
from app.api.pdf_processing import (
    list_pdf_files,
    get_pdf_process_status,
    upload_pdf,
    process_pdf_files
)
from app.api.vector_db import (
    check_database,
    create_vector_database,
    get_vector_db_status
)
from app.api.visualization import vector_visualization
from app.api.files import list_text_files, view_text_content, list_all_documents
from app.api.settings import set_api_key, check_api_key, get_groq_models
from app.api.chat import chat
from app.models.schemas import ApiKeyRequest
from app.services.llm_service import get_available_ollama_models, check_ollama_availability, get_stored_api_key, initialize_groq_client
from app.db.vector_db import create_database, process_all_texts_into_db

router = APIRouter(tags=["compatibility"])

# PDF processing compatibility routes
@router.get("/list-pdf-files")
async def compat_list_pdf_files():
    """Compatibility route for listing PDF files"""
    response = await list_pdf_files()
    
    # For compatibility: if files are objects with filename property, extract just the filenames
    if "files" in response and isinstance(response["files"], list):
        # Transform rich file objects into simple filenames for older frontend compatibility
        response["files"] = [f["filename"] if isinstance(f, dict) and "filename" in f else f for f in response["files"]]
    
    return response

@router.get("/process-pdf-status/{task_id}")
async def compat_get_pdf_process_status(task_id: str):
    """Compatibility route for getting PDF processing status"""
    return await get_pdf_process_status(task_id)

# Add dedicated Ollama endpoint
@router.post("/ollama-chat")
async def ollama_chat(request: OllamaChatRequest):
    """Dedicated endpoint for Ollama chat"""
    import logging
    import os
    from app.core.config import DB_PATH
    from app.db.database import get_similar_paragraphs
    from app.services.llm_service import generate_response, check_ollama_availability
    
    logger = logging.getLogger(__name__)
    logger.info(f"Ollama chat request for model: {request.model}")
    
    # Ensure Ollama is available
    if not check_ollama_availability():
        return {
            "answer": "Error: Ollama is not available. Please make sure the Ollama server is running.",
            "sources": [],
            "model": None
        }
    
    # Get similar paragraphs from the database
    if not os.path.exists(DB_PATH):
        logger.error(f"Vector database not found at {DB_PATH}")
        return {"answer": "Error: Vector database not found. Please process some documents first.", "sources": []}
        
    try:
        # Get similar contexts
        similar_results = get_similar_paragraphs(request.query, request.similarity_top_k)
        logger.info(f"Found {len(similar_results)} similar paragraphs in database")
        
        # Call Ollama directly
        ollama_result = generate_response(
            query=request.query,
            context=similar_results,  # Pass the full results list
            model_id=request.model,
            temperature=request.temperature,
            use_ollama=True  # Always use Ollama for this endpoint
        )
        
        # Extract answer from result
        answer = ollama_result.get("answer", "No response generated")
        
        # Extract sources for citation with proper format for the frontend
        sources = [{
            "text": result["text"], 
            "file": result["source"],
            "page": result.get("page", "1"),  # Default to page 1 if not available
            "similarity": result.get("similarity", 0.5)  # Include similarity score for UI
        } for result in similar_results]
        
        logger.info(f"Successfully generated Ollama response with model: {request.model}")
        return {"answer": answer, "sources": sources, "model": f"ollama:{request.model}"}
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in Ollama chat endpoint: {error_msg}")
        return {
            "answer": f"Error generating response with Ollama: {error_msg}",
            "sources": [],
            "model": None
        }

@router.post("/upload-pdf")
async def compat_upload_pdf(file: UploadFile = File(...)):
    """Compatibility route for uploading PDF files"""
    return await upload_pdf(file)

@router.post("/process-pdf-files")
async def compat_process_pdf_files(request: dict, background_tasks: BackgroundTasks):
    """Compatibility route for processing PDF files"""
    pdf_files = request.get("pdf_files", [])
    
    # For single files, immediately process and return status
    if len(pdf_files) == 1:
        from app.services.pdf_service import process_pdf_file
        from app.core.config import PDF_DIR, TEXT_DIR
        import os
        
        try:
            # Process the PDF file directly for immediate feedback
            pdf_path = os.path.join(PDF_DIR, pdf_files[0])
            text_filename = pdf_files[0].replace(".pdf", ".txt")
            text_path = os.path.join(TEXT_DIR, text_filename)
            
            # Ensure text directory exists
            os.makedirs(TEXT_DIR, exist_ok=True)
            
            # Process the file
            result = process_pdf_file(pdf_path, TEXT_DIR)
            
            return {
                "status": "success",
                "message": f"Processed 1 PDF file directly: {pdf_files[0]}",
                "task_id": "direct_processing",
                "processed_count": 1,
                "total_count": 1,
                "completed": True
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing PDF: {str(e)}",
                "task_id": "error",
                "completed": True
            }
    
    # For multiple files, use background task
    process_request = ProcessPdfRequest(pdf_files=pdf_files)
    return await process_pdf_files(process_request, background_tasks)

# Vector database compatibility routes
@router.get("/check-database")
async def compat_check_database():
    """Compatibility route for checking database"""
    from app.core.config import DB_PATH, VECTOR_DB_DIR
    import os
    
    # Ensure the vector DB directory exists
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # Check if database exists, if not create it immediately for better UX
    if not os.path.exists(DB_PATH):
        from app.db.vector_db import create_database
        create_database()
        
    return await check_database()

@router.get("/check-db-status")
async def compat_check_db_status():
    """Another compatibility route for checking database status"""
    return await check_database()

# Cache for embedding model to avoid reloading
_embedding_model_cache = None

def get_cached_embedding_model():
    """Return cached embedding model or load it if not cached"""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        from app.db.vector_db import load_embedding_model
        _embedding_model_cache = load_embedding_model()
        print("Loaded embedding model into cache")
    return _embedding_model_cache

@router.post("/create-vector-database")
async def compat_create_vector_database(background_tasks: BackgroundTasks):
    """Compatibility route for creating vector database"""
    # Preload the model before passing to the task for faster performance
    get_cached_embedding_model()
    return await create_vector_database(background_tasks)

@router.post("/build-vector-db")
async def compat_build_vector_database(background_tasks: BackgroundTasks):
    """Another compatibility route for creating vector database"""
    return await create_vector_database(background_tasks)

@router.get("/vector-db-status/{task_id}")
async def compat_get_vector_db_status(task_id: str):
    """Compatibility route for getting vector database creation status"""
    return await get_vector_db_status(task_id)

# Visualization compatibility routes
@router.get("/vector-visualization")
async def compat_vector_visualization(selected_sources: Optional[List[str]] = None):
    """Compatibility route for vector visualization"""
    return await vector_visualization(selected_sources)

# File management compatibility routes
@router.get("/list-text-files")
async def compat_list_text_files():
    """Compatibility route for listing text files"""
    response = await list_text_files()
    
    # For compatibility: if files are objects with filename property, extract just the filenames
    if "files" in response and isinstance(response["files"], list):
        # Transform rich file objects into simple filenames for older frontend compatibility
        response["files"] = [f["filename"] if isinstance(f, dict) and "filename" in f else f for f in response["files"]]
    
    return response

@router.get("/list-all-documents")
async def compat_list_all_documents():
    """Compatibility route for listing all documents (PDF and text files)"""
    # Get the directory paths
    from app.core.config import PDF_DIR, TEXT_DIR
    import os
    
    # Create an empty array for the flattened documents
    flat_documents = []
    
    # Directly gather files ourselves to improve reliability
    # PDF files
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        for filename in pdf_files:
            try:
                file_path = os.path.join(PDF_DIR, filename)
                file_size = os.path.getsize(file_path)
                
                # Format file size
                if file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                # Add to documents array
                flat_documents.append({
                    "name": filename,
                    "path": file_path,
                    "size": file_size,
                    "size_str": size_str,
                    "type": "pdf",
                    "modified": os.path.getmtime(file_path)
                })
            except Exception as e:
                print(f"Error adding PDF file {filename}: {str(e)}")
    
    # Text files
    if os.path.exists(TEXT_DIR):
        text_files = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith('.txt')]
        for filename in text_files:
            try:
                file_path = os.path.join(TEXT_DIR, filename)
                file_size = os.path.getsize(file_path)
                
                # Format file size
                if file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                # Check if PDF exists for this text file
                pdf_filename = filename.replace(".txt", ".pdf")
                pdf_path = os.path.join(PDF_DIR, pdf_filename)
                has_pdf = os.path.exists(pdf_path)
                
                # Add to documents array
                flat_documents.append({
                    "name": filename,
                    "path": file_path,
                    "size": file_size,
                    "size_str": size_str,
                    "type": "text",
                    "has_pdf": has_pdf,
                    "pdf_filename": pdf_filename if has_pdf else None,
                    "modified": os.path.getmtime(file_path)
                })
            except Exception as e:
                print(f"Error adding text file {filename}: {str(e)}")
    
    # Return the flat array structure expected by the frontend
    return {
        "status": "success",
        "message": f"Found {len(flat_documents)} documents",
        "documents": flat_documents
    }

@router.get("/view-text-content/{filename}")
async def compat_view_text_content(filename: str):
    """Compatibility route for viewing text file content"""
    return await view_text_content(filename)

# API key management compatibility routes
@router.post("/set-api-key")
async def compat_set_api_key(request: dict):
    """Compatibility route for setting the API key"""
    # Convert the legacy format {api_key: "key"} to the new ApiKeyRequest format
    api_key_request = ApiKeyRequest(api_key=request.get("api_key", ""))
    return await set_api_key(api_key_request)

@router.get("/check-api-key")
async def compat_check_api_key():
    """Compatibility route for checking API key status"""
    # First get basic availability check
    result = await check_api_key()
    
    # Import these from config and services
    from app.core.config import GROQ_MODEL
    from app.services.llm_service import groq_client, initialize_groq_client, get_stored_api_key
    
    # If we don't have a client yet but do have a key, try to initialize
    if result.get("available", False) and not groq_client:
        # Force a new initialization attempt
        print("API key exists but client not initialized, attempting initialization")
        stored_key = get_stored_api_key()
        if stored_key:
            initialize_groq_client(stored_key)
    
    # Check if client was successfully initialized
    client_initialized = groq_client is not None
    
    # Transform the response format to match what the original backend returned
    # The original returned key_exists, client_initialized, and model
    response = {
        "key_exists": result.get("available", False),
        "client_initialized": client_initialized,
        "model": GROQ_MODEL if client_initialized else None,
        # Also include the models array for dropdown in the frontend
        "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    }
    
    print(f"API Key Status: exists={response['key_exists']}, initialized={response['client_initialized']}, model={response['model']}")
    return response

# Add Groq models endpoint for compatibility
@router.get("/groq-models")
async def compat_groq_models():
    """Compatibility route for getting all models (Groq and Ollama)"""
    # For the React frontend, the model list needs to be simple strings
    from app.core.config import GROQ_PRODUCTION_MODELS, GROQ_PREVIEW_MODELS, GROQ_PREVIEW_SYSTEMS, OLLAMA_MODELS, GROQ_MODEL
    import logging
    import time
    
    logger = logging.getLogger(__name__)
    logger.info("Getting model list for frontend")
    start_time = time.time()
    
    # Get Ollama models if available
    ollama_model_list = []
    ollama_available = False
    try:
        # First check if Ollama is available
        ollama_available = check_ollama_availability()
        if ollama_available:
            logger.info("Ollama is available, fetching models...")
            # Get dynamically available models
            ollama_model_list = get_available_ollama_models()
            logger.info(f"Found {len(ollama_model_list)} dynamic Ollama models")
            
            # Add any predefined models that weren't found
            for model in OLLAMA_MODELS:
                if model not in ollama_model_list:
                    ollama_model_list.append(model)
            
            logger.info(f"Total Ollama models (including predefined): {len(ollama_model_list)}")
        else:
            logger.info("Ollama is not available")
    except Exception as e:
        logger.error(f"Error checking Ollama availability: {str(e)}")
    
    # Format expected by the updated frontend
    all_models = {
        "production_models": GROQ_PRODUCTION_MODELS,
        "preview_models": GROQ_PREVIEW_MODELS,
        "preview_systems": GROQ_PREVIEW_SYSTEMS,
        "ollama_models": ollama_model_list,  # Use actual found models
        "ollama_available": ollama_available,
        "default_model": GROQ_MODEL,
        "other_models": [],
    }
    
    # Log information about returned models
    end_time = time.time()
    logger.info(f"Returning {len(GROQ_PRODUCTION_MODELS)} production models, "
               f"{len(GROQ_PREVIEW_MODELS)} preview models, "
               f"{len(GROQ_PREVIEW_SYSTEMS)} systems, and "
               f"{len(ollama_model_list)} Ollama models in {end_time - start_time:.2f}s")
    
    return all_models

# Add chat endpoint compatibility
@router.post("/chat")
async def compat_chat(request: ChatRequest):
    """Compatibility route for chat functionality"""
    # Check database existence before attempting to chat
    from app.core.config import DB_PATH, VECTOR_DB_DIR, GROQ_PRODUCTION_MODELS, GROQ_PREVIEW_MODELS, GROQ_PREVIEW_SYSTEMS, OLLAMA_MODELS, GROQ_MODEL
    import os
    import logging
    from app.services.llm_service import groq_client, initialize_groq_client, get_stored_api_key, check_ollama_availability, get_available_ollama_models
    from app.db.vector_db import create_database, process_all_texts_into_db
    
    # We don't need global declarations with the new pattern
    # Client initialization is now handled inside generate_response
    
    logger = logging.getLogger(__name__)
    
    # Make sure the vector DB directory exists
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # If database doesn't exist, create it automatically
    db_created = False
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}, attempting to create it automatically...")
        try:
            create_database()
            # Check if there are text files to process
            from app.core.config import TEXT_DIR
            if os.path.exists(TEXT_DIR) and any(f.endswith('.txt') for f in os.listdir(TEXT_DIR)):
                print("Found text files, processing them into the database...")
                process_all_texts_into_db()
                db_created = True
            else:
                print("No text files found to process into the database")
                return {
                    "answer": "Database created, but no text files were found to process. Please upload and process PDF files first in the PDF Processing tab.",
                    "sources": [],
                    "model": None
                }
        except Exception as e:
            print(f"Error creating database: {str(e)}")
            return {
                "answer": f"Error creating database: {str(e)}. Please create the database manually in the Vector Database tab.",
                "sources": [],
                "model": None
            }
        
        if db_created:
            return {
                "answer": "Database was automatically created and populated with your text files. You can now chat with your documents.",
                "sources": [],
                "model": None
            }
    
    # First, explicitly check if we have an API key stored
    api_key = get_stored_api_key()
    if not api_key:
        return {
            "answer": "No API key found. Please set your Groq API key in the API Key Settings.",
            "sources": [],
            "model": None
        }
    
    # Ensure API key is initialized - force initialization with the stored key
    # Sometimes the API key is valid but the client initialization failed
    if not groq_client and api_key:
        print(f"API key found but client not initialized. Re-initializing with stored key...")
        client = initialize_groq_client(api_key)
        if client:
            print("Successfully initialized Groq client")
    
    # Even if API key initialization fails, don't return a 503 error
    # Instead, return a helpful message
    if not groq_client:
        return {
            "answer": "The LLM service is not available. Your API key might be invalid or there might be connectivity issues with the Groq API. Please check your API key in the API Key Settings.",
            "sources": [],
            "model": None
        }
    
    # Set default model if none specified
    if not request.model:
        request.model = GROQ_MODEL
        print(f"No model specified, using default: {request.model}")
    
    # Extract provider and model from the request
    provider = getattr(request, 'provider', None)
    model_id = request.model
    logger.info(f"Using model: {model_id}, Provider: {provider}")
    
    # Log the full request for debugging
    logger.info(f"Full request: {request}")
    
    # Check available Ollama models
    available_ollama_models = get_available_ollama_models()
    logger.info(f"Available Ollama models: {len(available_ollama_models)}")
    
    # Determine if this is an Ollama model request
    is_ollama_model = (provider == 'ollama')
    
    # If using Ollama, handle differently than Groq
    if is_ollama_model:
        if not check_ollama_availability():
            return {
                "answer": "Error: Ollama is not available. Please make sure the Ollama server is running.",
                "sources": [],
                "model": None
            }
        
        logger.info(f"Using Ollama model: {model_id}")
        
        # Check if model exists and fallback if not
        if model_id not in available_ollama_models:
            if len(available_ollama_models) > 0:
                fallback_model = available_ollama_models[0]
                logger.warning(f"Ollama model {model_id} not found, falling back to {fallback_model}")
                model_id = fallback_model
                request.model = fallback_model
            else:
                return {
                    "answer": f"Error: Ollama model '{model_id}' not found, and no fallback models available",
                    "sources": [],
                    "model": None
                }

        # For Ollama, we explicitly set the use_ollama flag
        try:
            # Force request parameters for Ollama
            request.temperature = getattr(request, 'temperature', 0.7)
            request.max_tokens = getattr(request, 'max_tokens', 1024)

            # Call a special Ollama chat function
            import logging
            logging.getLogger('app.api.compatibility').info(f"Explicitly calling Ollama with model: {model_id}")

            # Get similar paragraphs from the database
            from app.db.database import similarity_search
            from app.services.llm_service import generate_response
            import os
            
            if os.path.exists(DB_PATH):
                similar_paragraphs = similarity_search(request.query, request.similarity_top_k)
                context = "\n\n".join([paragraph for _, paragraph, _ in similar_paragraphs])
                logger.info(f"Found {len(similar_paragraphs)} similar paragraphs in database")
            else:
                logger.error(f"Vector database not found at {DB_PATH}")
                return {"answer": "Error: Vector database not found. Please process some documents first.", "sources": []}
            
            # Explicitly call generate_response with use_ollama=True
            answer = generate_response(
                request.query, 
                context, 
                model_id, 
                request.temperature,
                max_tokens=request.max_tokens,
                use_ollama=True  # Force using Ollama
            )
            
            # Extract sources for citation
            sources = [{"text": paragraph, "file": filename} for _, paragraph, filename in similar_paragraphs]
            
            # Return the answer directly
            logger.info(f"Successfully generated Ollama response with model: {model_id}")
            return {"answer": answer, "sources": sources, "model": f"ollama:{model_id}"}
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in Ollama chat endpoint: {error_msg}")
            return {
                "answer": f"Error generating response with Ollama: {error_msg}",
                "sources": [],
                "model": None
            }
    
    # If not Ollama, determine Groq model type
    is_production_model = model_id in GROQ_PRODUCTION_MODELS
    is_preview_model = model_id in GROQ_PREVIEW_MODELS
    is_preview_system = model_id in GROQ_PREVIEW_SYSTEMS
    
    # For system models like compound-beta, we need to use max_completion_tokens
    # For regular models, we use max_tokens
    if is_production_model:
        print(f"Using Production model: {model_id}")
    elif is_preview_model:
        print(f"Using Preview model: {model_id}") 
    elif is_preview_system:
        print(f"Using Preview System model: {model_id}")
    else:
        print(f"Model {model_id} not found in any category, will attempt to use anyway")
    
    try:
        # Call the modular endpoint
        response = await chat(request)
        print(f"Chat response generated successfully using model: {response.get('model', 'unknown')}")
        return response
    except Exception as e:
        # Catch any errors and return a helpful message
        error_msg = str(e)
        print(f"Error in chat endpoint: {error_msg}")
        return {
            "answer": f"Error generating response: {error_msg}",
            "sources": [],
            "model": None
        }
