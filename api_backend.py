import os
import io
import sqlite3
import pandas as pd
import numpy as np
import requests
import json
import time
import shutil
import tempfile
import re
import PyPDF2
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import subprocess
import sys
import platform
import os
from groq import Groq

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "bilanci_vectors.db")
PDF_DIR = os.path.join(SCRIPT_DIR, "bilanci_pdf")
RULES_DIR = os.path.join(SCRIPT_DIR, "rules")  # Additional directory where PDFs are stored
TEXT_DIR = os.path.join(SCRIPT_DIR, "bilanci_text")
BDAP_ROOT = "https://bdap-opendata.rgs.mef.gov.it/api/3/action"
BDAP_ALT_ROOT = "https://bdap-opendata.rgs.mef.gov.it/api/1/rest"

# Embedding model name for SentenceTransformer
MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast model that works well for semantic similarity

# Groq LLM API settings
# The API key should be set as an environment variable GROQ_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Default to Llama 3 70B model if available, otherwise use Mixtral 8x7B
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
# Fallback models if the main one is not available
FALLBACK_MODELS = ["mixtral-8x7b-32768", "llama3-8b-8192"]

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Check if the rules directory exists and has PDF files
if os.path.exists(RULES_DIR):
    rules_pdfs = [f for f in os.listdir(RULES_DIR) if f.lower().endswith('.pdf')]
    if rules_pdfs and not os.listdir(PDF_DIR):
        print(f"Found PDFs in rules directory: {rules_pdfs}")
        # Use the rules directory as the PDF directory since it has PDFs and bilanci_pdf is empty
        PDF_DIR = RULES_DIR
        print(f"Set PDF_DIR to: {PDF_DIR}")

# Model for text embeddings
MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller model suitable for this application

# Initialize FastAPI app
app = FastAPI(title="Multilingual Chatbot Backend API")

# Add CORS middleware to allow cross-origin requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you'd restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models for API endpoints
class ChatRequest(BaseModel):
    query: str
    language: str = "en"  # Default to English
    similarity_top_k: int = 5  # Default to 5 similar paragraphs
    temperature: float = 0.2  # Default to low temperature for more factual responses
    model: str = ""  # Model ID to use, if empty will use default

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    model: Optional[str] = None
    
class ApiKeyRequest(BaseModel):
    api_key: str
    
class ApiKeyResponse(BaseModel):
    success: bool
    message: str

# Background task status tracking
background_tasks_status = {}

# Path to store the API key (in a real production app, use a more secure method)
API_KEY_FILE = os.path.join(SCRIPT_DIR, ".groq_api_key")

# Function to get stored API key
def get_stored_api_key():
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "r") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading API key file: {str(e)}")
    return None

# Function to initialize Groq client
def initialize_groq_client(api_key=None):
    global groq_client
    
    # Use provided key, stored key, or environment variable (in that order)
    key_to_use = api_key or get_stored_api_key() or GROQ_API_KEY
    
    # Accept any non-empty key without validation
    if key_to_use and len(key_to_use.strip()) > 0:
        try:
            # Print debug info about the key
            print(f"Trying to initialize Groq client with key: {key_to_use[:5]}... (length: {len(key_to_use)})")
            
            # Just create the client without testing it
            # The actual validity will be checked when making a chat request
            groq_client = Groq(api_key=key_to_use)
            
            print(f"Groq client initialization seemed successful")
            return True
        except Exception as e:
            print(f"Error initializing Groq client: {str(e)}")
            import traceback
            traceback.print_exc()
            groq_client = None
            return False
    else:
        print("No API key provided or key is empty")
        return False

# Groq model lists (based on their documentation)
GROQ_PRODUCTION_MODELS = [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192"
]

GROQ_PREVIEW_MODELS = [
    "allam-2-7b",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "mistral-saba-24b",
    "qwen-qwq-32b"
]

GROQ_PREVIEW_SYSTEMS = [
    "compound-beta",
    "compound-beta-mini"
]

# Initialize Groq client if API key is available
groq_client = None
initialize_groq_client()


@app.get("/vector-visualization")
async def get_vector_visualization(selected_sources: List[str] = Query(None)):
    """
    Generate PCA and t-SNE visualizations from document embeddings.
    Returns visualization data with document-wise differentiation.
    """
    try:
        print(f"Vector visualization API called with sources: {selected_sources}")
        
        # Check if database exists
        if not os.path.exists(DB_PATH):
            print(f"Database not found at: {DB_PATH}")
            return {"error": "Database does not exist"}
            
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Database tables: {tables}")
        
        # Prepare document counts and source filter
        doc_counts = {}
        if selected_sources and len(selected_sources) > 0:
            source_filter = "WHERE source IN ('" + "', '".join(selected_sources) + "')"
        else:
            source_filter = ""
        
        print(f"Using source filter: {source_filter}")
        
        # Your database structure has documents and paragraphs tables
        # Get document counts for statistics
        if "documents" in tables:
            cursor.execute(f"SELECT source, COUNT(*) FROM documents {source_filter} GROUP BY source")
            doc_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # DIRECT APPROACH: Get embeddings from paragraphs table (this is the structure of this database)
        print("Direct query to paragraphs table for embeddings - known database structure")
        results = []
        
        try:
            # Prepare query with direct access to paragraphs table
            if selected_sources and len(selected_sources) > 0:
                # Handle partial source matches since database may have absolute paths
                # but frontend passes just filenames
                clauses = []
                for source in selected_sources:
                    # Get just the base filename for comparison
                    base_source = os.path.basename(source)
                    clauses.append(f"source LIKE '%{base_source}'")
                
                where_clause = " OR ".join(clauses)
                query = f"""
                    SELECT id, source, embedding 
                    FROM paragraphs
                    WHERE ({where_clause}) AND embedding IS NOT NULL
                    LIMIT 2000
                """
            else:
                query = """
                    SELECT id, source, embedding
                    FROM paragraphs 
                    WHERE embedding IS NOT NULL
                    LIMIT 2000
                """
            
            print(f"Executing direct paragraphs query: {query}")
            cursor.execute(query)
            results = cursor.fetchall()
            print(f"Direct query returned {len(results)} results")
        except Exception as e:
            print(f"Error querying paragraphs table: {str(e)}")
        
        conn.close()
        
        # Process embeddings for visualization
        if not results:
            print("No embedding results found in any table")
            return {
                "pca_data": [],
                "tsne_data": [],
                "explained_variance": 0,
                "document_counts": doc_counts
            }
        
        print(f"Processing {len(results)} embedding results")
        
        # Extract data from results
        doc_ids = []
        sources = []
        doc_source_map = {}  # Map to group by document source
        embeddings = []
        
        for row in results:
            try:
                doc_id, source, emb_bytes = row
                
                # Check if embedding is valid
                if not emb_bytes:
                    print(f"Warning: Empty embedding for document {doc_id}, source {source}")
                    continue
                
                # Get document ID from source path for grouping
                doc_source = os.path.basename(source)
                if doc_source not in doc_source_map:
                    doc_source_map[doc_source] = f"doc_{len(doc_source_map) + 1}"  # Create stable doc IDs
                
                # Use stable document ID for consistent coloring
                stable_doc_id = doc_source_map[doc_source]
                
                doc_ids.append(stable_doc_id)  # Use a stable document ID
                sources.append(source)
                
                # Convert bytes to numpy array - the embedding is stored as float32
                try:
                    emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
                    
                    # Sanity check on embedding dimensions
                    if len(emb_array) < 10:
                        print(f"Warning: Embedding has insufficient dimensions: {len(emb_array)}")
                        continue
                        
                    embeddings.append(emb_array)
                except Exception as e:
                    print(f"Error converting embedding: {str(e)}")                    
                    # Try fallback to float64 if float32 fails
                    try:
                        emb_array = np.frombuffer(emb_bytes, dtype=np.float64)
                        if len(emb_array) < 10:
                            print(f"Warning: Fallback embedding has insufficient dimensions: {len(emb_array)}")
                            continue
                    except Exception as e2:
                        print(f"Error fallback embedding conversion: {str(e2)}")
                        continue
                embeddings.append(emb_array)
            except Exception as e:
                print(f"Error processing row: {str(e)}")
        
        # Stack embeddings for processing
        if not embeddings:
            print("No valid embeddings found after processing")
            return {
                "pca_data": [],
                "tsne_data": [],
                "explained_variance": 0,
                "document_counts": doc_counts
            }
            
        print(f"Found {len(embeddings)} valid embeddings to visualize")
        embeddings_array = np.vstack(embeddings)
        
        # Create mappings for display names
        unique_sources = list(set(sources))
        source_to_display = {}
        doc_source_map = {}  # For consistent document IDs
        
        for idx, source in enumerate(sorted(unique_sources)):
            # Extract just the filename without path or extension
            basename = os.path.basename(source)
            display_name = basename
            if display_name.lower().endswith('.txt'):
                display_name = display_name[:-4]  # Remove .txt extension
                
            # Create stable document IDs based on basename
            if basename not in doc_source_map:
                doc_source_map[basename] = f"doc_{idx+1}"
                
            # Store display name mapping
            source_to_display[source] = display_name
        
        # Create stable mapping of document IDs to colors
        unique_doc_ids = list(set(doc_ids))
        print(f"Found {len(unique_doc_ids)} unique document IDs for coloring")
        
        # Generate PCA visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings_array)
        explained_variance = float(pca.explained_variance_ratio_.sum())
        
        pca_data = []
        for i, (doc_id, source, vector) in enumerate(zip(doc_ids, sources, pca_result)):
            # Get basename for more consistent document grouping
            basename = os.path.basename(source)
            stable_doc_id = doc_source_map.get(basename, doc_id)
            
            pca_data.append({
                "x": float(vector[0]),
                "y": float(vector[1]),
                "source": source,
                "basename": basename,  # Add basename for easier frontend handling
                "document_id": stable_doc_id,
                "display_name": source_to_display[source]
            })
        
        # Generate t-SNE visualization
        # Adjust perplexity based on data size
        perplexity = min(30, max(5, len(embeddings_array) // 10))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(embeddings_array)
        
        tsne_data = []
        for i, (doc_id, source, vector) in enumerate(zip(doc_ids, sources, tsne_result)):
            # Get basename for more consistent document grouping
            basename = os.path.basename(source)
            stable_doc_id = doc_source_map.get(basename, doc_id)
            
            tsne_data.append({
                "x": float(vector[0]),
                "y": float(vector[1]),
                "source": source,
                "basename": basename,  # Add basename for easier frontend handling
                "document_id": stable_doc_id,
                "display_name": source_to_display[source]
            })
        
        # Return all visualization data
        return {
            "pca_data": pca_data,
            "tsne_data": tsne_data,
            "explained_variance": explained_variance,
            "document_counts": doc_counts
        }
        
    except Exception as e:
        print(f"Error in vector visualization: {str(e)}")
        return {"error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the system using RAG (Retrieval-Augmented Generation).
    The system will search for similar paragraphs in the database and use them to
    generate an answer using the Groq LLM API.
    """
    try:
        # Check if database exists
        if not os.path.exists(DB_PATH):
            raise HTTPException(status_code=404, detail="Database not found")
            
        # Check if Groq client is available
        if not groq_client:
            return ChatResponse(
                answer="The Groq API key is not configured. Please set the GROQ_API_KEY environment variable.",
                sources=[]
            )
            
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Load query embedding model
        model = SentenceTransformer(MODEL_NAME)
        
        # Get query embedding
        query_embedding = model.encode(request.query)
        
        # Check which tables exist in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables in database: {tables}")
        
        # Search for similar paragraphs based on cosine similarity
        similar_paragraphs = []
        try:
            if "paragraphs" in tables:
                # Convert query embedding to bytes format matching the database
                query_embedding_bytes = query_embedding.astype(np.float32).tobytes()
                
                # More efficient approach: Get paragraphs with their embeddings in a single query
                cursor.execute(
                    """SELECT id, text, embedding, document_id, page, source FROM paragraphs 
                    WHERE embedding IS NOT NULL
                    LIMIT 500"""  # Reduced limit for better performance
                )
                
                paragraphs = cursor.fetchall()
                print(f"Found {len(paragraphs)} paragraphs with embeddings")
                
                # Calculate similarities manually
                similarities = []
                for p in paragraphs:
                    p_id, p_text, p_embedding_bytes, p_doc_id, p_page, p_source = p
                    
                    # Use embedding directly from the query result
                    
                    if p_embedding_bytes:
                        try:
                            # Convert bytes to numpy array
                            p_embedding = np.frombuffer(p_embedding_bytes, dtype=np.float32)
                            
                            # Make sure dimensions match
                            if len(p_embedding) != len(query_embedding):
                                print(f"Warning: Embedding dimension mismatch for paragraph {p_id}: {len(p_embedding)} vs {len(query_embedding)}")
                                continue
                                
                            # Calculate cosine similarity with safeguards for zero vectors
                            query_norm = np.linalg.norm(query_embedding)
                            p_norm = np.linalg.norm(p_embedding)
                            
                            if query_norm > 0 and p_norm > 0:
                                similarity = np.dot(query_embedding, p_embedding) / (query_norm * p_norm)
                                
                                # Add to similar paragraphs if similarity is meaningful
                                if not np.isnan(similarity) and similarity > 0.05:  # Lower minimum similarity threshold for better recall
                                    similarities.append({
                                        "id": p_id,
                                        "text": p_text,
                                        "document_id": p_doc_id,
                                        "page": p_page,
                                        "source": p_source,
                                        "similarity": float(similarity)
                                    })
                            else:
                                print(f"Warning: Zero norm detected for paragraph {p_id} or query")
                        except Exception as e:
                            print(f"Error processing paragraph {p_id}: {str(e)}")
                
                # Sort by similarity (highest first) and get top-k
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
                similar_paragraphs = similarities[:request.similarity_top_k]
                
                print(f"Found {len(similar_paragraphs)} similar paragraphs")
            else:
                print("No paragraphs table found in the database")
        except Exception as e:
            print(f"Error searching for similar paragraphs: {str(e)}")
        
        # Format context from similar paragraphs with additional information
        context_text = ""
        if similar_paragraphs:
            # Group paragraphs by source document for better context organization
            paragraphs_by_source = {}
            for p in similar_paragraphs:
                source_key = f"{os.path.basename(p['source'])}"
                if source_key not in paragraphs_by_source:
                    paragraphs_by_source[source_key] = []
                paragraphs_by_source[source_key].append(p)
            
            context_text = "I'm providing you with relevant passages from financial document sources to answer the question. The passages are grouped by document and ordered by relevance. Pay close attention to these sources:\n\n"
            
            # Provide context organized by document
            for doc_idx, (source, paragraphs) in enumerate(paragraphs_by_source.items()):
                # Sort paragraphs within each document by similarity
                paragraphs.sort(key=lambda x: x['similarity'], reverse=True)
                
                context_text += f"SOURCE {doc_idx+1}: {source}\n"
                for i, p in enumerate(paragraphs):
                    similarity_percent = round(p['similarity'] * 100)
                    context_text += f"Passage {i+1} (Page {p['page']}, Relevance: {similarity_percent}%):\n{p['text']}\n\n"
        else:
            context_text = "I couldn't find any relevant information in the financial documents to answer this question.\n"
        
        # Determine language for prompt
        lang_prompt = "in Italian" if request.language == "it" else "in English"
        
        # Create prompt for LLM with improved instructions
        system_prompt = f"""You are a financial expert specialized in Italian harmonized financial statements (bilanci armonizzati) and BDAP (Banca Dati Amministrazioni Pubbliche) regulations. You help users understand complex financial accounting concepts and regulations in the Italian public administration context.
        
        INSTRUCTIONS:
        1. Answer the user's question {lang_prompt} based ONLY on the provided context from financial documents.
        2. If the context doesn't contain sufficient information to answer the question, clearly state which parts you can answer based on the available information and which parts you cannot.
        3. DO NOT make up or hallucinate any information. Only use what's explicitly stated in the context.
        4. For technical accounting terms, budget classifications, or financial regulations, provide precise explanations referencing specific regulations when available.
        5. When discussing document formats, report structures, or accounting rules, be as specific as possible with examples from the context.
        6. If asked about classification codes, account numbers, or specific fields, provide the exact formatting and usage requirements.
        
        FORMAT YOUR RESPONSE:
        - Respond {lang_prompt} in a clear, structured manner.
        - Begin with a direct answer to the question, then provide supporting details.
        - If multiple sources contain relevant information, integrate them into a coherent explanation.
        - When referencing regulations or standards, mention them by name (e.g., "According to D.Lgs 118/2011" or "As specified in the BDAP technical document").
        - Use bullet points or numbered lists when explaining multi-step processes or listing requirements.
        - For complex topics, provide concrete examples from the context.
        - IMPORTANT: Provide accurate, actionable information that would be helpful to financial administrators working with harmonized financial statements.
        """
        
        user_prompt = f"""Context information:
        {context_text}
        
        Question: {request.query}
        """
        
        # Call Groq API for text generation
        try:
            # Use requested model or fall back to default
            current_model = request.model if request.model else GROQ_MODEL
            print(f"Attempting to use Groq with model: {current_model}")
            print(f"Temperature: {request.temperature}")
            print(f"System prompt length: {len(system_prompt)}")
            print(f"User prompt length: {len(user_prompt)}")
            
            try:
                # Add retries for robustness (up to 2 retries with exponential backoff)
                max_retries = 2
                retry_delay = 1  # Start with 1 second delay
                
                for attempt in range(max_retries + 1):
                    try:
                        chat_completion = groq_client.chat.completions.create(
                            model=current_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=request.temperature,
                            max_tokens=1024
                        )
                        answer = chat_completion.choices[0].message.content
                        model_used = current_model
                        print(f"Successfully used model: {current_model}")
                        break  # Success, exit retry loop
                    except Exception as api_error:
                        print(f"API Error attempt {attempt+1}/{max_retries+1}: {str(api_error)}")
                        if attempt < max_retries:
                            # Exponential backoff
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            # Last attempt failed, propagate the error
                            raise
            except Exception as e:
                print(f"Error with primary model {current_model} after retries: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        except Exception as e:
            # Try fallback models if main model fails
            print(f"Error with primary model {current_model}: {str(e)}")
            answer = None
            for fallback_model in FALLBACK_MODELS:
                try:
                    current_model = fallback_model
                    print(f"Trying fallback model: {current_model}")
                    chat_completion = groq_client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=request.temperature,
                        max_tokens=1024
                    )
                    answer = chat_completion.choices[0].message.content
                    model_used = current_model
                    if answer:
                        break
                except Exception as e2:
                    print(f"Error with fallback model {current_model}: {str(e2)}")
            
            # If all models fail or no context was found, return an appropriate response
            if not answer:
                if not similar_paragraphs:
                    if request.language == "it":
                        answer = "Non ho trovato informazioni rilevanti nei documenti finanziari per rispondere a questa domanda. Prova a riformulare la domanda o a chiedere qualcosa di diverso sui bilanci armonizzati."
                    else:
                        answer = "I couldn't find relevant information in the financial documents to answer this question. Try rephrasing your question or asking about a different aspect of harmonized financial statements."
                else:
                    if request.language == "it":
                        answer = "Mi dispiace, si è verificato un errore nel generare una risposta. Riprova più tardi."
                    else:
                        answer = "Sorry, there was an error generating a response. Please try again later."
        
        # Format sources for the response
        sources = []
        for p in similar_paragraphs:
            sources.append({
                "source": os.path.basename(p["source"]),  # Just the filename without path
                "page": p["page"],
                "text": p["text"],
                "similarity": p["similarity"]
            })
        
        conn.close()
        return ChatResponse(answer=answer, sources=sources, model=model_used if 'model_used' in locals() else None)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        if request.language == "it":
            error_msg = f"Errore: {str(e)}"
        else:
            error_msg = f"Error: {str(e)}"
        return ChatResponse(answer=error_msg, sources=[])
        
@app.post("/set-api-key", response_model=ApiKeyResponse)
async def set_api_key(request: ApiKeyRequest):
    """
    Save the Groq API key for future use
    """
    try:
        # Save the API key to a file
        with open(API_KEY_FILE, "w") as f:
            f.write(request.api_key)
            
        # Very basic validation - just make sure it's not empty
        if not request.api_key.strip():
            return ApiKeyResponse(success=False, message="API key cannot be empty")
            
        # Try to initialize Groq client with the new key
        success = initialize_groq_client(request.api_key)
        
        if success:
            return ApiKeyResponse(success=True, message="API key saved successfully. It will be validated when you make a chat request.")
        else:
            # If initialization somehow failed, remove the key
            if os.path.exists(API_KEY_FILE):
                os.remove(API_KEY_FILE)
            return ApiKeyResponse(success=False, message="Could not save API key. Please try again.")
            
    except Exception as e:
        return ApiKeyResponse(success=False, message=f"Error saving API key: {str(e)}")

@app.get("/check-api-key")
async def check_api_key():
    """
    Check if a Groq API key is available (not validating it)
    """
    key_exists = get_stored_api_key() is not None or GROQ_API_KEY != ""
    client_initialized = groq_client is not None
    
    return {
        "key_exists": key_exists,
        "client_initialized": client_initialized,
        "model": GROQ_MODEL if client_initialized else None
    }

@app.get("/groq-models")
async def get_groq_models():
    """
    Return available Groq models organized by category
    """
    models = []
    
    # Check if we have an active Groq client
    if groq_client:
        try:
            # Try to fetch models directly from the API
            models_response = groq_client.models.list()
            # Convert to a simple list of IDs
            models = [model.id for model in models_response.data]
            print(f"Successfully fetched {len(models)} models from Groq API")
        except Exception as e:
            print(f"Error fetching models from Groq API: {str(e)}")
            # Fall back to predefined lists
            models = GROQ_PRODUCTION_MODELS + GROQ_PREVIEW_MODELS + GROQ_PREVIEW_SYSTEMS
    else:
        # If no client, use the predefined lists
        models = GROQ_PRODUCTION_MODELS + GROQ_PREVIEW_MODELS + GROQ_PREVIEW_SYSTEMS
    
    # Organize into categories
    production_models = [m for m in models if m in GROQ_PRODUCTION_MODELS]
    preview_models = [m for m in models if m in GROQ_PREVIEW_MODELS]
    preview_systems = [m for m in models if m in GROQ_PREVIEW_SYSTEMS]
    
    # Include any models not in our predefined lists
    other_models = [m for m in models if m not in GROQ_PRODUCTION_MODELS and 
                                       m not in GROQ_PREVIEW_MODELS and 
                                       m not in GROQ_PREVIEW_SYSTEMS]
    
    return {
        "production_models": production_models,
        "preview_models": preview_models,
        "preview_systems": preview_systems,
        "other_models": other_models,
        "default_model": GROQ_MODEL
    }

# PDF Processing Functions (from 0_pdf_to_text.py)
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def process_pdf_file(pdf_path, output_dir):
    """Process a single PDF file and save its text"""
    try:
        start_time = time.time()
        filename = os.path.basename(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        
        # Skip if no text was extracted
        if not text.strip():
            return {
                "status": "error",
                "file": filename,
                "message": "No text could be extracted"
            }
        
        # Save text to file
        text_filename = os.path.splitext(filename)[0] + ".txt"
        text_path = os.path.join(output_dir, text_filename)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        duration = time.time() - start_time
        return {
            "status": "success",
            "file": filename,
            "text_file": text_filename,
            "duration": f"{duration:.2f}s",
            "text_length": len(text)
        }
    except Exception as e:
        return {
            "status": "error",
            "file": os.path.basename(pdf_path),
            "message": str(e)
        }

def process_all_pdfs(pdf_dir, output_dir):
    """Process all PDF files in a directory"""
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                if f.lower().endswith('.pdf')]
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pdf_file, pdf, output_dir) for pdf in pdf_files]
        for future in futures:
            result = future.result()
            results.append(result)
    
    return results

# Vector Database Functions (from 1_text_to_vector_db.py)
def create_database():
    """Create a new SQLite database for storing document embeddings"""
    # Check if the database exists and if so, check its schema
    db_exists = os.path.exists(DB_PATH)
    if db_exists:
        try:
            # Connect to existing database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if paragraphs table has the text column
            cursor.execute("PRAGMA table_info(paragraphs)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # If 'text' column is missing, alter the table
            if 'text' not in column_names and len(columns) > 0:
                print("Adding 'text' column to paragraphs table")
                cursor.execute("ALTER TABLE paragraphs ADD COLUMN text TEXT")
                conn.commit()
                
            conn.close()
        except Exception as e:
            print(f"Error checking database schema: {str(e)}")
            # If there was an error, we'll recreate the database
            os.remove(DB_PATH)
            db_exists = False
    
    # Create new database if it doesn't exist or was removed due to errors
    if not db_exists:
        print("Creating new database")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create table for documents
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create table for paragraphs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paragraphs (
            id INTEGER PRIMARY KEY,
            text TEXT,
            embedding BLOB,
            document_id INTEGER,
            page INTEGER,
            source TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        """)
        
        conn.commit()
        conn.close()

def load_embedding_model():
    """Load the sentence transformer model for creating embeddings"""
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def process_text_into_db(text_path, model):
    """Process a text file into the vector database"""
    conn = None
    try:
        # Extract source
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        source = text_path
        
        # First, check the database schema to ensure it's valid
        try:
            cursor.execute("PRAGMA table_info(paragraphs)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            if 'text' not in column_names:
                raise Exception(f"Database schema error: 'text' column missing from paragraphs table")
        except sqlite3.Error as e:
            raise Exception(f"Database error checking schema: {str(e)}")
        
        # Check if this source already exists
        cursor.execute("SELECT id FROM documents WHERE source = ?", (source,))
        existing = cursor.fetchone()
        if existing:
            # Delete existing entries for this source
            doc_id = existing[0]
            cursor.execute("DELETE FROM paragraphs WHERE document_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
        
        # Insert new document record
        cursor.execute("INSERT INTO documents (source) VALUES (?)", (source,))
        doc_id = cursor.lastrowid
        
        # Read text file
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            with open(text_path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        # Split into paragraphs and pages
        # For simplicity, using a simple split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Create embeddings for each paragraph
        for i, para in enumerate(paragraphs):
            # Skip very short paragraphs
            if len(para.split()) < 5:
                continue
                
            # Create embedding
            embedding = model.encode(para)
            
            # Convert embedding to bytes for storage
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            # Estimate page number (simplified)
            page = i // 5  # Rough estimate, 5 paragraphs per page
            
            # Store in database
            try:
                cursor.execute("""
                INSERT INTO paragraphs (text, embedding, document_id, page, source)
                VALUES (?, ?, ?, ?, ?)
                """, (para, embedding_bytes, doc_id, page, source))
            except sqlite3.Error as db_err:
                # Print specific error details for debugging
                print(f"Database error inserting paragraph: {str(db_err)}")
                raise Exception(f"Database error inserting paragraph: {str(db_err)}")
        
        conn.commit()
        conn.close()
        conn = None
        
        return {
            "status": "success",
            "file": os.path.basename(text_path),
            "paragraphs": len(paragraphs),
            "document_id": doc_id
        }
    except Exception as e:
        print(f"Error processing {os.path.basename(text_path)}: {str(e)}")
        if conn:
            conn.close()
        return {
            "status": "error",
            "file": os.path.basename(text_path),
            "message": str(e)
        }

def process_all_texts_into_db():
    """Process all text files into the vector database"""
    try:
        # First, forcibly recreate the database to ensure correct schema
        # Remove the existing database if it exists
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            print(f"Removed existing database: {DB_PATH}")
        
        # Create a fresh database with the correct schema
        create_database()
        print("Created fresh database with correct schema")
        
        # Load embedding model
        model = load_embedding_model()
        print(f"Loaded embedding model: {MODEL_NAME}")
        
        # Get all text files
        text_files = [os.path.join(TEXT_DIR, f) for f in os.listdir(TEXT_DIR) 
                    if f.lower().endswith('.txt')]
        print(f"Found {len(text_files)} text files to process")
        
        results = []
        for text_file in text_files:
            print(f"Processing: {os.path.basename(text_file)}")
            result = process_text_into_db(text_file, model)
            results.append(result)
        
        return results
    except Exception as e:
        print(f"Error in process_all_texts_into_db: {str(e)}")
        # Return error results for all text files
        text_files = [os.path.join(TEXT_DIR, f) for f in os.listdir(TEXT_DIR) 
                    if f.lower().endswith('.txt')]
        return [
            {
                "status": "error",
                "file": os.path.basename(text_file),
                "message": str(e)
            } for text_file in text_files
        ]

# Pydantic models for request/response data
class ChatRequest(BaseModel):
    query: str
    language: str
    similarity_top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class DocumentEmbedding(BaseModel):
    x: float
    y: float
    source: str

class VectorVisualizationResponse(BaseModel):
    pca_data: List[DocumentEmbedding]
    tsne_data: List[DocumentEmbedding]
    explained_variance: float
    document_counts: Dict[str, int]

class BDAPDatasetResponse(BaseModel):
    datasets: List[Dict[str, Any]]

class BDAPResourceResponse(BaseModel):
    resources: List[Dict[str, Any]]

# Helper functions
def execute_query(query: str, params=None):
    """Execute a SQLite query and return the results"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def get_all_sources():
    """Get all document sources from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check database structure
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        if "paragraphs" in tables:
            cursor.execute("SELECT DISTINCT source FROM paragraphs")
        elif "chunks" in tables:
            cursor.execute("SELECT DISTINCT source FROM chunks")
        elif "documents" in tables:
            cursor.execute("SELECT DISTINCT source FROM documents")
        else:
            return []
            
        sources = [row[0] for row in cursor.fetchall()]
        conn.close()
        return sources
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sources: {str(e)}")

def get_similar_paragraphs(query: str, top_k: int = 5, filters: Dict[str, Any] = None):
    """Get paragraphs similar to the query from the database"""
    # This is a simplified version - in a real app, you'd implement proper embeddings here
    # For now, we'll just return some mock data
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check which tables exist
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        # For demonstration, we'll just return the first few paragraphs
        if "paragraphs" in tables:
            cursor.execute("SELECT text, source, page FROM paragraphs LIMIT ?", (top_k,))
            results = [{"text": row[0], "source": row[1], "page": row[2]} for row in cursor.fetchall()]
        elif "chunks" in tables:
            cursor.execute("SELECT text, source, page FROM chunks LIMIT ?", (top_k,))
            results = [{"text": row[0], "source": row[1], "page": row[2]} for row in cursor.fetchall()]
        else:
            results = []
            
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching paragraphs: {str(e)}")

def bdap_api(endpoint: str, payload=None):
    """Make a request to the BDAP CKAN API"""
    try:
        url = f"{BDAP_ROOT}/{endpoint}"
        if payload:
            response = requests.post(url, json=payload)
        else:
            response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BDAP API error: {str(e)}")

def bdap_alt_api(endpoint: str):
    """Make a request to the alternative BDAP REST API"""
    try:
        url = f"{BDAP_ALT_ROOT}/{endpoint}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BDAP Alternative API error: {str(e)}")

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Multilingual Chatbot API is running"}

# Data models for PDF processing
class ProcessPdfRequest(BaseModel):
    files: Optional[List[str]] = None

def process_selected_pdfs(pdf_dir, output_dir, selected_files=None):
    """Process selected PDF files in a directory"""
    if selected_files:
        pdf_files = [os.path.join(pdf_dir, f) for f in selected_files 
                    if f.lower().endswith('.pdf') and os.path.exists(os.path.join(pdf_dir, f))]
    else:
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                    if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return []
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pdf_file, pdf, output_dir) for pdf in pdf_files]
        for future in futures:
            result = future.result()
            results.append(result)
    
    return results

# PDF Processing endpoints
@app.post("/process-pdf-files")
async def process_pdf_files(background_tasks: BackgroundTasks, request: ProcessPdfRequest = None):
    """Process PDF files in the PDF directory"""
    # Get selected files if any
    selected_files = request.files if request and request.files else None
    
    # Generate a task ID
    task_id = f"pdf_processing_{int(time.time())}"
    
    # Create a function to run in the background
    def process_pdfs_task():
        try:
            results = process_selected_pdfs(PDF_DIR, TEXT_DIR, selected_files)
            background_tasks_status[task_id] = {
                "status": "completed",
                "results": results,
                "timestamp": time.time()
            }
        except Exception as e:
            background_tasks_status[task_id] = {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    # Register the background task
    background_tasks_status[task_id] = {
        "status": "running",
        "timestamp": time.time(),
        "selected_files": selected_files
    }
    background_tasks.add_task(process_pdfs_task)
    
    return {"task_id": task_id, "status": "started"}

@app.get("/process-pdf-status/{task_id}")
def get_pdf_process_status(task_id: str):
    """Get the status of a PDF processing task"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save the uploaded PDF
    file_path = os.path.join(PDF_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        result = process_pdf_file(file_path, TEXT_DIR)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Vector database endpoints
@app.post("/create-vector-database")
async def create_vector_database(background_tasks: BackgroundTasks):
    """Create the vector database from text files"""
    # Generate a task ID
    task_id = f"vector_db_{int(time.time())}"
    
    # Create a function to run in the background
    def create_db_task():
        try:
            results = process_all_texts_into_db()
            background_tasks_status[task_id] = {
                "status": "completed",
                "results": results,
                "timestamp": time.time()
            }
        except Exception as e:
            background_tasks_status[task_id] = {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    # Register the background task
    background_tasks_status[task_id] = {
        "status": "running",
        "timestamp": time.time()
    }
    background_tasks.add_task(create_db_task)
    
    return {"task_id": task_id, "status": "started"}

@app.get("/vector-db-status/{task_id}")
def get_vector_db_status(task_id: str):
    """Get the status of a vector database creation task"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@app.get("/list-pdf-files")
def list_pdf_files():
    """List all PDF files in the PDF directory"""
    global PDF_DIR  # Declare global variable at the very beginning of the function
    
    print(f"Looking for PDFs in: {PDF_DIR}")
    
    if not os.path.exists(PDF_DIR):
        print(f"Directory does not exist: {PDF_DIR}")
        # Create the directory
        os.makedirs(PDF_DIR, exist_ok=True)
        return {"files": [], "directory": PDF_DIR, "exists": False}
    
    # List all files in the directory for debugging
    all_files = os.listdir(PDF_DIR)
    print(f"All files in directory: {all_files}")
    
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
    print(f"PDF files found: {pdf_files}")
    
    # If we don't find PDFs, check the rules directory
    if not pdf_files and PDF_DIR != RULES_DIR and os.path.exists(RULES_DIR):
        rules_files = os.listdir(RULES_DIR)
        rules_pdfs = [f for f in rules_files if f.lower().endswith('.pdf')]
        if rules_pdfs:
            print(f"Found PDFs in rules directory: {rules_pdfs}")
            # Update the global PDF directory
            PDF_DIR = RULES_DIR
            return {"files": rules_pdfs, "directory": PDF_DIR, "exists": True, "note": "Using 'rules' directory which contains PDF files"}
    
    # Check other possible locations
    if not pdf_files:
        # Check if bilanci_pdf is in the current working directory
        cwd = os.getcwd()
        possible_dirs = [
            os.path.join(cwd, "bilanci_pdf"),
            os.path.join(cwd, "rules"),
            os.path.join(SCRIPT_DIR, "bilanci_pdf"),
            os.path.join(SCRIPT_DIR, "rules")
        ]
        
        for alt_dir in possible_dirs:
            if os.path.exists(alt_dir) and alt_dir != PDF_DIR:
                alt_files = os.listdir(alt_dir)
                alt_pdfs = [f for f in alt_files if f.lower().endswith('.pdf')]
                if alt_pdfs:
                    print(f"Found PDFs in alternate directory: {alt_pdfs}")
                    # Set PDF_DIR to this directory
                    PDF_DIR = alt_dir
                    return {"files": alt_pdfs, "directory": PDF_DIR, "exists": True, "note": f"Using directory: {os.path.basename(alt_dir)}"}
    
    return {"files": pdf_files, "directory": PDF_DIR, "exists": True}

@app.get("/list-text-files")
def list_text_files():
    """List all text files in the text directory"""
    if not os.path.exists(TEXT_DIR):
        return {"files": []}
    
    text_files = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith('.txt')]
    return {"files": text_files}

@app.get("/list-all-documents")
def list_all_documents():
    """List all PDF and text files with their details"""
    documents = []
    
    # List PDF files
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        for file in pdf_files:
            file_path = os.path.join(PDF_DIR, file)
            file_stat = os.stat(file_path)
            documents.append({
                "name": file,
                "type": "pdf",
                "size": file_stat.st_size,
                "modified": file_stat.st_mtime,
                "path": file_path
            })
    
    # List text files
    if os.path.exists(TEXT_DIR):
        text_files = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith('.txt')]
        for file in text_files:
            file_path = os.path.join(TEXT_DIR, file)
            file_stat = os.stat(file_path)
            # Try to find the corresponding PDF
            pdf_name = os.path.splitext(file)[0] + ".pdf"
            has_pdf = os.path.exists(os.path.join(PDF_DIR, pdf_name))
            documents.append({
                "name": file,
                "type": "text",
                "size": file_stat.st_size,
                "modified": file_stat.st_mtime,
                "path": file_path,
                "has_pdf": has_pdf
            })

    return {"documents": documents}

@app.get("/view-text-content/{filename}")
def view_text_content(filename: str):
    """View the content of a text file"""
    file_path = os.path.join(TEXT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    return {"content": content}

@app.get("/check-database")
async def check_database():
    """Check if the vector database exists and get some basic stats"""
    if not os.path.exists(DB_PATH):
        return {"exists": False, "message": "Database not found"}

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check database structure
        # Check which tables exist
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        # Get document counts
        doc_counts = {}
        if "paragraphs" in tables:
            cursor.execute("SELECT COUNT(*) FROM paragraphs")
            doc_counts["paragraphs"] = cursor.fetchone()[0]
        if "chunks" in tables:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            doc_counts["chunks"] = cursor.fetchone()[0]
        if "documents" in tables:
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_counts["documents"] = cursor.fetchone()[0]
            
        conn.close()
        
        return {
            "exists": True,
            "tables": tables,
            "counts": doc_counts
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}

@app.get("/sources")
async def get_document_sources():
    """Get all document sources from the database"""
    sources = get_all_sources()
    return {"sources": sources}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint - get an answer to a query"""
    # This is a simplified version - in a real app, you'd implement proper RAG here
    query = request.query
    language = request.language
    
    # Get similar paragraphs
    similar_paragraphs = get_similar_paragraphs(query, request.similarity_top_k)
    
    # For demonstration, we'll just return a mock answer
    if language == "it":
        answer = f"Risposta alla domanda: {query}"
    else:
        answer = f"Answer to the query: {query}"
    
    return {"answer": answer, "sources": similar_paragraphs}

@app.get("/vector-visualization")
async def vector_visualization(selected_sources: Optional[List[str]] = Query(None)):
    """Get vector visualization data (PCA and t-SNE)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check database structure
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        # Build source filter if sources are selected
        source_filter = ""
        if selected_sources and len(selected_sources) > 0:
            source_list = "', '".join(selected_sources)
            source_filter = f"WHERE source IN ('{source_list}')"
        
        # Get document counts
        doc_counts = {}
        
        # Get embeddings based on database structure
        results = []
        if "paragraphs" in tables:
            # Get counts
            cursor.execute(f"SELECT source, COUNT(*) FROM paragraphs {source_filter} GROUP BY source")
            doc_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get embeddings
            viz_query = f"SELECT source, embedding FROM paragraphs {source_filter} LIMIT 200"
            cursor.execute(viz_query)
            results = cursor.fetchall()
        elif "chunks" in tables:
            # Get counts
            cursor.execute(f"SELECT source, COUNT(*) FROM chunks {source_filter} GROUP BY source")
            doc_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get embeddings
            viz_query = f"SELECT source, embedding FROM chunks {source_filter} LIMIT 200"
            cursor.execute(viz_query)
            results = cursor.fetchall()
        elif "documents" in tables and "embeddings" in tables:
            # Get counts
            cursor.execute(f"SELECT source, COUNT(*) FROM documents {source_filter} GROUP BY source")
            doc_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get embeddings
            if selected_sources and len(selected_sources) > 0:
                source_list = "', '".join(selected_sources)
                viz_query = f"""SELECT d.source, e.embedding 
                    FROM embeddings e 
                    JOIN documents d ON e.document_id = d.id 
                    WHERE d.source IN ('{source_list}')
                    LIMIT 200"""
            else:
                viz_query = """SELECT d.source, e.embedding 
                    FROM embeddings e 
                    JOIN documents d ON e.document_id = d.id 
                    LIMIT 200"""
            cursor.execute(viz_query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Process embeddings
        if not results:
            return {
                "pca_data": [],
                "tsne_data": [],
                "explained_variance": 0,
                "document_counts": doc_counts
            }
        
        sources = []
        embeddings = []
        
        for source, emb_bytes in results:
            sources.append(source)
            # Convert bytes to numpy array
            emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.append(emb_array)
        
        embeddings_array = np.vstack(embeddings)
        
        # Create a mapping of unique document IDs
        unique_sources = list(set(sources))
        source_to_id = {source: f"doc_{i}" for i, source in enumerate(unique_sources)}
        
        # Get document display names (just the filename without path or extension)
        source_to_display = {}
        for source in unique_sources:
            # Extract just the filename without path or extension for display
            display_name = source.split('/')[-1]
            if display_name.lower().endswith('.txt'):
                display_name = display_name[:-4]  # Remove .txt extension
            source_to_display[source] = display_name
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings_array)
        
        pca_data = []
        for i, (source, vector) in enumerate(zip(sources, pca_result)):
            pca_data.append({
                "x": float(vector[0]),
                "y": float(vector[1]),
                "source": source,  # Keep full source path for uniqueness
                "document_id": source_to_id[source],  # Add unique ID
                "display_name": source_to_display[source]  # Add display name
            })
        
        # t-SNE
        perplexity = min(30, len(embeddings_array)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(embeddings_array)
        
        tsne_data = []
        for i, (source, vector) in enumerate(zip(sources, tsne_result)):
            tsne_data.append({
                "x": float(vector[0]),
                "y": float(vector[1]),
                "source": source,  # Keep full source path for uniqueness
                "document_id": source_to_id[source],  # Add unique ID
                "display_name": source_to_display[source]  # Add display name
            })
        
        # Return data
        return {
            "pca_data": pca_data,
            "tsne_data": tsne_data,
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "document_counts": doc_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

@app.get("/bdap/datasets")
async def get_bdap_datasets(q: str = ""):
    """Get BDAP datasets using the primary CKAN API"""
    try:
        payload = {
            "q": q,
            "rows": 100
        }
        response = bdap_api("package_search", payload)
        datasets = response.get("result", {}).get("results", [])
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching BDAP datasets: {str(e)}")

@app.get("/bdap/alternative/datasets")
async def get_bdap_alt_datasets():
    """Get BDAP datasets using the alternative REST API"""
    try:
        response = bdap_alt_api("dataset")
        return {"datasets": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alternative BDAP datasets: {str(e)}")

@app.get("/bdap/alternative/dataset/{dataset_id}")
async def get_bdap_alt_dataset_details(dataset_id: str):
    """Get details for a specific BDAP dataset using the alternative REST API"""
    try:
        response = bdap_alt_api(f"dataset/{dataset_id}")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dataset details: {str(e)}")

@app.get("/bdap/resource/{resource_id}")
async def get_bdap_resource(resource_id: str):
    """Get a specific BDAP resource and return as CSV"""
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

@app.get("/check-data-formulator")
async def check_data_formulator():
    """Check if Data Formulator is installed"""
    try:
        # Try to import data_formulator
        result = subprocess.run(
            [sys.executable, "-c", "import data_formulator; print('installed')"],
            capture_output=True,
            text=True,
            check=False
        )
        
        is_installed = "installed" in result.stdout.strip()
        return {"installed": is_installed}
    except Exception:
        return {"installed": False}

@app.post("/launch-data-formulator")
async def launch_data_formulator():
    """Launch Data Formulator in a separate process"""
    try:
        # Check if data_formulator is installed
        check_result = await check_data_formulator()
        if not check_result["installed"]:
            return {"success": False, "message": "Data Formulator is not installed. Please install it first."}
        
        # Create a temporary launcher script
        launcher_code = """
import subprocess
import sys
import platform

try:
    # Import data_formulator and run it
    from data_formulator import main
    
    # Launch data formulator
    main()
except Exception as e:
    print(f"Error launching Data Formulator: {str(e)}")
    input("Press Enter to exit...")
"""
        
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(launcher_code)
            temp_path = f.name
        
        # Launch the script in a new window
        if platform.system() == "Windows":
            subprocess.Popen(
                ["start", "python", temp_path],
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # macOS or Linux
            subprocess.Popen(
                ["gnome-terminal", "--", "python3", temp_path],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
        
        return {
            "success": True,
            "message": "Data Formulator launched successfully in a new window."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error launching Data Formulator: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
