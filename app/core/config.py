"""
Application configuration settings.
"""
import os
from pathlib import Path

# Base directory setup
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_DB_DIR = os.path.join(SCRIPT_DIR, "vectordb")
DB_PATH = os.path.join(VECTOR_DB_DIR, "embeddings.db")
PDF_DIR = os.path.join(SCRIPT_DIR, "bilanci_pdf")
RULES_DIR = os.path.join(SCRIPT_DIR, "rules")
TEXT_DIR = os.path.join(SCRIPT_DIR, "bilanci_text")
API_KEY_FILE = os.path.join(SCRIPT_DIR, ".groq_api_key")

# BDAP API endpoints
BDAP_ROOT = "https://bdap-opendata.rgs.mef.gov.it/api/3/action"
BDAP_ALT_ROOT = "https://bdap-opendata.rgs.mef.gov.it/api/1/rest"

# Embedding model settings
MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast model that works well for semantic similarity

# Groq LLM configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
API_KEY_FILE = os.path.join(SCRIPT_DIR, ".groq_api_key")

# Default Groq model to use if none is specified
# Using a reliable model that's known to be stable
GROQ_MODEL = "llama-3.1-8b-instant"

# List of fallback models to try if the requested model fails
# These models are known to be reliable and available
FALLBACK_MODELS = ["mixtral-8x7b-32768", "gemma-7b-it"]

# Groq models - valid for April 2025
# Organization by type makes more sense than just a single list

# Production models - reliable for production use
GROQ_PRODUCTION_MODELS = [
    "llama-3.1-8b-instant",  # Our default model
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
    "mixtral-8x7b-32768",  # Added for compatibility with older references
]

# Preview models - use for testing, may be discontinued at short notice
GROQ_PREVIEW_MODELS = [
    "allam-2-7b",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "mistral-saba-24b",
    "playai-tts",
    "playai-tts-arabic",
    "qwen-qwq-32b",
]

# Preview systems - special models that require different parameters
GROQ_PREVIEW_SYSTEMS = [
    "compound-beta",
    "compound-beta-mini"
]

# Ollama models - common local models available through Ollama
OLLAMA_MODELS = [
    "llama3.2",             # Llama 3.2 (8B)
    "llama3.2:90b",         # Llama 3.2 (90B)
    "llama3.2-vision",      # Llama 3.2 Vision (11B)
    "llama3.2-vision:90b",  # Llama 3.2 Vision (90B)
    "llama3.1",             # Llama 3.1 (8B)
    "llama3.1:405b",        # Llama 3.1 (405B)
    "phi4",                 # Phi 4 (14B)
    "phi4-mini",            # Phi 4 Mini (3.8B)
    "mistral",              # Mistral (7B)
    "moondream",            # Moondream 2 (1.4B)
    "neural-chat",          # Neural Chat (7B)
    "starling-lm",          # Starling (7B)
    "codellama",            # Code Llama (7B)
    "gemma2",               # Gemma 2 (9B)
    "gemma2:27b",           # Gemma 2 (27B)
]

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Check if the rules directory exists and has PDF files
if os.path.exists(RULES_DIR):
    rules_pdfs = [f for f in os.listdir(RULES_DIR) if f.lower().endswith('.pdf')]
    if rules_pdfs and not os.listdir(PDF_DIR):
        print(f"Found PDFs in rules directory: {rules_pdfs}")
        # Use the rules directory as the PDF directory since it has PDFs and bilanci_pdf is empty
        PDF_DIR = RULES_DIR
        print(f"Set PDF_DIR to: {PDF_DIR}")

# Background task status tracking (for async operations)
background_tasks_status = {}
