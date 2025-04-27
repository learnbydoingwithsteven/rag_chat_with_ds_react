"""
Service for interacting with the Groq LLM API.
"""
import os
import time
import logging
from typing import Optional, Dict, Any, List
from groq import Groq
from app.core.config import GROQ_API_KEY, GROQ_MODEL, FALLBACK_MODELS
from app.core.config import GROQ_PRODUCTION_MODELS, GROQ_PREVIEW_MODELS, GROQ_PREVIEW_SYSTEMS
from app.core.config import API_KEY_FILE, OLLAMA_MODELS

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API settings
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

# Flag to indicate if Ollama is available
ollama_available = False

# Global Groq client
groq_client = None

# Check if Ollama is available at startup
def check_ollama_availability():
    """Check if Ollama is available by running the 'ollama list' command"""
    global ollama_available
    try:
        # First try API method
        import requests
        # Use the list endpoint as recommended in the Ollama docs
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                # Check if we got a valid response with models
                if "models" in data and isinstance(data["models"], list):
                    model_count = len(data["models"])
                    logger.info(f"Ollama service is available with {model_count} models (API check)")
                    ollama_available = True
                    return True
        except Exception as api_error:
            logger.info(f"API check failed, trying command line: {str(api_error)}")
        
        # If API fails, try command line method
        import subprocess
        try:
            # Run the command with a timeout
            result = subprocess.run(["ollama", "list"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   timeout=5)
            
            if result.returncode == 0:
                # Command was successful
                logger.info(f"Ollama is available (command line check)")
                logger.info(f"Models: {result.stdout.strip()}")
                ollama_available = True
                return True
            else:
                # Command failed
                logger.warning(f"'ollama list' failed with error: {result.stderr.strip()}")
                ollama_available = False
                return False
        except subprocess.TimeoutExpired:
            logger.warning("'ollama list' command timed out")
            ollama_available = False
            return False
        except FileNotFoundError:
            logger.warning("'ollama' command not found - may not be installed or not in PATH")
            ollama_available = False
            return False
    except Exception as e:
        logger.warning(f"Ollama availability check failed: {str(e)}")
        ollama_available = False
        return False

# Try to initialize Ollama at startup
try:
    check_ollama_availability()
except Exception as e:
    logger.warning(f"Failed to check Ollama availability at startup: {str(e)}")

def get_stored_api_key():
    """
    Get stored API key from file
    """
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "r") as f:
                key = f.read().strip()
                if key:
                    print(f"Retrieved API key from file: {key[:4]}...{key[-4:] if len(key) > 8 else '***'}")
                    return key
                else:
                    print("API key file exists but is empty")
        except Exception as e:
            print(f"Error reading API key file: {str(e)}")
    else:
        print(f"API key file not found at: {API_KEY_FILE}")
    return None

def initialize_groq_client(api_key=None):
    """
    Initialize Groq API client and return it
    """
    # Use provided key, stored key, or environment variable (in that order)
    key_to_use = api_key or get_stored_api_key() or GROQ_API_KEY
    
    if not key_to_use:
        print("No Groq API key available, LLM functionality will be limited")
        return None
    
    # Strip whitespace as this can cause validation issues
    key_to_use = key_to_use.strip()
    
    # Basic validation to catch common errors
    if not (key_to_use.startswith('gsk_') and len(key_to_use) > 10):
        print(f"API key appears invalid - doesn't start with 'gsk_' or is too short")
        # We'll still try to initialize, but log the warning
    
    # Print some debug info (masked key)
    masked_key = key_to_use[:4] + "***" + key_to_use[-4:] if len(key_to_use) > 8 else "***"
    print(f"Attempting to initialize Groq client with key {masked_key}")
    
    try:
        # Create the client - this step may not actually validate the key
        client = Groq(api_key=key_to_use)
        logger.info("Created Groq client")
        
        # Try to do a simple operation to really validate the key
        try:
            # Use a simple, lightweight model for validation from Groq's production models
            test_model = "llama-3.1-8b-instant"  # Current reliable production model
            
            print(f"Validating API key with model {test_model}...")
            test_response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model=test_model,
                max_tokens=5,  # Minimal tokens for quick validation
            )
            
            print(f"Groq client successfully validated: {test_response.model}")
            return client
            
        except Exception as test_error:
            # Log but continue - the key might still be valid for other operations
            print(f"Warning: Initial validation test failed: {str(test_error)}")
            
            # Try with a fallback model - using one of Groq's more reliable models
            try:
                fallback_model = "mixtral-8x7b-32768"  # Another reliable model
                print(f"Trying fallback validation with {fallback_model}...")
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "Test"}],
                    model=fallback_model,
                    max_tokens=5
                )
                print(f"Groq client validated with fallback model")
                return client
            except Exception as fallback_error:
                print(f"Fallback validation also failed: {str(fallback_error)}")
                # We'll still initialize the client object, even if validation failed
                # This way, it might work for subsequent operations
                print("Initializing client anyway, but it may not be fully operational")
                return client
        
    except Exception as e:
        print(f"Critical error initializing Groq client: {str(e)}")
        
        # If we can't even create the client object, the key is definitely invalid
        return None

def save_api_key(api_key):
    """
    Save API key to file for future use
    """
    try:
        # Clean the API key of any whitespace
        cleaned_key = api_key.strip()
        
        # Basic validation
        if not cleaned_key:
            print("Cannot save empty API key")
            return False
            
        # Make sure the directory exists
        api_key_dir = os.path.dirname(API_KEY_FILE)
        os.makedirs(api_key_dir, exist_ok=True)
        
        # Save the key to file
        with open(API_KEY_FILE, "w") as f:
            f.write(cleaned_key)
            
        print(f"Successfully saved API key to {API_KEY_FILE}")
        
        # Attempt to validate the key immediately
        client = initialize_groq_client(cleaned_key)
        if client:
            print("API key validated successfully")
        else:
            print("Warning: Saved API key but validation failed - the key may not be valid")
            
        return client is not None
        
    except Exception as e:
        print(f"Error saving API key: {str(e)}")
        return False

def get_available_models():
    """
    Get available Groq models by category
    """
    return {
        "production": GROQ_PRODUCTION_MODELS,
        "preview": GROQ_PREVIEW_MODELS,
        "systems": GROQ_PREVIEW_SYSTEMS,
        "ollama": get_available_ollama_models()
    }

def get_available_ollama_models():
    """
    Get available Ollama models using both API and command-line approaches
    """
    models = []
    
    # First try the API method
    try:
        import requests
        logger.info(f"Fetching available Ollama models from {OLLAMA_API_URL}/api/tags")
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            
            # Extract model names based on the API response format
            if 'models' in models_data and isinstance(models_data['models'], list):
                # Newer Ollama API format with 'models' key
                models = [model['name'] for model in models_data['models']]
                logger.info(f"Found {len(models)} Ollama models via API")
            elif isinstance(models_data, list):
                # Some versions might return a direct list
                models = [model['name'] for model in models_data if 'name' in model]
                logger.info(f"Found {len(models)} Ollama models via API (list format)")
    except Exception as e:
        logger.warning(f"API method failed, will try command line: {str(e)}")
    
    # If API didn't find any models or failed, try command line
    if not models:
        try:
            import subprocess
            import re
            
            # Run the 'ollama list' command
            result = subprocess.run(["ollama", "list"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                # Command was successful and returned output
                output = result.stdout.strip()
                logger.info(f"Ollama list command output: {output}")
                
                # Parse the output - typically has format like:
                # NAME                    ID              SIZE    MODIFIED
                # llama3.1:latest         ...
                
                lines = output.split('\n')
                if len(lines) > 1:  # Skip header line
                    for line in lines[1:]:  # Start from the second line
                        parts = line.split()
                        if parts:  # Make sure there's at least one part
                            # The model name is the first column, might have :tag
                            model_name = parts[0].strip()
                            models.append(model_name)
                
                logger.info(f"Found {len(models)} Ollama models via command line")
            else:
                logger.warning(f"'ollama list' didn't return valid data: {result.stderr.strip()}")
        except Exception as e:
            logger.error(f"Command line method failed: {str(e)}")
    
    # Add predefined models from config that weren't found
    from app.core.config import OLLAMA_MODELS
    for model in OLLAMA_MODELS:
        if model not in models:
            models.append(model)
    
    logger.info(f"Total available Ollama models: {len(models)}")
    return models

def get_all_available_models():
    """
    Get all available models from both Groq and Ollama
    """
    models = []
    
    # Add Groq predefined models
    models.extend(GROQ_PRODUCTION_MODELS)
    models.extend(GROQ_PREVIEW_MODELS)
    models.extend(GROQ_PREVIEW_SYSTEMS)
    
    # Add Ollama predefined models
    models.extend(OLLAMA_MODELS)
    
    # Add dynamically available Ollama models
    ollama_models = get_available_ollama_models()
    for model in ollama_models:
        if model not in models:
            models.append(model)
    
    return models

def generate_response(query: str, context, model_id: str = "", temperature: float = 0.2, max_tokens: int = 1024, use_ollama: bool = False):
    """
    Generate a response from the LLM using RAG context
    
    Args:
        query: The user's query
        context: Either a string or a list of dict/tuple results from database
        model_id: Model name to use
        temperature: Temperature for generation
        max_tokens: Maximum number of tokens to generate
        use_ollama: Whether to use Ollama instead of Groq
    """
    logger.info(f"Generating response with parameters: model={model_id}, use_ollama={use_ollama}")
    
    # If use_ollama flag is set, directly use Ollama
    if use_ollama:
        logger.info(f"Using Ollama model explicitly: {model_id}")
        return generate_response_with_ollama(query, context, model_id, temperature, max_tokens)
        
    # Determine if this is an Ollama model based on model ID
    is_ollama_model = (
        model_id in OLLAMA_MODELS or 
        (model_id and model_id not in GROQ_PRODUCTION_MODELS and 
                  model_id not in GROQ_PREVIEW_MODELS and 
                  model_id not in GROQ_PREVIEW_SYSTEMS)
    )
    
    # If it's an Ollama model and Ollama is available, use it
    if is_ollama_model and check_ollama_availability():
        logger.info(f"Model recognized as Ollama model: {model_id}")
        return generate_response_with_ollama(query, context, model_id, temperature, max_tokens)
        
    # Get Groq client (will be passed to the function)
    client = None
    
    # Try to initialize client if needed and not using Ollama
    if not use_ollama:
        # Initialize a client with default settings (will use saved API key)
        client = initialize_groq_client()
        
        if not client:
            logger.error("Failed to initialize Groq client - this will likely fail")
    
    model_to_use = model_id or GROQ_MODEL
    
    # Set the global client for compatibility with other functions
    # This isn't ideal but maintains compatibility with existing code
    global groq_client
    groq_client = client
    
    # Create context text from provided context items
    # Context can be either a string or a list of dict/tuple results from database
    if isinstance(context, str):
        context_text = context
    else:  # Assume it's a list of database results
        context_items = []
        for i, item in enumerate(context):
            # Handle different possible formats
            if isinstance(item, dict):
                text = item.get("text", "")
                source = item.get("source", f"Source {i+1}")
                context_items.append(f"[{i+1}] {text} (Source: {source})")
            elif isinstance(item, tuple) and len(item) >= 3:
                # Handle tuple format (id, text, source)
                _, text, source = item
                context_items.append(f"[{i+1}] {text} (Source: {source})")
            else:
                # Fallback for unknown format
                context_items.append(f"[{i+1}] {str(item)}")
        
        context_text = "\n\n".join(context_items)

    # Create system prompt
    system_prompt = f"""You are a helpful assistant that provides accurate information based on the given context. 
If the context doesn't contain information to answer the query, acknowledge that you don't know rather than making up information.
Always provide concise, accurate responses based only on the context provided.
When referencing sources, use the source numbers from the context e.g., [1], [2], etc."""
    
    # Build base message structure for all model types
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuery: {query}"}
    ]
    
    # Determine proper request parameters based on model type
    is_production_model = model_to_use in GROQ_PRODUCTION_MODELS
    is_preview_model = model_to_use in GROQ_PREVIEW_MODELS
    is_preview_system = model_to_use in GROQ_PREVIEW_SYSTEMS
    
    # Base parameters for all model types
    params = {
        "messages": messages,
        "temperature": temperature,
    }
    
    # Model-specific parameters
    if is_production_model or is_preview_model:
        # Regular language models - use model and max_tokens parameters
        params["model"] = model_to_use
        params["max_tokens"] = max_tokens
    elif is_preview_system:
        # Systems (like compound-beta) - use different parameter format
        # according to the Groq documentation
        params["model"] = model_to_use
        params["max_completion_tokens"] = max_tokens
    else:
        # Default behavior for unknown models
        params["model"] = model_to_use
        params["max_tokens"] = max_tokens
    
    try:
        # Send request to Groq
        start_time = time.time()
        
        logger.info(f"Generating response using Groq model: {model_to_use}")
        response = client.chat.completions.create(**params)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Extract and return response
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "model": model_to_use,
            "generation_time": generation_time
        }
    except Exception as e:
        logger.error(f"Error generating response with Groq model {model_to_use}: {str(e)}")
        
        # Try with a fallback model if the primary one failed
        for fallback_model in FALLBACK_MODELS:
            try:
                # Update the model in the parameters
                fallback_params = params.copy()
                fallback_params["model"] = fallback_model
                
                # Remove max_completion_tokens if exists when switching models
                if "max_completion_tokens" in fallback_params and fallback_model not in GROQ_PREVIEW_SYSTEMS:
                    fallback_params.pop("max_completion_tokens")
                    fallback_params["max_tokens"] = max_tokens
                
                logger.info(f"Trying fallback model {fallback_model}")
                response = client.chat.completions.create(**fallback_params)
                answer = response.choices[0].message.content
                
                return {
                    "answer": answer,
                    "model": fallback_model
                }
            except Exception as e2:
                logger.error(f"Error with fallback model {fallback_model}: {str(e2)}")
        
        # If Ollama is available, try using a default Ollama model as ultimate fallback
        if check_ollama_availability():
            try:
                ollama_fallback = "llama3.1"  # Popular default model
                logger.info(f"Trying Ollama fallback model: {ollama_fallback}")
                return generate_response_with_ollama(query, context, ollama_fallback, temperature, max_tokens)
            except Exception as e3:
                logger.error(f"Error with Ollama fallback: {str(e3)}")
        
        return {
            "answer": f"Error generating response: {str(e)}",
            "model": None
        }

def generate_response_with_ollama(query: str, context, model_id: str, temperature: float = 0.2, max_tokens: int = 1024):
    """
    Generate a response using the Ollama API
    Based on the API documentation in ollama.md
    """
    import requests
    import json
    import time
    
    logger.info(f"Using Ollama with model {model_id}")
    
    # Validate Ollama availability
    if not check_ollama_availability():
        logger.error("Ollama service is not available")
        return {
            "answer": "Ollama service is not available. Please ensure Ollama is running.",
            "model": None
        }
    
    # Create context text from provided context items
    # Context can be either a string or a list of dict results from database
    if isinstance(context, str):
        context_text = context
    else:  # Assume it's a list of database results
        context_items = []
        for i, item in enumerate(context):
            # Handle different possible formats
            if isinstance(item, dict):
                text = item.get("text", "")
                source = item.get("source", f"Source {i+1}")
                context_items.append(f"[{i+1}] {text} (Source: {source})")
            elif isinstance(item, tuple) and len(item) >= 3:
                # Handle tuple format (id, text, source)
                _, text, source = item
                context_items.append(f"[{i+1}] {text} (Source: {source})")
            else:
                # Fallback for unknown format
                context_items.append(f"[{i+1}] {str(item)}")
        
        context_text = "\n\n".join(context_items)
    
    # Create system prompt
    system_prompt = """You are a helpful assistant that provides accurate information based on the given context. 
If the context doesn't contain information to answer the query, acknowledge that you don't know rather than making up information.
Always provide concise, accurate responses based only on the context provided.
When referencing sources, use the source numbers from the context e.g., [1], [2], etc."""
    
    # Define the chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuery: {query}"}
    ]
    
    # Create options dictionary
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,  # Use the provided max_tokens parameter
        "top_k": 40,
        "top_p": 0.9
    }
    
    # Set up the payload for Ollama API according to docs
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,    # Non-streaming response
        "options": options
    }
    
    try:
        # Send request to Ollama
        start_time = time.time()
        
        logger.info(f"Generating response using Ollama model: {model_id}")
        logger.info(f"Sending request to {OLLAMA_API_URL}/api/chat")
        
        response = requests.post(f"{OLLAMA_API_URL}/api/chat", json=payload, timeout=120)  # Longer timeout for larger models
        end_time = time.time()
        generation_time = end_time - start_time
        
        if response.status_code == 200:
            response_json = response.json()
            logger.info(f"Received response from Ollama: {str(response_json)[:100]}...")
            
            # Extract message content based on documented response format
            if "message" in response_json and "content" in response_json["message"]:
                answer = response_json["message"]["content"]
            else:
                # Fallback extraction attempts
                answer = response_json.get("response", "") or response_json.get("content", "")
                if not answer:
                    answer = "No response content received from Ollama in expected format."
            
            return {
                "answer": answer,
                "model": model_id,
                "generation_time": generation_time
            }
        else:
            error_content = response.text[:200] if response.text else "No error details"
            logger.error(f"Ollama API error: HTTP {response.status_code}: {error_content}")
            return {
                "answer": f"Error: Ollama returned status code {response.status_code}. {error_content}",
                "model": None
            }
    except Exception as e:
        logger.error(f"Error generating response with Ollama model {model_id}: {str(e)}")
        return {
            "answer": f"Error generating response with Ollama: {str(e)}",
            "model": None
        }
