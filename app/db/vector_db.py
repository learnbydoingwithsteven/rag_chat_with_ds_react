"""
Vector database creation and management functions.
"""
import os
import sqlite3
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.core.config import DB_PATH, TEXT_DIR, MODEL_NAME

def create_database():
    """
    Create a new SQLite database for storing document embeddings
    """
    try:
        # Check if db already exists
        if os.path.exists(DB_PATH):
            print(f"Database already exists at {DB_PATH}")
            return False
        
        # Create a new database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create a table for documents
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        ''')
        
        # Create a table for metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        ''')
        
        # Add embedding model info
        cursor.execute('''
        INSERT INTO metadata (key, value) VALUES (?, ?)
        ''', ('embedding_model', MODEL_NAME))
        
        # Create index on source field for faster filtering
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_source ON documents(source)
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"Created new database at {DB_PATH}")
        return True
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        return False

def load_embedding_model():
    """
    Load the sentence transformer model for creating embeddings
    """
    try:
        # Load the model
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        print(f"Error loading embedding model: {str(e)}")
        return None

def process_text_into_db(text_path, model):
    """
    Process a text file into the vector database
    """
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get the source name from the filename
        filename = os.path.basename(text_path)
        source = os.path.splitext(filename)[0]
        
        # Check if this source already exists in the database
        cursor.execute("SELECT COUNT(*) FROM documents WHERE source = ?", (source,))
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"Source '{source}' already exists in the database with {count} chunks.")
            # Skip this file - uncomment below to remove existing and re-add
            # cursor.execute("DELETE FROM documents WHERE source = ?", (source,))
            # conn.commit()
            # print(f"Removed existing {count} chunks for source '{source}'.")
            conn.close()
            return False
        
        # Load the text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into paragraphs (chunks) - adjust the splitting logic as needed
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Alternative splitting approach (for more uniform chunks)
        # Split into sentences and then combine into chunks of ~200-300 words
        # This may be more appropriate for different document types
        
        # Create embeddings for each chunk
        for i, chunk in enumerate(chunks):
            # Skip very short chunks
            if len(chunk.split()) < 10:  # Skip chunks with fewer than 10 words
                continue
                
            # Create embedding
            embedding = model.encode(chunk)
            
            # Store as JSON string (SQLite doesn't support arrays directly)
            embedding_json = json.dumps(embedding.tolist())
            
            # Store in database
            cursor.execute('''
            INSERT INTO documents (source, chunk_index, chunk_text, embedding)
            VALUES (?, ?, ?, ?)
            ''', (source, i, chunk, embedding_json))
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print(f"Processed {len(chunks)} chunks from '{source}'")
        return True
    except Exception as e:
        print(f"Error processing text file {text_path}: {str(e)}")
        return False

def process_all_texts_into_db():
    """
    Process all text files into the vector database
    """
    try:
        # Check if DB exists, if not create it
        if not os.path.exists(DB_PATH):
            print(f"Database does not exist at {DB_PATH}, creating it...")
            create_result = create_database()
            if not create_result:
                return {
                    "status": "error", 
                    "message": "Failed to create database"
                }
            print("Database created successfully")
        else:
            print(f"Using existing database at {DB_PATH}")
        
        # Get all text files
        if not os.path.exists(TEXT_DIR):
            os.makedirs(TEXT_DIR, exist_ok=True)
            print(f"Created text directory at {TEXT_DIR}")
            return {
                "status": "warning", 
                "message": f"Text directory {TEXT_DIR} was empty. Process PDF files first."
            }
        
        text_files = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith('.txt')]
        
        if not text_files:
            return {
                "status": "warning", 
                "message": f"No text files found in {TEXT_DIR}. Process PDF files first."
            }
        
        print(f"Found {len(text_files)} text files to process")
        
        # Load embedding model - look for a cached model in compatibility layer first
        try:
            from app.api.compatibility import get_cached_embedding_model
            model = get_cached_embedding_model()
            print("Using cached embedding model")
        except (ImportError, AttributeError):
            print("Loading embedding model from scratch")
            model = load_embedding_model()
            
        if not model:
            return {
                "status": "error", 
                "message": "Failed to load embedding model"
            }
        
        # Process all text files with better error handling and logging
        total_files = len(text_files)
        processed_files = 0
        failed_files = []
        
        # Connect once to improve performance
        conn = sqlite3.connect(DB_PATH)
        
        for idx, filename in enumerate(text_files):
            try:
                text_path = os.path.join(TEXT_DIR, filename)
                print(f"Processing file {idx+1}/{total_files}: {filename}")
                
                # Check if file already processed to avoid duplicates
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents WHERE source = ?", (filename,))
                count = cursor.fetchone()[0]
                
                if count > 0:
                    print(f"File {filename} already in database with {count} paragraphs, skipping")
                    processed_files += 1
                    continue
                
                # Process the file
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into paragraphs (non-empty lines)
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                # Create embeddings for each paragraph
                for p_idx, paragraph in enumerate(paragraphs):
                    if len(paragraph) < 10:  # Skip very short paragraphs
                        continue
                        
                    # Create embedding
                    embedding = model.encode(paragraph)
                    
                    # Convert embedding to JSON string for storage
                    embedding_json = json.dumps(embedding.tolist())
                    
                    # Save to database
                    cursor.execute("""
                    INSERT INTO documents (source, paragraph_num, content, embedding)
                    VALUES (?, ?, ?, ?)
                    """, (filename, p_idx, paragraph, embedding_json))
                
                conn.commit()
                processed_files += 1
                print(f"Processed {filename} with {len(paragraphs)} paragraphs")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                failed_files.append(filename)
        
        # Close connection
        conn.close()
        
        # Return status
        if processed_files == total_files:
            return {
                "status": "success", 
                "message": f"Processed {processed_files} text files into the database"
            }
        elif processed_files > 0:
            return {
                "status": "partial", 
                "message": f"Processed {processed_files} of {total_files} text files. Failed: {', '.join(failed_files)}"
            }
        else:
            return {
                "status": "error", 
                "message": f"Failed to process any text files"
            }
    
    except Exception as e:
        print(f"Error processing text files: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"Error: {str(e)}"
        }
