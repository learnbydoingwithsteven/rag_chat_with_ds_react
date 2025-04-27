"""
Vector database API routes.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from typing import Dict, Any, List, Optional
import os
import sqlite3
import pandas as pd

from app.core.config import DB_PATH, TEXT_DIR, background_tasks_status
from app.db.vector_db import create_database, process_all_texts_into_db
from app.db.database import get_all_sources, execute_query

router = APIRouter(prefix="/vector-db", tags=["vector_database"])

@router.post("/create")
async def create_vector_database(background_tasks: BackgroundTasks):
    """
    Create the vector database from text files
    """
    # Generate a task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Set initial task status
    background_tasks_status[task_id] = {
        "status": "pending",
        "message": "Initializing vector database creation..."
    }
    
    # Define the task for background processing
    def create_db_task():
        try:
            background_tasks_status[task_id]["status"] = "running"
            background_tasks_status[task_id]["message"] = "Creating vector database..."
            
            # Check if there are any text files
            if not os.path.exists(TEXT_DIR):
                background_tasks_status[task_id]["status"] = "error"
                background_tasks_status[task_id]["message"] = f"Text directory {TEXT_DIR} does not exist"
                return
            
            text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]
            if not text_files:
                background_tasks_status[task_id]["status"] = "error"
                background_tasks_status[task_id]["message"] = f"No text files found in {TEXT_DIR}"
                return
            
            # Process all text files into the database
            success = process_all_texts_into_db()
            
            if success:
                background_tasks_status[task_id]["status"] = "completed"
                background_tasks_status[task_id]["message"] = "Vector database created successfully"
            else:
                background_tasks_status[task_id]["status"] = "error"
                background_tasks_status[task_id]["message"] = "Error creating vector database"
        except Exception as e:
            background_tasks_status[task_id]["status"] = "error"
            background_tasks_status[task_id]["message"] = f"Error: {str(e)}"
    
    # Add task to background tasks
    background_tasks.add_task(create_db_task)
    
    # Return task ID for status checking
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Vector database creation has been scheduled"
    }

@router.get("/status/{task_id}")
async def get_vector_db_status(task_id: str):
    """
    Get the status of a vector database creation task
    """
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@router.get("/check")
async def check_database():
    """
    Check if the vector database exists and get some basic stats
    """
    try:
        # Check if database file exists
        if not os.path.exists(DB_PATH):
            return {
                "exists": False,
                "message": f"Database file {DB_PATH} does not exist",
                "stats": None
            }
        
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if the documents table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        if not cursor.fetchone():
            conn.close()
            return {
                "exists": True,
                "message": "Database file exists but documents table not found",
                "stats": None
            }
        
        # Get database stats
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        # Get distinct sources
        cursor.execute("SELECT COUNT(DISTINCT source) FROM documents")
        source_count = cursor.fetchone()[0]
        
        # Get embedding model info if available
        model_info = "Unknown"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
        if cursor.fetchone():
            cursor.execute("SELECT value FROM metadata WHERE key='embedding_model'")
            result = cursor.fetchone()
            if result:
                model_info = result[0]
        
        conn.close()
        
        return {
            "exists": True,
            "message": f"Database contains {doc_count} document chunks from {source_count} sources",
            "stats": {
                "total_documents": doc_count,
                "total_sources": source_count,
                "embedding_model": model_info
            }
        }
    except Exception as e:
        return {
            "exists": False,
            "message": f"Error checking database: {str(e)}",
            "stats": None
        }

@router.get("/sources")
async def get_document_sources():
    """
    Get all document sources from the database
    """
    return {"sources": get_all_sources()}
