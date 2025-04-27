"""
PDF processing API routes.
"""
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from typing import Dict, Any, List, Optional
import os
import shutil

from app.core.config import PDF_DIR, TEXT_DIR, background_tasks_status
from app.models.schemas import ProcessPdfRequest
from app.services.pdf_processor import process_selected_pdfs, process_all_pdfs

router = APIRouter(prefix="/pdf", tags=["pdf_processing"])

@router.post("/process")
async def process_pdf_files(background_tasks: BackgroundTasks, request: ProcessPdfRequest = None):
    """
    Process PDF files in the PDF directory
    """
    # Generate a task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Set initial task status
    background_tasks_status[task_id] = {
        "status": "pending",
        "message": "Initializing PDF processing..."
    }
    
    # Define the task for background processing
    def process_pdfs_task():
        try:
            background_tasks_status[task_id]["status"] = "running"
            background_tasks_status[task_id]["message"] = "Processing PDF files..."
            
            # Process PDFs
            if request and request.files:
                result = process_selected_pdfs(PDF_DIR, TEXT_DIR, request.files)
            else:
                result = process_all_pdfs(PDF_DIR, TEXT_DIR)
            
            # Update status
            background_tasks_status[task_id]["status"] = "completed"
            background_tasks_status[task_id]["message"] = result["message"]
            background_tasks_status[task_id]["results"] = result["results"]
        except Exception as e:
            background_tasks_status[task_id]["status"] = "error"
            background_tasks_status[task_id]["message"] = f"Error: {str(e)}"
    
    # Add task to background tasks
    background_tasks.add_task(process_pdfs_task)
    
    # Return task ID for status checking
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "PDF processing has been scheduled"
    }

@router.get("/status/{task_id}")
async def get_pdf_process_status(task_id: str):
    """
    Get the status of a PDF processing task
    """
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        # Ensure PDF directory exists
        os.makedirs(PDF_DIR, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(PDF_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully",
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/list")
async def list_pdf_files():
    """
    List all PDF files in the PDF directory
    """
    try:
        # Check if directory exists
        if not os.path.exists(PDF_DIR):
            return {
                "status": "warning",
                "message": f"PDF directory {PDF_DIR} does not exist",
                "files": []
            }
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return {
                "status": "warning",
                "message": "No PDF files found",
                "files": []
            }
        
        # Get file details
        files = []
        for filename in pdf_files:
            file_path = os.path.join(PDF_DIR, filename)
            # Get file size in KB or MB
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            # Check if text file exists
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_path = os.path.join(TEXT_DIR, text_filename)
            has_text = os.path.exists(text_path)
            
            files.append({
                "filename": filename,
                "path": file_path,
                "size": size_bytes,
                "size_str": size_str,
                "has_text": has_text,
                "text_path": text_path if has_text else None
            })
        
        return {
            "status": "success",
            "message": f"Found {len(files)} PDF files",
            "files": files
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing PDF files: {str(e)}",
            "files": []
        }
