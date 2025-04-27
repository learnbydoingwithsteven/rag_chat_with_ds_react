"""
File management API routes.
"""
from fastapi import APIRouter, HTTPException
import os
from typing import Dict, Any, List

from app.core.config import TEXT_DIR, PDF_DIR

router = APIRouter(prefix="/files", tags=["files"])

@router.get("/text")
async def list_text_files():
    """
    List all text files in the text directory
    """
    try:
        # Check if directory exists
        if not os.path.exists(TEXT_DIR):
            return {
                "status": "warning",
                "message": f"Text directory {TEXT_DIR} does not exist",
                "files": []
            }
        
        # Get all text files
        text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]
        
        if not text_files:
            return {
                "status": "warning",
                "message": "No text files found",
                "files": []
            }
        
        # Get file details
        files = []
        for filename in text_files:
            file_path = os.path.join(TEXT_DIR, filename)
            # Get file size in KB or MB
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            files.append({
                "filename": filename,
                "path": file_path,
                "size": size_bytes,
                "size_str": size_str
            })
        
        return {
            "status": "success",
            "message": f"Found {len(files)} text files",
            "files": files
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing text files: {str(e)}",
            "files": []
        }

@router.get("/documents")
async def list_all_documents():
    """
    List all PDF and text files with their details
    """
    try:
        documents = {
            "pdf_files": [],
            "text_files": []
        }
        
        # Check PDF directory
        if os.path.exists(PDF_DIR):
            pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
            for filename in pdf_files:
                file_path = os.path.join(PDF_DIR, filename)
                size_bytes = os.path.getsize(file_path)
                if size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                
                # Check if corresponding text file exists
                text_filename = os.path.splitext(filename)[0] + ".txt"
                text_path = os.path.join(TEXT_DIR, text_filename)
                has_text = os.path.exists(text_path)
                
                documents["pdf_files"].append({
                    "filename": filename,
                    "path": file_path,
                    "size": size_bytes,
                    "size_str": size_str,
                    "has_text": has_text,
                    "text_filename": text_filename if has_text else None
                })
        
        # Check text directory
        if os.path.exists(TEXT_DIR):
            text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]
            for filename in text_files:
                file_path = os.path.join(TEXT_DIR, filename)
                size_bytes = os.path.getsize(file_path)
                if size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                
                # Check if corresponding PDF exists
                pdf_filename = os.path.splitext(filename)[0] + ".pdf"
                pdf_path = os.path.join(PDF_DIR, pdf_filename)
                has_pdf = os.path.exists(pdf_path)
                
                documents["text_files"].append({
                    "filename": filename,
                    "path": file_path,
                    "size": size_bytes,
                    "size_str": size_str,
                    "has_pdf": has_pdf,
                    "pdf_filename": pdf_filename if has_pdf else None
                })
        
        return {
            "status": "success",
            "message": f"Found {len(documents['pdf_files'])} PDF files and {len(documents['text_files'])} text files",
            "documents": documents
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing documents: {str(e)}",
            "documents": {"pdf_files": [], "text_files": []}
        }

@router.get("/text/{filename}")
async def view_text_content(filename: str):
    """
    View the content of a text file
    """
    try:
        # Check if the file exists
        file_path = os.path.join(TEXT_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Text file {filename} not found")
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "status": "success",
            "filename": filename,
            "content": content
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading text file: {str(e)}")
