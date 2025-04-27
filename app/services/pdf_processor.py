"""
PDF file processing and text extraction services.
"""
import os
import io
import shutil
import PyPDF2
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from app.core.config import PDF_DIR, TEXT_DIR

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def process_pdf_file(pdf_path, output_dir):
    """
    Process a single PDF file and save its text
    """
    try:
        print(f"Processing PDF: {pdf_path}")
        
        # Get the filename without extension
        filename = os.path.basename(pdf_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Output text file path
        text_file_path = os.path.join(output_dir, f"{name_without_ext}.txt")
        
        # Skip if text file already exists
        if os.path.exists(text_file_path):
            print(f"Text file already exists for {filename}, skipping...")
            return {
                "pdf": filename,
                "text_path": text_file_path,
                "status": "skipped",
                "message": "Text file already exists"
            }
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            return {
                "pdf": filename,
                "status": "error",
                "message": "No text extracted from PDF"
            }
        
        # Save text to file
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return {
            "pdf": filename,
            "text_path": text_file_path,
            "status": "success",
            "message": f"Extracted {len(text.split())} words"
        }
    except Exception as e:
        return {
            "pdf": os.path.basename(pdf_path),
            "status": "error",
            "message": str(e)
        }

def process_all_pdfs(pdf_dir, output_dir):
    """
    Process all PDF files in a directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return {"status": "error", "message": "No PDF files found", "results": []}
    
    # Process files in parallel with ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda pdf: process_pdf_file(pdf, output_dir), pdf_files))
    
    return {"status": "success", "message": f"Processed {len(pdf_files)} PDF files", "results": results}

def process_selected_pdfs(pdf_dir, output_dir, selected_files=None):
    """
    Process selected PDF files in a directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if selected_files:
        # Process only the selected files
        pdf_files = [os.path.join(pdf_dir, f) for f in selected_files if f.lower().endswith('.pdf')]
    else:
        # Process all PDF files
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return {"status": "error", "message": "No PDF files found to process", "results": []}
    
    # Process files in parallel with ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda pdf: process_pdf_file(pdf, output_dir), pdf_files))
    
    return {
        "status": "success", 
        "message": f"Processed {len(pdf_files)} PDF files", 
        "results": results
    }
