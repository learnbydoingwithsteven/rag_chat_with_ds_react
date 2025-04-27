"""
Entry point for the RAG Chatbot application.
This file starts the FastAPI server using the modular backend.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
