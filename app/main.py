"""
Main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import API routers
from app.api import chat, pdf_processing, vector_db, visualization
from app.api import bdap, files, settings, data_formulator, compatibility

# Create FastAPI app
app = FastAPI(title="Multilingual Chatbot Backend API")

# Add CORS middleware to allow cross-origin requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you'd restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Chatbot API"}

# Include all routers
app.include_router(chat.router)
app.include_router(pdf_processing.router)
app.include_router(vector_db.router)
app.include_router(visualization.router)
app.include_router(bdap.router)
app.include_router(files.router)
app.include_router(settings.router)
app.include_router(data_formulator.router)

# Include compatibility routes (for backward compatibility with the original API)
app.include_router(compatibility.router)

# Start server when script is run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
