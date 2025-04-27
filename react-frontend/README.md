# Multilingual Financial Statement RAG Chatbot - React Version

This is the React implementation of the Multilingual Financial Statement RAG Chatbot. This application provides the same functionality as the Streamlit version but with a modern React frontend and a FastAPI backend.

## Features

- **Multilingual Support**: Toggle between English and Italian
- **Chat Interface**: Ask questions about harmonized financial statements
- **Vector Visualization**: Interactive visualization of document embeddings with multi-document selection and color coding
- **Financial Data Access**: Access Italian government financial datasets via both the primary CKAN API and the alternative REST API
- **Data Formulator Integration**: Launch and use Data Formulator from within the app

## Project Structure

```
react-frontend/           # React frontend application
├── src/
│   ├── components/       # React components for each tab
│   ├── App.js            # Main application component
│   └── ...
└── package.json          # Frontend dependencies

api_backend.py            # FastAPI backend for the React frontend
```

## Setup and Installation

### Backend Setup

1. Install the required Python packages:

```bash
pip install fastapi uvicorn pandas numpy scikit-learn plotly
pip install requests sqlalchemy pydantic python-multipart
```

2. Start the FastAPI server:

```bash
python -m uvicorn api_backend:app --reload
```

The API will be available at http://localhost:8000.

### Frontend Setup

1. Navigate to the React frontend directory:

```bash
cd react-frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

The React app will open in your browser at http://localhost:3000.

## API Documentation

Once the FastAPI backend is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Using the Application

The application provides five main tabs:

1. **Chat**: Ask questions about harmonized financial statements in either English or Italian
2. **Documents**: Upload and manage documents for the vector database
3. **Vector Visualization**: Visualize document embeddings with interactive PCA and t-SNE plots
4. **Financial Data**: Access and visualize Italian government financial datasets
5. **Data Formulator**: Launch Data Formulator for advanced financial data analysis

## Development

To modify or extend the application:

- Frontend components are located in the `src/components/` directory
- Backend API endpoints are defined in `api_backend.py`
- To add new features, create new components and API endpoints as needed

## Notes

- This application requires the `bilanci_vectors.db` database file to be present in the root directory
- The backend and frontend should be running simultaneously for the application to work properly
