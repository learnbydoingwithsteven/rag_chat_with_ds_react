# Multilingual Financial Statement RAG Chatbot - React/FastAPI Version

This is the React/FastAPI implementation of the Multilingual Financial Statement RAG Chatbot for harmonized financial statements. This version separates the frontend (React) from the backend (FastAPI) for a more modern architectural approach.

## Features

- **Multilingual Support**: Toggle between English and Italian interfaces
- **Vector-based RAG**: Find and present the most relevant information from your financial documents
- **Document Management**: Upload and process PDF documents
- **Vector Visualization**: Interactive visualization of document embeddings with multi-document selection and color coding using Plotly
- **Financial Data Access**: Access Italian government financial datasets via:
  - Primary BDAP CKAN API
  - Alternative BDAP REST API
- **Data Formulator Integration**: Launch and use Data Formulator directly from the app
- **Modern UI**: Material UI components for a sleek user experience

## Project Structure

```
api_backend.py            # FastAPI backend server
react-frontend/           # React frontend application
├── src/
│   ├── components/       # React components for each tab
│   │   ├── ChatTab.js    # Chat interface component
│   │   ├── DocumentsTab.js # Document management component
│   │   ├── VectorVisualizationTab.js # Vector visualization component
│   │   ├── FinancialDataTab.js # Financial data access component
│   │   └── DataFormulatorTab.js # Data Formulator integration component
│   ├── App.js            # Main application component
│   └── ...
└── package.json          # Frontend dependencies
bilanci_vectors.db        # SQLite database with vector embeddings
bilanci_pdf/              # Directory for PDF documents
```

## Requirements

### Backend Requirements
- Python 3.8+
- FastAPI and Uvicorn
- Pandas, NumPy, Plotly
- Scikit-learn
- SQLite3
- Requests
- Optional: data_formulator package

### Frontend Requirements
- Node.js and npm
- React
- Material UI
- Plotly.js and React-Plotly.js
- Axios for API calls

## Setup and Installation

### Backend Setup

1. Install the required Python packages:

```bash
pip install fastapi uvicorn pandas numpy scikit-learn plotly
pip install requests python-multipart pydantic
pip install data-formulator # Optional, for Data Formulator integration
```

2. Make sure the vector database exists (create it using the preprocessing scripts if needed):

```bash
# Convert PDFs to text
python 0_pdf_to_text.py

# Create vector embeddings database
python 1_text_to_vector_db.py
```

3. Start the FastAPI server:

```bash
uvicorn api_backend:app --reload
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

## Using the Application

The application provides five main tabs with the same functionality as the Streamlit version:

1. **Chat**: Ask questions about harmonized financial statements in either English or Italian
2. **Documents**: Upload and manage documents for the vector database
3. **Vector Visualization**: Visualize document embeddings with interactive PCA and t-SNE plots
   - Select multiple documents with color coding
   - Explore document similarities visually with interactive Plotly charts
4. **Financial Data**: Access and visualize Italian government financial datasets
   - Switch between primary CKAN API and alternative REST API
   - Download and visualize CSV resources
5. **Data Formulator**: Launch Data Formulator for advanced financial data analysis

## API Documentation

Once the FastAPI backend is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Advantages of the React/FastAPI Version

- **Separation of Concerns**: Frontend and backend are cleanly separated
- **Scalability**: Can scale frontend and backend independently
- **Modern UI**: Material UI provides a polished user experience
- **Interactive Visualizations**: Plotly.js offers more interactive data visualizations
- **API-First Design**: Well-documented API can be used by other applications
- **Performance**: React's virtual DOM provides optimized rendering performance

## Notes

- Both backend and frontend must be running for the application to work
- Make sure the database file exists and is accessible to the backend
- For best results, upload high-quality, text-searchable PDFs
- The multilingual interface supports both English and Italian
