"""
Services for data visualization, including PCA and t-SNE for vector embeddings.
"""
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from app.core.config import DB_PATH
from app.db.database import execute_query

def get_vector_visualization(selected_sources: List[str] = None):
    """
    Generate PCA and t-SNE visualizations from document embeddings.
    Returns visualization data with document-wise differentiation.
    """
    try:
        # Connect to the database using standard sqlite3
        conn = sqlite3.connect(DB_PATH)
        
        # Build query based on whether we filter by sources
        if selected_sources and len(selected_sources) > 0:
            placeholders = ', '.join(['?' for _ in selected_sources])
            query = f"""
            SELECT id, source, embedding FROM documents
            WHERE source IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=selected_sources)
        else:
            query = """
            SELECT id, source, embedding FROM documents
            """
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return {
                "pca_data": [],
                "tsne_data": [],
                "explained_variance": 0,
                "document_counts": {}
            }
        
        # Convert embeddings from string to numpy arrays
        df['embedding_array'] = df['embedding'].apply(
            lambda x: np.array(json.loads(x))
        )
        
        # Get all embeddings as a matrix
        embeddings = np.vstack(df['embedding_array'].values)
        
        # Count documents per source
        document_counts = df['source'].value_counts().to_dict()
        
        # PCA to reduce dimensionality to 2D
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        
        # t-SNE for non-linear dimensionality reduction
        # Adjust perplexity based on dataset size
        perplexity = min(30, len(df) - 1) if len(df) > 10 else 5
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                   n_iter=1000, random_state=42)
        tsne_result = tsne.fit_transform(embeddings)
        
        # Create visualization results
        pca_data = []
        tsne_data = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            source = row['source']
            
            # PCA data point
            pca_data.append({
                "x": float(pca_result[i, 0]),
                "y": float(pca_result[i, 1]),
                "source": source
            })
            
            # t-SNE data point
            tsne_data.append({
                "x": float(tsne_result[i, 0]),
                "y": float(tsne_result[i, 1]),
                "source": source
            })
        
        # Calculate explained variance for PCA
        explained_variance = float(sum(pca.explained_variance_ratio_))
        
        return {
            "pca_data": pca_data,
            "tsne_data": tsne_data,
            "explained_variance": explained_variance,
            "document_counts": document_counts
        }
    except Exception as e:
        print(f"Error in vector visualization: {str(e)}")
        return {
            "pca_data": [],
            "tsne_data": [],
            "explained_variance": 0,
            "document_counts": {},
            "error": str(e)
        }
