"""
Database operations and vector search functionality.
"""
import sqlite3
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import DB_PATH, VECTOR_DB_DIR

def execute_query(query: str, params=None):
    """
    Execute a SQLite query and return the results
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        if params:
            results = pd.read_sql_query(query, conn, params=params)
        else:
            results = pd.read_sql_query(query, conn)
        conn.close()
        return results
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return pd.DataFrame()

def get_all_sources():
    """
    Get all document sources from the database
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        
        # Get distinct sources and count of chunks for each
        query = """
        SELECT 
            source, 
            COUNT(*) as chunk_count 
        FROM documents 
        GROUP BY source 
        ORDER BY source
        """
        
        results = pd.read_sql_query(query, conn)
        conn.close()
        
        # Format the results
        sources = []
        for _, row in results.iterrows():
            sources.append({
                "source": row['source'],
                "chunk_count": int(row['chunk_count'])
            })
            
        return sources
    except Exception as e:
        print(f"Error getting sources: {str(e)}")
        return []

def get_similar_paragraphs(query: str, top_k: int = 5, filters: Dict[str, Any] = None):
    """
    Get paragraphs similar to the query from the database
    """
    try:
        # First check if database exists
        if not os.path.exists(DB_PATH):
            # Create directory if needed
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
            
            # Alert that database doesn't exist
            print(f"Database not found at {DB_PATH}")
            return []
            
        from sentence_transformers import SentenceTransformer
        from app.core.config import MODEL_NAME
        
        # Load the model
        model = SentenceTransformer(MODEL_NAME)
        
        # Get query embedding
        query_embedding = model.encode(query)
        
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        
        # Get all documents and their embeddings
        if filters and 'sources' in filters and filters['sources']:
            placeholders = ', '.join(['?' for _ in filters['sources']])
            query = f"""
            SELECT id, source, chunk_text, embedding
            FROM documents
            WHERE source IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=filters['sources'])
        else:
            query = """
            SELECT id, source, chunk_text, embedding
            FROM documents
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if df.empty:
            return []
        
        # Convert embeddings from string to numpy arrays
        df['embedding_array'] = df['embedding'].apply(
            lambda x: np.array(json.loads(x))
        )
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            [query_embedding], 
            np.stack(df['embedding_array'].values)
        )[0]
        
        # Add similarities to dataframe
        df['similarity'] = similarities
        
        # Sort by similarity (descending)
        df = df.sort_values('similarity', ascending=False)
        
        # Get top k results
        top_results = df.head(top_k)
        
        # Format results
        results = []
        for _, row in top_results.iterrows():
            results.append({
                "id": int(row['id']),
                "source": row['source'],
                "text": row['chunk_text'],
                "similarity": float(row['similarity'])
            })
        
        return results
    except Exception as e:
        print(f"Error getting similar paragraphs: {str(e)}")
        return []
