import sqlite3
import os
import numpy as np
from app.core.config import DB_PATH

# Connect to the database using the path from config.py
db_path = DB_PATH
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check embeddings in documents table
    print('Checking embeddings in documents table:')
    try:
        # Count documents with non-null embeddings
        cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"Found {count} document chunks with non-null embeddings")
        
        # Check a sample embedding
        if count > 0:
            cursor.execute("SELECT id, embedding, source FROM documents WHERE embedding IS NOT NULL LIMIT 1")
            row = cursor.fetchone()
            if row:
                embedding_id, embedding_bytes, source = row
                if embedding_bytes:
                    try:
                        # In the new schema, embeddings are stored as JSON strings
                        import json
                        emb_array = np.array(json.loads(embedding_bytes))
                        print(f"Sample embedding dimensions: {len(emb_array)}")
                        print(f"Sample embedding source: {source}")
                        print(f"Sample embedding first 5 values: {emb_array[:5]}")
                    except Exception as e:
                        print(f"Error converting JSON embedding: {str(e)}")
                        # Try the old binary format as fallback
                        try:
                            emb_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                            print(f"Fallback binary embedding dimensions: {len(emb_array)}")
                        except Exception as e2:
                            print(f"Error with fallback binary conversion: {str(e2)}")
                else:
                    print(f"Embedding bytes is None for id {embedding_id}")
            else:
                print("No rows returned from sample query")
    except Exception as e:
        print(f"Error checking embeddings: {str(e)}")
    
    # Check full range of sources
    print('\nChecking sources:')
    try:
        cursor.execute("SELECT DISTINCT source FROM documents")
        sources = cursor.fetchall()
        print(f"Found {len(sources)} distinct sources")
        for i, source in enumerate(sources[:5]):  # Show first 5
            print(f"Source {i+1}: {source[0]}")
        if len(sources) > 5:
            print(f"... and {len(sources) - 5} more")
    except Exception as e:
        print(f"Error checking sources: {str(e)}")
    
    conn.close()
else:
    print(f"Database file {db_path} not found")
