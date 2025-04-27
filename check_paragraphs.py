import sqlite3
import os
import numpy as np

# Connect to the database
db_path = 'bilanci_vectors.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check paragraphs table
    print('\nTable paragraphs structure:')
    cursor.execute(f"PRAGMA table_info(paragraphs)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col}")
    
    # Count rows
    count = conn.execute(f"SELECT COUNT(*) FROM paragraphs").fetchone()[0]
    print(f'Row count: {count}')
    
    # Count rows with non-null embeddings
    count_with_embeddings = conn.execute(f"SELECT COUNT(*) FROM paragraphs WHERE embedding IS NOT NULL").fetchone()[0]
    print(f'Rows with embeddings: {count_with_embeddings}')
    
    # Sample data
    if count > 0:
        print(f'\nSample data from paragraphs (first 3 rows):')
        cursor.execute(f"SELECT id, text, document_id, page, source FROM paragraphs LIMIT 3")
        rows = cursor.fetchall()
        for i, row in enumerate(rows):
            print(f"  Row {i+1}: ID={row[0]}, Text={row[1][:50]}..., Doc={row[2]}, Page={row[3]}")
            
        # Check embedding dimensions
        cursor.execute("SELECT embedding FROM paragraphs WHERE embedding IS NOT NULL LIMIT 1")
        embedding = cursor.fetchone()
        if embedding and embedding[0]:
            try:
                embedding_array = np.frombuffer(embedding[0], dtype=np.float32)
                print(f"\nEmbedding dimensions: {len(embedding_array)}")
            except:
                print("Error converting embedding to array")
    
    conn.close()
else:
    print(f"Database file {db_path} not found")
