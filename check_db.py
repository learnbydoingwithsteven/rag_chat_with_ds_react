import sqlite3
import os

# Connect to the database
db_path = 'bilanci_vectors.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print('Tables in database:', tables)
    
    # Check each table structure and count
    for table in tables:
        print(f'\nTable {table} structure:')
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col}")
        
        # Count rows
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f'Row count: {count}')
        
        # Sample data
        if count > 0:
            print(f'\nSample data from {table} (first 3 rows):')
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            rows = cursor.fetchall()
            for i, row in enumerate(rows):
                print(f"  Row {i+1}: {row[:2]}... (truncated)")
    
    conn.close()
else:
    print(f"Database file {db_path} not found")
