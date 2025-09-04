import mysql.connector
from config import Config

def update_db_schema():
    # Connect to MySQL database
    conn = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB
    )
    cursor = conn.cursor()

    try:
        # Add session_duration column if it doesn't exist
        cursor.execute("""
            ALTER TABLE posture_records 
            ADD COLUMN IF NOT EXISTS session_duration INT DEFAULT 30
        """)
        
        # Add corrections_count column if it doesn't exist
        cursor.execute("""
            ALTER TABLE posture_records 
            ADD COLUMN IF NOT EXISTS corrections_count INT DEFAULT 0
        """)
        
        # Update foreign key constraint for user_face_embeddings to include CASCADE DELETE
        # First, drop the existing foreign key constraint
        cursor.execute("""
            ALTER TABLE user_face_embeddings 
            DROP FOREIGN KEY IF EXISTS user_face_embeddings_ibfk_1
        """)
        
        # Add the new foreign key constraint with CASCADE DELETE
        cursor.execute("""
            ALTER TABLE user_face_embeddings 
            ADD CONSTRAINT user_face_embeddings_ibfk_1 
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        """)
        
        conn.commit()
        print("Database schema updated successfully!")
        
    except Exception as e:
        print(f"Error updating database schema: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    update_db_schema() 