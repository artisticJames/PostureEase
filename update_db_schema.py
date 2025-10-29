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
        try:
            cursor.execute("""
                ALTER TABLE posture_records 
                ADD COLUMN session_duration INT DEFAULT 30
            """)
            print("Added session_duration column")
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("session_duration column already exists")
            else:
                raise e
        
        # Add corrections_count column if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE posture_records 
                ADD COLUMN corrections_count INT DEFAULT 0
            """)
            print("Added corrections_count column")
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("corrections_count column already exists")
            else:
                raise e
        
        # Add good_time column if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE posture_records 
                ADD COLUMN good_time INT DEFAULT 0
            """)
            print("Added good_time column")
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("good_time column already exists")
            else:
                raise e
        
        # Add bad_time column if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE posture_records 
                ADD COLUMN bad_time INT DEFAULT 0
            """)
            print("Added bad_time column")
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("bad_time column already exists")
            else:
                raise e
        
        # Add status column to users table if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN status BOOLEAN DEFAULT TRUE
            """)
            print("Added status column to users table")
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("status column already exists in users table")
            else:
                raise e
        
        # Add last_login column to users table if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN last_login TIMESTAMP NULL
            """)
            print("Added last_login column to users table")
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("last_login column already exists in users table")
            else:
                raise e
        
        
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